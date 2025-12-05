"""
Independent Sentiment Strategy
==============================

This strategy tests whether sentiment features have predictive power
for volatility expansion and price direction, INDEPENDENT of the
existing volatility breakout strategy.

Key Hypothesis (from literature):
- Sentiment extremes predict volatility expansion
- Extreme fear (high VIX, bearish AAII, high PCR) is contrarian bullish
- Extreme greed (low VIX, bullish AAII, low PCR) is contrarian bearish

Following project methodology:
1. Validate sentiment has standalone predictive power
2. Only ensemble with vol breakout if this strategy passes validation
3. Use same walk-forward framework and QC checks

References:
- Baker & Wurgler (2006): Investor sentiment predicts returns
- Tetlock (2007): Media pessimism predicts market
- MacroMicro research: PCR >1.1 = market trough, <0.8 = market top

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.historical_sentiment_loader import (
    HistoricalSentimentLoader,
    HistoricalSentimentConfig
)

logger = logging.getLogger(__name__)


@dataclass
class SentimentStrategyConfig:
    """Configuration for the sentiment strategy."""

    # Data paths
    data_dir: Path = None

    # Sentiment thresholds (contrarian signals)
    # High VIX = fear = contrarian bullish
    vix_fear_threshold: float = 25.0
    vix_complacency_threshold: float = 15.0

    # Composite contrarian threshold (adjusted for actual data distribution)
    # VIX contrarian ranges ~0.2-0.5 in low VIX environment
    vix_contrarian_bullish_threshold: float = 0.4   # High fear = contrarian bullish
    vix_contrarian_bearish_threshold: float = 0.25  # Low fear = contrarian bearish

    # Use VIX percentile for relative thresholds
    vix_percentile_high_threshold: float = 0.7  # VIX above 70th percentile = fear
    vix_percentile_low_threshold: float = 0.3   # VIX below 30th percentile = complacency

    # Regime requirements
    require_regime_confirmation: bool = False  # Don't require absolute thresholds

    # Position sizing
    base_contracts: int = 1
    max_contracts: int = 3

    # Exit parameters (ATR-based)
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.0
    max_holding_bars: int = 20  # ~100 minutes at 5-min bars

    # Trading costs (ES futures)
    commission_per_side: float = 1.29  # NinjaTrader rate
    slippage_ticks: float = 0.5
    tick_size: float = 0.25
    point_value: float = 50.0

    # Walk-forward settings
    train_days: int = 60
    test_days: int = 5
    embargo_bars: int = 20

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent.parent / 'data'


class SentimentStrategy:
    """
    Independent sentiment-based trading strategy.

    This strategy uses sentiment features to:
    1. Predict volatility expansion (WHEN to trade)
    2. Determine direction via contrarian logic (WHICH direction)
    3. Set dynamic exits based on ATR

    The strategy is designed to be validated independently before
    being combined with the volatility breakout strategy.
    """

    def __init__(self, config: Optional[SentimentStrategyConfig] = None):
        self.config = config or SentimentStrategyConfig()
        self.sentiment_loader = HistoricalSentimentLoader()
        self.models: Dict = {}
        self.feature_columns: List[str] = []

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load and prepare price data with sentiment features."""
        # Load ES price data
        es_file = self.config.data_dir / 'raw' / 'market' / 'ES_1min_databento.csv'

        if not es_file.exists():
            raise FileNotFoundError(f"ES data not found at {es_file}")

        logger.info(f"Loading ES data from {es_file}")
        df = pd.read_csv(es_file)

        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['ts_event'], utc=True).dt.tz_localize(None)
        df = df.set_index('timestamp').sort_index()

        # Filter date range if specified
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        # Resample to 5-minute bars
        df_5min = df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        logger.info(f"Loaded {len(df_5min)} 5-minute bars")

        # Add technical features
        df_5min = self._add_technical_features(df_5min)

        # Add sentiment features
        df_5min = self._add_sentiment_features(df_5min)

        # Add target labels
        df_5min = self._add_targets(df_5min)

        # Drop rows with NaN
        df_5min = df_5min.dropna()

        logger.info(f"Final dataset: {len(df_5min)} bars with {len(df_5min.columns)} columns")

        return df_5min

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features."""
        # ATR for position sizing
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        df['atr_14'] = tr.rolling(14).mean()
        df['atr_20'] = tr.rolling(20).mean()

        # Realized volatility
        returns = df['close'].pct_change()
        df['rv_10'] = returns.rolling(10).std() * np.sqrt(288)  # Annualized
        df['rv_20'] = returns.rolling(20).std() * np.sqrt(288)

        # Volatility ratio (current vs recent)
        df['vol_ratio'] = df['rv_10'] / df['rv_20']

        # Price position
        df['close_vs_high_20'] = (df['close'] - df['low'].rolling(20).min()) / \
                                  (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-8)

        # Momentum
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)

        # Volume
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-8)

        return df

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add aligned sentiment features."""
        # Load and align sentiment data
        self.sentiment_loader.load_all()
        sentiment_features = self.sentiment_loader.align_to_bars(df.index)

        # Join sentiment features
        for col in sentiment_features.columns:
            df[col] = sentiment_features[col].values

        # Store feature columns for modeling
        self.feature_columns = [col for col in df.columns if col.startswith('sent_')]

        # Add sentiment-based derived features
        df['sent_vol_alignment'] = (
            (df['vol_ratio'] > 1.0) &
            (df.get('sent_vix_fear_regime', 0) == 1)
        ).astype(int)

        df['sent_contrarian_strength'] = np.abs(df.get('sent_composite_contrarian', 0))

        return df

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add target labels for the sentiment strategy.

        Targets:
        1. vol_expansion: Did volatility expand in the next N bars?
        2. direction: Did price go up or down?
        3. breakout_up/down: Did price break recent highs/lows?
        """
        # Volatility expansion (our primary target)
        # Did realized vol in next 10 bars exceed current RV?
        future_rv = df['rv_10'].shift(-10)
        df['target_vol_expansion'] = (future_rv > df['rv_10'] * 1.2).astype(int)

        # Direction (secondary - expect low predictability)
        future_close = df['close'].shift(-10)
        df['target_direction'] = (future_close > df['close']).astype(int)

        # Breakout targets
        future_high = df['high'].rolling(10).max().shift(-10)
        future_low = df['low'].rolling(10).min().shift(-10)
        current_high = df['high'].rolling(20).max()
        current_low = df['low'].rolling(20).min()

        df['target_breakout_up'] = (future_high > current_high).astype(int)
        df['target_breakout_down'] = (future_low < current_low).astype(int)

        return df

    def generate_signals(self, df: pd.DataFrame, use_model: bool = True) -> pd.DataFrame:
        """
        Generate trading signals based on sentiment features.

        If use_model=True (default): Uses trained ML model to predict vol expansion
        If use_model=False: Uses simple contrarian rules (for baseline comparison)

        Signal logic:
        - LONG when: Model predicts vol expansion + contrarian bullish (high VIX percentile)
        - SHORT when: Model predicts vol expansion + contrarian bearish (low VIX percentile)
        """
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['position_size'] = 0
        signals['tp_price'] = 0.0
        signals['sl_price'] = 0.0
        signals['vol_expansion_prob'] = 0.0

        # Get model predictions if available
        vol_expansion_probs = None
        if use_model and 'vol_expansion' in self.models:
            feature_cols = self.feature_columns
            X = df[feature_cols].copy()
            valid_idx = ~X.isna().any(axis=1)
            probs = np.zeros(len(df))
            probs[valid_idx] = self.models['vol_expansion'].predict_proba(X[valid_idx])[:, 1]
            vol_expansion_probs = probs

        min_vol_prob = 0.55  # Threshold for vol expansion prediction

        for i in range(20, len(df)):
            row = df.iloc[i]

            # Get sentiment values
            vix_percentile = row.get('sent_vix_percentile_20d', 0.5)
            vix_contrarian = row.get('sent_vix_contrarian_signal', 0)

            # Get vol expansion probability from model
            vol_prob = vol_expansion_probs[i] if vol_expansion_probs is not None else 0.5
            signals.iloc[i, signals.columns.get_loc('vol_expansion_prob')] = vol_prob

            signal = 0

            # Only trade when volatility expansion is predicted
            if vol_prob > min_vol_prob:
                # Direction from contrarian sentiment
                # High VIX percentile = fear = contrarian bullish
                if vix_percentile > self.config.vix_percentile_high_threshold:
                    signal = 1  # LONG
                # Low VIX percentile = complacency = contrarian bearish
                elif vix_percentile < self.config.vix_percentile_low_threshold:
                    signal = -1  # SHORT

            if signal != 0:
                # Position sizing based on vol expansion probability
                strength = vol_prob - min_vol_prob
                contracts = min(
                    max(1, int(1 + strength * 4)),
                    self.config.max_contracts
                )

                # ATR-based exits
                atr = row.get('atr_14', row.get('atr_20', 5.0))
                entry_price = row['close']

                if signal == 1:  # LONG
                    tp = entry_price + (atr * self.config.tp_atr_mult)
                    sl = entry_price - (atr * self.config.sl_atr_mult)
                else:  # SHORT
                    tp = entry_price - (atr * self.config.tp_atr_mult)
                    sl = entry_price + (atr * self.config.sl_atr_mult)

                signals.iloc[i, signals.columns.get_loc('signal')] = signal
                signals.iloc[i, signals.columns.get_loc('position_size')] = contracts
                signals.iloc[i, signals.columns.get_loc('tp_price')] = tp
                signals.iloc[i, signals.columns.get_loc('sl_price')] = sl

        return signals

    def backtest(self, df: pd.DataFrame, signals: pd.DataFrame) -> Dict:
        """
        Run backtest on generated signals.

        Returns:
            Dictionary with backtest metrics and trade log
        """
        trades = []
        position = 0
        entry_price = 0
        entry_time = None
        entry_contracts = 0
        tp_price = 0
        sl_price = 0
        bars_held = 0

        for i in range(len(df)):
            current_time = df.index[i]
            current_close = df.iloc[i]['close']
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']

            # Check for exit if in position
            if position != 0:
                bars_held += 1
                exit_price = None
                exit_reason = None

                # Check TP/SL
                if position == 1:  # LONG
                    if current_high >= tp_price:
                        exit_price = tp_price
                        exit_reason = 'TP'
                    elif current_low <= sl_price:
                        exit_price = sl_price
                        exit_reason = 'SL'
                else:  # SHORT
                    if current_low <= tp_price:
                        exit_price = tp_price
                        exit_reason = 'TP'
                    elif current_high >= sl_price:
                        exit_price = sl_price
                        exit_reason = 'SL'

                # Max holding time
                if bars_held >= self.config.max_holding_bars:
                    exit_price = current_close
                    exit_reason = 'MAX_HOLD'

                # Exit position
                if exit_price is not None:
                    # Calculate P&L
                    if position == 1:
                        gross_pnl = (exit_price - entry_price) * self.config.point_value * entry_contracts
                    else:
                        gross_pnl = (entry_price - exit_price) * self.config.point_value * entry_contracts

                    # Apply costs
                    commission = 2 * self.config.commission_per_side * entry_contracts
                    slippage = 2 * self.config.slippage_ticks * self.config.tick_size * \
                               self.config.point_value * entry_contracts
                    net_pnl = gross_pnl - commission - slippage

                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'direction': 'LONG' if position == 1 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'contracts': entry_contracts,
                        'gross_pnl': gross_pnl,
                        'commission': commission,
                        'slippage': slippage,
                        'net_pnl': net_pnl,
                        'exit_reason': exit_reason,
                        'bars_held': bars_held
                    })

                    position = 0
                    entry_contracts = 0
                    bars_held = 0

            # Check for new entry
            if position == 0:
                signal = signals.iloc[i]['signal']
                if signal != 0:
                    position = signal
                    entry_price = current_close
                    entry_time = current_time
                    entry_contracts = int(signals.iloc[i]['position_size'])
                    tp_price = signals.iloc[i]['tp_price']
                    sl_price = signals.iloc[i]['sl_price']
                    bars_held = 0

        # Calculate metrics
        if not trades:
            return {'trades': [], 'metrics': {}}

        trades_df = pd.DataFrame(trades)

        total_trades = len(trades_df)
        winners = trades_df[trades_df['net_pnl'] > 0]
        losers = trades_df[trades_df['net_pnl'] <= 0]

        total_pnl = trades_df['net_pnl'].sum()
        win_rate = len(winners) / total_trades if total_trades > 0 else 0

        avg_win = winners['net_pnl'].mean() if len(winners) > 0 else 0
        avg_loss = losers['net_pnl'].mean() if len(losers) > 0 else 0

        profit_factor = abs(winners['net_pnl'].sum() / losers['net_pnl'].sum()) \
            if len(losers) > 0 and losers['net_pnl'].sum() != 0 else 0

        # Daily returns for Sharpe
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_time']).dt.date
        daily_pnl = trades_df.groupby('exit_date')['net_pnl'].sum()
        sharpe = (daily_pnl.mean() / (daily_pnl.std() + 1e-8)) * np.sqrt(252)

        # Max drawdown
        cumulative_pnl = trades_df['net_pnl'].cumsum()
        rolling_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - rolling_max
        max_drawdown = drawdown.min()

        metrics = {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'avg_bars_held': trades_df['bars_held'].mean(),
            'long_trades': len(trades_df[trades_df['direction'] == 'LONG']),
            'short_trades': len(trades_df[trades_df['direction'] == 'SHORT'])
        }

        return {'trades': trades, 'trades_df': trades_df, 'metrics': metrics}

    def train_models(self, df: pd.DataFrame) -> Dict:
        """
        Train ML models to predict sentiment-based targets.

        Models:
        1. Volatility expansion predictor
        2. Breakout predictor

        Uses LightGBM following project best practices.
        """
        try:
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import roc_auc_score
        except ImportError:
            logger.error("LightGBM not installed. Skipping model training.")
            return {}

        # Get feature columns (sentiment + technical)
        feature_cols = [col for col in df.columns if
                        col.startswith('sent_') or
                        col in ['atr_14', 'rv_10', 'vol_ratio', 'rsi_14', 'volume_ratio', 'close_vs_high_20']]

        X = df[feature_cols].copy()
        y_vol = df['target_vol_expansion'].copy()

        # Remove NaN
        valid_idx = ~(X.isna().any(axis=1) | y_vol.isna())
        X = X[valid_idx]
        y_vol = y_vol[valid_idx]

        # Train/test split (temporal)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y_vol.iloc[:split_idx], y_vol.iloc[split_idx:]

        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)}")

        # Train volatility expansion model
        vol_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )

        vol_model.fit(X_train, y_train)
        y_pred = vol_model.predict_proba(X_test)[:, 1]
        vol_auc = roc_auc_score(y_test, y_pred)

        logger.info(f"Vol Expansion Model - Test AUC: {vol_auc:.4f}")

        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': vol_model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(f"Top 10 features:\n{importance.head(10)}")

        self.models['vol_expansion'] = vol_model
        self.feature_columns = feature_cols

        return {
            'vol_auc': vol_auc,
            'feature_importance': importance
        }


def run_sentiment_strategy_test():
    """Run a test of the sentiment strategy."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("SENTIMENT STRATEGY TEST")
    print("=" * 70)

    # Initialize strategy
    print("\n[1] Initializing strategy...")
    strategy = SentimentStrategy()

    # Load data
    print("\n[2] Loading data...")
    try:
        df = strategy.load_data(start_date='2024-01-01', end_date='2024-06-30')
        print(f"    Loaded {len(df)} bars")
        print(f"    Sentiment features: {len([c for c in df.columns if c.startswith('sent_')])}")
    except Exception as e:
        print(f"    Error loading data: {e}")
        return

    # Train models
    print("\n[3] Training models...")
    try:
        model_results = strategy.train_models(df)
        if model_results:
            print(f"    Vol Expansion AUC: {model_results.get('vol_auc', 'N/A'):.4f}")
    except Exception as e:
        print(f"    Error training models: {e}")

    # Generate signals
    print("\n[4] Generating signals...")
    signals = strategy.generate_signals(df)
    signal_count = (signals['signal'] != 0).sum()
    print(f"    Total signals: {signal_count}")
    print(f"    Long signals: {(signals['signal'] == 1).sum()}")
    print(f"    Short signals: {(signals['signal'] == -1).sum()}")

    # Run backtest
    print("\n[5] Running backtest...")
    results = strategy.backtest(df, signals)

    if results['metrics']:
        print(f"\n    BACKTEST RESULTS:")
        print(f"    ================")
        for key, value in results['metrics'].items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_sentiment_strategy_test()
