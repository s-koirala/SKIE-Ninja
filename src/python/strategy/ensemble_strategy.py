"""
Ensemble Strategy: Volatility Breakout + Sentiment
===================================================

Combines the validated volatility breakout strategy with sentiment features
to potentially improve vol expansion prediction and trade filtering.

Architecture:
1. Original Vol Breakout (validated: +$763K across 5 years)
   - Vol expansion filter (AUC 0.84)
   - Breakout direction (AUC 0.72)
   - ATR-based exits (R² 0.36)

2. Sentiment Enhancement
   - Additional VIX-based vol expansion signal (AUC 0.77)
   - Sentiment regime filtering
   - Agreement-based entry (both models must agree)

Integration Methods:
- Method A: Agreement Filter - Both vol models must predict expansion
- Method B: Weighted Average - Combine probabilities with learned weights
- Method C: Stacking - Meta-model on strategy outputs

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score
import lightgbm as lgb

from feature_engineering.multi_target_labels import MultiTargetLabeler
from data_collection.historical_sentiment_loader import HistoricalSentimentLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble strategy."""

    # Entry filters
    min_vol_expansion_prob: float = 0.50      # Technical vol model threshold
    min_sentiment_vol_prob: float = 0.55      # Sentiment vol model threshold
    min_breakout_prob: float = 0.50           # Breakout threshold

    # Ensemble method: 'agreement', 'weighted', 'either'
    ensemble_method: str = 'agreement'

    # Weights for weighted method
    technical_weight: float = 0.6
    sentiment_weight: float = 0.4

    # Position sizing
    base_contracts: int = 1
    max_contracts: int = 3

    # Dynamic exits
    tp_atr_mult_base: float = 2.0
    sl_atr_mult_base: float = 1.0
    max_holding_bars: int = 20

    # Trading costs
    commission_per_side: float = 1.29
    slippage_ticks: float = 0.5
    tick_size: float = 0.25
    point_value: float = 50.0

    # Walk-forward settings
    train_days: int = 60
    test_days: int = 5
    embargo_bars: int = 20


class EnsembleStrategy:
    """
    Ensemble strategy combining vol breakout with sentiment.

    Uses the validated vol breakout architecture and adds sentiment
    as an additional vol expansion predictor for confirmation.
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()

        # Models
        self.vol_model = None           # Technical vol expansion
        self.sentiment_vol_model = None  # Sentiment-based vol expansion
        self.breakout_high_model = None
        self.breakout_low_model = None
        self.atr_model = None

        # Data loaders
        self.target_labeler = MultiTargetLabeler()
        self.sentiment_loader = HistoricalSentimentLoader()

        # Feature tracking
        self.technical_features = []
        self.sentiment_features = []
        self.all_features = []

    def generate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate technical features (same as vol breakout)."""
        features = pd.DataFrame(index=df.index)

        # Returns (lagged)
        for lag in [1, 2, 3, 5, 10, 20]:
            features[f'return_lag{lag}'] = df['close'].pct_change(lag)

        # ATR calculation
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        for period in [5, 10, 14, 20]:
            features[f'rv_{period}'] = df['close'].pct_change().rolling(period).std()
            features[f'atr_{period}'] = tr.rolling(period).mean()
            features[f'atr_pct_{period}'] = features[f'atr_{period}'] / df['close']

        # Price position
        for period in [10, 20, 50]:
            features[f'close_vs_high_{period}'] = (
                df['close'] - df['high'].rolling(period).max()
            ) / df['close']
            features[f'close_vs_low_{period}'] = (
                df['close'] - df['low'].rolling(period).min()
            ) / df['close']

        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['close'].pct_change(period)
            ma = df['close'].rolling(period).mean()
            features[f'ma_dist_{period}'] = (df['close'] - ma) / ma

        # RSI
        for period in [7, 14]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Bollinger Band position
        mid = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        features['bb_pct_20'] = (df['close'] - lower) / (upper - lower + 1e-10)

        # Volume features
        if 'volume' in df.columns:
            for period in [5, 10, 20]:
                features[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
                features[f'volume_ratio_{period}'] = df['volume'] / (
                    features[f'volume_sma_{period}'] + 1
                )

        self.technical_features = list(features.columns)
        return features

    def generate_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate sentiment features aligned to price bars."""
        self.sentiment_loader.load_all()
        sentiment = self.sentiment_loader.align_to_bars(df.index)

        self.sentiment_features = list(sentiment.columns)
        return sentiment

    def prepare_data(self, prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare all features and targets."""
        logger.info("Generating technical features...")
        tech_features = self.generate_technical_features(prices)

        logger.info("Generating sentiment features...")
        sent_features = self.generate_sentiment_features(prices)

        logger.info("Generating targets...")
        targets = self.target_labeler.generate_all_targets(prices)

        # Combine features
        all_features = pd.concat([tech_features, sent_features], axis=1)
        self.all_features = list(all_features.columns)

        # Align indices
        common_idx = all_features.index.intersection(targets.index)
        all_features = all_features.loc[common_idx]
        targets = targets.loc[common_idx]
        prices_aligned = prices.loc[common_idx]

        # Remove NaN
        valid_mask = ~(all_features.isna().any(axis=1) | targets.isna().any(axis=1))
        all_features = all_features[valid_mask]
        targets = targets[valid_mask]
        prices_aligned = prices_aligned[valid_mask]

        logger.info(f"Prepared {len(all_features)} samples with {len(self.all_features)} features")
        logger.info(f"  Technical: {len(self.technical_features)}, Sentiment: {len(self.sentiment_features)}")

        return all_features, targets, prices_aligned

    def train_models(
        self,
        X_train: np.ndarray,
        targets_train: pd.DataFrame,
        X_test: np.ndarray,
        targets_test: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Train all prediction models."""
        metrics = {}

        # Identify feature indices
        tech_idx = [i for i, f in enumerate(feature_names) if f in self.technical_features]
        sent_idx = [i for i, f in enumerate(feature_names) if f in self.sentiment_features]

        # 1. Technical vol expansion model (using technical features only)
        logger.info("Training technical vol expansion model...")
        y_vol = targets_train['vol_expansion_5'].values
        y_vol_test = targets_test['vol_expansion_5'].values

        X_train_tech = X_train[:, tech_idx]
        X_test_tech = X_test[:, tech_idx]

        self.vol_model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            random_state=42, verbose=-1
        )
        self.vol_model.fit(X_train_tech, y_vol)

        vol_probs = self.vol_model.predict_proba(X_test_tech)[:, 1]
        metrics['tech_vol_auc'] = roc_auc_score(y_vol_test, vol_probs)
        logger.info(f"  Technical vol AUC: {metrics['tech_vol_auc']:.4f}")

        # 2. Sentiment vol expansion model (using sentiment + key technical features)
        logger.info("Training sentiment vol expansion model...")

        # Use sentiment features + a few key technical features
        key_tech = ['rv_10', 'atr_14', 'rsi_14']
        key_tech_idx = [i for i, f in enumerate(feature_names) if f in key_tech]
        combined_idx = sent_idx + key_tech_idx

        X_train_sent = X_train[:, combined_idx]
        X_test_sent = X_test[:, combined_idx]

        self.sentiment_vol_model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            random_state=42, verbose=-1
        )
        self.sentiment_vol_model.fit(X_train_sent, y_vol)

        sent_vol_probs = self.sentiment_vol_model.predict_proba(X_test_sent)[:, 1]
        metrics['sent_vol_auc'] = roc_auc_score(y_vol_test, sent_vol_probs)
        logger.info(f"  Sentiment vol AUC: {metrics['sent_vol_auc']:.4f}")

        # 3. Combined vol expansion (ensemble of both)
        combined_probs = (
            self.config.technical_weight * vol_probs +
            self.config.sentiment_weight * sent_vol_probs
        )
        metrics['combined_vol_auc'] = roc_auc_score(y_vol_test, combined_probs)
        logger.info(f"  Combined vol AUC: {metrics['combined_vol_auc']:.4f}")

        # 4. New high model
        logger.info("Training breakout high model...")
        y_high = targets_train['new_high_10'].values
        y_high_test = targets_test['new_high_10'].values

        self.breakout_high_model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            random_state=42, verbose=-1
        )
        self.breakout_high_model.fit(X_train, y_high)

        high_probs = self.breakout_high_model.predict_proba(X_test)[:, 1]
        metrics['high_auc'] = roc_auc_score(y_high_test, high_probs)
        logger.info(f"  New high AUC: {metrics['high_auc']:.4f}")

        # 5. New low model
        logger.info("Training breakout low model...")
        y_low = targets_train['new_low_10'].values
        y_low_test = targets_test['new_low_10'].values

        self.breakout_low_model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            random_state=42, verbose=-1
        )
        self.breakout_low_model.fit(X_train, y_low)

        low_probs = self.breakout_low_model.predict_proba(X_test)[:, 1]
        metrics['low_auc'] = roc_auc_score(y_low_test, low_probs)
        logger.info(f"  New low AUC: {metrics['low_auc']:.4f}")

        # 6. ATR forecast
        logger.info("Training ATR forecast model...")
        y_atr = targets_train['future_atr_5'].values
        y_atr_test = targets_test['future_atr_5'].values

        self.atr_model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            random_state=42, verbose=-1
        )
        self.atr_model.fit(X_train, y_atr)

        atr_pred = self.atr_model.predict(X_test)
        metrics['atr_r2'] = r2_score(y_atr_test, atr_pred)
        logger.info(f"  ATR forecast R²: {metrics['atr_r2']:.4f}")

        # Store feature indices for prediction
        self._tech_idx = tech_idx
        self._sent_combined_idx = combined_idx

        return metrics

    def generate_signal(
        self,
        features: np.ndarray,
        current_atr: float
    ) -> Tuple[bool, int, int, float, float, Dict]:
        """
        Generate trading signal using ensemble approach.

        Returns:
            (should_trade, direction, contracts, tp_offset, sl_offset, debug_info)
        """
        debug = {}

        # Get technical vol probability
        tech_features = features[self._tech_idx]
        tech_vol_prob = self.vol_model.predict_proba(tech_features.reshape(1, -1))[0, 1]
        debug['tech_vol_prob'] = tech_vol_prob

        # Get sentiment vol probability
        sent_features = features[self._sent_combined_idx]
        sent_vol_prob = self.sentiment_vol_model.predict_proba(sent_features.reshape(1, -1))[0, 1]
        debug['sent_vol_prob'] = sent_vol_prob

        # Apply ensemble method
        if self.config.ensemble_method == 'agreement':
            # Both models must agree on vol expansion
            vol_signal = (
                tech_vol_prob >= self.config.min_vol_expansion_prob and
                sent_vol_prob >= self.config.min_sentiment_vol_prob
            )
            combined_vol_prob = min(tech_vol_prob, sent_vol_prob)

        elif self.config.ensemble_method == 'weighted':
            # Weighted average of probabilities
            combined_vol_prob = (
                self.config.technical_weight * tech_vol_prob +
                self.config.sentiment_weight * sent_vol_prob
            )
            vol_signal = combined_vol_prob >= self.config.min_vol_expansion_prob

        else:  # 'either'
            # Either model predicting expansion is sufficient
            vol_signal = (
                tech_vol_prob >= self.config.min_vol_expansion_prob or
                sent_vol_prob >= self.config.min_sentiment_vol_prob
            )
            combined_vol_prob = max(tech_vol_prob, sent_vol_prob)

        debug['combined_vol_prob'] = combined_vol_prob
        debug['vol_signal'] = vol_signal

        if not vol_signal:
            return False, 0, 0, 0, 0, debug

        # Get breakout probabilities (using all features)
        high_prob = self.breakout_high_model.predict_proba(features.reshape(1, -1))[0, 1]
        low_prob = self.breakout_low_model.predict_proba(features.reshape(1, -1))[0, 1]
        debug['high_prob'] = high_prob
        debug['low_prob'] = low_prob

        # Determine direction
        high_signal = high_prob >= self.config.min_breakout_prob
        low_signal = low_prob >= self.config.min_breakout_prob

        if high_signal and not low_signal:
            direction = 1
            breakout_prob = high_prob
        elif low_signal and not high_signal:
            direction = -1
            breakout_prob = low_prob
        elif high_signal and low_signal:
            if high_prob > low_prob:
                direction = 1
                breakout_prob = high_prob
            else:
                direction = -1
                breakout_prob = low_prob
        else:
            return False, 0, 0, 0, 0, debug

        debug['direction'] = direction
        debug['breakout_prob'] = breakout_prob

        # Position sizing
        predicted_atr = self.atr_model.predict(features.reshape(1, -1))[0]
        vol_factor = np.clip(current_atr / (predicted_atr + 1e-10), 0.5, 2.0)
        contracts = max(1, min(int(self.config.base_contracts * vol_factor), self.config.max_contracts))

        # Dynamic exits
        tp_offset = self.config.tp_atr_mult_base * predicted_atr * direction
        sl_offset = -self.config.sl_atr_mult_base * predicted_atr * direction

        return True, direction, contracts, tp_offset, sl_offset, debug

    def simulate_trade(
        self,
        prices: pd.DataFrame,
        entry_idx: int,
        direction: int,
        contracts: int,
        tp_offset: float,
        sl_offset: float,
        debug_info: Dict
    ) -> Optional[Dict]:
        """Simulate a single trade."""
        if entry_idx >= len(prices) - 1:
            return None

        entry_bar = prices.iloc[entry_idx]
        entry_price = entry_bar['close']
        entry_time = prices.index[entry_idx]

        tp_price = entry_price + tp_offset
        sl_price = entry_price + sl_offset

        exit_idx = entry_idx + 1
        exit_reason = "time"

        for i in range(entry_idx + 1, min(entry_idx + self.config.max_holding_bars + 1, len(prices))):
            bar = prices.iloc[i]

            if direction == 1:
                if bar['high'] >= tp_price:
                    exit_idx = i
                    exit_reason = "tp"
                    break
                elif bar['low'] <= sl_price:
                    exit_idx = i
                    exit_reason = "sl"
                    break
            else:
                if bar['low'] <= tp_price:
                    exit_idx = i
                    exit_reason = "tp"
                    break
                elif bar['high'] >= sl_price:
                    exit_idx = i
                    exit_reason = "sl"
                    break
            exit_idx = i

        exit_bar = prices.iloc[exit_idx]
        exit_time = prices.index[exit_idx]

        if exit_reason == "tp":
            exit_price = tp_price
        elif exit_reason == "sl":
            exit_price = sl_price
        else:
            exit_price = exit_bar['close']

        price_diff = (exit_price - entry_price) * direction
        gross_pnl = price_diff * self.config.point_value * contracts

        commission = self.config.commission_per_side * 2 * contracts
        slippage = self.config.slippage_ticks * self.config.tick_size * self.config.point_value * 2 * contracts
        net_pnl = gross_pnl - commission - slippage

        return {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'contracts': contracts,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'gross_pnl': gross_pnl,
            'commission': commission,
            'slippage': slippage,
            'net_pnl': net_pnl,
            'exit_reason': exit_reason,
            'bars_held': exit_idx - entry_idx,
            'tech_vol_prob': debug_info.get('tech_vol_prob', 0),
            'sent_vol_prob': debug_info.get('sent_vol_prob', 0),
            'breakout_prob': debug_info.get('breakout_prob', 0)
        }

    def calculate_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate backtest metrics."""
        if not trades:
            return {}

        trades_df = pd.DataFrame(trades)

        total = len(trades_df)
        winners = trades_df[trades_df['net_pnl'] > 0]
        losers = trades_df[trades_df['net_pnl'] <= 0]

        net_pnl = trades_df['net_pnl'].sum()
        win_rate = len(winners) / total

        avg_win = winners['net_pnl'].mean() if len(winners) > 0 else 0
        avg_loss = losers['net_pnl'].mean() if len(losers) > 0 else 0

        gross_profit = winners['net_pnl'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['net_pnl'].sum()) if len(losers) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Drawdown
        cumulative = trades_df['net_pnl'].cumsum()
        running_max = cumulative.expanding().max()
        max_drawdown = (running_max - cumulative).max()

        # Sharpe
        trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date
        daily_pnl = trades_df.groupby('date')['net_pnl'].sum()
        sharpe = (daily_pnl.mean() / (daily_pnl.std() + 1e-8)) * np.sqrt(252) if len(daily_pnl) > 1 else 0

        return {
            'total_trades': total,
            'net_pnl': net_pnl,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'avg_bars_held': trades_df['bars_held'].mean(),
            'long_trades': len(trades_df[trades_df['direction'] == 'LONG']),
            'short_trades': len(trades_df[trades_df['direction'] == 'SHORT'])
        }


def run_ensemble_backtest(method: str = 'agreement', verbose: bool = True):
    """Run walk-forward backtest of ensemble strategy.

    Args:
        method: Ensemble method ('agreement', 'weighted', 'either')
        verbose: Whether to print detailed output
    """
    # Suppress logging if not verbose
    if not verbose:
        logging.getLogger().setLevel(logging.WARNING)

    if verbose:
        print("=" * 80)
        print(" ENSEMBLE STRATEGY BACKTEST")
        print(" Volatility Breakout + Sentiment")
        print("=" * 80)

    from data_collection.ninjatrader_loader import load_sample_data

    # Load data
    if verbose:
        logger.info("\n--- Loading Data ---")
    prices, _ = load_sample_data(source="databento")
    if verbose:
        logger.info(f"Loaded {len(prices)} bars")

    # Filter RTH and resample
    if hasattr(prices.index, 'hour'):
        prices = prices[
            (prices.index.hour >= 9) &
            ((prices.index.hour < 16) | ((prices.index.hour == 9) & (prices.index.minute >= 30)))
        ]

    prices = prices.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    if verbose:
        logger.info(f"After RTH + 5-min resample: {len(prices)} bars")

    # Initialize strategy with specified method
    config = EnsembleConfig(ensemble_method=method)
    strategy = EnsembleStrategy(config)

    # Prepare data
    logger.info("\n--- Preparing Data ---")
    features, targets, prices_aligned = strategy.prepare_data(prices)

    # Walk-forward backtest
    logger.info("\n--- Walk-Forward Backtest ---")

    bars_per_day = 78
    train_bars = config.train_days * bars_per_day
    test_bars = config.test_days * bars_per_day

    all_trades = []
    all_metrics = []

    fold = 0
    start_idx = 0

    while start_idx + train_bars + config.embargo_bars + test_bars <= len(features):
        fold += 1

        train_end = start_idx + train_bars
        test_start = train_end + config.embargo_bars
        test_end = test_start + test_bars

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features.iloc[start_idx:train_end].values)
        X_test = scaler.transform(features.iloc[test_start:test_end].values)

        targets_train = targets.iloc[start_idx:train_end]
        targets_test = targets.iloc[test_start:test_end]

        # Train models
        metrics = strategy.train_models(
            X_train, targets_train,
            X_test, targets_test,
            list(features.columns)
        )

        # Simulate trades
        test_prices = prices_aligned.iloc[test_start:test_end]
        current_atr = features.iloc[test_start:test_end]['atr_14'].values

        for i in range(len(X_test)):
            should_trade, direction, contracts, tp_offset, sl_offset, debug = \
                strategy.generate_signal(X_test[i], current_atr[i])

            if should_trade:
                trade = strategy.simulate_trade(
                    test_prices, i, direction, contracts,
                    tp_offset, sl_offset, debug
                )
                if trade:
                    all_trades.append(trade)

        all_metrics.append(metrics)

        if fold % 5 == 0 or fold == 1:
            logger.info(f"  Fold {fold}: Tech Vol={metrics['tech_vol_auc']:.3f}, "
                       f"Sent Vol={metrics['sent_vol_auc']:.3f}, "
                       f"Combined={metrics['combined_vol_auc']:.3f}, "
                       f"Trades: {len(all_trades)}")

        start_idx += test_bars

    # Calculate final metrics
    if verbose:
        logger.info("\n--- Final Results ---")
    results = strategy.calculate_metrics(all_trades)

    # Average model metrics
    avg_tech_vol = np.mean([m['tech_vol_auc'] for m in all_metrics])
    avg_sent_vol = np.mean([m['sent_vol_auc'] for m in all_metrics])
    avg_combined = np.mean([m['combined_vol_auc'] for m in all_metrics])
    avg_breakout = np.mean([(m['high_auc'] + m['low_auc']) / 2 for m in all_metrics])

    # Add normalized keys for comparison function
    results['trades'] = results.get('total_trades', 0)
    results['win_rate'] = results.get('win_rate', 0) * 100  # Convert to percentage
    results['sharpe'] = results.get('sharpe_ratio', 0)

    if verbose:
        print("\n" + "=" * 80)
        print(" ENSEMBLE BACKTEST RESULTS")
        print("=" * 80)

        print(f"\nEnsemble Method: {config.ensemble_method}")
        print(f"Tech Weight: {config.technical_weight}, Sent Weight: {config.sentiment_weight}")

        print(f"\n--- Model Performance ---")
        print(f"  Technical Vol AUC:  {avg_tech_vol:.4f}")
        print(f"  Sentiment Vol AUC:  {avg_sent_vol:.4f}")
        print(f"  Combined Vol AUC:   {avg_combined:.4f}")
        print(f"  Breakout AUC:       {avg_breakout:.4f}")

        print(f"\n--- Trade Statistics ---")
        print(f"  Total Trades:       {results.get('total_trades', 0)}")
        print(f"  Win Rate:           {results.get('win_rate', 0):.1f}%")
        print(f"  Avg Bars Held:      {results.get('avg_bars_held', 0):.1f}")

        print(f"\n--- P&L ---")
        print(f"  Net P&L:            ${results.get('net_pnl', 0):,.2f}")
        print(f"  Profit Factor:      {results.get('profit_factor', 0):.2f}")
        print(f"  Max Drawdown:       ${results.get('max_drawdown', 0):,.2f}")
        print(f"  Sharpe Ratio:       {results.get('sharpe', 0):.2f}")

        print(f"\n--- Comparison to Baseline ---")
        print(f"  Vol Breakout Baseline: +$209,351 (in-sample)")
        print(f"  Ensemble Result:       ${results.get('net_pnl', 0):,.2f}")

        improvement = results.get('net_pnl', 0) - 209351
        print(f"  Difference:            ${improvement:,.2f} ({improvement/209351*100:.1f}%)")

    # Save results (only when running standalone with verbose)
    if verbose:
        output_dir = project_root / 'data' / 'backtest_results'
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv(output_dir / f'ensemble_trades_{timestamp}.csv', index=False)
        logger.info(f"\nTrades saved to: ensemble_trades_{timestamp}.csv")

    return results, all_metrics


def compare_ensemble_methods():
    """Compare all ensemble methods by running three backtests."""
    print("=" * 80)
    print(" ENSEMBLE METHOD COMPARISON")
    print(" Testing: agreement, weighted, either")
    print("=" * 80)

    methods = ['agreement', 'weighted', 'either']
    results_comparison = []

    for method in methods:
        print(f"\n{'=' * 60}")
        print(f"Testing method: {method.upper()}")
        print(f"{'=' * 60}")

        # Run full backtest for this method
        results, metrics = run_ensemble_backtest(method=method, verbose=False)

        results_comparison.append({
            'method': method,
            'net_pnl': results['net_pnl'],
            'trades': results['trades'],
            'win_rate': results['win_rate'],
            'profit_factor': results['profit_factor'],
            'sharpe': results['sharpe'],
        })

        print(f"\n{method.upper()} Results:")
        print(f"  Net P&L:       ${results['net_pnl']:,.2f}")
        print(f"  Trades:        {results['trades']}")
        print(f"  Win Rate:      {results['win_rate']:.1f}%")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print(f"  Sharpe:        {results['sharpe']:.2f}")

    # Print comparison table
    print("\n" + "=" * 80)
    print(" ENSEMBLE METHOD COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<12} {'Net P&L':>15} {'Trades':>8} {'Win%':>8} {'PF':>8} {'Sharpe':>8}")
    print("-" * 60)

    baseline_pnl = 209351
    for r in results_comparison:
        print(f"{r['method']:<12} ${r['net_pnl']:>13,.0f} {r['trades']:>8} "
              f"{r['win_rate']:>7.1f}% {r['profit_factor']:>7.2f} {r['sharpe']:>8.2f}")

    print("-" * 60)
    print(f"{'Baseline':<12} ${baseline_pnl:>13,.0f} {'4,560':>8} {'39.9':>7}% {'1.29':>7} {'3.22':>8}")

    # Find best method
    best = max(results_comparison, key=lambda x: x['net_pnl'])
    print(f"\nBest Method: {best['method'].upper()}")
    improvement = best['net_pnl'] - baseline_pnl
    print(f"vs Baseline: ${improvement:+,.0f} ({improvement/baseline_pnl*100:+.1f}%)")

    return results_comparison


if __name__ == "__main__":
    compare_ensemble_methods()  # Run comparison of all methods
    # run_ensemble_backtest()  # Run single method (default: agreement)
