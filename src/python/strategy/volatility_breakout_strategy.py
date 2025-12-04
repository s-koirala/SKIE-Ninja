"""
Volatility Breakout Strategy

A strategy based on the Session 8 breakthrough findings:
- Volatility expansion is highly predictable (AUC 0.84)
- New highs/lows are predictable (AUC 0.72)
- Direction alone is NOT predictable (AUC 0.50)

Strategy Logic:
1. FILTER: Only trade when vol_expansion is predicted (AUC 0.84)
2. DIRECTION: Use new_high/new_low prediction for direction (AUC 0.72)
3. SIZING: Inverse volatility sizing (smaller in high vol)
4. EXITS: Dynamic TP/SL based on predicted ATR (R² 0.36)

This approach predicts WHEN and WHERE, not IF (direction).

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score
import lightgbm as lgb

from feature_engineering.multi_target_labels import (
    MultiTargetLabeler, MultiTargetConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for volatility breakout strategy."""

    # Entry filters (lowered from initial targets to account for WF variance)
    min_vol_expansion_prob: float = 0.50  # Min probability for vol expansion
    min_breakout_prob: float = 0.50      # Min probability for new high/low

    # Position sizing
    base_contracts: int = 1
    max_contracts: int = 3
    vol_sizing_factor: float = 1.0       # Inverse vol sizing strength

    # Dynamic exits
    tp_atr_mult_base: float = 2.0        # Base TP in ATR multiples
    sl_atr_mult_base: float = 1.0        # Base SL in ATR multiples
    tp_adjustment_factor: float = 0.25   # Adjust TP based on reach probability
    max_holding_bars: int = 20           # Maximum bars to hold

    # Trading costs (REALISTIC - NinjaTrader research)
    commission_per_side: float = 1.29    # NinjaTrader official rate
    slippage_ticks: float = 0.5          # Conservative RTH slippage
    tick_size: float = 0.25              # ES tick size
    point_value: float = 50.0            # ES point value

    # Walk-forward settings
    train_days: int = 60  # Reduced to allow more folds with available data
    test_days: int = 5
    embargo_bars: int = 20  # Reduced embargo

    # QC thresholds
    min_vol_auc: float = 0.60            # Minimum vol model AUC
    min_breakout_auc: float = 0.55       # Minimum breakout model AUC


@dataclass
class TradeResult:
    """Single trade result."""
    entry_time: datetime
    exit_time: datetime
    direction: str
    contracts: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    vol_prob: float
    breakout_prob: float
    predicted_atr: float
    bars_held: int
    exit_reason: str


@dataclass
class BacktestResults:
    """Comprehensive backtest results."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    vol_model_auc: float = 0.0
    breakout_model_auc: float = 0.0
    atr_model_r2: float = 0.0
    avg_bars_held: float = 0.0


class VolatilityBreakoutStrategy:
    """
    Volatility-based breakout strategy.

    Uses multiple predictable targets:
    1. vol_expansion_5 (AUC 0.84) - Entry filter
    2. new_high_10/new_low_10 (AUC 0.72) - Direction
    3. future_atr_5 (R² 0.36) - Dynamic exits
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.vol_model = None
        self.breakout_high_model = None
        self.breakout_low_model = None
        self.atr_model = None
        self.scaler = None
        self.target_labeler = MultiTargetLabeler()

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features from price data."""
        features = pd.DataFrame(index=df.index)

        # Returns (lagged)
        for lag in [1, 2, 3, 5, 10, 20]:
            features[f'return_lag{lag}'] = df['close'].pct_change(lag)

        # Volatility features
        # Calculate ATR first (used in features and for current_atr)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        for period in [5, 10, 14, 20]:  # Added 14 for current_atr lookup
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
            features[f'range_pct_{period}'] = (
                df['high'].rolling(period).max() - df['low'].rolling(period).min()
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
        for period in [20]:
            mid = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            upper = mid + 2 * std
            lower = mid - 2 * std
            features[f'bb_pct_{period}'] = (df['close'] - lower) / (upper - lower + 1e-10)

        # Volume features (if available)
        if 'volume' in df.columns:
            for period in [5, 10, 20]:
                features[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
                features[f'volume_ratio_{period}'] = df['volume'] / (
                    features[f'volume_sma_{period}'] + 1
                )

        return features

    def train_models(
        self,
        X_train: np.ndarray,
        targets_train: pd.DataFrame,
        X_test: np.ndarray,
        targets_test: pd.DataFrame
    ) -> Dict[str, float]:
        """Train all prediction models."""
        metrics = {}

        # 1. Volatility expansion model (classification)
        logger.info("Training volatility expansion model...")
        y_vol_train = targets_train['vol_expansion_5'].values
        y_vol_test = targets_test['vol_expansion_5'].values

        self.vol_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        self.vol_model.fit(X_train, y_vol_train)

        vol_probs_test = self.vol_model.predict_proba(X_test)[:, 1]
        metrics['vol_auc'] = roc_auc_score(y_vol_test, vol_probs_test)
        logger.info(f"  Vol expansion AUC: {metrics['vol_auc']:.4f}")

        # 2. New high model (classification)
        logger.info("Training new high model...")
        y_high_train = targets_train['new_high_10'].values
        y_high_test = targets_test['new_high_10'].values

        self.breakout_high_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        self.breakout_high_model.fit(X_train, y_high_train)

        high_probs_test = self.breakout_high_model.predict_proba(X_test)[:, 1]
        metrics['high_auc'] = roc_auc_score(y_high_test, high_probs_test)
        logger.info(f"  New high AUC: {metrics['high_auc']:.4f}")

        # 3. New low model (classification)
        logger.info("Training new low model...")
        y_low_train = targets_train['new_low_10'].values
        y_low_test = targets_test['new_low_10'].values

        self.breakout_low_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        self.breakout_low_model.fit(X_train, y_low_train)

        low_probs_test = self.breakout_low_model.predict_proba(X_test)[:, 1]
        metrics['low_auc'] = roc_auc_score(y_low_test, low_probs_test)
        logger.info(f"  New low AUC: {metrics['low_auc']:.4f}")

        # 4. ATR forecast model (regression)
        logger.info("Training ATR forecast model...")
        y_atr_train = targets_train['future_atr_5'].values
        y_atr_test = targets_test['future_atr_5'].values

        self.atr_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        self.atr_model.fit(X_train, y_atr_train)

        atr_pred_test = self.atr_model.predict(X_test)
        metrics['atr_r2'] = r2_score(y_atr_test, atr_pred_test)
        logger.info(f"  ATR forecast R²: {metrics['atr_r2']:.4f}")

        return metrics

    def generate_signal(
        self,
        features: np.ndarray,
        current_atr: float
    ) -> Tuple[bool, int, int, float, float, float]:
        """
        Generate trading signal based on predictions.

        Returns:
            (should_trade, direction, contracts, tp_price_offset, sl_price_offset, vol_prob)
        """
        # Get predictions
        vol_prob = self.vol_model.predict_proba(features.reshape(1, -1))[0, 1]
        high_prob = self.breakout_high_model.predict_proba(features.reshape(1, -1))[0, 1]
        low_prob = self.breakout_low_model.predict_proba(features.reshape(1, -1))[0, 1]
        predicted_atr = self.atr_model.predict(features.reshape(1, -1))[0]

        # 1. Volatility filter - must expect vol expansion
        if vol_prob < self.config.min_vol_expansion_prob:
            return False, 0, 0, 0, 0, vol_prob

        # 2. Direction from breakout probabilities
        high_signal = high_prob >= self.config.min_breakout_prob
        low_signal = low_prob >= self.config.min_breakout_prob

        if high_signal and not low_signal:
            direction = 1  # LONG
            breakout_prob = high_prob
        elif low_signal and not high_signal:
            direction = -1  # SHORT
            breakout_prob = low_prob
        elif high_signal and low_signal:
            # Both signals - take stronger one
            if high_prob > low_prob:
                direction = 1
                breakout_prob = high_prob
            else:
                direction = -1
                breakout_prob = low_prob
        else:
            # No clear direction
            return False, 0, 0, 0, 0, vol_prob

        # 3. Position sizing (inverse volatility)
        vol_factor = current_atr / (predicted_atr + 1e-10)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)

        contracts = int(
            self.config.base_contracts *
            vol_factor *
            self.config.vol_sizing_factor
        )
        contracts = max(1, min(contracts, self.config.max_contracts))

        # 4. Dynamic exits based on predicted ATR
        # Adjust TP based on breakout probability
        tp_mult = self.config.tp_atr_mult_base * (
            1 + self.config.tp_adjustment_factor * (breakout_prob - 0.5)
        )
        sl_mult = self.config.sl_atr_mult_base

        tp_offset = tp_mult * predicted_atr * direction
        sl_offset = -sl_mult * predicted_atr * direction

        return True, direction, contracts, tp_offset, sl_offset, vol_prob

    def simulate_trade(
        self,
        prices: pd.DataFrame,
        entry_idx: int,
        direction: int,
        contracts: int,
        tp_offset: float,
        sl_offset: float,
        vol_prob: float,
        breakout_prob: float,
        predicted_atr: float
    ) -> Optional[TradeResult]:
        """Simulate a single trade with TP/SL."""
        if entry_idx >= len(prices) - 1:
            return None

        entry_bar = prices.iloc[entry_idx]
        entry_price = entry_bar['close']
        entry_time = prices.index[entry_idx]

        tp_price = entry_price + tp_offset
        sl_price = entry_price + sl_offset

        # Simulate trade execution
        exit_idx = entry_idx + 1
        exit_reason = "time"

        for i in range(entry_idx + 1, min(entry_idx + self.config.max_holding_bars + 1, len(prices))):
            bar = prices.iloc[i]

            if direction == 1:  # LONG
                if bar['high'] >= tp_price:
                    exit_idx = i
                    exit_reason = "tp"
                    break
                elif bar['low'] <= sl_price:
                    exit_idx = i
                    exit_reason = "sl"
                    break
            else:  # SHORT
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

        # Determine exit price
        if exit_reason == "tp":
            exit_price = tp_price
        elif exit_reason == "sl":
            exit_price = sl_price
        else:
            exit_price = exit_bar['close']

        # Calculate P&L
        price_diff = (exit_price - entry_price) * direction
        gross_pnl = price_diff * self.config.point_value * contracts

        commission = self.config.commission_per_side * 2 * contracts
        slippage = (
            self.config.slippage_ticks *
            self.config.tick_size *
            self.config.point_value * 2 * contracts
        )

        net_pnl = gross_pnl - commission - slippage

        return TradeResult(
            entry_time=entry_time,
            exit_time=exit_time,
            direction="LONG" if direction == 1 else "SHORT",
            contracts=contracts,
            entry_price=entry_price,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            commission=commission,
            slippage=slippage,
            net_pnl=net_pnl,
            vol_prob=vol_prob,
            breakout_prob=breakout_prob,
            predicted_atr=predicted_atr,
            bars_held=exit_idx - entry_idx,
            exit_reason=exit_reason
        )

    def calculate_metrics(self, trades: List[TradeResult]) -> BacktestResults:
        """Calculate backtest metrics."""
        results = BacktestResults()

        if not trades:
            return results

        results.total_trades = len(trades)
        results.winning_trades = sum(1 for t in trades if t.net_pnl > 0)
        results.losing_trades = sum(1 for t in trades if t.net_pnl < 0)

        results.gross_pnl = sum(t.gross_pnl for t in trades)
        results.net_pnl = sum(t.net_pnl for t in trades)
        results.total_commission = sum(t.commission for t in trades)
        results.total_slippage = sum(t.slippage for t in trades)

        results.win_rate = results.winning_trades / results.total_trades

        winners = [t.net_pnl for t in trades if t.net_pnl > 0]
        losers = [t.net_pnl for t in trades if t.net_pnl < 0]

        results.avg_win = np.mean(winners) if winners else 0
        results.avg_loss = np.mean(losers) if losers else 0

        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 1
        results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Drawdown
        cumulative = np.cumsum([t.net_pnl for t in trades])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        results.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Sharpe (using daily returns)
        trades_df = pd.DataFrame([{
            'date': t.entry_time.date() if hasattr(t.entry_time, 'date') else t.entry_time,
            'pnl': t.net_pnl
        } for t in trades])

        if len(trades_df) > 0:
            daily_pnl = trades_df.groupby('date')['pnl'].sum()
            if len(daily_pnl) > 1:
                avg_daily = daily_pnl.mean()
                std_daily = daily_pnl.std()
                results.sharpe_ratio = (avg_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0

        results.avg_bars_held = np.mean([t.bars_held for t in trades])

        return results


def run_backtest():
    """Run full walk-forward backtest."""
    print("=" * 80)
    print(" VOLATILITY BREAKOUT STRATEGY BACKTEST")
    print(" Using Multi-Target Predictions (Vol AUC 0.84, Breakout AUC 0.72)")
    print("=" * 80)

    # Import data loader
    from data_collection.ninjatrader_loader import load_sample_data

    # Load data
    logger.info("\n--- Loading Data ---")
    prices, _ = load_sample_data(source="databento")
    logger.info(f"Loaded {len(prices)} bars")

    # Filter RTH
    if hasattr(prices.index, 'hour'):
        prices = prices[
            (prices.index.hour >= 9) &
            ((prices.index.hour < 16) | ((prices.index.hour == 9) & (prices.index.minute >= 30)))
        ]

    # Resample to 5-min
    prices = prices.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    logger.info(f"After RTH + 5-min resample: {len(prices)} bars")

    # Initialize strategy
    config = StrategyConfig()
    strategy = VolatilityBreakoutStrategy(config)

    # Generate features and targets
    logger.info("\n--- Generating Features & Targets ---")
    features = strategy.generate_features(prices)
    targets = strategy.target_labeler.generate_all_targets(prices)

    # Align data
    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]
    prices_aligned = prices.loc[common_idx]

    # Remove NaN
    valid_mask = ~(features.isna().any(axis=1) | targets.isna().any(axis=1))
    features = features[valid_mask]
    targets = targets[valid_mask]
    prices_aligned = prices_aligned[valid_mask]

    logger.info(f"Valid samples: {len(features)}")

    # Walk-forward backtest
    logger.info("\n--- Walk-Forward Backtest ---")

    bars_per_day = 78  # 5-min RTH
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

        # Get data slices
        X_train = features.iloc[start_idx:train_end].values
        targets_train = targets.iloc[start_idx:train_end]
        X_test = features.iloc[test_start:test_end].values
        targets_test = targets.iloc[test_start:test_end]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train models
        metrics = strategy.train_models(
            X_train_scaled, targets_train,
            X_test_scaled, targets_test
        )

        strategy.scaler = scaler

        # Simulate trades on test set
        test_prices = prices_aligned.iloc[test_start:test_end]
        current_atr = features.iloc[test_start:test_end]['atr_14'].values

        for i in range(len(X_test_scaled)):
            should_trade, direction, contracts, tp_offset, sl_offset, vol_prob = \
                strategy.generate_signal(X_test_scaled[i], current_atr[i])

            if should_trade:
                # Get breakout probability for logging
                if direction == 1:
                    breakout_prob = strategy.breakout_high_model.predict_proba(
                        X_test_scaled[i].reshape(1, -1)
                    )[0, 1]
                else:
                    breakout_prob = strategy.breakout_low_model.predict_proba(
                        X_test_scaled[i].reshape(1, -1)
                    )[0, 1]

                predicted_atr = strategy.atr_model.predict(X_test_scaled[i].reshape(1, -1))[0]

                trade = strategy.simulate_trade(
                    test_prices, i, direction, contracts,
                    tp_offset, sl_offset, vol_prob, breakout_prob, predicted_atr
                )

                if trade:
                    all_trades.append(trade)

        all_metrics.append(metrics)

        if fold % 5 == 0 or fold == 1:
            logger.info(f"  Fold {fold}: Vol AUC={metrics['vol_auc']:.4f}, "
                       f"High AUC={metrics['high_auc']:.4f}, "
                       f"Low AUC={metrics['low_auc']:.4f}, "
                       f"Trades so far: {len(all_trades)}")

        start_idx += test_bars

    # Calculate final metrics
    logger.info("\n--- Final Results ---")
    results = strategy.calculate_metrics(all_trades)

    # Average model metrics
    results.vol_model_auc = np.mean([m['vol_auc'] for m in all_metrics])
    results.breakout_model_auc = np.mean([
        (m['high_auc'] + m['low_auc']) / 2 for m in all_metrics
    ])
    results.atr_model_r2 = np.mean([m['atr_r2'] for m in all_metrics])

    # Print results
    print("\n" + "=" * 80)
    print(" BACKTEST RESULTS")
    print("=" * 80)

    print(f"\nStrategy: Volatility Breakout")
    print(f"Config: Vol prob > {config.min_vol_expansion_prob}, "
          f"Breakout prob > {config.min_breakout_prob}")
    print(f"Costs: ${config.commission_per_side}/side commission, "
          f"{config.slippage_ticks} tick slippage")

    print(f"\n--- Model Performance ---")
    print(f"  Vol Expansion AUC:  {results.vol_model_auc:.4f} (target: 0.84)")
    print(f"  Breakout AUC:       {results.breakout_model_auc:.4f} (target: 0.72)")
    print(f"  ATR Forecast R²:    {results.atr_model_r2:.4f} (target: 0.36)")

    print(f"\n--- Trade Statistics ---")
    print(f"  Total Trades:       {results.total_trades}")
    print(f"  Win Rate:           {results.win_rate*100:.1f}%")
    print(f"  Avg Bars Held:      {results.avg_bars_held:.1f}")

    print(f"\n--- P&L ---")
    print(f"  Gross P&L:          ${results.gross_pnl:,.2f}")
    print(f"  Commission:         ${results.total_commission:,.2f}")
    print(f"  Slippage:           ${results.total_slippage:,.2f}")
    print(f"  Net P&L:            ${results.net_pnl:,.2f}")

    print(f"\n--- Risk Metrics ---")
    print(f"  Profit Factor:      {results.profit_factor:.2f}")
    print(f"  Max Drawdown:       ${results.max_drawdown:,.2f}")
    print(f"  Sharpe Ratio:       {results.sharpe_ratio:.2f}")

    print(f"\n--- Win/Loss ---")
    print(f"  Avg Win:            ${results.avg_win:,.2f}")
    print(f"  Avg Loss:           ${results.avg_loss:,.2f}")
    print(f"  Payoff Ratio:       {abs(results.avg_win/results.avg_loss) if results.avg_loss != 0 else 0:.2f}")

    # Save results
    output_dir = project_root / 'data' / 'backtest_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save trades
    trades_df = pd.DataFrame([{
        'entry_time': t.entry_time,
        'exit_time': t.exit_time,
        'direction': t.direction,
        'contracts': t.contracts,
        'entry_price': t.entry_price,
        'exit_price': t.exit_price,
        'gross_pnl': t.gross_pnl,
        'commission': t.commission,
        'slippage': t.slippage,
        'net_pnl': t.net_pnl,
        'vol_prob': t.vol_prob,
        'breakout_prob': t.breakout_prob,
        'predicted_atr': t.predicted_atr,
        'bars_held': t.bars_held,
        'exit_reason': t.exit_reason
    } for t in all_trades])

    trades_df.to_csv(output_dir / f'vol_breakout_trades_{timestamp}.csv', index=False)
    logger.info(f"\nTrades saved to: {output_dir / f'vol_breakout_trades_{timestamp}.csv'}")

    return results


if __name__ == "__main__":
    run_backtest()
