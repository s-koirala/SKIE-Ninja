"""
SKIE-Ninja Comprehensive Walk-Forward Backtesting System

Full-featured backtesting with:
- Walk-forward validation (train on past, trade on future)
- Comprehensive trade metrics (P&L, drawdown, time, contracts)
- RTH-only trading enforcement
- Data leakage prevention and detection
- Detailed reporting with all KPIs

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
import json
import warnings
from enum import Enum

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# ES Futures Contract Specifications
ES_POINT_VALUE = 50.0       # $50 per point
ES_TICK_SIZE = 0.25         # Minimum price movement
ES_TICK_VALUE = 12.50       # $12.50 per tick
ES_MARGIN_INTRADAY = 500    # Typical intraday margin per contract

# Regular Trading Hours (RTH) - Eastern Time
RTH_START = time(9, 30)     # 9:30 AM ET
RTH_END = time(16, 0)       # 4:00 PM ET


class TradeDirection(Enum):
    LONG = 1
    SHORT = -1


class ExitReason(Enum):
    HOLD_BARS = "hold_bars"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    END_OF_DAY = "end_of_day"
    SIGNAL_REVERSAL = "signal_reversal"


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class BacktestConfig:
    """Comprehensive backtest configuration."""

    # Position Sizing
    contracts_per_trade: int = 1
    max_contracts: int = 5
    scale_with_equity: bool = False

    # Entry Rules
    long_threshold: float = 0.55    # Probability > 0.55 for long
    short_threshold: float = 0.45   # Probability < 0.45 for short
    min_signal_strength: float = 0.1  # Minimum |prob - 0.5|

    # Exit Rules
    hold_bars: int = 3              # Bars to hold position
    stop_loss_ticks: int = 20       # 20 ticks = 5 points = $250/contract
    take_profit_ticks: int = 40     # 40 ticks = 10 points = $500/contract
    trailing_stop_ticks: Optional[int] = None

    # Risk Management
    max_daily_trades: int = 10
    max_daily_loss: float = 1000.0  # Stop trading if daily loss exceeds
    max_drawdown_pct: float = 0.10  # Stop if drawdown exceeds 10%
    max_consecutive_losses: int = 5

    # Costs
    commission_per_side: float = 2.50   # Per contract, per side
    slippage_ticks: float = 0.5         # Expected slippage

    # Walk-Forward Parameters
    train_days: int = 180
    test_days: int = 5
    embargo_bars: int = 42      # ~3.5 hours for 5-min bars
    bars_per_day: int = 78      # 5-min RTH bars

    # RTH Enforcement
    rth_only: bool = True
    rth_start: time = RTH_START
    rth_end: time = RTH_END

    # Contract Specs
    point_value: float = ES_POINT_VALUE
    tick_size: float = ES_TICK_SIZE

    # Data Leakage
    check_leakage: bool = True
    leakage_correlation_threshold: float = 0.95


# ============================================================================
# TRADE RECORD
# ============================================================================

@dataclass
class Trade:
    """Detailed trade record."""
    # Identification
    trade_id: int
    model_name: str
    fold: int

    # Timing
    entry_time: datetime
    exit_time: datetime
    entry_bar: int
    exit_bar: int
    bars_held: int = 0
    time_held_minutes: float = 0.0

    # Prices
    entry_price: float = 0.0
    exit_price: float = 0.0
    high_price_during: float = 0.0
    low_price_during: float = 0.0

    # Position
    direction: TradeDirection = TradeDirection.LONG
    contracts: int = 1

    # P&L
    gross_pnl: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0
    net_pnl: float = 0.0
    pnl_per_contract: float = 0.0

    # Excursions
    max_favorable_excursion: float = 0.0    # Best unrealized P&L
    max_adverse_excursion: float = 0.0      # Worst unrealized P&L
    mfe_ticks: int = 0
    mae_ticks: int = 0

    # Signal Info
    entry_signal: float = 0.0
    exit_reason: ExitReason = ExitReason.HOLD_BARS

    # Running Totals (filled during backtest)
    cumulative_pnl: float = 0.0
    drawdown_at_entry: float = 0.0
    equity_at_entry: float = 0.0

    def calculate_metrics(
        self,
        prices: pd.DataFrame,
        point_value: float,
        tick_size: float
    ):
        """Calculate MFE, MAE, and P&L metrics from price data during trade."""
        if prices.empty:
            return

        self.bars_held = len(prices)

        if hasattr(self.entry_time, 'timestamp') and hasattr(self.exit_time, 'timestamp'):
            self.time_held_minutes = (self.exit_time - self.entry_time).total_seconds() / 60

        # High/Low during trade
        self.high_price_during = prices['high'].max() if 'high' in prices.columns else prices['close'].max()
        self.low_price_during = prices['low'].min() if 'low' in prices.columns else prices['close'].min()

        # Calculate MFE/MAE
        if self.direction == TradeDirection.LONG:
            self.max_favorable_excursion = (self.high_price_during - self.entry_price) * point_value * self.contracts
            self.max_adverse_excursion = (self.entry_price - self.low_price_during) * point_value * self.contracts
            self.mfe_ticks = int((self.high_price_during - self.entry_price) / tick_size)
            self.mae_ticks = int((self.entry_price - self.low_price_during) / tick_size)
        else:
            self.max_favorable_excursion = (self.entry_price - self.low_price_during) * point_value * self.contracts
            self.max_adverse_excursion = (self.high_price_during - self.entry_price) * point_value * self.contracts
            self.mfe_ticks = int((self.entry_price - self.low_price_during) / tick_size)
            self.mae_ticks = int((self.high_price_during - self.entry_price) / tick_size)

        # Calculate P&L
        price_diff = (self.exit_price - self.entry_price) * self.direction.value
        self.gross_pnl = price_diff * point_value * self.contracts
        self.net_pnl = self.gross_pnl - self.commission - self.slippage_cost
        self.pnl_per_contract = self.net_pnl / self.contracts if self.contracts > 0 else 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame/JSON."""
        return {
            'trade_id': self.trade_id,
            'model': self.model_name,
            'fold': self.fold,
            'entry_time': str(self.entry_time),
            'exit_time': str(self.exit_time),
            'bars_held': self.bars_held,
            'time_held_min': round(self.time_held_minutes, 1),
            'direction': self.direction.name,
            'contracts': self.contracts,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'gross_pnl': round(self.gross_pnl, 2),
            'commission': round(self.commission, 2),
            'slippage': round(self.slippage_cost, 2),
            'net_pnl': round(self.net_pnl, 2),
            'pnl_per_contract': round(self.pnl_per_contract, 2),
            'mfe': round(self.max_favorable_excursion, 2),
            'mae': round(self.max_adverse_excursion, 2),
            'mfe_ticks': self.mfe_ticks,
            'mae_ticks': self.mae_ticks,
            'entry_signal': round(self.entry_signal, 4),
            'exit_reason': self.exit_reason.value,
            'cumulative_pnl': round(self.cumulative_pnl, 2),
            'high_during': self.high_price_during,
            'low_during': self.low_price_during
        }


# ============================================================================
# COMPREHENSIVE METRICS
# ============================================================================

@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""

    # === Trade Counts ===
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    long_trades: int = 0
    short_trades: int = 0
    long_winners: int = 0
    short_winners: int = 0

    # === P&L Metrics ===
    total_gross_pnl: float = 0.0
    total_net_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0

    # === Win/Loss Statistics ===
    win_rate: float = 0.0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0

    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0

    avg_win_ticks: float = 0.0
    avg_loss_ticks: float = 0.0

    # === Key Performance Indicators (KPIs) ===
    profit_factor: float = 0.0          # Gross profit / Gross loss
    payoff_ratio: float = 0.0           # Avg win / Avg loss
    expectancy: float = 0.0             # Average P&L per trade
    expectancy_per_contract: float = 0.0

    # === Drawdown Metrics ===
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_bars: int = 0
    max_drawdown_duration_days: float = 0.0
    avg_drawdown: float = 0.0
    max_runup: float = 0.0

    # === MFE/MAE Analysis ===
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    avg_mfe_ticks: float = 0.0
    avg_mae_ticks: float = 0.0
    max_mfe: float = 0.0
    max_mae: float = 0.0

    # === Trade Duration ===
    avg_bars_held: float = 0.0
    avg_time_held_minutes: float = 0.0
    min_bars_held: int = 0
    max_bars_held: int = 0
    min_time_held_minutes: float = 0.0
    max_time_held_minutes: float = 0.0

    # === Contract Statistics ===
    avg_contracts_per_trade: float = 0.0
    max_contracts_per_trade: int = 0
    total_contracts_traded: int = 0

    # === Risk-Adjusted Returns ===
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # === Time Analysis ===
    total_days: int = 0
    trading_days: int = 0
    trades_per_day: float = 0.0
    avg_daily_pnl: float = 0.0
    best_day_pnl: float = 0.0
    worst_day_pnl: float = 0.0
    winning_days: int = 0
    losing_days: int = 0

    # === Consecutive Stats ===
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_consecutive_wins: float = 0.0
    avg_consecutive_losses: float = 0.0

    # === Model Performance ===
    accuracy: float = 0.0
    auc_roc: float = 0.0
    f1_score: float = 0.0

    # === Walk-Forward Info ===
    n_folds: int = 0
    train_samples: int = 0
    test_samples: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            # Trade Counts
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'breakeven_trades': self.breakeven_trades,
            'long_trades': self.long_trades,
            'short_trades': self.short_trades,

            # P&L
            'total_gross_pnl': round(self.total_gross_pnl, 2),
            'total_net_pnl': round(self.total_net_pnl, 2),
            'total_commission': round(self.total_commission, 2),
            'total_slippage': round(self.total_slippage, 2),

            # Win/Loss Stats
            'win_rate_pct': round(self.win_rate * 100, 2),
            'long_win_rate_pct': round(self.long_win_rate * 100, 2),
            'short_win_rate_pct': round(self.short_win_rate * 100, 2),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'max_win': round(self.max_win, 2),
            'max_loss': round(self.max_loss, 2),

            # KPIs
            'profit_factor': round(self.profit_factor, 2),
            'payoff_ratio': round(self.payoff_ratio, 2),
            'expectancy': round(self.expectancy, 2),

            # Drawdown
            'max_drawdown': round(self.max_drawdown, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct * 100, 2),
            'max_drawdown_duration_days': round(self.max_drawdown_duration_days, 1),

            # MFE/MAE
            'avg_mfe': round(self.avg_mfe, 2),
            'avg_mae': round(self.avg_mae, 2),
            'avg_mfe_ticks': round(self.avg_mfe_ticks, 1),
            'avg_mae_ticks': round(self.avg_mae_ticks, 1),

            # Duration
            'avg_bars_held': round(self.avg_bars_held, 1),
            'avg_time_held_minutes': round(self.avg_time_held_minutes, 1),
            'max_bars_held': self.max_bars_held,
            'min_bars_held': self.min_bars_held,

            # Contracts
            'avg_contracts_per_trade': round(self.avg_contracts_per_trade, 1),
            'total_contracts_traded': self.total_contracts_traded,

            # Risk-Adjusted
            'sharpe_ratio': round(self.sharpe_ratio, 3),
            'sortino_ratio': round(self.sortino_ratio, 3),
            'calmar_ratio': round(self.calmar_ratio, 3),

            # Time
            'total_days': self.total_days,
            'trading_days': self.trading_days,
            'trades_per_day': round(self.trades_per_day, 2),
            'avg_daily_pnl': round(self.avg_daily_pnl, 2),
            'best_day_pnl': round(self.best_day_pnl, 2),
            'worst_day_pnl': round(self.worst_day_pnl, 2),

            # Consecutive
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,

            # Model
            'accuracy_pct': round(self.accuracy * 100, 2),
            'auc_roc_pct': round(self.auc_roc * 100, 2),

            # Walk-Forward
            'n_folds': self.n_folds,
            'train_samples': self.train_samples,
            'test_samples': self.test_samples
        }


# ============================================================================
# DATA LEAKAGE CHECKER
# ============================================================================

class DataLeakageChecker:
    """Detect and prevent data leakage in features."""

    def __init__(self, correlation_threshold: float = 0.95):
        self.threshold = correlation_threshold
        self.suspicious_features: List[str] = []
        self.warnings: List[str] = []

    def check(
        self,
        features: pd.DataFrame,
        target_col: str,
        feature_cols: List[str]
    ) -> bool:
        """
        Check for data leakage.

        Returns True if data is clean, False if leakage detected.
        """
        self.suspicious_features = []
        self.warnings = []

        target = features[target_col].values

        for col in feature_cols:
            if col == target_col:
                continue

            # Check correlation with target
            corr = np.corrcoef(features[col].values, target)[0, 1]

            if abs(corr) > self.threshold:
                self.suspicious_features.append(col)
                self.warnings.append(
                    f"LEAKAGE: {col} has {corr:.3f} correlation with target"
                )
                logger.warning(f"DATA LEAKAGE DETECTED: {col} (corr={corr:.3f})")

            # Check for future data in feature name
            leakage_keywords = ['future', 'target', 'label', 'y_', 'next_']
            for keyword in leakage_keywords:
                if keyword in col.lower() and not col.startswith('target_'):
                    self.warnings.append(
                        f"SUSPICIOUS: {col} contains '{keyword}' in name"
                    )

        if self.suspicious_features:
            logger.warning(f"Found {len(self.suspicious_features)} potentially leaking features")
            return False

        logger.info("Data leakage check passed")
        return True


# ============================================================================
# RTH FILTER
# ============================================================================

class RTHFilter:
    """Filter data to Regular Trading Hours only."""

    def __init__(
        self,
        start_time: time = RTH_START,
        end_time: time = RTH_END,
        timezone: str = 'America/New_York'
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.timezone = timezone

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to RTH only."""
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, skipping RTH filter")
            return df

        # Convert to Eastern Time if needed
        if df.index.tz is None:
            # Assume UTC and convert
            df_tz = df.copy()
            df_tz.index = df_tz.index.tz_localize('UTC').tz_convert(self.timezone)
        else:
            df_tz = df.copy()
            df_tz.index = df_tz.index.tz_convert(self.timezone)

        # Filter by time
        mask = (df_tz.index.time >= self.start_time) & (df_tz.index.time < self.end_time)

        # Also filter by weekday (Mon-Fri = 0-4)
        mask = mask & (df_tz.index.weekday < 5)

        filtered = df.loc[mask]
        logger.info(f"RTH filter: {len(df):,} -> {len(filtered):,} bars")

        return filtered

    def is_rth(self, timestamp: datetime) -> bool:
        """Check if timestamp is within RTH."""
        if hasattr(timestamp, 'time'):
            t = timestamp.time()
            return self.start_time <= t < self.end_time
        return True


# ============================================================================
# COMPREHENSIVE BACKTESTER
# ============================================================================

class ComprehensiveBacktester:
    """
    Walk-forward backtester with comprehensive metrics.

    Features:
    - Walk-forward validation (train, embargo, test, repeat)
    - RTH-only trading enforcement
    - Data leakage prevention
    - Detailed trade logging
    - Comprehensive KPI calculation
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.daily_pnl: Dict[str, float] = {}
        self.predictions_log: List[Dict] = []
        self.rth_filter = RTHFilter()
        self.leakage_checker = DataLeakageChecker(self.config.leakage_correlation_threshold)

    def run_backtest(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        target_col: str,
        selected_features: List[str],
        model_type: str = 'lightgbm',
        model_params: Optional[Dict] = None
    ) -> Tuple[List[Trade], BacktestMetrics]:
        """
        Run comprehensive walk-forward backtest.

        Parameters:
        -----------
        prices : pd.DataFrame
            OHLCV price data with DatetimeIndex
        features : pd.DataFrame
            Feature matrix with same index
        target_col : str
            Target column name
        selected_features : List[str]
            Feature columns to use
        model_type : str
            'lightgbm', 'xgboost', or 'randomforest'
        model_params : Dict, optional
            Model hyperparameters

        Returns:
        --------
        trades : List[Trade]
            All executed trades
        metrics : BacktestMetrics
            Comprehensive performance metrics
        """
        logger.info("="*70)
        logger.info(f"COMPREHENSIVE WALK-FORWARD BACKTEST: {model_type.upper()}")
        logger.info("="*70)

        # Reset state
        self.trades = []
        self.equity_curve = [0.0]
        self.daily_pnl = {}
        self.predictions_log = []

        # Align data
        common_idx = prices.index.intersection(features.index)
        prices = prices.loc[common_idx].copy()
        features = features.loc[common_idx].copy()

        # Apply RTH filter if configured
        if self.config.rth_only:
            prices = self.rth_filter.filter(prices)
            features = features.loc[features.index.isin(prices.index)]
            logger.info(f"RTH Filter Applied: {len(prices):,} bars remaining")

        # Check for data leakage
        if self.config.check_leakage:
            is_clean = self.leakage_checker.check(features, target_col, selected_features)
            if not is_clean:
                logger.warning("Proceeding despite potential data leakage warnings")

        logger.info(f"Data: {len(prices):,} bars")
        logger.info(f"Date Range: {prices.index[0]} to {prices.index[-1]}")
        logger.info(f"Features: {len(selected_features)}")

        # Calculate walk-forward windows
        train_bars = self.config.train_days * self.config.bars_per_day
        test_bars = self.config.test_days * self.config.bars_per_day
        embargo_bars = self.config.embargo_bars

        n_samples = len(prices)
        n_folds = max(1, (n_samples - train_bars - embargo_bars) // test_bars)

        logger.info(f"\nWalk-Forward Configuration:")
        logger.info(f"  Train Window:  {self.config.train_days} days ({train_bars:,} bars)")
        logger.info(f"  Test Window:   {self.config.test_days} days ({test_bars:,} bars)")
        logger.info(f"  Embargo:       {embargo_bars} bars")
        logger.info(f"  Expected Folds: {n_folds}")

        trade_id = 0
        all_predictions = []
        all_actuals = []
        total_train_samples = 0
        total_test_samples = 0

        # Walk-forward loop
        for fold in range(n_folds):
            # Define windows
            train_start = fold * test_bars
            train_end = train_start + train_bars
            test_start = train_end + embargo_bars
            test_end = min(test_start + test_bars, n_samples)

            if test_end > n_samples or test_start >= test_end:
                break

            # Get train/test splits
            X_train = features.iloc[train_start:train_end][selected_features].values
            y_train = features.iloc[train_start:train_end][target_col].values
            X_test = features.iloc[test_start:test_end][selected_features].values
            y_test = features.iloc[test_start:test_end][target_col].values

            test_prices = prices.iloc[test_start:test_end]
            test_times = features.index[test_start:test_end]

            total_train_samples += len(X_train)
            total_test_samples += len(X_test)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model and predict
            y_proba = self._train_and_predict(
                X_train_scaled, y_train, X_test_scaled,
                model_type, model_params
            )

            all_predictions.extend(y_proba)
            all_actuals.extend(y_test)

            # Generate trades for this fold
            fold_trades, fold_trade_id = self._generate_trades(
                y_proba=y_proba,
                prices=test_prices,
                times=test_times,
                start_trade_id=trade_id,
                model_name=model_type,
                fold=fold + 1
            )

            trade_id = fold_trade_id
            self.trades.extend(fold_trades)

            # Update equity curve
            for trade in fold_trades:
                new_equity = self.equity_curve[-1] + trade.net_pnl
                self.equity_curve.append(new_equity)
                trade.cumulative_pnl = new_equity

            # Progress logging
            fold_pnl = sum(t.net_pnl for t in fold_trades)
            if (fold + 1) % 5 == 0 or fold == 0:
                logger.info(
                    f"Fold {fold + 1}/{n_folds}: "
                    f"{len(fold_trades)} trades, "
                    f"Fold P&L: ${fold_pnl:,.2f}, "
                    f"Cumulative: ${self.equity_curve[-1]:,.2f}"
                )

        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(all_predictions, all_actuals, n_folds)
        metrics.train_samples = total_train_samples
        metrics.test_samples = total_test_samples

        # Log summary
        logger.info("\n" + "="*70)
        logger.info("BACKTEST COMPLETE")
        logger.info("="*70)
        logger.info(f"Total Trades:     {metrics.total_trades}")
        logger.info(f"Net P&L:          ${metrics.total_net_pnl:,.2f}")
        logger.info(f"Win Rate:         {metrics.win_rate*100:.1f}%")
        logger.info(f"Profit Factor:    {metrics.profit_factor:.2f}")
        logger.info(f"Sharpe Ratio:     {metrics.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown:     ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct*100:.1f}%)")

        return self.trades, metrics

    def _train_and_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        model_type: str,
        model_params: Optional[Dict]
    ) -> np.ndarray:
        """Train model and generate predictions."""

        if model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                params = model_params or {
                    'objective': 'binary',
                    'metric': 'auc',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'verbose': -1
                }
                train_data = lgb.Dataset(X_train, label=y_train)
                model = lgb.train(params, train_data, num_boost_round=100)
                return model.predict(X_test)
            except ImportError:
                logger.warning("LightGBM not available, falling back to RandomForest")
                model_type = 'randomforest'

        if model_type == 'xgboost':
            try:
                import xgboost as xgb
                params = model_params or {
                    'objective': 'binary:logistic',
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'eval_metric': 'auc',
                    'verbosity': 0
                }
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test)
                model = xgb.train(params, dtrain, num_boost_round=100)
                return model.predict(dtest)
            except ImportError:
                logger.warning("XGBoost not available, falling back to RandomForest")
                model_type = 'randomforest'

        if model_type == 'randomforest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            return model.predict_proba(X_test)[:, 1]

        raise ValueError(f"Unknown model type: {model_type}")

    def _generate_trades(
        self,
        y_proba: np.ndarray,
        prices: pd.DataFrame,
        times: pd.DatetimeIndex,
        start_trade_id: int,
        model_name: str,
        fold: int
    ) -> Tuple[List[Trade], int]:
        """Generate trades from predictions with full trade management."""
        trades = []
        trade_id = start_trade_id

        in_position = False
        current_trade: Optional[Trade] = None
        entry_bar_idx = 0
        daily_trade_count: Dict[str, int] = {}
        daily_loss: Dict[str, float] = {}
        consecutive_losses = 0

        for i in range(len(y_proba) - self.config.hold_bars - 1):
            timestamp = times[i]
            date_key = str(timestamp.date()) if hasattr(timestamp, 'date') else str(timestamp)[:10]

            # Initialize daily counters
            if date_key not in daily_trade_count:
                daily_trade_count[date_key] = 0
                daily_loss[date_key] = 0.0

            # Check RTH if enabled
            if self.config.rth_only and not self.rth_filter.is_rth(timestamp):
                continue

            current_price = prices.iloc[i]['close']

            if in_position and current_trade is not None:
                bars_held = i - entry_bar_idx

                # Check exit conditions
                should_exit = False
                exit_reason = ExitReason.HOLD_BARS

                # 1. Time-based exit
                if bars_held >= self.config.hold_bars:
                    should_exit = True
                    exit_reason = ExitReason.HOLD_BARS

                # 2. Stop loss
                if self.config.stop_loss_ticks:
                    price_change = (current_price - current_trade.entry_price) * current_trade.direction.value
                    stop_price_change = self.config.stop_loss_ticks * self.config.tick_size
                    if price_change <= -stop_price_change:
                        should_exit = True
                        exit_reason = ExitReason.STOP_LOSS

                # 3. Take profit
                if self.config.take_profit_ticks and not should_exit:
                    price_change = (current_price - current_trade.entry_price) * current_trade.direction.value
                    tp_price_change = self.config.take_profit_ticks * self.config.tick_size
                    if price_change >= tp_price_change:
                        should_exit = True
                        exit_reason = ExitReason.TAKE_PROFIT

                if should_exit:
                    # Apply slippage to exit
                    slippage = self.config.slippage_ticks * self.config.tick_size
                    exit_price = current_price - (slippage * current_trade.direction.value)

                    current_trade.exit_time = timestamp
                    current_trade.exit_price = exit_price
                    current_trade.exit_bar = i
                    current_trade.exit_reason = exit_reason

                    # Calculate metrics
                    trade_prices = prices.iloc[entry_bar_idx:i+1]
                    current_trade.calculate_metrics(
                        trade_prices,
                        self.config.point_value,
                        self.config.tick_size
                    )

                    # Update daily P&L tracking
                    daily_loss[date_key] += min(0, current_trade.net_pnl)

                    # Track consecutive losses
                    if current_trade.net_pnl < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0

                    trades.append(current_trade)
                    trade_id += 1

                    in_position = False
                    current_trade = None

            else:
                # Check entry conditions

                # Skip if daily limits exceeded
                if daily_trade_count[date_key] >= self.config.max_daily_trades:
                    continue

                if abs(daily_loss[date_key]) >= self.config.max_daily_loss:
                    continue

                if consecutive_losses >= self.config.max_consecutive_losses:
                    continue

                signal = y_proba[i]
                signal_strength = abs(signal - 0.5)

                if signal_strength < self.config.min_signal_strength:
                    continue

                # Determine direction
                if signal > self.config.long_threshold:
                    direction = TradeDirection.LONG
                elif signal < self.config.short_threshold:
                    direction = TradeDirection.SHORT
                else:
                    continue

                # Apply slippage to entry
                slippage = self.config.slippage_ticks * self.config.tick_size
                entry_price = current_price + (slippage * direction.value)

                # Calculate costs
                commission = self.config.commission_per_side * 2 * self.config.contracts_per_trade
                slippage_cost = slippage * self.config.point_value * self.config.contracts_per_trade * 2

                # Create trade
                current_trade = Trade(
                    trade_id=trade_id,
                    model_name=model_name,
                    fold=fold,
                    entry_time=timestamp,
                    exit_time=timestamp,
                    entry_bar=i,
                    exit_bar=i,
                    entry_price=entry_price,
                    exit_price=entry_price,
                    direction=direction,
                    contracts=self.config.contracts_per_trade,
                    commission=commission,
                    slippage_cost=slippage_cost,
                    entry_signal=signal,
                    equity_at_entry=self.equity_curve[-1] if self.equity_curve else 0.0
                )

                in_position = True
                entry_bar_idx = i
                daily_trade_count[date_key] += 1

        return trades, trade_id

    def _calculate_metrics(
        self,
        predictions: List[float],
        actuals: List[int],
        n_folds: int
    ) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        metrics = BacktestMetrics()
        metrics.n_folds = n_folds

        if not self.trades:
            return metrics

        # === Trade Counts ===
        metrics.total_trades = len(self.trades)
        metrics.winning_trades = sum(1 for t in self.trades if t.net_pnl > 0)
        metrics.losing_trades = sum(1 for t in self.trades if t.net_pnl < 0)
        metrics.breakeven_trades = sum(1 for t in self.trades if t.net_pnl == 0)

        metrics.long_trades = sum(1 for t in self.trades if t.direction == TradeDirection.LONG)
        metrics.short_trades = sum(1 for t in self.trades if t.direction == TradeDirection.SHORT)
        metrics.long_winners = sum(1 for t in self.trades if t.direction == TradeDirection.LONG and t.net_pnl > 0)
        metrics.short_winners = sum(1 for t in self.trades if t.direction == TradeDirection.SHORT and t.net_pnl > 0)

        # === P&L Metrics ===
        metrics.total_gross_pnl = sum(t.gross_pnl for t in self.trades)
        metrics.total_net_pnl = sum(t.net_pnl for t in self.trades)
        metrics.total_commission = sum(t.commission for t in self.trades)
        metrics.total_slippage = sum(t.slippage_cost for t in self.trades)

        # === Win/Loss Statistics ===
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0
        metrics.long_win_rate = metrics.long_winners / metrics.long_trades if metrics.long_trades > 0 else 0
        metrics.short_win_rate = metrics.short_winners / metrics.short_trades if metrics.short_trades > 0 else 0

        winning_pnls = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        losing_pnls = [t.net_pnl for t in self.trades if t.net_pnl < 0]

        metrics.avg_win = np.mean(winning_pnls) if winning_pnls else 0
        metrics.avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        metrics.max_win = max(winning_pnls) if winning_pnls else 0
        metrics.max_loss = min(losing_pnls) if losing_pnls else 0

        # === KPIs ===
        total_wins = sum(winning_pnls) if winning_pnls else 0
        total_losses = abs(sum(losing_pnls)) if losing_pnls else 0

        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        metrics.payoff_ratio = abs(metrics.avg_win / metrics.avg_loss) if metrics.avg_loss != 0 else float('inf')
        metrics.expectancy = metrics.total_net_pnl / metrics.total_trades if metrics.total_trades > 0 else 0
        metrics.expectancy_per_contract = metrics.expectancy / self.config.contracts_per_trade

        # === Drawdown Metrics ===
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity

        metrics.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        metrics.max_runup = equity.max() if len(equity) > 0 else 0

        if peak.max() > 0:
            metrics.max_drawdown_pct = metrics.max_drawdown / peak.max()

        metrics.avg_drawdown = np.mean(drawdown[drawdown > 0]) if np.any(drawdown > 0) else 0

        # Drawdown duration
        in_drawdown = drawdown > 0
        if np.any(in_drawdown):
            dd_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0]
            dd_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0]
            if len(dd_starts) > 0 and len(dd_ends) > 0:
                durations = []
                for start in dd_starts:
                    ends_after = dd_ends[dd_ends > start]
                    if len(ends_after) > 0:
                        durations.append(ends_after[0] - start)
                if durations:
                    metrics.max_drawdown_duration_bars = max(durations)
                    metrics.max_drawdown_duration_days = metrics.max_drawdown_duration_bars / self.config.bars_per_day

        # === MFE/MAE Analysis ===
        mfes = [t.max_favorable_excursion for t in self.trades]
        maes = [t.max_adverse_excursion for t in self.trades]
        mfe_ticks = [t.mfe_ticks for t in self.trades]
        mae_ticks = [t.mae_ticks for t in self.trades]

        metrics.avg_mfe = np.mean(mfes) if mfes else 0
        metrics.avg_mae = np.mean(maes) if maes else 0
        metrics.max_mfe = max(mfes) if mfes else 0
        metrics.max_mae = max(maes) if maes else 0
        metrics.avg_mfe_ticks = np.mean(mfe_ticks) if mfe_ticks else 0
        metrics.avg_mae_ticks = np.mean(mae_ticks) if mae_ticks else 0

        # === Trade Duration ===
        bars_held = [t.bars_held for t in self.trades]
        time_held = [t.time_held_minutes for t in self.trades]

        metrics.avg_bars_held = np.mean(bars_held) if bars_held else 0
        metrics.min_bars_held = min(bars_held) if bars_held else 0
        metrics.max_bars_held = max(bars_held) if bars_held else 0
        metrics.avg_time_held_minutes = np.mean(time_held) if time_held else 0
        metrics.min_time_held_minutes = min(time_held) if time_held else 0
        metrics.max_time_held_minutes = max(time_held) if time_held else 0

        # === Contract Statistics ===
        contracts = [t.contracts for t in self.trades]
        metrics.avg_contracts_per_trade = np.mean(contracts) if contracts else 0
        metrics.max_contracts_per_trade = max(contracts) if contracts else 0
        metrics.total_contracts_traded = sum(contracts)

        # === Time Analysis ===
        if self.trades:
            dates = set()
            for t in self.trades:
                if hasattr(t.entry_time, 'date'):
                    dates.add(t.entry_time.date())

            metrics.trading_days = len(dates)

            if dates:
                first_date = min(dates)
                last_date = max(dates)
                metrics.total_days = (last_date - first_date).days + 1

            metrics.trades_per_day = metrics.total_trades / metrics.trading_days if metrics.trading_days > 0 else 0

            # Daily P&L analysis
            daily_pnls = {}
            for t in self.trades:
                date_key = str(t.entry_time.date()) if hasattr(t.entry_time, 'date') else str(t.entry_time)[:10]
                daily_pnls[date_key] = daily_pnls.get(date_key, 0) + t.net_pnl

            if daily_pnls:
                pnl_values = list(daily_pnls.values())
                metrics.avg_daily_pnl = np.mean(pnl_values)
                metrics.best_day_pnl = max(pnl_values)
                metrics.worst_day_pnl = min(pnl_values)
                metrics.winning_days = sum(1 for p in pnl_values if p > 0)
                metrics.losing_days = sum(1 for p in pnl_values if p < 0)

        # === Consecutive Stats ===
        if self.trades:
            is_winner = [t.net_pnl > 0 for t in self.trades]

            # Max consecutive wins/losses
            max_wins = 0
            max_losses = 0
            current_wins = 0
            current_losses = 0

            for win in is_winner:
                if win:
                    current_wins += 1
                    current_losses = 0
                    max_wins = max(max_wins, current_wins)
                else:
                    current_losses += 1
                    current_wins = 0
                    max_losses = max(max_losses, current_losses)

            metrics.max_consecutive_wins = max_wins
            metrics.max_consecutive_losses = max_losses

        # === Risk-Adjusted Returns ===
        trade_returns = [t.net_pnl for t in self.trades]
        if len(trade_returns) > 1:
            avg_return = np.mean(trade_returns)
            std_return = np.std(trade_returns)

            # Annualized Sharpe (assuming ~250 trading days)
            trades_per_year = metrics.trades_per_day * 250 if metrics.trades_per_day > 0 else 250
            metrics.sharpe_ratio = (avg_return / std_return) * np.sqrt(trades_per_year) if std_return > 0 else 0

            # Sortino (downside deviation)
            negative_returns = [r for r in trade_returns if r < 0]
            downside_std = np.std(negative_returns) if negative_returns else 0
            metrics.sortino_ratio = (avg_return / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

            # Calmar (return / max drawdown)
            metrics.calmar_ratio = metrics.total_net_pnl / metrics.max_drawdown if metrics.max_drawdown > 0 else float('inf')

        # === Model Performance ===
        if predictions and actuals:
            pred_binary = [1 if p > 0.5 else 0 for p in predictions]
            metrics.accuracy = accuracy_score(actuals, pred_binary)
            try:
                metrics.auc_roc = roc_auc_score(actuals, predictions)
            except Exception:
                metrics.auc_roc = 0.5
            metrics.f1_score = f1_score(actuals, pred_binary)

        return metrics

    def get_trade_log(self) -> pd.DataFrame:
        """Get detailed trade log as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        records = [t.to_dict() for t in self.trades]
        return pd.DataFrame(records)

    def get_equity_curve(self) -> pd.Series:
        """Get equity curve."""
        return pd.Series(self.equity_curve, name='equity')

    def get_daily_pnl(self) -> pd.DataFrame:
        """Get daily P&L summary."""
        if not self.trades:
            return pd.DataFrame()

        daily_data = {}
        for t in self.trades:
            date_key = str(t.entry_time.date()) if hasattr(t.entry_time, 'date') else str(t.entry_time)[:10]
            if date_key not in daily_data:
                daily_data[date_key] = {'pnl': 0, 'trades': 0, 'wins': 0}
            daily_data[date_key]['pnl'] += t.net_pnl
            daily_data[date_key]['trades'] += 1
            if t.net_pnl > 0:
                daily_data[date_key]['wins'] += 1

        df = pd.DataFrame.from_dict(daily_data, orient='index')
        df.index.name = 'date'
        df['win_rate'] = df['wins'] / df['trades']
        df['cumulative_pnl'] = df['pnl'].cumsum()
        return df.sort_index()


# ============================================================================
# REPORT GENERATOR
# ============================================================================

def generate_comprehensive_report(
    trades: List[Trade],
    metrics: BacktestMetrics,
    model_name: str,
    config: BacktestConfig,
    output_path: Optional[str] = None
) -> str:
    """Generate comprehensive backtest report."""

    lines = [
        "=" * 80,
        f"COMPREHENSIVE BACKTEST REPORT",
        f"Model: {model_name.upper()}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
        "CONFIGURATION",
        "-" * 40,
        f"Train Window:          {config.train_days} days",
        f"Test Window:           {config.test_days} days",
        f"Embargo Bars:          {config.embargo_bars}",
        f"Hold Bars:             {config.hold_bars}",
        f"Contracts per Trade:   {config.contracts_per_trade}",
        f"Stop Loss:             {config.stop_loss_ticks} ticks",
        f"Take Profit:           {config.take_profit_ticks} ticks",
        f"Commission per Side:   ${config.commission_per_side}",
        f"Slippage:              {config.slippage_ticks} ticks",
        f"RTH Only:              {config.rth_only}",
        "",
        "=" * 80,
        "TRADE SUMMARY",
        "=" * 80,
        "",
        f"Total Trades:          {metrics.total_trades:,}",
        f"  - Winning:           {metrics.winning_trades:,} ({metrics.win_rate*100:.1f}%)",
        f"  - Losing:            {metrics.losing_trades:,}",
        f"  - Breakeven:         {metrics.breakeven_trades:,}",
        "",
        f"Long Trades:           {metrics.long_trades:,} ({metrics.long_win_rate*100:.1f}% win rate)",
        f"Short Trades:          {metrics.short_trades:,} ({metrics.short_win_rate*100:.1f}% win rate)",
        "",
        "=" * 80,
        "PROFIT & LOSS",
        "=" * 80,
        "",
        f"Gross P&L:             ${metrics.total_gross_pnl:>12,.2f}",
        f"Total Commission:      ${metrics.total_commission:>12,.2f}",
        f"Total Slippage:        ${metrics.total_slippage:>12,.2f}",
        f"NET P&L:               ${metrics.total_net_pnl:>12,.2f}",
        "",
        f"Average Win:           ${metrics.avg_win:>12,.2f}",
        f"Average Loss:          ${metrics.avg_loss:>12,.2f}",
        f"Largest Win:           ${metrics.max_win:>12,.2f}",
        f"Largest Loss:          ${metrics.max_loss:>12,.2f}",
        "",
        "=" * 80,
        "KEY PERFORMANCE INDICATORS (KPIs)",
        "=" * 80,
        "",
        f"Profit Factor:         {metrics.profit_factor:>12.2f}",
        f"Payoff Ratio:          {metrics.payoff_ratio:>12.2f}",
        f"Expectancy per Trade:  ${metrics.expectancy:>11,.2f}",
        f"Expectancy per Contract: ${metrics.expectancy_per_contract:>9,.2f}",
        "",
        "=" * 80,
        "DRAWDOWN ANALYSIS",
        "=" * 80,
        "",
        f"Max Drawdown:          ${metrics.max_drawdown:>12,.2f} ({metrics.max_drawdown_pct*100:.1f}%)",
        f"Max Drawdown Duration: {metrics.max_drawdown_duration_days:>10.1f} days",
        f"Average Drawdown:      ${metrics.avg_drawdown:>12,.2f}",
        f"Max Run-up:            ${metrics.max_runup:>12,.2f}",
        "",
        "=" * 80,
        "EXCURSION ANALYSIS (MFE/MAE)",
        "=" * 80,
        "",
        f"Avg Max Favorable:     ${metrics.avg_mfe:>12,.2f} ({metrics.avg_mfe_ticks:.1f} ticks)",
        f"Avg Max Adverse:       ${metrics.avg_mae:>12,.2f} ({metrics.avg_mae_ticks:.1f} ticks)",
        f"Max MFE:               ${metrics.max_mfe:>12,.2f}",
        f"Max MAE:               ${metrics.max_mae:>12,.2f}",
        "",
        "=" * 80,
        "TRADE DURATION",
        "=" * 80,
        "",
        f"Average Bars Held:     {metrics.avg_bars_held:>12.1f}",
        f"Average Time Held:     {metrics.avg_time_held_minutes:>12.1f} minutes",
        f"Min Bars Held:         {metrics.min_bars_held:>12}",
        f"Max Bars Held:         {metrics.max_bars_held:>12}",
        "",
        "=" * 80,
        "CONTRACT STATISTICS",
        "=" * 80,
        "",
        f"Avg Contracts/Trade:   {metrics.avg_contracts_per_trade:>12.1f}",
        f"Max Contracts/Trade:   {metrics.max_contracts_per_trade:>12}",
        f"Total Contracts:       {metrics.total_contracts_traded:>12,}",
        "",
        "=" * 80,
        "RISK-ADJUSTED RETURNS",
        "=" * 80,
        "",
        f"Sharpe Ratio:          {metrics.sharpe_ratio:>12.3f}",
        f"Sortino Ratio:         {metrics.sortino_ratio:>12.3f}",
        f"Calmar Ratio:          {metrics.calmar_ratio:>12.3f}",
        "",
        "=" * 80,
        "TIME ANALYSIS",
        "=" * 80,
        "",
        f"Total Calendar Days:   {metrics.total_days:>12}",
        f"Trading Days:          {metrics.trading_days:>12}",
        f"Trades per Day:        {metrics.trades_per_day:>12.2f}",
        f"Avg Daily P&L:         ${metrics.avg_daily_pnl:>11,.2f}",
        f"Best Day:              ${metrics.best_day_pnl:>11,.2f}",
        f"Worst Day:             ${metrics.worst_day_pnl:>11,.2f}",
        f"Winning Days:          {metrics.winning_days:>12}",
        f"Losing Days:           {metrics.losing_days:>12}",
        "",
        "=" * 80,
        "CONSECUTIVE STATISTICS",
        "=" * 80,
        "",
        f"Max Consecutive Wins:  {metrics.max_consecutive_wins:>12}",
        f"Max Consecutive Losses:{metrics.max_consecutive_losses:>12}",
        "",
        "=" * 80,
        "MODEL PERFORMANCE",
        "=" * 80,
        "",
        f"Accuracy:              {metrics.accuracy*100:>11.2f}%",
        f"AUC-ROC:               {metrics.auc_roc*100:>11.2f}%",
        f"F1 Score:              {metrics.f1_score:>12.4f}",
        "",
        f"Walk-Forward Folds:    {metrics.n_folds:>12}",
        f"Train Samples:         {metrics.train_samples:>12,}",
        f"Test Samples:          {metrics.test_samples:>12,}",
        "",
        "=" * 80
    ]

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")

    return report


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_comprehensive_backtest(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    target_col: str = 'target_direction_1',
    selected_features: Optional[List[str]] = None,
    model_type: str = 'lightgbm',
    config: Optional[BacktestConfig] = None,
    output_dir: Optional[str] = None
) -> Tuple[List[Trade], BacktestMetrics, str]:
    """
    Run comprehensive backtest and generate full report.

    Returns trades, metrics, and report text.
    """
    config = config or BacktestConfig()

    if selected_features is None:
        selected_features = [c for c in features.columns if not c.startswith('target_')]

    backtester = ComprehensiveBacktester(config)
    trades, metrics = backtester.run_backtest(
        prices, features, target_col, selected_features, model_type
    )

    report = generate_comprehensive_report(trades, metrics, model_type, config)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save trade log
        trade_log = backtester.get_trade_log()
        if not trade_log.empty:
            trade_log.to_csv(output_path / f'trades_{model_type}_{timestamp}.csv', index=False)

        # Save metrics
        with open(output_path / f'metrics_{model_type}_{timestamp}.json', 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)

        # Save report
        with open(output_path / f'report_{model_type}_{timestamp}.txt', 'w') as f:
            f.write(report)

        # Save equity curve
        equity = backtester.get_equity_curve()
        equity.to_csv(output_path / f'equity_{model_type}_{timestamp}.csv')

        # Save daily P&L
        daily_pnl = backtester.get_daily_pnl()
        if not daily_pnl.empty:
            daily_pnl.to_csv(output_path / f'daily_pnl_{model_type}_{timestamp}.csv')

        logger.info(f"All results saved to {output_path}")

    return trades, metrics, report


# Exports
__all__ = [
    'BacktestConfig',
    'Trade',
    'TradeDirection',
    'ExitReason',
    'BacktestMetrics',
    'DataLeakageChecker',
    'RTHFilter',
    'ComprehensiveBacktester',
    'generate_comprehensive_report',
    'run_comprehensive_backtest'
]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("="*70)
    print("COMPREHENSIVE WALK-FORWARD BACKTEST SYSTEM")
    print("="*70)
    print("\nUsage:")
    print("  from comprehensive_backtest import run_comprehensive_backtest")
    print("  trades, metrics, report = run_comprehensive_backtest(")
    print("      prices, features, target_col='target_direction_1',")
    print("      model_type='lightgbm', output_dir='data/backtest_results'")
    print("  )")
    print("\nFeatures:")
    print("  - Walk-forward validation with configurable windows")
    print("  - RTH-only trading enforcement")
    print("  - Data leakage detection")
    print("  - Comprehensive metrics (P&L, drawdown, MFE/MAE, duration)")
    print("  - Detailed trade logging with entry/exit times")
    print("  - Risk-adjusted returns (Sharpe, Sortino, Calmar)")
    print("  - Full report generation")
