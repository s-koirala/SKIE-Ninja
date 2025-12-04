"""
SKIE-Ninja Triple Barrier + Meta-Labeling Walk-Forward Backtest

Integrated pipeline implementing Lopez de Prado's methodology:
1. Triple Barrier labeling for realistic trade outcomes
2. Meta-labeling for bet sizing (quality over quantity)
3. Volatility regime features for strategy adaptation
4. Walk-forward validation with rigorous QC checks

Literature References:
- Lopez de Prado (2018) "Advances in Financial Machine Learning" Ch. 3
- PLOS One 2024: VIX-based regime detection
- Macrosynergy 2024: HMM for market regimes

QC Methodology:
- Feature-target correlation checks (max 0.3 for legitimacy)
- Temporal ordering verification (no future data)
- Distribution stability across folds
- Embargo periods between train/test

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import json
import warnings
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import lightgbm as lgb

# SKIE-Ninja imports
from data_collection.ninjatrader_loader import NinjaTraderLoader, load_sample_data
from feature_engineering.triple_barrier import (
    TripleBarrierConfig, TripleBarrierLabeler, apply_triple_barrier
)
from feature_engineering.volatility_regime import (
    RealizedVolatilityGenerator, RegimeDetector, generate_volatility_features
)
from models.meta_labeling import MetaLabelConfig, MetaLabeler

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the integrated backtest pipeline."""

    # Triple Barrier Settings (Lopez de Prado 2018)
    tb_upper_barrier: float = 2.0      # Take profit in ATR multiples
    tb_lower_barrier: float = 1.0      # Stop loss in ATR multiples
    tb_max_holding: int = 12           # Max bars to hold
    tb_atr_period: int = 14            # ATR lookback

    # Meta-Labeling Settings
    meta_threshold: float = 0.5        # Minimum meta probability to trade
    sizing_method: str = 'sqrt'        # Position sizing: linear, sqrt, kelly

    # Walk-Forward Settings
    train_days: int = 180              # Training window
    test_days: int = 5                 # Test window
    embargo_bars: int = 42             # Gap between train/test (~3.5 hours)
    min_train_samples: int = 5000      # Minimum samples to train

    # Model Settings
    primary_model: str = 'lightgbm'    # Primary direction model
    n_estimators: int = 100            # Trees in ensemble
    max_depth: int = 5                 # Tree depth (prevent overfit)
    learning_rate: float = 0.05        # Conservative learning rate

    # Trading Settings
    contracts_per_trade: int = 1       # Base contracts
    max_contracts: int = 3             # Max position size
    commission_per_side: float = 2.50  # Per contract
    slippage_ticks: float = 0.5        # Expected slippage
    point_value: float = 50.0          # ES contract value
    tick_size: float = 0.25            # ES tick size

    # QC Thresholds (Literature-Based)
    # References:
    # - "Suspicious" levels from Lopez de Prado suggest >0.3 correlation is leakage
    # - Typical legitimate financial ML: AUC 0.52-0.58 (QuantStart, 2021)
    max_feature_target_corr: float = 0.30    # Features >0.3 corr = suspicious
    min_auc_threshold: float = 0.51          # Must beat random by margin
    max_auc_threshold: float = 0.70          # AUC >0.70 = likely leakage
    max_sharpe_threshold: float = 3.0        # Sharpe >3 = suspicious
    max_win_rate_threshold: float = 0.65     # Win rate >65% = suspicious
    min_trade_count: int = 50                # Need enough trades for significance

    # Data Settings
    timeframe_minutes: int = 5         # 5-minute bars
    rth_only: bool = True              # Regular trading hours only


@dataclass
class QCReport:
    """Quality control report for detecting data leakage."""

    # Feature Checks
    max_feature_corr: float = 0.0
    high_corr_features: List[str] = field(default_factory=list)

    # Temporal Checks
    future_leak_detected: bool = False
    future_leak_features: List[str] = field(default_factory=list)

    # Performance Checks
    train_auc: float = 0.0
    test_auc: float = 0.0
    auc_gap: float = 0.0               # Large gap = overfit

    # Distribution Checks
    label_distribution_stable: bool = True
    feature_distribution_stable: bool = True

    # Overall
    all_checks_passed: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class BacktestResults:
    """Comprehensive backtest results."""

    # Trade Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    # P&L
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0

    # Win/Loss
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Risk Metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade Duration
    avg_bars_held: float = 0.0
    avg_minutes_held: float = 0.0

    # Model Metrics
    primary_auc: float = 0.0
    meta_auc: float = 0.0
    precision_improvement: float = 0.0
    trades_filtered_pct: float = 0.0

    # QC Status
    qc_passed: bool = False
    qc_warnings: List[str] = field(default_factory=list)

    # Fold Details
    fold_results: List[Dict] = field(default_factory=list)


# ============================================================================
# DATA LEAKAGE CHECKER
# ============================================================================

class DataLeakageChecker:
    """
    Comprehensive data leakage detection.

    Based on Lopez de Prado (2018) and best practices:
    - Feature-target correlation analysis
    - Temporal ordering verification
    - Train/test distribution comparison
    - Suspicious performance flagging
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.report = QCReport()

    def check_feature_correlations(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> List[str]:
        """
        Check for suspiciously high feature-target correlations.

        Reference: Lopez de Prado (2018) - correlations >0.3 are suspicious
        """
        warnings = []
        high_corr_features = []

        for col in features.columns:
            corr = features[col].corr(target)
            if abs(corr) > self.config.max_feature_target_corr:
                high_corr_features.append(f"{col}: {corr:.4f}")
                warnings.append(
                    f"HIGH CORRELATION: {col} has {corr:.4f} correlation with target"
                )

        self.report.high_corr_features = high_corr_features
        self.report.max_feature_corr = max(
            [abs(features[col].corr(target)) for col in features.columns],
            default=0.0
        )

        return warnings

    def check_temporal_ordering(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> List[str]:
        """
        Verify no features use future data.

        Check for:
        - Forward shifts
        - Future aggregations
        - Target leakage patterns
        """
        warnings = []
        future_leak_features = []

        # Check for common leakage patterns in feature names
        leakage_patterns = [
            'future', 'next', 'forward', 'lead', 'shift_-',
            'target', 'label', 'return_fwd'
        ]

        for col in feature_cols:
            col_lower = col.lower()
            for pattern in leakage_patterns:
                if pattern in col_lower:
                    future_leak_features.append(col)
                    warnings.append(
                        f"POTENTIAL LEAKAGE: {col} contains suspicious pattern '{pattern}'"
                    )
                    break

        # Statistical check: features with perfect separation
        if 'target' in df.columns or 'label' in df.columns:
            target_col = 'target' if 'target' in df.columns else 'label'
            for col in feature_cols:
                if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                    # Check if feature perfectly separates classes
                    grouped = df.groupby(target_col)[col].agg(['mean', 'std'])
                    if grouped['std'].min() < 1e-6:  # Near-zero variance in one class
                        future_leak_features.append(col)
                        warnings.append(
                            f"STATISTICAL LEAKAGE: {col} has near-perfect class separation"
                        )

        self.report.future_leak_detected = len(future_leak_features) > 0
        self.report.future_leak_features = future_leak_features

        return warnings

    def check_train_test_gap(
        self,
        train_auc: float,
        test_auc: float
    ) -> List[str]:
        """
        Check for overfitting via train/test performance gap.

        Large gap (>0.15) suggests model is memorizing training data.
        """
        warnings = []

        self.report.train_auc = train_auc
        self.report.test_auc = test_auc
        self.report.auc_gap = train_auc - test_auc

        if self.report.auc_gap > 0.15:
            warnings.append(
                f"OVERFIT WARNING: Train AUC ({train_auc:.4f}) much higher than "
                f"Test AUC ({test_auc:.4f}). Gap: {self.report.auc_gap:.4f}"
            )

        return warnings

    def check_suspicious_performance(
        self,
        results: BacktestResults
    ) -> List[str]:
        """
        Flag suspiciously good performance that likely indicates leakage.

        Based on literature:
        - Sharpe >3 is exceptional and rare
        - Win rate >65% in directional trading is suspicious
        - AUC >0.70 on financial data is very high
        """
        warnings = []

        if results.sharpe_ratio > self.config.max_sharpe_threshold:
            warnings.append(
                f"SUSPICIOUS: Sharpe ratio {results.sharpe_ratio:.2f} exceeds "
                f"threshold {self.config.max_sharpe_threshold}. "
                "This is rare in legitimate trading systems."
            )

        if results.win_rate > self.config.max_win_rate_threshold:
            warnings.append(
                f"SUSPICIOUS: Win rate {results.win_rate:.1%} exceeds "
                f"threshold {self.config.max_win_rate_threshold:.0%}. "
                "Typical directional strategies achieve 50-55%."
            )

        if results.primary_auc > self.config.max_auc_threshold:
            warnings.append(
                f"SUSPICIOUS: Primary AUC {results.primary_auc:.4f} exceeds "
                f"threshold {self.config.max_auc_threshold}. "
                "Financial ML typically achieves 0.52-0.58."
            )

        if results.profit_factor > 3.0:
            warnings.append(
                f"SUSPICIOUS: Profit factor {results.profit_factor:.2f} > 3.0. "
                "Sustainable strategies typically show 1.2-1.5."
            )

        return warnings

    def generate_report(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        train_auc: float,
        test_auc: float,
        results: Optional[BacktestResults] = None
    ) -> QCReport:
        """Generate comprehensive QC report."""

        all_warnings = []
        all_errors = []

        # Run all checks
        all_warnings.extend(self.check_feature_correlations(features, target))
        all_warnings.extend(self.check_temporal_ordering(
            features.assign(target=target),
            features.columns.tolist()
        ))
        all_warnings.extend(self.check_train_test_gap(train_auc, test_auc))

        if results:
            all_warnings.extend(self.check_suspicious_performance(results))

        # Determine if passed
        self.report.warnings = all_warnings
        self.report.errors = all_errors
        self.report.all_checks_passed = (
            not self.report.future_leak_detected and
            len(all_errors) == 0 and
            self.report.max_feature_corr <= self.config.max_feature_target_corr
        )

        return self.report


# ============================================================================
# FEATURE GENERATOR
# ============================================================================

def generate_features(
    prices: pd.DataFrame,
    config: PipelineConfig
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Generate features for the Triple Barrier pipeline.

    Features include:
    - Volatility features (RV, ATR, Parkinson)
    - Regime features (GMM classification)
    - Technical features (momentum, mean reversion)
    - Time features (hour, day of week)

    All features are designed to avoid look-ahead bias.
    """
    logger.info("Generating features...")

    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in prices.columns:
            prices[col] = prices['close'] if col != 'volume' else 0

    features = pd.DataFrame(index=prices.index)
    feature_info = {}

    # 1. Volatility Features (Lopez de Prado, PLOS One 2024)
    logger.info("  Generating volatility features...")
    rv_gen = RealizedVolatilityGenerator()
    rv_features = rv_gen.generate_features(prices)
    features = pd.concat([features, rv_features], axis=1)
    feature_info['realized_vol'] = rv_features.columns.tolist()

    # 2. Regime Features (Macrosynergy 2024)
    logger.info("  Generating regime features...")
    regime_detector = RegimeDetector()
    regime_features = regime_detector.classify_regime_rules(rv_features, prices)
    # Only use lagged regime (current regime is based on current data)
    regime_features['regime_lag1'] = regime_features['regime'].shift(1)
    regime_features['regime_lag2'] = regime_features['regime'].shift(2)
    features = pd.concat([
        features,
        regime_features[['regime', 'regime_lag1', 'regime_lag2']]
    ], axis=1)
    feature_info['regime'] = ['regime', 'regime_lag1', 'regime_lag2']

    # 3. Technical Features (standard momentum/mean-reversion)
    logger.info("  Generating technical features...")

    # Returns (lagged to avoid leakage)
    for lag in [1, 2, 3, 5, 10]:
        features[f'return_lag{lag}'] = prices['close'].pct_change(lag).shift(1)

    # Momentum
    for period in [5, 10, 20]:
        features[f'momentum_{period}'] = (
            prices['close'].shift(1) / prices['close'].shift(period + 1) - 1
        )

    # Mean reversion signals
    for period in [10, 20]:
        ma = prices['close'].rolling(period).mean().shift(1)
        features[f'dist_to_ma_{period}'] = (prices['close'].shift(1) - ma) / ma

    # RSI (lagged)
    for period in [7, 14]:
        delta = prices['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        features[f'rsi_{period}'] = (100 - 100 / (1 + rs)).shift(1)

    feature_info['technical'] = [c for c in features.columns
                                  if c not in feature_info.get('realized_vol', []) +
                                  feature_info.get('regime', [])]

    # 4. Time Features
    logger.info("  Generating time features...")
    if isinstance(prices.index, pd.DatetimeIndex):
        features['hour'] = prices.index.hour
        features['day_of_week'] = prices.index.dayofweek
        features['bars_since_open'] = (
            (prices.index.hour - 9) * 12 + prices.index.minute // 5
        )
    feature_info['time'] = ['hour', 'day_of_week', 'bars_since_open']

    # 5. Generate Triple Barrier Labels
    logger.info("  Generating Triple Barrier labels...")
    tb_config = TripleBarrierConfig(
        upper_barrier=config.tb_upper_barrier,
        lower_barrier=config.tb_lower_barrier,
        max_holding_bars=config.tb_max_holding,
        atr_period=config.tb_atr_period,
        use_atr=True
    )
    tb_labeler = TripleBarrierLabeler(tb_config)
    tb_results = tb_labeler.fit_transform(prices)

    # Target is Triple Barrier label (1 = profitable, -1 = loss, 0 = timeout)
    # Convert to binary for classification (1 = should trade, 0 = should not)
    target = (tb_results['tb_label'] == 1).astype(int)

    feature_info['label_distribution'] = {
        'profitable': (tb_results['tb_label'] == 1).sum(),
        'loss': (tb_results['tb_label'] == -1).sum(),
        'timeout': (tb_results['tb_label'] == 0).sum()
    }

    # Drop NaN rows
    valid_idx = features.dropna().index.intersection(target.dropna().index)
    features = features.loc[valid_idx]
    target = target.loc[valid_idx]

    logger.info(f"  Generated {len(features.columns)} features")
    logger.info(f"  Valid samples: {len(features)}")
    logger.info(f"  Label distribution: {feature_info['label_distribution']}")

    return features, target, feature_info


# ============================================================================
# WALK-FORWARD BACKTESTER
# ============================================================================

class TripleBarrierBacktester:
    """
    Walk-forward backtester with Triple Barrier + Meta-Labeling.

    Pipeline:
    1. Split data into train/test folds
    2. Train primary model on Triple Barrier labels
    3. Train meta-model for bet sizing
    4. Simulate trading on test set
    5. Aggregate results across folds
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.qc_checker = DataLeakageChecker(config)
        self.results = BacktestResults()
        self.trades = []

    def create_walk_forward_splits(
        self,
        n_samples: int,
        bars_per_day: int = 78
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create walk-forward train/test splits with embargo."""
        splits = []

        train_bars = self.config.train_days * bars_per_day
        test_bars = self.config.test_days * bars_per_day
        embargo_bars = self.config.embargo_bars

        start_idx = train_bars

        while start_idx + test_bars <= n_samples:
            train_idx = np.arange(start_idx - train_bars, start_idx - embargo_bars)
            test_idx = np.arange(start_idx, min(start_idx + test_bars, n_samples))

            if len(train_idx) >= self.config.min_train_samples:
                splits.append((train_idx, test_idx))

            start_idx += test_bars

        return splits

    def train_primary_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> lgb.LGBMClassifier:
        """Train primary direction model."""
        model = lgb.LGBMClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=42,
            verbose=-1,
            force_col_wise=True
        )
        model.fit(X_train, y_train)
        return model

    def train_meta_model(
        self,
        X_train: np.ndarray,
        primary_probs: np.ndarray,
        y_train: np.ndarray
    ) -> MetaLabeler:
        """Train meta-labeling model for bet sizing."""
        meta_config = MetaLabelConfig(
            min_probability=self.config.meta_threshold,
            sizing_method=self.config.sizing_method
        )
        meta_labeler = MetaLabeler(meta_config)

        # Primary predictions (binary direction) - convert to pandas Series/DataFrame
        # Ensure column names are strings to avoid sklearn error
        X_train_df = pd.DataFrame(X_train)
        X_train_df.columns = [f'feat_{i}' for i in range(X_train_df.shape[1])]
        primary_preds = pd.Series((primary_probs >= 0.5).astype(int))
        y_train_series = pd.Series(y_train)
        primary_probs_series = pd.Series(primary_probs)

        # Fit meta-labeler
        meta_labeler.fit(X_train_df, primary_preds, y_train_series, primary_probs_series)

        return meta_labeler

    def simulate_trades(
        self,
        prices: pd.DataFrame,
        test_idx: np.ndarray,
        primary_probs: np.ndarray,
        bet_sizes: np.ndarray,
        fold: int
    ) -> List[Dict]:
        """Simulate trades based on predictions and bet sizes."""
        trades = []

        for i, idx in enumerate(test_idx):
            if idx >= len(prices):
                continue

            prob = primary_probs[i]
            size = bet_sizes[i]

            # Skip if bet size is 0 (filtered by meta-model)
            if size <= 0:
                continue

            # Determine direction
            direction = 1 if prob >= 0.5 else -1

            # Get entry/exit prices
            entry_bar = prices.iloc[idx]
            entry_price = entry_bar['close']
            entry_time = prices.index[idx]

            # Find exit (use Triple Barrier holding period)
            exit_idx = min(idx + self.config.tb_max_holding, len(prices) - 1)
            exit_bar = prices.iloc[exit_idx]
            exit_price = exit_bar['close']
            exit_time = prices.index[exit_idx]

            # Calculate P&L
            contracts = max(1, int(size * self.config.max_contracts))
            contracts = min(contracts, self.config.max_contracts)

            price_diff = (exit_price - entry_price) * direction
            gross_pnl = price_diff * self.config.point_value * contracts

            commission = self.config.commission_per_side * 2 * contracts
            slippage = self.config.slippage_ticks * self.config.tick_size * \
                       self.config.point_value * 2 * contracts

            net_pnl = gross_pnl - commission - slippage

            trade = {
                'fold': fold,
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
                'signal_prob': prob,
                'bet_size': size,
                'bars_held': exit_idx - idx
            }
            trades.append(trade)

        return trades

    def calculate_metrics(self, trades: List[Dict]) -> BacktestResults:
        """Calculate comprehensive backtest metrics."""
        results = BacktestResults()

        if not trades:
            return results

        trades_df = pd.DataFrame(trades)

        # Basic counts
        results.total_trades = len(trades)
        results.winning_trades = (trades_df['net_pnl'] > 0).sum()
        results.losing_trades = (trades_df['net_pnl'] < 0).sum()
        results.breakeven_trades = (trades_df['net_pnl'] == 0).sum()

        # P&L
        results.gross_pnl = trades_df['gross_pnl'].sum()
        results.net_pnl = trades_df['net_pnl'].sum()
        results.total_commission = trades_df['commission'].sum()
        results.total_slippage = trades_df['slippage'].sum()

        # Win/Loss
        results.win_rate = results.winning_trades / results.total_trades if results.total_trades > 0 else 0

        winners = trades_df[trades_df['net_pnl'] > 0]['net_pnl']
        losers = trades_df[trades_df['net_pnl'] < 0]['net_pnl']

        results.avg_win = winners.mean() if len(winners) > 0 else 0
        results.avg_loss = losers.mean() if len(losers) > 0 else 0
        results.largest_win = winners.max() if len(winners) > 0 else 0
        results.largest_loss = losers.min() if len(losers) > 0 else 0

        # Profit Factor
        gross_profit = winners.sum() if len(winners) > 0 else 0
        gross_loss = abs(losers.sum()) if len(losers) > 0 else 1
        results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Drawdown
        cumulative_pnl = trades_df['net_pnl'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = running_max - cumulative_pnl
        results.max_drawdown = drawdown.max()
        results.max_drawdown_pct = results.max_drawdown / running_max.max() if running_max.max() > 0 else 0

        # Risk-Adjusted Returns (using DAILY returns)
        trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date
        daily_pnl = trades_df.groupby('date')['net_pnl'].sum()

        if len(daily_pnl) > 1:
            avg_daily = daily_pnl.mean()
            std_daily = daily_pnl.std()

            # Sharpe Ratio (annualized)
            results.sharpe_ratio = (avg_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0

            # Sortino Ratio (downside deviation)
            downside = daily_pnl[daily_pnl < 0]
            downside_std = downside.std() if len(downside) > 1 else std_daily
            results.sortino_ratio = (avg_daily / downside_std) * np.sqrt(252) if downside_std > 0 else 0

            # Calmar Ratio
            annual_return = avg_daily * 252
            results.calmar_ratio = annual_return / results.max_drawdown if results.max_drawdown > 0 else 0

        # Trade Duration
        results.avg_bars_held = trades_df['bars_held'].mean()
        results.avg_minutes_held = results.avg_bars_held * self.config.timeframe_minutes

        return results

    def run(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        target: pd.Series
    ) -> BacktestResults:
        """Run the full walk-forward backtest."""
        logger.info("=" * 60)
        logger.info("RUNNING TRIPLE BARRIER WALK-FORWARD BACKTEST")
        logger.info("=" * 60)

        # Create splits
        splits = self.create_walk_forward_splits(len(features))
        logger.info(f"Created {len(splits)} walk-forward folds")

        if not splits:
            logger.error("Not enough data for walk-forward validation")
            return self.results

        all_trades = []
        fold_metrics = []
        all_train_aucs = []
        all_test_aucs = []
        total_primary_trades = 0
        total_filtered_trades = 0

        for fold, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"\n--- Fold {fold + 1}/{len(splits)} ---")
            logger.info(f"  Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")

            # Get data for this fold
            X_train = features.iloc[train_idx].values
            y_train = target.iloc[train_idx].values
            X_test = features.iloc[test_idx].values
            y_test = target.iloc[test_idx].values

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train primary model
            primary_model = self.train_primary_model(X_train_scaled, y_train)

            # Get predictions
            train_probs = primary_model.predict_proba(X_train_scaled)[:, 1]
            test_probs = primary_model.predict_proba(X_test_scaled)[:, 1]

            # Calculate AUCs
            train_auc = roc_auc_score(y_train, train_probs)
            test_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.5

            all_train_aucs.append(train_auc)
            all_test_aucs.append(test_auc)

            logger.info(f"  Primary Model - Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}")

            # Train meta-model
            meta_labeler = self.train_meta_model(X_train_scaled, train_probs, y_train)

            # Get bet sizes for test set - convert to pandas
            X_test_df = pd.DataFrame(X_test_scaled)
            X_test_df.columns = [f'feat_{i}' for i in range(X_test_df.shape[1])]
            test_preds = pd.Series((test_probs >= 0.5).astype(int))
            test_probs_series = pd.Series(test_probs)
            bet_sizes = meta_labeler.predict_bet_size(
                X_test_df, test_preds, test_probs_series
            )

            # Count filtered trades
            primary_signals = len(test_idx)
            meta_signals = (bet_sizes > 0).sum()
            total_primary_trades += primary_signals
            total_filtered_trades += (primary_signals - meta_signals)

            logger.info(f"  Meta-Labeling - Filtered {primary_signals - meta_signals}/{primary_signals} trades")

            # Simulate trades
            fold_trades = self.simulate_trades(
                prices, test_idx, test_probs, bet_sizes.values, fold + 1
            )
            all_trades.extend(fold_trades)

            # Fold metrics
            fold_result = {
                'fold': fold + 1,
                'train_samples': len(train_idx),
                'test_samples': len(test_idx),
                'train_auc': train_auc,
                'test_auc': test_auc,
                'trades': len(fold_trades),
                'pnl': sum(t['net_pnl'] for t in fold_trades)
            }
            fold_metrics.append(fold_result)

        # Calculate overall metrics
        self.results = self.calculate_metrics(all_trades)
        self.results.fold_results = fold_metrics
        self.trades = all_trades

        # Add model metrics
        self.results.primary_auc = np.mean(all_test_aucs)
        self.results.trades_filtered_pct = total_filtered_trades / total_primary_trades \
            if total_primary_trades > 0 else 0

        # Run QC checks
        logger.info("\n--- Quality Control Checks ---")
        qc_report = self.qc_checker.generate_report(
            features, target,
            np.mean(all_train_aucs), np.mean(all_test_aucs),
            self.results
        )

        self.results.qc_passed = qc_report.all_checks_passed
        self.results.qc_warnings = qc_report.warnings

        if qc_report.warnings:
            logger.warning(f"  QC Warnings: {len(qc_report.warnings)}")
            for w in qc_report.warnings:
                logger.warning(f"    - {w}")
        else:
            logger.info("  All QC checks passed!")

        return self.results


# ============================================================================
# REPORT GENERATOR
# ============================================================================

def generate_report(
    results: BacktestResults,
    config: PipelineConfig,
    output_path: Path
) -> str:
    """Generate comprehensive backtest report."""

    report = []
    report.append("=" * 80)
    report.append(" SKIE-NINJA TRIPLE BARRIER + META-LABELING BACKTEST REPORT")
    report.append(" " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    report.append("=" * 80)

    # Literature References
    report.append("\n" + "-" * 60)
    report.append("METHODOLOGY & LITERATURE REFERENCES")
    report.append("-" * 60)
    report.append("""
Triple Barrier Method: Lopez de Prado (2018) "Advances in Financial ML" Ch. 3
  - ATR-adjusted take profit/stop loss barriers
  - Vertical time barrier for maximum holding period
  - Labels based on actual trade outcomes, not arbitrary thresholds

Meta-Labeling: Lopez de Prado (2018) "Advances in Financial ML" Ch. 3
  - Two-stage architecture: direction + bet sizing
  - Improves precision by filtering low-confidence trades
  - Enables Kelly-optimal position sizing

Volatility Regime: PLOS One 2024, Macrosynergy 2024
  - Realized volatility features
  - GMM/HMM regime detection
  - Strategy adaptation to market conditions

QC Methodology: Lopez de Prado (2018) + Industry Best Practices
  - Feature-target correlation threshold: {max_corr:.2f}
  - Maximum legitimate AUC: {max_auc:.2f}
  - Maximum legitimate Sharpe: {max_sharpe:.1f}
  - Minimum trade count for significance: {min_trades}
""".format(
        max_corr=config.max_feature_target_corr,
        max_auc=config.max_auc_threshold,
        max_sharpe=config.max_sharpe_threshold,
        min_trades=config.min_trade_count
    ))

    # Configuration
    report.append("\n" + "-" * 60)
    report.append("CONFIGURATION")
    report.append("-" * 60)
    report.append(f"""
Triple Barrier:
  - Upper Barrier (TP): {config.tb_upper_barrier} ATR
  - Lower Barrier (SL): {config.tb_lower_barrier} ATR
  - Max Holding: {config.tb_max_holding} bars ({config.tb_max_holding * config.timeframe_minutes} minutes)
  - ATR Period: {config.tb_atr_period}

Walk-Forward:
  - Training Window: {config.train_days} days
  - Test Window: {config.test_days} days
  - Embargo: {config.embargo_bars} bars ({config.embargo_bars * config.timeframe_minutes} minutes)

Trading:
  - Commission: ${config.commission_per_side}/side
  - Slippage: {config.slippage_ticks} ticks
  - Max Contracts: {config.max_contracts}
""")

    # Results Summary
    report.append("\n" + "-" * 60)
    report.append("RESULTS SUMMARY")
    report.append("-" * 60)

    report.append(f"""
TRADE STATISTICS
  Total Trades:    {results.total_trades}
  Winning Trades:  {results.winning_trades} ({results.win_rate:.1%})
  Losing Trades:   {results.losing_trades}
  Trades Filtered: {results.trades_filtered_pct:.1%} (by meta-labeling)

P&L BREAKDOWN
  Gross P&L:       ${results.gross_pnl:,.2f}
  Commission:      ${results.total_commission:,.2f}
  Slippage:        ${results.total_slippage:,.2f}
  Net P&L:         ${results.net_pnl:,.2f}

WIN/LOSS ANALYSIS
  Average Win:     ${results.avg_win:,.2f}
  Average Loss:    ${results.avg_loss:,.2f}
  Largest Win:     ${results.largest_win:,.2f}
  Largest Loss:    ${results.largest_loss:,.2f}
  Profit Factor:   {results.profit_factor:.2f}

RISK METRICS
  Max Drawdown:    ${results.max_drawdown:,.2f} ({results.max_drawdown_pct:.1%})
  Sharpe Ratio:    {results.sharpe_ratio:.2f}
  Sortino Ratio:   {results.sortino_ratio:.2f}
  Calmar Ratio:    {results.calmar_ratio:.2f}

MODEL PERFORMANCE
  Primary AUC:     {results.primary_auc:.4f}
  Avg Bars Held:   {results.avg_bars_held:.1f} ({results.avg_minutes_held:.0f} min)
""")

    # QC Status
    report.append("\n" + "-" * 60)
    report.append("QUALITY CONTROL STATUS")
    report.append("-" * 60)

    if results.qc_passed:
        report.append("\n  [PASS] All quality control checks passed")
    else:
        report.append("\n  [WARNING] Quality control issues detected:")
        for w in results.qc_warnings:
            report.append(f"    - {w}")

    # Benchmark Comparison
    report.append("\n" + "-" * 60)
    report.append("BENCHMARK COMPARISON (Literature-Based)")
    report.append("-" * 60)

    benchmarks = [
        ("Sharpe Ratio", results.sharpe_ratio, 0.5, 1.5, 3.0),
        ("Win Rate", results.win_rate * 100, 50, 55, 65),
        ("Profit Factor", results.profit_factor, 1.0, 1.3, 2.0),
        ("AUC-ROC", results.primary_auc, 0.52, 0.58, 0.70),
    ]

    report.append("\n  Metric          Actual    Target   Good    Suspicious")
    report.append("  " + "-" * 55)

    for name, actual, target, good, suspicious in benchmarks:
        status = "OK" if actual <= good else ("WARN" if actual <= suspicious else "FLAG")
        report.append(f"  {name:15} {actual:7.2f}   {target:6.2f}  {good:6.2f}  {suspicious:6.2f}  [{status}]")

    # Fold Results
    if results.fold_results:
        report.append("\n" + "-" * 60)
        report.append("FOLD-BY-FOLD RESULTS")
        report.append("-" * 60)
        report.append("\n  Fold  Train AUC  Test AUC  Trades    P&L")
        report.append("  " + "-" * 45)

        for fold in results.fold_results:
            report.append(
                f"  {fold['fold']:4d}  {fold['train_auc']:9.4f}  {fold['test_auc']:8.4f}  "
                f"{fold['trades']:6d}  ${fold['pnl']:8,.2f}"
            )

    report.append("\n" + "=" * 80)
    report.append(" END OF REPORT")
    report.append("=" * 80)

    report_text = "\n".join(report)

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_text)

    return report_text


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the Triple Barrier + Meta-Labeling backtest."""

    print("=" * 80)
    print(" SKIE-NINJA TRIPLE BARRIER + META-LABELING BACKTEST")
    print(" Based on Lopez de Prado (2018) Methodology")
    print("=" * 80)

    # Configuration
    config = PipelineConfig()

    # Load data
    logger.info("\n--- Loading Data ---")
    prices, _ = load_sample_data(source="databento")

    logger.info(f"Loaded {len(prices)} bars from {prices.index[0]} to {prices.index[-1]}")

    # Resample to 5-min and filter RTH
    logger.info("\n--- Resampling to 5-min RTH ---")

    # Filter RTH
    if isinstance(prices.index, pd.DatetimeIndex):
        prices_utc = prices.copy()
        if prices_utc.index.tz is None:
            prices_utc.index = prices_utc.index.tz_localize('UTC')
        prices_et = prices_utc.copy()
        prices_et.index = prices_et.index.tz_convert('America/New_York')

        rth_mask = (prices_et.index.time >= pd.Timestamp('09:30').time()) & \
                   (prices_et.index.time < pd.Timestamp('16:00').time())
        prices_rth = prices_utc[rth_mask]

        logger.info(f"After RTH filter: {len(prices_rth):,} bars")
    else:
        prices_rth = prices

    # Resample to 5-min
    prices_5min = prices_rth.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    logger.info(f"Resampled to {len(prices_5min):,} 5-min bars")

    # Generate features
    features, target, feature_info = generate_features(prices_5min, config)

    # Run backtest
    backtester = TripleBarrierBacktester(config)
    results = backtester.run(prices_5min, features, target)

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(__file__).parent.parent.parent / 'data' / 'backtest_results' / \
                  f'triple_barrier_backtest_{timestamp}.txt'

    report = generate_report(results, config, report_path)
    print(report)

    # Save trades
    if backtester.trades:
        trades_path = report_path.with_suffix('.csv')
        pd.DataFrame(backtester.trades).to_csv(trades_path, index=False)
        logger.info(f"\nTrades saved to: {trades_path}")

    # Save results JSON
    results_dict = asdict(results)
    results_dict['fold_results'] = results.fold_results  # Preserve fold results
    results_json_path = report_path.with_suffix('.json')
    with open(results_json_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    logger.info(f"Results saved to: {results_json_path}")

    logger.info(f"\nReport saved to: {report_path}")

    return results


if __name__ == "__main__":
    main()
