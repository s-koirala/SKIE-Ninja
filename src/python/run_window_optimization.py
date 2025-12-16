"""
Data-Driven Window Size Optimization
=====================================

Selects optimal train/test window sizes by minimizing IS-OOS performance gap.
Based on Lopez de Prado (2018) recommendations for avoiding overfitting.

Method:
1. For each window size combination, run walk-forward validation
2. Measure IS-OOS performance gap (Sharpe difference)
3. Select window with smallest gap and acceptable OOS performance

Usage:
    python run_window_optimization.py

Author: SKIE_Ninja Development Team
Created: 2025-12-15
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
import sys

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """Results for a single window configuration."""
    train_days: int
    test_days: int
    is_sharpe: float
    oos_sharpe: float
    gap: float
    gap_pct: float  # Percentage degradation
    n_folds: int
    is_auc: float
    oos_auc: float
    total_trades: int
    oos_win_rate: float


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load price and feature data."""
    # Look for preprocessed data first
    features_path = data_dir / 'processed' / 'features_5min.parquet'
    prices_path = data_dir / 'processed' / 'prices_5min.parquet'

    if features_path.exists() and prices_path.exists():
        features = pd.read_parquet(features_path)
        prices = pd.read_parquet(prices_path)
        logger.info(f"Loaded preprocessed data: {len(features):,} samples")
        return prices, features

    # Fallback: Load raw data and create minimal features
    raw_dir = data_dir / 'raw' / 'market'
    csv_files = list(raw_dir.glob('ES_*1min*.csv'))

    if not csv_files:
        raise FileNotFoundError(f"No ES data files found in {raw_dir}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.lower()
        dfs.append(df)

    prices = pd.concat(dfs, ignore_index=True)

    # Parse timestamp
    if 'ts_event' in prices.columns:
        prices['timestamp'] = pd.to_datetime(prices['ts_event'])
    elif 'timestamp' in prices.columns:
        prices['timestamp'] = pd.to_datetime(prices['timestamp'])
    else:
        prices['timestamp'] = pd.to_datetime(prices.iloc[:, 0])

    prices.set_index('timestamp', inplace=True)
    prices = prices.sort_index()

    # Resample to 5-min
    prices = prices.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Create minimal features for testing
    features = create_minimal_features(prices)

    logger.info(f"Created features from raw data: {len(features):,} samples")
    return prices, features


def create_minimal_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Create minimal feature set for window optimization testing."""
    features = pd.DataFrame(index=prices.index)

    # Returns
    for lag in [1, 5, 10, 20]:
        features[f'return_lag{lag}'] = prices['close'].pct_change(lag)

    # Volatility
    for period in [5, 10, 20]:
        features[f'rv_{period}'] = prices['close'].pct_change().rolling(period).std()

    # ATR
    tr = pd.concat([
        prices['high'] - prices['low'],
        abs(prices['high'] - prices['close'].shift(1)),
        abs(prices['low'] - prices['close'].shift(1))
    ], axis=1).max(axis=1)

    for period in [5, 14, 20]:
        features[f'atr_{period}'] = tr.rolling(period).mean()

    # Momentum
    for period in [5, 10, 20]:
        features[f'momentum_{period}'] = prices['close'].pct_change(period)

    # RSI
    delta = prices['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features['rsi_14'] = 100 - (100 / (1 + rs))

    # Volume
    features['volume_ratio_20'] = prices['volume'] / (prices['volume'].rolling(20).mean() + 1)

    # Target: Future volatility expansion (consistent with main strategy)
    future_rv = prices['close'].pct_change().rolling(12).std().shift(-12)
    current_rv = prices['close'].pct_change().rolling(12).std()
    features['target_vol_expansion'] = (future_rv > current_rv * 1.5).astype(int)

    return features.dropna()


def calculate_sharpe(returns: np.ndarray, trades_per_year: float = 2500) -> float:
    """Calculate annualized Sharpe ratio from trade returns."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return (np.mean(returns) / np.std(returns)) * np.sqrt(trades_per_year)


def run_single_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    prices_test: pd.DataFrame
) -> Tuple[float, float, float, int, float]:
    """
    Run single fold and return IS/OOS metrics.

    Returns:
        is_sharpe, oos_sharpe, oos_auc, n_trades, win_rate
    """
    import lightgbm as lgb

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'verbose': -1,
        'seed': 42
    }

    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=100)

    # Get predictions
    is_proba = model.predict(X_train_scaled)
    oos_proba = model.predict(X_test_scaled)

    # Calculate AUC
    try:
        is_auc = roc_auc_score(y_train, is_proba)
        oos_auc = roc_auc_score(y_test, oos_proba)
    except Exception:
        is_auc = 0.5
        oos_auc = 0.5

    # Simulate simple trades for Sharpe calculation
    # Long when prob > 0.55, Short when prob < 0.45
    is_returns = []
    for i, prob in enumerate(is_proba[:-1]):
        if prob > 0.55:
            ret = (y_train[i+1] - 0.5) * 0.01  # Simplified return
            is_returns.append(ret)
        elif prob < 0.45:
            ret = -(y_train[i+1] - 0.5) * 0.01
            is_returns.append(ret)

    oos_returns = []
    wins = 0
    for i, prob in enumerate(oos_proba[:-1]):
        if prob > 0.55:
            ret = (y_test[i+1] - 0.5) * 0.01
            oos_returns.append(ret)
            if ret > 0:
                wins += 1
        elif prob < 0.45:
            ret = -(y_test[i+1] - 0.5) * 0.01
            oos_returns.append(ret)
            if ret > 0:
                wins += 1

    is_sharpe = calculate_sharpe(np.array(is_returns)) if is_returns else 0
    oos_sharpe = calculate_sharpe(np.array(oos_returns)) if oos_returns else 0
    n_trades = len(oos_returns)
    win_rate = wins / n_trades if n_trades > 0 else 0

    return is_sharpe, oos_sharpe, oos_auc, n_trades, win_rate


def optimize_window_sizes(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    train_days_range: List[int] = [30, 45, 60, 90, 120],
    test_days_range: List[int] = [3, 5, 7, 10],
    min_oos_sharpe: float = 0.5,
    target_col: str = 'target_vol_expansion'
) -> Tuple[WindowResult, List[WindowResult]]:
    """
    Optimize train/test window sizes.

    Args:
        prices: OHLCV price data
        features: Feature matrix with target
        train_days_range: List of training window sizes (days)
        test_days_range: List of test window sizes (days)
        min_oos_sharpe: Minimum acceptable OOS Sharpe
        target_col: Target column name

    Returns:
        optimal: Best window configuration
        all_results: All tested configurations
    """
    logger.info("=" * 60)
    logger.info("WINDOW SIZE OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Train days to test: {train_days_range}")
    logger.info(f"Test days to test: {test_days_range}")

    # Get feature columns
    feature_cols = [c for c in features.columns if not c.startswith('target_')]

    # Align data
    common_idx = prices.index.intersection(features.index)
    prices = prices.loc[common_idx]
    features = features.loc[common_idx]

    bars_per_day = 78  # 5-min RTH bars
    embargo_bars = 20  # Fixed embargo

    results = []
    total_combos = len(train_days_range) * len(test_days_range)
    combo_idx = 0

    for train_days in train_days_range:
        for test_days in test_days_range:
            combo_idx += 1

            train_bars = train_days * bars_per_day
            test_bars = test_days * bars_per_day

            n_samples = len(prices)
            n_folds = (n_samples - train_bars - embargo_bars) // test_bars

            if n_folds < 3:
                logger.warning(f"Skipping train={train_days}, test={test_days}: insufficient folds ({n_folds})")
                continue

            logger.info(f"\n[{combo_idx}/{total_combos}] Testing train={train_days}d, test={test_days}d ({n_folds} folds)")

            fold_is_sharpes = []
            fold_oos_sharpes = []
            fold_oos_aucs = []
            total_trades = 0
            total_wins = 0

            # Run walk-forward
            for fold in range(min(n_folds, 20)):  # Cap at 20 folds for speed
                train_start = fold * test_bars
                train_end = train_start + train_bars
                test_start = train_end + embargo_bars
                test_end = min(test_start + test_bars, n_samples)

                if test_end > n_samples:
                    break

                X_train = features.iloc[train_start:train_end][feature_cols].values
                y_train = features.iloc[train_start:train_end][target_col].values
                X_test = features.iloc[test_start:test_end][feature_cols].values
                y_test = features.iloc[test_start:test_end][target_col].values
                prices_test = prices.iloc[test_start:test_end]

                try:
                    is_sharpe, oos_sharpe, oos_auc, n_trades, win_rate = run_single_fold(
                        X_train, y_train, X_test, y_test, prices_test
                    )

                    fold_is_sharpes.append(is_sharpe)
                    fold_oos_sharpes.append(oos_sharpe)
                    fold_oos_aucs.append(oos_auc)
                    total_trades += n_trades
                    total_wins += int(win_rate * n_trades)

                except Exception as e:
                    logger.warning(f"Fold {fold} failed: {e}")
                    continue

            if not fold_oos_sharpes:
                continue

            avg_is_sharpe = np.mean(fold_is_sharpes)
            avg_oos_sharpe = np.mean(fold_oos_sharpes)
            avg_oos_auc = np.mean(fold_oos_aucs)
            gap = avg_is_sharpe - avg_oos_sharpe
            gap_pct = gap / (avg_is_sharpe + 1e-10) * 100
            oos_win_rate = total_wins / total_trades if total_trades > 0 else 0

            result = WindowResult(
                train_days=train_days,
                test_days=test_days,
                is_sharpe=avg_is_sharpe,
                oos_sharpe=avg_oos_sharpe,
                gap=gap,
                gap_pct=gap_pct,
                n_folds=len(fold_oos_sharpes),
                is_auc=np.mean([roc_auc_score([1,0,1], [0.6,0.4,0.7]) for _ in range(1)]),  # Placeholder
                oos_auc=avg_oos_auc,
                total_trades=total_trades,
                oos_win_rate=oos_win_rate
            )
            results.append(result)

            logger.info(f"  IS Sharpe: {avg_is_sharpe:.2f}, OOS Sharpe: {avg_oos_sharpe:.2f}, "
                       f"Gap: {gap:.2f} ({gap_pct:.1f}%)")

    if not results:
        raise ValueError("No valid window configurations found")

    # Select optimal: smallest gap with acceptable OOS performance
    valid_results = [r for r in results if r.oos_sharpe >= min_oos_sharpe]

    if valid_results:
        optimal = min(valid_results, key=lambda x: x.gap)
        logger.info(f"\nOptimal (gap-minimizing): train={optimal.train_days}d, test={optimal.test_days}d")
    else:
        # Fallback: best OOS Sharpe if none meet threshold
        optimal = max(results, key=lambda x: x.oos_sharpe)
        logger.warning(f"\nNo config met min_oos_sharpe={min_oos_sharpe}. Using best OOS Sharpe.")
        logger.info(f"Selected: train={optimal.train_days}d, test={optimal.test_days}d")

    return optimal, results


def print_optimization_report(optimal: WindowResult, all_results: List[WindowResult]) -> str:
    """Generate optimization report."""
    lines = [
        "=" * 70,
        "WINDOW SIZE OPTIMIZATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "OPTIMAL CONFIGURATION",
        "-" * 40,
        f"Training Window:   {optimal.train_days} days",
        f"Test Window:       {optimal.test_days} days",
        f"IS Sharpe:         {optimal.is_sharpe:.3f}",
        f"OOS Sharpe:        {optimal.oos_sharpe:.3f}",
        f"IS-OOS Gap:        {optimal.gap:.3f} ({optimal.gap_pct:.1f}%)",
        f"OOS AUC:           {optimal.oos_auc:.3f}",
        f"Folds Tested:      {optimal.n_folds}",
        "",
        "ALL CONFIGURATIONS TESTED",
        "-" * 70,
        f"{'Train':>8} {'Test':>6} {'IS_Sharpe':>10} {'OOS_Sharpe':>11} {'Gap':>8} {'Gap%':>8}",
        "-" * 70,
    ]

    # Sort by gap
    sorted_results = sorted(all_results, key=lambda x: x.gap)

    for r in sorted_results:
        marker = " *" if r.train_days == optimal.train_days and r.test_days == optimal.test_days else ""
        lines.append(
            f"{r.train_days:>8}d {r.test_days:>5}d {r.is_sharpe:>10.3f} {r.oos_sharpe:>11.3f} "
            f"{r.gap:>8.3f} {r.gap_pct:>7.1f}%{marker}"
        )

    lines.extend([
        "",
        "INTERPRETATION",
        "-" * 40,
        "* Smaller gap = less overfitting",
        "* Optimal balances gap minimization with acceptable OOS performance",
        "* If gap > 50%, model may be overfit",
        "",
        "RECOMMENDATIONS",
        "-" * 40,
    ])

    if optimal.gap_pct < 20:
        lines.append("✓ Low gap indicates robust model generalization")
    elif optimal.gap_pct < 35:
        lines.append("⚠ Moderate gap - monitor OOS performance closely")
    else:
        lines.append("✗ High gap - consider simpler model or more data")

    if optimal.oos_sharpe >= 2.0:
        lines.append("✓ Strong OOS Sharpe suggests real predictive edge")
    elif optimal.oos_sharpe >= 1.0:
        lines.append("⚠ Moderate OOS Sharpe - edge present but modest")
    else:
        lines.append("✗ Low OOS Sharpe - edge may be insufficient")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    """Run window optimization analysis."""
    print("=" * 70)
    print("SKIE_NINJA WINDOW SIZE OPTIMIZATION")
    print("=" * 70)
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Load data
    data_dir = project_root / 'data'

    try:
        prices, features = load_data(data_dir)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.info("Creating synthetic data for demonstration...")

        # Create synthetic data for testing
        np.random.seed(42)
        n_samples = 50000

        dates = pd.date_range('2023-01-01', periods=n_samples, freq='5min')
        prices = pd.DataFrame({
            'open': 4000 + np.cumsum(np.random.randn(n_samples) * 0.5),
            'high': 4000 + np.cumsum(np.random.randn(n_samples) * 0.5) + 2,
            'low': 4000 + np.cumsum(np.random.randn(n_samples) * 0.5) - 2,
            'close': 4000 + np.cumsum(np.random.randn(n_samples) * 0.5),
            'volume': np.random.randint(1000, 10000, n_samples)
        }, index=dates)

        features = create_minimal_features(prices)

    # Run optimization
    optimal, all_results = optimize_window_sizes(
        prices,
        features,
        train_days_range=[30, 45, 60, 90, 120],
        test_days_range=[3, 5, 7, 10],
        min_oos_sharpe=0.5
    )

    # Print report
    report = print_optimization_report(optimal, all_results)
    print(report)

    # Save results
    output_dir = project_root / 'data' / 'validation_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save report
    report_file = output_dir / f'window_optimization_{datetime.now():%Y%m%d_%H%M%S}.txt'
    with open(report_file, 'w') as f:
        f.write(report)

    # Save results as CSV
    results_df = pd.DataFrame([{
        'train_days': r.train_days,
        'test_days': r.test_days,
        'is_sharpe': r.is_sharpe,
        'oos_sharpe': r.oos_sharpe,
        'gap': r.gap,
        'gap_pct': r.gap_pct,
        'n_folds': r.n_folds,
        'oos_auc': r.oos_auc,
        'total_trades': r.total_trades
    } for r in all_results])

    csv_file = output_dir / f'window_optimization_{datetime.now():%Y%m%d_%H%M%S}.csv'
    results_df.to_csv(csv_file, index=False)

    print(f"\nResults saved to:")
    print(f"  Report: {report_file}")
    print(f"  CSV: {csv_file}")

    return optimal, all_results


if __name__ == '__main__':
    optimal, results = main()
