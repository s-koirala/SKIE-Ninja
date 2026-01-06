"""
Run Canonical Validation on SKIE-Ninja Backtest Results

This script applies rigorous statistical validation methods to assess
whether the reported trading edge is statistically significant after
accounting for:

1. Multiple testing (81 threshold combinations tested)
2. High variance of single-path walk-forward
3. Selection bias from parameter optimization
4. Non-stationarity of return series

References:
- Lopez de Prado (2018). "Advances in Financial Machine Learning"
- Bailey et al. (2014). "The Probability of Backtest Overfitting"
- Bailey & Lopez de Prado (2014). "The Deflated Sharpe Ratio"

Author: SKIE_Ninja Development Team
Created: 2026-01-05
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

from quality_control.canonical_validation import (
    run_canonical_validation,
    quick_validation_check,
    calculate_proper_embargo,
    CPCVConfig,
    CombinatorialPurgedKFold,
    deflated_sharpe_ratio,
    expected_max_sharpe
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_backtest_results() -> pd.DataFrame:
    """Load existing backtest trade results."""
    results_dir = project_root / 'data' / 'backtest_results'

    # Find most recent ensemble trades file
    trade_files = list(results_dir.glob('ensemble_trades_*.csv'))
    oos_files = list(results_dir.glob('ensemble_oos_trades_*.csv'))

    all_trades = []

    for f in trade_files + oos_files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = f.name
            all_trades.append(df)
            logger.info(f"Loaded {len(df)} trades from {f.name}")
        except Exception as e:
            logger.warning(f"Could not load {f.name}: {e}")

    if not all_trades:
        logger.error("No trade files found!")
        return pd.DataFrame()

    combined = pd.concat(all_trades, ignore_index=True)
    logger.info(f"Total trades loaded: {len(combined)}")

    return combined


def calculate_daily_returns(trades: pd.DataFrame) -> np.ndarray:
    """Convert trade P&Ls to daily returns series."""
    if 'entry_time' not in trades.columns:
        logger.warning("No entry_time column, using row order")
        trades['entry_time'] = pd.date_range(
            start='2020-01-01', periods=len(trades), freq='5min'
        )

    trades['date'] = pd.to_datetime(trades['entry_time']).dt.date
    daily_pnl = trades.groupby('date')['net_pnl'].sum()

    # Convert to daily returns (assuming $100K starting capital)
    starting_capital = 100000
    daily_returns = daily_pnl / starting_capital

    logger.info(f"Daily returns series: {len(daily_returns)} days")
    logger.info(f"  Mean daily return: {daily_returns.mean():.6f}")
    logger.info(f"  Std daily return: {daily_returns.std():.6f}")

    return daily_returns.values


def main():
    """Run canonical validation on existing results."""
    print("=" * 80)
    print(" CANONICAL VALIDATION OF SKIE-NINJA BACKTEST RESULTS")
    print(" Applying Lopez de Prado (2018) & Bailey et al. (2014) Methods")
    print("=" * 80)

    # 1. Calculate proper embargo
    print("\n" + "=" * 60)
    print("1. EMBARGO CALCULATION")
    print("=" * 60)

    feature_lookbacks = [1, 2, 3, 5, 10, 14, 20, 50, 100, 200]  # All MA periods used
    label_horizons = [5, 10, 20, 30]  # All target horizons

    proper_embargo = calculate_proper_embargo(feature_lookbacks, label_horizons)

    print(f"\nCurrent embargo values in codebase:")
    print(f"  walk_forward_backtest.py: 42 bars")
    print(f"  ensemble_strategy.py: 20 bars")
    print(f"\nREQUIRED embargo: {proper_embargo} bars")
    print(f"\nDISCREPANCY: {proper_embargo - 42} bars under-specified")

    # 2. Load trade results
    print("\n" + "=" * 60)
    print("2. LOADING BACKTEST RESULTS")
    print("=" * 60)

    trades = load_backtest_results()

    if len(trades) == 0:
        print("\nNo trade data found. Creating synthetic example...")
        # Create synthetic data for demonstration
        np.random.seed(42)
        n_trades = 4500
        trades = pd.DataFrame({
            'entry_time': pd.date_range('2020-01-01', periods=n_trades, freq='30min'),
            'net_pnl': np.random.normal(10, 200, n_trades),  # ~$10 avg, $200 std
            'gross_pnl': np.random.normal(15, 200, n_trades),
        })

    # 3. Calculate daily returns
    print("\n" + "=" * 60)
    print("3. CALCULATING DAILY RETURNS")
    print("=" * 60)

    daily_returns = calculate_daily_returns(trades)

    # 4. Quick validation check
    print("\n" + "=" * 60)
    print("4. DEFLATED SHARPE RATIO ANALYSIS")
    print("=" * 60)

    # Number of trials from threshold optimization
    n_trials = 81  # 3x3x3x3 grid from run_ensemble_threshold_optimization.py

    quick_results = quick_validation_check(
        daily_returns,
        n_trials=n_trials,
        verbose=True
    )

    # 5. Expected Sharpe Analysis
    print("\n" + "=" * 60)
    print("5. EXPECTED SHARPE UNDER NULL HYPOTHESIS")
    print("=" * 60)

    print(f"\nIf testing {n_trials} random strategies with no skill:")
    expected_max = expected_max_sharpe(n_trials)
    print(f"  Expected max Sharpe (luck alone): {expected_max:.4f}")
    print(f"  Observed Sharpe:                  {quick_results['raw_sharpe']:.4f}")
    print(f"  Excess over expected:             {quick_results['raw_sharpe'] - expected_max:.4f}")

    if quick_results['raw_sharpe'] < expected_max * 1.5:
        print(f"\n  WARNING: Observed Sharpe is within 1.5x of expected max under null.")
        print(f"           This suggests the edge may be due to selection bias.")

    # 6. Full validation (if path data available)
    print("\n" + "=" * 60)
    print("6. FULL CANONICAL VALIDATION")
    print("=" * 60)

    trade_pnls = trades['net_pnl'].values if 'net_pnl' in trades.columns else None

    result = run_canonical_validation(
        returns=daily_returns,
        trade_pnls=trade_pnls,
        n_trials=n_trials,
        n_bootstrap=10000,
        random_state=42
    )

    # 7. Summary Report
    print("\n" + "=" * 80)
    print(" VALIDATION SUMMARY")
    print("=" * 80)

    print(result.summary())

    # 8. Recommendations
    print("\n" + "=" * 60)
    print("7. RECOMMENDATIONS")
    print("=" * 60)

    recommendations = []

    if quick_results['dsr_pvalue'] > 0.05:
        recommendations.append(
            "DSR p-value > 0.05: Edge NOT statistically significant after "
            "correcting for multiple testing. Consider reducing n_trials or "
            "improving strategy to increase true Sharpe."
        )

    if not quick_results['is_stationary']:
        recommendations.append(
            "Returns are non-stationary (ADF p > 0.05): Strategy performance "
            "may vary significantly across regimes. Consider regime detection "
            "and conditional trading."
        )

    if proper_embargo > 42:
        recommendations.append(
            f"Embargo is under-specified: Current = 42 bars, Required = {proper_embargo} bars. "
            f"This creates potential data leakage. MUST FIX before re-validation."
        )

    if quick_results['raw_sharpe'] > 2.5:
        recommendations.append(
            f"Sharpe > 2.5 is unusual for non-HFT strategies. Verify no "
            f"look-ahead bias, verify costs are realistic, verify data quality."
        )

    if len(recommendations) == 0:
        recommendations.append("All validation checks passed. Proceed with paper trading.")

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")

    # 9. Save results
    output_dir = project_root / 'data' / 'validation_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'canonical_validation_{timestamp}.txt'

    with open(output_file, 'w') as f:
        f.write("CANONICAL VALIDATION REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(result.summary())
        f.write("\n\nRECOMMENDATIONS:\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"\n{i}. {rec}\n")

    print(f"\n\nResults saved to: {output_file}")

    return result, quick_results


if __name__ == "__main__":
    result, quick_results = main()
