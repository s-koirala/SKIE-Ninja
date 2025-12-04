"""
SKIE-Ninja Validated Backtest Runner

Comprehensive backtest execution with full quality control validation.

Methodology:
============
1. Data Loading & Validation
   - Load ES 1-minute data from Databento
   - Validate OHLCV data quality
   - Resample to 5-minute RTH bars

2. Feature Engineering & Validation
   - Build feature matrix using pipeline
   - Validate for data leakage
   - Select top 75 features from rankings

3. Walk-Forward Backtest
   - Train: 180 days rolling window
   - Test: 5 days
   - Embargo: 42 bars (~3.5 hours)
   - RTH only: 9:30 AM - 4:00 PM ET

4. Quality Control Validation
   - Data quality checks
   - Feature leakage detection
   - Backtest realism checks
   - Statistical validation

5. Report Generation
   - Comprehensive trade log
   - Performance metrics (Sharpe, Sortino, Calmar)
   - Quality control report

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'python'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'data' / 'backtest_results' / 'backtest_log.txt', mode='w')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run validated backtest."""

    print("=" * 80)
    print("SKIE-NINJA VALIDATED BACKTEST")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Create output directory
    output_dir = PROJECT_ROOT / 'data' / 'backtest_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ========================================================================
    # STEP 1: Load and Validate Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING & VALIDATION")
    print("=" * 80)

    data_path = PROJECT_ROOT / 'data' / 'raw' / 'market' / 'ES_1min_databento.csv'

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        print(f"ERROR: Data file not found: {data_path}")
        return

    logger.info(f"Loading data from {data_path}")
    prices_raw = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(prices_raw):,} raw 1-minute bars")

    print(f"\nRaw Data Summary:")
    print(f"  Bars: {len(prices_raw):,}")
    print(f"  Date Range: {prices_raw.index[0]} to {prices_raw.index[-1]}")
    print(f"  Columns: {list(prices_raw.columns)}")

    # ========================================================================
    # STEP 2: Resample to 5-min RTH
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: RESAMPLING TO 5-MIN RTH")
    print("=" * 80)

    try:
        from utils.data_resampler import DataResampler
        resampler = DataResampler()
        prices = resampler.resample(prices_raw, '5min', rth_only=True)
        logger.info(f"Resampled to {len(prices):,} 5-min RTH bars")
    except ImportError as e:
        logger.warning(f"DataResampler not available: {e}")
        # Simple fallback resampling
        prices = prices_raw.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        logger.info(f"Fallback resampled to {len(prices):,} 5-min bars")

    print(f"\nResampled Data:")
    print(f"  Bars: {len(prices):,}")
    print(f"  Date Range: {prices.index[0]} to {prices.index[-1]}")

    # ========================================================================
    # STEP 3: Build Features
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 80)

    try:
        from feature_engineering.feature_pipeline import build_feature_matrix
        features = build_feature_matrix(
            prices,
            symbol='ES',
            include_lagged=True,
            include_interactions=True,
            include_targets=True,
            include_macro=False,
            include_sentiment=False,
            include_intermarket=False,
            include_alternative=False,
            dropna=True
        )
        logger.info(f"Built {features.shape[1]} features for {features.shape[0]} samples")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        print(f"ERROR: Feature engineering failed: {e}")
        return

    print(f"\nFeature Matrix:")
    print(f"  Shape: {features.shape}")
    print(f"  Target column: target_direction_1")

    # Load selected features
    rankings_path = PROJECT_ROOT / 'data' / 'processed' / 'feature_rankings.csv'
    if rankings_path.exists():
        rankings = pd.read_csv(rankings_path)
        selected_features = rankings['feature'].tolist()[:75]
        # Filter to available features
        selected_features = [f for f in selected_features if f in features.columns]
        logger.info(f"Using {len(selected_features)} selected features")
    else:
        selected_features = [c for c in features.columns if not c.startswith('target_')][:75]
        logger.warning(f"Using default feature selection ({len(selected_features)} features)")

    print(f"  Selected features: {len(selected_features)}")

    # ========================================================================
    # STEP 4: Data Validation
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: DATA QUALITY VALIDATION")
    print("=" * 80)

    try:
        from quality_control.validation_framework import (
            DataValidator, ValidationConfig, run_full_validation
        )

        config = ValidationConfig()
        validator = DataValidator(config)

        # Validate OHLCV
        data_result = validator.validate_ohlcv(prices)
        print(f"\nOHLCV Validation: {'PASSED' if data_result['passed'] else 'FAILED'}")
        for check, result in data_result.get('checks', {}).items():
            status = "✓" if result.get('passed', False) else "✗"
            print(f"  {status} {check}")

        # Validate features
        feature_result = validator.validate_features(features, 'target_direction_1')
        print(f"\nFeature Validation: {'PASSED' if feature_result['passed'] else 'FAILED'}")
        for check, result in feature_result.get('checks', {}).items():
            status = "✓" if result.get('passed', False) else "✗"
            print(f"  {status} {check}")

        if feature_result.get('issues'):
            print("\nISSUES:")
            for issue in feature_result['issues']:
                print(f"  ❌ {issue}")

        if feature_result.get('warnings'):
            print("\nWARNINGS:")
            for warning in feature_result['warnings'][:5]:
                print(f"  ⚠️ {warning}")

    except ImportError as e:
        logger.warning(f"Validation framework not available: {e}")
        print("Skipping detailed validation (module not found)")

    # ========================================================================
    # STEP 5: Run Backtest
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: WALK-FORWARD BACKTEST")
    print("=" * 80)

    try:
        from backtesting.comprehensive_backtest import (
            BacktestConfig, run_comprehensive_backtest
        )

        config = BacktestConfig(
            # Position sizing
            contracts_per_trade=1,

            # Entry/Exit rules
            long_threshold=0.55,
            short_threshold=0.45,
            hold_bars=3,
            stop_loss_ticks=20,
            take_profit_ticks=40,

            # Risk management
            max_daily_trades=10,
            max_daily_loss=1000.0,

            # Walk-forward (optimal from grid search)
            train_days=180,
            test_days=5,
            embargo_bars=42,
            bars_per_day=78,

            # Costs
            commission_per_side=2.50,
            slippage_ticks=0.5,

            # RTH enforcement
            rth_only=True,

            # Data leakage
            check_leakage=True
        )

        trades, metrics, report = run_comprehensive_backtest(
            prices=prices,
            features=features,
            target_col='target_direction_1',
            selected_features=selected_features,
            model_type='lightgbm',
            config=config,
            output_dir=str(output_dir)
        )

        # Print report
        print(report)

        # Save additional artifacts
        metrics_dict = metrics.to_dict()

        # Save metrics summary
        with open(output_dir / f'metrics_summary_{timestamp}.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)

    except ImportError as e:
        logger.error(f"Backtest module not available: {e}")
        print(f"ERROR: {e}")

        # Fallback: Run simpler backtest
        print("\nRunning simplified backtest...")
        trades = []
        metrics = None
        report = "Simplified backtest - module import failed"

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # STEP 6: Quality Control Validation
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: BACKTEST QUALITY VALIDATION")
    print("=" * 80)

    try:
        from quality_control.validation_framework import (
            BacktestValidator, run_full_validation
        )

        # Get trades as DataFrame
        if trades:
            trades_df = pd.DataFrame([t.to_dict() for t in trades])

            backtest_validator = BacktestValidator()
            bt_validation = backtest_validator.validate_trades(trades_df)

            print(f"\nBacktest Validation: {'PASSED' if bt_validation['passed'] else 'FAILED'}")
            for check, result in bt_validation.get('checks', {}).items():
                status = "✓" if result.get('passed', False) else "✗"
                value = result.get('value', result.get('count', 'N/A'))
                print(f"  {status} {check}: {value}")

            if bt_validation.get('issues'):
                print("\nISSUES:")
                for issue in bt_validation['issues']:
                    print(f"  ❌ {issue}")

            if bt_validation.get('warnings'):
                print("\nWARNINGS:")
                for warning in bt_validation['warnings']:
                    print(f"  ⚠️ {warning}")

            # Run full validation
            qc_report = run_full_validation(
                prices=prices,
                features=features,
                target_col='target_direction_1',
                trades_df=trades_df,
                metrics=metrics.to_dict() if metrics else None,
                output_path=str(output_dir / f'qc_report_{timestamp}.txt')
            )

            print(f"\nQuality Control Report saved to: {output_dir / f'qc_report_{timestamp}.txt'}")

    except Exception as e:
        logger.warning(f"Validation failed: {e}")
        print(f"Validation error: {e}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("BACKTEST EXECUTION SUMMARY")
    print("=" * 80)

    if metrics:
        print(f"""
Results Summary:
---------------
Total Trades:        {metrics.total_trades:,}
Win Rate:            {metrics.win_rate*100:.1f}%
Net P&L:             ${metrics.total_net_pnl:,.2f}
Profit Factor:       {metrics.profit_factor:.2f}
Sharpe Ratio:        {metrics.sharpe_ratio:.3f}
Max Drawdown:        ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct*100:.1f}%)
Avg Trade Duration:  {metrics.avg_bars_held:.1f} bars ({metrics.avg_time_held_minutes:.1f} min)

Model Performance:
-----------------
Accuracy:            {metrics.accuracy*100:.2f}%
AUC-ROC:             {metrics.auc_roc*100:.2f}%

Output Files:
------------
  - {output_dir / f'trades_lightgbm_{timestamp}.csv'}
  - {output_dir / f'metrics_lightgbm_{timestamp}.json'}
  - {output_dir / f'report_lightgbm_{timestamp}.txt'}
  - {output_dir / f'equity_lightgbm_{timestamp}.csv'}
  - {output_dir / f'qc_report_{timestamp}.txt'}
""")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
