"""
SKIE-Ninja Backtest Analysis Runner

Runs comprehensive walk-forward backtesting with:
1. Data loading and preprocessing
2. RTH filtering
3. Feature engineering
4. Model training with walk-forward validation
5. Comprehensive reporting

Usage:
    python run_backtest_analysis.py --model lightgbm --output data/backtest_results

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_resampler import DataResampler
from feature_engineering.feature_pipeline import build_feature_matrix
from backtesting.comprehensive_backtest import (
    BacktestConfig,
    run_comprehensive_backtest
)
from models.purged_cv_rnn_trainer import (
    PurgedCVConfig,
    train_rnn_with_purged_cv
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(
    data_path: str,
    timeframe: str = '5min',
    rth_only: bool = True
) -> pd.DataFrame:
    """Load and resample data to target timeframe."""
    logger.info(f"Loading data from {data_path}")

    # Load raw data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df):,} raw bars")

    # Resample if needed
    if timeframe != '1min':
        resampler = DataResampler()
        df = resampler.resample(df, timeframe, rth_only=rth_only)
        logger.info(f"Resampled to {len(df):,} {timeframe} bars (RTH={rth_only})")

    return df


def build_features(
    prices: pd.DataFrame,
    symbol: str = 'ES',
    include_macro: bool = False
) -> pd.DataFrame:
    """Build feature matrix from price data."""
    logger.info("Building feature matrix...")

    features = build_feature_matrix(
        prices,
        symbol=symbol,
        include_lagged=True,
        include_interactions=True,
        include_targets=True,
        include_macro=include_macro,
        include_sentiment=False,
        include_intermarket=False,
        include_alternative=False,
        dropna=True
    )

    logger.info(f"Feature matrix shape: {features.shape}")
    return features


def load_selected_features(
    rankings_path: str = 'data/processed/feature_rankings.csv',
    top_n: int = 75
) -> list:
    """Load top N features from rankings file."""
    try:
        rankings = pd.read_csv(rankings_path)
        selected = rankings['feature'].tolist()[:top_n]
        logger.info(f"Loaded {len(selected)} selected features")
        return selected
    except FileNotFoundError:
        logger.warning(f"Rankings file not found: {rankings_path}")
        return None


def run_gradient_boosting_backtest(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    selected_features: list,
    model_type: str = 'lightgbm',
    output_dir: str = 'data/backtest_results'
):
    """Run backtest with gradient boosting model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING {model_type.upper()} BACKTEST")
    logger.info(f"{'='*60}")

    # Configure backtest
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

        # Walk-forward parameters (optimal from grid search)
        train_days=180,
        test_days=5,
        embargo_bars=42,

        # Costs
        commission_per_side=2.50,
        slippage_ticks=0.5,

        # RTH enforcement
        rth_only=True,

        # Data leakage
        check_leakage=True
    )

    # Run backtest
    trades, metrics, report = run_comprehensive_backtest(
        prices=prices,
        features=features,
        target_col='target_direction_1',
        selected_features=selected_features,
        model_type=model_type,
        config=config,
        output_dir=output_dir
    )

    # Print report
    print("\n" + report)

    return trades, metrics


def run_rnn_cv_analysis(
    features: pd.DataFrame,
    selected_features: list,
    model_type: str = 'lstm',
    output_dir: str = 'data/models'
):
    """Run RNN training with Purged K-Fold CV."""
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING {model_type.upper()} WITH PURGED CV")
    logger.info(f"{'='*60}")

    # Configure for reduced overfitting
    config = PurgedCVConfig(
        n_splits=5,
        purge_bars=200,
        embargo_bars=42,

        # Reduced model complexity
        hidden_size=64,
        num_layers=1,
        dropout=0.5,

        # Training
        sequence_length=20,
        batch_size=128,
        learning_rate=0.0005,
        weight_decay=1e-4,
        epochs=30,
        early_stopping_patience=5,

        # Regularization
        use_batch_norm=True
    )

    # Run CV training
    results = train_rnn_with_purged_cv(
        features=features,
        target_col='target_direction_1',
        selected_features=selected_features,
        model_type=model_type,
        output_dir=output_dir,
        config=config
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"{model_type.upper()} PURGED CV RESULTS")
    print(f"{'='*60}")
    print(f"Mean AUC-ROC:  {results.mean_auc:.4f} (+/- {results.std_auc:.4f})")
    print(f"Mean Accuracy: {results.mean_accuracy:.4f} (+/- {results.std_accuracy:.4f})")
    print(f"Mean F1:       {results.mean_f1:.4f}")
    print(f"Mean Log Loss: {results.mean_log_loss:.4f}")
    print(f"Avg Epochs:    {results.avg_epochs:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='SKIE-Ninja Backtest Analysis'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/raw/market/ES_1min_databento.csv',
        help='Path to price data CSV'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['lightgbm', 'xgboost', 'randomforest', 'lstm', 'gru', 'all'],
        default='lightgbm',
        help='Model type to test'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/backtest_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='5min',
        help='Resampling timeframe (1min, 5min, 15min)'
    )
    parser.add_argument(
        '--features',
        type=int,
        default=75,
        help='Number of top features to use'
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("SKIE-NINJA BACKTEST ANALYSIS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data:       {args.data}")
    print(f"  Model:      {args.model}")
    print(f"  Timeframe:  {args.timeframe}")
    print(f"  Features:   {args.features}")
    print(f"  Output:     {args.output}")

    # Load and prepare data
    prices = load_and_prepare_data(args.data, args.timeframe)

    # Build features
    features = build_features(prices)

    # Load selected features
    selected_features = load_selected_features(top_n=args.features)
    if selected_features is None:
        # Use all non-target features
        selected_features = [c for c in features.columns if not c.startswith('target_')]

    # Filter to available features
    selected_features = [f for f in selected_features if f in features.columns]
    print(f"\nUsing {len(selected_features)} features")

    # Run backtests based on model selection
    if args.model in ['lightgbm', 'xgboost', 'randomforest']:
        run_gradient_boosting_backtest(
            prices, features, selected_features,
            model_type=args.model,
            output_dir=args.output
        )

    elif args.model in ['lstm', 'gru']:
        run_rnn_cv_analysis(
            features, selected_features,
            model_type=args.model,
            output_dir=args.output
        )

    elif args.model == 'all':
        # Run all models
        print("\n" + "="*70)
        print("RUNNING ALL MODELS")
        print("="*70)

        results_summary = {}

        # Gradient boosting models
        for model in ['lightgbm', 'xgboost']:
            try:
                trades, metrics = run_gradient_boosting_backtest(
                    prices, features, selected_features,
                    model_type=model,
                    output_dir=args.output
                )
                results_summary[model] = {
                    'total_pnl': metrics.total_net_pnl,
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor,
                    'sharpe': metrics.sharpe_ratio,
                    'max_dd': metrics.max_drawdown
                }
            except Exception as e:
                logger.error(f"Error running {model}: {e}")

        # RNN models
        for model in ['lstm', 'gru']:
            try:
                results = run_rnn_cv_analysis(
                    features, selected_features,
                    model_type=model,
                    output_dir=args.output
                )
                results_summary[model] = {
                    'mean_auc': results.mean_auc,
                    'std_auc': results.std_auc,
                    'mean_accuracy': results.mean_accuracy
                }
            except Exception as e:
                logger.error(f"Error running {model}: {e}")

        # Print summary
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)

        for model, stats in results_summary.items():
            print(f"\n{model.upper()}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    print("\n" + "="*70)
    print("BACKTEST ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
