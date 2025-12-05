"""
2025 Forward Test for Ensemble Strategy

Tests the ensemble strategy (vol breakout + sentiment) on 2025 data.
This is TRUE forward test data - completely unseen during development.

Baseline to beat: Vol Breakout 2025 = +$57,394

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score
import lightgbm as lgb

from feature_engineering.multi_target_labels import MultiTargetLabeler
from strategy.ensemble_strategy import EnsembleStrategy, EnsembleConfig
from data_collection.historical_sentiment_loader import HistoricalSentimentLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_2025_data() -> pd.DataFrame:
    """Load 2025 forward test data."""
    data_dir = project_root / 'data' / 'raw' / 'market'
    file_path = data_dir / 'ES_2025_1min_databento.csv'

    if not file_path.exists():
        raise FileNotFoundError(f"2025 data not found: {file_path}")

    logger.info(f"Loading {file_path.name}...")
    df = pd.read_csv(file_path)

    # Parse timestamp
    if 'ts_event' in df.columns:
        df['timestamp'] = pd.to_datetime(df['ts_event'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        df['timestamp'] = pd.to_datetime(df.iloc[:, 0])

    df.set_index('timestamp', inplace=True)

    # Standardize columns
    col_mapping = {
        'open': 'open', 'high': 'high', 'low': 'low',
        'close': 'close', 'volume': 'volume'
    }

    for orig, new in col_mapping.items():
        if orig not in df.columns:
            for col in df.columns:
                if orig in col.lower():
                    df[new] = df[col]
                    break

    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    logger.info(f"Loaded {len(df)} bars for 2025")

    return df


def run_ensemble_2025_forward_test(method: str = 'either'):
    """Run 2025 forward test of ensemble strategy."""
    print("=" * 80)
    print(" ENSEMBLE 2025 FORWARD TEST")
    print(f" Method: {method.upper()}")
    print(" Baseline: Vol Breakout 2025 = +$57,394")
    print("=" * 80)

    # Load 2025 data
    logger.info("\n--- Loading 2025 Data ---")
    prices = load_2025_data()
    logger.info(f"Total 2025 bars: {len(prices)}")

    # Filter RTH and resample to 5min
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

    logger.info(f"After RTH + 5-min resample: {len(prices)} bars")

    # Initialize strategy
    config = EnsembleConfig(ensemble_method=method)
    strategy = EnsembleStrategy(config)

    # Prepare data (features + sentiment + targets)
    logger.info("\n--- Preparing Data ---")
    features, targets, prices_aligned = strategy.prepare_data(prices)

    # Walk-forward backtest
    logger.info("\n--- Walk-Forward 2025 Forward Test ---")

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

        all_metrics.append(metrics)

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

        if fold % 10 == 0:
            logger.info(f"  Fold {fold}: Tech Vol AUC={metrics['tech_vol_auc']:.3f}, "
                       f"Sent Vol AUC={metrics['sent_vol_auc']:.3f}, "
                       f"Trades: {len(all_trades)}")

        start_idx += test_bars

    # Calculate final metrics
    logger.info("\n--- 2025 Forward Test Results ---")
    results = strategy.calculate_metrics(all_trades)

    # Average model metrics
    avg_tech_vol = np.mean([m['tech_vol_auc'] for m in all_metrics])
    avg_sent_vol = np.mean([m['sent_vol_auc'] for m in all_metrics])
    avg_combined = np.mean([m['combined_vol_auc'] for m in all_metrics])
    avg_breakout = np.mean([(m['high_auc'] + m['low_auc']) / 2 for m in all_metrics])

    print("\n" + "=" * 80)
    print(" ENSEMBLE 2025 FORWARD TEST RESULTS")
    print("=" * 80)

    print(f"\nEnsemble Method: {config.ensemble_method}")

    print(f"\n--- Model Performance ---")
    print(f"  Technical Vol AUC:  {avg_tech_vol:.4f}")
    print(f"  Sentiment Vol AUC:  {avg_sent_vol:.4f}")
    print(f"  Combined Vol AUC:   {avg_combined:.4f}")
    print(f"  Breakout AUC:       {avg_breakout:.4f}")

    print(f"\n--- Trade Statistics ---")
    print(f"  Total Trades:       {results.get('total_trades', 0)}")
    print(f"  Win Rate:           {results.get('win_rate', 0)*100:.1f}%")
    print(f"  Avg Bars Held:      {results.get('avg_bars_held', 0):.1f}")

    print(f"\n--- P&L ---")
    print(f"  Net P&L:            ${results.get('net_pnl', 0):,.2f}")
    print(f"  Profit Factor:      {results.get('profit_factor', 0):.2f}")
    print(f"  Max Drawdown:       ${results.get('max_drawdown', 0):,.2f}")
    print(f"  Sharpe Ratio:       {results.get('sharpe_ratio', 0):.2f}")

    print(f"\n--- Comparison to Vol Breakout 2025 ---")
    baseline_2025 = 57394
    print(f"  Vol Breakout 2025:  +${baseline_2025:,}")
    print(f"  Ensemble 2025:      ${results.get('net_pnl', 0):,.2f}")

    improvement = results.get('net_pnl', 0) - baseline_2025
    pct_improvement = improvement / baseline_2025 * 100 if baseline_2025 > 0 else 0
    print(f"  Difference:         ${improvement:,.2f} ({pct_improvement:+.1f}%)")

    if improvement > 0:
        print(f"\n  [PASS] ENSEMBLE OUTPERFORMS on 2025 forward test!")
        print(f"    Recommendation: Optimize ensemble thresholds")
    else:
        print(f"\n  [FAIL] Ensemble underperforms on 2025 forward test")
        print(f"    Recommendation: Optimize vol breakout thresholds only")

    # Save results
    output_dir = project_root / 'data' / 'backtest_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv(output_dir / f'ensemble_2025_forward_trades_{timestamp}.csv', index=False)
    logger.info(f"\nTrades saved to: ensemble_2025_forward_trades_{timestamp}.csv")

    return results, all_metrics, improvement > 0


if __name__ == "__main__":
    results, metrics, outperforms = run_ensemble_2025_forward_test(method='either')
