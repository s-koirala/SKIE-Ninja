"""
Threshold Optimization for Volatility Breakout Strategy

Optimizes entry/exit thresholds using walk-forward validation.
Uses 2023-2024 in-sample data for optimization.

Parameters to optimize:
- min_vol_expansion_prob: Volatility filter threshold
- min_breakout_prob: Breakout filter threshold
- tp_atr_mult_base: Take profit ATR multiplier
- sl_atr_mult_base: Stop loss ATR multiplier

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from itertools import product
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from feature_engineering.multi_target_labels import MultiTargetLabeler
from strategy.volatility_breakout_strategy import (
    VolatilityBreakoutStrategy, StrategyConfig, BacktestResults
)
from data_collection.ninjatrader_loader import load_sample_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_backtest_with_config(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    config: StrategyConfig,
    max_folds: int = 20
) -> BacktestResults:
    """Run backtest with specific configuration."""

    strategy = VolatilityBreakoutStrategy(config)

    bars_per_day = 78
    train_bars = config.train_days * bars_per_day
    test_bars = config.test_days * bars_per_day

    all_trades = []
    all_metrics = []

    fold = 0
    start_idx = 0

    while start_idx + train_bars + config.embargo_bars + test_bars <= len(features):
        if fold >= max_folds:
            break

        fold += 1

        train_end = start_idx + train_bars
        test_start = train_end + config.embargo_bars
        test_end = test_start + test_bars

        X_train = features.iloc[start_idx:train_end].values
        targets_train = targets.iloc[start_idx:train_end]
        X_test = features.iloc[test_start:test_end].values
        targets_test = targets.iloc[test_start:test_end]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        try:
            metrics = strategy.train_models(
                X_train_scaled, targets_train,
                X_test_scaled, targets_test
            )
        except Exception:
            start_idx += test_bars
            continue

        strategy.scaler = scaler

        test_prices = prices.iloc[test_start:test_end]
        current_atr = features.iloc[test_start:test_end]['atr_14'].values

        for i in range(len(X_test_scaled)):
            try:
                should_trade, direction, contracts, tp_offset, sl_offset, vol_prob = \
                    strategy.generate_signal(X_test_scaled[i], current_atr[i])

                if should_trade:
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
            except Exception:
                continue

        all_metrics.append(metrics)
        start_idx += test_bars

    results = strategy.calculate_metrics(all_trades)

    if all_metrics:
        results.vol_model_auc = np.mean([m.get('vol_auc', 0) for m in all_metrics])
        results.breakout_model_auc = np.mean([
            (m.get('high_auc', 0) + m.get('low_auc', 0)) / 2 for m in all_metrics
        ])

    return results


def run_optimization():
    """Run threshold optimization."""
    print("=" * 80)
    print(" THRESHOLD OPTIMIZATION")
    print(" Finding optimal entry/exit parameters")
    print("=" * 80)

    # Load data
    logger.info("\n--- Loading In-Sample Data (2023-2024) ---")
    prices, _ = load_sample_data(source="databento")

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

    logger.info(f"Total bars: {len(prices)}")

    # Generate features and targets once
    logger.info("\n--- Generating Features & Targets ---")
    base_config = StrategyConfig()
    strategy = VolatilityBreakoutStrategy(base_config)

    features = strategy.generate_features(prices)
    targets = strategy.target_labeler.generate_all_targets(prices)

    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]
    prices_aligned = prices.loc[common_idx]

    valid_mask = ~(features.isna().any(axis=1) | targets.isna().any(axis=1))
    features = features[valid_mask]
    targets = targets[valid_mask]
    prices_aligned = prices_aligned[valid_mask]

    logger.info(f"Valid samples: {len(features)}")

    # Define parameter grid
    param_grid = {
        'min_vol_expansion_prob': [0.45, 0.50, 0.55, 0.60],
        'min_breakout_prob': [0.45, 0.50, 0.55, 0.60],
        'tp_atr_mult_base': [1.5, 2.0, 2.5, 3.0],
        'sl_atr_mult_base': [0.75, 1.0, 1.25, 1.5],
    }

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    print(f"\n--- Testing {len(combinations)} Parameter Combinations ---")
    print(f"Parameters: {param_names}")

    results_list = []
    best_sharpe = -np.inf
    best_config = None
    best_results = None

    for i, combo in enumerate(combinations):
        # Create config with current parameters
        config = StrategyConfig(
            min_vol_expansion_prob=combo[0],
            min_breakout_prob=combo[1],
            tp_atr_mult_base=combo[2],
            sl_atr_mult_base=combo[3],
        )

        # Run backtest (limited folds for speed)
        results = run_backtest_with_config(
            prices_aligned, features, targets, config, max_folds=15
        )

        # Store results
        result_dict = {
            'vol_prob': combo[0],
            'breakout_prob': combo[1],
            'tp_mult': combo[2],
            'sl_mult': combo[3],
            'net_pnl': results.net_pnl,
            'total_trades': results.total_trades,
            'win_rate': results.win_rate,
            'profit_factor': results.profit_factor,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown': results.max_drawdown,
        }
        results_list.append(result_dict)

        # Track best
        if results.sharpe_ratio > best_sharpe and results.total_trades >= 100:
            best_sharpe = results.sharpe_ratio
            best_config = combo
            best_results = results

        # Progress update
        if (i + 1) % 16 == 0 or i == 0:
            print(f"  Tested {i + 1}/{len(combinations)}: "
                  f"vol={combo[0]}, breakout={combo[1]}, "
                  f"tp={combo[2]}, sl={combo[3]} -> "
                  f"PnL=${results.net_pnl:,.0f}, Sharpe={results.sharpe_ratio:.2f}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    # Sort by Sharpe ratio (with minimum trades filter)
    valid_results = results_df[results_df['total_trades'] >= 100].copy()
    valid_results = valid_results.sort_values('sharpe_ratio', ascending=False)

    # Print top configurations
    print("\n" + "=" * 80)
    print(" TOP 10 CONFIGURATIONS (by Sharpe Ratio)")
    print("=" * 80)

    print(f"\n{'Rank':<6} {'Vol':<6} {'Brk':<6} {'TP':<6} {'SL':<6} "
          f"{'PnL':>12} {'Trades':>8} {'Win%':>7} {'PF':>6} {'Sharpe':>7}")
    print("-" * 80)

    for rank, (idx, row) in enumerate(valid_results.head(10).iterrows(), 1):
        print(f"{rank:<6} {row['vol_prob']:<6.2f} {row['breakout_prob']:<6.2f} "
              f"{row['tp_mult']:<6.1f} {row['sl_mult']:<6.2f} "
              f"${row['net_pnl']:>10,.0f} {row['total_trades']:>8.0f} "
              f"{row['win_rate']*100:>6.1f}% {row['profit_factor']:>5.2f} "
              f"{row['sharpe_ratio']:>6.2f}")

    # Best configuration
    print("\n" + "=" * 80)
    print(" OPTIMAL CONFIGURATION")
    print("=" * 80)

    if best_config:
        print(f"\n  min_vol_expansion_prob: {best_config[0]}")
        print(f"  min_breakout_prob:      {best_config[1]}")
        print(f"  tp_atr_mult_base:       {best_config[2]}")
        print(f"  sl_atr_mult_base:       {best_config[3]}")

        print(f"\n  Expected Performance:")
        print(f"    Net P&L:        ${best_results.net_pnl:,.2f}")
        print(f"    Total Trades:   {best_results.total_trades}")
        print(f"    Win Rate:       {best_results.win_rate*100:.1f}%")
        print(f"    Profit Factor:  {best_results.profit_factor:.2f}")
        print(f"    Sharpe Ratio:   {best_results.sharpe_ratio:.2f}")
        print(f"    Max Drawdown:   ${best_results.max_drawdown:,.2f}")

    # Compare to default
    print("\n" + "=" * 80)
    print(" COMPARISON: DEFAULT vs OPTIMIZED")
    print("=" * 80)

    default_row = results_df[
        (results_df['vol_prob'] == 0.50) &
        (results_df['breakout_prob'] == 0.50) &
        (results_df['tp_mult'] == 2.0) &
        (results_df['sl_mult'] == 1.0)
    ]

    if len(default_row) > 0 and best_config:
        default = default_row.iloc[0]
        best = valid_results.iloc[0]

        print(f"\n{'Metric':<20} {'Default':<15} {'Optimized':<15} {'Change':<15}")
        print("-" * 65)
        print(f"{'Net P&L':<20} ${default['net_pnl']:>12,.0f} ${best['net_pnl']:>12,.0f} "
              f"${best['net_pnl'] - default['net_pnl']:>+12,.0f}")
        print(f"{'Win Rate':<20} {default['win_rate']*100:>11.1f}% {best['win_rate']*100:>11.1f}% "
              f"{(best['win_rate'] - default['win_rate'])*100:>+11.1f}%")
        print(f"{'Profit Factor':<20} {default['profit_factor']:>12.2f} {best['profit_factor']:>12.2f} "
              f"{best['profit_factor'] - default['profit_factor']:>+12.2f}")
        print(f"{'Sharpe Ratio':<20} {default['sharpe_ratio']:>12.2f} {best['sharpe_ratio']:>12.2f} "
              f"{best['sharpe_ratio'] - default['sharpe_ratio']:>+12.2f}")

    # Save results
    output_dir = project_root / 'data' / 'optimization_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(output_dir / f'threshold_optimization_{timestamp}.csv', index=False)
    logger.info(f"\nResults saved to: {output_dir / f'threshold_optimization_{timestamp}.csv'}")

    return best_config, results_df


if __name__ == "__main__":
    run_optimization()
