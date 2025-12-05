"""
Threshold Optimization for Ensemble Strategy

Optimizes entry/exit thresholds using walk-forward validation.
Uses 2023-2024 in-sample data for optimization.

Parameters to optimize:
- min_vol_expansion_prob: Technical volatility filter threshold
- min_sentiment_vol_prob: Sentiment volatility filter threshold
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
from dataclasses import replace

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from feature_engineering.multi_target_labels import MultiTargetLabeler
from strategy.ensemble_strategy import EnsembleStrategy, EnsembleConfig
from data_collection.ninjatrader_loader import load_sample_data
from data_collection.historical_sentiment_loader import HistoricalSentimentLoader

logging.basicConfig(level=logging.WARNING)  # Reduce noise during optimization
logger = logging.getLogger(__name__)


def run_quick_backtest_with_config(
    prices: pd.DataFrame,
    config: EnsembleConfig,
    max_folds: int = 8
) -> dict:
    """Run quick backtest with a specific config."""

    # Create strategy with config
    strategy = EnsembleStrategy(config)

    # Prepare data
    features, targets, prices_aligned = strategy.prepare_data(prices)

    bars_per_day = 78
    train_bars = config.train_days * bars_per_day
    test_bars = config.test_days * bars_per_day

    all_trades = []

    fold = 0
    start_idx = 0

    while start_idx + train_bars + config.embargo_bars + test_bars <= len(features):
        if fold >= max_folds:
            break

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
        try:
            metrics = strategy.train_models(
                X_train, targets_train,
                X_test, targets_test,
                list(features.columns)
            )
        except Exception as e:
            start_idx += test_bars
            continue

        # Simulate trades
        test_prices = prices_aligned.iloc[test_start:test_end]
        current_atr = features.iloc[test_start:test_end]['atr_14'].values

        for i in range(len(X_test)):
            try:
                should_trade, direction, contracts, tp_offset, sl_offset, debug = \
                    strategy.generate_signal(X_test[i], current_atr[i])

                if should_trade:
                    trade = strategy.simulate_trade(
                        test_prices, i, direction, contracts,
                        tp_offset, sl_offset, debug
                    )
                    if trade:
                        all_trades.append(trade)
            except Exception:
                continue

        start_idx += test_bars

    # Calculate metrics
    if not all_trades:
        return {'net_pnl': 0, 'trades': 0, 'win_rate': 0, 'profit_factor': 0, 'sharpe': 0}

    trades_df = pd.DataFrame(all_trades)
    net_pnl = trades_df['net_pnl'].sum()
    total_trades = len(trades_df)
    win_rate = (trades_df['net_pnl'] > 0).mean()

    gross_profit = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
    gross_loss = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Convert entry_time to date for daily grouping
    trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date
    daily_pnl = trades_df.groupby('date')['net_pnl'].sum()
    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0

    return {
        'net_pnl': net_pnl,
        'trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe': sharpe
    }


def optimize_ensemble_thresholds():
    """Run grid search optimization for ensemble thresholds."""
    print("=" * 80)
    print(" ENSEMBLE THRESHOLD OPTIMIZATION")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    prices, _ = load_sample_data(source="databento")

    # Filter RTH and resample
    if hasattr(prices.index, 'hour'):
        prices = prices[
            (prices.index.hour >= 9) &
            ((prices.index.hour < 16) | ((prices.index.hour == 9) & (prices.index.minute >= 30)))
        ]

    prices = prices.resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()

    print(f"    Loaded {len(prices)} bars")

    # Define parameter grid - reduced for faster testing
    param_grid = {
        'min_vol_expansion_prob': [0.40, 0.50, 0.60],
        'min_breakout_prob': [0.45, 0.50, 0.55],
        'tp_atr_mult_base': [1.5, 2.0, 2.5],
        'sl_atr_mult_base': [0.75, 1.0, 1.25]
    }

    # Calculate total combinations
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)

    print(f"\n[2] Running optimization...")
    print(f"    Testing {total_combinations} parameter combinations")
    print(f"    Using 8 walk-forward folds per config")
    print(f"    This may take a while...")

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    results = []
    best_pnl = -float('inf')
    best_params = None

    start_time = datetime.now()

    for i, combo in enumerate(product(*param_values)):
        params = dict(zip(param_names, combo))

        # Create config with these parameters
        config = EnsembleConfig(
            ensemble_method='either',
            min_vol_expansion_prob=params['min_vol_expansion_prob'],
            min_sentiment_vol_prob=0.55,  # Keep default for sentiment
            min_breakout_prob=params['min_breakout_prob'],
            tp_atr_mult_base=params['tp_atr_mult_base'],
            sl_atr_mult_base=params['sl_atr_mult_base']
        )

        # Run quick backtest with this config
        metrics = run_quick_backtest_with_config(prices, config, max_folds=8)

        # Store results
        result = {**params, **metrics}
        results.append(result)

        # Track best
        if metrics['net_pnl'] > best_pnl:
            best_pnl = metrics['net_pnl']
            best_params = params.copy()

        # Progress update
        if (i + 1) % 10 == 0 or i == total_combinations - 1:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (i + 1) / elapsed if elapsed > 0 else 1
            remaining = (total_combinations - i - 1) / rate if rate > 0 else 0
            print(f"    Progress: {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.1f}%) "
                  f"- Best P&L: ${best_pnl:,.0f} - ETA: {remaining/60:.1f} min")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('net_pnl', ascending=False)

    # Save results
    output_dir = project_root / 'data' / 'optimization_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(output_dir / f'ensemble_optimization_{timestamp}.csv', index=False)

    # Print results
    print("\n" + "=" * 80)
    print(" OPTIMIZATION RESULTS")
    print("=" * 80)

    print("\n--- Top 10 Configurations ---")
    print(results_df.head(10).to_string(index=False))

    print("\n--- Best Configuration ---")
    best_row = results_df.iloc[0]
    print(f"  min_vol_expansion_prob: {best_row['min_vol_expansion_prob']}")
    print(f"  min_breakout_prob:      {best_row['min_breakout_prob']}")
    print(f"  tp_atr_mult_base:       {best_row['tp_atr_mult_base']}")
    print(f"  sl_atr_mult_base:       {best_row['sl_atr_mult_base']}")
    print(f"\n  Net P&L:      ${best_row['net_pnl']:,.2f}")
    print(f"  Trades:       {best_row['trades']:.0f}")
    print(f"  Win Rate:     {best_row['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {best_row['profit_factor']:.2f}")
    print(f"  Sharpe:       {best_row['sharpe']:.2f}")

    print("\n--- Comparison to Default ---")
    default_row = results_df[
        (results_df['min_vol_expansion_prob'] == 0.50) &
        (results_df['min_breakout_prob'] == 0.50) &
        (results_df['tp_atr_mult_base'] == 2.0) &
        (results_df['sl_atr_mult_base'] == 1.0)
    ]

    if len(default_row) > 0:
        default_pnl = default_row.iloc[0]['net_pnl']
        if default_pnl != 0:
            improvement = best_row['net_pnl'] - default_pnl
            print(f"  Default P&L:  ${default_pnl:,.2f}")
            print(f"  Best P&L:     ${best_row['net_pnl']:,.2f}")
            print(f"  Improvement:  ${improvement:,.2f} ({improvement/default_pnl*100:+.1f}%)")
        else:
            print(f"  Default P&L:  ${default_pnl:,.2f}")
            print(f"  Best P&L:     ${best_row['net_pnl']:,.2f}")

    print(f"\nResults saved to: {output_dir / f'ensemble_optimization_{timestamp}.csv'}")

    return results_df, best_params


if __name__ == "__main__":
    results_df, best_params = optimize_ensemble_thresholds()
