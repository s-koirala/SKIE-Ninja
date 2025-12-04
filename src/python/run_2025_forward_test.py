"""
2025 Forward Test for Volatility Breakout Strategy

Tests the strategy on 2025 data - TRUE FORWARD TEST
This data was downloaded AFTER strategy development was complete.

Development: 2023-2024 data
OOS Validation: 2020-2022 data (PASSED: +$496K)
Forward Test: 2025 data (this script)

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
from strategy.volatility_breakout_strategy import (
    VolatilityBreakoutStrategy, StrategyConfig, TradeResult
)

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
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'open' in col_lower:
            col_map[col] = 'open'
        elif 'high' in col_lower:
            col_map[col] = 'high'
        elif 'low' in col_lower:
            col_map[col] = 'low'
        elif 'close' in col_lower:
            col_map[col] = 'close'
        elif 'volume' in col_lower:
            col_map[col] = 'volume'

    df.rename(columns=col_map, inplace=True)

    # Keep only OHLCV
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    available_cols = [c for c in required_cols if c in df.columns]
    df = df[available_cols]

    logger.info(f"Loaded {len(df)} bars")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

    return df


def run_forward_test():
    """Run forward test on 2025 data."""
    print("=" * 80)
    print(" 2025 FORWARD TEST")
    print(" Testing on 2025 Data (Downloaded AFTER strategy development)")
    print("=" * 80)

    # Load 2025 data
    logger.info("\n--- Loading 2025 Forward Test Data ---")
    prices = load_2025_data()
    logger.info(f"Total 2025 bars: {len(prices)}")

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
    logger.info("\n--- Walk-Forward Forward Test ---")

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
        try:
            metrics = strategy.train_models(
                X_train_scaled, targets_train,
                X_test_scaled, targets_test
            )
        except Exception as e:
            logger.warning(f"Fold {fold} training failed: {e}")
            start_idx += test_bars
            continue

        strategy.scaler = scaler

        # Simulate trades on test set
        test_prices = prices_aligned.iloc[test_start:test_end]
        current_atr = features.iloc[test_start:test_end]['atr_14'].values

        for i in range(len(X_test_scaled)):
            try:
                should_trade, direction, contracts, tp_offset, sl_offset, vol_prob = \
                    strategy.generate_signal(X_test_scaled[i], current_atr[i])

                if should_trade:
                    # Get breakout probability
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
            except Exception as e:
                continue

        all_metrics.append(metrics)

        if fold % 10 == 0 or fold == 1:
            logger.info(f"  Fold {fold}: Vol AUC={metrics.get('vol_auc', 0):.4f}, "
                       f"Trades so far: {len(all_trades)}")

        start_idx += test_bars

    # Calculate final metrics
    logger.info("\n--- Final Results ---")
    results = strategy.calculate_metrics(all_trades)

    # Average model metrics
    if all_metrics:
        results.vol_model_auc = np.mean([m.get('vol_auc', 0) for m in all_metrics])
        results.breakout_model_auc = np.mean([
            (m.get('high_auc', 0) + m.get('low_auc', 0)) / 2 for m in all_metrics
        ])
        results.atr_model_r2 = np.mean([m.get('atr_r2', 0) for m in all_metrics])

    # Print results
    print("\n" + "=" * 80)
    print(" 2025 FORWARD TEST RESULTS")
    print("=" * 80)

    print(f"\nData Period: 2025 (TRUE FORWARD TEST)")
    print(f"Strategy was developed on: 2023-2024 data")
    print(f"OOS validation was on: 2020-2022 data")

    print(f"\n--- Model Performance ---")
    print(f"  Vol Expansion AUC:  {results.vol_model_auc:.4f}")
    print(f"  Breakout AUC:       {results.breakout_model_auc:.4f}")
    print(f"  ATR Forecast R2:    {results.atr_model_r2:.4f}")

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

    # Compare to previous periods
    print("\n" + "=" * 80)
    print(" COMPARISON: ALL TEST PERIODS")
    print("=" * 80)

    print(f"\n{'Metric':<20} {'In-Sample':<15} {'OOS (2020-22)':<15} {'Forward (2025)':<15}")
    print("-" * 65)

    # Reference values
    is_net_pnl = 209351
    is_win_rate = 0.399
    is_profit_factor = 1.29
    is_sharpe = 3.22

    oos_net_pnl = 496380
    oos_win_rate = 0.404
    oos_profit_factor = 1.28
    oos_sharpe = 3.09

    print(f"{'Net P&L':<20} ${is_net_pnl:>13,} ${oos_net_pnl:>13,} ${results.net_pnl:>13,.0f}")
    print(f"{'Win Rate':<20} {is_win_rate*100:>12.1f}% {oos_win_rate*100:>12.1f}% {results.win_rate*100:>12.1f}%")
    print(f"{'Profit Factor':<20} {is_profit_factor:>13.2f} {oos_profit_factor:>13.2f} {results.profit_factor:>13.2f}")
    print(f"{'Sharpe Ratio':<20} {is_sharpe:>13.2f} {oos_sharpe:>13.2f} {results.sharpe_ratio:>13.2f}")

    # Validation verdict
    print("\n" + "=" * 80)
    print(" FORWARD TEST VERDICT")
    print("=" * 80)

    if results.net_pnl > 0 and results.profit_factor > 1.0:
        print("\n[PASS] Strategy is PROFITABLE on 2025 forward test!")
        print("       Edge confirmed across ALL time periods:")
        print("       - In-Sample (2023-24): +$209K")
        print("       - OOS (2020-22): +$496K")
        print(f"       - Forward (2025): +${results.net_pnl:,.0f}")
    elif results.net_pnl > 0:
        print("\n[WARN] Strategy is profitable but borderline on forward test.")
    else:
        print("\n[FAIL] Strategy is NOT profitable on 2025 forward test.")
        print("       Market conditions may have changed.")

    # Save results
    output_dir = project_root / 'data' / 'backtest_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save trades
    if all_trades:
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

        trades_df.to_csv(output_dir / f'forward_test_2025_trades_{timestamp}.csv', index=False)
        logger.info(f"\nTrades saved to: {output_dir / f'forward_test_2025_trades_{timestamp}.csv'}")

    return results


if __name__ == "__main__":
    run_forward_test()
