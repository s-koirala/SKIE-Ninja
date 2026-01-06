"""
Feature Parity Verification: Python vs C#
==========================================

Generates reference feature values from Python and outputs expected values
for verifying C# calculations match exactly.

This script documents the EXACT feature calculations used in Python
for comparison against C# implementations.

Author: SKIE_Ninja Development Team
Date: 2026-01-06
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))


def generate_reference_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Generate features using EXACT Python calculations.

    This function documents the precise formulas for each feature.
    C# must replicate these exactly.
    """
    features = pd.DataFrame(index=prices.index)

    # ================================================================
    # 1. RETURN FEATURES (6 features)
    # Formula: (Close[t] - Close[t-lag]) / Close[t-lag]
    # ================================================================
    for lag in [1, 2, 3, 5, 10, 20]:
        features[f'return_lag{lag}'] = prices['close'].pct_change(lag)

    # ================================================================
    # 2. VOLATILITY FEATURES (12 features: 3 per period × 4 periods)
    # ATR = mean(TrueRange) over period
    # TrueRange = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    # RV = std(pct_change) over period (NOT ANNUALIZED)
    # ATR_pct = ATR / Close
    # ================================================================
    tr1 = prices['high'] - prices['low']
    tr2 = abs(prices['high'] - prices['close'].shift(1))
    tr3 = abs(prices['low'] - prices['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    for period in [5, 10, 14, 20]:
        # Realized Volatility: std of percent changes (NOT annualized)
        features[f'rv_{period}'] = prices['close'].pct_change().rolling(period).std()
        # ATR: simple moving average of True Range
        features[f'atr_{period}'] = tr.rolling(period).mean()
        # ATR as percent of close price
        features[f'atr_pct_{period}'] = features[f'atr_{period}'] / prices['close']

    # ================================================================
    # 3. PRICE POSITION FEATURES (9 features: 3 per period × 3 periods)
    # close_vs_high = (Close - Highest_High) / Close
    # close_vs_low = (Close - Lowest_Low) / Close
    # range_pct = (Highest_High - Lowest_Low) / Close
    # ================================================================
    for period in [10, 20, 50]:
        period_high = prices['high'].rolling(period).max()
        period_low = prices['low'].rolling(period).min()
        features[f'close_vs_high_{period}'] = (prices['close'] - period_high) / prices['close']
        features[f'close_vs_low_{period}'] = (prices['close'] - period_low) / prices['close']
        features[f'range_pct_{period}'] = (period_high - period_low) / prices['close']

    # ================================================================
    # 4. MOMENTUM FEATURES (6 features: 2 per period × 3 periods)
    # momentum = pct_change(period) = (Close[t] - Close[t-period]) / Close[t-period]
    # ma_dist = (Close - SMA) / SMA  <-- NORMALIZED BY MA, NOT CLOSE
    # ================================================================
    for period in [5, 10, 20]:
        # Momentum is percent change, NOT price difference
        features[f'momentum_{period}'] = prices['close'].pct_change(period)
        ma = prices['close'].rolling(period).mean()
        # MA distance normalized by MA (not by Close)
        features[f'ma_dist_{period}'] = (prices['close'] - ma) / ma

    # ================================================================
    # 5. RSI FEATURES (2 features)
    # Standard RSI formula
    # ================================================================
    for period in [7, 14]:
        delta = prices['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # ================================================================
    # 6. BOLLINGER BAND POSITION (1 feature)
    # bb_pct = (Close - Lower) / (Upper - Lower)
    # Upper = MA20 + 2*std, Lower = MA20 - 2*std
    # ================================================================
    mid = prices['close'].rolling(20).mean()
    std = prices['close'].rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    features['bb_pct_20'] = (prices['close'] - lower) / (upper - lower + 1e-10)

    # ================================================================
    # 7. VOLUME FEATURES (6 features: 2 per period × 3 periods)
    # volume_sma = SMA(Volume, period)
    # volume_ratio = Volume / volume_sma
    # ================================================================
    if 'volume' in prices.columns:
        for period in [5, 10, 20]:
            features[f'volume_sma_{period}'] = prices['volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = prices['volume'] / (features[f'volume_sma_{period}'] + 1)

    return features


def run_verification():
    """Run feature parity verification."""
    print("=" * 80)
    print("FEATURE PARITY VERIFICATION")
    print("Python Reference Implementation")
    print("=" * 80)

    # Load data
    from data_collection.ninjatrader_loader import load_sample_data
    prices, _ = load_sample_data(source="databento")

    # Filter RTH and resample to 5-min
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

    print(f"\nData: {len(prices)} bars")
    print(f"Range: {prices.index[0]} to {prices.index[-1]}")

    # Generate features
    features = generate_reference_features(prices)
    features = features.dropna()

    print(f"Valid samples: {len(features)}")

    # Print feature formulas and sample values
    print("\n" + "=" * 80)
    print("FEATURE REFERENCE VALUES (Sample Bar)")
    print("=" * 80)

    # Pick a sample bar in the middle
    sample_idx = len(features) // 2
    sample_row = features.iloc[sample_idx]
    sample_prices = prices.iloc[sample_idx - 50:sample_idx + 1]

    print(f"\nSample bar: {features.index[sample_idx]}")
    print(f"Close: {sample_prices.iloc[-1]['close']:.2f}")
    print(f"High: {sample_prices.iloc[-1]['high']:.2f}")
    print(f"Low: {sample_prices.iloc[-1]['low']:.2f}")
    print(f"Volume: {sample_prices.iloc[-1]['volume']:.0f}")

    print("\n--- Feature Values ---")
    expected_order = [
        'return_lag1', 'return_lag2', 'return_lag3', 'return_lag5', 'return_lag10', 'return_lag20',
        'rv_5', 'atr_5', 'atr_pct_5',
        'rv_10', 'atr_10', 'atr_pct_10',
        'rv_14', 'atr_14', 'atr_pct_14',
        'rv_20', 'atr_20', 'atr_pct_20',
        'close_vs_high_10', 'close_vs_low_10', 'range_pct_10',
        'close_vs_high_20', 'close_vs_low_20', 'range_pct_20',
        'close_vs_high_50', 'close_vs_low_50', 'range_pct_50',
        'momentum_5', 'ma_dist_5',
        'momentum_10', 'ma_dist_10',
        'momentum_20', 'ma_dist_20',
        'rsi_7', 'rsi_14',
        'bb_pct_20',
        'volume_sma_5', 'volume_ratio_5',
        'volume_sma_10', 'volume_ratio_10',
        'volume_sma_20', 'volume_ratio_20'
    ]

    for i, name in enumerate(expected_order):
        if name in sample_row:
            value = sample_row[name]
            print(f"  [{i:2d}] {name:20s} = {value:15.8f}")

    # Key verification points
    print("\n" + "=" * 80)
    print("KEY VERIFICATION POINTS")
    print("=" * 80)

    print("\n1. MOMENTUM CALCULATION:")
    print("   Python: pct_change(N) = (Close[t] - Close[t-N]) / Close[t-N]")
    print("   C# BUG: Close[0] - Close[N] (price difference, not fraction)")
    print("   C# FIX: (Close[0] - Close[N]) / Close[N]")

    print("\n2. REALIZED VOLATILITY:")
    print("   Python: rolling(N).std() on pct_change (NOT annualized)")
    print("   C# BUG: std * sqrt(252) (annualized)")
    print("   C# FIX: std only, NO sqrt(252)")

    print("\n3. MA_DIST NORMALIZATION:")
    print("   Python: (Close - MA) / MA")
    print("   C# BUG: (Close - MA) / Close")
    print("   C# FIX: Divide by MA, not Close")

    # Distribution analysis
    print("\n" + "=" * 80)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    key_features = ['rv_14', 'momentum_10', 'ma_dist_10', 'return_lag1']
    for feat in key_features:
        if feat in features.columns:
            values = features[feat].dropna()
            print(f"\n{feat}:")
            print(f"  Min:    {values.min():.8f}")
            print(f"  Max:    {values.max():.8f}")
            print(f"  Mean:   {values.mean():.8f}")
            print(f"  Std:    {values.std():.8f}")
            print(f"  P5:     {values.quantile(0.05):.8f}")
            print(f"  P95:    {values.quantile(0.95):.8f}")

    # Save reference values
    output_dir = project_root / 'data' / 'validation_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = {
        'feature_order': expected_order,
        'sample_values': {name: float(sample_row[name]) for name in expected_order if name in sample_row},
        'distributions': {
            feat: {
                'min': float(features[feat].min()),
                'max': float(features[feat].max()),
                'mean': float(features[feat].mean()),
                'std': float(features[feat].std())
            }
            for feat in expected_order if feat in features.columns
        }
    }

    with open(output_dir / 'feature_reference_values.json', 'w') as f:
        json.dump(reference, f, indent=2)

    print(f"\n\nReference values saved to: feature_reference_values.json")

    return features


if __name__ == '__main__':
    run_verification()
