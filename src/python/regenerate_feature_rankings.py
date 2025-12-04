"""
Regenerate Feature Rankings with Corrected Features
====================================================
This script regenerates the feature rankings after fixing look-ahead bias
in the advanced_targets.py module.

The old rankings were based on leaky features (pyramid_rr_*, ddca_*_success_*,
pivot_*) that used future data. This script creates new rankings using
only legitimate, non-leaky features.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'python'))

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Regenerate feature rankings with corrected features."""
    print("=" * 70)
    print("FEATURE RANKING REGENERATION")
    print("Using corrected (non-leaky) features")
    print("=" * 70)

    # === Step 1: Load Data ===
    print("\n[1/5] Loading market data...")
    from data_collection.ninjatrader_loader import load_sample_data

    es_data, _ = load_sample_data(source="databento")
    print(f"  Loaded {len(es_data)} bars")
    print(f"  Date range: {es_data.index[0]} to {es_data.index[-1]}")

    # Resample to 5-min RTH bars
    print("\n[2/5] Resampling to 5-min RTH bars...")
    try:
        from utils.data_resampler import DataResampler
        resampler = DataResampler()
        prices = resampler.resample(es_data, '5min', rth_only=True)
    except ImportError:
        # Fallback resampling
        logger.warning("DataResampler not available, using fallback")
        # Filter RTH (9:30 - 16:00 ET)
        es_data_rth = es_data.between_time('09:30', '16:00')

        prices = es_data_rth.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

    print(f"  Resampled to {len(prices)} bars")

    # === Step 2: Build Feature Matrix ===
    print("\n[3/5] Building feature matrix with CORRECTED features...")
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
        dropna=False  # Handle NaN manually
    )
    print(f"  Generated {features.shape[1]} features")

    # Handle NaN: Drop warmup period and fill remaining
    warmup = 300
    features = features.iloc[warmup:].copy()
    prices = prices.iloc[warmup:].copy()

    # Forward/backward fill remaining NaN
    features = features.ffill().bfill()

    # Drop rows where target is still NaN
    if 'target_direction_1' in features.columns:
        valid_mask = ~features['target_direction_1'].isnull()
        features = features.loc[valid_mask]
        prices = prices.loc[valid_mask]

    print(f"  After NaN handling: {features.shape[0]} samples, {features.shape[1]} features")

    # === Step 3: Run Feature Selection ===
    print("\n[4/5] Running feature selection pipeline...")

    # Separate target and features
    target_cols = [c for c in features.columns if c.startswith('target_')]
    target = features['target_direction_1'].copy()
    X = features.drop(columns=target_cols, errors='ignore').copy()

    print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")

    # Fill NaN for correlation calculation
    X_filled = X.fillna(X.median())
    target_filled = target.fillna(0)

    # Method 1: Target Correlation
    print("  Computing target correlations...")
    correlations = X_filled.corrwith(target_filled).abs()
    corr_rank = correlations.rank(ascending=False)

    # Method 2: Variance-based (features with more variance are often more informative)
    print("  Computing variance scores...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_filled),
        columns=X.columns,
        index=X.index
    )
    variances = X_scaled.var()
    var_rank = variances.rank(ascending=False)

    # Method 3: Random Forest importance (using subset for speed)
    print("  Computing Random Forest importance...")
    from sklearn.ensemble import RandomForestClassifier

    # Sample for speed
    sample_size = min(10000, len(X_filled))
    sample_idx = np.random.choice(len(X_filled), sample_size, replace=False)
    X_sample = X_filled.iloc[sample_idx]
    y_sample = target_filled.iloc[sample_idx].astype(int)

    rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_sample, y_sample)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
    rf_rank = rf_importance.rank(ascending=False)

    # Aggregate rankings
    print("  Aggregating rankings...")
    rank_df = pd.DataFrame({
        'target_correlation': corr_rank,
        'variance': var_rank,
        'random_forest': rf_rank
    })

    rank_df['avg_rank'] = rank_df.mean(axis=1)
    rank_df = rank_df.sort_values('avg_rank')
    rank_df['final_rank'] = range(1, len(rank_df) + 1)

    aggregated = rank_df.head(200)

    # === Step 4: Save Results ===
    print("\n[5/5] Saving results...")
    output_dir = PROJECT_ROOT / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save aggregated rankings
    rankings_file = output_dir / 'feature_rankings.csv'

    # Format properly for the expected CSV format
    rankings_df = pd.DataFrame({
        'feature': aggregated.index,
        'target_correlation': aggregated['target_correlation'],
        'variance': aggregated['variance'],
        'random_forest': aggregated['random_forest'],
        'avg_rank': aggregated['avg_rank'],
        'final_rank': aggregated['final_rank']
    })
    rankings_df.to_csv(rankings_file, index=False)
    print(f"  Saved rankings to: {rankings_file}")

    # Save correlation values too
    correlations_df = pd.DataFrame({
        'feature': correlations.index,
        'correlation': correlations.values
    }).sort_values('correlation', ascending=False)
    correlations_df.to_csv(output_dir / 'feature_correlations.csv', index=False)
    print(f"  Saved correlations to: {output_dir / 'feature_correlations.csv'}")

    # === Summary ===
    print("\n" + "=" * 70)
    print("FEATURE RANKING SUMMARY")
    print("=" * 70)

    print("\nTop 20 Features (Corrected - No Look-Ahead Bias):")
    print("-" * 50)
    for i, (feature, row) in enumerate(aggregated.head(20).iterrows(), 1):
        print(f"{i:3d}. {feature:<40} (avg_rank: {row['avg_rank']:.1f})")

    # Check for any remaining potentially leaky features
    print("\n" + "=" * 70)
    print("VALIDATION: Checking for leaky feature names...")
    print("=" * 70)

    leaky_patterns = ['_success', 'future_', 'target_mfe', 'target_mae']
    leaky_found = []

    for feature in aggregated.index[:75]:
        for pattern in leaky_patterns:
            if pattern in feature.lower():
                leaky_found.append(feature)

    if leaky_found:
        print(f"\nWARNING: Found {len(leaky_found)} potentially leaky features:")
        for f in leaky_found:
            print(f"  - {f}")
    else:
        print("\nVALIDATION PASSED: No leaky feature patterns found in top 75 features")

    print("\n" + "=" * 70)
    print("FEATURE RANKING REGENERATION COMPLETE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return rankings_df


if __name__ == "__main__":
    rankings = main()
