"""
Multi-Target Predictability Analysis

This script evaluates the predictability of different target types:
1. Volatility targets (expected: most predictable)
2. Trend targets (expected: moderate)
3. Price targets (expected: moderate)

Compares against the original binary direction prediction.

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import sys
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
import lightgbm as lgb

from data_collection.ninjatrader_loader import load_sample_data
from feature_engineering.multi_target_labels import (
    MultiTargetLabeler, MultiTargetConfig,
    get_classification_targets, get_regression_targets
)
from feature_engineering.volatility_regime import RealizedVolatilityGenerator

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features for prediction (past data only)."""
    features = pd.DataFrame(index=df.index)

    # Price-based features
    for lag in [1, 2, 3, 5, 10, 20]:
        features[f'return_lag{lag}'] = df['close'].pct_change(lag)

    # Volatility features
    rv_gen = RealizedVolatilityGenerator()
    rv_features = rv_gen.generate_features(df)
    features = pd.concat([features, rv_features], axis=1)

    # Technical features
    for period in [5, 10, 20, 50]:
        features[f'sma_{period}'] = df['close'].rolling(period).mean()
        features[f'sma_dist_{period}'] = (
            (df['close'] - features[f'sma_{period}']) / features[f'sma_{period}']
        )

    # RSI
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_hist'] = features['macd'] - features['macd_signal']

    # Bollinger Bands
    for period in [20]:
        mid = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        features[f'bb_upper_{period}'] = mid + 2 * std
        features[f'bb_lower_{period}'] = mid - 2 * std
        features[f'bb_pct_{period}'] = (
            (df['close'] - features[f'bb_lower_{period}']) /
            (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'] + 1e-10)
        )

    # ATR features
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    for period in [5, 10, 14, 20]:
        features[f'atr_{period}'] = tr.rolling(period).mean()
        features[f'atr_pct_{period}'] = features[f'atr_{period}'] / df['close']

    # Volume features
    if 'volume' in df.columns:
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / (features['volume_sma_20'] + 1)

    # Time features
    if hasattr(df.index, 'hour'):
        features['hour'] = df.index.hour
        features['minute'] = df.index.minute
        features['day_of_week'] = df.index.dayofweek

    return features


def train_and_evaluate_target(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_type: str
) -> dict:
    """Train model and evaluate for a single target."""

    if target_type == 'classification':
        # Binary classification
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)

        train_probs = model.predict_proba(X_train)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]

        train_auc = roc_auc_score(y_train, train_probs)
        test_auc = roc_auc_score(y_test, test_probs)

        return {
            'type': 'classification',
            'train_auc': train_auc,
            'test_auc': test_auc,
            'overfit_gap': train_auc - test_auc
        }

    else:  # Regression
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        return {
            'type': 'regression',
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'overfit_gap': train_r2 - test_r2
        }


def main():
    print("=" * 80)
    print(" MULTI-TARGET PREDICTABILITY ANALYSIS")
    print(" Comparing Different Prediction Targets")
    print("=" * 80)

    # Load data
    logger.info("\n--- Loading Data ---")
    prices, _ = load_sample_data(source="databento")
    logger.info(f"Loaded {len(prices)} bars")

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

    # Generate features
    logger.info("\n--- Generating Features ---")
    features = generate_features(prices)

    # Generate multi-targets
    logger.info("\n--- Generating Multi-Targets ---")
    labeler = MultiTargetLabeler()
    targets = labeler.generate_all_targets(prices)

    # Also add traditional binary target for comparison
    targets['traditional_direction'] = (
        prices['close'].shift(-1) > prices['close']
    ).astype(int)

    # Align features and targets
    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]

    # Drop NaN
    valid_mask = ~(features.isna().any(axis=1) | targets.isna().any(axis=1))
    features = features[valid_mask]
    targets = targets[valid_mask]

    logger.info(f"Final dataset: {len(features)} samples, {len(features.columns)} features")
    logger.info(f"Targets to evaluate: {len(targets.columns)}")

    # Train/test split (temporal)
    split_idx = int(len(features) * 0.8)
    X_train = features.iloc[:split_idx].values
    X_test = features.iloc[split_idx:].values

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Evaluate each target
    logger.info("\n--- Evaluating Target Predictability ---")

    results = {}
    classification_targets = get_classification_targets(targets) + ['traditional_direction']
    regression_targets = get_regression_targets(targets)

    # Evaluate classification targets
    print("\n" + "=" * 80)
    print(" CLASSIFICATION TARGETS (AUC-ROC)")
    print("=" * 80)
    print(f"{'Target':<40} {'Train AUC':>12} {'Test AUC':>12} {'Overfit':>10}")
    print("-" * 80)

    for target_name in classification_targets:
        y = targets[target_name]
        y_train = y.iloc[:split_idx].values
        y_test = y.iloc[split_idx:].values

        # Skip if not binary
        if len(np.unique(y_train[~np.isnan(y_train)])) < 2:
            continue

        try:
            result = train_and_evaluate_target(
                X_train, y_train, X_test, y_test, 'classification'
            )
            results[target_name] = result

            print(f"{target_name:<40} {result['train_auc']:>12.4f} {result['test_auc']:>12.4f} {result['overfit_gap']:>10.4f}")

        except Exception as e:
            logger.warning(f"Error evaluating {target_name}: {e}")

    # Evaluate regression targets (subset)
    print("\n" + "=" * 80)
    print(" REGRESSION TARGETS (R² Score)")
    print("=" * 80)
    print(f"{'Target':<40} {'Train R²':>12} {'Test R²':>12} {'Overfit':>10}")
    print("-" * 80)

    # Select key regression targets
    key_regression = [t for t in regression_targets if any(
        k in t for k in ['future_rv', 'future_atr', 'trend_strength', 'upside_pct', 'downside_pct']
    )][:15]

    for target_name in key_regression:
        y = targets[target_name]
        y_train = y.iloc[:split_idx].values
        y_test = y.iloc[split_idx:].values

        # Skip if invalid
        if np.isnan(y_train).all():
            continue

        try:
            result = train_and_evaluate_target(
                X_train, y_train, X_test, y_test, 'regression'
            )
            results[target_name] = result

            print(f"{target_name:<40} {result['train_r2']:>12.4f} {result['test_r2']:>12.4f} {result['overfit_gap']:>10.4f}")

        except Exception as e:
            logger.warning(f"Error evaluating {target_name}: {e}")

    # Summary by category
    print("\n" + "=" * 80)
    print(" SUMMARY BY TARGET CATEGORY")
    print("=" * 80)

    categories = {
        'Traditional (1-bar direction)': ['traditional_direction'],
        'Volatility (expansion/regime)': [t for t in results if 'vol_' in t or 'expansion' in t or 'contraction' in t],
        'Trend Direction': [t for t in results if 'trend_dir' in t],
        'Trend Persistence': [t for t in results if 'trend_persist' in t],
        'Price Reach': [t for t in results if 'reach_' in t],
        'New Highs/Lows': [t for t in results if 'new_high' in t or 'new_low' in t],
        'Volatility Forecast (R²)': [t for t in results if 'future_rv' in t or 'future_atr' in t],
    }

    for category, target_list in categories.items():
        if not target_list:
            continue

        aucs = [results[t]['test_auc'] for t in target_list if t in results and 'test_auc' in results[t]]
        r2s = [results[t]['test_r2'] for t in target_list if t in results and 'test_r2' in results[t]]

        if aucs:
            print(f"\n{category}:")
            print(f"  Best AUC: {max(aucs):.4f}")
            print(f"  Avg AUC:  {np.mean(aucs):.4f}")
            print(f"  Count:    {len(aucs)}")

        if r2s:
            print(f"\n{category}:")
            print(f"  Best R²:  {max(r2s):.4f}")
            print(f"  Avg R²:   {np.mean(r2s):.4f}")
            print(f"  Count:    {len(r2s)}")

    # Key insights
    print("\n" + "=" * 80)
    print(" KEY INSIGHTS")
    print("=" * 80)

    # Find best classification targets
    class_results = {k: v for k, v in results.items() if v.get('type') == 'classification'}
    if class_results:
        best_class = max(class_results.items(), key=lambda x: x[1]['test_auc'])
        print(f"\nBest Classification Target: {best_class[0]}")
        print(f"  Test AUC: {best_class[1]['test_auc']:.4f}")

        traditional_auc = results.get('traditional_direction', {}).get('test_auc', 0.5)
        improvement = best_class[1]['test_auc'] - traditional_auc
        print(f"\nCompared to Traditional Direction (AUC {traditional_auc:.4f}):")
        print(f"  Improvement: +{improvement:.4f} ({improvement/traditional_auc*100:.1f}%)")

    # Find best regression targets
    reg_results = {k: v for k, v in results.items() if v.get('type') == 'regression'}
    if reg_results:
        best_reg = max(reg_results.items(), key=lambda x: x[1]['test_r2'])
        print(f"\nBest Regression Target: {best_reg[0]}")
        print(f"  Test R²: {best_reg[1]['test_r2']:.4f}")

    # Save results
    output_dir = project_root / 'data' / 'validation_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results).T
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(output_dir / f'multi_target_analysis_{timestamp}.csv')

    print(f"\nResults saved to: {output_dir / f'multi_target_analysis_{timestamp}.csv'}")


if __name__ == "__main__":
    main()
