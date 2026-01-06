"""
Model Divergence Diagnostic: Python vs ONNX
============================================

Investigates trade frequency collapse root cause per TRADE_FREQUENCY_INVESTIGATION_20260106.md

Hypothesis H2 Confirmed: Vol filter rejecting 99.9% of signals
- NT8 logs show vol_prob in range 0.02-0.09 (2-9%)
- Threshold is 0.50 (50%)
- Pass rate: 0.007%

This script determines whether the issue is:
A) ONNX export divergence (different outputs for same input)
B) Model calibration issue (model outputs low probabilities by design)
C) Feature calculation divergence (C# features differ from Python)

Author: SKIE_Ninja Development Team
Date: 2026-01-06
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
import sys

# Setup paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

import onnxruntime as ort
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_onnx_models(models_dir: Path):
    """Load all ONNX models."""
    sessions = {}
    for model_name in ['vol_expansion_model', 'breakout_high_model', 'breakout_low_model', 'atr_forecast_model']:
        model_path = models_dir / f'{model_name}.onnx'
        if model_path.exists():
            sessions[model_name] = ort.InferenceSession(str(model_path))
            logger.info(f"  Loaded: {model_name}")
    return sessions


def load_scaler_params(models_dir: Path):
    """Load scaler parameters."""
    with open(models_dir / 'scaler_params.json', 'r') as f:
        params = json.load(f)
    return params


def generate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate technical features (matching ensemble_strategy.py)."""
    features = pd.DataFrame(index=df.index)

    # Returns (lagged)
    for lag in [1, 2, 3, 5, 10, 20]:
        features[f'return_lag{lag}'] = df['close'].pct_change(lag)

    # ATR calculation
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    for period in [5, 10, 14, 20]:
        features[f'rv_{period}'] = df['close'].pct_change().rolling(period).std()
        features[f'atr_{period}'] = tr.rolling(period).mean()
        features[f'atr_pct_{period}'] = features[f'atr_{period}'] / df['close']

    # Price position
    for period in [10, 20, 50]:
        features[f'close_vs_high_{period}'] = (
            df['close'] - df['high'].rolling(period).max()
        ) / df['close']
        features[f'close_vs_low_{period}'] = (
            df['close'] - df['low'].rolling(period).min()
        ) / df['close']

    # Range percent (not in original but needed for feature parity)
    for period in [10, 20, 50]:
        period_high = df['high'].rolling(period).max()
        period_low = df['low'].rolling(period).min()
        features[f'range_pct_{period}'] = (period_high - period_low) / df['close']

    # Momentum
    for period in [5, 10, 20]:
        features[f'momentum_{period}'] = df['close'].pct_change(period)
        ma = df['close'].rolling(period).mean()
        features[f'ma_dist_{period}'] = (df['close'] - ma) / ma

    # RSI
    for period in [7, 14]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # Bollinger Band position
    mid = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    features['bb_pct_20'] = (df['close'] - lower) / (upper - lower + 1e-10)

    # Volume features
    if 'volume' in df.columns:
        for period in [5, 10, 20]:
            features[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = df['volume'] / (features[f'volume_sma_{period}'] + 1)

    return features


def run_onnx_inference(session, features_scaled: np.ndarray) -> np.ndarray:
    """Run ONNX inference and extract class 1 probabilities."""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: features_scaled.astype(np.float32)})

    # For classifiers, output[1] contains probabilities
    if len(outputs) > 1:
        prob_output = outputs[1]
        # Handle dict output format from LightGBM ONNX
        if isinstance(prob_output, list) and isinstance(prob_output[0], dict):
            probs = np.array([p.get(1, 0.0) for p in prob_output])
        else:
            probs = np.array(prob_output)[:, 1] if prob_output.ndim > 1 else np.array(prob_output)
    else:
        probs = outputs[0]

    return probs


def train_fresh_lgb_model(X_train, y_train, X_test):
    """Train a fresh LightGBM model and get predictions."""
    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    return probs, model


def run_diagnostic():
    """Run full model divergence diagnostic."""
    print("=" * 80)
    print("MODEL DIVERGENCE DIAGNOSTIC")
    print("Python vs ONNX Comparison")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Paths
    models_dir = project_root / 'data' / 'models' / 'onnx'

    # Load ONNX models
    print("\n--- Loading ONNX Models ---")
    onnx_sessions = load_onnx_models(models_dir)
    scaler_params = load_scaler_params(models_dir)

    print(f"\n  Feature count: {scaler_params['n_features']}")
    print(f"  Feature names: {scaler_params['feature_names'][:5]}...")

    # Load market data
    print("\n--- Loading Market Data ---")
    from data_collection.ninjatrader_loader import load_sample_data
    prices, _ = load_sample_data(source="databento")
    print(f"  Raw bars: {len(prices)}")

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
    print(f"  After RTH + 5-min resample: {len(prices)} bars")

    # Generate features
    print("\n--- Generating Features ---")
    features = generate_technical_features(prices)

    # Align feature columns to scaler order
    expected_features = scaler_params['feature_names']
    missing = set(expected_features) - set(features.columns)
    if missing:
        print(f"  WARNING: Missing features: {missing}")

    features = features[expected_features].dropna()
    print(f"  Valid samples: {len(features)}")

    # Generate targets for training fresh model
    print("\n--- Generating Targets ---")
    from feature_engineering.multi_target_labels import MultiTargetLabeler
    labeler = MultiTargetLabeler()
    targets = labeler.generate_all_targets(prices)

    # Align
    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]

    # Remove NaN
    valid_mask = ~(features.isna().any(axis=1) | targets.isna().any(axis=1))
    features = features[valid_mask]
    targets = targets[valid_mask]
    print(f"  Aligned samples: {len(features)}")

    # Scale features using stored scaler params
    print("\n--- Scaling Features ---")
    mean = np.array(scaler_params['mean'])
    scale = np.array(scaler_params['scale'])
    X_scaled = (features.values - mean) / scale
    print(f"  Scaled shape: {X_scaled.shape}")

    # Run ONNX inference on all data
    print("\n--- ONNX Inference ---")
    onnx_vol_probs = run_onnx_inference(onnx_sessions['vol_expansion_model'], X_scaled)
    onnx_high_probs = run_onnx_inference(onnx_sessions['breakout_high_model'], X_scaled)
    onnx_low_probs = run_onnx_inference(onnx_sessions['breakout_low_model'], X_scaled)

    print(f"\n  Vol Expansion ONNX Probabilities:")
    print(f"    Min:    {onnx_vol_probs.min():.4f}")
    print(f"    Max:    {onnx_vol_probs.max():.4f}")
    print(f"    Mean:   {onnx_vol_probs.mean():.4f}")
    print(f"    Median: {np.median(onnx_vol_probs):.4f}")
    print(f"    Std:    {onnx_vol_probs.std():.4f}")

    # Analyze threshold pass rates
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    print(f"\n  Pass Rates at Different Thresholds:")
    for thresh in thresholds:
        pass_rate = (onnx_vol_probs >= thresh).mean() * 100
        print(f"    >= {thresh:.2f}: {pass_rate:6.2f}%")

    # Train fresh Python model for comparison
    print("\n--- Training Fresh Python LightGBM Model ---")

    # Use walk-forward style: train on first 80%, test on last 20%
    split_idx = int(len(features) * 0.8)

    X_train = features.iloc[:split_idx].values
    X_test = features.iloc[split_idx:].values
    y_train = targets.iloc[:split_idx]['vol_expansion_5'].values
    y_test = targets.iloc[split_idx:]['vol_expansion_5'].values

    # Scale using fresh scaler (not stored params)
    fresh_scaler = StandardScaler()
    X_train_scaled = fresh_scaler.fit_transform(X_train)
    X_test_scaled = fresh_scaler.transform(X_test)

    python_probs, python_model = train_fresh_lgb_model(X_train_scaled, y_train, X_test_scaled)

    print(f"\n  Python LightGBM Probabilities (test set):")
    print(f"    Min:    {python_probs.min():.4f}")
    print(f"    Max:    {python_probs.max():.4f}")
    print(f"    Mean:   {python_probs.mean():.4f}")
    print(f"    Median: {np.median(python_probs):.4f}")
    print(f"    Std:    {python_probs.std():.4f}")

    print(f"\n  Python Pass Rates at Different Thresholds:")
    for thresh in thresholds:
        pass_rate = (python_probs >= thresh).mean() * 100
        print(f"    >= {thresh:.2f}: {pass_rate:6.2f}%")

    # Compare ONNX vs Python on SAME test data with SAME scaling
    print("\n--- Direct Comparison: ONNX vs Python (Same Inputs) ---")

    # Scale test data using stored ONNX scaler
    X_test_onnx_scaled = (X_test - mean) / scale
    onnx_test_probs = run_onnx_inference(onnx_sessions['vol_expansion_model'], X_test_onnx_scaled)

    print(f"\n  ONNX on test set (using stored scaler):")
    print(f"    Min:    {onnx_test_probs.min():.4f}")
    print(f"    Max:    {onnx_test_probs.max():.4f}")
    print(f"    Mean:   {onnx_test_probs.mean():.4f}")

    print(f"\n  Python on test set (using fresh scaler):")
    print(f"    Min:    {python_probs.min():.4f}")
    print(f"    Max:    {python_probs.max():.4f}")
    print(f"    Mean:   {python_probs.mean():.4f}")

    # Compute divergence
    # Note: Direct comparison requires same scaling, so retrain Python with stored scaler
    print("\n--- Strict Comparison: Both Using Stored Scaler ---")
    X_train_stored_scaled = (X_train - mean) / scale
    X_test_stored_scaled = (X_test - mean) / scale

    python_stored_probs, _ = train_fresh_lgb_model(X_train_stored_scaled, y_train, X_test_stored_scaled)

    print(f"\n  Python (stored scaler):")
    print(f"    Mean:   {python_stored_probs.mean():.4f}")
    print(f"    Std:    {python_stored_probs.std():.4f}")

    print(f"\n  ONNX (stored scaler):")
    print(f"    Mean:   {onnx_test_probs.mean():.4f}")
    print(f"    Std:    {onnx_test_probs.std():.4f}")

    # Correlation
    corr = np.corrcoef(python_stored_probs, onnx_test_probs)[0, 1]
    delta = np.abs(python_stored_probs - onnx_test_probs)

    print(f"\n  Divergence Metrics:")
    print(f"    Correlation:   {corr:.4f}")
    print(f"    Mean |delta|:  {delta.mean():.4f}")
    print(f"    Max |delta|:   {delta.max():.4f}")
    print(f"    Std |delta|:   {delta.std():.4f}")

    # Diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    if onnx_vol_probs.max() < 0.15:
        print("\n[CRITICAL] ONNX model outputs are severely compressed (max < 0.15)")
        print("  This indicates the ONNX export may have lost probability calibration.")
        print("  The LightGBM ZipMap output may not be correctly handled.")

    if python_probs.mean() > 0.3 and onnx_vol_probs.mean() < 0.1:
        print("\n[CRITICAL] Python outputs normal range, ONNX severely compressed")
        print("  Root cause: ONNX export/runtime probability extraction issue")
        print("  Remediation: Re-export models with verified probability output format")

    if corr < 0.5:
        print(f"\n[WARNING] Low correlation ({corr:.2f}) between Python and ONNX")
        print("  Models may have diverged during ONNX export")

    # Target distribution analysis
    print("\n--- Target Distribution Analysis ---")
    vol_target = targets['vol_expansion_5']
    print(f"  Vol expansion base rate: {vol_target.mean():.4f} ({vol_target.mean()*100:.1f}%)")
    print(f"  Class 0 (no expansion): {(vol_target == 0).sum():,}")
    print(f"  Class 1 (expansion):    {(vol_target == 1).sum():,}")

    # If base rate is low, model might be well-calibrated but threshold is wrong
    if vol_target.mean() < 0.15:
        print(f"\n[INFO] Vol expansion is a rare event (~{vol_target.mean()*100:.0f}% base rate)")
        print("  A well-calibrated model SHOULD output low probabilities on average.")
        print("  The issue may be threshold calibration, not model divergence.")
        print(f"  Recommended threshold: {vol_target.mean():.2f} to {vol_target.mean()*1.5:.2f}")

    # Final recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    optimal_thresh = np.percentile(onnx_vol_probs, 90)  # Top 10% of predictions
    print(f"\n1. IMMEDIATE: Lower vol threshold from 0.50 to {optimal_thresh:.2f}")
    print(f"   This captures top 10% of predictions, matching ~10% base rate")

    print(f"\n2. VERIFY: Check C# feature calculations match Python")
    print(f"   Export feature values from NT8 for comparison")

    print(f"\n3. RE-EXPORT: If divergence confirmed, re-export ONNX with verification")

    # Save diagnostic results
    output_dir = project_root / 'data' / 'validation_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'timestamp': timestamp,
        'onnx_vol_mean': float(onnx_vol_probs.mean()),
        'onnx_vol_std': float(onnx_vol_probs.std()),
        'onnx_vol_max': float(onnx_vol_probs.max()),
        'python_vol_mean': float(python_probs.mean()),
        'python_vol_std': float(python_probs.std()),
        'python_vol_max': float(python_probs.max()),
        'correlation': float(corr),
        'mean_delta': float(delta.mean()),
        'target_base_rate': float(vol_target.mean()),
        'recommended_threshold': float(optimal_thresh),
        'samples_analyzed': len(features)
    }

    with open(output_dir / f'model_divergence_diagnostic_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: model_divergence_diagnostic_{timestamp}.json")

    return results


if __name__ == '__main__':
    results = run_diagnostic()
