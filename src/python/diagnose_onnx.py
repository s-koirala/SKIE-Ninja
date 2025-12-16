"""
Diagnostic script to verify ONNX model outputs and feature calculations.
Run this to compare Python vs C# implementations.
"""

import numpy as np
import json
import onnxruntime as ort
from pathlib import Path

# Paths
project_root = Path(__file__).parent.parent.parent
models_dir = project_root / 'data' / 'models' / 'onnx'

def load_scaler():
    """Load scaler parameters."""
    with open(models_dir / 'scaler_params.json', 'r') as f:
        params = json.load(f)
    return params

def test_onnx_models():
    """Test ONNX model loading and output format."""
    print("=" * 60)
    print("ONNX MODEL DIAGNOSTIC")
    print("=" * 60)

    # Load scaler
    scaler = load_scaler()
    n_features = scaler['n_features']
    print(f"\nNumber of features: {n_features}")
    print(f"Feature names: {scaler['feature_names'][:5]}... (showing first 5)")

    # Create dummy input (zeros, scaled)
    test_input = np.zeros((1, n_features), dtype=np.float32)

    # Test each model
    models = [
        ('vol_expansion_model.onnx', 'classifier'),
        ('breakout_high_model.onnx', 'classifier'),
        ('breakout_low_model.onnx', 'classifier'),
        ('atr_forecast_model.onnx', 'regressor'),
    ]

    print("\n" + "-" * 40)
    print("Testing ONNX Models")
    print("-" * 40)

    for model_name, model_type in models:
        model_path = models_dir / model_name
        print(f"\n{model_name}:")

        session = ort.InferenceSession(str(model_path))

        # Get input/output info
        input_info = session.get_inputs()[0]
        print(f"  Input name: {input_info.name}")
        print(f"  Input shape: {input_info.shape}")

        for i, output in enumerate(session.get_outputs()):
            print(f"  Output[{i}] name: {output.name}, type: {output.type}")

        # Run inference
        outputs = session.run(None, {input_info.name: test_input})

        if model_type == 'classifier':
            print(f"  Output[0] (labels): {outputs[0]}")
            print(f"  Output[1] (probabilities): {outputs[1]}")

            # Show how to extract probability for class 1
            prob_output = outputs[1]
            print(f"  Probability output type: {type(prob_output)}")
            if isinstance(prob_output, list):
                print(f"  First element: {prob_output[0]}")
                if isinstance(prob_output[0], dict):
                    print(f"  Keys: {prob_output[0].keys()}")
                    if 1 in prob_output[0]:
                        print(f"  P(class=1): {prob_output[0][1]}")
        else:
            print(f"  Output (prediction): {outputs[0][0]}")

    return True

def test_feature_scaling():
    """Test feature scaling matches expected values."""
    print("\n" + "=" * 60)
    print("FEATURE SCALING TEST")
    print("=" * 60)

    scaler = load_scaler()

    # Show mean/scale for key features
    print("\nScaler statistics for key features:")
    for i, name in enumerate(scaler['feature_names'][:10]):
        print(f"  {name}: mean={scaler['mean'][i]:.6f}, scale={scaler['scale'][i]:.6f}")

    # Test scaling a sample feature vector
    sample_features = np.array([
        0.001,  # return_lag1 (0.1% return)
        0.002,  # return_lag2
        0.003,  # return_lag3
        0.005,  # return_lag5
        0.01,   # return_lag10
        0.02,   # return_lag20
        0.0007, # rv_5
        5.0,    # atr_5 (5 points)
        0.001,  # atr_pct_5
        0.0007, # rv_10
        4.9,    # atr_10
        0.001,  # atr_pct_10
        0.0007, # rv_14
        4.8,    # atr_14
        0.001,  # atr_pct_14
        0.0007, # rv_20
        4.7,    # atr_20
        0.001,  # atr_pct_20
        -0.002, # close_vs_high_10
        0.002,  # close_vs_low_10
        0.004,  # range_pct_10
        -0.003, # close_vs_high_20
        0.003,  # close_vs_low_20
        0.006,  # range_pct_20
        -0.004, # close_vs_high_50
        0.004,  # close_vs_low_50
        0.008,  # range_pct_50
        0.005,  # momentum_5
        0.001,  # ma_dist_5
        0.01,   # momentum_10
        0.002,  # ma_dist_10
        0.02,   # momentum_20
        0.003,  # ma_dist_20
        50.0,   # rsi_7
        50.0,   # rsi_14
        0.5,    # bb_pct_20
        10000,  # volume_sma_5
        1.0,    # volume_ratio_5
        10000,  # volume_sma_10
        1.0,    # volume_ratio_10
        10000,  # volume_sma_20
        1.0,    # volume_ratio_20
    ])

    # Scale features
    mean = np.array(scaler['mean'])
    scale = np.array(scaler['scale'])
    scaled = (sample_features - mean) / scale

    print("\nSample feature values (first 10):")
    for i in range(10):
        print(f"  {scaler['feature_names'][i]}: raw={sample_features[i]:.6f} -> scaled={scaled[i]:.4f}")

    return scaled

def run_full_prediction(scaled_features):
    """Run a full prediction with scaled features."""
    print("\n" + "=" * 60)
    print("FULL PREDICTION TEST")
    print("=" * 60)

    scaled_input = scaled_features.reshape(1, -1).astype(np.float32)

    # Load and run each model
    vol_session = ort.InferenceSession(str(models_dir / 'vol_expansion_model.onnx'))
    high_session = ort.InferenceSession(str(models_dir / 'breakout_high_model.onnx'))
    low_session = ort.InferenceSession(str(models_dir / 'breakout_low_model.onnx'))
    atr_session = ort.InferenceSession(str(models_dir / 'atr_forecast_model.onnx'))

    input_name = vol_session.get_inputs()[0].name

    # Get predictions
    vol_out = vol_session.run(None, {input_name: scaled_input})
    high_out = high_session.run(None, {input_name: scaled_input})
    low_out = low_session.run(None, {input_name: scaled_input})
    atr_out = atr_session.run(None, {input_name: scaled_input})

    # Extract probabilities
    vol_prob = vol_out[1][0].get(1, 0.0) if isinstance(vol_out[1][0], dict) else 0.0
    high_prob = high_out[1][0].get(1, 0.0) if isinstance(high_out[1][0], dict) else 0.0
    low_prob = low_out[1][0].get(1, 0.0) if isinstance(low_out[1][0], dict) else 0.0
    atr_pred = float(atr_out[0].flatten()[0])

    print(f"\nPredictions:")
    print(f"  Vol expansion probability: {vol_prob:.4f}")
    print(f"  Breakout high probability: {high_prob:.4f}")
    print(f"  Breakout low probability:  {low_prob:.4f}")
    print(f"  ATR forecast:              {atr_pred:.4f}")

    # Trading signal logic
    print(f"\nTrading Signal Logic:")
    min_vol_prob = 0.40
    min_breakout_prob = 0.45

    print(f"  Min vol prob threshold: {min_vol_prob}")
    print(f"  Min breakout threshold: {min_breakout_prob}")

    if vol_prob < min_vol_prob:
        print(f"  -> NO TRADE (vol prob {vol_prob:.3f} < {min_vol_prob})")
    elif high_prob >= min_breakout_prob and low_prob < min_breakout_prob:
        print(f"  -> LONG (high prob {high_prob:.3f} >= threshold)")
    elif low_prob >= min_breakout_prob and high_prob < min_breakout_prob:
        print(f"  -> SHORT (low prob {low_prob:.3f} >= threshold)")
    elif high_prob >= min_breakout_prob and low_prob >= min_breakout_prob:
        if high_prob > low_prob:
            print(f"  -> LONG (both signals, high {high_prob:.3f} > low {low_prob:.3f})")
        else:
            print(f"  -> SHORT (both signals, low {low_prob:.3f} >= high {high_prob:.3f})")
    else:
        print(f"  -> NO TRADE (no breakout signal)")

if __name__ == '__main__':
    test_onnx_models()
    scaled = test_feature_scaling()
    run_full_prediction(scaled)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
