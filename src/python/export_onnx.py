"""
ONNX Model Export for NinjaTrader 8 Integration

Exports trained LightGBM models to ONNX format for use in NinjaTrader.

Models exported:
1. vol_expansion_model.onnx - Volatility expansion classifier
2. breakout_high_model.onnx - New high classifier
3. breakout_low_model.onnx - New low classifier
4. atr_forecast_model.onnx - ATR regressor

Author: SKIE_Ninja Development Team
Created: 2025-12-05
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import sys
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# ONNX conversion
try:
    import onnx
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("WARNING: onnxmltools not installed. Run: pip install onnxmltools onnx")

from feature_engineering.multi_target_labels import MultiTargetLabeler
from strategy.volatility_breakout_strategy import VolatilityBreakoutStrategy, StrategyConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXExporter:
    """Export LightGBM models to ONNX format for NinjaTrader."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or (project_root / 'data' / 'models' / 'onnx')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.strategy = VolatilityBreakoutStrategy(StrategyConfig())
        self.scaler = None
        self.feature_names = []

    def load_and_prepare_data(self) -> tuple:
        """Load data and prepare for training."""
        from data_collection.ninjatrader_loader import load_sample_data

        logger.info("Loading data...")
        prices, _ = load_sample_data(source="databento")

        # Filter RTH
        if hasattr(prices.index, 'hour'):
            prices = prices[
                (prices.index.hour >= 9) &
                ((prices.index.hour < 16) |
                 ((prices.index.hour == 9) & (prices.index.minute >= 30)))
            ]

        # Resample to 5-min
        prices = prices.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        logger.info(f"Prepared {len(prices)} bars")

        # Generate features and targets
        features = self.strategy.generate_features(prices)
        targets = self.strategy.target_labeler.generate_all_targets(prices)

        # Align data
        common_idx = features.index.intersection(targets.index)
        features = features.loc[common_idx]
        targets = targets.loc[common_idx]

        # Remove NaN
        valid_mask = ~(features.isna().any(axis=1) | targets.isna().any(axis=1))
        features = features[valid_mask]
        targets = targets[valid_mask]

        self.feature_names = list(features.columns)
        logger.info(f"Features: {len(self.feature_names)}")

        return features, targets

    def train_production_models(self, features: pd.DataFrame, targets: pd.DataFrame):
        """Train models on full dataset for production deployment."""
        logger.info("\n--- Training Production Models ---")

        # Use last 180 days for training (matching walk-forward config)
        bars_per_day = 78
        train_bars = 180 * bars_per_day

        if len(features) > train_bars:
            features = features.iloc[-train_bars:]
            targets = targets.iloc[-train_bars:]
            logger.info(f"Using last {train_bars} bars for training")

        X = features.values

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 1. Volatility expansion model
        logger.info("Training volatility expansion model...")
        y_vol = targets['vol_expansion_5'].values
        self.strategy.vol_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        self.strategy.vol_model.fit(X_scaled, y_vol)
        logger.info(f"  Trained on {len(y_vol)} samples")

        # 2. Breakout high model
        logger.info("Training breakout high model...")
        y_high = targets['new_high_10'].values
        self.strategy.breakout_high_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        self.strategy.breakout_high_model.fit(X_scaled, y_high)

        # 3. Breakout low model
        logger.info("Training breakout low model...")
        y_low = targets['new_low_10'].values
        self.strategy.breakout_low_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        self.strategy.breakout_low_model.fit(X_scaled, y_low)

        # 4. ATR forecast model
        logger.info("Training ATR forecast model...")
        y_atr = targets['future_atr_5'].values
        self.strategy.atr_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        self.strategy.atr_model.fit(X_scaled, y_atr)

        logger.info("All models trained successfully")

    def export_to_onnx(self):
        """Export all models to ONNX format."""
        if not ONNX_AVAILABLE:
            raise ImportError("onnxmltools not installed. Run: pip install onnxmltools onnx")

        logger.info("\n--- Exporting to ONNX ---")

        n_features = len(self.feature_names)
        initial_type = [('features', FloatTensorType([None, n_features]))]

        # Export vol expansion model
        logger.info("Exporting vol_expansion_model.onnx...")
        onnx_vol = convert_lightgbm(
            self.strategy.vol_model,
            initial_types=initial_type,
            target_opset=12
        )
        onnx.save_model(onnx_vol, str(self.output_dir / 'vol_expansion_model.onnx'))

        # Export breakout high model
        logger.info("Exporting breakout_high_model.onnx...")
        onnx_high = convert_lightgbm(
            self.strategy.breakout_high_model,
            initial_types=initial_type,
            target_opset=12
        )
        onnx.save_model(onnx_high, str(self.output_dir / 'breakout_high_model.onnx'))

        # Export breakout low model
        logger.info("Exporting breakout_low_model.onnx...")
        onnx_low = convert_lightgbm(
            self.strategy.breakout_low_model,
            initial_types=initial_type,
            target_opset=12
        )
        onnx.save_model(onnx_low, str(self.output_dir / 'breakout_low_model.onnx'))

        # Export ATR model
        logger.info("Exporting atr_forecast_model.onnx...")
        onnx_atr = convert_lightgbm(
            self.strategy.atr_model,
            initial_types=initial_type,
            target_opset=12
        )
        onnx.save_model(onnx_atr, str(self.output_dir / 'atr_forecast_model.onnx'))

        logger.info(f"\nAll models exported to: {self.output_dir}")

    def export_scaler_params(self):
        """Export scaler parameters for feature normalization in C#."""
        logger.info("Exporting scaler parameters...")

        scaler_params = {
            'feature_names': self.feature_names,
            'mean': self.scaler.mean_.tolist(),
            'scale': self.scaler.scale_.tolist(),
            'n_features': len(self.feature_names)
        }

        with open(self.output_dir / 'scaler_params.json', 'w') as f:
            json.dump(scaler_params, f, indent=2)

        logger.info(f"Scaler params saved to: {self.output_dir / 'scaler_params.json'}")

    def export_strategy_config(self):
        """Export strategy configuration for NinjaScript."""
        config = self.strategy.config

        strategy_config = {
            'min_vol_expansion_prob': config.min_vol_expansion_prob,
            'min_breakout_prob': config.min_breakout_prob,
            'tp_atr_mult_base': config.tp_atr_mult_base,
            'sl_atr_mult_base': config.sl_atr_mult_base,
            'tp_adjustment_factor': config.tp_adjustment_factor,
            'max_holding_bars': config.max_holding_bars,
            'base_contracts': config.base_contracts,
            'max_contracts': config.max_contracts,
            'vol_sizing_factor': config.vol_sizing_factor
        }

        with open(self.output_dir / 'strategy_config.json', 'w') as f:
            json.dump(strategy_config, f, indent=2)

        logger.info(f"Strategy config saved to: {self.output_dir / 'strategy_config.json'}")

    def validate_onnx_models(self):
        """Validate exported ONNX models work correctly."""
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not installed. Skipping validation.")
            return

        logger.info("\n--- Validating ONNX Models ---")

        # Create test input
        n_features = len(self.feature_names)
        test_input = np.random.randn(1, n_features).astype(np.float32)

        models = [
            ('vol_expansion_model.onnx', 'classifier'),
            ('breakout_high_model.onnx', 'classifier'),
            ('breakout_low_model.onnx', 'classifier'),
            ('atr_forecast_model.onnx', 'regressor')
        ]

        for model_name, model_type in models:
            model_path = self.output_dir / model_name
            session = ort.InferenceSession(str(model_path))

            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: test_input})

            if model_type == 'classifier':
                # LightGBM classifier outputs: label, probabilities (as list of dicts)
                logger.info(f"  {model_name}: OK (outputs: {len(outputs)})")
            else:
                logger.info(f"  {model_name}: OK (output: {outputs[0][0]:.4f})")

        logger.info("All models validated successfully!")


def main():
    """Main export pipeline."""
    print("=" * 80)
    print(" ONNX MODEL EXPORT FOR NINJATRADER")
    print("=" * 80)

    if not ONNX_AVAILABLE:
        print("\nERROR: Required packages not installed.")
        print("Run: pip install onnxmltools onnx onnxruntime")
        return

    exporter = ONNXExporter()

    # Load and prepare data
    features, targets = exporter.load_and_prepare_data()

    # Train production models
    exporter.train_production_models(features, targets)

    # Export to ONNX
    exporter.export_to_onnx()

    # Export supporting files
    exporter.export_scaler_params()
    exporter.export_strategy_config()

    # Validate
    exporter.validate_onnx_models()

    print("\n" + "=" * 80)
    print(" EXPORT COMPLETE")
    print("=" * 80)
    print(f"\nFiles created in: {exporter.output_dir}")
    print("  - vol_expansion_model.onnx")
    print("  - breakout_high_model.onnx")
    print("  - breakout_low_model.onnx")
    print("  - atr_forecast_model.onnx")
    print("  - scaler_params.json")
    print("  - strategy_config.json")
    print("\nNext step: Copy these files to your NinjaTrader installation")


if __name__ == "__main__":
    main()
