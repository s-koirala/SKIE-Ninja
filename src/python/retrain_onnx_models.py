"""
ONNX Model Retraining Script for NinjaTrader

Per project methodology (BACKTEST_METHODOLOGY.md):
- Train window: 180 days
- Test window: 5 days
- Embargo: 42 bars
- Retraining frequency: Every 5 trading days

This script should be run weekly/bi-weekly to keep ONNX models fresh.
Static models decay rapidly in financial markets.

Usage:
    python retrain_onnx_models.py [--data-end-date YYYY-MM-DD]

Author: SKIE_Ninja Development Team
Created: 2025-12-06
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

from export_onnx import ONNXExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Retrain and export ONNX models for NinjaTrader'
    )
    parser.add_argument(
        '--data-end-date',
        type=str,
        default=None,
        help='End date for training data (YYYY-MM-DD). Default: use all available data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for ONNX files. Default: data/models/onnx'
    )
    parser.add_argument(
        '--copy-to-ninjatrader',
        action='store_true',
        help='Copy exported models to NinjaTrader models folder'
    )

    args = parser.parse_args()

    print("=" * 80)
    print(" ONNX MODEL RETRAINING FOR NINJATRADER")
    print(" Following project methodology: 180-day train, 5-day valid window")
    print("=" * 80)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / 'data' / 'models' / 'onnx'

    # Create timestamped backup of existing models
    if output_dir.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = output_dir.parent / f'onnx_backup_{timestamp}'
        logger.info(f"Backing up existing models to: {backup_dir}")
        import shutil
        if not backup_dir.exists():
            shutil.copytree(output_dir, backup_dir)

    # Run export
    logger.info("\nRetraining models...")
    exporter = ONNXExporter(output_dir)

    # Load and prepare data
    features, targets = exporter.load_and_prepare_data()

    # Log training period
    logger.info(f"\nData range: {features.index[0]} to {features.index[-1]}")
    logger.info(f"Training on last 180 days: ~{features.index[-180*78]} to {features.index[-1]}")

    # Train production models
    exporter.train_production_models(features, targets)

    # Export to ONNX
    exporter.export_to_onnx()

    # Export supporting files
    exporter.export_scaler_params()
    exporter.export_strategy_config()

    # Validate
    exporter.validate_onnx_models()

    # Add metadata about training
    metadata = {
        'training_date': datetime.now().isoformat(),
        'data_start': str(features.index[0]),
        'data_end': str(features.index[-1]),
        'training_bars': len(features),
        'methodology': {
            'train_window_days': 180,
            'test_window_days': 5,
            'embargo_bars': 42,
            'valid_until': (datetime.now() + timedelta(days=5)).strftime('%Y-%m-%d'),
            'recommendation': 'Retrain every 5 trading days for optimal performance'
        }
    }

    import json
    with open(output_dir / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"\nTraining metadata saved to: {output_dir / 'training_metadata.json'}")

    # Copy to NinjaTrader if requested
    if args.copy_to_ninjatrader:
        import os
        nt_models_dir = Path(os.path.expanduser('~')) / 'Documents' / 'SKIE_Ninja' / 'models'

        if nt_models_dir.exists():
            import shutil
            for file in output_dir.glob('*'):
                shutil.copy2(file, nt_models_dir / file.name)
            logger.info(f"\nModels copied to NinjaTrader: {nt_models_dir}")
        else:
            logger.warning(f"\nNinjaTrader models folder not found: {nt_models_dir}")
            logger.warning("Please manually copy models from: {output_dir}")

    print("\n" + "=" * 80)
    print(" RETRAINING COMPLETE")
    print("=" * 80)
    print(f"\nModels valid for approximately 5 trading days from: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Recommended next retraining: {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}")
    print("\nFor walk-forward equivalent performance, retrain weekly.")
    print("=" * 80)


if __name__ == "__main__":
    main()
