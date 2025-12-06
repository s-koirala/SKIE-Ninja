"""
Walk-Forward ONNX Model Export for NinjaTrader 8

This script generates multiple ONNX model snapshots at different points in time,
enabling proper walk-forward backtesting in NinjaTrader 8.

The script:
1. Splits data into walk-forward folds (180-day train, 5-day test)
2. Trains and exports ONNX models for each fold
3. Creates a schedule showing which models to use for which dates
4. Enables NT8 to achieve results comparable to Python walk-forward

Usage:
    python export_walkforward_onnx.py                    # Export all folds
    python export_walkforward_onnx.py --start 2024-01-01 # Start from specific date
    python export_walkforward_onnx.py --max-folds 10     # Limit number of folds

Author: SKIE_Ninja Development Team
Created: 2025-12-06
"""

import sys
import os
import json
import logging
import argparse
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'python'))

# Import project modules
from data_collection.ninjatrader_loader import load_sample_data
from data_collection.historical_sentiment_loader import HistoricalSentimentLoader, HistoricalSentimentConfig
from strategy.volatility_breakout_strategy import VolatilityBreakoutStrategy, StrategyConfig
from sklearn.preprocessing import StandardScaler

# Check for ONNX dependencies
try:
    import onnx
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    import onnxruntime as ort
    import lightgbm as lgb
    ONNX_AVAILABLE = True
except ImportError as e:
    ONNX_AVAILABLE = False
    print(f"ONNX import error: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WalkForwardONNXExporter:
    """Exports ONNX models for each walk-forward fold."""

    def __init__(
        self,
        train_days: int = 180,
        test_days: int = 5,
        output_base_dir: Optional[Path] = None
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.output_base_dir = output_base_dir or PROJECT_ROOT / 'data' / 'models' / 'walkforward_onnx'
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Model parameters (matching production settings)
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 5,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        }

        self.feature_names = None
        self.sentiment_feature_names = None
        self.technical_feature_names = None
        self.schedule = []

        # Sentiment loader
        self.sentiment_loader = HistoricalSentimentLoader(HistoricalSentimentConfig())

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and prepare all data."""
        logger.info("Loading market data...")

        # Load data using same method as export_onnx.py
        prices, _ = load_sample_data(source="databento")

        # Filter RTH (Regular Trading Hours)
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

        # Generate technical features and targets using strategy
        strategy = VolatilityBreakoutStrategy(StrategyConfig())
        tech_features = strategy.generate_features(prices)
        targets = strategy.target_labeler.generate_all_targets(prices)

        # Generate sentiment features from VIX
        logger.info("Loading VIX sentiment data...")
        self.sentiment_loader.load_all()
        sent_features = self.sentiment_loader.align_to_bars(prices.index)

        # Align all data
        common_idx = tech_features.index.intersection(targets.index).intersection(sent_features.index)
        tech_features = tech_features.loc[common_idx]
        sent_features = sent_features.loc[common_idx]
        targets = targets.loc[common_idx]

        # Remove NaN
        valid_mask = ~(
            tech_features.isna().any(axis=1) |
            sent_features.isna().any(axis=1) |
            targets.isna().any(axis=1)
        )
        tech_features = tech_features[valid_mask]
        sent_features = sent_features[valid_mask]
        targets = targets[valid_mask]

        # Store feature names
        self.technical_feature_names = list(tech_features.columns)
        self.sentiment_feature_names = list(sent_features.columns)
        self.feature_names = self.technical_feature_names  # For backward compatibility

        logger.info(f"Loaded {len(tech_features)} samples")
        logger.info(f"  Technical features: {len(self.technical_feature_names)}")
        logger.info(f"  Sentiment features: {len(self.sentiment_feature_names)}")

        return tech_features, sent_features, targets

    def get_fold_ranges(
        self,
        data_start: pd.Timestamp,
        data_end: pd.Timestamp,
        start_date: Optional[str] = None
    ) -> List[Dict]:
        """Calculate all walk-forward fold date ranges."""
        folds = []

        # Start after enough data for first training window
        first_test_start = data_start + pd.Timedelta(days=self.train_days)

        if start_date:
            start_ts = pd.Timestamp(start_date)
            # Match timezone if needed
            if data_start.tz is not None and start_ts.tz is None:
                start_ts = start_ts.tz_localize(data_start.tz)
            first_test_start = max(first_test_start, start_ts)

        current_test_start = first_test_start
        fold_num = 0

        while current_test_start + pd.Timedelta(days=self.test_days) <= data_end:
            train_start = current_test_start - pd.Timedelta(days=self.train_days)
            train_end = current_test_start - pd.Timedelta(days=1)
            test_start = current_test_start
            test_end = current_test_start + pd.Timedelta(days=self.test_days - 1)

            folds.append({
                'fold': fold_num,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'model_id': test_start.strftime('%Y%m%d')
            })

            current_test_start += pd.Timedelta(days=self.test_days)
            fold_num += 1

        logger.info(f"Created {len(folds)} walk-forward folds")
        return folds

    def train_fold_models(
        self,
        tech_features: pd.DataFrame,
        sent_features: pd.DataFrame,
        targets: pd.DataFrame,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp
    ) -> Tuple[Dict, StandardScaler, StandardScaler]:
        """Train models for a single fold including sentiment model."""
        # Filter to training window
        mask = (tech_features.index >= train_start) & (tech_features.index <= train_end)
        X_tech_train = tech_features.loc[mask].values
        X_sent_train = sent_features.loc[mask].values

        if len(X_tech_train) < 100:
            logger.warning(f"Insufficient training data: {len(X_tech_train)} samples")
            return None, None, None

        # Fit scalers
        tech_scaler = StandardScaler()
        sent_scaler = StandardScaler()
        X_tech_scaled = tech_scaler.fit_transform(X_tech_train)
        X_sent_scaled = sent_scaler.fit_transform(X_sent_train)

        # Combined features for sentiment model (tech + sent)
        X_combined_scaled = np.hstack([X_tech_scaled, X_sent_scaled])

        models = {}

        # Train technical volatility expansion model (tech features only)
        y_vol = targets.loc[mask, 'vol_expansion_5'].values
        vol_model = lgb.LGBMClassifier(**self.lgb_params)
        vol_model.fit(X_tech_scaled, y_vol)
        models['vol_expansion'] = vol_model

        # Train SENTIMENT volatility model (combined features)
        # This is the key addition for ensemble parity!
        sent_vol_model = lgb.LGBMClassifier(**self.lgb_params)
        sent_vol_model.fit(X_combined_scaled, y_vol)
        models['sentiment_vol'] = sent_vol_model

        # Train breakout high model (tech features)
        y_high = targets.loc[mask, 'new_high_10'].values
        high_model = lgb.LGBMClassifier(**self.lgb_params)
        high_model.fit(X_tech_scaled, y_high)
        models['breakout_high'] = high_model

        # Train breakout low model (tech features)
        y_low = targets.loc[mask, 'new_low_10'].values
        low_model = lgb.LGBMClassifier(**self.lgb_params)
        low_model.fit(X_tech_scaled, y_low)
        models['breakout_low'] = low_model

        # Train ATR forecast model (tech features)
        y_atr = targets.loc[mask, 'future_atr_5'].values
        atr_params = self.lgb_params.copy()
        atr_params['objective'] = 'regression'
        atr_params['metric'] = 'rmse'
        atr_model = lgb.LGBMRegressor(**atr_params)
        atr_model.fit(X_tech_scaled, y_atr)
        models['atr_forecast'] = atr_model

        return models, tech_scaler, sent_scaler

    def export_fold_to_onnx(
        self,
        models: Dict,
        tech_scaler: StandardScaler,
        sent_scaler: StandardScaler,
        fold_info: Dict,
        output_dir: Path
    ):
        """Export a fold's models to ONNX format including sentiment model."""
        output_dir.mkdir(parents=True, exist_ok=True)

        n_tech_features = len(self.technical_feature_names)
        n_sent_features = len(self.sentiment_feature_names)
        n_combined_features = n_tech_features + n_sent_features

        tech_initial_type = [('features', FloatTensorType([None, n_tech_features]))]
        combined_initial_type = [('features', FloatTensorType([None, n_combined_features]))]

        # Export technical models (use tech features only)
        tech_model_files = {
            'vol_expansion': 'vol_expansion_model.onnx',
            'breakout_high': 'breakout_high_model.onnx',
            'breakout_low': 'breakout_low_model.onnx',
            'atr_forecast': 'atr_forecast_model.onnx'
        }

        for model_name, filename in tech_model_files.items():
            model = models[model_name]
            onnx_model = convert_lightgbm(
                model,
                initial_types=tech_initial_type,
                target_opset=12
            )
            onnx.save_model(onnx_model, str(output_dir / filename))

        # Export sentiment vol model (uses combined features)
        sent_vol_model = models['sentiment_vol']
        sent_onnx_model = convert_lightgbm(
            sent_vol_model,
            initial_types=combined_initial_type,
            target_opset=12
        )
        onnx.save_model(sent_onnx_model, str(output_dir / 'sentiment_vol_model.onnx'))

        # Export technical scaler parameters
        scaler_params = {
            'feature_names': self.technical_feature_names,
            'means': tech_scaler.mean_.tolist(),
            'scales': tech_scaler.scale_.tolist(),
            'n_features': n_tech_features
        }
        with open(output_dir / 'scaler_params.json', 'w') as f:
            json.dump(scaler_params, f, indent=2)

        # Export sentiment scaler parameters
        sent_scaler_params = {
            'feature_names': self.sentiment_feature_names,
            'means': sent_scaler.mean_.tolist(),
            'scales': sent_scaler.scale_.tolist(),
            'n_features': n_sent_features
        }
        with open(output_dir / 'sentiment_scaler_params.json', 'w') as f:
            json.dump(sent_scaler_params, f, indent=2)

        # Export strategy config with ensemble thresholds
        # NOTE: Keys must match C# StrategyConfig property names exactly
        strategy_config = {
            'min_vol_expansion_prob': 0.40,       # Technical vol threshold
            'min_sentiment_vol_prob': 0.55,       # Sentiment vol threshold (NEW!)
            'min_breakout_prob': 0.45,
            'tp_atr_mult_base': 2.5,
            'sl_atr_mult_base': 1.25,
            'tp_adjustment_factor': 0.25,
            'max_holding_bars': 20,
            'base_contracts': 1,
            'max_contracts': 3,
            'vol_sizing_factor': 1.0,
            'ensemble_mode': 'agreement',          # Both filters must pass
            'fold_info': {
                'train_start': fold_info['train_start'].isoformat(),
                'train_end': fold_info['train_end'].isoformat(),
                'valid_from': fold_info['test_start'].isoformat(),
                'valid_until': fold_info['test_end'].isoformat()
            }
        }
        with open(output_dir / 'strategy_config.json', 'w') as f:
            json.dump(strategy_config, f, indent=2)

        logger.info(f"  Exported models to {output_dir}")

    def export_all_folds(
        self,
        start_date: Optional[str] = None,
        max_folds: Optional[int] = None
    ) -> List[Dict]:
        """Export ONNX models for all walk-forward folds including sentiment model."""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX dependencies not available. Install: pip install onnxmltools onnx onnxruntime lightgbm")

        # Load data (now returns tech + sentiment features)
        tech_features, sent_features, targets = self.load_data()

        # Get fold ranges
        data_start = tech_features.index.min()
        data_end = tech_features.index.max()
        folds = self.get_fold_ranges(data_start, data_end, start_date)

        if max_folds:
            folds = folds[:max_folds]
            logger.info(f"Limited to {max_folds} folds")

        # Process each fold
        successful_folds = []
        for fold_info in folds:
            fold_id = fold_info['model_id']
            logger.info(f"\nProcessing fold {fold_info['fold']}: {fold_id}")
            logger.info(f"  Train: {fold_info['train_start'].date()} to {fold_info['train_end'].date()}")
            logger.info(f"  Valid: {fold_info['test_start'].date()} to {fold_info['test_end'].date()}")

            # Train models (including sentiment model)
            models, tech_scaler, sent_scaler = self.train_fold_models(
                tech_features, sent_features, targets,
                fold_info['train_start'],
                fold_info['train_end']
            )

            if models is None:
                logger.warning(f"  Skipping fold {fold_id} - insufficient data")
                continue

            # Export to ONNX (including sentiment model)
            fold_dir = self.output_base_dir / f"fold_{fold_id}"
            self.export_fold_to_onnx(models, tech_scaler, sent_scaler, fold_info, fold_dir)

            successful_folds.append({
                'fold': fold_info['fold'],
                'model_id': fold_id,
                'folder': str(fold_dir),
                'valid_from': fold_info['test_start'].strftime('%Y-%m-%d'),
                'valid_until': fold_info['test_end'].strftime('%Y-%m-%d'),
                'train_samples': len(tech_features[(tech_features.index >= fold_info['train_start']) &
                                                    (tech_features.index <= fold_info['train_end'])]),
                'has_sentiment_model': True
            })

        # Save schedule
        self.schedule = successful_folds
        schedule_df = pd.DataFrame(successful_folds)
        schedule_path = self.output_base_dir / 'model_schedule.csv'
        schedule_df.to_csv(schedule_path, index=False)
        logger.info(f"\nSaved model schedule to {schedule_path}")

        # Create master schedule JSON
        master_schedule = {
            'created': datetime.now().isoformat(),
            'methodology': {
                'train_window_days': self.train_days,
                'test_window_days': self.test_days,
                'total_folds': len(successful_folds)
            },
            'folds': successful_folds
        }
        with open(self.output_base_dir / 'master_schedule.json', 'w') as f:
            json.dump(master_schedule, f, indent=2, default=str)

        return successful_folds


def create_nt8_backtest_instructions(output_dir: Path, folds: List[Dict]):
    """Create instructions for running walk-forward backtest in NT8."""
    instructions = """
================================================================================
NINJATRADER 8 WALK-FORWARD BACKTEST INSTRUCTIONS
================================================================================

This folder contains {num_folds} sets of ONNX models, each valid for a specific
date range. To perform a proper walk-forward backtest in NinjaTrader 8:

METHOD 1: MANUAL (Recommended for accuracy)
-------------------------------------------
For each fold in the schedule below:
1. Copy the fold's models to Documents\\SKIE_Ninja\\models\\
2. Restart NinjaTrader 8 (or reload strategy)
3. Run backtest for the "Valid From" to "Valid Until" date range only
4. Record the results
5. Repeat for next fold
6. Sum all results for total walk-forward performance

MODEL SCHEDULE:
{schedule}

METHOD 2: BATCH SCRIPT (Faster but requires multiple NT8 sessions)
------------------------------------------------------------------
A PowerShell script has been created to automate model swapping:
  .\\swap_models.ps1 -FoldId <fold_id>

Example workflow:
  1. Run: .\\swap_models.ps1 -FoldId 20240101
  2. Start NT8, run backtest for 2024-01-01 to 2024-01-05
  3. Close NT8
  4. Repeat for each fold

IMPORTANT NOTES:
----------------
- Each model set is ONLY valid for its specified date range
- Using models outside their valid range will give incorrect results
- The total walk-forward result = sum of all fold results
- This methodology matches the Python walk-forward that achieved +$502K

FOLDER STRUCTURE:
{folder_structure}

For questions, see: docs/NINJATRADER_INSTALLATION.md
================================================================================
""".format(
        num_folds=len(folds),
        schedule="\n".join([
            f"  Fold {f['fold']:3d}: {f['model_id']} | Valid: {f['valid_from']} to {f['valid_until']}"
            for f in folds
        ]),
        folder_structure="\n".join([
            f"  fold_{f['model_id']}/  <- Use for {f['valid_from']} to {f['valid_until']}"
            for f in folds[:5]
        ]) + "\n  ..." if len(folds) > 5 else ""
    )

    with open(output_dir / 'README_NT8_BACKTEST.txt', 'w') as f:
        f.write(instructions)

    logger.info(f"Created backtest instructions: {output_dir / 'README_NT8_BACKTEST.txt'}")


def create_model_swap_script(output_dir: Path, folds: List[Dict]):
    """Create PowerShell script for swapping models."""
    script = '''# SKIE_Ninja Walk-Forward Model Swap Script
# Usage: .\\swap_models.ps1 -FoldId 20240101

param(
    [Parameter(Mandatory=$true)]
    [string]$FoldId,

    [string]$TargetDir = "$env:USERPROFILE\\Documents\\SKIE_Ninja\\models"
)

$SourceDir = Join-Path $PSScriptRoot "fold_$FoldId"

if (-not (Test-Path $SourceDir)) {
    Write-Error "Fold not found: $SourceDir"
    Write-Host "Available folds:"
    Get-ChildItem $PSScriptRoot -Directory -Filter "fold_*" | ForEach-Object { Write-Host "  $($_.Name)" }
    exit 1
}

# Backup current models
$BackupDir = Join-Path $TargetDir "backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
if (Test-Path $TargetDir) {
    Write-Host "Backing up current models to: $BackupDir"
    Copy-Item $TargetDir $BackupDir -Recurse
}

# Create target directory if needed
New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

# Copy new models
Write-Host "Copying models from fold_$FoldId..."
Copy-Item "$SourceDir\\*" $TargetDir -Force

# Read and display validity info
$config = Get-Content "$TargetDir\\strategy_config.json" | ConvertFrom-Json
Write-Host ""
Write-Host "Models swapped successfully!" -ForegroundColor Green
Write-Host "Valid from: $($config.fold_info.valid_from)"
Write-Host "Valid until: $($config.fold_info.valid_until)"
Write-Host ""
Write-Host "IMPORTANT: Restart NinjaTrader 8 before backtesting!"
'''

    with open(output_dir / 'swap_models.ps1', 'w') as f:
        f.write(script)

    logger.info(f"Created model swap script: {output_dir / 'swap_models.ps1'}")


def main():
    parser = argparse.ArgumentParser(description='Export walk-forward ONNX models for NT8')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--max-folds', type=int, help='Maximum number of folds to export')
    parser.add_argument('--train-days', type=int, default=180, help='Training window in days')
    parser.add_argument('--test-days', type=int, default=5, help='Test window in days')
    args = parser.parse_args()

    print("=" * 80)
    print(" WALK-FORWARD ONNX EXPORT FOR NINJATRADER 8")
    print(" Generates multiple model snapshots for proper walk-forward backtesting")
    print("=" * 80)

    if not ONNX_AVAILABLE:
        print("\nERROR: Required packages not installed.")
        print("Run: pip install onnxmltools onnx onnxruntime lightgbm")
        return

    exporter = WalkForwardONNXExporter(
        train_days=args.train_days,
        test_days=args.test_days
    )

    folds = exporter.export_all_folds(
        start_date=args.start,
        max_folds=args.max_folds
    )

    if folds:
        # Create helper files
        create_nt8_backtest_instructions(exporter.output_base_dir, folds)
        create_model_swap_script(exporter.output_base_dir, folds)

        print("\n" + "=" * 80)
        print(" EXPORT COMPLETE")
        print("=" * 80)
        print(f"\nExported {len(folds)} walk-forward model sets to:")
        print(f"  {exporter.output_base_dir}")
        print(f"\nFiles created:")
        print(f"  - {len(folds)} fold directories (fold_YYYYMMDD/)")
        print(f"  - model_schedule.csv (date ranges for each fold)")
        print(f"  - master_schedule.json (complete metadata)")
        print(f"  - README_NT8_BACKTEST.txt (instructions)")
        print(f"  - swap_models.ps1 (automation script)")
        print(f"\nSee README_NT8_BACKTEST.txt for backtest instructions")
    else:
        print("\nNo folds exported. Check data availability.")


if __name__ == "__main__":
    main()
