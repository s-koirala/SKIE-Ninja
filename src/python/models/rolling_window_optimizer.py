"""
SKIE-Ninja Rolling Window CV Optimizer

Grid search optimization for rolling window cross-validation parameters.
Tests different train/test window sizes to find optimal configuration.

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss
)
import xgboost as xgb

logger = logging.getLogger(__name__)


@dataclass
class GridSearchResult:
    """Container for grid search results."""
    timeframe: str
    train_window: int
    test_window: int
    embargo: int
    n_folds: int
    mean_auc: float
    std_auc: float
    mean_accuracy: float
    mean_f1: float
    mean_precision: float
    mean_recall: float
    fold_aucs: List[float]
    total_samples: int
    train_samples_per_fold: int


class RollingWindowOptimizer:
    """
    Optimizes rolling window CV parameters for time series ML.

    Grid searches over:
    - Train window size (days)
    - Test window size (days)
    - Embargo period (bars)
    """

    def __init__(
        self,
        bars_per_day: int = 78,  # RTH bars for 5-min
        min_train_days: int = 30,
        max_train_days: int = 180,
        min_test_days: int = 5,
        max_test_days: int = 30,
        n_jobs: int = -1
    ):
        """
        Initialize optimizer.

        Args:
            bars_per_day: Number of bars per trading day (RTH)
            min_train_days: Minimum training window in days
            max_train_days: Maximum training window in days
            min_test_days: Minimum test window in days
            max_test_days: Maximum test window in days
            n_jobs: Number of parallel jobs
        """
        self.bars_per_day = bars_per_day
        self.min_train_days = min_train_days
        self.max_train_days = max_train_days
        self.min_test_days = min_test_days
        self.max_test_days = max_test_days
        self.n_jobs = n_jobs

        self.scaler = StandardScaler()

    def create_grid(
        self,
        train_days_options: Optional[List[int]] = None,
        test_days_options: Optional[List[int]] = None,
        embargo_options: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Create parameter grid for search.

        Returns:
            List of parameter dictionaries
        """
        if train_days_options is None:
            # Default: 30, 60, 90, 120, 180 days
            train_days_options = [30, 60, 90, 120, 180]

        if test_days_options is None:
            # Default: 5, 10, 20 days
            test_days_options = [5, 10, 20]

        if embargo_options is None:
            # Default: based on feature lookback
            # For 5-min: 42 bars = ~3.5 hours
            # For 15-min: 14 bars = ~3.5 hours
            embargo_options = [max(1, 210 // (self.bars_per_day // 78 * 5))]

        grid = []
        for train_days in train_days_options:
            for test_days in test_days_options:
                for embargo in embargo_options:
                    grid.append({
                        'train_days': train_days,
                        'test_days': test_days,
                        'embargo': embargo,
                        'train_bars': train_days * self.bars_per_day,
                        'test_bars': test_days * self.bars_per_day
                    })

        return grid

    def rolling_window_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_size: int,
        test_size: int,
        embargo: int = 0,
        min_folds: int = 3
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate rolling window train/test splits.

        Args:
            X: Feature matrix
            y: Target array
            train_size: Training window size in bars
            test_size: Test window size in bars
            embargo: Gap between train and test
            min_folds: Minimum number of folds required

        Returns:
            List of (train_idx, test_idx) tuples
        """
        n_samples = len(X)
        splits = []

        # Calculate how many folds we can create
        total_needed = train_size + embargo + test_size
        remaining = n_samples - total_needed

        if remaining < 0:
            logger.warning(f"Not enough data for train={train_size}, test={test_size}, embargo={embargo}")
            return []

        # Calculate step size to get reasonable number of folds
        n_possible_folds = remaining // test_size + 1

        if n_possible_folds < min_folds:
            logger.warning(f"Only {n_possible_folds} folds possible, need {min_folds}")
            return []

        # Generate folds
        current_start = 0
        while True:
            train_end = current_start + train_size
            test_start = train_end + embargo
            test_end = test_start + test_size

            if test_end > n_samples:
                break

            train_idx = np.arange(current_start, train_end)
            test_idx = np.arange(test_start, test_end)

            splits.append((train_idx, test_idx))

            # Slide window by test_size
            current_start += test_size

        return splits

    def evaluate_config(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: Dict,
        selected_features: Optional[List[int]] = None,
        verbose: bool = False
    ) -> Optional[GridSearchResult]:
        """
        Evaluate a single configuration.

        Args:
            X: Feature matrix
            y: Target array
            config: Configuration dictionary
            selected_features: Indices of features to use
            verbose: Print progress

        Returns:
            GridSearchResult or None if evaluation failed
        """
        train_bars = config['train_bars']
        test_bars = config['test_bars']
        embargo = config['embargo']

        # Use subset of features if specified
        if selected_features is not None:
            X_subset = X[:, selected_features]
        else:
            X_subset = X

        # Generate splits
        splits = self.rolling_window_cv(X_subset, y, train_bars, test_bars, embargo)

        if len(splits) < 3:
            return None

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            try:
                # Get data
                X_train, X_test = X_subset[train_idx], X_subset[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train XGBoost (fast config for grid search)
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=self.n_jobs,
                    verbosity=0,
                    use_label_encoder=False,
                    eval_metric='auc'
                )

                model.fit(X_train_scaled, y_train)

                # Predict
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]

                # Calculate metrics
                fold_results.append({
                    'auc': roc_auc_score(y_test, y_prob),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'train_samples': len(train_idx),
                    'test_samples': len(test_idx)
                })

            except Exception as e:
                logger.warning(f"Fold {fold_idx} failed: {e}")
                continue

        if len(fold_results) < 3:
            return None

        # Aggregate results
        aucs = [r['auc'] for r in fold_results]

        result = GridSearchResult(
            timeframe=f"{self.bars_per_day}bars/day",
            train_window=config['train_days'],
            test_window=config['test_days'],
            embargo=embargo,
            n_folds=len(fold_results),
            mean_auc=np.mean(aucs),
            std_auc=np.std(aucs),
            mean_accuracy=np.mean([r['accuracy'] for r in fold_results]),
            mean_f1=np.mean([r['f1'] for r in fold_results]),
            mean_precision=np.mean([r['precision'] for r in fold_results]),
            mean_recall=np.mean([r['recall'] for r in fold_results]),
            fold_aucs=aucs,
            total_samples=len(X),
            train_samples_per_fold=fold_results[0]['train_samples']
        )

        if verbose:
            logger.info(f"Train={config['train_days']}d, Test={config['test_days']}d: "
                       f"AUC={result.mean_auc:.4f} (+/- {result.std_auc:.4f}), "
                       f"Folds={result.n_folds}")

        return result

    def run_grid_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        grid: Optional[List[Dict]] = None,
        selected_features: Optional[List[int]] = None,
        verbose: bool = True
    ) -> List[GridSearchResult]:
        """
        Run grid search over all configurations.

        Args:
            X: Feature matrix
            y: Target array
            grid: Parameter grid (uses default if None)
            selected_features: Feature indices to use
            verbose: Print progress

        Returns:
            List of GridSearchResult sorted by mean AUC
        """
        if grid is None:
            grid = self.create_grid()

        logger.info(f"Running grid search with {len(grid)} configurations...")
        logger.info(f"Data shape: X={X.shape}, y={y.shape}")

        results = []

        for i, config in enumerate(grid):
            if verbose:
                logger.info(f"\n[{i+1}/{len(grid)}] Testing train={config['train_days']}d, "
                           f"test={config['test_days']}d, embargo={config['embargo']}")

            result = self.evaluate_config(X, y, config, selected_features, verbose=False)

            if result is not None:
                results.append(result)
                if verbose:
                    logger.info(f"  -> AUC: {result.mean_auc:.4f} (+/- {result.std_auc:.4f}), "
                               f"Acc: {result.mean_accuracy:.4f}, Folds: {result.n_folds}")
            else:
                if verbose:
                    logger.warning(f"  -> Skipped (not enough data)")

        # Sort by mean AUC
        results.sort(key=lambda x: x.mean_auc, reverse=True)

        return results

    def save_results(
        self,
        results: List[GridSearchResult],
        output_path: Path,
        timeframe: str
    ) -> None:
        """Save grid search results to CSV and JSON."""
        # Convert to DataFrame
        rows = []
        for r in results:
            rows.append({
                'timeframe': timeframe,
                'train_days': r.train_window,
                'test_days': r.test_window,
                'embargo_bars': r.embargo,
                'n_folds': r.n_folds,
                'mean_auc': r.mean_auc,
                'std_auc': r.std_auc,
                'mean_accuracy': r.mean_accuracy,
                'mean_f1': r.mean_f1,
                'mean_precision': r.mean_precision,
                'mean_recall': r.mean_recall,
                'total_samples': r.total_samples,
                'train_samples_per_fold': r.train_samples_per_fold
            })

        df = pd.DataFrame(rows)
        csv_path = output_path / f'rolling_window_grid_{timeframe}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to {csv_path}")

        # Also save full results as JSON
        json_path = output_path / f'rolling_window_grid_{timeframe}.json'
        with open(json_path, 'w') as f:
            json.dump([{
                'timeframe': r.timeframe,
                'train_days': r.train_window,
                'test_days': r.test_window,
                'embargo': r.embargo,
                'n_folds': r.n_folds,
                'mean_auc': r.mean_auc,
                'std_auc': r.std_auc,
                'fold_aucs': r.fold_aucs,
                'mean_accuracy': r.mean_accuracy,
                'mean_f1': r.mean_f1
            } for r in results], f, indent=2)


def run_optimization(
    timeframe: str = '5min',
    output_dir: str = 'data/processed'
) -> List[GridSearchResult]:
    """
    Run rolling window optimization for a given timeframe.

    Args:
        timeframe: '5min' or '15min'
        output_dir: Output directory for results

    Returns:
        List of GridSearchResult
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from utils.data_resampler import DataResampler
    from feature_engineering.feature_pipeline import build_feature_matrix

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load and resample data
    logger.info(f"\n{'='*60}")
    logger.info(f"ROLLING WINDOW OPTIMIZATION - {timeframe.upper()}")
    logger.info(f"{'='*60}")

    data_path = Path('data/raw/market/ES_1min_databento.csv')
    logger.info(f"Loading data from {data_path}...")

    es_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(es_data):,} 1-min bars")

    # Resample to target timeframe
    logger.info(f"\nResampling to {timeframe} RTH...")
    resampler = DataResampler()
    resampled = resampler.resample(es_data, timeframe, rth_only=True)
    logger.info(f"Resampled to {len(resampled):,} {timeframe} bars")

    # Build features
    logger.info("\nBuilding features...")
    features = build_feature_matrix(
        resampled,
        symbol='ES',
        include_lagged=True,
        include_interactions=True,
        include_targets=True,
        include_macro=False,
        include_sentiment=False,
        include_intermarket=False,
        include_alternative=False,
        dropna=False
    )

    # Clean NaN
    if 'hurst_20' in features.columns:
        features = features.drop(columns=['hurst_20'])
    features_clean = features.dropna()
    logger.info(f"Clean features: {features_clean.shape}")

    # Prepare X and y
    target_col = 'target_direction_1'
    feature_cols = [c for c in features_clean.columns if not c.startswith('target_')]

    X = features_clean[feature_cols].values
    y = features_clean[target_col].values

    # Load selected features from previous analysis (if available)
    rankings_path = Path('data/processed/feature_rankings.csv')
    if rankings_path.exists():
        rankings = pd.read_csv(rankings_path)
        top_features = rankings['feature'].tolist()[:75]
        # Find indices of these features in our current feature set
        selected_idx = [feature_cols.index(f) for f in top_features if f in feature_cols]
        logger.info(f"Using {len(selected_idx)} pre-selected features")
    else:
        selected_idx = None
        logger.info("Using all features")

    # Set up optimizer
    if timeframe == '5min':
        bars_per_day = 78  # 390 min / 5 min
        embargo = 42  # ~3.5 hours
    elif timeframe == '15min':
        bars_per_day = 26  # 390 min / 15 min
        embargo = 14  # ~3.5 hours
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    optimizer = RollingWindowOptimizer(bars_per_day=bars_per_day)

    # Create grid with various train/test combinations
    grid = optimizer.create_grid(
        train_days_options=[30, 60, 90, 120, 180],
        test_days_options=[5, 10, 20],
        embargo_options=[embargo]
    )

    # Run grid search
    results = optimizer.run_grid_search(
        X, y, grid,
        selected_features=selected_idx,
        verbose=True
    )

    # Save results
    optimizer.save_results(results, output_path, timeframe)

    # Print top 5 configurations
    logger.info(f"\n{'='*60}")
    logger.info(f"TOP 5 CONFIGURATIONS - {timeframe.upper()}")
    logger.info(f"{'='*60}")

    for i, r in enumerate(results[:5], 1):
        logger.info(f"\n#{i}: Train={r.train_window}d, Test={r.test_window}d, Embargo={r.embargo}")
        logger.info(f"    AUC:      {r.mean_auc:.4f} (+/- {r.std_auc:.4f})")
        logger.info(f"    Accuracy: {r.mean_accuracy:.4f}")
        logger.info(f"    F1:       {r.mean_f1:.4f}")
        logger.info(f"    Folds:    {r.n_folds}")

    return results


def compare_timeframes(results_5min: List[GridSearchResult],
                       results_15min: List[GridSearchResult]) -> None:
    """Compare best results between timeframes."""
    print("\n" + "="*70)
    print("TIMEFRAME COMPARISON - BEST CONFIGURATIONS")
    print("="*70)

    print("\n5-MIN TIMEFRAME (Best):")
    if results_5min:
        r = results_5min[0]
        print(f"  Train={r.train_window}d, Test={r.test_window}d, Embargo={r.embargo}")
        print(f"  AUC: {r.mean_auc:.4f} (+/- {r.std_auc:.4f})")
        print(f"  Accuracy: {r.mean_accuracy:.4f}, F1: {r.mean_f1:.4f}")
        print(f"  Folds: {r.n_folds}, Train samples/fold: {r.train_samples_per_fold:,}")
    else:
        print("  No valid results")

    print("\n15-MIN TIMEFRAME (Best):")
    if results_15min:
        r = results_15min[0]
        print(f"  Train={r.train_window}d, Test={r.test_window}d, Embargo={r.embargo}")
        print(f"  AUC: {r.mean_auc:.4f} (+/- {r.std_auc:.4f})")
        print(f"  Accuracy: {r.mean_accuracy:.4f}, F1: {r.mean_f1:.4f}")
        print(f"  Folds: {r.n_folds}, Train samples/fold: {r.train_samples_per_fold:,}")
    else:
        print("  No valid results")

    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    if results_5min and results_15min:
        best_5 = results_5min[0]
        best_15 = results_15min[0]

        if best_5.mean_auc > best_15.mean_auc:
            winner = "5-MIN"
            diff = best_5.mean_auc - best_15.mean_auc
        else:
            winner = "15-MIN"
            diff = best_15.mean_auc - best_5.mean_auc

        print(f"\n  Winner: {winner} timeframe (+{diff:.4f} AUC)")
        print(f"\n  Note: Consider trade frequency vs. signal quality tradeoff")
        print(f"  - 5-min: More trades, faster signals, more noise")
        print(f"  - 15-min: Fewer trades, stronger signals, less noise")


# Exports
__all__ = [
    'RollingWindowOptimizer',
    'GridSearchResult',
    'run_optimization',
    'compare_timeframes',
]


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run for both timeframes
    print("\n" + "="*70)
    print("SKIE-NINJA ROLLING WINDOW GRID OPTIMIZATION")
    print("="*70)

    results_5min = run_optimization('5min')
    results_15min = run_optimization('15min')

    compare_timeframes(results_5min, results_15min)
