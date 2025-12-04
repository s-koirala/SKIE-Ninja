"""
Model Training Module for SKIE-Ninja
=====================================
Implements walk-forward cross-validation and multiple ML models for
directional prediction on futures markets.

Models:
- LightGBM (fast, SOTA for tabular data)
- XGBoost (robust gradient boosting)
- Random Forest (interpretable baseline)

Key Features:
- Walk-forward cross-validation (proper time-series validation)
- Class imbalance handling
- Feature importance analysis
- Model calibration
- ONNX export for NinjaTrader integration
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import joblib
import json
from datetime import datetime

# ML imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    log_loss, brier_score_loss
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# Gradient boosting
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logging.warning("LightGBM not installed. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logging.warning("XGBoost not installed. Install with: pip install xgboost")

# ONNX export
try:
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    logging.warning("ONNX not installed. Install with: pip install onnx skl2onnx")

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model training."""
    # LightGBM parameters
    lgb_params: Dict = field(default_factory=lambda: {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    })

    # XGBoost parameters
    xgb_params: Dict = field(default_factory=lambda: {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'tree_method': 'hist',
        'random_state': 42
    })

    # Random Forest parameters
    rf_params: Dict = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 50,
        'min_samples_leaf': 25,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': 42
    })

    # Training settings
    n_estimators: int = 500
    early_stopping_rounds: int = 50
    n_splits: int = 5  # Number of walk-forward splits
    test_size_pct: float = 0.20  # Hold out last 20% for final test
    scale_features: bool = True

    # Enhanced CV settings
    cv_window_type: str = 'expanding'  # 'expanding' or 'rolling'
    cv_embargo_size: int = 210  # Embargo period (bars between train/test)
    cv_max_train_size: Optional[int] = None  # Max train window for rolling
    rth_only: bool = True  # Filter to Regular Trading Hours only
    check_leakage: bool = True  # Run data leakage checks


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    log_loss_val: float
    brier_score: float
    confusion_mat: np.ndarray

    def to_dict(self) -> Dict:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'auc_roc': self.auc_roc,
            'log_loss': self.log_loss_val,
            'brier_score': self.brier_score,
            'confusion_matrix': self.confusion_mat.tolist()
        }


class WalkForwardValidator:
    """
    Walk-forward cross-validation for time series.

    Unlike regular k-fold, this maintains temporal ordering:
    - Train on historical data
    - Validate on future (out-of-sample) data
    - Supports both expanding window and rolling (sliding) window
    - Includes embargo period to prevent data leakage

    IMPORTANT: We only trade during Regular Trading Hours (RTH).
    For ES futures: 9:30 AM - 4:00 PM ET (6:30 AM - 1:00 PM PT)
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        min_train_size: Optional[int] = None,
        max_train_size: Optional[int] = None,  # NEW: For rolling window
        embargo_size: int = 0,  # NEW: Gap between train and test
        window_type: str = 'expanding'  # NEW: 'expanding' or 'rolling'
    ):
        """
        Initialize walk-forward validator.

        Parameters:
        -----------
        n_splits : int
            Number of cross-validation folds
        test_size : int, optional
            Size of each test fold
        min_train_size : int, optional
            Minimum training window size
        max_train_size : int, optional
            Maximum training window size (for rolling window)
        embargo_size : int
            Number of samples to skip between train and test
            to prevent data leakage from features using future data
        window_type : str
            'expanding' - train window grows each fold (default)
            'rolling' - train window maintains fixed size (slides)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.embargo_size = embargo_size
        self.window_type = window_type

        if window_type not in ['expanding', 'rolling']:
            raise ValueError("window_type must be 'expanding' or 'rolling'")

        if window_type == 'rolling' and max_train_size is None:
            logger.warning("Rolling window requires max_train_size. Using min_train_size.")

    def split(self, X: np.ndarray, y: np.ndarray = None
              ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate train/test indices for walk-forward validation."""
        n_samples = len(X)

        # Calculate test size if not specified
        test_size = self.test_size or n_samples // (self.n_splits + 1)

        # Minimum training size
        min_train = self.min_train_size or test_size * 2

        # Maximum training size (for rolling window)
        max_train = self.max_train_size or min_train

        # Generate splits
        for i in range(self.n_splits):
            # Test window starts after embargo
            test_start = min_train + self.embargo_size + i * test_size
            test_end = test_start + test_size

            if test_end > n_samples:
                break

            # Training window ends before embargo
            train_end = test_start - self.embargo_size

            if self.window_type == 'expanding':
                # Expanding window: train from start to train_end
                train_start = 0
            else:
                # Rolling window: train from (train_end - max_train) to train_end
                train_start = max(0, train_end - max_train)

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

    def get_n_splits(self, X: np.ndarray = None) -> int:
        """Return the number of splits."""
        return self.n_splits


def filter_rth_only(df: pd.DataFrame, timezone: str = 'America/New_York') -> pd.DataFrame:
    """
    Filter DataFrame to Regular Trading Hours (RTH) only.

    IMPORTANT: We only trade during RTH for ES futures.
    RTH for ES/NQ: 9:30 AM - 4:00 PM Eastern Time (ET)

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    timezone : str
        Timezone for RTH hours (default: America/New_York for ET)

    Returns:
    --------
    pd.DataFrame
        Filtered to RTH only
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # Convert to Eastern Time if needed
    if df.index.tz is None:
        # Assume UTC if no timezone
        idx_et = df.index.tz_localize('UTC').tz_convert(timezone)
    else:
        idx_et = df.index.tz_convert(timezone)

    # RTH: 9:30 AM - 4:00 PM ET
    hour = idx_et.hour
    minute = idx_et.minute

    # Create RTH mask
    rth_mask = (
        ((hour == 9) & (minute >= 30)) |  # 9:30 AM onwards
        ((hour >= 10) & (hour < 16)) |     # 10:00 AM - 3:59 PM
        ((hour == 16) & (minute == 0))     # 4:00 PM exactly
    )

    # Also exclude weekends
    weekday_mask = idx_et.dayofweek < 5  # Monday=0 to Friday=4

    final_mask = rth_mask & weekday_mask

    logger.info(f"RTH filter: {final_mask.sum():,} of {len(df):,} bars "
                f"({100*final_mask.sum()/len(df):.1f}%)")

    return df[final_mask]


def check_data_leakage(
    features: pd.DataFrame,
    target_col: str = 'target_direction_1',
    lag_features: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Check for potential data leakage in feature matrix.

    Data leakage sources:
    1. Target leakage: Features correlated > 0.95 with target
    2. Future leakage: Features using forward-looking data
    3. Lookahead bias: Features calculated with future prices

    Parameters:
    -----------
    features : pd.DataFrame
        Feature matrix with targets
    target_col : str
        Name of target column
    lag_features : List[str], optional
        List of known lagged features to check

    Returns:
    --------
    Dict with leakage analysis results
    """
    logger.info("Checking for data leakage...")
    results = {
        'target_leakage': [],
        'high_correlation': [],
        'suspicious_features': [],
        'warnings': [],
        'passed': True
    }

    if target_col not in features.columns:
        results['warnings'].append(f"Target column '{target_col}' not found")
        return results

    target = features[target_col]
    feature_cols = [c for c in features.columns if not c.startswith('target_')]

    # Check 1: Features highly correlated with target (potential leakage)
    for col in feature_cols:
        if col in features.columns:
            corr = features[col].corr(target)
            if abs(corr) > 0.95:
                results['target_leakage'].append((col, corr))
                results['passed'] = False
                logger.warning(f"TARGET LEAKAGE: {col} has {corr:.3f} correlation with target")
            elif abs(corr) > 0.80:
                results['high_correlation'].append((col, corr))
                logger.warning(f"High correlation: {col} = {corr:.3f}")

    # Check 2: Suspicious feature names (forward-looking)
    forward_patterns = ['future', 'forward', 'next', 'tomorrow', 'lead']
    for col in feature_cols:
        col_lower = col.lower()
        for pattern in forward_patterns:
            if pattern in col_lower and 'target' not in col_lower:
                results['suspicious_features'].append(col)
                results['warnings'].append(f"Suspicious feature name: {col}")

    # Check 3: Target columns that shouldn't be features
    target_features = [c for c in feature_cols if 'target' in c.lower()]
    if target_features:
        results['target_leakage'].extend([(f, 1.0) for f in target_features])
        results['warnings'].append(f"Target columns in features: {target_features}")
        results['passed'] = False

    # Check 4: Perfect prediction features
    for col in feature_cols[:50]:  # Check first 50 for speed
        if col in features.columns:
            # Check if feature perfectly predicts target
            unique_combos = features.groupby(col)[target_col].nunique()
            if (unique_combos == 1).all():
                results['target_leakage'].append((col, 'perfect_predictor'))
                results['passed'] = False
                logger.warning(f"PERFECT PREDICTOR (likely leakage): {col}")

    # Summary
    if results['passed']:
        logger.info("Data leakage check PASSED - no obvious leakage detected")
    else:
        logger.error("Data leakage check FAILED - review flagged features")

    return results


def calculate_embargo_size(
    df: pd.DataFrame,
    max_feature_lookback: int = 200,
    safety_margin: int = 10
) -> int:
    """
    Calculate appropriate embargo size based on feature lookback periods.

    The embargo prevents data leakage from:
    - Rolling features (e.g., 200-bar MA uses future data if no embargo)
    - Lagged targets that may peek into test set

    Parameters:
    -----------
    df : pd.DataFrame
        Data with DatetimeIndex
    max_feature_lookback : int
        Maximum lookback period used in features (default 200 for MA200)
    safety_margin : int
        Additional buffer for safety

    Returns:
    --------
    int
        Recommended embargo size in bars
    """
    # For 1-minute data, embargo should be at least max_lookback
    # For daily data, might need fewer bars

    # Detect data frequency
    if len(df) > 1:
        time_diff = (df.index[1] - df.index[0]).total_seconds()
        if time_diff <= 60:  # 1-minute or less
            # For intraday, use bars directly
            embargo = max_feature_lookback + safety_margin
        elif time_diff <= 3600:  # Hourly
            embargo = max(max_feature_lookback // 60 + safety_margin, 10)
        else:  # Daily or longer
            embargo = max(max_feature_lookback // 390 + safety_margin, 5)  # 390 min/day
    else:
        embargo = max_feature_lookback + safety_margin

    logger.info(f"Calculated embargo size: {embargo} bars")
    return embargo


class ModelTrainer:
    """
    Main model training class with walk-forward validation.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.scaler = StandardScaler() if self.config.scale_features else None
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, List[ModelMetrics]] = {}
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        self.best_model: Optional[Tuple[str, Any]] = None

        logger.info("ModelTrainer initialized")

    def prepare_data(self, features: pd.DataFrame, target_col: str,
                     selected_features: Optional[List[str]] = None
                     ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target for training.

        Parameters:
        -----------
        features : pd.DataFrame
            Full feature matrix including target
        target_col : str
            Name of target column
        selected_features : List[str], optional
            Subset of features to use

        Returns:
        --------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        feature_names : List[str]
            Names of features used
        """
        # Get target
        y = features[target_col].values

        # Get feature columns
        if selected_features:
            feature_cols = [f for f in selected_features if f in features.columns]
        else:
            # Exclude target columns
            feature_cols = [c for c in features.columns if not c.startswith('target_')]

        X = features[feature_cols].values

        logger.info(f"Prepared data: X={X.shape}, y={y.shape}")
        logger.info(f"Target distribution: {np.bincount(y.astype(int))}")

        return X, y, feature_cols

    def train_test_split_temporal(self, X: np.ndarray, y: np.ndarray
                                   ) -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
        """
        Split data temporally (last N% for testing).
        """
        n_samples = len(X)
        split_idx = int(n_samples * (1 - self.config.test_size_pct))

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def _scale_features(self, X_train: np.ndarray, X_test: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using StandardScaler."""
        if self.scaler is None:
            return X_train, X_test

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                        y_proba: np.ndarray) -> ModelMetrics:
        """Evaluate model predictions."""
        return ModelMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0),
            auc_roc=roc_auc_score(y_true, y_proba),
            log_loss_val=log_loss(y_true, y_proba),
            brier_score=brier_score_loss(y_true, y_proba),
            confusion_mat=confusion_matrix(y_true, y_pred)
        )

    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       feature_names: List[str]) -> Tuple[Any, ModelMetrics]:
        """Train LightGBM model."""
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed")

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names,
                               reference=train_data)

        # Train model
        model = lgb.train(
            self.config.lgb_params,
            train_data,
            num_boost_round=self.config.n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
        )

        # Predict
        y_proba = model.predict(X_val)
        y_pred = (y_proba > 0.5).astype(int)

        # Evaluate
        metrics = self._evaluate_model(y_val, y_pred, y_proba)

        return model, metrics

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      feature_names: List[str]) -> Tuple[Any, ModelMetrics]:
        """Train XGBoost model."""
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")

        # Create DMatrices
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

        # Train model
        model = xgb.train(
            self.config.xgb_params,
            dtrain,
            num_boost_round=self.config.n_estimators,
            evals=[(dtrain, 'train'), (dval, 'valid')],
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose_eval=100
        )

        # Predict
        y_proba = model.predict(dval)
        y_pred = (y_proba > 0.5).astype(int)

        # Evaluate
        metrics = self._evaluate_model(y_val, y_pred, y_proba)

        return model, metrics

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            feature_names: List[str]) -> Tuple[Any, ModelMetrics]:
        """Train Random Forest model."""
        model = RandomForestClassifier(**self.config.rf_params)
        model.fit(X_train, y_train)

        # Predict
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        # Evaluate
        metrics = self._evaluate_model(y_val, y_pred, y_proba)

        return model, metrics

    def train_all_models(self, features: pd.DataFrame, target_col: str,
                         selected_features: Optional[List[str]] = None,
                         output_dir: Optional[str] = None
                         ) -> Dict[str, ModelMetrics]:
        """
        Train all available models with walk-forward validation.

        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix with targets
        target_col : str
            Target column name
        selected_features : List[str], optional
            Subset of features to use
        output_dir : str, optional
            Directory to save models and results

        Returns:
        --------
        Dict[str, ModelMetrics]
            Final test set metrics for each model
        """
        logger.info("="*60)
        logger.info("MODEL TRAINING PIPELINE")
        logger.info("="*60)

        # Check for data leakage if enabled
        if self.config.check_leakage:
            logger.info("\n--- Data Leakage Check ---")
            leakage_results = check_data_leakage(features, target_col)
            if not leakage_results['passed']:
                logger.error("DATA LEAKAGE DETECTED! Review flagged features before proceeding.")
                logger.error(f"Flagged: {leakage_results['target_leakage']}")

        # Prepare data
        X, y, feature_names = self.prepare_data(features, target_col, selected_features)

        # Temporal train/test split
        X_train, X_test, y_train, y_test = self.train_test_split_temporal(X, y)

        # Scale features
        X_train_scaled, X_test_scaled = self._scale_features(X_train, X_test)

        # Walk-forward validation on training set with enhanced options
        logger.info(f"\n--- Cross-Validation Setup ---")
        logger.info(f"Window type: {self.config.cv_window_type}")
        logger.info(f"Embargo size: {self.config.cv_embargo_size} bars")
        logger.info(f"Number of folds: {self.config.n_splits}")

        wfv = WalkForwardValidator(
            n_splits=self.config.n_splits,
            embargo_size=self.config.cv_embargo_size,
            window_type=self.config.cv_window_type,
            max_train_size=self.config.cv_max_train_size
        )

        results = {}

        # Train each model type
        model_trainers = []

        if HAS_LIGHTGBM:
            model_trainers.append(('LightGBM', self.train_lightgbm, False))

        if HAS_XGBOOST:
            model_trainers.append(('XGBoost', self.train_xgboost, False))

        model_trainers.append(('RandomForest', self.train_random_forest, True))

        for model_name, train_fn, needs_scaling in model_trainers:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_name}")
            logger.info("="*60)

            X_tr = X_train_scaled if needs_scaling else X_train
            X_te = X_test_scaled if needs_scaling else X_test

            # Walk-forward cross-validation
            fold_metrics = []
            for fold, (train_idx, val_idx) in enumerate(wfv.split(X_tr)):
                logger.info(f"\n--- Fold {fold + 1}/{self.config.n_splits} ---")

                X_fold_train, y_fold_train = X_tr[train_idx], y_train[train_idx]
                X_fold_val, y_fold_val = X_tr[val_idx], y_train[val_idx]

                logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

                try:
                    model, metrics = train_fn(
                        X_fold_train, y_fold_train,
                        X_fold_val, y_fold_val,
                        feature_names
                    )
                    fold_metrics.append(metrics)

                    logger.info(f"Fold {fold + 1} - AUC: {metrics.auc_roc:.4f}, "
                               f"Acc: {metrics.accuracy:.4f}, F1: {metrics.f1:.4f}")
                except Exception as e:
                    logger.error(f"Error in fold {fold + 1}: {e}")

            # Average CV metrics
            if fold_metrics:
                avg_auc = np.mean([m.auc_roc for m in fold_metrics])
                avg_acc = np.mean([m.accuracy for m in fold_metrics])
                avg_f1 = np.mean([m.f1 for m in fold_metrics])

                logger.info(f"\n{model_name} CV Average:")
                logger.info(f"  AUC: {avg_auc:.4f} (+/- {np.std([m.auc_roc for m in fold_metrics]):.4f})")
                logger.info(f"  Accuracy: {avg_acc:.4f}")
                logger.info(f"  F1: {avg_f1:.4f}")

            # Train final model on full training set
            logger.info(f"\nTraining final {model_name} on full training set...")

            # Use last 10% of training for early stopping
            es_split = int(len(X_tr) * 0.9)
            X_train_final = X_tr[:es_split]
            y_train_final = y_train[:es_split]
            X_es = X_tr[es_split:]
            y_es = y_train[es_split:]

            final_model, _ = train_fn(
                X_train_final, y_train_final,
                X_es, y_es,
                feature_names
            )

            # Evaluate on held-out test set
            logger.info(f"\nEvaluating {model_name} on test set...")

            if model_name == 'LightGBM':
                y_proba_test = final_model.predict(X_te)
            elif model_name == 'XGBoost':
                dtest = xgb.DMatrix(X_te, feature_names=feature_names)
                y_proba_test = final_model.predict(dtest)
            else:
                y_proba_test = final_model.predict_proba(X_te)[:, 1]

            y_pred_test = (y_proba_test > 0.5).astype(int)
            test_metrics = self._evaluate_model(y_test, y_pred_test, y_proba_test)

            logger.info(f"\n{model_name} TEST SET RESULTS:")
            logger.info(f"  Accuracy:  {test_metrics.accuracy:.4f}")
            logger.info(f"  Precision: {test_metrics.precision:.4f}")
            logger.info(f"  Recall:    {test_metrics.recall:.4f}")
            logger.info(f"  F1 Score:  {test_metrics.f1:.4f}")
            logger.info(f"  AUC-ROC:   {test_metrics.auc_roc:.4f}")
            logger.info(f"  Log Loss:  {test_metrics.log_loss_val:.4f}")
            logger.info(f"  Brier:     {test_metrics.brier_score:.4f}")

            # Store model and metrics
            self.models[model_name] = final_model
            self.metrics[model_name] = fold_metrics
            results[model_name] = test_metrics

            # Feature importance
            if model_name == 'LightGBM':
                importance = final_model.feature_importance(importance_type='gain')
            elif model_name == 'XGBoost':
                importance = list(final_model.get_score(importance_type='gain').values())
                # Pad with zeros for missing features
                if len(importance) < len(feature_names):
                    importance = [final_model.get_score(importance_type='gain').get(f, 0)
                                  for f in feature_names]
            else:
                importance = final_model.feature_importances_

            self.feature_importance[model_name] = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

        # Find best model
        best_name = max(results.keys(), key=lambda k: results[k].auc_roc)
        self.best_model = (best_name, self.models[best_name])
        logger.info(f"\nBest model: {best_name} (AUC: {results[best_name].auc_roc:.4f})")

        # Save results
        if output_dir:
            self.save_results(output_dir, feature_names)

        return results

    def save_results(self, output_dir: str, feature_names: List[str]):
        """Save models, metrics, and feature importance."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save models
        models_dir = output_path / 'models'
        models_dir.mkdir(exist_ok=True)

        for name, model in self.models.items():
            model_file = models_dir / f"{name.lower()}_{timestamp}.pkl"
            joblib.dump(model, model_file)
            logger.info(f"Saved {name} model to {model_file}")

        # Save scaler
        if self.scaler is not None:
            scaler_file = models_dir / f"scaler_{timestamp}.pkl"
            joblib.dump(self.scaler, scaler_file)

        # Save feature names
        feature_file = models_dir / f"features_{timestamp}.json"
        with open(feature_file, 'w') as f:
            json.dump(feature_names, f, indent=2)

        # Save feature importance
        for name, importance_df in self.feature_importance.items():
            imp_file = output_path / f"{name.lower()}_feature_importance.csv"
            importance_df.to_csv(imp_file, index=False)

        # Save metrics summary
        metrics_summary = {}
        for name, metrics in self.metrics.items():
            if metrics:
                metrics_summary[name] = {
                    'cv_auc_mean': np.mean([m.auc_roc for m in metrics]),
                    'cv_auc_std': np.std([m.auc_roc for m in metrics]),
                    'cv_accuracy_mean': np.mean([m.accuracy for m in metrics]),
                    'cv_f1_mean': np.mean([m.f1 for m in metrics])
                }

        metrics_file = output_path / f"cv_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)

        logger.info(f"\nResults saved to {output_path}")

    def export_onnx(self, model_name: str, output_path: str,
                    feature_names: List[str]) -> bool:
        """Export model to ONNX format for NinjaTrader integration."""
        if not HAS_ONNX:
            logger.error("ONNX not installed")
            return False

        model = self.models.get(model_name)
        if model is None:
            logger.error(f"Model {model_name} not found")
            return False

        n_features = len(feature_names)

        try:
            if model_name == 'RandomForest':
                # sklearn models can be converted directly
                initial_type = [('float_input', FloatTensorType([None, n_features]))]
                onnx_model = convert_sklearn(model, initial_types=initial_type)

                with open(output_path, 'wb') as f:
                    f.write(onnx_model.SerializeToString())

                logger.info(f"Exported {model_name} to ONNX: {output_path}")
                return True
            else:
                logger.warning(f"ONNX export not supported for {model_name}")
                return False

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False


def train_models(features: pd.DataFrame, target_col: str = 'target_direction_1',
                 selected_features: Optional[List[str]] = None,
                 output_dir: str = 'data/models',
                 config: Optional[ModelConfig] = None,
                 rth_only: bool = False) -> Dict[str, ModelMetrics]:
    """
    Convenience function to train all models.

    IMPORTANT: We only trade during Regular Trading Hours (RTH).
    For ES futures: 9:30 AM - 4:00 PM Eastern Time.

    Parameters:
    -----------
    features : pd.DataFrame
        Feature matrix with targets (must have DatetimeIndex)
    target_col : str
        Target column name
    selected_features : List[str], optional
        Subset of features to use
    output_dir : str
        Directory to save models
    config : ModelConfig, optional
        Training configuration
    rth_only : bool
        If True, filter to Regular Trading Hours only

    Returns:
    --------
    Dict[str, ModelMetrics]
        Test set metrics for each model
    """
    # Filter to RTH if requested
    if rth_only:
        logger.info("Filtering to Regular Trading Hours (RTH) only...")
        logger.info("RTH for ES/NQ: 9:30 AM - 4:00 PM Eastern Time")
        features = filter_rth_only(features)

    trainer = ModelTrainer(config)
    results = trainer.train_all_models(
        features, target_col, selected_features, output_dir
    )
    return results


# Export all public functions
__all__ = [
    'ModelConfig',
    'ModelMetrics',
    'ModelTrainer',
    'WalkForwardValidator',
    'train_models',
    'filter_rth_only',
    'check_data_leakage',
    'calculate_embargo_size',
]


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load data
    print("Loading ES data...")
    es_data = pd.read_csv('data/raw/market/ES_1min_databento.csv',
                          index_col=0, parse_dates=True)

    # Build features
    print("Building features...")
    import sys
    sys.path.insert(0, 'src/python')
    from feature_engineering.feature_pipeline import build_feature_matrix

    features = build_feature_matrix(
        es_data,
        symbol='ES',
        include_lagged=True,
        include_interactions=True,
        include_targets=True,
        include_macro=False,
        dropna=False
    )

    # Drop hurst_20 and clean
    if 'hurst_20' in features.columns:
        features = features.drop(columns=['hurst_20'])
    features = features.dropna()

    # Load selected features
    rankings = pd.read_csv('data/processed/feature_rankings.csv')
    selected = rankings['feature'].tolist()[:75]

    # Train models
    print("\nTraining models...")
    results = train_models(features, 'target_direction_1', selected)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  AUC-ROC: {metrics.auc_roc:.4f}")
        print(f"  Accuracy: {metrics.accuracy:.4f}")
        print(f"  F1: {metrics.f1:.4f}")
