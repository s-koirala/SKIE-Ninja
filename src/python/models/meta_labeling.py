"""
Meta-Labeling for Bet Sizing

Implementation of the Meta-Labeling technique from Marcos Lopez de Prado's
"Advances in Financial Machine Learning" (2018).

Meta-labeling uses a two-stage approach:
1. Primary Model: Determines trade DIRECTION (long/short/no trade)
   - Optimized for HIGH RECALL (catch all opportunities)

2. Secondary Model (Meta-Model): Determines trade SIZE
   - Predicts if primary model is correct (1) or wrong (0)
   - Output probability used for position sizing
   - Improves PRECISION by filtering false positives

Benefits:
- Decouples direction from sizing (reduces overfitting)
- Works with any primary model (ML or rule-based)
- Enables Kelly-optimal bet sizing
- Higher F1 score than single-model approach

References:
- Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
- https://hudsonthames.org/meta-labeling-a-toy-example/
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

logger = logging.getLogger(__name__)


@dataclass
class MetaLabelConfig:
    """Configuration for meta-labeling."""
    # Primary model settings
    primary_threshold: float = 0.5      # Threshold for primary predictions
    recall_threshold: float = 0.3       # Threshold to achieve high recall

    # Secondary model settings
    meta_model_type: str = 'rf'         # 'rf' (Random Forest) or 'lr' (Logistic Regression)
    n_estimators: int = 100             # For Random Forest
    max_depth: int = 5                  # For Random Forest
    class_weight: str = 'balanced'      # Handle class imbalance

    # Bet sizing
    max_position_size: float = 1.0      # Maximum position size
    min_probability: float = 0.55       # Minimum probability to trade
    sizing_method: str = 'linear'       # 'linear', 'kelly', or 'sqrt'


class MetaLabeler:
    """
    Meta-Labeling system for bet sizing.

    This class implements a two-stage prediction system:
    1. Takes predictions from a primary model (direction)
    2. Trains a secondary model to predict if primary is correct
    3. Uses secondary probability for position sizing

    Example:
        >>> # Stage 1: Primary model predictions
        >>> primary_preds = primary_model.predict(X)

        >>> # Stage 2: Meta-labeling
        >>> meta_labeler = MetaLabeler(config)
        >>> meta_labeler.fit(X, primary_preds, tb_labels)
        >>> bet_sizes = meta_labeler.predict_bet_size(X_test, primary_preds_test)
    """

    def __init__(self, config: Optional[MetaLabelConfig] = None):
        """
        Initialize the Meta-Labeler.

        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or MetaLabelConfig()
        self.meta_model = None
        self.is_fitted = False
        self._feature_importance = None

    def _create_meta_model(self):
        """Create the secondary (meta) model."""
        if self.config.meta_model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                class_weight=self.config.class_weight,
                random_state=42,
                n_jobs=-1
            )
        elif self.config.meta_model_type == 'lr':
            return LogisticRegression(
                class_weight=self.config.class_weight,
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown meta model type: {self.config.meta_model_type}")

    def _create_meta_labels(
        self,
        primary_predictions: pd.Series,
        tb_labels: pd.Series
    ) -> pd.Series:
        """
        Create meta-labels indicating if primary model was correct.

        Args:
            primary_predictions: Primary model direction predictions (+1, -1)
            tb_labels: Triple barrier labels from actual outcomes

        Returns:
            Meta-labels: 1 if primary was correct, 0 if wrong
        """
        # Primary was correct if prediction matches outcome
        # Long prediction (+1) correct if tb_label is +1 (hit take profit)
        # Short prediction (-1) correct if tb_label is -1 (hit stop loss target)

        meta_labels = pd.Series(0, index=primary_predictions.index)

        # Correct long predictions
        correct_long = (primary_predictions == 1) & (tb_labels == 1)

        # Correct short predictions
        correct_short = (primary_predictions == -1) & (tb_labels == -1)

        meta_labels[correct_long | correct_short] = 1

        return meta_labels

    def _prepare_meta_features(
        self,
        X: pd.DataFrame,
        primary_predictions: pd.Series,
        primary_probabilities: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Prepare feature matrix for meta-model.

        Combines original features with primary model output.

        Args:
            X: Original feature matrix
            primary_predictions: Primary model predictions
            primary_probabilities: Optional probability scores

        Returns:
            Enhanced feature matrix for meta-model
        """
        meta_X = X.copy()

        # Add primary model prediction
        meta_X['primary_pred'] = primary_predictions.values

        # Add primary probability if available
        if primary_probabilities is not None:
            meta_X['primary_prob'] = primary_probabilities.values
            meta_X['primary_prob_dist_from_half'] = abs(primary_probabilities.values - 0.5)

        return meta_X

    def fit(
        self,
        X: pd.DataFrame,
        primary_predictions: pd.Series,
        tb_labels: pd.Series,
        primary_probabilities: Optional[pd.Series] = None
    ) -> 'MetaLabeler':
        """
        Train the meta-labeling model.

        Args:
            X: Feature matrix
            primary_predictions: Primary model direction predictions
            tb_labels: Triple barrier labels (ground truth)
            primary_probabilities: Optional primary model probabilities

        Returns:
            Self for chaining
        """
        logger.info("Training meta-labeling model...")

        # Create meta-labels
        meta_labels = self._create_meta_labels(primary_predictions, tb_labels)

        # Only train on samples where primary made a prediction
        active_mask = primary_predictions != 0
        if active_mask.sum() == 0:
            logger.warning("No active predictions to train on")
            return self

        # Prepare features
        meta_X = self._prepare_meta_features(
            X.loc[active_mask],
            primary_predictions.loc[active_mask],
            primary_probabilities.loc[active_mask] if primary_probabilities is not None else None
        )
        meta_y = meta_labels.loc[active_mask]

        # Handle NaN
        valid_mask = ~meta_X.isna().any(axis=1) & ~meta_y.isna()
        meta_X = meta_X.loc[valid_mask]
        meta_y = meta_y.loc[valid_mask]

        if len(meta_X) < 100:
            logger.warning(f"Only {len(meta_X)} samples for meta-model training")

        # Train meta-model
        self.meta_model = self._create_meta_model()
        self.meta_model.fit(meta_X, meta_y)

        # Store feature importance
        if hasattr(self.meta_model, 'feature_importances_'):
            self._feature_importance = pd.Series(
                self.meta_model.feature_importances_,
                index=meta_X.columns
            ).sort_values(ascending=False)

        self.is_fitted = True

        # Log training results
        train_pred = self.meta_model.predict(meta_X)
        accuracy = accuracy_score(meta_y, train_pred)
        precision = precision_score(meta_y, train_pred, zero_division=0)
        recall = recall_score(meta_y, train_pred, zero_division=0)
        f1 = f1_score(meta_y, train_pred, zero_division=0)

        logger.info(f"Meta-model training complete:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")

        return self

    def predict_meta_probability(
        self,
        X: pd.DataFrame,
        primary_predictions: pd.Series,
        primary_probabilities: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Predict probability that primary model is correct.

        Args:
            X: Feature matrix
            primary_predictions: Primary model predictions
            primary_probabilities: Optional primary probabilities

        Returns:
            Probability that primary prediction is correct
        """
        if not self.is_fitted:
            raise ValueError("Meta-labeler not fitted. Call fit() first.")

        # Prepare features
        meta_X = self._prepare_meta_features(
            X, primary_predictions, primary_probabilities
        )

        # Handle NaN
        meta_X = meta_X.fillna(0)

        # Predict probabilities
        probs = self.meta_model.predict_proba(meta_X)[:, 1]

        return pd.Series(probs, index=X.index)

    def predict_bet_size(
        self,
        X: pd.DataFrame,
        primary_predictions: pd.Series,
        primary_probabilities: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Predict optimal bet size based on meta-probability.

        Args:
            X: Feature matrix
            primary_predictions: Primary model predictions
            primary_probabilities: Optional primary probabilities

        Returns:
            Series of bet sizes (0 to max_position_size)
        """
        meta_probs = self.predict_meta_probability(
            X, primary_predictions, primary_probabilities
        )

        # Convert probability to bet size
        bet_sizes = self._probability_to_size(meta_probs)

        # Zero out bets below minimum probability
        bet_sizes[meta_probs < self.config.min_probability] = 0

        # Zero out bets where primary predicted no trade
        bet_sizes[primary_predictions == 0] = 0

        return bet_sizes

    def _probability_to_size(self, probabilities: pd.Series) -> pd.Series:
        """
        Convert meta-probability to position size.

        Args:
            probabilities: Meta-model probabilities

        Returns:
            Position sizes
        """
        if self.config.sizing_method == 'linear':
            # Linear scaling: size = (prob - 0.5) * 2 * max_size
            sizes = (probabilities - 0.5) * 2 * self.config.max_position_size
            sizes = sizes.clip(0, self.config.max_position_size)

        elif self.config.sizing_method == 'kelly':
            # Kelly criterion: size = (p - (1-p)) / 1 = 2p - 1
            # Assumes 1:1 risk/reward for simplicity
            sizes = (2 * probabilities - 1) * self.config.max_position_size
            sizes = sizes.clip(0, self.config.max_position_size)

        elif self.config.sizing_method == 'sqrt':
            # Square root scaling for more conservative sizing
            sizes = np.sqrt((probabilities - 0.5) * 2) * self.config.max_position_size
            sizes = sizes.clip(0, self.config.max_position_size)

        else:
            raise ValueError(f"Unknown sizing method: {self.config.sizing_method}")

        return sizes

    def evaluate(
        self,
        X: pd.DataFrame,
        primary_predictions: pd.Series,
        tb_labels: pd.Series,
        primary_probabilities: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Evaluate meta-labeling performance.

        Args:
            X: Feature matrix
            primary_predictions: Primary model predictions
            tb_labels: Triple barrier labels (ground truth)
            primary_probabilities: Optional primary probabilities

        Returns:
            Dictionary of evaluation metrics
        """
        # Get meta-predictions
        meta_probs = self.predict_meta_probability(
            X, primary_predictions, primary_probabilities
        )
        meta_preds = (meta_probs >= 0.5).astype(int)

        # True meta-labels
        meta_labels = self._create_meta_labels(primary_predictions, tb_labels)

        # Filter to active predictions
        active_mask = primary_predictions != 0
        meta_preds = meta_preds.loc[active_mask]
        meta_labels = meta_labels.loc[active_mask]
        meta_probs = meta_probs.loc[active_mask]

        # Calculate metrics
        metrics = {
            'meta_accuracy': accuracy_score(meta_labels, meta_preds),
            'meta_precision': precision_score(meta_labels, meta_preds, zero_division=0),
            'meta_recall': recall_score(meta_labels, meta_preds, zero_division=0),
            'meta_f1': f1_score(meta_labels, meta_preds, zero_division=0),
        }

        try:
            metrics['meta_auc'] = roc_auc_score(meta_labels, meta_probs)
        except:
            metrics['meta_auc'] = 0.5

        # Calculate improvement in precision
        # Without meta: precision = correct / total_predictions
        primary_correct = (
            ((primary_predictions == 1) & (tb_labels == 1)) |
            ((primary_predictions == -1) & (tb_labels == -1))
        ).sum()
        primary_total = (primary_predictions != 0).sum()
        primary_precision = primary_correct / primary_total if primary_total > 0 else 0

        # With meta: only trade when meta says trade
        meta_trade = meta_preds == 1
        meta_correct = (
            ((primary_predictions == 1) & (tb_labels == 1) & meta_trade) |
            ((primary_predictions == -1) & (tb_labels == -1) & meta_trade)
        ).sum()
        meta_total = meta_trade.sum()
        meta_precision = meta_correct / meta_total if meta_total > 0 else 0

        metrics['primary_precision'] = primary_precision
        metrics['meta_filtered_precision'] = meta_precision
        metrics['precision_improvement'] = meta_precision - primary_precision
        metrics['trades_filtered_pct'] = 1 - (meta_total / primary_total) if primary_total > 0 else 0

        return metrics

    @property
    def feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance from meta-model."""
        return self._feature_importance


class MetaLabelingPipeline:
    """
    End-to-end meta-labeling pipeline.

    Combines primary model training with meta-labeling for complete
    trading signal generation with bet sizing.
    """

    def __init__(
        self,
        primary_model: Any,
        meta_config: Optional[MetaLabelConfig] = None
    ):
        """
        Initialize the pipeline.

        Args:
            primary_model: Sklearn-compatible classifier for direction
            meta_config: Configuration for meta-labeling
        """
        self.primary_model = primary_model
        self.meta_labeler = MetaLabeler(meta_config)
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        tb_labels: pd.Series,
        cv_splits: int = 5
    ) -> 'MetaLabelingPipeline':
        """
        Train the complete pipeline.

        Uses time-series cross-validation to generate out-of-sample
        predictions for meta-label training.

        Args:
            X: Feature matrix
            tb_labels: Triple barrier labels
            cv_splits: Number of CV splits

        Returns:
            Self for chaining
        """
        logger.info("Training meta-labeling pipeline...")

        # Create binary target for primary model (direction)
        # 1 = long (tb_label == 1), 0 = short or no trade
        primary_target = (tb_labels == 1).astype(int)

        # Generate OOS predictions using time-series CV
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        oos_predictions = pd.Series(index=X.index, dtype=float)
        oos_probabilities = pd.Series(index=X.index, dtype=float)

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = primary_target.iloc[train_idx]

            # Train primary model
            self.primary_model.fit(X_train, y_train)

            # Predict on test fold
            preds = self.primary_model.predict(X_test)
            probs = self.primary_model.predict_proba(X_test)[:, 1]

            # Convert to direction: prob > 0.5 = long (+1), else short (-1)
            directions = np.where(probs > 0.5, 1, -1)

            oos_predictions.iloc[test_idx] = directions
            oos_probabilities.iloc[test_idx] = probs

        # Train final primary model on all data
        self.primary_model.fit(X, primary_target)

        # Train meta-labeler on OOS predictions
        valid_mask = ~oos_predictions.isna()
        self.meta_labeler.fit(
            X.loc[valid_mask],
            oos_predictions.loc[valid_mask].astype(int),
            tb_labels.loc[valid_mask],
            oos_probabilities.loc[valid_mask]
        )

        self.is_fitted = True
        logger.info("Meta-labeling pipeline training complete")

        return self

    def predict(
        self,
        X: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate trading signals with bet sizes.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (directions, bet_sizes, meta_probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        # Primary predictions
        primary_probs = self.primary_model.predict_proba(X)[:, 1]
        directions = pd.Series(
            np.where(primary_probs > 0.5, 1, -1),
            index=X.index
        )
        primary_probs = pd.Series(primary_probs, index=X.index)

        # Meta-labeling for bet sizes
        bet_sizes = self.meta_labeler.predict_bet_size(
            X, directions, primary_probs
        )
        meta_probs = self.meta_labeler.predict_meta_probability(
            X, directions, primary_probs
        )

        return directions, bet_sizes, meta_probs


if __name__ == "__main__":
    print("=" * 70)
    print("META-LABELING - TEST")
    print("=" * 70)

    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    # Simulate features
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
        'feature_4': np.random.randn(n_samples),
        'feature_5': np.random.randn(n_samples),
    })

    # Simulate primary model predictions (noisy)
    true_signal = (X['feature_1'] + X['feature_2'] > 0).astype(int)
    noise = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    primary_predictions = pd.Series(
        np.where(np.random.rand(n_samples) > 0.5, 1, -1)
    )

    # Simulate triple barrier labels (some correlation with primary)
    tb_labels = pd.Series(
        np.where(
            (primary_predictions == 1) & (np.random.rand(n_samples) > 0.4),
            1,
            np.where(
                (primary_predictions == -1) & (np.random.rand(n_samples) > 0.4),
                -1,
                np.random.choice([-1, 0, 1], size=n_samples)
            )
        )
    )

    print("\n[1] Testing MetaLabeler...")
    config = MetaLabelConfig(
        meta_model_type='rf',
        n_estimators=50,
        max_depth=4
    )

    meta_labeler = MetaLabeler(config)

    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    preds_train = primary_predictions.iloc[:train_size]
    preds_test = primary_predictions.iloc[train_size:]
    tb_train = tb_labels.iloc[:train_size]
    tb_test = tb_labels.iloc[train_size:]

    # Fit
    meta_labeler.fit(X_train, preds_train, tb_train)

    # Evaluate
    metrics = meta_labeler.evaluate(X_test, preds_test, tb_test)
    print("\nMeta-labeling metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Get bet sizes
    bet_sizes = meta_labeler.predict_bet_size(X_test, preds_test)
    print(f"\nBet size statistics:")
    print(f"  Mean: {bet_sizes.mean():.4f}")
    print(f"  Std: {bet_sizes.std():.4f}")
    print(f"  Zero bets: {(bet_sizes == 0).sum()}")
    print(f"  Non-zero bets: {(bet_sizes > 0).sum()}")

    print("\n[2] Testing MetaLabelingPipeline...")
    from sklearn.ensemble import GradientBoostingClassifier

    pipeline = MetaLabelingPipeline(
        primary_model=GradientBoostingClassifier(n_estimators=50, max_depth=3),
        meta_config=config
    )

    # Fit pipeline
    pipeline.fit(X_train, tb_train, cv_splits=3)

    # Predict
    directions, sizes, probs = pipeline.predict(X_test)
    print(f"\nPipeline predictions:")
    print(f"  Long signals: {(directions == 1).sum()}")
    print(f"  Short signals: {(directions == -1).sum()}")
    print(f"  Non-zero bet sizes: {(sizes > 0).sum()}")

    print("\n" + "=" * 70)
    print("META-LABELING TEST COMPLETE")
    print("=" * 70)
