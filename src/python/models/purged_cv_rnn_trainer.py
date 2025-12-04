"""
SKIE-Ninja Purged K-Fold CV for RNN Models

Addresses overfitting in LSTM/GRU through time-series aware cross-validation.

Literature-Based Approach:
=========================
1. **Purged K-Fold CV** (de Prado, 2018 - "Advances in Financial Machine Learning")
   - Removes observations from training that could leak information into test
   - Purge period = max feature lookback (e.g., 200 bars for 200-bar features)

2. **Embargo Period** (de Prado, 2018)
   - Gap after test set to prevent target leakage
   - Embargo = target horizon (e.g., if predicting 1-bar ahead, embargo = 1)

3. **Walk-Forward Validation** (Tashman, 2000)
   - Train on expanding/rolling window
   - Test on next unseen period
   - Mimics real trading deployment

Key Changes from Standard CV:
- No random shuffling (preserves temporal order)
- Purge period between train/test prevents feature leakage
- Embargo period prevents target leakage
- Multiple folds provide variance estimate

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss, brier_score_loss
)
from typing import Dict, List, Optional, Tuple, Any, Generator
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class PurgedCVConfig:
    """Configuration for Purged K-Fold Cross-Validation."""
    # CV Parameters
    n_splits: int = 5
    purge_bars: int = 200  # Bars to remove between train/test (max feature lookback)
    embargo_bars: int = 42  # Bars after test to embargo (~3.5 hours for 5-min)

    # Model Architecture (reduced for regularization)
    hidden_size: int = 64  # Smaller than typical - prevents overfitting
    num_layers: int = 1    # Single layer often sufficient
    dropout: float = 0.5   # Higher dropout for financial data

    # Sequence Parameters
    sequence_length: int = 20
    batch_size: int = 128  # Smaller batches for regularization

    # Training Parameters
    learning_rate: float = 0.0005  # Lower LR for stability
    weight_decay: float = 1e-4     # Stronger L2 regularization
    gradient_clip: float = 0.5     # Tighter gradient clipping
    epochs: int = 30
    early_stopping_patience: int = 5  # Quick early stopping

    # Regularization Flags
    use_batch_norm: bool = True
    bidirectional: bool = False

    # Data Leakage Checks
    check_leakage: bool = True
    leakage_threshold: float = 0.95  # Correlation threshold for leakage detection


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation for Time Series.

    Implements the approach from de Prado (2018):
    - Purge: Remove training samples within purge_bars of test set
    - Embargo: Remove test samples within embargo_bars of training set
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_bars: int = 200,
        embargo_bars: int = 42
    ):
        self.n_splits = n_splits
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices with purging and embargo.

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        test_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            # Define test boundaries
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n_samples)

            # Test indices
            test_idx = np.arange(test_start, test_end)

            # Train indices: everything NOT in test, purge, or embargo zones
            train_idx = []

            for j in range(n_samples):
                # Check if in purge zone (before test)
                in_purge = (j >= test_start - self.purge_bars) and (j < test_start)

                # Check if in embargo zone (after test)
                in_embargo = (j >= test_end) and (j < test_end + self.embargo_bars)

                # Check if in test set
                in_test = (j >= test_start) and (j < test_end)

                if not (in_purge or in_embargo or in_test):
                    train_idx.append(j)

            train_idx = np.array(train_idx)

            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_splits


class WalkForwardCV:
    """
    Walk-Forward Cross-Validation for Time Series.

    More realistic for trading - trains on historical data,
    tests on next period, then rolls forward.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_bars: int = 14040,  # 180 days of 5-min RTH bars
        test_bars: int = 390,     # 5 days
        embargo_bars: int = 42
    ):
        self.n_splits = n_splits
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.embargo_bars = embargo_bars

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate walk-forward splits."""
        n_samples = len(X)

        for i in range(self.n_splits):
            # Calculate boundaries
            train_start = 0  # Expanding window
            train_end = self.train_bars + (i * self.test_bars)

            test_start = train_end + self.embargo_bars
            test_end = min(test_start + self.test_bars, n_samples)

            if test_end > n_samples:
                break

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx


class SequenceDataset(Dataset):
    """PyTorch Dataset for creating sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.seq_len]
        y_target = self.y[idx + self.seq_len]
        return x_seq, y_target


class RegularizedLSTM(nn.Module):
    """LSTM with aggressive regularization for financial data."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.5,
        use_batch_norm: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm

        # Layer Normalization on input
        self.input_norm = nn.LayerNorm(input_size)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Batch Normalization after LSTM
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_size)

        # FC layers with dropout
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Normalize input
        x = self.input_norm(x)

        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]

        # Batch norm
        if self.use_batch_norm:
            h_last = self.batch_norm(h_last)

        # Output
        return self.fc(h_last).squeeze()


class RegularizedGRU(nn.Module):
    """GRU with aggressive regularization for financial data."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.5,
        use_batch_norm: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm

        self.input_norm = nn.LayerNorm(input_size)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_size)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_norm(x)
        gru_out, h_n = self.gru(x)
        h_last = h_n[-1]

        if self.use_batch_norm:
            h_last = self.batch_norm(h_last)

        return self.fc(h_last).squeeze()


@dataclass
class CVFoldResult:
    """Results from a single CV fold."""
    fold: int
    train_samples: int
    test_samples: int
    accuracy: float
    auc_roc: float
    f1: float
    precision: float
    recall: float
    log_loss: float
    brier_score: float
    epochs_trained: int
    best_val_loss: float


@dataclass
class CVResults:
    """Aggregated cross-validation results."""
    model_type: str
    n_folds: int
    fold_results: List[CVFoldResult]

    # Aggregated metrics
    mean_auc: float = 0.0
    std_auc: float = 0.0
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    mean_f1: float = 0.0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_log_loss: float = 0.0
    mean_brier: float = 0.0

    # Training info
    total_train_samples: int = 0
    total_test_samples: int = 0
    avg_epochs: float = 0.0

    def calculate_aggregates(self):
        """Calculate aggregate statistics from fold results."""
        aucs = [f.auc_roc for f in self.fold_results]
        accs = [f.accuracy for f in self.fold_results]

        self.mean_auc = np.mean(aucs)
        self.std_auc = np.std(aucs)
        self.mean_accuracy = np.mean(accs)
        self.std_accuracy = np.std(accs)
        self.mean_f1 = np.mean([f.f1 for f in self.fold_results])
        self.mean_precision = np.mean([f.precision for f in self.fold_results])
        self.mean_recall = np.mean([f.recall for f in self.fold_results])
        self.mean_log_loss = np.mean([f.log_loss for f in self.fold_results])
        self.mean_brier = np.mean([f.brier_score for f in self.fold_results])
        self.total_train_samples = sum(f.train_samples for f in self.fold_results)
        self.total_test_samples = sum(f.test_samples for f in self.fold_results)
        self.avg_epochs = np.mean([f.epochs_trained for f in self.fold_results])

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_type': self.model_type,
            'n_folds': self.n_folds,
            'mean_auc': round(self.mean_auc, 4),
            'std_auc': round(self.std_auc, 4),
            'mean_accuracy': round(self.mean_accuracy, 4),
            'std_accuracy': round(self.std_accuracy, 4),
            'mean_f1': round(self.mean_f1, 4),
            'mean_precision': round(self.mean_precision, 4),
            'mean_recall': round(self.mean_recall, 4),
            'mean_log_loss': round(self.mean_log_loss, 4),
            'mean_brier': round(self.mean_brier, 4),
            'total_train_samples': self.total_train_samples,
            'total_test_samples': self.total_test_samples,
            'avg_epochs': round(self.avg_epochs, 1),
            'fold_details': [
                {
                    'fold': f.fold,
                    'auc_roc': round(f.auc_roc, 4),
                    'accuracy': round(f.accuracy, 4),
                    'f1': round(f.f1, 4)
                }
                for f in self.fold_results
            ]
        }


class PurgedCVRNNTrainer:
    """
    RNN Trainer with Purged Cross-Validation.

    Key features to prevent overfitting:
    1. Purged K-Fold CV - removes samples that could leak information
    2. Aggressive regularization - high dropout, weight decay, small models
    3. Early stopping - quick termination when validation loss increases
    4. Gradient clipping - prevents exploding gradients
    5. Data leakage checks - validates no high correlations with target
    """

    def __init__(self, config: Optional[PurgedCVConfig] = None):
        self.config = config or PurgedCVConfig()
        self.scaler = StandardScaler()
        self.device = DEVICE
        self.feature_names: List[str] = []

    def check_data_leakage(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[str]:
        """
        Check for data leakage by detecting high correlations with target.

        Returns list of suspicious features.
        """
        if not self.config.check_leakage:
            return []

        suspicious = []

        for i, name in enumerate(feature_names):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            if abs(corr) > self.config.leakage_threshold:
                suspicious.append(f"{name} (corr={corr:.3f})")
                logger.warning(f"LEAKAGE DETECTED: {name} has {corr:.3f} correlation with target")

        return suspicious

    def _train_fold(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fold: int
    ) -> Tuple[nn.Module, float, int]:
        """Train model for one fold with early stopping."""
        model = model.to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )

        criterion = nn.BCELoss()

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        epochs_trained = 0

        for epoch in range(self.config.epochs):
            # Training
            model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.gradient_clip
                )

                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            model.eval()
            val_losses = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)
            scheduler.step(avg_val_loss)
            epochs_trained = epoch + 1

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Fold {fold}: Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model, best_val_loss, epochs_trained

    def _evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluate model on test set."""
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = model(batch_X)
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        pred_binary = (all_preds >= 0.5).astype(int)

        metrics = {
            'accuracy': accuracy_score(all_targets, pred_binary),
            'auc_roc': roc_auc_score(all_targets, all_preds),
            'f1': f1_score(all_targets, pred_binary),
            'precision': precision_score(all_targets, pred_binary, zero_division=0),
            'recall': recall_score(all_targets, pred_binary, zero_division=0),
            'log_loss': log_loss(all_targets, all_preds),
            'brier_score': brier_score_loss(all_targets, all_preds)
        }

        return metrics, all_preds, all_targets

    def train_with_purged_cv(
        self,
        features: pd.DataFrame,
        target_col: str,
        selected_features: Optional[List[str]] = None,
        model_type: str = 'lstm',
        cv_type: str = 'purged'  # 'purged' or 'walk_forward'
    ) -> CVResults:
        """
        Train RNN model with Purged K-Fold Cross-Validation.

        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix with target column
        target_col : str
            Name of target column
        selected_features : List[str], optional
            Features to use (default: all non-target columns)
        model_type : str
            'lstm' or 'gru'
        cv_type : str
            'purged' for Purged K-Fold, 'walk_forward' for Walk-Forward CV

        Returns:
        --------
        CVResults with fold-by-fold and aggregate metrics
        """
        logger.info("="*60)
        logger.info(f"PURGED CV TRAINING: {model_type.upper()}")
        logger.info("="*60)

        # Prepare data
        if selected_features is None:
            self.feature_names = [c for c in features.columns if not c.startswith('target_')]
        else:
            self.feature_names = [f for f in selected_features if f in features.columns]

        X = features[self.feature_names].values
        y = features[target_col].values

        logger.info(f"Data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Features: {len(self.feature_names)}")

        # Check for data leakage
        suspicious = self.check_data_leakage(X, y, self.feature_names)
        if suspicious:
            logger.warning(f"Found {len(suspicious)} potentially leaking features!")
            for s in suspicious[:5]:  # Show first 5
                logger.warning(f"  - {s}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create CV splitter
        if cv_type == 'purged':
            cv = PurgedKFold(
                n_splits=self.config.n_splits,
                purge_bars=self.config.purge_bars,
                embargo_bars=self.config.embargo_bars
            )
        else:
            cv = WalkForwardCV(
                n_splits=self.config.n_splits,
                embargo_bars=self.config.embargo_bars
            )

        logger.info(f"CV Type: {cv_type}")
        logger.info(f"N Splits: {self.config.n_splits}")
        logger.info(f"Purge Bars: {self.config.purge_bars}")
        logger.info(f"Embargo Bars: {self.config.embargo_bars}")

        # Run cross-validation
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled)):
            logger.info(f"\n--- Fold {fold + 1}/{self.config.n_splits} ---")
            logger.info(f"Train: {len(train_idx):,} samples, Test: {len(test_idx):,} samples")

            # Get fold data
            X_train, y_train = X_scaled[train_idx], y[train_idx]
            X_test, y_test = X_scaled[test_idx], y[test_idx]

            # Create datasets
            train_dataset = SequenceDataset(X_train, y_train, self.config.sequence_length)
            test_dataset = SequenceDataset(X_test, y_test, self.config.sequence_length)

            if len(train_dataset) < self.config.batch_size:
                logger.warning(f"Fold {fold + 1}: Not enough training samples, skipping")
                continue

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )

            # Create model
            input_size = X_train.shape[1]
            if model_type == 'lstm':
                model = RegularizedLSTM(
                    input_size=input_size,
                    hidden_size=self.config.hidden_size,
                    num_layers=self.config.num_layers,
                    dropout=self.config.dropout,
                    use_batch_norm=self.config.use_batch_norm
                )
            else:
                model = RegularizedGRU(
                    input_size=input_size,
                    hidden_size=self.config.hidden_size,
                    num_layers=self.config.num_layers,
                    dropout=self.config.dropout,
                    use_batch_norm=self.config.use_batch_norm
                )

            # Train
            model, best_val_loss, epochs_trained = self._train_fold(
                model, train_loader, test_loader, fold + 1
            )

            # Evaluate
            metrics, preds, targets = self._evaluate(model, test_loader)

            fold_result = CVFoldResult(
                fold=fold + 1,
                train_samples=len(train_idx),
                test_samples=len(test_idx),
                accuracy=metrics['accuracy'],
                auc_roc=metrics['auc_roc'],
                f1=metrics['f1'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                log_loss=metrics['log_loss'],
                brier_score=metrics['brier_score'],
                epochs_trained=epochs_trained,
                best_val_loss=best_val_loss
            )
            fold_results.append(fold_result)

            logger.info(f"Fold {fold + 1} Results:")
            logger.info(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1']:.4f}")
            logger.info(f"  Epochs:   {epochs_trained}")

        # Aggregate results
        cv_results = CVResults(
            model_type=model_type,
            n_folds=len(fold_results),
            fold_results=fold_results
        )
        cv_results.calculate_aggregates()

        logger.info("\n" + "="*60)
        logger.info("CROSS-VALIDATION RESULTS")
        logger.info("="*60)
        logger.info(f"Mean AUC-ROC:  {cv_results.mean_auc:.4f} (+/- {cv_results.std_auc:.4f})")
        logger.info(f"Mean Accuracy: {cv_results.mean_accuracy:.4f} (+/- {cv_results.std_accuracy:.4f})")
        logger.info(f"Mean F1:       {cv_results.mean_f1:.4f}")
        logger.info(f"Mean Log Loss: {cv_results.mean_log_loss:.4f}")
        logger.info(f"Avg Epochs:    {cv_results.avg_epochs:.1f}")

        return cv_results

    def save_results(
        self,
        cv_results: CVResults,
        output_dir: str = 'data/models'
    ):
        """Save CV results to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save results
        with open(output_path / f'purged_cv_results_{cv_results.model_type}_{timestamp}.json', 'w') as f:
            json.dump(cv_results.to_dict(), f, indent=2)

        logger.info(f"Results saved to {output_path}")


def train_rnn_with_purged_cv(
    features: pd.DataFrame,
    target_col: str = 'target_direction_1',
    selected_features: Optional[List[str]] = None,
    model_type: str = 'lstm',
    output_dir: str = 'data/models',
    config: Optional[PurgedCVConfig] = None
) -> CVResults:
    """
    Convenience function to train RNN with Purged K-Fold CV.

    This addresses RNN overfitting by:
    1. Using time-series aware CV (no random splitting)
    2. Purging samples that could leak information
    3. Adding embargo periods after test sets
    4. Aggressive regularization (high dropout, weight decay)
    5. Early stopping
    """
    trainer = PurgedCVRNNTrainer(config)
    results = trainer.train_with_purged_cv(
        features, target_col, selected_features, model_type
    )
    trainer.save_results(results, output_dir)
    return results


# Exports
__all__ = [
    'PurgedCVConfig',
    'PurgedKFold',
    'WalkForwardCV',
    'RegularizedLSTM',
    'RegularizedGRU',
    'CVResults',
    'PurgedCVRNNTrainer',
    'train_rnn_with_purged_cv'
]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("="*60)
    print("PURGED K-FOLD CV FOR RNN MODELS")
    print("="*60)
    print("\nUsage:")
    print("  from purged_cv_rnn_trainer import train_rnn_with_purged_cv")
    print("  results = train_rnn_with_purged_cv(features, 'target_direction_1', model_type='lstm')")
