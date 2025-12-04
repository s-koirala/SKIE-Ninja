"""
SKIE-Ninja Deep Learning Models

LSTM and GRU models for time series classification of ES futures.
Designed for 5-minute RTH bars with walk-forward validation.

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import json
from datetime import datetime
import joblib

logger = logging.getLogger(__name__)

# Check for GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")


@dataclass
class DeepLearningConfig:
    """Configuration for deep learning models."""
    # Architecture
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False

    # Training
    sequence_length: int = 20  # Number of bars to look back
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs: int = 50
    early_stopping_patience: int = 10

    # Data
    train_ratio: float = 0.8
    val_ratio: float = 0.1  # Remaining 0.1 for test

    # Regularization
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int):
        """
        Create sequences from feature matrix.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target array (n_samples,)
            sequence_length: Number of timesteps per sequence
        """
        self.sequence_length = sequence_length
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        # Get sequence of features
        x_seq = self.X[idx:idx + self.sequence_length]
        # Target is the label at the end of the sequence
        y_target = self.y[idx + self.sequence_length]
        return x_seq, y_target


class LSTMClassifier(nn.Module):
    """LSTM-based binary classifier for time series."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Fully connected layers
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_last = h_n[-1]

        # Classification
        out = self.fc(h_last)
        return out.squeeze()


class GRUClassifier(nn.Module):
    """GRU-based binary classifier for time series."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Fully connected layers
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # GRU forward pass
        gru_out, h_n = self.gru(x)

        # Use last hidden state
        if self.bidirectional:
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_last = h_n[-1]

        # Classification
        out = self.fc(h_last)
        return out.squeeze()


@dataclass
class DLMetrics:
    """Metrics for deep learning models."""
    accuracy: float
    auc_roc: float
    f1: float
    precision: float
    recall: float
    train_loss: float
    val_loss: float
    epochs_trained: int


class DeepLearningTrainer:
    """Trainer for LSTM/GRU models with walk-forward validation."""

    def __init__(self, config: Optional[DeepLearningConfig] = None):
        self.config = config or DeepLearningConfig()
        self.scaler = StandardScaler()
        self.device = DEVICE

    def prepare_data(
        self,
        features: pd.DataFrame,
        target_col: str,
        selected_features: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and targets for training."""
        # Get feature columns
        if selected_features:
            feature_cols = [f for f in selected_features if f in features.columns]
        else:
            feature_cols = [c for c in features.columns if not c.startswith('target_')]

        X = features[feature_cols].values
        y = features[target_col].values

        logger.info(f"Data shape: X={X.shape}, y={y.shape}")

        return X, y, feature_cols

    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str = "Model"
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """Train a PyTorch model with early stopping."""
        model = model.to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Training phase
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
                    model.parameters(), self.config.gradient_clip
                )

                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Validation phase
            model.eval()
            val_losses = []
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_losses.append(loss.item())
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())

            avg_val_loss = np.mean(val_losses)
            val_auc = roc_auc_score(val_targets, val_preds)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_auc'].append(val_auc)

            scheduler.step(avg_val_loss)

            # Logging
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"{model_name} Epoch {epoch+1}/{self.config.epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                    f"Val AUC: {val_auc:.4f}"
                )

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model, history

    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Tuple[DLMetrics, np.ndarray]:
        """Evaluate model on test set."""
        model.eval()
        all_preds = []
        all_targets = []
        test_losses = []
        criterion = nn.BCELoss()

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                test_losses.append(loss.item())
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        pred_labels = (all_preds >= 0.5).astype(int)

        metrics = DLMetrics(
            accuracy=accuracy_score(all_targets, pred_labels),
            auc_roc=roc_auc_score(all_targets, all_preds),
            f1=f1_score(all_targets, pred_labels),
            precision=precision_score(all_targets, pred_labels),
            recall=recall_score(all_targets, pred_labels),
            train_loss=0.0,  # Will be set by caller
            val_loss=np.mean(test_losses),
            epochs_trained=0  # Will be set by caller
        )

        return metrics, all_preds

    def train_all_models(
        self,
        features: pd.DataFrame,
        target_col: str = 'target_direction_1',
        selected_features: Optional[List[str]] = None,
        output_dir: str = 'data/models'
    ) -> Dict[str, DLMetrics]:
        """Train LSTM and GRU models."""
        logger.info("="*60)
        logger.info("DEEP LEARNING MODEL TRAINING")
        logger.info("="*60)

        # Prepare data
        X, y, feature_names = self.prepare_data(features, target_col, selected_features)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Temporal split
        n_samples = len(X_scaled)
        train_end = int(n_samples * self.config.train_ratio)
        val_end = int(n_samples * (self.config.train_ratio + self.config.val_ratio))

        X_train = X_scaled[:train_end]
        y_train = y[:train_end]
        X_val = X_scaled[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X_scaled[val_end:]
        y_test = y[val_end:]

        logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train, self.config.sequence_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, self.config.sequence_length)
        test_dataset = TimeSeriesDataset(X_test, y_test, self.config.sequence_length)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False
        )

        logger.info(f"Sequence length: {self.config.sequence_length}")
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        results = {}
        input_size = X_train.shape[1]

        # Train LSTM
        logger.info("\n--- Training LSTM ---")
        lstm_model = LSTMClassifier(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            bidirectional=self.config.bidirectional
        )

        lstm_model, lstm_history = self.train_model(
            lstm_model, train_loader, val_loader, "LSTM"
        )

        lstm_metrics, lstm_preds = self.evaluate_model(lstm_model, test_loader)
        lstm_metrics.train_loss = lstm_history['train_loss'][-1]
        lstm_metrics.epochs_trained = len(lstm_history['train_loss'])
        results['LSTM'] = lstm_metrics

        logger.info(f"LSTM Test AUC: {lstm_metrics.auc_roc:.4f}, Accuracy: {lstm_metrics.accuracy:.4f}")

        # Train GRU
        logger.info("\n--- Training GRU ---")
        gru_model = GRUClassifier(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            bidirectional=self.config.bidirectional
        )

        gru_model, gru_history = self.train_model(
            gru_model, train_loader, val_loader, "GRU"
        )

        gru_metrics, gru_preds = self.evaluate_model(gru_model, test_loader)
        gru_metrics.train_loss = gru_history['train_loss'][-1]
        gru_metrics.epochs_trained = len(gru_history['train_loss'])
        results['GRU'] = gru_metrics

        logger.info(f"GRU Test AUC: {gru_metrics.auc_roc:.4f}, Accuracy: {gru_metrics.accuracy:.4f}")

        # Save models
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        torch.save(lstm_model.state_dict(), output_path / f'lstm_{timestamp}.pt')
        torch.save(gru_model.state_dict(), output_path / f'gru_{timestamp}.pt')
        joblib.dump(self.scaler, output_path / f'dl_scaler_{timestamp}.pkl')

        # Save config and metrics
        config_dict = {
            'config': self.config.__dict__,
            'feature_names': feature_names,
            'input_size': input_size,
            'device': str(self.device),
            'results': {
                name: {
                    'accuracy': m.accuracy,
                    'auc_roc': m.auc_roc,
                    'f1': m.f1,
                    'precision': m.precision,
                    'recall': m.recall,
                    'epochs_trained': m.epochs_trained
                }
                for name, m in results.items()
            }
        }

        with open(output_path / f'dl_config_{timestamp}.json', 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"\nModels saved to {output_path}")

        return results


def train_deep_learning_models(
    features: pd.DataFrame,
    target_col: str = 'target_direction_1',
    selected_features: Optional[List[str]] = None,
    output_dir: str = 'data/models',
    config: Optional[DeepLearningConfig] = None
) -> Dict[str, DLMetrics]:
    """
    Convenience function to train LSTM and GRU models.

    Parameters:
    -----------
    features : pd.DataFrame
        Feature matrix with targets
    target_col : str
        Target column name
    selected_features : List[str], optional
        Subset of features to use
    output_dir : str
        Directory to save models
    config : DeepLearningConfig, optional
        Training configuration

    Returns:
    --------
    Dict[str, DLMetrics]
        Test metrics for each model
    """
    trainer = DeepLearningTrainer(config)
    return trainer.train_all_models(features, target_col, selected_features, output_dir)


# Exports
__all__ = [
    'DeepLearningConfig',
    'DLMetrics',
    'LSTMClassifier',
    'GRUClassifier',
    'TimeSeriesDataset',
    'DeepLearningTrainer',
    'train_deep_learning_models',
]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("="*60)
    print("SKIE-Ninja Deep Learning Training")
    print("="*60)
    print(f"\nDevice: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")

    # Load 5-min resampled data
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from utils.data_resampler import DataResampler
    from feature_engineering.feature_pipeline import build_feature_matrix

    print("\nLoading ES 1-min data...")
    es_data = pd.read_csv('data/raw/market/ES_1min_databento.csv',
                          index_col=0, parse_dates=True)
    print(f"Loaded {len(es_data):,} 1-min bars")

    # Resample to 5-min RTH
    print("\nResampling to 5-min RTH...")
    resampler = DataResampler()
    es_5min = resampler.resample(es_data, '5min', rth_only=True)
    print(f"Resampled to {len(es_5min):,} 5-min bars")

    # Build features
    print("\nBuilding features...")
    features = build_feature_matrix(
        es_5min,
        symbol='ES',
        include_lagged=True,
        include_interactions=True,
        include_targets=True,
        include_macro=False,
        include_sentiment=False,
        include_intermarket=False,
        include_alternative=False,
        dropna=True
    )
    print(f"Feature matrix: {features.shape}")

    # Load selected features
    rankings = pd.read_csv('data/processed/feature_rankings.csv')
    selected = rankings['feature'].tolist()[:75]

    # Train models
    config = DeepLearningConfig(
        sequence_length=20,
        hidden_size=128,
        num_layers=2,
        epochs=30,
        batch_size=256
    )

    results = train_deep_learning_models(
        features,
        target_col='target_direction_1',
        selected_features=selected,
        output_dir='data/models',
        config=config
    )

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  AUC-ROC:   {metrics.auc_roc:.4f}")
        print(f"  Accuracy:  {metrics.accuracy:.4f}")
        print(f"  F1:        {metrics.f1:.4f}")
        print(f"  Precision: {metrics.precision:.4f}")
        print(f"  Recall:    {metrics.recall:.4f}")
