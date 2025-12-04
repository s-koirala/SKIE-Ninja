"""
SKIE-Ninja RNN Hyperparameter Optimizer

Grid search cross-validation for LSTM/GRU models with time-series aware validation.
Addresses common RNN overfitting issues in financial prediction.

Literature Review on RNN Overfitting in Financial Prediction:
============================================================

1. **Problem: RNNs Overfit on Financial Time Series**
   - Fischer & Krauss (2018) - "Deep Learning with Long Short-Term Memory Networks
     for Financial Market Predictions" found LSTM models prone to overfitting on
     noisy financial data, recommending dropout rates of 0.2-0.5

2. **Key Causes of RNN Overfitting in Finance:**
   - Low signal-to-noise ratio in financial data
   - Non-stationary distributions (regime changes)
   - Limited training samples relative to model complexity
   - Sequential correlation creates data leakage if not handled properly

3. **Recommended Solutions from Literature:**
   a) **Dropout Regularization** (Srivastava et al., 2014)
      - Apply dropout to recurrent connections (recurrent_dropout)
      - Typical range: 0.2-0.5 for financial data

   b) **Early Stopping** (Prechelt, 1998)
      - Monitor validation loss, stop when it starts increasing
      - Patience of 5-10 epochs commonly used

   c) **Weight Decay / L2 Regularization** (Krogh & Hertz, 1992)
      - Add L2 penalty to loss function
      - Typical values: 1e-4 to 1e-6

   d) **Reduced Model Complexity**
      - Fewer LSTM/GRU layers (1-2 typically sufficient)
      - Smaller hidden sizes (32-128)
      - Shorter sequence lengths

   e) **Walk-Forward Validation** (Tashman, 2000)
      - Use rolling/expanding window CV instead of random splits
      - Prevent look-ahead bias

   f) **Batch Normalization** (Ioffe & Szegedy, 2015)
      - Normalize layer inputs to reduce internal covariate shift

   g) **Gradient Clipping** (Pascanu et al., 2013)
      - Prevent exploding gradients, typical max norm: 1.0-5.0

4. **Why Gradient Boosting Often Beats Deep Learning on Tabular Data:**
   - Shwartz-Ziv & Armon (2022) - "Tabular Data: Deep Learning is Not All You Need"
   - Grinsztajn et al. (2022) - "Why do tree-based models still outperform deep
     learning on typical tabular data?"
   - Key insight: Deep learning excels when data has spatial/temporal structure
     that can be exploited; for engineered tabular features, this structure is
     already captured, making boosting methods more effective.

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import json
from datetime import datetime
from itertools import product
import warnings

logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class RNNHyperparams:
    """Hyperparameters for RNN models."""
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    recurrent_dropout: float = 0.0  # Applied within recurrent connections
    sequence_length: int = 20
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    epochs: int = 50
    early_stopping_patience: int = 10
    bidirectional: bool = False
    use_batch_norm: bool = False


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int):
        self.sequence_length = sequence_length
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length]
        return x_seq, y_target


class LSTMWithRegularization(nn.Module):
    """LSTM with enhanced regularization options."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_batch_norm: bool = False
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_batch_norm = use_batch_norm
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

        # Batch normalization
        fc_input_size = hidden_size * self.num_directions
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(fc_input_size)

        # Fully connected layers with dropout
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_last = h_n[-1]

        if self.use_batch_norm:
            h_last = self.batch_norm(h_last)

        out = self.fc(h_last)
        return out.squeeze()


class GRUWithRegularization(nn.Module):
    """GRU with enhanced regularization options."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_batch_norm: bool = False
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_batch_norm = use_batch_norm
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        fc_input_size = hidden_size * self.num_directions
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(fc_input_size)

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        gru_out, h_n = self.gru(x)

        if self.bidirectional:
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_last = h_n[-1]

        if self.use_batch_norm:
            h_last = self.batch_norm(h_last)

        out = self.fc(h_last)
        return out.squeeze()


class RNNHyperparameterOptimizer:
    """
    Grid search cross-validation for LSTM/GRU hyperparameters.

    Uses walk-forward validation to prevent look-ahead bias.
    """

    def __init__(
        self,
        param_grid: Optional[Dict[str, List]] = None,
        n_folds: int = 3,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ):
        """
        Initialize optimizer.

        Parameters:
        -----------
        param_grid : Dict[str, List]
            Hyperparameter search space
        n_folds : int
            Number of walk-forward folds
        train_ratio : float
            Proportion of data for training in each fold
        val_ratio : float
            Proportion of data for validation in each fold
        """
        self.param_grid = param_grid or self._default_param_grid()
        self.n_folds = n_folds
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.results = []
        self.best_params = None
        self.best_score = 0.0

    def _default_param_grid(self) -> Dict[str, List]:
        """Default parameter grid based on literature recommendations."""
        return {
            'hidden_size': [64, 128],
            'num_layers': [1, 2],
            'dropout': [0.2, 0.3, 0.5],
            'sequence_length': [10, 20, 30],
            'learning_rate': [0.001, 0.0005],
            'weight_decay': [1e-5, 1e-4],
            'batch_size': [128, 256],
            'bidirectional': [False],  # Usually doesn't help for financial data
            'use_batch_norm': [False, True]
        }

    def _generate_param_combinations(self) -> List[Dict]:
        """Generate all parameter combinations."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]

    def _create_walk_forward_splits(
        self,
        n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Create walk-forward validation splits."""
        splits = []

        # Calculate fold sizes
        fold_size = n_samples // (self.n_folds + 2)  # Leave room for test

        for fold in range(self.n_folds):
            train_end = (fold + 1) * fold_size + fold_size
            val_start = train_end
            val_end = val_start + fold_size

            if val_end > n_samples:
                break

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)

            splits.append((train_idx, val_idx))

        return splits

    def _train_single_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict,
        model_type: str = 'lstm',
        max_epochs: int = 30
    ) -> Tuple[float, float, int]:
        """Train a single model and return validation metrics."""
        # Create datasets
        train_dataset = TimeSeriesDataset(
            X_train, y_train, params['sequence_length']
        )
        val_dataset = TimeSeriesDataset(
            X_val, y_val, params['sequence_length']
        )

        if len(train_dataset) < params['batch_size']:
            return 0.5, 0.5, 0  # Return baseline if not enough data

        train_loader = DataLoader(
            train_dataset, batch_size=params['batch_size'], shuffle=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=params['batch_size'], shuffle=False
        )

        # Create model
        input_size = X_train.shape[1]
        if model_type == 'lstm':
            model = LSTMWithRegularization(
                input_size=input_size,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                bidirectional=params.get('bidirectional', False),
                use_batch_norm=params.get('use_batch_norm', False)
            )
        else:
            model = GRUWithRegularization(
                input_size=input_size,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                bidirectional=params.get('bidirectional', False),
                use_batch_norm=params.get('use_batch_norm', False)
            )

        model = model.to(DEVICE)

        # Optimizer with weight decay
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params.get('weight_decay', 1e-5)
        )
        criterion = nn.BCELoss()

        # Training with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        epochs_trained = 0

        for epoch in range(max_epochs):
            # Train
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validate
            model.eval()
            val_losses = []
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)

                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_losses.append(loss.item())
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())

            avg_val_loss = np.mean(val_losses)
            epochs_trained = epoch + 1

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:  # Quick early stopping for grid search
                    break

        # Calculate final metrics
        try:
            val_auc = roc_auc_score(val_targets, val_preds)
            val_acc = accuracy_score(
                val_targets,
                [1 if p > 0.5 else 0 for p in val_preds]
            )
        except Exception:
            val_auc = 0.5
            val_acc = 0.5

        return val_auc, val_acc, epochs_trained

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = 'lstm',
        verbose: bool = True
    ) -> Dict:
        """
        Run grid search optimization.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (already scaled)
        y : np.ndarray
            Target array
        model_type : str
            'lstm' or 'gru'
        verbose : bool
            Print progress

        Returns:
        --------
        Dict with best parameters and all results
        """
        logger.info("="*60)
        logger.info(f"RNN HYPERPARAMETER OPTIMIZATION: {model_type.upper()}")
        logger.info("="*60)

        param_combinations = self._generate_param_combinations()
        logger.info(f"Testing {len(param_combinations)} parameter combinations")

        splits = self._create_walk_forward_splits(len(X))
        logger.info(f"Using {len(splits)} walk-forward folds")

        self.results = []
        best_score = 0.0
        best_params = None

        for i, params in enumerate(param_combinations):
            fold_aucs = []
            fold_accs = []

            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                auc, acc, epochs = self._train_single_model(
                    X_train, y_train, X_val, y_val, params, model_type
                )

                fold_aucs.append(auc)
                fold_accs.append(acc)

            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            mean_acc = np.mean(fold_accs)

            result = {
                'params': params,
                'mean_auc': mean_auc,
                'std_auc': std_auc,
                'mean_accuracy': mean_acc,
                'fold_aucs': fold_aucs
            }
            self.results.append(result)

            if mean_auc > best_score:
                best_score = mean_auc
                best_params = params

            if verbose and (i + 1) % 5 == 0:
                logger.info(
                    f"Combo {i+1}/{len(param_combinations)}: "
                    f"AUC={mean_auc:.4f} (+/-{std_auc:.4f}), "
                    f"hidden={params['hidden_size']}, "
                    f"dropout={params['dropout']}, "
                    f"seq_len={params['sequence_length']}"
                )

        self.best_params = best_params
        self.best_score = best_score

        logger.info(f"\nBest AUC: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': self.results
        }

    def save_results(self, output_dir: str):
        """Save optimization results to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save results
        results_df = pd.DataFrame([
            {
                **r['params'],
                'mean_auc': r['mean_auc'],
                'std_auc': r['std_auc'],
                'mean_accuracy': r['mean_accuracy']
            }
            for r in self.results
        ])
        results_df = results_df.sort_values('mean_auc', ascending=False)
        results_df.to_csv(
            output_path / f'rnn_grid_search_{timestamp}.csv',
            index=False
        )

        # Save best params
        with open(output_path / f'rnn_best_params_{timestamp}.json', 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'best_score': self.best_score,
                'timestamp': timestamp
            }, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def quick_rnn_grid_search(
    features: pd.DataFrame,
    target_col: str = 'target_direction_1',
    selected_features: Optional[List[str]] = None,
    model_type: str = 'lstm',
    output_dir: str = 'data/models'
) -> Dict:
    """
    Quick grid search for RNN hyperparameters.

    Uses a reduced parameter grid for faster results.

    Parameters:
    -----------
    features : pd.DataFrame
        Feature matrix with targets
    target_col : str
        Target column name
    selected_features : List[str], optional
        Features to use
    model_type : str
        'lstm' or 'gru'
    output_dir : str
        Directory to save results

    Returns:
    --------
    Dict with best parameters
    """
    # Prepare data
    if selected_features:
        feature_cols = [f for f in selected_features if f in features.columns]
    else:
        feature_cols = [c for c in features.columns if not c.startswith('target_')]

    X = features[feature_cols].values
    y = features[target_col].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Quick parameter grid
    quick_grid = {
        'hidden_size': [64, 128],
        'num_layers': [1, 2],
        'dropout': [0.3, 0.5],
        'sequence_length': [10, 20],
        'learning_rate': [0.001],
        'weight_decay': [1e-5],
        'batch_size': [256],
        'bidirectional': [False],
        'use_batch_norm': [False, True]
    }

    # Run optimization
    optimizer = RNNHyperparameterOptimizer(
        param_grid=quick_grid,
        n_folds=3
    )

    results = optimizer.optimize(X_scaled, y, model_type)
    optimizer.save_results(output_dir)

    return results


# Exports
__all__ = [
    'RNNHyperparams',
    'RNNHyperparameterOptimizer',
    'LSTMWithRegularization',
    'GRUWithRegularization',
    'quick_rnn_grid_search'
]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("="*60)
    print("RNN HYPERPARAMETER OPTIMIZATION")
    print("="*60)

    # This would be run with actual data
    print("\nUsage:")
    print("  from rnn_hyperparameter_optimizer import quick_rnn_grid_search")
    print("  results = quick_rnn_grid_search(features, 'target_direction_1')")
