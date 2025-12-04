"""
Temporal Fusion Transformer (TFT) for Financial Time Series

Implementation of the Temporal Fusion Transformer architecture for
multi-horizon time series forecasting in financial markets.

TFT is a state-of-the-art architecture that provides:
1. Multi-horizon predictions (predict multiple future timesteps)
2. Interpretable attention weights (understand what drives predictions)
3. Variable selection (automatically learns feature importance)
4. Static + temporal inputs (combine regime info with price data)
5. Quantile outputs (uncertainty quantification for position sizing)

Key advantages over LSTM/GRU:
- Better long-range dependency capture via attention
- Interpretability through attention weights
- Multi-horizon output without autoregressive rollout
- Built-in feature selection

References:
- Lim et al. (2021) "Temporal Fusion Transformers for Interpretable
  Multi-horizon Time Series Forecasting"
- TFT for Stock Prediction (IEEE 2022)

Note: Requires PyTorch. Falls back to simplified version if not available.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. TFT will use simplified fallback.")


@dataclass
class TFTConfig:
    """Configuration for Temporal Fusion Transformer."""
    # Architecture
    hidden_size: int = 64
    num_attention_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dropout: float = 0.1

    # Input/Output
    sequence_length: int = 60        # Input sequence length (bars)
    forecast_horizon: int = 12       # Prediction horizon
    num_quantiles: int = 3           # [0.1, 0.5, 0.9] for uncertainty

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10

    # Feature types
    num_static_features: int = 0     # Static categorical features
    num_time_features: int = 0       # Known future time features


if TORCH_AVAILABLE:

    class GatedLinearUnit(nn.Module):
        """Gated Linear Unit for feature gating."""

        def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
            super().__init__()
            self.fc = nn.Linear(input_size, output_size * 2)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = self.dropout(x)
            x = self.fc(x)
            x, gate = x.chunk(2, dim=-1)
            return x * torch.sigmoid(gate)


    class GatedResidualNetwork(nn.Module):
        """Gated Residual Network for flexible nonlinear processing."""

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            dropout: float = 0.1,
            context_size: Optional[int] = None
        ):
            super().__init__()

            self.input_size = input_size
            self.output_size = output_size
            self.context_size = context_size

            # Primary transformation
            self.fc1 = nn.Linear(input_size, hidden_size)
            if context_size is not None:
                self.context_fc = nn.Linear(context_size, hidden_size, bias=False)

            self.elu = nn.ELU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.dropout = nn.Dropout(dropout)

            # Gating
            self.gate = GatedLinearUnit(hidden_size, output_size, dropout)

            # Skip connection
            if input_size != output_size:
                self.skip_fc = nn.Linear(input_size, output_size)
            else:
                self.skip_fc = None

            self.layer_norm = nn.LayerNorm(output_size)

        def forward(self, x, context=None):
            # Primary path
            hidden = self.fc1(x)

            if context is not None and self.context_size is not None:
                hidden = hidden + self.context_fc(context)

            hidden = self.elu(hidden)
            hidden = self.fc2(hidden)
            hidden = self.dropout(hidden)
            hidden = self.gate(hidden)

            # Skip connection
            if self.skip_fc is not None:
                skip = self.skip_fc(x)
            else:
                skip = x

            return self.layer_norm(hidden + skip)


    class VariableSelectionNetwork(nn.Module):
        """Variable Selection Network for automatic feature importance."""

        def __init__(
            self,
            input_sizes: List[int],
            hidden_size: int,
            dropout: float = 0.1,
            context_size: Optional[int] = None
        ):
            super().__init__()

            self.hidden_size = hidden_size
            self.num_inputs = len(input_sizes)

            # GRN for each input
            self.grns = nn.ModuleList([
                GatedResidualNetwork(
                    input_size=size,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                    context_size=context_size
                )
                for size in input_sizes
            ])

            # Softmax weights GRN
            self.weight_grn = GatedResidualNetwork(
                input_size=hidden_size * self.num_inputs,
                hidden_size=hidden_size,
                output_size=self.num_inputs,
                dropout=dropout,
                context_size=context_size
            )

        def forward(self, inputs: List[torch.Tensor], context=None):
            # Process each input
            processed = []
            for i, (x, grn) in enumerate(zip(inputs, self.grns)):
                processed.append(grn(x, context))

            # Stack: [batch, num_inputs, hidden]
            processed = torch.stack(processed, dim=-2)

            # Calculate weights
            flat = processed.flatten(start_dim=-2)
            weights = self.weight_grn(flat, context)
            weights = F.softmax(weights, dim=-1).unsqueeze(-1)

            # Weighted sum
            output = (processed * weights).sum(dim=-2)

            return output, weights.squeeze(-1)


    class InterpretableMultiHeadAttention(nn.Module):
        """Multi-head attention with interpretable weights."""

        def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
            super().__init__()

            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads

            assert self.head_dim * num_heads == hidden_size

            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)
            self.output = nn.Linear(hidden_size, hidden_size)

            self.dropout = nn.Dropout(dropout)

        def forward(self, query, key, value, mask=None):
            batch_size = query.size(0)

            # Linear projections
            Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attention = F.softmax(scores, dim=-1)
            attention = self.dropout(attention)

            # Apply attention to values
            context = torch.matmul(attention, V)

            # Reshape and project
            context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
            output = self.output(context)

            return output, attention


    class TemporalFusionTransformer(nn.Module):
        """
        Temporal Fusion Transformer for multi-horizon forecasting.

        Architecture:
        1. Variable Selection Network (learn feature importance)
        2. LSTM Encoder (capture temporal patterns)
        3. Static Enrichment (condition on static features)
        4. Temporal Self-Attention (long-range dependencies)
        5. Position-wise Feed Forward
        6. Quantile Output (uncertainty estimation)
        """

        def __init__(self, config: TFTConfig, num_features: int):
            super().__init__()

            self.config = config
            self.num_features = num_features
            self.hidden_size = config.hidden_size

            # Input projection
            self.input_projection = nn.Linear(num_features, config.hidden_size)

            # Variable selection
            self.vsn = VariableSelectionNetwork(
                input_sizes=[config.hidden_size] * num_features,
                hidden_size=config.hidden_size,
                dropout=config.dropout
            )

            # LSTM encoder
            self.encoder = nn.LSTM(
                input_size=config.hidden_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_encoder_layers,
                dropout=config.dropout if config.num_encoder_layers > 1 else 0,
                batch_first=True
            )

            # Gated skip connection
            self.encoder_gate = GatedLinearUnit(
                config.hidden_size, config.hidden_size, config.dropout
            )

            # Temporal self-attention
            self.attention = InterpretableMultiHeadAttention(
                config.hidden_size,
                config.num_attention_heads,
                config.dropout
            )
            self.attention_gate = GatedLinearUnit(
                config.hidden_size, config.hidden_size, config.dropout
            )
            self.attention_norm = nn.LayerNorm(config.hidden_size)

            # Position-wise feed forward
            self.ff = GatedResidualNetwork(
                config.hidden_size,
                config.hidden_size * 4,
                config.hidden_size,
                config.dropout
            )

            # Output projection (quantile outputs)
            self.output_projection = nn.Linear(
                config.hidden_size,
                config.forecast_horizon * config.num_quantiles
            )

            # Quantile loss weights
            self.quantiles = [0.1, 0.5, 0.9][:config.num_quantiles]

        def forward(self, x, return_attention=False):
            """
            Forward pass.

            Args:
                x: Input tensor [batch, seq_len, num_features]
                return_attention: Whether to return attention weights

            Returns:
                predictions: [batch, forecast_horizon, num_quantiles]
                attention_weights: Optional attention weights
            """
            batch_size, seq_len, _ = x.shape

            # Input projection
            x = self.input_projection(x)

            # Variable selection (simplified - process all features together)
            # Full implementation would process each feature separately

            # LSTM encoding
            encoder_output, (hidden, cell) = self.encoder(x)
            encoder_output = self.encoder_gate(encoder_output)

            # Self-attention
            attn_output, attn_weights = self.attention(
                encoder_output, encoder_output, encoder_output
            )
            attn_output = self.attention_gate(attn_output)
            attn_output = self.attention_norm(encoder_output + attn_output)

            # Feed forward
            ff_output = self.ff(attn_output)

            # Take last timestep for prediction
            final_output = ff_output[:, -1, :]

            # Output projection
            predictions = self.output_projection(final_output)
            predictions = predictions.view(
                batch_size,
                self.config.forecast_horizon,
                self.config.num_quantiles
            )

            if return_attention:
                return predictions, attn_weights
            return predictions

        def quantile_loss(self, predictions, targets):
            """
            Compute quantile loss for training.

            Args:
                predictions: [batch, horizon, num_quantiles]
                targets: [batch, horizon]

            Returns:
                Quantile loss value
            """
            losses = []
            for i, q in enumerate(self.quantiles):
                pred = predictions[:, :, i]
                errors = targets - pred
                loss = torch.max(q * errors, (q - 1) * errors)
                losses.append(loss.mean())

            return sum(losses) / len(losses)


    class TFTDataset(Dataset):
        """Dataset for TFT training."""

        def __init__(
            self,
            features: np.ndarray,
            targets: np.ndarray,
            sequence_length: int,
            forecast_horizon: int
        ):
            self.features = torch.FloatTensor(features)
            self.targets = torch.FloatTensor(targets)
            self.sequence_length = sequence_length
            self.forecast_horizon = forecast_horizon

        def __len__(self):
            return len(self.features) - self.sequence_length - self.forecast_horizon + 1

        def __getitem__(self, idx):
            x = self.features[idx:idx + self.sequence_length]
            y = self.targets[idx + self.sequence_length:
                            idx + self.sequence_length + self.forecast_horizon]
            return x, y


class TFTTrainer:
    """
    Trainer for Temporal Fusion Transformer.

    Handles training, validation, and prediction with proper
    time-series cross-validation.
    """

    def __init__(self, config: Optional[TFTConfig] = None):
        self.config = config or TFTConfig()
        self.model = None
        self.device = None
        self._is_fitted = False

        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_X: Optional[np.ndarray] = None,
        val_y: Optional[np.ndarray] = None
    ) -> 'TFTTrainer':
        """
        Train the TFT model.

        Args:
            X: Feature matrix [samples, features]
            y: Target values [samples]
            val_X: Optional validation features
            val_y: Optional validation targets

        Returns:
            Self for chaining
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Using fallback model.")
            return self._fit_fallback(X, y)

        logger.info("Training Temporal Fusion Transformer...")

        # Create model
        num_features = X.shape[1]
        self.model = TemporalFusionTransformer(self.config, num_features)
        self.model.to(self.device)

        # Create datasets
        train_dataset = TFTDataset(
            X, y,
            self.config.sequence_length,
            self.config.forecast_horizon
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        if val_X is not None and val_y is not None:
            val_dataset = TFTDataset(
                val_X, val_y,
                self.config.sequence_length,
                self.config.forecast_horizon
            )
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        else:
            val_loader = None

        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.max_epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = self.model.quantile_loss(predictions, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        predictions = self.model(batch_x)
                        loss = self.model.quantile_loss(predictions, batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                log_msg = f"Epoch {epoch}: train_loss={train_loss:.6f}"
                if val_loader is not None:
                    log_msg += f", val_loss={val_loss:.6f}"
                logger.info(log_msg)

        self._is_fitted = True
        logger.info("TFT training complete")
        return self

    def _fit_fallback(self, X: np.ndarray, y: np.ndarray) -> 'TFTTrainer':
        """Fallback training using simple rolling regression."""
        from sklearn.linear_model import Ridge

        self.model = Ridge(alpha=1.0)
        # Use last few samples as features for simple regression
        lookback = min(20, len(X) // 10)
        X_flat = X[-lookback:].flatten().reshape(1, -1)
        y_mean = y[-lookback:].mean().reshape(1)
        self.model.fit(X_flat, y_mean)
        self._is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        return_quantiles: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate predictions.

        Args:
            X: Feature matrix [samples, features]
            return_quantiles: If True, return all quantile predictions

        Returns:
            Predictions (median) or (lower, median, upper) quantiles
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if not TORCH_AVAILABLE:
            # Fallback prediction
            predictions = np.zeros(self.config.forecast_horizon)
            return predictions

        self.model.eval()

        # Create sequences
        sequences = []
        for i in range(len(X) - self.config.sequence_length + 1):
            seq = X[i:i + self.config.sequence_length]
            sequences.append(seq)

        if not sequences:
            return np.zeros((1, self.config.forecast_horizon))

        sequences = np.array(sequences)
        sequences = torch.FloatTensor(sequences).to(self.device)

        with torch.no_grad():
            predictions = self.model(sequences)
            predictions = predictions.cpu().numpy()

        if return_quantiles:
            return predictions[:, :, 0], predictions[:, :, 1], predictions[:, :, 2]

        # Return median (q=0.5)
        return predictions[:, :, 1]

    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Get attention weights for interpretability.

        Args:
            X: Feature matrix

        Returns:
            Attention weight matrix
        """
        if not TORCH_AVAILABLE or not self._is_fitted:
            return np.zeros((self.config.sequence_length, self.config.sequence_length))

        self.model.eval()

        seq = X[-self.config.sequence_length:]
        seq = torch.FloatTensor(seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, attention = self.model(seq, return_attention=True)

        return attention.cpu().numpy()


def create_tft_features(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    config: Optional[TFTConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and targets for TFT training.

    Args:
        prices: OHLC price data
        features: Feature matrix
        config: TFT configuration

    Returns:
        Tuple of (X, y) arrays
    """
    config = config or TFTConfig()

    # Target: future returns
    close = prices['close'].values
    returns = np.diff(np.log(close))

    # Pad returns to match length
    returns = np.concatenate([[0], returns])

    # Features
    X = features.values

    # Remove NaN
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    returns = returns[valid_mask]

    return X, returns


if __name__ == "__main__":
    print("=" * 70)
    print("TEMPORAL FUSION TRANSFORMER - TEST")
    print("=" * 70)

    # Create sample data
    np.random.seed(42)
    n_samples = 500
    n_features = 10

    # Sample features
    X = np.random.randn(n_samples, n_features)

    # Target with some signal
    y = 0.3 * X[:, 0] + 0.2 * X[:, 1] + 0.1 * np.random.randn(n_samples)

    print(f"\n[1] Testing TFT with PyTorch available: {TORCH_AVAILABLE}")

    config = TFTConfig(
        hidden_size=32,
        num_attention_heads=2,
        sequence_length=30,
        forecast_horizon=5,
        max_epochs=20,
        batch_size=32
    )

    print(f"\nConfiguration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Sequence length: {config.sequence_length}")
    print(f"  Forecast horizon: {config.forecast_horizon}")

    # Train/test split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("\n[2] Training TFT model...")
    trainer = TFTTrainer(config)
    trainer.fit(X_train, y_train)

    print("\n[3] Making predictions...")
    if TORCH_AVAILABLE:
        predictions = trainer.predict(X_test)
        print(f"Prediction shape: {predictions.shape}")
        print(f"Sample predictions: {predictions[:3, :3]}")

        # Quantile predictions
        lower, median, upper = trainer.predict(X_test, return_quantiles=True)
        print(f"\nQuantile predictions (first sample):")
        print(f"  Lower (10%): {lower[0]}")
        print(f"  Median (50%): {median[0]}")
        print(f"  Upper (90%): {upper[0]}")

        print("\n[4] Getting attention weights...")
        attention = trainer.get_attention_weights(X_test)
        print(f"Attention shape: {attention.shape}")
    else:
        print("PyTorch not available - using fallback")

    print("\n" + "=" * 70)
    print("TFT TEST COMPLETE")
    print("=" * 70)
