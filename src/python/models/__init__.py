"""
SKIE-Ninja Models Module

ML model training, validation, and export.

Supported models:
- Random Forest (scikit-learn)
- XGBoost (gradient boosting)
- LightGBM (gradient boosting)
- LSTM/GRU (PyTorch) with Purged CV for overfitting prevention
- Transformer (Quantformer-inspired)
- Regime-specific ensemble models

Export format: ONNX for NinjaTrader integration
"""

from .model_trainer import train_models
from .deep_learning_trainer import (
    DeepLearningConfig,
    LSTMClassifier,
    GRUClassifier,
    DeepLearningTrainer,
    train_deep_learning_models
)
from .purged_cv_rnn_trainer import (
    PurgedCVConfig,
    PurgedKFold,
    WalkForwardCV,
    RegularizedLSTM,
    RegularizedGRU,
    CVResults,
    PurgedCVRNNTrainer,
    train_rnn_with_purged_cv
)
from .rnn_hyperparameter_optimizer import (
    RNNHyperparams,
    RNNHyperparameterOptimizer,
    quick_rnn_grid_search
)

__all__ = [
    # Model Trainer
    'train_models',

    # Deep Learning
    'DeepLearningConfig',
    'LSTMClassifier',
    'GRUClassifier',
    'DeepLearningTrainer',
    'train_deep_learning_models',

    # Purged CV RNN (addresses overfitting)
    'PurgedCVConfig',
    'PurgedKFold',
    'WalkForwardCV',
    'RegularizedLSTM',
    'RegularizedGRU',
    'CVResults',
    'PurgedCVRNNTrainer',
    'train_rnn_with_purged_cv',

    # Hyperparameter Optimization
    'RNNHyperparams',
    'RNNHyperparameterOptimizer',
    'quick_rnn_grid_search'
]
