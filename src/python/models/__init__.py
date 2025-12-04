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

NEW (2025-12-04 - Research Phase Implementation):
- Meta-Labeling (Lopez de Prado) for bet sizing
- Temporal Fusion Transformer for multi-horizon prediction

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
# Meta-Labeling (Lopez de Prado)
from .meta_labeling import (
    MetaLabelConfig,
    MetaLabeler,
    MetaLabelingPipeline
)
# Temporal Fusion Transformer
from .temporal_fusion_transformer import (
    TFTConfig,
    TFTTrainer,
    create_tft_features
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
    'quick_rnn_grid_search',

    # Meta-Labeling (NEW)
    'MetaLabelConfig',
    'MetaLabeler',
    'MetaLabelingPipeline',

    # Temporal Fusion Transformer (NEW)
    'TFTConfig',
    'TFTTrainer',
    'create_tft_features'
]
