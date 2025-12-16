"""
Shared Feature Engineering Utilities
====================================

Consolidated utility functions for feature calculation to eliminate code duplication.
All strategy, deployment, and analysis scripts should import from here.

Usage:
    from feature_engineering.shared import (
        calculate_true_range,
        calculate_atr,
        calculate_rsi,
        calculate_returns,
        calculate_bollinger_bands,
        calculate_stochastic,
        calculate_volume_features,
        encode_cyclical_time
    )

Author: SKIE_Ninja Development Team
Created: 2025-12-15
"""

from .technical_utils import (
    calculate_true_range,
    calculate_atr,
    calculate_rsi,
    calculate_stochastic,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_momentum,
    calculate_ma_distance,
)

from .returns_utils import (
    calculate_simple_returns,
    calculate_log_returns,
    calculate_lagged_returns,
)

from .volume_utils import (
    calculate_volume_ratio,
    calculate_vwap,
    calculate_volume_momentum,
)

from .temporal_utils import (
    encode_cyclical_time,
    encode_hour,
    encode_day_of_week,
)

__all__ = [
    # Technical
    'calculate_true_range',
    'calculate_atr',
    'calculate_rsi',
    'calculate_stochastic',
    'calculate_bollinger_bands',
    'calculate_macd',
    'calculate_momentum',
    'calculate_ma_distance',
    # Returns
    'calculate_simple_returns',
    'calculate_log_returns',
    'calculate_lagged_returns',
    # Volume
    'calculate_volume_ratio',
    'calculate_vwap',
    'calculate_volume_momentum',
    # Temporal
    'encode_cyclical_time',
    'encode_hour',
    'encode_day_of_week',
]
