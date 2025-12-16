"""
Volume Feature Utilities
========================

Consolidated implementations of volume-based features.

Author: SKIE_Ninja Development Team
Created: 2025-12-15
"""

import numpy as np
import pandas as pd
from typing import Tuple

# Standard epsilon for division safety
EPSILON = 1e-10


def calculate_volume_ratio(
    volume: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Calculate volume relative to moving average.

    Volume Ratio = volume / SMA(volume, period)

    Args:
        volume: Volume series
        period: Lookback period (default 20)

    Returns:
        Volume ratio series (1.0 = average volume)
    """
    vol_ma = volume.rolling(period).mean()
    return volume / (vol_ma + EPSILON)  # EPSILON for division safety


def calculate_volume_momentum(
    volume: pd.Series,
    period: int = 10
) -> pd.Series:
    """
    Calculate volume momentum (rate of change).

    Args:
        volume: Volume series
        period: Lookback period (default 10)

    Returns:
        Volume momentum series
    """
    return volume.pct_change(period)


def calculate_volume_std(
    volume: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Calculate rolling standard deviation of volume.

    Args:
        volume: Volume series
        period: Lookback period (default 20)

    Returns:
        Volume standard deviation series
    """
    return volume.rolling(period).std()


def calculate_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = None
) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).

    VWAP = sum(typical_price * volume) / sum(volume)
    typical_price = (high + low + close) / 3

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume series
        period: Rolling period (None = cumulative)

    Returns:
        VWAP series
    """
    typical_price = (high + low + close) / 3
    tp_volume = typical_price * volume

    if period is None:
        # Cumulative VWAP (session-based)
        cum_tp_vol = tp_volume.cumsum()
        cum_vol = volume.cumsum()
        vwap = cum_tp_vol / (cum_vol + EPSILON)
    else:
        # Rolling VWAP
        rolling_tp_vol = tp_volume.rolling(period).sum()
        rolling_vol = volume.rolling(period).sum()
        vwap = rolling_tp_vol / (rolling_vol + EPSILON)

    return vwap


def calculate_price_volume_trend(
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """
    Calculate Price Volume Trend (PVT).

    PVT = previous_PVT + volume * (close - prev_close) / prev_close

    Args:
        close: Close prices
        volume: Volume series

    Returns:
        PVT series
    """
    returns = close.pct_change()
    pvt = (volume * returns).cumsum()
    return pvt


def calculate_obv(
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).

    OBV accumulates volume on up days, subtracts on down days.

    Args:
        close: Close prices
        volume: Volume series

    Returns:
        OBV series
    """
    direction = np.sign(close.diff())
    obv = (direction * volume).cumsum()
    return obv


def calculate_volume_profile_features(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate volume profile features.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume series
        period: Lookback period

    Returns:
        Tuple of (vwap, vwap_distance, relative_volume)
    """
    vwap = calculate_vwap(high, low, close, volume, period)
    vwap_distance = (close - vwap) / (vwap + EPSILON)
    relative_volume = calculate_volume_ratio(volume, period)

    return vwap, vwap_distance, relative_volume
