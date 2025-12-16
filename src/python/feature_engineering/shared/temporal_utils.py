"""
Temporal Feature Utilities
==========================

Consolidated implementations of time-based features.

Author: SKIE_Ninja Development Team
Created: 2025-12-15
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


def encode_cyclical_time(
    timestamps: pd.DatetimeIndex
) -> Dict[str, pd.Series]:
    """
    Encode time features as cyclical (sin/cos) values.

    This avoids the discontinuity problem (23:59 -> 00:00).

    Args:
        timestamps: DatetimeIndex of timestamps

    Returns:
        Dictionary with hour_sin, hour_cos, dow_sin, dow_cos
    """
    hours = timestamps.hour + timestamps.minute / 60
    dow = timestamps.dayofweek

    return {
        'hour_sin': np.sin(2 * np.pi * hours / 24),
        'hour_cos': np.cos(2 * np.pi * hours / 24),
        'dow_sin': np.sin(2 * np.pi * dow / 5),  # 5 trading days
        'dow_cos': np.cos(2 * np.pi * dow / 5),
    }


def encode_hour(
    timestamps: pd.DatetimeIndex
) -> Tuple[pd.Series, pd.Series]:
    """
    Encode hour of day as sin/cos.

    Args:
        timestamps: DatetimeIndex

    Returns:
        Tuple of (hour_sin, hour_cos) series
    """
    hours = timestamps.hour + timestamps.minute / 60
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    return pd.Series(hour_sin, index=timestamps), pd.Series(hour_cos, index=timestamps)


def encode_day_of_week(
    timestamps: pd.DatetimeIndex,
    trading_days: int = 5
) -> Tuple[pd.Series, pd.Series]:
    """
    Encode day of week as sin/cos.

    Args:
        timestamps: DatetimeIndex
        trading_days: Number of trading days (default 5)

    Returns:
        Tuple of (dow_sin, dow_cos) series
    """
    dow = timestamps.dayofweek
    dow_sin = np.sin(2 * np.pi * dow / trading_days)
    dow_cos = np.cos(2 * np.pi * dow / trading_days)
    return pd.Series(dow_sin, index=timestamps), pd.Series(dow_cos, index=timestamps)


def get_session_time_features(
    timestamps: pd.DatetimeIndex,
    market_open: float = 9.5,   # 9:30 AM
    market_close: float = 16.0  # 4:00 PM
) -> Dict[str, pd.Series]:
    """
    Calculate session-relative time features.

    Args:
        timestamps: DatetimeIndex
        market_open: Market open hour (decimal)
        market_close: Market close hour (decimal)

    Returns:
        Dictionary with session_progress, minutes_from_open, etc.
    """
    hours = timestamps.hour + timestamps.minute / 60

    session_length = market_close - market_open
    time_in_session = hours - market_open

    # Clip to session bounds
    time_in_session = np.clip(time_in_session, 0, session_length)

    return {
        'session_progress': time_in_session / session_length,
        'minutes_from_open': (hours - market_open) * 60,
        'minutes_to_close': (market_close - hours) * 60,
        'is_opening_hour': ((hours >= market_open) & (hours < market_open + 1)).astype(int),
        'is_closing_hour': ((hours >= market_close - 1) & (hours < market_close)).astype(int),
        'is_lunch_hour': ((hours >= 12) & (hours < 13)).astype(int),
    }


def get_calendar_features(
    timestamps: pd.DatetimeIndex
) -> Dict[str, pd.Series]:
    """
    Calculate calendar-based features.

    Args:
        timestamps: DatetimeIndex

    Returns:
        Dictionary with day_of_month, week_of_year, etc.
    """
    return {
        'day_of_month': timestamps.day,
        'week_of_year': timestamps.isocalendar().week.values,
        'month': timestamps.month,
        'quarter': timestamps.quarter,
        'is_month_start': timestamps.is_month_start.astype(int),
        'is_month_end': timestamps.is_month_end.astype(int),
        'is_quarter_end': timestamps.is_quarter_end.astype(int),
    }


def is_high_volatility_period(
    timestamps: pd.DatetimeIndex
) -> pd.Series:
    """
    Identify historically high-volatility periods.

    High volatility typically occurs:
    - First 30 min after open
    - Last 30 min before close
    - Major economic release times (8:30 AM)

    Args:
        timestamps: DatetimeIndex

    Returns:
        Boolean series indicating high-volatility periods
    """
    hours = timestamps.hour + timestamps.minute / 60

    # Opening volatility (9:30-10:00 AM)
    is_open = (hours >= 9.5) & (hours < 10.0)

    # Closing volatility (3:30-4:00 PM)
    is_close = (hours >= 15.5) & (hours < 16.0)

    # Economic releases (8:30 AM)
    is_release = (hours >= 8.5) & (hours < 9.0)

    return is_open | is_close | is_release
