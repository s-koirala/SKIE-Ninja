"""
Technical Indicator Utilities
=============================

Consolidated implementations of technical indicators.
Eliminates 11+ duplications of TR, 10+ of RSI, etc.

IMPORTANT: All implementations use consistent parameters:
- EPSILON = 1e-10 (for division safety)
- Default periods match standard conventions

Author: SKIE_Ninja Development Team
Created: 2025-12-15
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple

# Standard epsilon for division safety - USE THIS EVERYWHERE
EPSILON = 1e-10


def calculate_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> pd.Series:
    """
    Calculate True Range (TR).

    TR = max(high - low, |high - prev_close|, |low - prev_close|)

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        True Range series

    Note:
        This consolidates 11+ duplicate implementations across the codebase.
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    ATR = SMA(TR, period)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default 14)

    Returns:
        ATR series
    """
    tr = calculate_true_range(high, low, close)
    atr = tr.rolling(period).mean()
    return atr


def calculate_rsi(
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    RSI = 100 - 100 / (1 + RS)
    RS = avg_gain / avg_loss

    Args:
        close: Close prices
        period: Lookback period (default 14)

    Returns:
        RSI series (0-100)

    Note:
        This consolidates 10+ duplicate implementations.
        Uses consistent EPSILON = 1e-10 for division safety.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + EPSILON)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator (%K and %D).

    %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
    %D = SMA(%K, d_period)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K lookback period (default 14)
        d_period: %D smoothing period (default 3)

    Returns:
        Tuple of (stoch_k, stoch_d) series
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + EPSILON)
    stoch_d = stoch_k.rolling(d_period).mean()
    return stoch_k, stoch_d


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Args:
        close: Close prices
        period: Lookback period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)

    Returns:
        Tuple of (middle, upper, lower, bb_pct) series
        bb_pct = (close - lower) / (upper - lower)
    """
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    bb_pct = (close - lower) / (upper - lower + EPSILON)
    return middle, upper, lower, bb_pct


def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        close: Close prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram) series
    """
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_momentum(
    close: pd.Series,
    period: int = 10
) -> pd.Series:
    """
    Calculate price momentum (rate of change).

    Momentum = (close - close[n]) / close[n]

    Args:
        close: Close prices
        period: Lookback period (default 10)

    Returns:
        Momentum series (as percentage)
    """
    return close.pct_change(period)


def calculate_ma_distance(
    close: pd.Series,
    period: int = 20,
    ma_type: str = 'sma'
) -> pd.Series:
    """
    Calculate distance from moving average.

    Distance = (close - MA) / MA

    Args:
        close: Close prices
        period: MA period (default 20)
        ma_type: 'sma' or 'ema' (default 'sma')

    Returns:
        Distance from MA as percentage
    """
    if ma_type == 'ema':
        ma = close.ewm(span=period, adjust=False).mean()
    else:
        ma = close.rolling(period).mean()

    return (close - ma) / (ma + EPSILON)


def calculate_realized_volatility(
    close: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Calculate realized volatility (rolling standard deviation of returns).

    Args:
        close: Close prices
        period: Lookback period (default 20)

    Returns:
        Realized volatility series
    """
    returns = close.pct_change()
    return returns.rolling(period).std()


def calculate_parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Calculate Parkinson volatility estimator.

    More efficient than close-to-close volatility.

    Args:
        high: High prices
        low: Low prices
        period: Lookback period (default 20)

    Returns:
        Parkinson volatility series
    """
    log_hl = np.log(high / low)
    parkinson = np.sqrt((1 / (4 * np.log(2))) * (log_hl ** 2).rolling(period).mean())
    return parkinson
