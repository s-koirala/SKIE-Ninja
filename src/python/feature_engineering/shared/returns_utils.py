"""
Returns Calculation Utilities
=============================

Consolidated implementations of return calculations.

Author: SKIE_Ninja Development Team
Created: 2025-12-15
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def calculate_simple_returns(
    close: pd.Series,
    period: int = 1
) -> pd.Series:
    """
    Calculate simple (arithmetic) returns.

    Return = (close - close[n]) / close[n]

    Args:
        close: Close prices
        period: Lookback period (default 1)

    Returns:
        Simple returns series
    """
    return close.pct_change(period)


def calculate_log_returns(
    close: pd.Series,
    period: int = 1
) -> pd.Series:
    """
    Calculate logarithmic returns.

    Log Return = ln(close / close[n])

    Args:
        close: Close prices
        period: Lookback period (default 1)

    Returns:
        Log returns series
    """
    return np.log(close / close.shift(period))


def calculate_lagged_returns(
    close: pd.Series,
    lags: List[int] = [1, 2, 3, 5, 10, 20]
) -> Dict[str, pd.Series]:
    """
    Calculate returns at multiple lags.

    Args:
        close: Close prices
        lags: List of lag periods

    Returns:
        Dictionary of {f'return_lag{lag}': series} for each lag
    """
    returns = {}
    for lag in lags:
        returns[f'return_lag{lag}'] = close.pct_change(lag)
    return returns


def calculate_cumulative_returns(
    close: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Calculate cumulative returns over rolling window.

    Args:
        close: Close prices
        period: Rolling window size

    Returns:
        Cumulative returns series
    """
    return (close / close.shift(period)) - 1


def calculate_returns_volatility(
    close: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Calculate rolling volatility of returns.

    Args:
        close: Close prices
        period: Rolling window size

    Returns:
        Rolling standard deviation of returns
    """
    returns = close.pct_change()
    return returns.rolling(period).std()


def calculate_sharpe_component(
    close: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Calculate rolling Sharpe component (mean/std of returns).

    Args:
        close: Close prices
        period: Rolling window size

    Returns:
        Rolling Sharpe component (not annualized)
    """
    returns = close.pct_change()
    mean_ret = returns.rolling(period).mean()
    std_ret = returns.rolling(period).std()
    return mean_ret / (std_ret + 1e-10)
