"""
SKIE-Ninja Backtesting Module

Walk-forward backtesting framework with comprehensive trade metrics.
"""

from .walk_forward_backtest import (
    BacktestConfig,
    Trade,
    BacktestMetrics,
    WalkForwardBacktester,
    run_backtest,
    generate_backtest_report
)

__all__ = [
    'BacktestConfig',
    'Trade',
    'BacktestMetrics',
    'WalkForwardBacktester',
    'run_backtest',
    'generate_backtest_report'
]
