"""
SKIE-Ninja Backtesting Module

Walk-forward backtesting framework with comprehensive trade metrics.

Features:
- Walk-forward validation (train, embargo, test, repeat)
- RTH-only trading enforcement
- Data leakage prevention and detection
- Comprehensive metrics (P&L, drawdown, MFE/MAE, Sharpe, Sortino, Calmar)
- Detailed trade logging with entry/exit times, contracts, duration
"""

# Original walk-forward backtest
from .walk_forward_backtest import (
    BacktestConfig as WFBacktestConfig,
    Trade as WFTrade,
    BacktestMetrics as WFBacktestMetrics,
    WalkForwardBacktester,
    run_backtest,
    generate_backtest_report
)

# Comprehensive backtest (enhanced version)
from .comprehensive_backtest import (
    BacktestConfig,
    Trade,
    TradeDirection,
    ExitReason,
    BacktestMetrics,
    DataLeakageChecker,
    RTHFilter,
    ComprehensiveBacktester,
    generate_comprehensive_report,
    run_comprehensive_backtest
)

__all__ = [
    # Original walk-forward
    'WFBacktestConfig',
    'WFTrade',
    'WFBacktestMetrics',
    'WalkForwardBacktester',
    'run_backtest',
    'generate_backtest_report',

    # Comprehensive backtest (recommended)
    'BacktestConfig',
    'Trade',
    'TradeDirection',
    'ExitReason',
    'BacktestMetrics',
    'DataLeakageChecker',
    'RTHFilter',
    'ComprehensiveBacktester',
    'generate_comprehensive_report',
    'run_comprehensive_backtest'
]
