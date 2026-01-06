"""
Validation module for SKIE_Ninja strategy.

Implements rigorous statistical validation methods per canonical literature:
- CPCV (Combinatorial Purged Cross-Validation) per Lopez de Prado (2018) Ch. 7
- PBO (Probability of Backtest Overfitting) per Bailey et al. (2014)
- DSR (Deflated Sharpe Ratio) per Bailey & Lopez de Prado (2014)

References:
    - Lopez de Prado (2018) "Advances in Financial Machine Learning"
    - Bailey et al. (2014) "The Probability of Backtest Overfitting" SSRN 2326253
    - Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio" SSRN 2460551
"""

from .cpcv_pbo import (
    CPCVConfig,
    CPCVResult,
    CombinatorialPurgedKFold,
    calculate_pbo,
    calculate_dsr,
    run_cpcv_validation,
    run_pbo_analysis,
    run_dsr_analysis,
    print_validation_report
)

__all__ = [
    'CPCVConfig',
    'CPCVResult',
    'CombinatorialPurgedKFold',
    'calculate_pbo',
    'calculate_dsr',
    'run_cpcv_validation',
    'run_pbo_analysis',
    'run_dsr_analysis',
    'print_validation_report'
]
