"""
Overfitting Detection Framework
===============================

Data-driven overfitting detection based on Lopez de Prado (2018) methodology.

Implements:
1. Deflated Sharpe Ratio (DSR) - Accounts for multiple testing
2. Combinatorially Symmetric Cross-Validation (CSCV) - Overfit probability
3. Performance Stability Ratio (PSR) - Consistency across periods

References:
- Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
- Bailey, D., et al. (2014). The Deflated Sharpe Ratio. Journal of Portfolio Management.

Author: SKIE_Ninja Development Team
Created: 2025-12-15
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


@dataclass
class DSRResult:
    """Results from Deflated Sharpe Ratio calculation."""
    observed_sharpe: float
    deflated_sharpe: float
    p_value: float
    trials: int
    expected_max_sharpe: float
    is_significant: bool  # p_value < 0.05


@dataclass
class CSCVResult:
    """Results from Combinatorially Symmetric Cross-Validation."""
    overfit_probability: float
    n_combinations: int
    is_likely_overfit: bool  # overfit_prob > 0.55
    interpretation: str


@dataclass
class PSRResult:
    """Results from Performance Stability Ratio."""
    psr: float  # Proportion of periods with positive Sharpe
    stability_cv: float  # Coefficient of variation
    sharpes_by_period: Dict[str, float]
    is_stable: bool  # psr > 0.8 and cv < 0.5


def deflated_sharpe_ratio(
    returns: np.ndarray,
    trials: int = 256,
    annualization_factor: float = np.sqrt(252)
) -> DSRResult:
    """
    Compute Deflated Sharpe Ratio to account for multiple testing.

    The DSR adjusts the observed Sharpe ratio for the number of trials
    (parameter combinations) tested, providing a p-value for the null
    hypothesis that the observed Sharpe is due to chance.

    Args:
        returns: Array of strategy returns
        trials: Number of parameter combinations tested (default: 256 for 4^4 grid)
        annualization_factor: Factor to annualize Sharpe (default: sqrt(252))

    Returns:
        DSRResult with deflated Sharpe and p-value

    Reference:
        Bailey, D., et al. (2014). "The Deflated Sharpe Ratio"
    """
    returns = np.array(returns)
    n = len(returns)

    if n < 30:
        logger.warning(f"Sample size {n} is small for DSR calculation")

    # Calculate observed Sharpe ratio
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    observed_sharpe = (mean_return / std_return) * annualization_factor if std_return > 0 else 0

    # Calculate higher moments for adjustment
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis

    # Expected maximum Sharpe under null hypothesis (multiple testing)
    # E[max(Z_1, ..., Z_k)] ≈ sqrt(2 * ln(k)) for standard normal
    expected_max_sharpe = np.sqrt(2 * np.log(trials)) * (std_return / mean_return if mean_return != 0 else 1)

    # Variance of Sharpe ratio estimator
    # Var(SR) ≈ (1 + 0.5*SR^2 - skew*SR + (kurt-3)/4*SR^2) / n
    sr = observed_sharpe / annualization_factor  # De-annualize for calculation
    var_sr = (1 + 0.5 * sr**2 - skewness * sr + (kurtosis) / 4 * sr**2) / n

    # Deflated Sharpe Ratio
    # DSR = (SR - E[max SR]) / sqrt(Var(SR))
    if var_sr > 0:
        dsr = (sr - expected_max_sharpe / annualization_factor) / np.sqrt(var_sr)
    else:
        dsr = 0

    # P-value: probability of observing this DSR under null
    p_value = 1 - stats.norm.cdf(dsr)

    # Re-annualize deflated Sharpe
    deflated_sharpe = dsr * annualization_factor

    return DSRResult(
        observed_sharpe=observed_sharpe,
        deflated_sharpe=deflated_sharpe,
        p_value=p_value,
        trials=trials,
        expected_max_sharpe=expected_max_sharpe,
        is_significant=p_value < 0.05
    )


def cscv_overfit_probability(
    returns: np.ndarray,
    n_splits: int = 16
) -> CSCVResult:
    """
    Estimate probability of overfitting using Combinatorially Symmetric CV.

    Method:
    1. Split data into n_splits parts
    2. For each combination, use half as IS, half as OOS
    3. Calculate probability that IS performance > OOS
    4. If P(IS > OOS) >> 50%, likely overfit

    Args:
        returns: Array of strategy returns
        n_splits: Number of splits (default: 16)

    Returns:
        CSCVResult with overfit probability

    Reference:
        Bailey, D., et al. (2017). "The Probability of Backtest Overfitting"
    """
    returns = np.array(returns)
    n = len(returns)

    # Split into n parts
    split_size = n // n_splits
    if split_size < 10:
        logger.warning(f"Split size {split_size} is small, results may be unreliable")

    parts = [returns[i*split_size:(i+1)*split_size] for i in range(n_splits)]

    overfit_count = 0
    total_combinations = 0

    # Test all combinations where half are IS, half are OOS
    half = n_splits // 2
    for is_indices in combinations(range(n_splits), half):
        oos_indices = [i for i in range(n_splits) if i not in is_indices]

        # Combine returns for IS and OOS
        is_returns = np.concatenate([parts[i] for i in is_indices])
        oos_returns = np.concatenate([parts[i] for i in oos_indices])

        # Calculate Sharpe for each
        is_sharpe = np.mean(is_returns) / (np.std(is_returns, ddof=1) + 1e-10)
        oos_sharpe = np.mean(oos_returns) / (np.std(oos_returns, ddof=1) + 1e-10)

        if is_sharpe > oos_sharpe:
            overfit_count += 1
        total_combinations += 1

    overfit_prob = overfit_count / total_combinations if total_combinations > 0 else 0.5

    # Interpretation
    if overfit_prob < 0.55:
        interpretation = "ROBUST - No significant overfitting detected"
    elif overfit_prob < 0.70:
        interpretation = "BORDERLINE - Some overfitting risk, proceed with caution"
    else:
        interpretation = "LIKELY OVERFIT - Strategy may not generalize"

    return CSCVResult(
        overfit_probability=overfit_prob,
        n_combinations=total_combinations,
        is_likely_overfit=overfit_prob > 0.55,
        interpretation=interpretation
    )


def performance_stability_ratio(
    returns_by_period: Dict[str, np.ndarray],
    annualization_factor: float = np.sqrt(252)
) -> PSRResult:
    """
    Measure consistency of performance across different time periods.

    Args:
        returns_by_period: Dict mapping period names to return arrays
        annualization_factor: Factor to annualize Sharpe

    Returns:
        PSRResult with stability metrics
    """
    sharpes = {}

    for period, returns in returns_by_period.items():
        returns = np.array(returns)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns, ddof=1)) * annualization_factor
            sharpes[period] = sharpe

    if len(sharpes) == 0:
        return PSRResult(
            psr=0,
            stability_cv=float('inf'),
            sharpes_by_period={},
            is_stable=False
        )

    sharpe_values = list(sharpes.values())

    # PSR: proportion of periods with positive Sharpe
    psr = sum(1 for s in sharpe_values if s > 0) / len(sharpe_values)

    # Coefficient of variation (stability measure)
    mean_sharpe = np.mean(sharpe_values)
    std_sharpe = np.std(sharpe_values, ddof=1) if len(sharpe_values) > 1 else 0
    stability_cv = std_sharpe / (abs(mean_sharpe) + 1e-10)

    return PSRResult(
        psr=psr,
        stability_cv=stability_cv,
        sharpes_by_period=sharpes,
        is_stable=psr >= 0.8 and stability_cv < 0.5
    )


def comprehensive_overfit_assessment(
    is_returns: np.ndarray,
    oos_returns: np.ndarray,
    forward_returns: Optional[np.ndarray] = None,
    trials: int = 256,
    returns_by_year: Optional[Dict[str, np.ndarray]] = None
) -> Dict:
    """
    Run comprehensive overfitting assessment combining all tests.

    Args:
        is_returns: In-sample returns
        oos_returns: Out-of-sample returns
        forward_returns: Forward test returns (optional)
        trials: Number of parameter combinations tested
        returns_by_year: Dict of returns by year for stability analysis

    Returns:
        Dict with all test results and overall assessment
    """
    results = {}

    # 1. Deflated Sharpe Ratio on OOS
    logger.info("Running Deflated Sharpe Ratio test...")
    results['dsr'] = deflated_sharpe_ratio(oos_returns, trials=trials)

    # 2. CSCV on combined IS+OOS
    logger.info("Running CSCV overfit probability test...")
    combined_returns = np.concatenate([is_returns, oos_returns])
    results['cscv'] = cscv_overfit_probability(combined_returns)

    # 3. Performance Stability
    if returns_by_year:
        logger.info("Running Performance Stability test...")
        results['psr'] = performance_stability_ratio(returns_by_year)

    # 4. IS-OOS Gap Analysis
    is_sharpe = (np.mean(is_returns) / np.std(is_returns, ddof=1)) * np.sqrt(252)
    oos_sharpe = (np.mean(oos_returns) / np.std(oos_returns, ddof=1)) * np.sqrt(252)
    gap_pct = (is_sharpe - oos_sharpe) / is_sharpe * 100 if is_sharpe != 0 else 0

    results['sharpe_gap'] = {
        'is_sharpe': is_sharpe,
        'oos_sharpe': oos_sharpe,
        'gap_percent': gap_pct,
        'is_acceptable': gap_pct < 40  # <40% degradation is acceptable
    }

    # 5. Forward test consistency (if provided)
    if forward_returns is not None and len(forward_returns) > 0:
        fwd_sharpe = (np.mean(forward_returns) / np.std(forward_returns, ddof=1)) * np.sqrt(252)
        fwd_gap = (oos_sharpe - fwd_sharpe) / oos_sharpe * 100 if oos_sharpe != 0 else 0
        results['forward_test'] = {
            'forward_sharpe': fwd_sharpe,
            'oos_to_forward_gap': fwd_gap,
            'is_consistent': abs(fwd_gap) < 30  # <30% deviation from OOS
        }

    # Overall Assessment
    robust_count = 0
    total_tests = 0

    if 'dsr' in results:
        total_tests += 1
        if results['dsr'].is_significant:
            robust_count += 1

    if 'cscv' in results:
        total_tests += 1
        if not results['cscv'].is_likely_overfit:
            robust_count += 1

    if 'psr' in results:
        total_tests += 1
        if results['psr'].is_stable:
            robust_count += 1

    if 'sharpe_gap' in results:
        total_tests += 1
        if results['sharpe_gap']['is_acceptable']:
            robust_count += 1

    if 'forward_test' in results:
        total_tests += 1
        if results['forward_test']['is_consistent']:
            robust_count += 1

    results['overall'] = {
        'robust_tests': robust_count,
        'total_tests': total_tests,
        'robustness_score': robust_count / total_tests if total_tests > 0 else 0,
        'assessment': 'ROBUST' if robust_count >= total_tests * 0.8 else
                      'ACCEPTABLE' if robust_count >= total_tests * 0.6 else 'CONCERN'
    }

    return results


def print_assessment_report(results: Dict) -> str:
    """Format assessment results as readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("OVERFITTING DETECTION REPORT")
    lines.append("=" * 60)

    # DSR Results
    if 'dsr' in results:
        dsr = results['dsr']
        lines.append("\n1. DEFLATED SHARPE RATIO (DSR)")
        lines.append(f"   Observed Sharpe:   {dsr.observed_sharpe:.3f}")
        lines.append(f"   Deflated Sharpe:   {dsr.deflated_sharpe:.3f}")
        lines.append(f"   P-value:           {dsr.p_value:.4f}")
        lines.append(f"   Trials tested:     {dsr.trials}")
        lines.append(f"   Significant:       {'YES' if dsr.is_significant else 'NO'}")

    # CSCV Results
    if 'cscv' in results:
        cscv = results['cscv']
        lines.append("\n2. COMBINATORIAL SYMMETRIC CV (CSCV)")
        lines.append(f"   Overfit Probability: {cscv.overfit_probability:.1%}")
        lines.append(f"   Combinations tested: {cscv.n_combinations}")
        lines.append(f"   Assessment:          {cscv.interpretation}")

    # PSR Results
    if 'psr' in results:
        psr = results['psr']
        lines.append("\n3. PERFORMANCE STABILITY RATIO (PSR)")
        lines.append(f"   Periods profitable: {psr.psr:.1%}")
        lines.append(f"   Stability (CV):     {psr.stability_cv:.2f}")
        lines.append(f"   Is Stable:          {'YES' if psr.is_stable else 'NO'}")
        if psr.sharpes_by_period:
            lines.append("   Sharpe by period:")
            for period, sharpe in psr.sharpes_by_period.items():
                lines.append(f"     {period}: {sharpe:.2f}")

    # Sharpe Gap
    if 'sharpe_gap' in results:
        gap = results['sharpe_gap']
        lines.append("\n4. IS-OOS SHARPE GAP")
        lines.append(f"   In-Sample Sharpe:  {gap['is_sharpe']:.2f}")
        lines.append(f"   OOS Sharpe:        {gap['oos_sharpe']:.2f}")
        lines.append(f"   Gap:               {gap['gap_percent']:.1f}%")
        lines.append(f"   Acceptable:        {'YES' if gap['is_acceptable'] else 'NO'}")

    # Forward Test
    if 'forward_test' in results:
        fwd = results['forward_test']
        lines.append("\n5. FORWARD TEST CONSISTENCY")
        lines.append(f"   Forward Sharpe:    {fwd['forward_sharpe']:.2f}")
        lines.append(f"   OOS-Forward Gap:   {fwd['oos_to_forward_gap']:.1f}%")
        lines.append(f"   Consistent:        {'YES' if fwd['is_consistent'] else 'NO'}")

    # Overall
    if 'overall' in results:
        overall = results['overall']
        lines.append("\n" + "=" * 60)
        lines.append("OVERALL ASSESSMENT")
        lines.append("=" * 60)
        lines.append(f"   Tests Passed:      {overall['robust_tests']}/{overall['total_tests']}")
        lines.append(f"   Robustness Score:  {overall['robustness_score']:.1%}")
        lines.append(f"   Assessment:        {overall['assessment']}")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo with synthetic data
    print("Overfitting Detection Framework - Demo")
    print("=" * 50)

    # Simulate returns
    np.random.seed(42)
    is_returns = np.random.normal(0.001, 0.02, 500)  # Slight positive edge
    oos_returns = np.random.normal(0.0008, 0.02, 300)  # Slight degradation
    fwd_returns = np.random.normal(0.0007, 0.02, 200)

    returns_by_year = {
        '2020': np.random.normal(0.0009, 0.02, 250),
        '2021': np.random.normal(0.0008, 0.02, 250),
        '2022': np.random.normal(0.0007, 0.02, 250),
        '2023': np.random.normal(0.0010, 0.02, 250),
        '2024': np.random.normal(0.0008, 0.02, 250),
    }

    results = comprehensive_overfit_assessment(
        is_returns=is_returns,
        oos_returns=oos_returns,
        forward_returns=fwd_returns,
        trials=256,
        returns_by_year=returns_by_year
    )

    print(print_assessment_report(results))
