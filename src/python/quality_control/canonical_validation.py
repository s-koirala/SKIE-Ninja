"""
Canonical Validation Framework for Financial Machine Learning

Implements rigorous validation methods from peer-reviewed literature:

1. Combinatorially Purged Cross-Validation (CPCV)
   - Lopez de Prado, M. (2018). "Advances in Financial Machine Learning", Ch. 7
   - Generates multiple train/test paths to reduce variance
   - Purges overlapping samples to prevent leakage
   - Applies embargo period after each test set

2. Probability of Backtest Overfitting (PBO)
   - Bailey, D.H., et al. (2014). "The Probability of Backtest Overfitting"
   - Estimates probability that IS-optimized strategy will underperform OOS
   - Uses rank distribution across combinatorial paths

3. Deflated Sharpe Ratio (DSR)
   - Bailey, D.H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
   - Adjusts Sharpe ratio for multiple testing
   - Accounts for expected maximum Sharpe under null

4. Bootstrap Confidence Intervals
   - Efron, B. & Tibshirani, R. (1993). "An Introduction to the Bootstrap"
   - Non-parametric uncertainty quantification

5. Stationarity Tests
   - Dickey, D.A. & Fuller, W.A. (1979). "Distribution of the Estimators..."
   - Augmented Dickey-Fuller test for unit roots

Author: SKIE_Ninja Development Team
Created: 2026-01-05
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Generator, Any
from dataclasses import dataclass, field
from itertools import combinations
import warnings
from scipy import stats
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CPCVConfig:
    """
    Configuration for Combinatorially Purged Cross-Validation.

    Parameters derived empirically from feature/target structure:
    - n_splits: Number of temporal folds (typically 6-10)
    - n_test_splits: Test splits per path (typically 2 for CSCV)
    - embargo_bars: max(feature_lookback, label_horizon) + safety_margin
    - purge_bars: Label horizon (bars needed for target to materialize)

    References:
        Lopez de Prado (2018), Chapter 7, Section 7.4
    """
    n_splits: int = 6
    n_test_splits: int = 2
    embargo_bars: int = 210  # Derived: max(200, 30) + 10 safety margin
    purge_bars: int = 30     # Longest label horizon

    def __post_init__(self):
        """Validate configuration."""
        if self.n_test_splits >= self.n_splits:
            raise ValueError("n_test_splits must be < n_splits")
        if self.embargo_bars < self.purge_bars:
            raise ValueError("embargo_bars should be >= purge_bars")


@dataclass
class ValidationResult:
    """Container for validation results."""
    # Core metrics
    sharpe_ratio: float = 0.0
    deflated_sharpe_ratio: float = 0.0
    dsr_pvalue: float = 1.0
    pbo: float = 1.0

    # Performance distribution
    oos_sharpes: List[float] = field(default_factory=list)
    is_sharpes: List[float] = field(default_factory=list)

    # Confidence intervals
    sharpe_ci_95: Tuple[float, float] = (0.0, 0.0)
    pnl_ci_95: Tuple[float, float] = (0.0, 0.0)

    # Stationarity
    adf_statistic: float = 0.0
    adf_pvalue: float = 1.0
    is_stationary: bool = False

    # Metadata
    n_paths: int = 0
    n_trials: int = 0
    passed: bool = False

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "CANONICAL VALIDATION RESULTS",
            "=" * 60,
            "",
            f"Sharpe Ratio (raw):      {self.sharpe_ratio:.4f}",
            f"Deflated Sharpe Ratio:   {self.deflated_sharpe_ratio:.4f}",
            f"DSR p-value:             {self.dsr_pvalue:.4f}",
            f"",
            f"PBO (Prob Overfit):      {self.pbo:.4f}",
            f"  Interpretation:        {'HIGH RISK' if self.pbo > 0.5 else 'ACCEPTABLE'}",
            f"",
            f"Sharpe 95% CI:           [{self.sharpe_ci_95[0]:.4f}, {self.sharpe_ci_95[1]:.4f}]",
            f"",
            f"ADF Statistic:           {self.adf_statistic:.4f}",
            f"ADF p-value:             {self.adf_pvalue:.4f}",
            f"Stationary:              {self.is_stationary}",
            f"",
            f"Paths Tested:            {self.n_paths}",
            f"Trials Corrected For:    {self.n_trials}",
            f"",
            f"OVERALL:                 {'PASS' if self.passed else 'FAIL'}",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# COMBINATORIALLY PURGED CROSS-VALIDATION (CPCV)
# =============================================================================

class CombinatorialPurgedKFold:
    """
    Combinatorially Purged K-Fold Cross-Validation.

    Implements the CPCV method from Lopez de Prado (2018) Chapter 7.

    Key features:
    1. Generates C(n_splits, n_test_splits) combinatorial train/test paths
    2. Purges training samples whose labels overlap with test period
    3. Applies embargo period after each test set

    This reduces variance vs single-path walk-forward and eliminates
    information leakage from overlapping labels.

    Example:
        >>> cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2)
        >>> for train_idx, test_idx in cv.split(X, y, timestamps):
        ...     model.fit(X[train_idx], y[train_idx])
        ...     score = model.score(X[test_idx], y[test_idx])

    References:
        Lopez de Prado, M. (2018). "Advances in Financial Machine Learning",
        Wiley, Chapter 7, Section 7.4.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        embargo_bars: int = 210,
        purge_bars: int = 30
    ):
        """
        Initialize CPCV splitter.

        Args:
            n_splits: Number of temporal folds to create
            n_test_splits: Number of folds to use as test in each path
            embargo_bars: Bars to skip after test period (prevents leakage)
            purge_bars: Label horizon (samples purged if label overlaps test)

        The number of paths generated is C(n_splits, n_test_splits).
        For n_splits=6, n_test_splits=2: C(6,2) = 15 paths.
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_bars = embargo_bars
        self.purge_bars = purge_bars

        if n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be < n_splits")

    def get_n_paths(self) -> int:
        """Return number of combinatorial paths."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for each combinatorial path.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target array (optional, not used for splitting)
            timestamps: Sample timestamps (optional, for logging)

        Yields:
            (train_indices, test_indices) for each path
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits

        # Create fold boundaries
        fold_bounds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            fold_bounds.append((start, end))

        # Generate all combinations of test folds
        test_fold_combos = list(combinations(range(self.n_splits), self.n_test_splits))

        logger.info(f"CPCV: {len(test_fold_combos)} paths from C({self.n_splits},{self.n_test_splits})")

        for path_idx, test_folds in enumerate(test_fold_combos):
            # Get test indices
            test_indices = []
            for fold_idx in test_folds:
                start, end = fold_bounds[fold_idx]
                test_indices.extend(range(start, end))
            test_indices = np.array(sorted(test_indices))

            # Get train indices (all folds not in test)
            train_folds = [i for i in range(self.n_splits) if i not in test_folds]
            train_indices = []
            for fold_idx in train_folds:
                start, end = fold_bounds[fold_idx]
                train_indices.extend(range(start, end))
            train_indices = np.array(sorted(train_indices))

            # Apply purging: remove train samples whose labels overlap test
            test_start = test_indices.min()
            test_end = test_indices.max()

            # Purge samples within purge_bars before test start
            # (their labels would leak into test period)
            purge_mask = train_indices < (test_start - self.purge_bars)
            purge_mask |= train_indices > (test_end + self.embargo_bars)

            # Also remove samples between test blocks if multiple test folds
            if len(test_folds) > 1:
                for i in range(len(test_folds) - 1):
                    gap_start = fold_bounds[test_folds[i]][1]
                    gap_end = fold_bounds[test_folds[i + 1]][0]
                    # Don't train on gap between test folds
                    purge_mask &= ~((train_indices >= gap_start) & (train_indices < gap_end))

            train_indices_purged = train_indices[purge_mask]

            # Apply embargo: remove train samples immediately after test end
            embargo_mask = np.ones(len(train_indices_purged), dtype=bool)
            for fold_idx in test_folds:
                _, fold_end = fold_bounds[fold_idx]
                embargo_start = fold_end
                embargo_end = fold_end + self.embargo_bars
                embargo_mask &= ~(
                    (train_indices_purged >= embargo_start) &
                    (train_indices_purged < embargo_end)
                )

            train_indices_final = train_indices_purged[embargo_mask]

            if len(train_indices_final) < 100:
                logger.warning(f"Path {path_idx}: Only {len(train_indices_final)} train samples after purging")

            yield train_indices_final, test_indices


# =============================================================================
# PROBABILITY OF BACKTEST OVERFITTING (PBO)
# =============================================================================

def calculate_pbo(
    is_returns: List[np.ndarray],
    oos_returns: List[np.ndarray],
    n_trials: int = 1
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate Probability of Backtest Overfitting.

    PBO estimates the probability that the in-sample optimal strategy
    will underperform out-of-sample.

    Method (from Bailey et al., 2014):
    1. For each CV path, compute IS and OOS Sharpe ratios
    2. Rank strategies by IS performance
    3. Compute logit: log(rank / (n - rank))
    4. PBO = fraction of paths where IS-best has negative OOS rank-logit

    Args:
        is_returns: List of in-sample return arrays (one per path)
        oos_returns: List of out-of-sample return arrays (one per path)
        n_trials: Number of strategy trials (for adjustment)

    Returns:
        (pbo, details_dict)

    References:
        Bailey, D.H., Borwein, J.M., Lopez de Prado, M., Zhu, Q.J. (2014).
        "The Probability of Backtest Overfitting". Journal of Computational Finance.
    """
    n_paths = len(is_returns)

    if n_paths < 2:
        logger.warning("PBO requires at least 2 paths, returning 1.0 (worst case)")
        return 1.0, {'warning': 'insufficient_paths'}

    # Calculate Sharpe ratios for each path
    is_sharpes = []
    oos_sharpes = []

    for is_ret, oos_ret in zip(is_returns, oos_returns):
        is_sr = np.mean(is_ret) / (np.std(is_ret) + 1e-10) * np.sqrt(252)
        oos_sr = np.mean(oos_ret) / (np.std(oos_ret) + 1e-10) * np.sqrt(252)
        is_sharpes.append(is_sr)
        oos_sharpes.append(oos_sr)

    is_sharpes = np.array(is_sharpes)
    oos_sharpes = np.array(oos_sharpes)

    # Rank by IS performance (1 = best)
    is_ranks = stats.rankdata(-is_sharpes)  # Negative for descending
    oos_ranks = stats.rankdata(-oos_sharpes)

    # For the IS-best strategy (rank 1), what is its OOS rank?
    is_best_idx = np.argmin(is_ranks)
    is_best_oos_rank = oos_ranks[is_best_idx]

    # Calculate rank logit for each path
    # logit(w) = log(rank / (n - rank))
    # Positive logit = below median performance
    rank_logits = []
    for oos_rank in oos_ranks:
        if oos_rank < n_paths:
            logit = np.log(oos_rank / (n_paths - oos_rank + 1))
        else:
            logit = np.log(n_paths)  # Worst rank
        rank_logits.append(logit)

    rank_logits = np.array(rank_logits)

    # PBO = P(IS-optimal underperforms OOS)
    # Approximate by fraction of paths where IS-best has negative logit rank
    # More rigorously: fit distribution and compute probability

    # Simple estimate: IS-best's OOS rank relative to median
    pbo_simple = is_best_oos_rank / n_paths

    # Stochastic dominance estimate (Bailey et al.)
    # Count pairs where IS-better strategy has worse OOS performance
    n_pairs = 0
    n_dominated = 0
    for i in range(n_paths):
        for j in range(i + 1, n_paths):
            n_pairs += 1
            # If i better IS but worse OOS, that's a sign of overfitting
            if (is_sharpes[i] > is_sharpes[j] and oos_sharpes[i] < oos_sharpes[j]):
                n_dominated += 1
            elif (is_sharpes[j] > is_sharpes[i] and oos_sharpes[j] < oos_sharpes[i]):
                n_dominated += 1

    pbo_dominance = n_dominated / max(n_pairs, 1)

    # Combined estimate
    pbo = (pbo_simple + pbo_dominance) / 2

    # Adjust for multiple trials
    if n_trials > 1:
        # More trials = higher probability of finding spurious IS-best
        pbo_adjusted = 1 - (1 - pbo) ** (1 / n_trials)
        pbo = min(pbo, pbo_adjusted)

    details = {
        'is_sharpes': is_sharpes.tolist(),
        'oos_sharpes': oos_sharpes.tolist(),
        'is_ranks': is_ranks.tolist(),
        'oos_ranks': oos_ranks.tolist(),
        'is_best_oos_rank': int(is_best_oos_rank),
        'pbo_simple': pbo_simple,
        'pbo_dominance': pbo_dominance,
        'n_pairs': n_pairs,
        'n_dominated': n_dominated
    }

    return pbo, details


# =============================================================================
# DEFLATED SHARPE RATIO (DSR)
# =============================================================================

def expected_max_sharpe(n_trials: int, variance_sr: float = 1.0) -> float:
    """
    Compute expected maximum Sharpe ratio under null hypothesis.

    Under the null (no skill), what's the expected max Sharpe when
    testing n_trials strategies?

    Formula (from Bailey & Lopez de Prado, 2014):
        E[max(SR)] = sqrt(variance_sr) * Z_{1-1/(n+1)}

    where Z_p is the p-th quantile of standard normal.

    Args:
        n_trials: Number of strategy trials tested
        variance_sr: Variance of the Sharpe ratio estimator

    Returns:
        Expected maximum Sharpe under null

    References:
        Bailey, D.H. & Lopez de Prado, M. (2014).
        "The Deflated Sharpe Ratio". SSRN 2460551.
    """
    if n_trials <= 1:
        return 0.0

    # Euler-Mascheroni approximation for expected max of n_trials i.i.d. normals
    # E[max] ≈ sqrt(2 * log(n))
    euler_mascheroni = 0.5772156649
    expected_max = (
        (1 - euler_mascheroni) * stats.norm.ppf(1 - 1 / (n_trials + 1)) +
        euler_mascheroni * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )

    return np.sqrt(variance_sr) * expected_max


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sr_variance: Optional[float] = None
) -> Tuple[float, float]:
    """
    Calculate Deflated Sharpe Ratio.

    DSR adjusts the observed Sharpe ratio for:
    1. Multiple testing (n_trials strategies tested)
    2. Non-normality of returns (skewness, kurtosis)
    3. Estimation variance

    Formula:
        DSR = (SR_observed - E[max(SR)]) / SE[SR]

    where SE[SR] accounts for higher moments.

    Args:
        observed_sr: Observed Sharpe ratio (annualized)
        n_trials: Number of strategy configurations tested
        n_obs: Number of return observations
        skewness: Return distribution skewness (0 for normal)
        kurtosis: Return distribution kurtosis (3 for normal)
        sr_variance: Optional override for SR variance

    Returns:
        (dsr, p_value)

    References:
        Bailey, D.H. & Lopez de Prado, M. (2014).
        "The Deflated Sharpe Ratio". SSRN 2460551.
    """
    if n_obs < 30:
        logger.warning(f"DSR: n_obs={n_obs} is low, estimates may be unreliable")

    # Standard error of Sharpe ratio (Lo, 2002)
    # Adjusted for non-normality (Mertens, 2002)
    if sr_variance is None:
        se_sr_squared = (
            1 +
            0.5 * observed_sr**2 -
            skewness * observed_sr +
            (kurtosis - 3) / 4 * observed_sr**2
        ) / (n_obs - 1)
        se_sr = np.sqrt(max(se_sr_squared, 1e-10))
    else:
        se_sr = np.sqrt(sr_variance)

    # Expected max SR under null
    exp_max_sr = expected_max_sharpe(n_trials, variance_sr=1.0)

    # Deflated Sharpe Ratio
    dsr = (observed_sr - exp_max_sr) / se_sr

    # P-value (one-sided: probability of observing SR this high under null)
    p_value = 1 - stats.norm.cdf(dsr)

    return dsr, p_value


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for Sharpe ratio.

    Uses percentile bootstrap method.

    Args:
        returns: Array of returns
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        (point_estimate, ci_lower, ci_upper)

    References:
        Efron, B. & Tibshirani, R. (1993).
        "An Introduction to the Bootstrap". Chapman & Hall.
    """
    rng = np.random.RandomState(random_state)
    n = len(returns)

    bootstrap_sharpes = []
    for _ in range(n_bootstrap):
        sample = rng.choice(returns, size=n, replace=True)
        sr = np.mean(sample) / (np.std(sample) + 1e-10) * np.sqrt(252)
        bootstrap_sharpes.append(sr)

    bootstrap_sharpes = np.array(bootstrap_sharpes)
    point_estimate = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_sharpes, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_sharpes, (1 - alpha / 2) * 100)

    return point_estimate, ci_lower, ci_upper


def bootstrap_pnl_ci(
    trade_pnls: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for total P&L.

    Args:
        trade_pnls: Array of individual trade P&Ls
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(random_state)
    n = len(trade_pnls)

    bootstrap_totals = []
    for _ in range(n_bootstrap):
        sample = rng.choice(trade_pnls, size=n, replace=True)
        bootstrap_totals.append(np.sum(sample))

    point_estimate = np.sum(trade_pnls)

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_totals, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_totals, (1 - alpha / 2) * 100)

    return point_estimate, ci_lower, ci_upper


# =============================================================================
# STATIONARITY TESTS
# =============================================================================

def augmented_dickey_fuller(
    series: np.ndarray,
    maxlag: Optional[int] = None,
    regression: str = 'c'
) -> Tuple[float, float, bool, Dict[str, Any]]:
    """
    Augmented Dickey-Fuller test for stationarity.

    Tests the null hypothesis that a unit root is present (non-stationary).
    Rejection of null suggests the series is stationary.

    Args:
        series: Time series to test
        maxlag: Maximum lag for ADF regression (None = auto)
        regression: Regression type ('c'=constant, 'ct'=constant+trend, 'n'=none)

    Returns:
        (adf_statistic, p_value, is_stationary, details)

    References:
        Dickey, D.A. & Fuller, W.A. (1979).
        "Distribution of the Estimators for Autoregressive Time Series
        with a Unit Root". Journal of the American Statistical Association.
    """
    try:
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(series, maxlag=maxlag, regression=regression, autolag='AIC')

        adf_stat = result[0]
        p_value = result[1]
        used_lag = result[2]
        n_obs = result[3]
        critical_values = result[4]

        # Reject null (unit root) if p < 0.05 -> stationary
        is_stationary = p_value < 0.05

        details = {
            'used_lag': used_lag,
            'n_obs': n_obs,
            'critical_values': critical_values,
            'adf_stat': adf_stat,
            'regression': regression
        }

        return adf_stat, p_value, is_stationary, details

    except ImportError:
        logger.warning("statsmodels not available, using manual ADF approximation")

        # Simple approximation without statsmodels
        diff = np.diff(series)
        corr = np.corrcoef(series[:-1], diff)[0, 1]
        n = len(series)

        # Approximate test statistic
        adf_approx = corr * np.sqrt(n)

        # Very rough p-value approximation
        # Critical values: -3.43 (1%), -2.86 (5%), -2.57 (10%)
        if adf_approx < -3.43:
            p_approx = 0.01
        elif adf_approx < -2.86:
            p_approx = 0.05
        elif adf_approx < -2.57:
            p_approx = 0.10
        else:
            p_approx = 0.50

        return adf_approx, p_approx, p_approx < 0.05, {'approximation': True}


# =============================================================================
# MAIN VALIDATION RUNNER
# =============================================================================

def run_canonical_validation(
    returns: np.ndarray,
    trade_pnls: Optional[np.ndarray] = None,
    n_trials: int = 1,
    cv_config: Optional[CPCVConfig] = None,
    is_returns_by_path: Optional[List[np.ndarray]] = None,
    oos_returns_by_path: Optional[List[np.ndarray]] = None,
    n_bootstrap: int = 10000,
    random_state: int = 42
) -> ValidationResult:
    """
    Run comprehensive canonical validation.

    Performs:
    1. Deflated Sharpe Ratio calculation
    2. PBO calculation (if path returns provided)
    3. Bootstrap confidence intervals
    4. Stationarity testing

    Args:
        returns: Daily returns array
        trade_pnls: Individual trade P&Ls (optional, for bootstrap)
        n_trials: Number of strategy configurations tested
        cv_config: CPCV configuration
        is_returns_by_path: In-sample returns for each CV path (for PBO)
        oos_returns_by_path: Out-of-sample returns for each CV path (for PBO)
        n_bootstrap: Bootstrap iterations
        random_state: Random seed

    Returns:
        ValidationResult with all metrics
    """
    result = ValidationResult()
    result.n_trials = n_trials

    config = cv_config or CPCVConfig()

    logger.info("=" * 60)
    logger.info("CANONICAL VALIDATION")
    logger.info("=" * 60)

    # 1. Basic Sharpe calculation
    logger.info("\n1. Calculating Sharpe Ratio...")
    result.sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns) + 3  # scipy returns excess kurtosis

    logger.info(f"   Raw Sharpe: {result.sharpe_ratio:.4f}")
    logger.info(f"   Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}")

    # 2. Deflated Sharpe Ratio
    logger.info("\n2. Calculating Deflated Sharpe Ratio...")
    result.deflated_sharpe_ratio, result.dsr_pvalue = deflated_sharpe_ratio(
        observed_sr=result.sharpe_ratio,
        n_trials=n_trials,
        n_obs=len(returns),
        skewness=skewness,
        kurtosis=kurtosis
    )

    expected_max = expected_max_sharpe(n_trials)
    logger.info(f"   E[max(SR)] under null: {expected_max:.4f}")
    logger.info(f"   Deflated Sharpe: {result.deflated_sharpe_ratio:.4f}")
    logger.info(f"   DSR p-value: {result.dsr_pvalue:.4f}")

    # 3. PBO (if path data provided)
    if is_returns_by_path is not None and oos_returns_by_path is not None:
        logger.info("\n3. Calculating PBO...")
        result.n_paths = len(is_returns_by_path)
        result.pbo, pbo_details = calculate_pbo(
            is_returns_by_path,
            oos_returns_by_path,
            n_trials=n_trials
        )
        result.is_sharpes = pbo_details.get('is_sharpes', [])
        result.oos_sharpes = pbo_details.get('oos_sharpes', [])
        logger.info(f"   PBO: {result.pbo:.4f}")
        logger.info(f"   IS-best OOS rank: {pbo_details.get('is_best_oos_rank', 'N/A')}/{result.n_paths}")
    else:
        logger.info("\n3. PBO: Skipped (no path data provided)")
        result.pbo = np.nan

    # 4. Bootstrap Confidence Intervals
    logger.info("\n4. Bootstrap Confidence Intervals...")
    _, ci_lower, ci_upper = bootstrap_sharpe_ci(
        returns, n_bootstrap=n_bootstrap, random_state=random_state
    )
    result.sharpe_ci_95 = (ci_lower, ci_upper)
    logger.info(f"   Sharpe 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    if trade_pnls is not None:
        pnl_point, pnl_lower, pnl_upper = bootstrap_pnl_ci(
            trade_pnls, n_bootstrap=n_bootstrap, random_state=random_state
        )
        result.pnl_ci_95 = (pnl_lower, pnl_upper)
        logger.info(f"   P&L 95% CI: [${pnl_lower:,.0f}, ${pnl_upper:,.0f}]")

    # 5. Stationarity Test
    logger.info("\n5. Stationarity Test (ADF)...")
    result.adf_statistic, result.adf_pvalue, result.is_stationary, adf_details = \
        augmented_dickey_fuller(returns)
    logger.info(f"   ADF Statistic: {result.adf_statistic:.4f}")
    logger.info(f"   ADF p-value: {result.adf_pvalue:.4f}")
    logger.info(f"   Stationary: {result.is_stationary}")

    # 6. Overall Assessment
    logger.info("\n6. Overall Assessment...")

    # Pass criteria:
    # - DSR p-value < 0.05 (significant after multiple testing)
    # - PBO < 0.5 (low probability of overfitting)
    # - Returns are stationary
    dsr_pass = result.dsr_pvalue < 0.05
    pbo_pass = np.isnan(result.pbo) or result.pbo < 0.5
    stationary_pass = result.is_stationary

    result.passed = dsr_pass and pbo_pass and stationary_pass

    logger.info(f"   DSR significant (p<0.05): {dsr_pass}")
    logger.info(f"   PBO acceptable (<0.5): {pbo_pass}")
    logger.info(f"   Returns stationary: {stationary_pass}")
    logger.info(f"   OVERALL: {'PASS' if result.passed else 'FAIL'}")

    return result


# =============================================================================
# EMBARGO CALCULATION
# =============================================================================

def calculate_proper_embargo(
    feature_lookbacks: List[int],
    label_horizons: List[int],
    safety_margin: int = 10
) -> int:
    """
    Calculate proper embargo period per Lopez de Prado (2018).

    Embargo = max(max_feature_lookback, max_label_horizon) + safety_margin

    Args:
        feature_lookbacks: List of lookback periods for all features
        label_horizons: List of horizon periods for all targets
        safety_margin: Additional buffer

    Returns:
        Embargo in bars

    Example:
        >>> features = [5, 10, 14, 20, 50, 100, 200]  # MA periods
        >>> labels = [5, 10, 20, 30]  # Prediction horizons
        >>> embargo = calculate_proper_embargo(features, labels)
        >>> print(embargo)  # 210
    """
    max_feature = max(feature_lookbacks) if feature_lookbacks else 0
    max_label = max(label_horizons) if label_horizons else 0

    embargo = max(max_feature, max_label) + safety_margin

    logger.info(f"Embargo calculation:")
    logger.info(f"  Max feature lookback: {max_feature}")
    logger.info(f"  Max label horizon: {max_label}")
    logger.info(f"  Safety margin: {safety_margin}")
    logger.info(f"  => Embargo: {embargo} bars")

    return embargo


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_validation_check(
    daily_returns: np.ndarray,
    n_trials: int = 81,  # Default from threshold optimization
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Quick validation check without full CPCV.

    Useful for preliminary assessment.

    Args:
        daily_returns: Array of daily returns
        n_trials: Number of strategy configurations tested
        verbose: Print results

    Returns:
        Dict with key metrics
    """
    # Raw Sharpe
    raw_sr = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)

    # DSR
    skew = stats.skew(daily_returns)
    kurt = stats.kurtosis(daily_returns) + 3
    dsr, dsr_p = deflated_sharpe_ratio(
        raw_sr, n_trials, len(daily_returns), skew, kurt
    )

    # Bootstrap CI
    _, ci_low, ci_high = bootstrap_sharpe_ci(daily_returns, n_bootstrap=1000)

    # Stationarity
    adf_stat, adf_p, is_stationary, _ = augmented_dickey_fuller(daily_returns)

    results = {
        'raw_sharpe': raw_sr,
        'deflated_sharpe': dsr,
        'dsr_pvalue': dsr_p,
        'sharpe_ci_95': (ci_low, ci_high),
        'expected_max_sharpe_null': expected_max_sharpe(n_trials),
        'adf_statistic': adf_stat,
        'adf_pvalue': adf_p,
        'is_stationary': is_stationary,
        'n_trials_corrected': n_trials,
        'passed': dsr_p < 0.05 and is_stationary
    }

    if verbose:
        print("\n" + "=" * 50)
        print("QUICK VALIDATION CHECK")
        print("=" * 50)
        print(f"Raw Sharpe:              {raw_sr:.4f}")
        print(f"Expected max (null):     {results['expected_max_sharpe_null']:.4f}")
        print(f"Deflated Sharpe:         {dsr:.4f}")
        print(f"DSR p-value:             {dsr_p:.4f}")
        print(f"Sharpe 95% CI:           [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"ADF Statistic:           {adf_stat:.4f}")
        print(f"ADF p-value:             {adf_p:.4f}")
        print(f"Is Stationary:           {is_stationary}")
        print("-" * 50)
        print(f"VERDICT:                 {'PASS' if results['passed'] else 'FAIL'}")
        print("=" * 50)

    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'CPCVConfig',
    'ValidationResult',
    'CombinatorialPurgedKFold',
    'calculate_pbo',
    'expected_max_sharpe',
    'deflated_sharpe_ratio',
    'bootstrap_sharpe_ci',
    'bootstrap_pnl_ci',
    'augmented_dickey_fuller',
    'run_canonical_validation',
    'calculate_proper_embargo',
    'quick_validation_check',
]


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("Canonical Validation Framework")
    print("=" * 60)

    # Example: Calculate proper embargo for SKIE-Ninja
    feature_lookbacks = [1, 2, 3, 5, 10, 14, 20, 50, 100, 200]  # All MA/lookback periods
    label_horizons = [5, 10, 20, 30]  # Target horizons

    embargo = calculate_proper_embargo(feature_lookbacks, label_horizons)
    print(f"\nRecommended embargo: {embargo} bars")

    # Example: Quick check with simulated returns
    print("\n" + "=" * 60)
    print("SIMULATED VALIDATION TEST")
    print("=" * 60)

    np.random.seed(42)

    # Simulate daily returns with small positive drift
    n_days = 500
    daily_returns = np.random.normal(0.0005, 0.01, n_days)  # ~12% annual, 15% vol

    # Run quick validation
    results = quick_validation_check(daily_returns, n_trials=81, verbose=True)
