"""
Combinatorial Purged Cross-Validation (CPCV), PBO, and DSR
==========================================================

Implements canonical validation methods from:
1. Lopez de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 7
2. Bailey, D.H., et al. (2014). "The Probability of Backtest Overfitting"
3. Bailey, D.H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"

CPCV generates all possible train/test combinations with proper purging and embargo
to prevent data leakage in time-series financial data.

PBO estimates the probability that a selected backtest strategy would underperform
out-of-sample, detecting overfitting through combinatorial analysis.

DSR adjusts the observed Sharpe ratio for multiple testing bias.

Author: SKIE_Ninja Development Team
Created: 2026-01-06
Updated: 2026-01-06 (Canonical fixes per CPCV_PBO_IMPLEMENTATION_AUDIT_20260106.md)

References:
    - Lopez de Prado (2018) Ch. 7: https://doi.org/10.1002/9781119482086.ch7
    - Bailey et al. (2014): SSRN 2326253
    - Bailey & Lopez de Prado (2014): SSRN 2460551

IMPLEMENTATION NOTES:
    - PBO uses Monte Carlo approximation (not exhaustive CSCV) due to computational
      constraints. For T=500 periods, exhaustive C(500,250) ~ 10^149 is infeasible.
      Monte Carlo with n_trials >= 1000 provides adequate approximation.
    - CPCV supports both index-based purging (fallback) and canonical t1-based
      purging when label end times are provided.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Callable, Union
from itertools import combinations
from dataclasses import dataclass, field
from scipy import stats
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

logger = logging.getLogger(__name__)

# Euler-Mascheroni constant for DSR calculation
EULER_MASCHERONI = 0.5772156649015329


@dataclass
class CPCVConfig:
    """Configuration for CPCV.

    Parameters derived from Lopez de Prado (2018) Chapter 7.

    Attributes:
        n_splits: Number of time-series splits (N in paper)
        n_test_splits: Test splits per combination (k in paper)
        purge_pct: Percentage of train to purge before test (fallback if t1 not provided)
        embargo_pct: Percentage to embargo after test
        min_train_size: Minimum training samples required
    """
    n_splits: int = 6
    n_test_splits: int = 2
    purge_pct: float = 0.01
    embargo_pct: float = 0.01
    min_train_size: int = 100


@dataclass
class CPCVResult:
    """Result from a single CPCV fold.

    Attributes:
        train_indices: Array of training sample indices
        test_indices: Array of test sample indices
        fold_id: Fold number
        combination: Tuple of test group indices
        sample_weights: Optional weights for training samples (for uneven representation)
    """
    train_indices: np.ndarray
    test_indices: np.ndarray
    fold_id: int
    combination: Tuple[int, ...]
    sample_weights: Optional[np.ndarray] = None


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged K-Fold Cross-Validation.

    Implements the CPCV method from Lopez de Prado (2018) Chapter 7.

    Key features:
    1. Generates all C(N,k) combinations of test groups
    2. Purges training samples whose labels overlap with test set (requires t1)
    3. Embargoes samples immediately after test to prevent leakage
    4. Computes sample weights for uneven fold representation

    The purging process (Section 7.4.1):
        For observation i with label spanning [t_i, t1_i], purge from training
        if t1_i > min(t_test). This prevents leakage from labels that extend
        into the test period.

    The embargo process (Section 7.4.2):
        Remove samples immediately following the test set to prevent
        information leakage through serial correlation.

    Args:
        n_splits: Number of groups to split data into (N)
        n_test_splits: Number of groups to use for testing in each combination (k)
        purge_pct: Fraction of training data to purge before test set (fallback)
        embargo_pct: Fraction of data to embargo after test set

    Example:
        With N=6, k=2:
        - Total combinations: C(6,2) = 15
        - Each combination uses 2 groups for testing, 4 for training
        - Provides 15 different train/test scenarios for validation

    Reference:
        Lopez de Prado (2018), Sections 7.4.1-7.4.3, pp. 105-109.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge_pct: float = 0.01,
        embargo_pct: float = 0.01
    ):
        if n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be less than n_splits")

        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct

        # Calculate total number of combinations: C(N, k)
        from math import factorial
        self.n_combinations = int(
            factorial(n_splits) /
            (factorial(n_test_splits) * factorial(n_splits - n_test_splits))
        )

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        t1: Optional[pd.Series] = None,
        groups: Optional[np.ndarray] = None
    ) -> List[CPCVResult]:
        """
        Generate CPCV splits with proper purging and embargo.

        Per Lopez de Prado (2018) Section 7.4.1, purging should be based on
        label end times (t1), not just index proximity. When t1 is provided,
        canonical label-based purging is used. Otherwise, falls back to
        index-based purging.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target array (optional)
            t1: Series mapping sample index to label end time/index.
                CRITICAL for canonical purging. If sample i has label computed
                from data up to t1[i], then t1[i] should be provided.
                For a label like "will price increase in next 10 bars",
                t1[i] = i + 10.
            groups: Group labels (optional, for grouped splitting)

        Returns:
            List of CPCVResult objects containing train/test indices and weights
        """
        n_samples = len(X)

        # Calculate group boundaries
        group_size = n_samples // self.n_splits
        group_indices = []

        for i in range(self.n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_splits - 1 else n_samples
            group_indices.append(np.arange(start, end))

        # Calculate fallback purge and embargo sizes (index-based)
        purge_size = int(n_samples * self.purge_pct)
        embargo_size = int(n_samples * self.embargo_pct)

        # Generate all C(N, k) combinations
        test_combinations = list(combinations(range(self.n_splits), self.n_test_splits))

        # Track sample appearances for weight calculation
        sample_appearances = np.zeros(n_samples)

        results = []
        for fold_id, test_groups in enumerate(test_combinations):
            # Get test indices
            test_indices = np.concatenate([group_indices[g] for g in test_groups])

            # Get train indices (all groups not in test)
            train_groups = [g for g in range(self.n_splits) if g not in test_groups]
            train_indices = np.concatenate([group_indices[g] for g in train_groups])

            test_min, test_max = test_indices.min(), test_indices.max()

            # === PURGING (Section 7.4.1) ===
            if t1 is not None:
                # Canonical label-based purging per Lopez de Prado (2018)
                # Implements BIDIRECTIONAL purging:
                #   Forward: Purge train samples whose labels extend into test period
                #   Backward: Purge test samples whose labels extend into train period
                if isinstance(t1, pd.Series):
                    t1_values = t1.values
                else:
                    t1_values = np.array(t1)

                # FORWARD PURGING: Purge train samples where t1[i] > test_min
                # These training samples have labels that extend into the test period
                purge_mask_train = t1_values[train_indices] > test_min
                train_indices = train_indices[~purge_mask_train]

                # BACKWARD PURGING: Purge test samples where label extends into train
                # Per Section 7.4.1: "We also need to purge from the test set those
                # observations whose labels depend on information used to train"
                #
                # NOTE: This implementation uses contiguous bounds [train_min, train_max]
                # as an approximation. For non-contiguous CPCV combinations (e.g., test
                # groups [1,3] with train groups [0,2,4,5]), this is CONSERVATIVE
                # (over-purges rather than under-purges).
                #
                # Per Lopez de Prado (2018) §7.4.1, over-purging is acceptable;
                # under-purging introduces data leakage and is not.
                #
                # For fully group-aware purging, use group_aware_backward_purge().
                # See Appendix A in CANONICAL_FIXES_REVIEW_20260106.md.
                if len(train_indices) > 0:
                    train_min = train_indices.min()
                    train_max = train_indices.max()

                    # Identify test samples that come before training ends
                    # and whose labels extend into training period
                    backward_purge_mask = (
                        (test_indices < train_max) &  # Test sample comes before train ends
                        (t1_values[test_indices] > train_min)  # Label extends into train
                    )
                    test_indices = test_indices[~backward_purge_mask]

                    if backward_purge_mask.sum() > 0:
                        logger.debug(f"Fold {fold_id}: Backward purged {backward_purge_mask.sum()} test samples")
            else:
                # Fallback: Index-based purging (less precise but functional)
                # Purge samples just before test set
                purge_mask = (train_indices >= test_min - purge_size) & (train_indices < test_min)
                train_indices = train_indices[~purge_mask]

                logger.debug(f"Fold {fold_id}: Using index-based purging (t1 not provided)")

            # === EMBARGO (Section 7.4.2) ===
            # Remove samples immediately after test set
            embargo_mask = (train_indices > test_max) & (train_indices <= test_max + embargo_size)
            train_indices = train_indices[~embargo_mask]

            # Track appearances for weight calculation
            sample_appearances[train_indices] += 1

            results.append(CPCVResult(
                train_indices=train_indices,
                test_indices=test_indices,
                fold_id=fold_id,
                combination=test_groups,
                sample_weights=None  # Computed below
            ))

        # === SAMPLE WEIGHTS (Section 7.4.3) ===
        # Compute weights to correct for uneven sample representation
        # Samples appearing more frequently should have lower weight
        max_appearances = sample_appearances.max()
        if max_appearances > 0:
            for result in results:
                appearances = sample_appearances[result.train_indices]
                # Weight inversely proportional to appearances
                weights = max_appearances / (appearances + 1e-10)
                # Normalize to sum to n_train
                weights = weights / weights.sum() * len(result.train_indices)
                result.sample_weights = weights

        purge_method = "t1-based bidirectional (canonical)" if t1 is not None else "index-based (fallback)"
        logger.info(f"Generated {len(results)} CPCV splits (N={self.n_splits}, k={self.n_test_splits}), purging: {purge_method}")

        return results

    def get_n_splits(self) -> int:
        """Return total number of combinations."""
        return self.n_combinations


def group_aware_backward_purge(
    test_indices: np.ndarray,
    train_indices: np.ndarray,
    t1_values: np.ndarray
) -> np.ndarray:
    """
    Canonical backward purging for non-contiguous CPCV splits.

    Purges test sample j if label interval [j, t1_j] overlaps
    with ANY training index (not just min/max bounds).

    This is the fully canonical implementation per Lopez de Prado (2018)
    Section 7.4.1. Use when exact compliance is required for non-contiguous
    CPCV combinations.

    The default implementation in CombinatorialPurgedKFold.split() uses
    contiguous bounds [train_min, train_max] as a conservative approximation
    that over-purges rather than under-purges.

    Args:
        test_indices: Array of test sample indices
        train_indices: Array of training sample indices (may be non-contiguous)
        t1_values: Array where t1_values[i] = label end index for sample i

    Returns:
        Filtered test_indices with overlapping samples removed

    Complexity: O(n_test * h) where h = max label horizon

    Reference:
        Lopez de Prado (2018) Section 7.4.1, pp. 105-107
        "We also need to purge from the test set those observations whose
        labels depend on information that was used to train the model."
    """
    train_set = set(train_indices)
    keep_mask = np.ones(len(test_indices), dtype=bool)

    for j, test_idx in enumerate(test_indices):
        t1_j = int(t1_values[test_idx])
        # Check if any index in [test_idx, t1_j] is in training set
        for idx in range(int(test_idx), t1_j + 1):
            if idx in train_set:
                keep_mask[j] = False
                break

    return test_indices[keep_mask]


def calculate_dsr(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    returns_skewness: float = 0.0,
    returns_kurtosis: float = 3.0,
    var_sharpe: Optional[float] = None
) -> Dict:
    """
    Calculate Deflated Sharpe Ratio per Bailey & Lopez de Prado (2014).

    The DSR adjusts the observed Sharpe ratio for multiple testing bias.
    When many strategy variants are tested, the maximum observed Sharpe
    will be inflated due to selection bias. DSR provides a haircut.

    DSR = (SR_observed - E[max(SR_0)]) / SE(SR)

    Where:
        - SR_observed: The Sharpe ratio of the selected strategy
        - E[max(SR_0)]: Expected maximum Sharpe under null hypothesis
        - SE(SR): Standard error of Sharpe ratio estimate

    A DSR p-value > 0.05 indicates the observed Sharpe may be due to chance.

    Args:
        observed_sharpe: Observed annualized Sharpe ratio
        n_trials: Number of strategy trials/variants tested
        n_observations: Number of return observations
        returns_skewness: Skewness of returns (0 for normal)
        returns_kurtosis: Kurtosis of returns (3 for normal)
        var_sharpe: Variance of Sharpe estimate (computed if not provided)

    Returns:
        Dictionary with:
            - dsr: Deflated Sharpe Ratio
            - p_value: Probability of observing DSR under null
            - e_max_sr: Expected maximum Sharpe under null
            - se_sr: Standard error of Sharpe estimate
            - haircut: Reduction from observed Sharpe

    Reference:
        Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", SSRN 2460551
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_observations < 2:
        raise ValueError("n_observations must be >= 2")

    # Expected maximum Sharpe under null (Euler-Mascheroni approximation)
    # Per Bailey & Lopez de Prado (2014), Eq. 6
    if n_trials > 1:
        log_n = np.log(n_trials)
        e_max_sr = np.sqrt(2 * log_n) - (
            (EULER_MASCHERONI + np.log(2 * log_n)) / (2 * np.sqrt(2 * log_n))
        )
    else:
        e_max_sr = 0.0

    # Standard error of Sharpe ratio
    # Per Lo (2002) and Bailey & Lopez de Prado (2014), Eq. 4
    # SE(SR) = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt-3)/4*SR^2) / (n-1))
    sr_sq = observed_sharpe ** 2

    if var_sharpe is not None:
        se_sr = np.sqrt(var_sharpe)
    else:
        # Compute from moments
        se_sr = np.sqrt(
            (1 + 0.5 * sr_sq - returns_skewness * observed_sharpe +
             (returns_kurtosis - 3) / 4 * sr_sq) / (n_observations - 1)
        )

    # Deflated Sharpe Ratio
    dsr = (observed_sharpe - e_max_sr) / (se_sr + 1e-10)

    # P-value (one-tailed, testing if DSR > 0)
    p_value = 1 - stats.norm.cdf(dsr)

    # Haircut = how much was "lost" to multiple testing
    haircut = e_max_sr

    return {
        'dsr': dsr,
        'p_value': p_value,
        'e_max_sr': e_max_sr,
        'se_sr': se_sr,
        'haircut': haircut,
        'observed_sharpe': observed_sharpe,
        'n_trials': n_trials,
        'n_observations': n_observations,
        'significant': p_value < 0.05
    }


def calculate_pbo(
    is_returns: np.ndarray,
    oos_returns: np.ndarray,
    n_trials: int = 1000,
    strategy_selection_func: Optional[Callable] = None,
    periods_per_year: int = 252
) -> Dict:
    """
    Calculate Probability of Backtest Overfitting (PBO).

    Implements the method from Bailey et al. (2014) "The Probability of Backtest Overfitting"
    using Monte Carlo approximation of Combinatorial Symmetric Cross-Validation (CSCV).

    IMPLEMENTATION NOTE:
        Canonical CSCV generates all C(T, T/2) possible partitions, which is
        computationally infeasible for large T (e.g., C(500,250) ~ 10^149).
        This implementation uses Monte Carlo sampling with n_trials random
        partitions as a pragmatic approximation. With n_trials >= 1000,
        this provides adequate estimation of PBO.

    The algorithm:
    1. Combine IS and OOS returns
    2. For each trial, randomly partition into two halves
    3. Select best strategy by IS Sharpe, measure its OOS rank
    4. PBO = proportion where best IS strategy underperforms median OOS

    A PBO > 0.50 indicates likely overfitting.

    Args:
        is_returns: In-sample returns matrix (n_periods, n_strategies)
        oos_returns: Out-of-sample returns matrix (n_periods, n_strategies)
        n_trials: Number of Monte Carlo trials (>=1000 recommended)
        strategy_selection_func: Function(returns) -> array of scores
        periods_per_year: Annualization factor (252 for daily, 252*78 for 5-min)

    Returns:
        Dictionary with PBO results

    Reference:
        Bailey et al. (2014), Section 3.2, "Combinatorial Symmetric Cross-Validation"
    """
    n_periods_is, n_strategies = is_returns.shape
    n_periods_oos = oos_returns.shape[0]

    if strategy_selection_func is None:
        # Default: use Sharpe ratio for ranking
        def strategy_selection_func(returns):
            mean_ret = np.mean(returns, axis=0)
            std_ret = np.std(returns, axis=0) + 1e-10
            return mean_ret / std_ret * np.sqrt(periods_per_year)

    # Combine returns for combinatorial splitting
    all_returns = np.vstack([is_returns, oos_returns])
    total_periods = len(all_returns)

    # Monte Carlo approximation of CSCV
    logits = []
    is_sharpes = []
    oos_ranks = []

    for trial in range(n_trials):
        # Random partition into two halves (Monte Carlo approximation)
        perm = np.random.permutation(total_periods)
        half = total_periods // 2

        trial_is = all_returns[perm[:half], :]
        trial_oos = all_returns[perm[half:], :]

        # Rank strategies by IS performance
        is_performance = strategy_selection_func(trial_is)
        best_is_idx = np.argmax(is_performance)
        best_is_sharpe = is_performance[best_is_idx]

        # Calculate OOS performance
        oos_performance = strategy_selection_func(trial_oos)

        # Calculate rank of best IS strategy in OOS
        # Rank 1 = best, Rank n = worst
        oos_rank = n_strategies - np.searchsorted(
            np.sort(oos_performance),
            oos_performance[best_is_idx]
        )

        # Calculate relative rank (0 to 1, where 0 = best)
        relative_rank = (oos_rank - 1) / (n_strategies - 1) if n_strategies > 1 else 0

        # Logit transformation of rank (with boundary handling)
        if relative_rank <= 0:
            logit = -10  # Capped to prevent infinity
        elif relative_rank >= 1:
            logit = 10   # Capped to prevent infinity
        else:
            logit = np.log(relative_rank / (1 - relative_rank))

        logits.append(logit)
        is_sharpes.append(best_is_sharpe)
        oos_ranks.append(relative_rank)

    logits = np.array(logits)
    is_sharpes = np.array(is_sharpes)
    oos_ranks = np.array(oos_ranks)

    # PBO = proportion of trials where best IS strategy is below median OOS
    pbo = np.mean(oos_ranks > 0.5)

    # Bootstrap confidence interval
    bootstrap_pbos = []
    for _ in range(1000):
        boot_idx = np.random.choice(len(oos_ranks), len(oos_ranks), replace=True)
        bootstrap_pbos.append(np.mean(oos_ranks[boot_idx] > 0.5))

    pbo_ci_lower = np.percentile(bootstrap_pbos, 2.5)
    pbo_ci_upper = np.percentile(bootstrap_pbos, 97.5)

    return {
        'pbo': pbo,
        'pbo_ci_lower': pbo_ci_lower,
        'pbo_ci_upper': pbo_ci_upper,
        'logits': logits,
        'logits_mean': np.mean(logits),
        'logits_std': np.std(logits),
        'is_sharpes': is_sharpes,
        'oos_ranks': oos_ranks,
        'n_trials': n_trials,
        'n_strategies': n_strategies,
        'periods_per_year': periods_per_year,
        'method': 'monte_carlo_cscv'  # Document approximation method
    }


def run_cpcv_validation(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable,
    metric_func: Callable,
    config: Optional[CPCVConfig] = None,
    t1: Optional[pd.Series] = None,
    use_sample_weights: bool = False
) -> Dict:
    """
    Run CPCV validation on a model.

    Args:
        X: Feature matrix
        y: Target vector
        model_factory: Function that returns a new model instance
        metric_func: Function(y_true, y_pred) -> float
        config: CPCV configuration
        t1: Label end times for canonical purging (recommended)
        use_sample_weights: Whether to use computed sample weights in training

    Returns:
        Dictionary with validation results
    """
    if config is None:
        config = CPCVConfig()

    cpcv = CombinatorialPurgedKFold(
        n_splits=config.n_splits,
        n_test_splits=config.n_test_splits,
        purge_pct=config.purge_pct,
        embargo_pct=config.embargo_pct
    )

    splits = cpcv.split(X, y, t1=t1)

    metrics = []
    fold_details = []

    for result in splits:
        # Get train/test data
        X_train = X[result.train_indices]
        y_train = y[result.train_indices]
        X_test = X[result.test_indices]
        y_test = y[result.test_indices]

        if len(X_train) < config.min_train_size:
            logger.warning(f"Fold {result.fold_id}: Insufficient training data ({len(X_train)})")
            continue

        # Train model
        model = model_factory()

        if use_sample_weights and result.sample_weights is not None:
            # Use sample weights if model supports it
            try:
                model.fit(X_train, y_train, sample_weight=result.sample_weights)
            except TypeError:
                # Model doesn't support sample_weight
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        # Predict and evaluate
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(X_test)[:, 1]
        else:
            y_pred = model.predict(X_test)

        metric = metric_func(y_test, y_pred)

        metrics.append(metric)
        fold_details.append({
            'fold_id': result.fold_id,
            'combination': result.combination,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'metric': metric
        })

    metrics = np.array(metrics)

    # Statistical summary
    mean_metric = np.mean(metrics)
    std_metric = np.std(metrics)

    # 95% confidence interval
    ci_lower = np.percentile(metrics, 2.5)
    ci_upper = np.percentile(metrics, 97.5)

    # T-test against 0.5 for AUC (random classifier baseline)
    t_stat, p_value = stats.ttest_1samp(metrics, 0.5)

    return {
        'mean': mean_metric,
        'std': std_metric,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        't_statistic': t_stat,
        'p_value': p_value,
        'n_folds': len(metrics),
        'metrics': metrics,
        'fold_details': fold_details,
        'config': {
            'n_splits': config.n_splits,
            'n_test_splits': config.n_test_splits,
            'purge_pct': config.purge_pct,
            'embargo_pct': config.embargo_pct
        },
        't1_provided': t1 is not None,
        'sample_weights_used': use_sample_weights
    }


def run_pbo_analysis(
    returns_matrix: np.ndarray,
    n_trials: int = 1000,
    train_ratio: float = 0.5,
    periods_per_year: int = 252
) -> Dict:
    """
    Run PBO analysis on a matrix of strategy returns.

    Args:
        returns_matrix: Matrix of returns (n_periods, n_strategies)
                       Each column is a different strategy variant
        n_trials: Number of Monte Carlo trials (>=1000 recommended)
        train_ratio: Ratio of data for in-sample vs out-of-sample
        periods_per_year: Annualization factor (252=daily, 19656=5min bars)

    Returns:
        PBO analysis results including interpretation
    """
    n_periods, n_strategies = returns_matrix.shape

    # Split into IS and OOS
    split_idx = int(n_periods * train_ratio)
    is_returns = returns_matrix[:split_idx, :]
    oos_returns = returns_matrix[split_idx:, :]

    logger.info(f"PBO Analysis: {n_strategies} strategies, {n_periods} periods")
    logger.info(f"  IS: {split_idx} periods, OOS: {n_periods - split_idx} periods")
    logger.info(f"  Annualization: sqrt({periods_per_year})")
    logger.info(f"  Method: Monte Carlo CSCV ({n_trials} trials)")

    result = calculate_pbo(
        is_returns, oos_returns, n_trials,
        periods_per_year=periods_per_year
    )

    # Interpretation per Bailey et al. (2014)
    if result['pbo'] < 0.3:
        interpretation = "LOW OVERFITTING RISK - Strategy likely robust"
    elif result['pbo'] < 0.5:
        interpretation = "MODERATE OVERFITTING RISK - Some concern"
    elif result['pbo'] < 0.7:
        interpretation = "HIGH OVERFITTING RISK - Strategy likely overfit"
    else:
        interpretation = "VERY HIGH OVERFITTING RISK - Strategy almost certainly overfit"

    result['interpretation'] = interpretation

    return result


def run_dsr_analysis(
    returns: np.ndarray,
    n_trials: int,
    periods_per_year: int = 252
) -> Dict:
    """
    Run DSR analysis on strategy returns.

    Args:
        returns: Array of strategy returns
        n_trials: Number of strategy variants tested during development
        periods_per_year: Annualization factor

    Returns:
        DSR analysis results
    """
    n_observations = len(returns)

    # Calculate observed Sharpe
    mean_ret = np.mean(returns)
    std_ret = np.std(returns) + 1e-10
    observed_sharpe = mean_ret / std_ret * np.sqrt(periods_per_year)

    # Calculate moments for SE calculation
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns) + 3  # scipy returns excess kurtosis

    result = calculate_dsr(
        observed_sharpe=observed_sharpe,
        n_trials=n_trials,
        n_observations=n_observations,
        returns_skewness=skewness,
        returns_kurtosis=kurtosis
    )

    # Interpretation
    if result['p_value'] < 0.01:
        interpretation = "HIGHLY SIGNIFICANT - Strong evidence of true skill"
    elif result['p_value'] < 0.05:
        interpretation = "SIGNIFICANT - Evidence of true skill"
    elif result['p_value'] < 0.10:
        interpretation = "MARGINALLY SIGNIFICANT - Weak evidence"
    else:
        interpretation = "NOT SIGNIFICANT - Observed Sharpe likely due to chance"

    result['interpretation'] = interpretation

    return result


def print_validation_report(
    cpcv_result: Dict,
    pbo_result: Optional[Dict] = None,
    dsr_result: Optional[Dict] = None
):
    """Print formatted validation report."""
    print("=" * 70)
    print("COMBINATORIAL PURGED CROSS-VALIDATION REPORT")
    print("Per Lopez de Prado (2018), Bailey et al. (2014)")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Splits (N):        {cpcv_result['config']['n_splits']}")
    print(f"  Test splits (k):   {cpcv_result['config']['n_test_splits']}")
    print(f"  Purge %:           {cpcv_result['config']['purge_pct']:.1%}")
    print(f"  Embargo %:         {cpcv_result['config']['embargo_pct']:.1%}")
    print(f"  Total folds:       {cpcv_result['n_folds']}")
    print(f"  t1 provided:       {cpcv_result.get('t1_provided', False)}")

    print(f"\nPerformance Metrics:")
    print(f"  Mean:              {cpcv_result['mean']:.4f}")
    print(f"  Std:               {cpcv_result['std']:.4f}")
    print(f"  95% CI:            [{cpcv_result['ci_lower']:.4f}, {cpcv_result['ci_upper']:.4f}]")

    print(f"\nStatistical Significance (vs 0.5 baseline):")
    print(f"  t-statistic:       {cpcv_result['t_statistic']:.4f}")
    print(f"  p-value:           {cpcv_result['p_value']:.4f}")

    if cpcv_result['p_value'] < 0.05:
        print(f"  Result:            SIGNIFICANT (p < 0.05)")
    else:
        print(f"  Result:            NOT SIGNIFICANT (p >= 0.05)")

    if pbo_result:
        print("\n" + "-" * 70)
        print("PROBABILITY OF BACKTEST OVERFITTING (PBO)")
        print(f"Method: {pbo_result.get('method', 'monte_carlo_cscv')}")
        print("-" * 70)

        print(f"\n  PBO:               {pbo_result['pbo']:.3f}")
        print(f"  95% CI:            [{pbo_result['pbo_ci_lower']:.3f}, {pbo_result['pbo_ci_upper']:.3f}]")
        print(f"  Trials:            {pbo_result['n_trials']}")
        print(f"  Strategies:        {pbo_result['n_strategies']}")
        print(f"  Periods/year:      {pbo_result.get('periods_per_year', 252)}")
        print(f"\n  Interpretation:    {pbo_result['interpretation']}")

    if dsr_result:
        print("\n" + "-" * 70)
        print("DEFLATED SHARPE RATIO (DSR)")
        print("Per Bailey & Lopez de Prado (2014)")
        print("-" * 70)

        print(f"\n  Observed Sharpe:   {dsr_result['observed_sharpe']:.4f}")
        print(f"  E[max(SR)]:        {dsr_result['e_max_sr']:.4f} (haircut)")
        print(f"  DSR:               {dsr_result['dsr']:.4f}")
        print(f"  SE(SR):            {dsr_result['se_sr']:.4f}")
        print(f"  p-value:           {dsr_result['p_value']:.4f}")
        print(f"  Trials tested:     {dsr_result['n_trials']}")
        print(f"  Observations:      {dsr_result['n_observations']}")
        print(f"\n  Interpretation:    {dsr_result['interpretation']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("CPCV / PBO / DSR VALIDATION MODULE TEST")
    print("=" * 70)

    # Test CPCV with synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)

    # Create synthetic t1 (label ends 10 bars after observation)
    t1 = pd.Series(np.arange(n_samples) + 10)

    # Test CombinatorialPurgedKFold with t1
    print("\n[1] Testing CombinatorialPurgedKFold with t1 (canonical purging)...")
    cpcv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, purge_pct=0.01, embargo_pct=0.01)
    splits = cpcv.split(X, y, t1=t1)

    print(f"  Generated {len(splits)} splits")
    print(f"  First split: train={len(splits[0].train_indices)}, test={len(splits[0].test_indices)}")
    print(f"  Sample weights computed: {splits[0].sample_weights is not None}")

    # Test with a simple model
    print("\n[2] Testing run_cpcv_validation...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    result = run_cpcv_validation(
        X, y,
        model_factory=lambda: LogisticRegression(max_iter=1000),
        metric_func=roc_auc_score,
        config=CPCVConfig(n_splits=6, n_test_splits=2),
        t1=t1
    )

    print(f"  Mean AUC: {result['mean']:.4f}")
    print(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  t1 provided: {result['t1_provided']}")

    # Test PBO with different annualization
    print("\n[3] Testing PBO with parameterized annualization...")
    n_periods = 500
    n_strategies = 20

    returns = np.random.randn(n_periods, n_strategies) * 0.01
    returns[:, 0] += 0.002  # Add small alpha

    # Test with 5-minute data (252 days * 78 bars)
    pbo_result = run_pbo_analysis(
        returns, n_trials=500,
        periods_per_year=252 * 78  # 5-minute bars
    )

    print(f"  PBO: {pbo_result['pbo']:.3f}")
    print(f"  Annualization: sqrt({pbo_result['periods_per_year']})")
    print(f"  Method: {pbo_result['method']}")
    print(f"  Interpretation: {pbo_result['interpretation']}")

    # Test DSR
    print("\n[4] Testing DSR calculation...")
    strategy_returns = np.random.randn(252) * 0.01 + 0.0005  # Daily returns with small alpha

    dsr_result = run_dsr_analysis(
        strategy_returns,
        n_trials=100,  # Tested 100 strategy variants
        periods_per_year=252
    )

    print(f"  Observed Sharpe: {dsr_result['observed_sharpe']:.4f}")
    print(f"  E[max(SR)]: {dsr_result['e_max_sr']:.4f}")
    print(f"  DSR: {dsr_result['dsr']:.4f}")
    print(f"  p-value: {dsr_result['p_value']:.4f}")
    print(f"  Interpretation: {dsr_result['interpretation']}")

    # Print full report
    print("\n[5] Full Validation Report:")
    print_validation_report(result, pbo_result, dsr_result)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
