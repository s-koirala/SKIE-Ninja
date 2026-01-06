# CPCV/PBO Implementation Audit Report

**Audit Date:** 2026-01-06
**Auditor:** Quantitative Review
**Reference Literature:**
- Lopez de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 7
- Bailey, D.H., et al. (2014). "The Probability of Backtest Overfitting", SSRN 2326253
- Bailey, D.H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio", SSRN 2460551

**Status:** FUNCTIONAL WITH METHODOLOGICAL DEVIATIONS

---

## Executive Summary

The implemented CPCV/PBO validation framework is **functional for preliminary analysis** but contains methodological deviations from canonical implementations described in the financial machine learning literature. The signal server modifications are **acceptable for deployment**.

| Component | Status | Canonical Compliance |
|-----------|--------|---------------------|
| Signal Server | ACCEPTABLE | N/A |
| CPCV Implementation | FUNCTIONAL | **PARTIAL** |
| PBO Implementation | FUNCTIONAL | **APPROXIMATION** |
| DSR Implementation | NOT IMPLEMENTED | **MISSING** |

---

## 1. CPCV Implementation Audit (`cpcv_pbo.py`)

### 1.1 Purging Implementation

**Severity: HIGH**

| Aspect | Canonical Requirement | Implementation | Compliance |
|--------|----------------------|----------------|------------|
| Purge basis | Label horizon (t1) overlap | Index proximity | **INCORRECT** |
| Purge direction | Bidirectional (before AND after) | Before test only | **INCOMPLETE** |
| t1 parameter | Required for each sample | Not used | **MISSING** |

**Code Analysis (lines 153-158):**
```python
# Current implementation - index-based purging only
purge_mask = (train_indices >= test_min - purge_size) & (train_indices < test_min)
train_indices = train_indices[~purge_mask]
```

**Canonical Requirement per Lopez de Prado (2018) Section 7.4.1:**

> "The purging process removes from the training set all observations whose labels overlap with the test set. For observation i with label spanning [t_i, t1_i], purge if t1_i > min(t_test)."

The implementation purges based on **index distance** rather than **label horizon overlap**. This can result in:
- **Under-purging:** Training samples with labels extending into test period are retained
- **Over-purging:** Samples with non-overlapping labels are incorrectly removed

**Required Fix:**
```python
def split(self, X, y=None, t1=None):  # Add t1 parameter
    """
    Args:
        t1: Series mapping sample index to label end time
    """
    # For each test sample, purge train samples where t1[train] > test.min()
    if t1 is not None:
        test_start_time = t1.iloc[test_indices].min()
        purge_mask = t1.iloc[train_indices] > test_start_time
        train_indices = train_indices[~purge_mask]
```

**Reference:** Lopez de Prado (2018), Section 7.4.1, "Purging", pp. 105-107.

---

### 1.2 Embargo Implementation

**Severity: MEDIUM**

| Aspect | Canonical Requirement | Implementation | Compliance |
|--------|----------------------|----------------|------------|
| Embargo direction | After test set | After test set | **CORRECT** |
| Embargo basis | Time-based (embargo_td) | Percentage-based | **ACCEPTABLE** |

**Code Analysis (lines 160-162):**
```python
# Embargo samples immediately after test set
embargo_mask = (train_indices > test_max) & (train_indices <= test_max + embargo_size)
train_indices = train_indices[~embargo_mask]
```

**Assessment:** The embargo implementation is directionally correct. Using percentage-based embargo (`embargo_pct`) rather than time-delta (`embargo_td`) is an acceptable simplification for fixed-frequency data.

**Reference:** Lopez de Prado (2018), Section 7.4.2, "Embargo", pp. 107-108.

---

### 1.3 Sample Weights

**Severity: LOW**

| Aspect | Canonical Requirement | Implementation | Compliance |
|--------|----------------------|----------------|------------|
| Sample weights | Compute weights for uneven representation | Not implemented | **MISSING** |

**Canonical Requirement:**

CPCV should produce sample weights to correct for samples appearing in different numbers of training folds. Current implementation returns indices only.

**Impact:** Minor bias in model training if some samples are over/under-represented across folds.

**Reference:** Lopez de Prado (2018), Section 7.4.3, "Sample Weights", pp. 108-109.

---

## 2. PBO Implementation Audit

### 2.1 Combinatorial Partitioning

**Severity: MEDIUM**

| Aspect | Canonical (Bailey et al. 2014) | Implementation | Compliance |
|--------|-------------------------------|----------------|------------|
| Partition method | CSCV (all C(T,T/2) combinations) | Monte Carlo random sampling | **APPROXIMATION** |
| Determinism | Deterministic | Stochastic | **DIFFERENT** |
| Coverage | Exhaustive | Probabilistic | **APPROXIMATION** |

**Code Analysis (lines 233-239):**
```python
# Monte Carlo approximation - NOT canonical CSCV
for trial in range(n_trials):
    perm = np.random.permutation(total_periods)
    half = total_periods // 2
    trial_is = all_returns[perm[:half], :]
    trial_oos = all_returns[perm[half:], :]
```

**Canonical CSCV per Bailey et al. (2014) Section 3.2:**

> "CSCV generates all C(T, T/2) possible ways of partitioning T periods into two equal subsets of size T/2."

For T=500 periods, C(500,250) ~ 10^149, making exhaustive enumeration computationally infeasible. The Monte Carlo approximation with n_trials=1000 is a **pragmatic compromise**.

**Assessment:** Acceptable for practical use. Should be documented as Monte Carlo approximation, not canonical CSCV.

**Reference:** Bailey et al. (2014), Section 3.2, "Combinatorial Symmetric Cross-Validation".

---

### 2.2 Sharpe Ratio Annualization

**Severity: MEDIUM**

| Aspect | Requirement | Implementation | Compliance |
|--------|-------------|----------------|------------|
| Annualization factor | Match data frequency | Hardcoded sqrt(252) | **POTENTIALLY INCORRECT** |

**Code Analysis (line 222):**
```python
return mean_ret / std_ret * np.sqrt(252)  # Assumes daily data
```

**Issue:** The factor `sqrt(252)` assumes daily returns. For 5-minute bar data:
- Correct factor: `sqrt(252 * 78)` = `sqrt(19,656)` ~ 140.2

**Impact:** Sharpe ratios will be understated by factor of ~3.2x for 5-minute data.

**Required Fix:**
```python
def __init__(self, ..., periods_per_year: int = 252):
    self.periods_per_year = periods_per_year

def strategy_selection_func(returns):
    return mean_ret / std_ret * np.sqrt(self.periods_per_year)
```

**Reference:** Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier".

---

### 2.3 Logit Transformation

**Severity: LOW**

**Code Analysis (lines 260-266):**
```python
if relative_rank <= 0:
    logit = -10  # Capped
elif relative_rank >= 1:
    logit = 10   # Capped
else:
    logit = np.log(relative_rank / (1 - relative_rank))
```

**Assessment:** Boundary handling for logit transformation is reasonable. Capping at +/-10 prevents numerical instability.

---

## 3. Validation Runner Audit (`run_cpcv_pbo_validation.py`)

### 3.1 Strategy Return Generation

**Severity: MEDIUM**

**Code Analysis (lines 218-223):**
```python
signals = (vol_probs[:-1] >= vol_thresh).astype(float)
strat_returns = signals * returns  # Simplified model
```

**Issues:**
1. Long-only assumption (no short signals in return generation)
2. No transaction costs modeled
3. No slippage modeled
4. Single-period holding assumption
5. Does not use full ensemble logic (vol + breakout + sentiment)

**Impact:** PBO analysis is performed on simplified strategy proxies. Results may not transfer to actual trading strategy.

**Recommendation:** Generate returns using full `EnsembleStrategy.simulate_trade()` logic for accurate PBO estimation.

---

### 3.2 CPCV Pass Criteria

**Code Analysis (lines 256-261):**
```python
cpcv_pass = (
    cpcv_result['ci_lower'] > 0.5 and  # AUC > random
    cpcv_result['p_value'] < 0.05       # Significant
)
pbo_pass = pbo_result['pbo'] < 0.50
```

**Assessment:**
- `ci_lower > 0.5` for AUC: **CORRECT** (better than random classifier)
- `p_value < 0.05`: **CORRECT** (conventional significance threshold)
- `pbo < 0.50`: **CORRECT** per Bailey et al. (2014)

---

### 3.3 DSR Not Implemented

**Severity: MEDIUM**

The prior audit (NINJATRADER_DEPLOYMENT_AUDIT_20260106.md) identified DSR as a critical metric. The canonical validation report showed DSR p-value = 0.978 (not significant). This implementation does not compute DSR.

**Required Addition:**
```python
def calculate_dsr(observed_sharpe, n_trials, variance_sharpe, skewness, kurtosis):
    """
    Deflated Sharpe Ratio per Bailey & Lopez de Prado (2014).

    DSR = (SR - E[max(SR)]) / std(SR)

    Where E[max(SR)] is expected maximum Sharpe under null hypothesis
    of n_trials independent tests.
    """
    # Expected maximum under null (Euler-Mascheroni approximation)
    e_max_sr = np.sqrt(2 * np.log(n_trials)) - (
        np.euler_gamma + np.log(2 * np.log(n_trials))
    ) / (2 * np.sqrt(2 * np.log(n_trials)))

    # Standard error of Sharpe
    se_sr = np.sqrt((1 + 0.5 * observed_sharpe**2 - skewness * observed_sharpe +
                     (kurtosis - 3) / 4 * observed_sharpe**2) / (n_observations - 1))

    dsr = (observed_sharpe - e_max_sr) / se_sr
    p_value = 1 - stats.norm.cdf(dsr)

    return {'dsr': dsr, 'p_value': p_value, 'e_max_sr': e_max_sr}
```

**Reference:** Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", SSRN 2460551.

---

## 4. Signal Server Modifications Audit

### 4.1 Threshold Alignment

**Status: CORRECT**

| Parameter | Old Value | New Value | Reference |
|-----------|-----------|-----------|-----------|
| min_vol_prob | 0.40 | 0.50 | ensemble_strategy.py:57 |
| min_sent_prob | 0.55 | 0.55 | ensemble_strategy.py:58 |
| min_breakout_prob | 0.45 | 0.50 | ensemble_strategy.py:59 |

Thresholds now match validated backtest configuration.

---

### 4.2 Short Signal Disable

**Status: APPROPRIATE**

**Justification:**
- Observed win rate: 9.1% (1/11 trades)
- 95% CI (Wilson score): [0.5%, 37.5%]
- Upper bound below acceptable threshold (40%)
- P(true win rate > 40%) < 0.05

Disabling is statistically justified risk mitigation.

---

### 4.3 Logging Infrastructure

**Status: WELL IMPLEMENTED**

| Feature | Implementation | Assessment |
|---------|---------------|------------|
| File rotation | 10MB max, 5 backups | Appropriate |
| Signal logging | Full probability vector | Comprehensive |
| Rejection tracking | By reason category | Useful for diagnosis |
| Client tracking | Connect/disconnect events | Useful for debugging |
| Heartbeat | 5-minute interval | Adequate for gap detection |

---

### 4.4 Gap Detection

**Status: REASONABLE**

| Parameter | Value | Assessment |
|-----------|-------|------------|
| Heartbeat interval | 300 seconds (5 min) | Adequate |
| Gap warning threshold | 3600 seconds (1 hour) | Reasonable |

---

## 5. Literature Compliance Matrix

| Method | Reference | Section | Compliance | Deviation |
|--------|-----------|---------|------------|-----------|
| CPCV Split | Lopez de Prado (2018) | 7.4 | **PARTIAL** | No t1-based purging |
| Purging | Lopez de Prado (2018) | 7.4.1 | **INCORRECT** | Index-based, not label-based |
| Embargo | Lopez de Prado (2018) | 7.4.2 | **CORRECT** | Percentage vs time-delta acceptable |
| Sample Weights | Lopez de Prado (2018) | 7.4.3 | **MISSING** | Not computed |
| PBO | Bailey et al. (2014) | 3.2 | **APPROXIMATION** | MC instead of exhaustive CSCV |
| DSR | Bailey & Lopez de Prado (2014) | - | **NOT IMPLEMENTED** | Omitted |

---

## 6. Summary of Issues by Severity

### HIGH Severity
| Issue | Location | Impact | Recommendation |
|-------|----------|--------|----------------|
| Purging lacks t1/label horizon | cpcv_pbo.py:153-158 | Potential data leakage | Add t1 parameter |

### MEDIUM Severity
| Issue | Location | Impact | Recommendation |
|-------|----------|--------|----------------|
| PBO uses MC not CSCV | cpcv_pbo.py:233-239 | Approximation | Document deviation |
| Annualization hardcoded | cpcv_pbo.py:222 | Incorrect Sharpe for non-daily | Parameterize frequency |
| DSR not implemented | run_cpcv_pbo_validation.py | Key metric omitted | Add DSR calculation |
| Simplified strategy returns | run_cpcv_pbo_validation.py:218-223 | PBO may not reflect actual strategy | Use full ensemble logic |

### LOW Severity
| Issue | Location | Impact | Recommendation |
|-------|----------|--------|----------------|
| Sample weights not computed | cpcv_pbo.py | Minor training bias | Add weight computation |

---

## 7. Recommendations

### Immediate (Before Next Validation Run)
1. **Document Monte Carlo approximation** in PBO docstrings
2. **Parameterize annualization factor** based on data frequency

### Short-Term (Before Live Deployment)
1. **Add t1 parameter** to CPCV for proper label-based purging
2. **Implement DSR calculation** per Bailey & Lopez de Prado (2014)
3. **Generate strategy returns** using full ensemble logic for PBO

### Medium-Term (Model Improvement)
1. **Add sample weight computation** to CPCV
2. **Consider exhaustive CSCV** for small T (if computationally feasible)

---

## 8. Verdict

**CPCV/PBO Implementation:** FUNCTIONAL but NOT CANONICAL. Suitable for preliminary analysis; results should be interpreted with awareness of methodological deviations.

**Signal Server Modifications:** ACCEPTABLE for deployment. All changes are appropriate risk mitigation measures.

**Overall Assessment:** The implementation provides useful validation tooling but should not be represented as fully canonical Lopez de Prado / Bailey et al. methodology until the HIGH severity issues are addressed.

---

## 9. References

1. Lopez de Prado, M. (2018). "Advances in Financial Machine Learning". Wiley.
   - Chapter 7: Cross-Validation in Finance
   - DOI: 10.1002/9781119482086.ch7

2. Bailey, D.H., Borwein, J.M., Lopez de Prado, M., Zhu, Q.J. (2014). "The Probability of Backtest Overfitting". Journal of Computational Finance.
   - SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253

3. Bailey, D.H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio".
   - SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551

4. Bailey, D.H. & Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier". Journal of Risk.
   - SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643

---

*Audit compiled: 2026-01-06*
*This audit should be reviewed prior to any claims of canonical methodology compliance.*
