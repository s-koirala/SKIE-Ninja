# SKIE-Ninja Methodology Audit Report

**Audit Date:** 2026-01-05
**Auditor:** Quantitative Review
**Status:** CRITICAL ISSUES IDENTIFIED
**Verdict:** Validation methodology requires fundamental revision before capital deployment

---

## Executive Summary

The current validation framework fails to implement canonical methods from the financial machine learning literature. The reported Sharpe ratio of 3.09-3.22 and 100% Monte Carlo probability of profit are **statistically suspicious** and likely overstated due to:

1. **No Combinatorially Purged Cross-Validation (CPCV)** - Single-path walk-forward has high variance
2. **No Probability of Backtest Overfitting (PBO)** - Selection bias unquantified
3. **No Deflated Sharpe Ratio (DSR)** - Multiple testing not corrected
4. **Inconsistent embargo periods** - Data leakage risk
5. **Arbitrary hyperparameters** - No empirical justification

**True edge likely exists** (AUC 0.72-0.84 on vol/breakout) but magnitude is unknown until proper validation.

---

## 1. Arbitrary Parameters Inventory

### 1.1 Walk-Forward Configuration

| File | Parameter | Value | Justification | Status |
|------|-----------|-------|---------------|--------|
| `walk_forward_backtest.py:55` | `train_days` | 180 | None provided | **ARBITRARY** |
| `walk_forward_backtest.py:56` | `test_days` | 5 | None provided | **ARBITRARY** |
| `walk_forward_backtest.py:57` | `embargo_bars` | 42 | Comment: "~3.5 hours" | **ARBITRARY** |
| `ensemble_strategy.py:84` | `train_days` | 60 | None provided | **ARBITRARY** (conflicts) |
| `ensemble_strategy.py:86` | `embargo_bars` | 20 | None provided | **ARBITRARY** (conflicts) |

**Critical Issue**: Two different embargo periods (20 vs 42 bars) exist. Per Lopez de Prado (2018, Ch. 7), embargo should equal `max(feature_lookback, label_horizon)`.

**Correct Calculation**:
```
max_feature_lookback = 200  # MA_200 is longest feature
label_horizon = 30          # longest target horizon
safety_margin = 10
embargo_bars = max(200, 30) + 10 = 210 bars
```

### 1.2 LightGBM Hyperparameters

| File | Parameter | Value | Justification |
|------|-----------|-------|---------------|
| `model_trainer.py:73` | `num_leaves` | 31 | None |
| `model_trainer.py:74` | `learning_rate` | 0.05 | None |
| `model_trainer.py:75` | `feature_fraction` | 0.8 | None |
| `model_trainer.py:77` | `min_child_samples` | 50 | None |
| `model_trainer.py:118` | `n_estimators` | 500 | None |
| `model_trainer.py:119` | `early_stopping_rounds` | 50 | None |
| `ensemble_strategy.py:239` | `n_estimators` | 100 | None (conflicts with 500!) |
| `ensemble_strategy.py:239` | `max_depth` | 5 | None |

### 1.3 Trading Thresholds

| File | Parameter | Value | Selection Method |
|------|-----------|-------|------------------|
| `ensemble_strategy.py:57` | `min_vol_expansion_prob` | 0.50 | Grid search (81 combos) |
| `ensemble_strategy.py:58` | `min_sentiment_vol_prob` | 0.55 | **Fixed, not searched** |
| `ensemble_strategy.py:59` | `min_breakout_prob` | 0.50 | Grid search |
| `ensemble_strategy.py:73` | `tp_atr_mult_base` | 2.0 | Grid search |
| `ensemble_strategy.py:74` | `sl_atr_mult_base` | 1.0 | Grid search |

### 1.4 Target Generation

| File | Parameter | Value | Justification |
|------|-----------|-------|---------------|
| `multi_target_labels.py:48` | `vol_expansion_threshold` | 1.2 | Comment: "20% increase" |
| `multi_target_labels.py:52` | `min_trend_return` | 0.001 | Comment: "0.1%" |
| `multi_target_labels.py:55` | `atr_multiples` | (1.0, 1.5, 2.0, 2.5) | None |

---

## 2. Validation Methodology Failures

### 2.1 No CPCV Implementation

**What exists**: Simple walk-forward with embargo

**What's missing per Lopez de Prado (2018)**:
- Combinatorial path generation
- Purging of overlapping labels
- Multiple test set evaluation

**Current code** (`model_trainer.py:157-254`):
```python
# Single-path walk-forward (HIGH VARIANCE)
for i in range(self.n_splits):
    train_idx = np.arange(train_start, train_end)
    test_idx = np.arange(test_start, test_end)
    yield train_idx, test_idx
```

**Required per canonical CPCV**:
```python
# Generates C(N,k) combinatorial train/test paths
# Purges samples where label horizon overlaps test set
# Reports distribution of OOS performance, not single path
from mlfinlab.cross_validation import CombinatorialPurgedKFold

cv = CombinatorialPurgedKFold(
    n_splits=6,
    n_test_splits=2,  # Creates C(6,2)=15 paths
    embargo_td=pd.Timedelta(hours=3)
)
```

**Reference**: Lopez de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 7.

### 2.2 No PBO Calculation

**What exists**: Monte Carlo resampling of trades

**What's missing per Bailey et al. (2014)**:
- Probability of Backtest Overfitting computation
- Combinatorially Symmetric Cross-Validation (CSCV)
- Rank distribution analysis

**Current Monte Carlo** (`run_monte_carlo_simulation.py:61-70`):
```python
# Only tests trade order sensitivity, NOT overfitting
def bootstrap_resample(self, trades):
    indices = np.random.choice(len(trades), size=n_samples, replace=True)
    return trades.iloc[indices]
```

This tests **execution variance**, not **overfitting probability**.

**Reference**: Bailey, D.H., Borwein, J.M., Lopez de Prado, M., Zhu, Q.J. (2014). "The Probability of Backtest Overfitting". Journal of Computational Finance.

### 2.3 No Deflated Sharpe Ratio

**What's missing per Bailey & Lopez de Prado (2014)**:
```
DSR = (SR_observed - E[max(SR)]) / sigma[SR]
```

Where `E[max(SR)]` accounts for multiple trials. The reported Sharpe of 3.09-3.22 is **not deflated** for:
- 81 threshold combinations tested
- Multiple model types evaluated
- Multiple target formulations tried

**Reference**: Bailey, D.H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio". SSRN 2460551.

### 2.4 Threshold Optimization Flaw

`run_ensemble_threshold_optimization.py:167-172`:
```python
param_grid = {
    'min_vol_expansion_prob': [0.40, 0.50, 0.60],  # Only 3 values
    'min_breakout_prob': [0.45, 0.50, 0.55],
    'tp_atr_mult_base': [1.5, 2.0, 2.5],
    'sl_atr_mult_base': [0.75, 1.0, 1.25]
}
# 81 combinations, 8 folds, NO nested CV
```

**Issues**:
1. Grid search selects on same data used for validation (no nested CV)
2. Only 8 folds = high variance estimate
3. Best configuration selection introduces selection bias
4. No adjustment for multiple comparisons

---

## 3. Statistical Issues

### 3.1 Suspicious Metrics

| Metric | Reported Value | Literature Threshold | Status |
|--------|---------------|---------------------|--------|
| Sharpe Ratio | 3.09-3.22 | < 2.0 typical for non-HFT | **SUSPICIOUS** |
| MC P(Profit>0) | 100% | Should be < 99% | **SUSPICIOUS** |
| Profit Factor | 1.29+ | 1.1-1.5 typical | Plausible |
| Win Rate | 40% | 35-55% typical | Plausible |

Per `validation_framework.py:95`:
```python
max_sharpe: float = 3.0  # Sharpe > 3 is suspicious
```
The framework correctly flags this but current results exceed threshold.

### 3.2 Missing Statistical Tests

| Test | Purpose | Status |
|------|---------|--------|
| Bonferroni correction | Adjust p-values for multiple comparisons | **MISSING** |
| False Discovery Rate (FDR) | Control expected false positives | **MISSING** |
| Bootstrap CI on returns | Uncertainty quantification | **MISSING** |
| Stationarity tests (ADF) | Verify return series stability | **MISSING** |

---

## 4. Model Architecture Assessment

### 4.1 Current Architecture

| Component | Model | Hyperparameters | Validation |
|-----------|-------|-----------------|------------|
| Vol Expansion | LightGBM | n_est=100, depth=5 | Single path WF |
| Breakout High | LightGBM | n_est=100, depth=5 | Single path WF |
| Breakout Low | LightGBM | n_est=100, depth=5 | Single path WF |
| ATR Forecast | LightGBM | n_est=100, depth=5 | Single path WF |

### 4.2 Architecture Issues

1. **No hyperparameter tuning per fold**: Same hyperparameters used across all walk-forward folds
2. **No model selection validation**: LightGBM chosen without comparison to alternatives
3. **Ensemble weights fixed**: `technical_weight=0.6, sentiment_weight=0.4` are arbitrary
4. **Different n_estimators**: 100 in ensemble vs 500 in model_trainer

---

## 5. Required Remediation

### 5.1 Immediate Actions (Before Any Capital Deployment)

| Priority | Action | Reference |
|----------|--------|-----------|
| P0 | Calculate proper embargo (210 bars) | Lopez de Prado (2018) Ch. 7 |
| P0 | Implement CPCV with purging | Lopez de Prado (2018) Ch. 7 |
| P0 | Calculate PBO | Bailey et al. (2014) |
| P0 | Calculate Deflated Sharpe Ratio | Bailey & Lopez de Prado (2014) |
| P1 | Add bootstrap CI on strategy returns | Efron & Tibshirani (1993) |
| P1 | Add stationarity tests (ADF) | Dickey & Fuller (1979) |
| P2 | Implement nested CV for hyperparameters | Varma & Simon (2006) |
| P2 | Use Bayesian optimization | Snoek et al. (2012) |

### 5.2 Expected Outcome

After proper validation:
- Sharpe ratio likely **0.8-1.5** (vs reported 3.09)
- P(Profit>0) likely **70-90%** (vs reported 100%)
- PBO should be **< 0.5** to proceed

---

## 6. References

1. Lopez de Prado, M. (2018). "Advances in Financial Machine Learning". Wiley.
   - Chapter 7: Cross-Validation in Finance
   - Chapter 8: Feature Importance

2. Bailey, D.H., Borwein, J.M., Lopez de Prado, M., Zhu, Q.J. (2014). "The Probability of Backtest Overfitting". Journal of Computational Finance.
   - URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253

3. Bailey, D.H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio". SSRN.
   - URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551

4. Wikipedia. "Purged Cross-Validation".
   - URL: https://en.wikipedia.org/wiki/Purged_cross-validation

5. ScienceDirect. "Backtest overfitting in the machine learning era".
   - URL: https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110

6. CRAN R Package "pbo" Documentation.
   - URL: https://cran.r-project.org/web/packages/pbo/vignettes/pbo.html

---

## 7. Audit Trail

| Date | Action | Result |
|------|--------|--------|
| 2026-01-05 | Initial audit | Critical issues identified |
| TBD | Implement CPCV | Pending |
| TBD | Calculate PBO | Pending |
| TBD | Calculate DSR | Pending |
| TBD | Re-validation | Pending |

---

*This audit supersedes all previous validation reports until remediation is complete.*
