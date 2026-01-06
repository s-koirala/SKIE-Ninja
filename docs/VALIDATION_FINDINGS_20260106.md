# CPCV/PBO/DSR Validation Findings

**Date:** 2026-01-06
**Validation Script:** `src/python/run_cpcv_pbo_validation.py`
**Framework Status:** SUBSTANTIALLY CANONICAL
**Methodology:** Trade-based returns simulation (2026-01-06 update)

---

## Executive Summary

| Metric | Threshold | Result | Status |
|--------|-----------|--------|--------|
| CPCV Vol AUC | > 0.60 | 0.760 | **PASS** |
| CPCV High AUC | > 0.60 | 0.721 | **PASS** |
| CPCV Low AUC | > 0.60 | 0.721 | **PASS** |
| PBO | < 0.50 | 0.627 | **FAIL** |
| DSR p-value | < 0.10 | 1.000 | **FAIL** |
| DSR Observed Sharpe | > 0 | -6.68 | **FAIL** |

**Verdict:** Models demonstrate strong predictive power (CPCV pass). However, when simulating actual trades with ATR-based exits, the strategy shows HIGH overfitting risk (PBO = 0.627) and negative risk-adjusted returns (Sharpe = -6.68).

**Live Capital Status:** NOT READY

---

## 1. CPCV Results Analysis

### Model Performance Across Folds

The CPCV validation with N=6 groups, k=2 test groups produced 15 train/test combinations.

| Model | AUC | 95% CI Lower | p-value | Interpretation |
|-------|-----|--------------|---------|----------------|
| Volatility | 0.760 | 0.730 | 1.6e-11 | Strong predictive power |
| High Breakout | 0.721 | - | - | Strong predictive power |
| Low Breakout | 0.721 | - | - | Strong predictive power |

**Assessment:** All three models significantly outperform random chance (AUC > 0.5). The volatility model shows the strongest predictive power with AUC 0.76.

### t1-Based Purging Statistics

- Forward purging: Active (t1 provided = True)
- Backward purging: Active with contiguity approximation
- Folds with insufficient training data after purging: 5/15

**Note:** The high number of folds with 0 samples post-purging is expected with aggressive bidirectional purging and variable t1 horizons. This is conservative (over-purging) per Lopez de Prado (2018) Section 7.4.1.

---

## 2. PBO Results Analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| PBO | 0.627 | HIGH overfitting probability |
| 95% CI | [0.597, 0.657] | Narrow confidence interval |
| Method | monte_carlo_cscv | 1000 trials |
| Periods/Year | 19,656 | Correct for 5-min RTH data |

**Assessment:** PBO = 0.627 indicates 62.7% of the time, the best in-sample strategy configuration ranked WORSE out-of-sample. This exceeds the 0.50 threshold, indicating HIGH overfitting risk.

**Interpretation per Bailey et al. (2014):**
- PBO < 0.30: Low overfitting risk
- PBO < 0.50: Moderate overfitting risk
- PBO >= 0.50: HIGH overfitting risk (CURRENT STATUS)

**Key Finding:** The strategy's performance is NOT consistent across IS/OOS splits. The threshold optimization (vol_thresh, break_thresh) appears to be selecting configurations that don't generalize.

---

## 3. DSR Results Analysis

### Raw Statistics (Trade-Based Methodology)

| Metric | Value |
|--------|-------|
| Observed Sharpe | -6.68 |
| E[max(SR)] | 2.06 |
| SE(SR) | 0.303 |
| DSR Statistic | -28.83 |
| p-value | 1.000 |
| Haircut | 2.06 |
| Observations | 8,970 |

### Methodology Update (2026-01-06)

The returns simulation was updated to use **trade-based** methodology:

```python
# Trade-based implementation
# Entry: Apply half of transaction cost
# Hold: Accumulate returns without transaction cost
# Exit: Apply half of transaction cost + check ATR stops/targets
```

**Trade Simulation Parameters:**
- ATR Stop Loss: 2.0x ATR
- ATR Profit Target: 3.0x ATR
- Maximum Hold Time: 30 bars
- Transaction Cost: 0.05% round-trip (split between entry/exit)

### Root Cause Analysis

**Finding:** Even with trade-based returns, the strategy produces negative Sharpe ratio.

**Possible Causes:**
1. **ATR exit rules not optimal:** The 2x stop / 3x target may not match the strategy's edge
2. **Direction signal accuracy insufficient:** CPCV shows predictive power for breakout direction, but the volatility filter may be reducing trade frequency too much
3. **Market regime dependency:** Strategy may work in specific regimes but fail overall

---

## 4. Remediation Options

Given CPCV passes but PBO and DSR fail, the following options are available:

### Option A: Fix Overfitting (Address PBO Failure)

**Issue:** Threshold optimization is selecting non-generalizing configurations.

**Remediation:**
1. Use fixed, literature-backed thresholds instead of grid search
2. Reduce strategy variants in PBO analysis (fewer threshold combinations)
3. Apply regularization in model training
4. Use nested cross-validation for threshold selection

```python
# Instead of 25 threshold combinations (5x5 grid)
# Use literature-backed defaults
vol_threshold = 0.50  # Single value, not optimized
break_threshold = 0.50  # Single value, not optimized
```

### Option B: Re-engineer Exit Rules (Address DSR Failure)

**Issue:** ATR-based exits (2x stop, 3x target) may not match the strategy's edge.

**Remediation:**
1. Analyze actual trade outcomes to determine optimal exit parameters
2. Consider time-based exits only (simpler, less curve-fit)
3. Remove short positions (9.1% win rate in backtest)

```python
# Simplified exit rules
MAX_HOLD_BARS = 10  # Time exit only
# OR
ATR_STOP_MULT = 1.5   # Tighter stop
ATR_TARGET_MULT = 1.5  # Lower target, higher win rate
```

### Option C: Paper Trading Focus (Empirical Validation)

**Issue:** Statistical tests may not capture real-world dynamics.

**Remediation:**
1. Continue paper trading on Sim101 to accumulate n=100 trades
2. Compare live execution to backtest at trade level
3. Use forward performance as primary validation

**Rationale:** If paper trading shows positive edge, statistical tests may be overly conservative.

### Option D: Strategy Re-design

**Issue:** Current ensemble design may have fundamental edge limitations.

**Remediation:**
1. Reduce model complexity (simpler features)
2. Focus on volatility model only (highest AUC at 0.76)
3. Consider different trading universe (NQ, RTY)
4. Re-examine feature engineering for label leakage

---

## 5. Production Implications

### Current Gate Status

| Gate | Threshold | Status | Blocker? |
|------|-----------|--------|----------|
| CPCV AUC | > 0.60 | PASS (0.72-0.76) | No |
| PBO | < 0.50 | FAIL (0.627) | **YES** |
| DSR p-value | < 0.10 | FAIL (1.00) | **YES** |
| Paper Trades | >= 100 | 18 | **YES** |

### Recommended Path Forward

**DO NOT PROCEED TO LIVE CAPITAL.**

Per the Production Roadmap, all gates must pass before live deployment.

**Immediate Actions:**
1. Continue paper trading on Sim101 (accumulate trade data)
2. Analyze paper trade outcomes vs backtest predictions
3. Consider Option A (fix overfitting via fixed thresholds)

**Medium-term Actions:**
1. Re-evaluate exit rule parameters using paper trade data
2. Consider removing short trades entirely
3. Investigate regime-dependent performance

---

## 6. Technical Notes

### Data Specifications

| Parameter | Value | Source |
|-----------|-------|--------|
| Features file | `data/processed/features_full.csv` | 36,000+ samples |
| Targets file | `data/processed/targets_full.csv` | Matching samples |
| Target horizons | 5, 10 bars | Detected from column names |
| Train size | 800 bars minimum | Walk-forward parameter |
| Embargo | 210 bars | Per walk-forward config |
| Periods/year | 19,656 | 252 days * 78 bars/day |

### Purging Configuration

| Parameter | Value | Reference |
|-----------|-------|-----------|
| n_splits | 6 | CPCV groups |
| n_test_splits | 2 | Test groups per combination |
| embargo_size | 10 | Bars after test |
| t1 horizon | 10 | Max from target columns |
| Purging direction | Bidirectional | Lopez de Prado 7.4.1 |

---

## References

1. Lopez de Prado, M. (2018). "Advances in Financial Machine Learning". Wiley. Ch. 7.
2. Bailey, D.H., et al. (2014). "The Probability of Backtest Overfitting". SSRN 2326253.
3. Bailey, D.H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio". SSRN 2460551.

---

*Report generated: 2026-01-06*
*Framework version: SUBSTANTIALLY CANONICAL*
*Validation script: run_cpcv_pbo_validation.py*
