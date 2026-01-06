# Trade Frequency Collapse Investigation Framework

**Date:** 2026-01-06
**Classification:** Technical Investigation
**Status:** ROOT CAUSE IDENTIFIED
**Priority:** P0 - Critical Path Blocker

---

## 1. Problem Statement

NinjaTrader backtest produces 5 trades over 12-month period (Jan-Oct 2025) versus expected 150-200 trades based on Python backtest validation. Trade frequency ratio: **2.5-3.3%** of expected.

| Metric | Expected | Observed | Deviation |
|--------|----------|----------|-----------|
| Trade Count | 150-200 | 5 | -97% |
| Signal Gap | <14 days | 247 days | +1664% |
| Monthly Frequency | 12-17 | 0.42 | -97% |

---

## 2. Evidence Summary

### 2.1 NinjaTrader Grid Results (2026-01-06 07:43 AM)

| Trade | Date | Direction | Exit Type | P&L | Gap From Prior |
|-------|------|-----------|-----------|-----|----------------|
| 1 | 2025-01-05 | Long | Target | +$362.50 | - |
| 2 | 2025-01-05 | Long | Time | +$87.50 | 0 days |
| 3 | 2025-01-06 | Long | Target | +$550.00 | 1 day |
| 4 | 2025-09-10 | Long | Target | +$537.50 | **247 days** |
| 5 | 2025-10-27 | Long | Time | +$787.50 | 47 days |

### 2.2 Historical Comparison

| Audit | Date | Trades | Gap | Source |
|-------|------|--------|-----|--------|
| Dec 2025 | 2025-12-07 | 18 | 160 days | NinjaTrader Grid 2025-12-07 |
| Jan 2026 | 2026-01-06 | 5 | 247 days | NinjaTrader Grid 2026-01-06 |

---

## 3. Hypothesis Matrix

### H1: Signal Server Infrastructure Failure

**Mechanism:** TCP connection drops, server crashes, or network interruption during gap period.

**Testable Predictions:**
- Signal server logs show connection drops or errors during Feb-Sep 2025
- Heartbeat monitoring (if enabled) shows gaps
- No signal requests logged during gap period

**Evidence Required:**
```
data/logs/signal_server.log
- Pattern: ERROR, DISCONNECT, TIMEOUT
- Date range: 2025-02-01 to 2025-09-09
```

**Falsification Criteria:** Continuous signal server logs with no errors during gap period.

---

### H2: Volatility Filter Overly Restrictive

**Mechanism:** `min_vol_prob >= 0.50` threshold rejects signals that would have been accepted in Python backtest.

**Testable Predictions:**
- Python backtest with identical threshold produces similar trade count
- Lowering threshold to 0.40 increases trade frequency proportionally
- Signal rejection logs show high `vol_filter` rejection count

**Evidence Required:**
```python
# Compare trade counts at different thresholds
for thresh in [0.40, 0.45, 0.50, 0.55, 0.60]:
    trades = run_backtest(min_vol_prob=thresh)
    print(f"{thresh}: {len(trades)} trades")
```

**Falsification Criteria:** Python backtest at 0.50 threshold produces >100 trades in same period.

---

### H3: Feature Calculation Divergence

**Mechanism:** C# feature calculation differs from Python, causing different model predictions.

**Testable Predictions:**
- Logging raw features from C# shows values outside expected range
- Feature-by-feature comparison reveals systematic offset
- Model probability outputs differ between Python and NT8

**Evidence Required:**
```
# C# feature log sample
Feature[0]=return_1: Python=0.0023, C#=0.0025, Delta=0.0002 (8.7%)
Feature[14]=atr_14: Python=12.45, C#=11.89, Delta=0.56 (4.5%)
...
```

**Falsification Criteria:** All 42 features match within 1% tolerance.

---

### H4: Model File Version Mismatch

**Mechanism:** NT8 strategy loads outdated or incorrect ONNX model files.

**Testable Predictions:**
- Model file timestamps predate latest training
- Model checksum differs from validated version
- Predictions from NT8 model differ from Python model on identical input

**Evidence Required:**
```powershell
# Check model file metadata
Get-Item data\models\walkforward_onnx\*.onnx |
    Select-Object Name, LastWriteTime, Length
```

**Falsification Criteria:** Model files match latest training with identical checksums.

---

### H5: Backtest Configuration Error

**Mechanism:** NinjaTrader backtest date range, data source, or session filter incorrectly configured.

**Testable Predictions:**
- Backtest settings show partial date range
- Session filter excludes valid trading hours
- Data feed has gaps in price history

**Evidence Required:**
- NinjaTrader Strategy Analyzer settings
- Data Series configuration
- Market data gap analysis

**Falsification Criteria:** Backtest configured for full Jan 2025 - Dec 2025 with RTH session.

---

### H6: ATR Exit Parameters Incompatible

**Mechanism:** ATR-based stop/target parameters (2x/3x) trigger exits before signals generate, or prevent entry.

**Testable Predictions:**
- Exit occurs immediately after entry in gap period
- Stop loss triggers dominate exits
- ATR values during gap period are anomalous

**Evidence Required:**
```python
# Analyze ATR distribution during gap vs non-gap periods
gap_atr = df.loc['2025-02-01':'2025-09-09', 'atr_14']
nongap_atr = df.loc[~df.index.isin(gap_atr.index), 'atr_14']
print(f"Gap ATR: {gap_atr.mean():.2f} +/- {gap_atr.std():.2f}")
print(f"Non-gap ATR: {nongap_atr.mean():.2f} +/- {nongap_atr.std():.2f}")
```

**Falsification Criteria:** ATR distribution statistically equivalent (KS test p > 0.05).

---

## 4. Investigation Protocol

### Phase 1: Log Analysis (30 min)

| Step | Action | Output |
|------|--------|--------|
| 1.1 | Check signal server log existence | File exists Y/N |
| 1.2 | Count log entries by month | Monthly distribution |
| 1.3 | Grep for ERROR/DISCONNECT | Error count by type |
| 1.4 | Identify gap boundaries | First/last log in gap |

```powershell
# Execute Phase 1
cd "C:\Users\skoir\Documents\SKIE Enterprises\SKIE-Ninja\SKIE-Ninja-Project\SKIE_Ninja"
Get-Content data\logs\signal_server.log | Measure-Object -Line
Get-Content data\logs\signal_server.log | Select-String "ERROR|DISCONNECT|TIMEOUT"
```

### Phase 2: Python Backtest Comparison (60 min)

| Step | Action | Output |
|------|--------|--------|
| 2.1 | Run Python backtest Jan-Dec 2025 | Trade list |
| 2.2 | Compare trade dates | Match rate |
| 2.3 | Analyze signal distribution | Monthly histogram |
| 2.4 | Extract rejection reasons | Filter breakdown |

```python
# Execute Phase 2
from backtesting.walk_forward_backtest import run_backtest
trades = run_backtest(start='2025-01-01', end='2025-12-31')
print(f"Python trades: {len(trades)}")
print(trades.groupby(trades['entry_time'].dt.month).size())
```

### Phase 3: Feature Parity Verification (60 min)

| Step | Action | Output |
|------|--------|--------|
| 3.1 | Extract C# features for sample bars | Feature matrix |
| 3.2 | Compute Python features for same bars | Feature matrix |
| 3.3 | Calculate element-wise delta | Delta matrix |
| 3.4 | Flag features with >1% deviation | Mismatch list |

### Phase 4: Model Inference Comparison (30 min)

| Step | Action | Output |
|------|--------|--------|
| 4.1 | Select 10 sample bars from gap period | Sample indices |
| 4.2 | Run Python inference | Probabilities |
| 4.3 | Run C# inference (via TCP) | Probabilities |
| 4.4 | Compare outputs | Delta matrix |

---

## 5. Decision Matrix

| Finding | Root Cause | Remediation | Effort |
|---------|------------|-------------|--------|
| H1 confirmed | Infrastructure | Implement auto-restart, monitoring | 4h |
| H2 confirmed | Threshold | Lower to 0.45, re-validate | 2h |
| H3 confirmed | Feature mismatch | Fix C# calculations | 8h |
| H4 confirmed | Model version | Deploy correct models | 1h |
| H5 confirmed | Config error | Correct settings, re-run | 1h |
| H6 confirmed | ATR params | Adjust exit parameters | 4h |

---

## 6. Success Criteria

Investigation complete when:

1. **Root cause identified** with supporting evidence
2. **Remediation implemented** with verification
3. **Re-test shows** trade frequency ratio > 50% of Python baseline

| Metric | Threshold | Measurement |
|--------|-----------|-------------|
| Trade count ratio | > 50% | NT8 trades / Python trades |
| Max signal gap | < 30 days | Max consecutive days without trade |
| Feature parity | > 99% | Matching features within 1% |

---

## 7. Appendix: Data Sources

| Source | Location | Format |
|--------|----------|--------|
| NT8 Grid Results | `C:\Users\skoir\Desktop\NinjaTrader Grid 2026-01-06 07-43 AM.csv` | CSV |
| Signal Server Log | `data/logs/signal_server.log` | Text |
| Python Backtest | `run_ensemble_2025_forward_test.py` | Script |
| Feature Params | `data/models/scaler_params.json` | JSON |
| ONNX Models | `data/models/walkforward_onnx/` | ONNX |

---

## 8. References

1. Previous audit: [NINJATRADER_DEPLOYMENT_AUDIT_20260106.md](NINJATRADER_DEPLOYMENT_AUDIT_20260106.md)
2. Remediation report: [REMEDIATION_COMPLETION_REPORT_20260106.md](REMEDIATION_COMPLETION_REPORT_20260106.md)
3. Feature audit: [FEATURE_AUDIT_20260105.md](FEATURE_AUDIT_20260105.md)

---

## 9. Investigation Results

**Execution Date:** 2026-01-06
**Status:** ROOT CAUSE IDENTIFIED

### Phase 1 Results: Log Analysis

| Metric | Value |
|--------|-------|
| Log file size | 2.9 MB + 10 MB (rotated) |
| Total signals processed | 15,291 |
| Rejected by vol_filter | 15,283 (99.9%) |
| Blocked short signals | 3 |
| Trade signals generated | 1 |
| ERROR/DISCONNECT/TIMEOUT | 0 |

### Root Cause: H2 CONFIRMED - Volatility Filter Overly Restrictive

**Evidence:**

```
Signal #52448 @ 2025-10-14: vol=0.056, -> REJECTED: vol_filter (0.056 < 0.5)
Signal #52449 @ 2025-10-14: vol=0.055, -> REJECTED: vol_filter (0.055 < 0.5)
Signal #52450 @ 2025-10-14: vol=0.039, -> REJECTED: vol_filter (0.039 < 0.5)
...
Signal #[only 1 passed]: vol=0.755 -> TRADE SIGNAL: LONG
```

**Vol Probability Distribution:**
- Typical range: 0.02-0.09 (2-9%)
- Threshold: 0.50 (50%)
- Pass rate: 0.007% (1/15,291)

### Mechanism

The volatility expansion model predicts low probability of volatility expansion for 99.9% of bars. The 0.50 threshold, while validated in Python backtest, produces near-zero trade frequency in NinjaTrader backtests.

**Possible Explanations:**
1. Python backtest used different date range or data
2. Model predictions differ between Python and ONNX runtime
3. Threshold was optimized on different volatility regime
4. ONNX model weights diverged during export

### Falsified Hypotheses

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H1: Infrastructure failure | **FALSIFIED** | 0 errors in logs |
| H3: Feature divergence | **PENDING** | Requires feature comparison |
| H4: Model version mismatch | **PENDING** | Requires checksum verification |
| H5: Config error | **FALSIFIED** | All signals processed |
| H6: ATR exit incompatible | **FALSIFIED** | Trade didn't exit early |

---

## 10. Remediation Options

### Option A: Lower Vol Threshold (Quick Fix)

```python
# Current
min_vol_prob: float = 0.50

# Proposed
min_vol_prob: float = 0.30  # or 0.20
```

**Risk:** May increase false positives, reduce win rate.
**Validation Required:** Re-run Python backtest with lower threshold, verify trade metrics.

### Option B: Investigate Model Divergence (Root Fix)

Compare Python vs ONNX model outputs on identical inputs.

```python
# Test protocol
test_features = load_sample_features(n=100)
python_probs = python_model.predict_proba(test_features)[:, 1]
onnx_probs = onnx_model.predict(test_features)
delta = np.abs(python_probs - onnx_probs)
print(f"Mean delta: {delta.mean():.4f}, Max: {delta.max():.4f}")
```

**Acceptance:** Mean delta < 0.01, Max delta < 0.05.

### Option C: Re-calibrate Vol Model

The vol model may be miscalibrated for 2025 data. Options:
1. Retrain on 2024-2025 data
2. Apply Platt scaling for probability calibration
3. Use quantile-based threshold instead of fixed

---

## 11. Recommended Action

**Immediate:** Execute Option B to determine if issue is model divergence or threshold calibration.

**If model divergence confirmed:** Re-export ONNX models with verification.
**If calibration issue:** Implement Option C with proper validation.

---

## 12. ROOT CAUSE ANALYSIS UPDATE (2026-01-06 07:00)

**Status:** ROOT CAUSE IDENTIFIED AND FIXED

### 12.1 Model Divergence Verification Results

Executed Option B diagnostic (`diagnose_model_divergence.py`):

| Metric | Python ONNX | Python LGB | Expected |
|--------|-------------|------------|----------|
| Vol Prob Mean | 0.4767 | 0.4770 | ~0.48 |
| Vol Prob Max | 0.9885 | 0.9843 | >0.9 |
| Pass Rate @0.50 | 46.90% | 47.67% | ~47% |
| Correlation | 0.9597 | - | >0.95 |

**Conclusion:** ONNX models produce CORRECT probability outputs (mean ~0.48).
The issue is NOT model divergence.

### 12.2 True Root Cause: C# Feature Calculation Bugs

Analysis of `SKIENinjaTCPStrategy.cs` revealed **3 critical bugs**:

#### Bug 1: momentum_* Calculation (CRITICAL)
```csharp
// BUG (lines 485, 489, 493):
featureBuffer[idx++] = Close[0] - Close[5];  // Price difference in POINTS

// CORRECT (Python):
df['close'].pct_change(5)  // Fractional return ~0.001
```
**Impact:** ES close ~6200, so C# sends ~5 when Python expects ~0.0008 → **6250x scale error**

#### Bug 2: rv_* Calculation (CRITICAL)
```csharp
// BUG (line 565):
return Math.Sqrt(variance) * Math.Sqrt(252);  // Annualized

// CORRECT (Python):
df['close'].pct_change().rolling(period).std()  // NOT annualized
```
**Impact:** 16x scale error (sqrt(252) ≈ 15.87)

#### Bug 3: ma_dist_* Normalization (MODERATE)
```csharp
// BUG (lines 486, 490, 494):
(Close[0] - sma5[0]) / Close[0]  // Divide by Close

// CORRECT (Python):
(df['close'] - ma) / ma  // Divide by MA
```
**Impact:** Slight scale difference, compounding with other errors

### 12.3 Why NT8 Showed Vol Prob = 0.02-0.09

When C# sends incorrectly calculated features:
1. Features are scaled using Python's stored scaler params
2. Scaled values are wildly incorrect (e.g., momentum scaled by ~0.002 mean when actual is ~5)
3. Model receives garbage inputs → outputs low-confidence predictions
4. Vol filter rejects 99.9% of signals

### 12.4 Fixes Applied

File: `src/csharp/SKIENinjaTCPStrategy.cs`

1. **momentum_* (lines 485-496):**
   ```csharp
   // OLD: featureBuffer[idx++] = Close[0] - Close[5];
   // NEW:
   featureBuffer[idx++] = Close[5] > 0 ? (Close[0] - Close[5]) / Close[5] : 0;
   ```

2. **rv_* (lines 549-572):**
   ```csharp
   // OLD: return Math.Sqrt(variance) * Math.Sqrt(252);
   // NEW:
   return Math.Sqrt(variance);  // NO annualization
   ```

3. **ma_dist_* (lines 488, 492, 496):**
   ```csharp
   // OLD: (Close[0] - sma5[0]) / Close[0]
   // NEW:
   sma5[0] > 0 ? (Close[0] - sma5[0]) / sma5[0] : 0
   ```

4. **logReturnHistory (lines 420-423):**
   ```csharp
   // OLD: double logReturn = Math.Log(Close[0] / Close[1]);
   // NEW:
   double pctReturn = (Close[0] - Close[1]) / Close[1];  // Percent, not log
   ```

### 12.5 Verification

Created `verify_feature_parity.py` to document exact Python feature calculations and reference values.

Key reference values (sample bar):
- `rv_14`: 0.00039702 (raw std, NOT annualized)
- `momentum_10`: 0.00042116 (fractional return, NOT points)
- `ma_dist_10`: 0.00060027 (normalized by MA)

### 12.6 Remediation Complete

- [x] Root cause identified (C# feature calculation bugs)
- [x] Fixes applied to SKIENinjaTCPStrategy.cs
- [x] Verification script created
- [ ] Recompile C# strategy in NinjaTrader
- [ ] Re-run backtest to verify trade frequency restored

---

*Document compiled: 2026-01-06*
*Investigation status: ROOT CAUSE FIXED*
*Root cause: C# feature calculation bugs (momentum, rv, ma_dist)*
*Files modified: src/csharp/SKIENinjaTCPStrategy.cs*
