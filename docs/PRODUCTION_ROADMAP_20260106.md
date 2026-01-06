# SKIE-Ninja Production Roadmap

**Date:** 2026-01-06
**Framework Status:** SUBSTANTIALLY CANONICAL
**Live Capital Status:** NOT READY

---

## Executive Summary

The CPCV/PBO/DSR validation framework is now substantially canonical per Lopez de Prado (2018) and Bailey et al. (2014). Validation completed 2026-01-06 with trade-based returns simulation.

| Gate | Threshold | Current | Status |
|------|-----------|---------|--------|
| Validation Framework | Canonical | Substantially Canonical | **PASS** |
| CPCV AUC | > 0.60 | 0.72-0.76 | **PASS** |
| PBO | < 0.50 | 0.627 | **FAIL** |
| DSR p-value | < 0.10 | 1.000 | **FAIL** |
| Paper Trades | >= 100 | 18 | **FAIL** |

**Key Findings:**
- Models show strong predictive power (CPCV passes with AUC 0.72-0.76)
- Threshold optimization causes overfitting (PBO = 0.627 > 0.50)
- Trade-based simulation produces negative Sharpe (-6.68)
- **Live Capital Status: NOT READY**

---

## Phase 1: Run Canonical Validation (IMMEDIATE)

### Step 1.1: Execute CPCV/PBO/DSR Validation

```powershell
cd "C:\Users\skoir\Documents\SKIE Enterprises\SKIE-Ninja\SKIE-Ninja-Project\SKIE_Ninja"
python src/python/run_cpcv_pbo_validation.py
```

**Expected Output:**
- CPCV results for 15 train/test combinations (N=6, k=2)
- PBO estimate with 95% confidence interval
- DSR statistic and p-value
- Sample weights for each fold

**Interpretation Thresholds:**

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| DSR p-value | < 0.05 | Strong evidence against overfitting |
| DSR p-value | < 0.10 | Acceptable for cautious deployment |
| DSR p-value | >= 0.10 | Cannot reject null hypothesis |
| PBO | < 0.30 | Low overfitting risk |
| PBO | < 0.50 | Moderate overfitting risk |
| PBO | >= 0.50 | High overfitting risk |

### Step 1.2: Review Results

If DSR p-value >= 0.10 or PBO >= 0.50:
- **Do NOT proceed to live capital**
- Review model assumptions
- Consider additional feature engineering
- Extend paper trading period

---

## Phase 2: Paper Trading (REQUIRED)

### Step 2.1: Start Signal Server

```powershell
cd "C:\Users\skoir\Documents\SKIE Enterprises\SKIE-Ninja\SKIE-Ninja-Project\SKIE_Ninja"
python -m src.python.signal_server --host 127.0.0.1 --port 5555
```

**Server Configuration (as remediated):**
- Short signals: **DISABLED** (9.1% win rate)
- Vol threshold: 0.50 (aligned to backtest)
- Breakout threshold: 0.50 (aligned to backtest)
- Heartbeat monitoring: 5-minute intervals
- Log file: `data/logs/signal_server.log`

### Step 2.2: Enable NinjaTrader Strategy

1. Open NinjaTrader 8
2. Apply **SKIENinjaTCPStrategy** to ES 5-minute chart
3. Set account to **Sim101** (demo account)
4. Enable strategy

### Step 2.3: Monitor Daily

- Check `data/logs/signal_server.log` for errors
- Record all trades in tracking spreadsheet
- Watch for signal gaps (heartbeat warnings)

### Step 2.4: Minimum Sample Requirements

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Trade count | >= 100 | Statistical power for 95% CI |
| Duration | >= 90 days | Capture varied market conditions |
| Gap tolerance | < 14 days | Per remediation audit |

---

## Phase 3: Pre-Live Checklist

Before deploying live capital, verify ALL criteria:

### 3.1 Statistical Criteria

- [ ] DSR p-value < 0.10
- [ ] PBO < 0.50
- [ ] Paper trades >= 100
- [ ] Win rate 95% CI lower bound > 35%

### 3.2 Operational Criteria

- [ ] Live/Backtest trade frequency > 50%
- [ ] No signal gaps > 14 days in paper period
- [ ] Server uptime > 95% during RTH
- [ ] No systematic execution errors

### 3.3 Risk Criteria

- [ ] Max drawdown < 150% of expected
- [ ] Profit factor > 1.2
- [ ] No 5+ consecutive losing trades

---

## Phase 4: Live Deployment (CONDITIONAL)

**GATE:** ALL Phase 3 criteria must be met.

### 4.1 Initial Capital Allocation

| Stage | Contracts | Duration | Gate |
|-------|-----------|----------|------|
| Live Phase 1 | 1 | 4 weeks | Execution matches paper |
| Live Phase 2 | 2 | 4 weeks | Performance holds |
| Live Phase 3 | 3+ | Ongoing | Scale per Kelly criterion |

### 4.2 Circuit Breakers

| Trigger | Action |
|---------|--------|
| Daily loss > $2,000 | Halt trading for day |
| Weekly loss > $10,000 | Pause and review |
| Monthly loss > $25,000 | Full system review |
| 5 consecutive losses | Reduce to 1 contract |

---

## Current Blockers

### Blocker 1: PBO = 0.627 (Threshold: < 0.50)

**Interpretation:** 62.7% probability that best in-sample configuration underperforms out-of-sample. HIGH overfitting risk.

**Potential Causes:**
1. Threshold grid search (5x5 = 25 configurations) overfits to in-sample period
2. ATR exit parameters may be data-mined
3. Strategy edge may be regime-specific

**Remediation Options:**
- Use fixed thresholds (0.50, 0.50) instead of grid search
- Reduce strategy variants in PBO analysis
- Apply regularization or simpler model architecture

### Blocker 2: DSR p-value = 1.000 (Threshold: < 0.10)

**Interpretation:** Observed Sharpe ratio (-6.68) is significantly worse than expected by chance alone. Cannot reject null hypothesis.

**Potential Causes:**
1. Negative edge after transaction costs
2. ATR exit rules (2x stop, 3x target) not optimal
3. Volatility filter reducing trade frequency without improving win rate

**Remediation Options:**
- Re-engineer exit rules using empirical trade outcomes
- Remove short positions (historically 9.1% win rate)
- Focus on paper trading for forward validation

### Blocker 3: Paper Trades = 18/100

**Interpretation:** Insufficient sample size for statistical inference.

**Remediation:**
- Continue paper trading on Sim101
- Expected time to n=100: ~6 months at current frequency
- Use paper trade outcomes to calibrate exit rules

---

## Timeline Estimate

| Phase | Duration | Completion |
|-------|----------|------------|
| Phase 1: Validation | 1 day | 2026-01-07 |
| Phase 2: Paper Trading | 6+ months | 2026-07-06* |
| Phase 3: Pre-Live | 1 week | 2026-07-13* |
| Phase 4: Live Phase 1 | 4 weeks | 2026-08-10* |

*Dates conditional on achieving statistical thresholds.

---

## Quick Reference Commands

### Run Canonical Validation
```powershell
cd "C:\Users\skoir\Documents\SKIE Enterprises\SKIE-Ninja\SKIE-Ninja-Project\SKIE_Ninja"
python src/python/run_cpcv_pbo_validation.py
```

### Start Signal Server
```powershell
cd "C:\Users\skoir\Documents\SKIE Enterprises\SKIE-Ninja\SKIE-Ninja-Project\SKIE_Ninja"
python -m src.python.signal_server --host 127.0.0.1 --port 5555
```

### Weekly Retrain (Manual)
```powershell
cd "C:\Users\skoir\Documents\SKIE Enterprises\SKIE-Ninja\SKIE-Ninja-Project\SKIE_Ninja"
.\scripts\weekly_retrain.bat
```

### Check Server Logs
```powershell
Get-Content "data\logs\signal_server.log" -Tail 50
```

---

## References

1. Lopez de Prado, M. (2018). "Advances in Financial Machine Learning". Wiley. Ch. 7.
2. Bailey, D.H., et al. (2014). "The Probability of Backtest Overfitting". SSRN 2326253.
3. Bailey, D.H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio". SSRN 2460551.

---

*Roadmap compiled: 2026-01-06*
*Framework status: SUBSTANTIALLY CANONICAL per CANONICAL_FIXES_REVIEW_20260106.md*
*Next action: Run canonical validation (Phase 1, Step 1.1)*
