# SKIE_Ninja Paper Trading Guide

**Created:** 2026-01-05
**Updated:** 2026-01-06
**Status:** LIVE ON DEMO ACCOUNT

---

## Current Session Status (2026-01-06)

| Component | Status |
|-----------|--------|
| Signal Server | ✅ Running |
| NT8 Strategy | ✅ Connected |
| Walk-Forward Models | ✅ 70 folds loaded |
| Last Backtest | +$1,287.50 (12 months) |

---

## Pre-Launch Checklist

### 1. Data Status (Verified 2026-01-05)

| Data Source | Status | Date Range | Notes |
|-------------|--------|------------|-------|
| VIX Daily | OK | 2023-12-01 to 2025-12-01 | Auto-updates via yfinance |
| AAII Sentiment | OK | 2023-12-01 to 2025-12-01 | Proxy from VIX if gaps |
| PCR (Put/Call) | PROXY | Uses VIX-based proxy | Historical ends 2019 |
| ES 1-min bars | OK | Handled by NinjaTrader | Real-time feed |

### 2. Model Status

- **Walk-forward folds loaded:** 70
- **Current fold:** fold_20241211
- **Sentiment model:** Included
- **All ONNX models:** Validated

### 3. System Requirements

- Python 3.8+ with packages: `onnxruntime`, `pandas`, `numpy`, `lightgbm`
- NinjaTrader 8 with active data subscription
- Network: localhost TCP port 5555 available

---

## Launch Procedure

### Step 1: Start Python Signal Server

Open PowerShell/Command Prompt:

```powershell
cd "C:\Users\skoir\Documents\SKIE Enterprises\SKIE-Ninja\SKIE-Ninja-Project\SKIE_Ninja"
python -m src.python.signal_server --host 127.0.0.1 --port 5555
```

Expected output:
```
============================================================
SKIE_Ninja Signal Server
============================================================
Address: 127.0.0.1:5555
Models: C:\...\data\models\walkforward_onnx
Folds loaded: 70
Sentiment: ENABLED
Ensemble mode: agreement
============================================================
Waiting for connections... (Ctrl+C to stop)
```

### Step 2: Configure NinjaTrader Strategy

1. Open NinjaTrader 8
2. Go to **Strategies** tab
3. Find **SKIENinjaTCPStrategy**
4. Add to ES 5-minute chart

**Strategy Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| ServerHost | 127.0.0.1 | Signal server IP |
| ServerPort | 5555 | Signal server port |
| AutoReconnect | True | Reconnect if disconnected |
| MaxDailyLoss | 2000 | Stop trading after $2K loss |
| MaxPositionSize | 1 | Start with 1 contract |

### Step 3: Enable Strategy

1. Set Account to **Sim101** (simulation account)
2. Set State to **Enabled**
3. Verify connection message in Output window:
   ```
   SKIENinjaTCPStrategy: Connected to signal server
   ```

### Step 4: Monitor First Signals

Watch for:
- Signal server console shows incoming requests
- NT8 Output window shows responses
- Check probabilities are in expected range (0.4-0.95)

---

## Daily Operations

### Morning Checklist (Before 9:30 AM ET)

- [ ] Signal server running (no errors in console)
- [ ] NinjaTrader connected to data feed
- [ ] Strategy enabled on ES 5-min chart
- [ ] Previous day's P&L recorded
- [ ] No scheduled maintenance on broker platform

### End of Day Checklist (After 4:00 PM ET)

- [ ] Review all trades executed
- [ ] Record P&L in tracking spreadsheet
- [ ] Check for any error messages
- [ ] Compare to expected performance
- [ ] Note any unusual market conditions

### Trade Logging Template

| Date | Time | Direction | Entry | Exit | Bars | P&L | Vol Prob | Sent Prob |
|------|------|-----------|-------|------|------|-----|----------|-----------|
| | | | | | | | | |

---

## Weekly Maintenance

### Sunday: Model Retraining (6:00 PM ET)

Option A: **Automated** (recommended)
```powershell
# Run as Administrator to set up scheduled task
.\scripts\setup_weekly_task.ps1
```

Option B: **Manual**
```powershell
cd "C:\Users\skoir\Documents\SKIE Enterprises\SKIE-Ninja\SKIE-Ninja-Project\SKIE_Ninja"
.\scripts\weekly_retrain.bat
```

### Weekly Performance Review

Compare actual results to expected (note: strategy is highly selective):

| Metric | Expected (Monthly) | Your Actual |
|--------|-------------------|-------------|
| Net P&L | ~$100 | $ |
| Trades | ~1-2 | |
| Win Rate | 35-45% | % |
| Avg Win | ~$450 | $ |
| Avg Loss | ~$200 | $ |
| Profit Factor | >1.5 | |

**Action thresholds:**
- Profit factor below 1.2 for 3 consecutive months: Review model performance
- Max DD exceeds $2,000: Pause and investigate
- 5 consecutive losing trades: Full system review

---

## Expected Performance

Based on latest NT8 backtest (2026-01-05, feature fix applied):

| Metric | NT8 Backtest | Notes |
|--------|--------------|-------|
| Annual P&L | +$1,287.50 | 12-month backtest |
| Trades/Year | ~18 | Highly selective |
| Win Rate | 39% | Low rate, high R:R |
| Profit Factor | 1.71 | Winners > Losers |
| Avg Winner | +$442 | Target hits pay well |
| Avg Loser | -$165 | Tight stops |

### Trade Characteristics

| Exit Type | Frequency | Typical P&L |
|-----------|-----------|-------------|
| Profit Target | 28% | +$550 avg |
| Time Exit | 44% | +$67 avg |
| Stop Loss | 28% | -$250 avg |

**IMPORTANT:** Strategy is selective (~1.5 trades/month). Most edge comes from large winners vs small losers.

---

## Troubleshooting

### Signal Server Won't Start

**Error:** `Address already in use`
```powershell
# Find process using port 5555
netstat -ano | findstr :5555
# Kill the process
taskkill /PID <pid> /F
```

**Error:** `Module not found`
```powershell
pip install onnxruntime pandas numpy lightgbm
```

### NinjaTrader Connection Issues

**Error:** `Unable to connect to signal server`
1. Verify signal server is running
2. Check firewall allows localhost connections
3. Verify port 5555 is not blocked
4. Try restarting both signal server and NT8

**Error:** `Invalid response from server`
1. Check signal server console for errors
2. Verify model files exist in `data/models/walkforward_onnx/`
3. Check log files in `logs/` directory

### Model Loading Failures

**Error:** `No fold found for date`
1. Models may be too old - run weekly retraining
2. Check `data/models/walkforward_onnx/` has recent folds

---

## Scaling Up (After Paper Trading)

### Transition to Live Trading

**Minimum paper trading period:** 90 days OR n >= 100 trades

**MANDATORY Criteria for going live (per audit 2026-01-06):**
- [ ] Paper trade sample size: n >= 100 trades
- [ ] Win rate 95% CI lower bound: > 35%
- [ ] DSR p-value < 0.10 (statistical significance)
- [ ] PBO < 0.50 (moderate overfitting risk or lower)
- [ ] Live/Backtest trade frequency ratio > 50%
- [ ] No systematic execution issues

**Current Status (as of 2026-01-06):**
| Criterion | Threshold | Actual | Status |
|-----------|-----------|--------|--------|
| Paper trades | >= 100 | 18 | **FAIL** |
| DSR p-value | < 0.10 | 0.978 | **FAIL** |
| Validation run | Complete | Pending | **PENDING** |

**WARNING:** Do NOT proceed to live capital until ALL mandatory criteria are met.

### Position Sizing Progression

| Stage | Contracts | Duration | Exit Criteria |
|-------|-----------|----------|---------------|
| Paper | 1 | 4-8 weeks | Meet criteria above |
| Live Phase 1 | 1 | 4 weeks | Verify execution matches paper |
| Live Phase 2 | 2 | 4 weeks | Performance holds |
| Live Phase 3 | 3 | Ongoing | Scale based on results |

### Risk Management Rules

1. **Daily loss limit:** $2,000 (stop trading for the day)
2. **Weekly loss limit:** $10,000 (pause and review)
3. **Monthly loss limit:** $25,000 (full system review)
4. **Never risk more than 2% of account per trade**

---

## Contact & Support

- **Documentation:** `docs/` directory
- **Validation Results:** `data/validation_results/`
- **Model Logs:** `logs/` directory

---

*Last Updated: 2026-01-06*
*Generated by SKIE_Ninja Development Team*
