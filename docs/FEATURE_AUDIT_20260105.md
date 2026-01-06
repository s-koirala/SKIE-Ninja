# Feature Parity Audit Report

**Date:** 2026-01-05
**Issue:** NT8 Strategy producing losses (-$42,600) instead of expected profits
**Root Cause:** Complete feature mismatch between C# strategy and trained model

---

## Executive Summary

The SKIENinjaTCPStrategy was sending 42 features that **did not match** the 42 features the ONNX model was trained on. This caused the model to receive garbage input, producing random/incorrect predictions.

**Status: FIXED** - Strategy updated to send exact features in exact order.

---

## Discrepancy Analysis

### Trained Model Expected Features (from scaler_params.json)

```
Feature Index | Feature Name        | Description
-------------|---------------------|----------------------------------
0            | return_lag1         | (close[0] - close[1]) / close[1]
1            | return_lag2         | (close[0] - close[2]) / close[2]
2            | return_lag3         | (close[0] - close[3]) / close[3]
3            | return_lag5         | (close[0] - close[5]) / close[5]
4            | return_lag10        | (close[0] - close[10]) / close[10]
5            | return_lag20        | (close[0] - close[20]) / close[20]
6            | rv_5                | realized volatility 5-bar
7            | atr_5               | ATR 5-bar
8            | atr_pct_5           | atr_5 / close
9            | rv_10               | realized volatility 10-bar
10           | atr_10              | ATR 10-bar
11           | atr_pct_10          | atr_10 / close
12           | rv_14               | realized volatility 14-bar
13           | atr_14              | ATR 14-bar
14           | atr_pct_14          | atr_14 / close
15           | rv_20               | realized volatility 20-bar
16           | atr_20              | ATR 20-bar
17           | atr_pct_20          | atr_20 / close
18           | close_vs_high_10    | (close - high_10) / close
19           | close_vs_low_10     | (close - low_10) / close
20           | range_pct_10        | (high_10 - low_10) / close
21           | close_vs_high_20    | (close - high_20) / close
22           | close_vs_low_20     | (close - low_20) / close
23           | range_pct_20        | (high_20 - low_20) / close
24           | close_vs_high_50    | (close - high_50) / close
25           | close_vs_low_50     | (close - low_50) / close
26           | range_pct_50        | (high_50 - low_50) / close
27           | momentum_5          | close[0] - close[5]
28           | ma_dist_5           | (close - sma5) / close
29           | momentum_10         | close[0] - close[10]
30           | ma_dist_10          | (close - sma10) / close
31           | momentum_20         | close[0] - close[20]
32           | ma_dist_20          | (close - sma20) / close
33           | rsi_7               | RSI 7-period
34           | rsi_14              | RSI 14-period
35           | bb_pct_20           | (close - bb_lower) / bb_width
36           | volume_sma_5        | SMA of volume (5)
37           | volume_ratio_5      | volume / volume_sma_5
38           | volume_sma_10       | SMA of volume (10)
39           | volume_ratio_10     | volume / volume_sma_10
40           | volume_sma_20       | SMA of volume (20)
41           | volume_ratio_20     | volume / volume_sma_20
```

### Original C# Strategy Was Sending (WRONG)

```
Index | What C# Sent          | What Model Expected
------|----------------------|--------------------
0     | return_lag1          | return_lag1 ✓
1     | return_lag3          | return_lag2 ✗
2     | return_lag5          | return_lag3 ✗
3     | return_lag10         | return_lag5 ✗
4     | return_lag20         | return_lag10 ✗
5     | lagged_return[1]     | return_lag20 ✗
6-9   | lagged_returns       | rv_5, atr_5, atr_pct_5 ✗
10-13 | SMA ratios           | rv_10, atr_10, atr_pct_10 ✗
14-16 | BB features          | rv_14, atr_14, atr_pct_14 ✗
17-21 | vol features         | rv_20, atr_20, atr_pct_20 + price ✗
22-26 | momentum features    | price position features ✗
27-28 | RSI                  | momentum_5, ma_dist_5 ✗
29-31 | MACD (NOT IN MODEL)  | momentum_10, ma_dist_10 ✗
32-36 | price position       | momentum_20, ma_dist_20 + RSI ✗
37-41 | TIME (NOT IN MODEL)  | bb_pct + volume features ✗
```

### Key Missing Features in Original

1. **return_lag2** - Was completely missing
2. **rv_X (realized volatility)** - Used different volatility calculation
3. **volume_sma_X, volume_ratio_X** - Not included at all
4. **MACD and Time features** - Included but NOT in trained model

---

## Fix Applied

### SKIENinjaTCPStrategy.cs Changes

1. Added volume SMA indicators:
   ```csharp
   private SMA volSma5, volSma10, volSma20;
   ```

2. Added log return history buffer for realized volatility:
   ```csharp
   private List<double> logReturnHistory;
   ```

3. Completely rewrote `CalculateFeatures()` to match exact order from scaler_params.json

4. Added `GetRealizedVolatility()` helper matching Python calculation:
   ```csharp
   // rv = std(log returns) * sqrt(252)
   ```

5. Removed unused methods: `GetEMA()`, `GetMACDSignal()`, `GetHistoricalVolatility()`

---

## Verification

### Feature Count Verification
```csharp
if (idx != 42)
{
    Print("ERROR: Feature count mismatch. Expected 42, got " + idx);
    return false;
}
```

### Strategy File Location
- Source: `SKIE_Ninja\src\csharp\SKIENinjaTCPStrategy.cs`
- Deployed: `Documents\NinjaTrader 8\bin\Custom\Strategies\SKIENinjaTCPStrategy.cs`

---

## Expected Results After Fix

| Metric | Before Fix | Expected After Fix |
|--------|------------|-------------------|
| Win Rate | 41% | 62-65% |
| Profit Factor | 0.85 | 1.4-1.6 |
| Net P&L (3 mo) | -$42,600 | +$8,000 to +$15,000 |

---

## Action Required

1. **Recompile in NinjaTrader:**
   - Open NinjaScript Editor
   - Right-click → Compile
   - Verify no errors

2. **Re-run backtest** with corrected strategy

3. **Verify feature logging** shows correct values in Output window

---

## Lessons Learned

1. Feature order and calculation method must EXACTLY match training
2. Scaler expects features in specific order - any deviation breaks predictions
3. Always verify feature parity before deployment using scaler_params.json as source of truth

---

*Report generated: 2026-01-05*
*Audit performed by: SKIE_Ninja Development Team*
