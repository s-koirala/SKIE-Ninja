# SKIE-Ninja Backtest Investigation Report

**Date**: 2025-12-04
**Analyst**: SKIE-Ninja Development Team
**Status**: FIXES APPLIED - Ready for Re-testing

---

## Executive Summary

The comprehensive backtest results showed metrics that are statistically implausible and require correction before any production deployment. The QC system correctly flagged these issues. This document details the findings and recommended fixes.

---

## 1. Suspicious Metrics Flagged by QC

| Metric | Reported Value | Realistic Range | Status |
|--------|----------------|-----------------|--------|
| Win Rate | 86.0% | 50-65% | SUSPICIOUS |
| Profit Factor | 18.17 | 1.2-3.0 | SUSPICIOUS |
| Sharpe Ratio | 42.68 | 0.5-3.0 | CRITICAL BUG |
| Sortino Ratio | 78.50 | 1.0-5.0 | CRITICAL BUG |
| Worst Day P&L | +$92.50 | Should have losing days | SUSPICIOUS |
| Max Consecutive Wins | 50 | Typically <15 | SUSPICIOUS |

---

## 2. Critical Bug: Sharpe Ratio Calculation

### Location
`SKIE_Ninja/src/python/backtesting/comprehensive_backtest.py` lines 1106-1108

### Current (Incorrect) Implementation
```python
# Annualized Sharpe (assuming ~250 trading days)
trades_per_year = metrics.trades_per_day * 250 if metrics.trades_per_day > 0 else 250
metrics.sharpe_ratio = (avg_return / std_return) * np.sqrt(trades_per_year) if std_return > 0 else 0
```

### Problem Analysis
The formula calculates Sharpe on **per-trade returns** and annualizes using `sqrt(trades_per_day * 250)`.

With 10.95 trades/day:
- trades_per_year = 10.95 * 250 = 2,737.5
- sqrt(2,737.5) = 52.3x multiplier

This incorrectly inflates the Sharpe ratio by ~52x.

### Correct Implementation
Sharpe ratio should be calculated on **daily returns**, not per-trade returns:

```python
# Calculate daily returns first
daily_pnls = list(daily_pnl_dict.values())
avg_daily_return = np.mean(daily_pnls)
std_daily_return = np.std(daily_pnls)

# Annualized Sharpe (standard: sqrt(252) for daily data)
metrics.sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
```

### Expected Corrected Value
With the correct calculation, expected Sharpe should be approximately:
- Avg daily P&L: $2,305.74
- If std dev is similar magnitude, corrected Sharpe would be ~1-3

---

## 3. Industry Benchmarks (Literature Review)

### Realistic Sharpe Ratios by Strategy Type

| Strategy Type | Minimum | Good | Excellent | Reference |
|--------------|---------|------|-----------|-----------|
| Retail Algo Trading | 1.0 | 1.5-2.0 | >2.0 | QuantStart |
| Quant Hedge Funds | 2.0 | 2.5-3.0 | >3.0 | QuantStart |
| High-Frequency Trading | 4.0 | 6-10 | >10 | Industry |
| Prop Desk (Live Trading) | 1.0 | 1.5+ | 2.0+ | Quant SE |

### Key Insights from Literature:
1. "Empirically, the Sharpe Ratio for active trading strategies rarely exceeds 2"
2. "Strategies performing above this threshold over extended periods are highly unlikely to be sustainable"
3. "No strategy can consistently outperform (Sharpe Ratio > 2) over the long term" - citing LTCM example

### Sources:
- https://www.quantstart.com/articles/Sharpe-Ratio-for-Algorithmic-Trading-Performance-Measurement/
- https://highstrike.com/what-is-a-good-sharpe-ratio/
- https://quant.stackexchange.com/questions/21120/what-is-an-acceptable-sharpe-ratio-for-a-prop-desk
- https://blog.quantinsti.com/sharpe-ratio-applications-algorithmic-trading/

---

## 4. Additional Concerns

### 4.1 No Losing Days
The worst day showed +$92.50 profit. In 309 trading days, having zero losing days is statistically improbable for any real trading strategy. This suggests:
- Potential look-ahead bias in features
- Target leakage in feature engineering
- Overfitting to historical data

### 4.2 Time Calculation Inconsistency
Trade #8 shows:
- `bars_held: 2`
- `time_held_min: 3935.0`

This inconsistency occurs when trades span weekends/overnight. The bars_held correctly counts trading bars, but time_held calculates clock time including non-trading hours.

### 4.3 Model vs Trading Performance Gap
- Model AUC-ROC: 83.5% (reasonable)
- Model Accuracy: 74.0% (reasonable)
- Trading Win Rate: 86.0% (suspiciously higher)

The trading win rate exceeding model accuracy suggests the trade execution logic may have issues.

---

## 5. Recommendations

### Immediate Actions Required

1. **Fix Sharpe Ratio Calculation**
   - Calculate on daily returns, not per-trade returns
   - Use standard sqrt(252) annualization for daily data

2. **Fix Sortino Ratio Calculation**
   - Same issue as Sharpe - should use daily returns

3. **Investigate Feature Leakage**
   - Review feature_pipeline.py for potential look-ahead bias
   - Check if any features use future information
   - Verify all technical indicators use only past data

4. **Add More QC Checks**
   - Alert if zero losing days over >50 trading days
   - Alert if trading win rate significantly exceeds model accuracy
   - Add statistical tests for return distribution

### Before Production
- Re-run backtest after fixes
- Validate with out-of-sample data
- Paper trade before live deployment
- Start with minimal position sizes

---

## 6. CRITICAL: Feature Engineering Look-Ahead Bias

**Investigation Date**: 2025-12-04

### Root Cause Identified

The top-ranked features used in training contain **direct look-ahead bias**:

#### Top 3 Features Are Leaky!

| Rank | Feature | Issue | Code Reference |
|------|---------|-------|----------------|
| 1 | `pyramid_rr_5` | Uses `shift(-5)` - future data | advanced_targets.py:81-101 |
| 2 | `pyramid_rr_10` | Uses `shift(-10)` - future data | advanced_targets.py:81-101 |
| 3 | `pyramid_rr_20` | Uses `shift(-20)` - future data | advanced_targets.py:81-101 |
| 14 | `ddca_sell_success_10` | Uses future close price | advanced_targets.py:188-195 |
| 21 | `ddca_buy_success_10` | Uses future close price | advanced_targets.py:188-195 |

#### How The Leak Works

```python
# From advanced_targets.py lines 81-86:
future_max = high.rolling(horizon).max().shift(-horizon)  # LOOKS INTO FUTURE!
future_min = low.rolling(horizon).min().shift(-horizon)   # LOOKS INTO FUTURE!
mfe = (future_max - close) / close  # Max favorable excursion
mae = (close - future_min) / close  # Max adverse excursion
pyramid_rr = mfe / (mae + 1e-10)    # THIS IS THE #1 FEATURE!
```

The `shift(-5)` operation takes the rolling maximum from the NEXT 5 bars and shifts it backwards to the current bar. This gives the model perfect knowledge of:
- How high the price will go in the next N bars
- How low the price will go in the next N bars
- The reward-to-risk ratio of any position

**This is why the model achieves 86% win rate** - it literally knows the future!

### Additional Leaky Features Identified

1. **Pivot Detection** (`pivot_high_*`, `pivot_low_*`):
   - Uses forward-looking window to confirm pivots
   - Ranks 4-12 in feature importance

2. **DDCA Success Features**:
   - `ddca_buy_success_*` and `ddca_sell_success_*`
   - Directly uses `close.shift(-horizon)` (future price)

3. **Percentile Rankings** (minor bias):
   - Include current bar in ranking calculation
   - Creates circular dependency

### Impact Assessment

| Metric | With Leaky Features | Expected Without Leak |
|--------|---------------------|----------------------|
| Win Rate | 86% | 50-60% |
| Profit Factor | 18.17 | 1.2-2.0 |
| Sharpe Ratio | 42.68 (now fixed) | 0.5-2.0 |
| AUC-ROC | 83.5% | 55-65% |

### Immediate Action Required

**DO NOT DEPLOY** any model trained with these features. They will fail catastrophically in live trading.

### Features That Must Be Removed

```
pyramid_rr_5, pyramid_rr_10, pyramid_rr_20
pyramid_long_5, pyramid_long_10, pyramid_long_20
pyramid_short_5, pyramid_short_10, pyramid_short_20
ddca_buy_success_5, ddca_buy_success_10, ddca_buy_success_20
ddca_sell_success_5, ddca_sell_success_10, ddca_sell_success_20
target_mfe_*, target_mae_*
```

### Pivot Features (Require Restructuring)

```
pivot_high_5_5, pivot_high_5_10, pivot_high_10_5, pivot_high_10_10
pivot_high_20_5, pivot_high_20_10
pivot_low_5_5, pivot_low_5_10, pivot_low_10_5, pivot_low_10_10
pivot_low_20_5, pivot_low_20_10
```

---

## 7. Model Comparison Summary (INVALIDATED)

| Model | AUC-ROC | Accuracy | Method | Notes |
|-------|---------|----------|--------|-------|
| LightGBM | 83.50% | 74.00% | Walk-Forward | **INVALID - trained on leaky features** |
| XGBoost | 83.42% | 73.84% | Walk-Forward | **INVALID - trained on leaky features** |
| LSTM | 66.26% | 62.17% | Purged K-Fold | More realistic (fewer leaky features) |
| GRU | 66.73% | 62.58% | Purged K-Fold | More realistic (fewer leaky features) |

The deep learning models (LSTM, GRU) using purged k-fold CV show more realistic performance metrics. This suggests:
1. RNNs may be less susceptible to exploiting leaky features
2. The purged CV methodology prevents some forms of leakage
3. Tree models (LightGBM/XGBoost) excel at exploiting any signal, including leakage

---

## 7. Conclusion

The backtest framework produces reasonable trade logs and metrics in structure, but contains a critical bug in the Sharpe ratio calculation that inflates the reported value by approximately 50x. Additional investigation is needed to understand why no losing days occurred and why trading win rate exceeds model accuracy.

The QC system correctly identified the suspicious metrics, demonstrating the importance of automated quality checks in trading system development.

**Next Steps:**
1. Apply Sharpe ratio fix to comprehensive_backtest.py
2. Re-run backtest with corrected metrics
3. Deep dive into feature engineering for potential leakage
4. Consider more conservative position sizing given metric uncertainty

---

*Report generated as part of SKIE-Ninja comprehensive testing phase.*

---

## 8. FIXES APPLIED (2025-12-04)

### 8.1 Sharpe/Sortino Ratio Fix - COMPLETED

**File**: `comprehensive_backtest.py` lines 1100-1141

Changed from per-trade returns to daily returns:
```python
# Aggregate P&L by day
daily_pnl_dict = {}
for t in self.trades:
    date_key = str(t.entry_time.date())
    daily_pnl_dict[date_key] = daily_pnl_dict.get(date_key, 0) + t.net_pnl

daily_returns = list(daily_pnl_dict.values())
avg_daily_return = np.mean(daily_returns)
std_daily_return = np.std(daily_returns, ddof=1)

# Annualized Sharpe: sqrt(252) for daily data
metrics.sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252)
```

### 8.2 Feature Engineering Look-Ahead Bias - COMPLETED

**File**: `advanced_targets.py`

All leaky features were FIXED (not removed) to preserve trading concepts:

#### Pyramiding Features (lines 67-168)
- **Before**: Used `shift(-horizon)` to access future MFE/MAE
- **After**: Uses `shift(1)` to calculate historical MFE/MAE patterns
- **Contract sizing**: Changed from 5/10/20 to 1/2/3 (max 5)

```python
# FIXED: Uses past data only
past_max = high.rolling(lookback).max().shift(1)  # Exclude current bar
past_min = low.rolling(lookback).min().shift(1)   # Exclude current bar
hist_mfe = (past_max - past_close) / (past_close + 1e-10)
hist_mae = (past_close - past_min) / (past_close + 1e-10)
df[f'pyramid_rr_{lookback}'] = hist_mfe / (hist_mae + 1e-10)
```

#### DDCA Features (lines 220-241)
- **Before**: Used `close.shift(-horizon)` for future price
- **After**: Uses historical pattern effectiveness

```python
# FIXED: Measures how past DDCA setups performed
past_buy_zone = df['ddca_buy_zone_20'].shift(lookback)
past_close = close.shift(lookback)
df[f'ddca_buy_pattern_{lookback}'] = (
    (past_buy_zone == 1) & (close.shift(1) > past_close)
).astype(int)
df[f'ddca_buy_effectiveness_{lookback}'] = df[f'ddca_buy_pattern_{lookback}'].rolling(50).mean()
```

#### Pivot Detection (lines 252-293)
- **Before**: Used `shift(-lookforward)` for forward confirmation
- **After**: Confirms pivots with delay (detects N bars after occurrence)

```python
# FIXED: Confirmed pivot using past data only
total_window = lookback + confirm_bars
window_max = high.rolling(total_window).max().shift(1)
bar_high_confirm_ago = high.shift(confirm_bars)
is_confirmed_pivot_high = (bar_high_confirm_ago >= window_max)
```

### 8.3 QC Checks Added - COMPLETED

**File**: `validation_framework.py`

New validation checks added:
- `max_sharpe: 3.0` - Flag if Sharpe > 3 (literature benchmark)
- `max_sortino: 5.0` - Flag if Sortino > 5
- `max_consecutive_wins: 20` - Flag if >20 consecutive wins
- `min_losing_days_pct: 0.10` - Alert if <10% losing days

### 8.4 Feature Count

After fixes, the module generates **126 features** (down from previous count due to restructuring).

### 8.5 Next Steps

1. Re-train models with fixed features
2. Re-run walk-forward backtest
3. Verify realistic performance metrics (expected: 50-60% win rate, 1-2 Sharpe)
4. Paper trade before live deployment

---

## 9. RE-TEST RESULTS WITH CORRECTED FEATURES (2025-12-04)

**Test Date**: 2025-12-04 10:40-10:45 UTC
**Feature Rankings**: Regenerated with corrected (non-leaky) features

### 9.1 Critical Finding: All Predictive Power Was From Leakage

The re-test with corrected features confirms the catastrophic impact of look-ahead bias:

#### Performance Comparison

| Model | Metric | With Leaky Features | Corrected Features | Change |
|-------|--------|---------------------|-------------------|--------|
| **LightGBM** | Win Rate | 86.0% | 45.1% | -40.9% |
| | Sharpe Ratio | 42.68 | -0.34 | -43.02 |
| | Profit Factor | 18.17 | 0.96 | -17.21 |
| | Net P&L | +$712,475 | -$4,953 | -$717,428 |
| | Total Trades | 3,385 | 893 | -74% |
| **XGBoost** | Win Rate | 86.3% | 46.0% | -40.3% |
| | Sharpe Ratio | 42.63 | -1.88 | -44.51 |
| | Profit Factor | 19.40 | 0.85 | -18.55 |
| | Net P&L | +$718,138 | -$24,548 | -$742,686 |
| | Total Trades | 3,395 | 1,167 | -66% |
| **LSTM** | AUC-ROC | 66.26% | ~49% | -17% |
| | Accuracy | 62.17% | ~49% | -13% |

### 9.2 Key Findings

1. **Zero True Predictive Power**: The corrected models perform at or slightly below random chance (50%), confirming that ALL apparent predictive ability came from look-ahead bias.

2. **Loss-Making Strategy**: Without future information, both LightGBM and XGBoost produce negative returns, with profit factors below 1.0.

3. **Negative Sharpe Ratios**: The corrected Sharpe calculations show -0.34 (LightGBM) and -1.88 (XGBoost), indicating the strategy destroys capital.

4. **Fewer Trades Taken**: Trade count dropped 66-74% because the model is less confident without future data, resulting in fewer high-probability signals.

5. **Deep Learning More Honest**: The LSTM model with corrected features shows ~49% AUC (random), while the leaky version showed 66.26%. This confirms even RNNs were exploiting the leakage to some degree.

### 9.3 New Top Features (Non-Leaky)

After regenerating feature rankings with corrected features, the top features are now legitimate:

| Rank | Feature | Category |
|------|---------|----------|
| 1 | upper_wick_ratio | Price-Based |
| 2 | di_diff_14 | Technical Indicator |
| 3 | macd_signal | Technical Indicator |
| 4 | estimated_buy_volume | Microstructure |
| 5 | close_open_ratio | Price-Based |
| 6 | keltner_mid_20 | Technical Indicator |
| 7 | volume_imbalance | Microstructure |
| 8 | bb_mid_20 | Technical Indicator |

### 9.4 QC Validation Results

The QC system now correctly validates realistic metrics:

```
BACKTEST VALIDATION:
  [PASS] trade_count
  [PASS] rth_compliance
  [PASS] win_rate (45.1% within realistic range)
  [PASS] profit_factor (0.96 within realistic range)
  [PASS] has_commission
  [PASS] has_slippage
  [PASS] consecutive_wins

FEATURE VALIDATION:
  [PASS] data_leakage (no forward-looking features detected)
```

### 9.5 Implications

1. **The Current Feature Set Has No Edge**: The technical indicators and microstructure features used provide NO predictive value for the target variable.

2. **Strategy Redesign Required**: Need to:
   - Develop new predictive features based on market research
   - Consider different target definitions (not just next-bar direction)
   - Explore alternative modeling approaches
   - Potentially increase feature engineering scope

3. **Positive Outcome**: The QC system successfully detected the issues, and the fix process worked correctly. The system is now producing honest, realistic metrics.

### 9.6 Walk-Forward Configuration Used

```
Train Window:    180 days (14,040 bars)
Test Window:     5 days (390 bars)
Embargo:         42 bars
Expected Folds:  61
Data Points:     37,986 samples
Features:        75 (selected from 423 total)
```

---

## 10. Recommendations & Next Steps

### Immediate Actions

1. **Feature Engineering Research Phase**
   - Conduct literature review on predictive features for ES futures
   - Explore order flow/tape reading features
   - Consider regime-based features (volatility regimes, trend regimes)
   - Evaluate higher timeframe confluence features

2. **Target Engineering**
   - Consider multi-bar targets instead of next-bar
   - Explore volatility-adjusted targets
   - Test classification thresholds (not just up/down)

3. **Model Architecture**
   - Current ML models may not be suitable for this problem
   - Consider reinforcement learning approaches
   - Evaluate ensemble methods with diverse feature sets

### Before Any Deployment

1. Achieve consistent positive expectancy in walk-forward tests
2. Paper trade for minimum 3 months
3. Start live with minimal position size (1 contract)
4. Gradual scale-up only with proven live performance

---

*Report updated: 2025-12-04*
*Status: RE-TEST COMPLETE - Strategy requires fundamental redesign*
