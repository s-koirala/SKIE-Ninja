# SKIE-Ninja Backtest Investigation Report

**Date**: 2025-12-04
**Analyst**: SKIE-Ninja Development Team
**Status**: Investigation Complete - Critical Issues Identified

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

## 6. Model Comparison Summary

| Model | AUC-ROC | Accuracy | Method | Notes |
|-------|---------|----------|--------|-------|
| LightGBM | 83.50% | 74.00% | Walk-Forward | Metrics need correction |
| XGBoost | 83.42% | 73.84% | Walk-Forward | Metrics need correction |
| LSTM | 66.26% | 62.17% | Purged K-Fold | More realistic metrics |
| GRU | 66.73% | 62.58% | Purged K-Fold | More realistic metrics |

The deep learning models (LSTM, GRU) using purged k-fold CV show more realistic performance metrics, though lower accuracy. This suggests the tree models may be benefiting from some form of leakage in the walk-forward setup.

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
