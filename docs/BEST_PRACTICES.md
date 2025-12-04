# SKIE_Ninja Best Practices & Lessons Learned

**Created**: 2025-12-04
**Purpose**: Critical lessons from development to prevent common ML trading pitfalls

---

## 1. DATA LEAKAGE PREVENTION

### What is Data Leakage?
Using future information to make predictions about the past. In trading ML, this creates unrealistically high accuracy that won't work in live trading.

### Common Leakage Sources We Encountered

#### 1.1 Feature Look-Ahead Bias
**BAD - Uses future data:**
```python
# LEAKY: shift(-N) looks into the future
df['pyramid_rr_5'] = df['close'].shift(-5) / df['close'] - 1
df['pivot_high'] = df['high'].rolling(window=5, center=True).max()  # center=True uses future
df['ddca_success'] = df['close'].shift(-horizon) > df['close']  # Uses future close
```

**GOOD - Uses only past data:**
```python
# SAFE: shift(N) or no shift only uses past/current data
df['return_lag5'] = df['close'].pct_change(5)  # Returns from 5 bars ago
df['rolling_high'] = df['high'].rolling(window=20).max()  # Only past 20 bars
df['atr_14'] = tr.rolling(14).mean()  # Only past 14 bars
```

#### 1.2 Target Construction
**BAD - Target leaks into features:**
```python
# If your target is "price goes up in 10 bars", don't include momentum
# features that span those 10 bars
target = (close.shift(-10) > close).astype(int)
feature = close.pct_change(15)  # Overlaps with target horizon!
```

**GOOD - Clean separation:**
```python
# Target horizon = 10 bars, features only use data BEFORE prediction point
target = (close.shift(-10) > close).astype(int)
feature = close.pct_change(5).shift(1)  # Ends before prediction point
```

### 1.3 Detection Methods

1. **Suspiciously High Accuracy**: Win rate > 65%, AUC > 0.85, Sharpe > 4.0
2. **Feature-Target Correlation > 0.30**: Run correlation checks
3. **Temporal Leakage Test**: Train on future, predict past - if AUC > 0.60, you have leakage
4. **Code Review**: Search for `shift(-` patterns in feature code

---

## 2. PROPER WALK-FORWARD VALIDATION

### Why Standard Train/Test Split Fails
Financial data is time-ordered. Random splits cause future data to leak into training.

### Our Walk-Forward Setup
```python
# Configuration that worked
train_days = 60       # 60-day training window
test_days = 5         # 5-day test window (1 week)
embargo_bars = 20     # Gap to prevent leakage (~100 minutes)

# Walk forward loop
for fold in folds:
    train_end = start_idx + train_bars
    test_start = train_end + embargo_bars  # CRITICAL: gap prevents leakage
    test_end = test_start + test_bars

    # Train on PAST data only
    X_train = features.iloc[start_idx:train_end]

    # Test on FUTURE data only
    X_test = features.iloc[test_start:test_end]

    # Move forward in time
    start_idx += test_bars
```

### Embargo Period
- Creates gap between training and testing
- Prevents autocorrelation from creating false edges
- 20 bars @ 5-min = ~100 minutes is sufficient for intraday

---

## 3. REALISTIC COST MODELING

### Costs That Kill Strategies

| Cost | Typical Value | Impact |
|------|---------------|--------|
| Commission | $1.29/side (NinjaTrader) | $2.58 round-trip |
| Slippage | 0.5-1.0 ticks RTH | $12.50-25.00 per trade |
| Bid-Ask Spread | Usually 0.25 (1 tick ES) | Already in slippage |

### Our Cost Implementation
```python
# Per-trade costs
commission = 2 * 1.29 * contracts  # Both sides
slippage = 2 * 0.5 * 0.25 * 50 * contracts  # Entry + Exit, ticks * tick_value

# Net P&L
net_pnl = gross_pnl - commission - slippage
```

### Common Mistakes
1. **Zero slippage** - Unrealistic, always assume at least 0.25-0.5 ticks
2. **Ignoring commission** - $2.58 per trade adds up with 4000+ trades
3. **Not scaling costs with contracts** - More contracts = more slippage

---

## 4. MULTI-TARGET PREDICTION APPROACH

### The Breakthrough Insight
**Direction prediction is impossible** (AUC 0.50 = random).

Instead, predict MORE TRACTABLE market characteristics:

| Target | AUC | Use |
|--------|-----|-----|
| Volatility Expansion | 0.84 | WHEN to trade (filter) |
| New High/Low Breakout | 0.72 | WHERE price goes (direction) |
| ATR Forecast | R² 0.36 | HOW MUCH it moves (sizing/exits) |

### The Strategy Logic
```python
# 1. Only trade when volatility is expanding
if vol_expansion_prob > 0.50:

    # 2. Determine direction from breakout prediction
    if breakout_high_prob > breakout_low_prob:
        direction = 1  # Long
    else:
        direction = -1  # Short

    # 3. Set dynamic exits based on ATR forecast
    tp = predicted_atr * 2.0
    sl = predicted_atr * 1.0
```

---

## 5. QC CHECKS TO RUN

### Before Trusting Any Result

1. **Feature Look-Ahead Check**
   - Search code for `shift(-` patterns
   - Verify `center=False` in all rolling operations

2. **Correlation Check**
   - Max feature-target correlation < 0.30
   - Higher suggests information leakage

3. **Temporal Leakage Test**
   - Train on future, predict past
   - If AUC > 0.55, investigate

4. **Result Suspiciousness Check**
   - Win rate < 65% (ours: 40%)
   - Sharpe < 4.0 (ours: 3.2)
   - Profit factor 1.0-2.0 (ours: 1.28)

### Our QC Script
```bash
python src/python/run_qc_check.py
```

---

## 6. MODEL SELECTION

### What Worked
- **LightGBM** for classification (vol expansion, breakouts)
- **LightGBM Regressor** for ATR forecasting
- Simple models with few hyperparameters

### What Didn't Work
- LSTM/GRU (overfitted quickly)
- XGBoost (similar to LightGBM, no advantage)
- Random Forest (slower, no better)

### Key Hyperparameters
```python
lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    num_leaves=31,
    random_state=42
)
```

---

## 7. FEATURE ENGINEERING

### Safe Features (No Leakage)

| Category | Features | Notes |
|----------|----------|-------|
| Returns | `return_lag1/2/3/5/10/20` | Always use positive lag |
| Volatility | `rv_N`, `atr_N`, `atr_pct_N` | Rolling lookback only |
| Price Position | `close_vs_high_N`, `close_vs_low_N` | Rolling max/min |
| Momentum | `rsi_7/14/21`, `stoch_k` | Standard indicators |
| Volume | `volume_ma_ratio_N` | Volume vs moving avg |
| Time | `hour_sin/cos`, `dow_sin/cos` | Cyclical encoding |

### Dangerous Features (Avoid)
- Anything with `shift(-N)` (negative shift)
- `center=True` in rolling windows
- Features that span into target horizon
- "Success" features based on future prices

---

## 8. DEBUGGING CHECKLIST

When results look too good:

1. [ ] Check win rate: > 60% is suspicious
2. [ ] Check Sharpe: > 4.0 is suspicious
3. [ ] Check profit factor: > 2.0 is suspicious
4. [ ] Run `grep -r "shift(-" src/` to find leaky code
5. [ ] Run QC check script
6. [ ] Compare in-sample vs out-of-sample performance
7. [ ] If OOS is BETTER than in-sample, something is wrong

---

## 9. VALIDATED RESULTS REFERENCE

These are our legitimate, validated results:

| Period | Net P&L | Trades | Win Rate | PF | Sharpe |
|--------|---------|--------|----------|-----|--------|
| In-Sample (2023-24) | +$209,351 | 4,560 | 39.9% | 1.29 | 3.22 |
| OOS (2020-22) | +$496,380 | 9,481 | 40.4% | 1.28 | 3.09 |
| Forward (2025) | +$57,394 | 1,187 | 39.5% | 1.24 | 2.66 |

**Note**: ~40% win rate with ~2:1 reward/risk is realistic for breakout strategies.

---

## 10. NEXT SESSION PRIORITIES

### Immediate Tasks
1. [ ] Run threshold optimization on more powerful machine
2. [ ] Update default parameters based on optimization
3. [ ] Run Monte Carlo simulation (1000+ iterations)

### Future Work
1. [ ] NinjaTrader ONNX integration
2. [ ] Paper trading validation
3. [ ] Live trading deployment

---

*Document maintained by SKIE_Ninja Development Team*
*Last Updated: 2025-12-04*
