# SKIE_Ninja Project Handoff Document

**Date**: 2025-12-04
**Purpose**: Complete context for continuing development on new machine

---

## PROJECT STATUS: PRODUCTION READY

The volatility breakout strategy has been **fully validated** across all test periods:

| Test Period | Net P&L | Status |
|-------------|---------|--------|
| In-Sample (2023-24) | +$209,351 | PASSED |
| Out-of-Sample (2020-22) | +$496,380 | PASSED |
| Forward Test (2025) | +$57,394 | PASSED |

**Total Validated Edge**: $763,125 across 5 years of data

---

## IMMEDIATE NEXT STEPS

### 1. Run Threshold Optimization (HIGH PRIORITY)
The optimization was started but took too long on previous machine. Run on more powerful hardware:

```bash
cd SKIE_Ninja
python src/python/run_threshold_optimization.py
```

**What it does**: Tests 256 parameter combinations to find optimal:
- `min_vol_expansion_prob` (default 0.50)
- `min_breakout_prob` (default 0.50)
- `tp_atr_mult_base` (default 2.0)
- `sl_atr_mult_base` (default 1.0)

**Expected output**: CSV file in `data/optimization_results/` with rankings

### 2. Monte Carlo Simulation
After optimization, run statistical validation:
- 1000+ bootstrap iterations
- Calculate confidence intervals
- Verify robustness to random variation

### 3. NinjaTrader Integration
Export models to ONNX format for NinjaTrader 8 deployment.

---

## KEY FILES TO KNOW

### Active Strategy Files
| File | Purpose |
|------|---------|
| `src/python/strategy/volatility_breakout_strategy.py` | **MAIN STRATEGY** |
| `src/python/feature_engineering/multi_target_labels.py` | Target generation (73 targets) |
| `src/python/run_oos_backtest.py` | OOS validation script |
| `src/python/run_2025_forward_test.py` | Forward test script |
| `src/python/run_threshold_optimization.py` | Parameter optimization |
| `src/python/run_qc_check.py` | Quality control validation |

### Documentation
| File | Purpose |
|------|---------|
| `config/CANONICAL_REFERENCE.md` | Single source of truth |
| `config/project_memory.md` | Decision log & history |
| `docs/BEST_PRACTICES.md` | Lessons learned |
| `docs/methodology/BACKTEST_METHODOLOGY.md` | Methodology reference |

### Data Files
| File | Period | Use |
|------|--------|-----|
| `data/raw/market/ES_1min_databento.csv` | 2023-2024 | In-sample development |
| `data/raw/market/ES_2020_1min_databento.csv` | 2020 | OOS validation |
| `data/raw/market/ES_2021_1min_databento.csv` | 2021 | OOS validation |
| `data/raw/market/ES_2022_1min_databento.csv` | 2022 | OOS validation |
| `data/raw/market/ES_2025_1min_databento.csv` | 2025 | Forward test |

---

## STRATEGY OVERVIEW

### Core Insight
**Don't predict direction** (impossible, AUC 0.50). Instead predict:
1. **WHEN** to trade - volatility expansion (AUC 0.84)
2. **WHERE** price goes - breakout high/low (AUC 0.72)
3. **HOW MUCH** it moves - ATR forecast (R² 0.36)

### Entry Logic
```python
# Only enter when volatility is expanding
if vol_expansion_prob > 0.50:
    # Direction from breakout prediction
    if breakout_high_prob > breakout_low_prob:
        direction = LONG
    else:
        direction = SHORT

    # Dynamic exits based on ATR
    tp = entry + (direction * predicted_atr * 2.0)
    sl = entry - (direction * predicted_atr * 1.0)
```

### Model Stack
- LightGBM Classifier: Volatility expansion
- LightGBM Classifier: Breakout high
- LightGBM Classifier: Breakout low
- LightGBM Regressor: ATR forecast

---

## CRITICAL WARNINGS

### Data Leakage Prevention
**NEVER use these patterns**:
```python
# BAD - looks into future
df['feature'] = df['close'].shift(-5)  # Negative shift = LEAKAGE
df['feature'] = df['high'].rolling(5, center=True).max()  # center=True = LEAKAGE
```

**ALWAYS use**:
```python
# GOOD - only uses past data
df['feature'] = df['close'].shift(5)  # Positive shift = SAFE
df['feature'] = df['high'].rolling(5).max()  # Default center=False = SAFE
```

### Result Validation
If you see these, **STOP and investigate**:
- Win rate > 60% (ours is 40%)
- Sharpe > 4.0 (ours is 3.2)
- Profit factor > 2.0 (ours is 1.28)
- AUC > 0.85 for any directional prediction

### Files to NOT Use (Deprecated)
See `config/CANONICAL_REFERENCE.md` Section 3 for full list of deprecated files.

---

## ENVIRONMENT SETUP

### Python Dependencies
```bash
pip install numpy pandas scikit-learn lightgbm ta
```

### Required Python Version
Python 3.9+ (tested on 3.11)

### Run Main Backtest
```bash
cd SKIE_Ninja
python src/python/strategy/volatility_breakout_strategy.py
```

### Run QC Check
```bash
python src/python/run_qc_check.py
```

---

## VALIDATED PERFORMANCE METRICS

### Model Performance
| Model | In-Sample | OOS | Forward |
|-------|-----------|-----|---------|
| Vol Expansion AUC | 0.84 | 0.79 | 0.77 |
| Breakout AUC | 0.72 | 0.73 | 0.71 |
| ATR Forecast R² | 0.36 | 0.30 | 0.28 |

### Trading Metrics
| Metric | In-Sample | OOS | Forward |
|--------|-----------|-----|---------|
| Net P&L | $209,351 | $496,380 | $57,394 |
| Total Trades | 4,560 | 9,481 | 1,187 |
| Win Rate | 39.9% | 40.4% | 39.5% |
| Profit Factor | 1.29 | 1.28 | 1.24 |
| Sharpe Ratio | 3.22 | 3.09 | 2.66 |
| Max Drawdown | $30,142 | $33,596 | $12,845 |

---

## TRADING COSTS USED

| Cost | Value | Notes |
|------|-------|-------|
| Commission | $1.29/side | NinjaTrader official rate |
| Slippage | 0.5 ticks | Conservative RTH estimate |
| Tick Size | $0.25 | ES futures |
| Point Value | $50 | ES futures |

---

## PHASE 10 CHECKLIST

- [x] Out-of-sample validation (2020-2022): PASSED
- [x] Forward test (2025): PASSED
- [x] QC checks: PASSED
- [ ] **Threshold optimization** - RUN ON NEW MACHINE
- [ ] Monte Carlo simulation
- [ ] NinjaTrader ONNX export
- [ ] Paper trading validation

---

## QUESTIONS?

Read these files in order:
1. `docs/BEST_PRACTICES.md` - Lessons learned
2. `config/CANONICAL_REFERENCE.md` - Active vs deprecated files
3. `config/project_memory.md` - Full decision history
4. `research/04_multi_target_prediction_strategy.md` - Strategy design

---

*Good luck with the next iteration!*
*SKIE_Ninja Development Team*
