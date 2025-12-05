# SKIE_Ninja Project Handoff Document

**Date**: 2025-12-05
**Purpose**: Complete context for continuing development on new machine

---

## PROJECT STATUS: STATISTICALLY VALIDATED & PRODUCTION READY

### Ensemble Strategy (RECOMMENDED - Best Performance)

The **Ensemble Strategy** (Vol Breakout + Sentiment) has been **fully validated** including Monte Carlo simulation:

| Test Period | Vol Breakout | Ensemble | Improvement |
|-------------|-------------|----------|-------------|
| In-Sample (2023-24) | +$209,351 | +$224,813 | +7.4% |
| Out-of-Sample (2020-22) | +$496,380 | +$502,219 | +1.2% |
| Forward Test (2025) | +$57,394 | +$59,847 | +4.3% |

**Total Validated Edge**: $786,879 across 5 years of data (Ensemble Strategy)

### Monte Carlo Validation: PASSED (2025-12-05)

| Metric | Original | 95% CI (Combined) | Assessment |
|--------|----------|-------------------|------------|
| Net P&L | $502,219 | [$361K, $573K] | **100% prob positive** |
| Sharpe Ratio | 3.16 | [2.24, 3.33] | **Significantly > 0** |
| Profit Factor | 1.25 | [1.15, 1.35] | **Significantly > 1** |

**Statistical Significance**: ROBUST (10,000 Monte Carlo iterations)

---

## COMPLETED PHASES

### Phase 11: Sentiment Strategy COMPLETE (2025-12-04)

**Key Finding**: Sentiment (VIX-based) predicts WHEN (vol expansion AUC 0.77) but not WHICH WAY.
Best used as additional filter for vol breakout, not as standalone strategy.

**Files Created**:
- `src/python/strategy/ensemble_strategy.py` - **PRODUCTION STRATEGY** with 3 methods
- `src/python/strategy/sentiment_strategy.py` - Standalone sentiment (research only)
- `src/python/data_collection/historical_sentiment_loader.py` - VIX data loader
- `src/python/run_ensemble_oos_backtest.py` - OOS validation script

**Best Method**: `"either"` - Enter if EITHER technical OR sentiment vol model predicts expansion

---

### Phase 12: Threshold Optimization COMPLETE (2025-12-05)

**Optimized Parameters** (96% improvement over defaults):
```python
# OPTIMIZED ENSEMBLE CONFIG
min_vol_expansion_prob = 0.40   # (default was 0.50)
min_breakout_prob = 0.45        # (default was 0.50)
tp_atr_mult_base = 2.5          # (default was 2.0)
sl_atr_mult_base = 1.25         # (default was 1.0)
```

| Config | Net P&L | Win Rate | Sharpe |
|--------|---------|----------|--------|
| Default | $73,215 | 39.9% | 3.22 |
| **Optimized** | **$143,475** | 43.3% | 4.56 |

**Script**: `src/python/run_ensemble_threshold_optimization.py`
**Results**: `data/optimization_results/ensemble_optimization_*.csv`

---

### Phase 13: Monte Carlo Simulation COMPLETE (2025-12-05)

**Script**: `src/python/run_monte_carlo_simulation.py`

Monte Carlo tests performed (10,000 iterations each):
1. **Bootstrap resampling** - Tests order-dependency
2. **Trade dropout** (0-15%) - Tests sensitivity to missing trades
3. **Cost variance** (±25% slippage, ±10% commission) - Tests cost sensitivity
4. **Combined** - All factors together

**Results**:
- **100% probability of profit** across all simulations
- 95% CI for Net P&L: [$361K, $573K]
- 95% CI for Sharpe: [2.24, 3.33]

**Assessment**: STATISTICALLY ROBUST

---

## NEXT STEPS

### 1. NinjaTrader Integration (PRIORITY)
Export models to ONNX format for NinjaTrader 8 deployment.

### 2. Paper Trading Validation
Deploy ensemble strategy in paper trading environment for live validation.

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

### Enhanced Feature Modules (NEW)
| File | Purpose |
|------|---------|
| `feature_engineering/multi_timeframe_features.py` | MTF analysis (15m, 1h, 4h) |
| `feature_engineering/enhanced_cross_market.py` | Real Databento cross-market data |
| `feature_engineering/social_news_sentiment.py` | Twitter/News/Reddit sentiment |
| `feature_engineering/enhanced_feature_pipeline.py` | Unified feature pipeline |
| `run_enhanced_feature_qc.py` | Enhanced QC validation |

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

### Generate Enhanced Features (NEW)
```bash
python src/python/feature_engineering/enhanced_feature_pipeline.py
```

### Run Enhanced Feature QC (NEW)
```bash
python src/python/run_enhanced_feature_qc.py
```

---

## ENHANCED FEATURES (NEW)

The following enhanced feature modules have been added with strict leakage prevention:

### Multi-Timeframe Analysis
- **Purpose**: Higher timeframe context (15m, 1h, 4h)
- **Features**: Trend alignment, HTF RSI, HTF ATR, support/resistance levels
- **Leakage Prevention**: Uses only COMPLETED HTF bars (lagged)

### Cross-Market Features (Real Data)
- **Purpose**: Cross-market correlations and relationships
- **Data Used**: Real Databento data for NQ, YM, GC, CL, ZN, VIX
- **Features**: Rolling correlations, lead/lag, spreads, regime detection
- **Leakage Prevention**: Proper alignment with lag

### Social/News Sentiment
- **Purpose**: Market sentiment from Twitter, News, Reddit
- **APIs**: Twitter API v2, Alpha Vantage News, PRAW (Reddit)
- **Features**: Sentiment aggregation, momentum, extremes
- **Leakage Prevention**: Minimum 5-minute lag, uses data BEFORE each bar

### Unified Pipeline
```python
from feature_engineering.enhanced_feature_pipeline import generate_enhanced_features

features, validation = generate_enhanced_features(
    prices,
    validate=True  # Runs QC checks
)
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

## VALIDATION CHECKLIST

### Phase 10-13 (All Complete)
- [x] Out-of-sample validation (2020-2022): PASSED
- [x] Forward test (2025): PASSED
- [x] QC checks: PASSED
- [x] **Threshold optimization**: COMPLETE (+96% improvement)
- [x] **Monte Carlo simulation**: COMPLETE (100% prob profit)
- [ ] NinjaTrader ONNX export
- [ ] Paper trading validation

### Strategy Validation Summary
| Strategy | Status | Recommendation |
|----------|--------|----------------|
| `volatility_breakout_strategy.py` | ✓ Validated | Baseline (use as fallback) |
| `sentiment_strategy.py` | ⚠ Research only | Do not use standalone |
| **`ensemble_strategy.py`** | ✓✓ **PRODUCTION** | **Primary strategy** |

### Scripts Reference
| Script | Purpose |
|--------|---------|
| `run_ensemble_threshold_optimization.py` | Find optimal parameters |
| `run_monte_carlo_simulation.py` | Statistical validation |
| `run_qc_check.py` | Data leakage detection |
| `run_ensemble_2025_forward_test.py` | Forward test |
| `run_ensemble_oos_backtest.py` | OOS validation |

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
