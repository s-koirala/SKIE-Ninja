# SKIE_Ninja Canonical Reference

**Created**: 2025-12-04
**Purpose**: Single source of truth for active files, models, and knowledge

---

## 1. DATA AVAILABILITY

### Available Data (Databento)

| File | Period | Bars | Status |
|------|--------|------|--------|
| `ES_1min_databento.csv` | 2023-01-02 to 2024-12-19 | ~684K | **IN-SAMPLE** (development) |
| `ES_2020_1min_databento.csv` | 2020 | ~340K | **OUT-OF-SAMPLE** (validated) |
| `ES_2021_1min_databento.csv` | 2021 | ~340K | **OUT-OF-SAMPLE** (validated) |
| `ES_2022_1min_databento.csv` | 2022 | ~341K | **OUT-OF-SAMPLE** (validated) |
| `ES_2025_1min_databento.csv` | 2025-01-01 to 2025-12-03 | **326K** | **FORWARD TEST** (newest, unseen) |

### Data NOT Available

| Period | Notes |
|--------|-------|
| Pre-2020 | Not downloaded |

---

## 2. ACTIVE FILES (USE THESE)

### Strategy Implementation

| File | Purpose | Status |
|------|---------|--------|
| `src/python/strategy/volatility_breakout_strategy.py` | **MAIN STRATEGY** - Vol filter + breakout | **ACTIVE** |
| `src/python/feature_engineering/multi_target_labels.py` | 73-target generator (vol, breakout, ATR) | **ACTIVE** |
| `src/python/run_oos_backtest.py` | Out-of-sample validation script | **ACTIVE** |
| `src/python/run_2025_forward_test.py` | Forward test on 2025 data | **ACTIVE** |
| `src/python/run_threshold_optimization.py` | Parameter grid search | **ACTIVE** |
| `src/python/run_qc_check.py` | Quality control validation | **ACTIVE** |

### Feature Engineering (Active)

| File | Purpose | Status |
|------|---------|--------|
| `feature_engineering/volatility_regime.py` | VIX + regime detection | **ACTIVE** |
| `feature_engineering/triple_barrier.py` | Triple barrier labeling | **ACTIVE** |

### Documentation (Active)

| File | Purpose | Status |
|------|---------|--------|
| `HANDOFF.md` | **START HERE** - Next session handoff | **ACTIVE** |
| `docs/BEST_PRACTICES.md` | Lessons learned & anti-patterns | **ACTIVE** |
| `config/project_memory.md` | Project decision log | **ACTIVE** |
| `config/CANONICAL_REFERENCE.md` | This file - canonical reference | **ACTIVE** |
| `research/04_multi_target_prediction_strategy.md` | Multi-target strategy design | **ACTIVE** |
| `docs/methodology/BACKTEST_METHODOLOGY.md` | Backtest methodology | **ACTIVE** |

---

## 3. DEPRECATED FILES (DO NOT USE)

### Old Models - DELETED (2025-12-04)

The following files were deleted because they used leaky features with look-ahead bias:
- `xgboost_20251203_*.pkl`, `randomforest_20251203_*.pkl` - Used `pyramid_rr_*` features
- `lightgbm_5min_*.txt`, `lstm_*.pt`, `gru_*.pt` - Trained on leaky data
- All feature importance CSVs from Dec 3-4

### Old Feature Sets (DO NOT USE)

| Features | Reason |
|----------|--------|
| `pyramid_rr_5/10/20` | **LEAKY** - uses shift(-N) future data |
| `pivot_high_*`, `pivot_low_*` | **LEAKY** - forward-looking window |
| `ddca_buy/sell_success_*` | **LEAKY** - uses close.shift(-horizon) |

### Old Backtest Results - DELETED (2025-12-04)

Invalid backtest files were deleted:
- All `trades_lightgbm_*.csv`, `trades_xgboost_*.csv` (86% win rate = leakage)
- Associated metrics, equity, and daily P&L files
- Old purged CV results and comparison reports

**Remaining valid results:**
- `vol_breakout_trades_20251204_134623.csv` - Valid strategy backtest (39.9% win rate)
- `oos_backtest_trades_20251204_*.csv` - OOS validation trades
- `triple_barrier_backtest_*.csv/json/txt` - Triple barrier validation

### Old Research Documents

| File | Status |
|------|--------|
| `research/01_initial_research.md` | Historical context only |
| `research/02_comprehensive_variables_research.md` | Superseded by multi-target approach |
| `research/03_advanced_strategy_research.md` | Partially implemented, use 04 instead |

---

## 4. ACTIVE FEATURE SET

The volatility breakout strategy uses these **NON-LEAKY** features:

### Returns (Lagged - Safe)
- `return_lag1/2/3/5/10/20` - Historical returns

### Volatility (Safe)
- `rv_5/10/14/20` - Realized volatility
- `atr_5/10/14/20` - Average true range
- `atr_pct_5/10/14/20` - ATR as percent of close

### Price Position (Safe)
- `close_vs_high_10/20/50` - Close vs rolling high
- `close_vs_low_10/20/50` - Close vs rolling low
- `close_vs_ma_10/20/50` - Close vs moving average

### Momentum (Safe)
- `rsi_7/14/21` - RSI indicators
- `stoch_k_14` - Stochastic oscillator

### Volume (Safe)
- `volume_ma_ratio_10/20` - Volume vs moving average
- `volume_std_10` - Volume volatility

### Time Features (Safe)
- `hour_sin`, `hour_cos` - Time of day encoding
- `dow_sin`, `dow_cos` - Day of week encoding

---

## 5. ACTIVE STRATEGY CONFIGURATION

```python
@dataclass
class StrategyConfig:
    # Entry filters (NOT optimized - see Phase 10 TODO)
    min_vol_expansion_prob: float = 0.50  # Threshold for vol filter
    min_breakout_prob: float = 0.50       # Threshold for breakout filter

    # Position sizing
    base_contracts: int = 1
    max_contracts: int = 3

    # Dynamic exits
    tp_atr_mult_base: float = 2.0         # Take profit = 2x ATR
    sl_atr_mult_base: float = 1.0         # Stop loss = 1x ATR
    max_holding_bars: int = 20            # ~100 minutes max hold

    # Trading costs (REALISTIC)
    commission_per_side: float = 1.29     # NinjaTrader rate
    slippage_ticks: float = 0.5           # Conservative RTH

    # Walk-forward (for model training)
    train_days: int = 60                  # 60-day training window
    test_days: int = 5                    # 5-day test window
    embargo_bars: int = 20                # 20-bar embargo (~100 min)
```

### Threshold Optimization Status

| Parameter | Current | Optimized? | Notes |
|-----------|---------|------------|-------|
| `min_vol_expansion_prob` | 0.50 | **NO** | Needs grid search |
| `min_breakout_prob` | 0.50 | **NO** | Needs grid search |
| `tp_atr_mult_base` | 2.0 | **NO** | Based on literature |
| `sl_atr_mult_base` | 1.0 | **NO** | Based on literature |

**TODO**: Run parameter optimization using walk-forward validation.

---

## 6. VALIDATED RESULTS

### In-Sample (2023-2024)

| Metric | Value |
|--------|-------|
| Net P&L | **$209,351** |
| Total Trades | 4,560 |
| Win Rate | 39.9% |
| Profit Factor | 1.29 |
| Sharpe Ratio | 3.22 |

### Out-of-Sample (2020-2022) - VALIDATED

| Metric | Value |
|--------|-------|
| Net P&L | **$496,380** |
| Total Trades | 9,481 |
| Win Rate | 40.4% |
| Profit Factor | 1.28 |
| Sharpe Ratio | 3.09 |

### Forward Test (2025) - VALIDATED

| Metric | Value |
|--------|-------|
| Net P&L | **$57,394** |
| Total Trades | 1,187 |
| Win Rate | 39.5% |
| Profit Factor | 1.24 |
| Sharpe Ratio | 2.66 |

**Total Validated Edge Across All Periods: $763,125**

### Model Performance

| Model | In-Sample AUC | OOS AUC | Assessment |
|-------|---------------|---------|------------|
| Vol Expansion | 0.84 | 0.79 | Strong |
| Breakout High/Low | 0.72 | 0.73 | Consistent |
| ATR Forecast R² | 0.36 | 0.30 | Good |

---

## 7. PHASE COMPLETION STATUS

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Environment Setup | COMPLETE |
| Phase 2 | Data Collection | COMPLETE |
| Phase 3 | Feature Engineering | COMPLETE (with corrections) |
| Phase 4 | Initial Models | DEPRECATED (leaky features) |
| Phase 5 | Research & Literature | COMPLETE |
| Phase 6 | Advanced Features | COMPLETE |
| Phase 7 | Walk-Forward Validation | COMPLETE |
| Phase 8 | Multi-Target Pivot | **BREAKTHROUGH** |
| Phase 9 | Vol Breakout Strategy | COMPLETE |
| Phase 10 | Production Readiness | **IN PROGRESS** |

### Phase 10 TODO

- [ ] **Optimize entry thresholds** (vol_prob, breakout_prob) - Script ready, needs faster machine
- [ ] **Optimize exit parameters** (TP/SL multipliers)
- [ ] Run Monte Carlo simulation (1000+ runs)
- [x] Out-of-sample validation (PASSED: +$496K)
- [x] Download 2025 data for true forward test (PASSED: +$57K)
- [ ] NinjaTrader ONNX integration
- [ ] Paper trading validation

---

## 8. CLEANUP RECOMMENDATIONS

### Files to Delete (Optional)

```
# Old models with leaky features
data/models/models/xgboost_20251203_220559.pkl
data/models/models/randomforest_20251203_220559.pkl
data/models/models/features_20251203_220559.json
data/models/lightgbm_5min_20251203_231817.txt
data/models/lstm_20251203_232323.pt
data/models/gru_20251203_232323.pt

# Old backtest results (invalid)
data/backtest_results/trades_lightgbm_20251204_094512.csv
data/backtest_results/trades_xgboost_20251204_094553.csv
data/backtest_results/metrics_lightgbm_20251204_094512.json
data/backtest_results/metrics_xgboost_20251204_094553.json
```

### Code Files - Keep but Mark Deprecated

| File | Status |
|------|--------|
| `feature_engineering/advanced_targets.py` | DEPRECATED - contains leaky features |
| `run_comprehensive_model_test.py` | DEPRECATED - uses old models |
| `run_validated_backtest.py` | DEPRECATED - uses old features |
| `run_backtest_analysis.py` | DEPRECATED - uses old pipeline |

---

## 9. QUICK REFERENCE

### To Run Current Strategy Backtest

```bash
python src/python/strategy/volatility_breakout_strategy.py
```

### To Run OOS Validation

```bash
python src/python/run_oos_backtest.py
```

### To Run QC Check

```bash
python src/python/run_qc_check.py
```

### Key Insight

**The breakthrough**: Don't predict direction (impossible, AUC 0.50). Instead predict:
1. **WHEN** to trade - volatility expansion (AUC 0.84)
2. **WHERE** price will go - new high/low (AUC 0.72)
3. **HOW MUCH** it will move - ATR forecast (R² 0.36)

---

*Last Updated: 2025-12-04*
*Maintained by: SKIE_Ninja Development Team*
