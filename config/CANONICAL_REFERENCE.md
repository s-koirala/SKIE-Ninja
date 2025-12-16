# SKIE_Ninja Canonical Reference

**Created**: 2025-12-04
**Updated**: 2025-12-15
**Purpose**: Single source of truth for active files, models, and knowledge

---

## AUDIT FINDINGS (2025-12-15) - **ALL CRITICAL ISSUES FIXED**

**A comprehensive audit was performed on 2025-12-15. All critical issues have been resolved.**
**See `docs/AUDIT_REPORT.md` for full details.**

### Socket Bridge Issues - ALL FIXED

| Issue | Severity | File | Status |
|-------|----------|------|--------|
| VIX buffer lag (T-2 vs T-1) | CRITICAL | ninja_signal_server.py:175 | **FIXED** |
| Feature count mismatch | CRITICAL | ninja_signal_server.py | **FIXED** |
| VIX percentile broken in live | HIGH | ninja_signal_server.py:179-185 | **FIXED** |
| No feature validation | HIGH | ninja_signal_server.py | **FIXED** |

**✓ READY FOR PAPER TRADING** - All critical issues resolved (2025-12-15)

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
| `src/python/strategy/ensemble_strategy.py` | **PRODUCTION** - Ensemble (recommended) | **ACTIVE** |
| `src/python/feature_engineering/multi_target_labels.py` | 73-target generator (vol, breakout, ATR) | **ACTIVE** |
| `src/python/run_oos_backtest.py` | Out-of-sample validation (baseline) | **ACTIVE** |
| `src/python/run_ensemble_oos_backtest.py` | Out-of-sample validation (ensemble) | **ACTIVE** |
| `src/python/run_2025_forward_test.py` | Forward test (baseline) | **ACTIVE** |
| `src/python/run_ensemble_2025_forward_test.py` | Forward test (ensemble - use this) | **ACTIVE** |
| `src/python/run_threshold_optimization.py` | Parameter grid search | **ACTIVE** |
| `src/python/run_qc_check.py` | Quality control validation | **ACTIVE** |

### Feature Engineering (Active)

| File | Purpose | Status |
|------|---------|--------|
| `feature_engineering/volatility_regime.py` | VIX + regime detection | **ACTIVE** |
| `feature_engineering/triple_barrier.py` | Triple barrier labeling | **ACTIVE** |
| `feature_engineering/multi_timeframe_features.py` | MTF analysis (15m, 1h, 4h) | **ACTIVE** |
| `feature_engineering/enhanced_cross_market.py` | Real Databento cross-market | **ACTIVE** |
| `feature_engineering/social_news_sentiment.py` | Twitter/News/Reddit sentiment | **ACTIVE** |
| `feature_engineering/enhanced_feature_pipeline.py` | Unified feature pipeline | **ACTIVE** |
| `run_enhanced_feature_qc.py` | Enhanced QC validation | **ACTIVE** |

### Shared Utilities (NEW - Phase C)

| File | Purpose | Status |
|------|---------|--------|
| `feature_engineering/shared/technical_utils.py` | TR, ATR, RSI, BB, MACD | **NEW** ✓ |
| `feature_engineering/shared/returns_utils.py` | Return calculations | **NEW** ✓ |
| `feature_engineering/shared/volume_utils.py` | Volume features, VWAP | **NEW** ✓ |
| `feature_engineering/shared/temporal_utils.py` | Time encoding | **NEW** ✓ |

**Note:** Shared utilities consolidate 11+ duplicate TR, 10+ RSI, 6+ ATR implementations.

### NinjaTrader Deployment (NEW - Phase 15)

| File | Purpose | Status |
|------|---------|--------|
| `src/python/deployment/ninja_signal_server.py` | Python TCP signal server | **ACTIVE** ✓ |
| `src/ninjatrader/SKIENinjaStrategy.cs` | NinjaScript client strategy | **ACTIVE** ✓ |
| `docs/DEPLOYMENT_INFRASTRUCTURE.md` | Production deployment guide | **ACTIVE** |
| `requirements.txt` | Python dependencies | **NEW** ✓ |

**✓ Socket Bridge ready for paper trading** - All critical issues fixed (2025-12-15)

### Documentation (Active)

| File | Purpose | Status |
|------|---------|--------|
| `HANDOFF.md` | **START HERE** - Next session handoff | **ACTIVE** |
| `CHANGELOG.md` | **NEW** Version history and changes | **ACTIVE** |
| `docs/BEST_PRACTICES.md` | Lessons learned & anti-patterns | **ACTIVE** |
| `docs/AUDIT_REPORT.md` | Comprehensive audit findings | **ACTIVE** |
| `docs/DATA_DRIVEN_DECISIONS.md` | Parameter justification & overfitting detection | **ACTIVE** |
| `docs/VALIDATION_REPORT.md` | Stress testing & sensitivity analysis | **ACTIVE** |
| `config/project_memory.md` | Project decision log | **ACTIVE** |
| `config/CANONICAL_REFERENCE.md` | This file - canonical reference | **ACTIVE** |
| `research/04_multi_target_prediction_strategy.md` | Multi-target strategy design | **ACTIVE** |
| `research/05_sentiment_strategy_plan.md` | Sentiment strategy & ensemble plan | **ACTIVE** |
| `docs/methodology/BACKTEST_METHODOLOGY.md` | Backtest methodology | **ACTIVE** |

### Quality Control & Testing (Phase 15)

| File | Purpose | Status |
|------|---------|--------|
| `src/python/quality_control/overfitting_detection.py` | DSR, CSCV, PSR implementations | **ACTIVE** ✓ |
| `src/python/run_overfitting_assessment.py` | Run comprehensive overfitting tests | **ACTIVE** ✓ |
| `src/python/run_window_optimization.py` | Data-driven train/test window selection | **NEW** ✓ |
| `src/python/run_embargo_analysis.py` | Autocorrelation-based embargo justification | **NEW** ✓ |
| `tests/test_critical_functions.py` | Pytest suite for critical functions | **ACTIVE** ✓ |

### Configuration (Updated)

| File | Purpose | Status |
|------|---------|--------|
| `config/api_keys.py` | **SECURE** - Loads keys from environment | **UPDATED** ✓ |
| `config/api_keys.env.template` | Template for API keys | **ACTIVE** |

**API Key Security**: Copy `api_keys.env.template` to `api_keys.env` and fill in your keys. Never commit `api_keys.env`.

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

### Multi-Timeframe Features (NEW - Safe)
- `htf_15m_*`, `htf_1h_*`, `htf_4h_*` - Higher timeframe indicators
- `mtf_trend_alignment` - Cross-timeframe trend agreement
- `mtf_all_bullish`, `mtf_all_bearish` - Alignment signals
- `mtf_rsi_avg`, `mtf_vol_expansion_score` - Aggregated HTF metrics

### Cross-Market Features (NEW - Safe)
- `corr_NQ/YM/GC/CL/ZN_*` - Rolling correlations with related markets
- `lead_*_lag*` - Lead/lag relationships (market leads ES?)
- `es_nq_spread_*` - Tech vs Broad market spread
- `stock_bond_spread_*` - Risk-on/Risk-off indicators
- `vix_*` - VIX-based sentiment features
- `risk_off_score` - Combined regime indicator

### Social/News Sentiment Features (NEW - Safe)
- `social_sentiment_*min` - Aggregated sentiment (5/15/30/60/240 min windows)
- `social_count_*min` - News/tweet volume
- `social_sentiment_momentum` - Short vs long-term sentiment
- `social_extreme_bullish/bearish` - Extreme sentiment signals

**CRITICAL**: All new features follow strict leakage prevention:
- HTF features use completed bars only (lagged)
- Cross-market data aligned with proper lag
- Sentiment uses data from BEFORE each bar (min 5-minute lag)
- No `shift(-N)` or `center=True` patterns

---

## 5. ACTIVE STRATEGY CONFIGURATION

```python
@dataclass
class StrategyConfig:
    # Entry filters (OPTIMIZED - Phase 12 Complete)
    min_vol_expansion_prob: float = 0.40  # Optimized (was 0.50)
    min_breakout_prob: float = 0.45       # Optimized (was 0.50)

    # Position sizing
    base_contracts: int = 1
    max_contracts: int = 3

    # Dynamic exits (OPTIMIZED)
    tp_atr_mult_base: float = 2.5         # Optimized (was 2.0)
    sl_atr_mult_base: float = 1.25        # Optimized (was 1.0)
    max_holding_bars: int = 20            # ~100 minutes max hold

    # Trading costs (REALISTIC)
    commission_per_side: float = 1.29     # NinjaTrader rate
    slippage_ticks: float = 0.5           # Conservative RTH

    # Walk-forward (for model training)
    train_days: int = 60                  # 60-day training window
    test_days: int = 5                    # 5-day test window
    embargo_bars: int = 20                # 20-bar embargo (~100 min)
```

### Threshold Optimization Status (Phase 12 - COMPLETE)

| Parameter | Default | Optimized | Improvement |
|-----------|---------|-----------|-------------|
| `min_vol_expansion_prob` | 0.50 | **0.40** | More trades captured |
| `min_breakout_prob` | 0.50 | **0.45** | Better signal quality |
| `tp_atr_mult_base` | 2.0 | **2.5** | Larger winners |
| `sl_atr_mult_base` | 1.0 | **1.25** | Reduced whipsaws |

**Optimization Results**: +96% improvement in Net P&L ($73K → $143K in-sample)

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

**Vol Breakout Baseline Total: $763,125** (IS + OOS + Forward)

### Ensemble Strategy Results (PRODUCTION)

The ensemble strategy combines vol breakout with VIX-based sentiment features using the "either" method (enter if either technical OR sentiment vol model predicts expansion).

| Period | Vol Breakout | Ensemble | Improvement |
|--------|-------------|----------|-------------|
| In-Sample (2023-24) | $209,351 | **$224,813** | **+7.4%** |
| OOS (2020-22) | $496,380 | **$502,219** | **+1.2%** |
| Forward (2025) | $57,394 | **$59,847** | **+4.3%** |

**Ensemble Total Validated: $786,879** (recommended production strategy)

### Model Performance

| Model | In-Sample AUC | OOS AUC | Assessment |
|-------|---------------|---------|------------|
| Vol Expansion (Tech) | 0.84 | 0.79 | Strong |
| Vol Expansion (Sent) | 0.77 | 0.65 | Moderate |
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
| Phase 10 | OOS & Forward Test | **COMPLETE** |
| Phase 11 | Sentiment Strategy & Ensemble | **COMPLETE** (+7.4% improvement) |
| Phase 12 | Threshold Optimization | **COMPLETE** (+96% improvement) |
| Phase 13 | Monte Carlo Validation | **COMPLETE** (100% prob profit) |

### Phase 10-13 Summary (All Complete)

- [x] **Optimize entry thresholds** (vol_prob, breakout_prob) - DONE (0.40, 0.45)
- [x] **Optimize exit parameters** (TP/SL multipliers) - DONE (2.5x, 1.25x)
- [x] Run Monte Carlo simulation (10,000 iterations) - PASSED
- [x] Out-of-sample validation (PASSED: +$496K)
- [x] Download 2025 data for true forward test (PASSED: +$57K)
- [ ] NinjaTrader ONNX integration
- [ ] Paper trading validation

### Phase 11 COMPLETE (Sentiment Strategy)

- [x] Phase 1: Collect and validate sentiment data sources (VIX-based)
- [x] Phase 2: Engineer sentiment features with leakage prevention
- [x] Phase 3: Build independent sentiment-only strategy (tested, not profitable standalone)
- [x] Phase 4-5: Skipped (sentiment not profitable as standalone)
- [x] Phase 6: Create ensemble with volatility breakout strategy (**+7.4% in-sample, +1.2% OOS**)
- [x] Phase 7: Document and commit to GitHub

**Key Finding**: Sentiment predicts WHEN (vol expansion AUC 0.77) but not WHICH WAY.
Best as filter for vol breakout, not standalone strategy.

**Files Created**:
- `src/python/strategy/ensemble_strategy.py` - Main ensemble implementation
- `src/python/strategy/sentiment_strategy.py` - Standalone sentiment (for testing)
- `src/python/data_collection/historical_sentiment_loader.py` - VIX data loader
- `src/python/run_ensemble_oos_backtest.py` - OOS validation script
- `research/05_sentiment_strategy_plan.md` - Phase 11 planning document

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

### To Generate Enhanced Features (NEW)

```bash
python src/python/feature_engineering/enhanced_feature_pipeline.py
```

### To Run Enhanced Feature QC (NEW)

```bash
python src/python/run_enhanced_feature_qc.py
```

### Key Insight

**The breakthrough**: Don't predict direction (impossible, AUC 0.50). Instead predict:
1. **WHEN** to trade - volatility expansion (AUC 0.84)
2. **WHERE** price will go - new high/low (AUC 0.72)
3. **HOW MUCH** it will move - ATR forecast (R² 0.36)

---

*Last Updated: 2025-12-15*
*Maintained by: SKIE_Ninja Development Team*
