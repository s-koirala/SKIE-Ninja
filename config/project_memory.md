# SKIE_Ninja Project Memory Base

**Created**: 2025-11-30
**Last Updated**: 2025-12-04
**Status**: Phase 7 - ML Model Development (RESEARCH IMPLEMENTATION COMPLETE)

---

## SESSION 6 UPDATE (2025-12-04) - RESEARCH IMPLEMENTATION & VALIDATION COMPLETE

### Implementation Status

All advanced features from the literature review have been successfully implemented and validated.

| Module | File | Lines | Status | Literature Reference |
|--------|------|-------|--------|---------------------|
| **Triple Barrier** | `feature_engineering/triple_barrier.py` | 463 | ✅ Validated | Lopez de Prado (2018) Ch. 3 |
| **Meta-Labeling** | `models/meta_labeling.py` | 638 | ✅ Validated | Lopez de Prado (2018) Ch. 3 |
| **Volatility Regime** | `feature_engineering/volatility_regime.py` | 728 | ✅ Validated | PLOS One 2024, Macrosynergy |
| **FinBERT Sentiment** | `feature_engineering/finbert_sentiment.py` | 660 | ✅ Implemented | ACM 2024, ScienceDirect 2024 |
| **Temporal Fusion Transformer** | `models/temporal_fusion_transformer.py` | 736 | ✅ Implemented | IEEE 2022 |

**Total New Code**: 3,225 lines implementing 5 advanced ML modules

### Validation Results (2025-12-04 12:41)

```
============================================================
OVERALL VALIDATION STATUS
============================================================
  Total Checks: 13
  Passed: 13
  Failed: 0
  Pass Rate: 100.0%
  STATUS: VALIDATION SUCCESSFUL
============================================================
```

#### 1. Triple Barrier Labeling - 4/4 Checks Passed

| Metric | Value | Reference |
|--------|-------|-----------|
| Label Distribution | Long: 38.4%, Short: 60.5%, Flat: 1.2% | Lopez de Prado (2018) |
| Avg Holding Period | 5.7 bars (~28 minutes) | ATR-adjusted barriers |
| Barrier Types | Upper: 10,289, Lower: 22,209, Vertical: 5,788 | |
| Avg Return | 0.0055% per trade | |
| Features Generated | 24 across 4 configurations | |

#### 2. Meta-Labeling - 4/4 Checks Passed

| Metric | Value | Improvement |
|--------|-------|-------------|
| Meta AUC-ROC | 0.6453 | Above random (0.5) |
| Precision (Primary) | 0.5422 | Baseline |
| Precision (With Meta) | 0.6767 | **+13.45%** |
| F1 Score | 0.6474 | |
| Trades Filtered | 50.3% | Quality over quantity |

**Key Insight**: Meta-labeling improved precision by 13.45% by filtering low-confidence trades.

#### 3. Volatility Regime Detection - 5/5 Checks Passed

| Feature Type | Count | Description |
|--------------|-------|-------------|
| Realized Volatility | 19 | Close-to-close, Parkinson, ATR |
| Regime Classification | 7 | Rule-based + GMM + HMM |
| Total | 26 | |

**Regime Distribution (GMM)**:
- Regime 2 (Low-Med Vol): 43.0%
- Regime 1 (Low Vol): 37.9%
- Regime 3 (High Vol): 16.9%
- Regime 0 (Extreme Vol): 2.2%

**HMM Transition Matrix** (rows = from, cols = to):
```
Regime 0: [0.75 0.25 0.00 0.00]  ← Stable extreme regime
Regime 1: [0.04 0.93 0.02 0.01]  ← Very stable low vol
Regime 2: [0.00 0.04 0.87 0.09]  ← Stable low-med vol
Regime 3: [0.00 0.07 0.20 0.73]  ← Moderate stability high vol
```

#### 4. Comprehensive Feature Integration

| Category | Features | Validation |
|----------|----------|------------|
| Triple Barrier | 6 | ✅ No look-ahead bias |
| Realized Volatility | 12 | ✅ Proper lag |
| ATR Features | 7 | ✅ Past data only |
| Regime Features | 7 | ✅ Current + lagged |
| **Total** | **32** | ✅ All validated |

**Top 10 Correlated Features with Target**:
1. `atr_21_pct`: +0.0305
2. `rv_21`: +0.0302
3. `atr_14_pct`: +0.0288
4. `parkinson_vol_21`: +0.0267
5. `atr_21`: +0.0242

### Files Created (Session 6)

| File | Purpose | Lines |
|------|---------|-------|
| `src/python/feature_engineering/triple_barrier.py` | Triple Barrier labeling | 463 |
| `src/python/feature_engineering/volatility_regime.py` | VIX + Regime detection | 728 |
| `src/python/feature_engineering/finbert_sentiment.py` | FinBERT NLP sentiment | 660 |
| `src/python/models/meta_labeling.py` | Meta-labeling for bet sizing | 638 |
| `src/python/models/temporal_fusion_transformer.py` | TFT architecture | 736 |
| `src/python/run_advanced_feature_validation.py` | Comprehensive test suite | 500+ |
| `research/03_advanced_strategy_research.md` | Literature review | 698 |

### Git Commits (Session 6)

- `a289243`: Research phase implementation - 5 advanced ML modules (3,309 lines)
- `ecf8ace`: Previous session commits merged

### Next Steps (Priority Order)

1. **Immediate**: Run full backtest with new Triple Barrier + Meta-labeling pipeline
2. **Short-term**: Add VIX data feed for real-time regime detection
3. **Medium-term**: Integrate FinBERT with live news API (Polygon.io or Alpha Vantage)
4. **Long-term**: Evaluate TFT vs LightGBM ensemble

### Validation Results Location

```
data/validation_results/feature_validation_20251204_124137.txt
```

---

## CRITICAL SESSION UPDATE (2025-12-04 Session 4) - LOOK-AHEAD BIAS CONFIRMED

### CATASTROPHIC FINDING: Zero Predictive Power Without Leaky Features

**The entire model's predictive capability was an illusion caused by look-ahead bias.**

#### Corrected Feature Test Results

| Model | Metric | With Leaky Features | Corrected Features | Change |
|-------|--------|---------------------|-------------------|--------|
| **LightGBM** | Win Rate | 86.0% | 45.1% | -40.9% |
| | Sharpe Ratio | 42.68 | -0.34 | -43.02 |
| | Profit Factor | 18.17 | 0.96 | -17.21 |
| | Net P&L | +$712,475 | -$4,953 | -$717,428 |
| **XGBoost** | Win Rate | 86.3% | 46.0% | -40.3% |
| | Sharpe Ratio | 42.63 | -1.88 | -44.51 |
| | Profit Factor | 19.40 | 0.85 | -18.55 |
| | Net P&L | +$718,138 | -$24,548 | -$742,686 |
| **LSTM** | AUC-ROC | 66.26% | ~49% | -17% |

### Key Conclusions

1. **ALL predictive power came from look-ahead bias** - Models perform at random chance without it
2. **Current feature set has NO edge** - Technical indicators and microstructure features provide zero value
3. **Strategy requires complete redesign** - New features needed from market research
4. **QC system works correctly** - Successfully caught suspicious metrics

### Fixes Applied

1. **Sharpe/Sortino Calculation** - Now uses daily returns with sqrt(252) annualization
2. **Pyramiding Features** - Changed from `shift(-N)` to `shift(N)` (past data only)
3. **DDCA Features** - Measures historical pattern effectiveness instead of future success
4. **Pivot Detection** - Confirms pivots with delay (past data only)
5. **Feature Rankings** - Regenerated with corrected features

### Status
- [x] Sharpe ratio bug fixed
- [x] Look-ahead bias eliminated from advanced_targets.py
- [x] Feature rankings regenerated
- [x] Model re-tested with corrected features
- [x] Feature engineering research phase (COMPLETE)
- [ ] Strategy redesign implementation (NEXT)

---

## SESSION 5 UPDATE (2025-12-04) - RESEARCH PHASE COMPLETE

### Comprehensive Literature Review Completed

A thorough literature review was conducted covering:
1. **ML Features for Futures Prediction** - Order flow imbalance, cross-asset features
2. **Sentiment Analysis** - FinBERT, Twitter/X, Reddit WSB (contrarian indicator)
3. **Target Engineering** - Triple Barrier Method, Meta-labeling (Lopez de Prado)
4. **Arbitrage Strategies** - ES-SPY basis, statistical arbitrage
5. **Volatility Regime Detection** - VIX features, Hidden Markov Models
6. **Novel ML Architectures** - Temporal Fusion Transformer, TFT-GNN
7. **Reinforcement Learning** - DQN, PPO, DDPG for trading

### Key Research Findings

| Category | Approach | Priority | Expected Impact |
|----------|----------|----------|-----------------|
| Target Engineering | Triple Barrier + Meta-labeling | HIGH | Improves F1, realistic sizing |
| Volatility Features | VIX regime detection | HIGH | Strategy adaptation |
| Sentiment | FinBERT on news | MEDIUM | New information source |
| Novel ML | Temporal Fusion Transformer | MEDIUM | Better sequential patterns |
| Reinforcement Learning | Ensemble DRL | LOW | Long-term research |

### Research Document Created

Full research documentation: `research/03_advanced_strategy_research.md`

Includes:
- Academic references (14+ papers from 2021-2024)
- Implementation code examples
- Realistic performance expectations
- 8-week implementation roadmap

### Next Steps (Priority Order)

1. **Week 1-2**: Implement Triple Barrier labeling + Meta-labeling
2. **Week 3-4**: Add VIX-based volatility regime features
3. **Week 5-6**: Integrate FinBERT sentiment analysis
4. **Week 7-8**: Evaluate Temporal Fusion Transformer

### Realistic Performance Targets

Based on literature review:

| Metric | Target Range | Current (Corrected) |
|--------|-------------|---------------------|
| Sharpe Ratio | 1.0 - 1.5 | -0.34 |
| Win Rate | 52% - 55% | 45.1% |
| Profit Factor | 1.2 - 1.3 | 0.96 |
| Max Drawdown | < 20% | N/A |

**Any backtest showing Sharpe > 3 or Win Rate > 60% should be treated with suspicion.**

---

## Previous Session Progress (2025-12-04) - SESSION 3 COMPLETE

### Session 3 Goals (COMPLETED)
**Objective**: Test current models using full metrics backtesting framework and QC checks prior to retraining LSTM/GRU with purged k-fold CV.

**Workflow**:
1. [x] Run comprehensive backtest on LightGBM (84.21% AUC)
2. [x] Run comprehensive backtest on XGBoost (84.07% AUC)
3. [x] Execute quality control validation framework
4. [x] Run baseline LSTM/GRU backtest for comparison
5. [x] Retrain LSTM with purged k-fold CV
6. [x] Retrain GRU with purged k-fold CV
7. [x] Compare pre/post purged CV results
8. [x] Update documentation with findings

### CRITICAL FINDING: Sharpe Ratio Calculation Bug

**The backtest framework contains a critical bug in risk-adjusted metric calculations.**

#### Reported vs Expected Metrics
| Metric | Reported | Expected | Issue |
|--------|----------|----------|-------|
| Sharpe Ratio | 42.68 | 1-3 | ~50x inflated |
| Sortino Ratio | 78.50 | 2-5 | ~30x inflated |
| Win Rate | 86.0% | 50-65% | Suspiciously high |
| Worst Day P&L | +$92.50 | Should have losses | No losing days |

#### Root Cause (Lines 1106-1108 in comprehensive_backtest.py)
```python
# INCORRECT: Annualizes per-trade returns
trades_per_year = metrics.trades_per_day * 250  # = 2,737.5
sharpe = (avg_return / std_return) * np.sqrt(trades_per_year)  # 52x multiplier!
```

#### Correct Implementation
```python
# CORRECT: Should use daily returns
avg_daily_return = np.mean(daily_pnls)
std_daily_return = np.std(daily_pnls)
sharpe = (avg_daily_return / std_daily_return) * np.sqrt(252)
```

#### Industry Benchmarks (Literature Review)
- Retail algo trading: Sharpe 1.0-2.0 is good
- Quant hedge funds: Sharpe 2.0-3.0 is excellent
- HFT strategies: Sharpe 4-10 is realistic
- **No sustainable strategy exceeds Sharpe 2-3 long-term**

Sources:
- QuantStart: https://www.quantstart.com/articles/Sharpe-Ratio-for-Algorithmic-Trading-Performance-Measurement/
- Quant SE: https://quant.stackexchange.com/questions/21120/what-is-an-acceptable-sharpe-ratio-for-a-prop-desk

### Model Performance Summary (Session 3)

| Model | AUC-ROC | Accuracy | Method | Status |
|-------|---------|----------|--------|--------|
| LightGBM | 83.50% | 74.00% | Walk-Forward | Metrics need correction |
| XGBoost | 83.42% | 73.84% | Walk-Forward | Metrics need correction |
| LSTM | 66.26% | 62.17% | Purged K-Fold | Realistic metrics |
| GRU | 66.73% | 62.58% | Purged K-Fold | Realistic metrics |

### Files Generated (Session 3)
- `data/backtest_results/trades_lightgbm_20251204_094512.csv`
- `data/backtest_results/metrics_lightgbm_20251204_094512.json`
- `data/backtest_results/trades_xgboost_20251204_094553.csv`
- `data/backtest_results/metrics_xgboost_20251204_094553.json`
- `data/backtest_results/purged_cv_results_lstm_20251204_094953.json`
- `data/backtest_results/purged_cv_results_gru_20251204_095503.json`
- `data/backtest_results/model_comparison.json`
- `data/backtest_results/backtest_investigation_report.md`

### Action Items Before Production
1. [x] Fix Sharpe/Sortino ratio calculation in comprehensive_backtest.py
2. [ ] Re-run backtests with corrected metrics and clean features
3. [x] Investigate feature engineering for potential look-ahead bias - **CRITICAL ISSUES FOUND**
4. [x] Add QC check for "no losing days" anomaly
5. [ ] Remove leaky features from feature pipeline
6. [ ] Paper trade before live deployment

### CRITICAL: Look-Ahead Bias in Top Features (Session 3 Continued)

**ROOT CAUSE OF 86% WIN RATE IDENTIFIED:**

The top-ranked features (`pyramid_rr_5/10/20`) use `shift(-N)` which accesses FUTURE bars:

```python
# advanced_targets.py lines 81-86 - LEAKY CODE:
future_max = high.rolling(horizon).max().shift(-horizon)  # FUTURE DATA!
future_min = low.rolling(horizon).min().shift(-horizon)   # FUTURE DATA!
pyramid_rr = (future_max - close) / (close - future_min)  # RANK #1 FEATURE!
```

**Features That Must Be Removed:**
- `pyramid_rr_5/10/20` - Top 3 features, all leaky
- `pyramid_long/short_*` - Uses future MFE/MAE
- `ddca_buy/sell_success_*` - Uses `close.shift(-horizon)`
- `pivot_high/low_*` - Uses forward-looking window

**Impact:** Model AUC will drop from ~84% to ~55-65% when leaky features removed.

**Git Status**: Committed (d808220), 2 commits ahead of origin

---

### Latest Accomplishments (Session 2)
1. ✅ **Purged K-Fold CV for RNNs** - Addresses overfitting in LSTM/GRU models
   - Implements de Prado (2018) approach with purge and embargo periods
   - Prevents data leakage from feature lookbacks and target labels
   - Uses aggressive regularization (50% dropout, weight decay 1e-4)
   - File: `src/python/models/purged_cv_rnn_trainer.py`

2. ✅ **Comprehensive Walk-Forward Backtesting** - Full metrics framework
   - P&L analysis (gross, net, commission, slippage)
   - Drawdown metrics (max DD, duration, avg DD)
   - Trade duration (bars held, time in minutes)
   - MFE/MAE analysis (max favorable/adverse excursion)
   - Risk-adjusted returns (Sharpe, Sortino, Calmar)
   - KPIs (profit factor, payoff ratio, expectancy)
   - File: `src/python/backtesting/comprehensive_backtest.py`

3. ✅ **RTH Enforcement** - All trades during Regular Trading Hours only
   - 9:30 AM - 4:00 PM Eastern Time
   - Automatic filtering in backtest framework
   - Better liquidity, tighter spreads

4. ✅ **Data Leakage Detection** - Automatic checks before training
   - Detects features with >95% correlation to target
   - Warns on suspicious feature names (future, target, next_)
   - Integrated into backtesting pipeline

5. ✅ **Comprehensive Report Generator** - Full backtest reports with:
   - Trade summary (total, wins, losses, long/short)
   - P&L breakdown (gross, commission, slippage, net)
   - Win/loss statistics (avg win/loss, max win/loss)
   - KPIs (profit factor, payoff ratio, expectancy)
   - Drawdown analysis (max DD, duration)
   - Trade duration (bars, minutes, min/max)
   - Risk-adjusted returns (Sharpe, Sortino, Calmar)
   - Time analysis (daily P&L, winning/losing days)

6. ✅ **Runner Script** - `run_backtest_analysis.py` for easy execution

### RNN Overfitting Mitigation (Literature-Based)
Based on de Prado (2018), Fischer & Krauss (2018), and Grinsztajn et al. (2022):

| Technique | Implementation | Rationale |
|-----------|---------------|-----------|
| **Purged K-Fold CV** | 200-bar purge between train/test | Prevents feature leakage |
| **Embargo Period** | 42-bar gap after test set | Prevents target leakage |
| **High Dropout** | 50% (vs typical 30%) | Reduces overfitting |
| **Weight Decay** | 1e-4 (vs 1e-5) | Stronger L2 regularization |
| **Reduced Model Size** | 64 hidden units, 1 layer | Less capacity = less overfitting |
| **Early Stopping** | 5 epochs patience | Quick termination |
| **Batch Normalization** | After LSTM/GRU | Stabilizes training |
| **Layer Normalization** | On input | Normalizes features |

### Backtest Metrics Available

| Category | Metrics |
|----------|---------|
| **Trade Counts** | Total, wins, losses, long, short, breakeven |
| **P&L** | Gross, net, commission, slippage |
| **Win/Loss** | Win rate, avg win, avg loss, max win, max loss |
| **KPIs** | Profit factor, payoff ratio, expectancy |
| **Drawdown** | Max DD ($), max DD (%), duration (days), avg DD |
| **Excursion** | Avg MFE, avg MAE, MFE/MAE in ticks |
| **Duration** | Avg/min/max bars held, avg/min/max time (minutes) |
| **Contracts** | Avg per trade, max per trade, total traded |
| **Risk-Adjusted** | Sharpe ratio, Sortino ratio, Calmar ratio |
| **Time** | Trading days, trades/day, daily P&L, best/worst day |
| **Consecutive** | Max wins streak, max losses streak |
| **Model** | Accuracy, AUC-ROC, F1 score |

### Previous Accomplishments (Session 1)
1. ✅ **Rolling Window Grid Optimization Complete** - Tested 15 configurations each for 5-min and 15-min
2. ✅ **Timeframe Decision: 5-MIN BARS** - Outperforms 15-min by ~1.3% AUC
3. ✅ **Optimal CV Config: Train=180d, Test=5d** - 83.30% AUC-ROC
4. ✅ Data resampler utility created for RTH filtering
5. ✅ Rolling window optimizer script created
6. ✅ **LightGBM Trained** - 84.21% AUC-ROC (best model yet!)
7. ✅ **LSTM/GRU Deep Learning** - 65.28%/65.60% AUC (underperforms gradient boosting)
8. ✅ Deep learning trainer module created (PyTorch-based)
9. ✅ **Walk-Forward Backtesting Framework** - Comprehensive trade simulation
10. ✅ **RNN Hyperparameter Optimizer** - Grid search CV for LSTM/GRU with literature review

### New Modules Created (2025-12-04)

#### Walk-Forward Backtester (`src/python/backtesting/`)
- **walk_forward_backtest.py** - Complete backtesting framework
- Features:
  - Walk-forward validation (train, validate, trade)
  - Trade metrics: entry/exit times, P&L, drawdown, run-up
  - Summary statistics: Win rate, Sharpe, Sortino, Calmar, profit factor
  - Multi-model support (LightGBM, XGBoost, LSTM, GRU)
  - ES futures contract specs ($50/point)
  - Comprehensive reporting

#### RNN Hyperparameter Optimizer (`src/python/models/rnn_hyperparameter_optimizer.py`)
- Grid search CV for LSTM/GRU with time-series aware validation
- Includes literature review on RNN overfitting in financial prediction
- Key references:
  - Fischer & Krauss (2018) - LSTM for financial prediction
  - Shwartz-Ziv & Armon (2022) - Why gradient boosting beats deep learning on tabular data
  - Grinsztajn et al. (2022) - Tree-based models vs deep learning

### Why RNNs Underperform on This Data
Based on literature research:
1. **Low signal-to-noise ratio** - Financial data is inherently noisy
2. **Pre-engineered features** - Our 75 features already capture temporal patterns
3. **Tabular data advantage** - Gradient boosting excels when data lacks spatial/temporal structure
4. **Overfitting risk** - RNNs have more parameters, prone to overfitting on small samples

### Model Comparison (5-min RTH data)

| Model | AUC-ROC | Accuracy | F1 Score | Precision | Recall |
|-------|---------|----------|----------|-----------|--------|
| **LightGBM** | **84.21%** | 74.40% | 73.36% | 75.62% | 71.22% |
| XGBoost | 84.07% | 75.23% | 67.22% | 76.22% | 60.12% |
| GRU | 65.60% | 62.03% | 62.63% | 59.36% | 66.28% |
| LSTM | 65.28% | 62.00% | 60.99% | 60.13% | 61.88% |
| RandomForest | 76.83% | 71.07% | 63.13% | 68.40% | 58.62% |

**Key Finding**: Gradient boosting (LightGBM, XGBoost) significantly outperforms deep learning (LSTM, GRU) for tabular feature-based financial prediction. This is consistent with ML literature on tabular data.

### LightGBM Top Important Features
1. pyramid_rr_5 (92,069)
2. pyramid_rr_10 (11,025)
3. bars_in_session (3,995)
4. pyramid_rr_20 (3,044)
5. dist_to_support (592)
6. stoch_diff_14 (545)
7. dist_to_resistance (542)
8. estimated_sell_volume (511)
9. atr_20 (507)
10. rsi_dist_50_7 (459)

### Grid Optimization Results Summary

#### 5-MIN BARS (WINNER)
| Rank | Train Days | Test Days | AUC-ROC | Accuracy | Folds |
|------|------------|-----------|---------|----------|-------|
| 1 | **180** | **5** | **83.30%** | 73.70% | 61 |
| 2 | 180 | 10 | 83.27% | 73.47% | 30 |
| 3 | 180 | 20 | 83.20% | 73.43% | 15 |
| 4 | 120 | 10 | 83.09% | 73.38% | 36 |
| 5 | 120 | 5 | 83.06% | 73.33% | 73 |

#### 15-MIN BARS
| Rank | Train Days | Test Days | AUC-ROC | Accuracy | Folds |
|------|------------|-----------|---------|----------|-------|
| 1 | 180 | 5 | 82.03% | 72.18% | 60 |
| 2 | 180 | 20 | 82.01% | 71.87% | 15 |
| 3 | 120 | 5 | 81.83% | 72.08% | 72 |
| 4 | 180 | 10 | 81.78% | 71.74% | 30 |
| 5 | 120 | 10 | 81.66% | 71.70% | 36 |

### Key Finding: 5-Min Outperforms 15-Min
- **5-min best AUC: 83.30%** vs 15-min best: 82.03% (+1.27%)
- Consistent ~1-1.5% advantage across all train/test configurations
- 5-min provides 78 bars/day (RTH) vs 26 for 15-min = more granular signals
- Trade-off: 5-min = more signals but potentially more noise

---

## Previous Session Progress (2025-12-03)

### Accomplishments
1. ✅ **Model Training Complete** - XGBoost 84% AUC, RandomForest 77% AUC
2. ✅ **Feature Selection Complete** - 75 top features from 100+ candidates
3. ✅ Walk-forward cross-validation implemented
4. ✅ Model serialization with joblib
5. ✅ Feature importance analysis saved

### Model Performance Summary
| Model | AUC-ROC | Accuracy | Precision | Recall | F1 Score |
|-------|---------|----------|-----------|--------|----------|
| **XGBoost** | **84.07%** | 75.23% | 76.22% | 60.12% | 67.22% |
| RandomForest | 76.83% | 71.07% | 68.40% | 58.62% | 63.13% |

### Top 10 Predictive Features
1. `pyramid_rr_5` - Pyramiding reward-to-risk (5-bar)
2. `pyramid_rr_10` - Pyramiding reward-to-risk (10-bar)
3. `pyramid_rr_20` - Pyramiding reward-to-risk (20-bar)
4. `pivot_high_5_5` - 5-bar pivot high detection
5. `pivot_low_5_5` - 5-bar pivot low detection
6. `pivot_high_5_10` - 5/10 pivot high
7. `pivot_high_10_5` - 10/5 pivot high
8. `pivot_low_5_10` - 5/10 pivot low
9. `pivot_low_10_5` - 10/5 pivot low
10. `pivot_high_10_10` - 10-bar pivot high

---

## Previous Session Progress (2025-12-01)

### Accomplishments
1. ✅ Cloned SKIE-Ninja repository from GitHub
2. ✅ Verified NinjaTrader 8.1.6.0 and Visual Studio installations
3. ✅ Downloaded PortaraNinja sample data (ES 1-min, NQ tick)
4. ✅ Built Python data loader for NinjaTrader format
5. ✅ Implemented 474 features across 12 categories (95% complete)
6. ✅ Built modular feature engineering architecture
7. ✅ Configured FRED API key (real macroeconomic data ready)
8. ✅ Implemented microstructure features (71 features)
9. ✅ Implemented sentiment features - VIX, COT (43 features)
10. ✅ Implemented intermarket features (84 features)
11. ✅ Implemented alternative data - Reddit, News, Fear&Greed (31 features)
12. ✅ Downloaded free Yahoo Finance data (ES, NQ, VIX + 5 more, 2 years daily)
13. ✅ Downloaded 1,126 hourly bars for ES (60 days)
14. ✅ **Downloaded Databento ES 1-min data: 684,410 bars (2023-2024)**
15. ✅ **Downloaded Databento NQ 1-min data: 684,432 bars (2023-2024)**
16. ✅ **Downloaded Databento ES/NQ 1-min data: 2020, 2021, 2022**
17. ✅ **Downloaded Databento YM, GC, CL, ZN 1-min (2023-2024)**
18. ✅ **Downloaded Databento ES MBP-10 Level 2 sample**

---

## Feature Engineering Status

| Category | Features | Status |
|----------|----------|--------|
| 1. Price-Based | 79 | ✅ Complete |
| 2. Technical Indicators | 105 | ✅ Complete |
| 3. Macroeconomic | 12 | ✅ Complete (FRED API ready) |
| 4. Microstructure | 71 | ✅ Complete |
| 5. Sentiment & Positioning | 43 | ✅ Complete (VIX, COT) |
| 6. Intermarket | 84 | ✅ Complete |
| 7. Seasonality & Calendar | 58 | ✅ Complete |
| 8. Regime & Fractal | 19 | ✅ Complete (Hurst) |
| 9. Alternative Data | 31 | ✅ Complete (Reddit, News, Fear&Greed) |
| 10. Lagged & Transformed | 67 | ✅ Complete |
| 11. Interaction Features | 8 | ✅ Complete |
| 12. Target Labels | 11 | ✅ Complete |
| **TOTAL** | **474/~500** | **~95% Complete** |

---

## Data Available

| Source | Data | Timeframe | Bars | Status |
|--------|------|-----------|------|--------|
| **Databento** | ES 1-min | 2023-2024 | **684,410** | ✅ Downloaded |
| **Databento** | NQ 1-min | 2023-2024 | **684,432** | ✅ Downloaded |
| **Databento** | ES 1-min | 2020 | ~340,000 | ✅ Downloaded |
| **Databento** | ES 1-min | 2021 | ~340,000 | ✅ Downloaded |
| **Databento** | ES 1-min | 2022 | ~340,000 | ✅ Downloaded |
| **Databento** | NQ 1-min | 2020-2022 | ~1M | ✅ Downloaded |
| **Databento** | YM 1-min | 2023-2024 | Various | ✅ Downloaded |
| **Databento** | GC 1-min | 2023-2024 | Various | ✅ Downloaded |
| **Databento** | CL 1-min | 2023-2024 | Various | ✅ Downloaded |
| **Databento** | ZN 1-min | 2023-2024 | Various | ✅ Downloaded |
| **Databento** | ES MBP-10 | Sample | Sample | ✅ Downloaded |
| PortaraNinja | ES 1-min | Sample | 67,782 | ✅ Downloaded |
| PortaraNinja | NQ tick | Sample | 42,649 | ✅ Downloaded |
| Yahoo Finance | ES daily | 2 years | 500 | ✅ Downloaded |
| Yahoo Finance | NQ daily | 2 years | 500 | ✅ Downloaded |
| Yahoo Finance | VIX, GC, CL, ZN, DX daily | 2 years | ~500 each | ✅ Downloaded |

---

## Databento Budget

- **API Key**: Configured (db-L8vcArDDsTpeVUW5x...)
- **Starting Credits**: $125.00
- **Used**: ~$3.17 (ES + NQ 1-min OHLCV 2020-2024)
- **Remaining**: ~$121.83

---

## Key Decisions Made

- **Primary Market**: ES (S&P 500 E-mini) - phased approach, NQ as secondary
- **Data Source**: Databento for 1-minute data, PortaraNinja for samples
- **FRED API Key**: Configured and tested (416c373c...13912)
- **Primary ML Model**: XGBoost (84% AUC-ROC)
- **Feature Selection**: 75 features from multi-method ranking (4 methods)
- **Training Data**: 684,410 bars (2023-2024 ES)
- **Validation**: Walk-forward with 3 folds, 80/20 temporal split

---

## Section 1: Development Environment

### 1.1 Visual Studio
- **Installed**: YES (User confirmed 2025-11-30)
- **VS Code Path**: C:\Users\skoir\AppData\Local\Programs\Microsoft VS Code\
- **Visual Studio**: Installed (user confirmed)
- **Workflow**: Claude Code (VS Code) → writes code → User compiles in Visual Studio/NinjaTrader

### 1.2 Python
- **Installed**: YES
- **Version**: Python 3.9.13
- **Path**: C:\Python39\python.exe
- **Environment Manager**: pip (no Anaconda/Miniconda detected)
- **ML Libraries Installed**: EXCELLENT - see details below

#### Verified ML Stack:
| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.26.4 | Numerical computing |
| pandas | 2.3.3 | Data manipulation |
| scikit-learn | 1.3.2 | ML algorithms (Random Forest, etc.) |
| xgboost | 2.0.2 | Gradient boosting |
| tensorflow | (via keras 3.10.0) | Deep learning |
| keras | 3.10.0 | Neural networks |
| torch | 2.8.0 | PyTorch deep learning |
| transformers | 4.57.1 | NLP/Transformer models |
| matplotlib | 3.9.4 | Visualization |
| onnxruntime | 1.19.2 | ONNX model inference |
| nltk | 3.9.2 | Natural language processing |
| sentence-transformers | 5.1.2 | Sentence embeddings |

### 1.3 NinjaTrader 8
- **Installed**: YES - Version 8.1.6.0 64-bit (confirmed 2025-11-30)
- **License Type**: NinjaTrader account (unfunded, simulation mode)
- **Brokerage Connection**: Simulation account active
- **User Documents**: C:\Users\skoir\Documents\NinjaTrader 8\

---

## Section 2: Trained Models

### 2.1 XGBoost Model (Best)
- **File**: data/models/models/xgboost_20251203_220559.pkl
- **Features**: 75 selected features
- **Performance**: 84.07% AUC-ROC, 75.23% Accuracy
- **Training Samples**: 547,572 (80% of 684,410)
- **Test Samples**: 136,838 (20%)

### 2.2 RandomForest Model (Baseline)
- **File**: data/models/models/randomforest_20251203_220559.pkl
- **Features**: 75 selected features
- **Performance**: 76.83% AUC-ROC, 71.07% Accuracy

### 2.3 Supporting Files
- Scaler: data/models/models/scaler_20251203_220559.pkl
- Feature list: data/models/models/features_20251203_220559.json
- CV metrics: data/models/cv_metrics_20251203_220559.json
- XGBoost importance: data/models/xgboost_feature_importance.csv
- RF importance: data/models/randomforest_feature_importance.csv

---

## Section 3: Environment Paths

### Verified Paths
| Component | Path | Status |
|-----------|------|--------|
| Python | C:\Python39\python.exe | Verified |
| VS Code | C:\Users\skoir\AppData\Local\Programs\Microsoft VS Code\ | Verified |
| Project Root | C:\Users\skoir\Documents\SKIE Enterprises\SKIE_Ninja\ | Verified |
| Visual Studio 2022 | Installed (user confirmed) | Verified |
| NinjaTrader 8 | C:\Users\skoir\Documents\NinjaTrader 8\ | Verified (8.1.6.0 64-bit) |

---

## Section 4: Next Steps

### CV Enhancements (2025-12-03)
- ✅ **Rolling Window CV** - Added `window_type='rolling'` option (vs expanding)
- ✅ **Embargo Periods** - Default 210 bars between train/test to prevent leakage
- ✅ **Data Leakage Checks** - Automated detection of target correlation >0.95
- ✅ **RTH Filtering** - Filter to Regular Trading Hours only
- ✅ **Data Resampler** - Utility to convert 1-min → 5-min/15-min bars with RTH filter

### Data Resampler (2025-12-04)
**File**: `src/python/utils/data_resampler.py`
**Usage**: `from utils import DataResampler, resample_ohlcv`
- Converts 1-min OHLCV to any timeframe (5-min, 15-min, 30-min, 1h)
- RTH filtering built-in (9:30 AM - 4:00 PM ET)
- Auto-calculates CV embargo/window sizes for target timeframe
- Tested: 684,410 1-min bars → 38,286 5-min RTH bars

### Trading Hours Constraint
**IMPORTANT: We only trade during Regular Trading Hours (RTH)**
- **ES/NQ RTH**: 9:30 AM - 4:00 PM Eastern Time (ET)
- **Timezone**: America/New_York
- **Reason**: Better liquidity, tighter spreads, more reliable signals
- **Implementation**: `filter_rth_only(df)` in model_trainer.py

### Timeframe Decision (DECIDED)
**SELECTED: 5-MIN BARS** for live trading
| Timeframe | Bars/Day (RTH) | Embargo (bars) | Best AUC | Decision |
|-----------|----------------|----------------|----------|----------|
| **5-min** | 78             | 42             | **83.30%** | **SELECTED** |
| 15-min    | 26             | 14             | 82.03%   | Rejected |

**Optimal Rolling Window Configuration:**
- **Training Window**: 180 days (~14,040 5-min bars)
- **Test Window**: 5 days (~390 5-min bars)
- **Embargo Period**: 42 bars (~3.5 hours)
- **Expected Folds**: ~61 walk-forward folds

### Phase 7 Remaining Tasks
- [x] **Decide trading timeframe** → 5-min bars selected
- [x] Resample data to chosen timeframe → 38,062 5-min RTH bars
- [x] Rolling window CV optimization → Complete (15 configs tested)
- [x] **Install LightGBM and train model** → 84.21% AUC-ROC (BEST)
- [x] **Develop LSTM/GRU time series models** → 65-66% AUC (underperforms)
- [ ] Implement Transformer-based models
- [ ] Create model ensemble (LightGBM + XGBoost)
- [ ] ONNX export for NinjaTrader integration
- [ ] Retrain final model with full historical data

### Phase 8 (Validation)
- [ ] Extended walk-forward testing
- [ ] Monte Carlo simulations (1000+ runs)
- [ ] Out-of-sample testing on 2020-2022 data
- [ ] Regime-specific performance analysis

---

## Section 5: Decision Log

| Date | Decision | Reasoning | Reference |
|------|----------|-----------|-----------|
| 2025-12-04 | **5-min bars selected** | +1.27% AUC over 15-min (83.30% vs 82.03%) | Grid optimization |
| 2025-12-04 | Train=180d, Test=5d | Best AUC config across all tested combinations | rolling_window_grid_5min.csv |
| 2025-12-04 | 42-bar embargo for 5-min | ~3.5 hours gap to prevent data leakage | rolling_window_optimizer.py |
| 2025-12-03 | RTH-only trading | Better liquidity, tighter spreads, reliable signals | model_trainer.py |
| 2025-12-03 | 210-bar embargo in CV | Prevent data leakage from 200-bar features | model_trainer.py |
| 2025-12-03 | Rolling window CV option | Test regime stability, avoid overfitting | model_trainer.py |
| 2025-12-03 | Data leakage auto-check | Detect >0.95 correlation with target | model_trainer.py |
| 2025-12-03 | XGBoost as primary model | Best AUC-ROC (84.07%) | Model training results |
| 2025-12-03 | 75 features selected | Multi-method ranking, best predictive power | Feature selection |
| 2025-12-03 | Pyramiding R:R top feature | Highest correlation with target | Feature rankings |
| 2025-12-01 | Databento for historical data | Cost-effective, high quality | $3.17 for 5 years ES+NQ |
| 2025-11-30 | Project memory base created | Track configuration decisions | - |
| 2025-11-30 | Python 3.9.13 verified with full ML stack | Ready for model development | System scan |
| 2025-11-30 | Rithmic recommended for live algo trading | Lowest latency, full order book | Data feed comparison |
| 2025-11-30 | PortaraNinja recommended for historical data | Native format, extensive coverage | Official NT partner |

---

*This document serves as the central memory base for all project decisions and configurations.*
*Auto-updated by SKIE_Ninja development process.*
