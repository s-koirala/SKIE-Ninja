# SKIE_Ninja Project Handoff Document

**Date**: 2026-01-06
**Purpose**: Complete context for continuing development on new machine
**Last Updated**: Phase 16 - Live Demo Trading Active (Feature Fix Applied)

---

## PROJECT STATUS: VALIDATION REMEDIATION COMPLETE - PAPER TRADING PENDING

### CRITICAL AUDIT FINDINGS (2026-01-06)

**Multiple audits identified critical issues requiring remediation before live capital deployment.**

| Audit Document | Key Finding | Status |
|----------------|-------------|--------|
| [NINJATRADER_DEPLOYMENT_AUDIT_20260106.md](docs/NINJATRADER_DEPLOYMENT_AUDIT_20260106.md) | 97.6% trade frequency collapse, 9.1% short win rate | **REMEDIATED** |
| [CPCV_PBO_IMPLEMENTATION_AUDIT_20260106.md](docs/CPCV_PBO_IMPLEMENTATION_AUDIT_20260106.md) | CPCV purging lacks t1 parameter, PBO uses MC approximation | **DOCUMENTED** |
| [CANONICAL_VALIDATION_RESULTS_20260105.md](data/validation_results/CANONICAL_VALIDATION_RESULTS_20260105.md) | DSR p=0.978 (not significant), 50% P&L inflation from embargo error | **ACKNOWLEDGED** |

### Latest NT8 Results (2026-01-05) - AUDIT COMPLETED

| Metric | Value | Assessment |
|--------|-------|------------|
| **Period** | Jan 2025 - Dec 2025 (12 months) | - |
| **Total Trades** | 18 | **97.6% below expected** |
| **Net P&L** | +$1,287.50 | **96.3% below expected** |
| **Win Rate** | 38.9% (7/18) | Within tolerance |
| **Long Win Rate** | 85.7% (6/7) | Acceptable |
| **Short Win Rate** | **9.1% (1/11)** | **CRITICAL FAILURE** |
| **Signal Gap** | 160 days (Feb-Jul) | **Infrastructure failure** |

### Remediation Actions Completed

| Action | Status | Reference |
|--------|--------|-----------|
| Short signals disabled | **COMPLETE** | signal_server.py:76 |
| Thresholds aligned to backtest | **COMPLETE** | signal_server.py:66-68 |
| Diagnostic logging added | **COMPLETE** | signal_server.py:235-261 |
| Heartbeat monitoring added | **COMPLETE** | signal_server.py:548-575 |
| CPCV/PBO framework implemented | **COMPLETE** | validation/cpcv_pbo.py |

### Statistical Validation Status

| Metric | Threshold | Current | Status |
|--------|-----------|---------|--------|
| DSR p-value | < 0.10 | 0.978 | **FAIL** |
| PBO | < 0.50 | Pending run | **PENDING** |
| Paper trade n | >= 100 | 18 | **FAIL** |
| Short win rate | > 30% or disabled | **DISABLED** | **FIXED** |

---

## PREVIOUS VALIDATION (SUPERSEDED BY AUDIT)

**WARNING:** The following results were obtained with **incorrect embargo (20-42 bars)**. Per Lopez de Prado (2018) Ch. 7, correct embargo is **210 bars** (max feature lookback + label horizon + safety margin).

### Ensemble Strategy - ORIGINAL (INFLATED)

| Test Period | Original P&L | Corrected P&L | Inflation |
|-------------|--------------|---------------|-----------|
| In-Sample (2023-24) | +$114,447 | +$158,212 | -38% (improved) |
| Out-of-Sample (2020-22) | +$502,219 | +$142,867 | **+71.6% inflated** |
| Forward Test (2025) | +$57,394 | +$34,771 | **+39.4% inflated** |
| **TOTAL** | **$674,060** | **$335,850** | **50.2% inflated** |

### Monte Carlo Validation: SUPERSEDED

The original Monte Carlo validation was performed on inflated results. The "100% probability of profit" finding is **not valid** for corrected data.

### DSR Analysis (CANONICAL_VALIDATION_RESULTS_20260105.md)

| Period | Sharpe | E[max(SR)] | DSR p-value | Status |
|--------|--------|------------|-------------|--------|
| In-Sample 2023-24 | 3.48 | 2.46 | 0.000 | **Significant** |
| OOS 2020-22 | 1.67 | 2.46 | 1.000 | **Not Significant** |
| Forward 2025 | 2.15 | 2.46 | 0.932 | **Not Significant** |
| **Combined** | 2.27 | 2.46 | **0.978** | **Not Significant** |

**Interpretation:** The model cannot reject the null hypothesis that observed OOS/Forward performance is due to chance. Edge likely exists but is smaller than originally reported.

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

## CURRENT PHASE - LIVE DEMO TRADING

### Phase 16: Paper Trading Active (2026-01-06)

**Status**: LIVE on demo account - monitoring for live signal generation

#### Quick Start Commands

```powershell
# Terminal 1: Start Python signal server
cd "C:\Users\skoir\Documents\SKIE Enterprises\SKIE-Ninja\SKIE-Ninja-Project\SKIE_Ninja"
python -m src.python.signal_server

# Terminal 2: NinjaTrader 8
# - Open ES 5-min chart
# - Enable SKIENinjaTCPStrategy
# - Monitor Output window for signals
```

#### System Components

| Component | Location | Status |
|-----------|----------|--------|
| Python Signal Server | `src/python/signal_server.py` | ✅ Running |
| NT8 TCP Strategy | `src/csharp/SKIENinjaTCPStrategy.cs` | ✅ Fixed & Compiled |
| Walk-Forward Models | `data/models/walkforward_onnx/` | ✅ 70 folds loaded |
| Sentiment Data | `data/raw/sentiment/` | ✅ VIX + AAII + PCR |
| Scaler Parameters | `data/models/scaler_params.json` | ✅ 42 features defined |

---

## COMPLETED PHASES

### Phase 15: Python-NT8 Bridge Architecture (COMPLETE - 2026-01-05)

**Objective**: Deploy validated models using hybrid Python-NT8 architecture

#### Critical Fix Applied (2026-01-05)

**Issue**: NT8 backtest showed -$42,600 loss instead of expected profit.

**Root Cause**: Complete feature mismatch - C# was sending wrong 42 features:
- Missing: `return_lag2`, all `rv_X` (realized volatility), all `volume_sma_X`, all `volume_ratio_X`
- Extra (wrong): MACD features, time features, wrong return calculations

**Fix**: Complete rewrite of `CalculateFeatures()` in SKIENinjaTCPStrategy.cs to match exact order from `scaler_params.json`.

**Result**: Backtest now shows +$1,287.50 profit with 1.71 profit factor.

#### 15B: Socket Bridge Implementation (CURRENT)
Deploy Python strategy directly via TCP socket bridge - preserves validated code exactly.

| Task | Status | Files |
|------|--------|-------|
| 1. VIX Data Access Fix | ✅ Complete | `BarsArray[1].GetClose(index)` method |
| 2. Identify Fake Data Issue | ✅ Complete | C# was using VIX proxies for PCR/AAII |
| 3. Python Signal Server | ✅ Complete | `src/python/signal_server.py` |
| 4. Sentiment Data Downloader | ✅ Complete | `src/python/data_collection/sentiment_data_downloader.py` |
| 5. NT8 TCP Client Strategy | ✅ Complete | `src/csharp/SKIENinjaTCPStrategy.cs` |
| 6. Feature Mismatch Fix | ✅ Complete | `CalculateFeatures()` rewritten |
| 7. GitHub Documentation | ✅ Complete | README.md, HANDOFF.md, FEATURE_AUDIT |

#### Critical Finding 1: VIX Data Access (SOLVED)

**Problem**: NT8 daily bars don't "complete" until market close, causing `CurrentBars[1] = -1`.

**Solution**: Use direct array access:
```csharp
// OLD (broken)
double vixClose = Closes[1][barsAgo];  // Returns -1 for incomplete bars

// NEW (works)
int vixBarIndex = BarsArray[1].Count - 1 - barsAgo;
double vixClose = BarsArray[1].GetClose(vixBarIndex);
```

---

#### Critical Finding 2: Fake Sentiment Data (SOLVED)

**Problem**: NT8 showed ~4,467 trades with -$105K loss vs Python's ~2,044 trades with +$88K profit.

**Root Cause**: C# was using **VIX-derived PROXIES** for PCR and AAII sentiment features:
```csharp
// C# was generating FAKE features like this:
pcr_5d_ma = vixClose * 0.05;      // NOT real PCR data!
aaii_bull = 0.35 - (vixClose/100); // NOT real AAII data!
```

The sentiment model was **trained on REAL PCR/AAII data**, so these proxies produced garbage predictions that passed ~100% of the time.

**Solution**: Python Signal Server architecture that uses actual historical sentiment data.

---

#### Python-NT8 Bridge Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT ARCHITECTURE                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌─────────────────┐        TCP/5555        ┌─────────────────┐ │
│   │  Python Server  │◄──────────────────────►│  NinjaTrader 8  │ │
│   │  (Signal Gen)   │                        │  (Execution)    │ │
│   └────────┬────────┘                        └────────┬────────┘ │
│            │                                          │          │
│   ┌────────┴────────┐                        ┌────────┴────────┐ │
│   │ - 70 ONNX models│                        │ - Receives JSON │ │
│   │ - Walk-forward  │                        │ - Executes OCO  │ │
│   │ - Real PCR/AAII │                        │ - Position mgmt │ │
│   │ - Ensemble logic│                        │ - Broker conn   │ │
│   └─────────────────┘                        └─────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

**Why This Architecture**:
- Python holds all 70 walk-forward ONNX models
- Python has access to REAL historical PCR/AAII sentiment data
- NT8 handles order execution and broker connection (cannot be bypassed)
- Clean separation: Python = brain, NT8 = hands

**Data Flow**:
1. NT8 calculates 42 technical features from live bars
2. Sends JSON request to Python server via TCP socket
3. Python loads correct walk-forward model, adds sentiment features
4. Returns signal: `{should_trade, direction, tp_mult, sl_mult}`
5. NT8 executes the trade with TP/SL orders

---

#### Data Requirements

| Data Source | Cost | Purpose | Status |
|-------------|------|---------|--------|
| CBOE PCR | Free (historical only to 2019) | Put/Call Ratio | Partial |
| Barchart PCR | $30-50/mo | Real-time PCR | Recommended |
| AAII Sentiment | $29/year | Investor sentiment survey | **Required** |

**To Get Data**:
1. **AAII**: https://www.aaii.com/membership ($29/year)
2. **PCR**: Check https://data.nasdaq.com/databases/EOD first (may be free), else https://www.barchart.com/solutions/data

---

#### Remaining Tasks

| Task | Status | Priority |
|------|--------|----------|
| Obtain AAII subscription | Pending | HIGH |
| Obtain PCR data (Barchart/Quandl) | Pending | HIGH |
| Test integration end-to-end | Pending | HIGH |
| Validate trade counts (~2,044 expected) | Pending | HIGH |
| Paper trading validation | Pending | MEDIUM |

---

#### Walk-Forward NT8 Integration

**Solution**: Two approaches implemented:

1. **Weekly Retraining** (Manual): Run `retrain_onnx_models.py` weekly to keep static models fresh
2. **Walk-Forward Backtest** (NEW): Automatic model switching in NT8 for true walk-forward simulation

#### Walk-Forward NT8 Integration (NEW - 2025-12-06)

Created a complete walk-forward backtesting system for NinjaTrader 8:

| Component | File | Purpose |
|-----------|------|---------|
| Walk-Forward ONNX Export | `src/python/export_walkforward_onnx.py` | Generates 70 model sets for 2024 |
| Walk-Forward Predictor | `src/csharp/SKIENinjaML/WalkForwardPredictor.cs` | Auto-switches models by date |
| Walk-Forward Strategy | `src/csharp/SKIENinjaWalkForwardStrategy.cs` | NT8 strategy with walk-forward mode |
| Model Schedule | `data/models/walkforward_onnx/model_schedule.csv` | Date ranges for each fold |

**How it works:**
1. Pre-generate 70 ONNX model sets (one per 5-day test period)
2. Each model is trained on the prior 180 days
3. NT8 strategy automatically loads correct model based on current bar date
4. Run a SINGLE backtest covering 2024 - models switch automatically

**Usage:**
```powershell
# Step 1: Generate walk-forward models (run once)
python src/python/export_walkforward_onnx.py --start 2024-01-01

# Step 2: Copy walkforward_onnx folder to Documents\SKIE_Ninja\walkforward_models

# Step 3: In NT8, use SKIENinjaWalkForwardStrategy with:
#   - WalkForwardMode = true
#   - WalkForwardPath = path to walkforward_models folder
#   - Run backtest from 2024-01-01 to 2024-12-15

# Expected: Performance matching Python walk-forward (+$502K)
```

#### Technical Fixes Applied

1. **ATR Calculation** - Changed from NinjaTrader's Wilder smoothing to SMA (matching Python)
2. **Realized Volatility** - Changed from population std dev (n) to sample std dev (n-1)
3. **ONNX Output Parsing** - Fixed handling of LightGBM's nested DisposableList output format
4. **SetProfitTarget/SetStopLoss Order** - Must be called BEFORE entry per NT8 docs

#### Files Created/Modified

| File | Purpose |
|------|---------|
| `src/python/deployment/ninja_signal_server.py` | Python signal server |
| `src/ninjatrader/SKIENinjaStrategy.cs` | NinjaScript client |

**Why Socket Bridge over ONNX**:
- Preserves validated Python code exactly as tested
- Eliminates C# feature parity risk
- ~5-10ms latency is negligible for 5-min bars
- Faster to implement (1-2 days vs 1-2 weeks)

#### 15C: Platform Walk-Forward Validation
- Run through NinjaTrader Market Replay (2020-2022)
- Compare trade-by-trade vs Python backtest
- Document discrepancies >5%

#### 15D: Paper Trading (30-60 days)
- Deploy to simulation account
- Monitor daily vs backtest benchmarks
- Implement kill switch (daily loss >$5K)

#### 15E: Controlled Live Trading
- Start with 1 MES contract
- Scale to ES after 30 profitable days

### Exit Parameter Warning (DO NOT RE-OPTIMIZE)

Exit parameters are **FRAGILE** - further optimization risks overfitting.

| Parameter | Validated | Safe Range | Danger Zone |
|-----------|-----------|------------|-------------|
| tp_atr_mult | 2.5 | 2.0-3.0 | <2.0 (losses) |
| sl_atr_mult | 1.25 | 1.0-1.5 | >1.5 (losses) |

**To enable automated weekly retraining:**
```powershell
# Run as Administrator
.\scripts\setup_weekly_task.ps1
```

This creates a scheduled task that runs every Sunday at 6:00 PM.

---

## NEXT STEPS - ACTION PLAN

### Phase 15 Remaining: Data Acquisition (PRIORITY 1)

**Required Data Subscriptions**:

1. **AAII Sentiment Survey** ($29/year)
   - URL: https://www.aaii.com/membership
   - Download weekly CSV with Bull/Bear/Neutral percentages
   - Place in: `data/raw/sentiment/aaii_sentiment.csv`

2. **Put/Call Ratio (PCR)**
   - **Option A**: Quandl/Nasdaq Data Link (check if free tier available)
     - URL: https://data.nasdaq.com/databases/EOD
   - **Option B**: Barchart ($30-50/month)
     - URL: https://www.barchart.com/solutions/data
   - Place in: `data/raw/sentiment/pcr_data.csv`

### Phase 15 Remaining: Integration Testing (PRIORITY 2)

**To Test End-to-End**:
```bash
# Terminal 1: Start Python server
python src/python/signal_server.py

# Terminal 2 (or NinjaTrader):
# - Import SKIENinjaTCPStrategy.cs
# - Apply to ES 1-minute chart
# - Run backtest for 2024 data
# - Expected: ~2,044 trades (matching Python)
```

### Phase 16: Paper Trading Validation (PRIORITY 3)

**Objective**: Validate live execution matches backtest expectations

| Metric | Backtest | Target (Paper) | Tolerance |
|--------|----------|----------------|-----------|
| Win Rate | 40.4% | 38-43% | ±3% |
| Avg Trade | $26.30 | $20-30 | ±25% |
| Sharpe | 3.09 | 2.5-3.5 | ±15% |
| Daily Trades | ~15 | 12-18 | ±20% |

**Duration**: 30-60 trading days minimum

**Success Criteria**:
- Metrics within tolerance bands
- No execution errors
- Slippage matches assumptions ($0.125/tick)

### Phase 17: Live Trading (PRIORITY 4)

**Prerequisites**:
- Paper trading validation PASSED
- VPS infrastructure ready
- Risk management implemented

**Scaling Plan**:
1. Start with 1 contract
2. Scale based on performance (Kelly criterion)
3. Max position: TBD based on account size

### Maintenance & Monitoring

| Task | Frequency | Description |
|------|-----------|-------------|
| **ONNX Model Retrain** | **Weekly** | Run `retrain_onnx_models.py` (CRITICAL for walk-forward equivalent) |
| Performance Review | Weekly | Compare to backtest expectations |
| Feature Drift Check | Monthly | Validate feature distributions |
| Risk Review | Daily | Max drawdown, position limits |
| Full Model Retrain | Quarterly | Complete retraining with expanded dataset |

**CRITICAL**: ONNX models are static and decay rapidly. Per project methodology (180-day train, 5-day test), models should be retrained **weekly** to approximate walk-forward performance.

---

## KEY FILES TO KNOW

### Live Trading System (Phase 16 - ACTIVE)

| File | Purpose | Status |
|------|---------|--------|
| `src/python/signal_server.py` | **TCP signal server - MUST BE RUNNING** | ✅ Active |
| `src/csharp/SKIENinjaTCPStrategy.cs` | **NT8 TCP client - FIXED 2026-01-05** | ✅ Fixed |
| `data/models/walkforward_onnx/` | 70 walk-forward ONNX model sets | ✅ Loaded |
| `data/models/scaler_params.json` | Feature scaling parameters (42 features) | ✅ Reference |
| `docs/FEATURE_AUDIT_20260105.md` | Feature mismatch fix documentation | ✅ Complete |
| `docs/PAPER_TRADING_GUIDE.md` | Complete launch procedure | ✅ Reference |

### Python Strategy Files
| File | Purpose |
|------|---------|
| `src/python/strategy/volatility_breakout_strategy.py` | **MAIN STRATEGY** |
| `src/python/feature_engineering/multi_target_labels.py` | Target generation (73 targets) |
| `src/python/backtesting/walk_forward_backtest.py` | Walk-forward backtesting engine |
| `src/python/run_oos_backtest.py` | OOS validation script |
| `src/python/run_qc_check.py` | Quality control validation |

### NinjaTrader Integration Files
| File | Purpose |
|------|---------|
| `src/python/signal_server.py` | **TCP signal server (PRODUCTION)** |
| `src/python/data_collection/sentiment_data_downloader.py` | PCR/AAII data downloader |
| `src/python/data_collection/historical_sentiment_loader.py` | VIX/AAII/PCR loader |
| `src/python/export_onnx.py` | Export LightGBM models to ONNX |
| `src/python/retrain_onnx_models.py` | Weekly retraining script |
| `src/csharp/SKIENinjaTCPStrategy.cs` | **NT8 TCP client (PRODUCTION)** |
| `src/csharp/SKIENinjaWalkForwardStrategy.cs` | NT8 walk-forward ONNX strategy |
| `src/csharp/SKIENinjaML/WalkForwardPredictor.cs` | C# walk-forward predictor |
| `data/models/walkforward_onnx/` | 70 walk-forward ONNX model sets |
| `docs/VALIDATION_REPORT.md` | Complete validation results |
| `docs/NINJATRADER_INSTALLATION.md` | NT8 setup guide |

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
| `CHANGELOG.md` | **NEW** Version history and changes |
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

### Phase 10-14 (All Complete)
- [x] Out-of-sample validation (2020-2022): PASSED
- [x] Forward test (2025): PASSED
- [x] QC checks: PASSED
- [x] **Threshold optimization**: COMPLETE (+96% improvement)
- [x] **Monte Carlo simulation**: COMPLETE (100% prob profit)
- [x] **NinjaTrader ONNX export**: COMPLETE (70 walk-forward models)
- [x] **VIX data access fix**: COMPLETE (BarsArray method)
- [x] **Identify fake data issue**: COMPLETE (PCR/AAII proxies)
- [x] **Python-NT8 bridge architecture**: COMPLETE

### Phase 15 (In Progress)
- [ ] Obtain AAII subscription ($29/year)
- [ ] Obtain PCR data subscription
- [ ] Test integration end-to-end
- [ ] Validate trade counts (~2,044 expected)
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
| `run_overfitting_assessment.py` | DSR + CSCV overfitting tests |
| `run_window_optimization.py` | **NEW** Data-driven train/test window selection |
| `run_embargo_analysis.py` | **NEW** Autocorrelation-based embargo justification |

### Quality Control (Phase 15)
| File | Purpose |
|------|---------|
| `quality_control/overfitting_detection.py` | DSR, CSCV, PSR implementations |
| `tests/test_critical_functions.py` | Pytest suite for validation |

### Shared Utilities (Phase C)
| File | Purpose |
|------|---------|
| `feature_engineering/shared/technical_utils.py` | TR, ATR, RSI, Bollinger, MACD |
| `feature_engineering/shared/returns_utils.py` | Return calculations |
| `feature_engineering/shared/volume_utils.py` | Volume features, VWAP |
| `feature_engineering/shared/temporal_utils.py` | Time encoding |

**Note:** Shared utilities consolidate 11+ duplicate implementations across the codebase.

### Configuration (Updated)
| File | Purpose |
|------|---------|
| `config/api_keys.py` | **SECURE** - Loads keys from environment variables |
| `config/api_keys.env.template` | Template for API keys (copy to api_keys.env) |

**API Key Security**:
- Copy `api_keys.env.template` to `api_keys.env`
- Fill in your actual keys in `api_keys.env`
- Never commit `api_keys.env` (it's gitignored)

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
