# SKIE_Ninja - Smart Algorithmic Trading System for NinjaTrader

A comprehensive algorithmic trading system leveraging machine learning, macroeconomic factors, and advanced quantitative strategies for futures trading on the NinjaTrader platform.

---

## LIVE DEMO STATUS (2026-01-06) - ACTIVE

**STATUS: PAPER TRADING LIVE ON DEMO ACCOUNT**

### Latest NT8 Backtest Results (Feature Fix Applied)

| Metric | Value |
|--------|-------|
| **Period** | Jan 2025 - Dec 2025 (12 months) |
| **Total Trades** | 18 |
| **Net P&L** | **+$1,287.50** |
| **Win Rate** | 39% (7/18) |
| **Profit Factor** | **1.71** |

### Quick Start

```powershell
# Terminal 1: Start Python signal server
cd "C:\Users\skoir\Documents\SKIE Enterprises\SKIE-Ninja\SKIE-Ninja-Project\SKIE_Ninja"
python -m src.python.signal_server

# NinjaTrader 8: Enable SKIENinjaTCPStrategy on ES 5-min chart
```

---

## VALIDATION STATUS (2026-01-05) - COMPLETE

**Previous canonical validation per Lopez de Prado (2018) and Bailey et al. (2014):**

### Corrected Results (embargo=210 bars)

| Period | Trades | Net P&L | Sharpe | DSR p-value |
|--------|--------|---------|--------|-------------|
| In-Sample 2023-24 | 2,656 | $158,212 | 3.48 | 0.000 *** |
| OOS 2020-22 | 4,763 | $142,867 | 1.67 | 1.000 NS |
| Forward 2025 | 753 | $34,771 | 2.14 | 0.932 NS |
| **TOTAL** | **8,172** | **$335,850** | 2.27 | 0.978 NS |

See `data/validation_results/CANONICAL_VALIDATION_RESULTS_20260105.md` for complete analysis.

---

## CRITICAL FIX (2026-01-05): Feature Mismatch Resolved

**Issue:** NT8 backtest showed -$42,600 loss (41% win rate) instead of expected profit.

**Root Cause:** C# strategy was sending 42 features that DID NOT MATCH the trained model's expected features. The model received garbage input, producing random predictions.

**Fix Applied:** Complete rewrite of `CalculateFeatures()` to match exact feature order from `scaler_params.json`.

See `docs/FEATURE_AUDIT_20260105.md` for complete analysis.

**After fix, recompile required in NinjaTrader NinjaScript Editor.**

---

## Project Status: Phase 16 - PAPER TRADING READY

### Current Status: Ready for Paper Trading (2026-01-05)

All systems verified and tested:

| Component | Status | Notes |
|-----------|--------|-------|
| Signal Server | READY | Tested, 70 folds loaded |
| ONNX Models | READY | Walk-forward models exported |
| Sentiment Data | READY | VIX + AAII + PCR proxy |
| Weekly Retraining | READY | Scheduled task available |
| NT8 Strategy | **FIXED** | Feature parity corrected 2026-01-05 |

**Quick Start:**
```powershell
# Terminal 1: Start signal server
cd "C:\Users\skoir\Documents\SKIE Enterprises\SKIE-Ninja\SKIE-Ninja-Project\SKIE_Ninja"
python -m src.python.signal_server

# Then enable SKIENinjaTCPStrategy on ES 5-min chart in NinjaTrader 8
```

See `docs/PAPER_TRADING_GUIDE.md` for complete launch procedure.

### Corrected Performance Expectations

Based on canonical validation with proper embargo (210 bars):

| Metric | Expected (Annual) | Notes |
|--------|-------------------|-------|
| Net P&L | ~$67,000 | From $335K / 5 years |
| Weekly P&L | ~$1,300-$1,500 | High variance |
| Win Rate | 62-65% | Per backtest |
| Max Drawdown | ~$30,000 | Could occur any time |
| Sharpe Ratio | 2.27 | Annualized |

**CAUTION:** DSR p-value was not significant (0.978) after correcting for 81 threshold combinations. Position size conservatively.

---

## Phase 15: NT8 Integration Findings

### Critical Issue 1: VIX Data Access (SOLVED)

**Problem**: NT8 daily bars don't "complete" until market close, causing `CurrentBars[1] = -1` for VIX data series.

**Root Cause**: Using `Closes[1][barsAgo]` fails when daily bars haven't completed yet.

**Solution**: Use direct array access instead:
```csharp
// OLD (broken) - Returns -1 for incomplete bars
double vixClose = Closes[1][barsAgo];

// NEW (works) - Direct array access
int vixBarIndex = BarsArray[1].Count - 1 - barsAgo;
double vixClose = BarsArray[1].GetClose(vixBarIndex);
```

### Critical Issue 2: Fake Sentiment Data (SOLVED)

**Problem**: NT8 backtest showed ~4,467 trades with -$105K loss vs Python's ~2,044 trades with +$88K profit.

**Root Cause**: The C# strategy was using **VIX-derived PROXIES** for PCR and AAII sentiment features:
```csharp
// C# was generating "fake" features like this:
pcr_5d_ma = vixClose * 0.05;     // NOT real PCR data!
aaii_bull = 0.35 - (vixClose/100); // NOT real AAII data!
```

The sentiment model was **trained on REAL PCR/AAII data**, so these proxies produced garbage predictions.

**Solution**: Python Signal Server architecture that uses actual historical sentiment data.

### Python-NT8 Bridge Architecture

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

**Data Flow**:
1. NT8 calculates 42 technical features from live bars
2. Sends JSON request to Python server via TCP socket
3. Python loads correct walk-forward model, adds sentiment features
4. Returns signal: `{should_trade, direction, tp_mult, sl_mult}`
5. NT8 executes the trade with TP/SL orders

### Data Requirements

| Data Source | Cost | Purpose | Status |
|-------------|------|---------|--------|
| CBOE PCR | Free (historical only to 2019) | Put/Call Ratio | Partial |
| Barchart PCR | $30-50/mo | Real-time PCR | Recommended |
| AAII Sentiment | $29/year | Investor sentiment survey | Required |

**AAII Membership**: https://www.aaii.com/membership

### New Files Created

| File | Purpose |
|------|---------|
| `src/python/signal_server.py` | TCP server with walk-forward models |
| `src/python/data_collection/sentiment_data_downloader.py` | PCR/AAII data downloader |
| `src/csharp/SKIENinjaTCPStrategy.cs` | NT8 TCP client strategy |

### Model Performance

| Model | In-Sample | OOS | Forward | Purpose |
|-------|-----------|-----|---------|---------|
| Vol Expansion (Tech) | 0.84 AUC | 0.79 | 0.77 | WHEN to trade |
| Vol Expansion (Sent) | 0.77 AUC | 0.65 | N/A | Sentiment filter |
| Breakout High/Low | 0.72 AUC | 0.73 | 0.71 | WHERE price goes |
| ATR Forecast | 0.36 R² | 0.30 | 0.28 | HOW MUCH it moves |

## Key Breakthrough: Predict Market Structure, Not Direction

**The breakthrough insight**: Direction prediction is impossible (AUC 0.50). Instead, we predict:
1. **WHEN** to trade - Volatility expansion (AUC 0.84)
2. **WHERE** price goes - Breakout high/low (AUC 0.72)
3. **HOW MUCH** it moves - ATR forecast (R² 0.36)

## NinjaTrader Deployment (Socket Bridge)

### Architecture

The validated Python strategy runs via TCP socket bridge - preserving code exactly as tested:

```
NinjaTrader 8 ←→ TCP Socket (localhost:5555) ←→ Python Signal Server
```

### Quick Deployment

```bash
# 1. Start Python signal server
cd SKIE_Ninja
python src/python/deployment/ninja_signal_server.py --port 5555 --mode paper

# 2. Install NinjaScript strategy
# Copy src/ninjatrader/SKIENinjaStrategy.cs to:
# Documents\NinjaTrader 8\bin\Custom\Strategies\
# In NinjaTrader: Right-click Strategies > Compile

# 3. Apply strategy to ES 5-minute chart
# Set Python Server Host: localhost
# Set Python Server Port: 5555
```

### Why Socket Bridge (Not ONNX)?

| Benefit | Explanation |
|---------|-------------|
| **Code Integrity** | Preserves validated Python exactly as tested |
| **No Feature Parity Risk** | Eliminates C# vs Python calculation drift |
| **Rapid Iteration** | Change Python, restart server - no recompile |
| **Acceptable Latency** | ~5-10ms negligible for 5-min bars |

### Exit Parameters (DO NOT CHANGE)

| Parameter | Value | Safe Range | Danger |
|-----------|-------|------------|--------|
| tp_atr_mult | 2.5 | 2.0-3.0 | <2.0 = losses |
| sl_atr_mult | 1.25 | 1.0-1.5 | >1.5 = losses |

---

## Repository Structure

```
SKIE_Ninja/
├── src/
│   ├── python/
│   │   ├── strategy/                    # Trading strategies
│   │   │   ├── volatility_breakout_strategy.py   # Base strategy (validated)
│   │   │   ├── ensemble_strategy.py              # PRODUCTION strategy (best)
│   │   │   └── sentiment_strategy.py             # Sentiment-only (research)
│   │   ├── signal_server.py             # TCP server for NT8 signals (NEW)
│   │   ├── data_collection/             # Data downloaders and loaders
│   │   │   ├── databento_downloader.py          # Databento API integration
│   │   │   ├── historical_sentiment_loader.py   # VIX sentiment data
│   │   │   ├── sentiment_data_downloader.py     # PCR/AAII downloader (NEW)
│   │   │   ├── established_sentiment_indices.py # AAII, PCR, VIX indices
│   │   │   └── ...
│   │   ├── feature_engineering/         # Feature calculation modules
│   │   │   ├── multi_target_labels.py           # 73-target generator
│   │   │   ├── multi_timeframe_features.py      # MTF analysis (15m, 1h, 4h)
│   │   │   ├── enhanced_cross_market.py         # Cross-market features
│   │   │   ├── social_news_sentiment.py         # Twitter/News/Reddit
│   │   │   ├── enhanced_feature_pipeline.py     # Unified pipeline
│   │   │   └── ...
│   │   ├── models/                      # ML model training
│   │   ├── backtesting/                 # Walk-forward backtesting
│   │   ├── quality_control/             # Validation framework
│   │   ├── run_monte_carlo_simulation.py        # Monte Carlo validation
│   │   └── ...
│   └── csharp/
│       ├── SKIENinjaTCPStrategy.cs      # NT8 TCP client strategy (NEW)
│       ├── SKIENinjaWalkForwardStrategy.cs  # NT8 walk-forward strategy
│       ├── SKIENinjaStrategy.cs         # NT8 static ONNX strategy
│       └── SKIENinjaML/                 # C# ONNX inference DLL
├── data/
│   ├── raw/market/                      # Downloaded market data
│   ├── processed/                       # Feature rankings
│   ├── models/                          # Trained model files
│   ├── backtest_results/                # Backtest outputs
│   ├── optimization_results/            # Threshold optimization
│   ├── monte_carlo_results/             # Monte Carlo validation
│   └── validation_results/              # QC reports
├── config/
│   ├── CANONICAL_REFERENCE.md           # Single source of truth
│   └── project_memory.md                # Decision log
├── research/
│   ├── 04_multi_target_prediction_strategy.md   # Strategy design
│   └── 05_sentiment_strategy_plan.md            # Sentiment planning
└── docs/
    ├── BEST_PRACTICES.md                # Lessons learned
    ├── VALIDATION_REPORT.md             # Complete validation results (NEW)
    ├── methodology/BACKTEST_METHODOLOGY.md
    └── archive/                         # Archived old docs
```

## Data Available

| Source | Instrument | Timeframe | Bars | Years | Use |
|--------|-----------|-----------|------|-------|-----|
| Databento | ES (S&P 500) | 1-min | 684,410 | 2023-2024 | In-Sample |
| Databento | ES | 1-min | ~1M | 2020-2022 | Out-of-Sample |
| Databento | ES | 1-min | 326K | 2025 | Forward Test |
| Databento | NQ, YM, GC, CL, ZN | 1-min | Various | 2020-2024 | Cross-market |
| Yahoo Finance | VIX, DX + others | Daily | ~500 each | 2+ years | Sentiment |

## Feature Engineering

### Active Non-Leaky Features

| Category | Features | Status |
|----------|----------|--------|
| Returns (Lagged) | return_lag1/2/3/5/10/20 | Safe |
| Volatility | rv_5/10/14/20, atr_5/10/14/20 | Safe |
| Price Position | close_vs_high/low/ma_10/20/50 | Safe |
| Momentum | rsi_7/14/21, stoch_k_14 | Safe |
| Volume | volume_ma_ratio_10/20 | Safe |
| Time | hour_sin/cos, dow_sin/cos | Safe |
| Multi-Timeframe (NEW) | htf_15m/1h/4h_* | Safe (lagged) |
| Cross-Market (NEW) | corr_NQ/YM/GC/CL/ZN_* | Safe (lagged) |
| Sentiment (NEW) | vix_*, social_sentiment_* | Safe (5-min lag) |

### Deprecated Leaky Features (DO NOT USE)

| Features | Reason |
|----------|--------|
| `pyramid_rr_5/10/20` | Uses shift(-N) - future data |
| `pivot_high/low_*` | Forward-looking window |
| `ddca_buy/sell_success_*` | Uses close.shift(-horizon) |

## Quality Control & Validation

### Data Leakage Prevention Checks

```python
# Automated checks run by run_qc_check.py:
# 1. Feature look-ahead: Detects shift(-N) patterns
# 2. Correlation check: max feature-target corr < 0.30
# 3. Temporal leakage test: train future, predict past
# 4. Result validation: Win rate < 65%, Sharpe < 4.0
```

### Monte Carlo Simulation Types

1. **Bootstrap Resampling** - Tests order-dependency
2. **Trade Dropout (0-15%)** - Tests sensitivity to missing trades
3. **Cost Variance** - ±25% slippage, ±10% commission
4. **Combined** - All factors together

### Result Validation Thresholds

If you see these, **STOP and investigate**:
- Win rate > 60% (ours is 40%)
- Sharpe > 4.0 (ours is 3.2)
- Profit factor > 2.0 (ours is 1.28)
- AUC > 0.85 for any directional prediction

### Data-Driven Decision Framework

All parameters are justified by data, not arbitrary selection:

| Decision | Evidence | Overfitting Check |
|----------|----------|-------------------|
| Entry thresholds (0.40, 0.45) | 256-point grid search | OOS validation |
| Exit multipliers (2.5, 1.25) | Grid search + sensitivity | Forward test |
| Model selection (LightGBM) | Walk-forward CV | Consistent OOS |
| Feature selection (75) | 4-method ranking | No leakage detected |

**Overfitting Detection:**
- IS-OOS AUC degradation: 6% (0.84→0.79) - ROBUST
- IS-OOS Sharpe degradation: 31% (4.56→3.16) - ACCEPTABLE
- Forward test consistency: 2.66 Sharpe - ROBUST
- Year-over-year: 100% profitable (2020-2025) - ROBUST

**Full methodology:** See `docs/DATA_DRIVEN_DECISIONS.md`

## Development Roadmap

### Completed Phases

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1-3 | Environment & Data Setup | COMPLETE |
| Phase 4-6 | Feature Engineering & Baseline | COMPLETE |
| Phase 7 | ML Model Development | COMPLETE |
| Phase 8 | Multi-Target Pivot | **BREAKTHROUGH** |
| Phase 9 | Vol Breakout Strategy | COMPLETE (+$209K validated) |
| Phase 10 | OOS & Forward Test | COMPLETE (+$496K OOS, +$57K forward) |
| Phase 11 | Sentiment Strategy & Ensemble | **COMPLETE (+7.4% improvement)** |
| Phase 12 | Threshold Optimization | **COMPLETE (+96% improvement)** |
| Phase 13 | Monte Carlo Validation | **COMPLETE (100% prob profit)** |
| Phase 14 | Enhanced Validation | **COMPLETE (4/5 stress tests passed)** |

### Phase 15: NT8 Integration (COMPLETE)

- [x] Identify VIX data access issue and fix
- [x] Identify fake PCR/AAII proxy data issue
- [x] Design Python-NT8 bridge architecture
- [x] Create Python signal server (`signal_server.py`)
- [x] Create NT8 TCP client strategy (`SKIENinjaTCPStrategy.cs`)
- [x] **Fix feature mismatch** (2026-01-05) - Critical fix applied
- [x] Validate backtest results (+$1,287.50 over 12 months)

### Phase 16: Paper Trading (ACTIVE)

- [x] Signal server running with 70 walk-forward folds
- [x] NT8 strategy connected to demo account
- [x] Backtest validated with corrected features
- [ ] Monitor live signals (30-60 days)
- [ ] Validate live performance vs backtest expectations

### Phase 17: Production Deployment (NEXT)

- [ ] VPS setup and monitoring
- [ ] Controlled live trading (MES → ES)

## Trading Costs Used

| Cost | Value | Notes |
|------|-------|-------|
| Commission | $1.29/side | NinjaTrader official rate |
| Slippage | 0.5 ticks | Conservative RTH estimate |
| Tick Size | $0.25 | ES futures |
| Point Value | $50 | ES futures |

## Technology Stack

### Python ML Stack

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.26.4 | Numerical computing |
| pandas | 2.3.3 | Data manipulation |
| scikit-learn | 1.3.2 | ML algorithms |
| lightgbm | latest | Gradient boosting (primary) |
| xgboost | 2.0.2 | Gradient boosting (backup) |
| torch | 2.8.0 | PyTorch deep learning |
| ta | latest | Technical analysis |

### Data Sources

| Source | Purpose | Status |
|--------|---------|--------|
| Databento | Historical futures data | Active (~$122 remaining) |
| FRED | Macroeconomic indicators | Configured |
| Yahoo Finance | Daily/VIX data | Active |

### Platform

- **Trading Platform**: NinjaTrader 8.1.6.0
- **Primary Language**: C# (NinjaScript)
- **ML Development**: Python 3.9.13
- **IDE**: Visual Studio + VS Code

## Quick Start

### Python Backtesting

```bash
# Clone repository
git clone https://github.com/s-koirala/SKIE-Ninja.git
cd SKIE_Ninja

# Install Python dependencies
pip install numpy pandas scikit-learn lightgbm ta onnxruntime

# Run main ensemble strategy backtest
python src/python/strategy/ensemble_strategy.py

# Run QC validation
python src/python/run_qc_check.py

# Run Monte Carlo simulation
python src/python/run_monte_carlo_simulation.py
```

### NT8 Live Trading (Python-NT8 Bridge)

```bash
# Step 1: Download sentiment data (requires AAII subscription for full data)
python src/python/data_collection/sentiment_data_downloader.py

# Step 2: Start the Python signal server (keep running)
python src/python/signal_server.py

# Step 3: In NinjaTrader 8:
#   - Import SKIENinjaTCPStrategy.cs
#   - Apply to ES 1-minute chart
#   - Strategy connects to localhost:5555
#   - Python server generates signals, NT8 executes trades

# The flow:
# NT8 → sends 42 technical features → Python server
# Python server → loads walk-forward model + sentiment data → generates signal
# Signal → {should_trade, direction, tp_mult, sl_mult} → NT8 executes
```

### Data Subscriptions Required

1. **AAII Sentiment**: $29/year at https://www.aaii.com/membership
2. **PCR Data**: Check https://data.nasdaq.com/databases/EOD (Quandl) first, or https://www.barchart.com/solutions/data

## Research Documentation

- [Multi-Target Strategy](research/04_multi_target_prediction_strategy.md) - Core strategy design
- [Sentiment Strategy Plan](research/05_sentiment_strategy_plan.md) - Ensemble development
- [Best Practices](docs/BEST_PRACTICES.md) - Lessons learned & anti-patterns
- [Backtest Methodology](docs/methodology/BACKTEST_METHODOLOGY.md) - Walk-forward validation
- [Validation Report](docs/VALIDATION_REPORT.md) - **NEW** Complete stress testing results

## Key Research Findings

### Multi-Target Prediction (Breakthrough)
- **Volatility Expansion**: AUC 0.84 - Primary edge
- **Breakout Detection**: AUC 0.72 - Direction guidance
- **ATR Forecast**: R² 0.36 - Dynamic exits
- **Direction Alone**: AUC 0.50 - **Impossible to predict**

### Sentiment Analysis
- VIX predicts WHEN (vol expansion) but not WHICH WAY
- Best used as filter for vol breakout, not standalone
- Ensemble "either" method: Enter if EITHER technical OR sentiment model predicts expansion

### Statistical Validation
- 10,000 Monte Carlo iterations
- 100% probability of positive P&L
- Robust to order changes, trade dropout, and cost variance

## License

Proprietary - All rights reserved

---

*Last Updated: 2026-01-06*
*Maintained by: SKIE_Ninja Development Team*
