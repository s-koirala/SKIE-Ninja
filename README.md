# SKIE_Ninja - Smart Algorithmic Trading System for NinjaTrader

A comprehensive algorithmic trading system leveraging machine learning, macroeconomic factors, and advanced quantitative strategies for futures trading on the NinjaTrader platform.

## Project Status: Phase 15 - PRODUCTION DEPLOYMENT IN PROGRESS

**Ensemble Strategy (Vol Breakout + VIX Sentiment) - RECOMMENDED**

| Test Period | Vol Breakout | Ensemble | Improvement |
|-------------|-------------|----------|-------------|
| In-Sample (2023-24) | +$209,351 | **+$224,813** | **+7.4%** |
| Out-of-Sample (2020-22) | +$496,380 | **+$502,219** | **+1.2%** |
| Forward Test (2025) | +$57,394 | **+$59,847** | **+4.3%** |

**Total Validated Edge**: $786,879 across 5 years of data

### Monte Carlo Validation: PASSED (5,000 iterations per test)

| Metric | Original | 95% CI (Combined) | Assessment |
|--------|----------|-------------------|------------|
| Net P&L | $502,219 | [$361K, $573K] | **100% prob positive** |
| Sharpe Ratio | 3.16 | [2.24, 3.33] | **Significantly > 0** |
| Profit Factor | 1.25 | [1.15, 1.35] | **Significantly > 1** |

### Enhanced Stress Testing: 4/5 PASSED

| Test | Condition | P(Profit>0) | Result |
|------|-----------|-------------|--------|
| Slippage | 3x baseline | 100% | **PASS** |
| Dropout | 50% trades | 100% | **PASS** |
| Adverse Selection | 20% winners removed | 100% | **PASS** |
| Black Swan | 5% frequency | 100% | **PASS** |
| Combined Extreme | All above | 0% | FAIL (expected) |

**Note**: Combined extreme test represents catastrophic conditions (3x slippage + 30% dropout + 20% adverse + black swan simultaneously) - unlikely to persist in real markets.

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
│   │   ├── deployment/                  # NinjaTrader integration (NEW)
│   │   │   └── ninja_signal_server.py           # Python signal server
│   │   ├── data_collection/             # Data downloaders and loaders
│   │   ├── feature_engineering/         # Feature calculation modules
│   │   ├── models/                      # ML model training
│   │   ├── backtesting/                 # Walk-forward backtesting
│   │   ├── quality_control/             # Validation framework
│   │   └── run_*.py                     # Various run scripts
│   └── ninjatrader/                     # NinjaScript strategies (NEW)
│       └── SKIENinjaStrategy.cs                 # NinjaTrader client
├── models/
│   └── production/                      # Trained model files for deployment
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

### Phase 15: Production Deployment (IN PROGRESS)

- [x] NinjaTrader account setup
- [x] Socket Bridge architecture implementation
- [ ] Platform walk-forward validation (Market Replay)
- [ ] Paper trading validation (30-60 days)
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

```bash
# Clone repository
git clone https://github.com/s-koirala/SKIE-Ninja.git
cd SKIE_Ninja

# Install Python dependencies
pip install numpy pandas scikit-learn lightgbm ta

# Run main ensemble strategy backtest
python src/python/strategy/ensemble_strategy.py

# Run QC validation
python src/python/run_qc_check.py

# Run Monte Carlo simulation
python src/python/run_monte_carlo_simulation.py

# Run OOS backtest
python src/python/run_ensemble_oos_backtest.py

# Run 2025 forward test
python src/python/run_ensemble_2025_forward_test.py
```

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

*Last Updated: 2025-12-15*
*Maintained by: SKIE_Ninja Development Team*
