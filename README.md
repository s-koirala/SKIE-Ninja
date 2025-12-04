# SKIE_Ninja - Smart Algorithmic Trading System for NinjaTrader

A comprehensive algorithmic trading system leveraging machine learning, macroeconomic factors, and advanced quantitative strategies for futures trading on the NinjaTrader platform.

## Project Status: Phase 7 - ML Model Development (IN PROGRESS)

**Current Results:**
| Model | AUC-ROC | Accuracy | F1 Score | Notes |
|-------|---------|----------|----------|-------|
| **LightGBM** | **84.21%** | 74.40% | 73.36% | Best overall |
| XGBoost | 84.07% | 75.23% | 67.22% | Strong baseline |
| RandomForest | 76.83% | 71.07% | 63.13% | Baseline |
| GRU | 65.60% | 62.03% | 62.63% | Deep learning |
| LSTM | 65.28% | 62.00% | 60.99% | Deep learning |

**Training Configuration:**
- **Data**: 684,410 bars of ES 1-minute data (2023-2024)
- **Timeframe**: 5-minute RTH bars (optimal from grid search)
- **Walk-Forward**: 180-day train, 5-day test, 42-bar embargo
- **Features**: 75 top-performing from 474 candidates

**Latest Additions (2025-12-04):**
- Comprehensive walk-forward backtesting with full metrics
- Purged K-Fold CV for LSTM/GRU (addresses overfitting)
- Quality control validation framework
- RTH-only trading enforcement
- Data leakage detection

## Repository Structure

```
SKIE_Ninja/
├── src/
│   └── python/
│       ├── data_collection/       # Data downloaders and loaders
│       │   ├── databento_downloader.py   # Databento API integration
│       │   ├── ninjatrader_loader.py     # NinjaTrader format parser
│       │   ├── fred_collector.py         # FRED macroeconomic data
│       │   ├── free_data_collector.py    # Yahoo Finance data
│       │   └── alternative_data_collector.py  # Reddit, News, Fear&Greed
│       ├── feature_engineering/   # Feature calculation modules
│       │   ├── price_features.py         # 79 price-based features
│       │   ├── technical_indicators.py   # 105 technical indicators
│       │   ├── microstructure_features.py # 71 microstructure features
│       │   ├── sentiment_features.py     # VIX, COT, sentiment proxies
│       │   ├── intermarket_features.py   # Cross-asset correlations
│       │   ├── alternative_features.py   # Alternative data features
│       │   ├── advanced_targets.py       # Pyramiding/DDCA targets
│       │   ├── feature_pipeline.py       # Unified feature builder
│       │   └── feature_selection.py      # Multi-method feature ranking
│       ├── models/                # ML model training
│       │   ├── model_trainer.py          # XGBoost, RF, LightGBM training
│       │   ├── deep_learning_trainer.py  # LSTM/GRU models
│       │   ├── purged_cv_rnn_trainer.py  # Purged K-Fold CV for RNNs (NEW)
│       │   └── rnn_hyperparameter_optimizer.py  # Grid search for RNNs
│       ├── backtesting/           # Walk-forward backtesting (NEW)
│       │   ├── walk_forward_backtest.py  # Original WF backtest
│       │   └── comprehensive_backtest.py # Full metrics backtest (NEW)
│       ├── quality_control/       # Validation framework (NEW)
│       │   └── validation_framework.py   # Data & model validation
│       └── utils/                 # Utility functions
│           └── data_resampler.py        # OHLCV resampling utilities
├── data/
│   ├── raw/
│   │   └── market/               # Downloaded market data (ES, NQ, etc.)
│   ├── processed/                # Feature rankings and selections
│   ├── models/                   # Trained model files (.pkl, .pt)
│   └── backtest_results/         # Backtest outputs (NEW)
├── config/
│   ├── feature_config.yaml       # Feature configuration
│   ├── api_keys.py              # API key management
│   └── project_memory.md        # Project decisions log
├── research/
│   ├── 01_initial_research.md   # Platform research (494 lines)
│   └── 02_comprehensive_variables_research.md  # Variables (2,692 lines)
└── docs/
    ├── architecture/             # System architecture docs
    └── methodology/              # Backtesting methodology (NEW)
        └── BACKTEST_METHODOLOGY.md
```

## Data Available

| Source | Instrument | Timeframe | Bars | Years |
|--------|-----------|-----------|------|-------|
| Databento | ES (S&P 500) | 1-min | 684,410 | 2023-2024 |
| Databento | NQ (Nasdaq) | 1-min | 684,432 | 2023-2024 |
| Databento | ES | 1-min | ~340K each | 2020, 2021, 2022 |
| Databento | NQ | 1-min | ~340K each | 2020, 2021, 2022 |
| Databento | YM, GC, CL, ZN | 1-min | Various | 2023-2024 |
| Databento | ES MBP-10 | L2 sample | Sample | 2024 |
| Yahoo Finance | ES, NQ, VIX, GC, CL, ZN, DX | Daily | ~500 each | 2+ years |
| PortaraNinja | ES, NQ | Sample | 67,782 + 42,649 | Sample |

## Feature Engineering (474 Features Implemented)

| Category | Features | Status |
|----------|----------|--------|
| 1. Price-Based | 79 | ✅ Complete |
| 2. Technical Indicators | 105 | ✅ Complete |
| 3. Macroeconomic (FRED) | 12 | ✅ Complete |
| 4. Microstructure | 71 | ✅ Complete |
| 5. Sentiment & Positioning | 43 | ✅ Complete |
| 6. Intermarket | 84 | ✅ Complete |
| 7. Seasonality & Calendar | 58 | ✅ Complete |
| 8. Regime & Fractal | 19 | ✅ Complete |
| 9. Alternative Data | 31 | ✅ Complete |
| 10. Lagged Features | 67 | ✅ Complete |
| 11. Interaction Features | 8 | ✅ Complete |
| 12. Target Labels | 11 | ✅ Complete |
| **TOTAL** | **~500** | **95% Complete** |

### Top Performing Features (by multi-method ranking)
1. `pyramid_rr_5/10/20` - Pyramiding reward-to-risk ratios
2. `pivot_high_*` / `pivot_low_*` - Support/Resistance pivot detection
3. `ddca_sell_success_*` - DDCA trading signals
4. `sell_pressure` / `buy_pressure` - Order flow pressure
5. `dist_to_R1_*` - Distance to resistance levels
6. `atr_20` - Average True Range
7. `stoch_diff_14` - Stochastic oscillator divergence
8. `return_rolling_std_*` - Rolling volatility

## ML Model Results

### XGBoost (Best Performer)
```
Accuracy:  75.23%
Precision: 76.22%
Recall:    60.12%
F1 Score:  67.22%
AUC-ROC:   84.07%
Log Loss:  0.4699
Brier:     0.1592
```

### RandomForest (Baseline)
```
Accuracy:  71.07%
Precision: 68.40%
Recall:    58.62%
F1 Score:  63.13%
AUC-ROC:   76.83%
```

### Training Configuration
- Walk-forward validation with 61 folds (180-day train, 5-day test)
- 42-bar embargo period (~3.5 hours) between train/test
- 300 estimators per model
- Early stopping for gradient boosting
- Feature scaling with StandardScaler

## Backtesting Framework

### Comprehensive Walk-Forward Backtest

The backtesting system provides detailed metrics for strategy evaluation:

| Category | Metrics |
|----------|---------|
| **P&L** | Gross, Net, Commission, Slippage |
| **Win/Loss** | Win Rate, Avg Win, Avg Loss, Max Win/Loss |
| **KPIs** | Profit Factor, Payoff Ratio, Expectancy |
| **Drawdown** | Max DD ($, %), Duration, Avg DD |
| **Duration** | Bars Held, Time in Trade (min) |
| **Risk-Adjusted** | Sharpe, Sortino, Calmar Ratios |
| **MFE/MAE** | Max Favorable/Adverse Excursion |

### Quality Control Validation

Automatic checks for:
- Data quality (OHLCV relationships, missing values)
- Feature quality (no leakage, no infinite values)
- Backtest realism (costs, RTH compliance)
- Statistical validity (suspicious metrics detection)

### Usage

```bash
# Run comprehensive backtest
python src/python/run_validated_backtest.py

# Or programmatically
from backtesting import run_comprehensive_backtest
trades, metrics, report = run_comprehensive_backtest(
    prices, features,
    target_col='target_direction_1',
    model_type='lightgbm'
)
```

## Development Roadmap

### ✅ Phase 1: Foundation - COMPLETED
- [x] Research NinjaTrader ecosystem
- [x] Identify programming languages and API methods
- [x] Research ML model feasibility
- [x] Identify data sources
- [x] Document trading strategies and best practices

### ✅ Phase 2: Extended Research - COMPLETED
- [x] Deep dive into quantitative literature
- [x] Comprehensive variable identification (500+ features)
- [x] Macroeconomic and microeconomic factors
- [x] Alternative data source research
- [x] Market microstructure analysis

### ✅ Phase 3: Environment Setup - COMPLETED
- [x] NinjaTrader 8.1.6.0 installed and configured
- [x] Visual Studio installed
- [x] Python 3.9.13 with full ML stack
- [x] Databento API configured ($125 budget)
- [x] FRED API configured

### ✅ Phase 4: Data Collection - COMPLETED
- [x] Databento 1-minute data: ES, NQ (2020-2024)
- [x] Databento 1-minute data: YM, GC, CL, ZN (2023-2024)
- [x] Databento Level 2 MBP-10 sample data
- [x] Yahoo Finance daily data (8 instruments)
- [x] FRED macroeconomic data integration

### ✅ Phase 5: Feature Engineering - COMPLETED
- [x] 474 features implemented across 12 categories
- [x] Price, technical, microstructure features
- [x] Sentiment (VIX, COT) features
- [x] Intermarket correlation features
- [x] Alternative data (Reddit, News, Fear&Greed)
- [x] Advanced targets (Pyramiding, DDCA, S/R pivots)
- [x] Feature selection pipeline (4 ranking methods)
- [x] 75 top features selected

### ✅ Phase 6: Baseline Strategy - COMPLETED
- [x] XGBoost classification model trained
- [x] RandomForest baseline model trained
- [x] Walk-forward cross-validation implemented
- [x] Model performance: 84% AUC-ROC achieved

### 🔄 Phase 7: ML Model Development - IN PROGRESS
- [x] XGBoost and RandomForest models
- [x] LightGBM model (84.21% AUC - best performer)
- [x] LSTM/GRU time series models (with Purged CV)
- [x] Comprehensive walk-forward backtesting
- [x] Quality control validation framework
- [x] RTH-only trading enforcement
- [x] Data leakage detection
- [ ] Transformer-based models
- [ ] Model ensembling and stacking
- [ ] ONNX export for NinjaTrader

### 📋 Phase 8: Validation
- [ ] Extended walk-forward testing
- [ ] Monte Carlo simulations (1000+ runs)
- [ ] Out-of-sample testing on 2020-2022 data
- [ ] Regime-specific performance analysis

### 📋 Phase 9: Paper Trading
- [ ] Deploy to NinjaTrader simulation
- [ ] Monitor 30-60 days
- [ ] Compare live vs backtest performance

### 📋 Phase 10: Production Deployment
- [ ] Deploy with minimal capital
- [ ] VPS setup and monitoring
- [ ] Risk management implementation
- [ ] Scale based on performance

## Technology Stack

### Python ML Stack
| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.26.4 | Numerical computing |
| pandas | 2.3.3 | Data manipulation |
| scikit-learn | 1.3.2 | ML algorithms |
| xgboost | 2.0.2 | Gradient boosting |
| keras | 3.10.0 | Neural networks |
| torch | 2.8.0 | PyTorch deep learning |
| transformers | 4.57.1 | Transformer models |
| onnxruntime | 1.19.2 | ONNX model inference |

### Data Sources
| Source | Purpose | Status |
|--------|---------|--------|
| Databento | Historical futures data | ✅ Active (~$122 remaining) |
| FRED | Macroeconomic indicators | ✅ API configured |
| Yahoo Finance | Daily/intermarket data | ✅ Active |
| Reddit/News | Sentiment analysis | ✅ Ready |

### Platform
- **Trading Platform**: NinjaTrader 8.1.6.0
- **Primary Language**: C# (NinjaScript)
- **ML Development**: Python 3.9.13
- **IDE**: Visual Studio + VS Code

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/SKIE_Ninja.git
cd SKIE_Ninja

# Install Python dependencies
pip install pandas numpy scikit-learn xgboost

# Configure API keys
cp config/api_keys.env.template config/api_keys.env
# Edit api_keys.env with your keys

# Build features and train model
python -c "
from src.python.feature_engineering.feature_pipeline import build_feature_matrix
from src.python.models.model_trainer import train_models
import pandas as pd

# Load data
es_data = pd.read_csv('data/raw/market/ES_1min_databento.csv', index_col=0, parse_dates=True)

# Build features
features = build_feature_matrix(es_data, symbol='ES')

# Train models
results = train_models(features, target_col='target_direction_1')
print(results)
"
```

## Research Documentation

- [Initial Research](research/01_initial_research.md) - 494 lines on NinjaTrader ecosystem
- [Comprehensive Variables](research/02_comprehensive_variables_research.md) - 2,692 lines on 500+ features

## Key Research Findings

- **Pyramiding R:R Features**: Top predictive power (avg rank 1-3)
- **S/R Pivot Detection**: Strong signals (avg rank 4-12)
- **Order Flow Pressure**: Key microstructure signal (avg rank 15)
- **Volatility (ATR)**: Important regime indicator (avg rank 22)
- **Social Media Sentiment**: 87% forecast accuracy (literature)
- **Fractal Analysis**: +12.71% vs +7.06% traditional (literature)

## License

Proprietary - All rights reserved
