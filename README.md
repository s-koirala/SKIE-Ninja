# SKIE_Ninja - Smart Algorithmic Trading System for NinjaTrader

A comprehensive algorithmic trading system leveraging machine learning, macroeconomic factors, and advanced quantitative strategies for futures trading on the NinjaTrader platform.

## Project Overview

This project aims to research, develop, train, and deploy a sophisticated algorithmic trading system that combines:
- Machine learning models (Random Forest, Neural Networks)
- Traditional technical analysis
- Macroeconomic and microeconomic factors
- Alternative data sources
- Robust risk management

## Repository Structure

```
SKIE_Ninja/
├── research/           # Research findings and literature reviews
├── data/              # Data collection scripts and storage
├── strategies/        # NinjaScript strategy implementations
├── models/            # Machine learning model development
├── backtests/         # Backtesting results and analysis
├── docs/              # Documentation and guides
└── utils/             # Utility scripts and helpers
```

## Research Documentation

### Phase 1: Platform & Strategy Research
**[research/01_initial_research.md](research/01_initial_research.md)** - 494 lines
- NinjaTrader programming (C# / NinjaScript)
- API connection methods (6 different approaches)
- ML model feasibility (Random Forest vs Neural Networks)
- Historical data sources (PortaraNinja, Kinetick, CQG)
- Trading strategies (scalping, day trading, swing trading)
- Best practices for backtesting and deployment

### Phase 2: Comprehensive Variables & Factors
**[research/02_comprehensive_variables_research.md](research/02_comprehensive_variables_research.md)** - 2,692 lines
- **500-1,000+ features identified** across 13 major categories
- Quantitative trading publications (2024-2025 academic papers)
- Top quant forums (Wilmott, Elite Trader, QuantConnect, futures.io)
- Macroeconomic variables (GDP, inflation, labor market, monetary policy)
- Market microstructure (order flow, bid-ask spread, liquidity)
- Alternative data sources (sentiment, satellite, transaction data)
- Technical indicators and ML features
- Sentiment & positioning (COT, put/call, VIX, gamma exposure)
- Intermarket relationships (bonds, commodities, currencies)
- Seasonality & calendar effects
- Statistical arbitrage (cointegration, pairs trading)
- Fractal analysis & regime detection (Hurst exponent)
- Options market & dealer positioning (GEX, dark pools)
- Time-based features (hour-of-day, intraday patterns)
- Order book microstructure (LOB, depth, OFI)

### Key Research Findings
- **Social media sentiment**: 87% forecast accuracy
- **Alternative data adoption**: 65% of hedge funds, +3% annual returns
- **Satellite imagery**: 18% better earnings estimates
- **Fractal model**: +12.71% vs +7.06% traditional strategy
- **Gamma exposure**: Real-time volatility prediction
- **Hurst exponent**: Regime detection (trending vs mean-reverting)

## Technology Stack

- **Platform**: NinjaTrader 8
- **Primary Language**: C# (NinjaScript)
- **ML Frameworks**:
  - ML.NET (officially supported by NinjaTrader)
  - Python: scikit-learn, TensorFlow, Keras
  - Accord.NET, Encog (C# ML libraries)
- **Data Sources**:
  - PortaraNinja (official historical data, 1899-present)
  - Kinetick (5+ years minute data)
  - Alternative data vendors
  - Real-time market microstructure (Level 2)
- **Development**: Visual Studio, Git
- **Deployment**: VPS hosting for 24/7 operation

## Development Roadmap

### ✅ Phase 1: Foundation (Weeks 1-2) - COMPLETED
- [x] Research NinjaTrader ecosystem
- [x] Identify programming languages and API methods
- [x] Research ML model feasibility
- [x] Identify data sources
- [x] Document trading strategies and best practices

### ✅ Phase 2: Extended Research (Week 2) - COMPLETED
- [x] Deep dive into quantitative literature
- [x] Comprehensive variable identification (500-1,000+ features)
- [x] Macroeconomic and microeconomic factors
- [x] Alternative data source research
- [x] Market microstructure analysis
- [x] Regime detection methodologies

### 🔄 Phase 3: Environment Setup (Week 3) - IN PLANNING
- [ ] Set up NinjaTrader 8 development environment
- [ ] Install Visual Studio and configure NinjaScript
- [ ] Acquire historical data subscriptions
- [ ] Set up Python ML development environment
- [ ] Configure data pipeline architecture

### 📋 Phase 4: Data Collection (Weeks 4-5)
- [ ] Implement data collection scripts
- [ ] Historical data acquisition (5-10 years)
- [ ] Real-time data feed integration
- [ ] Alternative data API integration
- [ ] Data storage and preprocessing pipeline

### 📋 Phase 5: Feature Engineering (Weeks 6-8)
- [ ] Implement priority features (top 100)
- [ ] Technical indicators in NinjaScript
- [ ] Macroeconomic data integration
- [ ] Market microstructure features
- [ ] Time-based and regime features

### 📋 Phase 6: Baseline Strategy (Weeks 9-10)
- [ ] Develop simple mean reversion strategy on ES
- [ ] Implement in NinjaScript
- [ ] Backtest with realistic costs
- [ ] Establish performance baseline

### 📋 Phase 7: ML Model Development (Weeks 11-14)
- [ ] Train Random Forest models
- [ ] Develop LSTM for time series
- [ ] Feature selection and importance analysis
- [ ] Regime-specific model training
- [ ] Model ensemble and stacking

### 📋 Phase 8: Validation (Weeks 15-16)
- [ ] Walk-forward testing
- [ ] Monte Carlo simulations (1000+ runs)
- [ ] Out-of-sample testing
- [ ] Regime-specific performance analysis

### 📋 Phase 9: Paper Trading (Weeks 17-20)
- [ ] Deploy to simulation account
- [ ] Monitor 30-60 days minimum
- [ ] Compare live vs backtest performance
- [ ] Refine based on real market behavior

### 📋 Phase 10: Production Deployment (Week 21+)
- [ ] Deploy with minimal capital
- [ ] VPS setup and monitoring
- [ ] Risk management implementation
- [ ] Scale gradually based on performance
- [ ] Continuous monitoring and optimization

## Variable Categories (13 Major Categories)

1. **Price-Based Features** (~50): OHLC, returns, ratios, ranges
2. **Technical Indicators** (~100): Trend, momentum, volatility, volume
3. **Macroeconomic** (~50): GDP, inflation, employment, monetary policy
4. **Microstructure** (~40): Order flow, spread, depth, liquidity
5. **Sentiment** (~30): COT, put/call, VIX, surveys
6. **Intermarket** (~25): Correlations, cross-asset signals
7. **Seasonality** (~30): Calendar effects, intraday patterns
8. **Statistical Arbitrage** (~20): Cointegration, pairs trading
9. **Regime Detection** (~15): Hurst exponent, fractal analysis
10. **Alternative Data** (~50): Sentiment, satellite, transaction data
11. **Options Market** (~25): Gamma exposure, dealer positioning
12. **Time-Based** (~30): Hour-of-day, session periods
13. **Order Book** (~40): LOB depth, OFI, liquidity metrics

**Total: 500-1,000+ features** (including lagged and interaction terms)

## Getting Started

1. **Review Research**: Start with [research/01_initial_research.md](research/01_initial_research.md)
2. **Explore Variables**: Deep dive into [research/02_comprehensive_variables_research.md](research/02_comprehensive_variables_research.md)
3. **Environment Setup**: Coming soon - development environment guide
4. **Data Collection**: Coming soon - data pipeline implementation
5. **Model Training**: Coming soon - ML model development guide

## License

Proprietary - All rights reserved
