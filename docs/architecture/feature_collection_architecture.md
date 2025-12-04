# SKIE_Ninja Feature Collection Architecture

**Created**: 2025-11-30
**Status**: Design Phase
**Target**: 1000+ Features Across 13 Categories

---

## Executive Summary

This document outlines the architecture for collecting, processing, and storing 1000+ features from multiple data sources for ML-based algorithmic trading on NinjaTrader.

---

## 1. Data Source Strategy

### Challenge: Training vs Live Trading Data Consistency

The user requirement is to use the **same data source for training and live trading**. However:

| Source | Historical Depth | Live Capability | Cost |
|--------|------------------|-----------------|------|
| Rithmic | ~1 year tick | Yes (low latency) | Paid |
| CQG | ~1 month tick | Yes | Often free |
| PortaraNinja | 1899-present | No (historical only) | Paid |

### Recommended Hybrid Approach

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA SOURCE ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TRAINING PHASE                    LIVE TRADING PHASE               │
│  ──────────────                    ──────────────────               │
│                                                                      │
│  ┌─────────────────┐               ┌─────────────────┐              │
│  │  PortaraNinja   │               │     Rithmic     │              │
│  │  (5-10 years)   │               │  (Real-time)    │              │
│  │  Historical     │               │  + 1 year hist  │              │
│  └────────┬────────┘               └────────┬────────┘              │
│           │                                  │                       │
│           ▼                                  ▼                       │
│  ┌─────────────────────────────────────────────────────┐            │
│  │           UNIFIED FEATURE PIPELINE                   │            │
│  │                                                      │            │
│  │  • Same feature calculations                         │            │
│  │  • Same normalization                                │            │
│  │  • Consistent data format                            │            │
│  └──────────────────────────────────────────────────────┘            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Why This Works:**
1. PortaraNinja uses CQG Data Factory - same source as Rithmic/CQG feeds
2. Both produce clean, adjusted futures data
3. Feature calculations are deterministic (same input → same output)
4. Walk-forward validation bridges training/live gap

---

## 2. Feature Categories & Data Sources

### 13-Category Feature Matrix

| # | Category | Feature Count | Primary Source | Secondary Source |
|---|----------|---------------|----------------|------------------|
| 1 | Price-Based | ~50 | NinjaTrader/Rithmic | - |
| 2 | Technical Indicators | ~100 | NinjaTrader (built-in) | Python (TA-Lib) |
| 3 | Macroeconomic | ~50 | FRED API | Trading Economics |
| 4 | Microstructure | ~40 | Rithmic Level 2 | NinjaTrader DOM |
| 5 | Sentiment & Positioning | ~30 | CFTC COT, CBOE | News APIs |
| 6 | Intermarket | ~25 | NinjaTrader (multi-symbol) | - |
| 7 | Seasonality & Calendar | ~30 | Python (computed) | - |
| 8 | Statistical Arbitrage | ~20 | Python (statsmodels) | - |
| 9 | Regime & Fractal | ~15 | Python (custom Hurst) | - |
| 10 | Alternative Data | ~50 | Twitter/Reddit APIs | Google Trends |
| 11 | Lagged & Transformed | ~200 | Python (computed) | - |
| 12 | Interaction Features | ~50-100 | Python (computed) | - |
| 13 | Target Labels | ~10 | Python (computed) | - |

**Total: 670-720 base features → 1000+ with lags/interactions**

---

## 3. Data Collection Pipelines

### Pipeline 1: Market Data (NinjaTrader + Rithmic)

```
Source: Rithmic → NinjaTrader 8
├── Level 1: Bid, Ask, Last, Volume
├── Level 2: Full order book depth (10+ levels)
├── Time & Sales: Every trade with direction
└── Output: NinjaTrader .ntd files or CSV export

Features Generated:
├── Category 1: Price-Based (~50)
├── Category 2: Technical Indicators (~100)
├── Category 4: Microstructure (~40)
├── Category 6: Intermarket (~25)
└── Subtotal: ~215 features
```

### Pipeline 2: Macroeconomic Data (FRED API)

```python
# Python Pipeline using fredapi
Source: Federal Reserve Economic Data (FRED)
├── GDP growth rate, surprises
├── CPI (headline & core)
├── Non-farm payrolls, unemployment
├── Federal Funds Rate
├── Treasury yields (2Y, 10Y, 30Y)
├── Yield curve slope
└── Output: Pandas DataFrame with vintage dates

Features Generated:
├── Category 3: Macroeconomic (~50)
└── Subtotal: ~50 features

Key Consideration: FRED provides revision history
(vintage dates) to avoid look-ahead bias in backtesting
```

### Pipeline 3: Sentiment & Positioning (Multiple APIs)

```
Sources:
├── CFTC: COT Report (weekly, Tuesdays)
│   └── Commercial, Non-Commercial, Non-Reportable positions
├── CBOE: VIX, Put/Call ratios, SKEW
│   └── Daily/Intraday
├── Twitter/X API: Sentiment analysis
│   └── NLP processing with transformers
├── Reddit API: WallStreetBets mentions
│   └── Keyword tracking + sentiment
└── News APIs: Bloomberg, RavenPack (premium)

Features Generated:
├── Category 5: Sentiment & Positioning (~30)
├── Category 10: Alternative Data (~50)
└── Subtotal: ~80 features
```

### Pipeline 4: Computed Features (Python)

```python
# Pure computation - no external data needed
Sources: Previously collected data
├── Lagged values (1, 2, 3, 5, 10, 20 periods)
├── Rolling statistics (mean, std, min, max)
├── Cyclical encoding (sin/cos of time)
├── Hurst exponent calculation
├── Cointegration statistics
└── Interaction terms

Features Generated:
├── Category 7: Seasonality & Calendar (~30)
├── Category 8: Statistical Arbitrage (~20)
├── Category 9: Regime & Fractal (~15)
├── Category 11: Lagged & Transformed (~200)
├── Category 12: Interaction Features (~50-100)
├── Category 13: Target Labels (~10)
└── Subtotal: ~325-375 features
```

---

## 4. System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SKIE_NINJA ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  NinjaTrader │  │   FRED API   │  │  CFTC/CBOE   │  │ Social APIs  │ │
│  │   + Rithmic  │  │   (Python)   │  │   (Python)   │  │  (Python)    │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                 │                 │                 │          │
│         ▼                 ▼                 ▼                 ▼          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                     DATA INGESTION LAYER                            ││
│  │  • CSV/Parquet storage                                              ││
│  │  • Timestamp alignment                                              ││
│  │  • Missing data handling                                            ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                   FEATURE ENGINEERING LAYER                         ││
│  │                                                                     ││
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       ││
│  │  │Technical│ │  Macro  │ │Sentiment│ │ Regime  │ │  Lags   │       ││
│  │  │Indicators│ │Features │ │ Scores  │ │Detection│ │& Interact│      ││
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘       ││
│  │                                                                     ││
│  │  Output: 1000+ normalized features per timestamp                    ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                      FEATURE STORE (Parquet/SQLite)                 ││
│  │                                                                     ││
│  │  • Historical features for training                                 ││
│  │  • Real-time features for inference                                 ││
│  │  • Feature versioning and metadata                                  ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                    │                                     │
│                    ┌───────────────┴───────────────┐                    │
│                    ▼                               ▼                    │
│  ┌────────────────────────────┐  ┌────────────────────────────────────┐│
│  │      ML TRAINING           │  │       LIVE INFERENCE               ││
│  │      (Python)              │  │       (NinjaTrader)                ││
│  │                            │  │                                    ││
│  │  • Random Forest           │  │  • Load trained model              ││
│  │  • XGBoost                 │  │  • Real-time feature calc          ││
│  │  • LSTM/Transformer        │  │  • Signal generation               ││
│  │  • Regime-specific models  │  │  • Order execution                 ││
│  │                            │  │                                    ││
│  │  Output: .onnx models      │  │  Input: .onnx models               ││
│  └────────────────────────────┘  └────────────────────────────────────┘│
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Flow: Training vs Live

### Training Mode

```
1. Historical Data Collection
   └── PortaraNinja (5-10 years) → NinjaTrader → CSV Export

2. External Data Collection (Python)
   ├── FRED API → Macroeconomic features
   ├── CFTC → COT data
   ├── CBOE → VIX, options data
   └── Social APIs → Sentiment scores

3. Feature Engineering (Python)
   └── Merge all sources by timestamp
   └── Calculate derived features
   └── Handle missing data
   └── Normalize features

4. Model Training (Python)
   └── Train/validation/test splits
   └── Walk-forward validation
   └── Export model to ONNX

5. Output
   └── Trained model (.onnx)
   └── Feature scaler parameters
   └── Feature importance rankings
```

### Live Mode

```
1. Real-time Market Data
   └── Rithmic → NinjaTrader (continuous)

2. Scheduled Data Updates
   ├── FRED → Every release (8:30 AM ET typical)
   ├── COT → Weekly (Friday 3:30 PM ET)
   ├── VIX → Real-time
   └── Sentiment → Configurable (5min-1hr)

3. Feature Calculation (NinjaScript + Python)
   └── Calculate same features as training
   └── Apply same normalization
   └── Feed to loaded ONNX model

4. Signal Generation
   └── Model prediction
   └── Confidence threshold
   └── Regime filter

5. Order Execution
   └── NinjaTrader order management
```

---

## 6. Technology Stack

### NinjaTrader Side (C#/NinjaScript)
- Market data collection
- Technical indicator calculation
- Order execution
- ONNX model inference (via ML.NET)

### Python Side
- Data collection from external APIs
- Feature engineering
- Model training
- Backtesting

### Communication Bridge
- **Option A**: File-based (CSV/Parquet)
  - NinjaTrader exports data → Python reads
  - Python exports model → NinjaTrader loads
  - Simple, reliable, some latency

- **Option B**: TCP Socket
  - Real-time communication
  - Lower latency
  - More complex implementation

- **Option C**: Shared Database (SQLite/PostgreSQL)
  - Best for large datasets
  - ACID compliance
  - Moderate complexity

**Recommendation**: Start with Option A (file-based) for development, migrate to Option C for production.

---

## 7. API Keys & Access Required

| Source | API Type | Cost | Sign-up URL |
|--------|----------|------|-------------|
| FRED | REST API | FREE | https://fred.stlouisfed.org/docs/api/api_key.html |
| CFTC COT | Public data | FREE | https://www.cftc.gov/MarketReports/CommitmentsofTraders |
| CBOE | Market data | Varies | https://www.cboe.com/market_data_services/ |
| Twitter/X | API v2 | $100/mo+ | https://developer.twitter.com/ |
| Reddit | API | FREE (limited) | https://www.reddit.com/dev/api/ |
| Google Trends | Unofficial | FREE | pytrends library |
| Rithmic | Broker-based | Included | Via NinjaTrader brokerage |

---

## 8. Implementation Phases

### Phase 1: Core Market Data (Week 1-2)
- [ ] Configure Rithmic connection in NinjaTrader
- [ ] Export historical data to CSV
- [ ] Build Python data loader
- [ ] Implement Category 1-2 features (Price, Technical)

### Phase 2: External Data Integration (Week 3-4)
- [ ] Set up FRED API access
- [ ] Implement macroeconomic feature pipeline
- [ ] Set up CFTC COT data collection
- [ ] Implement Category 3, 5-6 features

### Phase 3: Advanced Features (Week 5-6)
- [ ] Implement Hurst exponent calculation
- [ ] Build regime detection system
- [ ] Implement lagged/interaction features
- [ ] Implement Category 7-12 features

### Phase 4: Alternative Data (Week 7-8)
- [ ] Set up Twitter/Reddit APIs
- [ ] Build NLP sentiment pipeline
- [ ] Implement Category 10 features
- [ ] Complete feature store

### Phase 5: Model Training (Week 9-10)
- [ ] Feature selection (importance analysis)
- [ ] Train baseline Random Forest
- [ ] Train regime-specific models
- [ ] Export to ONNX

### Phase 6: NinjaTrader Integration (Week 11-12)
- [ ] Build NinjaScript strategy shell
- [ ] Implement ONNX model loading
- [ ] Real-time feature calculation
- [ ] Paper trading validation

---

## 9. File Structure

```
SKIE_Ninja/
├── config/
│   ├── project_memory.md
│   ├── api_keys.env (gitignored)
│   └── feature_config.yaml
├── data/
│   ├── raw/
│   │   ├── market/          # NinjaTrader exports
│   │   ├── macro/           # FRED data
│   │   ├── sentiment/       # COT, VIX, social
│   │   └── alternative/     # Twitter, Reddit
│   ├── processed/
│   │   └── features/        # Computed features
│   └── models/
│       └── *.onnx           # Trained models
├── src/
│   ├── python/
│   │   ├── data_collection/
│   │   ├── feature_engineering/
│   │   ├── models/
│   │   └── utils/
│   └── ninjatrader/
│       ├── indicators/
│       ├── strategies/
│       └── addons/
├── notebooks/
│   └── exploration/
├── tests/
├── docs/
│   └── architecture/
└── research/
```

---

*Next Step: Create folder structure and begin implementation*
