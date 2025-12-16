# SKIE Ninja Deployment Infrastructure Guide

**Created**: 2025-12-15
**Updated**: 2025-12-15
**Purpose**: Data-driven analysis of optimal brokerage, data feeds, and infrastructure for production deployment
**Status**: Phase 15 - Active Implementation

---

## Executive Summary

This document provides a comprehensive analysis of infrastructure requirements for deploying the SKIE Ninja ensemble strategy to live trading. Recommendations are based on 2024-2025 benchmark data, academic research, and industry best practices.

**Key Recommendation**: AMP Futures or NinjaTrader Brokerage with Rithmic data feed, Kinetick CBOE add-on for VIX, and VPS co-location for optimal execution.

**Implementation Approach**: Socket Bridge architecture for direct Python strategy execution with minimal latency.

---

## 0. NinjaTrader Integration Architecture (RECOMMENDED)

### 0.1 Implementation Options Comparison

| Approach | Development Time | Latency | Maintenance | Recommendation |
|----------|------------------|---------|-------------|----------------|
| **Socket Bridge** | 1-2 days | ~5-10ms | LOW | **RECOMMENDED** |
| ONNX Export | 1-2 weeks | ~1ms | HIGH | Not recommended |
| Python.NET | 3-5 days | ~20-50ms | MEDIUM | Alternative |
| Full C# Port | 4-8 weeks | ~1ms | HIGH | Not recommended |

**Rationale**: For 5-minute bars with ~20 bar hold time, the ~5-10ms Socket Bridge latency is negligible. This approach:
- Preserves validated Python code exactly as tested
- Eliminates feature parity risk (Python vs C# calculations)
- Enables rapid iteration and debugging
- Reduces deployment complexity

### 0.2 Socket Bridge Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     NINJATRADER 8                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  SKIENinjaStrategy.cs (NinjaScript)                      │   │
│  │  - Receives bar data from Rithmic                        │   │
│  │  - Sends OHLCV + VIX to Python via TCP socket            │   │
│  │  - Receives trade signals (LONG/SHORT/FLAT)              │   │
│  │  - Executes orders via NinjaTrader order API             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ TCP Socket (localhost:5555)
                              │ JSON messages, ~5ms round-trip
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PYTHON SIGNAL SERVER                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ninja_signal_server.py                                   │   │
│  │  - Receives OHLCV data from NinjaTrader                  │   │
│  │  - Maintains rolling feature window (200 bars)           │   │
│  │  - Runs ensemble_strategy.py prediction                  │   │
│  │  - Returns trade signal with TP/SL levels                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Trained Models (loaded at startup)                       │   │
│  │  - vol_expansion_model.pkl                               │   │
│  │  - breakout_model.pkl                                    │   │
│  │  - atr_forecast_model.pkl                                │   │
│  │  - sentiment_vol_model.pkl                               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 0.3 Message Protocol

**Request (NinjaTrader → Python)**:
```json
{
  "type": "BAR_UPDATE",
  "timestamp": "2025-12-15T10:30:00",
  "symbol": "ES 03-25",
  "timeframe": "5min",
  "open": 6050.25,
  "high": 6052.50,
  "low": 6049.00,
  "close": 6051.75,
  "volume": 12500,
  "vix_close": 14.25,
  "position": 0
}
```

**Response (Python → NinjaTrader)**:
```json
{
  "type": "SIGNAL",
  "action": "LONG",
  "confidence": 0.72,
  "vol_expansion_prob": 0.65,
  "breakout_prob": 0.58,
  "tp_price": 6061.75,
  "sl_price": 6046.50,
  "contracts": 1,
  "reason": "Vol expansion + bullish breakout"
}
```

### 0.4 Exit Parameter Configuration

**IMPORTANT**: Exit parameters are FRAGILE - do not re-optimize.

| Parameter | Validated Value | Safe Range | DO NOT USE |
|-----------|-----------------|------------|------------|
| tp_atr_mult | 2.5 | 2.0 - 3.0 | < 2.0 (losses) |
| sl_atr_mult | 1.25 | 1.0 - 1.5 | > 1.5 (losses) |
| max_holding_bars | 20 | 15 - 25 | - |

**Rationale**: Validation report showed TP/SL sensitivity can swing P&L by $3M+. Current values are in stable zone. Further optimization risks overfitting.

### 0.5 File Locations

| File | Purpose |
|------|---------|
| `src/ninjatrader/SKIENinjaStrategy.cs` | NinjaScript client strategy |
| `src/python/deployment/ninja_signal_server.py` | Python signal server |
| `src/python/deployment/feature_calculator.py` | Real-time feature calculation |
| `models/production/*.pkl` | Trained model files |

---

## 1. Data Delivery Methods

### 1.1 VIX Data: CSV Upload vs Real-Time Feed

The ensemble strategy uses VIX-based sentiment features with T-1 (previous day) lag to prevent look-ahead bias. This creates flexibility in data delivery approach:

| Approach | Latency | Reliability | Cost | Suitability |
|----------|---------|-------------|------|-------------|
| **Daily CSV Upload** | ~24hr delay | Manual/error-prone | Free | Backtesting only |
| **Kinetick + CBOE** | Real-time | High | $7.50/mo + $69/mo base | Production |
| **Interactive Brokers** | Real-time | High | Included with IB feed | Production |

**Recommendation**: Use real-time VIX feed for production. Even though features use T-1 data, real-time feed enables:
- Automatic capture of prior day's close at market open
- Elimination of manual data management
- Reduced operational risk

**Implementation**: Configure NinjaTrader to subscribe to `^VIX` via Kinetick with CBOE Indexes add-on ($7.50/month).

### 1.2 PCR (Put/Call Ratio) Data

| Source | Update Frequency | API Access | Cost | Notes |
|--------|------------------|------------|------|-------|
| **CBOE Direct** | End-of-day | Enterprise only | $$$ | Contact for pricing |
| **Barchart** | 5-min delay | Yes | $50-200/mo | Good for real-time |
| **MacroMicro** | Daily | Web scraping | Free | Manual process |
| **YCharts** | Daily | API available | $400+/mo | Expensive |

**Current Status**: PCR historical data unavailable; VIX-based proxy implemented in `historical_sentiment_loader.py`:

```python
# VIX-to-PCR proxy formula (lines 239-240)
vix_normalized = np.clip((vix_close - 12) / 25, 0, 1)
pcr_total = 0.6 + (vix_normalized * 0.7)  # Range: 0.6 - 1.3
```

**Recommendation**: Continue using VIX proxy for initial deployment. VIX and PCR are highly correlated during fear/greed regimes. Add Barchart API later as enhancement.

### 1.3 AAII Sentiment Data

| Source | Frequency | Availability | Cost |
|--------|-----------|--------------|------|
| **AAII.com** | Weekly (Thursday) | Web scraping | Free |
| **Quandl** | Weekly | API | $50+/mo |

**Current Status**: VIX-based proxy implemented:

```python
# VIX-to-AAII proxy formula (lines 177-184)
vix_pct = vix_data['vix_percentile_20d']
aaii_bearish = 25 + (vix_pct * 30)  # Range: 25-55%
aaii_bullish = 50 - (vix_pct * 30)  # Range: 20-50%
```

**Recommendation**: VIX proxy is academically defensible. AAII is weekly data with limited intraday value.

### 1.4 Social/News Sentiment Data

| Provider | Latency | Coverage | API Cost | Notes |
|----------|---------|----------|----------|-------|
| **Twitter/X API v2** | Seconds | Twitter only | $100-5000/mo | Rate limits apply |
| **Polygon.io News** | Seconds | News aggregation | $29-199/mo | Good for NLP |
| **Alpha Vantage** | 15-min delay | News sentiment | Free-$50/mo | Sufficient for T-1 |
| **Custom LLM Pipeline** | Variable | Configurable | Compute cost | Most flexible |

**Academic Support**: Research from Alpaca Markets confirms "Twitter posts correspond with DJIA, NASDAQ, S&P 500 outcomes... emotional responses provide indication of next-day market behavior."

**Recommendation**: Alpha Vantage News API (free tier) is sufficient for T-1 sentiment features.

---

## 2. Data Feed Comparison

### 2.1 Latency Benchmarks (December 2024)

| Feed | Latency | Data Type | Tick Quality | Cost/Contract |
|------|---------|-----------|--------------|---------------|
| **Rithmic** | **~1ms** | MBO (Market By Order) | Unfiltered | $0.25/side |
| **CQG Continuum** | ~5ms | MBP (Market By Price) | Aggregated | $0.10/side |
| **Kinetick** | ~10-15ms | L1 | Delayed | Included |
| **Interactive Brokers** | ~5-10ms | L1/L2 | Standard | $0.25/side |

**Source**: Confluence Trading (December 2024) benchmark study:
> "Rithmic at 1ms latency vs CQG's 5ms—critical for scalpers. Rithmic's MBO granularity crushes CQG's MBP for order flow analysis."

### 2.2 Data Quality Comparison

| Feed | Order Book Depth | Cumulative Delta | Historical Data | Reliability |
|------|------------------|------------------|-----------------|-------------|
| **Rithmic** | Full MBO | Excellent | Limited | Very High |
| **CQG** | MBP only | Good | Extensive | Very High |
| **Kinetick** | L1 only | N/A | Good | High |

**Key Finding**: Rithmic provides unfiltered price data directly from exchange. CQG applies compression which may affect high-frequency calculations.

### 2.3 VIX Data Access by Feed

| Feed | VIX Index | VIX Futures (VX) | CBOE Add-on Required |
|------|-----------|------------------|----------------------|
| **Rithmic** | No | No | Yes (via Kinetick) |
| **CQG** | Yes | Yes | No |
| **Kinetick** | Yes | No | Yes ($7.50/mo) |
| **Interactive Brokers** | Yes | Yes | No |

**Note**: For VIX data on NinjaTrader 8, enable 'multi provider' option via Tools > Options > General.

---

## 3. Brokerage Analysis

### 3.1 Commission Comparison (ES Futures, 2025)

| Broker | Commission | Clearing | Exchange | All-In Cost | Notes |
|--------|------------|----------|----------|-------------|-------|
| **NinjaTrader** | $0.09/micro | Included | $0.47 | ~$0.56/micro | Lifetime license |
| **AMP Futures** | $0.25 | $0.50 | $0.47 | ~$1.22/ES | Volume discounts |
| **Interactive Brokers** | $0.25 | Included | $0.47 | ~$0.72/ES | Tiered pricing |
| **Optimus Futures** | $0.25 | $0.50 | $0.47 | ~$1.22/ES | Multiple platforms |

**Intraday Margins**:
- NinjaTrader: $500 (ES), $50 (MES)
- AMP Futures: $400 (ES), $40 (MES)
- Interactive Brokers: $500+ (ES)

### 3.2 Platform & Algo Trading Support

| Broker | NinjaTrader Support | Rithmic Available | API Access | Algo Trading |
|--------|---------------------|-------------------|------------|--------------|
| **NinjaTrader** | Native | Yes | NinjaScript | Excellent |
| **AMP Futures** | Full | Yes | Multiple | Excellent |
| **Interactive Brokers** | Via adapter | No | TWS API | Excellent |
| **Optimus Futures** | Full | Yes | Multiple | Good |

### 3.3 Industry Recognition

- **AMP Futures**: "Best Futures Broker" 2023 & 2024 (TradingView)
- **NinjaTrader**: "#1 Futures Broker" 2025 (BrokerChooser)
- **Interactive Brokers**: Best for institutional/multi-asset

---

## 4. Recommended Configuration

### 4.1 Primary Recommendation: NinjaTrader + Rithmic

| Component | Selection | Cost | Rationale |
|-----------|-----------|------|-----------|
| **Brokerage** | NinjaTrader Brokerage | $0 platform | Native integration |
| **Data Feed** | Rithmic | $20/mo min | 1ms latency, MBO data |
| **VIX Data** | Kinetick CBOE | $7.50/mo | Real-time index data |
| **Sentiment** | Alpha Vantage | $0-50/mo | Sufficient for T-1 |
| **VPS** | QuantVPS/AWS | $50-100/mo | CME co-location |

**Total Monthly**: ~$80-180 + commissions ($1.29/side ES)

### 4.2 Alternative: AMP Futures + Rithmic

| Component | Selection | Cost | Rationale |
|-----------|-----------|------|-----------|
| **Brokerage** | AMP Futures | $0 platform | Lowest commissions |
| **Data Feed** | Rithmic | $20/mo min | 1ms latency |
| **VIX Data** | Kinetick CBOE | $7.50/mo | Real-time index data |
| **Sentiment** | Alpha Vantage | $0-50/mo | Sufficient for T-1 |
| **VPS** | QuantVPS/AWS | $50-100/mo | CME co-location |

**Total Monthly**: ~$80-180 + commissions ($0.49/side ES all-in)

### 4.3 Budget Option: NinjaTrader + CQG

| Component | Selection | Cost | Rationale |
|-----------|-----------|------|-----------|
| **Brokerage** | NinjaTrader Brokerage | $0 platform | Native integration |
| **Data Feed** | CQG Continuum | $10/mo min | Lower cost, includes VIX |
| **Sentiment** | VIX proxy only | $0 | Use existing implementation |
| **VPS** | Local machine | $0 | Higher latency acceptable |

**Total Monthly**: ~$10 + commissions

---

## 5. Implementation Checklist

### Phase 15A: Infrastructure Setup (COMPLETE if NinjaTrader account exists)

- [x] Select brokerage (NinjaTrader account exists)
- [ ] Configure Rithmic data feed
- [ ] Add Kinetick CBOE subscription for VIX
- [ ] Enable multi-provider in NinjaTrader (Tools > Options > General)
- [ ] Set up VPS with CME proximity (Chicago) - *optional for paper trading*

### Phase 15B: Socket Bridge Implementation (1-2 days)

- [ ] Create Python signal server (`ninja_signal_server.py`)
- [ ] Export trained models to `models/production/` directory
- [ ] Create NinjaScript client strategy (`SKIENinjaStrategy.cs`)
- [ ] Test socket communication locally
- [ ] Implement heartbeat/reconnection logic

### Phase 15C: Platform Walk-Forward Validation (Critical)

- [ ] Run strategy through NinjaTrader Market Replay (2020-2022 data)
- [ ] Compare trade-by-trade results vs Python backtest
- [ ] Document any discrepancies >5%
- [ ] Validate feature calculations match Python output
- [ ] Measure simulated slippage vs 0.5 tick assumption

### Phase 15D: Paper Trading (30-60 days)

- [ ] Deploy to NinjaTrader simulation account
- [ ] Monitor daily P&L vs backtest benchmarks
- [ ] Implement kill switch (halt if daily loss >$5K)
- [ ] Track metrics: actual slippage, fill rate, latency
- [ ] Weekly performance review

### Phase 15E: Controlled Live Trading

- [ ] Start with 1 MES contract (1/10th ES exposure)
- [ ] Scale to 1 ES contract after 30 profitable days
- [ ] Quarterly full validation re-run
- [ ] Monthly parameter stability review

---

## 6. Latency Optimization

### 6.1 Network Architecture

```
[Exchange (CME Globex)]
        |
        | <1ms (co-located)
        v
[Rithmic Gateway - Chicago]
        |
        | 1-5ms (fiber)
        v
[VPS - Chicago Data Center]
        |
        | Local
        v
[NinjaTrader 8 + Strategy]
```

### 6.2 VPS Recommendations

| Provider | Location | Latency to CME | Cost |
|----------|----------|----------------|------|
| **QuantVPS** | Chicago | <1ms | $50-100/mo |
| **AWS us-east-2** | Ohio | 5-10ms | $50-150/mo |
| **Rithmic Co-lo** | CME | <250μs | $500+/mo |

**Note**: For the SKIE Ninja strategy (5-min bars, ~20 bar hold time), sub-millisecond latency is not critical. Chicago-based VPS is sufficient.

### 6.3 Expected Execution Quality

Based on backtest assumptions vs realistic expectations:

| Metric | Backtest | Expected Live | Notes |
|--------|----------|---------------|-------|
| Slippage | 0.5 tick | 0.25-0.75 tick | RTH, ES liquid |
| Commission | $1.29/side | $0.49-1.29/side | Depends on broker |
| Fill Rate | 100% | 98-99% | Rare partial fills |
| Latency Impact | None | Minimal | 5-min bars |

---

## 7. Risk Considerations

### 7.1 Data Feed Risks

| Risk | Mitigation |
|------|------------|
| Feed disconnection | Configure automatic reconnect, alerts |
| Data gaps | Implement missing bar detection |
| VIX data delay | Use previous valid value with staleness flag |
| API rate limits | Cache sentiment data, respect limits |

### 7.2 Execution Risks

| Risk | Mitigation |
|------|------------|
| Slippage spike | Max slippage parameter in strategy |
| Partial fills | Handle in position management logic |
| Platform crash | VPS auto-restart, position reconciliation |
| Broker outage | Have backup broker account ready |

---

## 8. Cost Summary

### Monthly Operating Costs (Recommended Setup)

| Item | Low | High | Notes |
|------|-----|------|-------|
| Data Feed (Rithmic) | $20 | $20 | Minimum monthly |
| VIX Data (Kinetick CBOE) | $7.50 | $7.50 | Fixed |
| VPS | $50 | $100 | Chicago location |
| Sentiment API | $0 | $50 | Alpha Vantage |
| **Subtotal (Fixed)** | **$77.50** | **$177.50** | |

### Variable Costs (Per Trade)

| Broker | Commission | Clearing | Exchange | Total/Side |
|--------|------------|----------|----------|------------|
| NinjaTrader | $1.29 | Incl. | $0.47 | $1.76 |
| AMP Futures | $0.25 | $0.50 | $0.47 | $1.22 |

**Example**: 100 round-trip trades/month on ES
- NinjaTrader: 100 × 2 × $1.76 = $352
- AMP Futures: 100 × 2 × $1.22 = $244

---

## 9. References

### Data Feed & Latency
- [Confluence Trading - Rithmic vs CQG (Dec 2024)](https://confluence-trading.com/index.php/en/blog/185-2024-12-21-compare-data-feeds-rithmic-versus-cqg)
- [Optimus Futures - Data Feeds Comparison](https://optimusfutures.com/Datafeeds.php)
- [QuantVPS - Low Latency Futures Trading](https://www.quantvps.com/blog/low-latency-futures-trading)

### Brokerage
- [QuantVPS - Lowest Commission Brokers](https://www.quantvps.com/blog/lowest-commission-futures-trading-platforms-compared)
- [BrokerChooser - AMP vs NinjaTrader](https://brokerchooser.com/compare/amp-futures-vs-ninjatrader)
- [BrokerChooser - Best Futures Brokers 2025](https://brokerchooser.com/best-brokers/best-futures-brokers)

### VIX & Sentiment Data
- [NinjaTrader Forum - Real-Time VIX Data](https://forum.ninjatrader.com/forum/ninjatrader-8/platform-technical-support-aa/1335569-real-time-vix-data)
- [CBOE Market Statistics](https://www.cboe.com/us/options/market_statistics/daily/)
- [Alpaca - Twitter Sentiment Trading](https://alpaca.markets/learn/algorithmic-trading-with-twitter-sentiment-analysis)

### Academic Research
- Baker & Wurgler (2006) - Investor sentiment predicts returns
- Bollen, Mao, Zeng (2011) - Twitter mood predicts DJIA
- MacroMicro Research - Put/Call ratio thresholds

---

*Last Updated: 2025-12-15*
*Document Status: Phase 15 Planning*
*Maintained by: SKIE_Ninja Development Team*
