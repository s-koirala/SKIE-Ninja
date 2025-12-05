# Sentiment Strategy Development Plan

**Created**: 2025-12-04
**Author**: SKIE_Ninja Development Team
**Status**: PLANNING

---

## Executive Summary

This document outlines the development of an **independent sentiment-based trading strategy** for ES futures, following the project's established methodology of validating components independently before combining them. The strategy will be tested using walk-forward validation with strict data leakage prevention, then ensembled with the existing volatility breakout strategy if successful.

---

## 1. LITERATURE REVIEW & THEORETICAL FOUNDATION

### 1.1 Academic Support for Sentiment-Based Trading

| Source | Finding | Application |
|--------|---------|-------------|
| **Baker & Wurgler (2006)** | Investor sentiment predicts cross-section of stock returns | Sentiment as timing indicator |
| **Tetlock (2007)** | Media pessimism predicts downward pressure on market | News sentiment features |
| **Da, Engelberg, Gao (2015)** | FEARS index (search volume) predicts volatility | Alternative sentiment data |
| **Bollen, Mao, Zeng (2011)** | Twitter mood predicts DJIA movements | Social media sentiment |
| **AAII Research** | Contrarian indicator - extreme readings precede reversals | AAII survey as contrarian |
| **MacroMicro Research** | Put/Call ratio >1.1 signals market troughs | PCR thresholds |
| **Huang et al. (2015)** | Aligned sentiment across measures is more predictive | Composite sentiment signals |

### 1.2 Key Insight: Sentiment as CONTRARIAN Indicator

Research consistently shows sentiment works best as a **contrarian** indicator:
- **Extreme bullish sentiment** → Market often reverses DOWN
- **Extreme bearish sentiment** → Market often reverses UP
- **Neutral sentiment** → Lower predictive value

This aligns with our project philosophy: predict **WHEN** to trade (volatility), not **direction** directly.

### 1.3 Sentiment + Volatility Connection

Literature suggests sentiment extremes predict **volatility expansion**:
- High fear (VIX spike, bearish sentiment) → Volatility increases
- Low fear (complacency) → Precedes volatility spikes
- This directly connects to our existing vol_expansion prediction (AUC 0.84)

---

## 2. DATA SOURCES & AVAILABILITY

### 2.1 Established Sentiment Indices (Already Implemented)

| Source | Frequency | Type | File |
|--------|-----------|------|------|
| **AAII Survey** | Weekly (Thursday) | Contrarian | `established_sentiment_indices.py` |
| **Put/Call Ratio** | Daily | Contrarian | `established_sentiment_indices.py` |
| **VIX Level/Structure** | Intraday | Volatility proxy | `established_sentiment_indices.py` |
| **VIX Term Structure** | Daily | Fear gauge | `established_sentiment_indices.py` |

### 2.2 Social/News Sentiment (Already Implemented)

| Source | Frequency | Type | File |
|--------|-----------|------|------|
| **Twitter/X API** | Real-time | Social sentiment | `social_news_sentiment.py` |
| **Alpha Vantage News** | Near real-time | News sentiment | `social_news_sentiment.py` |
| **Reddit (WSB, stocks)** | Real-time | Retail contrarian | `social_news_sentiment.py` |

### 2.3 Data Collection Tasks

1. **Historical AAII Data**: Download weekly survey data (2020-2024)
   - Source: AAII.com (archived data)
   - Frequency: Weekly, released Thursday

2. **Historical Put/Call Ratio**: Download daily CBOE data
   - Source: CBOE website, Quandl
   - Frequency: Daily

3. **VIX Historical**: Already available in `data/raw/market/VIX_daily.csv`

4. **News Sentiment Backfill**: Optional - expensive API calls
   - For backtesting, use placeholder or proxy data

---

## 3. FEATURE ENGINEERING PLAN

### 3.1 Established Sentiment Features

From `established_sentiment_indices.py`:

```python
# AAII Features (Weekly, forward-filled to bars)
'aaii_bullish'                # % bullish investors
'aaii_bearish'                # % bearish investors
'aaii_spread'                 # Bull - Bear spread
'aaii_contrarian_signal'      # Normalized contrarian (-1 to +1)
'aaii_extreme_bullish'        # Binary: extreme bullish (contrarian bearish)
'aaii_extreme_bearish'        # Binary: extreme bearish (contrarian bullish)

# Put/Call Ratio Features (Daily)
'pcr_total'                   # Raw put/call ratio
'pcr_bullish_extreme'         # PCR > 1.1 (contrarian bullish)
'pcr_bearish_extreme'         # PCR < 0.8 (contrarian bearish)
'pcr_contrarian_signal'       # Normalized signal

# VIX Features (Daily/Intraday)
'vix_level'                   # Current VIX
'vix_vs_ma10'                 # VIX vs 10-day MA
'vix_vs_ma20'                 # VIX vs 20-day MA
'vix_percentile_20d'          # VIX percentile (20-day)
'vix_fear_regime'             # VIX > 25
'vix_extreme_fear'            # VIX > 30
'vix_complacency_regime'      # VIX < 15
'vix_spike'                   # 15% daily increase
'vix_sentiment'               # Normalized sentiment

# Composite
'composite_contrarian_signal' # Average of all contrarian signals
```

### 3.2 Social/News Sentiment Features

From `social_news_sentiment.py`:

```python
# Aggregated sentiment (multiple windows)
'social_sentiment_5min'       # 5-minute window sentiment
'social_sentiment_15min'      # 15-minute window
'social_sentiment_30min'      # 30-minute window
'social_sentiment_60min'      # 1-hour window
'social_sentiment_240min'     # 4-hour window

# Activity metrics
'social_count_Nmin'           # Number of posts/articles
'social_engagement_Nmin'      # Total engagement (likes, shares)

# Derived features
'social_sentiment_momentum'   # Short vs long-term sentiment
'social_sentiment_trend'      # 5-bar sentiment change
'social_extreme_bullish'      # Sentiment > 0.5
'social_extreme_bearish'      # Sentiment < -0.5
'social_high_activity'        # Activity > 2x moving average
```

### 3.3 NEW Features to Engineer

```python
# Cross-source agreement (stronger signals)
'sentiment_agreement_score'   # % of sources agreeing on direction
'all_sources_bullish'         # Binary: all sources bullish
'all_sources_bearish'         # Binary: all sources bearish

# Sentiment divergence (potential reversal)
'sentiment_price_divergence'  # Sentiment vs price direction mismatch

# Regime-based features
'sentiment_regime'            # 0=neutral, 1=bullish, -1=bearish
'regime_duration'             # Bars since regime change

# Sentiment velocity
'aaii_spread_change'          # Week-over-week AAII change
'pcr_ma5_slope'               # 5-day PCR moving average slope
'vix_velocity'                # Rate of VIX change
```

### 3.4 DATA LEAKAGE PREVENTION CHECKLIST

**CRITICAL**: All features must follow these rules:

1. **No negative shifts**: Never use `shift(-N)` - this accesses future data
2. **No center=True**: Rolling windows must use `center=False` (default)
3. **Proper lag**: Sentiment data lagged by at least 5 minutes
4. **Point-in-time**: Only use data available at prediction time
5. **Forward-fill carefully**: Weekly data forward-filled to bars must use PREVIOUS week's value

```python
# CORRECT: Use previous week's AAII for current bar
# Bar on Monday uses LAST Thursday's survey
df['aaii_bullish'] = aaii_data.reindex(df.index, method='ffill')

# WRONG: Using current week's survey for bars before Thursday
# This is look-ahead bias!
```

---

## 4. STRATEGY DESIGN

### 4.1 Primary Hypothesis

**Sentiment extremes predict volatility expansion**, which aligns with our existing vol_expansion model.

Strategy Approach:
1. Use sentiment as a **filter** for vol_expansion trades
2. Use sentiment as a **confirmation** for breakout direction
3. Test sentiment as **standalone** predictor first

### 4.2 Independent Sentiment Strategy (Phase 1)

```python
class SentimentOnlyStrategy:
    """
    Pure sentiment-based strategy for validation.
    Tests if sentiment alone has predictive power.
    """

    def generate_signal(self, row):
        # Composite contrarian signal
        contrarian = row['composite_contrarian_signal']

        # VIX regime
        fear_regime = row['vix_fear_regime']
        complacency = row['vix_complacency_regime']

        # Entry conditions (contrarian logic)
        if contrarian > 0.3 and fear_regime:  # High fear = contrarian bullish
            return 1  # LONG
        elif contrarian < -0.3 and complacency:  # Complacency = contrarian bearish
            return -1  # SHORT
        else:
            return 0  # No trade
```

### 4.3 Targets for Sentiment Strategy

Following project methodology, predict multiple targets:

| Target | Description | Expected AUC |
|--------|-------------|--------------|
| `sentiment_vol_expansion` | Does vol expand after sentiment extreme? | 0.60-0.70 |
| `sentiment_reversal` | Does price reverse after extreme? | 0.55-0.65 |
| `sentiment_regime_change` | Regime shift imminent? | 0.55-0.60 |

### 4.4 Model Selection

Based on project experience:
- **LightGBM Classifier** for binary targets
- Simple hyperparameters (avoid overfitting)
- Walk-forward training

---

## 5. VALIDATION METHODOLOGY

### 5.1 Walk-Forward Configuration

Following `docs/BEST_PRACTICES.md`:

```python
@dataclass
class SentimentWalkForwardConfig:
    train_days: int = 60        # 60-day training window
    test_days: int = 5          # 5-day test window
    embargo_bars: int = 20      # 20-bar gap (~100 min at 5-min bars)
    min_train_samples: int = 500
```

### 5.2 Out-of-Sample Periods

| Period | Bars | Use |
|--------|------|-----|
| 2023-2024 | 684K | In-Sample Development |
| 2020-2022 | 1.02M | Out-of-Sample Validation |
| 2025 YTD | 326K | Forward Test |

### 5.3 Success Criteria

Strategy passes if:
- [ ] OOS AUC > 0.55 (above random chance)
- [ ] OOS Sharpe > 0.5 (positive risk-adjusted return)
- [ ] Consistent across all OOS years (2020, 2021, 2022)
- [ ] Forward test (2025) confirms edge
- [ ] No suspicious metrics (see Best Practices)

### 5.4 QC Checks

```python
# Run before trusting results
python src/python/run_qc_check.py

# Specific checks:
# 1. Feature look-ahead: grep -r "shift(-" in sentiment code
# 2. Correlation check: max feature-target corr < 0.30
# 3. Temporal leakage test: train future, predict past
# 4. Result validation: Win rate < 65%, Sharpe < 4.0
```

---

## 6. ENSEMBLE DESIGN

### 6.1 Ensemble Architecture (If Sentiment Strategy Passes)

```
                    ┌──────────────────────┐
                    │   Vol Breakout       │
                    │   Strategy           │
                    │   (Validated)        │
                    │   AUC: 0.84 vol      │
                    │   AUC: 0.72 breakout │
                    └─────────┬────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────┐
│              ENSEMBLE DECISION LAYER               │
│                                                    │
│  Signal = f(vol_signal, sentiment_signal, weight) │
│                                                    │
│  Options:                                          │
│  1. Stacking (meta-model)                         │
│  2. Weighted average                              │
│  3. Voting (both agree)                           │
│  4. Hierarchical (sentiment filters vol)          │
└────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │   Sentiment          │
                    │   Strategy           │
                    │   (To Validate)      │
                    │   Target AUC: 0.55+  │
                    └──────────────────────┘
```

### 6.2 Ensemble Methods to Test

#### Method 1: Hierarchical Filter
```python
# Sentiment as additional filter for vol breakout
if vol_expansion_prob > threshold:
    if sentiment_confirmation:  # Sentiment agrees with direction
        execute_trade()
```

#### Method 2: Weighted Combination
```python
# Combine probabilities with learned weights
combined_prob = (w1 * vol_prob) + (w2 * sentiment_prob)
# Weights learned via validation
```

#### Method 3: Stacking Meta-Model
```python
# Train meta-model on strategy outputs
meta_features = [vol_expansion_prob, breakout_prob, sentiment_prob]
meta_model = LightGBM(meta_features)
final_signal = meta_model.predict()
```

### 6.3 Ensemble Success Criteria

Ensemble passes if:
- [ ] Net P&L > Vol Breakout alone
- [ ] Sharpe Ratio improved
- [ ] Max Drawdown reduced OR unchanged
- [ ] OOS performance consistent with in-sample

---

## 7. IMPLEMENTATION PHASES

### Phase 1: Data Collection & Validation (Day 1)

**Tasks:**
1. [ ] Verify `established_sentiment_indices.py` works
2. [ ] Download historical AAII data (2020-2024)
3. [ ] Download historical Put/Call ratio data
4. [ ] Verify VIX data alignment
5. [ ] Create unified sentiment data loader

**Deliverables:**
- `data/raw/sentiment/aaii_historical.csv`
- `data/raw/sentiment/pcr_historical.csv`
- `src/python/data_collection/sentiment_data_loader.py`

### Phase 2: Feature Engineering (Day 1-2)

**Tasks:**
1. [ ] Generate all sentiment features
2. [ ] Run leakage prevention checks
3. [ ] Align sentiment to 5-minute bars
4. [ ] Calculate feature statistics

**Deliverables:**
- `src/python/feature_engineering/unified_sentiment_features.py`
- `data/processed/sentiment_features_2023_2024.csv`
- Feature statistics report

### Phase 3: Independent Sentiment Strategy (Day 2-3)

**Tasks:**
1. [ ] Implement `SentimentStrategy` class
2. [ ] Define sentiment-based targets
3. [ ] Train LightGBM models
4. [ ] Run in-sample backtest

**Deliverables:**
- `src/python/strategy/sentiment_strategy.py`
- `data/backtest_results/sentiment_strategy_insample.csv`

### Phase 4: Walk-Forward Validation (Day 3-4)

**Tasks:**
1. [ ] Run walk-forward backtest
2. [ ] Calculate per-fold metrics
3. [ ] Run OOS validation (2020-2022)
4. [ ] Run forward test (2025)

**Deliverables:**
- `src/python/run_sentiment_backtest.py`
- `data/backtest_results/sentiment_oos_*.csv`
- `data/backtest_results/sentiment_forward_2025.csv`

### Phase 5: Statistical Validation (Day 4)

**Tasks:**
1. [ ] Run Monte Carlo simulation (1000 iterations)
2. [ ] Calculate confidence intervals
3. [ ] Compare to random baseline
4. [ ] Run QC checks

**Deliverables:**
- `data/validation_results/sentiment_monte_carlo.csv`
- Validation report

### Phase 6: Ensemble Development (Day 5)

**Tasks:**
1. [ ] Implement ensemble methods
2. [ ] Test hierarchical filter
3. [ ] Test weighted combination
4. [ ] Test stacking meta-model
5. [ ] Select best ensemble

**Deliverables:**
- `src/python/strategy/ensemble_strategy.py`
- Ensemble comparison report

### Phase 7: Documentation & Deployment (Day 6)

**Tasks:**
1. [ ] Update CANONICAL_REFERENCE.md
2. [ ] Update project_memory.md
3. [ ] Create ensemble documentation
4. [ ] Commit all changes to GitHub
5. [ ] Update HANDOFF.md

**Deliverables:**
- Updated documentation
- GitHub commit with all changes
- Production-ready ensemble strategy

---

## 8. RISK ASSESSMENT

### 8.1 Potential Issues

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Sentiment has no edge | Medium | High | Test independently first |
| Data quality issues | Medium | Medium | Validate data sources |
| Overfitting to AAII timing | Medium | High | Use multiple sentiment sources |
| API rate limits | Low | Low | Cache data, use historical |
| Ensemble degrades performance | Low | Medium | Keep vol breakout as fallback |

### 8.2 Fallback Plan

If sentiment strategy fails validation:
1. Document negative results (valuable information)
2. Continue with vol breakout strategy (already validated)
3. Consider sentiment as auxiliary filter only
4. Explore alternative data sources

---

## 9. SUCCESS METRICS SUMMARY

### Independent Sentiment Strategy

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| OOS AUC | 0.55 | 0.60 | 0.65 |
| OOS Sharpe | 0.5 | 1.0 | 1.5 |
| Win Rate | 35% | 40% | 45% |
| Profit Factor | 1.05 | 1.15 | 1.25 |

### Ensemble Strategy

| Metric | Vol Breakout Baseline | Ensemble Target |
|--------|----------------------|-----------------|
| Net P&L | $763,125 | > $850,000 |
| Sharpe | 3.09 | > 3.20 |
| Max DD | $33,596 | < $35,000 |
| Win Rate | 40% | 40-42% |

---

## 10. REFERENCES

### Academic Papers
1. Baker, M., & Wurgler, J. (2006). "Investor sentiment and the cross-section of stock returns." *Journal of Finance*, 61(4), 1645-1680.
2. Tetlock, P. C. (2007). "Giving content to investor sentiment." *Journal of Finance*, 62(3), 1139-1168.
3. Da, Z., Engelberg, J., & Gao, P. (2015). "The sum of all FEARS investor sentiment and asset prices." *Review of Financial Studies*, 28(1), 1-32.
4. Bollen, J., Mao, H., & Zeng, X. (2011). "Twitter mood predicts the stock market." *Journal of Computational Science*, 2(1), 1-8.
5. Huang, D., Jiang, F., Tu, J., & Zhou, G. (2015). "Investor sentiment aligned: A powerful predictor of stock returns." *Review of Financial Studies*, 28(3), 791-837.

### Industry Sources
- AAII Sentiment Survey: https://www.aaii.com/sentimentsurvey
- CBOE Put/Call Ratio: https://www.cboe.com/
- MacroMicro PCR Analysis: https://en.macromicro.me/

### Project Documentation
- [HANDOFF.md](../HANDOFF.md) - Current session context
- [CANONICAL_REFERENCE.md](../config/CANONICAL_REFERENCE.md) - Active files
- [BEST_PRACTICES.md](../docs/BEST_PRACTICES.md) - Lessons learned
- [04_multi_target_prediction_strategy.md](./04_multi_target_prediction_strategy.md) - Multi-target approach

---

## 11. PHASE 10 INTEGRATION

This sentiment strategy work integrates with existing Phase 10 TODOs:

### Existing TODOs (from HANDOFF.md)
- [ ] **Threshold optimization** - Still HIGH PRIORITY (can run in parallel)
- [ ] Monte Carlo simulation - Will run for sentiment strategy too
- [ ] NinjaTrader ONNX export - Ensemble will need ONNX export
- [ ] Paper trading - Test ensemble in paper trading

### New TODOs (from this plan)
- [ ] Phase 1: Data collection & validation
- [ ] Phase 2: Feature engineering
- [ ] Phase 3: Independent sentiment strategy
- [ ] Phase 4: Walk-forward validation
- [ ] Phase 5: Statistical validation
- [ ] Phase 6: Ensemble development
- [ ] Phase 7: Documentation & deployment

---

*Document created: 2025-12-04*
*Next update: After Phase 1 completion*
*Maintained by: SKIE_Ninja Development Team*
