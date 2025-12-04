# Advanced Strategy Research: Feature Engineering, Sentiment, and Novel ML Approaches

**Date**: 2025-12-04
**Purpose**: Comprehensive literature review for SKIE-Ninja ES futures ML trading system redesign
**Context**: Current feature set shows zero predictive power after eliminating look-ahead bias. This research explores new feature engineering approaches, alternative data sources, and innovative ML strategies.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [ML Features for Futures Prediction](#2-ml-features-for-futures-prediction)
3. [Sentiment Analysis](#3-sentiment-analysis)
4. [Target Engineering](#4-target-engineering)
5. [Arbitrage and Spread Strategies](#5-arbitrage-and-spread-strategies)
6. [Volatility Regime Detection](#6-volatility-regime-detection)
7. [Novel ML Architectures](#7-novel-ml-architectures)
8. [Reinforcement Learning](#8-reinforcement-learning)
9. [Implementation Recommendations](#9-implementation-recommendations)
10. [References](#10-references)

---

## 1. Executive Summary

### Current State
- **Problem**: All predictive power in current models came from look-ahead bias
- **Result**: LightGBM with corrected features: 45.1% win rate, -0.34 Sharpe, -$4,953 P&L
- **Conclusion**: Technical indicators and basic microstructure features provide NO edge

### Research Findings

| Category | Promising Approaches | Expected Alpha | Data Availability |
|----------|---------------------|----------------|-------------------|
| **Order Flow** | Multi-level OFI, Hawkes processes | Medium-High | Requires L2/L3 data |
| **Sentiment** | FinBERT + LSTM hybrid | Medium | Available (APIs) |
| **Target Engineering** | Triple Barrier + Meta-labeling | Improves sizing | Self-generated |
| **Volatility Regime** | Hidden Markov Models, VIX features | High | Available |
| **Arbitrage** | ES-SPY basis, cross-market | Low (HFT dominated) | Available |
| **Novel ML** | Temporal Fusion Transformer | Medium | Standard data |
| **Reinforcement Learning** | PPO, Ensemble DRL | Experimental | Requires simulator |

### Key Recommendations

1. **Immediate Priority**: Implement Triple Barrier labeling + Meta-labeling
2. **Short-term**: Add volatility regime features (VIX, realized vol regimes)
3. **Medium-term**: Integrate sentiment analysis (FinBERT on financial news)
4. **Long-term**: Explore Temporal Fusion Transformers and reinforcement learning

---

## 2. ML Features for Futures Prediction

### 2.1 Academic Research Summary

Recent studies (2022-2024) demonstrate ML effectiveness for futures prediction:

- **XGBoost + SMOTE + NSGA-II** combination for futures direction prediction shows promise
- **SVR and ANN** remain popular for price forecasting
- **Random Forest** effective for assessing microstructure features across 87 futures contracts
- Walk-forward testing with 6.3 basis point break-even transaction costs documented

**Key Reference**: [Machine Learning in Futures Markets (MDPI 2021)](https://www.mdpi.com/1911-8074/14/3/119)

### 2.2 Order Flow Imbalance (OFI) Features

Order Flow Imbalance models quantify buying/selling pressure and predict short-term price movements.

#### Multi-Level Order Flow Imbalance (MLOFI)

| Feature Type | Description | Improvement Over OFI |
|-------------|-------------|---------------------|
| Level 1 OFI | Best bid/ask imbalance | Baseline |
| Level 2-10 OFI | Deeper book imbalance | 60-70% RMSE improvement (large-tick) |
| Aggregated OFI | Weighted sum across levels | 15-35% improvement (small-tick) |

**Implementation**: Use Ridge regression on MLOFI for short-term price prediction.

**Key Reference**: [Deep Order Flow Imbalance (Kolm & Turiel)](https://www.semanticscholar.org/paper/Deep-order-flow-imbalance:-Extracting-alpha-at-from-Kolm-Turiel/977e72a246b1a2b374288e2409694eb67d5dfbca)

#### Hawkes Process for OFI Forecasting

Hawkes processes model self-exciting order flow with lagged dependencies:
- Account for bid-ask order flow correlation
- Sum of Exponentials kernel gives best forecasts
- Useful for near-term distribution forecasting

**Key Reference**: [Forecasting High Frequency Order Flow Imbalance (arXiv 2024)](https://arxiv.org/html/2408.03594v1)

#### Derivative Indicators

| Indicator | Formula | Use Case |
|-----------|---------|----------|
| **Micro-Price** | Weighted mid by volume at best | Fair value estimation |
| **VAMP** | Volume-adjusted mid price | Price impact estimation |
| **Static OBI** | (Bid Vol - Ask Vol) / Total Vol | Direction prediction |
| **Trade Flow Imbalance** | Buy trades - Sell trades | Momentum signal |

### 2.3 Cross-Asset Features

| Feature | Description | ES Relevance |
|---------|-------------|--------------|
| VIX Level | CBOE Volatility Index | Risk-off indicator |
| VIX Term Structure | Contango/backwardation | Regime indicator |
| Bond Yields (ZN, ZB) | Treasury futures | Flight to safety |
| NQ/ES Ratio | Nasdaq vs S&P | Tech sentiment |
| Currency (DX) | Dollar index | Risk appetite |
| Gold (GC) | Safe haven | Risk-off flows |

### 2.4 What Doesn't Work

Based on literature and our own testing:

1. **Simple technical indicators** - Provide minimal edge in isolation
2. **Lagging indicators** - By definition, too slow for intraday
3. **Single-timeframe features** - Need multi-timeframe confluence
4. **Overfit pattern matching** - Fails out-of-sample
5. **Static thresholds** - Need volatility-adjusted levels

---

## 3. Sentiment Analysis

### 3.1 FinBERT for Financial Text

FinBERT is a pre-trained NLP model specifically designed for financial sentiment analysis.

#### Performance Comparison

| Model | F1-Score Improvement | Best Use Case |
|-------|---------------------|---------------|
| FinBERT | Baseline | Financial news, earnings calls |
| FinBERT-LSTM | +4-5% vs alternatives | Time-series sentiment integration |
| GPT-4 sentiment | Comparable | Ad-hoc analysis, not real-time |
| BERT (general) | Lower | Non-financial text |

#### FinBERT-LSTM Pipeline

```
1. News/Tweet → FinBERT → Sentiment Score [-1, 1]
2. Sentiment Score → Rolling Average → Sentiment Feature
3. Price Data + Sentiment Feature → LSTM → Prediction
```

**Key Finding**: FinBERT + LSTM outperforms standalone LSTM and DNN models.

**Key Reference**: [FinBERT-LSTM Stock Prediction (ACM 2024)](https://dl.acm.org/doi/10.1145/3694860.3694870)

### 3.2 Twitter/X Sentiment

| Data Source | Processing | Alpha Signal |
|-------------|-----------|--------------|
| $SPY cashtag | FinBERT sentiment | Weak correlation with next-day returns |
| $ES_F cashtag | Volume + sentiment | Volume spikes more predictive |
| Financial influencers | Weighted by followers | Mixed results |

**Latency Considerations**:
- Real-time processing required for intraday
- Minimum 1-5 minute sentiment aggregation
- Alpha decays rapidly (minutes to hours)

### 3.3 Reddit WallStreetBets Sentiment

**Critical Finding**: WSB sentiment is a CONTRARIAN indicator.

| WSB Sentiment | Market Response | Trading Signal |
|--------------|-----------------|----------------|
| Extreme bullish | Stocks often decline | Fade the crowd |
| Extreme bearish | Stocks often recover | Contrarian long |
| Neutral | No signal | No trade |

**Academic Evidence**:
- WSB attention increases risk levels but REDUCES returns
- Long/short portfolio following WSB recommendations produces NO alpha
- WSB is inversely correlated with VIX

**Key Reference**: [Reddit WSB Sentiment Research (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S1057521924006537)

### 3.4 News Sentiment Sources

| Provider | Coverage | Latency | Cost |
|----------|----------|---------|------|
| **RavenPack** | Comprehensive | Real-time | $$$$ |
| **Bloomberg Terminal** | Premium | Real-time | $$$ |
| **GDELT** | Global events | Minutes | Free |
| **Alpha Vantage News** | Basic | Minutes | Free tier |
| **Polygon.io** | Market news | Real-time | $$ |

### 3.5 Implementation Recommendation

```python
# Proposed Sentiment Feature Pipeline
class SentimentFeatureGenerator:
    def __init__(self):
        self.finbert = AutoModel.from_pretrained("ProsusAI/finbert")

    def generate_features(self, news_df, price_df):
        # 1. Score each news item
        news_df['sentiment'] = self.finbert.predict(news_df['text'])

        # 2. Aggregate by time window (5min, 15min, 1hr)
        for window in ['5T', '15T', '1H']:
            price_df[f'news_sentiment_{window}'] = (
                news_df.resample(window)['sentiment'].mean()
            )
            price_df[f'news_volume_{window}'] = (
                news_df.resample(window).size()
            )

        # 3. Sentiment momentum
        price_df['sentiment_momentum'] = (
            price_df['news_sentiment_5T'] -
            price_df['news_sentiment_1H']
        )

        return price_df
```

---

## 4. Target Engineering

### 4.1 Triple Barrier Method (Lopez de Prado)

The Triple Barrier Method creates labels based on price action rather than fixed time horizons.

#### The Three Barriers

| Barrier | Trigger Condition | Label |
|---------|------------------|-------|
| **Upper** (Take Profit) | Price rises to target | +1 (Win) |
| **Lower** (Stop Loss) | Price falls to stop | -1 (Loss) |
| **Vertical** (Time Limit) | Time expires | Sign of return |

#### Advantages Over Binary Labels

1. **Volatility-adjusted**: Barriers scale with ATR/volatility
2. **Realistic**: Matches actual trading with stops/targets
3. **Provides odds**: Barrier distances define risk/reward
4. **Works with Kelly**: Enables optimal bet sizing

#### Implementation

```python
def triple_barrier_label(price_series, upper_barrier, lower_barrier, max_holding):
    """
    Apply triple barrier labeling.

    Args:
        price_series: Close prices
        upper_barrier: Take profit level (e.g., 2 * ATR)
        lower_barrier: Stop loss level (e.g., -1 * ATR)
        max_holding: Maximum bars to hold

    Returns:
        labels: 1 (profit), -1 (loss), 0 (time exit with direction)
    """
    labels = []
    for i in range(len(price_series) - max_holding):
        entry_price = price_series.iloc[i]

        for j in range(1, max_holding + 1):
            future_price = price_series.iloc[i + j]
            ret = (future_price - entry_price) / entry_price

            if ret >= upper_barrier:
                labels.append(1)
                break
            elif ret <= lower_barrier:
                labels.append(-1)
                break
        else:
            # Time barrier hit - label by final direction
            final_ret = (price_series.iloc[i + max_holding] - entry_price) / entry_price
            labels.append(1 if final_ret > 0 else -1)

    return labels
```

**Key Reference**: [Triple Barrier Method (mlfinlab docs)](https://www.mlfinlab.com/en/latest/labeling/tb_meta_labeling.html)

### 4.2 Meta-Labeling

Meta-labeling uses a secondary model to decide bet SIZE, not direction.

#### Two-Stage Architecture

```
Stage 1: Primary Model (Side)
    Input: Features → Output: Long (+1) or Short (-1)
    Goal: High RECALL (catch all opportunities)

Stage 2: Meta Model (Size)
    Input: Features + Primary Prediction → Output: Trade (1) or Pass (0)
    Goal: High PRECISION (filter false positives)
```

#### Benefits

| Aspect | Without Meta-Labeling | With Meta-Labeling |
|--------|----------------------|-------------------|
| F1 Score | Lower | Higher |
| False Positives | Many | Filtered |
| Bet Sizing | Binary | Probability-based |
| Overfitting Risk | Higher | Lower (decoupled) |

#### Position Sizing from Meta-Label Probability

```python
def position_size_from_meta(probability, max_size=1.0):
    """
    Convert meta-label probability to position size.

    Uses modified Kelly criterion:
    size = (p * b - q) / b
    where p = probability, q = 1-p, b = win/loss ratio
    """
    if probability < 0.5:
        return 0  # Don't trade

    # Scale linearly from 0.5 to 1.0 probability
    size = (probability - 0.5) * 2 * max_size
    return min(size, max_size)
```

**Key Reference**: [Meta-Labeling (Hudson & Thames)](https://hudsonthames.org/meta-labeling-a-toy-example/)

### 4.3 Alternative Target Definitions

| Target Type | Description | Pros | Cons |
|-------------|-------------|------|------|
| Binary Direction | Next bar up/down | Simple | Noisy, ignores magnitude |
| Multi-class | Strong/weak up/down/neutral | More nuanced | Class imbalance |
| Regression | Predict exact return | Full information | Harder to optimize |
| MFE/MAE Target | Predict max favorable/adverse | Trade-relevant | Complex labels |
| Volatility-normalized | Return / ATR | Comparable across regimes | Lagging |
| Multi-horizon | Ensemble 5/15/30 bar targets | Robust | Complexity |

---

## 5. Arbitrage and Spread Strategies

### 5.1 ES-SPY Basis Trading

The E-mini S&P 500 (ES) and SPDR S&P 500 ETF (SPY) track the same index but have pricing differences.

#### Fair Value Calculation

```
ES Fair Value = SPY * Multiplier * e^((r - d) * T)

Where:
- r = risk-free rate
- d = dividend yield (~1.5%)
- T = time to expiration
```

#### Arbitrage Opportunity Detection

| Condition | Action | Expected |
|-----------|--------|----------|
| ES >> Fair Value | Short ES, Long SPY | Convergence profit |
| ES << Fair Value | Long ES, Short SPY | Convergence profit |

**Reality Check**:
- HFT firms dominate this space
- Opportunities last milliseconds, not minutes
- Retail traders cannot compete on latency

**Key Reference**: [ES-SPY Arbitrage Research (ResearchGate)](https://www.researchgate.net/publication/228880120_Index_Arbitrage_between_Futures_and_ETFs_Evidence_on_the_limits_to_arbitrage_from_SP_500_Futures_and_SPDRs)

### 5.2 Statistical Arbitrage / Pairs Trading

More feasible for non-HFT traders:

| Strategy | Instruments | Signal |
|----------|-------------|--------|
| Sector Rotation | ES vs XLF, XLK, XLE | Relative strength |
| Index Spread | ES vs NQ | Mean reversion |
| Cross-market | ES vs DAX, FTSE | Correlation breakdown |
| Calendar Spread | ES front vs back month | Roll dynamics |

#### Implementation Approach

```python
def pairs_signal(es_price, nq_price, lookback=20, zscore_threshold=2):
    """
    Generate pairs trading signal for ES vs NQ.
    """
    # Calculate spread ratio
    ratio = es_price / nq_price

    # Z-score of ratio
    ratio_mean = ratio.rolling(lookback).mean()
    ratio_std = ratio.rolling(lookback).std()
    zscore = (ratio - ratio_mean) / ratio_std

    # Generate signals
    if zscore > zscore_threshold:
        return "SHORT_ES_LONG_NQ"  # Spread will contract
    elif zscore < -zscore_threshold:
        return "LONG_ES_SHORT_NQ"  # Spread will expand
    else:
        return "NO_TRADE"
```

### 5.3 Calendar Spread Dynamics

ES futures exhibit predictable patterns around roll dates:

| Phase | Timing | Pattern |
|-------|--------|---------|
| Pre-roll | 2 weeks before expiry | Spread widening |
| Roll week | Expiration week | High volume, volatility |
| Post-roll | After expiry | Spread normalization |

---

## 6. Volatility Regime Detection

### 6.1 Why Regimes Matter

Different market conditions require different strategies:

| Regime | Characteristics | Optimal Strategy |
|--------|----------------|------------------|
| Low Vol Trending | VIX < 15, sustained direction | Trend following |
| Low Vol Range | VIX < 15, choppy | Mean reversion |
| High Vol Trending | VIX > 25, strong direction | Momentum |
| High Vol Chaos | VIX > 30, erratic | Reduce size / wait |

### 6.2 Regime Detection Methods

| Method | Description | Speed |
|--------|-------------|-------|
| **Hidden Markov Model** | Latent state estimation | Moderate |
| **Gaussian Mixture Model** | Cluster current conditions | Fast |
| **Random Forest Classifier** | Supervised regime labels | Fast |
| **Wasserstein k-means** | Distribution-aware clustering | Moderate |

**Key Reference**: [Classifying Market Regimes (Macrosynergy)](https://macrosynergy.com/research/classifying-market-regimes/)

### 6.3 VIX-Based Features

| Feature | Calculation | Signal |
|---------|-------------|--------|
| VIX Level | Raw VIX | Risk-off above 25 |
| VIX Percentile | 252-day rolling percentile | Regime indicator |
| VIX Term Structure | VX1 - VX2 | Contango = normal, Backwardation = fear |
| VIX Rate of Change | VIX pct change | Panic spikes |
| VIX/VIX3M Ratio | Short vs medium term vol | Short-term fear |

### 6.4 ML for VIX Prediction

Recent research (Quantitative Finance 2024) shows:

- ML can predict VIX with higher accuracy than previously documented
- Dynamic training and nonlinear methods outperform static models
- Weekly jobless claims are a pivotal predictor
- ML methods outperform HAR model at 1-month horizon

**Key Reference**: [Predicting VIX with Adaptive ML (Taylor & Francis 2024)](https://www.tandfonline.com/doi/full/10.1080/14697688.2024.2439458)

---

## 7. Novel ML Architectures

### 7.1 Temporal Fusion Transformer (TFT)

TFT is a state-of-the-art architecture for multi-horizon time series forecasting.

#### Key Advantages for Trading

| Feature | Benefit |
|---------|---------|
| Multi-horizon output | Predict 5, 15, 30 bars simultaneously |
| Interpretable attention | Understand what drives predictions |
| Static + temporal inputs | Combine regime info with price data |
| Uncertainty quantification | Confidence intervals for sizing |

#### TFT Architecture

```
Inputs:
├── Static Covariates (regime, day of week, contract)
├── Known Future Inputs (time features)
└── Observed Inputs (price, volume, indicators)
     ↓
Variable Selection Network (learn feature importance)
     ↓
LSTM Encoder (capture temporal patterns)
     ↓
Multi-Head Attention (long-range dependencies)
     ↓
Gated Residual Networks (skip connections)
     ↓
Quantile Outputs (10th, 50th, 90th percentile)
```

#### Performance

- Outperforms LSTM, ARIMA, DeepAR on financial data
- Provides interpretable attention weights
- Multi-horizon capability reduces model count

**Key Reference**: [TFT for Stock Prediction (IEEE 2022)](https://ieeexplore.ieee.org/document/9731073/)

### 7.2 TFT-GNN Hybrid

Combines TFT with Graph Neural Networks to model cross-asset relationships:

```
Asset Graph:
ES ←→ NQ (high correlation)
ES ←→ ZN (moderate negative)
ES ←→ GC (low correlation)
     ↓
GNN Layer (learns relationship dynamics)
     ↓
TFT (temporal forecasting)
```

### 7.3 Transformer vs Gradient Boosting

| Aspect | Transformer | LightGBM/XGBoost |
|--------|-------------|------------------|
| Sequential patterns | Excellent | Limited |
| Tabular features | Good | Excellent |
| Training data needed | More | Less |
| Interpretability | Attention weights | SHAP values |
| Compute cost | High | Low |

**Recommendation**: Use ensemble of both for complementary strengths.

---

## 8. Reinforcement Learning

### 8.1 DRL Agents for Trading

| Algorithm | Action Space | Best For |
|-----------|-------------|----------|
| **DQN** | Discrete (buy/sell/hold) | Volatile markets |
| **PPO** | Discrete or Continuous | Trending markets |
| **DDPG** | Continuous (position size) | Fine-grained control |
| **A2C** | Either | Balance speed/stability |

### 8.2 Ensemble DRL Strategy

Research shows combining multiple agents improves robustness:

```python
class EnsembleDRLTrader:
    def __init__(self):
        self.dqn = DQNAgent()   # Good in high volatility
        self.ppo = PPOAgent()   # Good in trending markets
        self.ddpg = DDPGAgent() # Good for position sizing

    def select_agent(self, market_state):
        """
        Switch between agents based on market conditions.
        """
        if market_state['volatility'] > 0.02:
            return self.dqn
        elif market_state['trend_strength'] > 0.7:
            return self.ppo
        else:
            return self.ddpg
```

**Key Reference**: [Deep RL for Algorithmic Trading (ResearchGate)](https://www.researchgate.net/publication/340644261_An_Application_of_Deep_Reinforcement_Learning_to_Algorithmic_Trading)

### 8.3 Reward Engineering

Critical for trading RL:

| Reward Type | Formula | Pros | Cons |
|-------------|---------|------|------|
| Raw P&L | Trade profit | Simple | Encourages risk |
| Sharpe-based | Return / Vol | Risk-adjusted | Slow feedback |
| Asymmetric | Penalize losses more | Drawdown control | Conservative |
| MFE-based | Reward good entries | Entry optimization | Complex |

### 8.4 Implementation Frameworks

| Framework | Description | Ease of Use |
|-----------|-------------|-------------|
| **Stable-Baselines3** | Standard RL algorithms | High |
| **FinRL** | Finance-specific RL | High |
| **RLlib** | Distributed training | Medium |
| **Custom** | Full control | Low |

---

## 9. Implementation Recommendations

### 9.1 Priority Roadmap

#### Phase 1: Target Engineering (Week 1-2)
```
1. Implement Triple Barrier labeling
2. Train meta-labeling secondary model
3. Validate with walk-forward testing
4. Expected improvement: Better F1, realistic trade counts
```

#### Phase 2: Volatility Features (Week 3-4)
```
1. Add VIX-based features (level, term structure, percentile)
2. Implement Hidden Markov Model for regime detection
3. Condition model on detected regime
4. Expected improvement: Strategy adapts to market conditions
```

#### Phase 3: Sentiment Integration (Week 5-6)
```
1. Set up FinBERT sentiment pipeline
2. Integrate financial news API (Alpha Vantage or Polygon)
3. Create sentiment features with proper lag
4. Expected improvement: Capture information not in price
```

#### Phase 4: Architecture Upgrade (Week 7-8)
```
1. Implement Temporal Fusion Transformer
2. Compare with LightGBM baseline
3. Ensemble if beneficial
4. Expected improvement: Better sequential pattern capture
```

### 9.2 Data Requirements

| Data Type | Source | Priority |
|-----------|--------|----------|
| ES 1-min bars | Databento (have) | Essential |
| VIX data | CBOE / Yahoo Finance | High |
| Financial news | Alpha Vantage / Polygon | Medium |
| Options flow | CBOE / broker | Low |
| Order book L2 | Rithmic / exchange | Low |

### 9.3 Realistic Expectations

Based on academic literature:

| Metric | Academic Papers Report | Realistic Target |
|--------|----------------------|------------------|
| Sharpe Ratio | 0.5 - 2.0 (after costs) | 1.0 - 1.5 |
| Win Rate | 50-55% | 52-55% |
| Profit Factor | 1.1 - 1.5 | 1.2 - 1.3 |
| Max Drawdown | 10-25% | < 20% |
| Annual Return | 10-30% (levered) | 15-25% |

**Warning**: Any backtest showing Sharpe > 3, Win Rate > 60%, or Profit Factor > 2 should be treated with extreme suspicion.

---

## 10. References

### Academic Papers

1. [Machine Learning in Futures Markets (MDPI 2021)](https://www.mdpi.com/1911-8074/14/3/119)
2. [High-Frequency Direction Forecasting (REPEC 2022)](https://ideas.repec.org/a/gam/jftint/v14y2022i6p180-d835486.html)
3. [Forecasting Futures with ML (Diva Portal 2024)](http://www.diva-portal.org/smash/get/diva2:1967848/FULLTEXT01.pdf)
4. [Deep Order Flow Imbalance (Semantic Scholar)](https://www.semanticscholar.org/paper/Deep-order-flow-imbalance:-Extracting-alpha-at-from-Kolm-Turiel/977e72a246b1a2b374288e2409694eb67d5dfbca)
5. [Forecasting High Frequency OFI (arXiv 2024)](https://arxiv.org/html/2408.03594v1)
6. [FinBERT-LSTM Stock Prediction (ACM 2024)](https://dl.acm.org/doi/10.1145/3694860.3694870)
7. [FinBERT Sentiment with SHAP (MDPI 2024)](https://www.mdpi.com/2227-7390/13/17/2747)
8. [Reddit WSB Sentiment (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S1057521924006537)
9. [ES-SPY Arbitrage (ResearchGate)](https://www.researchgate.net/publication/228880120_Index_Arbitrage_between_Futures_and_ETFs_Evidence_on_the_limits_to_arbitrage_from_SP_500_Futures_and_SPDRs)
10. [VIX ML Trading (PLOS One 2024)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0302289)
11. [Predicting VIX with Adaptive ML (Taylor & Francis 2024)](https://www.tandfonline.com/doi/full/10.1080/14697688.2024.2439458)
12. [TFT for Stock Prediction (IEEE 2022)](https://ieeexplore.ieee.org/document/9731073/)
13. [Deep RL for Algorithmic Trading (ResearchGate)](https://www.researchgate.net/publication/340644261_An_Application_of_Deep_Reinforcement_Learning_to_Algorithmic_Trading)
14. [Ensemble DRL for Trading (MDPI 2024)](https://www.mdpi.com/1911-8074/18/7/347)

### Books

1. **Lopez de Prado, M.** - *Advances in Financial Machine Learning* (2018) - Wiley
   - Triple Barrier Method, Meta-Labeling, Fractional Differentiation

2. **Chan, E.** - *Machine Trading* (2017) - Wiley
   - Practical ML for trading, backtesting methodology

### Libraries and Tools

| Library | Purpose | Link |
|---------|---------|------|
| mlfinlab | Financial ML (Lopez de Prado methods) | [GitHub](https://github.com/hudson-and-thames/mlfinlab) |
| FinBERT | Financial NLP | [HuggingFace](https://huggingface.co/ProsusAI/finbert) |
| Stable-Baselines3 | Reinforcement Learning | [GitHub](https://github.com/DLR-RM/stable-baselines3) |
| PyTorch Forecasting | TFT implementation | [GitHub](https://github.com/jdb78/pytorch-forecasting) |
| hftbacktest | HFT backtesting with order book | [GitHub](https://github.com/nkaz001/hftbacktest) |

---

*Document created: 2025-12-04*
*For: SKIE-Ninja ES Futures ML Trading System*
*Status: Research Complete - Implementation Phase Next*
