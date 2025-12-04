# SKIE-Ninja Backtesting Methodology

**Document Version**: 1.0
**Last Updated**: 2025-12-04
**Author**: SKIE_Ninja Development Team

---

## Table of Contents

1. [Overview](#overview)
2. [Walk-Forward Validation](#walk-forward-validation)
3. [Data Handling](#data-handling)
4. [Model Training](#model-training)
5. [Trade Execution](#trade-execution)
6. [Risk Management](#risk-management)
7. [Quality Control](#quality-control)
8. [Performance Metrics](#performance-metrics)
9. [Best Practices](#best-practices)
10. [References](#references)

---

## Overview

This document describes the methodology used in the SKIE-Ninja walk-forward backtesting framework for algorithmic trading on ES (S&P 500 E-mini) futures.

### Key Principles

1. **Temporal Integrity**: No look-ahead bias; train only on past data
2. **Realistic Costs**: Include commission ($2.50/side) and slippage (0.5 ticks)
3. **RTH Only**: Trade only during Regular Trading Hours (9:30 AM - 4:00 PM ET)
4. **Data Leakage Prevention**: Embargo periods and feature validation
5. **Statistical Validation**: Multiple quality control checks

---

## Walk-Forward Validation

### Methodology

Walk-forward validation (WFV) is the gold standard for backtesting trading strategies. Unlike random cross-validation, WFV respects the temporal ordering of financial data.

```
[=== TRAIN (180 days) ===][EMBARGO (42 bars)][= TEST (5 days) =]
                                              ↓
                         [=== TRAIN (180 days) ===][EMBARGO][= TEST =]
                                                             ↓
                                      [=== TRAIN (180 days) ===][EMBARGO][= TEST =]
```

### Configuration (Optimal from Grid Search)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Train Window | 180 days | Captures sufficient market regimes |
| Test Window | 5 days | Balances bias-variance tradeoff |
| Embargo Period | 42 bars (~3.5 hours) | Prevents target leakage |
| Bars per Day | 78 (5-min RTH) | Regular trading hours only |
| Expected Folds | ~61 | Provides statistical significance |

### Reference

> "Walk-forward analysis is the only testing methodology that can reasonably ensure that a trading system is not curve-fitted."
> — Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*

---

## Data Handling

### Data Sources

| Source | Instrument | Timeframe | Period |
|--------|-----------|-----------|--------|
| Databento | ES | 1-minute | 2020-2024 |
| Databento | NQ, YM, GC, CL, ZN | 1-minute | 2023-2024 |
| Yahoo Finance | VIX, DX | Daily | 2+ years |
| FRED | Macro indicators | Various | Various |

### Data Pipeline

1. **Load Raw Data**: 1-minute OHLCV from Databento
2. **Resample**: Convert to 5-minute bars
3. **RTH Filter**: Keep only 9:30 AM - 4:00 PM ET
4. **Validation**: Check OHLCV relationships, missing values
5. **Feature Engineering**: Calculate 474 features

### RTH (Regular Trading Hours) Specification

| Parameter | Value |
|-----------|-------|
| Start Time | 9:30 AM ET |
| End Time | 4:00 PM ET |
| Timezone | America/New_York |
| Trading Days | Monday - Friday |
| Bars per Day | 78 (5-minute) |

### Data Quality Checks

- [ ] No missing values in OHLCV columns
- [ ] OHLC relationship: Low ≤ Open, Close ≤ High
- [ ] Timestamps are monotonically increasing
- [ ] No duplicate timestamps
- [ ] Volume is non-negative

---

## Model Training

### Supported Models

| Model | Library | Purpose |
|-------|---------|---------|
| LightGBM | lightgbm | Primary (best AUC) |
| XGBoost | xgboost | Secondary |
| RandomForest | sklearn | Baseline |
| LSTM/GRU | PyTorch | Deep learning |

### Feature Selection

**Method**: Multi-method ranking with aggregation

1. F-Test (ANOVA)
2. Mutual Information
3. Random Forest Importance
4. Target Correlation

**Selected Features**: Top 75 from 474 candidates

### Top 10 Predictive Features

1. `pyramid_rr_5` - Pyramiding reward-to-risk (5-bar)
2. `pyramid_rr_10` - Pyramiding reward-to-risk (10-bar)
3. `bars_in_session` - Time context
4. `pyramid_rr_20` - Pyramiding reward-to-risk (20-bar)
5. `dist_to_support` - Distance to support level
6. `stoch_diff_14` - Stochastic divergence
7. `dist_to_resistance` - Distance to resistance level
8. `estimated_sell_volume` - Order flow
9. `atr_20` - 20-bar Average True Range
10. `rsi_dist_50_7` - RSI distance from 50

### Training Configuration

```python
# Gradient Boosting (LightGBM/XGBoost)
params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'num_boost_round': 100
}

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler!
```

### RNN Overfitting Prevention (Purged K-Fold CV)

For LSTM/GRU models, we use Purged K-Fold CV based on de Prado (2018):

| Technique | Implementation | Purpose |
|-----------|---------------|---------|
| Purge Period | 200 bars | Remove samples near test set |
| Embargo Period | 42 bars | Gap after test set |
| Dropout | 50% | Reduce overfitting |
| Weight Decay | 1e-4 | L2 regularization |
| Early Stopping | 5 epochs | Prevent overtraining |

---

## Triple Barrier Labeling (Advanced)

### Overview

Triple Barrier Labeling (Lopez de Prado, 2018) replaces traditional binary labels with a more realistic labeling scheme that accounts for:
- **Take Profit** (upper barrier)
- **Stop Loss** (lower barrier)
- **Time Expiration** (vertical barrier)

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Upper Barrier | 2.0 × ATR | Dynamic profit target based on volatility |
| Lower Barrier | 1.0 × ATR | Risk-adjusted stop loss |
| Vertical Barrier | 10 bars | Maximum holding period (~50 minutes) |
| Min Holding Period | 1 bar | Prevents immediate exits |
| ATR Period | 14 bars | Standard volatility lookback |

### Label Distribution

For ES 5-min RTH data (2023-2024):
- **Long (Upper Hit)**: ~38% - Price rises to take profit
- **Short (Lower Hit)**: ~60% - Price falls to stop loss
- **Flat (Time Exit)**: ~2% - Neither barrier hit within window

### Literature Reference

> "The triple barrier method produces labels that better reflect the realities of trading... By labeling based on what happens first (barrier breach or timeout), we capture the path-dependent nature of trading."
> — Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 3

---

## Meta-Labeling (Bet Sizing)

### Overview

Meta-labeling (Lopez de Prado, 2018) is a two-stage approach:
1. **Primary Model**: Predicts direction (long/short/flat)
2. **Meta Model**: Predicts whether to take the trade (bet sizing)

### Architecture

```
Primary Model (LightGBM)     Meta Model (Secondary Classifier)
       ↓                              ↓
   Direction                    Bet Size (0-1)
       ↓                              ↓
       └────────────┬────────────────┘
                    ↓
            Final Trade Decision
```

### Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Min Probability | 0.55 | Minimum confidence to trade |
| Sizing Method | 'kelly' | Kelly criterion for position sizing |
| Max Position Size | 1.0 | Maximum fraction of capital |

### Benefits

1. **Precision Over Recall**: Filters low-confidence trades
2. **Bet Sizing**: Adjusts position size based on confidence
3. **Decoupled Training**: Primary and meta models train independently
4. **Reduced Overfitting**: Meta model acts as regularization

### Validation Results (Session 7)

| Metric | Primary Only | With Meta | Impact |
|--------|--------------|-----------|--------|
| Total Trades | 23,750 | 1,376 | -94.2% filtered |
| Win Rate | 48.1% | 48.5% | Neutral |
| Net P&L | Negative | -$12,385 | No edge |

**Interpretation**: Meta-labeling correctly filters trades, but underlying primary model has no predictive edge (AUC 0.5084).

### Literature Reference

> "Meta-labeling adds a layer that learns which predictions are worth acting upon... This approach can significantly improve the precision of a strategy."
> — Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 3.6

---

## Trade Execution

### Entry Rules

| Condition | Value |
|-----------|-------|
| Long Entry | Probability > 0.55 |
| Short Entry | Probability < 0.45 |
| Min Signal Strength | |prob - 0.5| > 0.10 |
| Max Daily Trades | 10 |

### Exit Rules

| Condition | Value |
|-----------|-------|
| Hold Bars | 3 bars (15 minutes) |
| Stop Loss | 20 ticks (5 points = $250/contract) |
| Take Profit | 40 ticks (10 points = $500/contract) |
| End of Day | Close all positions |

### Cost Model

| Cost | Value | Notes |
|------|-------|-------|
| Commission | $2.50/side | Per contract |
| Slippage | 0.5 ticks | $6.25/contract |
| Total Round Trip | ~$11.25/contract | Conservative estimate |

### Position Sizing

| Parameter | Value |
|-----------|-------|
| Contracts per Trade | 1 |
| Max Contracts | 5 |
| Margin (Intraday) | $500/contract |

---

## Risk Management

### Position Limits

- Maximum 1 position at a time
- Maximum 10 trades per day
- Maximum 5 consecutive losses before pause

### Loss Limits

- Daily loss limit: $1,000
- Maximum drawdown: 10% of peak equity

### Trade Validation

All trades must pass:
- [ ] Entry within RTH
- [ ] Exit within RTH
- [ ] Valid signal strength
- [ ] Daily trade limit not exceeded
- [ ] Daily loss limit not exceeded

---

## Quality Control

### Pre-Backtest Validation

1. **Data Quality**
   - OHLCV relationship validation
   - Missing value check (< 1%)
   - Timestamp monotonicity

2. **Feature Quality**
   - No infinite values
   - No extreme outliers (> 10 std)
   - No data leakage (correlation > 0.95)

3. **Model Quality**
   - Minimum train samples: 10,000
   - Minimum test samples: 1,000
   - Embargo period: ≥ 20 bars

### Post-Backtest Validation

1. **Trade Validation**
   - RTH compliance: 100%
   - Minimum trades: 100
   - Win rate: 30% - 80% (suspicious if outside)

2. **Metric Validation**
   - Profit factor: < 5.0 (suspicious if higher)
   - Sharpe ratio: < 3.0 (suspicious if higher)
   - AUC-ROC: < 0.95 (suspicious if higher)

### Data Leakage Detection

**Literature-Based Thresholds** (Lopez de Prado, 2018; Bailey et al., 2014):

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| Feature-Target Correlation | < 0.30 | Higher suggests leakage or overfitting |
| AUC-ROC Upper Bound | < 0.70 | AUC > 0.70 is suspicious for financial data |
| AUC-ROC Lower Bound | > 0.51 | Must beat random by meaningful margin |
| Sharpe Ratio | < 3.0 | Sharpe > 3 is unrealistic for most strategies |
| Win Rate | < 65% | Win rate > 65% suggests unrealistic assumptions |

**Signs of Data Leakage:**
- Feature correlation with target > 0.30
- AUC-ROC > 0.70 (too good to be true)
- Win rate > 65%
- Profit factor > 3.0
- Suspiciously named features (future_, target_, next_)

```python
# Comprehensive leakage detection (run_triple_barrier_backtest.py)
class DataLeakageChecker:
    def check_feature_target_correlation(self, X, y):
        for i, feature in enumerate(X.T):
            corr = np.abs(np.corrcoef(feature, y)[0, 1])
            if corr > 0.30:
                self.warnings.append(f"High correlation: feature_{i} = {corr:.3f}")
        return len(self.warnings) == 0

    def check_suspicious_features(self, feature_names):
        suspicious = ['future', 'target', 'next_', 'forward', '_y', 'label']
        for name in feature_names:
            if any(s in name.lower() for s in suspicious):
                self.warnings.append(f"Suspicious feature name: {name}")
        return len(self.warnings) == 0
```

### QC Report Format

Each backtest generates a comprehensive QC report:

```
============================================================
QUALITY CONTROL REPORT
============================================================
Date: 2025-12-04 14:30:00
Model: LightGBM + Meta-Labeling

Feature Checks:
  - Max feature-target correlation: 0.0305 ✅ (< 0.30)
  - Suspicious feature names: None ✅

Model Checks:
  - Primary AUC: 0.5084 ⚠️ (borderline)
  - Not too high: ✅ (< 0.70)

Performance Checks:
  - Sharpe Ratio: -0.12 ✅ (< 3.0)
  - Win Rate: 48.5% ✅ (< 65%)

OVERALL STATUS: PASSED (No data leakage detected)
============================================================
```

---

## Performance Metrics

### Trade Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Win Rate | Winning trades / Total trades | 45-55% |
| Profit Factor | Gross profit / Gross loss | > 1.2 |
| Payoff Ratio | Avg win / |Avg loss| | > 1.5 |
| Expectancy | Avg P&L per trade | > $0 |

### Risk Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Max Drawdown | Largest peak-to-trough decline | < 10% |
| Sharpe Ratio | Risk-adjusted return | > 1.0 |
| Sortino Ratio | Downside risk-adjusted return | > 1.5 |
| Calmar Ratio | Return / Max Drawdown | > 1.0 |

### Trade Analysis

| Metric | Description |
|--------|-------------|
| MFE | Max Favorable Excursion (best unrealized P&L) |
| MAE | Max Adverse Excursion (worst unrealized P&L) |
| Avg Bars Held | Average trade duration |
| Time in Trade | Average minutes per trade |

### Model Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| AUC-ROC | Area Under ROC Curve | > 0.60 |
| Accuracy | Correct predictions / Total | > 55% |
| F1 Score | Harmonic mean of precision/recall | > 0.55 |

---

## Best Practices

### Do's

1. **Always use walk-forward validation** - never random splits
2. **Include realistic costs** - commission AND slippage
3. **Trade only during RTH** - better liquidity
4. **Use embargo periods** - prevent data leakage
5. **Validate before and after** - quality control checks
6. **Document everything** - reproducibility

### Don'ts

1. **Don't optimize on full dataset** - leads to overfitting
2. **Don't ignore transaction costs** - unrealistic results
3. **Don't trade overnight** - different regime
4. **Don't trust suspiciously good results** - probably leakage
5. **Don't use future data** - look-ahead bias
6. **Don't ignore drawdowns** - risk matters

### Warning Signs

| Sign | Possible Issue |
|------|---------------|
| AUC > 0.95 | Data leakage |
| Win rate > 80% | Unrealistic costs |
| Sharpe > 3.0 | Overfitting |
| Profit factor > 5.0 | Survivorship bias |
| No losing streaks | Simulation error |

---

## References

1. **de Prado, M. L. (2018)**. *Advances in Financial Machine Learning*. Wiley.
   - Chapter 7: Cross-Validation in Finance
   - Chapter 8: Feature Importance

2. **Pardo, R. (2008)**. *The Evaluation and Optimization of Trading Strategies*. Wiley.
   - Walk-forward analysis methodology

3. **Bailey, D. H., et al. (2014)**. "The Probability of Backtest Overfitting". *Journal of Computational Finance*.
   - Statistical validation of strategies

4. **Harvey, C. R., et al. (2016)**. "... and the Cross-Section of Expected Returns". *Review of Financial Studies*.
   - Multiple testing correction

5. **Fischer, T. & Krauss, C. (2018)**. "Deep Learning with Long Short-Term Memory Networks for Financial Market Predictions". *European Journal of Operational Research*.
   - LSTM regularization for finance

6. **Grinsztajn, L., et al. (2022)**. "Why do tree-based models still outperform deep learning on typical tabular data?"
   - Gradient boosting vs deep learning

---

## Appendix: File Reference

| File | Purpose |
|------|---------|
| `comprehensive_backtest.py` | Walk-forward backtesting engine |
| `purged_cv_rnn_trainer.py` | RNN training with Purged CV |
| `validation_framework.py` | Quality control checks |
| `run_validated_backtest.py` | Full pipeline runner |
| `run_triple_barrier_backtest.py` | Triple Barrier + Meta-labeling pipeline |
| `feature_pipeline.py` | Feature engineering |
| `data_resampler.py` | OHLCV resampling utilities |
| `triple_barrier.py` | Triple Barrier labeling implementation |
| `meta_labeling.py` | Meta-labeling for bet sizing |
| `volatility_regime.py` | Volatility regime detection (VIX, HMM) |

---

*Document maintained by SKIE_Ninja Development Team*
