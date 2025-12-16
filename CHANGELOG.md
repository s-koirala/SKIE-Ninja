# SKIE_Ninja Changelog

All notable changes to the SKIE_Ninja project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- Shared utility modules for feature engineering (Phase C)
  - `feature_engineering/shared/technical_utils.py` - TR, ATR, RSI, Stochastic, BB, MACD
  - `feature_engineering/shared/returns_utils.py` - Return calculations
  - `feature_engineering/shared/volume_utils.py` - Volume features, VWAP, OBV
  - `feature_engineering/shared/temporal_utils.py` - Cyclical time encoding

### Changed
- Documentation updated to resolve inconsistencies
- CANONICAL_REFERENCE.md clarified ensemble vs baseline totals

---

## [0.15.0] - 2025-12-15

### Added - Phase A (Overfitting Detection)
- `quality_control/overfitting_detection.py` - DSR, CSCV, PSR implementations
- `run_overfitting_assessment.py` - Comprehensive overfitting tests
- `tests/test_critical_functions.py` - Pytest validation suite

### Added - Phase B (Parameter Justification)
- `run_window_optimization.py` - Data-driven train/test window selection
- `run_embargo_analysis.py` - Autocorrelation-based embargo justification

### Changed - API Key Security
- `config/api_keys.py` - Now loads from environment variables (no hardcoded keys)
- `config/api_keys.env.template` - Template for secure key configuration

### Fixed - Socket Bridge (Critical)
- VIX buffer lag: Changed T-2 to T-1 in `ninja_signal_server.py`
- Feature count mismatch: Added 27 sentiment features
- VIX percentile: Now uses historical VIX reference data
- Feature validation: Added count/order validation before prediction
- P&L tracking: Fixed to use `GetProfitLoss()` in `SKIENinjaStrategy.cs`
- Heartbeat: Added 30-second heartbeat timer for connection monitoring

### Added
- `requirements.txt` - Python dependencies with pinned versions
- `docs/DATA_DRIVEN_DECISIONS.md` - Parameter justification framework
- `docs/AUDIT_REPORT.md` - Comprehensive audit findings

---

## [0.14.0] - 2025-12-05

### Added - Enhanced Validation
- Monte Carlo simulation with 10,000 iterations
- Bootstrap resampling validation
- Trade dropout sensitivity analysis
- Cost variance testing

### Results
- 100% probability of profit across all simulations
- 95% CI for Net P&L: [$361K, $573K]
- Statistical robustness confirmed

---

## [0.13.0] - 2025-12-05

### Added - Threshold Optimization
- `run_threshold_optimization.py` - 256-point grid search
- `run_ensemble_threshold_optimization.py` - Ensemble parameter optimization

### Changed
- Entry thresholds optimized: `min_vol_expansion_prob=0.40`, `min_breakout_prob=0.45`
- Exit multipliers optimized: `tp_atr_mult=2.5`, `sl_atr_mult=1.25`
- Performance improvement: +96% over defaults

---

## [0.12.0] - 2025-12-04

### Added - Ensemble Strategy
- `strategy/ensemble_strategy.py` - Combined vol + sentiment strategy
- `run_ensemble_oos_backtest.py` - Ensemble OOS validation
- `run_ensemble_2025_forward_test.py` - Ensemble forward test

### Results
- Ensemble improves over baseline by 7.4% (IS), 1.2% (OOS), 4.3% (Forward)
- Total validated edge: $786,879 across 5 years

---

## [0.11.0] - 2025-12-04

### Added - Sentiment Strategy
- `strategy/sentiment_strategy.py` - VIX-based sentiment strategy
- `data_collection/historical_sentiment_loader.py` - Historical VIX loader

### Findings
- Sentiment predicts WHEN (vol expansion AUC 0.77) but not WHICH WAY
- Best used as filter, not standalone

---

## [0.10.0] - 2025-12-04

### Added - Forward Testing
- `run_2025_forward_test.py` - 2025 forward test script
- ES_2025_1min_databento.csv - 326K bars of 2025 data

### Results
- Forward test (2025): $57,394 net P&L, 2.66 Sharpe
- Consistent with OOS performance (no degradation)

---

## [0.9.0] - 2025-12-04

### Added - Out-of-Sample Validation
- `run_oos_backtest.py` - OOS validation script
- ES_2020/2021/2022 1min data downloaded

### Results
- OOS (2020-2022): $496,380 net P&L, 3.09 Sharpe
- All years profitable (PSR = 100%)

---

## [0.8.0] - 2025-12-04

### Fixed - Data Leakage (CRITICAL)
- Removed `pyramid_rr_*` features (used shift(-N))
- Removed `pivot_high_*`, `pivot_low_*` (forward-looking window)
- Removed `ddca_buy/sell_success_*` (used close.shift(-horizon))
- Deleted all models trained on leaky data

### Changed
- Win rate dropped from 86% to 40% (expected, real)
- All features now use only historical data

---

## [0.7.0] - 2025-12-04

### Added - Volatility Breakout Strategy
- `strategy/volatility_breakout_strategy.py` - Main production strategy
- Multi-target prediction (vol expansion, breakout, ATR)

### Core Insight
- Don't predict direction (AUC ~0.50)
- Predict WHEN (vol expansion AUC 0.84)
- Predict WHERE (breakout high/low AUC 0.72)
- Predict HOW MUCH (ATR forecast R² 0.36)

---

## [0.6.0] - 2025-12-03

### Added - Feature Engineering
- `feature_engineering/multi_target_labels.py` - 73-target generator
- `feature_engineering/volatility_regime.py` - VIX regime detection
- `feature_engineering/triple_barrier.py` - Triple barrier labeling

---

## [0.5.0] - 2025-12-02

### Added - Quality Control
- `run_qc_check.py` - Data leakage detection
- Feature-target correlation checks
- Suspicious performance flagging

---

## [0.4.0] - 2025-12-01

### Added - Data Collection
- Databento ES futures data (2020-2025)
- VIX historical data
- FRED macro indicators

---

## [0.3.0] - 2025-11-30

### Added - Project Structure
- Initial directory structure
- Configuration files
- Documentation templates

---

## [0.2.0] - 2025-11-29

### Added - Research
- `research/04_multi_target_prediction_strategy.md` - Strategy design
- `research/05_sentiment_strategy_plan.md` - Sentiment integration plan

---

## [0.1.0] - 2025-11-28

### Added
- Initial project setup
- Environment configuration
- Basic documentation

---

## Performance Summary

| Version | Period | Net P&L | Sharpe | Win Rate |
|---------|--------|---------|--------|----------|
| 0.15.0 | IS 2023-24 | $224,813 | 4.56 | 43.3% |
| 0.15.0 | OOS 2020-22 | $502,219 | 3.16 | 40.4% |
| 0.15.0 | Forward 2025 | $59,847 | 2.66 | 39.5% |
| **Total** | **5 Years** | **$786,879** | - | - |

---

*Maintained by SKIE_Ninja Development Team*
