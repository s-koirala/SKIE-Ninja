"""
Run CPCV, PBO, and DSR Validation on SKIE_Ninja Strategy
=========================================================

Executes rigorous statistical validation per canonical literature:
- Lopez de Prado (2018) "Advances in Financial Machine Learning" Ch. 7
- Bailey et al. (2014) "The Probability of Backtest Overfitting"
- Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio"

This script validates the ensemble strategy to determine:
1. Whether performance is statistically significant across all combinatorial splits
2. The probability that the strategy is overfit to historical data
3. Whether the observed Sharpe survives multiple testing adjustment

Updated: 2026-01-06 (Canonical fixes per CPCV_PBO_IMPLEMENTATION_AUDIT_20260106.md)

Author: SKIE_Ninja Development Team
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

from validation.cpcv_pbo import (
    CPCVConfig, CombinatorialPurgedKFold,
    run_cpcv_validation, run_pbo_analysis, run_dsr_analysis,
    print_validation_report
)
from strategy.ensemble_strategy import EnsembleStrategy, EnsembleConfig
from data_collection.ninjatrader_loader import load_sample_data
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for 5-minute ES data
BARS_PER_DAY = 78  # RTH 9:30-16:00 = 6.5 hours = 78 5-min bars
PERIODS_PER_YEAR = 252 * BARS_PER_DAY  # ~19,656 bars/year


def generate_t1_series(n_samples: int, label_horizon: int = 10) -> pd.Series:
    """
    Generate t1 series for canonical CPCV purging (fixed horizon).

    t1[i] = the bar index where the label for sample i ends.
    For a label "will price move X bars forward", t1[i] = i + X.

    Args:
        n_samples: Number of samples
        label_horizon: Number of bars forward used in label calculation

    Returns:
        Series mapping sample index to label end index
    """
    return pd.Series(np.arange(n_samples) + label_horizon)


def generate_variable_t1_series(
    n_samples: int,
    target_columns: list,
    default_horizon: int = 10
) -> pd.Series:
    """
    Generate variable t1 series based on actual target horizons used.

    Per Lopez de Prado (2018) Section 7.4.1, t1 should reflect when the label
    for each sample ends. When multiple targets with different horizons are
    used, we take the MAXIMUM horizon per sample to be conservative.

    Target horizon extraction:
        - vol_expansion_{H}: H bars
        - new_high_{H}, new_low_{H}: H bars
        - trend_dir_{H}: H bars
        - reach_{mult}atr_up_{H}: H bars
        - future_return_{H}: H bars

    Args:
        n_samples: Number of samples
        target_columns: List of target column names being used
        default_horizon: Fallback horizon if parsing fails

    Returns:
        Series mapping sample index to label end index (i + max_horizon)
    """
    import re

    # Extract horizons from target column names
    horizons = []
    horizon_pattern = re.compile(r'_(\d+)(?:$|_)')

    for col in target_columns:
        # Find all numeric suffixes that represent horizons
        matches = horizon_pattern.findall(col)
        for match in matches:
            horizon = int(match)
            # Sanity check: horizons are typically 5-60 bars
            if 1 <= horizon <= 100:
                horizons.append(horizon)

    if horizons:
        max_horizon = max(horizons)
        logger.info(f"Variable t1: Detected horizons {sorted(set(horizons))}, using max={max_horizon}")
    else:
        max_horizon = default_horizon
        logger.warning(f"Variable t1: No horizons detected, using default={default_horizon}")

    # For fixed-horizon labels across all samples, t1 is uniform
    # For variable per-sample horizons, would need sample-level mapping
    return pd.Series(np.arange(n_samples) + max_horizon)


def generate_ensemble_strategy_returns(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    prices: pd.DataFrame,
    config: EnsembleConfig,
    thresholds_grid: list
) -> np.ndarray:
    """
    Generate strategy returns using trade-based simulation for accurate PBO.

    METHODOLOGY UPDATE (2026-01-06):
    Previous implementation applied transaction costs per-bar, which penalized
    strategies that generate frequent signals. This trade-based approach:
    1. Tracks position state (flat, long, short)
    2. Applies transaction cost only on position changes
    3. Uses ATR-based exit conditions matching live strategy
    4. Accumulates returns while in position

    Per Bailey et al. (2014), returns simulation must match actual trading
    behavior for PBO to be meaningful.

    Args:
        features: Feature DataFrame
        targets: Target DataFrame
        prices: Price DataFrame aligned with features
        config: Ensemble configuration
        thresholds_grid: List of (vol_thresh, breakout_thresh) tuples

    Returns:
        Returns matrix (n_periods, n_strategies) with trade-based returns
    """
    bars_per_day = BARS_PER_DAY
    train_days = 60
    test_days = 5
    embargo_bars = 210  # max(feature_lookback, label_horizon) + buffer

    train_bars = train_days * bars_per_day
    test_bars = test_days * bars_per_day

    X = features.values
    y_vol = targets['vol_expansion_5'].values
    y_high = targets['new_high_10'].values
    y_low = targets['new_low_10'].values
    y_atr = targets['future_atr_5'].values  # For ATR-based position sizing

    # Extract current ATR from features (atr_14 is in feature set)
    atr_col_idx = list(features.columns).index('atr_14') if 'atr_14' in features.columns else None
    current_atr_values = features['atr_14'].values if atr_col_idx is not None else None

    n_strategies = len(thresholds_grid)
    all_strategy_returns = [[] for _ in range(n_strategies)]

    # Position sizing parameters (from EnsembleConfig)
    base_contracts = config.base_contracts  # Default: 1
    max_contracts = config.max_contracts    # Default: 3

    # Transaction cost: 0.05% round-trip (2 ES ticks @ $12.50 / ~$5000 contract)
    TRANSACTION_COST = 0.0005

    # ATR-based exit parameters (matching live strategy)
    ATR_STOP_MULT = 2.0   # Stop loss at 2x ATR
    ATR_TARGET_MULT = 3.0  # Profit target at 3x ATR
    MAX_HOLD_BARS = 30    # Maximum bars to hold position

    # Walk-forward simulation
    start_idx = 0
    while start_idx + train_bars + embargo_bars + test_bars <= len(X):
        train_end = start_idx + train_bars
        test_start = train_end + embargo_bars
        test_end = test_start + test_bars

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[start_idx:train_end])
        X_test = scaler.transform(X[test_start:test_end])

        # Train all models including ATR model for position sizing
        vol_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
        high_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
        low_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
        atr_model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)

        vol_model.fit(X_train, y_vol[start_idx:train_end])
        high_model.fit(X_train, y_high[start_idx:train_end])
        low_model.fit(X_train, y_low[start_idx:train_end])
        atr_model.fit(X_train, y_atr[start_idx:train_end])

        # Get predictions for test period
        vol_probs = vol_model.predict_proba(X_test)[:, 1]
        high_probs = high_model.predict_proba(X_test)[:, 1]
        low_probs = low_model.predict_proba(X_test)[:, 1]
        predicted_atr = atr_model.predict(X_test)  # ATR predictions for sizing

        # Get current ATR for test period (for position sizing calculation)
        if current_atr_values is not None:
            test_current_atr = current_atr_values[test_start:test_end]
        else:
            # Fallback: use predicted ATR as current (neutral sizing)
            test_current_atr = predicted_atr

        # Get price data for test period
        test_prices = prices.iloc[test_start:test_end]
        close_prices = test_prices['close'].values
        high_prices = test_prices['high'].values
        low_prices = test_prices['low'].values

        # Generate returns for each strategy configuration using TRADE-BASED simulation
        for strat_idx, (vol_thresh, break_thresh) in enumerate(thresholds_grid):
            strategy_returns = []

            # Position state tracking
            position = 0        # 0 = flat, 1 = long, -1 = short
            entry_price = None
            entry_atr = None
            hold_bars = 0
            position_scale = 1.0

            for i in range(len(close_prices) - 1):
                bar_return = 0.0  # Return for this bar

                if position == 0:
                    # FLAT: Check for entry signal
                    vol_pass = vol_probs[i] >= vol_thresh

                    if vol_pass:
                        # Determine direction using full ensemble logic
                        high_signal = high_probs[i] >= break_thresh
                        low_signal = low_probs[i] >= break_thresh

                        if high_signal and not low_signal:
                            new_direction = 1
                        elif low_signal and not high_signal:
                            new_direction = -1
                        elif high_signal and low_signal:
                            new_direction = 1 if high_probs[i] > low_probs[i] else -1
                        else:
                            new_direction = 0

                        if new_direction != 0:
                            # ENTER POSITION
                            position = new_direction
                            entry_price = close_prices[i]
                            entry_atr = test_current_atr[i] if i < len(test_current_atr) else predicted_atr[i]
                            hold_bars = 0

                            # ATR-based position sizing
                            pred_atr = predicted_atr[i] + 1e-10
                            curr_atr = entry_atr
                            vol_factor = np.clip(curr_atr / pred_atr, 0.5, 2.0)
                            contracts = max(1, min(int(base_contracts * vol_factor), max_contracts))
                            position_scale = contracts / base_contracts

                            # Apply entry transaction cost (half of round-trip)
                            bar_return = -TRANSACTION_COST * position_scale * 0.5

                else:
                    # IN POSITION: Check for exit or accumulate return
                    hold_bars += 1
                    current_price = close_prices[i + 1]  # Next bar close
                    bar_high = high_prices[i + 1]
                    bar_low = low_prices[i + 1]

                    # Calculate price change return
                    price_return = (current_price - close_prices[i]) / close_prices[i]
                    bar_return = position * price_return * position_scale

                    # Check exit conditions
                    exit_trade = False
                    exit_reason = None

                    # Stop loss check (2x ATR)
                    stop_distance = ATR_STOP_MULT * entry_atr / entry_price
                    if position == 1:  # Long
                        if bar_low <= entry_price * (1 - stop_distance):
                            exit_trade = True
                            exit_reason = 'stop_loss'
                    else:  # Short
                        if bar_high >= entry_price * (1 + stop_distance):
                            exit_trade = True
                            exit_reason = 'stop_loss'

                    # Profit target check (3x ATR)
                    target_distance = ATR_TARGET_MULT * entry_atr / entry_price
                    if position == 1:  # Long
                        if bar_high >= entry_price * (1 + target_distance):
                            exit_trade = True
                            exit_reason = 'target'
                    else:  # Short
                        if bar_low <= entry_price * (1 - target_distance):
                            exit_trade = True
                            exit_reason = 'target'

                    # Max hold time exit
                    if hold_bars >= MAX_HOLD_BARS:
                        exit_trade = True
                        exit_reason = 'max_hold'

                    if exit_trade:
                        # EXIT POSITION: Apply exit transaction cost (half of round-trip)
                        bar_return -= TRANSACTION_COST * position_scale * 0.5
                        position = 0
                        entry_price = None
                        entry_atr = None
                        hold_bars = 0

                strategy_returns.append(bar_return)

            # Close any open position at end of test period
            if position != 0:
                # Exit at final close with transaction cost
                final_return = -TRANSACTION_COST * position_scale * 0.5
                strategy_returns.append(final_return)
            else:
                strategy_returns.append(0.0)

            all_strategy_returns[strat_idx].extend(strategy_returns)

        start_idx += test_bars

    # Convert to matrix
    min_len = min(len(r) for r in all_strategy_returns)
    returns_matrix = np.array([r[:min_len] for r in all_strategy_returns]).T

    return returns_matrix


def run_full_validation():
    """Run complete CPCV, PBO, and DSR validation on the ensemble strategy."""

    print("=" * 80)
    print("SKIE_NINJA STRATEGY VALIDATION")
    print("CPCV + PBO + DSR Analysis")
    print("Per Lopez de Prado (2018), Bailey et al. (2014)")
    print("=" * 80)

    # Load data
    logger.info("\n--- Loading Data ---")
    prices, _ = load_sample_data(source="databento")
    logger.info(f"Loaded {len(prices)} bars")

    # Filter RTH and resample
    if hasattr(prices.index, 'hour'):
        prices = prices[
            (prices.index.hour >= 9) &
            ((prices.index.hour < 16) | ((prices.index.hour == 9) & (prices.index.minute >= 30)))
        ]

    prices = prices.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    logger.info(f"After RTH + 5-min resample: {len(prices)} bars")

    # Initialize strategy
    config = EnsembleConfig(
        embargo_bars=210  # max(feature_lookback=200, label_horizon=30) + 10
    )
    strategy = EnsembleStrategy(config)

    # Prepare data
    logger.info("\n--- Preparing Features and Targets ---")
    features, targets, prices_aligned = strategy.prepare_data(prices)

    X = features.values
    y_vol = targets['vol_expansion_5'].values
    y_high = targets['new_high_10'].values
    y_low = targets['new_low_10'].values

    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Target samples: {len(y_vol)}")

    # Generate t1 for canonical purging using variable horizon detection
    # This extracts the maximum horizon from actual target column names:
    #   - vol_expansion_5 -> 5 bars
    #   - new_high_10, new_low_10 -> 10 bars
    # Maximum horizon is used for conservative purging
    target_columns_used = ['vol_expansion_5', 'new_high_10', 'new_low_10']
    t1 = generate_variable_t1_series(len(X), target_columns_used, default_horizon=10)
    logger.info(f"Generated variable t1 series from targets: {target_columns_used}")

    # =========================================================================
    # CPCV Validation
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)")
    print("Per Lopez de Prado (2018) Ch. 7")
    print("=" * 80)

    # CPCV Configuration per Lopez de Prado recommendations
    cpcv_config = CPCVConfig(
        n_splits=6,
        n_test_splits=2,
        purge_pct=0.01,
        embargo_pct=0.01,
        min_train_size=1000
    )

    logger.info(f"\nCPCV Configuration:")
    logger.info(f"  N (splits): {cpcv_config.n_splits}")
    logger.info(f"  k (test splits): {cpcv_config.n_test_splits}")
    logger.info(f"  Combinations: C({cpcv_config.n_splits},{cpcv_config.n_test_splits}) = 15")
    logger.info(f"  Purging: t1-based (canonical)")
    logger.info(f"  Embargo: {cpcv_config.embargo_pct:.1%}")

    # Run CPCV on vol expansion model with t1
    def model_factory():
        return lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )

    logger.info("\n--- Running CPCV on Vol Expansion Model (with t1) ---")
    cpcv_result = run_cpcv_validation(
        X, y_vol,
        model_factory=model_factory,
        metric_func=roc_auc_score,
        config=cpcv_config,
        t1=t1,  # Canonical purging
        use_sample_weights=True  # Use computed weights
    )

    print(f"\nVol Expansion Model CPCV Results:")
    print(f"  Mean AUC:     {cpcv_result['mean']:.4f}")
    print(f"  Std AUC:      {cpcv_result['std']:.4f}")
    print(f"  95% CI:       [{cpcv_result['ci_lower']:.4f}, {cpcv_result['ci_upper']:.4f}]")
    print(f"  t-statistic:  {cpcv_result['t_statistic']:.4f}")
    print(f"  p-value:      {cpcv_result['p_value']:.6f}")
    print(f"  t1 provided:  {cpcv_result['t1_provided']}")

    if cpcv_result['ci_lower'] > 0.5:
        print(f"  Status:       PASS - Lower CI bound > 0.5 (random)")
    else:
        print(f"  Status:       FAIL - Lower CI bound <= 0.5")

    # Test breakout models
    logger.info("\n--- Running CPCV on Breakout High Model ---")
    cpcv_high = run_cpcv_validation(X, y_high, model_factory, roc_auc_score, cpcv_config, t1=t1)

    logger.info("--- Running CPCV on Breakout Low Model ---")
    cpcv_low = run_cpcv_validation(X, y_low, model_factory, roc_auc_score, cpcv_config, t1=t1)

    print(f"\nBreakout High Model CPCV Results:")
    print(f"  Mean AUC:     {cpcv_high['mean']:.4f}")
    print(f"  95% CI:       [{cpcv_high['ci_lower']:.4f}, {cpcv_high['ci_upper']:.4f}]")
    print(f"  p-value:      {cpcv_high['p_value']:.6f}")

    print(f"\nBreakout Low Model CPCV Results:")
    print(f"  Mean AUC:     {cpcv_low['mean']:.4f}")
    print(f"  95% CI:       [{cpcv_low['ci_lower']:.4f}, {cpcv_low['ci_upper']:.4f}]")
    print(f"  p-value:      {cpcv_low['p_value']:.6f}")

    # =========================================================================
    # PBO Validation with Full Ensemble Logic
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: PROBABILITY OF BACKTEST OVERFITTING (PBO)")
    print("Per Bailey et al. (2014) - Using Full Ensemble Logic")
    print("=" * 80)

    # Generate strategy variants using full ensemble
    logger.info("\n--- Generating Strategy Returns with Full Ensemble Logic ---")

    # Create threshold grid
    vol_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]
    break_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]
    thresholds_grid = [(v, b) for v in vol_thresholds for b in break_thresholds]

    logger.info(f"  Strategy variants: {len(thresholds_grid)}")
    logger.info(f"  Vol thresholds: {vol_thresholds}")
    logger.info(f"  Breakout thresholds: {break_thresholds}")

    # Generate returns using full ensemble logic
    returns_matrix = generate_ensemble_strategy_returns(
        features, targets, prices_aligned, config, thresholds_grid
    )

    logger.info(f"  Returns matrix shape: {returns_matrix.shape}")
    logger.info(f"  Periods: {returns_matrix.shape[0]}")
    logger.info(f"  Strategies: {returns_matrix.shape[1]}")

    # Run PBO analysis with correct annualization for 5-min data
    logger.info("\n--- Running PBO Analysis ---")
    pbo_result = run_pbo_analysis(
        returns_matrix,
        n_trials=1000,
        periods_per_year=PERIODS_PER_YEAR  # Correct for 5-min bars
    )

    print(f"\nPBO Analysis Results:")
    print(f"  PBO:              {pbo_result['pbo']:.3f}")
    print(f"  95% CI:           [{pbo_result['pbo_ci_lower']:.3f}, {pbo_result['pbo_ci_upper']:.3f}]")
    print(f"  Strategies:       {pbo_result['n_strategies']}")
    print(f"  Trials:           {pbo_result['n_trials']}")
    print(f"  Method:           {pbo_result['method']}")
    print(f"  Periods/year:     {pbo_result['periods_per_year']}")
    print(f"  Interpretation:   {pbo_result['interpretation']}")

    # =========================================================================
    # DSR Validation
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: DEFLATED SHARPE RATIO (DSR)")
    print("Per Bailey & Lopez de Prado (2014)")
    print("=" * 80)

    # Use the "best" strategy configuration (0.50, 0.50) returns
    best_config_idx = thresholds_grid.index((0.50, 0.50))
    best_strategy_returns = returns_matrix[:, best_config_idx]

    # Number of trials = number of strategy variants tested
    n_trials_tested = len(thresholds_grid)

    logger.info("\n--- Running DSR Analysis ---")
    dsr_result = run_dsr_analysis(
        best_strategy_returns,
        n_trials=n_trials_tested,
        periods_per_year=PERIODS_PER_YEAR
    )

    print(f"\nDSR Analysis Results:")
    print(f"  Observed Sharpe:  {dsr_result['observed_sharpe']:.4f}")
    print(f"  E[max(SR)]:       {dsr_result['e_max_sr']:.4f} (haircut for {n_trials_tested} trials)")
    print(f"  DSR:              {dsr_result['dsr']:.4f}")
    print(f"  SE(SR):           {dsr_result['se_sr']:.4f}")
    print(f"  p-value:          {dsr_result['p_value']:.4f}")
    print(f"  Observations:     {dsr_result['n_observations']}")
    print(f"  Interpretation:   {dsr_result['interpretation']}")

    # =========================================================================
    # Summary Report
    # =========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    # Determine overall pass/fail
    cpcv_pass = (
        cpcv_result['ci_lower'] > 0.5 and
        cpcv_result['p_value'] < 0.05
    )

    pbo_pass = pbo_result['pbo'] < 0.50

    dsr_pass = dsr_result['p_value'] < 0.10  # More lenient for DSR

    print(f"\nCPCV Validation (vs AUC=0.5 baseline):")
    print(f"  Vol Model:        {'PASS' if cpcv_result['ci_lower'] > 0.5 else 'FAIL'} (AUC CI lower: {cpcv_result['ci_lower']:.3f})")
    print(f"  High Model:       {'PASS' if cpcv_high['ci_lower'] > 0.5 else 'FAIL'} (AUC CI lower: {cpcv_high['ci_lower']:.3f})")
    print(f"  Low Model:        {'PASS' if cpcv_low['ci_lower'] > 0.5 else 'FAIL'} (AUC CI lower: {cpcv_low['ci_lower']:.3f})")
    print(f"  t1 purging used:  {cpcv_result['t1_provided']}")

    print(f"\nPBO Validation (threshold: <0.50):")
    print(f"  PBO Score:        {'PASS' if pbo_pass else 'FAIL'} (PBO: {pbo_result['pbo']:.3f})")
    print(f"  Method:           {pbo_result['method']}")

    print(f"\nDSR Validation (threshold: p<0.10):")
    print(f"  DSR p-value:      {'PASS' if dsr_pass else 'FAIL'} (p: {dsr_result['p_value']:.4f})")
    print(f"  Haircut applied:  {dsr_result['e_max_sr']:.4f} (for {n_trials_tested} trials)")

    print(f"\n{'='*80}")
    all_pass = cpcv_pass and pbo_pass and dsr_pass
    if all_pass:
        print("OVERALL STATUS: VALIDATED FOR PAPER TRADING")
        print("  - CPCV shows statistically significant predictive power with canonical purging")
        print("  - PBO indicates low overfitting risk with full ensemble returns")
        print("  - DSR survives multiple testing adjustment")
    else:
        print("OVERALL STATUS: NOT FULLY VALIDATED")
        if not cpcv_pass:
            print("  - CPCV: Model lacks statistically significant predictive power")
        if not pbo_pass:
            print("  - PBO: High probability of backtest overfitting")
        if not dsr_pass:
            print("  - DSR: Observed Sharpe does not survive multiple testing adjustment")
    print("=" * 80)

    # Print detailed report
    print("\n" + "=" * 80)
    print("DETAILED VALIDATION REPORT")
    print_validation_report(cpcv_result, pbo_result, dsr_result)

    # Save results
    output_dir = project_root / 'data' / 'validation_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df = pd.DataFrame([{
        'timestamp': timestamp,
        'vol_auc': cpcv_result['mean'],
        'vol_auc_ci_lower': cpcv_result['ci_lower'],
        'vol_p_value': cpcv_result['p_value'],
        't1_provided': cpcv_result['t1_provided'],
        'high_auc': cpcv_high['mean'],
        'low_auc': cpcv_low['mean'],
        'pbo': pbo_result['pbo'],
        'pbo_method': pbo_result['method'],
        'pbo_periods_per_year': pbo_result['periods_per_year'],
        'pbo_interpretation': pbo_result['interpretation'],
        'dsr': dsr_result['dsr'],
        'dsr_p_value': dsr_result['p_value'],
        'dsr_observed_sharpe': dsr_result['observed_sharpe'],
        'dsr_haircut': dsr_result['e_max_sr'],
        'dsr_interpretation': dsr_result['interpretation'],
        'cpcv_pass': cpcv_pass,
        'pbo_pass': pbo_pass,
        'dsr_pass': dsr_pass,
        'overall_pass': all_pass
    }])

    results_df.to_csv(output_dir / f'cpcv_pbo_dsr_validation_{timestamp}.csv', index=False)
    logger.info(f"\nResults saved to: cpcv_pbo_dsr_validation_{timestamp}.csv")

    return {
        'cpcv': cpcv_result,
        'cpcv_high': cpcv_high,
        'cpcv_low': cpcv_low,
        'pbo': pbo_result,
        'dsr': dsr_result,
        'overall_pass': all_pass
    }


if __name__ == "__main__":
    run_full_validation()
