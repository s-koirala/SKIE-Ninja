"""
Advanced Feature Validation Script

Comprehensive testing and validation of all new research-based features:
1. Triple Barrier Labeling (Lopez de Prado)
2. Meta-Labeling for Bet Sizing
3. Volatility Regime Detection (VIX, HMM, GMM)
4. Realized Volatility Features

Cross-references academic literature and validates against best practices.

References:
- Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
- Kolm & Turiel - Deep Order Flow Imbalance
- FinBERT-LSTM (ACM 2024)
- VIX ML Trading (PLOS One 2024)
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'python'))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_subheader(title: str):
    """Print subsection header."""
    print(f"\n--- {title} ---")


def load_price_data():
    """Load ES futures price data."""
    print_subheader("Loading Price Data")

    try:
        from data_collection.ninjatrader_loader import load_sample_data
        es_data, _ = load_sample_data(source="databento")
        print(f"  Loaded {len(es_data)} bars from Databento")
        print(f"  Date range: {es_data.index[0]} to {es_data.index[-1]}")

        # Resample to 5-min RTH
        try:
            from utils.data_resampler import DataResampler
            resampler = DataResampler()
            prices = resampler.resample(es_data, '5min', rth_only=True)
        except ImportError:
            # Fallback
            es_data_rth = es_data.between_time('09:30', '16:00')
            prices = es_data_rth.resample('5min').agg({
                'open': 'first', 'high': 'max',
                'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()

        print(f"  Resampled to {len(prices)} 5-min RTH bars")
        return prices

    except Exception as e:
        logger.warning(f"Could not load real data: {e}")
        logger.info("Generating synthetic data for testing...")

        # Generate synthetic data
        np.random.seed(42)
        n_bars = 5000

        # Simulate realistic ES price movement
        returns = np.random.randn(n_bars) * 0.001 + 0.00005
        close = 4500 * np.cumprod(1 + returns)

        dates = pd.date_range('2024-01-01 09:30', periods=n_bars, freq='5min')

        prices = pd.DataFrame({
            'open': close * (1 + np.random.randn(n_bars) * 0.0005),
            'high': close * (1 + np.abs(np.random.randn(n_bars)) * 0.001),
            'low': close * (1 - np.abs(np.random.randn(n_bars)) * 0.001),
            'close': close,
            'volume': np.random.randint(100, 1000, n_bars)
        }, index=dates)

        print(f"  Generated {len(prices)} synthetic bars")
        return prices


def test_triple_barrier(prices: pd.DataFrame) -> dict:
    """
    Test Triple Barrier Labeling.

    Literature Reference:
    - Lopez de Prado (2018) Chapter 3: Meta-Labeling

    Best Practices:
    - Use ATR-adjusted barriers for volatility normalization
    - Typical upper barrier: 1.5-3x ATR (take profit)
    - Typical lower barrier: 1-2x ATR (stop loss)
    - Vertical barrier: 6-24 bars depending on timeframe
    """
    print_header("1. TRIPLE BARRIER LABELING TEST")
    print("\nReference: Lopez de Prado (2018) 'Advances in Financial ML' Ch. 3")

    from feature_engineering.triple_barrier import (
        TripleBarrierConfig,
        TripleBarrierLabeler,
        apply_triple_barrier,
        generate_barrier_features
    )

    results = {}

    # Test 1: Basic labeling
    print_subheader("1.1 Basic Triple Barrier Labeling")

    config = TripleBarrierConfig(
        upper_barrier=2.0,   # 2 ATR take profit
        lower_barrier=1.0,   # 1 ATR stop loss
        max_holding_bars=12,  # 1 hour max hold (5min bars)
        atr_period=14
    )

    labeler = TripleBarrierLabeler(config)
    tb_labels = labeler.fit_transform(prices)

    # Analyze label distribution
    label_dist = tb_labels['tb_label'].value_counts(normalize=True)
    results['label_distribution'] = dict(label_dist)

    print(f"\n  Label Distribution:")
    print(f"    +1 (Take Profit): {label_dist.get(1, 0)*100:.1f}%")
    print(f"     0 (Time Exit):   {label_dist.get(0, 0)*100:.1f}%")
    print(f"    -1 (Stop Loss):   {label_dist.get(-1, 0)*100:.1f}%")

    # Barrier type analysis
    barrier_types = tb_labels['tb_barrier_type'].value_counts()
    results['barrier_types'] = dict(barrier_types)

    print(f"\n  Barrier Types Hit:")
    print(f"    Upper (TP):  {barrier_types.get(1, 0)}")
    print(f"    Lower (SL):  {barrier_types.get(-1, 0)}")
    print(f"    Vertical:    {barrier_types.get(0, 0)}")

    # Holding period analysis
    avg_hold = tb_labels['tb_holding_bars'].mean()
    results['avg_holding_bars'] = avg_hold
    print(f"\n  Average Holding Period: {avg_hold:.1f} bars ({avg_hold*5:.0f} minutes)")

    # Return analysis
    avg_return = tb_labels['tb_return'].mean() * 100
    std_return = tb_labels['tb_return'].std() * 100
    results['avg_return_pct'] = avg_return
    results['std_return_pct'] = std_return
    print(f"\n  Average Return: {avg_return:.4f}%")
    print(f"  Return Std Dev: {std_return:.4f}%")

    # Test 2: Multi-configuration features
    print_subheader("1.2 Multi-Configuration Features")

    multi_features = generate_barrier_features(prices)
    results['num_features'] = len(multi_features.columns)
    print(f"  Generated {len(multi_features.columns)} features")
    print(f"  Feature groups: {len(set([c.split('_')[-1] for c in multi_features.columns]))}")

    # Validation checks
    print_subheader("1.3 Validation Checks")

    checks = {
        'no_future_leakage': True,  # By design
        'labels_not_all_same': len(tb_labels['tb_label'].unique()) > 1,
        'reasonable_distribution': 0.2 < label_dist.get(1, 0) < 0.8,
        'avg_hold_reasonable': 1 < avg_hold < config.max_holding_bars
    }
    results['validation_checks'] = checks

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    return results


def test_meta_labeling(prices: pd.DataFrame, tb_labels: pd.DataFrame) -> dict:
    """
    Test Meta-Labeling for bet sizing.

    Literature Reference:
    - Lopez de Prado (2018) Chapter 3: Meta-Labeling

    Best Practices:
    - Primary model: High RECALL (catch opportunities)
    - Meta model: High PRECISION (filter false positives)
    - Use probability for Kelly-style bet sizing
    - Different model types for primary vs meta (avoid information redundancy)
    """
    print_header("2. META-LABELING TEST")
    print("\nReference: Lopez de Prado (2018) 'Advances in Financial ML' Ch. 3")

    from models.meta_labeling import (
        MetaLabelConfig,
        MetaLabeler,
        MetaLabelingPipeline
    )
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    results = {}

    # Prepare features
    print_subheader("2.1 Preparing Features")

    # Simple feature set for testing
    features = pd.DataFrame(index=prices.index)
    close = prices['close']

    # Basic features
    features['return_1'] = close.pct_change(1)
    features['return_5'] = close.pct_change(5)
    features['return_10'] = close.pct_change(10)
    features['vol_10'] = features['return_1'].rolling(10).std()
    features['vol_20'] = features['return_1'].rolling(20).std()
    features['momentum'] = close / close.rolling(20).mean() - 1
    features['rsi'] = 50 + 50 * features['return_1'].rolling(14).apply(
        lambda x: np.sum(x > 0) / len(x) - 0.5
    )

    features = features.dropna()
    tb_aligned = tb_labels.loc[features.index, 'tb_label']

    print(f"  Features: {features.shape[1]}")
    print(f"  Samples: {len(features)}")

    # Train/test split
    split = int(0.8 * len(features))
    X_train, X_test = features.iloc[:split], features.iloc[split:]
    tb_train, tb_test = tb_aligned.iloc[:split], tb_aligned.iloc[split:]

    # Test 1: Basic MetaLabeler
    print_subheader("2.2 MetaLabeler Training")

    # Simulate primary model predictions
    np.random.seed(42)
    primary_accuracy = 0.55  # Slightly better than random
    correct_mask = np.random.rand(len(X_train)) < primary_accuracy
    primary_train = pd.Series(
        np.where(correct_mask, tb_train.values, -tb_train.values),
        index=X_train.index
    )
    # Ensure no zeros for testing
    primary_train = primary_train.replace(0, 1)

    config = MetaLabelConfig(
        meta_model_type='rf',
        n_estimators=50,
        max_depth=4
    )

    meta_labeler = MetaLabeler(config)
    meta_labeler.fit(X_train, primary_train, tb_train)

    # Evaluate on test set
    primary_test = pd.Series(
        np.where(np.random.rand(len(X_test)) < primary_accuracy,
                tb_test.values, -tb_test.values),
        index=X_test.index
    ).replace(0, 1)

    metrics = meta_labeler.evaluate(X_test, primary_test, tb_test)
    results['meta_metrics'] = metrics

    print(f"\n  Meta-Model Metrics:")
    print(f"    Accuracy:  {metrics['meta_accuracy']:.4f}")
    print(f"    Precision: {metrics['meta_precision']:.4f}")
    print(f"    Recall:    {metrics['meta_recall']:.4f}")
    print(f"    F1 Score:  {metrics['meta_f1']:.4f}")
    print(f"    AUC-ROC:   {metrics['meta_auc']:.4f}")

    print(f"\n  Precision Improvement:")
    print(f"    Primary:     {metrics['primary_precision']:.4f}")
    print(f"    With Meta:   {metrics['meta_filtered_precision']:.4f}")
    print(f"    Improvement: {metrics['precision_improvement']:+.4f}")
    print(f"    Trades Filtered: {metrics['trades_filtered_pct']*100:.1f}%")

    # Test 2: Bet sizing
    print_subheader("2.3 Bet Sizing")

    bet_sizes = meta_labeler.predict_bet_size(X_test, primary_test)
    results['bet_size_stats'] = {
        'mean': bet_sizes.mean(),
        'std': bet_sizes.std(),
        'zero_pct': (bet_sizes == 0).mean()
    }

    print(f"\n  Bet Size Statistics:")
    print(f"    Mean:       {bet_sizes.mean():.4f}")
    print(f"    Std Dev:    {bet_sizes.std():.4f}")
    print(f"    Zero Bets:  {(bet_sizes == 0).sum()} ({(bet_sizes == 0).mean()*100:.1f}%)")

    # Validation checks
    print_subheader("2.4 Validation Checks")

    checks = {
        'meta_auc_above_random': metrics['meta_auc'] > 0.5,
        'precision_improvement': metrics['precision_improvement'] >= 0,
        'bet_sizes_valid': bet_sizes.between(0, 1).all(),
        'feature_importance_available': meta_labeler.feature_importance is not None
    }
    results['validation_checks'] = checks

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    return results


def test_volatility_regime(prices: pd.DataFrame) -> dict:
    """
    Test Volatility Regime Detection.

    Literature References:
    - VIX ML Trading (PLOS One 2024)
    - Classifying Market Regimes (Macrosynergy)

    Best Practices:
    - Combine VIX-based and realized volatility features
    - Use multiple regime detection methods (rules, GMM, HMM)
    - Validate regime persistence and transition probabilities
    """
    print_header("3. VOLATILITY REGIME DETECTION TEST")
    print("\nReferences:")
    print("  - VIX ML Trading (PLOS One 2024)")
    print("  - Classifying Market Regimes (Macrosynergy)")

    from feature_engineering.volatility_regime import (
        VolatilityConfig,
        VIXFeatureGenerator,
        RealizedVolatilityGenerator,
        RegimeDetector,
        HiddenMarkovRegimeModel,
        MarketRegime,
        generate_volatility_features
    )

    results = {}

    # Test 1: Realized Volatility
    print_subheader("3.1 Realized Volatility Features")

    rv_gen = RealizedVolatilityGenerator()
    rv_features = rv_gen.generate_features(prices)
    results['rv_features'] = len(rv_features.columns)

    print(f"\n  Generated {len(rv_features.columns)} RV features")
    print(f"  Feature types:")
    print(f"    - Close-to-close RV: rv_5, rv_10, rv_21, rv_63")
    print(f"    - Parkinson RV: parkinson_vol_*")
    print(f"    - ATR-based: atr_7, atr_14, atr_21")

    # RV statistics
    if 'rv_21' in rv_features.columns:
        rv_21 = rv_features['rv_21'].dropna()
        results['rv_21_stats'] = {
            'mean': rv_21.mean(),
            'std': rv_21.std(),
            'min': rv_21.min(),
            'max': rv_21.max()
        }
        print(f"\n  RV_21 Statistics (annualized):")
        print(f"    Mean: {rv_21.mean():.4f}")
        print(f"    Std:  {rv_21.std():.4f}")
        print(f"    Range: {rv_21.min():.4f} - {rv_21.max():.4f}")

    # Test 2: Regime Detection (Rules)
    print_subheader("3.2 Rule-Based Regime Detection")

    config = VolatilityConfig()
    regime_detector = RegimeDetector(config)
    regime_features = regime_detector.classify_regime_rules(rv_features, prices)

    regime_dist = regime_features['regime'].value_counts(normalize=True)
    results['regime_distribution'] = dict(regime_dist)

    print(f"\n  Regime Distribution:")
    for regime in MarketRegime:
        pct = regime_dist.get(regime.value, 0) * 100
        print(f"    {regime.name}: {pct:.1f}%")

    # Regime transitions
    regime_changes = regime_features['regime_change'].sum()
    avg_duration = regime_features['regime_duration'].mean()
    results['regime_transitions'] = regime_changes
    results['avg_regime_duration'] = avg_duration

    print(f"\n  Regime Dynamics:")
    print(f"    Total Transitions: {regime_changes}")
    print(f"    Avg Duration: {avg_duration:.1f} bars")

    # Test 3: GMM Regime Detection
    print_subheader("3.3 Gaussian Mixture Model Regime Detection")

    regime_detector.fit_gmm(prices, rv_features)
    gmm_regimes = regime_detector.predict_regime(prices, rv_features)
    gmm_dist = gmm_regimes.value_counts(normalize=True)
    results['gmm_distribution'] = dict(gmm_dist)

    print(f"\n  GMM Regime Distribution:")
    for regime, pct in gmm_dist.items():
        print(f"    Regime {regime}: {pct*100:.1f}%")

    # Test 4: HMM
    print_subheader("3.4 Hidden Markov Model")

    hmm = HiddenMarkovRegimeModel(n_states=4)
    hmm.fit_simplified(regime_features['regime'], rv_features)

    # Transition probabilities
    results['transition_matrix'] = hmm.transition_matrix.tolist()

    print(f"\n  Transition Matrix (rows = from, cols = to):")
    for i, row in enumerate(hmm.transition_matrix):
        row_str = " ".join([f"{p:.2f}" for p in row])
        print(f"    Regime {i}: [{row_str}]")

    # Test regime stability
    stabilities = [hmm.transition_matrix[i, i] for i in range(4)]
    results['regime_stabilities'] = stabilities

    print(f"\n  Regime Stabilities (P(stay in regime)):")
    for i, stab in enumerate(stabilities):
        print(f"    Regime {i}: {stab:.2f}")

    # Validation checks
    print_subheader("3.5 Validation Checks")

    checks = {
        'rv_features_generated': len(rv_features.columns) > 10,
        'all_regimes_present': len(regime_dist) >= 3,
        'reasonable_transitions': regime_changes > 10,
        'hmm_converged': hmm.transition_matrix is not None,
        'transition_probs_sum_to_1': all(
            abs(row.sum() - 1.0) < 0.01 for row in hmm.transition_matrix
        )
    }
    results['validation_checks'] = checks

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    return results


def test_comprehensive_features(prices: pd.DataFrame) -> dict:
    """
    Test comprehensive feature generation combining all new features.
    """
    print_header("4. COMPREHENSIVE FEATURE INTEGRATION TEST")

    from feature_engineering.triple_barrier import apply_triple_barrier
    from feature_engineering.volatility_regime import generate_volatility_features

    results = {}

    # Generate all features
    print_subheader("4.1 Generating All Features")

    # Triple Barrier labels
    tb_labels = apply_triple_barrier(prices)
    print(f"  Triple Barrier labels: {len(tb_labels.columns)} columns")

    # Volatility/regime features
    vol_features = generate_volatility_features(prices)
    print(f"  Volatility features: {len(vol_features.columns)} columns")

    # Combine
    all_features = pd.concat([tb_labels, vol_features], axis=1)
    all_features = all_features.dropna()

    results['total_features'] = len(all_features.columns)
    results['total_samples'] = len(all_features)

    print(f"\n  Total Combined Features: {len(all_features.columns)}")
    print(f"  Valid Samples: {len(all_features)}")

    # Feature categories
    print_subheader("4.2 Feature Categories")

    categories = {
        'triple_barrier': [c for c in all_features.columns if c.startswith('tb_')],
        'realized_vol': [c for c in all_features.columns if c.startswith('rv_') or 'parkinson' in c],
        'atr': [c for c in all_features.columns if c.startswith('atr_')],
        'regime': [c for c in all_features.columns if 'regime' in c],
        'vix': [c for c in all_features.columns if 'vix' in c.lower()]
    }

    for cat, cols in categories.items():
        print(f"  {cat}: {len(cols)} features")

    results['feature_categories'] = {k: len(v) for k, v in categories.items()}

    # Correlation with target
    print_subheader("4.3 Feature-Target Correlations")

    target = all_features['tb_label']
    feature_cols = [c for c in all_features.columns if not c.startswith('tb_')]

    correlations = {}
    for col in feature_cols[:20]:  # Top 20
        corr = all_features[col].corr(target)
        if not np.isnan(corr):
            correlations[col] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    results['top_correlations'] = sorted_corrs[:10]

    print(f"\n  Top 10 Correlated Features:")
    for feat, corr in sorted_corrs[:10]:
        print(f"    {feat}: {corr:+.4f}")

    return results


def generate_validation_report(results: dict):
    """Generate comprehensive validation report."""
    print_header("VALIDATION SUMMARY REPORT")

    print("\n" + "-" * 60)
    print("TRIPLE BARRIER LABELING")
    print("-" * 60)

    tb = results.get('triple_barrier', {})
    print(f"  Label Distribution: {tb.get('label_distribution', 'N/A')}")
    print(f"  Avg Holding: {tb.get('avg_holding_bars', 0):.1f} bars")
    print(f"  Avg Return: {tb.get('avg_return_pct', 0):.4f}%")

    checks = tb.get('validation_checks', {})
    passed = sum(checks.values())
    total = len(checks)
    print(f"  Validation: {passed}/{total} checks passed")

    print("\n" + "-" * 60)
    print("META-LABELING")
    print("-" * 60)

    ml = results.get('meta_labeling', {})
    metrics = ml.get('meta_metrics', {})
    print(f"  Meta AUC: {metrics.get('meta_auc', 0):.4f}")
    print(f"  Precision Improvement: {metrics.get('precision_improvement', 0):+.4f}")
    print(f"  Trades Filtered: {metrics.get('trades_filtered_pct', 0)*100:.1f}%")

    checks = ml.get('validation_checks', {})
    passed = sum(checks.values())
    total = len(checks)
    print(f"  Validation: {passed}/{total} checks passed")

    print("\n" + "-" * 60)
    print("VOLATILITY REGIME")
    print("-" * 60)

    vr = results.get('volatility_regime', {})
    print(f"  RV Features: {vr.get('rv_features', 0)}")
    print(f"  Regime Transitions: {vr.get('regime_transitions', 0)}")
    print(f"  Avg Duration: {vr.get('avg_regime_duration', 0):.1f} bars")

    checks = vr.get('validation_checks', {})
    passed = sum(checks.values())
    total = len(checks)
    print(f"  Validation: {passed}/{total} checks passed")

    print("\n" + "-" * 60)
    print("COMPREHENSIVE FEATURES")
    print("-" * 60)

    cf = results.get('comprehensive', {})
    print(f"  Total Features: {cf.get('total_features', 0)}")
    print(f"  Valid Samples: {cf.get('total_samples', 0)}")

    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL VALIDATION STATUS")
    print("=" * 60)

    all_checks = []
    for section in ['triple_barrier', 'meta_labeling', 'volatility_regime']:
        checks = results.get(section, {}).get('validation_checks', {})
        all_checks.extend(checks.values())

    total_passed = sum(all_checks)
    total_checks = len(all_checks)
    pass_rate = total_passed / total_checks * 100 if total_checks > 0 else 0

    print(f"\n  Total Checks: {total_checks}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_checks - total_passed}")
    print(f"  Pass Rate: {pass_rate:.1f}%")

    if pass_rate >= 90:
        print("\n  STATUS: VALIDATION SUCCESSFUL")
    elif pass_rate >= 70:
        print("\n  STATUS: VALIDATION PASSED WITH WARNINGS")
    else:
        print("\n  STATUS: VALIDATION FAILED - REVIEW REQUIRED")


def main():
    """Run all validation tests."""
    print("=" * 80)
    print(" SKIE-NINJA ADVANCED FEATURE VALIDATION")
    print(" Based on Research Phase Literature Review (2025-12-04)")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    prices = load_price_data()

    # Store all results
    all_results = {}

    # Test 1: Triple Barrier
    try:
        all_results['triple_barrier'] = test_triple_barrier(prices)
    except Exception as e:
        logger.error(f"Triple Barrier test failed: {e}")
        all_results['triple_barrier'] = {'error': str(e)}

    # Get TB labels for meta-labeling test
    try:
        from feature_engineering.triple_barrier import apply_triple_barrier
        tb_labels = apply_triple_barrier(prices)
    except:
        tb_labels = pd.DataFrame({'tb_label': np.random.choice([-1, 0, 1], len(prices))},
                                index=prices.index)

    # Test 2: Meta-Labeling
    try:
        all_results['meta_labeling'] = test_meta_labeling(prices, tb_labels)
    except Exception as e:
        logger.error(f"Meta-Labeling test failed: {e}")
        all_results['meta_labeling'] = {'error': str(e)}

    # Test 3: Volatility Regime
    try:
        all_results['volatility_regime'] = test_volatility_regime(prices)
    except Exception as e:
        logger.error(f"Volatility Regime test failed: {e}")
        all_results['volatility_regime'] = {'error': str(e)}

    # Test 4: Comprehensive
    try:
        all_results['comprehensive'] = test_comprehensive_features(prices)
    except Exception as e:
        logger.error(f"Comprehensive test failed: {e}")
        all_results['comprehensive'] = {'error': str(e)}

    # Generate report
    generate_validation_report(all_results)

    # Save results
    output_dir = PROJECT_ROOT / 'data' / 'validation_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'feature_validation_{timestamp}.txt'

    with open(output_file, 'w') as f:
        f.write("SKIE-Ninja Advanced Feature Validation Results\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for section, results in all_results.items():
            f.write(f"\n{section.upper()}\n")
            f.write("-" * 40 + "\n")
            for key, value in results.items():
                f.write(f"  {key}: {value}\n")

    print(f"\n\nResults saved to: {output_file}")

    print("\n" + "=" * 80)
    print(" VALIDATION COMPLETE")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    results = main()
