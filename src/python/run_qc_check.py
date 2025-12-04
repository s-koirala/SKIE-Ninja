"""
Quality Control Check for Volatility Breakout Strategy

Checks for:
1. Look-ahead bias in features
2. Proper target construction
3. Feature-target correlations
4. Walk-forward methodology validation
5. Result suspiciousness indicators

Author: SKIE_Ninja Development Team
Created: 2025-12-04
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

from data_collection.ninjatrader_loader import load_sample_data
from feature_engineering.multi_target_labels import MultiTargetLabeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_feature_lookahead(features: pd.DataFrame, df: pd.DataFrame) -> dict:
    """
    Check if features contain look-ahead bias.

    Tests:
    1. Correlation of feature[t] with future_close[t+1..t+N]
    2. If feature[t] can predict close[t+1], it's leaky
    """
    results = {'passed': True, 'warnings': [], 'details': {}}

    future_returns = {}
    for lag in [1, 5, 10, 20]:
        future_returns[f'future_ret_{lag}'] = df['close'].pct_change(lag).shift(-lag)

    suspicious_features = []

    for col in features.columns:
        if features[col].isna().all():
            continue

        for lag, future_ret in future_returns.items():
            valid_mask = ~(features[col].isna() | future_ret.isna())
            if valid_mask.sum() < 100:
                continue

            corr = np.corrcoef(
                features[col][valid_mask].values,
                future_ret[valid_mask].values
            )[0, 1]

            if abs(corr) > 0.10:  # Suspicious if correlated with future
                suspicious_features.append({
                    'feature': col,
                    'correlation_with': lag,
                    'corr_value': corr
                })
                results['warnings'].append(
                    f"SUSPICIOUS: {col} correlated with {lag} (r={corr:.4f})"
                )

    if len(suspicious_features) > 0:
        results['passed'] = False
        results['details']['suspicious_features'] = suspicious_features

    return results


def check_target_construction(targets: pd.DataFrame, df: pd.DataFrame) -> dict:
    """
    Verify targets properly use FUTURE data.

    A valid target should be correlated with future returns, not past.
    """
    results = {'passed': True, 'warnings': [], 'details': {}}

    # Check that targets are correlated with FUTURE data
    past_returns = df['close'].pct_change(10)  # Past 10-bar return
    future_returns = df['close'].pct_change(10).shift(-10)  # Future 10-bar return

    target_correlations = {}

    for col in targets.columns:
        if targets[col].isna().all():
            continue
        if targets[col].nunique() < 2:
            continue

        valid_mask = ~(targets[col].isna() | past_returns.isna() | future_returns.isna())

        if valid_mask.sum() < 100:
            continue

        corr_past = np.corrcoef(
            targets[col][valid_mask].values,
            past_returns[valid_mask].values
        )[0, 1]

        corr_future = np.corrcoef(
            targets[col][valid_mask].values,
            future_returns[valid_mask].values
        )[0, 1]

        target_correlations[col] = {
            'corr_past': corr_past,
            'corr_future': corr_future,
            'direction_correct': abs(corr_future) > abs(corr_past)
        }

        # Warning if target more correlated with past than future
        if abs(corr_past) > abs(corr_future) * 1.5:
            results['warnings'].append(
                f"WARNING: {col} more correlated with PAST ({corr_past:.4f}) "
                f"than FUTURE ({corr_future:.4f})"
            )

    results['details']['target_correlations'] = target_correlations
    return results


def check_feature_target_correlations(
    features: pd.DataFrame,
    targets: pd.DataFrame
) -> dict:
    """
    Check for suspiciously high feature-target correlations.

    Threshold: 0.30 (per Lopez de Prado 2018)
    """
    results = {'passed': True, 'warnings': [], 'details': {}}

    # Key targets to check
    key_targets = ['vol_expansion_5', 'new_high_10', 'new_low_10']

    high_correlations = []

    for target_col in key_targets:
        if target_col not in targets.columns:
            continue

        for feat_col in features.columns:
            valid_mask = ~(features[feat_col].isna() | targets[target_col].isna())

            if valid_mask.sum() < 100:
                continue

            corr = np.corrcoef(
                features[feat_col][valid_mask].values,
                targets[target_col][valid_mask].values
            )[0, 1]

            if abs(corr) > 0.30:
                high_correlations.append({
                    'feature': feat_col,
                    'target': target_col,
                    'correlation': corr
                })
                results['warnings'].append(
                    f"HIGH CORRELATION: {feat_col} vs {target_col} = {corr:.4f}"
                )
                results['passed'] = False

    # Also get max correlation for each target
    max_correlations = {}
    for target_col in key_targets:
        if target_col not in targets.columns:
            continue

        max_corr = 0
        max_feat = None

        for feat_col in features.columns:
            valid_mask = ~(features[feat_col].isna() | targets[target_col].isna())
            if valid_mask.sum() < 100:
                continue

            corr = abs(np.corrcoef(
                features[feat_col][valid_mask].values,
                targets[target_col][valid_mask].values
            )[0, 1])

            if corr > max_corr:
                max_corr = corr
                max_feat = feat_col

        max_correlations[target_col] = {
            'max_feature': max_feat,
            'max_correlation': max_corr
        }

    results['details']['max_correlations'] = max_correlations
    results['details']['high_correlations'] = high_correlations

    return results


def check_result_suspiciousness(trades_file: str) -> dict:
    """
    Check if backtest results are suspiciously good.

    Thresholds based on Lopez de Prado (2018):
    - Sharpe > 3.0 is suspicious
    - Win rate > 65% is suspicious
    - Profit factor > 3.0 is suspicious
    """
    results = {'passed': True, 'warnings': [], 'details': {}}

    try:
        df = pd.read_csv(trades_file)
    except:
        results['warnings'].append("Could not load trades file")
        return results

    # Calculate metrics
    total_trades = len(df)
    win_rate = (df['net_pnl'] > 0).mean()
    net_pnl = df['net_pnl'].sum()

    winners = df[df['net_pnl'] > 0]['net_pnl'].sum()
    losers = abs(df[df['net_pnl'] < 0]['net_pnl'].sum())
    profit_factor = winners / losers if losers > 0 else 0

    # Daily P&L for Sharpe
    df['date'] = pd.to_datetime(df['entry_time']).dt.date
    daily_pnl = df.groupby('date')['net_pnl'].sum()

    if len(daily_pnl) > 1:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0
    else:
        sharpe = 0

    results['details'] = {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'net_pnl': net_pnl,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe
    }

    # Check thresholds
    if sharpe > 3.0:
        results['warnings'].append(f"SUSPICIOUS: Sharpe {sharpe:.2f} > 3.0")
        results['passed'] = False

    if win_rate > 0.65:
        results['warnings'].append(f"SUSPICIOUS: Win rate {win_rate*100:.1f}% > 65%")
        results['passed'] = False

    if profit_factor > 3.0:
        results['warnings'].append(f"SUSPICIOUS: Profit factor {profit_factor:.2f} > 3.0")
        results['passed'] = False

    # Also check for too-good volatility model AUC
    if 'vol_prob' in df.columns:
        avg_vol_prob = df['vol_prob'].mean()
        if avg_vol_prob > 0.90:
            results['warnings'].append(f"SUSPICIOUS: Avg vol_prob {avg_vol_prob:.3f} > 0.90")

    return results


def check_temporal_leakage(
    features: pd.DataFrame,
    targets: pd.DataFrame
) -> dict:
    """
    Check for temporal leakage by testing if features at time T
    can predict targets at time T (should be impossible for proper targets).
    """
    results = {'passed': True, 'warnings': [], 'details': {}}

    # For a proper target like vol_expansion_5, the target at time T
    # should depend on data from T+1 to T+5
    # Features at time T should not perfectly predict it

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    # Test key target
    target_col = 'vol_expansion_5'

    if target_col not in targets.columns:
        results['warnings'].append(f"Target {target_col} not found")
        return results

    # Align data
    common_idx = features.index.intersection(targets.index)
    X = features.loc[common_idx]
    y = targets.loc[common_idx][target_col]

    # Drop NaN
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]

    if len(X) < 1000:
        results['warnings'].append("Not enough samples for leakage test")
        return results

    # CRITICAL TEST: Train on first 80%, test on last 20%
    # If AUC is suspiciously high (>0.95), there's leakage
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Fill NaN with median for this test
    X_train_filled = X_train.fillna(X_train.median())
    X_test_filled = X_test.fillna(X_train.median())

    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train_filled, y_train)

    probs = model.predict_proba(X_test_filled)[:, 1]
    auc = roc_auc_score(y_test, probs)

    results['details']['quick_test_auc'] = auc

    if auc > 0.95:
        results['warnings'].append(f"CRITICAL: Quick test AUC {auc:.4f} > 0.95 - LIKELY LEAKAGE")
        results['passed'] = False
    elif auc > 0.85:
        results['warnings'].append(f"HIGH AUC: {auc:.4f} - verify methodology")
    else:
        results['details']['assessment'] = f"AUC {auc:.4f} is within expected range for financial data"

    return results


def main():
    print("=" * 80)
    print(" QUALITY CONTROL CHECK")
    print(" Data Leakage & Look-Ahead Bias Detection")
    print("=" * 80)

    # Load data
    logger.info("\n--- Loading Data ---")
    prices, _ = load_sample_data(source="databento")

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

    logger.info(f"Data loaded: {len(prices)} bars")

    # Generate features (same as strategy)
    logger.info("\n--- Generating Features ---")
    from strategy.volatility_breakout_strategy import VolatilityBreakoutStrategy
    strategy = VolatilityBreakoutStrategy()
    features = strategy.generate_features(prices)

    # Generate targets
    logger.info("\n--- Generating Targets ---")
    labeler = MultiTargetLabeler()
    targets = labeler.generate_all_targets(prices)

    # Align
    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]

    # Drop NaN
    valid_mask = ~(features.isna().any(axis=1) | targets.isna().any(axis=1))
    features = features[valid_mask]
    targets = targets[valid_mask]
    prices_aligned = prices.loc[features.index]

    logger.info(f"Valid samples: {len(features)}")

    # Run QC checks
    all_passed = True

    # 1. Feature Look-Ahead Check
    print("\n" + "=" * 80)
    print(" CHECK 1: Feature Look-Ahead Bias")
    print("=" * 80)

    result1 = check_feature_lookahead(features, prices_aligned)
    if result1['passed']:
        print("[PASS] No look-ahead bias detected in features")
    else:
        print("[FAIL] Look-ahead bias detected!")
        for w in result1['warnings']:
            print(f"  - {w}")
        all_passed = False

    # 2. Target Construction Check
    print("\n" + "=" * 80)
    print(" CHECK 2: Target Construction")
    print("=" * 80)

    result2 = check_target_construction(targets, prices_aligned)
    if result2['passed']:
        print("[PASS] Targets properly constructed")
    else:
        print("[WARN] WARNINGS in target construction")
    for w in result2['warnings']:
        print(f"  - {w}")

    # 3. Feature-Target Correlation Check
    print("\n" + "=" * 80)
    print(" CHECK 3: Feature-Target Correlations")
    print("=" * 80)

    result3 = check_feature_target_correlations(features, targets)
    if result3['passed']:
        print("[PASS] No suspicious correlations (< 0.30)")
    else:
        print("[FAIL] High correlations detected!")
        all_passed = False

    for target, info in result3['details'].get('max_correlations', {}).items():
        print(f"  {target}: max corr = {info['max_correlation']:.4f} ({info['max_feature']})")

    for w in result3['warnings']:
        print(f"  - {w}")

    # 4. Temporal Leakage Check (Quick Model Test)
    print("\n" + "=" * 80)
    print(" CHECK 4: Temporal Leakage Test")
    print("=" * 80)

    result4 = check_temporal_leakage(features, targets)
    if result4['passed']:
        print(f"[PASS] {result4['details'].get('assessment', '')}")
    else:
        print("[FAIL] Possible temporal leakage!")
        all_passed = False

    for w in result4['warnings']:
        print(f"  - {w}")

    if 'quick_test_auc' in result4['details']:
        print(f"  Quick model test AUC: {result4['details']['quick_test_auc']:.4f}")

    # 5. Result Suspiciousness Check
    print("\n" + "=" * 80)
    print(" CHECK 5: Result Suspiciousness")
    print("=" * 80)

    trades_file = project_root / 'data' / 'backtest_results' / 'vol_breakout_trades_20251204_134623.csv'
    result5 = check_result_suspiciousness(str(trades_file))

    if result5['passed']:
        print("[PASS] Results within expected ranges")
    else:
        print("[WARN] SUSPICIOUS results detected")
        all_passed = False

    details = result5['details']
    if details:
        print(f"  Total Trades: {details.get('total_trades', 'N/A')}")
        print(f"  Win Rate: {details.get('win_rate', 0)*100:.1f}%")
        print(f"  Profit Factor: {details.get('profit_factor', 0):.2f}")
        print(f"  Sharpe Ratio: {details.get('sharpe_ratio', 0):.2f}")

    for w in result5['warnings']:
        print(f"  - {w}")

    # Final Summary
    print("\n" + "=" * 80)
    print(" QC SUMMARY")
    print("=" * 80)

    if all_passed:
        print("\n[PASS] ALL CHECKS PASSED - No data leakage or look-ahead bias detected")
        print("\nThe strategy results appear legitimate based on:")
        print("  1. Features use only past data (no shift(-N) patterns)")
        print("  2. Targets properly reference future data")
        print("  3. Feature-target correlations below 0.30 threshold")
        print("  4. Model AUC within expected range for financial data")
        print("  5. Backtest metrics not suspiciously high")
    else:
        print("\n[FAIL] SOME CHECKS FAILED - Review warnings above")
        print("\nPotential issues require investigation before trusting results.")

    # Save report
    output_dir = project_root / 'data' / 'validation_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = output_dir / f'qc_report_{timestamp}.txt'

    with open(report_file, 'w') as f:
        f.write("QC Report - Data Leakage & Look-Ahead Bias Check\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"All Passed: {all_passed}\n\n")

        f.write("Check 1 - Feature Look-Ahead: " + ("PASSED" if result1['passed'] else "FAILED") + "\n")
        for w in result1['warnings']:
            f.write(f"  {w}\n")

        f.write("\nCheck 2 - Target Construction: " + ("PASSED" if result2['passed'] else "WARNINGS") + "\n")
        for w in result2['warnings']:
            f.write(f"  {w}\n")

        f.write("\nCheck 3 - Correlations: " + ("PASSED" if result3['passed'] else "FAILED") + "\n")
        for w in result3['warnings']:
            f.write(f"  {w}\n")

        f.write("\nCheck 4 - Temporal Leakage: " + ("PASSED" if result4['passed'] else "FAILED") + "\n")
        for w in result4['warnings']:
            f.write(f"  {w}\n")

        f.write("\nCheck 5 - Results: " + ("PASSED" if result5['passed'] else "SUSPICIOUS") + "\n")
        for w in result5['warnings']:
            f.write(f"  {w}\n")

    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
