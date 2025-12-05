"""
Enhanced Feature QC Validation Script
=====================================

Comprehensive quality control checks for all feature modules:
1. Data leakage detection (look-ahead bias)
2. Feature-target correlation analysis
3. Temporal leakage testing
4. Feature quality metrics

Based on BEST_PRACTICES.md guidelines.

Usage:
    python src/python/run_enhanced_feature_qc.py

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings

warnings.filterwarnings('ignore')

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'python'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedFeatureQC:
    """
    Quality control for enhanced feature pipeline.

    Performs comprehensive checks to detect data leakage and
    ensure features are suitable for trading strategy use.
    """

    # Thresholds from BEST_PRACTICES.md
    MAX_FEATURE_TARGET_CORR = 0.30
    SUSPICIOUS_AUC = 0.85
    SUSPICIOUS_WIN_RATE = 0.65
    SUSPICIOUS_SHARPE = 4.0
    SUSPICIOUS_PROFIT_FACTOR = 2.0

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or PROJECT_ROOT / 'data' / 'validation_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def check_code_patterns(self, source_dir: Path) -> Dict[str, Any]:
        """
        Check source code for dangerous patterns.

        Looks for:
        - shift(-N) patterns (look-ahead)
        - center=True in rolling operations
        - Other leaky patterns
        """
        logger.info("Checking source code for dangerous patterns...")

        results = {
            'files_checked': 0,
            'shift_negative': [],
            'center_true': [],
            'other_issues': []
        }

        py_files = list(source_dir.rglob('*.py'))

        for filepath in py_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                results['files_checked'] += 1

                for i, line in enumerate(lines, 1):
                    # Check for shift(-N) pattern
                    if 'shift(-' in line and '#' not in line.split('shift(-')[0]:
                        results['shift_negative'].append({
                            'file': str(filepath.relative_to(PROJECT_ROOT)),
                            'line': i,
                            'code': line.strip()
                        })

                    # Check for center=True
                    if 'center=True' in line and '#' not in line.split('center=True')[0]:
                        results['center_true'].append({
                            'file': str(filepath.relative_to(PROJECT_ROOT)),
                            'line': i,
                            'code': line.strip()
                        })

            except Exception as e:
                logger.warning(f"Could not read {filepath}: {e}")

        return results

    def check_feature_target_correlation(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> Dict[str, Any]:
        """
        Check correlation between features and target.

        High correlation (>0.30) may indicate data leakage.
        """
        logger.info("Checking feature-target correlations...")

        results = {
            'total_features': len(features.columns),
            'suspicious_features': [],
            'correlation_stats': {}
        }

        correlations = {}
        for col in features.columns:
            if features[col].isna().all():
                continue

            # Align indices
            valid_mask = features[col].notna() & target.notna()
            if valid_mask.sum() < 100:
                continue

            corr = features.loc[valid_mask, col].corr(target.loc[valid_mask])
            correlations[col] = corr

            if abs(corr) > self.MAX_FEATURE_TARGET_CORR:
                results['suspicious_features'].append({
                    'feature': col,
                    'correlation': corr
                })

        if correlations:
            corr_values = list(correlations.values())
            results['correlation_stats'] = {
                'mean': np.nanmean(corr_values),
                'max': np.nanmax(np.abs(corr_values)),
                'std': np.nanstd(corr_values)
            }

        return results

    def temporal_leakage_test(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> Dict[str, Any]:
        """
        Test for temporal leakage by training on future, predicting past.

        If we can predict the past from the future better than random,
        there's likely data leakage.
        """
        logger.info("Running temporal leakage test...")

        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.metrics import roc_auc_score
        except ImportError:
            logger.warning("sklearn not available for temporal leakage test")
            return {'error': 'sklearn not available'}

        results = {
            'test_performed': True,
            'forward_auc': None,
            'reverse_auc': None,
            'leakage_detected': False
        }

        # Prepare data
        valid_mask = features.notna().all(axis=1) & target.notna()
        X = features.loc[valid_mask].values
        y = (target.loc[valid_mask] > 0).astype(int).values

        if len(X) < 1000:
            results['error'] = 'Insufficient data for temporal test'
            return results

        # Split point
        split = len(X) // 2

        # Forward test (normal: train on past, predict future)
        X_train_fwd, X_test_fwd = X[:split], X[split:]
        y_train_fwd, y_test_fwd = y[:split], y[split:]

        # Reverse test (train on future, predict past)
        X_train_rev, X_test_rev = X[split:], X[:split]
        y_train_rev, y_test_rev = y[split:], y[:split]

        try:
            # Forward model
            model_fwd = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, random_state=42
            )
            model_fwd.fit(X_train_fwd, y_train_fwd)
            pred_fwd = model_fwd.predict_proba(X_test_fwd)[:, 1]
            results['forward_auc'] = roc_auc_score(y_test_fwd, pred_fwd)

            # Reverse model
            model_rev = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, random_state=42
            )
            model_rev.fit(X_train_rev, y_train_rev)
            pred_rev = model_rev.predict_proba(X_test_rev)[:, 1]
            results['reverse_auc'] = roc_auc_score(y_test_rev, pred_rev)

            # If reverse AUC > 0.55, there may be leakage
            if results['reverse_auc'] > 0.55:
                results['leakage_detected'] = True
                results['warning'] = f"Reverse AUC {results['reverse_auc']:.3f} > 0.55 suggests leakage"

        except Exception as e:
            results['error'] = str(e)

        return results

    def check_feature_quality(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Check overall feature quality metrics.
        """
        logger.info("Checking feature quality metrics...")

        results = {
            'total_features': len(features.columns),
            'nan_analysis': {},
            'constant_features': [],
            'infinite_values': 0,
            'highly_correlated_pairs': []
        }

        # NaN analysis
        nan_pct = features.isna().sum() / len(features) * 100
        results['nan_analysis'] = {
            'features_with_any_nan': (nan_pct > 0).sum(),
            'features_over_10pct_nan': (nan_pct > 10).sum(),
            'features_over_50pct_nan': (nan_pct > 50).sum(),
            'worst_features': nan_pct.nlargest(5).to_dict()
        }

        # Constant features
        for col in features.columns:
            if features[col].nunique() <= 1:
                results['constant_features'].append(col)

        # Infinite values
        numeric_df = features.select_dtypes(include=[np.number])
        results['infinite_values'] = np.isinf(numeric_df).sum().sum()

        # Highly correlated feature pairs (sample to reduce computation)
        sample_cols = list(features.columns)[:100]  # Limit for speed
        sample_df = features[sample_cols].dropna()

        if len(sample_df) > 100:
            corr_matrix = sample_df.corr()
            for i in range(len(sample_cols)):
                for j in range(i+1, len(sample_cols)):
                    corr = abs(corr_matrix.iloc[i, j])
                    if corr > 0.95:
                        results['highly_correlated_pairs'].append({
                            'feature1': sample_cols[i],
                            'feature2': sample_cols[j],
                            'correlation': corr
                        })

        return results

    def run_full_qc(
        self,
        features: pd.DataFrame,
        prices: pd.DataFrame,
        check_source_code: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete QC suite.

        Args:
            features: Feature DataFrame
            prices: Price DataFrame (for target calculation)
            check_source_code: Whether to check source code patterns

        Returns:
            Complete QC results
        """
        logger.info("=" * 70)
        logger.info("ENHANCED FEATURE QC - FULL VALIDATION SUITE")
        logger.info("=" * 70)

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'passed': True,
            'checks': {}
        }

        # 1. Source code pattern check
        if check_source_code:
            logger.info("\n[1/4] Source Code Pattern Check")
            code_results = self.check_code_patterns(
                PROJECT_ROOT / 'src' / 'python' / 'feature_engineering'
            )
            all_results['checks']['code_patterns'] = code_results

            if code_results['shift_negative']:
                all_results['passed'] = False
                logger.error(f"  FAIL: Found {len(code_results['shift_negative'])} shift(-N) patterns")
                for issue in code_results['shift_negative'][:3]:
                    logger.error(f"    {issue['file']}:{issue['line']}")
            else:
                logger.info("  PASS: No shift(-N) patterns found")

            if code_results['center_true']:
                all_results['passed'] = False
                logger.error(f"  FAIL: Found {len(code_results['center_true'])} center=True patterns")
            else:
                logger.info("  PASS: No center=True patterns found")

        # 2. Feature quality check
        logger.info("\n[2/4] Feature Quality Check")
        quality_results = self.check_feature_quality(features)
        all_results['checks']['feature_quality'] = quality_results

        if quality_results['constant_features']:
            logger.warning(f"  WARN: {len(quality_results['constant_features'])} constant features")
        else:
            logger.info("  PASS: No constant features")

        if quality_results['infinite_values'] > 0:
            logger.warning(f"  WARN: {quality_results['infinite_values']} infinite values")
        else:
            logger.info("  PASS: No infinite values")

        logger.info(f"  INFO: {quality_results['nan_analysis']['features_over_50pct_nan']} features with >50% NaN")

        # 3. Feature-target correlation check
        logger.info("\n[3/4] Feature-Target Correlation Check")

        # Create target (future returns for testing)
        target = prices['close'].pct_change(10).shift(-10)  # 10-bar future return

        corr_results = self.check_feature_target_correlation(features, target)
        all_results['checks']['correlation'] = corr_results

        if corr_results['suspicious_features']:
            all_results['passed'] = False
            logger.error(f"  FAIL: {len(corr_results['suspicious_features'])} features with correlation > {self.MAX_FEATURE_TARGET_CORR}")
            for sf in corr_results['suspicious_features'][:5]:
                logger.error(f"    {sf['feature']}: {sf['correlation']:.3f}")
        else:
            logger.info(f"  PASS: No features exceed correlation threshold ({self.MAX_FEATURE_TARGET_CORR})")

        # 4. Temporal leakage test
        logger.info("\n[4/4] Temporal Leakage Test")
        temporal_results = self.temporal_leakage_test(features, target)
        all_results['checks']['temporal_leakage'] = temporal_results

        if temporal_results.get('leakage_detected'):
            all_results['passed'] = False
            logger.error(f"  FAIL: Temporal leakage detected (reverse AUC: {temporal_results['reverse_auc']:.3f})")
        elif temporal_results.get('error'):
            logger.warning(f"  WARN: Could not complete test - {temporal_results['error']}")
        else:
            logger.info(f"  PASS: Forward AUC: {temporal_results.get('forward_auc', 'N/A'):.3f}, "
                       f"Reverse AUC: {temporal_results.get('reverse_auc', 'N/A'):.3f}")

        # Summary
        logger.info("\n" + "=" * 70)
        if all_results['passed']:
            logger.info("QC RESULT: PASSED - All checks passed")
        else:
            logger.error("QC RESULT: FAILED - Issues detected (see above)")
        logger.info("=" * 70)

        self.results = all_results
        return all_results

    def save_results(self, filename: str = None):
        """Save QC results to file."""
        if filename is None:
            filename = f"enhanced_qc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            f.write("ENHANCED FEATURE QC RESULTS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {self.results.get('timestamp', 'N/A')}\n")
            f.write(f"Overall Result: {'PASSED' if self.results.get('passed') else 'FAILED'}\n\n")

            for check_name, check_results in self.results.get('checks', {}).items():
                f.write(f"\n{check_name.upper()}\n")
                f.write("-" * 40 + "\n")

                if isinstance(check_results, dict):
                    for key, value in check_results.items():
                        if isinstance(value, (list, dict)) and len(str(value)) > 100:
                            f.write(f"  {key}: [truncated - {len(value) if isinstance(value, list) else 'complex'}]\n")
                        else:
                            f.write(f"  {key}: {value}\n")

        logger.info(f"Results saved to {filepath}")
        return filepath


def main():
    """Main entry point for QC validation."""
    print("=" * 70)
    print("ENHANCED FEATURE QC VALIDATION")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    try:
        from data_collection.ninjatrader_loader import load_sample_data
        prices, _ = load_sample_data(source="databento")

        # Resample to 5-min
        prices = prices.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Take subset for faster testing
        prices = prices.iloc[:10000]
        print(f"Loaded {len(prices)} bars")

    except Exception as e:
        print(f"Could not load data: {e}")
        print("Creating synthetic data...")

        dates = pd.date_range('2024-01-01 09:30', periods=10000, freq='5min')
        np.random.seed(42)
        close = 4500 + np.cumsum(np.random.randn(10000) * 2)

        prices = pd.DataFrame({
            'open': close + np.random.randn(10000),
            'high': close + abs(np.random.randn(10000)) * 3,
            'low': close - abs(np.random.randn(10000)) * 3,
            'close': close,
            'volume': np.random.randint(1000, 10000, 10000)
        }, index=dates)

    # Generate features
    print("\n[2] Generating features...")
    try:
        from feature_engineering.enhanced_feature_pipeline import (
            generate_enhanced_features, EnhancedPipelineConfig
        )

        config = EnhancedPipelineConfig(
            include_technical=True,
            include_mtf=True,
            include_cross_market=True,
            include_sentiment=True,
            include_volatility_regime=False
        )

        features, _ = generate_enhanced_features(prices, config, validate=False)
        print(f"Generated {len(features.columns)} features")

    except Exception as e:
        print(f"Error generating features: {e}")
        print("Using basic features for testing...")

        from feature_engineering.technical_indicators import calculate_technical_indicators
        features = calculate_technical_indicators(prices)
        print(f"Generated {len(features.columns)} basic features")

    # Run QC
    print("\n[3] Running QC validation...")
    qc = EnhancedFeatureQC()
    results = qc.run_full_qc(features, prices)

    # Save results
    print("\n[4] Saving results...")
    qc.save_results()

    return results


if __name__ == "__main__":
    main()
