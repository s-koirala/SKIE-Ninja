"""
SKIE-Ninja Validation and Quality Control Framework

Comprehensive validation for ML trading systems based on best practices:

References:
===========
1. de Prado, M. L. (2018). "Advances in Financial Machine Learning" - Wiley
   - Chapter 7: Cross-Validation in Finance
   - Chapter 8: Feature Importance

2. Bailey, D. H., et al. (2014). "The Probability of Backtest Overfitting"
   - Journal of Computational Finance
   - Methodology for detecting overfitted strategies

3. Harvey, C. R., et al. (2016). "... and the Cross-Section of Expected Returns"
   - Review of Financial Studies
   - Multiple testing correction (Bonferroni, FDR)

4. Aronson, D. R. (2006). "Evidence-Based Technical Analysis" - Wiley
   - Statistical validation of trading systems

Quality Control Checklist:
=========================
□ Data Quality
  - No missing values in critical columns
  - Timestamps are monotonically increasing
  - OHLC relationship: Low <= Open,Close <= High
  - Volume is non-negative

□ Feature Quality
  - No infinite values
  - No extreme outliers (>10 std)
  - No data leakage (correlation with target > 0.95)
  - Feature distributions are reasonable

□ Model Quality
  - Train/test split is temporal (no look-ahead bias)
  - Embargo period between train/test
  - Cross-validation is walk-forward
  - No features from future data

□ Backtest Quality
  - All trades within RTH
  - Realistic costs (commission, slippage)
  - Position sizing is reasonable
  - Drawdown limits are enforced

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, time
import json
import warnings

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ValidationConfig:
    """Configuration for validation checks."""
    # Data Quality
    max_missing_pct: float = 0.01  # Max 1% missing values
    max_duplicate_pct: float = 0.001  # Max 0.1% duplicates
    max_outlier_std: float = 10.0  # Flag values > 10 std

    # Feature Quality
    leakage_correlation_threshold: float = 0.95
    min_feature_variance: float = 1e-10
    max_feature_correlation: float = 0.99  # Between features

    # Model Quality
    min_train_samples: int = 10000
    min_test_samples: int = 1000
    min_embargo_bars: int = 20
    max_auc_suspicion: float = 0.95  # AUC > 95% is suspicious

    # Backtest Quality
    min_trades: int = 100
    min_win_rate: float = 0.30
    max_win_rate: float = 0.80  # Win rate > 80% is suspicious
    max_profit_factor: float = 5.0  # PF > 5 is suspicious
    min_sharpe: float = 0.0

    # RTH Parameters
    rth_start: time = time(9, 30)
    rth_end: time = time(16, 0)


# ============================================================================
# DATA VALIDATOR
# ============================================================================

class DataValidator:
    """Validate data quality."""

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.passed: bool = True

    def validate_ohlcv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate OHLCV data quality.

        Checks:
        1. Required columns exist
        2. No missing values
        3. OHLC relationship is valid
        4. Timestamps are monotonic
        5. Volume is non-negative
        """
        self.issues = []
        self.warnings = []
        results = {'passed': True, 'checks': {}}

        # Check required columns
        required = ['open', 'high', 'low', 'close']
        missing_cols = [c for c in required if c.lower() not in [col.lower() for col in df.columns]]
        if missing_cols:
            self.issues.append(f"Missing required columns: {missing_cols}")
            results['passed'] = False

        # Standardize column names
        df_check = df.copy()
        df_check.columns = df_check.columns.str.lower()

        # Check missing values
        missing_pct = df_check[['open', 'high', 'low', 'close']].isnull().mean()
        max_missing = missing_pct.max()
        results['checks']['missing_values'] = {
            'max_pct': float(max_missing),
            'passed': max_missing <= self.config.max_missing_pct
        }
        if max_missing > self.config.max_missing_pct:
            self.issues.append(f"Too many missing values: {max_missing:.2%}")
            results['passed'] = False

        # Check OHLC relationship
        if all(c in df_check.columns for c in ['open', 'high', 'low', 'close']):
            valid_ohlc = (
                (df_check['low'] <= df_check['open']) &
                (df_check['low'] <= df_check['close']) &
                (df_check['high'] >= df_check['open']) &
                (df_check['high'] >= df_check['close'])
            )
            invalid_pct = (~valid_ohlc).mean()
            results['checks']['ohlc_relationship'] = {
                'invalid_pct': float(invalid_pct),
                'passed': invalid_pct < 0.01
            }
            if invalid_pct >= 0.01:
                self.warnings.append(f"OHLC relationship invalid in {invalid_pct:.2%} of bars")

        # Check timestamp monotonicity
        if isinstance(df.index, pd.DatetimeIndex):
            is_monotonic = df.index.is_monotonic_increasing
            results['checks']['timestamp_monotonic'] = {
                'is_monotonic': is_monotonic,
                'passed': is_monotonic
            }
            if not is_monotonic:
                self.warnings.append("Timestamps are not monotonically increasing")

        # Check for duplicates
        if isinstance(df.index, pd.DatetimeIndex):
            dup_pct = df.index.duplicated().mean()
            results['checks']['duplicates'] = {
                'pct': float(dup_pct),
                'passed': dup_pct <= self.config.max_duplicate_pct
            }
            if dup_pct > self.config.max_duplicate_pct:
                self.warnings.append(f"Duplicate timestamps: {dup_pct:.4%}")

        # Check volume
        if 'volume' in df_check.columns:
            negative_vol = (df_check['volume'] < 0).mean()
            results['checks']['volume_positive'] = {
                'negative_pct': float(negative_vol),
                'passed': negative_vol == 0
            }
            if negative_vol > 0:
                self.warnings.append(f"Negative volume in {negative_vol:.2%} of bars")

        results['issues'] = self.issues
        results['warnings'] = self.warnings

        self.passed = results['passed']
        return results

    def validate_features(self, features: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Validate feature quality.

        Checks:
        1. No infinite values
        2. No extreme outliers
        3. No data leakage
        4. Reasonable variance
        """
        self.issues = []
        self.warnings = []
        results = {'passed': True, 'checks': {}}

        feature_cols = [c for c in features.columns if c != target_col and not c.startswith('target_')]

        # Check for infinite values
        inf_counts = {}
        for col in feature_cols:
            inf_count = np.isinf(features[col]).sum()
            if inf_count > 0:
                inf_counts[col] = int(inf_count)

        results['checks']['infinite_values'] = {
            'features_with_inf': inf_counts,
            'passed': len(inf_counts) == 0
        }
        if inf_counts:
            self.issues.append(f"Infinite values in {len(inf_counts)} features")
            results['passed'] = False

        # Check for extreme outliers
        outlier_features = []
        for col in feature_cols:
            if features[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                std = features[col].std()
                mean = features[col].mean()
                if std > 0:
                    max_zscore = abs((features[col] - mean) / std).max()
                    if max_zscore > self.config.max_outlier_std:
                        outlier_features.append((col, float(max_zscore)))

        results['checks']['extreme_outliers'] = {
            'features': outlier_features[:10],  # Top 10
            'passed': len(outlier_features) == 0
        }
        if outlier_features:
            self.warnings.append(f"Extreme outliers in {len(outlier_features)} features")

        # Check for data leakage
        leaky_features = []
        if target_col in features.columns:
            target = features[target_col].values
            for col in feature_cols:
                try:
                    corr = np.corrcoef(features[col].values, target)[0, 1]
                    if abs(corr) > self.config.leakage_correlation_threshold:
                        leaky_features.append((col, float(corr)))
                except Exception:
                    pass

        results['checks']['data_leakage'] = {
            'features': leaky_features,
            'passed': len(leaky_features) == 0
        }
        if leaky_features:
            self.issues.append(f"DATA LEAKAGE DETECTED in {len(leaky_features)} features")
            results['passed'] = False

        # Check for low variance
        low_var_features = []
        for col in feature_cols:
            var = features[col].var()
            if var < self.config.min_feature_variance:
                low_var_features.append(col)

        results['checks']['low_variance'] = {
            'features': low_var_features[:10],
            'passed': len(low_var_features) < len(feature_cols) * 0.1
        }
        if low_var_features:
            self.warnings.append(f"Low variance in {len(low_var_features)} features")

        results['issues'] = self.issues
        results['warnings'] = self.warnings
        self.passed = results['passed']

        return results


# ============================================================================
# MODEL VALIDATOR
# ============================================================================

class ModelValidator:
    """Validate model training quality."""

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.issues: List[str] = []
        self.warnings: List[str] = []

    def validate_cv_setup(
        self,
        n_samples: int,
        train_samples: int,
        test_samples: int,
        embargo_bars: int,
        n_folds: int
    ) -> Dict[str, Any]:
        """Validate cross-validation setup."""
        results = {'passed': True, 'checks': {}}

        # Check minimum samples
        results['checks']['train_samples'] = {
            'count': train_samples,
            'min_required': self.config.min_train_samples,
            'passed': train_samples >= self.config.min_train_samples
        }
        if train_samples < self.config.min_train_samples:
            self.warnings.append(f"Low train samples: {train_samples}")

        results['checks']['test_samples'] = {
            'count': test_samples,
            'min_required': self.config.min_test_samples,
            'passed': test_samples >= self.config.min_test_samples
        }
        if test_samples < self.config.min_test_samples:
            self.warnings.append(f"Low test samples: {test_samples}")

        # Check embargo
        results['checks']['embargo'] = {
            'bars': embargo_bars,
            'min_required': self.config.min_embargo_bars,
            'passed': embargo_bars >= self.config.min_embargo_bars
        }
        if embargo_bars < self.config.min_embargo_bars:
            self.issues.append(f"Insufficient embargo: {embargo_bars} bars")
            results['passed'] = False

        # Check fold count
        results['checks']['n_folds'] = {
            'count': n_folds,
            'passed': n_folds >= 3
        }

        results['issues'] = self.issues
        results['warnings'] = self.warnings

        return results

    def validate_model_metrics(
        self,
        accuracy: float,
        auc_roc: float,
        f1: float,
        model_name: str
    ) -> Dict[str, Any]:
        """Validate model performance metrics for suspicious results."""
        results = {'passed': True, 'checks': {}, 'model': model_name}

        # Check for suspiciously high AUC
        results['checks']['auc_suspicion'] = {
            'value': auc_roc,
            'threshold': self.config.max_auc_suspicion,
            'passed': auc_roc < self.config.max_auc_suspicion
        }
        if auc_roc >= self.config.max_auc_suspicion:
            self.warnings.append(f"SUSPICIOUS: {model_name} AUC-ROC {auc_roc:.4f} may indicate data leakage")

        # Check for random performance
        results['checks']['above_random'] = {
            'auc': auc_roc,
            'passed': auc_roc > 0.52
        }
        if auc_roc <= 0.52:
            self.warnings.append(f"{model_name} performance near random (AUC={auc_roc:.4f})")

        # Check accuracy vs AUC consistency
        expected_acc_min = 0.5 + (auc_roc - 0.5) * 0.5
        results['checks']['acc_auc_consistency'] = {
            'accuracy': accuracy,
            'auc': auc_roc,
            'passed': accuracy >= expected_acc_min * 0.9
        }

        results['issues'] = self.issues
        results['warnings'] = self.warnings

        return results


# ============================================================================
# BACKTEST VALIDATOR
# ============================================================================

class BacktestValidator:
    """Validate backtest quality and realism."""

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.issues: List[str] = []
        self.warnings: List[str] = []

    def validate_trades(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate trade log for quality and realism."""
        results = {'passed': True, 'checks': {}}

        n_trades = len(trades_df)

        # Check minimum trades
        results['checks']['trade_count'] = {
            'count': n_trades,
            'min_required': self.config.min_trades,
            'passed': n_trades >= self.config.min_trades
        }
        if n_trades < self.config.min_trades:
            self.warnings.append(f"Low trade count: {n_trades}")

        if n_trades == 0:
            results['passed'] = False
            self.issues.append("No trades generated")
            return results

        # Check RTH compliance
        if 'entry_time' in trades_df.columns:
            try:
                entry_times = pd.to_datetime(trades_df['entry_time'])
                rth_violations = 0
                for t in entry_times:
                    if hasattr(t, 'time'):
                        entry_time = t.time()
                        if entry_time < self.config.rth_start or entry_time >= self.config.rth_end:
                            rth_violations += 1

                rth_violation_pct = rth_violations / n_trades
                results['checks']['rth_compliance'] = {
                    'violations': rth_violations,
                    'violation_pct': rth_violation_pct,
                    'passed': rth_violation_pct == 0
                }
                if rth_violations > 0:
                    self.issues.append(f"RTH VIOLATIONS: {rth_violations} trades outside RTH")
                    results['passed'] = False
            except Exception as e:
                self.warnings.append(f"Could not validate RTH: {e}")

        # Check win rate bounds
        if 'net_pnl' in trades_df.columns:
            win_rate = (trades_df['net_pnl'] > 0).mean()
            results['checks']['win_rate'] = {
                'value': float(win_rate),
                'min': self.config.min_win_rate,
                'max': self.config.max_win_rate,
                'passed': self.config.min_win_rate <= win_rate <= self.config.max_win_rate
            }
            if win_rate > self.config.max_win_rate:
                self.warnings.append(f"SUSPICIOUS: Win rate {win_rate:.2%} unusually high")
            if win_rate < self.config.min_win_rate:
                self.warnings.append(f"Low win rate: {win_rate:.2%}")

        # Check profit factor
        if 'net_pnl' in trades_df.columns:
            wins = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
            losses = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
            pf = wins / losses if losses > 0 else float('inf')

            results['checks']['profit_factor'] = {
                'value': float(pf) if pf != float('inf') else 'inf',
                'max_threshold': self.config.max_profit_factor,
                'passed': pf <= self.config.max_profit_factor
            }
            if pf > self.config.max_profit_factor:
                self.warnings.append(f"SUSPICIOUS: Profit factor {pf:.2f} unusually high")

        # Check for costs
        if 'commission' in trades_df.columns:
            has_commission = trades_df['commission'].sum() > 0
            results['checks']['has_commission'] = {
                'has_costs': has_commission,
                'passed': has_commission
            }
            if not has_commission:
                self.warnings.append("No commission costs applied - unrealistic")

        if 'slippage' in trades_df.columns:
            has_slippage = trades_df['slippage'].sum() > 0
            results['checks']['has_slippage'] = {
                'has_costs': has_slippage,
                'passed': has_slippage
            }
            if not has_slippage:
                self.warnings.append("No slippage costs applied - unrealistic")

        results['issues'] = self.issues
        results['warnings'] = self.warnings

        return results

    def validate_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate backtest metrics for realism."""
        results = {'passed': True, 'checks': {}}

        # Check Sharpe ratio
        sharpe = metrics.get('sharpe_ratio', 0)
        results['checks']['sharpe'] = {
            'value': sharpe,
            'min': self.config.min_sharpe,
            'passed': sharpe >= self.config.min_sharpe
        }
        if sharpe > 3.0:
            self.warnings.append(f"SUSPICIOUS: Sharpe ratio {sharpe:.2f} unusually high")
        if sharpe < self.config.min_sharpe:
            self.warnings.append(f"Negative Sharpe ratio: {sharpe:.2f}")

        # Check max drawdown
        max_dd_pct = metrics.get('max_drawdown_pct', 0)
        results['checks']['max_drawdown'] = {
            'value': max_dd_pct,
            'passed': max_dd_pct < 0.5  # Less than 50%
        }
        if max_dd_pct >= 0.5:
            self.issues.append(f"Extreme drawdown: {max_dd_pct:.2%}")

        results['issues'] = self.issues
        results['warnings'] = self.warnings

        return results


# ============================================================================
# QUALITY REPORT
# ============================================================================

@dataclass
class QualityReport:
    """Comprehensive quality control report."""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Overall status
    passed: bool = True
    total_issues: int = 0
    total_warnings: int = 0

    # Component results
    data_validation: Optional[Dict] = None
    feature_validation: Optional[Dict] = None
    model_validation: Optional[Dict] = None
    backtest_validation: Optional[Dict] = None

    # Summary
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'passed': self.passed,
            'total_issues': self.total_issues,
            'total_warnings': self.total_warnings,
            'issues': self.issues,
            'warnings': self.warnings,
            'data_validation': self.data_validation,
            'feature_validation': self.feature_validation,
            'model_validation': self.model_validation,
            'backtest_validation': self.backtest_validation
        }

    def generate_report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 70,
            "QUALITY CONTROL REPORT",
            f"Generated: {self.timestamp}",
            "=" * 70,
            "",
            f"OVERALL STATUS: {'PASSED' if self.passed else 'FAILED'}",
            f"Issues: {self.total_issues}",
            f"Warnings: {self.total_warnings}",
            "",
        ]

        if self.issues:
            lines.append("ISSUES (Must Fix):")
            lines.append("-" * 40)
            for issue in self.issues:
                lines.append(f"  ❌ {issue}")
            lines.append("")

        if self.warnings:
            lines.append("WARNINGS (Review):")
            lines.append("-" * 40)
            for warning in self.warnings:
                lines.append(f"  ⚠️ {warning}")
            lines.append("")

        if self.data_validation:
            lines.append("DATA VALIDATION:")
            lines.append("-" * 40)
            for check, result in self.data_validation.get('checks', {}).items():
                status = "✓" if result.get('passed', False) else "✗"
                lines.append(f"  {status} {check}")
            lines.append("")

        if self.feature_validation:
            lines.append("FEATURE VALIDATION:")
            lines.append("-" * 40)
            for check, result in self.feature_validation.get('checks', {}).items():
                status = "✓" if result.get('passed', False) else "✗"
                lines.append(f"  {status} {check}")
            lines.append("")

        if self.model_validation:
            lines.append("MODEL VALIDATION:")
            lines.append("-" * 40)
            for check, result in self.model_validation.get('checks', {}).items():
                status = "✓" if result.get('passed', False) else "✗"
                lines.append(f"  {status} {check}")
            lines.append("")

        if self.backtest_validation:
            lines.append("BACKTEST VALIDATION:")
            lines.append("-" * 40)
            for check, result in self.backtest_validation.get('checks', {}).items():
                status = "✓" if result.get('passed', False) else "✗"
                lines.append(f"  {status} {check}")
            lines.append("")

        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        return "\n".join(lines)


# ============================================================================
# MAIN VALIDATION RUNNER
# ============================================================================

def run_full_validation(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    target_col: str,
    trades_df: Optional[pd.DataFrame] = None,
    metrics: Optional[Dict] = None,
    model_metrics: Optional[Dict] = None,
    config: Optional[ValidationConfig] = None,
    output_path: Optional[str] = None
) -> QualityReport:
    """
    Run comprehensive validation on all components.

    Parameters:
    -----------
    prices : pd.DataFrame
        OHLCV price data
    features : pd.DataFrame
        Feature matrix
    target_col : str
        Target column name
    trades_df : pd.DataFrame, optional
        Trade log from backtest
    metrics : Dict, optional
        Backtest metrics
    model_metrics : Dict, optional
        Model training metrics (accuracy, AUC, etc.)
    config : ValidationConfig, optional
        Validation configuration
    output_path : str, optional
        Path to save report

    Returns:
    --------
    QualityReport with all validation results
    """
    config = config or ValidationConfig()
    report = QualityReport()

    logger.info("="*60)
    logger.info("RUNNING QUALITY CONTROL VALIDATION")
    logger.info("="*60)

    # 1. Validate data
    logger.info("\n1. Validating data quality...")
    data_validator = DataValidator(config)
    report.data_validation = data_validator.validate_ohlcv(prices)
    report.issues.extend(report.data_validation.get('issues', []))
    report.warnings.extend(report.data_validation.get('warnings', []))

    # 2. Validate features
    logger.info("\n2. Validating feature quality...")
    feature_result = data_validator.validate_features(features, target_col)
    report.feature_validation = feature_result
    report.issues.extend(feature_result.get('issues', []))
    report.warnings.extend(feature_result.get('warnings', []))

    # 3. Validate model (if metrics provided)
    if model_metrics:
        logger.info("\n3. Validating model metrics...")
        model_validator = ModelValidator(config)
        report.model_validation = model_validator.validate_model_metrics(
            accuracy=model_metrics.get('accuracy', 0),
            auc_roc=model_metrics.get('auc_roc', 0),
            f1=model_metrics.get('f1', 0),
            model_name=model_metrics.get('model_name', 'Unknown')
        )
        report.issues.extend(report.model_validation.get('issues', []))
        report.warnings.extend(report.model_validation.get('warnings', []))

    # 4. Validate backtest (if trades provided)
    if trades_df is not None and len(trades_df) > 0:
        logger.info("\n4. Validating backtest results...")
        backtest_validator = BacktestValidator(config)
        report.backtest_validation = backtest_validator.validate_trades(trades_df)
        report.issues.extend(report.backtest_validation.get('issues', []))
        report.warnings.extend(report.backtest_validation.get('warnings', []))

        if metrics:
            metrics_result = backtest_validator.validate_metrics(metrics)
            report.issues.extend(metrics_result.get('issues', []))
            report.warnings.extend(metrics_result.get('warnings', []))

    # Finalize report
    report.total_issues = len(report.issues)
    report.total_warnings = len(report.warnings)
    report.passed = report.total_issues == 0

    # Generate and save report
    report_text = report.generate_report()
    logger.info(f"\n{report_text}")

    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(report_text)

        # Also save JSON
        json_path = str(output_path).replace('.txt', '.json')
        with open(json_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        logger.info(f"Report saved to {output_path}")

    return report


# Exports
__all__ = [
    'ValidationConfig',
    'DataValidator',
    'ModelValidator',
    'BacktestValidator',
    'QualityReport',
    'run_full_validation'
]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("Quality Control Validation Framework")
    print("Usage: from validation_framework import run_full_validation")
