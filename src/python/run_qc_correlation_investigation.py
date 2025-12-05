"""
QC Correlation Investigation

Investigates the high correlations flagged in the QC report:
1. Analyzes feature-target correlations
2. Determines if correlations indicate data leakage
3. Tests if correlations are legitimate predictive signals
4. Provides recommendations

Author: SKIE_Ninja Development Team
Created: 2025-12-05
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QCConfig:
    """Configuration for QC investigation."""

    # Correlation thresholds
    high_corr_threshold: float = 0.30
    suspicious_corr_threshold: float = 0.50
    leakage_corr_threshold: float = 0.80

    # Feature patterns that suggest leakage
    leakage_patterns: List[str] = None

    # Output
    output_dir: Path = None

    def __post_init__(self):
        if self.leakage_patterns is None:
            self.leakage_patterns = [
                'future', 'next', 'forward', 'target', 'label',
                'shift(-', 'center=True'
            ]
        if self.output_dir is None:
            self.output_dir = project_root / 'data' / 'qc_investigation'


class QCCorrelationInvestigator:
    """
    Investigates high correlations from QC report.
    """

    def __init__(self, config: Optional[QCConfig] = None):
        self.config = config or QCConfig()
        self.results = {}

    def load_qc_report(self) -> str:
        """Load most recent QC report."""
        qc_dir = project_root / 'data' / 'validation_results'
        qc_files = list(qc_dir.glob('qc_report_*.txt'))

        if not qc_files:
            raise FileNotFoundError("No QC reports found")

        qc_file = max(qc_files, key=lambda p: p.stat().st_mtime)
        with open(qc_file, 'r') as f:
            content = f.read()

        logger.info(f"Loaded QC report: {qc_file.name}")
        return content

    def parse_high_correlations(self, qc_content: str) -> List[Dict]:
        """Parse high correlations from QC report."""
        correlations = []

        for line in qc_content.split('\n'):
            if 'HIGH CORRELATION:' in line:
                # Parse: "HIGH CORRELATION: feature vs target = 0.3602"
                parts = line.split(':')[1].strip()
                feature_target = parts.split('=')[0].strip()
                corr_value = float(parts.split('=')[1].strip())

                feature, target = feature_target.split(' vs ')
                correlations.append({
                    'feature': feature.strip(),
                    'target': target.strip(),
                    'correlation': corr_value
                })

        return correlations

    def classify_correlation(self, feature: str, target: str, corr: float) -> Dict:
        """Classify a correlation as legitimate, suspicious, or leakage."""
        classification = {
            'feature': feature,
            'target': target,
            'correlation': corr,
            'abs_correlation': abs(corr),
            'classification': 'unknown',
            'reason': '',
            'recommendation': ''
        }

        # Check for obvious leakage patterns in feature name
        feature_lower = feature.lower()
        for pattern in self.config.leakage_patterns:
            if pattern.lower() in feature_lower:
                classification['classification'] = 'LEAKAGE'
                classification['reason'] = f"Feature name contains '{pattern}'"
                classification['recommendation'] = 'REMOVE feature immediately'
                return classification

        # Check correlation magnitude
        abs_corr = abs(corr)

        if abs_corr >= self.config.leakage_corr_threshold:
            classification['classification'] = 'LIKELY_LEAKAGE'
            classification['reason'] = f"Correlation {abs_corr:.3f} >= {self.config.leakage_corr_threshold}"
            classification['recommendation'] = 'Investigate feature construction for look-ahead bias'

        elif abs_corr >= self.config.suspicious_corr_threshold:
            classification['classification'] = 'SUSPICIOUS'
            classification['reason'] = f"Correlation {abs_corr:.3f} >= {self.config.suspicious_corr_threshold}"
            classification['recommendation'] = 'Review feature construction carefully'

        else:
            # Analyze if this is a legitimate predictive relationship
            classification = self._analyze_legitimate_correlation(feature, target, corr, classification)

        return classification

    def _analyze_legitimate_correlation(self, feature: str, target: str, corr: float, classification: Dict) -> Dict:
        """Analyze if a correlation represents a legitimate predictive signal."""

        # Known legitimate patterns
        legitimate_patterns = {
            # Momentum features predicting direction
            ('rsi', 'new_high'): "RSI measures overbought/oversold - naturally predictive of highs/lows",
            ('rsi', 'new_low'): "RSI measures overbought/oversold - naturally predictive of highs/lows",
            ('momentum', 'new_high'): "Momentum directly relates to price movement direction",
            ('momentum', 'new_low'): "Momentum directly relates to price movement direction",

            # Price position features
            ('close_vs_high', 'new_high'): "Close near high suggests continuation - legitimate signal",
            ('close_vs_low', 'new_low'): "Close near low suggests continuation - legitimate signal",

            # Bollinger bands
            ('bb_pct', 'new_high'): "BB position measures relative price - predictive of extremes",
            ('bb_pct', 'new_low'): "BB position measures relative price - predictive of extremes",

            # Moving average distance
            ('ma_dist', 'new_high'): "Distance from MA indicates trend strength - legitimate",
            ('ma_dist', 'new_low'): "Distance from MA indicates trend strength - legitimate",

            # Volatility and vol expansion
            ('rv', 'vol_expansion'): "Realized vol relates to vol expansion - but check direction",
            ('atr', 'vol_expansion'): "ATR relates to vol expansion - but check direction",

            # Return lags
            ('return_lag', 'new_high'): "Past returns predict future direction - classic momentum",
            ('return_lag', 'new_low'): "Past returns predict future direction - classic momentum",
        }

        # Check against known patterns
        feature_lower = feature.lower()
        target_lower = target.lower()

        for (feat_pattern, tgt_pattern), explanation in legitimate_patterns.items():
            if feat_pattern in feature_lower and tgt_pattern in target_lower:
                classification['classification'] = 'LEGITIMATE'
                classification['reason'] = explanation
                classification['recommendation'] = 'No action needed - expected relationship'
                return classification

        # Default: needs investigation
        classification['classification'] = 'NEEDS_INVESTIGATION'
        classification['reason'] = f"Correlation {abs(corr):.3f} not obviously legitimate or leakage"
        classification['recommendation'] = 'Manually review feature construction in source code'

        return classification

    def investigate_all_correlations(self, correlations: List[Dict]) -> pd.DataFrame:
        """Investigate all high correlations."""
        results = []

        for corr in correlations:
            classification = self.classify_correlation(
                corr['feature'],
                corr['target'],
                corr['correlation']
            )
            results.append(classification)

        return pd.DataFrame(results)

    def check_feature_code(self, feature_name: str) -> Dict:
        """Check feature construction code for leakage patterns."""
        # Search for feature in source code
        feature_engineering_dir = project_root / 'src' / 'python' / 'feature_engineering'

        result = {
            'feature': feature_name,
            'found_in_files': [],
            'potential_issues': [],
            'code_snippets': []
        }

        for py_file in feature_engineering_dir.glob('*.py'):
            with open(py_file, 'r') as f:
                content = f.read()
                lines = content.split('\n')

            # Search for feature name
            for i, line in enumerate(lines):
                if feature_name in line or feature_name.replace('_', '') in line.lower():
                    result['found_in_files'].append(str(py_file.name))

                    # Check for leakage patterns in surrounding context
                    start = max(0, i - 5)
                    end = min(len(lines), i + 5)
                    context = '\n'.join(lines[start:end])

                    if 'shift(-' in context:
                        result['potential_issues'].append(f"Found shift(-N) near {feature_name}")
                    if 'center=True' in context:
                        result['potential_issues'].append(f"Found center=True near {feature_name}")

                    result['code_snippets'].append({
                        'file': py_file.name,
                        'line': i + 1,
                        'context': context[:500]
                    })
                    break

        result['found_in_files'] = list(set(result['found_in_files']))
        return result

    def run_investigation(self) -> Dict:
        """Run full correlation investigation."""
        # Load QC report
        qc_content = self.load_qc_report()

        # Parse correlations
        correlations = self.parse_high_correlations(qc_content)
        logger.info(f"Found {len(correlations)} high correlations to investigate")

        # Classify each
        investigation_df = self.investigate_all_correlations(correlations)

        # Group by classification
        summary = investigation_df.groupby('classification').size().to_dict()

        # Check code for suspicious features
        code_checks = []
        suspicious = investigation_df[
            investigation_df['classification'].isin(['SUSPICIOUS', 'LIKELY_LEAKAGE', 'NEEDS_INVESTIGATION'])
        ]

        for _, row in suspicious.iterrows():
            check = self.check_feature_code(row['feature'])
            code_checks.append(check)

        self.results = {
            'investigation': investigation_df,
            'summary': summary,
            'code_checks': code_checks,
            'qc_content': qc_content
        }

        return self.results

    def generate_report(self) -> str:
        """Generate investigation report."""
        if not self.results:
            return "No results available. Run investigation first."

        lines = []
        lines.append("=" * 80)
        lines.append(" QC CORRELATION INVESTIGATION REPORT")
        lines.append(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)

        # Summary
        lines.append("\n--- CLASSIFICATION SUMMARY ---")
        for classification, count in sorted(self.results['summary'].items()):
            lines.append(f"  {classification}: {count}")

        # Detailed findings by category
        df = self.results['investigation']

        # LEAKAGE
        leakage = df[df['classification'].isin(['LEAKAGE', 'LIKELY_LEAKAGE'])]
        if len(leakage) > 0:
            lines.append("\n--- POTENTIAL LEAKAGE (CRITICAL) ---")
            for _, row in leakage.iterrows():
                lines.append(f"\n  Feature: {row['feature']}")
                lines.append(f"  Target:  {row['target']}")
                lines.append(f"  Corr:    {row['correlation']:.4f}")
                lines.append(f"  Reason:  {row['reason']}")
                lines.append(f"  Action:  {row['recommendation']}")

        # SUSPICIOUS
        suspicious = df[df['classification'] == 'SUSPICIOUS']
        if len(suspicious) > 0:
            lines.append("\n--- SUSPICIOUS (INVESTIGATE) ---")
            for _, row in suspicious.iterrows():
                lines.append(f"\n  Feature: {row['feature']}")
                lines.append(f"  Target:  {row['target']}")
                lines.append(f"  Corr:    {row['correlation']:.4f}")
                lines.append(f"  Reason:  {row['reason']}")

        # LEGITIMATE
        legitimate = df[df['classification'] == 'LEGITIMATE']
        if len(legitimate) > 0:
            lines.append("\n--- LEGITIMATE CORRELATIONS ---")
            for _, row in legitimate.iterrows():
                lines.append(f"  {row['feature']} vs {row['target']}: {row['correlation']:.4f}")
                lines.append(f"    -> {row['reason']}")

        # NEEDS INVESTIGATION
        needs_inv = df[df['classification'] == 'NEEDS_INVESTIGATION']
        if len(needs_inv) > 0:
            lines.append("\n--- NEEDS MANUAL REVIEW ---")
            for _, row in needs_inv.iterrows():
                lines.append(f"  {row['feature']} vs {row['target']}: {row['correlation']:.4f}")

        # Code check findings
        if self.results['code_checks']:
            issues_found = [c for c in self.results['code_checks'] if c['potential_issues']]
            if issues_found:
                lines.append("\n--- CODE ISSUES FOUND ---")
                for check in issues_found:
                    lines.append(f"\n  Feature: {check['feature']}")
                    for issue in check['potential_issues']:
                        lines.append(f"    - {issue}")

        # Overall assessment
        lines.append("\n" + "=" * 80)
        lines.append(" OVERALL ASSESSMENT")
        lines.append("=" * 80)

        leakage_count = len(df[df['classification'].isin(['LEAKAGE', 'LIKELY_LEAKAGE'])])
        suspicious_count = len(df[df['classification'] == 'SUSPICIOUS'])
        legitimate_count = len(df[df['classification'] == 'LEGITIMATE'])

        if leakage_count > 0:
            lines.append(f"\n[CRITICAL] Found {leakage_count} potential data leakage issues!")
            lines.append("  Action Required: Review and fix before production")
        elif suspicious_count > 0:
            lines.append(f"\n[WARNING] Found {suspicious_count} suspicious correlations")
            lines.append("  Action Required: Manual code review recommended")
        else:
            lines.append(f"\n[OK] No critical issues found")
            lines.append(f"  {legitimate_count} correlations appear legitimate")

        # Specific recommendations
        lines.append("\n--- RECOMMENDATIONS ---")

        if leakage_count > 0:
            lines.append("1. IMMEDIATE: Remove or fix features flagged as LEAKAGE")
            lines.append("2. Re-run backtest after removing leaky features")
            lines.append("3. Expect significant performance degradation")

        if suspicious_count > 0:
            lines.append("1. Review feature engineering code for:")
            lines.append("   - shift(-N) patterns (future data)")
            lines.append("   - center=True in rolling windows")
            lines.append("   - Any reference to future prices/targets")
            lines.append("2. Add unit tests for each suspicious feature")

        if legitimate_count == len(df):
            lines.append("1. High correlations appear to be legitimate signals")
            lines.append("2. Consider if correlations indicate tautological features")
            lines.append("3. Monitor for correlation decay in live trading")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def save_results(self, timestamp: str = None):
        """Save results to files."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save report
        report = self.generate_report()
        report_file = self.config.output_dir / f'qc_investigation_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_file}")

        # Save detailed CSV
        csv_file = self.config.output_dir / f'qc_investigation_details_{timestamp}.csv'
        self.results['investigation'].to_csv(csv_file, index=False)
        logger.info(f"Details saved to: {csv_file}")

        return report_file


def run_qc_investigation():
    """Main function to run QC correlation investigation."""
    print("=" * 80)
    print(" QC CORRELATION INVESTIGATION")
    print(" Analyzing High Feature-Target Correlations")
    print("=" * 80)

    # Initialize
    config = QCConfig()
    investigator = QCCorrelationInvestigator(config)

    # Run investigation
    print("\n[1] Loading QC report and investigating correlations...")
    try:
        results = investigator.run_investigation()
        print(f"    Investigated {len(results['investigation'])} correlations")
    except FileNotFoundError as e:
        print(f"    Error: {e}")
        return None

    # Print report
    print("\n" + investigator.generate_report())

    # Save results
    print("\n[2] Saving results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = investigator.save_results(timestamp)
    print(f"    Report: {report_file.name}")

    return investigator


if __name__ == "__main__":
    investigator = run_qc_investigation()
