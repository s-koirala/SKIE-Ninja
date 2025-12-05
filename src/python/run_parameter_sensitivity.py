"""
Parameter Sensitivity Analysis

Tests robustness of optimized parameters by:
1. Grid search around optimal values
2. Measuring performance variance across parameter space
3. Identifying fragile vs robust parameters
4. Generating heatmaps and sensitivity curves

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
from itertools import product
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis."""

    # Parameter ranges to test (around optimal)
    vol_expansion_probs: List[float] = None
    breakout_probs: List[float] = None
    tp_atr_mults: List[float] = None
    sl_atr_mults: List[float] = None

    # Baseline (optimized) values
    baseline_vol_prob: float = 0.40
    baseline_breakout_prob: float = 0.45
    baseline_tp_mult: float = 2.5
    baseline_sl_mult: float = 1.25

    # Output
    output_dir: Path = None

    def __post_init__(self):
        if self.vol_expansion_probs is None:
            self.vol_expansion_probs = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        if self.breakout_probs is None:
            self.breakout_probs = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        if self.tp_atr_mults is None:
            self.tp_atr_mults = [1.5, 2.0, 2.5, 3.0, 3.5]
        if self.sl_atr_mults is None:
            self.sl_atr_mults = [0.75, 1.0, 1.25, 1.5, 1.75]
        if self.output_dir is None:
            self.output_dir = project_root / 'data' / 'sensitivity_results'


class ParameterSensitivityAnalyzer:
    """
    Analyzes parameter sensitivity for the trading strategy.
    """

    def __init__(self, config: Optional[SensitivityConfig] = None):
        self.config = config or SensitivityConfig()
        self.trades_df = None
        self.results = {}

    def load_trades(self) -> pd.DataFrame:
        """Load original trades with full parameter info."""
        results_dir = project_root / 'data' / 'backtest_results'

        # Find trades file
        for pattern in ['oos_*trades*.csv', 'vol_breakout*trades*.csv', '*trades*.csv']:
            trade_files = list(results_dir.glob(pattern))
            if trade_files:
                break

        if not trade_files:
            raise FileNotFoundError("No trade files found")

        trades_file = max(trade_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading trades from: {trades_file.name}")

        self.trades_df = pd.read_csv(trades_file)
        logger.info(f"Loaded {len(self.trades_df)} trades")

        return self.trades_df

    def filter_trades_by_params(
        self,
        trades: pd.DataFrame,
        vol_prob_threshold: float,
        breakout_prob_threshold: float,
        tp_mult: float = None,
        sl_mult: float = None
    ) -> pd.DataFrame:
        """Filter trades based on parameter thresholds."""
        filtered = trades.copy()

        # Filter by probability thresholds
        if 'vol_prob' in filtered.columns:
            filtered = filtered[filtered['vol_prob'] >= vol_prob_threshold]
        elif 'vol_expansion_prob' in filtered.columns:
            filtered = filtered[filtered['vol_expansion_prob'] >= vol_prob_threshold]

        if 'breakout_prob' in filtered.columns:
            filtered = filtered[filtered['breakout_prob'] >= breakout_prob_threshold]

        # Adjust P&L based on TP/SL multipliers (approximation)
        if tp_mult is not None and sl_mult is not None and 'predicted_atr' in filtered.columns:
            # This is an approximation - in reality would need full backtest
            baseline_tp = self.config.baseline_tp_mult
            baseline_sl = self.config.baseline_sl_mult

            # Adjust winning trades by TP ratio
            tp_ratio = tp_mult / baseline_tp
            sl_ratio = sl_mult / baseline_sl

            # Winners get scaled by TP ratio, losers by SL ratio
            winners_mask = filtered['net_pnl'] > 0
            filtered.loc[winners_mask, 'net_pnl'] *= tp_ratio
            filtered.loc[~winners_mask, 'net_pnl'] *= sl_ratio

        return filtered

    def calculate_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate key performance metrics."""
        if len(trades) == 0:
            return {
                'net_pnl': 0, 'trades': 0, 'win_rate': 0,
                'profit_factor': 0, 'sharpe': 0, 'max_dd': 0
            }

        net_pnl = trades['net_pnl'].sum()
        total_trades = len(trades)
        win_rate = (trades['net_pnl'] > 0).mean()

        winners = trades[trades['net_pnl'] > 0]['net_pnl']
        losers = trades[trades['net_pnl'] < 0]['net_pnl']

        gross_profit = winners.sum() if len(winners) > 0 else 0
        gross_loss = abs(losers.sum()) if len(losers) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Max drawdown
        cumulative = trades['net_pnl'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = running_max - cumulative
        max_dd = drawdown.max()

        # Sharpe
        sharpe = 0
        if 'entry_time' in trades.columns:
            trades_copy = trades.copy()
            trades_copy['date'] = pd.to_datetime(trades_copy['entry_time']).dt.date
            daily_pnl = trades_copy.groupby('date')['net_pnl'].sum()
            if len(daily_pnl) > 1 and daily_pnl.std() > 0:
                sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)

        return {
            'net_pnl': net_pnl,
            'trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe': sharpe,
            'max_dd': max_dd
        }

    def run_1d_sensitivity(self, trades: pd.DataFrame, param_name: str, values: List[float]) -> pd.DataFrame:
        """Run 1D sensitivity analysis for a single parameter."""
        logger.info(f"Running 1D sensitivity for {param_name}...")

        results = []
        for val in values:
            if param_name == 'vol_expansion_prob':
                filtered = self.filter_trades_by_params(
                    trades, val, self.config.baseline_breakout_prob
                )
            elif param_name == 'breakout_prob':
                filtered = self.filter_trades_by_params(
                    trades, self.config.baseline_vol_prob, val
                )
            elif param_name == 'tp_atr_mult':
                filtered = self.filter_trades_by_params(
                    trades, self.config.baseline_vol_prob, self.config.baseline_breakout_prob,
                    tp_mult=val, sl_mult=self.config.baseline_sl_mult
                )
            elif param_name == 'sl_atr_mult':
                filtered = self.filter_trades_by_params(
                    trades, self.config.baseline_vol_prob, self.config.baseline_breakout_prob,
                    tp_mult=self.config.baseline_tp_mult, sl_mult=val
                )
            else:
                continue

            metrics = self.calculate_metrics(filtered)
            metrics['param_value'] = val
            metrics['param_name'] = param_name
            results.append(metrics)

        return pd.DataFrame(results)

    def run_2d_sensitivity(
        self,
        trades: pd.DataFrame,
        param1_name: str,
        param1_values: List[float],
        param2_name: str,
        param2_values: List[float]
    ) -> pd.DataFrame:
        """Run 2D sensitivity analysis for parameter pairs."""
        logger.info(f"Running 2D sensitivity for {param1_name} x {param2_name}...")

        results = []
        for val1, val2 in product(param1_values, param2_values):
            if param1_name == 'vol_expansion_prob' and param2_name == 'breakout_prob':
                filtered = self.filter_trades_by_params(trades, val1, val2)
            elif param1_name == 'tp_atr_mult' and param2_name == 'sl_atr_mult':
                filtered = self.filter_trades_by_params(
                    trades, self.config.baseline_vol_prob, self.config.baseline_breakout_prob,
                    tp_mult=val1, sl_mult=val2
                )
            else:
                continue

            metrics = self.calculate_metrics(filtered)
            metrics[param1_name] = val1
            metrics[param2_name] = val2
            results.append(metrics)

        return pd.DataFrame(results)

    def calculate_sensitivity_metrics(self, sensitivity_df: pd.DataFrame, param_name: str) -> Dict:
        """Calculate sensitivity metrics for a parameter."""
        if len(sensitivity_df) == 0:
            return {}

        pnl_values = sensitivity_df['net_pnl'].values
        param_values = sensitivity_df['param_value'].values if 'param_value' in sensitivity_df.columns else None

        # Basic statistics
        pnl_mean = np.mean(pnl_values)
        pnl_std = np.std(pnl_values)
        pnl_range = np.max(pnl_values) - np.min(pnl_values)

        # Coefficient of variation (normalized sensitivity)
        cv = pnl_std / abs(pnl_mean) if pnl_mean != 0 else np.inf

        # Robustness: % of parameter space that's profitable
        pct_profitable = (pnl_values > 0).mean() * 100

        return {
            'param_name': param_name,
            'pnl_mean': pnl_mean,
            'pnl_std': pnl_std,
            'pnl_range': pnl_range,
            'cv': cv,
            'pct_profitable': pct_profitable,
            'is_robust': cv < 0.5 and pct_profitable >= 80
        }

    def run_full_analysis(self, trades: pd.DataFrame) -> Dict:
        """Run complete sensitivity analysis."""
        results = {
            '1d': {},
            '2d': {},
            'sensitivity_metrics': []
        }

        # 1D sensitivity for each parameter
        for param_name, values in [
            ('vol_expansion_prob', self.config.vol_expansion_probs),
            ('breakout_prob', self.config.breakout_probs),
            ('tp_atr_mult', self.config.tp_atr_mults),
            ('sl_atr_mult', self.config.sl_atr_mults)
        ]:
            df = self.run_1d_sensitivity(trades, param_name, values)
            results['1d'][param_name] = df
            metrics = self.calculate_sensitivity_metrics(df, param_name)
            if metrics:
                results['sensitivity_metrics'].append(metrics)

        # 2D sensitivity for key pairs
        results['2d']['vol_breakout'] = self.run_2d_sensitivity(
            trades,
            'vol_expansion_prob', self.config.vol_expansion_probs,
            'breakout_prob', self.config.breakout_probs
        )

        results['2d']['tp_sl'] = self.run_2d_sensitivity(
            trades,
            'tp_atr_mult', self.config.tp_atr_mults,
            'sl_atr_mult', self.config.sl_atr_mults
        )

        self.results = results
        return results

    def generate_report(self) -> str:
        """Generate sensitivity analysis report."""
        if not self.results:
            return "No results available. Run analysis first."

        lines = []
        lines.append("=" * 80)
        lines.append(" PARAMETER SENSITIVITY ANALYSIS REPORT")
        lines.append(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)

        # Baseline parameters
        lines.append("\n--- BASELINE (OPTIMIZED) PARAMETERS ---")
        lines.append(f"  vol_expansion_prob: {self.config.baseline_vol_prob}")
        lines.append(f"  breakout_prob:      {self.config.baseline_breakout_prob}")
        lines.append(f"  tp_atr_mult:        {self.config.baseline_tp_mult}")
        lines.append(f"  sl_atr_mult:        {self.config.baseline_sl_mult}")

        # 1D Sensitivity Results
        lines.append("\n--- 1D SENSITIVITY ANALYSIS ---")

        for param_name, df in self.results['1d'].items():
            lines.append(f"\n  {param_name}:")
            lines.append(f"  {'Value':<8} {'Net P&L':>12} {'Trades':>8} {'Win Rate':>10} {'Sharpe':>8}")
            lines.append("  " + "-" * 50)

            for _, row in df.iterrows():
                lines.append(
                    f"  {row['param_value']:<8.2f} ${row['net_pnl']:>11,.0f} {row['trades']:>8.0f} "
                    f"{row['win_rate']*100:>9.1f}% {row['sharpe']:>8.2f}"
                )

        # Sensitivity metrics summary
        lines.append("\n--- SENSITIVITY METRICS SUMMARY ---")
        lines.append(f"| Parameter | CV | % Profitable | Robust? |")
        lines.append(f"|-----------|-----|--------------|---------|")

        for metrics in self.results['sensitivity_metrics']:
            robust_str = "YES" if metrics['is_robust'] else "NO"
            lines.append(
                f"| {metrics['param_name']:<20} | {metrics['cv']:.3f} | "
                f"{metrics['pct_profitable']:.1f}% | {robust_str} |"
            )

        # 2D Analysis - Best/Worst combinations
        lines.append("\n--- 2D SENSITIVITY: VOL_PROB x BREAKOUT_PROB ---")
        vol_bp = self.results['2d']['vol_breakout']
        if len(vol_bp) > 0:
            best = vol_bp.loc[vol_bp['net_pnl'].idxmax()]
            worst = vol_bp.loc[vol_bp['net_pnl'].idxmin()]

            lines.append(f"  Best:  vol_prob={best['vol_expansion_prob']:.2f}, "
                        f"breakout_prob={best['breakout_prob']:.2f} -> ${best['net_pnl']:,.0f}")
            lines.append(f"  Worst: vol_prob={worst['vol_expansion_prob']:.2f}, "
                        f"breakout_prob={worst['breakout_prob']:.2f} -> ${worst['net_pnl']:,.0f}")
            lines.append(f"  Range: ${vol_bp['net_pnl'].max() - vol_bp['net_pnl'].min():,.0f}")

        lines.append("\n--- 2D SENSITIVITY: TP_MULT x SL_MULT ---")
        tp_sl = self.results['2d']['tp_sl']
        if len(tp_sl) > 0:
            best = tp_sl.loc[tp_sl['net_pnl'].idxmax()]
            worst = tp_sl.loc[tp_sl['net_pnl'].idxmin()]

            lines.append(f"  Best:  tp_mult={best['tp_atr_mult']:.2f}, "
                        f"sl_mult={best['sl_atr_mult']:.2f} -> ${best['net_pnl']:,.0f}")
            lines.append(f"  Worst: tp_mult={worst['tp_atr_mult']:.2f}, "
                        f"sl_mult={worst['sl_atr_mult']:.2f} -> ${worst['net_pnl']:,.0f}")
            lines.append(f"  Range: ${tp_sl['net_pnl'].max() - tp_sl['net_pnl'].min():,.0f}")

        # Overall assessment
        lines.append("\n" + "=" * 80)
        lines.append(" ROBUSTNESS ASSESSMENT")
        lines.append("=" * 80)

        robust_params = sum(1 for m in self.results['sensitivity_metrics'] if m['is_robust'])
        total_params = len(self.results['sensitivity_metrics'])

        if robust_params == total_params:
            lines.append("\n[ROBUST] All parameters show low sensitivity - strategy is robust")
        elif robust_params >= total_params * 0.75:
            lines.append("\n[ACCEPTABLE] Most parameters are robust - minor sensitivities exist")
        elif robust_params >= total_params * 0.5:
            lines.append("\n[CAUTION] Mixed robustness - some parameters are fragile")
        else:
            lines.append("\n[CONCERN] High parameter sensitivity - strategy may be over-optimized")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def save_results(self, timestamp: str = None):
        """Save results to files."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save report
        report = self.generate_report()
        report_file = self.config.output_dir / f'sensitivity_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_file}")

        # Save 1D results
        for param_name, df in self.results['1d'].items():
            csv_file = self.config.output_dir / f'sensitivity_1d_{param_name}_{timestamp}.csv'
            df.to_csv(csv_file, index=False)

        # Save 2D results
        for name, df in self.results['2d'].items():
            csv_file = self.config.output_dir / f'sensitivity_2d_{name}_{timestamp}.csv'
            df.to_csv(csv_file, index=False)

        logger.info(f"All results saved to: {self.config.output_dir}")

        return report_file


def run_parameter_sensitivity():
    """Main function to run parameter sensitivity analysis."""
    print("=" * 80)
    print(" PARAMETER SENSITIVITY ANALYSIS")
    print(" Testing Robustness of Optimized Parameters")
    print("=" * 80)

    # Initialize
    config = SensitivityConfig()
    analyzer = ParameterSensitivityAnalyzer(config)

    # Load trades
    print("\n[1] Loading trades...")
    try:
        trades = analyzer.load_trades()
        print(f"    Loaded {len(trades)} trades")
    except FileNotFoundError as e:
        print(f"    Error: {e}")
        return None

    # Run analysis
    print("\n[2] Running sensitivity analysis...")
    print(f"    Testing {len(config.vol_expansion_probs)} vol_prob values")
    print(f"    Testing {len(config.breakout_probs)} breakout_prob values")
    print(f"    Testing {len(config.tp_atr_mults)} TP mult values")
    print(f"    Testing {len(config.sl_atr_mults)} SL mult values")

    results = analyzer.run_full_analysis(trades)

    # Print report
    print("\n" + analyzer.generate_report())

    # Save results
    print("\n[3] Saving results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = analyzer.save_results(timestamp)
    print(f"    Report: {report_file.name}")

    return analyzer


if __name__ == "__main__":
    analyzer = run_parameter_sensitivity()
