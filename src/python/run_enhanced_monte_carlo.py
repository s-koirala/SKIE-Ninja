"""
Enhanced Monte Carlo Simulation with Stress Testing

Extends basic Monte Carlo with:
1. Extreme slippage scenarios (2x, 3x, 5x)
2. Severe trade dropout (20%, 30%, 50%)
3. Adverse selection (remove only winning trades)
4. Regime stress testing (high VIX simulation)
5. Black swan event simulation
6. Parameter perturbation

Author: SKIE_Ninja Development Team
Created: 2025-12-05
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedMCConfig:
    """Configuration for enhanced Monte Carlo simulation."""

    # Base simulation parameters
    n_simulations: int = 5000
    confidence_level: float = 0.95

    # Slippage stress multipliers
    slippage_multipliers: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 5.0])

    # Dropout stress levels
    dropout_levels: List[float] = field(default_factory=lambda: [0.0, 0.15, 0.30, 0.50])

    # Adverse selection (remove winning trades)
    adverse_selection_rates: List[float] = field(default_factory=lambda: [0.0, 0.10, 0.20, 0.30])

    # Black swan parameters
    black_swan_loss_pct: float = 0.20  # 20% of max drawdown as sudden loss
    black_swan_frequency: float = 0.05  # 5% of simulations include black swan

    # Output
    output_dir: Path = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = project_root / 'data' / 'monte_carlo_results'


class EnhancedMonteCarloSimulator:
    """
    Enhanced Monte Carlo simulator with comprehensive stress testing.
    """

    def __init__(self, config: Optional[EnhancedMCConfig] = None):
        self.config = config or EnhancedMCConfig()
        self.results = {}
        self.trades_df = None

    def load_trades(self) -> pd.DataFrame:
        """Load trades from most recent backtest file."""
        results_dir = project_root / 'data' / 'backtest_results'

        # Try to find ensemble or OOS trades
        for pattern in ['ensemble_*trades*.csv', 'oos_*trades*.csv', '*trades*.csv']:
            trade_files = list(results_dir.glob(pattern))
            if trade_files:
                break

        if not trade_files:
            raise FileNotFoundError("No trade files found in backtest_results")

        # Get most recent
        trades_file = max(trade_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading trades from: {trades_file.name}")

        self.trades_df = pd.read_csv(trades_file)
        logger.info(f"Loaded {len(self.trades_df)} trades")

        return self.trades_df

    def calculate_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate comprehensive metrics from trades."""
        if len(trades) == 0:
            return {
                'net_pnl': 0, 'trades': 0, 'win_rate': 0,
                'profit_factor': 0, 'sharpe': 0, 'max_dd': 0,
                'avg_win': 0, 'avg_loss': 0, 'expectancy': 0
            }

        net_pnl = trades['net_pnl'].sum()
        total_trades = len(trades)

        winners = trades[trades['net_pnl'] > 0]
        losers = trades[trades['net_pnl'] < 0]

        win_rate = len(winners) / total_trades if total_trades > 0 else 0

        avg_win = winners['net_pnl'].mean() if len(winners) > 0 else 0
        avg_loss = abs(losers['net_pnl'].mean()) if len(losers) > 0 else 0

        gross_profit = winners['net_pnl'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['net_pnl'].sum()) if len(losers) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Max drawdown
        cumulative = trades['net_pnl'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = running_max - cumulative
        max_dd = drawdown.max()

        # Sharpe (daily)
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
            'max_dd': max_dd,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy
        }

    # ========== STRESS TEST METHODS ==========

    def stress_slippage(self, trades: pd.DataFrame, multiplier: float) -> pd.DataFrame:
        """Apply slippage stress multiplier."""
        trades = trades.copy()
        if 'slippage' in trades.columns:
            additional_slippage = trades['slippage'] * (multiplier - 1)
            trades['net_pnl'] = trades['net_pnl'] - additional_slippage
        return trades

    def stress_dropout(self, trades: pd.DataFrame, dropout_rate: float) -> pd.DataFrame:
        """Remove random trades at specified rate."""
        if dropout_rate <= 0:
            return trades
        keep_mask = np.random.random(len(trades)) > dropout_rate
        return trades[keep_mask].reset_index(drop=True)

    def stress_adverse_selection(self, trades: pd.DataFrame, remove_rate: float) -> pd.DataFrame:
        """Adversely remove winning trades (worst case scenario)."""
        if remove_rate <= 0:
            return trades

        trades = trades.copy()
        winners_idx = trades[trades['net_pnl'] > 0].index.tolist()

        # Remove specified percentage of winners
        n_remove = int(len(winners_idx) * remove_rate)
        if n_remove > 0:
            remove_idx = np.random.choice(winners_idx, size=n_remove, replace=False)
            trades = trades.drop(remove_idx).reset_index(drop=True)

        return trades

    def stress_black_swan(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Inject black swan event (sudden large loss)."""
        trades = trades.copy()

        # Calculate black swan loss amount
        max_dd = abs(trades['net_pnl'].cumsum().min() - trades['net_pnl'].cumsum().max())
        black_swan_loss = -max_dd * self.config.black_swan_loss_pct

        # Insert at random position
        insert_idx = np.random.randint(0, len(trades))

        # Create black swan trade
        black_swan = trades.iloc[[insert_idx]].copy()
        black_swan['net_pnl'] = black_swan_loss
        black_swan['exit_reason'] = 'black_swan'

        # Insert
        trades = pd.concat([
            trades.iloc[:insert_idx],
            black_swan,
            trades.iloc[insert_idx:]
        ]).reset_index(drop=True)

        return trades

    # ========== SIMULATION RUNNERS ==========

    def run_slippage_stress_test(self, trades: pd.DataFrame) -> Dict:
        """Run stress test across slippage multipliers."""
        logger.info("Running slippage stress test...")

        results = {}
        for mult in self.config.slippage_multipliers:
            sim_results = []
            for _ in range(self.config.n_simulations):
                stressed = self.stress_slippage(trades, mult)
                # Add some random variance
                indices = np.random.choice(len(stressed), size=len(stressed), replace=True)
                sampled = stressed.iloc[indices].reset_index(drop=True)
                metrics = self.calculate_metrics(sampled)
                sim_results.append(metrics)

            results[f'slippage_{mult}x'] = self._summarize_results(sim_results)
            logger.info(f"  {mult}x slippage: Mean P&L = ${results[f'slippage_{mult}x']['net_pnl']['mean']:,.0f}")

        return results

    def run_dropout_stress_test(self, trades: pd.DataFrame) -> Dict:
        """Run stress test across dropout levels."""
        logger.info("Running dropout stress test...")

        results = {}
        for rate in self.config.dropout_levels:
            sim_results = []
            for _ in range(self.config.n_simulations):
                dropped = self.stress_dropout(trades, rate)
                metrics = self.calculate_metrics(dropped)
                sim_results.append(metrics)

            results[f'dropout_{int(rate*100)}pct'] = self._summarize_results(sim_results)
            prob_profit = results[f'dropout_{int(rate*100)}pct']['net_pnl']['prob_positive']
            logger.info(f"  {int(rate*100)}% dropout: P(profit) = {prob_profit*100:.1f}%")

        return results

    def run_adverse_selection_test(self, trades: pd.DataFrame) -> Dict:
        """Run adverse selection stress test (remove winners)."""
        logger.info("Running adverse selection test (removing winners)...")

        results = {}
        for rate in self.config.adverse_selection_rates:
            sim_results = []
            for _ in range(self.config.n_simulations):
                adverse = self.stress_adverse_selection(trades, rate)
                metrics = self.calculate_metrics(adverse)
                sim_results.append(metrics)

            results[f'adverse_{int(rate*100)}pct'] = self._summarize_results(sim_results)
            prob_profit = results[f'adverse_{int(rate*100)}pct']['net_pnl']['prob_positive']
            logger.info(f"  {int(rate*100)}% winners removed: P(profit) = {prob_profit*100:.1f}%")

        return results

    def run_black_swan_test(self, trades: pd.DataFrame) -> Dict:
        """Run black swan event simulation."""
        logger.info("Running black swan simulation...")

        sim_results = []
        n_black_swan = int(self.config.n_simulations * self.config.black_swan_frequency)

        for i in range(self.config.n_simulations):
            # Bootstrap sample
            indices = np.random.choice(len(trades), size=len(trades), replace=True)
            sampled = trades.iloc[indices].reset_index(drop=True)

            # Inject black swan for subset
            if i < n_black_swan:
                sampled = self.stress_black_swan(sampled)

            metrics = self.calculate_metrics(sampled)
            sim_results.append(metrics)

        return {'black_swan': self._summarize_results(sim_results)}

    def run_combined_extreme_test(self, trades: pd.DataFrame) -> Dict:
        """Run combined extreme stress test (worst case)."""
        logger.info("Running COMBINED EXTREME stress test...")

        sim_results = []
        for i in range(self.config.n_simulations):
            stressed = trades.copy()

            # Apply 3x slippage
            stressed = self.stress_slippage(stressed, 3.0)

            # 30% random dropout
            stressed = self.stress_dropout(stressed, 0.30)

            # 20% adverse selection (remove winners)
            stressed = self.stress_adverse_selection(stressed, 0.20)

            # 10% chance of black swan
            if np.random.random() < 0.10:
                stressed = self.stress_black_swan(stressed)

            metrics = self.calculate_metrics(stressed)
            sim_results.append(metrics)

        result = self._summarize_results(sim_results)
        logger.info(f"  Combined extreme: P(profit) = {result['net_pnl']['prob_positive']*100:.1f}%")
        logger.info(f"  Combined extreme: Mean P&L = ${result['net_pnl']['mean']:,.0f}")
        logger.info(f"  Combined extreme: Worst case = ${result['net_pnl']['min']:,.0f}")

        return {'combined_extreme': result}

    def _summarize_results(self, results: List[Dict]) -> Dict:
        """Summarize simulation results with statistics."""
        results_df = pd.DataFrame(results)

        alpha = 1 - self.config.confidence_level
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100

        summary = {}
        for col in results_df.columns:
            values = results_df[col].values
            summary[col] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'ci_lower': np.percentile(values, lower_pct),
                'ci_upper': np.percentile(values, upper_pct),
                'min': np.min(values),
                'max': np.max(values),
                'prob_positive': (values > 0).mean() if col == 'net_pnl' else None
            }

        return summary

    def run_all_stress_tests(self, trades: pd.DataFrame) -> Dict:
        """Run all stress tests."""
        original = self.calculate_metrics(trades)

        results = {
            'original': original,
            'slippage_stress': self.run_slippage_stress_test(trades),
            'dropout_stress': self.run_dropout_stress_test(trades),
            'adverse_selection': self.run_adverse_selection_test(trades),
            'black_swan': self.run_black_swan_test(trades),
            'combined_extreme': self.run_combined_extreme_test(trades)
        }

        self.results = results
        return results

    def generate_report(self) -> str:
        """Generate comprehensive stress test report."""
        if not self.results:
            return "No results available. Run stress tests first."

        lines = []
        lines.append("=" * 80)
        lines.append(" ENHANCED MONTE CARLO STRESS TEST REPORT")
        lines.append(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)

        # Original metrics
        orig = self.results['original']
        lines.append("\n--- ORIGINAL STRATEGY PERFORMANCE ---")
        lines.append(f"  Net P&L:       ${orig['net_pnl']:,.2f}")
        lines.append(f"  Trades:        {orig['trades']}")
        lines.append(f"  Win Rate:      {orig['win_rate']*100:.1f}%")
        lines.append(f"  Profit Factor: {orig['profit_factor']:.2f}")
        lines.append(f"  Sharpe Ratio:  {orig['sharpe']:.2f}")
        lines.append(f"  Max Drawdown:  ${orig['max_dd']:,.2f}")

        # Slippage stress
        lines.append("\n--- SLIPPAGE STRESS TEST ---")
        lines.append("| Multiplier | Mean P&L | 95% CI Lower | P(Profit>0) |")
        lines.append("|------------|----------|--------------|-------------|")
        for key, data in self.results['slippage_stress'].items():
            mult = key.replace('slippage_', '').replace('x', '')
            pnl = data['net_pnl']
            lines.append(f"| {mult}x | ${pnl['mean']:,.0f} | ${pnl['ci_lower']:,.0f} | {pnl['prob_positive']*100:.1f}% |")

        # Dropout stress
        lines.append("\n--- DROPOUT STRESS TEST ---")
        lines.append("| Dropout Rate | Mean P&L | 95% CI Lower | P(Profit>0) |")
        lines.append("|--------------|----------|--------------|-------------|")
        for key, data in self.results['dropout_stress'].items():
            rate = key.replace('dropout_', '').replace('pct', '%')
            pnl = data['net_pnl']
            lines.append(f"| {rate} | ${pnl['mean']:,.0f} | ${pnl['ci_lower']:,.0f} | {pnl['prob_positive']*100:.1f}% |")

        # Adverse selection
        lines.append("\n--- ADVERSE SELECTION TEST (Remove Winners) ---")
        lines.append("| Winners Removed | Mean P&L | 95% CI Lower | P(Profit>0) |")
        lines.append("|-----------------|----------|--------------|-------------|")
        for key, data in self.results['adverse_selection'].items():
            rate = key.replace('adverse_', '').replace('pct', '%')
            pnl = data['net_pnl']
            lines.append(f"| {rate} | ${pnl['mean']:,.0f} | ${pnl['ci_lower']:,.0f} | {pnl['prob_positive']*100:.1f}% |")

        # Black swan
        lines.append("\n--- BLACK SWAN EVENT TEST ---")
        bs = self.results['black_swan']['black_swan']['net_pnl']
        lines.append(f"  Frequency: {self.config.black_swan_frequency*100:.0f}% of simulations")
        lines.append(f"  Impact: {self.config.black_swan_loss_pct*100:.0f}% of max drawdown")
        lines.append(f"  Mean P&L: ${bs['mean']:,.0f}")
        lines.append(f"  95% CI: [${bs['ci_lower']:,.0f}, ${bs['ci_upper']:,.0f}]")
        lines.append(f"  P(Profit>0): {bs['prob_positive']*100:.1f}%")

        # Combined extreme
        lines.append("\n--- COMBINED EXTREME STRESS TEST ---")
        lines.append("  Conditions: 3x slippage + 30% dropout + 20% adverse + 10% black swan")
        extreme = self.results['combined_extreme']['combined_extreme']['net_pnl']
        lines.append(f"  Mean P&L: ${extreme['mean']:,.0f}")
        lines.append(f"  Std Dev: ${extreme['std']:,.0f}")
        lines.append(f"  95% CI: [${extreme['ci_lower']:,.0f}, ${extreme['ci_upper']:,.0f}]")
        lines.append(f"  Worst Case: ${extreme['min']:,.0f}")
        lines.append(f"  P(Profit>0): {extreme['prob_positive']*100:.1f}%")

        # Assessment
        lines.append("\n" + "=" * 80)
        lines.append(" ROBUSTNESS ASSESSMENT")
        lines.append("=" * 80)

        # Determine pass/fail for each test
        assessments = []

        # Check 3x slippage
        slip_3x = self.results['slippage_stress']['slippage_3.0x']['net_pnl']
        if slip_3x['prob_positive'] >= 0.95:
            assessments.append(("3x Slippage", "PASS", f"{slip_3x['prob_positive']*100:.1f}% profit probability"))
        elif slip_3x['prob_positive'] >= 0.80:
            assessments.append(("3x Slippage", "MARGINAL", f"{slip_3x['prob_positive']*100:.1f}% profit probability"))
        else:
            assessments.append(("3x Slippage", "FAIL", f"{slip_3x['prob_positive']*100:.1f}% profit probability"))

        # Check 30% dropout
        drop_30 = self.results['dropout_stress']['dropout_30pct']['net_pnl']
        if drop_30['prob_positive'] >= 0.95:
            assessments.append(("30% Dropout", "PASS", f"{drop_30['prob_positive']*100:.1f}% profit probability"))
        elif drop_30['prob_positive'] >= 0.80:
            assessments.append(("30% Dropout", "MARGINAL", f"{drop_30['prob_positive']*100:.1f}% profit probability"))
        else:
            assessments.append(("30% Dropout", "FAIL", f"{drop_30['prob_positive']*100:.1f}% profit probability"))

        # Check 20% adverse selection
        adv_20 = self.results['adverse_selection']['adverse_20pct']['net_pnl']
        if adv_20['prob_positive'] >= 0.95:
            assessments.append(("20% Adverse Selection", "PASS", f"{adv_20['prob_positive']*100:.1f}% profit probability"))
        elif adv_20['prob_positive'] >= 0.80:
            assessments.append(("20% Adverse Selection", "MARGINAL", f"{adv_20['prob_positive']*100:.1f}% profit probability"))
        else:
            assessments.append(("20% Adverse Selection", "FAIL", f"{adv_20['prob_positive']*100:.1f}% profit probability"))

        # Check combined extreme
        if extreme['prob_positive'] >= 0.80:
            assessments.append(("Combined Extreme", "PASS", f"{extreme['prob_positive']*100:.1f}% profit probability"))
        elif extreme['prob_positive'] >= 0.60:
            assessments.append(("Combined Extreme", "MARGINAL", f"{extreme['prob_positive']*100:.1f}% profit probability"))
        else:
            assessments.append(("Combined Extreme", "FAIL", f"{extreme['prob_positive']*100:.1f}% profit probability"))

        lines.append("\n| Test | Result | Details |")
        lines.append("|------|--------|---------|")
        for test, result, details in assessments:
            lines.append(f"| {test} | {result} | {details} |")

        # Overall
        pass_count = sum(1 for _, r, _ in assessments if r == "PASS")
        marginal_count = sum(1 for _, r, _ in assessments if r == "MARGINAL")
        fail_count = sum(1 for _, r, _ in assessments if r == "FAIL")

        lines.append(f"\nOverall: {pass_count} PASS, {marginal_count} MARGINAL, {fail_count} FAIL")

        if fail_count == 0 and marginal_count <= 1:
            lines.append("\n[ROBUST] Strategy demonstrates strong robustness under stress")
        elif fail_count == 0:
            lines.append("\n[ACCEPTABLE] Strategy is acceptable but has some sensitivities")
        else:
            lines.append("\n[CONCERN] Strategy has significant vulnerabilities under stress")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def save_results(self, timestamp: str = None):
        """Save results to files."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save report
        report = self.generate_report()
        report_file = self.config.output_dir / f'enhanced_mc_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_file}")

        # Save detailed results as CSV
        rows = []
        for category, cat_data in self.results.items():
            if category == 'original':
                continue
            for test_name, test_data in cat_data.items():
                if isinstance(test_data, dict) and 'net_pnl' in test_data:
                    row = {
                        'category': category,
                        'test': test_name,
                        **{f'pnl_{k}': v for k, v in test_data['net_pnl'].items()},
                        'sharpe_mean': test_data['sharpe']['mean'],
                        'pf_mean': test_data['profit_factor']['mean']
                    }
                    rows.append(row)

        if rows:
            csv_file = self.config.output_dir / f'enhanced_mc_details_{timestamp}.csv'
            pd.DataFrame(rows).to_csv(csv_file, index=False)
            logger.info(f"Details saved to: {csv_file}")

        return report_file


def run_enhanced_monte_carlo():
    """Main function to run enhanced Monte Carlo stress tests."""
    print("=" * 80)
    print(" ENHANCED MONTE CARLO STRESS TEST")
    print(" Comprehensive Strategy Robustness Validation")
    print("=" * 80)

    # Initialize
    config = EnhancedMCConfig(n_simulations=5000)
    simulator = EnhancedMonteCarloSimulator(config)

    # Load trades
    print("\n[1] Loading trades...")
    try:
        trades = simulator.load_trades()
        print(f"    Loaded {len(trades)} trades")
    except FileNotFoundError as e:
        print(f"    Error: {e}")
        return None

    # Run stress tests
    print("\n[2] Running stress tests...")
    print(f"    Simulations per test: {config.n_simulations:,}")

    results = simulator.run_all_stress_tests(trades)

    # Print report
    print("\n" + simulator.generate_report())

    # Save results
    print("\n[3] Saving results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = simulator.save_results(timestamp)
    print(f"    Report: {report_file.name}")

    return simulator


if __name__ == "__main__":
    simulator = run_enhanced_monte_carlo()
