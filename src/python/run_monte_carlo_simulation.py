"""
Monte Carlo Simulation for Ensemble Strategy

Validates strategy robustness by:
1. Trade resampling (bootstrap) - Shuffle trade order
2. Random trade dropout - Remove random trades
3. Slippage/cost variance - Add cost uncertainty
4. Return distribution analysis - Confidence intervals

Following Lopez de Prado's guidelines for statistical significance.

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
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""

    # Simulation parameters
    n_simulations: int = 10000           # Number of MC iterations
    confidence_level: float = 0.95       # For confidence intervals

    # Bootstrap settings
    bootstrap_sample_pct: float = 1.0    # Sample 100% with replacement

    # Trade dropout settings
    dropout_min: float = 0.0             # Min trade dropout rate
    dropout_max: float = 0.15            # Max trade dropout rate (15%)

    # Cost variance settings
    slippage_variance: float = 0.25      # +/- 25% slippage variance
    commission_variance: float = 0.10    # +/- 10% commission variance

    # Output
    save_results: bool = True
    output_dir: Path = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = project_root / 'data' / 'monte_carlo_results'


class MonteCarloSimulator:
    """
    Monte Carlo simulator for trading strategy validation.

    Methods:
    1. Bootstrap resampling - Tests if results are order-dependent
    2. Trade dropout - Tests sensitivity to missing trades
    3. Cost variance - Tests sensitivity to transaction costs
    4. Combined simulation - All factors together
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None):
        self.config = config or MonteCarloConfig()
        self.results = {}

    def load_trades(self, trades_file: Optional[Path] = None) -> pd.DataFrame:
        """Load trades from CSV or find most recent."""
        if trades_file is None:
            # Find most recent ensemble trades file
            results_dir = project_root / 'data' / 'backtest_results'
            trade_files = list(results_dir.glob('ensemble_*trades*.csv'))

            if not trade_files:
                raise FileNotFoundError("No ensemble trade files found")

            # Sort by modification time and get most recent
            trades_file = max(trade_files, key=lambda p: p.stat().st_mtime)

        logger.info(f"Loading trades from: {trades_file.name}")
        trades_df = pd.read_csv(trades_file)
        logger.info(f"Loaded {len(trades_df)} trades")

        return trades_df

    def calculate_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate key metrics from trades."""
        if len(trades) == 0:
            return {
                'net_pnl': 0, 'trades': 0, 'win_rate': 0,
                'profit_factor': 0, 'sharpe': 0, 'max_dd': 0
            }

        net_pnl = trades['net_pnl'].sum()
        total_trades = len(trades)
        win_rate = (trades['net_pnl'] > 0).mean()

        gross_profit = trades[trades['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(trades[trades['net_pnl'] < 0]['net_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Max drawdown
        cumulative = trades['net_pnl'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = running_max - cumulative
        max_dd = drawdown.max()

        # Sharpe (daily)
        if 'entry_time' in trades.columns:
            trades_copy = trades.copy()
            trades_copy['date'] = pd.to_datetime(trades_copy['entry_time']).dt.date
            daily_pnl = trades_copy.groupby('date')['net_pnl'].sum()
            sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if daily_pnl.std() > 0 else 0
        else:
            sharpe = 0

        return {
            'net_pnl': net_pnl,
            'trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe': sharpe,
            'max_dd': max_dd
        }

    def bootstrap_resample(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Resample trades with replacement (bootstrap)."""
        n_samples = int(len(trades) * self.config.bootstrap_sample_pct)
        indices = np.random.choice(len(trades), size=n_samples, replace=True)
        return trades.iloc[indices].reset_index(drop=True)

    def apply_dropout(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Randomly drop trades to test sensitivity."""
        dropout_rate = np.random.uniform(
            self.config.dropout_min,
            self.config.dropout_max
        )
        keep_mask = np.random.random(len(trades)) > dropout_rate
        return trades[keep_mask].reset_index(drop=True)

    def apply_cost_variance(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Apply random variance to transaction costs."""
        trades = trades.copy()

        # Vary slippage
        slippage_mult = 1 + np.random.uniform(
            -self.config.slippage_variance,
            self.config.slippage_variance,
            len(trades)
        )

        # Vary commission
        commission_mult = 1 + np.random.uniform(
            -self.config.commission_variance,
            self.config.commission_variance,
            len(trades)
        )

        # Recalculate net P&L
        if 'slippage' in trades.columns and 'commission' in trades.columns:
            original_slippage = trades['slippage']
            original_commission = trades['commission']

            new_slippage = original_slippage * slippage_mult
            new_commission = original_commission * commission_mult

            cost_diff = (new_slippage - original_slippage) + (new_commission - original_commission)
            trades['net_pnl'] = trades['net_pnl'] - cost_diff

        return trades

    def run_bootstrap_simulation(self, trades: pd.DataFrame) -> Dict:
        """Run bootstrap resampling simulation."""
        logger.info(f"Running bootstrap simulation ({self.config.n_simulations} iterations)...")

        results = []
        for i in range(self.config.n_simulations):
            resampled = self.bootstrap_resample(trades)
            metrics = self.calculate_metrics(resampled)
            results.append(metrics)

            if (i + 1) % 1000 == 0:
                logger.info(f"  Bootstrap progress: {i+1}/{self.config.n_simulations}")

        return self._summarize_results(results, "Bootstrap")

    def run_dropout_simulation(self, trades: pd.DataFrame) -> Dict:
        """Run trade dropout simulation."""
        logger.info(f"Running dropout simulation ({self.config.n_simulations} iterations)...")

        results = []
        for i in range(self.config.n_simulations):
            dropped = self.apply_dropout(trades)
            metrics = self.calculate_metrics(dropped)
            results.append(metrics)

            if (i + 1) % 1000 == 0:
                logger.info(f"  Dropout progress: {i+1}/{self.config.n_simulations}")

        return self._summarize_results(results, "Dropout")

    def run_cost_variance_simulation(self, trades: pd.DataFrame) -> Dict:
        """Run cost variance simulation."""
        logger.info(f"Running cost variance simulation ({self.config.n_simulations} iterations)...")

        results = []
        for i in range(self.config.n_simulations):
            varied = self.apply_cost_variance(trades)
            metrics = self.calculate_metrics(varied)
            results.append(metrics)

            if (i + 1) % 1000 == 0:
                logger.info(f"  Cost variance progress: {i+1}/{self.config.n_simulations}")

        return self._summarize_results(results, "Cost Variance")

    def run_combined_simulation(self, trades: pd.DataFrame) -> Dict:
        """Run combined simulation with all factors."""
        logger.info(f"Running COMBINED simulation ({self.config.n_simulations} iterations)...")

        results = []
        for i in range(self.config.n_simulations):
            # Apply all transformations
            simulated = self.bootstrap_resample(trades)
            simulated = self.apply_dropout(simulated)
            simulated = self.apply_cost_variance(simulated)

            metrics = self.calculate_metrics(simulated)
            results.append(metrics)

            if (i + 1) % 1000 == 0:
                logger.info(f"  Combined progress: {i+1}/{self.config.n_simulations}")

        return self._summarize_results(results, "Combined")

    def _summarize_results(self, results: List[Dict], sim_type: str) -> Dict:
        """Summarize simulation results with confidence intervals."""
        results_df = pd.DataFrame(results)

        alpha = 1 - self.config.confidence_level
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100

        summary = {
            'type': sim_type,
            'n_simulations': len(results),
            'metrics': {}
        }

        for col in results_df.columns:
            values = results_df[col].values
            summary['metrics'][col] = {
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

    def run_all_simulations(self, trades: pd.DataFrame) -> Dict:
        """Run all Monte Carlo simulations."""
        original_metrics = self.calculate_metrics(trades)

        results = {
            'original': original_metrics,
            'bootstrap': self.run_bootstrap_simulation(trades),
            'dropout': self.run_dropout_simulation(trades),
            'cost_variance': self.run_cost_variance_simulation(trades),
            'combined': self.run_combined_simulation(trades)
        }

        self.results = results
        return results

    def print_results(self):
        """Print formatted results."""
        if not self.results:
            logger.warning("No results to print. Run simulations first.")
            return

        print("\n" + "=" * 80)
        print(" MONTE CARLO SIMULATION RESULTS")
        print(" Strategy: Ensemble (Volatility Breakout + Sentiment)")
        print("=" * 80)

        # Original metrics
        orig = self.results['original']
        print(f"\n--- Original Strategy Performance ---")
        print(f"  Net P&L:       ${orig['net_pnl']:,.2f}")
        print(f"  Trades:        {orig['trades']}")
        print(f"  Win Rate:      {orig['win_rate']*100:.1f}%")
        print(f"  Profit Factor: {orig['profit_factor']:.2f}")
        print(f"  Sharpe Ratio:  {orig['sharpe']:.2f}")
        print(f"  Max Drawdown:  ${orig['max_dd']:,.2f}")

        # Simulation summaries
        for sim_type in ['bootstrap', 'dropout', 'cost_variance', 'combined']:
            sim = self.results[sim_type]
            pnl = sim['metrics']['net_pnl']

            print(f"\n--- {sim['type']} Simulation ({sim['n_simulations']:,} iterations) ---")
            print(f"  Net P&L:")
            print(f"    Mean:           ${pnl['mean']:,.2f}")
            print(f"    Std Dev:        ${pnl['std']:,.2f}")
            print(f"    95% CI:         [${pnl['ci_lower']:,.2f}, ${pnl['ci_upper']:,.2f}]")
            print(f"    Min/Max:        [${pnl['min']:,.2f}, ${pnl['max']:,.2f}]")
            if pnl['prob_positive'] is not None:
                print(f"    P(Profit > 0):  {pnl['prob_positive']*100:.1f}%")

            sharpe = sim['metrics']['sharpe']
            print(f"  Sharpe Ratio:")
            print(f"    Mean:           {sharpe['mean']:.2f}")
            print(f"    95% CI:         [{sharpe['ci_lower']:.2f}, {sharpe['ci_upper']:.2f}]")

        # Statistical significance assessment
        print("\n" + "=" * 80)
        print(" STATISTICAL SIGNIFICANCE ASSESSMENT")
        print("=" * 80)

        combined = self.results['combined']['metrics']

        # Probability of profit
        prob_profit = combined['net_pnl']['prob_positive']
        print(f"\n  Probability of Positive Net P&L: {prob_profit*100:.1f}%")

        # Sharpe > 0
        sharpe_positive = (combined['sharpe']['ci_lower'] > 0)
        print(f"  95% CI Sharpe > 0: {'YES' if sharpe_positive else 'NO'}")

        # Profit factor > 1
        pf_above_1 = combined['profit_factor']['ci_lower'] > 1.0
        print(f"  95% CI Profit Factor > 1: {'YES' if pf_above_1 else 'NO'}")

        # Overall assessment
        print(f"\n  --- OVERALL ASSESSMENT ---")
        if prob_profit >= 0.95 and sharpe_positive and pf_above_1:
            print(f"  [PASS] Strategy is STATISTICALLY ROBUST")
            print(f"         - 95%+ probability of profit")
            print(f"         - Sharpe ratio significantly > 0")
            print(f"         - Profit factor significantly > 1")
        elif prob_profit >= 0.80:
            print(f"  [CAUTION] Strategy shows promise but needs monitoring")
            print(f"         - {prob_profit*100:.0f}% probability of profit")
        else:
            print(f"  [FAIL] Strategy may not be statistically robust")
            print(f"         - Only {prob_profit*100:.0f}% probability of profit")

        print("\n" + "=" * 80)

    def save_results(self, timestamp: str = None):
        """Save results to files."""
        if not self.results:
            logger.warning("No results to save.")
            return

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save summary
        summary_file = self.config.output_dir / f'monte_carlo_summary_{timestamp}.txt'

        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(" MONTE CARLO SIMULATION RESULTS\n")
            f.write(" Strategy: Ensemble (Volatility Breakout + Sentiment)\n")
            f.write(f" Timestamp: {timestamp}\n")
            f.write("=" * 80 + "\n\n")

            # Original metrics
            orig = self.results['original']
            f.write("--- Original Strategy Performance ---\n")
            f.write(f"  Net P&L:       ${orig['net_pnl']:,.2f}\n")
            f.write(f"  Trades:        {orig['trades']}\n")
            f.write(f"  Win Rate:      {orig['win_rate']*100:.1f}%\n")
            f.write(f"  Profit Factor: {orig['profit_factor']:.2f}\n")
            f.write(f"  Sharpe Ratio:  {orig['sharpe']:.2f}\n")
            f.write(f"  Max Drawdown:  ${orig['max_dd']:,.2f}\n\n")

            # Simulation summaries
            for sim_type in ['bootstrap', 'dropout', 'cost_variance', 'combined']:
                sim = self.results[sim_type]
                pnl = sim['metrics']['net_pnl']
                sharpe = sim['metrics']['sharpe']

                f.write(f"--- {sim['type']} Simulation ({sim['n_simulations']:,} iterations) ---\n")
                f.write(f"  Net P&L:\n")
                f.write(f"    Mean:           ${pnl['mean']:,.2f}\n")
                f.write(f"    Std Dev:        ${pnl['std']:,.2f}\n")
                f.write(f"    95% CI:         [${pnl['ci_lower']:,.2f}, ${pnl['ci_upper']:,.2f}]\n")
                f.write(f"    P(Profit > 0):  {pnl['prob_positive']*100:.1f}%\n")
                f.write(f"  Sharpe Ratio:\n")
                f.write(f"    Mean:           {sharpe['mean']:.2f}\n")
                f.write(f"    95% CI:         [{sharpe['ci_lower']:.2f}, {sharpe['ci_upper']:.2f}]\n\n")

            # Assessment
            combined = self.results['combined']['metrics']
            prob_profit = combined['net_pnl']['prob_positive']
            sharpe_positive = combined['sharpe']['ci_lower'] > 0
            pf_above_1 = combined['profit_factor']['ci_lower'] > 1.0

            f.write("=" * 80 + "\n")
            f.write(" STATISTICAL SIGNIFICANCE ASSESSMENT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"  Probability of Positive Net P&L: {prob_profit*100:.1f}%\n")
            f.write(f"  95% CI Sharpe > 0: {'YES' if sharpe_positive else 'NO'}\n")
            f.write(f"  95% CI Profit Factor > 1: {'YES' if pf_above_1 else 'NO'}\n\n")

            if prob_profit >= 0.95 and sharpe_positive and pf_above_1:
                f.write("  [PASS] Strategy is STATISTICALLY ROBUST\n")
            elif prob_profit >= 0.80:
                f.write("  [CAUTION] Strategy shows promise but needs monitoring\n")
            else:
                f.write("  [FAIL] Strategy may not be statistically robust\n")

        logger.info(f"Results saved to: {summary_file}")

        # Save detailed CSV
        csv_file = self.config.output_dir / f'monte_carlo_details_{timestamp}.csv'

        rows = []
        for sim_type in ['bootstrap', 'dropout', 'cost_variance', 'combined']:
            sim = self.results[sim_type]
            for metric, values in sim['metrics'].items():
                if isinstance(values, dict):
                    rows.append({
                        'simulation_type': sim_type,
                        'metric': metric,
                        **values
                    })

        pd.DataFrame(rows).to_csv(csv_file, index=False)
        logger.info(f"Details saved to: {csv_file}")

        return summary_file, csv_file


def run_monte_carlo():
    """Run Monte Carlo simulation on ensemble strategy trades."""
    print("=" * 80)
    print(" MONTE CARLO SIMULATION")
    print(" Ensemble Strategy Robustness Testing")
    print("=" * 80)

    # Initialize simulator
    config = MonteCarloConfig(
        n_simulations=10000,
        confidence_level=0.95,
        dropout_max=0.15,
        slippage_variance=0.25
    )
    simulator = MonteCarloSimulator(config)

    # Load trades
    print("\n[1] Loading trades...")
    try:
        trades = simulator.load_trades()
        print(f"    Loaded {len(trades)} trades")
    except FileNotFoundError as e:
        print(f"    Error: {e}")
        print("    Please run ensemble backtest first.")
        return None

    # Run simulations
    print("\n[2] Running Monte Carlo simulations...")
    print(f"    Iterations per simulation: {config.n_simulations:,}")

    results = simulator.run_all_simulations(trades)

    # Print results
    simulator.print_results()

    # Save results
    print("\n[3] Saving results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file, csv_file = simulator.save_results(timestamp)

    print(f"\n    Summary: {summary_file.name}")
    print(f"    Details: {csv_file.name}")

    return simulator


if __name__ == "__main__":
    simulator = run_monte_carlo()
