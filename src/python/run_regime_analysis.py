"""
Regime-Specific Performance Analysis

Analyzes strategy performance across different market regimes:
1. Bull market vs Bear market
2. High volatility vs Low volatility
3. Trending vs Range-bound
4. Time-based (by year, quarter, month)

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
class RegimeConfig:
    """Configuration for regime analysis."""

    # VIX thresholds for volatility regime
    vix_low_threshold: float = 15.0
    vix_high_threshold: float = 25.0
    vix_extreme_threshold: float = 35.0

    # Trend thresholds (20-day return)
    bull_threshold: float = 0.05   # 5% 20-day return = bull
    bear_threshold: float = -0.05  # -5% 20-day return = bear

    # Output
    output_dir: Path = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = project_root / 'data' / 'regime_analysis'


class RegimeAnalyzer:
    """
    Analyzes strategy performance across market regimes.
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.trades_df = None
        self.prices_df = None
        self.results = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load trades and price data."""
        # Load trades
        results_dir = project_root / 'data' / 'backtest_results'
        for pattern in ['oos_*trades*.csv', 'vol_breakout*trades*.csv', '*trades*.csv']:
            trade_files = list(results_dir.glob(pattern))
            if trade_files:
                break

        if not trade_files:
            raise FileNotFoundError("No trade files found")

        trades_file = max(trade_files, key=lambda p: p.stat().st_mtime)
        self.trades_df = pd.read_csv(trades_file)
        self.trades_df['entry_time'] = pd.to_datetime(self.trades_df['entry_time'])
        logger.info(f"Loaded {len(self.trades_df)} trades from {trades_file.name}")

        # Load price data for regime classification
        market_dir = project_root / 'data' / 'raw' / 'market'

        # Try to load VIX data
        vix_file = market_dir / 'VIX_daily.csv'
        if vix_file.exists():
            self.vix_df = pd.read_csv(vix_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded VIX data: {len(self.vix_df)} days")
        else:
            self.vix_df = None
            logger.warning("VIX data not found - using volatility proxy")

        # Load ES price data for trend analysis
        es_files = list(market_dir.glob('ES_*databento*.csv'))
        if es_files:
            # Load and combine ES files
            dfs = []
            for f in es_files:
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                dfs.append(df)
            self.prices_df = pd.concat(dfs).sort_index()
            self.prices_df = self.prices_df[~self.prices_df.index.duplicated(keep='first')]
            logger.info(f"Loaded ES price data: {len(self.prices_df)} bars")
        else:
            self.prices_df = None
            logger.warning("ES price data not found")

        return self.trades_df, self.prices_df

    def classify_volatility_regime(self, trade_date: datetime) -> str:
        """Classify volatility regime at trade date."""
        if self.vix_df is not None:
            try:
                # Find closest VIX value
                date_only = trade_date.date() if hasattr(trade_date, 'date') else trade_date
                date_ts = pd.Timestamp(date_only)

                # Ensure VIX index is datetime
                vix_index = pd.to_datetime(self.vix_df.index)
                mask = vix_index <= date_ts
                if not mask.any():
                    return 'unknown'

                closest_date = vix_index[mask].max()
                vix = self.vix_df.loc[closest_date, 'close']
            except (KeyError, IndexError, Exception):
                return 'unknown'

            if vix >= self.config.vix_extreme_threshold:
                return 'extreme_vol'
            elif vix >= self.config.vix_high_threshold:
                return 'high_vol'
            elif vix <= self.config.vix_low_threshold:
                return 'low_vol'
            else:
                return 'normal_vol'
        else:
            return 'unknown'

    def classify_trend_regime(self, trade_date: datetime) -> str:
        """Classify trend regime at trade date."""
        if self.prices_df is None:
            return 'unknown'

        try:
            # Get 20-day return ending at trade date
            date_only = trade_date.date() if hasattr(trade_date, 'date') else trade_date
            date_ts = pd.Timestamp(date_only)

            # Filter prices up to trade date
            prices_index = pd.to_datetime(self.prices_df.index)
            mask = prices_index <= date_ts
            if not mask.any():
                return 'unknown'

            filtered_prices = self.prices_df.loc[mask]

            # Resample to daily
            daily = filtered_prices['close'].resample('D').last().dropna()
            if len(daily) < 20:
                return 'unknown'

            return_20d = (daily.iloc[-1] / daily.iloc[-20]) - 1

            if return_20d >= self.config.bull_threshold:
                return 'bull'
            elif return_20d <= self.config.bear_threshold:
                return 'bear'
            else:
                return 'sideways'
        except Exception as e:
            logger.debug(f"Trend classification error: {e}")
            return 'unknown'

    def calculate_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate performance metrics for a subset of trades."""
        if len(trades) == 0:
            return {
                'net_pnl': 0, 'trades': 0, 'win_rate': 0,
                'profit_factor': 0, 'sharpe': 0, 'max_dd': 0,
                'avg_trade': 0, 'pct_of_total_pnl': 0
            }

        net_pnl = trades['net_pnl'].sum()
        total_trades = len(trades)
        win_rate = (trades['net_pnl'] > 0).mean()

        winners = trades[trades['net_pnl'] > 0]['net_pnl']
        losers = trades[trades['net_pnl'] < 0]['net_pnl']

        gross_profit = winners.sum() if len(winners) > 0 else 0
        gross_loss = abs(losers.sum()) if len(losers) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        avg_trade = net_pnl / total_trades if total_trades > 0 else 0

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
            'max_dd': max_dd,
            'avg_trade': avg_trade
        }

    def analyze_by_volatility_regime(self) -> pd.DataFrame:
        """Analyze performance by volatility regime."""
        logger.info("Analyzing by volatility regime...")

        # Classify each trade
        trades = self.trades_df.copy()
        trades['vol_regime'] = trades['entry_time'].apply(self.classify_volatility_regime)

        results = []
        total_pnl = trades['net_pnl'].sum()

        for regime in ['low_vol', 'normal_vol', 'high_vol', 'extreme_vol', 'unknown']:
            regime_trades = trades[trades['vol_regime'] == regime]
            if len(regime_trades) > 0:
                metrics = self.calculate_metrics(regime_trades)
                metrics['regime'] = regime
                metrics['pct_of_trades'] = len(regime_trades) / len(trades) * 100
                metrics['pct_of_pnl'] = metrics['net_pnl'] / total_pnl * 100 if total_pnl != 0 else 0
                results.append(metrics)

        return pd.DataFrame(results)

    def analyze_by_trend_regime(self) -> pd.DataFrame:
        """Analyze performance by trend regime."""
        logger.info("Analyzing by trend regime...")

        trades = self.trades_df.copy()
        trades['trend_regime'] = trades['entry_time'].apply(self.classify_trend_regime)

        results = []
        total_pnl = trades['net_pnl'].sum()

        for regime in ['bull', 'sideways', 'bear', 'unknown']:
            regime_trades = trades[trades['trend_regime'] == regime]
            if len(regime_trades) > 0:
                metrics = self.calculate_metrics(regime_trades)
                metrics['regime'] = regime
                metrics['pct_of_trades'] = len(regime_trades) / len(trades) * 100
                metrics['pct_of_pnl'] = metrics['net_pnl'] / total_pnl * 100 if total_pnl != 0 else 0
                results.append(metrics)

        return pd.DataFrame(results)

    def analyze_by_time_period(self) -> Dict[str, pd.DataFrame]:
        """Analyze performance by time periods."""
        logger.info("Analyzing by time period...")

        trades = self.trades_df.copy()
        trades['entry_date'] = pd.to_datetime(trades['entry_time'])
        trades['year'] = trades['entry_date'].dt.year
        trades['quarter'] = trades['entry_date'].dt.to_period('Q').astype(str)
        trades['month'] = trades['entry_date'].dt.to_period('M').astype(str)

        results = {}
        total_pnl = trades['net_pnl'].sum()

        # By year
        yearly_results = []
        for year in sorted(trades['year'].unique()):
            year_trades = trades[trades['year'] == year]
            metrics = self.calculate_metrics(year_trades)
            metrics['period'] = str(year)
            metrics['pct_of_trades'] = len(year_trades) / len(trades) * 100
            metrics['pct_of_pnl'] = metrics['net_pnl'] / total_pnl * 100 if total_pnl != 0 else 0
            yearly_results.append(metrics)
        results['yearly'] = pd.DataFrame(yearly_results)

        # By quarter
        quarterly_results = []
        for quarter in sorted(trades['quarter'].unique()):
            q_trades = trades[trades['quarter'] == quarter]
            metrics = self.calculate_metrics(q_trades)
            metrics['period'] = quarter
            quarterly_results.append(metrics)
        results['quarterly'] = pd.DataFrame(quarterly_results)

        return results

    def analyze_worst_periods(self, window_days: int = 30) -> pd.DataFrame:
        """Find worst performing periods."""
        logger.info(f"Analyzing worst {window_days}-day periods...")

        trades = self.trades_df.copy()
        trades['entry_date'] = pd.to_datetime(trades['entry_time']).dt.date
        trades = trades.sort_values('entry_date')

        # Calculate rolling window P&L
        daily_pnl = trades.groupby('entry_date')['net_pnl'].sum()

        if len(daily_pnl) < window_days:
            return pd.DataFrame()

        rolling_pnl = daily_pnl.rolling(window=window_days).sum()

        # Find worst periods
        worst_periods = rolling_pnl.nsmallest(10)

        results = []
        for end_date, pnl in worst_periods.items():
            start_date = end_date - pd.Timedelta(days=window_days)
            period_trades = trades[
                (trades['entry_date'] >= start_date) &
                (trades['entry_date'] <= end_date)
            ]
            results.append({
                'start_date': start_date,
                'end_date': end_date,
                'net_pnl': pnl,
                'trades': len(period_trades),
                'win_rate': (period_trades['net_pnl'] > 0).mean() if len(period_trades) > 0 else 0
            })

        return pd.DataFrame(results)

    def run_full_analysis(self) -> Dict:
        """Run complete regime analysis."""
        results = {}

        results['volatility_regime'] = self.analyze_by_volatility_regime()
        results['trend_regime'] = self.analyze_by_trend_regime()
        results['time_period'] = self.analyze_by_time_period()
        results['worst_periods'] = self.analyze_worst_periods()

        self.results = results
        return results

    def generate_report(self) -> str:
        """Generate regime analysis report."""
        if not self.results:
            return "No results available. Run analysis first."

        lines = []
        lines.append("=" * 80)
        lines.append(" REGIME-SPECIFIC PERFORMANCE ANALYSIS")
        lines.append(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)

        # Overall stats
        total_pnl = self.trades_df['net_pnl'].sum()
        total_trades = len(self.trades_df)
        lines.append(f"\n--- OVERALL PERFORMANCE ---")
        lines.append(f"  Total P&L: ${total_pnl:,.0f}")
        lines.append(f"  Total Trades: {total_trades:,}")

        # Volatility regime
        lines.append("\n--- PERFORMANCE BY VOLATILITY REGIME ---")
        vol_df = self.results['volatility_regime']
        if len(vol_df) > 0:
            lines.append(f"| {'Regime':<12} | {'Trades':>8} | {'Net P&L':>12} | {'Win Rate':>9} | {'Sharpe':>7} | {'% of P&L':>8} |")
            lines.append("|" + "-" * 74 + "|")
            for _, row in vol_df.iterrows():
                lines.append(
                    f"| {row['regime']:<12} | {row['trades']:>8.0f} | ${row['net_pnl']:>10,.0f} | "
                    f"{row['win_rate']*100:>8.1f}% | {row['sharpe']:>7.2f} | {row['pct_of_pnl']:>7.1f}% |"
                )

        # Trend regime
        lines.append("\n--- PERFORMANCE BY TREND REGIME ---")
        trend_df = self.results['trend_regime']
        if len(trend_df) > 0:
            lines.append(f"| {'Regime':<12} | {'Trades':>8} | {'Net P&L':>12} | {'Win Rate':>9} | {'Sharpe':>7} | {'% of P&L':>8} |")
            lines.append("|" + "-" * 74 + "|")
            for _, row in trend_df.iterrows():
                lines.append(
                    f"| {row['regime']:<12} | {row['trades']:>8.0f} | ${row['net_pnl']:>10,.0f} | "
                    f"{row['win_rate']*100:>8.1f}% | {row['sharpe']:>7.2f} | {row['pct_of_pnl']:>7.1f}% |"
                )

        # Yearly performance
        lines.append("\n--- PERFORMANCE BY YEAR ---")
        yearly_df = self.results['time_period']['yearly']
        if len(yearly_df) > 0:
            lines.append(f"| {'Year':<8} | {'Trades':>8} | {'Net P&L':>12} | {'Win Rate':>9} | {'Sharpe':>7} | {'Max DD':>10} |")
            lines.append("|" + "-" * 70 + "|")
            for _, row in yearly_df.iterrows():
                lines.append(
                    f"| {row['period']:<8} | {row['trades']:>8.0f} | ${row['net_pnl']:>10,.0f} | "
                    f"{row['win_rate']*100:>8.1f}% | {row['sharpe']:>7.2f} | ${row['max_dd']:>9,.0f} |"
                )

        # Worst periods
        lines.append("\n--- WORST 30-DAY PERIODS ---")
        worst_df = self.results['worst_periods']
        if len(worst_df) > 0:
            lines.append(f"| {'Start':<12} | {'End':<12} | {'Net P&L':>12} | {'Trades':>8} | {'Win Rate':>9} |")
            lines.append("|" + "-" * 62 + "|")
            for _, row in worst_df.head(5).iterrows():
                lines.append(
                    f"| {str(row['start_date']):<12} | {str(row['end_date']):<12} | ${row['net_pnl']:>10,.0f} | "
                    f"{row['trades']:>8.0f} | {row['win_rate']*100:>8.1f}% |"
                )

        # Assessment
        lines.append("\n" + "=" * 80)
        lines.append(" REGIME ROBUSTNESS ASSESSMENT")
        lines.append("=" * 80)

        # Check for concerning patterns
        concerns = []

        # Check if any regime has negative P&L
        for df_name, df in [('volatility', vol_df), ('trend', trend_df)]:
            if len(df) > 0:
                negative_regimes = df[df['net_pnl'] < 0]
                if len(negative_regimes) > 0:
                    for _, row in negative_regimes.iterrows():
                        if row['regime'] != 'unknown':
                            concerns.append(f"Negative P&L in {row['regime']} regime (${row['net_pnl']:,.0f})")

        # Check for highly concentrated P&L
        if len(vol_df) > 0:
            max_pct = vol_df['pct_of_pnl'].max()
            if max_pct > 80:
                concerns.append(f"P&L concentrated in one regime ({max_pct:.0f}%)")

        # Check for inconsistent yearly performance
        if len(yearly_df) > 0:
            yearly_pnls = yearly_df['net_pnl']
            if (yearly_pnls < 0).any():
                neg_years = yearly_df[yearly_df['net_pnl'] < 0]['period'].tolist()
                concerns.append(f"Negative years: {', '.join(neg_years)}")

        if concerns:
            lines.append("\nCONCERNS IDENTIFIED:")
            for concern in concerns:
                lines.append(f"  - {concern}")
        else:
            lines.append("\n[ROBUST] Strategy performs consistently across regimes")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def save_results(self, timestamp: str = None):
        """Save results to files."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save report
        report = self.generate_report()
        report_file = self.config.output_dir / f'regime_analysis_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_file}")

        # Save detailed results
        for name, data in self.results.items():
            if isinstance(data, pd.DataFrame):
                csv_file = self.config.output_dir / f'regime_{name}_{timestamp}.csv'
                data.to_csv(csv_file, index=False)
            elif isinstance(data, dict):
                for sub_name, sub_df in data.items():
                    if isinstance(sub_df, pd.DataFrame):
                        csv_file = self.config.output_dir / f'regime_{name}_{sub_name}_{timestamp}.csv'
                        sub_df.to_csv(csv_file, index=False)

        logger.info(f"All results saved to: {self.config.output_dir}")

        return report_file


def run_regime_analysis():
    """Main function to run regime analysis."""
    print("=" * 80)
    print(" REGIME-SPECIFIC PERFORMANCE ANALYSIS")
    print(" Analyzing Strategy Across Market Conditions")
    print("=" * 80)

    # Initialize
    config = RegimeConfig()
    analyzer = RegimeAnalyzer(config)

    # Load data
    print("\n[1] Loading data...")
    try:
        trades, prices = analyzer.load_data()
        print(f"    Loaded {len(trades)} trades")
    except FileNotFoundError as e:
        print(f"    Error: {e}")
        return None

    # Run analysis
    print("\n[2] Running regime analysis...")
    results = analyzer.run_full_analysis()

    # Print report
    print("\n" + analyzer.generate_report())

    # Save results
    print("\n[3] Saving results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = analyzer.save_results(timestamp)
    print(f"    Report: {report_file.name}")

    return analyzer


if __name__ == "__main__":
    analyzer = run_regime_analysis()
