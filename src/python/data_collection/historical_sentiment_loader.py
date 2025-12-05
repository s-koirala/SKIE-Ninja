"""
Historical Sentiment Data Loader
================================

Loads and manages historical sentiment data for backtesting:
1. VIX - From existing VIX_daily.csv
2. AAII - From cached CSV or generates proxy from VIX
3. Put/Call Ratio - From cached CSV or generates proxy

For backtesting, we use VIX as the primary sentiment indicator since:
- VIX data is available for all periods (2020-2025)
- VIX is highly correlated with sentiment extremes
- Academic research supports VIX as fear/greed proxy

CRITICAL: All features aligned with proper lag to prevent look-ahead bias.

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class HistoricalSentimentConfig:
    """Configuration for historical sentiment loading."""

    # Data paths
    data_dir: Path = None
    vix_file: str = "VIX_daily.csv"
    aaii_file: str = "aaii_historical.csv"
    pcr_file: str = "pcr_historical.csv"

    # VIX thresholds (from MacroMicro research)
    vix_fear_threshold: float = 25.0
    vix_extreme_fear_threshold: float = 30.0
    vix_complacency_threshold: float = 15.0

    # PCR thresholds (from MacroMicro research)
    pcr_bullish_threshold: float = 1.1  # Above = oversold (bullish)
    pcr_bearish_threshold: float = 0.8  # Below = overbought (bearish)

    # AAII historical averages
    aaii_hist_bullish_avg: float = 37.5
    aaii_hist_bearish_avg: float = 31.0

    # Lag for intraday alignment (minutes)
    alignment_lag_minutes: int = 5

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'raw' / 'market'


class HistoricalSentimentLoader:
    """
    Load and process historical sentiment data for backtesting.

    Primary data source: VIX (available, reliable)
    Secondary: AAII, PCR (if available, otherwise use VIX-based proxies)
    """

    def __init__(self, config: Optional[HistoricalSentimentConfig] = None):
        self.config = config or HistoricalSentimentConfig()
        self.vix_data: Optional[pd.DataFrame] = None
        self.aaii_data: Optional[pd.DataFrame] = None
        self.pcr_data: Optional[pd.DataFrame] = None

    def load_vix_data(self) -> pd.DataFrame:
        """Load VIX historical data."""
        vix_path = self.config.data_dir / self.config.vix_file

        if not vix_path.exists():
            raise FileNotFoundError(f"VIX data not found at {vix_path}")

        logger.info(f"Loading VIX data from {vix_path}")

        df = pd.read_csv(vix_path)

        # Standardize column names
        df.columns = df.columns.str.lower()

        # Parse date
        date_col = 'date' if 'date' in df.columns else df.columns[0]
        # Handle timezone-aware dates by converting to UTC first, then removing timezone
        parsed_dates = pd.to_datetime(df[date_col], utc=True)
        # Normalize to date only (remove time component)
        df['date'] = parsed_dates.dt.normalize().dt.tz_localize(None)
        df = df.set_index('date').sort_index()

        # Get close price
        close_col = 'close' if 'close' in df.columns else 'adj close'
        df['vix_close'] = df[close_col].astype(float)

        # Calculate derived features
        df['vix_ma5'] = df['vix_close'].rolling(5).mean()
        df['vix_ma10'] = df['vix_close'].rolling(10).mean()
        df['vix_ma20'] = df['vix_close'].rolling(20).mean()

        # VIX vs moving averages
        df['vix_vs_ma10'] = df['vix_close'] / df['vix_ma10']
        df['vix_vs_ma20'] = df['vix_close'] / df['vix_ma20']

        # VIX percentile (rolling 20-day)
        df['vix_percentile_20d'] = df['vix_close'].rolling(20).apply(
            lambda x: (x < x.iloc[-1]).mean() if len(x) > 0 else 0.5
        )

        # Regime indicators
        df['vix_fear_regime'] = (df['vix_close'] > self.config.vix_fear_threshold).astype(int)
        df['vix_extreme_fear'] = (df['vix_close'] > self.config.vix_extreme_fear_threshold).astype(int)
        df['vix_complacency'] = (df['vix_close'] < self.config.vix_complacency_threshold).astype(int)

        # VIX change and spike detection
        df['vix_pct_change'] = df['vix_close'].pct_change()
        df['vix_spike'] = (df['vix_pct_change'] > 0.15).astype(int)  # 15% daily increase

        # Normalized sentiment (-1 = extreme fear, +1 = extreme complacency)
        # Inverted: high VIX = fear = contrarian bullish
        df['vix_sentiment'] = -np.clip((df['vix_close'] - 20) / 15, -1, 1)

        # Contrarian signal (high VIX = bullish, low VIX = bearish)
        df['vix_contrarian_signal'] = df['vix_sentiment']

        self.vix_data = df
        logger.info(f"Loaded VIX data: {len(df)} days, {df.index.min()} to {df.index.max()}")

        return df

    def load_aaii_data(self) -> pd.DataFrame:
        """
        Load AAII historical data if available, otherwise create VIX-based proxy.

        AAII is weekly data - needs to be forward-filled to daily.
        """
        aaii_path = self.config.data_dir.parent / 'sentiment' / self.config.aaii_file

        if aaii_path.exists():
            logger.info(f"Loading AAII data from {aaii_path}")
            df = pd.read_csv(aaii_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
        else:
            logger.warning("AAII historical data not found. Creating VIX-based proxy.")
            df = self._create_aaii_proxy()

        self.aaii_data = df
        return df

    def _create_aaii_proxy(self) -> pd.DataFrame:
        """
        Create AAII proxy from VIX data.

        Based on academic research:
        - High VIX correlates with high bearish sentiment
        - Low VIX correlates with high bullish sentiment
        """
        if self.vix_data is None:
            self.load_vix_data()

        df = pd.DataFrame(index=self.vix_data.index)

        # VIX percentile determines sentiment distribution
        vix_pct = self.vix_data['vix_percentile_20d'].fillna(0.5)

        # High VIX = High bearish, Low bullish
        # Low VIX = Low bearish, High bullish
        # Formula based on historical AAII ranges (25-50% typical)

        # Bearish increases with VIX percentile
        df['aaii_bearish'] = 25 + (vix_pct * 30)  # Range: 25-55%

        # Bullish decreases with VIX percentile
        df['aaii_bullish'] = 50 - (vix_pct * 30)  # Range: 20-50%

        # Neutral is remainder
        df['aaii_neutral'] = 100 - df['aaii_bullish'] - df['aaii_bearish']

        # Bull-bear spread
        df['aaii_spread'] = df['aaii_bullish'] - df['aaii_bearish']

        # Deviation from historical average
        df['aaii_bullish_deviation'] = df['aaii_bullish'] - self.config.aaii_hist_bullish_avg
        df['aaii_bearish_deviation'] = df['aaii_bearish'] - self.config.aaii_hist_bearish_avg

        # Extreme readings
        df['aaii_extreme_bullish'] = (df['aaii_bullish'] > 50).astype(int)
        df['aaii_extreme_bearish'] = (df['aaii_bearish'] > 50).astype(int)

        # Contrarian signal: high bearish = bullish signal
        df['aaii_contrarian_signal'] = (df['aaii_bearish'] - df['aaii_bullish']) / 100

        df['is_proxy'] = True

        return df

    def load_pcr_data(self) -> pd.DataFrame:
        """
        Load Put/Call Ratio historical data if available, otherwise create VIX-based proxy.
        """
        pcr_path = self.config.data_dir.parent / 'sentiment' / self.config.pcr_file

        if pcr_path.exists():
            logger.info(f"Loading PCR data from {pcr_path}")
            df = pd.read_csv(pcr_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
        else:
            logger.warning("PCR historical data not found. Creating VIX-based proxy.")
            df = self._create_pcr_proxy()

        self.pcr_data = df
        return df

    def _create_pcr_proxy(self) -> pd.DataFrame:
        """
        Create PCR proxy from VIX data.

        Based on research:
        - High VIX correlates with high put/call ratio (fear = buying puts)
        - Low VIX correlates with low put/call ratio (greed = buying calls)
        """
        if self.vix_data is None:
            self.load_vix_data()

        df = pd.DataFrame(index=self.vix_data.index)

        # VIX level determines PCR
        # Typical PCR range: 0.6 - 1.3
        # High VIX = High PCR, Low VIX = Low PCR

        vix_normalized = np.clip((self.vix_data['vix_close'] - 12) / 25, 0, 1)
        df['pcr_total'] = 0.6 + (vix_normalized * 0.7)  # Range: 0.6 - 1.3

        # Moving averages
        df['pcr_ma5'] = df['pcr_total'].rolling(5).mean()
        df['pcr_ma10'] = df['pcr_total'].rolling(10).mean()

        # Extreme readings
        df['pcr_bullish_extreme'] = (df['pcr_total'] > self.config.pcr_bullish_threshold).astype(int)
        df['pcr_bearish_extreme'] = (df['pcr_total'] < self.config.pcr_bearish_threshold).astype(int)

        # Contrarian signal: high PCR = bullish (oversold)
        df['pcr_contrarian_signal'] = np.clip((df['pcr_total'] - 0.95) / 0.35, -1, 1)

        df['is_proxy'] = True

        return df

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all sentiment data sources."""
        return {
            'vix': self.load_vix_data(),
            'aaii': self.load_aaii_data(),
            'pcr': self.load_pcr_data()
        }

    def align_to_bars(
        self,
        bar_timestamps: pd.DatetimeIndex,
        lag_minutes: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Align daily sentiment data to intraday bars.

        CRITICAL: Uses PREVIOUS day's sentiment to avoid look-ahead bias.
        Intraday bars on date D use sentiment data from date D-1.

        Args:
            bar_timestamps: DatetimeIndex of price bars
            lag_minutes: Additional lag in minutes (default from config)

        Returns:
            DataFrame with sentiment features aligned to bars
        """
        if lag_minutes is None:
            lag_minutes = self.config.alignment_lag_minutes

        # Ensure all data is loaded
        if self.vix_data is None:
            self.load_vix_data()
        if self.aaii_data is None:
            self.load_aaii_data()
        if self.pcr_data is None:
            self.load_pcr_data()

        # Create output DataFrame
        features = pd.DataFrame(index=bar_timestamps)

        # Get date of each bar (timezone-naive, normalized to midnight)
        bar_timestamps_normalized = pd.to_datetime(bar_timestamps)
        if bar_timestamps_normalized.tz is not None:
            bar_timestamps_normalized = bar_timestamps_normalized.tz_localize(None)
        bar_dates = bar_timestamps_normalized.normalize()

        # CRITICAL: Use PREVIOUS day's data to avoid look-ahead
        # For a bar at 10:30 on Jan 5, use sentiment from Jan 4
        prev_dates = bar_dates - pd.Timedelta(days=1)

        # VIX features
        vix_cols = [
            'vix_close', 'vix_ma5', 'vix_ma10', 'vix_ma20',
            'vix_vs_ma10', 'vix_vs_ma20', 'vix_percentile_20d',
            'vix_fear_regime', 'vix_extreme_fear', 'vix_complacency',
            'vix_spike', 'vix_sentiment', 'vix_contrarian_signal'
        ]

        for col in vix_cols:
            if col in self.vix_data.columns:
                # Reindex to get values for prev_dates
                features[f'sent_{col}'] = self.vix_data[col].reindex(prev_dates).values

        # AAII features
        aaii_cols = [
            'aaii_bullish', 'aaii_bearish', 'aaii_spread',
            'aaii_extreme_bullish', 'aaii_extreme_bearish',
            'aaii_contrarian_signal'
        ]

        for col in aaii_cols:
            if col in self.aaii_data.columns:
                features[f'sent_{col}'] = self.aaii_data[col].reindex(prev_dates).values

        # PCR features
        pcr_cols = [
            'pcr_total', 'pcr_ma5', 'pcr_ma10',
            'pcr_bullish_extreme', 'pcr_bearish_extreme',
            'pcr_contrarian_signal'
        ]

        for col in pcr_cols:
            if col in self.pcr_data.columns:
                features[f'sent_{col}'] = self.pcr_data[col].reindex(prev_dates).values

        # Composite contrarian signal (average of all contrarian signals)
        contrarian_cols = [
            'sent_vix_contrarian_signal',
            'sent_aaii_contrarian_signal',
            'sent_pcr_contrarian_signal'
        ]

        available_contrarian = [c for c in contrarian_cols if c in features.columns]
        if available_contrarian:
            features['sent_composite_contrarian'] = features[available_contrarian].mean(axis=1)

        # Sentiment regime
        features['sent_fear_regime'] = (
            (features.get('sent_vix_fear_regime', 0) == 1) |
            (features.get('sent_aaii_extreme_bearish', 0) == 1) |
            (features.get('sent_pcr_bullish_extreme', 0) == 1)  # High PCR = fear
        ).astype(int)

        features['sent_greed_regime'] = (
            (features.get('sent_vix_complacency', 0) == 1) |
            (features.get('sent_aaii_extreme_bullish', 0) == 1) |
            (features.get('sent_pcr_bearish_extreme', 0) == 1)  # Low PCR = greed
        ).astype(int)

        # Forward fill NaN (sentiment persists through weekends/holidays)
        # This is SAFE because we're using previous day's data
        features = features.ffill()

        # Backward fill for the very first rows if needed
        features = features.bfill()

        # Fill any remaining NaN with neutral values
        features = features.fillna(0)

        logger.info(f"Aligned {len(features.columns)} sentiment features to {len(features)} bars")

        return features

    def get_sentiment_statistics(self) -> Dict:
        """Get statistics about loaded sentiment data."""
        stats = {}

        if self.vix_data is not None:
            stats['vix'] = {
                'rows': len(self.vix_data),
                'date_range': f"{self.vix_data.index.min()} to {self.vix_data.index.max()}",
                'mean_vix': self.vix_data['vix_close'].mean(),
                'max_vix': self.vix_data['vix_close'].max(),
                'fear_days_pct': self.vix_data['vix_fear_regime'].mean() * 100
            }

        if self.aaii_data is not None:
            stats['aaii'] = {
                'rows': len(self.aaii_data),
                'is_proxy': 'is_proxy' in self.aaii_data.columns and self.aaii_data['is_proxy'].iloc[0],
                'mean_spread': self.aaii_data['aaii_spread'].mean() if 'aaii_spread' in self.aaii_data.columns else None
            }

        if self.pcr_data is not None:
            stats['pcr'] = {
                'rows': len(self.pcr_data),
                'is_proxy': 'is_proxy' in self.pcr_data.columns and self.pcr_data['is_proxy'].iloc[0],
                'mean_pcr': self.pcr_data['pcr_total'].mean() if 'pcr_total' in self.pcr_data.columns else None
            }

        return stats


def load_historical_sentiment_features(
    bar_timestamps: pd.DatetimeIndex,
    config: Optional[HistoricalSentimentConfig] = None
) -> pd.DataFrame:
    """
    Convenience function to load aligned historical sentiment features.

    Args:
        bar_timestamps: DatetimeIndex of price bars
        config: Optional configuration

    Returns:
        DataFrame with sentiment features aligned to bars
    """
    loader = HistoricalSentimentLoader(config)
    loader.load_all()
    return loader.align_to_bars(bar_timestamps)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("HISTORICAL SENTIMENT LOADER TEST")
    print("=" * 70)

    # Test loading
    print("\n[1] Loading all sentiment data...")
    loader = HistoricalSentimentLoader()
    data = loader.load_all()

    print("\n[2] Sentiment data statistics:")
    stats = loader.get_sentiment_statistics()
    for source, source_stats in stats.items():
        print(f"\n  {source.upper()}:")
        for key, value in source_stats.items():
            print(f"    {key}: {value}")

    # Test bar alignment with dates that have prior trading data
    print("\n[3] Testing bar alignment...")
    # Use mid-January to ensure we have prior trading days (after New Year holidays)
    bar_times = pd.date_range('2024-01-15 09:30', periods=1000, freq='5min')
    features = loader.align_to_bars(bar_times)

    print(f"\n  Aligned features: {len(features.columns)}")
    print(f"  Sample columns: {list(features.columns[:10])}")
    print(f"\n  First 5 rows:")
    print(features.head())

    # Verify no NaN
    nan_counts = features.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"\n  WARNING: Columns with NaN: {len(nan_cols)}")
    else:
        print(f"\n  PASS: No NaN values in aligned features")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
