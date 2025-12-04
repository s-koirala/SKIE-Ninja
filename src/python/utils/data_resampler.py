"""
SKIE-Ninja Data Resampler

Utility for resampling 1-minute OHLCV data to higher timeframes (5-min, 15-min, etc.)
while preserving data integrity and applying RTH (Regular Trading Hours) filtering.

Author: SKIE_Ninja Development Team
Created: 2025-12-03
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
import logging

logger = logging.getLogger(__name__)


class DataResampler:
    """
    Resamples 1-minute OHLCV data to higher timeframes.

    Features:
    - Proper OHLCV aggregation (first open, max high, min low, last close, sum volume)
    - RTH (Regular Trading Hours) filtering
    - Timezone-aware processing
    - Quality validation and reporting
    """

    # ES/NQ Regular Trading Hours (Eastern Time)
    RTH_START = (9, 30)   # 9:30 AM ET
    RTH_END = (16, 0)     # 4:00 PM ET

    # Common timeframe mappings
    TIMEFRAME_MAP = {
        '5min': '5min',
        '5m': '5min',
        '5': '5min',
        '15min': '15min',
        '15m': '15min',
        '15': '15min',
        '30min': '30min',
        '30m': '30min',
        '30': '30min',
        '1h': '1h',
        '60min': '1h',
        '1hour': '1h',
    }

    def __init__(self, timezone: str = 'America/New_York'):
        """
        Initialize DataResampler.

        Args:
            timezone: Timezone for RTH filtering (default: America/New_York for ES/NQ)
        """
        self.timezone = timezone

    def resample(
        self,
        df: pd.DataFrame,
        target_timeframe: str,
        rth_only: bool = True,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to target timeframe.

        Args:
            df: DataFrame with OHLCV columns and DatetimeIndex
            target_timeframe: Target timeframe ('5min', '15min', '30min', '1h')
            rth_only: If True, filter to Regular Trading Hours only
            validate: If True, validate input data quality

        Returns:
            Resampled DataFrame with OHLCV columns
        """
        # Normalize timeframe string
        tf = self.TIMEFRAME_MAP.get(target_timeframe.lower(), target_timeframe)

        # Validate input
        if validate:
            self._validate_input(df)

        # Make a copy to avoid modifying original
        data = df.copy()

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Convert to timezone-aware if not already
        if data.index.tz is None:
            # Assume UTC if not specified
            data.index = data.index.tz_localize('UTC')

        # Convert to target timezone
        data.index = data.index.tz_convert(self.timezone)

        # Filter RTH if requested
        if rth_only:
            data = self._filter_rth(data)
            logger.info(f"After RTH filter: {len(data):,} bars")

        # Resample OHLCV data
        resampled = data.resample(tf).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        logger.info(f"Resampled from {len(df):,} to {len(resampled):,} bars ({tf})")

        return resampled

    def _filter_rth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to Regular Trading Hours only."""
        # Get hour and minute
        hours = df.index.hour
        minutes = df.index.minute

        # RTH: 9:30 AM - 4:00 PM ET (inclusive of 9:30, exclusive of 16:00)
        start_time = self.RTH_START[0] * 60 + self.RTH_START[1]  # 9:30 = 570 minutes
        end_time = self.RTH_END[0] * 60 + self.RTH_END[1]        # 16:00 = 960 minutes

        time_in_minutes = hours * 60 + minutes

        # Create RTH mask
        rth_mask = (time_in_minutes >= start_time) & (time_in_minutes < end_time)

        # Exclude weekends
        weekday_mask = df.index.weekday < 5

        return df[rth_mask & weekday_mask]

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        required_cols = ['open', 'high', 'low', 'close', 'volume']

        # Check for required columns
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Check for DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Index cannot be converted to datetime: {e}")

        # Check for NaN values
        nan_counts = df[required_cols].isna().sum()
        if nan_counts.any():
            logger.warning(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")

        # Check OHLC integrity
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        if invalid_ohlc.any():
            logger.warning(f"Found {invalid_ohlc.sum()} bars with invalid OHLC relationships")

    def get_resampling_stats(
        self,
        df: pd.DataFrame,
        target_timeframe: str,
        rth_only: bool = True
    ) -> Dict:
        """
        Get statistics about the resampling operation without performing it.

        Args:
            df: Input DataFrame
            target_timeframe: Target timeframe
            rth_only: Whether RTH filtering will be applied

        Returns:
            Dictionary with resampling statistics
        """
        tf = self.TIMEFRAME_MAP.get(target_timeframe.lower(), target_timeframe)

        # Parse timeframe to minutes
        if 'min' in tf:
            tf_minutes = int(tf.replace('min', ''))
        elif tf == '1h':
            tf_minutes = 60
        else:
            tf_minutes = int(tf)

        # Calculate expected bars
        input_bars = len(df)

        # RTH is 6.5 hours = 390 minutes
        rth_minutes_per_day = 390
        bars_per_day_rth = rth_minutes_per_day // tf_minutes

        # Non-RTH (globex): ~23 hours = 1380 minutes
        globex_minutes_per_day = 1380
        bars_per_day_globex = globex_minutes_per_day // tf_minutes

        # Estimate trading days in data
        if isinstance(df.index, pd.DatetimeIndex):
            trading_days = len(pd.unique(df.index.date))
        else:
            trading_days = len(pd.unique(pd.to_datetime(df.index).date))

        expected_bars_rth = trading_days * bars_per_day_rth
        expected_bars_all = trading_days * bars_per_day_globex

        # Calculate CV parameters
        embargo_bars = self._calculate_embargo(tf_minutes)
        rolling_window_min = bars_per_day_rth * 60  # ~60 trading days minimum
        rolling_window_max = bars_per_day_rth * 252  # ~1 year maximum

        return {
            'input_bars': input_bars,
            'target_timeframe': tf,
            'timeframe_minutes': tf_minutes,
            'trading_days': trading_days,
            'bars_per_day_rth': bars_per_day_rth,
            'bars_per_day_globex': bars_per_day_globex,
            'expected_output_rth': expected_bars_rth,
            'expected_output_all': expected_bars_all,
            'rth_hours': '9:30 AM - 4:00 PM ET',
            'cv_embargo_bars': embargo_bars,
            'cv_rolling_window_min': rolling_window_min,
            'cv_rolling_window_max': rolling_window_max,
        }

    def _calculate_embargo(self, timeframe_minutes: int) -> int:
        """Calculate embargo period in bars based on timeframe."""
        # We use features with up to 200-bar lookback
        # On 1-min: 200 bars = 200 minutes = 3.3 hours
        # Need to scale embargo proportionally
        base_embargo_minutes = 210  # ~3.5 hours worth of 1-min bars
        return max(1, base_embargo_minutes // timeframe_minutes)


def resample_ohlcv(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    target_timeframe: str = '5min',
    rth_only: bool = True
) -> pd.DataFrame:
    """
    Convenience function to resample a CSV file.

    Args:
        input_file: Path to input CSV with 1-min OHLCV data
        output_file: Optional path for output CSV (if None, returns DataFrame only)
        target_timeframe: Target timeframe ('5min', '15min', etc.)
        rth_only: Filter to RTH only

    Returns:
        Resampled DataFrame
    """
    input_path = Path(input_file)

    logger.info(f"Loading {input_path}...")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)

    # Ensure we have the right columns
    df.columns = df.columns.str.lower()

    # Resample
    resampler = DataResampler()
    resampled = resampler.resample(df, target_timeframe, rth_only=rth_only)

    # Save if output path provided
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resampled.to_csv(output_path)
        logger.info(f"Saved resampled data to {output_path}")

    return resampled


def compare_timeframes() -> None:
    """
    Print comparison table for different timeframes.
    Useful for deciding between 5-min and 15-min bars.
    """
    print("\n" + "="*70)
    print("TIMEFRAME COMPARISON FOR ES FUTURES (RTH ONLY)")
    print("="*70)
    print(f"{'Timeframe':<12} {'Bars/Day':<12} {'Embargo':<12} {'Rolling Win (min)':<18} {'Rolling Win (max)'}")
    print("-"*70)

    for tf, tf_min in [('1-min', 1), ('5-min', 5), ('15-min', 15), ('30-min', 30), ('1-hour', 60)]:
        bars_per_day = 390 // tf_min
        embargo = max(1, 210 // tf_min)
        rolling_min = bars_per_day * 60   # ~60 days
        rolling_max = bars_per_day * 252  # ~1 year

        print(f"{tf:<12} {bars_per_day:<12} {embargo:<12} {rolling_min:<18} {rolling_max}")

    print("-"*70)
    print("\nNotes:")
    print("- Embargo: Gap between train/test to prevent data leakage")
    print("- Rolling Window: Number of bars to use in rolling CV")
    print("- RTH = 6.5 hours (9:30 AM - 4:00 PM ET) = 390 minutes")
    print("="*70)


# Exports
__all__ = [
    'DataResampler',
    'resample_ohlcv',
    'compare_timeframes',
]


if __name__ == '__main__':
    # Demo usage
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    compare_timeframes()

    # Example resampling
    print("\n" + "="*70)
    print("EXAMPLE RESAMPLING")
    print("="*70)

    data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'raw' / 'market'
    es_file = data_dir / 'ES_1min_databento.csv'

    if es_file.exists():
        print(f"\nLoading {es_file}...")
        df = pd.read_csv(es_file, index_col=0, parse_dates=True)
        print(f"Input: {len(df):,} 1-min bars")

        resampler = DataResampler()

        # Get stats for 5-min and 15-min
        for tf in ['5min', '15min']:
            stats = resampler.get_resampling_stats(df, tf, rth_only=True)
            print(f"\n{tf} Statistics:")
            for key, val in stats.items():
                print(f"  {key}: {val:,}" if isinstance(val, int) else f"  {key}: {val}")
    else:
        print(f"Data file not found: {es_file}")
