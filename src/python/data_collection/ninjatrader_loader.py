"""
NinjaTrader Data Loader
=======================
Parses NinjaTrader's semicolon-separated data format from PortaraNinja.

Supports:
- 1-minute OHLCV data (continuous contracts)
- Tick data (trades with timestamps)

Reference: https://portaraninja.com/historical-intraday-tick-data/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Literal
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NinjaTraderLoader:
    """
    Load and parse NinjaTrader historical data files.

    Supports PortaraNinja semicolon-separated format for:
    - 1-minute bars: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
    - Tick data: YYYYMMDD HHMMSS Ticks;Price;Volume
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the loader.

        Args:
            data_dir: Base directory for data files. Defaults to project data/raw/market.
        """
        if data_dir is None:
            # Default to project structure
            self.data_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "market"
        else:
            self.data_dir = Path(data_dir)

        logger.info(f"NinjaTraderLoader initialized with data_dir: {self.data_dir}")

    def load_minute_data(
        self,
        filepath: Union[str, Path],
        symbol: str = "ES",
        timezone: str = "America/Chicago"
    ) -> pd.DataFrame:
        """
        Load 1-minute OHLCV bar data.

        Args:
            filepath: Path to the data file (relative to data_dir or absolute)
            symbol: Instrument symbol (for column naming)
            timezone: Timezone for timestamps (PortaraNinja uses Chicago time)

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume
        """
        filepath = self._resolve_path(filepath)
        logger.info(f"Loading minute data from: {filepath}")

        # Read semicolon-separated file
        # Format: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
        df = pd.read_csv(
            filepath,
            sep=';',
            header=None,
            names=['datetime_str', 'open', 'high', 'low', 'close', 'volume'],
            dtype={
                'datetime_str': str,
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': int
            }
        )

        # Parse datetime - format is "YYYYMMDD HHMMSS"
        df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y%m%d %H%M%S')
        df = df.drop('datetime_str', axis=1)

        # Set datetime as index
        df = df.set_index('datetime')

        # Localize to Chicago time (PortaraNinja default)
        df.index = df.index.tz_localize(timezone)

        # Add symbol column
        df['symbol'] = symbol

        # Reorder columns
        df = df[['symbol', 'open', 'high', 'low', 'close', 'volume']]

        logger.info(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")

        return df

    def load_tick_data(
        self,
        filepath: Union[str, Path],
        symbol: str = "NQ",
        timezone: str = "America/Chicago"
    ) -> pd.DataFrame:
        """
        Load tick (trade) data.

        Args:
            filepath: Path to the data file
            symbol: Instrument symbol
            timezone: Timezone for timestamps

        Returns:
            DataFrame with columns: datetime, price, volume
        """
        filepath = self._resolve_path(filepath)
        logger.info(f"Loading tick data from: {filepath}")

        # Read semicolon-separated file
        # Format: YYYYMMDD HHMMSS Ticks;Price;Volume
        # The "Ticks" portion contains subsecond precision

        df = pd.read_csv(
            filepath,
            sep=';',
            header=None,
            names=['datetime_str', 'price', 'volume'],
            dtype={
                'datetime_str': str,
                'price': float,
                'volume': int
            }
        )

        # Parse datetime with subsecond precision
        # Format: "YYYYMMDD HHMMSS Ticks" where Ticks is fractional seconds
        df['datetime'] = df['datetime_str'].apply(self._parse_tick_datetime)
        df = df.drop('datetime_str', axis=1)

        # Set datetime as index
        df = df.set_index('datetime')

        # Localize to Chicago time
        df.index = df.index.tz_localize(timezone)

        # Add symbol column
        df['symbol'] = symbol

        # Reorder columns
        df = df[['symbol', 'price', 'volume']]

        logger.info(f"Loaded {len(df)} ticks from {df.index.min()} to {df.index.max()}")

        return df

    def _parse_tick_datetime(self, dt_str: str) -> datetime:
        """
        Parse tick datetime string with subsecond precision.

        Format: "YYYYMMDD HHMMSS Ticks"
        Example: "20181220 000001 8960000"

        The Ticks portion represents fractional seconds (nanoseconds or similar).
        """
        parts = dt_str.split()
        date_str = parts[0]  # YYYYMMDD
        time_str = parts[1]  # HHMMSS

        # Parse subsecond if present
        if len(parts) > 2:
            subsec_str = parts[2]
            # Convert to microseconds (first 6 digits)
            microseconds = int(subsec_str[:6]) if len(subsec_str) >= 6 else int(subsec_str.ljust(6, '0'))
        else:
            microseconds = 0

        # Build datetime string
        full_str = f"{date_str} {time_str}"
        dt = datetime.strptime(full_str, "%Y%m%d %H%M%S")

        # Add microseconds
        dt = dt.replace(microsecond=microseconds)

        return dt

    def _resolve_path(self, filepath: Union[str, Path]) -> Path:
        """Resolve filepath relative to data_dir or as absolute."""
        filepath = Path(filepath)
        if filepath.is_absolute():
            return filepath
        return self.data_dir / filepath

    def resample_ticks_to_bars(
        self,
        tick_df: pd.DataFrame,
        freq: str = '1T',  # 1 minute
        price_col: str = 'price',
        volume_col: str = 'volume'
    ) -> pd.DataFrame:
        """
        Resample tick data to OHLCV bars.

        Args:
            tick_df: DataFrame with tick data (price, volume)
            freq: Pandas frequency string ('1T' = 1 minute, '5T' = 5 minutes, etc.)
            price_col: Column name for price
            volume_col: Column name for volume

        Returns:
            DataFrame with OHLCV bars
        """
        logger.info(f"Resampling ticks to {freq} bars")

        ohlcv = tick_df.resample(freq).agg({
            price_col: ['first', 'max', 'min', 'last'],
            volume_col: 'sum'
        })

        # Flatten column names
        ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']

        # Drop bars with no trades
        ohlcv = ohlcv.dropna()

        # Add symbol if present in original
        if 'symbol' in tick_df.columns:
            ohlcv['symbol'] = tick_df['symbol'].iloc[0]

        logger.info(f"Created {len(ohlcv)} bars")

        return ohlcv

    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate data quality and return statistics.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        stats = {
            'total_rows': len(df),
            'date_range': (df.index.min(), df.index.max()),
            'trading_days': df.index.normalize().nunique(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.index.duplicated().sum(),
        }

        if 'volume' in df.columns:
            stats['zero_volume_bars'] = (df['volume'] == 0).sum()
            stats['total_volume'] = df['volume'].sum()

        if 'close' in df.columns:
            stats['price_range'] = (df['close'].min(), df['close'].max())
            stats['avg_price'] = df['close'].mean()

        return stats


def load_databento_data(symbol: str = "ES", data_dir: str = None) -> pd.DataFrame:
    """
    Load Databento 1-minute data.

    Args:
        symbol: 'ES' or 'NQ'
        data_dir: Optional data directory

    Returns:
        DataFrame with OHLCV data
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "market"
    else:
        data_dir = Path(data_dir)

    filepath = data_dir / f"{symbol}_1min_databento.csv"

    if not filepath.exists():
        raise FileNotFoundError(f"Databento data not found: {filepath}")

    logger.info(f"Loading Databento data from {filepath}")

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    # Standardize columns
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    df['symbol'] = symbol

    # Ensure proper timezone handling
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    logger.info(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")

    return df


def load_sample_data(source: str = "portaraninja"):
    """
    Load sample data files.

    Args:
        source: 'portaraninja' or 'databento'

    Returns:
        Tuple of (ES minute data, NQ tick/minute data)
    """
    loader = NinjaTraderLoader()

    if source == "databento":
        # Load Databento 2-year 1-minute data
        es_data = load_databento_data("ES")
        nq_data = load_databento_data("NQ")
        return es_data, nq_data
    else:
        # Load PortaraNinja sample data
        es_minute = loader.load_minute_data("ES_1min_continuous.txt", symbol="ES")
        nq_ticks = loader.load_tick_data("NQ_tick_sample.txt", symbol="NQ")
        return es_minute, nq_ticks


def load_best_available_data(symbol: str = "ES") -> pd.DataFrame:
    """
    Load the best available data for a symbol.

    Priority:
    1. Databento (2 years 1-minute)
    2. PortaraNinja sample

    Returns:
        DataFrame with OHLCV data
    """
    try:
        return load_databento_data(symbol)
    except FileNotFoundError:
        logger.info(f"Databento data not found for {symbol}, using PortaraNinja sample")
        loader = NinjaTraderLoader()
        if symbol == "ES":
            return loader.load_minute_data("ES_1min_continuous.txt", symbol="ES")
        else:
            # For NQ, resample ticks
            nq_ticks = loader.load_tick_data("NQ_tick_sample.txt", symbol="NQ")
            return loader.resample_ticks_to_bars(nq_ticks)


if __name__ == "__main__":
    # Test the loader with sample data
    print("=" * 60)
    print("NinjaTrader Data Loader Test")
    print("=" * 60)

    try:
        es_data, nq_data = load_sample_data()

        print("\n--- ES 1-Minute Data ---")
        print(f"Shape: {es_data.shape}")
        print(f"Columns: {es_data.columns.tolist()}")
        print(f"\nFirst 5 rows:\n{es_data.head()}")
        print(f"\nLast 5 rows:\n{es_data.tail()}")

        # Validate
        loader = NinjaTraderLoader()
        es_stats = loader.validate_data(es_data)
        print(f"\nValidation Stats:\n{es_stats}")

        print("\n--- NQ Tick Data ---")
        print(f"Shape: {nq_data.shape}")
        print(f"Columns: {nq_data.columns.tolist()}")
        print(f"\nFirst 5 rows:\n{nq_data.head()}")

        # Resample ticks to 1-minute bars
        nq_bars = loader.resample_ticks_to_bars(nq_data)
        print(f"\nNQ resampled to 1-min bars: {nq_bars.shape}")
        print(nq_bars.head())

        print("\n" + "=" * 60)
        print("SUCCESS: Data loader working correctly!")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
