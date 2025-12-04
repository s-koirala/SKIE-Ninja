"""
Free Data Collector
====================
Collects free market data from various public sources.

Sources:
- Yahoo Finance (daily data for ES, NQ futures)
- Nasdaq Data Link (formerly Quandl)
- Investing.com (via scraping, limited)

Note: Free sources typically only provide daily data.
For minute/tick data, use paid sources like PortaraNinja or FirstRateData.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Run: pip install yfinance")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class YahooFinanceCollector:
    """Collect futures data from Yahoo Finance."""

    # Yahoo Finance symbols for futures
    SYMBOLS = {
        'ES': 'ES=F',       # E-mini S&P 500
        'NQ': 'NQ=F',       # E-mini Nasdaq 100
        'YM': 'YM=F',       # E-mini Dow
        'RTY': 'RTY=F',     # E-mini Russell 2000
        'MES': 'MES=F',     # Micro E-mini S&P 500
        'MNQ': 'MNQ=F',     # Micro E-mini Nasdaq 100
        'GC': 'GC=F',       # Gold futures
        'SI': 'SI=F',       # Silver futures
        'CL': 'CL=F',       # Crude oil futures
        'NG': 'NG=F',       # Natural gas futures
        'ZN': 'ZN=F',       # 10-year Treasury note
        'ZB': 'ZB=F',       # 30-year Treasury bond
        'DX': 'DX=F',       # US Dollar Index
        'VIX': '^VIX',      # VIX Index
        'SPX': '^GSPC',     # S&P 500 Index
        'NDX': '^NDX',      # Nasdaq 100 Index
    }

    def __init__(self):
        """Initialize Yahoo Finance collector."""
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

    def get_historical_data(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        period: str = "5y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.

        Args:
            symbol: Futures symbol (e.g., 'ES', 'NQ') or Yahoo symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Period if dates not specified ('1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max')
            interval: Data interval ('1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo')

        Returns:
            DataFrame with OHLCV data
        """
        # Map symbol if needed
        yf_symbol = self.SYMBOLS.get(symbol.upper(), symbol)

        logger.info(f"Fetching {yf_symbol} from Yahoo Finance...")

        try:
            ticker = yf.Ticker(yf_symbol)

            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)

            if len(df) == 0:
                logger.warning(f"No data returned for {yf_symbol}")
                return pd.DataFrame()

            # Standardize column names
            df.columns = df.columns.str.lower()

            # Rename columns to match our standard
            df = df.rename(columns={
                'stock splits': 'stock_splits',
                'capital gains': 'capital_gains'
            })

            # Keep only OHLCV
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[[c for c in ohlcv_cols if c in df.columns]]

            logger.info(f"Retrieved {len(df)} bars for {yf_symbol}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

            return df

        except Exception as e:
            logger.error(f"Error fetching {yf_symbol}: {e}")
            return pd.DataFrame()

    def get_multiple_symbols(
        self,
        symbols: List[str],
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of symbols
            **kwargs: Arguments passed to get_historical_data

        Returns:
            Dict mapping symbol to DataFrame
        """
        data = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol, **kwargs)
            if len(df) > 0:
                data[symbol] = df
            time.sleep(0.5)  # Rate limiting
        return data

    def get_es_nq_data(
        self,
        years: int = 5,
        save_path: Optional[Path] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get ES and NQ historical data.

        Args:
            years: Number of years of history
            save_path: Optional path to save data

        Returns:
            Dict with ES and NQ DataFrames
        """
        period = f"{years}y"

        data = {
            'ES': self.get_historical_data('ES', period=period),
            'NQ': self.get_historical_data('NQ', period=period)
        }

        # Optionally save to disk
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

            for symbol, df in data.items():
                if len(df) > 0:
                    filepath = save_path / f"{symbol}_daily_yahoo.csv"
                    df.to_csv(filepath)
                    logger.info(f"Saved {symbol} data to {filepath}")

        return data


class QuandlCollector:
    """Collect data from Nasdaq Data Link (formerly Quandl)."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Quandl collector.

        Args:
            api_key: Nasdaq Data Link API key
        """
        self.api_key = api_key
        self.base_url = "https://data.nasdaq.com/api/v3"

    def get_dataset(
        self,
        database_code: str,
        dataset_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch dataset from Nasdaq Data Link.

        Args:
            database_code: Database code (e.g., 'CHRIS')
            dataset_code: Dataset code (e.g., 'CME_ES1')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with data
        """
        if not REQUESTS_AVAILABLE:
            logger.error("requests library not available")
            return pd.DataFrame()

        url = f"{self.base_url}/datasets/{database_code}/{dataset_code}.json"

        params = {}
        if self.api_key:
            params['api_key'] = self.api_key
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            dataset = data.get('dataset', {})
            columns = dataset.get('column_names', [])
            rows = dataset.get('data', [])

            if not rows:
                logger.warning(f"No data returned for {database_code}/{dataset_code}")
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=columns)

            # Set date as index
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)

            logger.info(f"Retrieved {len(df)} rows from {database_code}/{dataset_code}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {database_code}/{dataset_code}: {e}")
            return pd.DataFrame()


class FreeDataCollector:
    """Main collector that aggregates free data sources."""

    def __init__(self, quandl_api_key: Optional[str] = None):
        """
        Initialize free data collector.

        Args:
            quandl_api_key: Optional Nasdaq Data Link API key
        """
        self.yahoo_collector = None
        self.quandl_collector = None

        if YFINANCE_AVAILABLE:
            self.yahoo_collector = YahooFinanceCollector()

        self.quandl_collector = QuandlCollector(quandl_api_key)

    def get_daily_futures_data(
        self,
        symbols: List[str] = None,
        years: int = 5,
        save_path: Optional[Path] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get daily futures data from Yahoo Finance.

        Args:
            symbols: List of symbols (default: ES, NQ, YM)
            years: Years of history
            save_path: Optional save path

        Returns:
            Dict of DataFrames
        """
        if self.yahoo_collector is None:
            logger.error("yfinance not available")
            return {}

        if symbols is None:
            symbols = ['ES', 'NQ', 'YM', 'VIX', 'GC', 'CL', 'ZN', 'DX']

        data = {}
        for symbol in symbols:
            df = self.yahoo_collector.get_historical_data(
                symbol,
                period=f"{years}y",
                interval="1d"
            )
            if len(df) > 0:
                data[symbol] = df
            time.sleep(0.5)

        # Save if path provided
        if save_path and data:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

            for symbol, df in data.items():
                filepath = save_path / f"{symbol}_daily.csv"
                df.to_csv(filepath)
                logger.info(f"Saved {symbol} to {filepath}")

        return data

    def get_intraday_data(
        self,
        symbol: str,
        interval: str = "1h",
        period: str = "60d"
    ) -> pd.DataFrame:
        """
        Get intraday data (limited by Yahoo Finance).

        Note: Yahoo Finance only provides limited intraday history:
        - 1m data: 7 days max
        - 5m data: 60 days max
        - 1h data: 730 days max

        Args:
            symbol: Futures symbol
            interval: '1m', '5m', '15m', '30m', '1h'
            period: Period to fetch

        Returns:
            DataFrame with intraday data
        """
        if self.yahoo_collector is None:
            logger.error("yfinance not available")
            return pd.DataFrame()

        return self.yahoo_collector.get_historical_data(
            symbol,
            period=period,
            interval=interval
        )


def download_free_futures_data(
    save_dir: str = None,
    symbols: List[str] = None,
    years: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Download free daily futures data from Yahoo Finance.

    Args:
        save_dir: Directory to save data
        symbols: List of symbols (default: ES, NQ, YM, etc.)
        years: Years of history

    Returns:
        Dict of DataFrames
    """
    if save_dir is None:
        save_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "market"

    collector = FreeDataCollector()
    return collector.get_daily_futures_data(
        symbols=symbols,
        years=years,
        save_path=save_dir
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Free Data Collector Test")
    print("=" * 60)

    # Test Yahoo Finance
    if YFINANCE_AVAILABLE:
        print("\n1. Testing Yahoo Finance...")
        collector = YahooFinanceCollector()

        # Get ES daily data
        es_data = collector.get_historical_data('ES', period='1y', interval='1d')
        print(f"\nES daily data: {len(es_data)} rows")
        if len(es_data) > 0:
            print(f"Date range: {es_data.index.min()} to {es_data.index.max()}")
            print(f"\nSample data:")
            print(es_data.tail())

        # Get NQ data
        nq_data = collector.get_historical_data('NQ', period='1y', interval='1d')
        print(f"\nNQ daily data: {len(nq_data)} rows")

        # Test intraday (limited)
        print("\n2. Testing intraday data (1h, 60 days)...")
        es_hourly = collector.get_historical_data('ES', period='60d', interval='1h')
        print(f"ES hourly data: {len(es_hourly)} rows")

        # Save data
        print("\n3. Downloading and saving data...")
        data = download_free_futures_data(years=2)
        print(f"\nDownloaded {len(data)} symbols")
        for symbol, df in data.items():
            print(f"  {symbol}: {len(df)} rows")

    else:
        print("yfinance not installed. Run: pip install yfinance")
