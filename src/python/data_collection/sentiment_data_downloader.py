"""
Sentiment Data Downloader
=========================

Downloads real PCR (Put/Call Ratio) and AAII sentiment data for the strategy.

Data Sources:
- PCR: CBOE Historical Data (free CSV download)
- AAII: AAII.com (requires membership for full history, uses proxy otherwise)

Output Files:
- data/raw/sentiment/pcr_historical.csv
- data/raw/sentiment/aaii_historical.csv

These files are then loaded by historical_sentiment_loader.py for backtesting
and can be exported for NT8 via the TCP bridge.

Author: SKIE_Ninja Development Team
Created: 2025-12-07
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import logging
import time

logger = logging.getLogger(__name__)


class SentimentDataDownloader:
    """
    Download and manage real sentiment data from external sources.
    """

    # CBOE Put/Call Ratio CSV URLs
    CBOE_URLS = {
        'total': 'https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv',
        'equity': 'https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv',
        'index': 'https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpc.csv',
    }

    # Alternative CBOE archive URLs (historical)
    CBOE_ARCHIVE_URLS = {
        'total_archive': 'https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpcarchive.csv',
        'index_archive': 'https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpcarchive.csv',
    }

    # Alternative: FRED has VIX and some sentiment data
    FRED_SERIES = {
        'vix': 'VIXCLS',
    }

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the downloader.

        Args:
            data_dir: Directory to save data. Defaults to data/raw/sentiment/
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'raw' / 'sentiment'
        else:
            self.data_dir = Path(data_dir)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Sentiment data directory: {self.data_dir}")

    def download_cboe_pcr(self, pcr_type: str = 'total') -> pd.DataFrame:
        """
        Download Put/Call Ratio data from CBOE.

        Args:
            pcr_type: Type of PCR ('total', 'equity', 'index')

        Returns:
            DataFrame with PCR data
        """
        if pcr_type not in self.CBOE_URLS:
            raise ValueError(f"Invalid PCR type: {pcr_type}. Choose from {list(self.CBOE_URLS.keys())}")

        url = self.CBOE_URLS[pcr_type]
        logger.info(f"Downloading {pcr_type} PCR from CBOE: {url}")

        try:
            # Try current data first
            df = self._fetch_cboe_csv(url)

            # Try to also get archive data and merge
            archive_key = f'{pcr_type}_archive'
            if archive_key in self.CBOE_ARCHIVE_URLS:
                try:
                    archive_df = self._fetch_cboe_csv(self.CBOE_ARCHIVE_URLS[archive_key])
                    df = pd.concat([archive_df, df]).drop_duplicates(subset=['date']).sort_values('date')
                    logger.info(f"Merged with archive data: {len(df)} total rows")
                except Exception as e:
                    logger.warning(f"Could not fetch archive data: {e}")

            return df

        except Exception as e:
            logger.error(f"Failed to download PCR data: {e}")
            raise

    def _fetch_cboe_csv(self, url: str) -> pd.DataFrame:
        """
        Fetch and parse a CBOE CSV file.

        Args:
            url: URL to fetch

        Returns:
            Parsed DataFrame
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse CSV - CBOE format has header rows we need to skip
        from io import StringIO
        content = response.text

        # Find the actual data header (usually contains 'DATE' or 'Trade Date')
        lines = content.strip().split('\n')
        header_idx = 0
        for i, line in enumerate(lines):
            if 'DATE' in line.upper() or 'TRADE' in line.upper():
                header_idx = i
                break

        # Read from the header row
        df = pd.read_csv(StringIO('\n'.join(lines[header_idx:])))

        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        df.columns = df.columns.str.replace(' ', '_')

        # Find date column
        date_col = None
        for col in df.columns:
            if 'date' in col:
                date_col = col
                break

        if date_col is None:
            date_col = df.columns[0]

        # Parse date
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['date'])

        # Find P/C ratio column
        pcr_col = None
        for col in df.columns:
            if 'p/c' in col or 'ratio' in col:
                pcr_col = col
                break

        if pcr_col and pcr_col != 'pcr_total':
            df['pcr_total'] = pd.to_numeric(df[pcr_col], errors='coerce')

        # Find volume columns if available
        for col in df.columns:
            if 'call' in col and 'put' not in col:
                df['call_volume'] = pd.to_numeric(df[col], errors='coerce')
            if 'put' in col and 'call' not in col:
                df['put_volume'] = pd.to_numeric(df[col], errors='coerce')
            if 'total' in col and 'ratio' not in col and 'p/c' not in col:
                df['total_volume'] = pd.to_numeric(df[col], errors='coerce')

        # Calculate PCR if we have volumes but not ratio
        if 'pcr_total' not in df.columns or df['pcr_total'].isna().all():
            if 'call_volume' in df.columns and 'put_volume' in df.columns:
                df['pcr_total'] = df['put_volume'] / df['call_volume'].replace(0, np.nan)

        # Keep only relevant columns
        keep_cols = ['date', 'pcr_total']
        for col in ['call_volume', 'put_volume', 'total_volume']:
            if col in df.columns:
                keep_cols.append(col)

        df = df[keep_cols].dropna(subset=['pcr_total'])
        df = df.sort_values('date').reset_index(drop=True)

        logger.info(f"Parsed {len(df)} rows, date range: {df['date'].min()} to {df['date'].max()}")

        return df

    def process_pcr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate PCR-derived features matching what the model expects.

        Args:
            df: Raw PCR DataFrame with 'date' and 'pcr_total' columns

        Returns:
            DataFrame with calculated features
        """
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)

        # Moving averages
        df['pcr_ma5'] = df['pcr_total'].rolling(5, min_periods=1).mean()
        df['pcr_ma10'] = df['pcr_total'].rolling(10, min_periods=1).mean()

        # Extreme readings (from config thresholds)
        pcr_bullish_threshold = 1.1  # High PCR = oversold = bullish
        pcr_bearish_threshold = 0.8  # Low PCR = overbought = bearish

        df['pcr_bullish_extreme'] = (df['pcr_total'] > pcr_bullish_threshold).astype(int)
        df['pcr_bearish_extreme'] = (df['pcr_total'] < pcr_bearish_threshold).astype(int)

        # Contrarian signal: high PCR = bullish (oversold)
        df['pcr_contrarian_signal'] = np.clip((df['pcr_total'] - 0.95) / 0.35, -1, 1)

        # Mark as real data (not proxy)
        df['is_proxy'] = False

        return df

    def save_pcr_data(self, df: pd.DataFrame) -> Path:
        """
        Save PCR data to CSV in the expected format.

        Args:
            df: DataFrame with PCR data and features

        Returns:
            Path to saved file
        """
        output_path = self.data_dir / 'pcr_historical.csv'

        # Ensure date is properly formatted
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        df.to_csv(output_path, index=False)
        logger.info(f"Saved PCR data to {output_path}: {len(df)} rows")

        return output_path

    def download_and_save_pcr(self, pcr_type: str = 'total') -> Tuple[pd.DataFrame, Path]:
        """
        Download PCR data, calculate features, and save.

        Args:
            pcr_type: Type of PCR to download

        Returns:
            Tuple of (DataFrame, output_path)
        """
        df = self.download_cboe_pcr(pcr_type)
        df = self.process_pcr_features(df)
        output_path = self.save_pcr_data(df)

        return df, output_path

    def create_aaii_from_available_sources(self) -> pd.DataFrame:
        """
        Create AAII data from available sources.

        Since AAII requires membership, we try:
        1. Load from existing file if available
        2. Use a minimal proxy based on VIX data

        For production, consider AAII membership ($29/year) for full historical data.

        Returns:
            DataFrame with AAII data (real or proxy)
        """
        # Check if we have existing AAII file
        aaii_path = self.data_dir / 'aaii_historical.csv'

        if aaii_path.exists():
            logger.info(f"Loading existing AAII data from {aaii_path}")
            df = pd.read_csv(aaii_path, parse_dates=['date'])
            if 'is_proxy' in df.columns and not df['is_proxy'].iloc[0]:
                logger.info("Using real AAII data")
                return df

        # Create proxy from VIX data
        logger.warning("Creating AAII proxy from VIX data. For real data, get AAII membership.")

        # Load VIX data
        vix_path = self.data_dir.parent / 'market' / 'VIX_daily.csv'
        if not vix_path.exists():
            raise FileNotFoundError(f"VIX data not found at {vix_path}")

        vix_df = pd.read_csv(vix_path)
        vix_df.columns = vix_df.columns.str.lower()

        date_col = 'date' if 'date' in vix_df.columns else vix_df.columns[0]
        # Handle timezone-aware dates by converting to UTC first
        vix_df['date'] = pd.to_datetime(vix_df[date_col], utc=True).dt.tz_localize(None)

        close_col = 'close' if 'close' in vix_df.columns else 'adj close'
        vix_df['vix_close'] = vix_df[close_col].astype(float)

        # Calculate VIX percentile
        vix_df['vix_percentile'] = vix_df['vix_close'].rolling(20, min_periods=1).apply(
            lambda x: (x < x.iloc[-1]).mean() if len(x) > 0 else 0.5
        )

        # Create AAII proxy features
        df = pd.DataFrame()
        df['date'] = vix_df['date']

        # High VIX = High bearish, Low bullish
        vix_pct = vix_df['vix_percentile'].fillna(0.5)
        df['aaii_bearish'] = 25 + (vix_pct * 30)  # Range: 25-55%
        df['aaii_bullish'] = 50 - (vix_pct * 30)  # Range: 20-50%
        df['aaii_neutral'] = 100 - df['aaii_bullish'] - df['aaii_bearish']
        df['aaii_spread'] = df['aaii_bullish'] - df['aaii_bearish']

        # Extreme readings
        df['aaii_extreme_bullish'] = (df['aaii_bullish'] > 50).astype(int)
        df['aaii_extreme_bearish'] = (df['aaii_bearish'] > 50).astype(int)

        # Contrarian signal
        df['aaii_contrarian_signal'] = (df['aaii_bearish'] - df['aaii_bullish']) / 100

        df['is_proxy'] = True

        return df

    def save_aaii_data(self, df: pd.DataFrame) -> Path:
        """
        Save AAII data to CSV.

        Args:
            df: DataFrame with AAII data

        Returns:
            Path to saved file
        """
        output_path = self.data_dir / 'aaii_historical.csv'

        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        df.to_csv(output_path, index=False)
        logger.info(f"Saved AAII data to {output_path}: {len(df)} rows")

        return output_path

    def create_nt8_sentiment_export(self) -> Path:
        """
        Create a combined sentiment file for NT8 to load.

        This file contains the latest sentiment values that NT8 can read
        as a CSV to get real PCR/AAII data instead of using VIX proxies.

        Returns:
            Path to exported file
        """
        # Load PCR data
        pcr_path = self.data_dir / 'pcr_historical.csv'
        if pcr_path.exists():
            pcr_df = pd.read_csv(pcr_path, parse_dates=['date'])
        else:
            logger.warning("PCR data not found, downloading...")
            pcr_df, _ = self.download_and_save_pcr()

        # Load AAII data
        aaii_path = self.data_dir / 'aaii_historical.csv'
        if aaii_path.exists():
            aaii_df = pd.read_csv(aaii_path, parse_dates=['date'])
        else:
            logger.warning("AAII data not found, creating proxy...")
            aaii_df = self.create_aaii_from_available_sources()
            self.save_aaii_data(aaii_df)

        # Merge on date
        df = pcr_df.merge(aaii_df, on='date', how='outer', suffixes=('', '_aaii'))
        df = df.sort_values('date').reset_index(drop=True)

        # Forward fill to handle missing dates
        df = df.ffill()

        # Export for NT8
        export_path = self.data_dir.parent.parent / 'nt8_sentiment_data.csv'

        # Select columns for NT8
        export_cols = [
            'date',
            'pcr_total', 'pcr_ma5', 'pcr_ma10',
            'pcr_bullish_extreme', 'pcr_bearish_extreme', 'pcr_contrarian_signal',
            'aaii_bullish', 'aaii_bearish', 'aaii_spread',
            'aaii_extreme_bullish', 'aaii_extreme_bearish', 'aaii_contrarian_signal'
        ]

        export_cols = [c for c in export_cols if c in df.columns]

        export_df = df[export_cols].copy()
        export_df['date'] = pd.to_datetime(export_df['date']).dt.strftime('%Y-%m-%d')

        export_df.to_csv(export_path, index=False)
        logger.info(f"Exported NT8 sentiment data to {export_path}: {len(export_df)} rows")

        return export_path

    def get_latest_sentiment(self) -> Dict:
        """
        Get the latest sentiment values for real-time trading.

        Returns:
            Dict with latest PCR and AAII values
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'pcr': {},
            'aaii': {}
        }

        # Load PCR
        pcr_path = self.data_dir / 'pcr_historical.csv'
        if pcr_path.exists():
            pcr_df = pd.read_csv(pcr_path, parse_dates=['date'])
            latest_pcr = pcr_df.iloc[-1]
            result['pcr'] = {
                'date': str(latest_pcr['date'].date()),
                'pcr_total': float(latest_pcr['pcr_total']),
                'pcr_ma5': float(latest_pcr['pcr_ma5']),
                'pcr_ma10': float(latest_pcr['pcr_ma10']),
                'pcr_contrarian_signal': float(latest_pcr['pcr_contrarian_signal']),
                'is_proxy': bool(latest_pcr.get('is_proxy', False))
            }

        # Load AAII
        aaii_path = self.data_dir / 'aaii_historical.csv'
        if aaii_path.exists():
            aaii_df = pd.read_csv(aaii_path, parse_dates=['date'])
            latest_aaii = aaii_df.iloc[-1]
            result['aaii'] = {
                'date': str(latest_aaii['date'].date()),
                'aaii_bullish': float(latest_aaii['aaii_bullish']),
                'aaii_bearish': float(latest_aaii['aaii_bearish']),
                'aaii_spread': float(latest_aaii['aaii_spread']),
                'aaii_contrarian_signal': float(latest_aaii['aaii_contrarian_signal']),
                'is_proxy': bool(latest_aaii.get('is_proxy', True))
            }

        return result


def main():
    """Main function to download and process sentiment data."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("SENTIMENT DATA DOWNLOADER")
    print("=" * 70)

    downloader = SentimentDataDownloader()

    # Download PCR data
    print("\n[1] Downloading PCR data from CBOE...")
    try:
        pcr_df, pcr_path = downloader.download_and_save_pcr('total')
        print(f"    SUCCESS: {len(pcr_df)} rows saved to {pcr_path}")
        print(f"    Date range: {pcr_df['date'].min()} to {pcr_df['date'].max()}")
        print(f"    Latest PCR: {pcr_df['pcr_total'].iloc[-1]:.3f}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Create AAII data
    print("\n[2] Creating AAII data...")
    try:
        aaii_df = downloader.create_aaii_from_available_sources()
        aaii_path = downloader.save_aaii_data(aaii_df)
        print(f"    {'PROXY' if aaii_df['is_proxy'].iloc[0] else 'REAL'}: {len(aaii_df)} rows saved to {aaii_path}")
        print(f"    Date range: {aaii_df['date'].min()} to {aaii_df['date'].max()}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Create NT8 export
    print("\n[3] Creating NT8 sentiment export...")
    try:
        export_path = downloader.create_nt8_sentiment_export()
        print(f"    SUCCESS: Exported to {export_path}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Show latest values
    print("\n[4] Latest sentiment values:")
    try:
        latest = downloader.get_latest_sentiment()
        print(f"\n    PCR ({latest['pcr'].get('date', 'N/A')}):")
        print(f"      Total: {latest['pcr'].get('pcr_total', 'N/A'):.3f}")
        print(f"      Contrarian Signal: {latest['pcr'].get('pcr_contrarian_signal', 'N/A'):.3f}")
        print(f"      Is Proxy: {latest['pcr'].get('is_proxy', 'N/A')}")

        print(f"\n    AAII ({latest['aaii'].get('date', 'N/A')}):")
        print(f"      Bullish: {latest['aaii'].get('aaii_bullish', 'N/A'):.1f}%")
        print(f"      Bearish: {latest['aaii'].get('aaii_bearish', 'N/A'):.1f}%")
        print(f"      Spread: {latest['aaii'].get('aaii_spread', 'N/A'):.1f}%")
        print(f"      Is Proxy: {latest['aaii'].get('is_proxy', 'N/A')}")
    except Exception as e:
        print(f"    ERROR: {e}")

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run historical_sentiment_loader.py to verify data loads correctly")
    print("2. For real AAII data, get membership at https://www.aaii.com ($29/year)")
    print("3. Set up scheduled task to run this script daily before market open")
    print("=" * 70)


if __name__ == "__main__":
    main()
