"""
Databento Data Downloader
=========================
Downloads historical futures data from Databento.

API Key stored in config/api_keys.py (gitignored)
"""

import databento as db
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "config"))
    from api_keys import DATABENTO_API_KEY
except ImportError:
    DATABENTO_API_KEY = None


class DatabentoDownloader:
    """Download futures data from Databento."""

    def __init__(self, api_key: str = None):
        """Initialize with API key."""
        self.api_key = api_key or DATABENTO_API_KEY
        if not self.api_key:
            raise ValueError("Databento API key required")

        self.client = db.Historical(self.api_key)
        self.data_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "market"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_cost_estimate(
        self,
        symbols: list,
        schema: str,
        start: str,
        end: str
    ) -> float:
        """Get cost estimate for download."""
        try:
            cost = self.client.metadata.get_cost(
                dataset='GLBX.MDP3',
                symbols=symbols,
                schema=schema,
                start=start,
                end=end
            )
            return cost
        except Exception as e:
            logger.error(f"Error getting cost: {e}")
            return 0.0

    def download_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        schema: str = 'ohlcv-1m'
    ) -> pd.DataFrame:
        """
        Download OHLCV data for a symbol.

        Args:
            symbol: Contract symbol (e.g., 'ESZ4')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            schema: Data schema ('ohlcv-1m', 'ohlcv-1h', 'ohlcv-1d')

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Downloading {symbol} {schema} from {start} to {end}...")

        try:
            # Download data
            data = self.client.timeseries.get_range(
                dataset='GLBX.MDP3',
                symbols=[symbol],
                schema=schema,
                start=start,
                end=end
            )

            # Convert to DataFrame
            df = data.to_df()

            if len(df) == 0:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            logger.info(f"Downloaded {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")
            return pd.DataFrame()

    def download_trades(
        self,
        symbol: str,
        start: str,
        end: str
    ) -> pd.DataFrame:
        """Download tick/trades data."""
        logger.info(f"Downloading {symbol} trades from {start} to {end}...")

        try:
            data = self.client.timeseries.get_range(
                dataset='GLBX.MDP3',
                symbols=[symbol],
                schema='trades',
                start=start,
                end=end
            )

            df = data.to_df()
            logger.info(f"Downloaded {len(df)} trades for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error downloading trades: {e}")
            return pd.DataFrame()

    def download_es_nq_2years(self) -> dict:
        """
        Download 2 years of ES and NQ 1-minute data.

        Returns:
            Dict with combined DataFrames for ES and NQ
        """
        # Contract definitions (symbol, start, end)
        contracts = {
            'ES': [
                ('ESH3', '2023-01-01', '2023-03-17'),
                ('ESM3', '2023-03-17', '2023-06-16'),
                ('ESU3', '2023-06-16', '2023-09-15'),
                ('ESZ3', '2023-09-15', '2023-12-15'),
                ('ESH4', '2024-01-01', '2024-03-15'),
                ('ESM4', '2024-03-15', '2024-06-21'),
                ('ESU4', '2024-06-21', '2024-09-20'),
                ('ESZ4', '2024-09-20', '2024-12-20'),
            ],
            'NQ': [
                ('NQH3', '2023-01-01', '2023-03-17'),
                ('NQM3', '2023-03-17', '2023-06-16'),
                ('NQU3', '2023-06-16', '2023-09-15'),
                ('NQZ3', '2023-09-15', '2023-12-15'),
                ('NQH4', '2024-01-01', '2024-03-15'),
                ('NQM4', '2024-03-15', '2024-06-21'),
                ('NQU4', '2024-06-21', '2024-09-20'),
                ('NQZ4', '2024-09-20', '2024-12-20'),
            ]
        }

        results = {}

        for instrument, contract_list in contracts.items():
            logger.info(f"\nDownloading {instrument} contracts...")
            all_data = []

            for symbol, start, end in contract_list:
                df = self.download_ohlcv(symbol, start, end, 'ohlcv-1m')
                if len(df) > 0:
                    all_data.append(df)

            if all_data:
                # Combine all contracts
                combined = pd.concat(all_data)
                combined = combined.sort_index()
                combined = combined[~combined.index.duplicated(keep='first')]

                results[instrument] = combined
                logger.info(f"{instrument}: {len(combined)} total bars")

                # Save to file
                filepath = self.data_dir / f"{instrument}_1min_databento.csv"
                combined.to_csv(filepath)
                logger.info(f"Saved to {filepath}")

        return results

    def create_continuous_contract(
        self,
        contract_data: list,
        adjust_method: str = 'ratio'
    ) -> pd.DataFrame:
        """
        Create continuous futures contract from individual contracts.

        Args:
            contract_data: List of (symbol, dataframe) tuples
            adjust_method: 'ratio' or 'difference'

        Returns:
            DataFrame with continuous contract
        """
        if not contract_data:
            return pd.DataFrame()

        # Sort by date
        all_data = []
        for symbol, df in contract_data:
            df = df.copy()
            df['contract'] = symbol
            all_data.append(df)

        combined = pd.concat(all_data)
        combined = combined.sort_index()

        # For now, just use unadjusted prices (ratio adjustment can be added later)
        return combined


    def download_historical_years(
        self,
        instrument: str,
        years: list,
        schema: str = 'ohlcv-1m'
    ) -> pd.DataFrame:
        """
        Download multiple years of data for an instrument.

        Args:
            instrument: Base instrument (ES, NQ, YM, GC, CL, ZN)
            years: List of years to download (e.g., [2020, 2021, 2022])
            schema: Data schema

        Returns:
            Combined DataFrame for all years
        """
        # Contract month codes: H=Mar, M=Jun, U=Sep, Z=Dec
        contract_months = [
            ('H', '01-01', '03-17'),  # March contract
            ('M', '03-17', '06-16'),  # June contract
            ('U', '06-16', '09-15'),  # September contract
            ('Z', '09-15', '12-20'),  # December contract
        ]

        all_data = []

        for year in years:
            year_suffix = str(year)[-1]  # Last digit of year

            for month_code, start_md, end_md in contract_months:
                symbol = f"{instrument}{month_code}{year_suffix}"
                start = f"{year}-{start_md}"
                end = f"{year}-{end_md}"

                df = self.download_ohlcv(symbol, start, end, schema)
                if len(df) > 0:
                    all_data.append(df)

        if all_data:
            combined = pd.concat(all_data)
            combined = combined.sort_index()
            combined = combined[~combined.index.duplicated(keep='first')]
            return combined

        return pd.DataFrame()

    def estimate_cost_for_years(
        self,
        instrument: str,
        years: list,
        schema: str = 'ohlcv-1m'
    ) -> float:
        """Estimate cost for downloading multiple years."""
        contract_months = ['H', 'M', 'U', 'Z']
        total_cost = 0.0

        for year in years:
            year_suffix = str(year)[-1]
            for month_code in contract_months:
                symbol = f"{instrument}{month_code}{year_suffix}"
                try:
                    cost = self.get_cost_estimate(
                        [symbol], schema,
                        f'{year}-01-01', f'{year}-12-31'
                    )
                    total_cost += cost
                except Exception as e:
                    logger.debug(f"Could not estimate cost for {symbol}: {e}")

        return total_cost

    def download_mbo_sample(
        self,
        symbol: str,
        start: str,
        end: str
    ) -> pd.DataFrame:
        """
        Download MBO (Market-By-Order) Level 2 data sample.

        This is order book data showing individual orders.
        NOTE: MBO data is much larger and more expensive than OHLCV.
        """
        logger.info(f"Downloading {symbol} MBO from {start} to {end}...")

        try:
            data = self.client.timeseries.get_range(
                dataset='GLBX.MDP3',
                symbols=[symbol],
                schema='mbo',
                start=start,
                end=end
            )

            df = data.to_df()
            logger.info(f"Downloaded {len(df)} MBO records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error downloading MBO: {e}")
            return pd.DataFrame()

    def download_mbp_sample(
        self,
        symbol: str,
        start: str,
        end: str,
        levels: int = 10
    ) -> pd.DataFrame:
        """
        Download MBP (Market-By-Price) data - aggregated order book.

        Args:
            symbol: Contract symbol
            start: Start date
            end: End date
            levels: Number of price levels (1, 5, or 10)
        """
        schema = f'mbp-{levels}'
        logger.info(f"Downloading {symbol} {schema} from {start} to {end}...")

        try:
            data = self.client.timeseries.get_range(
                dataset='GLBX.MDP3',
                symbols=[symbol],
                schema=schema,
                start=start,
                end=end
            )

            df = data.to_df()
            logger.info(f"Downloaded {len(df)} MBP records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error downloading MBP: {e}")
            return pd.DataFrame()


def estimate_all_costs(api_key: str = None, budget: float = 125.0, already_spent: float = 3.17):
    """
    Estimate costs for all potential downloads.

    Current situation:
    - Already downloaded: 2023-2024 ES and NQ 1-min data (~$3.17)
    - Remaining budget: ~$121.83
    """
    downloader = DatabentoDownloader(api_key)
    remaining = budget - already_spent

    print("=" * 70)
    print("DATABENTO COST ESTIMATION")
    print("=" * 70)
    print(f"Total Budget: ${budget:.2f}")
    print(f"Already Spent: ${already_spent:.2f}")
    print(f"Remaining: ${remaining:.2f}")
    print("=" * 70)

    estimates = {}

    # 1. More years of ES/NQ (2020-2022)
    print("\n1. Additional Years of ES/NQ (2020-2022):")
    for instrument in ['ES', 'NQ']:
        for year in [2020, 2021, 2022]:
            cost = downloader.estimate_cost_for_years(instrument, [year], 'ohlcv-1m')
            key = f"{instrument}_{year}"
            estimates[key] = cost
            print(f"   {instrument} {year}: ${cost:.4f}")

    es_nq_additional = sum(v for k, v in estimates.items())
    print(f"   SUBTOTAL (2020-2022 ES+NQ): ${es_nq_additional:.4f}")

    # 2. Other assets (YM, GC, CL, ZN) - just 2023-2024 for now
    print("\n2. Other Assets (2023-2024):")
    other_instruments = ['YM', 'GC', 'CL', 'ZN']
    other_costs = {}
    for instrument in other_instruments:
        cost = downloader.estimate_cost_for_years(instrument, [2023, 2024], 'ohlcv-1m')
        other_costs[instrument] = cost
        estimates[f"{instrument}_2023_2024"] = cost
        print(f"   {instrument} (2023-2024): ${cost:.4f}")

    other_total = sum(other_costs.values())
    print(f"   SUBTOTAL (Other Assets): ${other_total:.4f}")

    # 3. Level 2 data sample (very expensive - just estimate 1 day)
    print("\n3. Level 2 Order Book Data (1-day samples):")
    l2_costs = {}

    for schema in ['mbo', 'mbp-1', 'mbp-10']:
        try:
            cost = downloader.get_cost_estimate(
                ['ESH4'], schema,
                '2024-03-01', '2024-03-02'  # Just 1 day
            )
            l2_costs[schema] = cost
            print(f"   ES {schema} (1 day): ${cost:.4f}")
        except Exception as e:
            print(f"   ES {schema}: Error - {e}")

    # Summary
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    total_ohlcv = es_nq_additional + other_total
    print(f"\nTotal for all OHLCV data (2020-2024, all instruments): ${total_ohlcv:.4f}")

    if total_ohlcv <= remaining:
        print(f"[OK] This fits within remaining budget of ${remaining:.2f}")
        print(f"  Remaining after all OHLCV: ${remaining - total_ohlcv:.2f}")

        # Can we afford L2 sample?
        if 'mbp-10' in l2_costs:
            mbp_1week = l2_costs['mbp-10'] * 5
            if mbp_1week < (remaining - total_ohlcv):
                print(f"  Could also afford ~5 days of MBP-10 data: ${mbp_1week:.4f}")
    else:
        print(f"[X] Exceeds budget by ${total_ohlcv - remaining:.2f}")
        print("  Consider downloading fewer years or instruments")

    return estimates


def download_additional_data(api_key: str = None, budget: float = 125.0, already_spent: float = 3.17):
    """Download additional data within budget."""
    downloader = DatabentoDownloader(api_key)
    remaining = budget - already_spent

    print("=" * 70)
    print("DOWNLOADING ADDITIONAL DATA")
    print("=" * 70)
    print(f"Remaining Budget: ${remaining:.2f}")
    print("=" * 70)

    results = {}
    total_spent = 0.0

    # Priority 1: More years of ES (2020-2022)
    print("\n1. Downloading ES 2020-2022...")
    for year in [2022, 2021, 2020]:  # Most recent first
        est_cost = downloader.estimate_cost_for_years('ES', [year], 'ohlcv-1m')
        if total_spent + est_cost <= remaining:
            print(f"   Downloading ES {year} (est: ${est_cost:.4f})...")
            df = downloader.download_historical_years('ES', [year])
            if len(df) > 0:
                filepath = downloader.data_dir / f"ES_{year}_1min_databento.csv"
                df.to_csv(filepath)
                results[f'ES_{year}'] = df
                total_spent += est_cost
                print(f"   [OK] ES {year}: {len(df)} bars saved to {filepath}")
        else:
            print(f"   Skipping ES {year} - would exceed budget")

    # Priority 2: More years of NQ (2020-2022)
    print("\n2. Downloading NQ 2020-2022...")
    for year in [2022, 2021, 2020]:
        est_cost = downloader.estimate_cost_for_years('NQ', [year], 'ohlcv-1m')
        if total_spent + est_cost <= remaining:
            print(f"   Downloading NQ {year} (est: ${est_cost:.4f})...")
            df = downloader.download_historical_years('NQ', [year])
            if len(df) > 0:
                filepath = downloader.data_dir / f"NQ_{year}_1min_databento.csv"
                df.to_csv(filepath)
                results[f'NQ_{year}'] = df
                total_spent += est_cost
                print(f"   [OK] NQ {year}: {len(df)} bars saved to {filepath}")
        else:
            print(f"   Skipping NQ {year} - would exceed budget")

    # Priority 3: Other assets (2023-2024 only)
    print("\n3. Downloading other assets (2023-2024)...")
    for instrument in ['YM', 'GC', 'CL', 'ZN']:
        est_cost = downloader.estimate_cost_for_years(instrument, [2023, 2024], 'ohlcv-1m')
        if total_spent + est_cost <= remaining:
            print(f"   Downloading {instrument} 2023-2024 (est: ${est_cost:.4f})...")
            df = downloader.download_historical_years(instrument, [2023, 2024])
            if len(df) > 0:
                filepath = downloader.data_dir / f"{instrument}_1min_databento.csv"
                df.to_csv(filepath)
                results[instrument] = df
                total_spent += est_cost
                print(f"   [OK] {instrument}: {len(df)} bars saved to {filepath}")
        else:
            print(f"   Skipping {instrument} - would exceed budget")

    # Priority 4: MBP-10 sample (1 week)
    print("\n4. Downloading Level 2 sample...")
    try:
        mbp_cost = downloader.get_cost_estimate(['ESH4'], 'mbp-10', '2024-03-01', '2024-03-05')
        if total_spent + mbp_cost <= remaining:
            print(f"   Downloading ES MBP-10 sample (est: ${mbp_cost:.4f})...")
            df = downloader.download_mbp_sample('ESH4', '2024-03-01', '2024-03-05', levels=10)
            if len(df) > 0:
                filepath = downloader.data_dir / "ES_mbp10_sample_databento.csv"
                df.to_csv(filepath)
                results['ES_MBP10_sample'] = df
                total_spent += mbp_cost
                print(f"   [OK] MBP-10 sample: {len(df)} records saved")
        else:
            print(f"   Skipping MBP-10 - would exceed budget (cost: ${mbp_cost:.4f})")
    except Exception as e:
        print(f"   Error with MBP-10: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Total estimated cost: ${total_spent:.4f}")
    print(f"Remaining budget: ${remaining - total_spent:.2f}")

    for key, df in results.items():
        print(f"  {key}: {len(df)} records")

    return results


def download_databento_data(api_key: str = None):
    """Main function to download data."""
    downloader = DatabentoDownloader(api_key)

    print("=" * 60)
    print("Databento Data Download")
    print("=" * 60)

    # First, estimate total cost
    print("\nEstimating costs...")
    total_cost = 0

    contracts = [
        # 2023
        'ESH3', 'ESM3', 'ESU3', 'ESZ3',
        'NQH3', 'NQM3', 'NQU3', 'NQZ3',
        # 2024
        'ESH4', 'ESM4', 'ESU4', 'ESZ4',
        'NQH4', 'NQM4', 'NQU4', 'NQZ4',
    ]

    for contract in contracts:
        year = 2023 if contract[2] == '3' else 2024
        cost = downloader.get_cost_estimate(
            [contract], 'ohlcv-1m',
            f'{year}-01-01', f'{year}-12-31'
        )
        total_cost += cost

    print(f"\nEstimated total cost: ${total_cost:.4f}")

    if total_cost > 125:
        print("ERROR: Cost exceeds $125 budget!")
        return None

    print(f"Remaining budget after download: ${125 - total_cost:.2f}")
    print("\nProceeding with download...")

    # Download data
    data = downloader.download_es_nq_2years()

    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)

    for instrument, df in data.items():
        print(f"\n{instrument}:")
        print(f"  Rows: {len(df)}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Columns: {list(df.columns)}")

    return data


if __name__ == "__main__":
    # Use command line API key or from config
    import sys

    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = 'db-L8vcArDDsTpeVUW5x4yBJuD4iAagU'

    download_databento_data(api_key)
