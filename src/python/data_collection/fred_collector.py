"""
FRED (Federal Reserve Economic Data) Collector
===============================================
Collects macroeconomic data from the St. Louis Fed's FRED API.

Features collected (Category 3):
- GDP growth and surprises
- Labor market (NFP, unemployment, claims)
- Inflation (CPI, PPI, PCE)
- Monetary policy (Fed Funds, Treasury yields, yield curve)
- Housing and trade data

Reference: https://fred.stlouisfed.org/docs/api/fred/
Documentation: https://github.com/mortada/fredapi

To use this module:
1. Get a FREE API key from: https://fred.stlouisfed.org/docs/api/api_key.html
2. Set environment variable: FRED_API_KEY=your_key_here
   Or pass api_key to FREDCollector()
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

# Check if fredapi is installed
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    logger.warning("fredapi not installed. Run: pip install fredapi")


class FREDCollector:
    """
    Collect macroeconomic data from FRED API.

    Provides ~50 macroeconomic features for ML trading models.
    """

    # FRED Series IDs for key economic indicators
    SERIES_CONFIG = {
        # GDP & Growth
        'gdp': {
            'series_id': 'GDP',
            'name': 'Gross Domestic Product',
            'frequency': 'quarterly',
            'transform': 'pct_change'
        },
        'gdp_real': {
            'series_id': 'GDPC1',
            'name': 'Real GDP',
            'frequency': 'quarterly',
            'transform': 'pct_change'
        },
        'industrial_production': {
            'series_id': 'INDPRO',
            'name': 'Industrial Production Index',
            'frequency': 'monthly',
            'transform': 'pct_change'
        },
        'capacity_utilization': {
            'series_id': 'TCU',
            'name': 'Capacity Utilization',
            'frequency': 'monthly',
            'transform': 'level'
        },
        'ism_manufacturing': {
            'series_id': 'MANEMP',  # Manufacturing Employment as proxy
            'name': 'ISM Manufacturing (proxy)',
            'frequency': 'monthly',
            'transform': 'pct_change'
        },
        'retail_sales': {
            'series_id': 'RSXFS',
            'name': 'Retail Sales (ex food services)',
            'frequency': 'monthly',
            'transform': 'pct_change'
        },

        # Labor Market
        'nonfarm_payrolls': {
            'series_id': 'PAYEMS',
            'name': 'Total Nonfarm Payrolls',
            'frequency': 'monthly',
            'transform': 'diff'  # Change in thousands
        },
        'unemployment_rate': {
            'series_id': 'UNRATE',
            'name': 'Unemployment Rate',
            'frequency': 'monthly',
            'transform': 'level'
        },
        'initial_claims': {
            'series_id': 'ICSA',
            'name': 'Initial Jobless Claims',
            'frequency': 'weekly',
            'transform': 'level'
        },
        'continuing_claims': {
            'series_id': 'CCSA',
            'name': 'Continuing Claims',
            'frequency': 'weekly',
            'transform': 'level'
        },
        'avg_hourly_earnings': {
            'series_id': 'CES0500000003',
            'name': 'Average Hourly Earnings',
            'frequency': 'monthly',
            'transform': 'pct_change_yoy'
        },
        'labor_force_participation': {
            'series_id': 'CIVPART',
            'name': 'Labor Force Participation Rate',
            'frequency': 'monthly',
            'transform': 'level'
        },

        # Inflation
        'cpi_headline': {
            'series_id': 'CPIAUCSL',
            'name': 'CPI All Items',
            'frequency': 'monthly',
            'transform': 'pct_change_yoy'
        },
        'cpi_core': {
            'series_id': 'CPILFESL',
            'name': 'CPI Core (ex food & energy)',
            'frequency': 'monthly',
            'transform': 'pct_change_yoy'
        },
        'ppi_headline': {
            'series_id': 'PPIACO',
            'name': 'PPI All Commodities',
            'frequency': 'monthly',
            'transform': 'pct_change_yoy'
        },
        'pce_deflator': {
            'series_id': 'PCEPI',
            'name': 'PCE Price Index',
            'frequency': 'monthly',
            'transform': 'pct_change_yoy'
        },
        'breakeven_5y': {
            'series_id': 'T5YIE',
            'name': '5-Year Breakeven Inflation',
            'frequency': 'daily',
            'transform': 'level'
        },
        'breakeven_10y': {
            'series_id': 'T10YIE',
            'name': '10-Year Breakeven Inflation',
            'frequency': 'daily',
            'transform': 'level'
        },

        # Monetary Policy & Interest Rates
        'fed_funds_rate': {
            'series_id': 'FEDFUNDS',
            'name': 'Effective Federal Funds Rate',
            'frequency': 'monthly',
            'transform': 'level'
        },
        'fed_funds_target_upper': {
            'series_id': 'DFEDTARU',
            'name': 'Fed Funds Target Upper',
            'frequency': 'daily',
            'transform': 'level'
        },
        'treasury_3m': {
            'series_id': 'DTB3',
            'name': '3-Month Treasury Bill',
            'frequency': 'daily',
            'transform': 'level'
        },
        'treasury_2y': {
            'series_id': 'DGS2',
            'name': '2-Year Treasury Yield',
            'frequency': 'daily',
            'transform': 'level'
        },
        'treasury_5y': {
            'series_id': 'DGS5',
            'name': '5-Year Treasury Yield',
            'frequency': 'daily',
            'transform': 'level'
        },
        'treasury_10y': {
            'series_id': 'DGS10',
            'name': '10-Year Treasury Yield',
            'frequency': 'daily',
            'transform': 'level'
        },
        'treasury_30y': {
            'series_id': 'DGS30',
            'name': '30-Year Treasury Yield',
            'frequency': 'daily',
            'transform': 'level'
        },
        'fed_balance_sheet': {
            'series_id': 'WALCL',
            'name': 'Fed Total Assets',
            'frequency': 'weekly',
            'transform': 'pct_change'
        },

        # Credit & Spreads
        'corporate_baa': {
            'series_id': 'DBAA',
            'name': 'Moodys BAA Corporate Bond Yield',
            'frequency': 'daily',
            'transform': 'level'
        },
        'corporate_aaa': {
            'series_id': 'DAAA',
            'name': 'Moodys AAA Corporate Bond Yield',
            'frequency': 'daily',
            'transform': 'level'
        },
        'high_yield_spread': {
            'series_id': 'BAMLH0A0HYM2',
            'name': 'High Yield Spread',
            'frequency': 'daily',
            'transform': 'level'
        },
        'ted_spread': {
            'series_id': 'TEDRATE',
            'name': 'TED Spread',
            'frequency': 'daily',
            'transform': 'level'
        },

        # Housing
        'housing_starts': {
            'series_id': 'HOUST',
            'name': 'Housing Starts',
            'frequency': 'monthly',
            'transform': 'pct_change'
        },
        'building_permits': {
            'series_id': 'PERMIT',
            'name': 'Building Permits',
            'frequency': 'monthly',
            'transform': 'pct_change'
        },
        'existing_home_sales': {
            'series_id': 'EXHOSLUSM495S',
            'name': 'Existing Home Sales',
            'frequency': 'monthly',
            'transform': 'pct_change'
        },
        'case_shiller': {
            'series_id': 'CSUSHPINSA',
            'name': 'Case-Shiller Home Price Index',
            'frequency': 'monthly',
            'transform': 'pct_change_yoy'
        },

        # Trade
        'trade_balance': {
            'series_id': 'BOPGSTB',
            'name': 'Trade Balance',
            'frequency': 'monthly',
            'transform': 'level'
        },
        'dollar_index': {
            'series_id': 'DTWEXBGS',
            'name': 'Trade Weighted Dollar Index',
            'frequency': 'daily',
            'transform': 'pct_change'
        },

        # Sentiment & Leading Indicators
        'consumer_sentiment': {
            'series_id': 'UMCSENT',
            'name': 'U of Michigan Consumer Sentiment',
            'frequency': 'monthly',
            'transform': 'level'
        },
        'consumer_confidence': {
            'series_id': 'CSCICP03USM665S',
            'name': 'Consumer Confidence (OECD)',
            'frequency': 'monthly',
            'transform': 'level'
        },
        'leading_index': {
            'series_id': 'USSLIND',
            'name': 'Leading Index',
            'frequency': 'monthly',
            'transform': 'level'
        },

        # Financial Conditions
        'chicago_fed_index': {
            'series_id': 'NFCI',
            'name': 'Chicago Fed Financial Conditions Index',
            'frequency': 'weekly',
            'transform': 'level'
        },
        'st_louis_stress': {
            'series_id': 'STLFSI4',
            'name': 'St. Louis Fed Financial Stress Index',
            'frequency': 'weekly',
            'transform': 'level'
        },
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED collector.

        Args:
            api_key: FRED API key. If None, looks for FRED_API_KEY env variable.
        """
        if not FRED_AVAILABLE:
            raise ImportError("fredapi not installed. Run: pip install fredapi")

        self.api_key = api_key or os.environ.get('FRED_API_KEY')

        if not self.api_key:
            logger.warning(
                "No FRED API key provided. Get a FREE key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            self.fred = None
        else:
            self.fred = Fred(api_key=self.api_key)
            logger.info("FRED API connected successfully")

    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Fetch a single FRED series.

        Args:
            series_id: FRED series ID (e.g., 'GDP', 'UNRATE')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            pandas Series with datetime index
        """
        if self.fred is None:
            raise ValueError("FRED API key not configured")

        try:
            data = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            logger.info(f"Fetched {series_id}: {len(data)} observations")
            return data
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {e}")
            return pd.Series(dtype=float)

    def get_all_series(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        series_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch multiple FRED series and combine into DataFrame.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            series_list: List of series keys from SERIES_CONFIG. If None, fetch all.

        Returns:
            DataFrame with all series (daily frequency, forward-filled)
        """
        if series_list is None:
            series_list = list(self.SERIES_CONFIG.keys())

        all_data = {}

        for key in series_list:
            if key not in self.SERIES_CONFIG:
                logger.warning(f"Unknown series key: {key}")
                continue

            config = self.SERIES_CONFIG[key]
            series_id = config['series_id']

            try:
                data = self.get_series(series_id, start_date, end_date)
                if len(data) > 0:
                    all_data[key] = data
            except Exception as e:
                logger.error(f"Failed to fetch {key}: {e}")

        if not all_data:
            return pd.DataFrame()

        # Combine all series
        df = pd.DataFrame(all_data)

        # Resample to daily frequency and forward fill
        df.index = pd.to_datetime(df.index)
        df = df.resample('D').last().ffill()

        logger.info(f"Combined {len(df.columns)} series, {len(df)} days")
        return df

    def calculate_features(
        self,
        df: pd.DataFrame,
        include_derived: bool = True
    ) -> pd.DataFrame:
        """
        Calculate derived features from raw FRED data.

        Args:
            df: DataFrame with raw FRED series
            include_derived: Whether to add derived features

        Returns:
            DataFrame with transformed features
        """
        features = pd.DataFrame(index=df.index)

        # Apply transforms based on config
        for key, config in self.SERIES_CONFIG.items():
            if key not in df.columns:
                continue

            series = df[key]
            transform = config.get('transform', 'level')

            if transform == 'level':
                features[f'fred_{key}'] = series
            elif transform == 'pct_change':
                features[f'fred_{key}'] = series.pct_change()
            elif transform == 'pct_change_yoy':
                features[f'fred_{key}_yoy'] = series.pct_change(252)  # ~1 year of trading days
                features[f'fred_{key}_mom'] = series.pct_change(21)   # ~1 month
            elif transform == 'diff':
                features[f'fred_{key}_diff'] = series.diff()

        if include_derived:
            features = self._add_derived_features(features, df)

        return features

    def _add_derived_features(self, features: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Add derived/computed features."""

        # Yield Curve Features
        if 'treasury_10y' in raw_df.columns and 'treasury_2y' in raw_df.columns:
            features['fred_yield_curve_10y_2y'] = raw_df['treasury_10y'] - raw_df['treasury_2y']
            # Inverted yield curve signal
            features['fred_yield_curve_inverted'] = (features['fred_yield_curve_10y_2y'] < 0).astype(int)

        if 'treasury_10y' in raw_df.columns and 'treasury_3m' in raw_df.columns:
            features['fred_yield_curve_10y_3m'] = raw_df['treasury_10y'] - raw_df['treasury_3m']

        if 'treasury_30y' in raw_df.columns and 'treasury_2y' in raw_df.columns:
            features['fred_yield_curve_30y_2y'] = raw_df['treasury_30y'] - raw_df['treasury_2y']

        # Credit Spreads
        if 'corporate_baa' in raw_df.columns and 'treasury_10y' in raw_df.columns:
            features['fred_credit_spread_baa'] = raw_df['corporate_baa'] - raw_df['treasury_10y']

        if 'corporate_aaa' in raw_df.columns and 'treasury_10y' in raw_df.columns:
            features['fred_credit_spread_aaa'] = raw_df['corporate_aaa'] - raw_df['treasury_10y']

        if 'corporate_baa' in raw_df.columns and 'corporate_aaa' in raw_df.columns:
            features['fred_credit_spread_baa_aaa'] = raw_df['corporate_baa'] - raw_df['corporate_aaa']

        # Real Interest Rate (nominal - breakeven inflation)
        if 'treasury_10y' in raw_df.columns and 'breakeven_10y' in raw_df.columns:
            features['fred_real_rate_10y'] = raw_df['treasury_10y'] - raw_df['breakeven_10y']

        if 'treasury_5y' in raw_df.columns and 'breakeven_5y' in raw_df.columns:
            features['fred_real_rate_5y'] = raw_df['treasury_5y'] - raw_df['breakeven_5y']

        # Rate of change features for key indicators
        for col in ['treasury_10y', 'high_yield_spread', 'dollar_index']:
            if col in raw_df.columns:
                # 5-day momentum
                features[f'fred_{col}_mom_5d'] = raw_df[col].diff(5)
                # 20-day momentum
                features[f'fred_{col}_mom_20d'] = raw_df[col].diff(20)

        # Financial conditions changes
        for col in ['chicago_fed_index', 'st_louis_stress']:
            if col in raw_df.columns:
                features[f'fred_{col}_change'] = raw_df[col].diff()

        return features

    def align_to_trading_data(
        self,
        macro_df: pd.DataFrame,
        trading_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align macroeconomic data to trading data timestamps.

        Uses forward fill to ensure no look-ahead bias
        (only uses data that was available at each point in time).

        Args:
            macro_df: DataFrame with macroeconomic features
            trading_df: DataFrame with trading data (target index)

        Returns:
            Macroeconomic features aligned to trading timestamps
        """
        # Ensure both have datetime index
        macro_df.index = pd.to_datetime(macro_df.index)

        if hasattr(trading_df.index, 'tz'):
            # Remove timezone for alignment
            trading_dates = trading_df.index.tz_localize(None).normalize().unique()
        else:
            trading_dates = pd.to_datetime(trading_df.index).normalize().unique()

        # Reindex macro data to trading dates with forward fill
        # This ensures we only use data that was available at each point
        aligned = macro_df.reindex(trading_dates, method='ffill')

        logger.info(f"Aligned macro data: {len(aligned)} days")
        return aligned


def create_sample_macro_features() -> pd.DataFrame:
    """
    Create sample macroeconomic features without API key.

    Returns synthetic data for testing purposes.
    """
    logger.info("Creating sample macroeconomic features (no API key)")

    # Create date range
    dates = pd.date_range(start='2019-09-01', end='2019-11-30', freq='D')

    # Create synthetic data
    np.random.seed(42)
    n = len(dates)

    features = pd.DataFrame(index=dates)

    # Treasury yields (realistic ranges)
    features['fred_treasury_2y'] = 1.5 + np.random.randn(n).cumsum() * 0.01
    features['fred_treasury_10y'] = 1.8 + np.random.randn(n).cumsum() * 0.01
    features['fred_treasury_30y'] = 2.2 + np.random.randn(n).cumsum() * 0.01

    # Yield curve
    features['fred_yield_curve_10y_2y'] = features['fred_treasury_10y'] - features['fred_treasury_2y']
    features['fred_yield_curve_inverted'] = (features['fred_yield_curve_10y_2y'] < 0).astype(int)

    # Unemployment (stable around 3.5%)
    features['fred_unemployment_rate'] = 3.5 + np.random.randn(n) * 0.1

    # CPI YoY (around 2%)
    features['fred_cpi_headline_yoy'] = 0.02 + np.random.randn(n) * 0.002

    # Fed Funds rate
    features['fred_fed_funds_rate'] = 1.75 + np.random.randn(n).cumsum() * 0.001

    # Credit spreads
    features['fred_high_yield_spread'] = 3.5 + np.random.randn(n) * 0.2

    # Dollar index
    features['fred_dollar_index'] = np.random.randn(n) * 0.001

    # Financial conditions
    features['fred_chicago_fed_index'] = np.random.randn(n) * 0.1
    features['fred_st_louis_stress'] = np.random.randn(n) * 0.1

    return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("FRED Collector Test")
    print("=" * 60)

    # Check if API key is available
    api_key = os.environ.get('FRED_API_KEY')

    if api_key:
        print("\nFRED API key found. Fetching real data...")
        collector = FREDCollector(api_key)

        # Fetch subset of series for testing
        test_series = [
            'treasury_2y', 'treasury_10y', 'unemployment_rate',
            'cpi_headline', 'fed_funds_rate', 'high_yield_spread'
        ]

        raw_data = collector.get_all_series(
            start_date='2019-01-01',
            end_date='2019-12-31',
            series_list=test_series
        )

        print(f"\nRaw data shape: {raw_data.shape}")
        print(f"Columns: {raw_data.columns.tolist()}")

        # Calculate features
        features = collector.calculate_features(raw_data)
        print(f"\nFeatures shape: {features.shape}")
        print(f"Feature columns: {features.columns.tolist()}")

    else:
        print("\nNo FRED API key found. Using sample data...")
        print("To use real data, get a FREE key at:")
        print("  https://fred.stlouisfed.org/docs/api/api_key.html")
        print("\nThen set: FRED_API_KEY=your_key_here")

        features = create_sample_macro_features()
        print(f"\nSample features shape: {features.shape}")
        print(f"Columns: {features.columns.tolist()}")
        print(f"\nSample data:")
        print(features.head())

    print("\n" + "=" * 60)
    print("FRED Collector Test Complete")
    print("=" * 60)
