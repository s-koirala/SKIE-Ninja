"""
Price-Based Features (Category 1)
=================================
Implements ~50 price-based features for ML models.

Categories:
- Raw price data (OHLC, mid-price)
- Returns (simple, log, multi-period)
- Price ratios
- Range metrics

Reference: research/02_comprehensive_variables_research.md Section 14
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class PriceFeatures:
    """Calculate price-based features from OHLCV data."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        self.df = df.copy()
        self._validate_columns()

    def _validate_columns(self):
        """Ensure required columns exist."""
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def calculate_all(self) -> pd.DataFrame:
        """Calculate all price-based features."""
        logger.info("Calculating all price-based features...")

        # Start with OHLCV
        features = self.df[['open', 'high', 'low', 'close', 'volume']].copy()

        # Add derived features
        features = self._add_derived_prices(features)
        features = self._add_returns(features)
        features = self._add_log_returns(features)
        features = self._add_price_ratios(features)
        features = self._add_range_metrics(features)

        logger.info(f"Generated {len(features.columns)} price-based features")
        return features

    def _add_derived_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived price measures."""
        # Mid price (for futures with bid/ask, this approximates)
        df['mid_price'] = (df['high'] + df['low']) / 2

        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # Weighted close
        df['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4

        # OHLC average
        df['ohlc_avg'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        return df

    def _add_returns(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """Add simple returns for multiple periods."""
        if periods is None:
            periods = [1, 2, 3, 5, 10, 20, 60]

        for period in periods:
            df[f'return_{period}'] = df['close'].pct_change(period)

        return df

    def _add_log_returns(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """Add log returns for multiple periods."""
        if periods is None:
            periods = [1, 5, 20]

        for period in periods:
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))

        return df

    def _add_price_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price ratio features."""
        # High/Low ratio
        df['high_low_ratio'] = df['high'] / df['low']

        # Close/Open ratio
        df['close_open_ratio'] = df['close'] / df['open']

        # Current close vs previous close
        df['close_prev_ratio'] = df['close'] / df['close'].shift(1)

        # Close relative to day's range (normalized position)
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        # Open relative to previous close
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # Close relative to rolling averages
        for period in [10, 20, 50]:
            df[f'close_to_sma_{period}'] = df['close'] / df['close'].rolling(period).mean()

        return df

    def _add_range_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add range-based metrics."""
        # True Range components
        df['hl_range'] = df['high'] - df['low']
        df['hc_range'] = abs(df['high'] - df['close'].shift(1))
        df['lc_range'] = abs(df['low'] - df['close'].shift(1))

        # True Range
        df['true_range'] = df[['hl_range', 'hc_range', 'lc_range']].max(axis=1)

        # ATR (Average True Range) for multiple periods
        for period in [7, 14, 20]:
            df[f'atr_{period}'] = df['true_range'].rolling(period).mean()

        # Range as percentage of price
        df['range_pct'] = df['hl_range'] / df['close'] * 100

        # Daily range percentile (vs 20-day rolling)
        df['range_percentile_20'] = df['hl_range'].rolling(20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # Body size (absolute)
        df['body_size'] = abs(df['close'] - df['open'])

        # Body ratio (body / total range)
        df['body_ratio'] = df['body_size'] / (df['hl_range'] + 1e-10)

        # Upper/Lower shadow
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

        # Clean up intermediate columns
        df = df.drop(['hl_range', 'hc_range', 'lc_range'], axis=1)

        return df


def calculate_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to calculate all price-based features.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with all price-based features
    """
    calculator = PriceFeatures(df)
    return calculator.calculate_all()


if __name__ == "__main__":
    # Test with sample data
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_collection.ninjatrader_loader import load_sample_data

    print("=" * 60)
    print("Price Features Test")
    print("=" * 60)

    es_data, _ = load_sample_data()

    # Calculate features
    features = calculate_price_features(es_data)

    print(f"\nGenerated {len(features.columns)} features:")
    print(features.columns.tolist())

    print(f"\nSample output (first 5 rows, selected columns):")
    sample_cols = ['close', 'return_1', 'log_return_1', 'atr_14', 'range_pct', 'body_ratio']
    print(features[sample_cols].head())

    print(f"\nFeature statistics:")
    print(features.describe().T[['mean', 'std', 'min', 'max']])
