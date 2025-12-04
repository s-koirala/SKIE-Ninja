"""
Sentiment & Positioning Features (Category 5)
==============================================
Implements market sentiment and positioning features.

Data Sources:
- VIX: CBOE Volatility Index (from FRED)
- COT: Commitments of Traders (from CFTC)
- Put/Call Ratio: Options market sentiment
- VIX term structure: VIX futures contango/backwardation

Reference: research/02_comprehensive_variables_research.md Section 14
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from pathlib import Path
import logging
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SentimentFeatures:
    """Calculate sentiment and positioning features."""

    # VIX series from FRED
    VIX_SERIES = {
        'vix': 'VIXCLS',           # VIX daily close
        'vix_3m': 'VXVCLS',        # 3-month VIX
    }

    # COT data URLs from CFTC
    COT_URL = "https://www.cftc.gov/dea/newcot/deafut.txt"

    def __init__(
        self,
        trading_df: pd.DataFrame,
        fred_api_key: Optional[str] = None
    ):
        """
        Initialize sentiment features calculator.

        Args:
            trading_df: OHLCV DataFrame with datetime index
            fred_api_key: FRED API key for VIX data (optional)
        """
        self.trading_df = trading_df.copy()
        self.fred_api_key = fred_api_key
        self._validate_data()

    def _validate_data(self):
        """Validate input data."""
        if not isinstance(self.trading_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

    def calculate_all(self) -> pd.DataFrame:
        """Calculate all sentiment features."""
        logger.info("Calculating sentiment features...")

        features = pd.DataFrame(index=self.trading_df.index)

        # VIX-based features
        vix_features = self._calculate_vix_features()
        if vix_features is not None:
            features = pd.concat([features, vix_features], axis=1)

        # COT-based features
        cot_features = self._calculate_cot_features()
        if cot_features is not None:
            features = pd.concat([features, cot_features], axis=1)

        # Price-derived sentiment proxies
        sentiment_proxies = self._calculate_sentiment_proxies()
        features = pd.concat([features, sentiment_proxies], axis=1)

        logger.info(f"Generated {len(features.columns)} sentiment features")
        return features

    def _calculate_vix_features(self) -> Optional[pd.DataFrame]:
        """Calculate VIX-based features."""
        try:
            if self.fred_api_key:
                vix_data = self._fetch_vix_from_fred()
            else:
                vix_data = self._create_sample_vix()

            if vix_data is None or len(vix_data) == 0:
                return None

            df = pd.DataFrame(index=self.trading_df.index)

            # Align VIX to trading data
            if hasattr(self.trading_df.index, 'tz'):
                trading_dates = self.trading_df.index.tz_localize(None).normalize()
            else:
                trading_dates = pd.to_datetime(self.trading_df.index).normalize()

            vix_data.index = pd.to_datetime(vix_data.index)

            # Map VIX values to trading timestamps
            for col in vix_data.columns:
                date_values = vix_data[col].to_dict()
                df[col] = trading_dates.map(lambda d: date_values.get(d.normalize(), np.nan))
                df[col] = df[col].ffill()

            # VIX level
            if 'vix' in df.columns:
                vix = df['vix']

                # VIX change features
                df['vix_change_1d'] = vix.pct_change()
                df['vix_change_5d'] = vix.pct_change(5)

                # VIX relative to moving averages
                df['vix_sma10_ratio'] = vix / vix.rolling(10).mean()
                df['vix_sma20_ratio'] = vix / vix.rolling(20).mean()

                # VIX percentile (20-day rolling)
                df['vix_percentile_20'] = vix.rolling(20).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
                )

                # VIX regime (high/low volatility)
                df['vix_regime_high'] = (vix > 25).astype(int)
                df['vix_regime_extreme'] = (vix > 30).astype(int)
                df['vix_regime_low'] = (vix < 15).astype(int)

                # VIX mean reversion signals
                vix_ma20 = vix.rolling(20).mean()
                vix_std20 = vix.rolling(20).std()
                df['vix_zscore'] = (vix - vix_ma20) / (vix_std20 + 1e-10)

                # VIX spike detection (>2 std above mean)
                df['vix_spike'] = (df['vix_zscore'] > 2).astype(int)

            # VIX term structure (if 3-month VIX available)
            if 'vix_3m' in df.columns and 'vix' in df.columns:
                # Contango: VIX < VIX3M (normal), Backwardation: VIX > VIX3M (fear)
                df['vix_term_spread'] = df['vix_3m'] - df['vix']
                df['vix_contango'] = (df['vix_term_spread'] > 0).astype(int)
                df['vix_backwardation'] = (df['vix_term_spread'] < 0).astype(int)

            return df

        except Exception as e:
            logger.error(f"Error calculating VIX features: {e}")
            return None

    def _fetch_vix_from_fred(self) -> Optional[pd.DataFrame]:
        """Fetch VIX data from FRED API."""
        try:
            from fredapi import Fred
            fred = Fred(api_key=self.fred_api_key)

            start_date = self.trading_df.index.min().strftime('%Y-%m-%d')
            end_date = self.trading_df.index.max().strftime('%Y-%m-%d')

            data = {}
            for name, series_id in self.VIX_SERIES.items():
                try:
                    series = fred.get_series(
                        series_id,
                        observation_start=start_date,
                        observation_end=end_date
                    )
                    data[name] = series
                except Exception as e:
                    logger.warning(f"Could not fetch {series_id}: {e}")

            if data:
                return pd.DataFrame(data)
            return None

        except ImportError:
            logger.warning("fredapi not installed, using sample data")
            return None
        except Exception as e:
            logger.error(f"Error fetching VIX from FRED: {e}")
            return None

    def _create_sample_vix(self) -> pd.DataFrame:
        """Create sample VIX data for development."""
        logger.info("Using sample VIX data (no API key)")

        # Get date range from trading data
        if hasattr(self.trading_df.index, 'tz'):
            dates = self.trading_df.index.tz_localize(None).normalize().unique()
        else:
            dates = pd.to_datetime(self.trading_df.index).normalize().unique()

        # Generate realistic VIX values (typically 12-30, with occasional spikes)
        np.random.seed(42)
        n_days = len(dates)

        # Mean-reverting VIX with volatility clustering
        vix = np.zeros(n_days)
        vix[0] = 18  # Start at typical level

        for i in range(1, n_days):
            # Mean reversion + random shock
            mean_level = 18
            reversion_speed = 0.1
            volatility = 1.5

            vix[i] = vix[i-1] + reversion_speed * (mean_level - vix[i-1]) + \
                     volatility * np.random.randn()

            # Add occasional spikes
            if np.random.rand() < 0.02:
                vix[i] += np.random.uniform(5, 15)

            # Keep in realistic range
            vix[i] = np.clip(vix[i], 9, 80)

        # VIX 3-month (typically higher - contango)
        vix_3m = vix + np.random.uniform(1, 3, n_days)

        return pd.DataFrame({
            'vix': vix,
            'vix_3m': vix_3m
        }, index=dates)

    def _calculate_cot_features(self) -> Optional[pd.DataFrame]:
        """Calculate COT (Commitments of Traders) features."""
        try:
            # For development, create sample COT data
            # In production, fetch from CFTC
            cot_data = self._create_sample_cot()

            if cot_data is None or len(cot_data) == 0:
                return None

            df = pd.DataFrame(index=self.trading_df.index)

            # Align COT to trading data (COT released weekly on Friday)
            if hasattr(self.trading_df.index, 'tz'):
                trading_dates = self.trading_df.index.tz_localize(None).normalize()
            else:
                trading_dates = pd.to_datetime(self.trading_df.index).normalize()

            cot_data.index = pd.to_datetime(cot_data.index)

            # Map COT values to trading timestamps (forward fill for weekly data)
            for col in cot_data.columns:
                date_values = cot_data[col].to_dict()
                df[col] = trading_dates.map(lambda d: date_values.get(d.normalize(), np.nan))
                df[col] = df[col].ffill()

            # COT positioning features
            if 'commercial_net' in df.columns:
                # Commercial net position (hedgers - often contrarian indicator)
                df['cot_commercial_net'] = df['commercial_net']
                df['cot_commercial_change'] = df['commercial_net'].diff()

                # Commercial positioning percentile
                df['cot_commercial_pctl'] = df['commercial_net'].rolling(52).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
                )

            if 'noncommercial_net' in df.columns:
                # Non-commercial (speculators) - often trend-following
                df['cot_spec_net'] = df['noncommercial_net']
                df['cot_spec_change'] = df['noncommercial_net'].diff()

                # Speculator percentile
                df['cot_spec_pctl'] = df['noncommercial_net'].rolling(52).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
                )

            # COT sentiment spread (spec vs commercial)
            if 'commercial_net' in df.columns and 'noncommercial_net' in df.columns:
                df['cot_spread'] = df['noncommercial_net'] - df['commercial_net']
                df['cot_spread_zscore'] = (
                    df['cot_spread'] - df['cot_spread'].rolling(52).mean()
                ) / (df['cot_spread'].rolling(52).std() + 1e-10)

            return df

        except Exception as e:
            logger.error(f"Error calculating COT features: {e}")
            return None

    def _create_sample_cot(self) -> pd.DataFrame:
        """Create sample COT data for development."""
        logger.info("Using sample COT data")

        # Get date range from trading data
        if hasattr(self.trading_df.index, 'tz'):
            dates = self.trading_df.index.tz_localize(None).normalize().unique()
        else:
            dates = pd.to_datetime(self.trading_df.index).normalize().unique()

        # Generate weekly COT dates (Tuesdays)
        cot_dates = [d for d in dates if d.dayofweek == 1]
        if len(cot_dates) == 0:
            cot_dates = dates[::7]  # Every 7 days

        np.random.seed(123)
        n_weeks = len(cot_dates)

        # Commercial positions (hedgers) - typically short on average
        commercial_net = np.cumsum(np.random.randn(n_weeks) * 5000) - 50000

        # Non-commercial (speculators) - typically long on average
        noncommercial_net = np.cumsum(np.random.randn(n_weeks) * 5000) + 50000

        return pd.DataFrame({
            'commercial_net': commercial_net,
            'noncommercial_net': noncommercial_net
        }, index=cot_dates)

    def _calculate_sentiment_proxies(self) -> pd.DataFrame:
        """Calculate sentiment proxies from price action."""
        df = pd.DataFrame(index=self.trading_df.index)

        close = self.trading_df['close']
        high = self.trading_df['high']
        low = self.trading_df['low']
        volume = self.trading_df['volume']

        # Advance/Decline proxy (using up/down bars)
        returns = close.pct_change()
        up_bars = (returns > 0).astype(int)
        down_bars = (returns < 0).astype(int)

        # AD line proxy (cumulative up - down)
        df['ad_line_proxy'] = (up_bars - down_bars).cumsum()

        # McClellan-style oscillator (19/39 EMA of AD)
        ad_diff = up_bars - down_bars
        df['mcclellan_proxy'] = ad_diff.ewm(span=19).mean() - ad_diff.ewm(span=39).mean()

        # New Highs/Lows proxy (rolling 20-day)
        df['new_high_20'] = (close >= close.rolling(20).max()).astype(int)
        df['new_low_20'] = (close <= close.rolling(20).min()).astype(int)
        df['high_low_diff'] = df['new_high_20'] - df['new_low_20']

        # Momentum breadth (% of rolling returns positive)
        for period in [5, 10, 20]:
            pos_returns = (close.pct_change(period) > 0).astype(int)
            df[f'momentum_breadth_{period}'] = pos_returns.rolling(10).mean()

        # Fear/Greed proxy based on price action
        # High volatility + down trend = fear
        # Low volatility + up trend = greed
        vol_20 = returns.rolling(20).std()
        trend_20 = (close - close.shift(20)) / close.shift(20)

        df['fear_index_proxy'] = vol_20 * (-trend_20) * 100
        df['greed_index_proxy'] = (1 / (vol_20 + 0.001)) * trend_20 * 0.01

        # Put/Call ratio proxy (using volume on down vs up days)
        vol_up = volume * up_bars
        vol_down = volume * down_bars
        df['put_call_proxy'] = vol_down.rolling(5).sum() / (vol_up.rolling(5).sum() + 1)

        # Sentiment extremes
        df['extreme_fear'] = (df['fear_index_proxy'] > df['fear_index_proxy'].rolling(50).quantile(0.9)).astype(int)
        df['extreme_greed'] = (df['greed_index_proxy'] > df['greed_index_proxy'].rolling(50).quantile(0.9)).astype(int)

        return df


def calculate_sentiment_features(
    trading_df: pd.DataFrame,
    fred_api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to calculate all sentiment features.

    Args:
        trading_df: OHLCV DataFrame
        fred_api_key: Optional FRED API key

    Returns:
        DataFrame with sentiment features
    """
    calculator = SentimentFeatures(trading_df, fred_api_key)
    return calculator.calculate_all()


if __name__ == "__main__":
    # Test with sample data
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from data_collection.ninjatrader_loader import load_sample_data
    except ImportError:
        from src.python.data_collection.ninjatrader_loader import load_sample_data

    print("=" * 60)
    print("Sentiment Features Test")
    print("=" * 60)

    es_data, _ = load_sample_data()

    # Calculate features
    features = calculate_sentiment_features(es_data)

    print(f"\nGenerated {len(features.columns)} sentiment features:")
    for i, col in enumerate(features.columns, 1):
        print(f"  {i:2d}. {col}")

    print(f"\nSample output (row 500):")
    sample_row = features.iloc[500]
    for col, val in sample_row.items():
        print(f"  {col}: {val:.4f}" if pd.notna(val) else f"  {col}: NaN")

    print(f"\nFeature statistics:")
    print(features.describe().T[['mean', 'std', 'min', 'max']])
