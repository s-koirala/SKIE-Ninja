"""
Microstructure Features (Category 4)
====================================
Implements market microstructure features from OHLCV data.

When Level 2 order book data is available, additional features
can be calculated (bid-ask spread, order flow imbalance, etc.)

Categories:
- Volume-based metrics (from bars)
- Trade intensity proxies
- Price impact measures
- Volume profile analysis
- Intraday patterns

Reference: research/02_comprehensive_variables_research.md Section 14
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class MicrostructureFeatures:
    """Calculate microstructure features from OHLCV data."""

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
        """Calculate all microstructure features."""
        logger.info("Calculating microstructure features...")

        features = pd.DataFrame(index=self.df.index)

        # Volume-based metrics
        features = self._add_volume_metrics(features)

        # Trade intensity proxies
        features = self._add_trade_intensity(features)

        # Price impact measures
        features = self._add_price_impact(features)

        # Volume profile analysis
        features = self._add_volume_profile(features)

        # Intraday session metrics (if datetime index)
        features = self._add_intraday_metrics(features)

        # Order flow proxies (from bar data)
        features = self._add_order_flow_proxies(features)

        logger.info(f"Generated {len(features.columns)} microstructure features")
        return features

    def _add_volume_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based microstructure metrics."""
        volume = self.df['volume']
        close = self.df['close']

        # Relative volume (vs rolling average)
        for period in [10, 20, 50]:
            vol_ma = volume.rolling(period).mean()
            df[f'relative_volume_{period}'] = volume / (vol_ma + 1e-10)

        # Volume momentum
        df['volume_momentum_5'] = volume.pct_change(5)
        df['volume_momentum_20'] = volume.pct_change(20)

        # Volume acceleration
        df['volume_accel'] = volume.diff().diff()

        # Volume standard score (z-score)
        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std()
        df['volume_zscore'] = (volume - vol_mean) / (vol_std + 1e-10)

        # Volume concentration (high vs low volume bars)
        vol_rank = volume.rolling(20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        df['volume_percentile_20'] = vol_rank

        # Dollar volume (proxy for liquidity)
        df['dollar_volume'] = volume * close
        df['dollar_volume_ma20'] = df['dollar_volume'].rolling(20).mean()

        # Volume-weighted metrics
        df['volume_weighted_range'] = (self.df['high'] - self.df['low']) * volume
        df['volume_weighted_body'] = abs(close - self.df['open']) * volume

        return df

    def _add_trade_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trade intensity proxy metrics."""
        volume = self.df['volume']
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        # Volume per unit price change (liquidity metric)
        price_range = high - low + 1e-10
        df['volume_per_range'] = volume / price_range

        # Average trade size proxy (volume / implied trade count)
        # Higher values suggest larger institutional activity
        df['avg_trade_size_proxy'] = volume / (price_range * 100 + 1)

        # Trade intensity (volume velocity)
        df['trade_intensity_5'] = volume.rolling(5).sum()
        df['trade_intensity_20'] = volume.rolling(20).sum()

        # Volume distribution (up vs down bars)
        returns = close.pct_change()
        df['up_volume'] = np.where(returns > 0, volume, 0)
        df['down_volume'] = np.where(returns < 0, volume, 0)

        # Up/Down volume ratio (rolling)
        up_vol_sum = pd.Series(df['up_volume']).rolling(20).sum()
        down_vol_sum = pd.Series(df['down_volume']).rolling(20).sum()
        df['up_down_volume_ratio'] = up_vol_sum / (down_vol_sum + 1e-10)

        # Volume-weighted return direction
        df['volume_weighted_direction'] = returns * volume

        # Cumulative volume delta (proxy)
        df['cum_volume_delta'] = df['volume_weighted_direction'].cumsum()
        df['volume_delta_ma20'] = df['volume_weighted_direction'].rolling(20).mean()

        return df

    def _add_price_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price impact measures."""
        volume = self.df['volume']
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']

        # Amihud illiquidity measure (|return| / volume)
        returns = close.pct_change().abs()
        df['amihud_illiquidity'] = returns / (volume + 1e-10)
        df['amihud_ma20'] = df['amihud_illiquidity'].rolling(20).mean()

        # Kyle's lambda proxy (price impact per unit volume)
        price_change = close.diff().abs()
        df['kyle_lambda_proxy'] = price_change / (volume + 1e-10)
        df['kyle_lambda_ma20'] = df['kyle_lambda_proxy'].rolling(20).mean()

        # Range impact (range / volume)
        df['range_impact'] = (high - low) / (volume + 1e-10)
        df['range_impact_ma20'] = df['range_impact'].rolling(20).mean()

        # Volume-normalized return (standardized by volume)
        df['volume_normalized_return'] = close.pct_change() / (np.sqrt(volume) + 1e-10)

        # Price efficiency ratio (directional move / total range)
        total_range = (high - low).rolling(20).sum()
        net_move = close.diff(20).abs()
        df['price_efficiency'] = net_move / (total_range + 1e-10)

        return df

    def _add_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume profile analysis features."""
        volume = self.df['volume']
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']

        # VWAP calculation
        typical_price = (high + low + close) / 3
        cum_vol = volume.cumsum()
        cum_vwap = (typical_price * volume).cumsum()
        df['vwap'] = cum_vwap / (cum_vol + 1e-10)

        # Distance from VWAP
        df['vwap_distance'] = (close - df['vwap']) / df['vwap'] * 100
        df['vwap_distance_abs'] = df['vwap_distance'].abs()

        # Rolling VWAP (20-period)
        roll_vol = volume.rolling(20).sum()
        roll_vwap = (typical_price * volume).rolling(20).sum()
        df['vwap_20'] = roll_vwap / (roll_vol + 1e-10)
        df['vwap_20_distance'] = (close - df['vwap_20']) / df['vwap_20'] * 100

        # Volume-weighted average price bands (std)
        vwap_std = ((close - df['vwap_20']) ** 2 * volume).rolling(20).sum()
        vwap_std = np.sqrt(vwap_std / (roll_vol + 1e-10))
        df['vwap_upper_band'] = df['vwap_20'] + 2 * vwap_std
        df['vwap_lower_band'] = df['vwap_20'] - 2 * vwap_std
        df['vwap_band_position'] = (close - df['vwap_lower_band']) / (
            df['vwap_upper_band'] - df['vwap_lower_band'] + 1e-10
        )

        # Point of Control proxy (price level with highest volume)
        # Using rolling approximation
        df['poc_proxy'] = typical_price.rolling(20).apply(
            lambda x: x.iloc[x.values.argmax()] if len(x) > 0 else np.nan, raw=False
        )

        return df

    def _add_intraday_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add intraday session metrics if datetime index available."""
        if not isinstance(self.df.index, pd.DatetimeIndex):
            return df

        try:
            # Extract time components
            idx = self.df.index

            # Session volume metrics (cumulative within day)
            df['session_cumvol'] = self.df.groupby(idx.date)['volume'].cumsum()

            # Session high/low tracking
            df['session_high'] = self.df.groupby(idx.date)['high'].cummax()
            df['session_low'] = self.df.groupby(idx.date)['low'].cummin()
            df['session_range'] = df['session_high'] - df['session_low']

            # Close position within session range
            df['session_position'] = (self.df['close'] - df['session_low']) / (
                df['session_range'] + 1e-10
            )

            # Bars since session start
            df['bars_in_session'] = self.df.groupby(idx.date).cumcount() + 1

            # Volume rate (volume per bar in session)
            df['session_vol_rate'] = df['session_cumvol'] / df['bars_in_session']

            # Session VWAP
            typical = (self.df['high'] + self.df['low'] + self.df['close']) / 3
            df['session_vwap_num'] = (typical * self.df['volume']).groupby(idx.date).cumsum()
            df['session_vwap'] = df['session_vwap_num'] / (df['session_cumvol'] + 1e-10)
            df['session_vwap_dist'] = (self.df['close'] - df['session_vwap']) / df['session_vwap'] * 100

            # Clean up intermediate
            df = df.drop(['session_vwap_num'], axis=1)

        except Exception as e:
            logger.warning(f"Could not calculate intraday metrics: {e}")

        return df

    def _add_order_flow_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add order flow proxy features (from bar data)."""
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        open_price = self.df['open']
        volume = self.df['volume']

        # Buy/Sell volume estimation (using close position)
        close_position = (close - low) / (high - low + 1e-10)
        df['estimated_buy_volume'] = volume * close_position
        df['estimated_sell_volume'] = volume * (1 - close_position)

        # Order flow imbalance proxy
        df['ofi_proxy'] = df['estimated_buy_volume'] - df['estimated_sell_volume']
        df['ofi_proxy_ma10'] = df['ofi_proxy'].rolling(10).mean()
        df['ofi_proxy_ma20'] = df['ofi_proxy'].rolling(20).mean()

        # Cumulative OFI
        df['cum_ofi_proxy'] = df['ofi_proxy'].cumsum()

        # Trade pressure (body vs range)
        body = close - open_price
        range_size = high - low + 1e-10
        df['trade_pressure'] = body / range_size

        # Buying pressure strength
        df['buy_pressure'] = (close - low) / range_size
        df['sell_pressure'] = (high - close) / range_size

        # Volume-weighted trade pressure
        df['vw_trade_pressure'] = df['trade_pressure'] * volume
        df['vw_trade_pressure_ma10'] = df['vw_trade_pressure'].rolling(10).mean()

        # Tick rule proxy for trade direction
        # +1 if close > previous close, -1 if close < previous close
        df['tick_direction'] = np.sign(close.diff())
        df['tick_volume'] = df['tick_direction'] * volume
        df['cum_tick_volume'] = df['tick_volume'].cumsum()

        return df


def calculate_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to calculate all microstructure features.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with microstructure features
    """
    calculator = MicrostructureFeatures(df)
    return calculator.calculate_all()


if __name__ == "__main__":
    # Test with sample data
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from data_collection.ninjatrader_loader import load_sample_data
    except ImportError:
        from src.python.data_collection.ninjatrader_loader import load_sample_data

    print("=" * 60)
    print("Microstructure Features Test")
    print("=" * 60)

    es_data, _ = load_sample_data()

    # Calculate features
    features = calculate_microstructure_features(es_data)

    print(f"\nGenerated {len(features.columns)} microstructure features:")
    for i, col in enumerate(features.columns, 1):
        print(f"  {i:2d}. {col}")

    print(f"\nSample output (row 500):")
    print(features.iloc[500])

    print(f"\nFeature statistics:")
    print(features.describe().T[['mean', 'std', 'min', 'max']])
