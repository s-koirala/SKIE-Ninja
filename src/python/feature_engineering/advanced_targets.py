"""
Advanced Target Labels & Support/Resistance Features
=====================================================
Implements advanced trading targets for ML training:

1. Pyramiding Targets: Identify opportunities to add to winning positions
2. DDCA (Dynamic Dollar-Cost Averaging): Targets for scaling in/out
3. Support/Resistance Detection: Local extremes for S/R zones
4. Pivot Point Features: Classic and dynamic pivots

Reference: research/02_comprehensive_variables_research.md
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class AdvancedTargets:
    """Calculate advanced target labels and S/R features."""

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
        """Calculate all advanced targets and S/R features."""
        logger.info("Calculating advanced targets and S/R features...")

        features = pd.DataFrame(index=self.df.index)

        # Pyramiding targets
        pyramid_features = self._calculate_pyramiding_targets()
        features = pd.concat([features, pyramid_features], axis=1)

        # DDCA targets
        ddca_features = self._calculate_ddca_targets()
        features = pd.concat([features, ddca_features], axis=1)

        # Support/Resistance detection
        sr_features = self._calculate_support_resistance()
        features = pd.concat([features, sr_features], axis=1)

        # Pivot point features
        pivot_features = self._calculate_pivot_features()
        features = pd.concat([features, pivot_features], axis=1)

        logger.info(f"Generated {len(features.columns)} advanced features")
        return features

    def _calculate_pyramiding_targets(self) -> pd.DataFrame:
        """
        Calculate pyramiding opportunity targets.

        Pyramiding: Adding to a winning position when trend continues.
        Targets identify bars where adding to position would be profitable.
        """
        df = pd.DataFrame(index=self.df.index)
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']

        # Forward returns at different horizons
        for horizon in [5, 10, 20]:
            future_max = high.rolling(horizon).max().shift(-horizon)
            future_min = low.rolling(horizon).min().shift(-horizon)

            # MFE/MAE from current close
            mfe = (future_max - close) / close
            mae = (close - future_min) / close

            # Pyramiding UP target: Good to add to long when:
            # - Price continues higher (MFE > threshold)
            # - Drawdown limited (MAE < threshold)
            df[f'pyramid_long_{horizon}'] = (
                (mfe > 0.005) & (mae < 0.003)  # 0.5% profit, <0.3% drawdown
            ).astype(int)

            # Pyramiding DOWN target: Good to add to short when:
            df[f'pyramid_short_{horizon}'] = (
                (mae > 0.005) & (mfe < 0.003)
            ).astype(int)

            # Risk-reward ratio for pyramiding
            df[f'pyramid_rr_{horizon}'] = mfe / (mae + 1e-10)

        # Trend continuation strength (for pyramiding decisions)
        for period in [10, 20]:
            # Trend strength: how consistently price moves in one direction
            returns = close.pct_change()
            pos_returns = (returns > 0).rolling(period).sum() / period
            df[f'trend_consistency_{period}'] = pos_returns

            # Momentum for pyramiding
            momentum = close.pct_change(period)
            df[f'pyramid_momentum_{period}'] = momentum

            # Trend acceleration (2nd derivative)
            df[f'trend_accel_{period}'] = momentum.diff()

        # Optimal pyramid entry zones (pullbacks in trend)
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()

        # Long pyramid zone: uptrend + pullback to support
        uptrend = (sma_20 > sma_50) & (close > sma_50)
        pullback_to_sma20 = (close <= sma_20 * 1.005) & (close >= sma_20 * 0.995)
        df['pyramid_long_zone'] = (uptrend & pullback_to_sma20).astype(int)

        # Short pyramid zone: downtrend + rally to resistance
        downtrend = (sma_20 < sma_50) & (close < sma_50)
        rally_to_sma20 = (close >= sma_20 * 0.995) & (close <= sma_20 * 1.005)
        df['pyramid_short_zone'] = (downtrend & rally_to_sma20).astype(int)

        return df

    def _calculate_ddca_targets(self) -> pd.DataFrame:
        """
        Calculate DDCA (Dynamic Dollar-Cost Averaging) targets.

        DDCA UP: Buy more as price drops (average down)
        DDCA DOWN: Sell more as price rises (scale out)

        Targets identify optimal scaling points.
        """
        df = pd.DataFrame(index=self.df.index)
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']

        # Price levels relative to recent range
        for period in [20, 50]:
            period_high = high.rolling(period).max()
            period_low = low.rolling(period).min()
            period_range = period_high - period_low

            # Position in range (0 = at low, 1 = at high)
            df[f'range_position_{period}'] = (close - period_low) / (period_range + 1e-10)

            # DDCA Buy zones (lower part of range)
            df[f'ddca_buy_zone_{period}'] = (df[f'range_position_{period}'] < 0.3).astype(int)
            df[f'ddca_strong_buy_{period}'] = (df[f'range_position_{period}'] < 0.15).astype(int)

            # DDCA Sell zones (upper part of range)
            df[f'ddca_sell_zone_{period}'] = (df[f'range_position_{period}'] > 0.7).astype(int)
            df[f'ddca_strong_sell_{period}'] = (df[f'range_position_{period}'] > 0.85).astype(int)

        # ATR-based DDCA levels
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()

        # Distance from recent pivot in ATR multiples
        recent_high = high.rolling(20).max()
        recent_low = low.rolling(20).min()

        df['atr_from_high'] = (recent_high - close) / (atr_14 + 1e-10)
        df['atr_from_low'] = (close - recent_low) / (atr_14 + 1e-10)

        # DDCA scaling levels (1 ATR, 2 ATR, 3 ATR from entry)
        for mult in [1, 2, 3]:
            df[f'ddca_buy_level_{mult}atr'] = (df['atr_from_high'] >= mult).astype(int)
            df[f'ddca_sell_level_{mult}atr'] = (df['atr_from_low'] >= mult).astype(int)

        # Forward-looking DDCA success (did averaging work?)
        for horizon in [10, 20]:
            future_close = close.shift(-horizon)

            # If we averaged down, did price recover?
            df[f'ddca_buy_success_{horizon}'] = (
                (df['ddca_buy_zone_20'] == 1) & (future_close > close)
            ).astype(int)

            # If we scaled out, did price drop?
            df[f'ddca_sell_success_{horizon}'] = (
                (df['ddca_sell_zone_20'] == 1) & (future_close < close)
            ).astype(int)

        # Mean reversion probability (for DDCA effectiveness)
        returns = close.pct_change()
        for period in [5, 10, 20]:
            cumret = returns.rolling(period).sum()
            # Extreme moves tend to mean-revert
            df[f'mean_revert_signal_{period}'] = -cumret  # Negative = oversold, Positive = overbought

        return df

    def _calculate_support_resistance(self) -> pd.DataFrame:
        """
        Calculate support/resistance levels based on local extremes.

        A bar is a pivot high/low if it's the max/min among:
        - X bars lookback AND Y bars lookforward

        The wick/body of pivot bars form S/R zones.
        """
        df = pd.DataFrame(index=self.df.index)
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        open_price = self.df['open']

        # Detect pivot highs and lows with different lookback/forward
        for lookback in [5, 10, 20]:
            for lookforward in [5, 10]:
                # Pivot High: highest high in window
                past_max = high.rolling(lookback).max()
                future_max = high.rolling(lookforward).max().shift(-lookforward)

                is_pivot_high = (high >= past_max) & (high >= future_max)
                df[f'pivot_high_{lookback}_{lookforward}'] = is_pivot_high.astype(int)

                # Pivot Low: lowest low in window
                past_min = low.rolling(lookback).min()
                future_min = low.rolling(lookforward).min().shift(-lookforward)

                is_pivot_low = (low <= past_min) & (low <= future_min)
                df[f'pivot_low_{lookback}_{lookforward}'] = is_pivot_low.astype(int)

        # S/R zone features (distance to recent pivots)
        # Find most recent pivot high/low levels
        pivot_high_mask = df['pivot_high_10_5'] == 1
        pivot_low_mask = df['pivot_low_10_5'] == 1

        # Forward-fill pivot levels
        df['last_pivot_high'] = high.where(pivot_high_mask).ffill()
        df['last_pivot_low'] = low.where(pivot_low_mask).ffill()

        # Distance to S/R levels
        df['dist_to_resistance'] = (df['last_pivot_high'] - close) / close * 100
        df['dist_to_support'] = (close - df['last_pivot_low']) / close * 100

        # Proximity to S/R (within 0.2%)
        df['near_resistance'] = (abs(df['dist_to_resistance']) < 0.2).astype(int)
        df['near_support'] = (abs(df['dist_to_support']) < 0.2).astype(int)

        # S/R zone strength (number of touches)
        for window in [50, 100]:
            # Count how many times price touched resistance zone
            resistance_zone = (abs(high - df['last_pivot_high']) / df['last_pivot_high'] < 0.002)
            df[f'resistance_touches_{window}'] = resistance_zone.rolling(window).sum()

            # Count support touches
            support_zone = (abs(low - df['last_pivot_low']) / df['last_pivot_low'] < 0.002)
            df[f'support_touches_{window}'] = support_zone.rolling(window).sum()

        # Wick-based S/R (wicks often mark reversal zones)
        upper_wick = high - pd.concat([open_price, close], axis=1).max(axis=1)
        lower_wick = pd.concat([open_price, close], axis=1).min(axis=1) - low
        body = abs(close - open_price)

        # Large wick ratio indicates rejection
        df['upper_wick_ratio'] = upper_wick / (body + upper_wick + lower_wick + 1e-10)
        df['lower_wick_ratio'] = lower_wick / (body + upper_wick + lower_wick + 1e-10)

        # Rejection candles at pivots
        df['rejection_at_high'] = (
            (df['pivot_high_10_5'] == 1) & (df['upper_wick_ratio'] > 0.5)
        ).astype(int)
        df['rejection_at_low'] = (
            (df['pivot_low_10_5'] == 1) & (df['lower_wick_ratio'] > 0.5)
        ).astype(int)

        # Breakout detection
        for period in [20, 50]:
            period_high = high.rolling(period).max().shift(1)
            period_low = low.rolling(period).min().shift(1)

            df[f'breakout_high_{period}'] = (close > period_high).astype(int)
            df[f'breakout_low_{period}'] = (close < period_low).astype(int)

            # Failed breakout (reversal after breakout)
            df[f'failed_breakout_high_{period}'] = (
                (close.shift(1) > period_high.shift(1)) & (close < period_high)
            ).astype(int)
            df[f'failed_breakout_low_{period}'] = (
                (close.shift(1) < period_low.shift(1)) & (close > period_low)
            ).astype(int)

        return df

    def _calculate_pivot_features(self) -> pd.DataFrame:
        """Calculate classic and dynamic pivot point features."""
        df = pd.DataFrame(index=self.df.index)
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        # Classic Pivot Points (daily)
        # For intraday data, we'd normally use previous day's HLC
        # Here we use rolling windows as proxy

        for period in [20, 50]:  # Rolling "session" approximation
            period_high = high.rolling(period).max()
            period_low = low.rolling(period).min()
            period_close = close.shift(1)  # Previous bar close

            # Classic Pivot
            pivot = (period_high + period_low + period_close) / 3
            df[f'pivot_{period}'] = pivot

            # Support levels
            df[f'S1_{period}'] = 2 * pivot - period_high
            df[f'S2_{period}'] = pivot - (period_high - period_low)
            df[f'S3_{period}'] = period_low - 2 * (period_high - pivot)

            # Resistance levels
            df[f'R1_{period}'] = 2 * pivot - period_low
            df[f'R2_{period}'] = pivot + (period_high - period_low)
            df[f'R3_{period}'] = period_high + 2 * (pivot - period_low)

            # Distance to pivot levels
            df[f'dist_to_pivot_{period}'] = (close - pivot) / pivot * 100
            df[f'dist_to_R1_{period}'] = (df[f'R1_{period}'] - close) / close * 100
            df[f'dist_to_S1_{period}'] = (close - df[f'S1_{period}']) / close * 100

            # Position relative to pivots
            df[f'above_pivot_{period}'] = (close > pivot).astype(int)
            df[f'above_R1_{period}'] = (close > df[f'R1_{period}']).astype(int)
            df[f'below_S1_{period}'] = (close < df[f'S1_{period}']).astype(int)

        # Fibonacci Pivot Points
        period = 20
        period_high = high.rolling(period).max()
        period_low = low.rolling(period).min()
        period_range = period_high - period_low

        pivot = (period_high + period_low + close.shift(1)) / 3
        df['fib_pivot'] = pivot
        df['fib_S1'] = pivot - 0.382 * period_range
        df['fib_S2'] = pivot - 0.618 * period_range
        df['fib_S3'] = pivot - 1.0 * period_range
        df['fib_R1'] = pivot + 0.382 * period_range
        df['fib_R2'] = pivot + 0.618 * period_range
        df['fib_R3'] = pivot + 1.0 * period_range

        # Camarilla Pivots (good for intraday)
        df['cam_S1'] = close.shift(1) - period_range * 1.1 / 12
        df['cam_S2'] = close.shift(1) - period_range * 1.1 / 6
        df['cam_S3'] = close.shift(1) - period_range * 1.1 / 4
        df['cam_R1'] = close.shift(1) + period_range * 1.1 / 12
        df['cam_R2'] = close.shift(1) + period_range * 1.1 / 6
        df['cam_R3'] = close.shift(1) + period_range * 1.1 / 4

        return df


def calculate_advanced_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to calculate all advanced targets.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with advanced target features
    """
    calculator = AdvancedTargets(df)
    return calculator.calculate_all()


if __name__ == "__main__":
    # Test with sample data
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from data_collection.ninjatrader_loader import load_best_available_data
    except ImportError:
        from src.python.data_collection.ninjatrader_loader import load_best_available_data

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Advanced Targets & S/R Features Test")
    print("=" * 60)

    es_data = load_best_available_data("ES")

    # Calculate features
    features = calculate_advanced_targets(es_data)

    print(f"\nGenerated {len(features.columns)} features:")

    # Group by category
    pyramid_cols = [c for c in features.columns if 'pyramid' in c]
    ddca_cols = [c for c in features.columns if 'ddca' in c or 'range_position' in c or 'atr_from' in c]
    sr_cols = [c for c in features.columns if 'pivot' in c or 'support' in c or 'resistance' in c or 'breakout' in c]
    pivot_cols = [c for c in features.columns if c.startswith(('S1', 'S2', 'S3', 'R1', 'R2', 'R3', 'fib_', 'cam_', 'dist_to', 'above', 'below'))]

    print(f"\nPyramiding features ({len(pyramid_cols)}):")
    for col in pyramid_cols[:10]:
        print(f"  - {col}")

    print(f"\nDDCA features ({len(ddca_cols)}):")
    for col in ddca_cols[:10]:
        print(f"  - {col}")

    print(f"\nS/R features ({len(sr_cols)}):")
    for col in sr_cols[:10]:
        print(f"  - {col}")

    print(f"\nPivot features ({len(pivot_cols)}):")
    for col in pivot_cols[:10]:
        print(f"  - {col}")

    print(f"\nSample data (row 1000):")
    print(features.iloc[1000][['pyramid_long_10', 'ddca_buy_zone_20', 'pivot_high_10_5', 'near_resistance']])
