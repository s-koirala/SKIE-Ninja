"""
Advanced Target Labels & Support/Resistance Features
=====================================================
Implements advanced trading targets for ML training:

1. Pyramiding Targets: Identify opportunities to add to winning positions
2. DDCA (Dynamic Dollar-Cost Averaging): Targets for scaling in/out
3. Support/Resistance Detection: Local extremes for S/R zones
4. Pivot Point Features: Classic and dynamic pivots

Reference: research/02_comprehensive_variables_research.md

IMPORTANT - LOOK-AHEAD BIAS FIX (2025-12-04):
---------------------------------------------
This module was refactored to remove all look-ahead bias:

- Pyramiding: Changed from shift(-horizon) to shift(1), using historical
  MFE/MAE patterns. Contract sizing: 1, 2, 3 (max 5) instead of 5, 10, 20.

- DDCA: Removed ddca_*_success features that used future prices.
  Replaced with ddca_*_pattern and ddca_*_effectiveness using past data.

- Pivot Detection: Changed from forward-looking confirmation to confirmed
  pivots detected with a delay (detect pivots N bars after they occur).

All features now use only past data (shift(N) where N >= 1) and no
future data (no shift(-N) operations).
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
        Calculate pyramiding opportunity features using PAST DATA ONLY.

        Pyramiding: Adding to a winning position when trend continues.
        Features identify favorable conditions for pyramiding based on:
        - Historical volatility and MFE/MAE patterns
        - Trend strength and momentum
        - ATR-based risk assessment

        NOTE: Fixed to remove look-ahead bias (no shift(-N) operations).
        Contract scaling: 1, 2, 3 contracts (max 5) instead of 5, 10, 20.
        """
        df = pd.DataFrame(index=self.df.index)
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']

        # Calculate ATR for risk assessment (all past data)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        atr_20 = tr.rolling(20).mean()

        # Historical MFE/MAE ratios (past N bars, NOT future)
        # This measures how price moved AFTER similar conditions historically
        for lookback in [5, 10, 20]:
            # Past bar range extremes (what happened in previous N bars)
            past_max = high.rolling(lookback).max().shift(1)  # Exclude current bar
            past_min = low.rolling(lookback).min().shift(1)   # Exclude current bar

            # Historical MFE/MAE from entry at past_close
            past_close = close.shift(1)
            hist_mfe = (past_max - past_close) / (past_close + 1e-10)
            hist_mae = (past_close - past_min) / (past_close + 1e-10)

            # Risk-reward based on HISTORICAL price action (not future)
            df[f'pyramid_rr_{lookback}'] = hist_mfe / (hist_mae + 1e-10)

            # Favorable pyramid conditions based on past patterns
            # Long: Good historical upside, limited downside
            df[f'pyramid_long_{lookback}'] = (
                (hist_mfe > 0.005) & (hist_mae < 0.003)
            ).astype(int)

            # Short: Good historical downside, limited upside
            df[f'pyramid_short_{lookback}'] = (
                (hist_mae > 0.005) & (hist_mfe < 0.003)
            ).astype(int)

        # ATR-based pyramid sizing signals (contracts 1, 2, 3)
        # Favorable conditions for adding contracts based on volatility
        returns = close.pct_change()
        vol_20 = returns.rolling(20).std()

        # Low volatility = safer to pyramid (add up to 3 contracts)
        df['pyramid_vol_signal'] = (vol_20 < vol_20.rolling(50).mean()).astype(int)

        # Pyramid scale levels based on ATR distance from entry
        # These indicate when to add 1, 2, or 3 contracts
        for contracts in [1, 2, 3]:
            atr_mult = contracts * 0.5  # 0.5, 1.0, 1.5 ATR thresholds
            # Favorable to add long when price pulled back N ATR
            df[f'pyramid_add_{contracts}_long'] = (
                (close < close.rolling(10).mean() - atr_14 * atr_mult)
            ).astype(int)
            # Favorable to add short when price rallied N ATR
            df[f'pyramid_add_{contracts}_short'] = (
                (close > close.rolling(10).mean() + atr_14 * atr_mult)
            ).astype(int)

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

        # DDCA historical success patterns (using PAST data only)
        # Instead of looking forward, we measure how similar past DDCA setups performed
        for lookback in [10, 20]:
            # Rolling mean reversion success: when in buy zone, did price subsequently rise?
            # We use PAST pattern: shifted forward by lookback to see what happened AFTER buy zone
            past_buy_zone = df['ddca_buy_zone_20'].shift(lookback)
            past_close = close.shift(lookback)

            # Historical success rate: was price higher after buy zone entry?
            df[f'ddca_buy_pattern_{lookback}'] = (
                (past_buy_zone == 1) & (close.shift(1) > past_close)
            ).astype(int)

            # Same for sell zone
            past_sell_zone = df['ddca_sell_zone_20'].shift(lookback)
            df[f'ddca_sell_pattern_{lookback}'] = (
                (past_sell_zone == 1) & (close.shift(1) < past_close)
            ).astype(int)

            # Rolling DDCA effectiveness score (past N bars)
            df[f'ddca_buy_effectiveness_{lookback}'] = df[f'ddca_buy_pattern_{lookback}'].rolling(50).mean()
            df[f'ddca_sell_effectiveness_{lookback}'] = df[f'ddca_sell_pattern_{lookback}'].rolling(50).mean()

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

        FIXED: Using PAST DATA ONLY for pivot detection.
        A bar is a confirmed pivot high/low if it was the max/min among
        the surrounding N bars - we detect this AFTER confirmation
        (i.e., with a delay equal to the confirmation window).

        The wick/body of pivot bars form S/R zones.
        """
        df = pd.DataFrame(index=self.df.index)
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        open_price = self.df['open']

        # Detect CONFIRMED pivot highs and lows (using past data only)
        # A pivot is confirmed N bars AFTER it occurs
        for lookback in [5, 10, 20]:
            for confirm_bars in [5, 10]:
                # Total window = lookback + confirm_bars
                total_window = lookback + confirm_bars

                # Find the max high in the total window, shifted to exclude current bar
                window_max = high.rolling(total_window).max().shift(1)
                window_min = low.rolling(total_window).min().shift(1)

                # The pivot occurred at the bar that was the max/min
                # We detect it `confirm_bars` later (when confirmed)
                # Check if the bar `confirm_bars` ago was the peak
                bar_high_confirm_ago = high.shift(confirm_bars)
                bar_low_confirm_ago = low.shift(confirm_bars)

                # Confirmed pivot high: the bar `confirm_bars` ago was the highest
                # in a window of `lookback` before it and `confirm_bars` after
                is_confirmed_pivot_high = (bar_high_confirm_ago >= window_max)
                df[f'pivot_high_{lookback}_{confirm_bars}'] = is_confirmed_pivot_high.astype(int)

                # Confirmed pivot low: the bar `confirm_bars` ago was the lowest
                is_confirmed_pivot_low = (bar_low_confirm_ago <= window_min)
                df[f'pivot_low_{lookback}_{confirm_bars}'] = is_confirmed_pivot_low.astype(int)

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
