"""
Technical Indicators (Category 2)
=================================
Implements ~100 technical indicator features for ML models.

Categories:
- Trend Indicators (SMA, EMA, ADX, Aroon)
- Momentum Indicators (RSI, MACD, Stochastic, CCI, ROC)
- Volatility Indicators (Bollinger Bands, ATR, Keltner)
- Volume Indicators (OBV, VWAP, CMF, MFI)

Reference: research/02_comprehensive_variables_research.md Section 14
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicator features from OHLCV data."""

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
        """Calculate all technical indicators."""
        logger.info("Calculating all technical indicators...")

        features = pd.DataFrame(index=self.df.index)

        # Trend indicators
        features = self._add_moving_averages(features)
        features = self._add_ma_crossovers(features)
        features = self._add_adx(features)
        features = self._add_aroon(features)

        # Momentum indicators
        features = self._add_rsi(features)
        features = self._add_macd(features)
        features = self._add_stochastic(features)
        features = self._add_cci(features)
        features = self._add_roc(features)
        features = self._add_williams_r(features)

        # Volatility indicators
        features = self._add_bollinger_bands(features)
        features = self._add_keltner_channels(features)
        features = self._add_donchian_channels(features)

        # Volume indicators
        features = self._add_obv(features)
        features = self._add_vwap(features)
        features = self._add_cmf(features)
        features = self._add_mfi(features)
        features = self._add_volume_features(features)

        logger.info(f"Generated {len(features.columns)} technical indicator features")
        return features

    # ==================== TREND INDICATORS ====================

    def _add_moving_averages(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages."""
        if periods is None:
            periods = [5, 10, 20, 50, 100, 200]

        close = self.df['close']

        for period in periods:
            # SMA
            df[f'sma_{period}'] = close.rolling(period).mean()

            # EMA
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()

            # Distance from MA (normalized)
            df[f'dist_sma_{period}'] = (close - df[f'sma_{period}']) / df[f'sma_{period}']
            df[f'dist_ema_{period}'] = (close - df[f'ema_{period}']) / df[f'ema_{period}']

            # MA slope (rate of change)
            df[f'sma_slope_{period}'] = df[f'sma_{period}'].pct_change(5)

        return df

    def _add_ma_crossovers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MA crossover signals."""
        close = self.df['close']

        # Golden/Death cross indicators
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()

        df['golden_cross'] = ((sma_50 > sma_200) & (sma_50.shift(1) <= sma_200.shift(1))).astype(int)
        df['death_cross'] = ((sma_50 < sma_200) & (sma_50.shift(1) >= sma_200.shift(1))).astype(int)
        df['sma_50_above_200'] = (sma_50 > sma_200).astype(int)

        # Short-term crossovers
        ema_10 = close.ewm(span=10, adjust=False).mean()
        ema_20 = close.ewm(span=20, adjust=False).mean()
        df['ema_10_above_20'] = (ema_10 > ema_20).astype(int)

        return df

    def _add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average Directional Index (ADX) and DI+/DI-."""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        idx = self.df.index

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=idx)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=idx)

        # Smoothed averages
        atr = tr.rolling(period).mean()
        plus_di = 100 * plus_dm.rolling(period).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.rolling(period).mean() / (atr + 1e-10)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()

        df[f'adx_{period}'] = adx
        df[f'plus_di_{period}'] = plus_di
        df[f'minus_di_{period}'] = minus_di
        df[f'di_diff_{period}'] = plus_di - minus_di

        # Also add ADX with period 20
        if period != 20:
            df = self._add_adx_single(df, 20)

        return df

    def _add_adx_single(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Add ADX for a single period."""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        idx = self.df.index

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=idx)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=idx)

        atr = tr.rolling(period).mean()
        plus_di = 100 * plus_dm.rolling(period).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.rolling(period).mean() / (atr + 1e-10)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()

        df[f'adx_{period}'] = adx

        return df

    def _add_aroon(self, df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
        """Add Aroon Up/Down indicators."""
        high = self.df['high']
        low = self.df['low']

        # Aroon Up: ((period - periods since highest high) / period) * 100
        def aroon_up(x):
            return ((period - (period - 1 - x.argmax())) / period) * 100

        def aroon_down(x):
            return ((period - (period - 1 - x.argmin())) / period) * 100

        df[f'aroon_up_{period}'] = high.rolling(period).apply(aroon_up, raw=True)
        df[f'aroon_down_{period}'] = low.rolling(period).apply(aroon_down, raw=True)
        df[f'aroon_osc_{period}'] = df[f'aroon_up_{period}'] - df[f'aroon_down_{period}']

        return df

    # ==================== MOMENTUM INDICATORS ====================

    def _add_rsi(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """Add Relative Strength Index (RSI)."""
        if periods is None:
            periods = [7, 14, 21]

        close = self.df['close']
        delta = close.diff()

        for period in periods:
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            df[f'rsi_{period}'] = rsi

            # RSI divergence from 50 (neutral)
            df[f'rsi_dist_50_{period}'] = rsi - 50

            # Overbought/Oversold
            df[f'rsi_overbought_{period}'] = (rsi > 70).astype(int)
            df[f'rsi_oversold_{period}'] = (rsi < 30).astype(int)

        return df

    def _add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicator."""
        close = self.df['close']

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram

        # Normalized MACD
        df['macd_norm'] = macd_line / close * 100

        # MACD crossovers
        df['macd_cross_up'] = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))).astype(int)
        df['macd_cross_down'] = ((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))).astype(int)

        return df

    def _add_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()

        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        stoch_d = stoch_k.rolling(d_period).mean()

        df[f'stoch_k_{k_period}'] = stoch_k
        df[f'stoch_d_{k_period}'] = stoch_d
        df[f'stoch_diff_{k_period}'] = stoch_k - stoch_d

        # Overbought/Oversold
        df[f'stoch_overbought_{k_period}'] = (stoch_k > 80).astype(int)
        df[f'stoch_oversold_{k_period}'] = (stoch_k < 20).astype(int)

        return df

    def _add_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Commodity Channel Index (CCI)."""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(period).mean()
        mean_dev = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())

        cci = (typical_price - sma_tp) / (0.015 * mean_dev + 1e-10)

        df[f'cci_{period}'] = cci

        # CCI extremes
        df[f'cci_overbought_{period}'] = (cci > 100).astype(int)
        df[f'cci_oversold_{period}'] = (cci < -100).astype(int)

        return df

    def _add_roc(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """Add Rate of Change (ROC)."""
        if periods is None:
            periods = [5, 10, 20]

        close = self.df['close']

        for period in periods:
            df[f'roc_{period}'] = ((close - close.shift(period)) / close.shift(period)) * 100

        return df

    def _add_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Williams %R."""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()

        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)

        df[f'williams_r_{period}'] = williams_r

        return df

    # ==================== VOLATILITY INDICATORS ====================

    def _add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands."""
        close = self.df['close']

        sma = close.rolling(period).mean()
        std = close.rolling(period).std()

        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)

        df[f'bb_upper_{period}'] = upper
        df[f'bb_middle_{period}'] = sma
        df[f'bb_lower_{period}'] = lower

        # %B (position within bands)
        df[f'bb_pct_b_{period}'] = (close - lower) / (upper - lower + 1e-10)

        # Band width (volatility measure)
        df[f'bb_width_{period}'] = (upper - lower) / sma * 100

        # Squeeze indicator (low volatility)
        avg_width = df[f'bb_width_{period}'].rolling(50).mean()
        df[f'bb_squeeze_{period}'] = (df[f'bb_width_{period}'] < avg_width * 0.75).astype(int)

        return df

    def _add_keltner_channels(self, df: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
        """Add Keltner Channels."""
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']

        # EMA as middle line
        ema = close.ewm(span=period, adjust=False).mean()

        # ATR
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        upper = ema + (atr * multiplier)
        lower = ema - (atr * multiplier)

        df[f'keltner_upper_{period}'] = upper
        df[f'keltner_middle_{period}'] = ema
        df[f'keltner_lower_{period}'] = lower

        # Position within channel
        df[f'keltner_pct_{period}'] = (close - lower) / (upper - lower + 1e-10)

        return df

    def _add_donchian_channels(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Donchian Channels."""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        upper = high.rolling(period).max()
        lower = low.rolling(period).min()
        middle = (upper + lower) / 2

        df[f'donchian_upper_{period}'] = upper
        df[f'donchian_middle_{period}'] = middle
        df[f'donchian_lower_{period}'] = lower

        # Position within channel
        df[f'donchian_pct_{period}'] = (close - lower) / (upper - lower + 1e-10)

        return df

    # ==================== VOLUME INDICATORS ====================

    def _add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume (OBV)."""
        close = self.df['close']
        volume = self.df['volume']

        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()

        df['obv'] = obv

        # OBV trend (normalized)
        df['obv_sma_20'] = obv.rolling(20).mean()
        df['obv_trend'] = (obv - df['obv_sma_20']) / (df['obv_sma_20'].abs() + 1e-10)

        return df

    def _add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Volume Weighted Average Price (VWAP)."""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        volume = self.df['volume']

        typical_price = (high + low + close) / 3

        # Cumulative VWAP (resets daily in production, here we use rolling)
        for period in [20, 50]:
            cumsum_tp_vol = (typical_price * volume).rolling(period).sum()
            cumsum_vol = volume.rolling(period).sum()
            vwap = cumsum_tp_vol / (cumsum_vol + 1e-10)

            df[f'vwap_{period}'] = vwap
            df[f'vwap_dist_{period}'] = (close - vwap) / vwap * 100

        return df

    def _add_cmf(self, df: pd.DataFrame, period: int = 21) -> pd.DataFrame:
        """Add Chaikin Money Flow (CMF)."""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        volume = self.df['volume']

        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)

        # Money Flow Volume
        mfv = mfm * volume

        # CMF
        cmf = mfv.rolling(period).sum() / (volume.rolling(period).sum() + 1e-10)

        df[f'cmf_{period}'] = cmf

        return df

    def _add_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Money Flow Index (MFI)."""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        volume = self.df['volume']

        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        # Positive/Negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()

        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))

        df[f'mfi_{period}'] = mfi

        # Overbought/Oversold
        df[f'mfi_overbought_{period}'] = (mfi > 80).astype(int)
        df[f'mfi_oversold_{period}'] = (mfi < 20).astype(int)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        volume = self.df['volume']

        # Relative volume
        for period in [10, 20, 50]:
            avg_vol = volume.rolling(period).mean()
            df[f'rel_volume_{period}'] = volume / (avg_vol + 1e-10)

        # Volume rate of change
        for period in [5, 10]:
            df[f'vol_roc_{period}'] = volume.pct_change(period)

        # Volume trend
        df['vol_sma_20'] = volume.rolling(20).mean()
        df['vol_trend'] = (volume - df['vol_sma_20']) / (df['vol_sma_20'] + 1e-10)

        return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to calculate all technical indicators.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with all technical indicator features
    """
    calculator = TechnicalIndicators(df)
    return calculator.calculate_all()


if __name__ == "__main__":
    # Test with sample data
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_collection.ninjatrader_loader import load_sample_data

    print("=" * 60)
    print("Technical Indicators Test")
    print("=" * 60)

    es_data, _ = load_sample_data()

    # Calculate features
    features = calculate_technical_indicators(es_data)

    print(f"\nGenerated {len(features.columns)} features:")
    print(f"Columns: {features.columns.tolist()[:20]}... (showing first 20)")

    print(f"\nSample output (last 5 rows, selected columns):")
    sample_cols = ['sma_20', 'ema_20', 'rsi_14', 'macd', 'adx_14', 'bb_pct_b_20']
    available_cols = [c for c in sample_cols if c in features.columns]
    print(features[available_cols].tail())

    print("\n" + "=" * 60)
    print("SUCCESS: Technical indicators calculated!")
    print("=" * 60)
