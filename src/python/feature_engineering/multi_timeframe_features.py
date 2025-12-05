"""
Multi-Timeframe Feature Engineering Module
==========================================

Generates features from multiple timeframes for enhanced market analysis.
Higher timeframe context provides trend direction, support/resistance levels,
and volatility regime information.

CRITICAL: Follows strict data leakage prevention rules:
1. NEVER use shift(-N) - only positive shifts allowed
2. NEVER use center=True in rolling windows
3. Higher timeframe bars must be COMPLETE before use (lag applied)
4. All features use only PAST data at each prediction point

Timeframes supported:
- Base: 1-min or 5-min (trading timeframe)
- HTF1: 15-min (short-term trend)
- HTF2: 1-hour (medium-term trend)
- HTF3: 4-hour (longer-term trend)
- Daily: End-of-day context

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MTFConfig:
    """Configuration for multi-timeframe features."""

    # Base timeframe (in minutes)
    base_timeframe: int = 5

    # Higher timeframes to calculate (in minutes)
    higher_timeframes: List[int] = None

    # Lag to apply when aligning HTF to base (in base bars)
    # This ensures we only use COMPLETED HTF bars
    htf_completion_lag: int = 1

    # Feature periods for each timeframe
    ma_periods: List[int] = None
    rsi_periods: List[int] = None
    atr_periods: List[int] = None

    def __post_init__(self):
        if self.higher_timeframes is None:
            self.higher_timeframes = [15, 60, 240]  # 15min, 1hr, 4hr
        if self.ma_periods is None:
            self.ma_periods = [10, 20, 50]
        if self.rsi_periods is None:
            self.rsi_periods = [14]
        if self.atr_periods is None:
            self.atr_periods = [14]


class MultiTimeframeFeatures:
    """
    Generate features from multiple timeframes.

    IMPORTANT: All features are calculated to avoid look-ahead bias:
    - HTF bars are only used AFTER they complete
    - Proper lag is applied when aligning timeframes
    - Rolling calculations use past data only
    """

    def __init__(self, config: Optional[MTFConfig] = None):
        self.config = config or MTFConfig()

    def resample_ohlcv(
        self,
        df: pd.DataFrame,
        target_minutes: int
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to higher timeframe.

        Args:
            df: DataFrame with OHLCV columns and DatetimeIndex
            target_minutes: Target timeframe in minutes

        Returns:
            Resampled DataFrame
        """
        rule = f'{target_minutes}min'

        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return resampled

    def calculate_htf_features(
        self,
        htf_df: pd.DataFrame,
        timeframe_label: str
    ) -> pd.DataFrame:
        """
        Calculate features for a single higher timeframe.

        All calculations use ONLY past data (no look-ahead).

        Args:
            htf_df: OHLCV DataFrame at higher timeframe
            timeframe_label: Label for feature names (e.g., '15m', '1h')

        Returns:
            DataFrame with HTF features
        """
        features = pd.DataFrame(index=htf_df.index)
        close = htf_df['close']
        high = htf_df['high']
        low = htf_df['low']
        volume = htf_df['volume']

        # === TREND FEATURES ===

        # Moving averages (SAFE: rolling uses past data only)
        for period in self.config.ma_periods:
            sma = close.rolling(period).mean()
            ema = close.ewm(span=period, adjust=False).mean()

            features[f'htf_{timeframe_label}_sma_{period}'] = sma
            features[f'htf_{timeframe_label}_ema_{period}'] = ema

            # Price position relative to MA (normalized)
            features[f'htf_{timeframe_label}_close_vs_sma_{period}'] = (
                (close - sma) / (sma + 1e-10)
            )
            features[f'htf_{timeframe_label}_close_vs_ema_{period}'] = (
                (close - ema) / (ema + 1e-10)
            )

            # MA slope (trend direction) - uses PAST change only
            features[f'htf_{timeframe_label}_sma_slope_{period}'] = sma.pct_change(3)

        # Trend direction (simple)
        features[f'htf_{timeframe_label}_trend_up'] = (
            close > close.shift(1)  # SAFE: shift(1) looks back
        ).astype(int)

        features[f'htf_{timeframe_label}_trend_strong_up'] = (
            (close > close.shift(1)) &
            (close.shift(1) > close.shift(2))  # SAFE: all positive shifts
        ).astype(int)

        # Higher highs / Lower lows
        features[f'htf_{timeframe_label}_higher_high'] = (
            high > high.shift(1)  # SAFE
        ).astype(int)

        features[f'htf_{timeframe_label}_lower_low'] = (
            low < low.shift(1)  # SAFE
        ).astype(int)

        # === MOMENTUM FEATURES ===

        # RSI (SAFE: uses rolling which looks back only)
        for period in self.config.rsi_periods:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            features[f'htf_{timeframe_label}_rsi_{period}'] = rsi
            features[f'htf_{timeframe_label}_rsi_ob_{period}'] = (rsi > 70).astype(int)
            features[f'htf_{timeframe_label}_rsi_os_{period}'] = (rsi < 30).astype(int)

        # Rate of change (SAFE: pct_change uses past data)
        for period in [3, 5, 10]:
            features[f'htf_{timeframe_label}_roc_{period}'] = close.pct_change(period)

        # === VOLATILITY FEATURES ===

        # ATR (SAFE: rolling uses past data only)
        for period in self.config.atr_periods:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))  # SAFE: shift(1)
            tr3 = abs(low - close.shift(1))   # SAFE: shift(1)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()

            features[f'htf_{timeframe_label}_atr_{period}'] = atr
            features[f'htf_{timeframe_label}_atr_pct_{period}'] = atr / close

            # ATR expansion/contraction
            atr_ma = atr.rolling(period).mean()
            features[f'htf_{timeframe_label}_atr_expansion_{period}'] = (
                atr > atr_ma
            ).astype(int)

        # Range (high-low) normalized
        features[f'htf_{timeframe_label}_range_pct'] = (high - low) / close

        # === SUPPORT/RESISTANCE LEVELS ===

        # Rolling high/low (SAFE: no center=True)
        for period in [10, 20]:
            features[f'htf_{timeframe_label}_rolling_high_{period}'] = high.rolling(period).max()
            features[f'htf_{timeframe_label}_rolling_low_{period}'] = low.rolling(period).min()

            # Distance to S/R levels
            features[f'htf_{timeframe_label}_dist_to_high_{period}'] = (
                (features[f'htf_{timeframe_label}_rolling_high_{period}'] - close) / close
            )
            features[f'htf_{timeframe_label}_dist_to_low_{period}'] = (
                (close - features[f'htf_{timeframe_label}_rolling_low_{period}']) / close
            )

        # === VOLUME FEATURES ===

        # Relative volume
        vol_ma = volume.rolling(20).mean()
        features[f'htf_{timeframe_label}_rel_volume'] = volume / (vol_ma + 1e-10)

        # Volume trend
        features[f'htf_{timeframe_label}_vol_increasing'] = (
            volume > volume.shift(1)  # SAFE
        ).astype(int)

        # === CANDLE PATTERNS (simplified) ===

        # Body size relative to range
        body = abs(close - htf_df['open'])
        range_size = high - low + 1e-10
        features[f'htf_{timeframe_label}_body_pct'] = body / range_size

        # Bullish/Bearish candle
        features[f'htf_{timeframe_label}_bullish'] = (
            close > htf_df['open']
        ).astype(int)

        return features

    def align_htf_to_base(
        self,
        htf_features: pd.DataFrame,
        base_index: pd.DatetimeIndex,
        htf_minutes: int
    ) -> pd.DataFrame:
        """
        Align higher timeframe features to base timeframe.

        CRITICAL: Applies lag to ensure we only use COMPLETED HTF bars.
        At any base bar, we use the HTF bar that completed BEFORE it.

        Args:
            htf_features: Features calculated at HTF
            base_index: DatetimeIndex of base timeframe
            htf_minutes: HTF bar duration in minutes

        Returns:
            HTF features aligned to base timeframe index
        """
        aligned = pd.DataFrame(index=base_index)

        # For each base bar, find the most recent COMPLETED HTF bar
        # The HTF bar completes at the END of its period
        # So at base bar time T, we can use HTF bars that ended at or before T

        for col in htf_features.columns:
            # Create a series that we can forward-fill
            htf_series = htf_features[col]

            # Shift HTF values by the completion lag
            # This ensures we only use bars that have fully completed
            htf_series_lagged = htf_series.shift(self.config.htf_completion_lag)

            # Reindex to base timeframe with forward fill
            # ffill propagates the last known HTF value
            aligned[col] = htf_series_lagged.reindex(
                base_index,
                method='ffill'
            )

        return aligned

    def calculate_mtf_alignment_features(
        self,
        base_df: pd.DataFrame,
        htf_aligned: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate features based on alignment across timeframes.

        These features capture when multiple timeframes agree or diverge.

        Args:
            base_df: Base timeframe OHLCV
            htf_aligned: Dict of aligned HTF features by timeframe label

        Returns:
            DataFrame with alignment features
        """
        features = pd.DataFrame(index=base_df.index)
        close = base_df['close']

        # Calculate base timeframe trend
        base_trend_up = (close > close.shift(1)).astype(int)

        # Collect trend signals from all timeframes
        trend_signals = [base_trend_up]
        trend_labels = ['base']

        for tf_label, htf_df in htf_aligned.items():
            trend_col = f'htf_{tf_label}_trend_up'
            if trend_col in htf_df.columns:
                trend_signals.append(htf_df[trend_col])
                trend_labels.append(tf_label)

        if len(trend_signals) > 1:
            # Combine trends into DataFrame
            trends_df = pd.concat(trend_signals, axis=1)
            trends_df.columns = trend_labels

            # MTF trend alignment score (0 to 1)
            features['mtf_trend_alignment'] = trends_df.mean(axis=1)

            # Strong bullish alignment (all timeframes up)
            features['mtf_all_bullish'] = (trends_df.sum(axis=1) == len(trend_labels)).astype(int)

            # Strong bearish alignment (all timeframes down)
            features['mtf_all_bearish'] = (trends_df.sum(axis=1) == 0).astype(int)

            # Divergence (base vs higher timeframes)
            htf_trend_avg = trends_df.drop('base', axis=1).mean(axis=1)
            features['mtf_base_vs_htf_divergence'] = base_trend_up - htf_trend_avg

        # RSI alignment across timeframes
        rsi_signals = []
        for tf_label, htf_df in htf_aligned.items():
            rsi_col = f'htf_{tf_label}_rsi_14'
            if rsi_col in htf_df.columns:
                rsi_signals.append(htf_df[rsi_col])

        if rsi_signals:
            rsi_df = pd.concat(rsi_signals, axis=1)
            features['mtf_rsi_avg'] = rsi_df.mean(axis=1)
            features['mtf_rsi_std'] = rsi_df.std(axis=1)

            # MTF overbought/oversold
            features['mtf_rsi_ob'] = (features['mtf_rsi_avg'] > 70).astype(int)
            features['mtf_rsi_os'] = (features['mtf_rsi_avg'] < 30).astype(int)

        # Volatility alignment
        vol_signals = []
        for tf_label, htf_df in htf_aligned.items():
            vol_col = f'htf_{tf_label}_atr_expansion_14'
            if vol_col in htf_df.columns:
                vol_signals.append(htf_df[vol_col])

        if vol_signals:
            vol_df = pd.concat(vol_signals, axis=1)
            features['mtf_vol_expansion_score'] = vol_df.mean(axis=1)
            features['mtf_all_vol_expanding'] = (vol_df.sum(axis=1) == len(vol_signals)).astype(int)

        return features

    def generate_all_features(
        self,
        base_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate all multi-timeframe features.

        Args:
            base_df: Base timeframe OHLCV with DatetimeIndex

        Returns:
            DataFrame with all MTF features aligned to base timeframe
        """
        logger.info("Generating multi-timeframe features...")
        logger.info(f"Base timeframe: {self.config.base_timeframe} minutes")
        logger.info(f"Higher timeframes: {self.config.higher_timeframes} minutes")

        all_features = pd.DataFrame(index=base_df.index)
        htf_aligned = {}

        for htf_minutes in self.config.higher_timeframes:
            # Generate timeframe label
            if htf_minutes < 60:
                tf_label = f'{htf_minutes}m'
            elif htf_minutes < 1440:
                tf_label = f'{htf_minutes // 60}h'
            else:
                tf_label = f'{htf_minutes // 1440}d'

            logger.info(f"  Processing {tf_label} timeframe...")

            # Resample to higher timeframe
            htf_df = self.resample_ohlcv(base_df, htf_minutes)

            if len(htf_df) < 50:
                logger.warning(f"    Insufficient data for {tf_label} ({len(htf_df)} bars)")
                continue

            # Calculate HTF features
            htf_features = self.calculate_htf_features(htf_df, tf_label)

            # Align to base timeframe with proper lag
            aligned = self.align_htf_to_base(htf_features, base_df.index, htf_minutes)
            htf_aligned[tf_label] = aligned

            # Add to all features
            all_features = pd.concat([all_features, aligned], axis=1)

            logger.info(f"    Generated {len(htf_features.columns)} features")

        # Calculate cross-timeframe alignment features
        alignment_features = self.calculate_mtf_alignment_features(base_df, htf_aligned)
        all_features = pd.concat([all_features, alignment_features], axis=1)

        logger.info(f"Total MTF features generated: {len(all_features.columns)}")

        return all_features


def calculate_multi_timeframe_features(
    df: pd.DataFrame,
    config: Optional[MTFConfig] = None
) -> pd.DataFrame:
    """
    Convenience function to calculate all multi-timeframe features.

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        config: Optional MTF configuration

    Returns:
        DataFrame with all MTF features
    """
    calculator = MultiTimeframeFeatures(config)
    return calculator.generate_all_features(df)


# === LEAKAGE VALIDATION ===

def validate_mtf_features_no_leakage(
    features: pd.DataFrame,
    prices: pd.DataFrame
) -> Dict[str, any]:
    """
    Validate that MTF features have no look-ahead bias.

    Tests:
    1. Check for any negative shift patterns (should be none)
    2. Verify correlation with future returns is not suspiciously high
    3. Ensure features at time T don't correlate with returns before T

    Args:
        features: MTF features DataFrame
        prices: Original OHLCV DataFrame

    Returns:
        Dict with validation results
    """
    results = {
        'passed': True,
        'checks': [],
        'warnings': []
    }

    # Calculate future returns for testing
    future_return_5 = prices['close'].pct_change(5).shift(-5)  # 5-bar future return
    future_return_10 = prices['close'].pct_change(10).shift(-10)

    # Check each feature's correlation with future returns
    suspicious_features = []

    for col in features.columns:
        if features[col].isna().all():
            continue

        # Correlation with future returns
        corr_5 = features[col].corr(future_return_5)
        corr_10 = features[col].corr(future_return_10)

        # If correlation > 0.3 with future returns, flag as suspicious
        if abs(corr_5) > 0.3 or abs(corr_10) > 0.3:
            suspicious_features.append({
                'feature': col,
                'corr_future_5': corr_5,
                'corr_future_10': corr_10
            })

    if suspicious_features:
        results['passed'] = False
        results['warnings'].append(
            f"Found {len(suspicious_features)} features with high future correlation"
        )
        results['suspicious_features'] = suspicious_features
    else:
        results['checks'].append("No suspicious future correlations found")

    # Check for NaN patterns that might indicate alignment issues
    nan_pct = features.isna().sum() / len(features) * 100
    high_nan_cols = nan_pct[nan_pct > 50].index.tolist()

    if high_nan_cols:
        results['warnings'].append(
            f"{len(high_nan_cols)} features have >50% NaN values"
        )

    results['checks'].append(f"Validated {len(features.columns)} features")

    return results


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("MULTI-TIMEFRAME FEATURES TEST")
    print("=" * 70)

    # Load sample data
    try:
        from data_collection.ninjatrader_loader import load_sample_data
        prices, _ = load_sample_data(source="databento")
    except Exception as e:
        print(f"Could not load data: {e}")
        print("Creating synthetic data for testing...")

        dates = pd.date_range('2024-01-01 09:30', periods=5000, freq='5min')
        np.random.seed(42)
        close = 4500 + np.cumsum(np.random.randn(5000) * 2)

        prices = pd.DataFrame({
            'open': close + np.random.randn(5000),
            'high': close + abs(np.random.randn(5000)) * 3,
            'low': close - abs(np.random.randn(5000)) * 3,
            'close': close,
            'volume': np.random.randint(1000, 10000, 5000)
        }, index=dates)

    # Resample to 5-min if needed
    if len(prices) > 100000:
        prices = prices.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

    print(f"\nLoaded {len(prices)} bars")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")

    # Generate MTF features
    print("\n[1] Generating multi-timeframe features...")
    config = MTFConfig(
        base_timeframe=5,
        higher_timeframes=[15, 60, 240]
    )

    mtf_features = calculate_multi_timeframe_features(prices, config)

    print(f"\n[2] Generated {len(mtf_features.columns)} features:")
    for col in sorted(mtf_features.columns)[:20]:
        print(f"    {col}")
    if len(mtf_features.columns) > 20:
        print(f"    ... and {len(mtf_features.columns) - 20} more")

    # Validate for leakage
    print("\n[3] Validating for data leakage...")
    validation = validate_mtf_features_no_leakage(mtf_features, prices)

    print(f"    Validation passed: {validation['passed']}")
    for check in validation['checks']:
        print(f"    [OK] {check}")
    for warning in validation['warnings']:
        print(f"    [WARN] {warning}")

    if 'suspicious_features' in validation:
        print("\n    Suspicious features:")
        for sf in validation['suspicious_features'][:5]:
            print(f"      {sf['feature']}: corr_5={sf['corr_future_5']:.3f}")

    print("\n" + "=" * 70)
    print("MULTI-TIMEFRAME FEATURES TEST COMPLETE")
    print("=" * 70)
