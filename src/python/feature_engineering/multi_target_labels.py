"""
Multi-Target Label Generation for SKIE-Ninja

Instead of predicting a single binary "will trade be profitable" target,
this module generates multiple more tractable prediction targets:

1. VOLATILITY TARGETS - Predict future volatility (highly predictable)
   - Future ATR, realized volatility
   - Volatility expansion/contraction
   - Regime classification

2. TREND TARGETS - Predict direction over N-bar horizons (more stable)
   - Multi-horizon direction (10, 20, 30 bars)
   - Trend strength and persistence
   - Momentum signals

3. PRICE TARGETS - Predict price level attainment (structural)
   - Will price reach X*ATR above/below?
   - Support/resistance breakout probability
   - Reversal probability at levels

These targets are combined in strategy layer for selective trading.

References:
- Bollerslev (1986): GARCH - Volatility clustering and predictability
- Jegadeesh & Titman (1993): Momentum effect
- Lopez de Prado (2018): Financial ML methodology

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class MultiTargetConfig:
    """Configuration for multi-target label generation."""

    # Volatility targets
    vol_horizons: Tuple[int, ...] = (5, 10, 20)
    vol_expansion_threshold: float = 1.2  # 20% increase = expansion

    # Trend targets
    trend_horizons: Tuple[int, ...] = (10, 20, 30)
    min_trend_return: float = 0.001  # 0.1% minimum to count as directional

    # Price targets (ATR multiples)
    atr_multiples: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5)
    price_horizons: Tuple[int, ...] = (10, 20, 30)

    # General
    atr_period: int = 14
    min_samples: int = 50  # Minimum samples for valid target


class MultiTargetLabeler:
    """
    Generate multiple prediction targets from OHLCV price data.

    This class creates labels for three families of predictions:
    - Volatility (easiest to predict, use for sizing/timing)
    - Trend (moderate difficulty, use for direction)
    - Price targets (structural, use for TP/SL placement)

    Example:
        >>> labeler = MultiTargetLabeler()
        >>> targets = labeler.generate_all_targets(prices_df)
        >>> print(targets.columns.tolist())
    """

    def __init__(self, config: Optional[MultiTargetConfig] = None):
        self.config = config or MultiTargetConfig()
        self._atr_cache = None

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.rolling(window=period).mean()

    def _calculate_realized_vol(
        self,
        df: pd.DataFrame,
        window: int,
        annualize: bool = False
    ) -> pd.Series:
        """Calculate realized volatility from close-to-close returns."""
        returns = df['close'].pct_change()
        rv = returns.rolling(window).std()

        if annualize:
            # Annualize for 5-min bars, 78 bars/day RTH, 252 days/year
            rv = rv * np.sqrt(78 * 252)

        return rv

    def _calculate_parkinson_vol(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Parkinson volatility from high-low range."""
        log_hl = np.log(df['high'] / df['low'])
        factor = 1 / (4 * np.log(2))
        parkinson = np.sqrt(factor * (log_hl ** 2).rolling(window).mean())
        return parkinson

    # =========================================================================
    # VOLATILITY TARGETS
    # =========================================================================

    def generate_volatility_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate future volatility prediction targets.

        These are among the most predictable targets due to volatility clustering.

        Targets generated:
        - future_rv_{horizon}: Future realized volatility (regression)
        - future_atr_{horizon}: Future ATR (regression)
        - vol_expansion_{horizon}: Will volatility increase? (classification)
        - vol_regime_{horizon}: High/medium/low vol regime (classification)

        Returns:
            DataFrame with volatility targets
        """
        targets = pd.DataFrame(index=df.index)

        logger.info("Generating volatility targets...")

        for horizon in self.config.vol_horizons:
            # Future realized volatility (regression target)
            current_rv = self._calculate_realized_vol(df, horizon)
            future_rv = current_rv.shift(-horizon)
            targets[f'future_rv_{horizon}'] = future_rv

            # Future ATR (regression target)
            current_atr = self._calculate_atr(df, horizon)
            future_atr = current_atr.shift(-horizon)
            targets[f'future_atr_{horizon}'] = future_atr

            # Volatility expansion (classification: will vol increase?)
            threshold = self.config.vol_expansion_threshold
            targets[f'vol_expansion_{horizon}'] = (
                future_rv > current_rv * threshold
            ).astype(int)

            # Volatility contraction
            targets[f'vol_contraction_{horizon}'] = (
                future_rv < current_rv / threshold
            ).astype(int)

            # Parkinson volatility (more accurate for intraday)
            current_parkinson = self._calculate_parkinson_vol(df, horizon)
            future_parkinson = current_parkinson.shift(-horizon)
            targets[f'future_parkinson_{horizon}'] = future_parkinson

        # Volatility regime classification (based on current + future)
        rv_20 = self._calculate_realized_vol(df, 20)
        future_rv_20 = rv_20.shift(-20)

        # Define regime thresholds (percentiles)
        rv_33 = rv_20.rolling(252 * 78).quantile(0.33)  # ~1 year
        rv_67 = rv_20.rolling(252 * 78).quantile(0.67)

        # Future regime: 0=low, 1=medium, 2=high
        targets['future_vol_regime'] = np.where(
            future_rv_20 <= rv_33, 0,
            np.where(future_rv_20 <= rv_67, 1, 2)
        )

        logger.info(f"Generated {len(targets.columns)} volatility targets")
        return targets

    # =========================================================================
    # TREND TARGETS
    # =========================================================================

    def generate_trend_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate future trend prediction targets.

        Multi-horizon trend prediction is more stable than single-bar direction.

        Targets generated:
        - trend_dir_{horizon}: Direction over horizon (classification)
        - trend_strength_{horizon}: Return magnitude (regression)
        - trend_persist_{horizon}: Will current trend continue? (classification)
        - momentum_{horizon}: Momentum score (regression)

        Returns:
            DataFrame with trend targets
        """
        targets = pd.DataFrame(index=df.index)

        logger.info("Generating trend targets...")

        for horizon in self.config.trend_horizons:
            # Future return (regression target)
            future_return = (df['close'].shift(-horizon) - df['close']) / df['close']
            targets[f'future_return_{horizon}'] = future_return

            # Trend direction (classification: 1=up, 0=down)
            # Use threshold to avoid labeling noise as signal
            min_ret = self.config.min_trend_return
            targets[f'trend_dir_{horizon}'] = np.where(
                future_return > min_ret, 1,
                np.where(future_return < -min_ret, 0, np.nan)
            )

            # Trend strength (absolute return, regression)
            targets[f'trend_strength_{horizon}'] = future_return.abs()

            # Trend persistence (did prior trend continue?)
            past_trend = (df['close'] - df['close'].shift(horizon)) > 0
            future_trend = future_return > 0
            targets[f'trend_persist_{horizon}'] = (
                past_trend == future_trend
            ).astype(int)

            # Maximum favorable excursion (best price during horizon)
            future_high = df['high'].rolling(horizon).max().shift(-horizon)
            future_low = df['low'].rolling(horizon).min().shift(-horizon)
            targets[f'mfe_up_{horizon}'] = (future_high - df['close']) / df['close']
            targets[f'mfe_down_{horizon}'] = (df['close'] - future_low) / df['close']

        # Momentum indicators (for signal quality)
        for short, long in [(5, 20), (10, 30), (20, 60)]:
            short_ma = df['close'].rolling(short).mean()
            long_ma = df['close'].rolling(long).mean()
            targets[f'momentum_{short}_{long}'] = (short_ma - long_ma) / long_ma

        logger.info(f"Generated {len(targets.columns)} trend targets")
        return targets

    # =========================================================================
    # PRICE TARGET PREDICTIONS
    # =========================================================================

    def generate_price_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate price level attainment targets.

        Predicts whether price will reach specific levels (useful for TP/SL).

        Targets generated:
        - reach_{mult}atr_up_{horizon}: Will price reach mult*ATR above? (classification)
        - reach_{mult}atr_down_{horizon}: Will price reach mult*ATR below? (classification)
        - touch_high_{horizon}: Will price make new N-bar high? (classification)
        - touch_low_{horizon}: Will price make new N-bar low? (classification)

        Returns:
            DataFrame with price target predictions
        """
        targets = pd.DataFrame(index=df.index)

        logger.info("Generating price targets...")

        # Calculate ATR for level definitions
        atr = self._calculate_atr(df, self.config.atr_period)

        for mult in self.config.atr_multiples:
            for horizon in self.config.price_horizons:
                # Target level above current price
                target_up = df['close'] + mult * atr
                future_high = df['high'].rolling(horizon).max().shift(-horizon)
                targets[f'reach_{mult}atr_up_{horizon}'] = (
                    future_high >= target_up
                ).astype(int)

                # Target level below current price
                target_down = df['close'] - mult * atr
                future_low = df['low'].rolling(horizon).min().shift(-horizon)
                targets[f'reach_{mult}atr_down_{horizon}'] = (
                    future_low <= target_down
                ).astype(int)

        # Touch new highs/lows
        for horizon in self.config.price_horizons:
            lookback_high = df['high'].rolling(horizon).max()
            lookback_low = df['low'].rolling(horizon).min()
            future_high = df['high'].rolling(horizon).max().shift(-horizon)
            future_low = df['low'].rolling(horizon).min().shift(-horizon)

            # Will make new high relative to lookback?
            targets[f'new_high_{horizon}'] = (
                future_high > lookback_high
            ).astype(int)

            # Will make new low relative to lookback?
            targets[f'new_low_{horizon}'] = (
                future_low < lookback_low
            ).astype(int)

        # Risk/reward targets (for position sizing)
        for horizon in [10, 20]:
            future_high = df['high'].rolling(horizon).max().shift(-horizon)
            future_low = df['low'].rolling(horizon).min().shift(-horizon)

            # Upside potential
            targets[f'upside_pct_{horizon}'] = (future_high - df['close']) / df['close']

            # Downside risk
            targets[f'downside_pct_{horizon}'] = (df['close'] - future_low) / df['close']

            # Risk/Reward ratio (upside / downside)
            targets[f'rr_ratio_{horizon}'] = (
                targets[f'upside_pct_{horizon}'] /
                (targets[f'downside_pct_{horizon}'] + 0.0001)
            )

        logger.info(f"Generated {len(targets.columns)} price targets")
        return targets

    # =========================================================================
    # COMBINED GENERATION
    # =========================================================================

    def generate_all_targets(
        self,
        df: pd.DataFrame,
        include_volatility: bool = True,
        include_trend: bool = True,
        include_price: bool = True
    ) -> pd.DataFrame:
        """
        Generate all multi-target labels.

        Args:
            df: OHLCV DataFrame with datetime index
            include_volatility: Generate volatility targets
            include_trend: Generate trend targets
            include_price: Generate price targets

        Returns:
            DataFrame with all target columns
        """
        logger.info("=" * 60)
        logger.info("GENERATING MULTI-TARGET LABELS")
        logger.info("=" * 60)

        all_targets = pd.DataFrame(index=df.index)

        if include_volatility:
            vol_targets = self.generate_volatility_targets(df)
            all_targets = pd.concat([all_targets, vol_targets], axis=1)

        if include_trend:
            trend_targets = self.generate_trend_targets(df)
            all_targets = pd.concat([all_targets, trend_targets], axis=1)

        if include_price:
            price_targets = self.generate_price_targets(df)
            all_targets = pd.concat([all_targets, price_targets], axis=1)

        # Drop rows with NaN (at edges due to rolling/shifting)
        valid_count_before = len(all_targets)
        all_targets = all_targets.dropna()
        valid_count_after = len(all_targets)

        logger.info(f"\nTarget Generation Summary:")
        logger.info(f"  Total targets: {len(all_targets.columns)}")
        logger.info(f"  Valid samples: {valid_count_after} / {valid_count_before}")
        logger.info(f"  Sample loss from edge effects: {valid_count_before - valid_count_after}")

        return all_targets

    def get_target_summary(self, targets: pd.DataFrame) -> Dict[str, Dict]:
        """
        Get summary statistics for all generated targets.

        Returns:
            Dictionary with summary stats for each target
        """
        summary = {}

        for col in targets.columns:
            target = targets[col].dropna()

            if target.nunique() <= 3:  # Classification target
                summary[col] = {
                    'type': 'classification',
                    'n_classes': target.nunique(),
                    'class_distribution': target.value_counts(normalize=True).to_dict(),
                    'count': len(target)
                }
            else:  # Regression target
                summary[col] = {
                    'type': 'regression',
                    'mean': target.mean(),
                    'std': target.std(),
                    'min': target.min(),
                    'max': target.max(),
                    'median': target.median(),
                    'count': len(target)
                }

        return summary


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_multi_targets(
    df: pd.DataFrame,
    config: Optional[MultiTargetConfig] = None
) -> pd.DataFrame:
    """
    Convenience function to generate all multi-target labels.

    Args:
        df: OHLCV DataFrame
        config: Optional configuration

    Returns:
        DataFrame with all target labels
    """
    labeler = MultiTargetLabeler(config)
    return labeler.generate_all_targets(df)


def get_classification_targets(targets: pd.DataFrame) -> List[str]:
    """Get list of classification target column names."""
    classification_cols = []

    for col in targets.columns:
        if targets[col].nunique() <= 3:
            classification_cols.append(col)

    return classification_cols


def get_regression_targets(targets: pd.DataFrame) -> List[str]:
    """Get list of regression target column names."""
    regression_cols = []

    for col in targets.columns:
        if targets[col].nunique() > 3:
            regression_cols.append(col)

    return regression_cols


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    np.random.seed(42)
    n_bars = 10000

    dates = pd.date_range('2023-01-01', periods=n_bars, freq='5min')
    prices = 4500 + np.cumsum(np.random.randn(n_bars) * 0.5)

    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n_bars)) * 2,
        'low': prices - np.abs(np.random.randn(n_bars)) * 2,
        'close': prices + np.random.randn(n_bars) * 0.5,
        'volume': np.random.randint(1000, 10000, n_bars)
    }, index=dates)

    # Generate targets
    labeler = MultiTargetLabeler()
    targets = labeler.generate_all_targets(df)

    print("\n" + "=" * 60)
    print("TARGET GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal targets generated: {len(targets.columns)}")

    # Show classification vs regression split
    class_targets = get_classification_targets(targets)
    reg_targets = get_regression_targets(targets)

    print(f"\nClassification targets: {len(class_targets)}")
    print(f"Regression targets: {len(reg_targets)}")

    # Show sample of each type
    print("\nClassification targets (sample):")
    for t in class_targets[:5]:
        dist = targets[t].value_counts(normalize=True)
        print(f"  {t}: {dict(dist.round(3))}")

    print("\nRegression targets (sample):")
    for t in reg_targets[:5]:
        print(f"  {t}: mean={targets[t].mean():.4f}, std={targets[t].std():.4f}")
