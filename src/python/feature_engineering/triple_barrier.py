"""
Triple Barrier Labeling Method

Implementation of the Triple Barrier Method from Marcos Lopez de Prado's
"Advances in Financial Machine Learning" (2018).

The method creates labels based on which barrier is touched first:
1. Upper Barrier (Take Profit): Price rises to target -> Label: 1
2. Lower Barrier (Stop Loss): Price falls to stop -> Label: -1
3. Vertical Barrier (Time): Max holding period expires -> Label: sign of return

This provides more realistic labels than simple direction prediction because:
- Labels reflect actual trade outcomes with stops/targets
- Barrier widths can be volatility-adjusted (ATR-based)
- Works naturally with bet sizing via meta-labeling

References:
- Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
- https://www.mlfinlab.com/en/latest/labeling/tb_meta_labeling.html
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BarrierType(Enum):
    """Which barrier was touched to exit the trade."""
    UPPER = 1       # Take profit hit
    LOWER = -1      # Stop loss hit
    VERTICAL = 0    # Time barrier hit


@dataclass
class TripleBarrierConfig:
    """Configuration for Triple Barrier labeling."""
    # Barrier widths (in ATR multiples or absolute)
    upper_barrier: float = 2.0      # Take profit level (ATR multiples)
    lower_barrier: float = 1.0      # Stop loss level (ATR multiples)
    max_holding_bars: int = 12      # Maximum bars to hold (vertical barrier)

    # ATR configuration
    atr_period: int = 14            # Period for ATR calculation
    use_atr: bool = True            # If False, barriers are absolute returns

    # Label configuration
    min_return_threshold: float = 0.0001  # Minimum return to label as +1/-1

    # Side information (optional - for meta-labeling)
    use_side: bool = False          # If True, use provided side for labels


class TripleBarrierLabeler:
    """
    Triple Barrier labeling for ML training targets.

    Creates labels based on which barrier (upper/lower/vertical) is touched first.
    Supports both ATR-adjusted and absolute barrier widths.

    Example:
        >>> labeler = TripleBarrierLabeler(config)
        >>> labels = labeler.fit_transform(prices)
        >>> print(labels['tb_label'].value_counts())
    """

    def __init__(self, config: Optional[TripleBarrierConfig] = None):
        """
        Initialize the Triple Barrier labeler.

        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or TripleBarrierConfig()
        self._atr = None

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range for volatility adjustment."""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Average True Range
        atr = tr.rolling(window=self.config.atr_period).mean()
        return atr

    def _get_barrier_prices(
        self,
        entry_price: float,
        atr_value: float,
        side: int = 1
    ) -> Tuple[float, float]:
        """
        Calculate upper and lower barrier prices.

        Args:
            entry_price: Price at entry
            atr_value: ATR value for volatility adjustment
            side: 1 for long, -1 for short

        Returns:
            Tuple of (upper_barrier_price, lower_barrier_price)
        """
        if self.config.use_atr:
            upper_width = self.config.upper_barrier * atr_value
            lower_width = self.config.lower_barrier * atr_value
        else:
            # Absolute return percentages
            upper_width = entry_price * self.config.upper_barrier
            lower_width = entry_price * self.config.lower_barrier

        if side >= 0:  # Long position
            upper_price = entry_price + upper_width
            lower_price = entry_price - lower_width
        else:  # Short position
            upper_price = entry_price - upper_width  # Profit for short
            lower_price = entry_price + lower_width  # Loss for short

        return upper_price, lower_price

    def _label_single_event(
        self,
        idx: int,
        prices: pd.DataFrame,
        atr: pd.Series,
        side: int = 1
    ) -> Tuple[int, BarrierType, int, float]:
        """
        Label a single trading event.

        Args:
            idx: Index position of entry
            prices: DataFrame with OHLC data
            atr: ATR series
            side: Trade direction (1=long, -1=short)

        Returns:
            Tuple of (label, barrier_type, exit_bar, return)
        """
        entry_price = prices.iloc[idx]['close']
        entry_atr = atr.iloc[idx]

        if pd.isna(entry_atr):
            return 0, BarrierType.VERTICAL, 0, 0.0

        upper_barrier, lower_barrier = self._get_barrier_prices(
            entry_price, entry_atr, side
        )

        # Look forward up to max_holding_bars
        max_idx = min(idx + self.config.max_holding_bars + 1, len(prices))

        for j in range(idx + 1, max_idx):
            high_price = prices.iloc[j]['high']
            low_price = prices.iloc[j]['low']
            close_price = prices.iloc[j]['close']

            if side >= 0:  # Long position
                # Check upper barrier (take profit)
                if high_price >= upper_barrier:
                    ret = (upper_barrier - entry_price) / entry_price
                    return 1, BarrierType.UPPER, j - idx, ret

                # Check lower barrier (stop loss)
                if low_price <= lower_barrier:
                    ret = (lower_barrier - entry_price) / entry_price
                    return -1, BarrierType.LOWER, j - idx, ret
            else:  # Short position
                # Check upper barrier (take profit for short = price goes down)
                if low_price <= upper_barrier:
                    ret = (entry_price - upper_barrier) / entry_price
                    return 1, BarrierType.UPPER, j - idx, ret

                # Check lower barrier (stop loss for short = price goes up)
                if high_price >= lower_barrier:
                    ret = (entry_price - lower_barrier) / entry_price
                    return -1, BarrierType.LOWER, j - idx, ret

        # Vertical barrier hit (time expired)
        if max_idx > idx + 1:
            exit_price = prices.iloc[max_idx - 1]['close']
            ret = (exit_price - entry_price) / entry_price
            if side < 0:
                ret = -ret  # Invert for short

            # Label based on final return
            if abs(ret) < self.config.min_return_threshold:
                label = 0
            else:
                label = 1 if ret > 0 else -1

            return label, BarrierType.VERTICAL, max_idx - idx - 1, ret

        return 0, BarrierType.VERTICAL, 0, 0.0

    def fit_transform(
        self,
        prices: pd.DataFrame,
        sides: Optional[pd.Series] = None,
        events: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Generate Triple Barrier labels for the entire dataset.

        Args:
            prices: DataFrame with OHLC columns (open, high, low, close)
            sides: Optional series of trade directions (1=long, -1=short)
                  If None, assumes long for all entries
            events: Optional boolean series indicating which bars to label
                   If None, labels all bars

        Returns:
            DataFrame with columns:
            - tb_label: Triple barrier label (-1, 0, 1)
            - tb_barrier_type: Which barrier was hit (upper/lower/vertical)
            - tb_holding_bars: Number of bars held
            - tb_return: Return achieved
            - tb_upper_barrier: Upper barrier price
            - tb_lower_barrier: Lower barrier price
        """
        # Validate input
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in prices.columns:
                raise ValueError(f"Missing required column: {col}")

        # Calculate ATR
        self._atr = self._calculate_atr(prices)

        # Initialize output arrays
        n = len(prices)
        labels = np.zeros(n, dtype=np.int32)
        barrier_types = np.zeros(n, dtype=np.int32)
        holding_bars = np.zeros(n, dtype=np.int32)
        returns = np.zeros(n, dtype=np.float64)
        upper_barriers = np.zeros(n, dtype=np.float64)
        lower_barriers = np.zeros(n, dtype=np.float64)

        # Default sides
        if sides is None:
            sides = pd.Series(1, index=prices.index)

        # Default events (all bars)
        if events is None:
            events = pd.Series(True, index=prices.index)

        # Process each event
        for i in range(n - self.config.max_holding_bars):
            if not events.iloc[i]:
                continue

            side = sides.iloc[i] if self.config.use_side else 1
            label, barrier_type, bars, ret = self._label_single_event(
                i, prices, self._atr, side
            )

            labels[i] = label
            barrier_types[i] = barrier_type.value
            holding_bars[i] = bars
            returns[i] = ret

            # Record barrier prices
            entry_price = prices.iloc[i]['close']
            atr_val = self._atr.iloc[i]
            if not pd.isna(atr_val):
                upper, lower = self._get_barrier_prices(entry_price, atr_val, side)
                upper_barriers[i] = upper
                lower_barriers[i] = lower

        # Create output DataFrame
        result = pd.DataFrame({
            'tb_label': labels,
            'tb_barrier_type': barrier_types,
            'tb_holding_bars': holding_bars,
            'tb_return': returns,
            'tb_upper_barrier': upper_barriers,
            'tb_lower_barrier': lower_barriers
        }, index=prices.index)

        # Log statistics
        label_counts = pd.Series(labels).value_counts()
        logger.info(f"Triple Barrier Labels: {dict(label_counts)}")

        return result

    def get_meta_labels(
        self,
        tb_labels: pd.DataFrame,
        primary_predictions: pd.Series
    ) -> pd.Series:
        """
        Generate meta-labels for the secondary model.

        Meta-labels indicate whether the primary model's prediction was correct.
        Used for training the meta-labeling model that determines bet size.

        Args:
            tb_labels: Output from fit_transform()
            primary_predictions: Primary model predictions (1=long, -1=short, 0=no trade)

        Returns:
            Series of meta-labels (1=correct, 0=incorrect)
        """
        # Primary model was correct if:
        # - It predicted long (1) and tb_label is 1 (hit upper barrier)
        # - It predicted short (-1) and tb_label is -1 (hit lower barrier)
        # - It predicted no trade (0) and we count that as neutral

        meta_labels = pd.Series(0, index=tb_labels.index)

        # Long predictions that were correct
        long_correct = (primary_predictions == 1) & (tb_labels['tb_label'] == 1)

        # Short predictions that were correct
        short_correct = (primary_predictions == -1) & (tb_labels['tb_label'] == -1)

        meta_labels[long_correct | short_correct] = 1

        return meta_labels


def apply_triple_barrier(
    prices: pd.DataFrame,
    upper_atr: float = 2.0,
    lower_atr: float = 1.0,
    max_holding: int = 12,
    atr_period: int = 14
) -> pd.DataFrame:
    """
    Convenience function to apply Triple Barrier labeling.

    Args:
        prices: DataFrame with OHLC data
        upper_atr: Take profit in ATR multiples
        lower_atr: Stop loss in ATR multiples
        max_holding: Maximum holding period in bars
        atr_period: ATR calculation period

    Returns:
        DataFrame with triple barrier features added
    """
    config = TripleBarrierConfig(
        upper_barrier=upper_atr,
        lower_barrier=lower_atr,
        max_holding_bars=max_holding,
        atr_period=atr_period,
        use_atr=True
    )

    labeler = TripleBarrierLabeler(config)
    tb_labels = labeler.fit_transform(prices)

    return tb_labels


def generate_barrier_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Generate multiple Triple Barrier feature sets with different configurations.

    Creates labels with various barrier widths and holding periods for
    ensemble learning or multi-horizon prediction.

    Args:
        prices: DataFrame with OHLC data

    Returns:
        DataFrame with multiple TB label columns
    """
    features = pd.DataFrame(index=prices.index)

    # Different barrier configurations
    configs = [
        # (upper_atr, lower_atr, max_holding, suffix)
        (1.5, 1.0, 6, 'tight_short'),
        (2.0, 1.0, 12, 'standard'),
        (2.5, 1.5, 18, 'wide_medium'),
        (3.0, 2.0, 24, 'wide_long'),
    ]

    for upper, lower, holding, suffix in configs:
        config = TripleBarrierConfig(
            upper_barrier=upper,
            lower_barrier=lower,
            max_holding_bars=holding
        )

        labeler = TripleBarrierLabeler(config)
        tb_labels = labeler.fit_transform(prices)

        # Rename columns with suffix
        for col in tb_labels.columns:
            features[f'{col}_{suffix}'] = tb_labels[col]

    logger.info(f"Generated {len(features.columns)} Triple Barrier features")
    return features


if __name__ == "__main__":
    # Test with sample data
    import sys
    sys.path.insert(0, str(__file__).replace('feature_engineering/triple_barrier.py', ''))

    print("=" * 70)
    print("TRIPLE BARRIER LABELING - TEST")
    print("=" * 70)

    # Create sample data
    np.random.seed(42)
    n_bars = 1000

    # Simulate price data with some trend
    returns = np.random.randn(n_bars) * 0.002 + 0.0001
    close = 4500 * np.cumprod(1 + returns)

    prices = pd.DataFrame({
        'open': close * (1 + np.random.randn(n_bars) * 0.001),
        'high': close * (1 + np.abs(np.random.randn(n_bars)) * 0.002),
        'low': close * (1 - np.abs(np.random.randn(n_bars)) * 0.002),
        'close': close
    })

    # Test basic labeling
    print("\n[1] Testing basic Triple Barrier labeling...")
    config = TripleBarrierConfig(
        upper_barrier=2.0,
        lower_barrier=1.0,
        max_holding_bars=12
    )

    labeler = TripleBarrierLabeler(config)
    labels = labeler.fit_transform(prices)

    print(f"\nLabel distribution:")
    print(labels['tb_label'].value_counts().sort_index())

    print(f"\nBarrier type distribution:")
    barrier_names = {1: 'Upper (TP)', -1: 'Lower (SL)', 0: 'Vertical (Time)'}
    for val, name in barrier_names.items():
        count = (labels['tb_barrier_type'] == val).sum()
        print(f"  {name}: {count}")

    print(f"\nAverage holding period: {labels['tb_holding_bars'].mean():.1f} bars")
    print(f"Average return: {labels['tb_return'].mean() * 100:.4f}%")

    # Test multi-configuration
    print("\n[2] Testing multi-configuration barrier features...")
    multi_features = generate_barrier_features(prices)
    print(f"Generated {len(multi_features.columns)} features")
    print(f"Feature names: {list(multi_features.columns[:8])}...")

    print("\n" + "=" * 70)
    print("TRIPLE BARRIER TEST COMPLETE")
    print("=" * 70)
