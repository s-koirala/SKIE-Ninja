"""
Intermarket Features (Category 6)
=================================
Implements cross-market relationship features.

Captures correlations and lead/lag relationships between:
- Equity indices (ES, NQ, YM)
- Bond markets (ZN, ZB, TLT)
- Currencies (DX dollar index)
- Commodities (Gold, Oil)
- Volatility (VIX correlation)

Reference: research/02_comprehensive_variables_research.md Section 14
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class IntermarketFeatures:
    """Calculate intermarket relationship features."""

    # Default correlation pairs for ES
    DEFAULT_PAIRS = [
        'NQ',   # NASDAQ futures
        'YM',   # Dow futures
        'ZN',   # 10-year Treasury
        'GC',   # Gold
        'CL',   # Crude Oil
        'DX',   # Dollar Index
    ]

    def __init__(
        self,
        primary_df: pd.DataFrame,
        related_data: Optional[Dict[str, pd.DataFrame]] = None,
        primary_symbol: str = "ES"
    ):
        """
        Initialize intermarket features calculator.

        Args:
            primary_df: OHLCV DataFrame for primary instrument (ES)
            related_data: Dict of DataFrames for related instruments
            primary_symbol: Symbol of primary instrument
        """
        self.primary_df = primary_df.copy()
        self.related_data = related_data or {}
        self.primary_symbol = primary_symbol
        self._validate_data()

    def _validate_data(self):
        """Validate input data."""
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in self.primary_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not isinstance(self.primary_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

    def calculate_all(self) -> pd.DataFrame:
        """Calculate all intermarket features."""
        logger.info("Calculating intermarket features...")

        features = pd.DataFrame(index=self.primary_df.index)

        # If no related data provided, use synthetic correlations
        if not self.related_data:
            logger.info("No related data provided, using synthetic intermarket proxies")
            related = self._create_synthetic_related_data()
        else:
            related = self.related_data

        # Cross-market correlations
        corr_features = self._calculate_correlations(related)
        features = pd.concat([features, corr_features], axis=1)

        # Lead/Lag relationships
        leadlag_features = self._calculate_lead_lag(related)
        features = pd.concat([features, leadlag_features], axis=1)

        # Spread features
        spread_features = self._calculate_spreads(related)
        features = pd.concat([features, spread_features], axis=1)

        # Relative strength features
        rs_features = self._calculate_relative_strength(related)
        features = pd.concat([features, rs_features], axis=1)

        # Regime detection
        regime_features = self._calculate_correlation_regimes(related)
        features = pd.concat([features, regime_features], axis=1)

        logger.info(f"Generated {len(features.columns)} intermarket features")
        return features

    def _create_synthetic_related_data(self) -> Dict[str, pd.DataFrame]:
        """Create synthetic related instrument data for development."""
        related = {}
        primary_close = self.primary_df['close']
        primary_returns = primary_close.pct_change()

        np.random.seed(42)

        # NQ - highly correlated with ES (~0.95 correlation)
        nq_returns = primary_returns * 1.1 + np.random.randn(len(primary_returns)) * 0.001
        related['NQ'] = self._returns_to_ohlcv(nq_returns, base_price=7500)

        # YM - highly correlated with ES (~0.90 correlation)
        ym_returns = primary_returns * 0.9 + np.random.randn(len(primary_returns)) * 0.001
        related['YM'] = self._returns_to_ohlcv(ym_returns, base_price=28000)

        # ZN - negatively correlated with ES (~-0.3 correlation, flight to safety)
        zn_returns = -primary_returns * 0.3 + np.random.randn(len(primary_returns)) * 0.0005
        related['ZN'] = self._returns_to_ohlcv(zn_returns, base_price=130)

        # GC (Gold) - slight negative correlation (~-0.2), safe haven
        gc_returns = -primary_returns * 0.2 + np.random.randn(len(primary_returns)) * 0.002
        related['GC'] = self._returns_to_ohlcv(gc_returns, base_price=1500)

        # CL (Oil) - moderate positive correlation (~0.4), risk-on
        cl_returns = primary_returns * 0.4 + np.random.randn(len(primary_returns)) * 0.003
        related['CL'] = self._returns_to_ohlcv(cl_returns, base_price=55)

        # DX (Dollar) - negative correlation (~-0.4), dollar weakness = equity strength
        dx_returns = -primary_returns * 0.4 + np.random.randn(len(primary_returns)) * 0.001
        related['DX'] = self._returns_to_ohlcv(dx_returns, base_price=97)

        return related

    def _returns_to_ohlcv(
        self,
        returns: pd.Series,
        base_price: float = 100
    ) -> pd.DataFrame:
        """Convert returns series to synthetic OHLCV DataFrame."""
        # Generate price from returns
        price = base_price * (1 + returns).cumprod()
        price = price.fillna(base_price)

        # Create OHLCV with some noise
        noise = np.random.randn(len(price)) * 0.001
        df = pd.DataFrame(index=self.primary_df.index)
        df['close'] = price
        df['open'] = price.shift(1).fillna(base_price) * (1 + noise * 0.5)
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + abs(noise))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - abs(noise))
        df['volume'] = self.primary_df['volume'] * np.random.uniform(0.5, 1.5, len(price))

        return df

    def _calculate_correlations(
        self,
        related: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate rolling correlations between primary and related instruments."""
        df = pd.DataFrame(index=self.primary_df.index)
        primary_returns = self.primary_df['close'].pct_change()

        for symbol, data in related.items():
            if 'close' not in data.columns:
                continue

            related_returns = data['close'].pct_change()

            # Rolling correlations
            for window in [10, 20, 50]:
                corr = primary_returns.rolling(window).corr(related_returns)
                df[f'corr_{symbol}_{window}'] = corr

            # Correlation change
            corr_20 = primary_returns.rolling(20).corr(related_returns)
            corr_5 = primary_returns.rolling(5).corr(related_returns)
            df[f'corr_change_{symbol}'] = corr_5 - corr_20

        return df

    def _calculate_lead_lag(
        self,
        related: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate lead/lag relationships (cross-correlations at different lags)."""
        df = pd.DataFrame(index=self.primary_df.index)
        primary_returns = self.primary_df['close'].pct_change()

        for symbol, data in related.items():
            if 'close' not in data.columns:
                continue

            related_returns = data['close'].pct_change()

            # Lead correlations (does related market lead ES?)
            for lag in [1, 5, 10]:
                # Related returns lagged (occurred before)
                lagged_returns = related_returns.shift(lag)
                corr = primary_returns.rolling(20).corr(lagged_returns)
                df[f'lead_{symbol}_lag{lag}'] = corr

            # Does ES lead the related market?
            for lag in [1, 5]:
                lagged_primary = primary_returns.shift(lag)
                corr = related_returns.rolling(20).corr(lagged_primary)
                df[f'lag_{symbol}_lag{lag}'] = corr

        return df

    def _calculate_spreads(
        self,
        related: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate spread and ratio features between markets."""
        df = pd.DataFrame(index=self.primary_df.index)
        primary_returns = self.primary_df['close'].pct_change()

        # ES-NQ spread (tech vs broad market)
        if 'NQ' in related:
            nq_returns = related['NQ']['close'].pct_change()
            df['es_nq_return_spread'] = primary_returns - nq_returns
            df['es_nq_spread_ma5'] = df['es_nq_return_spread'].rolling(5).mean()
            df['es_nq_spread_zscore'] = (
                df['es_nq_return_spread'] - df['es_nq_return_spread'].rolling(20).mean()
            ) / (df['es_nq_return_spread'].rolling(20).std() + 1e-10)

        # Stock-Bond spread
        if 'ZN' in related:
            zn_returns = related['ZN']['close'].pct_change()
            df['stock_bond_spread'] = primary_returns - zn_returns
            df['stock_bond_ma10'] = df['stock_bond_spread'].rolling(10).mean()

            # Risk-on/Risk-off indicator
            df['risk_on'] = (df['stock_bond_spread'].rolling(5).mean() > 0).astype(int)

        # Gold-Equity spread (safe haven demand)
        if 'GC' in related:
            gc_returns = related['GC']['close'].pct_change()
            df['gold_equity_spread'] = gc_returns - primary_returns
            df['gold_equity_ma10'] = df['gold_equity_spread'].rolling(10).mean()

        # Dollar impact
        if 'DX' in related:
            dx_returns = related['DX']['close'].pct_change()
            df['dollar_equity_spread'] = dx_returns + primary_returns  # Typically inverse
            df['dollar_strength'] = dx_returns.rolling(10).mean()

        return df

    def _calculate_relative_strength(
        self,
        related: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate relative strength features."""
        df = pd.DataFrame(index=self.primary_df.index)
        primary_close = self.primary_df['close']

        for symbol, data in related.items():
            if 'close' not in data.columns:
                continue

            related_close = data['close']

            # Relative Strength (price ratio)
            ratio = primary_close / related_close
            df[f'rs_{symbol}'] = ratio

            # RS momentum
            df[f'rs_{symbol}_mom5'] = ratio.pct_change(5)
            df[f'rs_{symbol}_mom20'] = ratio.pct_change(20)

            # RS vs moving average
            df[f'rs_{symbol}_vs_ma20'] = ratio / ratio.rolling(20).mean()

        return df

    def _calculate_correlation_regimes(
        self,
        related: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Detect correlation regime changes."""
        df = pd.DataFrame(index=self.primary_df.index)
        primary_returns = self.primary_df['close'].pct_change()

        # Average cross-market correlation
        corr_list = []
        for symbol, data in related.items():
            if 'close' not in data.columns:
                continue
            related_returns = data['close'].pct_change()
            corr = primary_returns.rolling(20).corr(related_returns)
            corr_list.append(corr)

        if corr_list:
            avg_corr = pd.concat(corr_list, axis=1).mean(axis=1)
            df['avg_cross_corr'] = avg_corr

            # Correlation regime
            df['high_corr_regime'] = (avg_corr > 0.7).astype(int)
            df['low_corr_regime'] = (avg_corr < 0.3).astype(int)

            # Correlation breakout (sudden change)
            corr_change = avg_corr.diff(5)
            df['corr_spike_up'] = (corr_change > corr_change.rolling(20).std() * 2).astype(int)
            df['corr_spike_down'] = (corr_change < -corr_change.rolling(20).std() * 2).astype(int)

        # Stock-Bond correlation regime (important for risk management)
        if 'ZN' in related:
            zn_returns = related['ZN']['close'].pct_change()
            sb_corr = primary_returns.rolling(20).corr(zn_returns)
            df['stock_bond_corr'] = sb_corr

            # Normal: negative correlation, Crisis: positive (both crash or rally)
            df['sb_corr_normal'] = (sb_corr < 0).astype(int)
            df['sb_corr_crisis'] = (sb_corr > 0.3).astype(int)

        return df


def calculate_intermarket_features(
    primary_df: pd.DataFrame,
    related_data: Optional[Dict[str, pd.DataFrame]] = None,
    primary_symbol: str = "ES"
) -> pd.DataFrame:
    """
    Convenience function to calculate all intermarket features.

    Args:
        primary_df: OHLCV DataFrame for primary instrument
        related_data: Optional dict of related instrument DataFrames
        primary_symbol: Symbol of primary instrument

    Returns:
        DataFrame with intermarket features
    """
    calculator = IntermarketFeatures(primary_df, related_data, primary_symbol)
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
    print("Intermarket Features Test")
    print("=" * 60)

    es_data, _ = load_sample_data()

    # Calculate features (using synthetic related data)
    features = calculate_intermarket_features(es_data)

    print(f"\nGenerated {len(features.columns)} intermarket features:")
    for i, col in enumerate(features.columns, 1):
        print(f"  {i:2d}. {col}")

    print(f"\nSample output (row 500):")
    sample_row = features.iloc[500]
    for col, val in list(sample_row.items())[:20]:
        print(f"  {col}: {val:.4f}" if pd.notna(val) else f"  {col}: NaN")

    print(f"\nFeature statistics:")
    print(features.describe().T[['mean', 'std', 'min', 'max']])
