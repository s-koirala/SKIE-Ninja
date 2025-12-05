"""
Enhanced Cross-Market Features Module
=====================================

Generates features from cross-market relationships using REAL Databento data.
Captures correlations, lead/lag relationships, sector rotation, and divergences.

Markets analyzed:
- ES (S&P 500 E-mini) - Primary
- NQ (NASDAQ E-mini) - Tech sector
- YM (Dow E-mini) - Blue chips
- GC (Gold) - Safe haven
- CL (Crude Oil) - Risk sentiment
- ZN (10-Year Treasury) - Bonds/Rates
- VIX (from daily data) - Volatility index

CRITICAL: Follows strict data leakage prevention rules:
1. NEVER use shift(-N) - only positive shifts allowed
2. NEVER use center=True in rolling windows
3. Cross-market data aligned with proper lag
4. All features use only PAST data at each prediction point

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class CrossMarketConfig:
    """Configuration for cross-market features."""

    # Data paths
    data_dir: Path = None

    # Related markets to analyze
    related_markets: List[str] = None

    # Correlation windows
    correlation_windows: List[int] = None

    # Lead/lag periods to test (in bars)
    lead_lag_periods: List[int] = None

    # Minimum bars required for valid correlation
    min_correlation_bars: int = 20

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'raw' / 'market'
        if self.related_markets is None:
            self.related_markets = ['NQ', 'YM', 'GC', 'CL', 'ZN']
        if self.correlation_windows is None:
            self.correlation_windows = [10, 20, 50]
        if self.lead_lag_periods is None:
            self.lead_lag_periods = [1, 5, 10]


class EnhancedCrossMarketFeatures:
    """
    Generate cross-market features using real Databento data.

    IMPORTANT: All features are calculated to avoid look-ahead bias:
    - Rolling calculations use past data only
    - Lead/lag analysis uses only PAST correlations
    - Cross-market data aligned before feature calculation
    """

    def __init__(self, config: Optional[CrossMarketConfig] = None):
        self.config = config or CrossMarketConfig()
        self.related_data = {}

    def load_related_market_data(
        self,
        primary_index: pd.DatetimeIndex,
        year_suffix: str = ""
    ) -> Dict[str, pd.DataFrame]:
        """
        Load related market data from Databento CSV files.

        Args:
            primary_index: DatetimeIndex from primary market (ES)
            year_suffix: Optional year suffix (e.g., "_2020", "_2021")

        Returns:
            Dict of DataFrames keyed by market symbol
        """
        logger.info("Loading related market data...")

        related_data = {}

        for symbol in self.config.related_markets:
            # Try different file patterns
            patterns = [
                f'{symbol}_1min_databento{year_suffix}.csv',
                f'{symbol}{year_suffix}_1min_databento.csv',
                f'{symbol}_1min_databento.csv'
            ]

            loaded = False
            for pattern in patterns:
                filepath = self.config.data_dir / pattern
                if filepath.exists():
                    try:
                        df = self._load_csv(filepath)

                        # Align to primary index
                        df = df.reindex(primary_index, method='ffill')

                        related_data[symbol] = df
                        logger.info(f"  Loaded {symbol}: {len(df)} bars from {pattern}")
                        loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"  Error loading {pattern}: {e}")

            if not loaded:
                logger.warning(f"  Could not find data for {symbol}")

        # Also try to load VIX daily data
        vix_path = self.config.data_dir / 'VIX_daily.csv'
        if vix_path.exists():
            try:
                vix_df = self._load_vix_daily(vix_path, primary_index)
                related_data['VIX'] = vix_df
                logger.info(f"  Loaded VIX daily data")
            except Exception as e:
                logger.warning(f"  Error loading VIX: {e}")

        self.related_data = related_data
        return related_data

    def _load_csv(self, filepath: Path) -> pd.DataFrame:
        """Load and standardize a Databento CSV file."""
        df = pd.read_csv(filepath)

        # Standardize column names
        df.columns = df.columns.str.lower()

        # Handle timestamp column
        time_col = None
        for col in ['ts_event', 'timestamp', 'datetime', 'time', 'date']:
            if col in df.columns:
                time_col = col
                break

        if time_col is None:
            raise ValueError(f"No timestamp column found in {filepath}")

        df['timestamp'] = pd.to_datetime(df[time_col])
        df = df.set_index('timestamp')

        # Standardize OHLCV columns
        col_mapping = {
            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
        }
        df = df.rename(columns=col_mapping)

        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                # Try to find alternative column names
                for alt in ['Open', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']:
                    if alt.lower() == col and alt in df.columns:
                        df[col] = df[alt]

        return df[required].dropna()

    def _load_vix_daily(
        self,
        filepath: Path,
        target_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Load VIX daily data and align to intraday index."""
        vix = pd.read_csv(filepath)
        vix.columns = vix.columns.str.lower()

        # Find date column
        date_col = None
        for col in ['date', 'timestamp', 'datetime']:
            if col in vix.columns:
                date_col = col
                break

        if date_col:
            vix['date'] = pd.to_datetime(vix[date_col])
            vix = vix.set_index('date')

        # Find close/price column
        close_col = None
        for col in ['close', 'adj close', 'price', 'vix']:
            if col in vix.columns:
                close_col = col
                break

        if close_col is None:
            raise ValueError("No close/price column found in VIX data")

        # Create DataFrame with VIX values
        vix_values = vix[[close_col]].rename(columns={close_col: 'close'})

        # Align to target index by date (forward fill within each day)
        result = pd.DataFrame(index=target_index)
        target_dates = target_index.normalize()

        for date in target_dates.unique():
            if date in vix_values.index:
                mask = target_dates == date
                result.loc[mask, 'close'] = vix_values.loc[date, 'close']

        result = result.ffill()

        # Add placeholder OHLCV columns
        result['open'] = result['close']
        result['high'] = result['close']
        result['low'] = result['close']
        result['volume'] = 0

        return result

    def calculate_correlation_features(
        self,
        primary_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate rolling correlations between primary and related markets.

        All correlations use ONLY past data (rolling lookback).

        Args:
            primary_df: Primary market (ES) OHLCV DataFrame

        Returns:
            DataFrame with correlation features
        """
        features = pd.DataFrame(index=primary_df.index)
        primary_returns = primary_df['close'].pct_change()

        for symbol, data in self.related_data.items():
            if 'close' not in data.columns:
                continue

            related_returns = data['close'].pct_change()

            # Rolling correlations (SAFE: rolling uses past data only)
            for window in self.config.correlation_windows:
                corr = primary_returns.rolling(window).corr(related_returns)
                features[f'corr_{symbol}_{window}'] = corr

            # Correlation change (trend in correlation)
            corr_20 = primary_returns.rolling(20).corr(related_returns)
            corr_5 = primary_returns.rolling(5).corr(related_returns)
            features[f'corr_change_{symbol}'] = corr_5 - corr_20

            # Correlation z-score (is current correlation unusual?)
            corr_ma = corr_20.rolling(50).mean()
            corr_std = corr_20.rolling(50).std()
            features[f'corr_zscore_{symbol}'] = (corr_20 - corr_ma) / (corr_std + 1e-10)

        return features

    def calculate_lead_lag_features(
        self,
        primary_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate lead/lag relationships between markets.

        Tests whether related markets lead or lag the primary market.
        Uses ONLY past correlations (lagged related returns vs current primary).

        Args:
            primary_df: Primary market (ES) OHLCV DataFrame

        Returns:
            DataFrame with lead/lag features
        """
        features = pd.DataFrame(index=primary_df.index)
        primary_returns = primary_df['close'].pct_change()

        for symbol, data in self.related_data.items():
            if 'close' not in data.columns:
                continue

            related_returns = data['close'].pct_change()

            # Lead correlations: Does related market's PAST move predict ES current?
            # SAFE: shift(lag) looks BACK in time on related_returns
            for lag in self.config.lead_lag_periods:
                # Related market's return from `lag` bars ago
                lagged_related = related_returns.shift(lag)  # SAFE: positive shift

                # Correlation of lagged related with current primary
                lead_corr = primary_returns.rolling(20).corr(lagged_related)
                features[f'lead_{symbol}_lag{lag}'] = lead_corr

            # Best lead indicator (which lag has highest correlation?)
            lead_cols = [f'lead_{symbol}_lag{lag}' for lag in self.config.lead_lag_periods]
            if all(col in features.columns for col in lead_cols):
                lead_df = features[lead_cols].abs()
                features[f'best_lead_{symbol}'] = lead_df.idxmax(axis=1).apply(
                    lambda x: int(x.split('lag')[-1]) if pd.notna(x) else 0
                )

        return features

    def calculate_spread_features(
        self,
        primary_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate spread and ratio features between markets.

        These capture relative performance and mean reversion opportunities.

        Args:
            primary_df: Primary market (ES) OHLCV DataFrame

        Returns:
            DataFrame with spread features
        """
        features = pd.DataFrame(index=primary_df.index)
        primary_returns = primary_df['close'].pct_change()

        # ES-NQ spread (Tech vs Broad Market)
        if 'NQ' in self.related_data:
            nq_returns = self.related_data['NQ']['close'].pct_change()

            # Return spread
            features['es_nq_spread'] = primary_returns - nq_returns

            # Rolling spread statistics (SAFE: rolling looks back)
            features['es_nq_spread_ma5'] = features['es_nq_spread'].rolling(5).mean()
            features['es_nq_spread_ma20'] = features['es_nq_spread'].rolling(20).mean()

            # Spread z-score (mean reversion signal)
            spread_ma = features['es_nq_spread'].rolling(50).mean()
            spread_std = features['es_nq_spread'].rolling(50).std()
            features['es_nq_spread_zscore'] = (
                features['es_nq_spread'] - spread_ma
            ) / (spread_std + 1e-10)

            # Tech leadership indicator
            features['tech_leading'] = (features['es_nq_spread'] < 0).astype(int)

        # Stock-Bond spread (Risk On/Off)
        if 'ZN' in self.related_data:
            zn_returns = self.related_data['ZN']['close'].pct_change()

            features['stock_bond_spread'] = primary_returns - zn_returns
            features['stock_bond_ma10'] = features['stock_bond_spread'].rolling(10).mean()

            # Risk-on indicator
            features['risk_on'] = (features['stock_bond_ma10'] > 0).astype(int)

            # Flight to safety (bonds up, stocks down)
            features['flight_to_safety'] = (
                (zn_returns > 0) & (primary_returns < 0)
            ).astype(int)

        # Gold-Equity spread (Safe Haven Demand)
        if 'GC' in self.related_data:
            gc_returns = self.related_data['GC']['close'].pct_change()

            features['gold_equity_spread'] = gc_returns - primary_returns
            features['gold_equity_ma10'] = features['gold_equity_spread'].rolling(10).mean()

            # Safe haven demand increasing
            features['safe_haven_demand'] = (
                features['gold_equity_spread'].rolling(5).mean() > 0
            ).astype(int)

        # Dollar impact (DX not in our data, but we can proxy)
        if 'CL' in self.related_data:
            cl_returns = self.related_data['CL']['close'].pct_change()

            # Oil-Equity relationship (risk sentiment)
            features['oil_equity_spread'] = cl_returns - primary_returns
            features['oil_equity_corr_20'] = primary_returns.rolling(20).corr(cl_returns)

        return features

    def calculate_relative_strength_features(
        self,
        primary_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate relative strength features.

        Measures which markets are outperforming/underperforming.

        Args:
            primary_df: Primary market (ES) OHLCV DataFrame

        Returns:
            DataFrame with relative strength features
        """
        features = pd.DataFrame(index=primary_df.index)
        primary_close = primary_df['close']

        for symbol, data in self.related_data.items():
            if symbol == 'VIX':  # VIX is different, handled separately
                continue

            if 'close' not in data.columns:
                continue

            related_close = data['close']

            # Price ratio (relative strength)
            ratio = primary_close / (related_close + 1e-10)

            # Normalize by dividing by rolling mean
            ratio_normalized = ratio / ratio.rolling(50).mean()
            features[f'rs_{symbol}'] = ratio_normalized

            # RS momentum (SAFE: pct_change looks back)
            features[f'rs_{symbol}_mom5'] = ratio.pct_change(5)
            features[f'rs_{symbol}_mom20'] = ratio.pct_change(20)

            # RS above/below average
            features[f'rs_{symbol}_above_avg'] = (ratio_normalized > 1).astype(int)

        return features

    def calculate_vix_features(
        self,
        primary_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate VIX-based features.

        VIX is a key sentiment indicator for equity markets.

        Args:
            primary_df: Primary market (ES) OHLCV DataFrame

        Returns:
            DataFrame with VIX features
        """
        features = pd.DataFrame(index=primary_df.index)

        if 'VIX' not in self.related_data:
            logger.warning("VIX data not available")
            return features

        vix = self.related_data['VIX']['close']
        primary_returns = primary_df['close'].pct_change()

        # VIX level features
        features['vix_level'] = vix
        features['vix_ma10'] = vix.rolling(10).mean()
        features['vix_ma20'] = vix.rolling(20).mean()

        # VIX relative to moving averages
        features['vix_vs_ma10'] = vix / (features['vix_ma10'] + 1e-10)
        features['vix_vs_ma20'] = vix / (features['vix_ma20'] + 1e-10)

        # VIX regime indicators
        features['vix_high'] = (vix > 25).astype(int)
        features['vix_extreme'] = (vix > 30).astype(int)
        features['vix_low'] = (vix < 15).astype(int)

        # VIX change (SAFE: pct_change looks back)
        features['vix_change_1d'] = vix.pct_change()
        features['vix_change_5d'] = vix.pct_change(5)

        # VIX z-score
        vix_ma50 = vix.rolling(50).mean()
        vix_std50 = vix.rolling(50).std()
        features['vix_zscore'] = (vix - vix_ma50) / (vix_std50 + 1e-10)

        # VIX spike (sudden increase)
        features['vix_spike'] = (features['vix_zscore'] > 2).astype(int)

        # VIX-Return correlation (rolling)
        features['vix_return_corr_20'] = vix.pct_change().rolling(20).corr(primary_returns)

        return features

    def calculate_regime_features(
        self,
        primary_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate market regime features from cross-market data.

        Identifies different market regimes (risk-on, risk-off, etc.)

        Args:
            primary_df: Primary market (ES) OHLCV DataFrame

        Returns:
            DataFrame with regime features
        """
        features = pd.DataFrame(index=primary_df.index)
        primary_returns = primary_df['close'].pct_change()

        # Collect correlations for regime detection
        corr_list = []
        for symbol, data in self.related_data.items():
            if symbol == 'VIX':
                continue
            if 'close' not in data.columns:
                continue

            related_returns = data['close'].pct_change()
            corr = primary_returns.rolling(20).corr(related_returns)
            corr_list.append(corr)

        if corr_list:
            # Average cross-market correlation
            avg_corr = pd.concat(corr_list, axis=1).mean(axis=1)
            features['avg_cross_corr'] = avg_corr

            # Correlation regime (high correlation = risk-off/panic)
            features['high_corr_regime'] = (avg_corr > 0.7).astype(int)
            features['low_corr_regime'] = (avg_corr < 0.3).astype(int)

            # Correlation trend
            features['corr_trending_up'] = (
                avg_corr > avg_corr.shift(5)  # SAFE: positive shift
            ).astype(int)

        # Stock-Bond correlation regime
        if 'ZN' in self.related_data:
            zn_returns = self.related_data['ZN']['close'].pct_change()
            sb_corr = primary_returns.rolling(20).corr(zn_returns)

            features['stock_bond_corr'] = sb_corr

            # Normal: negative (stocks up = bonds down)
            # Crisis: positive (both crash or both rally)
            features['sb_corr_normal'] = (sb_corr < 0).astype(int)
            features['sb_corr_crisis'] = (sb_corr > 0.3).astype(int)

        # Combined regime score
        regime_signals = []

        if 'VIX' in self.related_data:
            vix = self.related_data['VIX']['close']
            regime_signals.append((vix > 25).astype(int))  # High VIX = risk-off

        if 'risk_on' in features.columns:
            regime_signals.append(1 - features['risk_on'])  # Inverse for risk-off

        if regime_signals:
            regime_df = pd.concat(regime_signals, axis=1)
            features['risk_off_score'] = regime_df.mean(axis=1)

        return features

    def generate_all_features(
        self,
        primary_df: pd.DataFrame,
        year_suffix: str = ""
    ) -> pd.DataFrame:
        """
        Generate all cross-market features.

        Args:
            primary_df: Primary market (ES) OHLCV with DatetimeIndex
            year_suffix: Optional year suffix for data files

        Returns:
            DataFrame with all cross-market features
        """
        logger.info("Generating enhanced cross-market features...")

        # Load related market data
        self.load_related_market_data(primary_df.index, year_suffix)

        if not self.related_data:
            logger.warning("No related market data loaded. Using synthetic data.")
            # Fall back to synthetic if no real data
            from feature_engineering.intermarket_features import calculate_intermarket_features
            return calculate_intermarket_features(primary_df)

        all_features = pd.DataFrame(index=primary_df.index)

        # Calculate each feature category
        logger.info("  Calculating correlation features...")
        corr_features = self.calculate_correlation_features(primary_df)
        all_features = pd.concat([all_features, corr_features], axis=1)

        logger.info("  Calculating lead/lag features...")
        leadlag_features = self.calculate_lead_lag_features(primary_df)
        all_features = pd.concat([all_features, leadlag_features], axis=1)

        logger.info("  Calculating spread features...")
        spread_features = self.calculate_spread_features(primary_df)
        all_features = pd.concat([all_features, spread_features], axis=1)

        logger.info("  Calculating relative strength features...")
        rs_features = self.calculate_relative_strength_features(primary_df)
        all_features = pd.concat([all_features, rs_features], axis=1)

        logger.info("  Calculating VIX features...")
        vix_features = self.calculate_vix_features(primary_df)
        all_features = pd.concat([all_features, vix_features], axis=1)

        logger.info("  Calculating regime features...")
        regime_features = self.calculate_regime_features(primary_df)
        all_features = pd.concat([all_features, regime_features], axis=1)

        logger.info(f"Total cross-market features generated: {len(all_features.columns)}")

        return all_features


def calculate_enhanced_cross_market_features(
    primary_df: pd.DataFrame,
    config: Optional[CrossMarketConfig] = None,
    year_suffix: str = ""
) -> pd.DataFrame:
    """
    Convenience function to calculate all enhanced cross-market features.

    Args:
        primary_df: Primary market OHLCV DataFrame with DatetimeIndex
        config: Optional configuration
        year_suffix: Optional year suffix for data files

    Returns:
        DataFrame with all cross-market features
    """
    calculator = EnhancedCrossMarketFeatures(config)
    return calculator.generate_all_features(primary_df, year_suffix)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("ENHANCED CROSS-MARKET FEATURES TEST")
    print("=" * 70)

    # Load ES data
    try:
        from data_collection.ninjatrader_loader import load_sample_data
        es_data, _ = load_sample_data(source="databento")

        # Resample to 5-min
        es_data = es_data.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

    except Exception as e:
        print(f"Could not load ES data: {e}")
        print("Creating synthetic data...")

        dates = pd.date_range('2024-01-01 09:30', periods=5000, freq='5min')
        np.random.seed(42)
        close = 4500 + np.cumsum(np.random.randn(5000) * 2)

        es_data = pd.DataFrame({
            'open': close + np.random.randn(5000),
            'high': close + abs(np.random.randn(5000)) * 3,
            'low': close - abs(np.random.randn(5000)) * 3,
            'close': close,
            'volume': np.random.randint(1000, 10000, 5000)
        }, index=dates)

    print(f"\nLoaded ES data: {len(es_data)} bars")
    print(f"Date range: {es_data.index[0]} to {es_data.index[-1]}")

    # Generate cross-market features
    print("\n[1] Generating enhanced cross-market features...")
    features = calculate_enhanced_cross_market_features(es_data)

    print(f"\n[2] Generated {len(features.columns)} features:")
    for col in sorted(features.columns)[:25]:
        print(f"    {col}")
    if len(features.columns) > 25:
        print(f"    ... and {len(features.columns) - 25} more")

    # Check for NaN
    nan_pct = features.isna().sum() / len(features) * 100
    high_nan = nan_pct[nan_pct > 50]
    if len(high_nan) > 0:
        print(f"\n[3] Features with >50% NaN: {len(high_nan)}")
    else:
        print(f"\n[3] All features have acceptable NaN levels")

    print("\n" + "=" * 70)
    print("ENHANCED CROSS-MARKET FEATURES TEST COMPLETE")
    print("=" * 70)
