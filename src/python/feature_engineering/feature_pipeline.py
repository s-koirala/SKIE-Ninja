"""
Feature Engineering Pipeline
============================
Main pipeline that orchestrates all feature calculations.

Combines features from all 14 categories:
1. Price-Based Features (~80) - ✅ Implemented
2. Technical Indicators (~105) - ✅ Implemented
3. Macroeconomic Variables (~15) - ✅ Implemented (FRED)
4. Microstructure Variables (~70) - ✅ Implemented
5. Sentiment & Positioning (~40) - ✅ Implemented (VIX, COT)
6. Intermarket Relationships (~95) - ✅ Implemented
7. Seasonality & Calendar (~60) - ✅ Implemented
8. Statistical Arbitrage (~20) - TODO
9. Regime & Fractal Features (~20) - ✅ Implemented (Hurst)
10. Alternative Data (~30) - ✅ Implemented (Reddit, News, Fear&Greed)
11. Lagged & Transformed (~70) - ✅ Implemented
12. Interaction Features (~10) - ✅ Implemented
13. Target Labels (~10) - ✅ Implemented
14. Advanced Targets (~115) - ✅ Implemented (Pyramiding, DDCA, S/R Pivots)

Current implementation: Categories 1-7, 9-14 (~700+ features)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
import logging
import warnings

warnings.filterwarnings('ignore')

# Import feature modules
try:
    from .price_features import PriceFeatures
    from .technical_indicators import TechnicalIndicators
    from .microstructure_features import MicrostructureFeatures
    from .sentiment_features import SentimentFeatures
    from .intermarket_features import IntermarketFeatures
    from .alternative_features import AlternativeFeatures
    from .advanced_targets import AdvancedTargets
except ImportError:
    from price_features import PriceFeatures
    from technical_indicators import TechnicalIndicators
    from microstructure_features import MicrostructureFeatures
    from sentiment_features import SentimentFeatures
    from intermarket_features import IntermarketFeatures
    from alternative_features import AlternativeFeatures
    from advanced_targets import AdvancedTargets

# Import FRED collector (optional - for macroeconomic features)
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_collection.fred_collector import FREDCollector, create_sample_macro_features
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Main feature engineering pipeline.

    Orchestrates calculation of all feature categories and combines
    them into a single feature matrix for ML training.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        symbol: str = "ES",
        include_lagged: bool = True,
        include_interactions: bool = True,
        include_targets: bool = True,
        include_macro: bool = True,
        include_sentiment: bool = True,
        include_intermarket: bool = True,
        include_alternative: bool = True,
        fred_api_key: Optional[str] = None,
        related_data: Optional[Dict] = None,
        use_live_alt_data: bool = False
    ):
        """
        Initialize the feature pipeline.

        Args:
            df: OHLCV DataFrame with datetime index
            symbol: Instrument symbol
            include_lagged: Whether to include lagged features
            include_interactions: Whether to include interaction features
            include_targets: Whether to include target labels
            include_macro: Whether to include macroeconomic features (FRED)
            include_sentiment: Whether to include sentiment features (VIX, COT)
            include_intermarket: Whether to include intermarket features
            include_alternative: Whether to include alternative data features
            fred_api_key: FRED API key (optional, uses sample data if None)
            related_data: Dict of related instrument DataFrames for intermarket features
            use_live_alt_data: Whether to fetch live alternative data (slower)
        """
        self.df = df.copy()
        self.symbol = symbol
        self.include_lagged = include_lagged
        self.include_interactions = include_interactions
        self.include_targets = include_targets
        self.include_macro = include_macro
        self.include_sentiment = include_sentiment
        self.include_intermarket = include_intermarket
        self.include_alternative = include_alternative
        self.fred_api_key = fred_api_key
        self.related_data = related_data
        self.use_live_alt_data = use_live_alt_data

        self._validate_data()

        logger.info(f"FeaturePipeline initialized for {symbol}")
        logger.info(f"Data shape: {self.df.shape}")
        logger.info(f"Date range: {self.df.index.min()} to {self.df.index.max()}")

    def _validate_data(self):
        """Validate input data."""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

    def calculate_all_features(self) -> pd.DataFrame:
        """
        Calculate all available features.

        Returns:
            DataFrame with all features
        """
        logger.info("Starting full feature calculation...")

        # Category 1: Price-Based Features
        logger.info("Calculating Category 1: Price-Based Features...")
        price_calc = PriceFeatures(self.df)
        price_features = price_calc.calculate_all()

        # Category 2: Technical Indicators
        logger.info("Calculating Category 2: Technical Indicators...")
        tech_calc = TechnicalIndicators(self.df)
        tech_features = tech_calc.calculate_all()

        # Category 7: Seasonality & Calendar
        logger.info("Calculating Category 7: Seasonality & Calendar...")
        calendar_features = self._calculate_calendar_features()

        # Category 9: Regime & Fractal Features (Hurst)
        logger.info("Calculating Category 9: Regime & Fractal Features...")
        regime_features = self._calculate_regime_features()

        # Category 3: Macroeconomic Features (FRED)
        macro_features = None
        if self.include_macro and FRED_AVAILABLE:
            logger.info("Calculating Category 3: Macroeconomic Features (FRED)...")
            macro_features = self._calculate_macro_features()

        # Category 4: Microstructure Features
        logger.info("Calculating Category 4: Microstructure Features...")
        micro_calc = MicrostructureFeatures(self.df)
        micro_features = micro_calc.calculate_all()

        # Category 5: Sentiment Features
        sentiment_features = None
        if self.include_sentiment:
            logger.info("Calculating Category 5: Sentiment Features...")
            sentiment_calc = SentimentFeatures(self.df, self.fred_api_key)
            sentiment_features = sentiment_calc.calculate_all()

        # Category 6: Intermarket Features
        intermarket_features = None
        if self.include_intermarket:
            logger.info("Calculating Category 6: Intermarket Features...")
            intermarket_calc = IntermarketFeatures(self.df, self.related_data, self.symbol)
            intermarket_features = intermarket_calc.calculate_all()

        # Category 10: Alternative Data Features
        alternative_features = None
        if self.include_alternative:
            logger.info("Calculating Category 10: Alternative Data Features...")
            alt_calc = AlternativeFeatures(
                self.df,
                use_live_data=self.use_live_alt_data
            )
            alternative_features = alt_calc.calculate_all()

        # Category 14: Advanced Targets (Pyramiding, DDCA, S/R)
        logger.info("Calculating Advanced Targets (Pyramiding, DDCA, S/R)...")
        advanced_calc = AdvancedTargets(self.df)
        advanced_features = advanced_calc.calculate_all()

        # Combine all features
        feature_list = [
            price_features,
            tech_features,
            calendar_features,
            regime_features,
            micro_features
        ]
        if macro_features is not None and len(macro_features) > 0:
            feature_list.append(macro_features)
        if sentiment_features is not None and len(sentiment_features) > 0:
            feature_list.append(sentiment_features)
        if intermarket_features is not None and len(intermarket_features) > 0:
            feature_list.append(intermarket_features)
        if alternative_features is not None and len(alternative_features) > 0:
            feature_list.append(alternative_features)
        if advanced_features is not None and len(advanced_features) > 0:
            feature_list.append(advanced_features)

        features = pd.concat(feature_list, axis=1)

        # Category 11: Lagged Features
        if self.include_lagged:
            logger.info("Calculating Category 11: Lagged Features...")
            lagged_features = self._calculate_lagged_features(features)
            features = pd.concat([features, lagged_features], axis=1)

        # Category 12: Interaction Features
        if self.include_interactions:
            logger.info("Calculating Category 12: Interaction Features...")
            interaction_features = self._calculate_interaction_features(features)
            features = pd.concat([features, interaction_features], axis=1)

        # Category 13: Target Labels
        if self.include_targets:
            logger.info("Calculating Category 13: Target Labels...")
            target_features = self._calculate_target_labels()
            features = pd.concat([features, target_features], axis=1)

        # Clean up
        features = self._clean_features(features)

        logger.info(f"Total features generated: {len(features.columns)}")
        logger.info(f"Features with NaN: {features.isnull().any().sum()}")

        return features

    def _calculate_calendar_features(self) -> pd.DataFrame:
        """
        Calculate seasonality and calendar features (Category 7).

        Includes cyclical encoding for time-based features.
        """
        df = pd.DataFrame(index=self.df.index)

        # Extract time components
        df['hour'] = self.df.index.hour
        df['minute'] = self.df.index.minute
        df['day_of_week'] = self.df.index.dayofweek
        df['day_of_month'] = self.df.index.day
        df['month'] = self.df.index.month
        df['quarter'] = self.df.index.quarter
        df['week_of_year'] = self.df.index.isocalendar().week.astype(int)

        # Cyclical encoding (sin/cos) for periodic features
        # Hour of day (0-23)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week (0-6)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Month (1-12)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Calendar effects
        # Turn of month (last day + first 3 days)
        df['turn_of_month'] = ((df['day_of_month'] <= 3) | (df['day_of_month'] >= 28)).astype(int)

        # First half vs second half of month
        df['first_half_month'] = (df['day_of_month'] <= 15).astype(int)

        # Trading session markers (for ES futures)
        # Regular trading hours: 9:30 AM - 4:00 PM CT
        # Overnight session: 6:00 PM - 9:30 AM CT
        hour = df['hour']
        minute = df['minute']

        # RTH (Regular Trading Hours)
        df['is_rth'] = (
            ((hour == 8) & (minute >= 30)) |
            ((hour >= 9) & (hour < 15)) |
            ((hour == 15) & (minute == 0))
        ).astype(int)

        # Opening hour (first hour of RTH)
        df['is_opening_hour'] = ((hour == 8) | (hour == 9)).astype(int)

        # Closing hour (last hour of RTH)
        df['is_closing_hour'] = ((hour == 14) | (hour == 15)).astype(int)

        # Drop raw components, keep encoded versions
        df = df.drop(['hour', 'minute', 'day_of_week', 'day_of_month', 'month'], axis=1)

        return df

    def _calculate_macro_features(self) -> pd.DataFrame:
        """
        Calculate macroeconomic features from FRED (Category 3).

        Uses FRED API if key available, otherwise generates sample data.
        """
        try:
            if self.fred_api_key:
                # Use real FRED data
                collector = FREDCollector(api_key=self.fred_api_key)

                # Get date range from trading data
                start_date = self.df.index.min().strftime('%Y-%m-%d')
                end_date = self.df.index.max().strftime('%Y-%m-%d')

                # Fetch key series
                raw_data = collector.get_all_series(
                    start_date=start_date,
                    end_date=end_date
                )

                if len(raw_data) > 0:
                    macro_features = collector.calculate_features(raw_data)
                else:
                    logger.warning("No FRED data retrieved, using sample data")
                    macro_features = create_sample_macro_features()
            else:
                # Use sample data for development
                logger.info("Using sample macroeconomic data (no API key)")
                macro_features = create_sample_macro_features()

            # Align macro data to trading data timestamps
            # Get unique dates from trading data
            if hasattr(self.df.index, 'tz'):
                trading_dates = self.df.index.tz_localize(None).normalize()
            else:
                trading_dates = pd.to_datetime(self.df.index).normalize()

            # Create a mapping from date to macro features
            macro_features.index = pd.to_datetime(macro_features.index)

            # For each trading timestamp, get the macro features for that date
            aligned = pd.DataFrame(index=self.df.index)
            for col in macro_features.columns:
                # Create date-to-value mapping
                date_values = macro_features[col].to_dict()
                # Map trading dates to values
                aligned[col] = trading_dates.map(lambda d: date_values.get(d.normalize(), np.nan))

            # Forward fill any missing values
            aligned = aligned.ffill()

            logger.info(f"Generated {len(aligned.columns)} macroeconomic features")
            return aligned

        except Exception as e:
            logger.error(f"Error calculating macro features: {e}")
            return pd.DataFrame(index=self.df.index)

    def _calculate_regime_features(self) -> pd.DataFrame:
        """
        Calculate regime and fractal features (Category 9).

        Includes Hurst exponent for trend/mean-reversion detection.
        """
        df = pd.DataFrame(index=self.df.index)
        close = self.df['close']

        # Hurst exponent calculation
        for window in [20, 50, 100]:
            df[f'hurst_{window}'] = close.rolling(window).apply(
                self._calculate_hurst, raw=False
            )

        # Regime classification based on Hurst
        # H > 0.5: Trending, H < 0.5: Mean-reverting, H ≈ 0.5: Random
        df['regime_trending'] = (df['hurst_50'] > 0.55).astype(int)
        df['regime_mean_reverting'] = (df['hurst_50'] < 0.45).astype(int)
        df['regime_random'] = ((df['hurst_50'] >= 0.45) & (df['hurst_50'] <= 0.55)).astype(int)

        # Volatility regime
        returns = close.pct_change()
        vol_20 = returns.rolling(20).std()
        vol_50 = returns.rolling(50).std()

        df['vol_regime_high'] = (vol_20 > vol_50 * 1.5).astype(int)
        df['vol_regime_low'] = (vol_20 < vol_50 * 0.75).astype(int)

        return df

    def _calculate_hurst(self, series: pd.Series) -> float:
        """
        Calculate Hurst exponent using R/S analysis.

        H > 0.5: Persistent (trending)
        H < 0.5: Anti-persistent (mean-reverting)
        H = 0.5: Random walk
        """
        try:
            n = len(series)
            if n < 20:
                return np.nan

            # Convert to numpy array
            ts = series.values

            # Calculate returns
            returns = np.diff(ts) / ts[:-1]

            # Use simplified R/S method
            max_k = min(n // 2, 100)
            if max_k < 8:
                return np.nan

            rs_list = []
            n_list = []

            for k in range(8, max_k + 1, 4):
                # Calculate mean and std
                mean = np.mean(returns[:k])
                std = np.std(returns[:k], ddof=1)

                if std == 0:
                    continue

                # Cumulative deviation from mean
                cum_dev = np.cumsum(returns[:k] - mean)

                # Range
                r = np.max(cum_dev) - np.min(cum_dev)

                # R/S statistic
                rs = r / std

                rs_list.append(rs)
                n_list.append(k)

            if len(rs_list) < 3:
                return np.nan

            # Linear regression of log(R/S) vs log(n)
            log_rs = np.log(rs_list)
            log_n = np.log(n_list)

            # Slope is Hurst exponent
            hurst = np.polyfit(log_n, log_rs, 1)[0]

            # Bound to reasonable range
            return np.clip(hurst, 0.0, 1.0)

        except Exception:
            return np.nan

    def _calculate_lagged_features(
        self,
        features: pd.DataFrame,
        lags: List[int] = None,
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Calculate lagged features (Category 11).

        Args:
            features: DataFrame with base features
            lags: List of lag periods
            columns: Columns to lag (defaults to key indicators)
        """
        if lags is None:
            lags = [1, 2, 3, 5, 10]

        if columns is None:
            # Select key features for lagging
            columns = [
                'return_1', 'return_5',
                'rsi_14', 'macd', 'macd_hist',
                'adx_14', 'bb_pct_b_20',
                'rel_volume_20', 'hurst_50'
            ]
            # Filter to existing columns
            columns = [c for c in columns if c in features.columns]

        lagged = pd.DataFrame(index=features.index)

        for col in columns:
            for lag in lags:
                lagged[f'{col}_lag_{lag}'] = features[col].shift(lag)

        # Rolling statistics on returns
        if 'return_1' in features.columns:
            returns = features['return_1']
            for window in [10, 20]:
                lagged[f'return_rolling_mean_{window}'] = returns.rolling(window).mean()
                lagged[f'return_rolling_std_{window}'] = returns.rolling(window).std()
                lagged[f'return_rolling_min_{window}'] = returns.rolling(window).min()
                lagged[f'return_rolling_max_{window}'] = returns.rolling(window).max()

                # Percentile rank
                lagged[f'return_rank_{window}'] = returns.rolling(window).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
                )

        return lagged

    def _calculate_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate interaction features (Category 12).

        Cross-feature products and conditional features.
        """
        df = pd.DataFrame(index=features.index)

        # Volume × Momentum interactions
        if 'rel_volume_20' in features.columns and 'return_1' in features.columns:
            df['volume_x_return'] = features['rel_volume_20'] * features['return_1']

        if 'rel_volume_20' in features.columns and 'rsi_14' in features.columns:
            df['volume_x_rsi'] = features['rel_volume_20'] * (features['rsi_14'] - 50) / 50

        # Volatility × Regime interactions
        if 'atr_14' in features.columns and 'hurst_50' in features.columns:
            df['atr_x_hurst'] = features['atr_14'] * features['hurst_50']

        # Time × Returns interactions
        if 'hour_sin' in features.columns and 'return_1' in features.columns:
            df['time_x_return'] = features['hour_sin'] * features['return_1']

        # Conditional features
        # Momentum strength when trending
        if 'regime_trending' in features.columns and 'return_1' in features.columns:
            df['momentum_if_trending'] = features['return_1'] * features['regime_trending']

        if 'regime_mean_reverting' in features.columns and 'return_1' in features.columns:
            df['momentum_if_reverting'] = features['return_1'] * features['regime_mean_reverting']

        # RSI extremes × Volume
        if 'rsi_oversold_14' in features.columns and 'rel_volume_20' in features.columns:
            df['oversold_x_volume'] = features['rsi_oversold_14'] * features['rel_volume_20']

        if 'rsi_overbought_14' in features.columns and 'rel_volume_20' in features.columns:
            df['overbought_x_volume'] = features['rsi_overbought_14'] * features['rel_volume_20']

        return df

    def _calculate_target_labels(self) -> pd.DataFrame:
        """
        Calculate target labels for supervised learning (Category 13).

        Includes classification and regression targets.
        """
        df = pd.DataFrame(index=self.df.index)
        close = self.df['close']

        # Classification labels
        # Direction next bar (binary)
        df['target_direction_1'] = (close.shift(-1) > close).astype(int)

        # Direction next 5 bars
        df['target_direction_5'] = (close.shift(-5) > close).astype(int)

        # 3-class: Strong up (>0.5%), Flat, Strong down (<-0.5%)
        future_return_5 = close.shift(-5) / close - 1
        df['target_class_3'] = pd.cut(
            future_return_5,
            bins=[-np.inf, -0.005, 0.005, np.inf],
            labels=[0, 1, 2]  # Down, Flat, Up
        ).astype(float)

        # Regression labels
        # Forward returns
        for period in [1, 5, 10, 20]:
            df[f'target_return_{period}'] = close.shift(-period) / close - 1

        # Maximum Favorable Excursion (MFE) - max gain in next N bars
        # Maximum Adverse Excursion (MAE) - max loss in next N bars
        for period in [5, 10]:
            high = self.df['high']
            low = self.df['low']

            # MFE: max((future highs - current close) / current close)
            future_highs = high.rolling(period).max().shift(-period)
            df[f'target_mfe_{period}'] = (future_highs - close) / close

            # MAE: min((future lows - current close) / current close)
            future_lows = low.rolling(period).min().shift(-period)
            df[f'target_mae_{period}'] = (future_lows - close) / close

        return df

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare final feature matrix."""
        # Remove duplicate columns
        features = features.loc[:, ~features.columns.duplicated()]

        # Replace infinite values with NaN
        features = features.replace([np.inf, -np.inf], np.nan)

        # Sort columns alphabetically for consistency
        features = features.reindex(sorted(features.columns), axis=1)

        return features

    def get_feature_summary(self, features: pd.DataFrame) -> Dict:
        """Get summary statistics for generated features."""
        summary = {
            'total_features': len(features.columns),
            'total_rows': len(features),
            'features_with_nan': features.isnull().any().sum(),
            'nan_percentages': features.isnull().mean().sort_values(ascending=False).head(10).to_dict(),
            'feature_groups': {
                'price_based': len([c for c in features.columns if any(
                    x in c for x in ['open', 'high', 'low', 'close', 'return', 'range', 'body']
                )]),
                'technical': len([c for c in features.columns if any(
                    x in c for x in ['sma', 'ema', 'rsi', 'macd', 'adx', 'bb_', 'stoch', 'cci', 'obv', 'cmf', 'mfi']
                )]),
                'calendar': len([c for c in features.columns if any(
                    x in c for x in ['hour', 'dow', 'month', 'quarter', 'week', 'turn', 'rth', 'opening', 'closing']
                )]),
                'regime': len([c for c in features.columns if any(
                    x in c for x in ['hurst', 'regime', 'vol_regime']
                )]),
                'macro': len([c for c in features.columns if 'fred_' in c]),
                'microstructure': len([c for c in features.columns if any(
                    x in c for x in ['volume_', 'vwap', 'amihud', 'kyle', 'ofi', 'trade_', 'session_', 'tick_', 'dollar_', 'buy_', 'sell_']
                )]),
                'sentiment': len([c for c in features.columns if any(
                    x in c for x in ['vix', 'cot_', 'fear', 'greed', 'put_call', 'ad_line', 'mcclellan', 'breadth']
                )]),
                'intermarket': len([c for c in features.columns if any(
                    x in c for x in ['corr_', 'lead_', 'rs_', 'spread', 'risk_on', 'sb_corr']
                )]),
                'alternative': len([c for c in features.columns if 'alt_' in c]),
                'advanced_targets': len([c for c in features.columns if any(
                    x in c for x in ['pyramid_', 'ddca_', 'pivot_', 'sr_', 'near_support', 'near_resistance', 'range_position']
                )]),
                'lagged': len([c for c in features.columns if 'lag_' in c or 'rolling' in c or 'rank' in c]),
                'interaction': len([c for c in features.columns if '_x_' in c or '_if_' in c]),
                'target': len([c for c in features.columns if 'target_' in c]),
            }
        }
        return summary


def build_feature_matrix(
    df: pd.DataFrame,
    symbol: str = "ES",
    include_lagged: bool = True,
    include_interactions: bool = True,
    include_targets: bool = True,
    include_macro: bool = True,
    include_sentiment: bool = True,
    include_intermarket: bool = True,
    include_alternative: bool = True,
    fred_api_key: Optional[str] = None,
    related_data: Optional[Dict] = None,
    use_live_alt_data: bool = False,
    dropna: bool = True
) -> pd.DataFrame:
    """
    Build complete feature matrix from OHLCV data.

    Args:
        df: OHLCV DataFrame with datetime index
        symbol: Instrument symbol
        include_lagged: Include lagged features
        include_interactions: Include interaction features
        include_targets: Include target labels
        include_macro: Include macroeconomic features from FRED
        include_sentiment: Include sentiment features (VIX, COT)
        include_intermarket: Include intermarket features
        include_alternative: Include alternative data features
        fred_api_key: FRED API key (uses sample data if None)
        related_data: Dict of related instrument DataFrames
        use_live_alt_data: Fetch live alternative data (slower)
        dropna: Drop rows with NaN values

    Returns:
        DataFrame with all features
    """
    pipeline = FeaturePipeline(
        df,
        symbol=symbol,
        include_lagged=include_lagged,
        include_interactions=include_interactions,
        include_targets=include_targets,
        include_macro=include_macro,
        include_sentiment=include_sentiment,
        include_intermarket=include_intermarket,
        include_alternative=include_alternative,
        fred_api_key=fred_api_key,
        related_data=related_data,
        use_live_alt_data=use_live_alt_data
    )

    features = pipeline.calculate_all_features()

    if dropna:
        before = len(features)
        features = features.dropna()
        after = len(features)
        logger.info(f"Dropped {before - after} rows with NaN ({(before-after)/before*100:.1f}%)")

    return features


if __name__ == "__main__":
    # Test the full pipeline
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_collection.ninjatrader_loader import load_sample_data

    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("Feature Pipeline Test")
    print("=" * 70)

    # Load sample data
    es_data, _ = load_sample_data()

    # Build feature matrix
    features = build_feature_matrix(es_data, symbol="ES", dropna=True)

    print(f"\n{'='*70}")
    print("FEATURE MATRIX SUMMARY")
    print("=" * 70)
    print(f"Shape: {features.shape}")

    # Get summary
    pipeline = FeaturePipeline(es_data)
    summary = pipeline.get_feature_summary(features)

    print(f"\nTotal Features: {summary['total_features']}")
    print(f"\nFeatures by Category:")
    for group, count in summary['feature_groups'].items():
        print(f"  {group}: {count}")

    print(f"\nSample Features (first 10):")
    print(features.columns.tolist()[:10])

    print(f"\nSample Data (last 3 rows, first 10 columns):")
    print(features.iloc[-3:, :10])

    # Save feature list
    feature_list_path = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "feature_list.txt"
    with open(feature_list_path, 'w') as f:
        f.write(f"SKIE_Ninja Feature List\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Total Features: {len(features.columns)}\n")
        f.write(f"{'='*50}\n\n")
        for col in sorted(features.columns):
            f.write(f"{col}\n")

    print(f"\nFeature list saved to: {feature_list_path}")
    print("\n" + "=" * 70)
    print("SUCCESS: Feature pipeline working correctly!")
    print("=" * 70)
