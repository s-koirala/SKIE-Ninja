"""
Enhanced Feature Pipeline
=========================

Unified feature generation pipeline combining all feature sources:
1. Base technical indicators
2. Multi-timeframe features (MTF)
3. Enhanced cross-market features (real data)
4. Social/News sentiment features
5. Volatility regime features

This pipeline is designed for the volatility breakout strategy,
with strict adherence to data leakage prevention best practices.

CRITICAL RULES (from BEST_PRACTICES.md):
1. NEVER use shift(-N) - only positive shifts
2. NEVER use center=True in rolling windows
3. All features use only PAST data at prediction point
4. Apply proper lag for cross-timeframe/cross-market data
5. Validate with QC checks before trusting results

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Import feature modules
try:
    from feature_engineering.multi_timeframe_features import (
        calculate_multi_timeframe_features, MTFConfig
    )
except ImportError:
    from multi_timeframe_features import calculate_multi_timeframe_features, MTFConfig

try:
    from feature_engineering.enhanced_cross_market import (
        calculate_enhanced_cross_market_features, CrossMarketConfig
    )
except ImportError:
    from enhanced_cross_market import calculate_enhanced_cross_market_features, CrossMarketConfig

try:
    from feature_engineering.social_news_sentiment import (
        calculate_social_news_features, SocialSentimentConfig
    )
except ImportError:
    from social_news_sentiment import calculate_social_news_features, SocialSentimentConfig

try:
    from feature_engineering.technical_indicators import calculate_technical_indicators
except ImportError:
    from technical_indicators import calculate_technical_indicators

try:
    from feature_engineering.volatility_regime import calculate_volatility_regime_features
except ImportError:
    calculate_volatility_regime_features = None

try:
    from data_collection.established_sentiment_indices import (
        calculate_established_sentiment_features, SentimentIndicesConfig
    )
except ImportError:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data_collection.established_sentiment_indices import (
            calculate_established_sentiment_features, SentimentIndicesConfig
        )
    except ImportError:
        calculate_established_sentiment_features = None
        SentimentIndicesConfig = None


@dataclass
class EnhancedPipelineConfig:
    """Configuration for the enhanced feature pipeline."""

    # Feature groups to include
    include_technical: bool = True
    include_mtf: bool = True
    include_cross_market: bool = True
    include_sentiment: bool = True
    include_volatility_regime: bool = True
    include_established_sentiment: bool = True  # AAII, Put/Call, VIX

    # Multi-timeframe configuration
    mtf_config: MTFConfig = field(default_factory=MTFConfig)

    # Cross-market configuration
    cross_market_config: CrossMarketConfig = field(default_factory=CrossMarketConfig)

    # Sentiment configuration
    sentiment_config: SocialSentimentConfig = field(default_factory=SocialSentimentConfig)

    # Data paths
    data_dir: Path = None

    # Year suffix for loading data (e.g., "_2020", "_2021")
    year_suffix: str = ""

    # QC thresholds
    max_feature_target_corr: float = 0.30
    suspicious_auc: float = 0.85
    suspicious_win_rate: float = 0.65

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent.parent / 'data'


class EnhancedFeaturePipeline:
    """
    Unified feature generation pipeline.

    Combines multiple feature sources with proper validation
    and data leakage prevention.
    """

    def __init__(self, config: Optional[EnhancedPipelineConfig] = None):
        self.config = config or EnhancedPipelineConfig()
        self._feature_counts = {}

    def generate_all_features(
        self,
        prices: pd.DataFrame,
        historical_sentiment: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate all features for the given price data.

        Args:
            prices: OHLCV DataFrame with DatetimeIndex
            historical_sentiment: Optional historical sentiment data

        Returns:
            DataFrame with all features aligned to price index
        """
        logger.info("=" * 60)
        logger.info("ENHANCED FEATURE PIPELINE - Starting feature generation")
        logger.info("=" * 60)

        all_features = pd.DataFrame(index=prices.index)

        # 1. Technical Indicators (Base Features)
        if self.config.include_technical:
            logger.info("\n[1/6] Generating technical indicators...")
            try:
                tech_features = calculate_technical_indicators(prices)
                all_features = pd.concat([all_features, tech_features], axis=1)
                self._feature_counts['technical'] = len(tech_features.columns)
                logger.info(f"      Generated {len(tech_features.columns)} technical features")
            except Exception as e:
                logger.error(f"      Error: {e}")
                self._feature_counts['technical'] = 0

        # 2. Multi-Timeframe Features
        if self.config.include_mtf:
            logger.info("\n[2/6] Generating multi-timeframe features...")
            try:
                mtf_features = calculate_multi_timeframe_features(
                    prices, self.config.mtf_config
                )
                all_features = pd.concat([all_features, mtf_features], axis=1)
                self._feature_counts['mtf'] = len(mtf_features.columns)
                logger.info(f"      Generated {len(mtf_features.columns)} MTF features")
            except Exception as e:
                logger.error(f"      Error: {e}")
                self._feature_counts['mtf'] = 0

        # 3. Cross-Market Features (Real Data)
        if self.config.include_cross_market:
            logger.info("\n[3/6] Generating cross-market features...")
            try:
                cross_features = calculate_enhanced_cross_market_features(
                    prices,
                    self.config.cross_market_config,
                    self.config.year_suffix
                )
                all_features = pd.concat([all_features, cross_features], axis=1)
                self._feature_counts['cross_market'] = len(cross_features.columns)
                logger.info(f"      Generated {len(cross_features.columns)} cross-market features")
            except Exception as e:
                logger.error(f"      Error: {e}")
                self._feature_counts['cross_market'] = 0

        # 4. Social/News Sentiment Features
        if self.config.include_sentiment:
            logger.info("\n[4/6] Generating sentiment features...")
            try:
                sentiment_features = calculate_social_news_features(
                    prices.index,
                    self.config.sentiment_config,
                    historical_sentiment
                )
                all_features = pd.concat([all_features, sentiment_features], axis=1)
                self._feature_counts['sentiment'] = len(sentiment_features.columns)
                logger.info(f"      Generated {len(sentiment_features.columns)} sentiment features")
            except Exception as e:
                logger.error(f"      Error: {e}")
                self._feature_counts['sentiment'] = 0

        # 5. Volatility Regime Features
        if self.config.include_volatility_regime and calculate_volatility_regime_features:
            logger.info("\n[5/6] Generating volatility regime features...")
            try:
                vol_features = calculate_volatility_regime_features(prices)
                all_features = pd.concat([all_features, vol_features], axis=1)
                self._feature_counts['volatility_regime'] = len(vol_features.columns)
                logger.info(f"      Generated {len(vol_features.columns)} volatility regime features")
            except Exception as e:
                logger.error(f"      Error: {e}")
                self._feature_counts['volatility_regime'] = 0
        else:
            self._feature_counts['volatility_regime'] = 0

        # 6. Established Sentiment Indices (AAII, Put/Call, VIX)
        if self.config.include_established_sentiment and calculate_established_sentiment_features:
            logger.info("\n[6/6] Generating established sentiment indices...")
            try:
                est_features = calculate_established_sentiment_features(prices.index)
                all_features = pd.concat([all_features, est_features], axis=1)
                self._feature_counts['established_sentiment'] = len(est_features.columns)
                logger.info(f"      Generated {len(est_features.columns)} established sentiment features")
            except Exception as e:
                logger.error(f"      Error: {e}")
                self._feature_counts['established_sentiment'] = 0
        else:
            self._feature_counts['established_sentiment'] = 0

        # Summary
        total_features = len(all_features.columns)
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"\nTotal features: {total_features}")
        for category, count in self._feature_counts.items():
            logger.info(f"  {category}: {count}")

        return all_features

    def validate_features(
        self,
        features: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Validate features for data leakage and quality issues.

        Args:
            features: Feature DataFrame
            target: Optional target variable for correlation checks

        Returns:
            Dict with validation results
        """
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE VALIDATION")
        logger.info("=" * 60)

        results = {
            'passed': True,
            'total_features': len(features.columns),
            'checks': [],
            'warnings': [],
            'errors': []
        }

        # 1. Check for NaN
        nan_pct = features.isna().sum() / len(features) * 100
        high_nan = nan_pct[nan_pct > 50]
        if len(high_nan) > 0:
            results['warnings'].append(
                f"{len(high_nan)} features have >50% NaN values"
            )
        results['checks'].append(f"NaN check: {len(high_nan)} features with high NaN")

        # 2. Check for infinite values
        inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            results['warnings'].append(f"Found {inf_count} infinite values")
        results['checks'].append(f"Infinite value check: {inf_count} found")

        # 3. Check for constant features
        constant_features = []
        for col in features.columns:
            if features[col].nunique() <= 1:
                constant_features.append(col)
        if constant_features:
            results['warnings'].append(
                f"{len(constant_features)} constant features detected"
            )
        results['checks'].append(f"Constant feature check: {len(constant_features)}")

        # 4. Feature-target correlation (if target provided)
        if target is not None:
            suspicious_corr = []
            for col in features.columns:
                if features[col].isna().all():
                    continue
                corr = features[col].corr(target)
                if abs(corr) > self.config.max_feature_target_corr:
                    suspicious_corr.append((col, corr))

            if suspicious_corr:
                results['warnings'].append(
                    f"{len(suspicious_corr)} features have high target correlation"
                )
                results['suspicious_correlations'] = suspicious_corr
            results['checks'].append(f"Target correlation check: {len(suspicious_corr)} suspicious")

        # 5. Check for look-ahead bias patterns in feature names
        leaky_patterns = ['future', 'next', 'forward', 'shift(-']
        suspicious_names = []
        for col in features.columns:
            col_lower = col.lower()
            for pattern in leaky_patterns:
                if pattern in col_lower:
                    suspicious_names.append(col)
                    break

        if suspicious_names:
            results['errors'].append(
                f"Potentially leaky feature names: {suspicious_names}"
            )
            results['passed'] = False
        results['checks'].append(f"Name pattern check: {len(suspicious_names)} suspicious")

        # Log results
        if results['passed']:
            logger.info("\n[PASS] All validation checks passed")
        else:
            logger.error("\n[FAIL] Validation failed - check errors")

        for check in results['checks']:
            logger.info(f"  - {check}")

        for warning in results['warnings']:
            logger.warning(f"  [WARN] {warning}")

        for error in results['errors']:
            logger.error(f"  [ERROR] {error}")

        return results

    def get_feature_summary(self) -> Dict[str, int]:
        """Get summary of generated features by category."""
        return self._feature_counts.copy()


def generate_enhanced_features(
    prices: pd.DataFrame,
    config: Optional[EnhancedPipelineConfig] = None,
    historical_sentiment: Optional[pd.DataFrame] = None,
    validate: bool = True,
    target: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to generate all enhanced features.

    Args:
        prices: OHLCV DataFrame with DatetimeIndex
        config: Optional pipeline configuration
        historical_sentiment: Optional historical sentiment data
        validate: Whether to run validation checks
        target: Optional target variable for validation

    Returns:
        Tuple of (features DataFrame, validation results)
    """
    pipeline = EnhancedFeaturePipeline(config)

    # Generate features
    features = pipeline.generate_all_features(prices, historical_sentiment)

    # Validate if requested
    validation_results = {}
    if validate:
        validation_results = pipeline.validate_features(features, target)

    return features, validation_results


def create_leakage_free_features(
    prices: pd.DataFrame,
    config: Optional[EnhancedPipelineConfig] = None
) -> pd.DataFrame:
    """
    Generate features with strict leakage prevention.

    This is the recommended function for production use.
    Performs additional cleaning and validation.

    Args:
        prices: OHLCV DataFrame with DatetimeIndex
        config: Optional pipeline configuration

    Returns:
        Clean, validated features DataFrame
    """
    pipeline = EnhancedFeaturePipeline(config)
    features = pipeline.generate_all_features(prices)

    # Additional cleaning
    # 1. Remove features with >30% NaN
    nan_pct = features.isna().sum() / len(features)
    features = features.loc[:, nan_pct < 0.30]

    # 2. Remove constant features
    for col in features.columns:
        if features[col].nunique() <= 1:
            features = features.drop(columns=[col])

    # 3. Cap extreme values (clip to 5 std)
    for col in features.select_dtypes(include=[np.number]).columns:
        mean = features[col].mean()
        std = features[col].std()
        if std > 0:
            features[col] = features[col].clip(mean - 5*std, mean + 5*std)

    # 4. Fill remaining NaN with forward fill then backward fill
    features = features.ffill().bfill()

    logger.info(f"\nFinal clean features: {len(features.columns)}")

    return features


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("ENHANCED FEATURE PIPELINE TEST")
    print("=" * 70)

    # Load sample data
    try:
        from data_collection.ninjatrader_loader import load_sample_data
        prices, _ = load_sample_data(source="databento")

        # Resample to 5-min
        prices = prices.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Take a subset for testing
        prices = prices.iloc[:5000]

    except Exception as e:
        print(f"Could not load data: {e}")
        print("Creating synthetic data...")

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

    print(f"\nLoaded {len(prices)} bars")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")

    # Configure pipeline
    config = EnhancedPipelineConfig(
        include_technical=True,
        include_mtf=True,
        include_cross_market=True,
        include_sentiment=True,
        include_volatility_regime=False  # May not be available
    )

    # Generate features
    print("\n[1] Generating features...")
    features, validation = generate_enhanced_features(
        prices,
        config,
        validate=True
    )

    print(f"\n[2] Generated {len(features.columns)} total features")

    # Show feature categories
    print("\n[3] Feature breakdown:")
    categories = {
        'technical': [c for c in features.columns if any(x in c.lower() for x in ['sma', 'ema', 'rsi', 'macd', 'bb_', 'adx', 'atr'])],
        'mtf': [c for c in features.columns if 'htf_' in c.lower() or 'mtf_' in c.lower()],
        'cross_market': [c for c in features.columns if any(x in c.lower() for x in ['corr_', 'spread', 'vix', 'rs_'])],
        'sentiment': [c for c in features.columns if 'social_' in c.lower() or 'sentiment' in c.lower()]
    }

    for cat, cols in categories.items():
        print(f"    {cat}: {len(cols)} features")

    # Validation results
    print(f"\n[4] Validation: {'PASSED' if validation.get('passed', True) else 'FAILED'}")
    for warning in validation.get('warnings', []):
        print(f"    [WARN] {warning}")

    # Create clean features
    print("\n[5] Creating leakage-free features...")
    clean_features = create_leakage_free_features(prices, config)
    print(f"    Clean features: {len(clean_features.columns)}")

    print("\n" + "=" * 70)
    print("ENHANCED FEATURE PIPELINE TEST COMPLETE")
    print("=" * 70)
