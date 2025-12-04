"""
Alternative Data Features (Category 10)
=======================================
Implements features from alternative data sources.

Sources:
- Reddit sentiment (wallstreetbets, stocks, futures)
- News headline sentiment
- CNN Fear & Greed Index
- Social media volume/engagement

Reference: research/02_comprehensive_variables_research.md
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Import alternative data collector
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_collection.alternative_data_collector import (
        AlternativeDataCollector,
        create_sample_alternative_features
    )
    ALT_DATA_AVAILABLE = True
except ImportError:
    ALT_DATA_AVAILABLE = False
    logger.warning("Alternative data collector not available")


class AlternativeFeatures:
    """Calculate features from alternative data sources."""

    def __init__(
        self,
        trading_df: pd.DataFrame,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        use_live_data: bool = True
    ):
        """
        Initialize alternative features calculator.

        Args:
            trading_df: OHLCV DataFrame with datetime index
            reddit_client_id: Reddit API client ID (optional)
            reddit_client_secret: Reddit API client secret (optional)
            use_live_data: If True, fetch live data; else use sample data
        """
        self.trading_df = trading_df.copy()
        self.reddit_client_id = reddit_client_id
        self.reddit_client_secret = reddit_client_secret
        self.use_live_data = use_live_data
        self._validate_data()

    def _validate_data(self):
        """Validate input data."""
        if not isinstance(self.trading_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

    def calculate_all(self) -> pd.DataFrame:
        """Calculate all alternative data features."""
        logger.info("Calculating alternative data features...")

        features = pd.DataFrame(index=self.trading_df.index)

        # Get alternative data
        if self.use_live_data and ALT_DATA_AVAILABLE:
            alt_data = self._fetch_live_data()
        else:
            alt_data = create_sample_alternative_features()

        # Create daily features (alternative data is typically daily)
        features = self._create_daily_features(alt_data)

        # Add derived features
        features = self._add_derived_features(features)

        logger.info(f"Generated {len(features.columns)} alternative data features")
        return features

    def _fetch_live_data(self) -> Dict[str, float]:
        """Fetch live alternative data."""
        try:
            collector = AlternativeDataCollector(
                reddit_client_id=self.reddit_client_id,
                reddit_client_secret=self.reddit_client_secret
            )
            # Use faster collection (exclude Reddit for intraday)
            return collector.collect_all(include_reddit=False)
        except Exception as e:
            logger.warning(f"Error fetching live data: {e}")
            return create_sample_alternative_features()

    def _create_daily_features(self, alt_data: Dict[str, float]) -> pd.DataFrame:
        """
        Align alternative data to trading timestamps.

        Alternative data is typically daily, so we forward-fill
        for intraday timestamps.
        """
        df = pd.DataFrame(index=self.trading_df.index)

        # Map alternative data to all timestamps
        for key, value in alt_data.items():
            if isinstance(value, (int, float)):
                df[f'alt_{key}'] = value

        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features from alternative data."""
        close = self.trading_df['close']
        returns = close.pct_change()

        # Fear & Greed regime features
        if 'alt_fear_greed_score' in df.columns:
            fg = df['alt_fear_greed_score'].iloc[0]

            # Fear/Greed zones
            df['alt_fg_extreme_fear_zone'] = (fg < 25).astype(int)
            df['alt_fg_fear_zone'] = ((fg >= 25) & (fg < 45)).astype(int)
            df['alt_fg_neutral_zone'] = ((fg >= 45) & (fg < 55)).astype(int)
            df['alt_fg_greed_zone'] = ((fg >= 55) & (fg < 75)).astype(int)
            df['alt_fg_extreme_greed_zone'] = (fg >= 75).astype(int)

            # Normalized score (0-1)
            df['alt_fg_normalized'] = fg / 100

        # Sentiment divergence (sentiment vs price)
        if 'alt_news_sentiment_compound' in df.columns:
            news_sent = df['alt_news_sentiment_compound'].iloc[0]

            # Price momentum (20-bar)
            price_mom = returns.rolling(20).mean()

            # Sentiment-price divergence
            # Positive divergence: bullish sentiment but price falling
            # Negative divergence: bearish sentiment but price rising
            df['alt_sentiment_price_divergence'] = np.where(
                price_mom > 0,
                -news_sent,  # If price rising, negative sentiment is bullish divergence
                news_sent    # If price falling, positive sentiment is bullish divergence
            )

        # Social volume relative to price volatility
        if 'alt_reddit_post_volume' in df.columns:
            vol = returns.rolling(20).std()
            post_vol = df['alt_reddit_post_volume'].iloc[0]

            # Normalize: high social volume during low price vol = unusual
            df['alt_social_vol_intensity'] = post_vol / (vol * 10000 + 1)

        # Composite sentiment score
        sentiment_cols = [
            col for col in df.columns
            if 'sentiment_compound' in col
        ]
        if sentiment_cols:
            df['alt_composite_sentiment'] = df[sentiment_cols].mean(axis=1)

        # Bullish/Bearish consensus
        bullish_cols = [col for col in df.columns if 'bullish_ratio' in col]
        bearish_cols = [col for col in df.columns if 'bearish_ratio' in col]

        if bullish_cols:
            df['alt_bullish_consensus'] = df[bullish_cols].mean(axis=1)
        if bearish_cols:
            df['alt_bearish_consensus'] = df[bearish_cols].mean(axis=1)

        return df


def calculate_alternative_features(
    trading_df: pd.DataFrame,
    use_live_data: bool = False,
    reddit_client_id: Optional[str] = None,
    reddit_client_secret: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to calculate all alternative data features.

    Args:
        trading_df: OHLCV DataFrame
        use_live_data: Whether to fetch live data
        reddit_client_id: Reddit API client ID
        reddit_client_secret: Reddit API secret

    Returns:
        DataFrame with alternative data features
    """
    calculator = AlternativeFeatures(
        trading_df,
        reddit_client_id=reddit_client_id,
        reddit_client_secret=reddit_client_secret,
        use_live_data=use_live_data
    )
    return calculator.calculate_all()


if __name__ == "__main__":
    # Test with sample data
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from data_collection.ninjatrader_loader import load_sample_data
    except ImportError:
        from src.python.data_collection.ninjatrader_loader import load_sample_data

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Alternative Data Features Test")
    print("=" * 60)

    es_data, _ = load_sample_data()

    # Test with sample data (faster)
    print("\n1. Testing with sample data...")
    features_sample = calculate_alternative_features(es_data, use_live_data=False)
    print(f"Generated {len(features_sample.columns)} features (sample)")

    # Test with live data
    print("\n2. Testing with live data...")
    features_live = calculate_alternative_features(es_data, use_live_data=True)
    print(f"Generated {len(features_live.columns)} features (live)")

    print(f"\nFeature names:")
    for i, col in enumerate(features_live.columns, 1):
        print(f"  {i:2d}. {col}")

    print(f"\nSample values (row 500):")
    for col in features_live.columns:
        val = features_live[col].iloc[500]
        if pd.notna(val):
            print(f"  {col}: {val:.4f}")
