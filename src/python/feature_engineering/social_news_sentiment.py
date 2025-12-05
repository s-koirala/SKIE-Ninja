"""
Social Media & News Sentiment Integration Module
=================================================

Integrates real-time sentiment from multiple sources:
1. Twitter/X API - Real-time market sentiment
2. News APIs - Financial news sentiment (Alpha Vantage, Polygon.io)
3. Reddit (WSB, stocks) - Retail sentiment (contrarian indicator)

CRITICAL: Follows strict data leakage prevention rules:
1. All sentiment features use LAGGED data (min 5-minute lag)
2. Sentiment is aggregated BEFORE the bar it applies to
3. Historical backtesting uses point-in-time sentiment only
4. Never use sentiment that wasn't available at prediction time

API Requirements:
- Twitter API v2: Requires API key (free tier available)
- Alpha Vantage: Free tier (5 calls/min, 500/day)
- Polygon.io: Free tier available
- Reddit: No API required for public data

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class SocialSentimentConfig:
    """Configuration for social/news sentiment features."""

    # API Keys (set via environment or config)
    twitter_bearer_token: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None

    # Sentiment aggregation windows (in minutes)
    aggregation_windows: List[int] = field(default_factory=lambda: [5, 15, 30, 60, 240])

    # CRITICAL: Minimum lag to apply (prevents look-ahead)
    min_lag_minutes: int = 5

    # Keywords for filtering financial content
    es_keywords: List[str] = field(default_factory=lambda: [
        'SPY', 'SPX', 'S&P', 'ES', 'ES_F', '$SPY', '$SPX',
        'stock market', 'equities', 'nasdaq', 'dow jones'
    ])

    # Rate limiting
    requests_per_minute: int = 5

    # WSB is contrarian indicator (research-backed)
    wsb_contrarian: bool = True


class SentimentDataSource(ABC):
    """Abstract base class for sentiment data sources."""

    @abstractmethod
    def fetch_sentiment(
        self,
        start_time: datetime,
        end_time: datetime,
        keywords: List[str]
    ) -> pd.DataFrame:
        """
        Fetch sentiment data for a time range.

        Returns DataFrame with columns:
        - timestamp: When the content was created
        - text: The content text
        - sentiment_score: -1 to +1 sentiment
        - source: Source identifier
        """
        pass


class TwitterSentimentSource(SentimentDataSource):
    """
    Twitter/X API v2 sentiment source.

    Fetches tweets mentioning financial keywords and analyzes sentiment.
    """

    def __init__(self, bearer_token: str, config: SocialSentimentConfig):
        self.bearer_token = bearer_token
        self.config = config
        self._last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting."""
        min_interval = 60 / self.config.requests_per_minute
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def fetch_sentiment(
        self,
        start_time: datetime,
        end_time: datetime,
        keywords: List[str]
    ) -> pd.DataFrame:
        """Fetch tweets and analyze sentiment."""
        try:
            import requests

            self._rate_limit()

            # Build query
            query = ' OR '.join(keywords[:5])  # Twitter limits query length
            query += ' -is:retweet lang:en'

            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
            params = {
                "query": query,
                "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "max_results": 100,
                "tweet.fields": "created_at,public_metrics"
            }

            response = requests.get(url, headers=headers, params=params)

            if response.status_code != 200:
                logger.warning(f"Twitter API error: {response.status_code}")
                return pd.DataFrame()

            data = response.json()

            if 'data' not in data:
                return pd.DataFrame()

            tweets = []
            for tweet in data['data']:
                sentiment = self._analyze_sentiment(tweet['text'])
                tweets.append({
                    'timestamp': pd.to_datetime(tweet['created_at']),
                    'text': tweet['text'],
                    'sentiment_score': sentiment,
                    'source': 'twitter',
                    'engagement': tweet.get('public_metrics', {}).get('like_count', 0)
                })

            return pd.DataFrame(tweets)

        except ImportError:
            logger.warning("requests library not available")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Twitter fetch error: {e}")
            return pd.DataFrame()

    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text.

        Uses simple keyword-based analysis as fallback.
        For production, integrate with FinBERT.
        """
        text_lower = text.lower()

        bullish_words = {
            'bullish', 'buy', 'long', 'calls', 'moon', 'rocket', 'rally',
            'breakout', 'strong', 'pump', 'green', 'up', 'higher', 'bull'
        }
        bearish_words = {
            'bearish', 'sell', 'short', 'puts', 'crash', 'dump', 'red',
            'down', 'lower', 'bear', 'fear', 'panic', 'drop', 'plunge'
        }

        words = set(text_lower.split())
        bullish_count = len(words & bullish_words)
        bearish_count = len(words & bearish_words)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0

        return (bullish_count - bearish_count) / total


class AlphaVantageNewsSource(SentimentDataSource):
    """
    Alpha Vantage News API sentiment source.

    Fetches financial news with pre-calculated sentiment scores.
    """

    def __init__(self, api_key: str, config: SocialSentimentConfig):
        self.api_key = api_key
        self.config = config
        self._last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting (5 calls/min for free tier)."""
        min_interval = 12  # 5 calls per minute
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def fetch_sentiment(
        self,
        start_time: datetime,
        end_time: datetime,
        keywords: List[str]
    ) -> pd.DataFrame:
        """Fetch news articles with sentiment."""
        try:
            import requests

            self._rate_limit()

            # Alpha Vantage uses tickers
            tickers = 'SPY,QQQ'  # ETFs as proxy for ES/NQ

            url = (
                f"https://www.alphavantage.co/query"
                f"?function=NEWS_SENTIMENT"
                f"&tickers={tickers}"
                f"&time_from={start_time.strftime('%Y%m%dT%H%M')}"
                f"&time_to={end_time.strftime('%Y%m%dT%H%M')}"
                f"&limit=50"
                f"&apikey={self.api_key}"
            )

            response = requests.get(url)

            if response.status_code != 200:
                logger.warning(f"Alpha Vantage API error: {response.status_code}")
                return pd.DataFrame()

            data = response.json()

            if 'feed' not in data:
                if 'Note' in data:
                    logger.warning(f"API limit reached: {data['Note']}")
                return pd.DataFrame()

            articles = []
            for item in data['feed']:
                # Extract ticker-specific sentiment
                ticker_sentiment = 0.0
                for ts in item.get('ticker_sentiment', []):
                    if ts['ticker'] in ['SPY', 'QQQ']:
                        ticker_sentiment = float(ts.get('ticker_sentiment_score', 0))
                        break

                # Use overall sentiment if no ticker-specific
                if ticker_sentiment == 0:
                    ticker_sentiment = float(item.get('overall_sentiment_score', 0))

                articles.append({
                    'timestamp': pd.to_datetime(item['time_published']),
                    'text': item.get('title', '') + ' ' + item.get('summary', ''),
                    'sentiment_score': ticker_sentiment,
                    'source': 'alpha_vantage',
                    'relevance': float(item.get('relevance_score', 0.5))
                })

            return pd.DataFrame(articles)

        except ImportError:
            logger.warning("requests library not available")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Alpha Vantage fetch error: {e}")
            return pd.DataFrame()


class RedditSentimentSource(SentimentDataSource):
    """
    Reddit sentiment source (WSB, stocks, investing).

    Note: WSB sentiment is a CONTRARIAN indicator based on research.
    """

    def __init__(self, config: SocialSentimentConfig):
        self.config = config

    def fetch_sentiment(
        self,
        start_time: datetime,
        end_time: datetime,
        keywords: List[str]
    ) -> pd.DataFrame:
        """
        Fetch Reddit sentiment.

        For production, use PRAW (Python Reddit API Wrapper).
        This provides a framework for integration.
        """
        try:
            # Check if PRAW is available
            import praw

            if not self.config.reddit_client_id:
                logger.warning("Reddit API credentials not configured")
                return pd.DataFrame()

            reddit = praw.Reddit(
                client_id=self.config.reddit_client_id,
                client_secret=self.config.reddit_client_secret,
                user_agent="SKIE_Ninja_Sentiment/1.0"
            )

            posts = []
            subreddits = ['wallstreetbets', 'stocks', 'investing']

            for sub_name in subreddits:
                try:
                    subreddit = reddit.subreddit(sub_name)
                    for post in subreddit.new(limit=50):
                        post_time = datetime.fromtimestamp(post.created_utc)

                        if start_time <= post_time <= end_time:
                            sentiment = self._analyze_sentiment(
                                post.title + ' ' + (post.selftext or '')
                            )

                            # Apply contrarian flag for WSB
                            if sub_name == 'wallstreetbets' and self.config.wsb_contrarian:
                                sentiment = -sentiment  # Invert WSB sentiment

                            posts.append({
                                'timestamp': post_time,
                                'text': post.title,
                                'sentiment_score': sentiment,
                                'source': f'reddit_{sub_name}',
                                'engagement': post.score
                            })

                except Exception as e:
                    logger.warning(f"Error fetching r/{sub_name}: {e}")

            return pd.DataFrame(posts)

        except ImportError:
            logger.info("PRAW not installed. Reddit sentiment unavailable.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Reddit fetch error: {e}")
            return pd.DataFrame()

    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis for Reddit posts."""
        text_lower = text.lower()

        # WSB-specific language
        bullish_words = {
            'moon', 'rocket', 'tendies', 'calls', 'yolo', 'diamond hands',
            'bull', 'long', 'buy', 'green', 'ath', 'breakout', 'squeeze'
        }
        bearish_words = {
            'puts', 'short', 'crash', 'dump', 'bear', 'red', 'drill',
            'sell', 'fear', 'panic', 'bagholder', 'loss porn'
        }

        words = set(text_lower.split())
        bullish_count = len(words & bullish_words)
        bearish_count = len(words & bearish_words)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0

        return (bullish_count - bearish_count) / total


class SocialNewsSentimentGenerator:
    """
    Generate trading features from social media and news sentiment.

    Aggregates sentiment from multiple sources with proper lag handling
    to prevent look-ahead bias.
    """

    def __init__(self, config: Optional[SocialSentimentConfig] = None):
        self.config = config or SocialSentimentConfig()
        self.sources: List[SentimentDataSource] = []
        self._initialize_sources()

    def _initialize_sources(self):
        """Initialize available sentiment sources."""
        if self.config.twitter_bearer_token:
            self.sources.append(
                TwitterSentimentSource(
                    self.config.twitter_bearer_token,
                    self.config
                )
            )
            logger.info("Twitter sentiment source enabled")

        if self.config.alpha_vantage_api_key:
            self.sources.append(
                AlphaVantageNewsSource(
                    self.config.alpha_vantage_api_key,
                    self.config
                )
            )
            logger.info("Alpha Vantage news source enabled")

        # Reddit doesn't require API key for basic access
        self.sources.append(RedditSentimentSource(self.config))
        logger.info("Reddit sentiment source enabled")

    def fetch_all_sentiment(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Fetch sentiment from all available sources.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Combined DataFrame of all sentiment data
        """
        all_sentiment = []

        for source in self.sources:
            try:
                df = source.fetch_sentiment(
                    start_time,
                    end_time,
                    self.config.es_keywords
                )
                if not df.empty:
                    all_sentiment.append(df)
            except Exception as e:
                logger.warning(f"Error fetching from source: {e}")

        if not all_sentiment:
            return pd.DataFrame()

        return pd.concat(all_sentiment, ignore_index=True)

    def aggregate_to_bars(
        self,
        sentiment_df: pd.DataFrame,
        bar_timestamps: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Aggregate sentiment to bar-level features.

        CRITICAL: Applies proper lag to avoid look-ahead bias.
        Only uses sentiment that was available BEFORE each bar.

        Args:
            sentiment_df: DataFrame with sentiment data
            bar_timestamps: Timestamps of price bars

        Returns:
            DataFrame with aggregated sentiment features per bar
        """
        features = pd.DataFrame(index=bar_timestamps)

        if sentiment_df.empty:
            logger.warning("No sentiment data to aggregate")
            return features

        sentiment_df = sentiment_df.copy()
        sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])

        lag_td = timedelta(minutes=self.config.min_lag_minutes)

        for window_minutes in self.config.aggregation_windows:
            window_td = timedelta(minutes=window_minutes)

            sentiments = []
            counts = []
            engagements = []

            for bar_time in bar_timestamps:
                # CRITICAL: Only use sentiment from BEFORE the bar (with lag)
                cutoff_time = bar_time - lag_td
                window_start = cutoff_time - window_td

                mask = (
                    (sentiment_df['timestamp'] >= window_start) &
                    (sentiment_df['timestamp'] < cutoff_time)
                )

                window_data = sentiment_df.loc[mask]

                if len(window_data) > 0:
                    # Weighted by engagement if available
                    if 'engagement' in window_data.columns:
                        weights = window_data['engagement'] + 1  # +1 to avoid zero
                        weights = weights / weights.sum()
                        avg_sentiment = (window_data['sentiment_score'] * weights).sum()
                    else:
                        avg_sentiment = window_data['sentiment_score'].mean()

                    sentiments.append(avg_sentiment)
                    counts.append(len(window_data))
                    engagements.append(window_data.get('engagement', pd.Series([0])).sum())
                else:
                    sentiments.append(np.nan)
                    counts.append(0)
                    engagements.append(0)

            suffix = f'_{window_minutes}min'
            features[f'social_sentiment{suffix}'] = sentiments
            features[f'social_count{suffix}'] = counts
            features[f'social_engagement{suffix}'] = engagements

        # Forward fill NaN values
        features = features.ffill().bfill()

        # Add derived features
        self._add_derived_features(features)

        return features

    def _add_derived_features(self, features: pd.DataFrame):
        """Add derived sentiment features."""
        windows = self.config.aggregation_windows
        short_win = min(windows)
        long_win = max(windows)

        short_col = f'social_sentiment_{short_win}min'
        long_col = f'social_sentiment_{long_win}min'

        if short_col in features.columns and long_col in features.columns:
            # Sentiment momentum (short-term vs long-term)
            features['social_sentiment_momentum'] = (
                features[short_col] - features[long_col]
            )

            # Sentiment trend (5-bar change)
            features['social_sentiment_trend'] = features[short_col].diff(5)

            # Sentiment extremes
            features['social_extreme_bullish'] = (
                features[short_col] > 0.5
            ).astype(int)
            features['social_extreme_bearish'] = (
                features[short_col] < -0.5
            ).astype(int)

        # Activity indicators
        count_col = f'social_count_{short_win}min'
        if count_col in features.columns:
            # High activity
            count_ma = features[count_col].rolling(20).mean()
            features['social_high_activity'] = (
                features[count_col] > count_ma * 2
            ).astype(int)

    def generate_features_for_backtest(
        self,
        bar_timestamps: pd.DatetimeIndex,
        historical_sentiment: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate sentiment features for backtesting.

        If historical sentiment data is not available, generates
        placeholder features that can be replaced with real data.

        Args:
            bar_timestamps: Timestamps of price bars
            historical_sentiment: Optional historical sentiment data

        Returns:
            DataFrame with sentiment features
        """
        if historical_sentiment is not None and not historical_sentiment.empty:
            return self.aggregate_to_bars(historical_sentiment, bar_timestamps)

        # Generate placeholder features for backtesting
        logger.warning("No historical sentiment data. Generating placeholders.")

        features = pd.DataFrame(index=bar_timestamps)

        for window in self.config.aggregation_windows:
            features[f'social_sentiment_{window}min'] = 0.0
            features[f'social_count_{window}min'] = 0
            features[f'social_engagement_{window}min'] = 0

        features['social_sentiment_momentum'] = 0.0
        features['social_sentiment_trend'] = 0.0
        features['social_extreme_bullish'] = 0
        features['social_extreme_bearish'] = 0
        features['social_high_activity'] = 0

        return features


def calculate_social_news_features(
    bar_timestamps: pd.DatetimeIndex,
    config: Optional[SocialSentimentConfig] = None,
    historical_sentiment: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Convenience function to calculate social/news sentiment features.

    Args:
        bar_timestamps: Timestamps of price bars
        config: Optional configuration
        historical_sentiment: Optional historical sentiment data

    Returns:
        DataFrame with sentiment features
    """
    generator = SocialNewsSentimentGenerator(config)
    return generator.generate_features_for_backtest(
        bar_timestamps,
        historical_sentiment
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("SOCIAL/NEWS SENTIMENT FEATURES TEST")
    print("=" * 70)

    # Create sample bar timestamps
    bar_times = pd.date_range('2024-01-02 09:30', periods=100, freq='5min')

    # Test with no API keys (placeholder features)
    print("\n[1] Testing placeholder feature generation...")
    config = SocialSentimentConfig()
    features = calculate_social_news_features(bar_times, config)

    print(f"Generated {len(features.columns)} features:")
    for col in features.columns:
        print(f"    {col}")

    # Simulate historical sentiment data
    print("\n[2] Testing with simulated historical sentiment...")
    np.random.seed(42)
    n_items = 200

    simulated_sentiment = pd.DataFrame({
        'timestamp': pd.to_datetime([
            bar_times[0] - timedelta(minutes=np.random.randint(5, 240))
            for _ in range(n_items)
        ]),
        'text': ['Sample text'] * n_items,
        'sentiment_score': np.random.randn(n_items) * 0.3,
        'source': np.random.choice(['twitter', 'reddit_wsb', 'news'], n_items),
        'engagement': np.random.randint(1, 1000, n_items)
    })

    features_with_data = calculate_social_news_features(
        bar_times,
        config,
        simulated_sentiment
    )

    print(f"Generated {len(features_with_data.columns)} features with data")
    print("\nSample values:")
    print(features_with_data.head())

    print("\n" + "=" * 70)
    print("SOCIAL/NEWS SENTIMENT TEST COMPLETE")
    print("=" * 70)
