"""
Alternative Data Collector
==========================
Collects alternative data from various sources for sentiment analysis.

Sources:
- Reddit (wallstreetbets, stocks, investing, futures)
- News headlines (via RSS feeds and free APIs)
- Fear & Greed Index
- Google Trends (search interest)

Reference: research/02_comprehensive_variables_research.md
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import logging
import json
import re
import time

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class RedditCollector:
    """Collect and analyze Reddit posts for market sentiment."""

    # Subreddits relevant for futures/equity trading
    DEFAULT_SUBREDDITS = [
        'wallstreetbets',
        'stocks',
        'investing',
        'futures',
        'options',
        'stockmarket',
        'trading'
    ]

    # Keywords for ES/NQ futures
    FUTURES_KEYWORDS = [
        'ES', 'NQ', 'SPY', 'QQQ', 'SPX', 'NDX',
        'S&P', 'nasdaq', 'futures', 'e-mini', 'emini',
        'bull', 'bear', 'calls', 'puts', 'rally', 'crash',
        'fed', 'fomc', 'jerome powell', 'inflation', 'recession'
    ]

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "SKIE_Ninja/1.0"
    ):
        """
        Initialize Reddit collector.

        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string for API requests
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.base_url = "https://www.reddit.com"
        self.oauth_url = "https://oauth.reddit.com"

        # Initialize sentiment analyzer
        self.sentiment_analyzer = None
        if NLTK_AVAILABLE:
            try:
                nltk.download('vader_lexicon', quiet=True)
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except Exception as e:
                logger.warning(f"Could not initialize NLTK sentiment: {e}")

    def get_subreddit_posts(
        self,
        subreddit: str,
        limit: int = 100,
        time_filter: str = 'day'
    ) -> List[Dict]:
        """
        Fetch posts from a subreddit using public JSON API.

        Args:
            subreddit: Subreddit name
            limit: Number of posts to fetch
            time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'

        Returns:
            List of post dictionaries
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available")
            return []

        url = f"{self.base_url}/r/{subreddit}/top.json"
        params = {
            'limit': min(limit, 100),
            't': time_filter
        }
        headers = {'User-Agent': self.user_agent}

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            posts = []
            for item in data.get('data', {}).get('children', []):
                post_data = item.get('data', {})
                posts.append({
                    'title': post_data.get('title', ''),
                    'selftext': post_data.get('selftext', ''),
                    'score': post_data.get('score', 0),
                    'upvote_ratio': post_data.get('upvote_ratio', 0.5),
                    'num_comments': post_data.get('num_comments', 0),
                    'created_utc': post_data.get('created_utc', 0),
                    'subreddit': subreddit
                })

            return posts

        except Exception as e:
            logger.error(f"Error fetching r/{subreddit}: {e}")
            return []

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using VADER.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        if self.sentiment_analyzer is None:
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}

        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            return scores
        except Exception:
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}

    def collect_market_sentiment(
        self,
        subreddits: List[str] = None,
        limit_per_sub: int = 50
    ) -> pd.DataFrame:
        """
        Collect and analyze sentiment from multiple subreddits.

        Args:
            subreddits: List of subreddit names
            limit_per_sub: Posts per subreddit

        Returns:
            DataFrame with sentiment data
        """
        if subreddits is None:
            subreddits = self.DEFAULT_SUBREDDITS

        all_posts = []
        for subreddit in subreddits:
            logger.info(f"Fetching r/{subreddit}...")
            posts = self.get_subreddit_posts(subreddit, limit_per_sub)

            for post in posts:
                # Combine title and text for analysis
                text = f"{post['title']} {post['selftext']}"

                # Check if post is relevant to futures
                is_relevant = any(
                    kw.lower() in text.lower()
                    for kw in self.FUTURES_KEYWORDS
                )

                # Analyze sentiment
                sentiment = self.analyze_sentiment(text)

                all_posts.append({
                    'subreddit': subreddit,
                    'title': post['title'][:100],
                    'score': post['score'],
                    'upvote_ratio': post['upvote_ratio'],
                    'num_comments': post['num_comments'],
                    'created_utc': datetime.fromtimestamp(post['created_utc']),
                    'is_futures_relevant': is_relevant,
                    'sentiment_compound': sentiment['compound'],
                    'sentiment_pos': sentiment['pos'],
                    'sentiment_neg': sentiment['neg'],
                    'sentiment_neu': sentiment['neu']
                })

            # Rate limiting
            time.sleep(1)

        df = pd.DataFrame(all_posts)
        return df

    def get_aggregated_sentiment(
        self,
        subreddits: List[str] = None,
        limit_per_sub: int = 50
    ) -> Dict[str, float]:
        """
        Get aggregated sentiment metrics.

        Returns:
            Dictionary with aggregated sentiment scores
        """
        df = self.collect_market_sentiment(subreddits, limit_per_sub)

        if len(df) == 0:
            return self._get_default_sentiment()

        # Filter to relevant posts
        relevant = df[df['is_futures_relevant']]
        if len(relevant) < 5:
            relevant = df  # Use all if not enough relevant

        # Weight by engagement (score + comments)
        weights = relevant['score'] + relevant['num_comments'] + 1
        weights = weights / weights.sum()

        return {
            'reddit_sentiment_compound': (relevant['sentiment_compound'] * weights).sum(),
            'reddit_sentiment_pos': (relevant['sentiment_pos'] * weights).sum(),
            'reddit_sentiment_neg': (relevant['sentiment_neg'] * weights).sum(),
            'reddit_bullish_ratio': (relevant['sentiment_compound'] > 0.05).mean(),
            'reddit_bearish_ratio': (relevant['sentiment_compound'] < -0.05).mean(),
            'reddit_post_volume': len(df),
            'reddit_avg_engagement': df['score'].mean() + df['num_comments'].mean()
        }

    def _get_default_sentiment(self) -> Dict[str, float]:
        """Return default neutral sentiment values."""
        return {
            'reddit_sentiment_compound': 0.0,
            'reddit_sentiment_pos': 0.0,
            'reddit_sentiment_neg': 0.0,
            'reddit_bullish_ratio': 0.5,
            'reddit_bearish_ratio': 0.5,
            'reddit_post_volume': 0,
            'reddit_avg_engagement': 0.0
        }


class NewsCollector:
    """Collect news headlines from free RSS feeds and APIs."""

    # Free financial news RSS feeds
    RSS_FEEDS = {
        'reuters_business': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best',
        'cnbc': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
        'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
        'seeking_alpha': 'https://seekingalpha.com/market_currents.xml',
        'yahoo_finance': 'https://finance.yahoo.com/news/rssindex',
    }

    def __init__(self):
        """Initialize news collector."""
        self.sentiment_analyzer = None
        if NLTK_AVAILABLE:
            try:
                nltk.download('vader_lexicon', quiet=True)
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except Exception as e:
                logger.warning(f"Could not initialize NLTK: {e}")

    def fetch_rss_headlines(self, feed_url: str, feed_name: str) -> List[Dict]:
        """
        Fetch headlines from RSS feed.

        Args:
            feed_url: RSS feed URL
            feed_name: Name identifier for the feed

        Returns:
            List of headline dictionaries
        """
        if not REQUESTS_AVAILABLE:
            return []

        try:
            response = requests.get(feed_url, timeout=10)
            response.raise_for_status()

            # Simple XML parsing for RSS
            headlines = []
            content = response.text

            # Extract titles using regex (simple approach)
            title_pattern = r'<title[^>]*>(?:<!\[CDATA\[)?([^<\]]+)(?:\]\]>)?</title>'
            titles = re.findall(title_pattern, content)

            for title in titles[:20]:  # Limit to 20 headlines
                if title and len(title) > 10:  # Filter out empty/short titles
                    headlines.append({
                        'source': feed_name,
                        'title': title.strip(),
                        'timestamp': datetime.now()
                    })

            return headlines

        except Exception as e:
            logger.warning(f"Error fetching {feed_name}: {e}")
            return []

    def collect_all_headlines(self) -> pd.DataFrame:
        """
        Collect headlines from all configured RSS feeds.

        Returns:
            DataFrame with headlines and sentiment
        """
        all_headlines = []

        for feed_name, feed_url in self.RSS_FEEDS.items():
            logger.info(f"Fetching {feed_name}...")
            headlines = self.fetch_rss_headlines(feed_url, feed_name)

            for headline in headlines:
                # Analyze sentiment
                if self.sentiment_analyzer:
                    sentiment = self.sentiment_analyzer.polarity_scores(headline['title'])
                else:
                    sentiment = {'compound': 0.0, 'pos': 0.0, 'neg': 0.0}

                all_headlines.append({
                    'source': headline['source'],
                    'title': headline['title'],
                    'timestamp': headline['timestamp'],
                    'sentiment_compound': sentiment['compound'],
                    'sentiment_pos': sentiment['pos'],
                    'sentiment_neg': sentiment['neg']
                })

            time.sleep(0.5)  # Rate limiting

        return pd.DataFrame(all_headlines)

    def get_news_sentiment_summary(self) -> Dict[str, float]:
        """
        Get aggregated news sentiment.

        Returns:
            Dictionary with sentiment metrics
        """
        df = self.collect_all_headlines()

        if len(df) == 0:
            return {
                'news_sentiment_compound': 0.0,
                'news_sentiment_pos': 0.0,
                'news_sentiment_neg': 0.0,
                'news_bullish_ratio': 0.5,
                'news_bearish_ratio': 0.5,
                'news_headline_count': 0
            }

        return {
            'news_sentiment_compound': df['sentiment_compound'].mean(),
            'news_sentiment_pos': df['sentiment_pos'].mean(),
            'news_sentiment_neg': df['sentiment_neg'].mean(),
            'news_bullish_ratio': (df['sentiment_compound'] > 0.05).mean(),
            'news_bearish_ratio': (df['sentiment_compound'] < -0.05).mean(),
            'news_headline_count': len(df)
        }


class FearGreedCollector:
    """Collect CNN Fear & Greed Index."""

    def __init__(self):
        """Initialize Fear & Greed collector."""
        pass

    def get_fear_greed_index(self) -> Dict[str, float]:
        """
        Fetch current Fear & Greed Index.

        Returns:
            Dictionary with Fear & Greed values
        """
        if not REQUESTS_AVAILABLE:
            return self._get_default_values()

        try:
            # CNN Fear & Greed API endpoint
            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Extract current value
            current = data.get('fear_and_greed', {})
            score = current.get('score', 50)
            rating = current.get('rating', 'Neutral')

            # Historical data for momentum
            history = data.get('fear_and_greed_historical', {})
            prev_close = history.get('previous_close', score)
            week_ago = history.get('one_week_ago', score)
            month_ago = history.get('one_month_ago', score)

            return {
                'fear_greed_score': score,
                'fear_greed_rating': rating,
                'fear_greed_change_1d': score - prev_close,
                'fear_greed_change_1w': score - week_ago,
                'fear_greed_change_1m': score - month_ago,
                'fear_greed_extreme_fear': 1 if score < 25 else 0,
                'fear_greed_extreme_greed': 1 if score > 75 else 0
            }

        except Exception as e:
            logger.warning(f"Error fetching Fear & Greed Index: {e}")
            return self._get_default_values()

    def _get_default_values(self) -> Dict[str, float]:
        """Return default neutral values."""
        return {
            'fear_greed_score': 50.0,
            'fear_greed_rating': 'Neutral',
            'fear_greed_change_1d': 0.0,
            'fear_greed_change_1w': 0.0,
            'fear_greed_change_1m': 0.0,
            'fear_greed_extreme_fear': 0,
            'fear_greed_extreme_greed': 0
        }


class AlternativeDataCollector:
    """Main collector that aggregates all alternative data sources."""

    def __init__(
        self,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None
    ):
        """
        Initialize the main alternative data collector.

        Args:
            reddit_client_id: Reddit API client ID
            reddit_client_secret: Reddit API client secret
        """
        self.reddit_collector = RedditCollector(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret
        )
        self.news_collector = NewsCollector()
        self.fear_greed_collector = FearGreedCollector()

    def collect_all(self, include_reddit: bool = True) -> Dict[str, float]:
        """
        Collect all alternative data.

        Args:
            include_reddit: Whether to include Reddit data (slower)

        Returns:
            Dictionary with all alternative data features
        """
        logger.info("Collecting alternative data...")

        data = {}

        # Fear & Greed Index (fastest)
        logger.info("Fetching Fear & Greed Index...")
        fg_data = self.fear_greed_collector.get_fear_greed_index()
        data.update(fg_data)

        # News sentiment
        logger.info("Collecting news headlines...")
        news_data = self.news_collector.get_news_sentiment_summary()
        data.update(news_data)

        # Reddit sentiment (slowest due to rate limits)
        if include_reddit:
            logger.info("Collecting Reddit sentiment...")
            reddit_data = self.reddit_collector.get_aggregated_sentiment(
                limit_per_sub=25  # Reduce for speed
            )
            data.update(reddit_data)

        # Aggregate sentiment score
        sentiment_scores = [
            data.get('news_sentiment_compound', 0),
            data.get('reddit_sentiment_compound', 0) if include_reddit else 0
        ]
        data['alt_data_sentiment_avg'] = np.mean([s for s in sentiment_scores if s != 0] or [0])

        logger.info(f"Collected {len(data)} alternative data features")
        return data


def create_sample_alternative_features() -> Dict[str, float]:
    """
    Create sample alternative data features for development.

    Returns:
        Dictionary with sample alternative data
    """
    logger.info("Creating sample alternative data features")

    np.random.seed(int(datetime.now().timestamp()) % 1000)

    return {
        # Fear & Greed
        'fear_greed_score': np.random.uniform(30, 70),
        'fear_greed_rating': 'Neutral',
        'fear_greed_change_1d': np.random.uniform(-5, 5),
        'fear_greed_change_1w': np.random.uniform(-10, 10),
        'fear_greed_change_1m': np.random.uniform(-15, 15),
        'fear_greed_extreme_fear': 0,
        'fear_greed_extreme_greed': 0,

        # News sentiment
        'news_sentiment_compound': np.random.uniform(-0.3, 0.3),
        'news_sentiment_pos': np.random.uniform(0.1, 0.4),
        'news_sentiment_neg': np.random.uniform(0.1, 0.4),
        'news_bullish_ratio': np.random.uniform(0.3, 0.7),
        'news_bearish_ratio': np.random.uniform(0.3, 0.7),
        'news_headline_count': np.random.randint(50, 150),

        # Reddit sentiment
        'reddit_sentiment_compound': np.random.uniform(-0.3, 0.3),
        'reddit_sentiment_pos': np.random.uniform(0.1, 0.4),
        'reddit_sentiment_neg': np.random.uniform(0.1, 0.4),
        'reddit_bullish_ratio': np.random.uniform(0.3, 0.7),
        'reddit_bearish_ratio': np.random.uniform(0.3, 0.7),
        'reddit_post_volume': np.random.randint(100, 500),
        'reddit_avg_engagement': np.random.uniform(50, 500),

        # Aggregate
        'alt_data_sentiment_avg': np.random.uniform(-0.2, 0.2)
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Alternative Data Collector Test")
    print("=" * 60)

    # Test with real data collection
    collector = AlternativeDataCollector()

    # Collect data (excluding Reddit for faster test)
    data = collector.collect_all(include_reddit=False)

    print(f"\nCollected {len(data)} features:")
    for key, value in data.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Test with Reddit (slower)...")
    print("=" * 60)

    # Test Reddit collector separately
    reddit = RedditCollector()
    sentiment = reddit.get_aggregated_sentiment(
        subreddits=['wallstreetbets', 'stocks'],
        limit_per_sub=10
    )

    print(f"\nReddit sentiment features:")
    for key, value in sentiment.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
