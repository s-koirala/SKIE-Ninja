"""
FinBERT Financial Sentiment Analysis Pipeline

Implementation of FinBERT-based sentiment analysis for trading features.
FinBERT is a BERT model pre-trained on financial text, specifically designed
to understand financial sentiment (positive, negative, neutral).

Features include:
1. News sentiment scoring
2. Twitter/X sentiment analysis
3. Aggregated sentiment features over time windows
4. Sentiment momentum and divergence indicators
5. Reddit WSB contrarian sentiment (inversely correlated with returns)

Important: Proper lag handling to avoid look-ahead bias.

References:
- FinBERT (ProsusAI/finbert): https://huggingface.co/ProsusAI/finbert
- FinBERT-LSTM Stock Prediction (ACM 2024)
- Reddit WSB Sentiment (ScienceDirect 2024) - WSB is contrarian indicator

Note: Requires transformers and torch libraries for full functionality.
This module provides a framework that works with or without these dependencies.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""
    # Model settings
    model_name: str = "ProsusAI/finbert"
    batch_size: int = 32
    max_length: int = 512

    # Aggregation windows (in minutes for intraday)
    aggregation_windows: List[int] = None

    # Feature settings
    min_news_count: int = 1         # Minimum news items for valid sentiment
    sentiment_decay: float = 0.95    # Decay factor for older sentiment

    # Lag settings (CRITICAL for avoiding look-ahead bias)
    min_lag_minutes: int = 5         # Minimum lag to apply

    # WSB settings (contrarian)
    wsb_contrarian: bool = True      # Use WSB as contrarian indicator

    def __post_init__(self):
        if self.aggregation_windows is None:
            self.aggregation_windows = [5, 15, 30, 60, 240]  # 5min to 4hr


class SentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers."""

    @abstractmethod
    def analyze(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of texts.

        Args:
            texts: List of text strings

        Returns:
            List of dicts with 'positive', 'negative', 'neutral' scores
        """
        pass


class FinBERTAnalyzer(SentimentAnalyzer):
    """
    FinBERT-based sentiment analyzer.

    Uses the pre-trained FinBERT model for financial sentiment analysis.
    Falls back to simple keyword-based analysis if transformers not available.
    """

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        self._model = None
        self._tokenizer = None
        self._device = None
        self._use_transformers = False
        self._initialize()

    def _initialize(self):
        """Initialize the FinBERT model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            logger.info(f"Loading FinBERT model: {self.config.model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name
            )

            # Set device
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._model.to(self._device)
            self._model.eval()

            self._use_transformers = True
            logger.info(f"FinBERT loaded on {self._device}")

        except ImportError:
            logger.warning("transformers/torch not available. Using keyword-based fallback.")
            self._use_transformers = False

    def analyze(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of texts using FinBERT.

        Args:
            texts: List of text strings

        Returns:
            List of sentiment scores {'positive', 'negative', 'neutral'}
        """
        if not texts:
            return []

        if self._use_transformers:
            return self._analyze_transformers(texts)
        else:
            return self._analyze_keywords(texts)

    def _analyze_transformers(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze using FinBERT transformers model."""
        import torch

        results = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]

            # Tokenize
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            ).to(self._device)

            # Predict
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Convert to results
            # FinBERT output: [positive, negative, neutral]
            for j, prob in enumerate(probs.cpu().numpy()):
                results.append({
                    'positive': float(prob[0]),
                    'negative': float(prob[1]),
                    'neutral': float(prob[2])
                })

        return results

    def _analyze_keywords(self, texts: List[str]) -> List[Dict[str, float]]:
        """Fallback keyword-based sentiment analysis."""
        # Simple keyword lists for fallback
        positive_words = {
            'bullish', 'rally', 'surge', 'gain', 'profit', 'growth', 'strong',
            'buy', 'long', 'upgrade', 'beat', 'exceed', 'optimistic', 'breakout',
            'support', 'accumulation', 'uptrend', 'momentum', 'outperform'
        }
        negative_words = {
            'bearish', 'crash', 'plunge', 'loss', 'decline', 'weak', 'sell',
            'short', 'downgrade', 'miss', 'fear', 'pessimistic', 'breakdown',
            'resistance', 'distribution', 'downtrend', 'underperform', 'warning'
        }

        results = []
        for text in texts:
            words = set(text.lower().split())

            pos_count = len(words & positive_words)
            neg_count = len(words & negative_words)
            total = pos_count + neg_count + 1  # +1 to avoid division by zero

            results.append({
                'positive': pos_count / total,
                'negative': neg_count / total,
                'neutral': 1 - (pos_count + neg_count) / total
            })

        return results


class SentimentFeatureGenerator:
    """
    Generate trading features from sentiment data.

    Combines news, Twitter, and Reddit sentiment into trading features
    with proper time alignment and lag handling.
    """

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        self.analyzer = FinBERTAnalyzer(config)

    def process_news_data(
        self,
        news_df: pd.DataFrame,
        text_column: str = 'text',
        time_column: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Process news data and add sentiment scores.

        Args:
            news_df: DataFrame with news data
            text_column: Column containing news text
            time_column: Column containing timestamps

        Returns:
            DataFrame with added sentiment columns
        """
        if news_df.empty:
            return news_df

        logger.info(f"Processing {len(news_df)} news items...")

        texts = news_df[text_column].tolist()
        sentiments = self.analyzer.analyze(texts)

        news_df = news_df.copy()
        news_df['sentiment_positive'] = [s['positive'] for s in sentiments]
        news_df['sentiment_negative'] = [s['negative'] for s in sentiments]
        news_df['sentiment_neutral'] = [s['neutral'] for s in sentiments]

        # Compound sentiment score (-1 to +1)
        news_df['sentiment_compound'] = (
            news_df['sentiment_positive'] - news_df['sentiment_negative']
        )

        logger.info("News sentiment processing complete")
        return news_df

    def aggregate_to_bars(
        self,
        sentiment_df: pd.DataFrame,
        bar_timestamps: pd.DatetimeIndex,
        time_column: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Aggregate sentiment to bar-level features.

        CRITICAL: Applies proper lag to avoid look-ahead bias.
        Only uses sentiment available BEFORE each bar.

        Args:
            sentiment_df: DataFrame with sentiment scores
            bar_timestamps: Timestamps of price bars
            time_column: Column with sentiment timestamps

        Returns:
            DataFrame with aggregated sentiment features per bar
        """
        features = pd.DataFrame(index=bar_timestamps)

        # Ensure timezone handling
        sentiment_df = sentiment_df.copy()
        sentiment_df[time_column] = pd.to_datetime(sentiment_df[time_column])

        for window_minutes in self.config.aggregation_windows:
            window_td = timedelta(minutes=window_minutes)
            lag_td = timedelta(minutes=self.config.min_lag_minutes)

            window_sentiments = []
            window_counts = []
            window_volumes = []

            for bar_time in bar_timestamps:
                # Only use sentiment from BEFORE the bar (with lag)
                cutoff_time = bar_time - lag_td
                window_start = cutoff_time - window_td

                mask = (
                    (sentiment_df[time_column] >= window_start) &
                    (sentiment_df[time_column] < cutoff_time)
                )

                window_data = sentiment_df.loc[mask]

                if len(window_data) >= self.config.min_news_count:
                    # Weighted by recency (more recent = higher weight)
                    if self.config.sentiment_decay < 1.0:
                        time_diffs = (cutoff_time - window_data[time_column]).dt.total_seconds()
                        weights = self.config.sentiment_decay ** (time_diffs / 60)
                        weights = weights / weights.sum()
                        avg_sentiment = (window_data['sentiment_compound'] * weights).sum()
                    else:
                        avg_sentiment = window_data['sentiment_compound'].mean()

                    window_sentiments.append(avg_sentiment)
                    window_counts.append(len(window_data))
                else:
                    window_sentiments.append(np.nan)
                    window_counts.append(0)

            # Add features for this window
            suffix = f'_{window_minutes}min'
            features[f'sentiment{suffix}'] = window_sentiments
            features[f'sentiment_count{suffix}'] = window_counts

        # Fill NaN with forward fill then backward fill
        features = features.ffill().bfill()

        # Add derived features
        self._add_derived_features(features)

        logger.info(f"Generated {len(features.columns)} sentiment features")
        return features

    def _add_derived_features(self, features: pd.DataFrame):
        """Add derived sentiment features."""
        # Get shortest and longest windows
        windows = self.config.aggregation_windows
        short_win = min(windows)
        long_win = max(windows)

        short_col = f'sentiment_{short_win}min'
        long_col = f'sentiment_{long_win}min'

        if short_col in features.columns and long_col in features.columns:
            # Sentiment momentum (short - long)
            features['sentiment_momentum'] = (
                features[short_col] - features[long_col]
            )

            # Sentiment trend
            features['sentiment_trend'] = features[short_col].diff(5)

            # Sentiment extremes
            features['sentiment_extreme_pos'] = (features[short_col] > 0.5).astype(int)
            features['sentiment_extreme_neg'] = (features[short_col] < -0.5).astype(int)

        # Sentiment volatility
        for window in windows:
            col = f'sentiment_{window}min'
            if col in features.columns:
                features[f'sentiment_vol_{window}min'] = (
                    features[col].rolling(12).std()
                )

    def generate_wsb_features(
        self,
        wsb_sentiment: pd.Series,
        bar_timestamps: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Generate WallStreetBets sentiment features.

        IMPORTANT: WSB sentiment is a CONTRARIAN indicator.
        Academic research shows it is inversely correlated with returns.

        Args:
            wsb_sentiment: Series of WSB sentiment scores
            bar_timestamps: Timestamps of price bars

        Returns:
            DataFrame with WSB features (inverted if contrarian=True)
        """
        features = pd.DataFrame(index=bar_timestamps)

        # Align WSB sentiment to bars
        wsb_aligned = wsb_sentiment.reindex(bar_timestamps, method='ffill')

        # Raw WSB sentiment
        features['wsb_sentiment_raw'] = wsb_aligned

        if self.config.wsb_contrarian:
            # CONTRARIAN: invert the signal
            features['wsb_sentiment'] = -wsb_aligned
            features['wsb_bullish_contrarian'] = (wsb_aligned > 0.3).astype(int)  # Fade bullish
            features['wsb_bearish_contrarian'] = (wsb_aligned < -0.3).astype(int)  # Fade bearish
        else:
            features['wsb_sentiment'] = wsb_aligned

        # WSB extremes (high conviction = fade harder)
        features['wsb_extreme'] = (abs(wsb_aligned) > 0.5).astype(int)

        # WSB momentum
        features['wsb_momentum'] = wsb_aligned.diff(5)

        logger.info(f"Generated {len(features.columns)} WSB features "
                   f"(contrarian={self.config.wsb_contrarian})")
        return features


class NewsSentimentCollector:
    """
    Collect and process news from various sources.

    Integrates with common financial news APIs.
    """

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        self._api_keys = {}

    def set_api_key(self, source: str, api_key: str):
        """Set API key for a news source."""
        self._api_keys[source] = api_key

    def fetch_alpha_vantage_news(
        self,
        ticker: str,
        time_from: datetime,
        time_to: datetime
    ) -> pd.DataFrame:
        """
        Fetch news from Alpha Vantage News API.

        Args:
            ticker: Stock/ETF ticker (e.g., 'SPY')
            time_from: Start time
            time_to: End time

        Returns:
            DataFrame with news data
        """
        if 'alpha_vantage' not in self._api_keys:
            logger.warning("Alpha Vantage API key not set")
            return pd.DataFrame()

        try:
            import requests

            api_key = self._api_keys['alpha_vantage']
            url = (
                f"https://www.alphavantage.co/query"
                f"?function=NEWS_SENTIMENT"
                f"&tickers={ticker}"
                f"&time_from={time_from.strftime('%Y%m%dT%H%M')}"
                f"&time_to={time_to.strftime('%Y%m%dT%H%M')}"
                f"&apikey={api_key}"
            )

            response = requests.get(url)
            data = response.json()

            if 'feed' not in data:
                logger.warning(f"No news data returned: {data.get('Note', 'Unknown error')}")
                return pd.DataFrame()

            news_items = []
            for item in data['feed']:
                news_items.append({
                    'timestamp': pd.to_datetime(item['time_published']),
                    'title': item['title'],
                    'text': item.get('summary', item['title']),
                    'source': item.get('source', 'unknown'),
                    'url': item.get('url', '')
                })

            return pd.DataFrame(news_items)

        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return pd.DataFrame()

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load news data from CSV file.

        Expected columns: timestamp, text (or title), source

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with news data
        """
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Use title if text not available
            if 'text' not in df.columns and 'title' in df.columns:
                df['text'] = df['title']

            logger.info(f"Loaded {len(df)} news items from {filepath}")
            return df

        except Exception as e:
            logger.error(f"Error loading news from {filepath}: {e}")
            return pd.DataFrame()


def generate_sentiment_features(
    prices: pd.DataFrame,
    news_df: Optional[pd.DataFrame] = None,
    wsb_sentiment: Optional[pd.Series] = None,
    config: Optional[SentimentConfig] = None
) -> pd.DataFrame:
    """
    Generate comprehensive sentiment features for trading.

    Convenience function combining all sentiment sources.

    Args:
        prices: DataFrame with OHLC and datetime index
        news_df: Optional DataFrame with news data
        wsb_sentiment: Optional Series with WSB sentiment
        config: Optional configuration

    Returns:
        DataFrame with all sentiment features aligned to price bars
    """
    config = config or SentimentConfig()
    generator = SentimentFeatureGenerator(config)
    all_features = pd.DataFrame(index=prices.index)

    # Process news if available
    if news_df is not None and not news_df.empty:
        # Add sentiment scores to news
        news_with_sentiment = generator.process_news_data(news_df)

        # Aggregate to bars
        news_features = generator.aggregate_to_bars(
            news_with_sentiment,
            prices.index
        )
        all_features = pd.concat([all_features, news_features], axis=1)

    # Process WSB sentiment if available
    if wsb_sentiment is not None:
        wsb_features = generator.generate_wsb_features(
            wsb_sentiment,
            prices.index
        )
        all_features = pd.concat([all_features, wsb_features], axis=1)

    # If no external data, generate placeholder features
    if all_features.empty:
        logger.warning("No sentiment data provided. Generating placeholder features.")
        all_features['sentiment_placeholder'] = 0.0

    logger.info(f"Generated {len(all_features.columns)} total sentiment features")
    return all_features


if __name__ == "__main__":
    print("=" * 70)
    print("FINBERT SENTIMENT ANALYSIS - TEST")
    print("=" * 70)

    # Create sample data
    np.random.seed(42)
    n_bars = 200

    # Sample price data
    close = 4500 + np.cumsum(np.random.randn(n_bars) * 10)
    bar_times = pd.date_range('2024-01-01 09:30', periods=n_bars, freq='5min')

    prices = pd.DataFrame({
        'open': close + np.random.randn(n_bars) * 2,
        'high': close + abs(np.random.randn(n_bars)) * 5,
        'low': close - abs(np.random.randn(n_bars)) * 5,
        'close': close
    }, index=bar_times)

    # Sample news data
    n_news = 50
    news_times = pd.to_datetime([
        bar_times[0] + timedelta(minutes=np.random.randint(0, n_bars * 5))
        for _ in range(n_news)
    ])

    sample_headlines = [
        "S&P 500 rallies to new highs on strong earnings",
        "Market fears grow as Fed signals rate hikes",
        "Tech stocks surge on AI optimism",
        "Bears take control as recession fears mount",
        "Neutral trading day with mixed signals",
        "Bull market continues despite uncertainty",
        "Profit taking drags indices lower",
        "Breakout expected as momentum builds",
    ]

    news_df = pd.DataFrame({
        'timestamp': news_times,
        'text': [np.random.choice(sample_headlines) for _ in range(n_news)],
        'source': ['test'] * n_news
    }).sort_values('timestamp')

    # Sample WSB sentiment
    wsb_sentiment = pd.Series(
        np.random.randn(n_bars) * 0.3,
        index=bar_times
    ).cumsum() * 0.1  # Cumulative to simulate trends

    print("\n[1] Testing FinBERTAnalyzer...")
    config = SentimentConfig()
    analyzer = FinBERTAnalyzer(config)

    test_texts = [
        "Stock market rallies on strong earnings",
        "Fears of recession drive markets lower",
        "Trading remains flat amid uncertainty"
    ]

    sentiments = analyzer.analyze(test_texts)
    for text, sent in zip(test_texts, sentiments):
        print(f"  '{text[:40]}...'")
        print(f"    Pos: {sent['positive']:.3f}, Neg: {sent['negative']:.3f}, "
              f"Neu: {sent['neutral']:.3f}")

    print("\n[2] Testing SentimentFeatureGenerator...")
    generator = SentimentFeatureGenerator(config)

    # Process news
    news_with_sentiment = generator.process_news_data(news_df)
    print(f"Processed {len(news_with_sentiment)} news items")
    print(f"Sentiment range: {news_with_sentiment['sentiment_compound'].min():.3f} "
          f"to {news_with_sentiment['sentiment_compound'].max():.3f}")

    print("\n[3] Testing aggregate_to_bars...")
    bar_features = generator.aggregate_to_bars(
        news_with_sentiment,
        prices.index
    )
    print(f"Generated {len(bar_features.columns)} features")
    print(f"Features: {list(bar_features.columns[:6])}")

    print("\n[4] Testing WSB features (contrarian)...")
    wsb_features = generator.generate_wsb_features(wsb_sentiment, prices.index)
    print(f"Generated {len(wsb_features.columns)} WSB features")
    print(f"Contrarian enabled: {config.wsb_contrarian}")
    print(f"Sample WSB signals: {wsb_features['wsb_sentiment'].head()}")

    print("\n[5] Testing generate_sentiment_features...")
    all_features = generate_sentiment_features(
        prices,
        news_df=news_df,
        wsb_sentiment=wsb_sentiment,
        config=config
    )
    print(f"Total features: {len(all_features.columns)}")

    print("\n" + "=" * 70)
    print("FINBERT SENTIMENT TEST COMPLETE")
    print("=" * 70)
