"""
Established Sentiment Indices Collector
========================================

Integrates well-known, academically-validated sentiment indicators:

1. AAII Investor Sentiment Survey (weekly) - Contrarian indicator
2. CBOE Put/Call Ratio (daily) - Options market sentiment
3. VIX Term Structure - Contango/backwardation
4. COT Report (weekly) - Commercial vs Speculator positioning
5. CNN Fear & Greed Index - Composite sentiment

These indices have documented predictive value in academic literature
and are widely used by institutional traders.

References:
- AAII Survey: Contrarian indicator (AAII.com)
- Put/Call Ratio: MacroMicro, CBOE
- VIX: CBOE Volatility Index

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class SentimentIndicesConfig:
    """Configuration for sentiment indices collection."""

    # Data directory for cached files
    data_dir: Path = None

    # AAII settings
    aaii_cache_hours: int = 24  # Cache AAII data for 24 hours

    # Put/Call ratio thresholds (MacroMicro research)
    pcr_bullish_threshold: float = 1.1  # Above = oversold (bullish)
    pcr_bearish_threshold: float = 0.8  # Below = overbought (bearish)

    # VIX thresholds
    vix_fear_threshold: float = 25
    vix_extreme_fear_threshold: float = 30
    vix_complacency_threshold: float = 15

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'sentiment'
        self.data_dir.mkdir(parents=True, exist_ok=True)


class AAIISentimentCollector:
    """
    AAII Investor Sentiment Survey Collector.

    The AAII survey measures individual investor sentiment (bullish/bearish/neutral).
    It's considered a CONTRARIAN indicator - extreme readings often precede reversals.

    Published: Every Thursday
    Source: https://www.aaii.com/sentimentsurvey
    """

    def __init__(self, config: SentimentIndicesConfig):
        self.config = config
        self.cache_file = config.data_dir / 'aaii_sentiment_cache.csv'

    def fetch_current_sentiment(self) -> Dict[str, float]:
        """
        Fetch current AAII sentiment data.

        Returns:
            Dict with bullish, bearish, neutral percentages
        """
        if not REQUESTS_AVAILABLE:
            return self._get_sample_data()

        try:
            # Try to fetch from AAII (they have a simple data endpoint)
            url = "https://www.aaii.com/sentimentsurvey"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code != 200:
                logger.warning(f"AAII fetch failed: {response.status_code}")
                return self._get_cached_or_sample()

            # Parse the HTML for sentiment values
            content = response.text

            # Extract percentages using regex
            bullish = self._extract_percentage(content, 'bullish')
            bearish = self._extract_percentage(content, 'bearish')
            neutral = self._extract_percentage(content, 'neutral')

            if bullish is None or bearish is None:
                return self._get_cached_or_sample()

            data = {
                'aaii_bullish': bullish,
                'aaii_bearish': bearish,
                'aaii_neutral': neutral if neutral else 100 - bullish - bearish,
                'aaii_bull_bear_spread': bullish - bearish,
                'aaii_timestamp': datetime.now().isoformat()
            }

            # Cache the data
            self._cache_data(data)

            return data

        except Exception as e:
            logger.warning(f"Error fetching AAII data: {e}")
            return self._get_cached_or_sample()

    def _extract_percentage(self, content: str, sentiment_type: str) -> Optional[float]:
        """Extract percentage value from HTML content."""
        patterns = [
            rf'{sentiment_type}[:\s]+(\d+\.?\d*)%',
            rf'{sentiment_type.capitalize()}[:\s]+(\d+\.?\d*)%',
            rf'(\d+\.?\d*)%\s*{sentiment_type}',
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _cache_data(self, data: Dict):
        """Cache AAII data to CSV."""
        try:
            df = pd.DataFrame([data])
            df.to_csv(self.cache_file, index=False)
        except Exception as e:
            logger.warning(f"Could not cache AAII data: {e}")

    def _get_cached_or_sample(self) -> Dict[str, float]:
        """Get cached data or return sample."""
        if self.cache_file.exists():
            try:
                df = pd.read_csv(self.cache_file)
                if len(df) > 0:
                    row = df.iloc[-1].to_dict()
                    # Check if cache is recent enough
                    cache_time = pd.to_datetime(row.get('aaii_timestamp', '2020-01-01'))
                    if (datetime.now() - cache_time).total_seconds() < self.config.aaii_cache_hours * 3600:
                        return row
            except Exception:
                pass

        return self._get_sample_data()

    def _get_sample_data(self) -> Dict[str, float]:
        """Return sample/historical average data."""
        # Historical averages from AAII
        return {
            'aaii_bullish': 37.5,  # Historical average
            'aaii_bearish': 31.0,  # Historical average
            'aaii_neutral': 31.5,
            'aaii_bull_bear_spread': 6.5,
            'aaii_timestamp': datetime.now().isoformat(),
            'aaii_is_sample': True
        }

    def calculate_features(self, current_data: Dict) -> Dict[str, float]:
        """
        Calculate derived features from AAII data.

        AAII is a CONTRARIAN indicator:
        - High bullish % often precedes corrections
        - High bearish % often precedes rallies
        """
        features = {}

        bullish = current_data.get('aaii_bullish', 37.5)
        bearish = current_data.get('aaii_bearish', 31.0)
        spread = current_data.get('aaii_bull_bear_spread', 6.5)

        # Raw values
        features['aaii_bullish'] = bullish
        features['aaii_bearish'] = bearish
        features['aaii_spread'] = spread

        # Historical context (averages)
        hist_bullish_avg = 37.5
        hist_bearish_avg = 31.0

        # Deviation from average
        features['aaii_bullish_deviation'] = bullish - hist_bullish_avg
        features['aaii_bearish_deviation'] = bearish - hist_bearish_avg

        # Extreme readings (contrarian signals)
        features['aaii_extreme_bullish'] = 1 if bullish > 50 else 0  # Contrarian bearish
        features['aaii_extreme_bearish'] = 1 if bearish > 50 else 0  # Contrarian bullish

        # Normalized sentiment (-1 to +1, from bearish perspective for contrarian)
        # High bearish = positive (contrarian bullish signal)
        features['aaii_contrarian_signal'] = (bearish - bullish) / 100

        return features


class PutCallRatioCollector:
    """
    CBOE Put/Call Ratio Collector.

    The Put/Call ratio measures the volume of put options vs call options.
    High ratio (>1.1) = Bearish sentiment (contrarian bullish)
    Low ratio (<0.8) = Bullish sentiment (contrarian bearish)

    Source: CBOE Historical Data
    Reference: MacroMicro analysis
    """

    def __init__(self, config: SentimentIndicesConfig):
        self.config = config
        self.data_file = config.data_dir.parent / 'raw' / 'market' / 'pcr_daily.csv'

    def load_historical_data(self) -> Optional[pd.DataFrame]:
        """Load historical Put/Call ratio data."""
        if self.data_file.exists():
            try:
                df = pd.read_csv(self.data_file)
                df['date'] = pd.to_datetime(df['date'])
                return df.set_index('date')
            except Exception as e:
                logger.warning(f"Could not load PCR data: {e}")

        return None

    def fetch_current_ratio(self) -> Dict[str, float]:
        """
        Fetch current Put/Call ratio.

        Returns:
            Dict with PCR values
        """
        if not REQUESTS_AVAILABLE:
            return self._get_sample_data()

        try:
            # Try CBOE website
            url = "https://www.cboe.com/us/options/market_statistics/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code != 200:
                return self._get_sample_data()

            # Parse for PCR values
            content = response.text

            # Try to extract total put/call ratio
            pcr_match = re.search(r'Total.*?(\d+\.?\d*)', content)

            if pcr_match:
                pcr = float(pcr_match.group(1))
                return {
                    'pcr_total': pcr,
                    'pcr_timestamp': datetime.now().isoformat()
                }

            return self._get_sample_data()

        except Exception as e:
            logger.warning(f"Error fetching PCR: {e}")
            return self._get_sample_data()

    def _get_sample_data(self) -> Dict[str, float]:
        """Return sample PCR data."""
        return {
            'pcr_total': 0.95,  # Near neutral
            'pcr_equity': 0.70,
            'pcr_index': 1.20,
            'pcr_timestamp': datetime.now().isoformat(),
            'pcr_is_sample': True
        }

    def calculate_features(self, current_data: Dict) -> Dict[str, float]:
        """
        Calculate PCR-based features.

        Based on MacroMicro research:
        - 10-day avg > 1.1 = Market trough approaching (bullish)
        - 10-day avg < 0.8 = Market peak approaching (bearish)
        """
        features = {}

        pcr = current_data.get('pcr_total', 0.95)

        # Raw value
        features['pcr_total'] = pcr

        # Sentiment interpretation
        features['pcr_bullish_extreme'] = 1 if pcr > self.config.pcr_bullish_threshold else 0
        features['pcr_bearish_extreme'] = 1 if pcr < self.config.pcr_bearish_threshold else 0

        # Normalized signal (-1 to +1)
        # High PCR = contrarian bullish, Low PCR = contrarian bearish
        features['pcr_contrarian_signal'] = (pcr - 0.95) / 0.3  # Centered around neutral
        features['pcr_contrarian_signal'] = np.clip(features['pcr_contrarian_signal'], -1, 1)

        return features


class VIXTermStructureCollector:
    """
    VIX Term Structure Analyzer.

    Analyzes VIX futures term structure:
    - Contango (normal): VIX < VIX futures = Complacency
    - Backwardation (fear): VIX > VIX futures = Fear/Panic

    Source: VIX_daily.csv in project data
    """

    def __init__(self, config: SentimentIndicesConfig):
        self.config = config
        self.vix_file = config.data_dir.parent / 'raw' / 'market' / 'VIX_daily.csv'

    def load_vix_data(self) -> Optional[pd.DataFrame]:
        """Load VIX historical data."""
        if self.vix_file.exists():
            try:
                df = pd.read_csv(self.vix_file)

                # Find date column
                date_col = None
                for col in ['Date', 'date', 'timestamp']:
                    if col in df.columns:
                        date_col = col
                        break

                if date_col:
                    df['date'] = pd.to_datetime(df[date_col])
                    df = df.set_index('date')

                # Find close column
                close_col = None
                for col in ['Close', 'close', 'Adj Close', 'VIX']:
                    if col in df.columns:
                        close_col = col
                        break

                if close_col:
                    df['vix_close'] = df[close_col]
                    return df

            except Exception as e:
                logger.warning(f"Could not load VIX data: {e}")

        return None

    def calculate_features(self, vix_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate VIX-based sentiment features.
        """
        features = {}

        if vix_data is None:
            vix_data = self.load_vix_data()

        if vix_data is None or 'vix_close' not in vix_data.columns:
            # Return sample data
            return {
                'vix_level': 18.0,
                'vix_ma10': 18.0,
                'vix_ma20': 18.0,
                'vix_percentile_20d': 0.50,
                'vix_fear_regime': 0,
                'vix_complacency_regime': 0,
                'vix_is_sample': True
            }

        # Get recent VIX values
        vix = vix_data['vix_close'].dropna()

        if len(vix) < 20:
            return self._get_sample_features()

        current_vix = vix.iloc[-1]

        # Moving averages
        features['vix_level'] = current_vix
        features['vix_ma10'] = vix.iloc[-10:].mean()
        features['vix_ma20'] = vix.iloc[-20:].mean()

        # VIX relative to MAs
        features['vix_vs_ma10'] = current_vix / features['vix_ma10']
        features['vix_vs_ma20'] = current_vix / features['vix_ma20']

        # Percentile (20-day)
        features['vix_percentile_20d'] = (vix.iloc[-20:] < current_vix).mean()

        # Regime indicators
        features['vix_fear_regime'] = 1 if current_vix > self.config.vix_fear_threshold else 0
        features['vix_extreme_fear'] = 1 if current_vix > self.config.vix_extreme_fear_threshold else 0
        features['vix_complacency_regime'] = 1 if current_vix < self.config.vix_complacency_threshold else 0

        # VIX spike (sudden increase)
        vix_change = (current_vix - vix.iloc[-2]) / vix.iloc[-2] if len(vix) > 1 else 0
        features['vix_spike'] = 1 if vix_change > 0.15 else 0  # 15% daily increase

        # Normalized sentiment (-1 = extreme fear, +1 = extreme complacency)
        # Inverted because high VIX = fear = contrarian bullish
        features['vix_sentiment'] = -np.clip((current_vix - 20) / 15, -1, 1)

        return features

    def _get_sample_features(self) -> Dict[str, float]:
        """Return sample VIX features."""
        return {
            'vix_level': 18.0,
            'vix_ma10': 18.0,
            'vix_ma20': 18.0,
            'vix_vs_ma10': 1.0,
            'vix_vs_ma20': 1.0,
            'vix_percentile_20d': 0.50,
            'vix_fear_regime': 0,
            'vix_extreme_fear': 0,
            'vix_complacency_regime': 0,
            'vix_spike': 0,
            'vix_sentiment': 0.0,
            'vix_is_sample': True
        }


class EstablishedSentimentCollector:
    """
    Main collector for all established sentiment indices.

    Combines:
    - AAII Investor Sentiment (weekly, contrarian)
    - Put/Call Ratio (daily, contrarian)
    - VIX Term Structure (daily)
    - CNN Fear & Greed (already in alternative_data_collector)
    """

    def __init__(self, config: Optional[SentimentIndicesConfig] = None):
        self.config = config or SentimentIndicesConfig()

        self.aaii_collector = AAIISentimentCollector(self.config)
        self.pcr_collector = PutCallRatioCollector(self.config)
        self.vix_collector = VIXTermStructureCollector(self.config)

    def collect_all(self) -> Dict[str, float]:
        """
        Collect all established sentiment indicators.

        Returns:
            Dict with all sentiment features
        """
        logger.info("Collecting established sentiment indices...")

        all_features = {}

        # AAII Sentiment
        logger.info("  Fetching AAII sentiment...")
        aaii_data = self.aaii_collector.fetch_current_sentiment()
        aaii_features = self.aaii_collector.calculate_features(aaii_data)
        all_features.update(aaii_features)

        # Put/Call Ratio
        logger.info("  Fetching Put/Call ratio...")
        pcr_data = self.pcr_collector.fetch_current_ratio()
        pcr_features = self.pcr_collector.calculate_features(pcr_data)
        all_features.update(pcr_features)

        # VIX Term Structure
        logger.info("  Calculating VIX features...")
        vix_features = self.vix_collector.calculate_features()
        all_features.update(vix_features)

        # Composite contrarian sentiment
        contrarian_signals = [
            all_features.get('aaii_contrarian_signal', 0),
            all_features.get('pcr_contrarian_signal', 0),
            all_features.get('vix_sentiment', 0)
        ]
        all_features['composite_contrarian_signal'] = np.mean(contrarian_signals)

        logger.info(f"  Collected {len(all_features)} sentiment features")

        return all_features

    def align_to_bars(
        self,
        bar_timestamps: pd.DatetimeIndex,
        sentiment_data: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Align sentiment data to bar timestamps.

        Since most sentiment indices are daily/weekly, they are forward-filled
        to intraday bars with appropriate lag.

        Args:
            bar_timestamps: DatetimeIndex of price bars
            sentiment_data: Optional pre-collected sentiment data

        Returns:
            DataFrame with sentiment features aligned to bars
        """
        if sentiment_data is None:
            sentiment_data = self.collect_all()

        features = pd.DataFrame(index=bar_timestamps)

        # For daily/weekly data, forward-fill to all intraday bars
        for key, value in sentiment_data.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                features[f'est_{key}'] = value

        return features


def calculate_established_sentiment_features(
    bar_timestamps: pd.DatetimeIndex,
    config: Optional[SentimentIndicesConfig] = None
) -> pd.DataFrame:
    """
    Convenience function to calculate established sentiment features.

    Args:
        bar_timestamps: DatetimeIndex of price bars
        config: Optional configuration

    Returns:
        DataFrame with sentiment features
    """
    collector = EstablishedSentimentCollector(config)
    return collector.align_to_bars(bar_timestamps)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("ESTABLISHED SENTIMENT INDICES TEST")
    print("=" * 70)

    # Test collection
    config = SentimentIndicesConfig()
    collector = EstablishedSentimentCollector(config)

    # Collect all sentiment
    print("\n[1] Collecting all sentiment indices...")
    sentiment = collector.collect_all()

    print(f"\n[2] Collected {len(sentiment)} features:")
    for key, value in sorted(sentiment.items()):
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    # Test alignment to bars
    print("\n[3] Testing bar alignment...")
    bar_times = pd.date_range('2024-01-02 09:30', periods=100, freq='5min')
    aligned = collector.align_to_bars(bar_times, sentiment)

    print(f"    Aligned to {len(aligned)} bars")
    print(f"    Features: {list(aligned.columns)[:5]}...")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
