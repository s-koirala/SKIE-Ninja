"""
Volatility Regime Detection and VIX-Based Features

This module implements volatility regime detection for adaptive trading strategies.
Different market regimes require different trading approaches:
- Low volatility trending: Trend following works best
- Low volatility range: Mean reversion works best
- High volatility trending: Momentum with tight stops
- High volatility chaos: Reduce size or stay flat

Features include:
1. VIX-based features (level, term structure, rate of change)
2. Realized volatility features (multiple timeframes)
3. Hidden Markov Model regime classification
4. Gaussian Mixture Model clustering
5. Regime-conditional indicators

References:
- VIX ML Trading (PLOS One 2024)
- Classifying Market Regimes (Macrosynergy)
- Predicting VIX with Adaptive ML (Taylor & Francis 2024)
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    LOW_VOL_TRENDING = 0
    LOW_VOL_RANGE = 1
    HIGH_VOL_TRENDING = 2
    HIGH_VOL_CHAOS = 3


@dataclass
class VolatilityConfig:
    """Configuration for volatility features."""
    # VIX thresholds
    low_vol_threshold: float = 15.0     # Below this = low volatility
    high_vol_threshold: float = 25.0    # Above this = high volatility
    extreme_vol_threshold: float = 35.0 # Above this = extreme fear

    # Realized volatility periods
    rv_periods: List[int] = None        # Periods for realized vol calculation

    # Regime detection
    n_regimes: int = 4                  # Number of regimes for GMM/HMM
    regime_lookback: int = 20           # Bars for regime features

    # Rolling windows
    vix_percentile_window: int = 252    # 1 year for VIX percentile

    def __post_init__(self):
        if self.rv_periods is None:
            self.rv_periods = [5, 10, 21, 63]  # Week, 2 weeks, month, quarter


class VIXFeatureGenerator:
    """
    Generate VIX-based features for volatility regime detection.

    VIX is the CBOE Volatility Index, representing market's expectation of
    30-day forward-looking volatility implied by S&P 500 options.
    """

    def __init__(self, config: Optional[VolatilityConfig] = None):
        self.config = config or VolatilityConfig()

    def generate_vix_features(
        self,
        vix_data: pd.Series,
        prices: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive VIX-based features.

        Args:
            vix_data: VIX index values (can be daily or intraday)
            prices: Optional price data for correlation features

        Returns:
            DataFrame with VIX features
        """
        features = pd.DataFrame(index=vix_data.index)

        # Basic VIX level features
        features['vix'] = vix_data
        features['vix_log'] = np.log(vix_data)

        # VIX zone classification
        features['vix_low'] = (vix_data < self.config.low_vol_threshold).astype(int)
        features['vix_mid'] = (
            (vix_data >= self.config.low_vol_threshold) &
            (vix_data < self.config.high_vol_threshold)
        ).astype(int)
        features['vix_high'] = (
            (vix_data >= self.config.high_vol_threshold) &
            (vix_data < self.config.extreme_vol_threshold)
        ).astype(int)
        features['vix_extreme'] = (vix_data >= self.config.extreme_vol_threshold).astype(int)

        # VIX percentile (rolling)
        features['vix_percentile'] = vix_data.rolling(
            self.config.vix_percentile_window
        ).apply(lambda x: (x.iloc[-1] > x).sum() / len(x) * 100, raw=False)

        # VIX rate of change
        for period in [1, 5, 10, 21]:
            features[f'vix_roc_{period}'] = vix_data.pct_change(period)

        # VIX momentum
        features['vix_momentum_5'] = vix_data - vix_data.rolling(5).mean()
        features['vix_momentum_21'] = vix_data - vix_data.rolling(21).mean()

        # VIX mean reversion signal
        vix_ma_20 = vix_data.rolling(20).mean()
        vix_std_20 = vix_data.rolling(20).std()
        features['vix_zscore'] = (vix_data - vix_ma_20) / (vix_std_20 + 1e-10)

        # VIX trend features
        features['vix_sma_5'] = vix_data.rolling(5).mean()
        features['vix_sma_10'] = vix_data.rolling(10).mean()
        features['vix_sma_21'] = vix_data.rolling(21).mean()
        features['vix_above_sma21'] = (vix_data > features['vix_sma_21']).astype(int)

        # VIX spikes (rapid increases)
        vix_1d_change = vix_data.pct_change()
        features['vix_spike_1d'] = (vix_1d_change > 0.10).astype(int)  # 10% spike
        features['vix_spike_magnitude'] = vix_1d_change.clip(lower=0)

        # VIX contraction (declining vol)
        features['vix_contraction'] = (vix_1d_change < -0.05).astype(int)

        # Rolling VIX stats
        features['vix_max_5d'] = vix_data.rolling(5).max()
        features['vix_min_5d'] = vix_data.rolling(5).min()
        features['vix_range_5d'] = features['vix_max_5d'] - features['vix_min_5d']

        # VIX regime persistence
        above_20 = (vix_data > 20).astype(int)
        features['vix_days_above_20'] = above_20.rolling(21).sum()

        logger.info(f"Generated {len(features.columns)} VIX features")
        return features

    def generate_vix_term_structure(
        self,
        vix_spot: pd.Series,
        vix_futures_1: pd.Series,
        vix_futures_2: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Generate VIX term structure features.

        VIX term structure indicates market fear:
        - Contango (VX1 > VIX): Normal conditions, complacency
        - Backwardation (VIX > VX1): Fear, hedging demand

        Args:
            vix_spot: VIX spot index
            vix_futures_1: Front month VIX futures (VX1)
            vix_futures_2: Optional second month VIX futures (VX2)

        Returns:
            DataFrame with term structure features
        """
        features = pd.DataFrame(index=vix_spot.index)

        # VIX basis (spot vs futures)
        features['vix_basis'] = vix_spot - vix_futures_1
        features['vix_basis_pct'] = (vix_spot - vix_futures_1) / vix_spot

        # Term structure state
        features['vix_contango'] = (vix_futures_1 > vix_spot).astype(int)
        features['vix_backwardation'] = (vix_spot > vix_futures_1).astype(int)

        # Basis percentile
        features['vix_basis_percentile'] = features['vix_basis'].rolling(252).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) * 100, raw=False
        )

        if vix_futures_2 is not None:
            # Second month spread
            features['vix_1_2_spread'] = vix_futures_1 - vix_futures_2
            features['vix_term_slope'] = features['vix_1_2_spread'] / vix_spot

        logger.info(f"Generated {len(features.columns)} VIX term structure features")
        return features


class RealizedVolatilityGenerator:
    """
    Generate realized volatility features from price data.

    Realized volatility measures actual historical volatility,
    complementing VIX (implied/expected volatility).
    """

    def __init__(self, config: Optional[VolatilityConfig] = None):
        self.config = config or VolatilityConfig()

    def generate_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate realized volatility features.

        Args:
            prices: DataFrame with OHLC data

        Returns:
            DataFrame with realized volatility features
        """
        features = pd.DataFrame(index=prices.index)
        close = prices['close']
        high = prices['high']
        low = prices['low']

        # Log returns
        log_returns = np.log(close / close.shift(1))

        # Realized volatility (standard deviation of returns)
        for period in self.config.rv_periods:
            # Close-to-close volatility
            rv = log_returns.rolling(period).std() * np.sqrt(252)  # Annualized
            features[f'rv_{period}'] = rv

            # Parkinson volatility (uses high-low)
            parkinson = np.sqrt(
                (np.log(high / low) ** 2).rolling(period).mean() / (4 * np.log(2))
            ) * np.sqrt(252)
            features[f'parkinson_vol_{period}'] = parkinson

        # Volatility ratios
        if len(self.config.rv_periods) >= 2:
            short_period = self.config.rv_periods[0]
            long_period = self.config.rv_periods[-1]
            features['rv_ratio'] = (
                features[f'rv_{short_period}'] /
                (features[f'rv_{long_period}'] + 1e-10)
            )

        # Volatility regime based on RV
        rv_21 = features.get('rv_21', log_returns.rolling(21).std() * np.sqrt(252))
        rv_mean = rv_21.rolling(63).mean()
        rv_std = rv_21.rolling(63).std()
        features['rv_zscore'] = (rv_21 - rv_mean) / (rv_std + 1e-10)

        # Volatility trend
        features['rv_trend'] = rv_21 - rv_21.rolling(21).mean()
        features['rv_increasing'] = (features['rv_trend'] > 0).astype(int)

        # ATR-based volatility
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        for period in [7, 14, 21]:
            atr = tr.rolling(period).mean()
            features[f'atr_{period}'] = atr
            features[f'atr_{period}_pct'] = atr / close * 100  # As percentage

        # ATR expansion/contraction
        features['atr_expansion'] = (
            features['atr_14'] > features['atr_14'].rolling(14).mean()
        ).astype(int)

        logger.info(f"Generated {len(features.columns)} realized volatility features")
        return features


class RegimeDetector:
    """
    Detect market regimes using statistical methods.

    Supports:
    1. Gaussian Mixture Models (unsupervised)
    2. Rule-based classification
    3. Feature-based regime scoring
    """

    def __init__(self, config: Optional[VolatilityConfig] = None):
        self.config = config or VolatilityConfig()
        self.gmm = None
        self.scaler = StandardScaler()
        self._regime_features = None

    def _prepare_regime_features(
        self,
        prices: pd.DataFrame,
        vol_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare features for regime detection."""
        close = prices['close']

        regime_features = pd.DataFrame(index=prices.index)

        # Volatility level
        if 'rv_21' in vol_features.columns:
            regime_features['volatility'] = vol_features['rv_21']
        else:
            log_ret = np.log(close / close.shift(1))
            regime_features['volatility'] = log_ret.rolling(21).std() * np.sqrt(252)

        # Trend strength (absolute momentum)
        returns_21 = close.pct_change(21)
        regime_features['trend_strength'] = abs(returns_21)

        # Trend direction
        regime_features['trend_direction'] = np.sign(returns_21)

        # Mean reversion (autocorrelation)
        log_ret = np.log(close / close.shift(1))
        regime_features['autocorr'] = log_ret.rolling(21).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )

        # Range expansion
        high_21 = prices['high'].rolling(21).max()
        low_21 = prices['low'].rolling(21).min()
        regime_features['range_expansion'] = (high_21 - low_21) / close

        return regime_features.dropna()

    def fit_gmm(
        self,
        prices: pd.DataFrame,
        vol_features: pd.DataFrame
    ) -> 'RegimeDetector':
        """
        Fit Gaussian Mixture Model for regime detection.

        Args:
            prices: OHLC price data
            vol_features: Volatility features

        Returns:
            Self for chaining
        """
        logger.info("Fitting GMM for regime detection...")

        # Prepare features
        self._regime_features = self._prepare_regime_features(prices, vol_features)

        # Scale features
        X = self.scaler.fit_transform(self._regime_features)

        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.config.n_regimes,
            covariance_type='full',
            n_init=10,
            random_state=42
        )
        self.gmm.fit(X)

        # Log regime characteristics
        regimes = self.gmm.predict(X)
        for i in range(self.config.n_regimes):
            mask = regimes == i
            if mask.sum() > 0:
                vol_mean = self._regime_features.loc[mask, 'volatility'].mean()
                trend_mean = self._regime_features.loc[mask, 'trend_strength'].mean()
                logger.info(f"Regime {i}: Vol={vol_mean:.4f}, Trend={trend_mean:.4f}, "
                           f"Count={mask.sum()}")

        return self

    def predict_regime(
        self,
        prices: pd.DataFrame,
        vol_features: pd.DataFrame
    ) -> pd.Series:
        """
        Predict regime for new data.

        Args:
            prices: OHLC price data
            vol_features: Volatility features

        Returns:
            Series of regime labels
        """
        if self.gmm is None:
            raise ValueError("GMM not fitted. Call fit_gmm() first.")

        regime_features = self._prepare_regime_features(prices, vol_features)
        X = self.scaler.transform(regime_features)
        regimes = self.gmm.predict(X)

        return pd.Series(regimes, index=regime_features.index)

    def classify_regime_rules(
        self,
        vol_features: pd.DataFrame,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Classify regimes using rule-based approach.

        This provides interpretable regime classification without ML.

        Args:
            vol_features: Volatility features with rv_21 or similar
            prices: OHLC data

        Returns:
            DataFrame with regime classifications
        """
        features = pd.DataFrame(index=prices.index)
        close = prices['close']

        # Get volatility measure
        if 'rv_21' in vol_features.columns:
            vol = vol_features['rv_21']
        else:
            log_ret = np.log(close / close.shift(1))
            vol = log_ret.rolling(21).std() * np.sqrt(252)

        # Trend measure
        returns_21 = close.pct_change(21)
        trend_strength = abs(returns_21)

        # Volatility classification
        vol_median = vol.rolling(252).median()
        is_high_vol = vol > vol_median * 1.2
        is_low_vol = vol < vol_median * 0.8

        # Trend classification
        trend_median = trend_strength.rolling(252).median()
        is_trending = trend_strength > trend_median

        # Regime classification
        features['regime'] = MarketRegime.LOW_VOL_RANGE.value  # Default

        # Low vol trending
        features.loc[is_low_vol & is_trending, 'regime'] = MarketRegime.LOW_VOL_TRENDING.value

        # Low vol range
        features.loc[is_low_vol & ~is_trending, 'regime'] = MarketRegime.LOW_VOL_RANGE.value

        # High vol trending
        features.loc[is_high_vol & is_trending, 'regime'] = MarketRegime.HIGH_VOL_TRENDING.value

        # High vol chaos
        features.loc[is_high_vol & ~is_trending, 'regime'] = MarketRegime.HIGH_VOL_CHAOS.value

        # One-hot encode regimes
        for regime in MarketRegime:
            features[f'regime_{regime.name.lower()}'] = (
                features['regime'] == regime.value
            ).astype(int)

        # Regime transition features
        features['regime_change'] = (features['regime'] != features['regime'].shift(1)).astype(int)
        features['regime_duration'] = features.groupby(
            (features['regime'] != features['regime'].shift(1)).cumsum()
        ).cumcount() + 1

        logger.info(f"Generated {len(features.columns)} regime classification features")
        return features


class HiddenMarkovRegimeModel:
    """
    Hidden Markov Model for regime detection.

    HMM treats regimes as latent (hidden) states that generate
    observable market data. Good for:
    - Capturing regime persistence
    - Probabilistic regime forecasts
    - Smooth transitions between regimes

    Note: Requires hmmlearn library for full implementation.
    This is a simplified version using transition matrices.
    """

    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        self.transition_matrix = None
        self.emission_params = None
        self._state_history = None

    def fit_simplified(
        self,
        regime_labels: pd.Series,
        features: pd.DataFrame
    ) -> 'HiddenMarkovRegimeModel':
        """
        Fit simplified HMM using transition counts.

        Args:
            regime_labels: Observed regime labels
            features: Features for each regime

        Returns:
            Self for chaining
        """
        logger.info("Fitting simplified HMM...")

        # Calculate transition matrix
        transitions = np.zeros((self.n_states, self.n_states))
        labels = regime_labels.dropna().values

        for i in range(len(labels) - 1):
            from_state = int(labels[i])
            to_state = int(labels[i + 1])
            if 0 <= from_state < self.n_states and 0 <= to_state < self.n_states:
                transitions[from_state, to_state] += 1

        # Normalize to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        self.transition_matrix = np.where(
            row_sums > 0,
            transitions / row_sums,
            1 / self.n_states
        )

        # Store emission parameters (mean/std for each state)
        self.emission_params = {}
        for state in range(self.n_states):
            mask = regime_labels == state
            if mask.sum() > 0:
                self.emission_params[state] = {
                    'mean': features.loc[mask].mean(),
                    'std': features.loc[mask].std()
                }

        self._state_history = regime_labels

        logger.info(f"Transition matrix:\n{self.transition_matrix}")
        return self

    def predict_next_regime(
        self,
        current_regime: int,
        n_steps: int = 1
    ) -> np.ndarray:
        """
        Predict probability of each regime n_steps ahead.

        Args:
            current_regime: Current regime (0 to n_states-1)
            n_steps: Number of steps ahead

        Returns:
            Array of probabilities for each regime
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit_simplified() first.")

        # Start with one-hot current state
        state_prob = np.zeros(self.n_states)
        state_prob[current_regime] = 1.0

        # Apply transition matrix n_steps times
        for _ in range(n_steps):
            state_prob = state_prob @ self.transition_matrix

        return state_prob

    def get_regime_features(self, current_regime: int) -> pd.DataFrame:
        """
        Generate features from HMM for current regime.

        Args:
            current_regime: Current regime label

        Returns:
            DataFrame with HMM-derived features
        """
        if self.transition_matrix is None:
            return pd.DataFrame()

        features = {}

        # Current regime one-hot
        for i in range(self.n_states):
            features[f'hmm_in_regime_{i}'] = 1 if current_regime == i else 0

        # Transition probabilities
        for i in range(self.n_states):
            features[f'hmm_prob_to_{i}_1step'] = self.transition_matrix[current_regime, i]

        # Multi-step predictions
        prob_5 = self.predict_next_regime(current_regime, 5)
        for i in range(self.n_states):
            features[f'hmm_prob_to_{i}_5step'] = prob_5[i]

        # Regime stability (probability of staying in current)
        features['hmm_regime_stability'] = self.transition_matrix[current_regime, current_regime]

        return pd.DataFrame([features])


def generate_volatility_features(
    prices: pd.DataFrame,
    vix_data: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Generate comprehensive volatility and regime features.

    Convenience function combining all volatility feature generators.

    Args:
        prices: DataFrame with OHLC data
        vix_data: Optional VIX index data (aligned with prices)

    Returns:
        DataFrame with all volatility features
    """
    config = VolatilityConfig()
    all_features = pd.DataFrame(index=prices.index)

    # Realized volatility features
    rv_gen = RealizedVolatilityGenerator(config)
    rv_features = rv_gen.generate_features(prices)
    all_features = pd.concat([all_features, rv_features], axis=1)

    # VIX features (if available)
    if vix_data is not None:
        vix_gen = VIXFeatureGenerator(config)
        vix_features = vix_gen.generate_vix_features(vix_data, prices)
        all_features = pd.concat([all_features, vix_features], axis=1)

    # Regime classification
    regime_detector = RegimeDetector(config)
    regime_features = regime_detector.classify_regime_rules(all_features, prices)
    all_features = pd.concat([all_features, regime_features], axis=1)

    logger.info(f"Generated {len(all_features.columns)} total volatility/regime features")
    return all_features


if __name__ == "__main__":
    print("=" * 70)
    print("VOLATILITY REGIME DETECTION - TEST")
    print("=" * 70)

    # Create sample data
    np.random.seed(42)
    n_bars = 1000

    # Simulate price with regime changes
    returns = np.zeros(n_bars)

    # Low vol trending (bars 0-300)
    returns[:300] = np.random.randn(300) * 0.005 + 0.001

    # High vol chaos (bars 300-500)
    returns[300:500] = np.random.randn(200) * 0.02

    # Low vol range (bars 500-700)
    returns[500:700] = np.random.randn(200) * 0.003

    # High vol trending (bars 700-1000)
    returns[700:] = np.random.randn(300) * 0.015 - 0.002

    close = 4500 * np.cumprod(1 + returns)

    prices = pd.DataFrame({
        'open': close * (1 + np.random.randn(n_bars) * 0.001),
        'high': close * (1 + np.abs(np.random.randn(n_bars)) * 0.003),
        'low': close * (1 - np.abs(np.random.randn(n_bars)) * 0.003),
        'close': close
    })

    # Simulate VIX
    vix = 15 + np.cumsum(np.random.randn(n_bars) * 0.5)
    vix[300:500] += 15  # Spike during chaos
    vix[700:] += 10     # Elevated during high vol trending
    vix = np.clip(vix, 10, 80)
    vix = pd.Series(vix, index=prices.index)

    print("\n[1] Testing RealizedVolatilityGenerator...")
    rv_gen = RealizedVolatilityGenerator()
    rv_features = rv_gen.generate_features(prices)
    print(f"Generated {len(rv_features.columns)} RV features")
    print(f"RV_21 range: {rv_features['rv_21'].min():.4f} to {rv_features['rv_21'].max():.4f}")

    print("\n[2] Testing VIXFeatureGenerator...")
    vix_gen = VIXFeatureGenerator()
    vix_features = vix_gen.generate_vix_features(vix)
    print(f"Generated {len(vix_features.columns)} VIX features")
    print(f"VIX zones: low={vix_features['vix_low'].sum()}, "
          f"mid={vix_features['vix_mid'].sum()}, "
          f"high={vix_features['vix_high'].sum()}")

    print("\n[3] Testing RegimeDetector (rules)...")
    regime_detector = RegimeDetector()
    regime_features = regime_detector.classify_regime_rules(rv_features, prices)
    print(f"Generated {len(regime_features.columns)} regime features")
    print("\nRegime distribution:")
    for regime in MarketRegime:
        count = (regime_features['regime'] == regime.value).sum()
        print(f"  {regime.name}: {count}")

    print("\n[4] Testing RegimeDetector (GMM)...")
    regime_detector.fit_gmm(prices, rv_features)
    gmm_regimes = regime_detector.predict_regime(prices, rv_features)
    print(f"GMM regime distribution: {dict(gmm_regimes.value_counts())}")

    print("\n[5] Testing HiddenMarkovRegimeModel...")
    hmm = HiddenMarkovRegimeModel(n_states=4)
    hmm.fit_simplified(regime_features['regime'], rv_features)
    print(f"Transition matrix shape: {hmm.transition_matrix.shape}")

    # Test prediction
    prob = hmm.predict_next_regime(current_regime=2, n_steps=1)
    print(f"From regime 2, 1-step probabilities: {prob}")

    print("\n[6] Testing generate_volatility_features...")
    all_features = generate_volatility_features(prices, vix)
    print(f"Total features generated: {len(all_features.columns)}")
    print(f"Sample features: {list(all_features.columns[:10])}")

    print("\n" + "=" * 70)
    print("VOLATILITY REGIME TEST COMPLETE")
    print("=" * 70)
