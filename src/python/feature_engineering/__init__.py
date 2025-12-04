# Feature Engineering Module
# Implements 1000+ features across 13 categories:
# 1. Price-Based Features (~50)
# 2. Technical Indicators (~100)
# 3. Macroeconomic Variables (~50)
# 4. Microstructure Variables (~40)
# 5. Sentiment & Positioning (~30)
# 6. Intermarket Relationships (~25)
# 7. Seasonality & Calendar (~30)
# 8. Statistical Arbitrage (~20)
# 9. Regime & Fractal Features (~15)
# 10. Alternative Data (~50)
# 11. Lagged & Transformed (~200)
# 12. Interaction Features (~50-100)
# 13. Target Labels (~10)
#
# NEW (2025-12-04 - Research Phase Implementation):
# 14. Triple Barrier Labels (Lopez de Prado)
# 15. Volatility Regime Features (VIX, HMM)
# 16. FinBERT Sentiment Features

# Triple Barrier Labeling
from .triple_barrier import (
    TripleBarrierConfig,
    TripleBarrierLabeler,
    BarrierType,
    apply_triple_barrier,
    generate_barrier_features
)

# Volatility Regime Detection
from .volatility_regime import (
    VolatilityConfig,
    VIXFeatureGenerator,
    RealizedVolatilityGenerator,
    RegimeDetector,
    HiddenMarkovRegimeModel,
    MarketRegime,
    generate_volatility_features
)

# FinBERT Sentiment
from .finbert_sentiment import (
    SentimentConfig,
    FinBERTAnalyzer,
    SentimentFeatureGenerator,
    NewsSentimentCollector,
    generate_sentiment_features
)

__all__ = [
    # Triple Barrier
    'TripleBarrierConfig',
    'TripleBarrierLabeler',
    'BarrierType',
    'apply_triple_barrier',
    'generate_barrier_features',
    # Volatility Regime
    'VolatilityConfig',
    'VIXFeatureGenerator',
    'RealizedVolatilityGenerator',
    'RegimeDetector',
    'HiddenMarkovRegimeModel',
    'MarketRegime',
    'generate_volatility_features',
    # Sentiment
    'SentimentConfig',
    'FinBERTAnalyzer',
    'SentimentFeatureGenerator',
    'NewsSentimentCollector',
    'generate_sentiment_features',
]
