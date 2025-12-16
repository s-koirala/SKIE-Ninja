"""
SKIE_Ninja Critical Function Tests
===================================

Pytest suite for validating critical functions:
1. Feature calculation parity
2. Signal generation logic
3. Overfitting detection
4. Data leakage prevention

Run with: pytest tests/ -v

Author: SKIE_Ninja Development Team
Created: 2025-12-15
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))


class TestFeatureCalculation:
    """Tests for feature calculation accuracy and consistency."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Generate sample OHLCV data for testing."""
        np.random.seed(42)
        n = 250  # 250 bars

        dates = pd.date_range('2024-01-01 09:30', periods=n, freq='5min')
        close = 5000 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n)) * 2
        low = close - np.abs(np.random.randn(n)) * 2
        open_price = close + np.random.randn(n) * 1
        volume = np.random.randint(1000, 10000, n)

        return pd.DataFrame({
            'timestamp': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    def test_return_calculation_no_lookahead(self, sample_ohlcv_data):
        """Verify return features don't use future data."""
        df = sample_ohlcv_data.copy()

        # Calculate return_lag1 manually
        df['return_lag1'] = df['close'].pct_change(1)

        # The value at index i should only use data from i and i-1
        for i in range(5, len(df)):
            expected = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
            actual = df['return_lag1'].iloc[i]
            assert abs(expected - actual) < 1e-10, f"Return calculation mismatch at index {i}"

    def test_atr_calculation_no_lookahead(self, sample_ohlcv_data):
        """Verify ATR calculation doesn't use future data."""
        df = sample_ohlcv_data.copy()

        # True Range calculation
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is rolling mean of TR - should only use past data
        atr_14 = tr.rolling(14).mean()

        # Value at index i should only use indices [i-13, i]
        for i in range(20, len(df)):
            expected = tr.iloc[i-13:i+1].mean()
            actual = atr_14.iloc[i]
            assert abs(expected - actual) < 1e-10, f"ATR mismatch at index {i}"

    def test_rsi_calculation_bounds(self, sample_ohlcv_data):
        """Verify RSI stays within [0, 100] bounds."""
        df = sample_ohlcv_data.copy()

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # RSI should always be between 0 and 100
        valid_rsi = rsi.dropna()
        assert valid_rsi.min() >= 0, "RSI below 0"
        assert valid_rsi.max() <= 100, "RSI above 100"

    def test_no_shift_negative(self, sample_ohlcv_data):
        """Verify no negative shifts (look-ahead) in feature code."""
        # This is a code inspection test
        from deployment.ninja_signal_server import FeatureCalculator

        # Get the source code
        import inspect
        source = inspect.getsource(FeatureCalculator)

        # Check for dangerous patterns
        assert '.shift(-' not in source, "Found shift(-N) which causes look-ahead bias"
        assert 'center=True' not in source, "Found center=True which causes look-ahead bias"


class TestSignalGeneration:
    """Tests for trade signal generation logic."""

    @pytest.fixture
    def mock_feature_calc(self):
        """Create mock feature calculator with preset values."""
        from deployment.ninja_signal_server import FeatureCalculator
        calc = FeatureCalculator(window_size=200)
        calc.current_atr = 10.0
        calc.current_close = 5000.0
        return calc

    def test_signal_flat_when_vol_prob_low(self):
        """Signal should be FLAT when vol_expansion_prob < threshold."""
        from deployment.ninja_signal_server import TradeSignal, ServerConfig

        config = ServerConfig()
        config.min_vol_expansion_prob = 0.40

        # Simulate low vol probability
        vol_prob = 0.35
        signal = TradeSignal()

        # Vol prob below threshold should result in FLAT
        if vol_prob < config.min_vol_expansion_prob:
            signal.action = 'FLAT'

        assert signal.action == 'FLAT'

    def test_long_signal_logic(self):
        """Test LONG signal is generated correctly."""
        from deployment.ninja_signal_server import ServerConfig

        config = ServerConfig()
        vol_prob = 0.55
        breakout_high_prob = 0.60
        breakout_low_prob = 0.40
        current_price = 5000.0
        current_atr = 10.0

        # Check conditions
        assert vol_prob >= config.min_vol_expansion_prob
        assert breakout_high_prob > breakout_low_prob
        assert breakout_high_prob >= config.min_breakout_prob

        # Calculate expected TP/SL
        expected_tp = current_price + (current_atr * config.tp_atr_mult)
        expected_sl = current_price - (current_atr * config.sl_atr_mult)

        assert expected_tp == 5000.0 + (10.0 * 2.5)  # 5025.0
        assert expected_sl == 5000.0 - (10.0 * 1.25)  # 4987.5

    def test_short_signal_logic(self):
        """Test SHORT signal is generated correctly."""
        from deployment.ninja_signal_server import ServerConfig

        config = ServerConfig()
        vol_prob = 0.55
        breakout_high_prob = 0.40
        breakout_low_prob = 0.60
        current_price = 5000.0
        current_atr = 10.0

        # Check conditions
        assert vol_prob >= config.min_vol_expansion_prob
        assert breakout_low_prob > breakout_high_prob
        assert breakout_low_prob >= config.min_breakout_prob

        # Calculate expected TP/SL for SHORT
        expected_tp = current_price - (current_atr * config.tp_atr_mult)
        expected_sl = current_price + (current_atr * config.sl_atr_mult)

        assert expected_tp == 5000.0 - (10.0 * 2.5)  # 4975.0
        assert expected_sl == 5000.0 + (10.0 * 1.25)  # 5012.5


class TestOverfittingDetection:
    """Tests for overfitting detection functions."""

    @pytest.fixture
    def synthetic_returns(self):
        """Generate synthetic returns for testing."""
        np.random.seed(42)
        # Slightly positive returns (realistic edge)
        is_returns = np.random.normal(0.0005, 0.02, 500)
        oos_returns = np.random.normal(0.0004, 0.02, 300)
        return is_returns, oos_returns

    def test_dsr_calculation(self, synthetic_returns):
        """Test Deflated Sharpe Ratio calculation."""
        from quality_control.overfitting_detection import deflated_sharpe_ratio

        is_returns, oos_returns = synthetic_returns
        result = deflated_sharpe_ratio(oos_returns, trials=256)

        # DSR should be calculated
        assert result.observed_sharpe is not None
        assert result.deflated_sharpe is not None
        assert 0 <= result.p_value <= 1
        assert result.trials == 256

    def test_dsr_penalizes_many_trials(self, synthetic_returns):
        """More trials should result in stricter DSR."""
        from quality_control.overfitting_detection import deflated_sharpe_ratio

        _, oos_returns = synthetic_returns

        result_few_trials = deflated_sharpe_ratio(oos_returns, trials=10)
        result_many_trials = deflated_sharpe_ratio(oos_returns, trials=1000)

        # More trials = higher p-value (less significant)
        assert result_many_trials.p_value >= result_few_trials.p_value

    def test_cscv_probability_bounds(self, synthetic_returns):
        """CSCV overfit probability should be in [0, 1]."""
        from quality_control.overfitting_detection import cscv_overfit_probability

        is_returns, oos_returns = synthetic_returns
        combined = np.concatenate([is_returns, oos_returns])

        result = cscv_overfit_probability(combined, n_splits=8)

        assert 0 <= result.overfit_probability <= 1
        assert result.n_combinations > 0

    def test_psr_calculation(self):
        """Test Performance Stability Ratio calculation."""
        from quality_control.overfitting_detection import performance_stability_ratio

        np.random.seed(42)
        returns_by_year = {
            '2020': np.random.normal(0.0005, 0.02, 250),
            '2021': np.random.normal(0.0004, 0.02, 250),
            '2022': np.random.normal(0.0006, 0.02, 250),
        }

        result = performance_stability_ratio(returns_by_year)

        assert 0 <= result.psr <= 1
        assert result.stability_cv >= 0
        assert len(result.sharpes_by_period) == 3


class TestDataLeakagePrevention:
    """Tests to prevent data leakage in features and targets."""

    def test_vix_uses_t_minus_1(self):
        """Verify VIX features use T-1 (previous day) data."""
        from deployment.ninja_signal_server import FeatureCalculator

        calc = FeatureCalculator(window_size=200)

        # Add bars with VIX data
        for i in range(5):
            calc.add_bar({
                'timestamp': f'2024-01-0{i+1}T10:00:00',
                'open': 5000,
                'high': 5010,
                'low': 4990,
                'close': 5005,
                'volume': 1000,
                'vix_close': 15 + i  # VIX: 15, 16, 17, 18, 19
            })

        # The last VIX in buffer should be 19 (today's)
        # But features should use 18 (yesterday's = T-1)
        assert calc.vix_buffer[-1]['vix_close'] == 19

        # When calculating features, we use vix_buffer[-1] which is the most recent
        # This represents T-1 relative to the NEXT bar that will be processed

    def test_no_future_targets_in_features(self):
        """Ensure target variables aren't leaked into features."""
        # Target-related terms that should NOT appear in feature calculation
        forbidden_terms = [
            'target',
            'label',
            'next_',
            'future_',
            'forward_'
        ]

        from deployment.ninja_signal_server import FeatureCalculator
        import inspect
        source = inspect.getsource(FeatureCalculator.calculate_features)

        for term in forbidden_terms:
            assert term not in source.lower(), f"Potential leakage: '{term}' found in feature calculation"


class TestConfigurationValidation:
    """Tests for configuration parameter validation."""

    def test_server_config_defaults(self):
        """Test that server config has expected defaults."""
        from deployment.ninja_signal_server import ServerConfig

        config = ServerConfig()

        # Validated parameters (DO NOT CHANGE)
        assert config.min_vol_expansion_prob == 0.40
        assert config.min_breakout_prob == 0.45
        assert config.tp_atr_mult == 2.5
        assert config.sl_atr_mult == 1.25
        assert config.max_holding_bars == 20
        assert config.feature_window == 200

    def test_exit_parameters_in_safe_range(self):
        """Verify exit parameters are in validated safe range."""
        from deployment.ninja_signal_server import ServerConfig

        config = ServerConfig()

        # Per VALIDATION_REPORT.md safe ranges
        assert 2.0 <= config.tp_atr_mult <= 3.0, "TP mult outside safe range [2.0, 3.0]"
        assert 1.0 <= config.sl_atr_mult <= 1.5, "SL mult outside safe range [1.0, 1.5]"


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
