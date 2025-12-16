"""
SKIE Ninja Signal Server for NinjaTrader Integration
=====================================================

TCP socket server that receives bar data from NinjaTrader and returns trade signals.
This preserves the validated Python strategy exactly as tested.

Architecture:
    NinjaTrader 8 -> TCP Socket (localhost:5555) -> This Server -> Trade Signals

Usage:
    python ninja_signal_server.py [--port 5555] [--mode paper|live]

Author: SKIE_Ninja Development Team
Created: 2025-12-15
"""

import socket
import json
import logging
import pickle
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import sys
import argparse

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = 'localhost'
    port: int = 5555
    mode: str = 'paper'  # 'paper' or 'live'
    buffer_size: int = 4096
    heartbeat_interval: int = 30

    # Strategy parameters (validated - DO NOT CHANGE)
    min_vol_expansion_prob: float = 0.40
    min_breakout_prob: float = 0.45
    tp_atr_mult: float = 2.5
    sl_atr_mult: float = 1.25
    max_holding_bars: int = 20

    # Feature window
    feature_window: int = 200  # bars needed for feature calculation


@dataclass
class TradeSignal:
    """Trade signal response."""
    type: str = 'SIGNAL'
    action: str = 'FLAT'  # LONG, SHORT, FLAT
    confidence: float = 0.0
    vol_expansion_prob: float = 0.0
    breakout_prob: float = 0.0
    tp_price: float = 0.0
    sl_price: float = 0.0
    contracts: int = 0
    reason: str = ''
    timestamp: str = ''


class FeatureCalculator:
    """Real-time feature calculation matching validated Python implementation."""

    # VIX thresholds (from MacroMicro research)
    VIX_FEAR_THRESHOLD = 25.0
    VIX_EXTREME_FEAR_THRESHOLD = 30.0
    VIX_COMPLACENCY_THRESHOLD = 15.0

    # PCR thresholds
    PCR_BULLISH_THRESHOLD = 1.1
    PCR_BEARISH_THRESHOLD = 0.8

    # AAII historical averages
    AAII_HIST_BULLISH_AVG = 37.5
    AAII_HIST_BEARISH_AVG = 31.0

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.bar_buffer = []
        self.vix_buffer = []
        self.historical_vix = None  # For percentile calculation
        self.feature_names = []  # Track feature order

    def load_historical_vix(self, vix_path: Path) -> bool:
        """Load historical VIX data for percentile calculation."""
        try:
            if vix_path.exists():
                df = pd.read_csv(vix_path)
                df.columns = df.columns.str.lower()
                date_col = 'date' if 'date' in df.columns else df.columns[0]
                df['date'] = pd.to_datetime(df[date_col])
                close_col = 'close' if 'close' in df.columns else 'adj close'
                self.historical_vix = df[close_col].dropna().values
                logger.info(f"Loaded {len(self.historical_vix)} days of historical VIX data")
                return True
        except Exception as e:
            logger.warning(f"Could not load historical VIX: {e}")
        return False

    def add_bar(self, bar: Dict) -> bool:
        """Add new bar to buffer. Returns True if enough data for features."""
        self.bar_buffer.append({
            'timestamp': bar['timestamp'],
            'open': bar['open'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close'],
            'volume': bar.get('volume', 0)
        })

        if bar.get('vix_close'):
            self.vix_buffer.append({
                'timestamp': bar['timestamp'],
                'vix_close': bar['vix_close']
            })

        # Keep only window_size bars
        if len(self.bar_buffer) > self.window_size:
            self.bar_buffer = self.bar_buffer[-self.window_size:]
        if len(self.vix_buffer) > self.window_size:
            self.vix_buffer = self.vix_buffer[-self.window_size:]

        return len(self.bar_buffer) >= self.window_size

    def calculate_features(self) -> Optional[np.ndarray]:
        """Calculate features from buffer. Returns feature vector or None."""
        if len(self.bar_buffer) < self.window_size:
            return None

        df = pd.DataFrame(self.bar_buffer)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        features = {}

        # ===== RETURNS (lagged) =====
        for lag in [1, 2, 3, 5, 10, 20]:
            features[f'return_lag{lag}'] = df['close'].pct_change(lag).iloc[-1]

        # ===== ATR CALCULATION =====
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        for period in [5, 10, 14, 20]:
            features[f'rv_{period}'] = df['close'].pct_change().rolling(period).std().iloc[-1]
            features[f'atr_{period}'] = tr.rolling(period).mean().iloc[-1]
            features[f'atr_pct_{period}'] = features[f'atr_{period}'] / df['close'].iloc[-1]

        # ===== PRICE POSITION =====
        for period in [10, 20, 50]:
            features[f'close_vs_high_{period}'] = (
                df['close'].iloc[-1] - df['high'].rolling(period).max().iloc[-1]
            ) / df['close'].iloc[-1]
            features[f'close_vs_low_{period}'] = (
                df['close'].iloc[-1] - df['low'].rolling(period).min().iloc[-1]
            ) / df['close'].iloc[-1]
            ma = df['close'].rolling(period).mean().iloc[-1]
            features[f'close_vs_ma_{period}'] = (df['close'].iloc[-1] - ma) / ma

        # ===== MOMENTUM =====
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['close'].pct_change(period).iloc[-1]
            ma = df['close'].rolling(period).mean().iloc[-1]
            features[f'ma_dist_{period}'] = (df['close'].iloc[-1] - ma) / ma

        # ===== RSI =====
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean().iloc[-1]
            rs = gain / (loss + 1e-10)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Stochastic
        low_14 = df['low'].rolling(14).min().iloc[-1]
        high_14 = df['high'].rolling(14).max().iloc[-1]
        features['stoch_k_14'] = 100 * (df['close'].iloc[-1] - low_14) / (high_14 - low_14 + 1e-10)

        # ===== BOLLINGER BANDS =====
        mid = df['close'].rolling(20).mean().iloc[-1]
        std = df['close'].rolling(20).std().iloc[-1]
        upper = mid + 2 * std
        lower = mid - 2 * std
        features['bb_pct_20'] = (df['close'].iloc[-1] - lower) / (upper - lower + 1e-10)

        # ===== VOLUME FEATURES =====
        for period in [10, 20]:
            vol_sma = df['volume'].rolling(period).mean().iloc[-1]
            features[f'volume_ma_ratio_{period}'] = df['volume'].iloc[-1] / (vol_sma + 1)
        features['volume_std_10'] = df['volume'].rolling(10).std().iloc[-1]

        # ===== TIME FEATURES =====
        current_time = df.index[-1]
        hour = current_time.hour + current_time.minute / 60
        dow = current_time.dayofweek
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['dow_sin'] = np.sin(2 * np.pi * dow / 5)
        features['dow_cos'] = np.cos(2 * np.pi * dow / 5)

        # ===== VIX SENTIMENT FEATURES (T-1 lag) =====
        # FIXED: Uses T-1 (previous day) to match backtest methodology
        if len(self.vix_buffer) > 0:
            vix_close = self.vix_buffer[-1]['vix_close']  # T-1 (previous day's close)

            # Core VIX features
            features['sent_vix_close'] = vix_close

            # VIX moving averages (from buffer or use current as proxy)
            vix_values = [v['vix_close'] for v in self.vix_buffer]
            if len(vix_values) >= 5:
                features['sent_vix_ma5'] = np.mean(vix_values[-5:])
            else:
                features['sent_vix_ma5'] = vix_close

            if len(vix_values) >= 10:
                features['sent_vix_ma10'] = np.mean(vix_values[-10:])
            else:
                features['sent_vix_ma10'] = vix_close

            if len(vix_values) >= 20:
                features['sent_vix_ma20'] = np.mean(vix_values[-20:])
            else:
                features['sent_vix_ma20'] = vix_close

            # VIX vs moving averages
            features['sent_vix_vs_ma10'] = vix_close / (features['sent_vix_ma10'] + 1e-10)
            features['sent_vix_vs_ma20'] = vix_close / (features['sent_vix_ma20'] + 1e-10)

            # VIX percentile (FIXED: use historical reference, not buffer)
            if self.historical_vix is not None and len(self.historical_vix) > 0:
                features['sent_vix_percentile_20d'] = (
                    np.sum(self.historical_vix < vix_close) / len(self.historical_vix)
                )
            elif len(vix_values) >= 20:
                # Fallback: use buffer if no historical data
                features['sent_vix_percentile_20d'] = (
                    sum(1 for v in vix_values[-20:] if v < vix_close) / 20
                )
            else:
                features['sent_vix_percentile_20d'] = 0.5

            # Regime indicators
            features['sent_vix_fear_regime'] = int(vix_close > self.VIX_FEAR_THRESHOLD)
            features['sent_vix_extreme_fear'] = int(vix_close > self.VIX_EXTREME_FEAR_THRESHOLD)
            features['sent_vix_complacency'] = int(vix_close < self.VIX_COMPLACENCY_THRESHOLD)

            # VIX spike detection (15% increase)
            if len(vix_values) >= 2:
                vix_pct_change = (vix_close - vix_values[-2]) / (vix_values[-2] + 1e-10)
                features['sent_vix_spike'] = int(vix_pct_change > 0.15)
            else:
                features['sent_vix_spike'] = 0

            # Normalized sentiment (-1 = extreme fear, +1 = extreme complacency)
            features['sent_vix_sentiment'] = -np.clip((vix_close - 20) / 15, -1, 1)

            # Contrarian signal (high VIX = bullish, low VIX = bearish)
            features['sent_vix_contrarian_signal'] = features['sent_vix_sentiment']

            # ===== AAII PROXY (VIX-based) =====
            vix_pct = features['sent_vix_percentile_20d']
            features['sent_aaii_bearish'] = 25 + (vix_pct * 30)  # Range: 25-55%
            features['sent_aaii_bullish'] = 50 - (vix_pct * 30)  # Range: 20-50%
            features['sent_aaii_spread'] = features['sent_aaii_bullish'] - features['sent_aaii_bearish']
            features['sent_aaii_extreme_bullish'] = int(features['sent_aaii_bullish'] > 50)
            features['sent_aaii_extreme_bearish'] = int(features['sent_aaii_bearish'] > 50)
            features['sent_aaii_contrarian_signal'] = (
                features['sent_aaii_bearish'] - features['sent_aaii_bullish']
            ) / 100

            # ===== PCR PROXY (VIX-based) =====
            vix_normalized = np.clip((vix_close - 12) / 25, 0, 1)
            features['sent_pcr_total'] = 0.6 + (vix_normalized * 0.7)  # Range: 0.6-1.3
            features['sent_pcr_bullish_extreme'] = int(features['sent_pcr_total'] > self.PCR_BULLISH_THRESHOLD)
            features['sent_pcr_bearish_extreme'] = int(features['sent_pcr_total'] < self.PCR_BEARISH_THRESHOLD)
            features['sent_pcr_contrarian_signal'] = np.clip(
                (features['sent_pcr_total'] - 0.95) / 0.35, -1, 1
            )

            # ===== COMPOSITE SENTIMENT =====
            features['sent_composite_contrarian'] = np.mean([
                features['sent_vix_contrarian_signal'],
                features['sent_aaii_contrarian_signal'],
                features['sent_pcr_contrarian_signal']
            ])

            # Sentiment regime
            features['sent_fear_regime'] = int(
                features['sent_vix_fear_regime'] == 1 or
                features['sent_aaii_extreme_bearish'] == 1 or
                features['sent_pcr_bullish_extreme'] == 1
            )
            features['sent_greed_regime'] = int(
                features['sent_vix_complacency'] == 1 or
                features['sent_aaii_extreme_bullish'] == 1 or
                features['sent_pcr_bearish_extreme'] == 1
            )
        else:
            # Default sentiment features when no VIX data
            self._add_default_sentiment_features(features)

        # Store feature names for validation
        self.feature_names = list(features.keys())

        # Store current ATR for exit calculation
        self.current_atr = features.get('atr_14', 0)
        self.current_close = df['close'].iloc[-1]

        # Handle NaN values
        feature_array = np.array(list(features.values()))
        feature_array = np.nan_to_num(feature_array, nan=0.0)

        return feature_array

    def _add_default_sentiment_features(self, features: Dict):
        """Add default/neutral sentiment features when VIX data unavailable."""
        features['sent_vix_close'] = 20.0
        features['sent_vix_ma5'] = 20.0
        features['sent_vix_ma10'] = 20.0
        features['sent_vix_ma20'] = 20.0
        features['sent_vix_vs_ma10'] = 1.0
        features['sent_vix_vs_ma20'] = 1.0
        features['sent_vix_percentile_20d'] = 0.5
        features['sent_vix_fear_regime'] = 0
        features['sent_vix_extreme_fear'] = 0
        features['sent_vix_complacency'] = 0
        features['sent_vix_spike'] = 0
        features['sent_vix_sentiment'] = 0.0
        features['sent_vix_contrarian_signal'] = 0.0
        features['sent_aaii_bearish'] = 40.0
        features['sent_aaii_bullish'] = 35.0
        features['sent_aaii_spread'] = -5.0
        features['sent_aaii_extreme_bullish'] = 0
        features['sent_aaii_extreme_bearish'] = 0
        features['sent_aaii_contrarian_signal'] = 0.05
        features['sent_pcr_total'] = 0.95
        features['sent_pcr_bullish_extreme'] = 0
        features['sent_pcr_bearish_extreme'] = 0
        features['sent_pcr_contrarian_signal'] = 0.0
        features['sent_composite_contrarian'] = 0.0
        features['sent_fear_regime'] = 0
        features['sent_greed_regime'] = 0

    def get_current_price(self) -> float:
        """Get current close price."""
        return self.current_close if hasattr(self, 'current_close') else 0

    def get_current_atr(self) -> float:
        """Get current ATR for exit calculation."""
        return self.current_atr if hasattr(self, 'current_atr') else 0


class SignalServer:
    """TCP socket server for NinjaTrader communication."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.feature_calc = FeatureCalculator(config.feature_window)
        self.models = {}
        self.scaler = None
        self.running = False
        self.current_position = 0

        self._load_models()
        self._load_historical_vix()

    def _load_models(self):
        """Load trained models from production directory."""
        models_dir = project_root / 'models' / 'production'

        model_files = {
            'vol_expansion': 'vol_expansion_model.pkl',
            'breakout_high': 'breakout_high_model.pkl',
            'breakout_low': 'breakout_low_model.pkl',
            'atr_forecast': 'atr_forecast_model.pkl',
            'sentiment_vol': 'sentiment_vol_model.pkl'
        }

        for name, filename in model_files.items():
            path = models_dir / filename
            if path.exists():
                with open(path, 'rb') as f:
                    self.models[name] = pickle.load(f)
                logger.info(f"Loaded model: {name}")
            else:
                logger.warning(f"Model not found: {path}")

        # Load scaler
        scaler_path = models_dir / 'feature_scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Loaded feature scaler")

    def _load_historical_vix(self):
        """Load historical VIX data for percentile calculation."""
        vix_path = project_root / 'data' / 'raw' / 'market' / 'VIX_daily.csv'
        if self.feature_calc.load_historical_vix(vix_path):
            logger.info("Historical VIX data loaded for percentile calculation")
        else:
            logger.warning("Historical VIX not loaded - percentile will use buffer fallback")

    def generate_signal(self, bar_data: Dict) -> TradeSignal:
        """Generate trade signal from bar data."""
        signal = TradeSignal(timestamp=datetime.now().isoformat())

        # Add bar to buffer
        has_enough_data = self.feature_calc.add_bar(bar_data)

        if not has_enough_data:
            signal.reason = f"Warming up: {len(self.feature_calc.bar_buffer)}/{self.config.feature_window} bars"
            return signal

        # Calculate features
        features = self.feature_calc.calculate_features()
        if features is None:
            signal.reason = "Feature calculation failed"
            return signal

        # Feature validation - ensure count matches trained model
        features = features.reshape(1, -1)
        if self.scaler:
            expected_features = len(self.scaler.feature_names_in_)
            actual_features = features.shape[1]
            if expected_features != actual_features:
                logger.error(f"Feature mismatch: got {actual_features}, expected {expected_features}")
                logger.error(f"Expected features: {list(self.scaler.feature_names_in_)}")
                signal.reason = f"Feature mismatch: {actual_features} vs {expected_features} expected"
                return signal
            features = self.scaler.transform(features)

        # Get model predictions
        vol_prob = 0.5
        breakout_high_prob = 0.5
        breakout_low_prob = 0.5

        if 'vol_expansion' in self.models:
            vol_prob = self.models['vol_expansion'].predict_proba(features)[0, 1]

        if 'breakout_high' in self.models:
            breakout_high_prob = self.models['breakout_high'].predict_proba(features)[0, 1]

        if 'breakout_low' in self.models:
            breakout_low_prob = self.models['breakout_low'].predict_proba(features)[0, 1]

        signal.vol_expansion_prob = float(vol_prob)

        # Check vol expansion threshold
        if vol_prob < self.config.min_vol_expansion_prob:
            signal.reason = f"Vol expansion prob {vol_prob:.2f} < {self.config.min_vol_expansion_prob}"
            return signal

        # Determine direction
        current_price = self.feature_calc.get_current_price()
        current_atr = self.feature_calc.get_current_atr()

        if breakout_high_prob > breakout_low_prob and breakout_high_prob >= self.config.min_breakout_prob:
            signal.action = 'LONG'
            signal.breakout_prob = float(breakout_high_prob)
            signal.tp_price = current_price + (current_atr * self.config.tp_atr_mult)
            signal.sl_price = current_price - (current_atr * self.config.sl_atr_mult)
            signal.contracts = 1
            signal.confidence = (vol_prob + breakout_high_prob) / 2
            signal.reason = f"Vol expansion ({vol_prob:.2f}) + bullish breakout ({breakout_high_prob:.2f})"

        elif breakout_low_prob > breakout_high_prob and breakout_low_prob >= self.config.min_breakout_prob:
            signal.action = 'SHORT'
            signal.breakout_prob = float(breakout_low_prob)
            signal.tp_price = current_price - (current_atr * self.config.tp_atr_mult)
            signal.sl_price = current_price + (current_atr * self.config.sl_atr_mult)
            signal.contracts = 1
            signal.confidence = (vol_prob + breakout_low_prob) / 2
            signal.reason = f"Vol expansion ({vol_prob:.2f}) + bearish breakout ({breakout_low_prob:.2f})"
        else:
            signal.reason = f"No clear breakout direction (high: {breakout_high_prob:.2f}, low: {breakout_low_prob:.2f})"

        return signal

    def handle_client(self, conn: socket.socket, addr: Tuple[str, int]):
        """Handle client connection."""
        logger.info(f"Client connected: {addr}")

        try:
            while self.running:
                data = conn.recv(self.config.buffer_size)
                if not data:
                    break

                try:
                    request = json.loads(data.decode('utf-8'))

                    if request.get('type') == 'BAR_UPDATE':
                        # Update position tracking
                        self.current_position = request.get('position', 0)

                        # Generate signal
                        signal = self.generate_signal(request)

                        # Don't signal same direction if already in position
                        if self.current_position > 0 and signal.action == 'LONG':
                            signal.action = 'FLAT'
                            signal.reason = 'Already long'
                        elif self.current_position < 0 and signal.action == 'SHORT':
                            signal.action = 'FLAT'
                            signal.reason = 'Already short'

                        response = asdict(signal)

                    elif request.get('type') == 'HEARTBEAT':
                        response = {'type': 'HEARTBEAT_ACK', 'timestamp': datetime.now().isoformat()}

                    elif request.get('type') == 'STATUS':
                        response = {
                            'type': 'STATUS',
                            'mode': self.config.mode,
                            'models_loaded': list(self.models.keys()),
                            'buffer_size': len(self.feature_calc.bar_buffer),
                            'current_position': self.current_position
                        }
                    else:
                        response = {'type': 'ERROR', 'message': f"Unknown request type: {request.get('type')}"}

                    conn.sendall(json.dumps(response).encode('utf-8'))

                except json.JSONDecodeError as e:
                    error_response = {'type': 'ERROR', 'message': f'Invalid JSON: {str(e)}'}
                    conn.sendall(json.dumps(error_response).encode('utf-8'))

        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            conn.close()
            logger.info(f"Client disconnected: {addr}")

    def start(self):
        """Start the server."""
        self.running = True

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.config.host, self.config.port))
            server.listen(1)
            server.settimeout(1.0)  # Allow checking self.running

            logger.info(f"SKIE Ninja Signal Server started on {self.config.host}:{self.config.port}")
            logger.info(f"Mode: {self.config.mode.upper()}")
            logger.info(f"Models loaded: {list(self.models.keys())}")
            logger.info("Waiting for NinjaTrader connection...")

            while self.running:
                try:
                    conn, addr = server.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(conn, addr),
                        daemon=True
                    )
                    client_thread.start()
                except socket.timeout:
                    continue
                except KeyboardInterrupt:
                    break

        logger.info("Server stopped")

    def stop(self):
        """Stop the server."""
        self.running = False


def main():
    parser = argparse.ArgumentParser(description='SKIE Ninja Signal Server')
    parser.add_argument('--port', type=int, default=5555, help='Server port')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper', help='Trading mode')
    args = parser.parse_args()

    config = ServerConfig(port=args.port, mode=args.mode)
    server = SignalServer(config)

    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__':
    main()
