"""
SKIE_Ninja Signal Server
========================

TCP server that receives technical features from NinjaTrader and returns
trading signals using the walk-forward ML models.

This approach solves the PCR/AAII data availability issue:
- Python has access to all historical sentiment data used in training
- NT8 only sends 42 technical features
- Python calculates sentiment features and runs the full ensemble model
- Returns ShouldTrade, Direction, and probability signals

Architecture:
    NT8 Strategy  <--TCP:5555-->  Python Signal Server
       |                              |
       | 42 tech features             | Walk-forward models
       |                              | Historical sentiment
       v                              v
    Execute trades            Generate signals

Protocol (JSON):
    Request:  {"timestamp": "...", "features": [42 floats], "atr": float}
    Response: {"should_trade": bool, "direction": int, "vol_prob": float,
               "sent_prob": float, "breakout_prob": float, "contracts": int,
               "tp_offset": float, "sl_offset": float}

Author: SKIE_Ninja Development Team
Created: 2025-12-07
"""

import socket
import json
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import time
import signal
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.python.data_collection.historical_sentiment_loader import HistoricalSentimentLoader

logger = logging.getLogger(__name__)


class SignalServerConfig:
    """Configuration for the signal server."""

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 5555,
        model_path: Optional[Path] = None,
        use_walk_forward: bool = True,
        enable_sentiment: bool = True,
        # Thresholds - ALIGNED TO VALIDATED BACKTEST (ensemble_strategy.py:57-59)
        # Per NINJATRADER_DEPLOYMENT_AUDIT_20260106.md Section 4.3
        min_vol_prob: float = 0.50,       # Was 0.40, validated: 0.50
        min_sent_prob: float = 0.55,      # Unchanged
        min_breakout_prob: float = 0.50,  # Was 0.45, validated: 0.50
        ensemble_mode: str = 'agreement',
        # Position sizing
        tp_atr_mult: float = 2.5,
        sl_atr_mult: float = 1.25,
        base_contracts: int = 1,
        max_contracts: int = 3,
        # Short signals - DISABLED per audit (9.1% win rate)
        enable_short_signals: bool = False,
        # Diagnostic logging
        log_file: Optional[Path] = None,
        log_all_signals: bool = True,
    ):
        self.host = host
        self.port = port
        self.model_path = model_path or (project_root / 'data' / 'models' / 'walkforward_onnx')
        self.use_walk_forward = use_walk_forward
        self.enable_sentiment = enable_sentiment
        self.min_vol_prob = min_vol_prob
        self.min_sent_prob = min_sent_prob
        self.min_breakout_prob = min_breakout_prob
        self.ensemble_mode = ensemble_mode
        self.tp_atr_mult = tp_atr_mult
        self.sl_atr_mult = sl_atr_mult
        self.base_contracts = base_contracts
        self.max_contracts = max_contracts
        self.enable_short_signals = enable_short_signals
        self.log_file = log_file or (project_root / 'data' / 'logs' / 'signal_server.log')
        self.log_all_signals = log_all_signals


class WalkForwardModelManager:
    """Manages walk-forward model loading and switching."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.fold_schedule: List[Dict] = []
        self.current_fold: Optional[Dict] = None
        self.models: Dict = {}
        self.scalers: Dict = {}
        self._load_fold_schedule()

    def _load_fold_schedule(self):
        """Load the fold schedule from model directories."""
        fold_dirs = sorted(self.model_path.glob('fold_*'))
        logger.info(f"Found {len(fold_dirs)} walk-forward folds")

        for fold_dir in fold_dirs:
            config_path = fold_dir / 'strategy_config.json'
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)

                fold_info = config.get('fold_info', {})
                valid_from = fold_info.get('valid_from', '')
                valid_until = fold_info.get('valid_until', '')

                if valid_from and valid_until:
                    self.fold_schedule.append({
                        'path': fold_dir,
                        'valid_from': pd.to_datetime(valid_from).tz_localize(None),
                        'valid_until': pd.to_datetime(valid_until).tz_localize(None),
                        'config': config
                    })

        self.fold_schedule.sort(key=lambda x: x['valid_from'])
        logger.info(f"Loaded {len(self.fold_schedule)} fold schedules")

    def get_fold_for_date(self, dt: datetime) -> Optional[Dict]:
        """Get the appropriate fold for a given date."""
        dt = pd.Timestamp(dt).tz_localize(None)

        for fold in self.fold_schedule:
            if fold['valid_from'] <= dt < fold['valid_until']:
                return fold

        # If no exact match, use the latest fold
        if self.fold_schedule:
            return self.fold_schedule[-1]

        return None

    def load_models_for_fold(self, fold: Dict) -> bool:
        """Load ONNX models for a specific fold."""
        try:
            import onnxruntime as ort

            fold_path = fold['path']
            logger.info(f"Loading models from {fold_path}")

            # Load models
            self.models = {
                'vol': ort.InferenceSession(str(fold_path / 'vol_expansion_model.onnx')),
                'high': ort.InferenceSession(str(fold_path / 'breakout_high_model.onnx')),
                'low': ort.InferenceSession(str(fold_path / 'breakout_low_model.onnx')),
                'atr': ort.InferenceSession(str(fold_path / 'atr_forecast_model.onnx')),
            }

            # Load sentiment model if available
            sent_model_path = fold_path / 'sentiment_vol_model.onnx'
            if sent_model_path.exists():
                self.models['sentiment'] = ort.InferenceSession(str(sent_model_path))
                logger.info("  Loaded sentiment volatility model")
            else:
                self.models['sentiment'] = None
                logger.warning("  Sentiment model not found")

            # Load scalers
            with open(fold_path / 'scaler_params.json') as f:
                self.scalers['tech'] = json.load(f)

            sent_scaler_path = fold_path / 'sentiment_scaler_params.json'
            if sent_scaler_path.exists():
                with open(sent_scaler_path) as f:
                    self.scalers['sentiment'] = json.load(f)
            else:
                self.scalers['sentiment'] = None

            self.current_fold = fold
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def ensure_model_for_date(self, dt: datetime) -> bool:
        """Ensure correct model is loaded for the given date."""
        fold = self.get_fold_for_date(dt)

        if fold is None:
            logger.error(f"No fold found for date {dt}")
            return False

        if self.current_fold is None or fold['path'] != self.current_fold['path']:
            return self.load_models_for_fold(fold)

        return True


class SignalServer:
    """TCP server that generates trading signals for NinjaTrader."""

    def __init__(self, config: SignalServerConfig):
        self.config = config
        self.model_manager = WalkForwardModelManager(config.model_path)
        self.sentiment_loader = HistoricalSentimentLoader() if config.enable_sentiment else None
        self.sentiment_data: Optional[pd.DataFrame] = None

        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.signal_count = 0
        self.trade_count = 0
        self.start_time: Optional[datetime] = None

        # Diagnostic counters for uptime monitoring
        self.long_signals = 0
        self.short_signals = 0
        self.short_signals_blocked = 0
        self.filter_rejections = {
            'vol_filter': 0,
            'sent_filter': 0,
            'no_direction': 0,
            'short_disabled': 0,
        }
        self.last_signal_time: Optional[datetime] = None
        self.last_heartbeat: Optional[datetime] = None
        self.client_connections = 0
        self.client_disconnections = 0

        # Setup file logging for diagnostics
        self._setup_file_logging()

        # Load sentiment data if enabled
        if self.sentiment_loader:
            self._load_sentiment_data()

    def _setup_file_logging(self):
        """Setup file-based logging for diagnostics and audit trail."""
        log_dir = self.config.log_file.parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create file handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.config.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Log startup
        logger.info("=" * 60)
        logger.info("SIGNAL SERVER STARTUP")
        logger.info(f"  Host: {self.config.host}:{self.config.port}")
        logger.info(f"  Short signals: {'ENABLED' if self.config.enable_short_signals else 'DISABLED'}")
        logger.info(f"  Thresholds: vol={self.config.min_vol_prob}, sent={self.config.min_sent_prob}, breakout={self.config.min_breakout_prob}")
        logger.info("=" * 60)

    def _load_sentiment_data(self):
        """Pre-load historical sentiment data."""
        try:
            logger.info("Loading historical sentiment data...")
            self.sentiment_loader.load_all()

            # Create a date-indexed sentiment lookup
            vix_data = self.sentiment_loader.vix_data
            aaii_data = self.sentiment_loader.aaii_data
            pcr_data = self.sentiment_loader.pcr_data

            # Merge all sentiment data
            self.sentiment_data = vix_data.copy()

            if aaii_data is not None:
                for col in aaii_data.columns:
                    if col not in self.sentiment_data.columns:
                        self.sentiment_data[col] = aaii_data[col]

            if pcr_data is not None:
                for col in pcr_data.columns:
                    if col not in self.sentiment_data.columns:
                        self.sentiment_data[col] = pcr_data[col]

            self.sentiment_data = self.sentiment_data.ffill()

            logger.info(f"Loaded sentiment data: {len(self.sentiment_data)} days")
            logger.info(f"  Date range: {self.sentiment_data.index.min()} to {self.sentiment_data.index.max()}")

        except Exception as e:
            logger.error(f"Failed to load sentiment data: {e}")
            self.sentiment_data = None

    def get_sentiment_features(self, dt: datetime) -> Optional[np.ndarray]:
        """Get 28 sentiment features for a given date (using previous day)."""
        if self.sentiment_data is None:
            return None

        try:
            # Use previous day to avoid look-ahead bias
            prev_date = (pd.Timestamp(dt) - pd.Timedelta(days=1)).normalize()

            if prev_date not in self.sentiment_data.index:
                # Find closest prior date
                valid_dates = self.sentiment_data.index[self.sentiment_data.index <= prev_date]
                if len(valid_dates) == 0:
                    return None
                prev_date = valid_dates[-1]

            row = self.sentiment_data.loc[prev_date]

            # Build 28-feature array matching C# order
            features = np.array([
                row.get('vix_close', 20.0),
                row.get('vix_ma5', 20.0),
                row.get('vix_ma10', 20.0),
                row.get('vix_ma20', 20.0),
                row.get('vix_vs_ma10', 1.0),
                row.get('vix_vs_ma20', 1.0),
                row.get('vix_percentile_20d', 0.5),
                row.get('vix_fear_regime', 0.0),
                row.get('vix_extreme_fear', 0.0),
                row.get('vix_complacency', 0.0),
                row.get('vix_spike', 0.0),
                row.get('vix_sentiment', 0.0),
                row.get('vix_contrarian_signal', 0.0),
                row.get('aaii_bullish', 37.5),
                row.get('aaii_bearish', 31.0),
                row.get('aaii_spread', 6.5),
                row.get('aaii_extreme_bullish', 0.0),
                row.get('aaii_extreme_bearish', 0.0),
                row.get('aaii_contrarian_signal', 0.0),
                row.get('pcr_total', 0.95),
                row.get('pcr_ma5', 0.95),
                row.get('pcr_ma10', 0.95),
                row.get('pcr_bullish_extreme', 0.0),
                row.get('pcr_bearish_extreme', 0.0),
                row.get('pcr_contrarian_signal', 0.0),
                row.get('sent_composite_contrarian', 0.0) if 'sent_composite_contrarian' in row else 0.0,
                row.get('sent_fear_regime', 0.0) if 'sent_fear_regime' in row else 0.0,
                row.get('sent_greed_regime', 0.0) if 'sent_greed_regime' in row else 0.0,
            ], dtype=np.float32)

            return features

        except Exception as e:
            logger.error(f"Error getting sentiment features: {e}")
            return None

    def scale_features(self, features: np.ndarray, scaler_type: str = 'tech') -> np.ndarray:
        """Scale features using loaded scaler params."""
        scaler = self.model_manager.scalers.get(scaler_type)
        if scaler is None:
            return features

        # Handle different naming conventions
        means = scaler.get('mean_', scaler.get('means', []))
        scales = scaler.get('scale_', scaler.get('scales', scaler.get('stds', [])))

        if not means or not scales:
            return features

        means = np.array(means, dtype=np.float32)
        scales = np.array(scales, dtype=np.float32)

        # Handle length mismatch
        if len(means) != len(features):
            logger.warning(f"Scaler length mismatch: {len(means)} vs {len(features)}")
            return features

        return (features - means) / scales

    def run_classifier(self, model, features: np.ndarray) -> float:
        """Run classifier model and get probability."""
        if model is None:
            return 0.5

        try:
            input_data = features.reshape(1, -1).astype(np.float32)
            outputs = model.run(None, {'features': input_data})

            # LightGBM format: outputs[1] is probabilities
            if len(outputs) > 1:
                probs = outputs[1]
                if isinstance(probs, list) and len(probs) > 0:
                    prob_dict = probs[0]
                    if isinstance(prob_dict, dict) and 1 in prob_dict:
                        return float(prob_dict[1])

            return 0.5

        except Exception as e:
            logger.error(f"Classifier error: {e}")
            return 0.5

    def run_regressor(self, model, features: np.ndarray) -> float:
        """Run regressor model and get prediction."""
        if model is None:
            return 0.0

        try:
            input_data = features.reshape(1, -1).astype(np.float32)
            outputs = model.run(None, {'features': input_data})
            return float(outputs[0][0].item())

        except Exception as e:
            logger.error(f"Regressor error: {e}")
            return 0.0

    def generate_signal(self, request: Dict) -> Dict:
        """Generate trading signal from request."""
        self.signal_count += 1
        self.last_signal_time = datetime.now()

        try:
            # Parse request
            timestamp_str = request.get('timestamp', datetime.now().isoformat())
            features = np.array(request.get('features', []), dtype=np.float32)
            current_atr = float(request.get('atr', 1.0))

            if len(features) != 42:
                logger.warning(f"Signal #{self.signal_count}: REJECTED - Expected 42 features, got {len(features)}")
                return self._error_response(f"Expected 42 features, got {len(features)}")

            # Parse timestamp
            dt = pd.to_datetime(timestamp_str)
            if dt.tzinfo is not None:
                dt = dt.tz_localize(None)

            # Ensure correct model is loaded
            if not self.model_manager.ensure_model_for_date(dt):
                logger.error(f"Signal #{self.signal_count}: REJECTED - Failed to load model for date {dt}")
                return self._error_response("Failed to load model for date")

            # Scale technical features
            scaled_tech = self.scale_features(features, 'tech')

            # Debug: log first few raw features for diagnosis
            if self.signal_count <= 5:
                logger.info(f"  Raw features (first 10): {features[:10].tolist()}")
                logger.info(f"  Scaled features (first 10): {scaled_tech[:10].tolist()}")

            # Run technical models
            vol_prob = self.run_classifier(self.model_manager.models['vol'], scaled_tech)
            high_prob = self.run_classifier(self.model_manager.models['high'], scaled_tech)
            low_prob = self.run_classifier(self.model_manager.models['low'], scaled_tech)
            predicted_atr = self.run_regressor(self.model_manager.models['atr'], scaled_tech)

            # Technical filter
            tech_pass = vol_prob >= self.config.min_vol_prob

            # Sentiment filter
            sent_prob = -1.0
            sent_pass = True  # Default pass if no sentiment

            if self.config.enable_sentiment and self.model_manager.models.get('sentiment'):
                sent_features = self.get_sentiment_features(dt)

                if sent_features is not None and len(sent_features) == 28:
                    # Scale sentiment features
                    scaled_sent = self.scale_features(sent_features, 'sentiment')

                    # Combine features
                    combined = np.concatenate([scaled_tech, scaled_sent])

                    # Run sentiment model
                    sent_prob = self.run_classifier(self.model_manager.models['sentiment'], combined)
                    sent_pass = sent_prob >= self.config.min_sent_prob

            # Apply ensemble mode
            if self.config.ensemble_mode == 'agreement':
                passed_filters = tech_pass and sent_pass
            else:
                passed_filters = tech_pass

            # Diagnostic logging for all signals
            if self.config.log_all_signals:
                logger.info(f"Signal #{self.signal_count} @ {timestamp_str}: vol={vol_prob:.3f}, high={high_prob:.3f}, low={low_prob:.3f}, sent={sent_prob:.3f}, atr={predicted_atr:.2f}")

            if not passed_filters:
                # Track rejection reason
                if not tech_pass:
                    self.filter_rejections['vol_filter'] += 1
                    rejection_reason = f"vol_filter ({vol_prob:.3f} < {self.config.min_vol_prob})"
                else:
                    self.filter_rejections['sent_filter'] += 1
                    rejection_reason = f"sent_filter ({sent_prob:.3f} < {self.config.min_sent_prob})"

                if self.config.log_all_signals:
                    logger.info(f"  -> REJECTED: {rejection_reason}")

                return self._no_trade_response(vol_prob, sent_prob, rejection_reason)

            # Direction from breakout probabilities
            high_signal = high_prob >= self.config.min_breakout_prob
            low_signal = low_prob >= self.config.min_breakout_prob

            if high_signal and not low_signal:
                direction = 1
                breakout_prob = high_prob
            elif low_signal and not high_signal:
                direction = -1
                breakout_prob = low_prob
            elif high_signal and low_signal:
                if high_prob > low_prob:
                    direction = 1
                    breakout_prob = high_prob
                else:
                    direction = -1
                    breakout_prob = low_prob
            else:
                self.filter_rejections['no_direction'] += 1
                rejection_reason = f"no_direction (high={high_prob:.3f}, low={low_prob:.3f} < {self.config.min_breakout_prob})"
                if self.config.log_all_signals:
                    logger.info(f"  -> REJECTED: {rejection_reason}")
                return self._no_trade_response(vol_prob, sent_prob, rejection_reason)

            # SHORT SIGNAL BLOCKING - Per audit finding (9.1% win rate)
            if direction == -1 and not self.config.enable_short_signals:
                self.short_signals_blocked += 1
                self.filter_rejections['short_disabled'] += 1
                rejection_reason = "short_disabled (per audit: 9.1% win rate)"
                logger.info(f"  -> BLOCKED SHORT: {rejection_reason}")
                return self._no_trade_response(vol_prob, sent_prob, rejection_reason)

            # Position sizing and exits
            atr = predicted_atr if predicted_atr > 0 else current_atr
            tp_offset = atr * self.config.tp_atr_mult
            sl_offset = atr * self.config.sl_atr_mult

            self.trade_count += 1
            if direction == 1:
                self.long_signals += 1
            else:
                self.short_signals += 1

            direction_str = "LONG" if direction == 1 else "SHORT"
            logger.info(f"  -> TRADE SIGNAL: {direction_str} | vol={vol_prob:.3f}, breakout={breakout_prob:.3f}, tp={tp_offset:.2f}, sl={sl_offset:.2f}")

            return {
                'should_trade': True,
                'direction': direction,
                'vol_prob': float(vol_prob),
                'sent_prob': float(sent_prob),
                'breakout_prob': float(breakout_prob),
                'contracts': self.config.base_contracts,
                'tp_offset': float(tp_offset),
                'sl_offset': float(sl_offset),
                'predicted_atr': float(predicted_atr),
                'model_fold': str(self.model_manager.current_fold['path'].name) if self.model_manager.current_fold else 'unknown'
            }

        except Exception as e:
            logger.error(f"Signal #{self.signal_count}: ERROR - {e}")
            return self._error_response(str(e))

    def _no_trade_response(self, vol_prob: float, sent_prob: float, rejection_reason: str = "") -> Dict:
        """Response when filters don't pass."""
        return {
            'should_trade': False,
            'direction': 0,
            'vol_prob': float(vol_prob),
            'sent_prob': float(sent_prob),
            'breakout_prob': 0.0,
            'contracts': 0,
            'tp_offset': 0.0,
            'sl_offset': 0.0,
            'predicted_atr': 0.0,
            'model_fold': str(self.model_manager.current_fold['path'].name) if self.model_manager.current_fold else 'unknown',
            'rejection_reason': rejection_reason
        }

    def _error_response(self, error: str) -> Dict:
        """Response on error."""
        return {
            'should_trade': False,
            'direction': 0,
            'error': error,
            'vol_prob': 0.0,
            'sent_prob': 0.0
        }

    def handle_client(self, client_socket: socket.socket, address: tuple):
        """Handle a client connection."""
        self.client_connections += 1
        logger.info(f"CLIENT CONNECTED: {address} (total connections: {self.client_connections})")

        buffer = ""
        client_signal_count = 0
        client_start_time = datetime.now()

        try:
            while self.running:
                data = client_socket.recv(4096).decode('utf-8')
                if not data:
                    break

                buffer += data

                # Process complete JSON messages (newline-delimited)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        request = json.loads(line)
                        response = self.generate_signal(request)
                        response_json = json.dumps(response) + '\n'
                        client_socket.send(response_json.encode('utf-8'))
                        client_signal_count += 1

                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON from {address}: {e}")
                        error_response = json.dumps({'error': f'Invalid JSON: {e}'}) + '\n'
                        client_socket.send(error_response.encode('utf-8'))

        except Exception as e:
            logger.error(f"Client handler error for {address}: {e}")

        finally:
            client_socket.close()
            self.client_disconnections += 1
            session_duration = datetime.now() - client_start_time
            logger.info(f"CLIENT DISCONNECTED: {address} | session={session_duration} | signals={client_signal_count}")

    def _heartbeat_thread(self):
        """Background thread for periodic heartbeat logging and uptime monitoring."""
        heartbeat_interval = 300  # 5 minutes
        while self.running:
            time.sleep(heartbeat_interval)
            if not self.running:
                break

            self.last_heartbeat = datetime.now()
            uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)

            # Calculate time since last signal
            if self.last_signal_time:
                time_since_signal = datetime.now() - self.last_signal_time
                signal_gap_warning = time_since_signal.total_seconds() > 3600  # >1 hour
            else:
                time_since_signal = None
                signal_gap_warning = True

            heartbeat_msg = (
                f"HEARTBEAT | uptime={uptime} | signals={self.signal_count} | trades={self.trade_count} | "
                f"longs={self.long_signals} | shorts_blocked={self.short_signals_blocked} | "
                f"last_signal={'NONE' if time_since_signal is None else str(time_since_signal)}"
            )
            logger.info(heartbeat_msg)

            if signal_gap_warning and self.signal_count > 0:
                logger.warning(f"SIGNAL GAP WARNING: No signals for {time_since_signal or 'since startup'}")

    def start(self):
        """Start the signal server."""
        self.running = True
        self.start_time = datetime.now()

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.config.host, self.config.port))
        self.server_socket.listen(5)

        # Start heartbeat thread for uptime monitoring
        heartbeat = threading.Thread(target=self._heartbeat_thread, daemon=True)
        heartbeat.start()

        logger.info(f"Signal server started on {self.config.host}:{self.config.port}")
        print(f"\n{'='*60}")
        print(f"SKIE_Ninja Signal Server")
        print(f"{'='*60}")
        print(f"Address: {self.config.host}:{self.config.port}")
        print(f"Models: {self.config.model_path}")
        print(f"Folds loaded: {len(self.model_manager.fold_schedule)}")
        print(f"Sentiment: {'ENABLED' if self.config.enable_sentiment else 'DISABLED'}")
        print(f"Ensemble mode: {self.config.ensemble_mode}")
        print(f"Short signals: {'ENABLED' if self.config.enable_short_signals else 'DISABLED (per audit)'}")
        print(f"Thresholds: vol={self.config.min_vol_prob}, sent={self.config.min_sent_prob}, breakout={self.config.min_breakout_prob}")
        print(f"Log file: {self.config.log_file}")
        print(f"{'='*60}")
        print("Waiting for connections... (Ctrl+C to stop)")

        try:
            while self.running:
                self.server_socket.settimeout(1.0)
                try:
                    client_socket, address = self.server_socket.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except socket.timeout:
                    continue

        except KeyboardInterrupt:
            logger.info("Shutdown requested via Ctrl+C")

        finally:
            self.stop()

    def stop(self):
        """Stop the server."""
        self.running = False

        if self.server_socket:
            self.server_socket.close()

        uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)

        # Comprehensive shutdown diagnostics
        summary = f"""
{'='*60}
SIGNAL SERVER SHUTDOWN
{'='*60}
Uptime:              {uptime}
Last Signal:         {self.last_signal_time or 'None'}

CONNECTION STATISTICS
---------------------
Total Connections:   {self.client_connections}
Disconnections:      {self.client_disconnections}

SIGNAL STATISTICS
-----------------
Total Signals:       {self.signal_count}
Trade Signals:       {self.trade_count}
  Long Signals:      {self.long_signals}
  Short Signals:     {self.short_signals}
Shorts Blocked:      {self.short_signals_blocked}

REJECTION BREAKDOWN
-------------------
Vol Filter:          {self.filter_rejections['vol_filter']}
Sent Filter:         {self.filter_rejections['sent_filter']}
No Direction:        {self.filter_rejections['no_direction']}
Short Disabled:      {self.filter_rejections['short_disabled']}
Total Rejections:    {sum(self.filter_rejections.values())}

CONFIGURATION
-------------
Short Signals:       {'ENABLED' if self.config.enable_short_signals else 'DISABLED'}
Vol Threshold:       {self.config.min_vol_prob}
Sent Threshold:      {self.config.min_sent_prob}
Breakout Threshold:  {self.config.min_breakout_prob}
{'='*60}
"""
        print(summary)
        logger.info(summary.replace('\n', ' | '))


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    import argparse
    parser = argparse.ArgumentParser(description='SKIE_Ninja Signal Server')
    parser.add_argument('--host', default='127.0.0.1', help='Server host')
    parser.add_argument('--port', type=int, default=5555, help='Server port')
    parser.add_argument('--no-sentiment', action='store_true', help='Disable sentiment filter')
    args = parser.parse_args()

    config = SignalServerConfig(
        host=args.host,
        port=args.port,
        enable_sentiment=not args.no_sentiment
    )

    server = SignalServer(config)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    server.start()


if __name__ == "__main__":
    main()
