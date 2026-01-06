"""
SKIE-Ninja Walk-Forward Backtesting Framework

Comprehensive backtesting with:
- Walk-forward validation (train, validate, trade)
- Detailed trade metrics (entry/exit, P&L, drawdown, run-up)
- Summary statistics and KPIs
- Multi-model comparison

Author: SKIE_Ninja Development Team
Created: 2025-12-04
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import warnings

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)

# ES futures contract specs
ES_POINT_VALUE = 50  # $50 per point for ES
ES_TICK_SIZE = 0.25  # Minimum price movement
ES_TICK_VALUE = 12.50  # $12.50 per tick


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Trade parameters
    position_size: int = 1  # Number of contracts
    point_value: float = ES_POINT_VALUE
    tick_size: float = ES_TICK_SIZE

    # Entry/Exit rules
    signal_threshold: float = 0.5  # Probability threshold for entry
    hold_bars: int = 1  # Bars to hold position
    max_daily_trades: int = 10  # Max trades per day

    # Risk management
    stop_loss_ticks: Optional[int] = None  # Stop loss in ticks
    take_profit_ticks: Optional[int] = None  # Take profit in ticks
    max_drawdown_pct: float = 0.10  # Max drawdown before stopping

    # Walk-forward parameters
    train_days: int = 180
    test_days: int = 5
    # Embargo = max(feature_lookback, label_horizon) + safety_margin
    # Per Lopez de Prado (2018), Ch. 7: max(200, 30) + 10 = 210
    # PREVIOUSLY: 42 (INCORRECT - caused potential data leakage)
    embargo_bars: int = 210

    # Costs
    commission_per_trade: float = 2.50  # Per side, per contract
    slippage_ticks: float = 0.5  # Expected slippage in ticks


@dataclass
class Trade:
    """Individual trade record."""
    trade_id: int
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: int  # 1 = long, -1 = short
    size: int  # Number of contracts

    # P&L
    gross_pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    net_pnl: float = 0.0

    # Trade metrics
    bars_held: int = 0
    time_held: timedelta = field(default_factory=lambda: timedelta(0))
    max_favorable_excursion: float = 0.0  # Max run-up
    max_adverse_excursion: float = 0.0  # Max drawdown during trade

    # Metadata
    entry_signal: float = 0.0  # Model probability
    model_name: str = ""

    def calculate_metrics(self, prices: pd.Series, point_value: float):
        """Calculate MFE, MAE, and P&L metrics."""
        if len(prices) == 0:
            return

        self.bars_held = len(prices)

        if self.direction == 1:  # Long
            # Price change from entry
            price_changes = prices - self.entry_price
            self.max_favorable_excursion = max(0, price_changes.max()) * point_value * self.size
            self.max_adverse_excursion = min(0, price_changes.min()) * point_value * self.size
        else:  # Short
            price_changes = self.entry_price - prices
            self.max_favorable_excursion = max(0, price_changes.max()) * point_value * self.size
            self.max_adverse_excursion = min(0, price_changes.min()) * point_value * self.size

        # Calculate P&L
        price_diff = (self.exit_price - self.entry_price) * self.direction
        self.gross_pnl = price_diff * point_value * self.size
        self.net_pnl = self.gross_pnl - self.commission - self.slippage


@dataclass
class BacktestMetrics:
    """Summary metrics for backtest results."""
    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    long_trades: int = 0
    short_trades: int = 0

    # P&L metrics
    total_gross_pnl: float = 0.0
    total_net_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0

    # Win/Loss metrics
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    payoff_ratio: float = 0.0
    expectancy: float = 0.0

    # Drawdown metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_runup: float = 0.0
    avg_mae: float = 0.0
    avg_mfe: float = 0.0

    # Trade duration
    avg_bars_held: float = 0.0
    avg_time_held_minutes: float = 0.0
    max_bars_held: int = 0
    min_bars_held: int = 0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Time analysis
    total_days: int = 0
    trading_days: int = 0
    trades_per_day: float = 0.0

    # Model metrics
    accuracy: float = 0.0
    auc_roc: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'long_trades': self.long_trades,
            'short_trades': self.short_trades,
            'total_gross_pnl': round(self.total_gross_pnl, 2),
            'total_net_pnl': round(self.total_net_pnl, 2),
            'total_commission': round(self.total_commission, 2),
            'total_slippage': round(self.total_slippage, 2),
            'win_rate': round(self.win_rate * 100, 2),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'profit_factor': round(self.profit_factor, 2),
            'payoff_ratio': round(self.payoff_ratio, 2),
            'expectancy': round(self.expectancy, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct * 100, 2),
            'max_runup': round(self.max_runup, 2),
            'avg_mae': round(self.avg_mae, 2),
            'avg_mfe': round(self.avg_mfe, 2),
            'avg_bars_held': round(self.avg_bars_held, 1),
            'avg_time_held_minutes': round(self.avg_time_held_minutes, 1),
            'max_bars_held': self.max_bars_held,
            'min_bars_held': self.min_bars_held,
            'sharpe_ratio': round(self.sharpe_ratio, 3),
            'sortino_ratio': round(self.sortino_ratio, 3),
            'calmar_ratio': round(self.calmar_ratio, 3),
            'total_days': self.total_days,
            'trading_days': self.trading_days,
            'trades_per_day': round(self.trades_per_day, 2),
            'accuracy': round(self.accuracy * 100, 2),
            'auc_roc': round(self.auc_roc * 100, 2)
        }


class WalkForwardBacktester:
    """
    Walk-forward backtesting engine.

    Implements true walk-forward validation:
    1. Train model on historical window
    2. Generate predictions on test window
    3. Simulate trades based on predictions
    4. Roll forward and repeat
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.daily_returns: List[float] = []
        self.predictions: List[Tuple[datetime, float, int]] = []

    def run_backtest(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        target_col: str,
        selected_features: List[str],
        model_type: str = 'lightgbm',
        model_params: Optional[Dict] = None
    ) -> Tuple[List[Trade], BacktestMetrics]:
        """
        Run walk-forward backtest.

        Parameters:
        -----------
        prices : pd.DataFrame
            OHLCV price data with DatetimeIndex
        features : pd.DataFrame
            Feature matrix with same index as prices
        target_col : str
            Target column name
        selected_features : List[str]
            Feature columns to use
        model_type : str
            'lightgbm', 'xgboost', 'lstm', or 'gru'
        model_params : Dict, optional
            Model hyperparameters

        Returns:
        --------
        trades : List[Trade]
            List of all trades
        metrics : BacktestMetrics
            Summary performance metrics
        """
        logger.info("="*60)
        logger.info(f"WALK-FORWARD BACKTEST: {model_type.upper()}")
        logger.info("="*60)

        self.trades = []
        self.equity_curve = [0.0]  # Start with 0 P&L
        self.predictions = []

        # Validate data alignment
        common_idx = prices.index.intersection(features.index)
        prices = prices.loc[common_idx]
        features = features.loc[common_idx]

        logger.info(f"Data: {len(prices):,} bars from {prices.index[0]} to {prices.index[-1]}")

        # Calculate walk-forward windows
        bars_per_day = 78  # 5-min RTH bars
        train_bars = self.config.train_days * bars_per_day
        test_bars = self.config.test_days * bars_per_day
        embargo_bars = self.config.embargo_bars

        n_samples = len(prices)
        n_folds = (n_samples - train_bars - embargo_bars) // test_bars

        logger.info(f"Walk-forward: {n_folds} folds, {train_bars} train, {test_bars} test, {embargo_bars} embargo")

        trade_id = 0
        all_predictions = []
        all_actuals = []

        # Walk-forward loop
        for fold in range(n_folds):
            train_start = fold * test_bars
            train_end = train_start + train_bars
            test_start = train_end + embargo_bars
            test_end = min(test_start + test_bars, n_samples)

            if test_end > n_samples:
                break

            # Get train/test splits
            X_train = features.iloc[train_start:train_end][selected_features].values
            y_train = features.iloc[train_start:train_end][target_col].values
            X_test = features.iloc[test_start:test_end][selected_features].values
            y_test = features.iloc[test_start:test_end][target_col].values

            test_prices = prices.iloc[test_start:test_end]
            test_times = features.index[test_start:test_end]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model and get predictions
            y_proba = self._train_and_predict(
                X_train_scaled, y_train, X_test_scaled,
                model_type, model_params
            )

            all_predictions.extend(y_proba)
            all_actuals.extend(y_test)

            # Generate trades
            fold_trades = self._generate_trades(
                y_proba, test_prices, test_times, trade_id, model_type
            )

            trade_id += len(fold_trades)
            self.trades.extend(fold_trades)

            # Update equity curve
            for trade in fold_trades:
                self.equity_curve.append(self.equity_curve[-1] + trade.net_pnl)

            if (fold + 1) % 10 == 0:
                logger.info(f"Fold {fold + 1}/{n_folds}: {len(fold_trades)} trades, "
                           f"Cumulative P&L: ${self.equity_curve[-1]:,.2f}")

        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_actuals)

        logger.info(f"\nBacktest complete: {len(self.trades)} trades")
        logger.info(f"Total P&L: ${metrics.total_net_pnl:,.2f}")
        logger.info(f"Win Rate: {metrics.win_rate*100:.1f}%")
        logger.info(f"Sharpe: {metrics.sharpe_ratio:.2f}")

        return self.trades, metrics

    def _train_and_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        model_type: str,
        model_params: Optional[Dict]
    ) -> np.ndarray:
        """Train model and generate predictions."""

        if model_type == 'lightgbm':
            import lightgbm as lgb
            params = model_params or {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'verbose': -1
            }

            train_data = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(params, train_data, num_boost_round=100)
            y_proba = model.predict(X_test)

        elif model_type == 'xgboost':
            import xgboost as xgb
            params = model_params or {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.05,
                'eval_metric': 'auc'
            }

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test)
            model = xgb.train(params, dtrain, num_boost_round=100)
            y_proba = model.predict(dtest)

        elif model_type in ['lstm', 'gru']:
            # Simplified RNN prediction (would need sequence data)
            # For now, use simple logistic regression as fallback
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return y_proba

    def _generate_trades(
        self,
        predictions: np.ndarray,
        prices: pd.DataFrame,
        times: pd.DatetimeIndex,
        start_trade_id: int,
        model_name: str
    ) -> List[Trade]:
        """Generate trades from predictions."""
        trades = []
        trade_id = start_trade_id

        in_position = False
        position_bars = 0
        current_trade = None
        daily_trades = {}

        for i in range(len(predictions) - self.config.hold_bars):
            timestamp = times[i]
            date_key = timestamp.date() if hasattr(timestamp, 'date') else str(timestamp)[:10]

            # Check daily trade limit
            if date_key not in daily_trades:
                daily_trades[date_key] = 0

            if in_position:
                position_bars += 1

                # Check exit conditions
                exit_trade = False
                exit_reason = ""

                # Time-based exit
                if position_bars >= self.config.hold_bars:
                    exit_trade = True
                    exit_reason = "hold_bars"

                # Stop loss / Take profit
                if self.config.stop_loss_ticks and current_trade:
                    current_price = prices.iloc[i]['close']
                    price_diff = (current_price - current_trade.entry_price) * current_trade.direction
                    if price_diff < -self.config.stop_loss_ticks * self.config.tick_size:
                        exit_trade = True
                        exit_reason = "stop_loss"
                    elif self.config.take_profit_ticks and price_diff > self.config.take_profit_ticks * self.config.tick_size:
                        exit_trade = True
                        exit_reason = "take_profit"

                if exit_trade and current_trade:
                    # Close position
                    exit_price = prices.iloc[i]['close']
                    exit_time = times[i]

                    # Apply slippage
                    slippage_cost = self.config.slippage_ticks * self.config.tick_size * current_trade.direction * -1
                    exit_price += slippage_cost

                    current_trade.exit_time = exit_time
                    current_trade.exit_price = exit_price
                    current_trade.slippage = abs(slippage_cost * self.config.point_value * current_trade.size) * 2  # Entry + exit
                    current_trade.time_held = exit_time - current_trade.entry_time

                    # Calculate MFE/MAE
                    trade_prices = prices.iloc[i-position_bars:i+1]['close']
                    current_trade.calculate_metrics(trade_prices, self.config.point_value)

                    trades.append(current_trade)
                    in_position = False
                    position_bars = 0
                    current_trade = None

            else:
                # Check entry conditions
                if daily_trades[date_key] >= self.config.max_daily_trades:
                    continue

                signal = predictions[i]

                if signal > self.config.signal_threshold:
                    # Long entry
                    direction = 1
                elif signal < (1 - self.config.signal_threshold):
                    # Short entry
                    direction = -1
                else:
                    continue

                # Enter position
                entry_price = prices.iloc[i]['close']

                # Apply slippage
                slippage_cost = self.config.slippage_ticks * self.config.tick_size * direction
                entry_price += slippage_cost

                current_trade = Trade(
                    trade_id=trade_id,
                    entry_time=times[i],
                    exit_time=times[i],  # Will be updated on exit
                    entry_price=entry_price,
                    exit_price=entry_price,  # Will be updated on exit
                    direction=direction,
                    size=self.config.position_size,
                    commission=self.config.commission_per_trade * 2 * self.config.position_size,
                    entry_signal=signal,
                    model_name=model_name
                )

                in_position = True
                position_bars = 0
                trade_id += 1
                daily_trades[date_key] += 1

        return trades

    def _calculate_metrics(
        self,
        predictions: List[float],
        actuals: List[int]
    ) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        metrics = BacktestMetrics()

        if not self.trades:
            return metrics

        # Trade counts
        metrics.total_trades = len(self.trades)
        metrics.winning_trades = sum(1 for t in self.trades if t.net_pnl > 0)
        metrics.losing_trades = sum(1 for t in self.trades if t.net_pnl <= 0)
        metrics.long_trades = sum(1 for t in self.trades if t.direction == 1)
        metrics.short_trades = sum(1 for t in self.trades if t.direction == -1)

        # P&L metrics
        metrics.total_gross_pnl = sum(t.gross_pnl for t in self.trades)
        metrics.total_net_pnl = sum(t.net_pnl for t in self.trades)
        metrics.total_commission = sum(t.commission for t in self.trades)
        metrics.total_slippage = sum(t.slippage for t in self.trades)

        # Win/Loss metrics
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0

        winning_pnls = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        losing_pnls = [t.net_pnl for t in self.trades if t.net_pnl <= 0]

        metrics.avg_win = np.mean(winning_pnls) if winning_pnls else 0
        metrics.avg_loss = np.mean(losing_pnls) if losing_pnls else 0

        total_wins = sum(winning_pnls) if winning_pnls else 0
        total_losses = abs(sum(losing_pnls)) if losing_pnls else 0

        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        metrics.payoff_ratio = abs(metrics.avg_win / metrics.avg_loss) if metrics.avg_loss != 0 else float('inf')
        metrics.expectancy = metrics.total_net_pnl / metrics.total_trades if metrics.total_trades > 0 else 0

        # Drawdown metrics
        equity_curve = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = peak - equity_curve
        metrics.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        if peak.max() > 0:
            metrics.max_drawdown_pct = metrics.max_drawdown / peak.max()

        metrics.max_runup = equity_curve.max() if len(equity_curve) > 0 else 0

        # MFE/MAE
        metrics.avg_mae = np.mean([t.max_adverse_excursion for t in self.trades]) if self.trades else 0
        metrics.avg_mfe = np.mean([t.max_favorable_excursion for t in self.trades]) if self.trades else 0

        # Trade duration
        bars_held = [t.bars_held for t in self.trades]
        metrics.avg_bars_held = np.mean(bars_held) if bars_held else 0
        metrics.max_bars_held = max(bars_held) if bars_held else 0
        metrics.min_bars_held = min(bars_held) if bars_held else 0

        time_held_minutes = [t.time_held.total_seconds() / 60 for t in self.trades if t.time_held]
        metrics.avg_time_held_minutes = np.mean(time_held_minutes) if time_held_minutes else 0

        # Time analysis
        if self.trades:
            start_date = self.trades[0].entry_time
            end_date = self.trades[-1].exit_time
            if hasattr(start_date, 'date') and hasattr(end_date, 'date'):
                metrics.total_days = (end_date.date() - start_date.date()).days + 1
            else:
                metrics.total_days = len(self.trades) // 10  # Estimate

            trading_dates = set()
            for t in self.trades:
                if hasattr(t.entry_time, 'date'):
                    trading_dates.add(t.entry_time.date())
            metrics.trading_days = len(trading_dates)
            metrics.trades_per_day = metrics.total_trades / metrics.trading_days if metrics.trading_days > 0 else 0

        # Risk-adjusted returns
        trade_returns = [t.net_pnl for t in self.trades]
        if trade_returns and len(trade_returns) > 1:
            avg_return = np.mean(trade_returns)
            std_return = np.std(trade_returns)

            # Sharpe Ratio (annualized, assuming ~250 trading days, ~10 trades/day)
            trades_per_year = metrics.trades_per_day * 250 if metrics.trades_per_day > 0 else 250
            metrics.sharpe_ratio = (avg_return / std_return) * np.sqrt(trades_per_year) if std_return > 0 else 0

            # Sortino Ratio (downside deviation only)
            negative_returns = [r for r in trade_returns if r < 0]
            downside_std = np.std(negative_returns) if negative_returns else 0
            metrics.sortino_ratio = (avg_return / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

            # Calmar Ratio (return / max drawdown)
            total_return = sum(trade_returns)
            metrics.calmar_ratio = total_return / metrics.max_drawdown if metrics.max_drawdown > 0 else float('inf')

        # Model accuracy
        if predictions and actuals:
            pred_binary = [1 if p > 0.5 else 0 for p in predictions]
            metrics.accuracy = accuracy_score(actuals, pred_binary)
            try:
                metrics.auc_roc = roc_auc_score(actuals, predictions)
            except Exception:
                metrics.auc_roc = 0.5

        return metrics

    def get_trade_log(self) -> pd.DataFrame:
        """Get detailed trade log as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                'trade_id': t.trade_id,
                'model': t.model_name,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': 'LONG' if t.direction == 1 else 'SHORT',
                'size': t.size,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'gross_pnl': t.gross_pnl,
                'commission': t.commission,
                'slippage': t.slippage,
                'net_pnl': t.net_pnl,
                'bars_held': t.bars_held,
                'time_held_min': t.time_held.total_seconds() / 60 if t.time_held else 0,
                'max_runup': t.max_favorable_excursion,
                'max_drawdown': t.max_adverse_excursion,
                'entry_signal': t.entry_signal
            })

        return pd.DataFrame(records)

    def get_equity_curve(self) -> pd.Series:
        """Get equity curve as Series."""
        return pd.Series(self.equity_curve, name='equity')


def run_backtest(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    target_col: str = 'target_direction_1',
    selected_features: Optional[List[str]] = None,
    model_type: str = 'lightgbm',
    config: Optional[BacktestConfig] = None,
    output_dir: Optional[str] = None
) -> Tuple[List[Trade], BacktestMetrics]:
    """
    Convenience function to run backtest.

    Parameters:
    -----------
    prices : pd.DataFrame
        OHLCV price data
    features : pd.DataFrame
        Feature matrix
    target_col : str
        Target column name
    selected_features : List[str], optional
        Features to use (default: all non-target columns)
    model_type : str
        Model type ('lightgbm', 'xgboost', 'lstm', 'gru')
    config : BacktestConfig, optional
        Backtest configuration
    output_dir : str, optional
        Directory to save results

    Returns:
    --------
    trades : List[Trade]
        All executed trades
    metrics : BacktestMetrics
        Performance metrics
    """
    # Get feature columns
    if selected_features is None:
        selected_features = [c for c in features.columns if not c.startswith('target_')]

    # Run backtest
    backtester = WalkForwardBacktester(config)
    trades, metrics = backtester.run_backtest(
        prices, features, target_col, selected_features, model_type
    )

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save trade log
        trade_log = backtester.get_trade_log()
        if not trade_log.empty:
            trade_log.to_csv(output_path / f'trades_{model_type}_{timestamp}.csv', index=False)

        # Save metrics
        with open(output_path / f'metrics_{model_type}_{timestamp}.json', 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)

        # Save equity curve
        equity = backtester.get_equity_curve()
        equity.to_csv(output_path / f'equity_{model_type}_{timestamp}.csv')

        logger.info(f"Results saved to {output_path}")

    return trades, metrics


def generate_backtest_report(
    trades: List[Trade],
    metrics: BacktestMetrics,
    model_name: str,
    output_path: Optional[str] = None
) -> str:
    """
    Generate comprehensive backtest report.

    Parameters:
    -----------
    trades : List[Trade]
        List of executed trades
    metrics : BacktestMetrics
        Performance metrics
    model_name : str
        Name of the model
    output_path : str, optional
        Path to save report

    Returns:
    --------
    report : str
        Formatted report text
    """
    report_lines = [
        "=" * 70,
        f"BACKTEST REPORT: {model_name.upper()}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "TRADE SUMMARY",
        "-" * 40,
        f"Total Trades:      {metrics.total_trades:,}",
        f"Winning Trades:    {metrics.winning_trades:,} ({metrics.win_rate*100:.1f}%)",
        f"Losing Trades:     {metrics.losing_trades:,}",
        f"Long Trades:       {metrics.long_trades:,}",
        f"Short Trades:      {metrics.short_trades:,}",
        "",
        "P&L SUMMARY",
        "-" * 40,
        f"Gross P&L:         ${metrics.total_gross_pnl:,.2f}",
        f"Commission:        ${metrics.total_commission:,.2f}",
        f"Slippage:          ${metrics.total_slippage:,.2f}",
        f"Net P&L:           ${metrics.total_net_pnl:,.2f}",
        "",
        "TRADE STATISTICS",
        "-" * 40,
        f"Average Win:       ${metrics.avg_win:,.2f}",
        f"Average Loss:      ${metrics.avg_loss:,.2f}",
        f"Profit Factor:     {metrics.profit_factor:.2f}",
        f"Payoff Ratio:      {metrics.payoff_ratio:.2f}",
        f"Expectancy:        ${metrics.expectancy:,.2f}",
        "",
        "RISK METRICS",
        "-" * 40,
        f"Max Drawdown:      ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct*100:.1f}%)",
        f"Max Run-up:        ${metrics.max_runup:,.2f}",
        f"Avg MAE:           ${metrics.avg_mae:,.2f}",
        f"Avg MFE:           ${metrics.avg_mfe:,.2f}",
        "",
        "TRADE DURATION",
        "-" * 40,
        f"Avg Bars Held:     {metrics.avg_bars_held:.1f}",
        f"Avg Time Held:     {metrics.avg_time_held_minutes:.1f} minutes",
        f"Max Bars Held:     {metrics.max_bars_held}",
        f"Min Bars Held:     {metrics.min_bars_held}",
        "",
        "RISK-ADJUSTED RETURNS",
        "-" * 40,
        f"Sharpe Ratio:      {metrics.sharpe_ratio:.3f}",
        f"Sortino Ratio:     {metrics.sortino_ratio:.3f}",
        f"Calmar Ratio:      {metrics.calmar_ratio:.3f}",
        "",
        "TIME ANALYSIS",
        "-" * 40,
        f"Total Days:        {metrics.total_days}",
        f"Trading Days:      {metrics.trading_days}",
        f"Trades/Day:        {metrics.trades_per_day:.2f}",
        "",
        "MODEL METRICS",
        "-" * 40,
        f"Accuracy:          {metrics.accuracy*100:.2f}%",
        f"AUC-ROC:           {metrics.auc_roc*100:.2f}%",
        "",
        "=" * 70
    ]

    report = "\n".join(report_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")

    return report


# Exports
__all__ = [
    'BacktestConfig',
    'Trade',
    'BacktestMetrics',
    'WalkForwardBacktester',
    'run_backtest',
    'generate_backtest_report'
]
