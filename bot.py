"""
Trading Bot - Simplified Live Trading Engine

A minimal trading bot for live/paper trading. Extracted from bot_1hour/bot.py.

Core functionality:
- Sync account and positions from broker
- Check entry signals using StrategyManager
- Check exit conditions (trailing stop, hard stop, take profit, max hold)
- Execute entry/exit orders via broker
- Kill switch on daily loss limit

Usage:
    python bot.py
    python bot.py --config custom_config.yaml
"""

import argparse
import logging
import time
import yaml
from datetime import datetime, timedelta
import pytz
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core import (
    YFinanceDataFetcher,
    TechnicalIndicators,
    RiskManager,
    ExitManager,
    EntryGate,
    create_broker,
    TradeLogger,
    MarketHours,
    VolatilityScanner,
    DailyDrawdownGuard,
    DrawdownTier,
    LosingStreakGuard,
)
from strategies import StrategyManager

# Configure logging with both console and file output
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format
)
logger = logging.getLogger('trading_bot')

# Add file handler for persistent logs
log_file = Path(__file__).parent / 'logs' / f'trading_{datetime.now().strftime("%Y%m%d")}.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)
logger.info(f"Logging to {log_file}")


class TradingBot:
    """
    Trading Bot for live/paper trading.

    Simplified from bot_1hour/bot.py. Strips out:
    - TradingDaemon (background mode)
    - ParityCapture (parity logging)
    - EnsembleMLSystem (ML system)
    - Promotion pipeline functions

    Implements:
    - Account and position syncing
    - Entry signal checking via StrategyManager
    - Exit condition checking (trailing stop, hard stop, take profit, max hold)
    - Order execution via broker
    - Kill switch on daily loss limit
    """

    def __init__(self, config_path: str = None, scanner_symbols: list = None):
        """
        Initialize the trading bot.

        Args:
            config_path: Path to config.yaml. Defaults to config.yaml in bot directory.
            scanner_symbols: Optional list of symbols from scanner (overrides config).
        """
        self.bot_dir = Path(__file__).parent

        # Load config
        if config_path is None:
            config_path = self.bot_dir / 'config.yaml'
        else:
            config_path = Path(config_path)

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.mode = self.config.get('mode', 'PAPER')
        self.timeframe = self.config.get('timeframe', '1Hour')

        # Database for trades
        db_path = self.config.get('logging', {}).get('database', 'logs/trades.db')

        # Load universe
        universe_path = self.bot_dir / self.config.get('trading', {}).get('watchlist_file', 'universe.yaml')
        with open(universe_path, 'r') as f:
            universe = yaml.safe_load(f)

        static_watchlist = universe.get('proven_symbols', [])

        # Check for scanner symbols override (from CLI)
        if scanner_symbols:
            self.scanner = None
            self.watchlist = scanner_symbols
            logger.info(f"Using scanner-provided watchlist: {self.watchlist}")
        # Check if volatility scanner is enabled in config
        elif self.config.get('volatility_scanner', {}).get('enabled', False):
            scanner_config = self.config.get('volatility_scanner', {})
            self.scanner = VolatilityScanner(scanner_config)
            scanned_symbols = self.scanner.scan()
            if scanned_symbols:
                self.watchlist = scanned_symbols
                logger.info(f"Scanner selected {len(self.watchlist)} symbols: {self.watchlist}")
            else:
                self.watchlist = static_watchlist
                logger.warning("Scanner returned no symbols, falling back to static universe")
        else:
            self.scanner = None
            self.watchlist = static_watchlist

        logger.info(f"Trading Bot initialized")
        logger.info(f"Mode: {self.mode}, Timeframe: {self.timeframe}")
        logger.info(f"Symbols: {self.watchlist}")
        logger.info(f"Risk: {self.config.get('risk_management', {}).get('max_position_size_pct')}% per trade")

        if not self.watchlist:
            logger.warning("No symbols in universe! Bot will have nothing to trade.")

        # Initialize shared components
        self.data_fetcher = YFinanceDataFetcher()
        self.indicators = TechnicalIndicators()
        self.broker = create_broker()
        self.trade_logger = TradeLogger(db_file=db_path)
        self.risk_manager = RiskManager(self.config.get('risk_management', {}))
        self.entry_gate = EntryGate(self.config.get('entry_gate', {}))

        # Initialize ExitManager with proper format (matching backtest.py)
        # ExitManager expects {'risk': {settings}} format with percentages as whole numbers
        # FIX (Jan 2026): Aligned default values to match backtest.py exactly
        exit_config = self.config.get('exit_manager', {})
        exit_settings = {
            'hard_stop_pct': abs(exit_config.get('tier_0_hard_stop', -0.02)) * 100,  # Default -2% matches backtest
            'profit_floor_pct': exit_config.get('tier_1_profit_floor', 0.02) * 100,
            'trailing_activation_pct': exit_config.get('tier_2_atr_trailing', 0.03) * 100,
            'partial_tp_pct': exit_config.get('tier_3_partial_take', 0.04) * 100,
            'partial_tp2_pct': exit_config.get('tier_4_partial_take2', 0.05) * 100,
            'partial_tp2_size': exit_config.get('tier_4_partial_take2_size', 1.0),
        }
        self.exit_manager = ExitManager({'risk': exit_settings})
        # FIX (Jan 2026): Track tiered exits enabled state to match backtest.py behavior
        self.use_tiered_exits = exit_config.get('enabled', True)
        # FIX (Jan 2026): Add EOD close logic to match backtest.py
        self.eod_close_enabled = exit_config.get('eod_close', False)
        self.eod_close_bar_hour = 15  # 3 PM ET (15:00) - close before market close
        self.market_hours = MarketHours()

        # Strategy Manager
        self.strategy_manager = StrategyManager(self.config)
        logger.info(f"Strategy Manager: {len(self.strategy_manager.strategies)} strategies initialized")

        self.running = False
        self.positions = {}

        # Trading state
        self.cash = 0.0
        self.portfolio_value = 0.0
        self.peak_value = 0.0
        self.daily_pnl = 0.0
        self.daily_starting_capital = 0.0
        self.current_trading_day = None
        self.kill_switch_triggered = False

        # Emergency stop - force close if unrealized loss exceeds threshold
        # FIX (Jan 2026): Added after $4K loss on single SHORT with no stop
        risk_config = self.config.get('risk_management', {})
        self.emergency_stop_pct = risk_config.get('emergency_stop_pct', 5.0) / 100
        self.emergency_stop_enabled = True

        # Daily Drawdown Guard (Jan 4, 2026)
        # Protects funded account capital with tiered drawdown limits
        self.drawdown_guard = DailyDrawdownGuard(self.config)
        if self.drawdown_guard.enabled:
            logger.info(f"DailyDrawdownGuard: enabled with {self.drawdown_guard.hard_limit_pct*100:.1f}% hard limit")

        # Losing Streak Guard (Jan 4, 2026)
        # Reduces position sizes after 2+ losing trades in 3 days
        self.losing_streak_guard = LosingStreakGuard(self.config)
        if self.losing_streak_guard.enabled:
            logger.info(f"LosingStreakGuard: enabled, throttle after {self.losing_streak_guard.min_losing_trades} losers in {self.losing_streak_guard.lookback_days} days")

        # Position tracking
        self.open_positions = {}  # {symbol: position_dict}
        self.pending_entries = {}  # {symbol: entry_dict}
        self.last_trade_time = {}  # {symbol: datetime}

        # Exit manager state tracking
        self.trailing_stops = {}  # {symbol: {'activated': bool, 'price': float}}
        self.highest_prices = {}  # {symbol: float}
        self.lowest_prices = {}   # {symbol: float}

        # Timing
        self.last_bar_time = None
        self.last_cycle_time = None

    def sync_account(self):
        """Sync account state from broker."""
        try:
            account = self.broker.get_account()
            self.cash = float(account.cash)
            self.portfolio_value = float(account.portfolio_value)

            # Track peak for drawdown
            if self.portfolio_value > self.peak_value:
                self.peak_value = self.portfolio_value

            # FIX (Jan 2026): Use Alpaca's last_equity for daily P&L calculation
            # Previously used self-tracked daily_starting_capital which reset on every bot restart,
            # causing false kill switch triggers (e.g., showing 4% loss when account was actually up).
            # Alpaca's last_equity is the official previous-day close, immune to bot restarts.
            self.daily_starting_capital = float(account.last_equity)
            self.daily_pnl = self.portfolio_value - self.daily_starting_capital

            # Reset kill switch on new trading day
            now = datetime.now()
            if self.current_trading_day != now.date():
                self.current_trading_day = now.date()
                self.kill_switch_triggered = False
                logger.info(f"New trading day: start equity=${self.daily_starting_capital:.2f}")

            # Kill switch check
            if self.daily_starting_capital > 0:
                daily_loss_pct = -self.daily_pnl / self.daily_starting_capital
                max_daily_loss = self.config.get('risk_management', {}).get('max_daily_loss_pct', 5.0) / 100
                if daily_loss_pct >= max_daily_loss:
                    if not self.kill_switch_triggered:
                        self.kill_switch_triggered = True
                        logger.warning(f"KILL SWITCH: Daily loss {daily_loss_pct*100:.2f}% >= {max_daily_loss*100}% (equity=${self.portfolio_value:.2f}, start=${self.daily_starting_capital:.2f})")

            logger.debug(f"Account synced: cash=${self.cash:.2f}, portfolio=${self.portfolio_value:.2f}")

        except Exception as e:
            logger.error(f"Failed to sync account: {e}")

    def sync_positions(self):
        """Sync open positions from broker."""
        try:
            broker_positions = self.broker.get_positions()
            synced = {}

            for pos in broker_positions:
                symbol = pos.symbol

                # FIX (Jan 2026): Sync ALL positions, not just watchlist symbols
                # We must manage exits for existing positions even if they're no longer
                # in today's scanner watchlist. Watchlist only limits NEW entries.
                # Previously this caused positions to be ignored and stops to not trigger!

                synced[symbol] = {
                    'symbol': symbol,
                    'qty': int(pos.qty),
                    'entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price) if hasattr(pos, 'current_price') else float(pos.avg_entry_price),
                    'unrealized_pnl': float(pos.unrealized_pl) if hasattr(pos, 'unrealized_pl') else 0.0,
                    'direction': 'LONG' if pos.side == 'long' else 'SHORT',
                    'entry_time': self.open_positions.get(symbol, {}).get('entry_time', datetime.now()),
                }

                # Initialize tracking if new position
                if symbol not in self.highest_prices:
                    self.highest_prices[symbol] = synced[symbol]['entry_price']
                if symbol not in self.lowest_prices:
                    self.lowest_prices[symbol] = synced[symbol]['entry_price']
                if symbol not in self.trailing_stops:
                    self.trailing_stops[symbol] = {'activated': False, 'price': 0.0}

                # FIX (Jan 2026): Register synced positions with ExitManager
                # Previously, positions synced from Alpaca were never registered with ExitManager,
                # causing all tiered exit logic (trailing stops, profit floors) to be bypassed.
                # evaluate_exit() returned None immediately for unregistered positions.
                if self.use_tiered_exits and self.exit_manager:
                    if symbol not in self.exit_manager.positions:
                        self.exit_manager.register_position(
                            symbol=symbol,
                            entry_price=synced[symbol]['entry_price'],
                            quantity=synced[symbol]['qty'],
                            entry_time=synced[symbol]['entry_time'],
                            direction=synced[symbol]['direction']
                        )
                        logger.info(f"EXIT_MGR | Registered synced position: {symbol} @ ${synced[symbol]['entry_price']:.2f}")

                # Update price tracking
                current = synced[symbol]['current_price']
                if current > self.highest_prices[symbol]:
                    self.highest_prices[symbol] = current
                if current < self.lowest_prices[symbol]:
                    self.lowest_prices[symbol] = current

            # Detect closed positions
            for symbol in list(self.open_positions.keys()):
                if symbol not in synced:
                    logger.info(f"Position {symbol} closed externally")
                    self._cleanup_position(symbol)

            self.open_positions = synced
            logger.debug(f"Positions synced: {len(self.open_positions)} open")

        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")

    def _cleanup_position(self, symbol: str):
        """Clean up tracking state for a closed position."""
        self.highest_prices.pop(symbol, None)
        self.lowest_prices.pop(symbol, None)
        self.trailing_stops.pop(symbol, None)
        self.pending_entries.pop(symbol, None)
        # FIX (Jan 2026): Unregister from ExitManager during cleanup
        if self.use_tiered_exits and self.exit_manager:
            self.exit_manager.unregister_position(symbol)

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate ATR (Average True Range) from historical data.

        Matches backtest.py implementation to ensure consistency.

        Args:
            data: DataFrame with high, low, close columns
            period: ATR period (default 14)

        Returns:
            ATR value or 0.0 if insufficient data
        """
        if data is None or len(data) < period + 1:
            return 0.0

        try:
            # Use last period+1 bars for calculation
            hist_data = data.tail(period + 1).copy()
            if len(hist_data) < period:
                return 0.0

            high = hist_data['high'].values
            low = hist_data['low'].values
            close = hist_data['close'].values

            # Calculate True Range
            tr = np.zeros(len(high))
            for j in range(1, len(high)):
                prev_close = close[j - 1]
                tr[j] = max(
                    high[j] - low[j],
                    abs(high[j] - prev_close),
                    abs(low[j] - prev_close)
                )

            # Average of last 'period' true ranges
            recent_tr = tr[-period:]
            atr = np.mean(recent_tr) if len(recent_tr) == period else 0.0

            return float(atr) if not np.isnan(atr) else 0.0

        except Exception as e:
            logger.debug(f"Error calculating ATR: {e}")
            return 0.0

    def check_entry(self, symbol: str, data: pd.DataFrame, current_price: float) -> dict:
        """
        Check for entry signal using StrategyManager.

        Args:
            symbol: Stock symbol
            data: Historical data with indicators
            current_price: Current price

        Returns:
            Signal dict with action, confidence, strategy, reasoning
        """
        # Default no-signal
        no_signal = {'action': 'HOLD', 'confidence': 0, 'strategy': '', 'reasoning': ''}

        if len(data) < 30:
            return {**no_signal, 'reasoning': 'Insufficient data'}

        # Kill switch check
        if self.kill_switch_triggered:
            return {**no_signal, 'reasoning': 'Kill switch active'}

        # Already have position in this symbol
        if symbol in self.open_positions:
            return {**no_signal, 'reasoning': 'Already have position'}

        # Cooldown check
        cooldown_minutes = self.config.get('entry_gate', {}).get('min_time_between_trades_minutes', 60)
        if symbol in self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time[symbol]).total_seconds() / 60
            if elapsed < cooldown_minutes:
                return {**no_signal, 'reasoning': f'Cooldown: {cooldown_minutes - elapsed:.0f}m remaining'}

        # Entry gate check
        if self.entry_gate:
            allowed, reason = self.entry_gate.check_entry_allowed(symbol, datetime.now())
            if not allowed:
                return {**no_signal, 'reasoning': f'Entry gate: {reason}'}

        try:
            # Get signal from strategy manager
            signal = self.strategy_manager.get_best_signal(
                symbol=symbol,
                data=data,
                current_price=current_price,
                indicators=self.indicators
            )

            if signal and signal.get('action') != 'HOLD':
                confidence = signal.get('confidence', 0)
                threshold = self.strategy_manager.confidence_threshold

                if signal['action'] == 'BUY' and confidence >= threshold:
                    return {
                        'action': 'BUY',
                        'confidence': confidence,
                        'strategy': signal.get('strategy', 'Unknown'),
                        'reasoning': signal.get('reasoning', ''),
                        'direction': 'LONG'
                    }
                elif signal['action'] == 'SELL' and confidence >= threshold:
                    # SHORT signal
                    return {
                        'action': 'SELL',
                        'confidence': confidence,
                        'strategy': signal.get('strategy', 'Unknown'),
                        'reasoning': signal.get('reasoning', ''),
                        'direction': 'SHORT'
                    }

            return no_signal

        except Exception as e:
            logger.error(f"Error checking entry for {symbol}: {e}")
            return {**no_signal, 'reasoning': f'Error: {e}'}

    def check_exit(self, symbol: str, position: dict, current_price: float,
                   bar_high: float = None, bar_low: float = None,
                   data: pd.DataFrame = None) -> Optional[dict]:
        """
        Check for exit conditions.

        Args:
            symbol: Stock symbol
            position: Position dict with entry_price, qty, direction, entry_time
            current_price: Current price
            bar_high: High of current bar (optional)
            bar_low: Low of current bar (optional)
            data: Historical OHLCV data for ATR calculation (optional)

        Returns:
            Exit dict with exit=True/False, reason, price, qty or None
        """
        if bar_high is None:
            bar_high = current_price
        if bar_low is None:
            bar_low = current_price

        entry_price = position['entry_price']
        direction = position.get('direction', 'LONG')
        entry_time = position.get('entry_time', datetime.now())
        qty = position['qty']

        exit_config = self.config.get('exit_manager', {})
        trailing_config = self.config.get('trailing_stop', {})
        risk_config = self.config.get('risk_management', {})

        # Get thresholds from config (matching backtest.py defaults)
        # FIX (Jan 2026): Aligned all defaults to match backtest.py exactly
        hard_stop_pct = abs(exit_config.get('tier_0_hard_stop', -0.02))  # 2% default matches backtest
        profit_floor_pct = exit_config.get('tier_1_profit_floor', 0.02)
        max_hold_hours = exit_config.get('max_hold_hours', 48)  # 48 hours default matches backtest
        take_profit_pct = risk_config.get('take_profit_pct', 8.0) / 100  # 8% default

        trailing_enabled = trailing_config.get('enabled', True)
        trailing_activation = trailing_config.get('activation_pct', 0.25) / 100  # 0.25% default
        trailing_trail = trailing_config.get('trail_pct', 0.25) / 100  # 0.25% default
        trailing_move_to_breakeven = trailing_config.get('move_to_breakeven', True)

        # Update price tracking
        if symbol not in self.highest_prices:
            self.highest_prices[symbol] = entry_price
        if symbol not in self.lowest_prices:
            self.lowest_prices[symbol] = entry_price

        if bar_high > self.highest_prices[symbol]:
            self.highest_prices[symbol] = bar_high
        if bar_low < self.lowest_prices[symbol]:
            self.lowest_prices[symbol] = bar_low

        highest = self.highest_prices[symbol]
        lowest = self.lowest_prices[symbol]

        # Initialize trailing stop state
        if symbol not in self.trailing_stops:
            self.trailing_stops[symbol] = {'activated': False, 'price': 0.0}

        # ============ EXIT CHECKS ============

        # 0. Emergency stop - force close if unrealized daily loss exceeds threshold
        # FIX (Jan 2026): Added after $4K loss on single SHORT with no stop
        if self.emergency_stop_enabled and self.daily_starting_capital > 0:
            if direction == 'LONG':
                unrealized_pnl = (current_price - entry_price) * qty
            else:  # SHORT
                unrealized_pnl = (entry_price - current_price) * qty

            total_daily_loss = self.daily_pnl + unrealized_pnl
            total_daily_loss_pct = -total_daily_loss / self.daily_starting_capital
            if total_daily_loss_pct >= self.emergency_stop_pct:
                logger.warning(f"EMERGENCY STOP: {symbol} - Daily loss {total_daily_loss_pct*100:.1f}% >= {self.emergency_stop_pct*100:.1f}%")
                return {
                    'exit': True,
                    'reason': 'emergency_stop',
                    'price': current_price,
                    'qty': qty
                }

        if direction == 'LONG':
            # 1. Trailing stop check (legacy - only when tiered exits disabled)
            # FIX (Jan 2026): Match backtest.py and SHORT logic - only run when tiered exits disabled
            if trailing_enabled and not self.use_tiered_exits:
                profit_pct = (highest - entry_price) / entry_price
                if not self.trailing_stops[symbol]['activated'] and profit_pct >= trailing_activation:
                    self.trailing_stops[symbol]['activated'] = True
                    # Use config to decide breakeven vs trailing from peak
                    if trailing_move_to_breakeven:
                        self.trailing_stops[symbol]['price'] = entry_price
                    else:
                        self.trailing_stops[symbol]['price'] = highest * (1 - trailing_trail)

                if self.trailing_stops[symbol]['activated']:
                    new_trail = highest * (1 - trailing_trail)
                    if new_trail > self.trailing_stops[symbol]['price']:
                        self.trailing_stops[symbol]['price'] = new_trail

                    if bar_low <= self.trailing_stops[symbol]['price']:
                        return {
                            'exit': True,
                            'reason': 'trailing_stop',
                            'price': self.trailing_stops[symbol]['price'],
                            'qty': qty
                        }

            # 2. Tiered exit via ExitManager (LONG and SHORT - Jan 2026)
            if self.use_tiered_exits and self.exit_manager:
                # Calculate ATR from historical data for proper trailing stops
                atr = self._calculate_atr(data, period=14) if data is not None else 0.0
                # Pass bar_high and bar_low for proper stop checking
                exit_action = self.exit_manager.evaluate_exit(
                    symbol, current_price, atr,
                    bar_high=bar_high, bar_low=bar_low
                )
                if exit_action:
                    reason = exit_action.get('reason', 'exit_manager')
                    if reason == 'hard_stop':
                        reason = 'stop_loss'
                    return {
                        'exit': True,
                        'reason': reason,
                        'price': exit_action.get('stop_price', current_price),
                        'qty': exit_action.get('qty', qty)
                    }

            # 3. Simple stop loss fallback - only when tiered exits disabled
            if not self.use_tiered_exits:
                stop_price = entry_price * (1 - hard_stop_pct)
                if bar_low <= stop_price:
                    return {
                        'exit': True,
                        'reason': 'stop_loss',
                        'price': stop_price,
                        'qty': qty
                    }

            # 4. Take profit - only when tiered exits disabled
            if not self.use_tiered_exits:
                tp_price = entry_price * (1 + take_profit_pct)
                if bar_high >= tp_price:
                    return {
                        'exit': True,
                        'reason': 'take_profit',
                        'price': tp_price,
                        'qty': qty
                    }

        else:  # SHORT
            # 1. Trailing stop check for SHORT (legacy - only when tiered exits disabled)
            if trailing_enabled and not self.use_tiered_exits:
                profit_pct = (entry_price - lowest) / entry_price
                if not self.trailing_stops[symbol]['activated'] and profit_pct >= trailing_activation:
                    self.trailing_stops[symbol]['activated'] = True
                    if trailing_move_to_breakeven:
                        self.trailing_stops[symbol]['price'] = entry_price
                    else:
                        self.trailing_stops[symbol]['price'] = lowest * (1 + trailing_trail)

                if self.trailing_stops[symbol]['activated']:
                    new_trail = lowest * (1 + trailing_trail)
                    if new_trail < self.trailing_stops[symbol]['price'] or self.trailing_stops[symbol]['price'] == 0:
                        self.trailing_stops[symbol]['price'] = new_trail

                    if bar_high >= self.trailing_stops[symbol]['price']:
                        return {
                            'exit': True,
                            'reason': 'trailing_stop',
                            'price': self.trailing_stops[symbol]['price'],
                            'qty': qty
                        }

            # 2. Tiered exit via ExitManager (SHORT - Jan 2026)
            if self.use_tiered_exits and self.exit_manager:
                atr = self._calculate_atr(data, period=14) if data is not None else 0.0
                exit_action = self.exit_manager.evaluate_exit(
                    symbol, current_price, atr,
                    bar_high=bar_high, bar_low=bar_low
                )
                if exit_action:
                    reason = exit_action.get('reason', 'exit_manager')
                    if reason == 'hard_stop':
                        reason = 'stop_loss'
                    return {
                        'exit': True,
                        'reason': reason,
                        'price': exit_action.get('stop_price', current_price),
                        'qty': exit_action.get('qty', qty)
                    }

            # 3. Hard stop for SHORT - only when tiered exits disabled
            if not self.use_tiered_exits:
                stop_price = entry_price * (1 + hard_stop_pct)
                if bar_high >= stop_price:
                    return {
                        'exit': True,
                        'reason': 'stop_loss',
                        'price': stop_price,
                        'qty': qty
                    }

            # 4. Take profit for SHORT - only when tiered exits disabled
            if not self.use_tiered_exits:
                tp_price = entry_price * (1 - take_profit_pct)
                if bar_low <= tp_price:
                    return {
                        'exit': True,
                        'reason': 'take_profit',
                        'price': tp_price,
                        'qty': qty
                    }

        # 5. EOD close check (matches backtest.py:729-736)
        # FIX (Jan 2026): Add EOD close logic to match backtest behavior
        if self.eod_close_enabled:
            import pytz
            market_tz = pytz.timezone('America/New_York')
            current_time = datetime.now(market_tz)
            if current_time.hour >= self.eod_close_bar_hour:
                return {
                    'exit': True,
                    'reason': 'eod_close',
                    'price': current_price,
                    'qty': qty
                }

        # 6. Max hold time check
        elapsed_hours = (datetime.now() - entry_time).total_seconds() / 3600
        if elapsed_hours >= max_hold_hours:
            return {
                'exit': True,
                'reason': 'max_hold',
                'price': current_price,
                'qty': qty
            }

        return None  # No exit

    def execute_entry(self, symbol: str, direction: str, price: float,
                      strategy: str, reasoning: str) -> dict:
        """
        Execute an entry order.

        Args:
            symbol: Stock symbol
            direction: 'LONG' or 'SHORT'
            price: Current price for position sizing
            strategy: Strategy name for logging
            reasoning: Entry reasoning for logging

        Returns:
            Result dict with filled=True/False, order details
        """
        try:
            # Calculate position size
            # FIX (Jan 2026): Default 5.0% matches backtest.py:151 (was 2.0%, causing mismatch)
            risk_config = self.config.get('risk_management', {})
            stop_loss_pct = risk_config.get('stop_loss_pct', 5.0) / 100

            # FIX (Jan 2026): Apply estimated slippage before position sizing (matches backtest.py:888-894)
            # This ensures position sizing uses realistic entry prices like backtest does
            exec_config = self.config.get('execution', {})
            slippage_bps = exec_config.get('slippage_bps', 5)  # Default 5 bps = 0.05%
            half_spread_bps = exec_config.get('half_spread_bps', 2)  # Default 2 bps = 0.02%
            entry_slippage = (slippage_bps + half_spread_bps) / 10000

            if direction == 'LONG':
                realistic_entry_price = price * (1 + entry_slippage)
                stop_price = realistic_entry_price * (1 - stop_loss_pct)
            else:
                realistic_entry_price = price * (1 - entry_slippage)
                stop_price = realistic_entry_price * (1 + stop_loss_pct)

            qty = self.risk_manager.calculate_position_size(
                self.portfolio_value, realistic_entry_price, stop_price
            )

            # Apply drawdown guard position size multiplier
            if self.drawdown_guard.enabled and self.drawdown_guard.position_size_multiplier < 1.0:
                original_qty = qty
                qty = int(qty * self.drawdown_guard.position_size_multiplier)
                if qty != original_qty:
                    logger.info(
                        f"DRAWDOWN_GUARD | Position size reduced: {original_qty} -> {qty} shares "
                        f"({self.drawdown_guard.position_size_multiplier*100:.0f}% multiplier)"
                    )

            # Apply losing streak guard position size multiplier (Jan 4, 2026)
            if self.losing_streak_guard.enabled and self.losing_streak_guard.position_size_multiplier < 1.0:
                original_qty = qty
                qty = int(qty * self.losing_streak_guard.position_size_multiplier)
                if qty != original_qty:
                    logger.info(
                        f"STREAK_GUARD | Position size reduced: {original_qty} -> {qty} shares "
                        f"({self.losing_streak_guard.position_size_multiplier*100:.0f}% multiplier)"
                    )

            # Calculate risk amount for losing streak tracking (Jan 4, 2026)
            # This is the R value - what we're risking on this trade
            risk_amount = abs(realistic_entry_price - stop_price) * qty

            if qty <= 0:
                return {'filled': False, 'reason': 'Position size too small'}

            # Submit order
            side = 'buy' if direction == 'LONG' else 'sell'
            order = self.broker.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )

            # FIX (Jan 2026): Track position immediately after order submission
            # Previously only tracked if status in ['filled', 'new', 'accepted'], but Alpaca
            # paper trading may return other statuses. Track optimistically to enforce position limits.
            if order:
                fill_price = float(order.filled_avg_price) if hasattr(order, 'filled_avg_price') and order.filled_avg_price else price
                fill_qty = int(order.filled_qty) if hasattr(order, 'filled_qty') and order.filled_qty else qty

                # Register with exit manager (LONG and SHORT - Jan 2026)
                if self.use_tiered_exits and self.exit_manager:
                    self.exit_manager.register_position(
                        symbol=symbol,
                        entry_price=fill_price,
                        quantity=fill_qty,
                        entry_time=datetime.now(),
                        direction=direction
                    )

                # Update tracking
                self.open_positions[symbol] = {
                    'symbol': symbol,
                    'qty': fill_qty,
                    'entry_price': fill_price,
                    'direction': direction,
                    'entry_time': datetime.now(),
                    'strategy': strategy,
                    'reasoning': reasoning,
                    'risk_amount': risk_amount,  # For losing streak guard (Jan 4, 2026)
                }
                self.highest_prices[symbol] = fill_price
                self.lowest_prices[symbol] = fill_price
                self.trailing_stops[symbol] = {'activated': False, 'price': 0.0}
                self.last_trade_time[symbol] = datetime.now()

                # Record entry in entry gate
                if self.entry_gate:
                    self.entry_gate.record_entry(symbol, datetime.now())

                # Log entry trade to database
                self.trade_logger.log_trade(
                    symbol=symbol,
                    action='BUY' if direction == 'LONG' else 'SELL',
                    quantity=fill_qty,
                    price=fill_price,
                    strategy=strategy,
                    pnl=0.0,  # No P&L on entry
                    exit_reason=None
                )

                logger.info(f"ENTRY: {direction} {fill_qty} {symbol} @ ${fill_price:.2f} [{strategy}]")

                return {
                    'filled': True,
                    'order_id': order.id,
                    'qty': fill_qty,
                    'price': fill_price,
                    'direction': direction
                }

            return {'filled': False, 'reason': 'Order not filled'}

        except Exception as e:
            logger.error(f"Error executing entry for {symbol}: {e}")
            return {'filled': False, 'reason': str(e)}

    def execute_exit(self, symbol: str, exit_signal: dict) -> dict:
        """
        Execute an exit order.

        Args:
            symbol: Stock symbol
            exit_signal: Exit dict from check_exit with reason, price, qty

        Returns:
            Result dict with filled=True/False, pnl, etc.
        """
        try:
            position = self.open_positions.get(symbol)
            if not position:
                return {'filled': False, 'reason': 'No position to exit'}

            direction = position.get('direction', 'LONG')
            entry_price = position['entry_price']
            qty = exit_signal.get('qty', position['qty'])
            exit_reason = exit_signal.get('reason', 'unknown')

            # Submit exit order
            side = 'sell' if direction == 'LONG' else 'buy'
            order = self.broker.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )

            if order and order.status in ['filled', 'new', 'accepted']:
                exit_price = float(order.filled_avg_price) if hasattr(order, 'filled_avg_price') and order.filled_avg_price else exit_signal.get('price', 0)

                # Calculate P&L
                if direction == 'LONG':
                    pnl = (exit_price - entry_price) * qty
                else:
                    pnl = (entry_price - exit_price) * qty

                # Log trade
                self.trade_logger.log_trade(
                    symbol=symbol,
                    action='SELL' if direction == 'LONG' else 'BUY',
                    quantity=qty,
                    price=exit_price,
                    strategy=position.get('strategy', 'Unknown'),
                    pnl=pnl,
                    exit_reason=exit_reason
                )

                # Record loss in entry gate
                if pnl < 0 and self.entry_gate:
                    self.entry_gate.record_loss(datetime.now())

                # Record realized P&L in drawdown guard
                if self.drawdown_guard.enabled:
                    self.drawdown_guard.record_realized_pnl(pnl)

                # Record trade in losing streak guard (Jan 4, 2026)
                if self.losing_streak_guard.enabled:
                    risk_amount = position.get('risk_amount', abs(pnl))  # Fallback to pnl if not stored
                    self.losing_streak_guard.record_trade(
                        symbol=symbol,
                        realized_pnl=pnl,
                        risk_amount=risk_amount,
                        close_time=datetime.now()
                    )

                # Unregister from exit manager (LONG and SHORT - Jan 2026)
                if self.use_tiered_exits and self.exit_manager:
                    self.exit_manager.unregister_position(symbol)

                # Cleanup
                self._cleanup_position(symbol)
                del self.open_positions[symbol]

                logger.info(f"EXIT: {symbol} @ ${exit_price:.2f} [{exit_reason}] P&L: ${pnl:+.2f}")

                return {
                    'filled': True,
                    'order_id': order.id,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'reason': exit_reason
                }

            return {'filled': False, 'reason': 'Order not filled'}

        except Exception as e:
            logger.error(f"Error executing exit for {symbol}: {e}")
            return {'filled': False, 'reason': str(e)}

    def fetch_data(self, symbol: str, bars: int = 200) -> Optional[pd.DataFrame]:
        """
        Fetch historical data with indicators.

        Args:
            symbol: Stock symbol
            bars: Number of bars to fetch (default 200 for SMA200)

        Returns:
            DataFrame with OHLCV and indicators, or None on error
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            # Add warmup buffer (extra days for weekends/holidays)
            warmup_days = int(bars / 6.5) + 10  # ~6.5 trading hours/day
            start_date = end_date - timedelta(days=warmup_days)

            # FIX (Jan 2026): Disable disk cache for live trading to ensure fresh data
            # Cache is useful for backtesting but causes stale data issues in live trading
            df = self.data_fetcher.get_historical_data_range(
                symbol=symbol,
                timeframe=self.timeframe,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                use_cache=False  # Always fetch fresh data for live trading
            )

            if df is None or df.empty:
                logger.warning(f"No data for {symbol}")
                return None

            # FIX (Jan 2026): Log the latest candle timestamp for debugging
            # This helps verify we're getting the correct completed candle
            if 'timestamp' in df.columns and len(df) > 0:
                latest_ts = df['timestamp'].iloc[-1]
                logger.debug(f"[{symbol}] Latest bar: {latest_ts} | Bars: {len(df)}")

            # Add indicators
            df = self.indicators.add_all_indicators(df)

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def run_trading_cycle(self):
        """
        Run one trading cycle.

        Called every hour (or on demand):
        1. Sync account and update drawdown guard
        2. Handle hard limit liquidation if needed
        3. Check exits for all positions
        4. Check entries for watchlist symbols (if allowed)
        """
        try:
            # FIX (Jan 2026): Log expected candle hour for verification
            eastern = pytz.timezone('America/New_York')
            now = datetime.now(eastern)
            expected_candle_hour = (now.hour - 1) % 24  # Previous hour's candle
            logger.info(f"=== Trading Cycle Start @ {now.strftime('%H:%M:%S')} EST ===")
            logger.info(f"Expecting candles from {expected_candle_hour}:00 hour")

            # 1. Sync state
            self.sync_account()
            self.sync_positions()

            # 2. Update drawdown guard (must be done after sync)
            if self.drawdown_guard.enabled:
                account = self.broker.get_account()
                tier = self.drawdown_guard.update_equity(account)

                # Log current drawdown status
                if tier != DrawdownTier.NORMAL:
                    status = self.drawdown_guard.get_status()
                    logger.warning(
                        f"DRAWDOWN_GUARD | {tier.name} | "
                        f"Drawdown: {status['drawdown_pct']:.2f}% | "
                        f"Entries: {'ALLOWED' if status['entries_allowed'] else 'BLOCKED'}"
                    )

                # Handle hard limit - full liquidation
                if tier == DrawdownTier.HARD_LIMIT:
                    logger.error("DRAWDOWN_GUARD | HARD_LIMIT REACHED - LIQUIDATING ALL POSITIONS")
                    positions_list = list(self.open_positions.values())
                    if positions_list:
                        result = self.drawdown_guard.force_liquidate_all(self.broker, positions_list)
                        if result['success']:
                            # Unregister from exit manager
                            for pos in result['liquidated']:
                                symbol = pos['symbol']
                                if self.use_tiered_exits and self.exit_manager:
                                    self.exit_manager.unregister_position(symbol)
                                self._cleanup_position(symbol)
                            self.open_positions.clear()
                    logger.error("DRAWDOWN_GUARD | DAY HALTED - No further trading")
                    return

                # Handle day halted state
                if self.drawdown_guard.day_halted:
                    logger.warning("DRAWDOWN_GUARD | Day halted - skipping cycle")
                    return

            # 2. Check exits for all positions
            logger.info(f"Checking exits for {len(self.open_positions)} positions")
            for symbol, position in list(self.open_positions.items()):
                # FIX (Jan 2026): Use 100 bars to ensure sufficient data for ATR and indicators
                # Previously 50 bars could cause insufficient warmup for some calculations
                data = self.fetch_data(symbol, bars=100)
                if data is None:
                    logger.warning(f"EXIT_CHECK | {symbol} | No data available")
                    continue

                current_price = data['close'].iloc[-1]
                bar_high = data['high'].iloc[-1]
                bar_low = data['low'].iloc[-1]
                entry_price = position['entry_price']
                direction = position.get('direction', 'LONG')

                # Calculate P&L for logging
                if direction == 'LONG':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100

                # FIX (Jan 2026): Increment bars_held for ExitManager minimum hold time
                # Backtest.py:726-727 does this - without it, min_hold_bars never triggers
                if self.use_tiered_exits and self.exit_manager and symbol in self.exit_manager.positions:
                    self.exit_manager.increment_bars_held(symbol)
                    bars_held = self.exit_manager.positions[symbol].bars_held
                    state = self.exit_manager.positions[symbol]
                    hard_stop = entry_price * (1 - state.hard_stop_pct) if direction == 'LONG' else entry_price * (1 + state.hard_stop_pct)
                    logger.info(f"EXIT_CHECK | {symbol} | ${current_price:.2f} ({pnl_pct:+.2f}%) | Stop: ${hard_stop:.2f} | Bars: {bars_held}")

                exit_signal = self.check_exit(symbol, position, current_price, bar_high, bar_low, data)

                if exit_signal and exit_signal.get('exit'):
                    logger.info(f"EXIT_TRIGGER | {symbol} | {exit_signal.get('reason', 'unknown')} @ ${exit_signal.get('price', current_price):.2f}")
                    self.execute_exit(symbol, exit_signal)

            # 3. Check entries for watchlist (only if not at position limit and entries allowed)
            max_positions = self.config.get('risk_management', {}).get('max_open_positions', 5)
            if len(self.open_positions) >= max_positions:
                logger.info(f"At max positions ({max_positions}), skipping entries")
                return

            # Check if drawdown guard blocks entries
            if self.drawdown_guard.enabled and not self.drawdown_guard.entries_allowed:
                logger.warning(f"DRAWDOWN_GUARD | Entries blocked at tier {self.drawdown_guard.tier.name}")
                return

            # Collect ALL qualifying signals first, then pick highest confidence
            # This ensures deterministic selection matching backtest behavior
            qualifying_signals = []
            first_candle_logged = False  # FIX (Jan 2026): Log first candle for verification
            bar_validated = False  # FIX (Jan 2026): Track if we've validated the bar

            for symbol in self.watchlist:
                if symbol in self.open_positions:
                    continue

                data = self.fetch_data(symbol)
                if data is None or len(data) < 30:
                    continue

                # FIX (Jan 2026): Log the first symbol's candle timestamp at INFO level
                # This verifies we're getting the correct completed candle
                if not first_candle_logged and 'timestamp' in data.columns:
                    latest_ts = data['timestamp'].iloc[-1]
                    logger.info(f"CANDLE_CHECK | Latest bar timestamp: {latest_ts}")
                    first_candle_logged = True

                # FIX (Jan 2026): Validate bar is COMPLETE before ANY entries
                # This prevents trading on incomplete forming bars
                if not bar_validated and 'timestamp' in data.columns:
                    if not validate_candle_timestamp(data, expected_candle_hour):
                        logger.warning("ENTRY_BLOCKED | Bar validation failed - skipping all entries this cycle")
                        break  # Exit the loop entirely, no entries this cycle
                    bar_validated = True

                current_price = data['close'].iloc[-1]
                entry_signal = self.check_entry(symbol, data, current_price)

                if entry_signal and entry_signal.get('action') in ['BUY', 'SELL']:
                    qualifying_signals.append({
                        'symbol': symbol,
                        'signal': entry_signal,
                        'price': current_price
                    })

            # Sort by confidence (highest first) for deterministic selection
            qualifying_signals.sort(key=lambda x: x['signal'].get('confidence', 0), reverse=True)

            # Execute trades for top signals up to position limit
            for entry in qualifying_signals:
                if len(self.open_positions) >= max_positions:
                    break

                symbol = entry['symbol']
                entry_signal = entry['signal']
                current_price = entry['price']
                direction = entry_signal.get('direction', 'LONG')

                logger.info(f"Selected {symbol} with confidence {entry_signal.get('confidence', 0):.1f} (best of {len(qualifying_signals)} signals)")

                self.execute_entry(
                    symbol=symbol,
                    direction=direction,
                    price=current_price,
                    strategy=entry_signal.get('strategy', 'Unknown'),
                    reasoning=entry_signal.get('reasoning', '')
                )

            logger.info(f"=== Cycle Complete: {len(self.open_positions)} positions ===")

        except Exception as e:
            logger.error(f"Trading cycle error: {e}")

    def start(self):
        """Start the trading bot."""
        if not self.watchlist:
            logger.error("Cannot start: No symbols in universe")
            return False

        # Sync initial state
        self.sync_account()
        self.sync_positions()

        self.running = True
        logger.info("Trading Bot started")
        return True

    def stop(self):
        """Stop the trading bot."""
        self.running = False
        logger.info("Trading Bot stopped")


def get_seconds_until_next_hour(buffer_minutes: int = 2) -> int:
    """
    Calculate seconds until X minutes past the next hour.

    FIX (Jan 2026): Smart hourly scheduling - align bot cycles to candle boundaries.
    For 1-hour bars, candles close at :00 (e.g., 9 AM candle closes at 10:00).
    We add a buffer (default 2 min) to ensure the candle is fully available.

    Args:
        buffer_minutes: Minutes after the hour to run (default 2)

    Returns:
        Seconds until next run time (e.g., 10:02, 11:02, etc.)
    """
    eastern = pytz.timezone('America/New_York')
    now = datetime.now(eastern)

    # Calculate target time: next hour + buffer minutes
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    target_time = next_hour + timedelta(minutes=buffer_minutes)

    # If we're already past the buffer (e.g., it's 10:05 and buffer is 2 min),
    # wait until the next hour
    if now >= target_time - timedelta(hours=1) and now.minute >= buffer_minutes:
        # We're past this hour's target, wait for next hour
        pass  # target_time is already set to next hour + buffer
    elif now.minute < buffer_minutes:
        # We're before this hour's buffer, target is this hour + buffer
        target_time = now.replace(minute=buffer_minutes, second=0, microsecond=0)

    seconds_until = (target_time - now).total_seconds()

    # Safety: ensure we wait at least 1 minute, at most ~60 minutes
    seconds_until = max(60, min(seconds_until, 3660))

    return int(seconds_until)


def validate_candle_timestamp(data: pd.DataFrame, expected_hour: int = None,
                               bar_duration_minutes: int = 60) -> bool:
    """
    Validate that we received a COMPLETED candle.

    FIX (Jan 2026): Ensure we're trading on completed candles only.
    A bar is incomplete if current_time < bar_start + bar_duration.

    For 1-hour bars:
    - 9:30 bar completes at 10:30 (first bar of day, market opens at 9:30)
    - 10:00 bar completes at 11:00
    - etc.

    Args:
        data: DataFrame with 'timestamp' column
        expected_hour: Expected hour of the latest bar (optional)
        bar_duration_minutes: Duration of each bar in minutes (default 60)

    Returns:
        True if candle is complete and valid, False if incomplete/stale/wrong
    """
    if data is None or len(data) == 0:
        return False

    if 'timestamp' not in data.columns:
        return True  # Can't validate without timestamp

    eastern = pytz.timezone('America/New_York')
    now = datetime.now(eastern)
    latest_ts = data['timestamp'].iloc[-1]

    # Ensure timezone-aware
    if latest_ts.tzinfo is None:
        latest_ts = eastern.localize(latest_ts)
    else:
        latest_ts = latest_ts.astimezone(eastern)

    # FIX (Jan 2026): Check if bar is COMPLETE
    # A bar starting at X:XX is not complete until X:XX + bar_duration
    bar_completion_time = latest_ts + timedelta(minutes=bar_duration_minutes)

    if now < bar_completion_time:
        # Bar is still forming - this is incomplete data!
        minutes_until_complete = (bar_completion_time - now).total_seconds() / 60
        logger.warning(
            f"INCOMPLETE_BAR | Bar {latest_ts.strftime('%H:%M')} not complete until "
            f"{bar_completion_time.strftime('%H:%M')} ({minutes_until_complete:.0f}m remaining) - SKIPPING"
        )
        return False

    # For 1-hour bars running at HH:02, we expect the (HH-1):00 candle
    # But handle 9:30 market open - first bar is 9:30, not 9:00
    if expected_hour is not None:
        bar_hour = latest_ts.hour
        bar_minute = latest_ts.minute

        # Special case: 9:30 bar is valid when expecting 9:00 hour
        # (market opens at 9:30, not 9:00)
        is_market_open_bar = (bar_hour == 9 and bar_minute == 30)
        expected_matches = (bar_hour == expected_hour) or (expected_hour == 9 and is_market_open_bar)

        if not expected_matches:
            logger.warning(f"CANDLE_VALIDATION | Expected {expected_hour}:00 bar, got {latest_ts.strftime('%H:%M')}")
            return False

    # Check that bar is from today (or yesterday if early morning)
    bar_date = latest_ts.date()
    today = now.date()
    yesterday = (now - timedelta(days=1)).date()

    if bar_date not in [today, yesterday]:
        logger.warning(f"CANDLE_VALIDATION | Bar date {bar_date} is too old (today={today})")
        return False

    # Check bar isn't too old (max 2 hours old for 1-hour bars)
    age_hours = (now - latest_ts).total_seconds() / 3600
    if age_hours > 2:
        logger.warning(f"CANDLE_VALIDATION | Bar is {age_hours:.1f} hours old - may be stale")
        # Don't return False here, just warn - market may have been closed

    return True


def main():
    """Main entry point for trading bot."""
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated list of symbols from scanner (overrides config)')
    parser.add_argument('--candle-delay', type=int, default=2,
                        help='Minutes after hour to run cycle (default: 2)')
    args = parser.parse_args()

    # Parse symbols if provided
    scanner_symbols = None
    if args.symbols:
        scanner_symbols = [s.strip() for s in args.symbols.split(',')]
        print(f"[SCANNER] Using {len(scanner_symbols)} symbols from scanner: {scanner_symbols}")

    bot = TradingBot(config_path=args.config, scanner_symbols=scanner_symbols)
    eastern = pytz.timezone('America/New_York')

    try:
        if bot.start():
            logger.info(f"Bot started with candle-delay={args.candle_delay} minutes")

            # FIX (Jan 2026): Smart hourly scheduling - align to candle boundaries
            # Instead of sleeping 3600s from start, wait until :02 past next hour
            while bot.running:
                now = datetime.now(eastern)

                # Run trading cycle immediately on first iteration
                logger.info(f"=== Running cycle at {now.strftime('%H:%M:%S')} EST ===")
                bot.run_trading_cycle()

                # Calculate wait time until next hour + buffer
                wait_seconds = get_seconds_until_next_hour(buffer_minutes=args.candle_delay)
                next_run = now + timedelta(seconds=wait_seconds)

                logger.info(f"Next cycle at {next_run.strftime('%H:%M')} EST (waiting {wait_seconds//60}m {wait_seconds%60}s)")

                # Sleep until next cycle
                time.sleep(wait_seconds)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        bot.stop()


if __name__ == '__main__':
    main()
