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
)
from strategies import StrategyManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('trading_bot')


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

            # Daily P&L tracking
            now = datetime.now()
            if self.current_trading_day != now.date():
                self.current_trading_day = now.date()
                self.daily_starting_capital = self.portfolio_value
                self.daily_pnl = 0.0
                self.kill_switch_triggered = False
            else:
                self.daily_pnl = self.portfolio_value - self.daily_starting_capital

            # Kill switch check
            if self.daily_starting_capital > 0:
                daily_loss_pct = -self.daily_pnl / self.daily_starting_capital
                max_daily_loss = self.config.get('risk_management', {}).get('max_daily_loss_pct', 3.0) / 100
                if daily_loss_pct >= max_daily_loss:
                    if not self.kill_switch_triggered:
                        self.kill_switch_triggered = True
                        logger.warning(f"KILL SWITCH: Daily loss {daily_loss_pct*100:.2f}% >= {max_daily_loss*100}%")

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

                # Only track symbols in our universe
                if symbol not in self.watchlist:
                    continue

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
            # 1. Trailing stop check (matches backtest.py logic)
            if trailing_enabled:
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

            # 2. Hard stop (tiered exit via exit manager)
            # FIX (Jan 2026): Only call ExitManager when tiered exits enabled (matches backtest.py:692)
            if self.use_tiered_exits and self.exit_manager:
                # Calculate ATR from historical data for proper trailing stops
                atr = self._calculate_atr(data, period=14) if data is not None else 0.0
                # FIX (Jan 2026): Use bar_low instead of current_price to match backtest behavior
                # This ensures stops trigger on intra-bar lows, not just on close
                exit_action = self.exit_manager.evaluate_exit(symbol, bar_low, atr)
                if exit_action:
                    # FIX (Jan 2026): Normalize 'hard_stop' to 'stop_loss' to match backtest.py:699-701
                    reason = exit_action.get('reason', 'exit_manager')
                    if reason == 'hard_stop':
                        reason = 'stop_loss'
                    return {
                        'exit': True,
                        'reason': reason,
                        'price': exit_action.get('stop_price', current_price),
                        'qty': exit_action.get('qty', qty)
                    }

            # 3. Simple stop loss fallback - only when tiered exits disabled (matches backtest.py)
            # FIX (Jan 2026): This was running even with ExitManager active, causing double-check
            # In backtest.py, this only runs when use_tiered_exits is False
            if not self.use_tiered_exits:
                stop_price = entry_price * (1 - hard_stop_pct)
                if bar_low <= stop_price:
                    return {
                        'exit': True,
                        'reason': 'stop_loss',  # FIX (Jan 2026): Match backtest.py naming
                        'price': stop_price,
                        'qty': qty
                    }

            # 4. Take profit (uses take_profit_pct from risk_management config)
            # FIX (Jan 2026): Only check take profit when tiered exits disabled (matches backtest.py:710)
            # With tiered exits enabled, ExitManager handles profit-taking via tiered system
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
            # 1. Trailing stop check for SHORT (matches backtest.py logic)
            if trailing_enabled:
                profit_pct = (entry_price - lowest) / entry_price
                if not self.trailing_stops[symbol]['activated'] and profit_pct >= trailing_activation:
                    self.trailing_stops[symbol]['activated'] = True
                    if trailing_move_to_breakeven:
                        self.trailing_stops[symbol]['price'] = entry_price
                    else:
                        self.trailing_stops[symbol]['price'] = lowest * (1 + trailing_trail)

                if self.trailing_stops[symbol]['activated']:
                    new_trail = lowest * (1 + trailing_trail)
                    # For shorts, lower trail price is better
                    if new_trail < self.trailing_stops[symbol]['price'] or self.trailing_stops[symbol]['price'] == 0:
                        self.trailing_stops[symbol]['price'] = new_trail

                    if bar_high >= self.trailing_stops[symbol]['price']:
                        return {
                            'exit': True,
                            'reason': 'trailing_stop',
                            'price': self.trailing_stops[symbol]['price'],
                            'qty': qty
                        }

            # 2. Hard stop for SHORT
            # FIX (Jan 2026): SHORT always needs hard stop - ExitManager only handles LONG!
            # BUG: Previous code only ran when tiered exits disabled, leaving SHORT unprotected
            stop_price = entry_price * (1 + hard_stop_pct)
            if bar_high >= stop_price:
                return {
                    'exit': True,
                    'reason': 'stop_loss',
                    'price': stop_price,
                    'qty': qty
                }

            # 3. Take profit for SHORT (uses take_profit_pct from risk_management config)
            # FIX (Jan 2026): SHORT take profit runs unconditionally (matches backtest.py:796-801)
            # LONG uses tiered exits (ExitManager) for profit taking, but SHORT doesn't have ExitManager
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

            # Wait for fill (simplified - real impl would poll)
            if order and order.status in ['filled', 'new', 'accepted']:
                fill_price = float(order.filled_avg_price) if hasattr(order, 'filled_avg_price') and order.filled_avg_price else price
                fill_qty = int(order.filled_qty) if hasattr(order, 'filled_qty') and order.filled_qty else qty

                # Register with exit manager (LONG only when tiered exits enabled)
                # FIX (Jan 2026): Match backtest.py:922-923 - only register LONG with tiered exits
                if direction == 'LONG' and self.use_tiered_exits and self.exit_manager:
                    self.exit_manager.register_position(
                        symbol=symbol,
                        entry_price=fill_price,
                        quantity=fill_qty,
                        entry_time=datetime.now()
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

                # Unregister from exit manager (LONG only when tiered exits enabled)
                # FIX (Jan 2026): Match registration logic - only LONG with tiered exits
                if direction == 'LONG' and self.use_tiered_exits and self.exit_manager:
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

            df = self.data_fetcher.get_historical_data_range(
                symbol=symbol,
                timeframe=self.timeframe,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            if df is None or df.empty:
                logger.warning(f"No data for {symbol}")
                return None

            # Add indicators
            df = self.indicators.add_all_indicators(df)

            logger.debug(f"Fetched {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def run_trading_cycle(self):
        """
        Run one trading cycle.

        Called every hour (or on demand):
        1. Sync account and positions
        2. Check exits for all positions
        3. Check entries for watchlist symbols
        """
        try:
            logger.info("=== Trading Cycle Start ===")

            # 1. Sync state
            self.sync_account()
            self.sync_positions()

            # 2. Check exits for all positions
            for symbol, position in list(self.open_positions.items()):
                # FIX (Jan 2026): Use 100 bars to ensure sufficient data for ATR and indicators
                # Previously 50 bars could cause insufficient warmup for some calculations
                data = self.fetch_data(symbol, bars=100)
                if data is None:
                    continue

                current_price = data['close'].iloc[-1]
                bar_high = data['high'].iloc[-1]
                bar_low = data['low'].iloc[-1]

                exit_signal = self.check_exit(symbol, position, current_price, bar_high, bar_low, data)

                if exit_signal and exit_signal.get('exit'):
                    self.execute_exit(symbol, exit_signal)

            # 3. Check entries for watchlist (only if not at position limit)
            max_positions = self.config.get('risk_management', {}).get('max_open_positions', 5)
            if len(self.open_positions) >= max_positions:
                logger.info(f"At max positions ({max_positions}), skipping entries")
                return

            for symbol in self.watchlist:
                if symbol in self.open_positions:
                    continue

                data = self.fetch_data(symbol)
                if data is None or len(data) < 30:
                    continue

                current_price = data['close'].iloc[-1]
                entry_signal = self.check_entry(symbol, data, current_price)

                if entry_signal and entry_signal.get('action') in ['BUY', 'SELL']:
                    direction = entry_signal.get('direction', 'LONG')
                    self.execute_entry(
                        symbol=symbol,
                        direction=direction,
                        price=current_price,
                        strategy=entry_signal.get('strategy', 'Unknown'),
                        reasoning=entry_signal.get('reasoning', '')
                    )

                    # Check if we've hit position limit
                    if len(self.open_positions) >= max_positions:
                        break

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


def main():
    """Main entry point for trading bot."""
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated list of symbols from scanner (overrides config)')
    args = parser.parse_args()

    # Parse symbols if provided
    scanner_symbols = None
    if args.symbols:
        scanner_symbols = [s.strip() for s in args.symbols.split(',')]
        print(f"[SCANNER] Using {len(scanner_symbols)} symbols from scanner: {scanner_symbols}")

    bot = TradingBot(config_path=args.config, scanner_symbols=scanner_symbols)

    try:
        if bot.start():
            # Keep running until interrupted
            while bot.running:
                # Run trading cycle
                bot.run_trading_cycle()
                # Wait for next cycle (1 hour by default)
                time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        bot.stop()


if __name__ == '__main__':
    main()
