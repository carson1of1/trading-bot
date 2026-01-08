"""
Backtest Runner - Trade Simulation Engine

Run backtests with realistic trade simulation.
Uses config.yaml and universe.yaml for settings.

Features:
- Full backtest loop with realistic trade simulation
- Entry/Exit management with slippage and commission
- Trailing stop logic
- Tiered exit management (hard_stop, profit_floor, atr_trailing)
- Max hold time enforcement
- EOD close simulation
- Daily P&L tracking and kill switch
- Both LONG and SHORT position handling

Usage:
    python backtest.py
    python backtest.py --symbols AAPL MSFT SPY
    python backtest.py --symbols SPY --start 2025-11-01 --end 2025-12-15
"""

import argparse
import logging
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
# statistics not needed - median removed as unused
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from core import (
    EntryGate,
    ExitManager,
    FakeBroker,
    RiskManager,
    TechnicalIndicators,
    VolatilityScanner,
    YFinanceDataFetcher,
)
from core.risk import DailyDrawdownGuard, DrawdownTier
from strategies import StrategyManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('backtest')


class Backtest1Hour:
    """
    Backtesting engine with realistic trade simulation.

    Implements:
    - Slippage and commission
    - Entry gate filtering
    - Exit manager (tiered exits)
    - Trailing stop logic
    - Max hold time enforcement
    - EOD close simulation
    - Daily P&L tracking and kill switch
    """

    # Default cost parameters
    DEFAULT_ENTRY_SLIPPAGE = 0.0005   # 0.05%
    DEFAULT_EXIT_SLIPPAGE = 0.0005    # 0.05%
    DEFAULT_STOP_SLIPPAGE = 0.002     # 0.20% - worse fills on stop losses
    DEFAULT_BID_ASK_SPREAD = 0.0002   # 0.02%
    COMMISSION = 0.0                   # Alpaca has no commission
    COOLDOWN_BARS = 1                  # 1 bar cooldown (1 hour)
    DEFAULT_MAX_HOLD_BARS = 48         # Fallback if config not available

    def __init__(
        self,
        initial_capital: float = 100000.0,
        config: Dict = None,
        longs_only: bool = False,
        shorts_only: bool = False,
        scanner_enabled: bool = None,
        kill_switch_trace: bool = False
    ):
        """
        Initialize the backtester.

        Args:
            initial_capital: Starting capital
            config: Configuration dict (loads from config.yaml if None)
            longs_only: Only take LONG positions
            shorts_only: Only take SHORT positions
            scanner_enabled: Override scanner enabled setting
        """
        self.initial_capital = initial_capital
        self.config = config or self._load_config()
        self.longs_only = longs_only
        self.shorts_only = shorts_only
        self.kill_switch_trace = kill_switch_trace
        self._kill_switch_trace_log = []

        # Initialize volatility scanner if enabled
        scanner_config = self.config.get('volatility_scanner', {})
        if scanner_enabled is not None:
            self.scanner_enabled = scanner_enabled
        else:
            self.scanner_enabled = scanner_config.get('enabled', False)

        if self.scanner_enabled:
            self.scanner = VolatilityScanner(scanner_config)
        else:
            self.scanner = None

        # Execution costs from config
        exec_config = self.config.get('execution', {})
        slippage_bps = exec_config.get('slippage_bps', 0)
        half_spread_bps = exec_config.get('half_spread_bps', 0)

        # Convert bps to decimal
        self.ENTRY_SLIPPAGE = slippage_bps / 10000 if slippage_bps else self.DEFAULT_ENTRY_SLIPPAGE
        self.EXIT_SLIPPAGE = slippage_bps / 10000 if slippage_bps else self.DEFAULT_EXIT_SLIPPAGE
        self.BID_ASK_SPREAD = half_spread_bps / 10000 if half_spread_bps else self.DEFAULT_BID_ASK_SPREAD
        self.STOP_SLIPPAGE = (slippage_bps * 4) / 10000 if slippage_bps else self.DEFAULT_STOP_SLIPPAGE

        # Initialize components
        self.indicators = TechnicalIndicators()
        self.data_fetcher = YFinanceDataFetcher()

        # Risk management from config
        risk_config = self.config.get('risk_management', {})
        entry_config = self.config.get('entry_gate', {})
        exit_config = self.config.get('exit_manager', {})

        # Position limits from config (defaults to 5)
        self.max_open_positions = risk_config.get('max_open_positions', 5)

        # Transform config keys for RiskManager compatibility
        # RiskManager expects: max_position_size (decimal), max_positions
        # Config may have: max_position_size_pct (percentage), max_open_positions
        risk_manager_config = risk_config.copy()
        if 'max_position_size_pct' in risk_manager_config:
            risk_manager_config['max_position_size'] = risk_manager_config['max_position_size_pct'] / 100
        if 'max_open_positions' in risk_manager_config:
            risk_manager_config['max_positions'] = risk_manager_config['max_open_positions']

        self.risk_manager = RiskManager(risk_manager_config)
        self.entry_gate = EntryGate(entry_config)

        # Build exit manager settings
        exit_settings = {
            'hard_stop_pct': abs(exit_config.get('tier_0_hard_stop', -0.02)) * 100,
            'profit_floor_activation_pct': exit_config.get('tier_1_profit_floor', 0.0125) * 100,  # FIX: correct key name
            'trailing_activation_pct': exit_config.get('tier_2_atr_trailing', 0.03) * 100,
            'partial_tp_pct': exit_config.get('tier_3_partial_take', 0.04) * 100,
            'partial_tp2_pct': exit_config.get('tier_4_partial_take2', 0.05) * 100,  # FIX: add tier 4 to match bot
            'partial_tp2_size': exit_config.get('tier_4_partial_take2_size', 1.0),
        }
        bot_settings = {'risk': exit_settings}
        self.exit_manager = ExitManager(bot_settings)
        self.use_tiered_exits = exit_config.get('enabled', True)

        # Strategy manager
        self.strategy_manager = StrategyManager(self.config)

        # Default risk settings
        self.default_stop_loss_pct = risk_config.get('stop_loss_pct', 2.0) / 100
        self.default_take_profit_pct = risk_config.get('take_profit_pct', 4.0) / 100

        # Max hold hours from config
        self.max_hold_hours = exit_config.get('max_hold_hours', self.DEFAULT_MAX_HOLD_BARS)

        # Daily loss kill switch (legacy - now using tiered DailyDrawdownGuard)
        self.max_daily_loss_pct = risk_config.get('max_daily_loss_pct', 3.0) / 100
        self.daily_loss_kill_switch_enabled = True

        # Tiered drawdown guard (ODE-118)
        self.drawdown_guard = DailyDrawdownGuard(self.config)

        # EOD close simulation
        self.eod_close_bar_hour = 15  # Close on bars starting at 3 PM or later
        self.eod_close_enabled = True

        # Trailing stop configuration
        trailing_config = self.config.get('trailing_stop', {})
        self.trailing_stop_enabled = trailing_config.get('enabled', True)
        self.trailing_activation_pct = trailing_config.get('activation_pct', 0.5) / 100
        self.trailing_trail_pct = trailing_config.get('trail_pct', 0.5) / 100
        self.trailing_move_to_breakeven = trailing_config.get('move_to_breakeven', True)

        # Backtest date range (set by run(), None for direct simulate_trades calls)
        self._backtest_start_date = None

        # State tracking
        self._reset_state()

    def _load_config(self) -> Dict:
        """Load configuration from config.yaml."""
        config_path = Path(__file__).parent / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def _reset_state(self):
        """Reset all state for a new backtest."""
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital

        self.trades = []
        self.equity_curve = []
        self.positions = {}
        # Note: Do NOT reset _backtest_start_date here - it's set by run() before calling simulate_trades_interleaved()

        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0

        # Daily P&L tracking
        self.daily_pnl = 0.0
        self.daily_starting_capital = self.initial_capital
        self.current_trading_day = None
        self.kill_switch_triggered = False

        # Reset drawdown guard for new backtest (ODE-118)
        if hasattr(self, 'drawdown_guard'):
            self.drawdown_guard = DailyDrawdownGuard(self.config)

        # Track partial liquidation state per day
        self._partial_liquidation_done_today = False

        # Reset entry gate
        if self.entry_gate:
            self.entry_gate.reset()

        # Diagnostics
        self._diag_exit_reasons = defaultdict(lambda: defaultdict(int))
        self._diag_entry_blocks = defaultdict(lambda: defaultdict(int))
        self._diag_kill_switch_blocks_per_day = defaultdict(int)

        # Scanner state
        self._daily_scanned_symbols = {}

        # Kill switch trace log (for debugging position limit blocking)
        self._kill_switch_trace_log = []

    def _build_daily_scan_results(
        self,
        historical_data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str
    ) -> Dict[str, List[str]]:
        """
        Build a dict of date -> scanned symbols for the backtest period.

        Uses the scanner's scan_historical method. NO LOOK-AHEAD BIAS.

        Args:
            historical_data: Dict mapping symbol -> DataFrame with OHLCV
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Dict mapping date_str -> list of symbols to trade that day
        """
        if not self.scanner:
            return {}

        daily_scans = {}
        symbols = list(historical_data.keys())

        # Get trading days from data
        all_timestamps = set()
        for df in historical_data.values():
            if 'timestamp' in df.columns:
                all_timestamps.update(pd.to_datetime(df['timestamp']).dt.date)

        # Sort and filter to backtest period
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date()
        trading_days = sorted([d for d in all_timestamps if start_dt <= d <= end_dt])

        logger.info(f"Building daily scan results for {len(trading_days)} trading days...")

        for date in trading_days:
            date_str = date.strftime('%Y-%m-%d')
            scanned = self.scanner.scan_historical(
                date=date_str,
                symbols=symbols,
                historical_data=historical_data
            )
            daily_scans[date_str] = scanned

        # Log summary
        unique_symbols = set()
        for syms in daily_scans.values():
            unique_symbols.update(syms)
        logger.info(f"Scanner selected {len(unique_symbols)} unique symbols across all days")

        return daily_scans

    def _is_symbol_scanned_for_date(self, symbol: str, timestamp) -> bool:
        """Check if a symbol was in the scanned list for a given date."""
        if not self.scanner_enabled:
            return True

        if isinstance(timestamp, (datetime, pd.Timestamp)):
            date_str = timestamp.strftime('%Y-%m-%d')
        elif isinstance(timestamp, str):
            date_str = pd.to_datetime(timestamp).strftime('%Y-%m-%d')
        else:
            return True

        scanned_symbols = self._daily_scanned_symbols.get(date_str, [])
        return symbol in scanned_symbols

    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch 1-hour historical data for a symbol with warmup for SMA200.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data and indicators, or None if unavailable
        """
        try:
            # Add 40 days warmup for SMA200 calculation
            warmup_days = 40
            warmup_start = (
                datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=warmup_days)
            ).strftime('%Y-%m-%d')

            df = self.data_fetcher.get_historical_data_range(
                symbol=symbol,
                timeframe='1Hour',
                start_date=warmup_start,
                end_date=end_date
            )

            if df is None or df.empty:
                logger.warning(f"No data for {symbol}")
                return None

            # Add indicators
            df = self.indicators.add_all_indicators(df)

            logger.info(f"Fetched {len(df)} 1-hour bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}", exc_info=True)
            return None

    def fetch_data_parallel(
        self, symbols: List[str], start_date: str, end_date: str, max_workers: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols in parallel.

        FIX (Jan 7, 2026): Added parallel data fetching for 5-10x speedup.
        Old: 100 symbols x 0.5s = 50+ seconds
        New: 100 symbols / 10 workers = ~5-10 seconds

        Args:
            symbols: List of stock ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_workers: Maximum parallel threads (default 10)

        Returns:
            Dict mapping symbol to DataFrame with OHLCV data
        """
        results = {}
        logger.info(f"Parallel fetching data for {len(symbols)} symbols with {max_workers} workers...")
        start_time = datetime.now()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.fetch_data, symbol, start_date, end_date): symbol
                for symbol in symbols
            }

            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1

                try:
                    data = future.result()
                    if data is not None and len(data) >= 30:
                        results[symbol] = data
                        logger.debug(f"[{completed}/{len(symbols)}] {symbol}: {len(data)} bars")
                    else:
                        logger.debug(f"[{completed}/{len(symbols)}] {symbol}: insufficient data")
                except Exception as e:
                    logger.warning(f"[{completed}/{len(symbols)}] {symbol}: {e}")

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Parallel fetch complete: {len(results)}/{len(symbols)} symbols "
            f"in {elapsed:.1f}s ({elapsed/len(symbols):.2f}s per symbol)"
        )

        return results

    def generate_signals_parallel(
        self, symbols_data: Dict[str, pd.DataFrame], max_workers: int = 8
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate signals for multiple symbols in parallel.

        FIX (Jan 7, 2026): Added parallel signal generation for 3-5x speedup.
        Signal generation is CPU-bound and independent per symbol.

        Args:
            symbols_data: Dict mapping symbol to DataFrame with OHLCV data
            max_workers: Maximum parallel threads (default 8)

        Returns:
            Dict mapping symbol to DataFrame with signals added
        """
        results = {}
        symbols = list(symbols_data.keys())
        logger.info(f"Parallel generating signals for {len(symbols)} symbols with {max_workers} workers...")
        start_time = datetime.now()

        def process_symbol(symbol: str) -> tuple:
            data = symbols_data[symbol]
            signals_df = self.generate_signals(symbol, data)
            return symbol, signals_df

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(process_symbol, symbol): symbol
                for symbol in symbols
            }

            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1

                try:
                    sym, signals_df = future.result()
                    if signals_df is not None:
                        results[sym] = signals_df
                        logger.debug(f"[{completed}/{len(symbols)}] {symbol}: signals generated")
                except Exception as e:
                    logger.warning(f"[{completed}/{len(symbols)}] {symbol}: signal error - {e}")

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Parallel signal generation complete: {len(results)}/{len(symbols)} symbols "
            f"in {elapsed:.1f}s ({elapsed/len(symbols):.2f}s per symbol)"
        )

        return results

    def generate_signals(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using strategies.

        If scanner is enabled, only generates signals for days when the symbol
        was in the scanned list.

        Args:
            symbol: Stock ticker symbol
            data: DataFrame with OHLCV data and indicators

        Returns:
            DataFrame with signal columns added
        """
        data = data.copy()
        data['signal'] = 0
        data['confidence'] = 0.0
        data['strategy'] = ''
        data['reasoning'] = ''

        # Warmup period for hourly bars
        MIN_WARMUP = min(20, max(10, int(len(data) * 0.2)))

        for i in range(MIN_WARMUP, len(data)):
            historical_data = data.iloc[:i].copy()
            current_price = data.iloc[i]['close']
            timestamp = data.iloc[i].get('timestamp', None)

            if pd.isna(current_price):
                continue

            # Scanner filter
            if self.scanner_enabled and timestamp is not None:
                if not self._is_symbol_scanned_for_date(symbol, timestamp):
                    self._diag_entry_blocks[symbol]['scanner_filtered'] += 1
                    continue

            try:
                signal = self.strategy_manager.get_best_signal(
                    symbol=symbol,
                    data=historical_data,
                    current_price=current_price,
                    indicators=self.indicators
                )

                if signal and signal.get('action') != 'HOLD':
                    confidence = signal.get('confidence', 0)
                    conf_threshold = self.strategy_manager.confidence_threshold

                    if signal['action'] == 'BUY' and confidence >= conf_threshold:
                        data.at[data.index[i], 'signal'] = 1
                    elif signal['action'] == 'BUY' and confidence < conf_threshold:
                        self._diag_entry_blocks[symbol]['confidence'] += 1
                    elif signal['action'] == 'SELL':
                        data.at[data.index[i], 'signal'] = -1

                    data.at[data.index[i], 'confidence'] = confidence
                    data.at[data.index[i], 'strategy'] = signal.get('strategy', 'Unknown')
                    data.at[data.index[i], 'reasoning'] = signal.get('reasoning', '')

            except Exception as e:
                logger.debug(f"Signal generation error at bar {i}: {e}")
                continue

        return data

    def _calculate_atr(self, data: pd.DataFrame, bar_index: int, period: int = 14) -> float:
        """Calculate ATR using only past data (no look-ahead)."""
        if bar_index < period + 1:
            return 0.0

        try:
            start_idx = max(0, bar_index - period - 1)
            end_idx = bar_index

            hist_data = data.iloc[start_idx:end_idx].copy()
            if len(hist_data) < period:
                return 0.0

            high = hist_data['high'].values
            low = hist_data['low'].values
            close = hist_data['close'].values

            tr = np.zeros(len(high))
            for j in range(1, len(high)):
                prev_close = close[j - 1]
                tr[j] = max(
                    high[j] - low[j],
                    abs(high[j] - prev_close),
                    abs(low[j] - prev_close)
                )

            recent_tr = tr[-period:]
            atr = np.mean(recent_tr) if len(recent_tr) == period else 0.0

            return float(atr) if not np.isnan(atr) else 0.0

        except Exception:
            return 0.0

    def simulate_trades(self, symbol: str, data: pd.DataFrame) -> List[Dict]:
        """
        Simulate trading on historical data with realistic costs.
        Supports BOTH long and short positions.

        Implements:
        - Entry slippage and commission
        - Exit via stop-loss, take-profit, trailing stop, signals
        - Max hold time enforcement
        - Tiered exit management
        - SHORT SELLING for downtrend profits

        Args:
            symbol: Stock ticker symbol
            data: DataFrame with OHLCV data, indicators, and signals

        Returns:
            List of trade dicts
        """
        trades = []
        position_direction = None  # 'LONG', 'SHORT', or None
        entry_price = 0.0
        entry_index = 0
        entry_time = None
        shares = 0
        stop_loss_price = 0.0
        take_profit_price = 0.0
        highest_price = 0.0
        lowest_price = float('inf')
        last_trade_bar = -999
        pending_entry = None
        entry_strategy = ''
        entry_reasoning = ''

        # Trailing stop state
        trailing_activated = False
        trailing_stop_price = 0.0

        for i in range(len(data)):
            row = data.iloc[i]
            current_price = row['close']
            timestamp = row.get('timestamp', i)

            # Skip NaN prices
            if pd.isna(current_price) or pd.isna(row.get('open')) or pd.isna(row.get('high')) or pd.isna(row.get('low')):
                continue

            bar_high = row.get('high', current_price)
            bar_low = row.get('low', current_price)

            # ============ DAILY RESET & EOD HANDLING ============
            bar_datetime = None
            bar_hour = None
            bar_date = None

            if isinstance(timestamp, (datetime, pd.Timestamp)):
                bar_datetime = timestamp
                bar_hour = timestamp.hour
                bar_date = timestamp.date() if hasattr(timestamp, 'date') else None
            elif isinstance(timestamp, str):
                try:
                    bar_datetime = pd.to_datetime(timestamp)
                    bar_hour = bar_datetime.hour
                    bar_date = bar_datetime.date()
                except:
                    pass

            # Reset daily P&L on new trading day
            if bar_date is not None and bar_date != self.current_trading_day:
                if self.current_trading_day is not None:
                    logger.debug(f"New trading day: {bar_date}, resetting daily P&L (was ${self.daily_pnl:.2f})")
                self.current_trading_day = bar_date
                self.daily_pnl = 0.0
                self.daily_starting_capital = self.portfolio_value
                self.kill_switch_triggered = False

            # ============ EXIT LOGIC (runs first, every bar) ============
            if position_direction is not None:
                exit_triggered = False
                exit_reason = ''
                exit_price = current_price
                exit_qty = shares

                # Update price tracking
                if not pd.isna(bar_high) and bar_high > highest_price:
                    highest_price = bar_high
                if not pd.isna(bar_low) and bar_low < lowest_price:
                    lowest_price = bar_low

                # ============ TRAILING STOP LOGIC ============
                if self.trailing_stop_enabled and not exit_triggered:
                    if position_direction == 'LONG':
                        current_profit_pct = (highest_price - entry_price) / entry_price

                        if not trailing_activated and current_profit_pct >= self.trailing_activation_pct:
                            trailing_activated = True
                            if self.trailing_move_to_breakeven:
                                trailing_stop_price = entry_price
                            else:
                                trailing_stop_price = highest_price * (1 - self.trailing_trail_pct)

                        if trailing_activated:
                            new_trail_price = highest_price * (1 - self.trailing_trail_pct)
                            if new_trail_price > trailing_stop_price:
                                trailing_stop_price = new_trail_price

                            if bar_low <= trailing_stop_price:
                                exit_triggered = True
                                exit_price = trailing_stop_price * (1 - self.STOP_SLIPPAGE - self.BID_ASK_SPREAD)
                                exit_reason = 'trailing_stop'

                    elif position_direction == 'SHORT':
                        current_profit_pct = (entry_price - lowest_price) / entry_price

                        if not trailing_activated and current_profit_pct >= self.trailing_activation_pct:
                            trailing_activated = True
                            if self.trailing_move_to_breakeven:
                                trailing_stop_price = entry_price
                            else:
                                trailing_stop_price = lowest_price * (1 + self.trailing_trail_pct)

                        if trailing_activated:
                            new_trail_price = lowest_price * (1 + self.trailing_trail_pct)
                            if new_trail_price < trailing_stop_price or trailing_stop_price == 0:
                                trailing_stop_price = new_trail_price

                            if bar_high >= trailing_stop_price:
                                exit_triggered = True
                                exit_price = trailing_stop_price * (1 + self.STOP_SLIPPAGE + self.BID_ASK_SPREAD)
                                exit_reason = 'trailing_stop'

                # Tiered exit logic via ExitManager (LONG and SHORT - Jan 2026)
                # FIX (Jan 8, 2026): Use tiered exits for both LONG and SHORT to match bot.py
                # bar_low/bar_high are passed as kwargs for stop checks, matching bot.py
                if not exit_triggered and self.use_tiered_exits and self.exit_manager:
                    current_atr = self._calculate_atr(data, i, period=14)
                    exit_action = self.exit_manager.evaluate_exit(
                        symbol, current_price, current_atr,
                        bar_high=bar_high, bar_low=bar_low
                    )

                    if exit_action:
                        exit_triggered = True
                        exit_reason = exit_action['reason']
                        exit_qty = exit_action.get('qty', shares)

                        # Apply slippage based on position direction
                        if position_direction == 'LONG':
                            if exit_reason in ['hard_stop', 'profit_floor', 'atr_trailing']:
                                exit_price = exit_action.get('stop_price', current_price) * (1 - self.STOP_SLIPPAGE - self.BID_ASK_SPREAD)
                            else:
                                exit_price = current_price * (1 - self.EXIT_SLIPPAGE - self.BID_ASK_SPREAD)
                        else:  # SHORT
                            if exit_reason in ['hard_stop', 'profit_floor', 'atr_trailing']:
                                exit_price = exit_action.get('stop_price', current_price) * (1 + self.STOP_SLIPPAGE + self.BID_ASK_SPREAD)
                            else:
                                exit_price = current_price * (1 + self.EXIT_SLIPPAGE + self.BID_ASK_SPREAD)

                elif not exit_triggered and position_direction == 'LONG':
                    # Legacy exit logic for LONG (fallback when tiered exits disabled)
                    if bar_high >= take_profit_price:
                        exit_triggered = True
                        exit_price = take_profit_price * (1 - self.EXIT_SLIPPAGE - self.BID_ASK_SPREAD)
                        exit_reason = 'take_profit'

                    elif bar_low <= stop_loss_price:
                        exit_triggered = True
                        exit_price = stop_loss_price * (1 - self.STOP_SLIPPAGE - self.BID_ASK_SPREAD)
                        exit_reason = 'stop_loss'

                    elif row['signal'] == -1 and current_price <= entry_price:
                        exit_triggered = True
                        exit_price = current_price * (1 - self.EXIT_SLIPPAGE - self.BID_ASK_SPREAD)
                        exit_reason = 'sell_signal'

                elif not exit_triggered and position_direction == 'SHORT':
                    # Legacy exit logic for SHORT (fallback when tiered exits disabled)
                    if bar_high >= stop_loss_price:
                        exit_triggered = True
                        exit_price = stop_loss_price * (1 + self.STOP_SLIPPAGE + self.BID_ASK_SPREAD)
                        exit_reason = 'stop_loss'

                    elif bar_low <= take_profit_price:
                        exit_triggered = True
                        exit_price = take_profit_price * (1 + self.EXIT_SLIPPAGE + self.BID_ASK_SPREAD)
                        exit_reason = 'take_profit'

                # EOD close simulation
                if not exit_triggered and self.eod_close_enabled:
                    if bar_hour is not None and bar_hour >= self.eod_close_bar_hour:
                        exit_triggered = True
                        if position_direction == 'LONG':
                            exit_price = current_price * (1 - self.EXIT_SLIPPAGE - self.BID_ASK_SPREAD)
                        else:
                            exit_price = current_price * (1 + self.EXIT_SLIPPAGE + self.BID_ASK_SPREAD)
                        exit_reason = 'eod_close'

                # Max hold time
                if not exit_triggered:
                    if entry_time is not None and bar_datetime is not None:
                        elapsed = bar_datetime - entry_time
                        elapsed_hours = elapsed.total_seconds() / 3600
                    else:
                        elapsed_hours = i - entry_index

                    if elapsed_hours >= self.max_hold_hours:
                        exit_triggered = True
                        if position_direction == 'LONG':
                            exit_price = current_price * (1 - self.EXIT_SLIPPAGE - self.BID_ASK_SPREAD)
                        else:
                            exit_price = current_price * (1 + self.EXIT_SLIPPAGE + self.BID_ASK_SPREAD)
                        exit_reason = 'max_hold'

                # Execute exit
                if exit_triggered:
                    self._diag_exit_reasons[symbol][exit_reason] += 1

                    if position_direction == 'LONG':
                        proceeds = shares * exit_price * (1 - self.COMMISSION)
                        self.cash += proceeds
                        entry_cost = shares * entry_price * (1 + self.COMMISSION)
                        pnl = proceeds - entry_cost
                    else:  # SHORT
                        cover_cost = shares * exit_price * (1 + self.COMMISSION)
                        self.cash -= cover_cost
                        pnl = (entry_price - exit_price) * shares - (self.COMMISSION * shares * (entry_price + exit_price))
                        entry_cost = shares * entry_price

                    last_trade_bar = i
                    pnl_pct = (pnl / entry_cost) * 100 if entry_cost > 0 else 0

                    self.total_pnl += pnl
                    self.total_trades += 1

                    if pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                        if self.entry_gate:
                            self.entry_gate.record_loss(timestamp)

                    # Track daily P&L for kill switch
                    self.daily_pnl += pnl

                    # Check kill switch after each trade
                    if self.daily_loss_kill_switch_enabled and self.daily_starting_capital > 0:
                        daily_loss_pct = -self.daily_pnl / self.daily_starting_capital
                        if daily_loss_pct >= self.max_daily_loss_pct:
                            if not self.kill_switch_triggered:
                                self.kill_switch_triggered = True
                                logger.info(f"KILL SWITCH TRIGGERED: Daily loss {daily_loss_pct*100:.2f}% >= {self.max_daily_loss_pct*100:.1f}%")
                                self._diag_entry_blocks[symbol]['kill_switch'] += 1

                    # MFE/MAE calculation
                    if position_direction == 'LONG':
                        mfe = highest_price - entry_price
                        mae = entry_price - lowest_price
                    else:
                        mfe = entry_price - lowest_price
                        mae = highest_price - entry_price
                    mfe_pct = (mfe / entry_price) * 100 if entry_price > 0 else 0
                    mae_pct = (mae / entry_price) * 100 if entry_price > 0 else 0

                    trades.append({
                        'symbol': symbol,
                        'direction': position_direction,
                        'entry_date': data.iloc[entry_index].get('timestamp', entry_index),
                        'exit_date': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'strategy': entry_strategy,
                        'reasoning': entry_reasoning,
                        'bars_held': i - entry_index,
                        'mfe': mfe,
                        'mae': mae,
                        'mfe_pct': mfe_pct,
                        'mae_pct': mae_pct
                    })

                    # Unregister from exit manager
                    if self.use_tiered_exits and self.exit_manager:
                        self.exit_manager.unregister_position(symbol)

                    # Reset trailing stop state
                    trailing_activated = False
                    trailing_stop_price = 0.0

                    position_direction = None

            # ============ UPDATE PORTFOLIO VALUE ============
            self.portfolio_value = self.cash
            if position_direction == 'LONG':
                self.portfolio_value += shares * current_price
            elif position_direction == 'SHORT':
                unrealized_pnl = (entry_price - current_price) * shares
                self.portfolio_value += unrealized_pnl

            # Record equity curve
            should_record = True
            if self._backtest_start_date is not None:
                bar_ts = pd.to_datetime(timestamp)
                if hasattr(bar_ts, 'tz') and bar_ts.tz is None:
                    bar_ts = bar_ts.tz_localize('UTC')
                should_record = bar_ts >= self._backtest_start_date

            if should_record:
                self.equity_curve.append({
                    'timestamp': timestamp,
                    'portfolio_value': self.portfolio_value,
                    'cash': self.cash,
                    'position_value': shares * current_price if position_direction else 0,
                    'direction': position_direction
                })

            # Track drawdown
            if self.portfolio_value > self.peak_value:
                self.peak_value = self.portfolio_value

            drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

            # ============ PROCESS PENDING ENTRY ============
            if pending_entry is not None and position_direction is None:
                open_price = row.get('open', current_price)
                direction = pending_entry.get('direction', 'LONG')

                if direction == 'LONG':
                    realistic_entry_price = open_price * (1 + self.ENTRY_SLIPPAGE + self.BID_ASK_SPREAD)
                    stop_loss_price = realistic_entry_price * (1 - self.default_stop_loss_pct)
                    take_profit_price = realistic_entry_price * (1 + self.default_take_profit_pct)
                else:
                    realistic_entry_price = open_price * (1 - self.ENTRY_SLIPPAGE - self.BID_ASK_SPREAD)
                    stop_loss_price = realistic_entry_price * (1 + self.default_stop_loss_pct)
                    take_profit_price = realistic_entry_price * (1 - self.default_take_profit_pct)

                # Calculate position size
                shares = self.risk_manager.calculate_position_size(
                    self.portfolio_value, realistic_entry_price, stop_loss_price
                )

                if direction == 'LONG':
                    cost = shares * realistic_entry_price * (1 + self.COMMISSION)
                    if cost <= self.cash and shares > 0:
                        self.cash -= cost
                        position_direction = 'LONG'
                else:
                    margin_required = shares * realistic_entry_price * 0.5
                    if margin_required <= self.cash and shares > 0:
                        self.cash += shares * realistic_entry_price * (1 - self.COMMISSION)
                        position_direction = 'SHORT'

                if position_direction is not None:
                    entry_price = realistic_entry_price
                    entry_index = i
                    entry_time = bar_datetime
                    highest_price = realistic_entry_price
                    lowest_price = realistic_entry_price
                    last_trade_bar = i
                    entry_strategy = pending_entry.get('strategy', 'Unknown')
                    entry_reasoning = pending_entry.get('reasoning', '')

                    # Reset trailing stop state
                    trailing_activated = False
                    trailing_stop_price = 0.0

                    # Register with exit manager (LONG and SHORT - Jan 2026)
                    if self.use_tiered_exits and self.exit_manager:
                        self.exit_manager.register_position(
                            symbol=symbol,
                            entry_price=entry_price,
                            quantity=shares,
                            entry_time=timestamp if isinstance(timestamp, datetime) else None,
                            direction=direction
                        )

                    # Record entry for gate
                    if self.entry_gate:
                        self.entry_gate.record_entry(symbol, timestamp)

                pending_entry = None

            # ============ CHECK FOR NEW ENTRY ============
            if position_direction is None and pending_entry is None:
                signal = row['signal']
                if signal in [1, -1]:
                    bars_since_last = i - last_trade_bar
                    if bars_since_last >= self.COOLDOWN_BARS:
                        # Kill switch check
                        if self.kill_switch_triggered:
                            self._diag_entry_blocks[symbol]['kill_switch'] += 1
                            if bar_date is not None:
                                self._diag_kill_switch_blocks_per_day[str(bar_date)] += 1
                            continue

                        # Check entry gate
                        entry_allowed = True
                        if self.entry_gate:
                            entry_allowed, reason = self.entry_gate.check_entry_allowed(symbol, timestamp)

                            if not entry_allowed:
                                if 'daily_loss_limit' in reason:
                                    self._diag_entry_blocks[symbol]['daily_loss_guard'] += 1
                                elif 'max_trades_per_day' in reason:
                                    self._diag_entry_blocks[symbol]['entry_gate'] += 1
                                elif 'time_filter' in reason:
                                    self._diag_entry_blocks[symbol]['time_filter'] += 1
                                else:
                                    self._diag_entry_blocks[symbol]['other'] += 1

                        if entry_allowed:
                            if signal == 1:  # BUY -> LONG
                                if self.shorts_only:
                                    continue

                                pending_entry = {
                                    'direction': 'LONG',
                                    'signal_price': current_price,
                                    'strategy': row.get('strategy', 'Unknown'),
                                    'reasoning': row.get('reasoning', '')
                                }
                            elif signal == -1:  # SELL -> SHORT
                                if self.longs_only:
                                    continue

                                pending_entry = {
                                    'direction': 'SHORT',
                                    'signal_price': current_price,
                                    'strategy': row.get('strategy', 'Unknown') + '_SHORT',
                                    'reasoning': row.get('reasoning', '')
                                }

        # ============ CLOSE REMAINING POSITION ============
        if position_direction is not None:
            self._diag_exit_reasons[symbol]['end_of_backtest'] += 1

            close_price = data.iloc[-1]['close']
            if position_direction == 'LONG':
                final_price = close_price * (1 - self.EXIT_SLIPPAGE - self.BID_ASK_SPREAD)
                proceeds = shares * final_price * (1 - self.COMMISSION)
                self.cash += proceeds
                entry_cost = shares * entry_price * (1 + self.COMMISSION)
                pnl = proceeds - entry_cost
            else:
                final_price = close_price * (1 + self.EXIT_SLIPPAGE + self.BID_ASK_SPREAD)
                cover_cost = shares * final_price * (1 + self.COMMISSION)
                self.cash -= cover_cost
                pnl = (entry_price - final_price) * shares - (self.COMMISSION * shares * (entry_price + final_price))
                entry_cost = shares * entry_price

            pnl_pct = (pnl / entry_cost) * 100 if entry_cost > 0 else 0

            self.total_pnl += pnl
            self.total_trades += 1

            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            # MFE/MAE calculation
            if position_direction == 'LONG':
                mfe = highest_price - entry_price
                mae = entry_price - lowest_price
            else:
                mfe = entry_price - lowest_price
                mae = highest_price - entry_price
            mfe_pct = (mfe / entry_price) * 100 if entry_price > 0 else 0
            mae_pct = (mae / entry_price) * 100 if entry_price > 0 else 0

            trades.append({
                'symbol': symbol,
                'direction': position_direction,
                'entry_date': data.iloc[entry_index].get('timestamp', entry_index),
                'exit_date': data.iloc[-1].get('timestamp', len(data) - 1),
                'entry_price': entry_price,
                'exit_price': final_price,
                'shares': shares,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': 'end_of_backtest',
                'strategy': entry_strategy,
                'reasoning': entry_reasoning,
                'bars_held': len(data) - 1 - entry_index,
                'mfe': mfe,
                'mae': mae,
                'mfe_pct': mfe_pct,
                'mae_pct': mae_pct
            })

            if self.use_tiered_exits and self.exit_manager:
                self.exit_manager.unregister_position(symbol)

        return trades

    def simulate_trades_interleaved(self, signals_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Simulate trading across multiple symbols with position limit enforcement.

        Unlike simulate_trades() which runs one symbol at a time, this method
        processes all symbols bar-by-bar in time order, enforcing max_open_positions
        across all symbols.

        Args:
            signals_data: Dict mapping symbol -> DataFrame with OHLCV, signals

        Returns:
            List of trade dicts from all symbols
        """
        self._reset_state()

        all_trades = []
        open_positions = {}  # symbol -> position dict

        # Risk tracking for analytics
        self._risk_analytics = {
            'max_positions_open': 0,
            'max_risk_exposure_pct': 0.0,
            'max_risk_exposure_dollars': 0.0,
            'times_at_position_limit': 0,
            'entries_blocked_by_limit': 0,
            'position_counts': [],  # (timestamp, count, exposure_pct)
            'daily_max_exposure': {},  # date -> max exposure that day
        }

        # Build unified timeline from all symbols
        all_events = []
        for symbol, df in signals_data.items():
            for idx in range(len(df)):
                row = df.iloc[idx]
                ts = row.get('timestamp', idx)
                all_events.append({
                    'timestamp': ts,
                    'symbol': symbol,
                    'bar_index': idx,
                    'row': row,
                    'df': df
                })

        # Sort by timestamp
        all_events.sort(key=lambda x: (x['timestamp'], x['symbol']))

        # Track position state per symbol
        position_state = {}  # symbol -> {'direction': 'LONG'/'SHORT', 'entry_price': float, ...}
        last_trade_bar = {}  # symbol -> bar index of last trade
        last_recorded_ts = None  # For equity curve recording
        latest_prices = {}  # symbol -> latest price for portfolio valuation

        # Daily tracking for drawdown guard (ODE-118)
        current_day = None
        partial_liquidation_done_today = False

        for event in all_events:
            symbol = event['symbol']
            row = event['row']
            df = event['df']
            bar_index = event['bar_index']
            timestamp = event['timestamp']

            current_price = row['close']
            signal = row.get('signal', 0)

            # Skip if price is NaN
            if pd.isna(current_price):
                continue

            # Initialize state for this symbol if needed
            if symbol not in position_state:
                position_state[symbol] = None
            if symbol not in last_trade_bar:
                last_trade_bar[symbol] = -999

            # ============ DRAWDOWN GUARD (ODE-118) ============
            # Extract date from timestamp
            bar_date = None
            if isinstance(timestamp, (datetime, pd.Timestamp)):
                bar_date = timestamp.date() if hasattr(timestamp, 'date') else None
            elif isinstance(timestamp, str):
                try:
                    bar_date = pd.to_datetime(timestamp).date()
                except:
                    pass

            # Reset guard on new day
            if bar_date is not None and bar_date != current_day:
                current_day = bar_date
                partial_liquidation_done_today = False
                # Calculate portfolio value for reset
                position_value = 0
                for sym, pos in open_positions.items():
                    if pos is not None:
                        pos_price = latest_prices.get(sym, pos['entry_price'])
                        if pos['direction'] == 'LONG':
                            unrealized_pnl = (pos_price - pos['entry_price']) * pos['shares']
                        else:
                            unrealized_pnl = (pos['entry_price'] - pos_price) * pos['shares']
                        position_value += unrealized_pnl
                day_start_equity = self.cash + position_value
                self.drawdown_guard.reset_day(day_start_equity, bar_date)

            # Update drawdown guard with current equity
            # Calculate current portfolio value
            position_value = 0
            for sym, pos in open_positions.items():
                if pos is not None:
                    pos_price = latest_prices.get(sym, pos['entry_price'])
                    if pos['direction'] == 'LONG':
                        unrealized_pnl = (pos_price - pos['entry_price']) * pos['shares']
                    else:
                        unrealized_pnl = (pos['entry_price'] - pos_price) * pos['shares']
                    position_value += unrealized_pnl
            portfolio_value = self.cash + position_value
            self.portfolio_value = portfolio_value

            # Create a simple object for guard.update_equity
            class EquityHolder:
                def __init__(self, equity):
                    self.equity = equity
            self.drawdown_guard.update_equity(EquityHolder(portfolio_value), current_date=bar_date)

            # ============ LIQUIDATION CHECKS (ODE-118) ============
            guard_tier = self.drawdown_guard.tier

            # HARD_LIMIT: Full liquidation
            if guard_tier == DrawdownTier.HARD_LIMIT and open_positions:
                logger.warning(f"DRAWDOWN_GUARD | HARD_LIMIT | Liquidating all {len(open_positions)} positions")
                for liq_symbol, pos in list(open_positions.items()):
                    if pos is None:
                        continue
                    direction = pos['direction']
                    entry_price = pos['entry_price']
                    shares = pos['shares']
                    liq_price = latest_prices.get(liq_symbol, entry_price)

                    if direction == 'LONG':
                        exit_price = liq_price * (1 - self.EXIT_SLIPPAGE)
                        pnl = (exit_price - entry_price) * shares
                    else:
                        exit_price = liq_price * (1 + self.EXIT_SLIPPAGE)
                        pnl = (entry_price - exit_price) * shares

                    pnl_pct = (pnl / (entry_price * shares)) * 100 if entry_price > 0 else 0

                    all_trades.append({
                        'symbol': liq_symbol,
                        'direction': direction,
                        'entry_date': pos['entry_time'],
                        'exit_date': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'hard_limit_liquidation',
                        'strategy': pos.get('strategy', 'Unknown'),
                        'reasoning': pos.get('reasoning', ''),
                        'bars_held': bar_index - pos['entry_bar'],
                        'mfe': 0, 'mae': 0, 'mfe_pct': 0, 'mae_pct': 0
                    })

                    self.cash += pnl
                    del open_positions[liq_symbol]
                    position_state[liq_symbol] = None

                # Skip to next event after full liquidation
                continue

            # MEDIUM: Partial liquidation (50% of each position)
            if guard_tier == DrawdownTier.MEDIUM and not partial_liquidation_done_today and open_positions:
                partial_liquidation_done_today = True
                logger.warning(f"DRAWDOWN_GUARD | MEDIUM | Partial liquidation: closing 50% of {len(open_positions)} positions")
                for liq_symbol, pos in list(open_positions.items()):
                    if pos is None:
                        continue
                    direction = pos['direction']
                    entry_price = pos['entry_price']
                    total_shares = pos['shares']
                    shares_to_close = total_shares // 2  # Close half (rounded down)

                    if shares_to_close <= 0:
                        continue

                    liq_price = latest_prices.get(liq_symbol, entry_price)

                    if direction == 'LONG':
                        exit_price = liq_price * (1 - self.EXIT_SLIPPAGE)
                        pnl = (exit_price - entry_price) * shares_to_close
                    else:
                        exit_price = liq_price * (1 + self.EXIT_SLIPPAGE)
                        pnl = (entry_price - exit_price) * shares_to_close

                    pnl_pct = (pnl / (entry_price * shares_to_close)) * 100 if entry_price > 0 else 0

                    all_trades.append({
                        'symbol': liq_symbol,
                        'direction': direction,
                        'entry_date': pos['entry_time'],
                        'exit_date': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': shares_to_close,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'partial_liquidation',
                        'strategy': pos.get('strategy', 'Unknown'),
                        'reasoning': pos.get('reasoning', ''),
                        'bars_held': bar_index - pos['entry_bar'],
                        'mfe': 0, 'mae': 0, 'mfe_pct': 0, 'mae_pct': 0
                    })

                    self.cash += pnl
                    # Reduce position size
                    pos['shares'] = total_shares - shares_to_close
                    if pos['shares'] <= 0:
                        del open_positions[liq_symbol]
                        position_state[liq_symbol] = None

            # ============ EXIT LOGIC ============
            if position_state[symbol] is not None:
                pos = position_state[symbol]
                direction = pos['direction']
                entry_price = pos['entry_price']
                entry_bar = pos['entry_bar']
                shares = pos['shares']

                bar_high = row.get('high', current_price)
                bar_low = row.get('low', current_price)

                exit_triggered = False
                exit_reason = ''
                exit_price = current_price

                # Check stop loss and take profit FIRST to determine effective high/low
                stop_loss = pos.get('stop_loss', 0)
                take_profit = pos.get('take_profit', 0)

                # Determine effective high/low for MFE/MAE (capped by exit triggers)
                effective_high = bar_high
                effective_low = bar_low

                if direction == 'LONG':
                    if bar_low <= stop_loss:
                        exit_triggered = True
                        exit_price = stop_loss * (1 - self.STOP_SLIPPAGE)
                        exit_reason = 'stop_loss'
                        effective_low = stop_loss  # Cap MAE at stop
                        effective_high = min(bar_high, entry_price * 1.001)  # Didn't run up much if stopped
                    elif bar_high >= take_profit:
                        exit_triggered = True
                        exit_price = take_profit * (1 - self.EXIT_SLIPPAGE)
                        exit_reason = 'take_profit'
                        effective_high = take_profit  # Cap MFE at take profit
                else:  # SHORT
                    if bar_high >= stop_loss:
                        exit_triggered = True
                        exit_price = stop_loss * (1 + self.STOP_SLIPPAGE)
                        exit_reason = 'stop_loss'
                        effective_high = stop_loss  # Cap MAE at stop
                        effective_low = max(bar_low, entry_price * 0.999)  # Didn't run down much if stopped
                    elif bar_low <= take_profit:
                        exit_triggered = True
                        exit_price = take_profit * (1 + self.EXIT_SLIPPAGE)
                        exit_reason = 'take_profit'
                        effective_low = take_profit  # Cap MFE at take profit

                # Update price tracking with effective values
                if effective_high > pos.get('highest_price', entry_price):
                    pos['highest_price'] = effective_high
                if effective_low < pos.get('lowest_price', entry_price):
                    pos['lowest_price'] = effective_low

                # Check max hold
                bars_held = bar_index - entry_bar
                if not exit_triggered and bars_held >= self.max_hold_hours:
                    exit_triggered = True
                    exit_reason = 'max_hold'
                    if direction == 'LONG':
                        exit_price = current_price * (1 - self.EXIT_SLIPPAGE)
                    else:
                        exit_price = current_price * (1 + self.EXIT_SLIPPAGE)

                # EOD close - no overnight holding
                if not exit_triggered and self.eod_close_enabled:
                    bar_hour = None
                    if isinstance(timestamp, (datetime, pd.Timestamp)):
                        bar_hour = timestamp.hour
                    elif isinstance(timestamp, str):
                        try:
                            bar_hour = pd.to_datetime(timestamp).hour
                        except:
                            pass
                    if bar_hour is not None and bar_hour >= self.eod_close_bar_hour:
                        exit_triggered = True
                        exit_reason = 'eod_close'
                        if direction == 'LONG':
                            exit_price = current_price * (1 - self.EXIT_SLIPPAGE)
                        else:
                            exit_price = current_price * (1 + self.EXIT_SLIPPAGE)

                if exit_triggered:
                    # Calculate P&L
                    if direction == 'LONG':
                        pnl = (exit_price - entry_price) * shares
                    else:
                        pnl = (entry_price - exit_price) * shares

                    pnl_pct = (pnl / (entry_price * shares)) * 100 if entry_price > 0 else 0

                    # Calculate MFE/MAE
                    if direction == 'LONG':
                        mfe = pos.get('highest_price', entry_price) - entry_price
                        mae = entry_price - pos.get('lowest_price', entry_price)
                    else:
                        mfe = entry_price - pos.get('lowest_price', entry_price)
                        mae = pos.get('highest_price', entry_price) - entry_price
                    mfe_pct = (mfe / entry_price) * 100 if entry_price > 0 else 0
                    mae_pct = (mae / entry_price) * 100 if entry_price > 0 else 0

                    all_trades.append({
                        'symbol': symbol,
                        'direction': direction,
                        'entry_date': pos['entry_time'],
                        'exit_date': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'strategy': pos.get('strategy', 'Unknown'),
                        'reasoning': pos.get('reasoning', ''),
                        'bars_held': bars_held,
                        'mfe': mfe,
                        'mae': mae,
                        'mfe_pct': mfe_pct,
                        'mae_pct': mae_pct
                    })

                    # Update realized P&L tracking (for equity curve)
                    self.cash += pnl  # Add realized P&L to cash

                    # Remove from open positions
                    del open_positions[symbol]
                    position_state[symbol] = None
                    last_trade_bar[symbol] = bar_index

            # ============ ENTRY LOGIC ============
            if position_state[symbol] is None and signal != 0:
                # Check cooldown
                bars_since_last = bar_index - last_trade_bar[symbol]
                if bars_since_last < self.COOLDOWN_BARS:
                    continue

                # Check drawdown guard entries_allowed (ODE-118)
                if not self.drawdown_guard.entries_allowed:
                    self._diag_entry_blocks[symbol]['drawdown_guard'] += 1
                    if self.kill_switch_trace:
                        self._kill_switch_trace_log.append({
                            'event': 'ENTRY_BLOCKED',
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'signal': signal,
                            'block_reason': f'drawdown_guard tier={self.drawdown_guard.tier.name}',
                            'drawdown_pct': self.drawdown_guard.drawdown_pct * 100
                        })
                    continue

                # Block entries at EOD - no new positions that would be immediately closed
                if self.eod_close_enabled:
                    entry_bar_hour = None
                    if isinstance(timestamp, (datetime, pd.Timestamp)):
                        entry_bar_hour = timestamp.hour
                    elif isinstance(timestamp, str):
                        try:
                            entry_bar_hour = pd.to_datetime(timestamp).hour
                        except:
                            pass
                    if entry_bar_hour is not None and entry_bar_hour >= self.eod_close_bar_hour:
                        continue  # Skip entry at EOD

                # Check position limit
                if len(open_positions) >= self.max_open_positions:
                    # Track blocked entries
                    self._risk_analytics['entries_blocked_by_limit'] += 1
                    self._risk_analytics['times_at_position_limit'] += 1
                    # Log blocked entry
                    if self.kill_switch_trace:
                        self._kill_switch_trace_log.append({
                            'event': 'ENTRY_BLOCKED',
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'signal': signal,
                            'block_reason': f'max_positions reached ({len(open_positions)}/{self.max_open_positions})',
                            'open_positions': list(open_positions.keys())
                        })
                    continue

                # Determine direction
                if signal == 1 and not self.shorts_only:
                    direction = 'LONG'
                elif signal == -1 and not self.longs_only:
                    direction = 'SHORT'
                else:
                    continue

                # Calculate entry price with slippage
                open_price = row.get('open', current_price)
                if direction == 'LONG':
                    entry_price = open_price * (1 + self.ENTRY_SLIPPAGE)
                    stop_loss = entry_price * (1 - self.default_stop_loss_pct)
                    take_profit = entry_price * (1 + self.default_take_profit_pct)
                else:
                    entry_price = open_price * (1 - self.ENTRY_SLIPPAGE)
                    stop_loss = entry_price * (1 + self.default_stop_loss_pct)
                    take_profit = entry_price * (1 - self.default_take_profit_pct)

                # Calculate position size
                shares = self.risk_manager.calculate_position_size(
                    self.portfolio_value, entry_price, stop_loss
                )

                # Apply drawdown guard position size multiplier (ODE-118)
                # WARNING tier reduces size to 50%
                size_multiplier = self.drawdown_guard.position_size_multiplier
                if size_multiplier < 1.0:
                    shares = int(shares * size_multiplier)

                if shares <= 0:
                    continue

                # Track position cost for portfolio calculation
                # Note: We don't modify self.cash here - portfolio value is calculated
                # as initial_capital + realized_pnl + unrealized_pnl

                # Open position
                position_state[symbol] = {
                    'direction': direction,
                    'entry_price': entry_price,
                    'entry_bar': bar_index,
                    'entry_time': timestamp,
                    'shares': shares,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'highest_price': entry_price,
                    'lowest_price': entry_price,
                    'strategy': row.get('strategy', 'Unknown'),
                    'reasoning': row.get('reasoning', '')
                }
                open_positions[symbol] = position_state[symbol]
                last_trade_bar[symbol] = bar_index

                # Track risk analytics after entry
                num_positions = len(open_positions)
                if num_positions > self._risk_analytics['max_positions_open']:
                    self._risk_analytics['max_positions_open'] = num_positions

                # Calculate current exposure
                total_exposure = sum(
                    p['shares'] * p['entry_price'] for p in open_positions.values()
                )
                exposure_pct = (total_exposure / self.initial_capital) * 100

                if exposure_pct > self._risk_analytics['max_risk_exposure_pct']:
                    self._risk_analytics['max_risk_exposure_pct'] = exposure_pct
                    self._risk_analytics['max_risk_exposure_dollars'] = total_exposure

                # Track daily max exposure
                if hasattr(timestamp, 'date'):
                    date_key = str(timestamp.date())
                elif hasattr(timestamp, 'strftime'):
                    date_key = timestamp.strftime('%Y-%m-%d')
                else:
                    date_key = str(timestamp)[:10]

                if date_key not in self._risk_analytics['daily_max_exposure']:
                    self._risk_analytics['daily_max_exposure'][date_key] = exposure_pct
                elif exposure_pct > self._risk_analytics['daily_max_exposure'][date_key]:
                    self._risk_analytics['daily_max_exposure'][date_key] = exposure_pct

            # Track latest price for this symbol
            latest_prices[symbol] = current_price

            # Record equity curve when timestamp changes (to avoid too many entries)
            # Only record if we're past the backtest start date
            should_record = True
            if self._backtest_start_date is not None:
                bar_ts = pd.to_datetime(timestamp)
                # Convert both to UTC for proper comparison
                if bar_ts.tz is None:
                    bar_ts = bar_ts.tz_localize('UTC')
                else:
                    bar_ts = bar_ts.tz_convert('UTC')
                should_record = bar_ts >= self._backtest_start_date

            if should_record and last_recorded_ts != timestamp:
                # Calculate current portfolio value
                position_value = 0
                for sym, pos in open_positions.items():
                    if pos is not None:
                        pos_price = latest_prices.get(sym, pos['entry_price'])
                        if pos['direction'] == 'LONG':
                            # Unrealized P&L for long
                            unrealized_pnl = (pos_price - pos['entry_price']) * pos['shares']
                            position_value += unrealized_pnl
                        else:  # SHORT
                            # Unrealized P&L for short
                            unrealized_pnl = (pos['entry_price'] - pos_price) * pos['shares']
                            position_value += unrealized_pnl

                portfolio_value = self.cash + position_value

                # Update peak and drawdown tracking
                if portfolio_value > self.peak_value:
                    self.peak_value = portfolio_value
                if self.peak_value > 0:
                    drawdown = (self.peak_value - portfolio_value) / self.peak_value
                    if drawdown > self.max_drawdown:
                        self.max_drawdown = drawdown

                self.equity_curve.append({
                    'timestamp': timestamp,
                    'portfolio_value': portfolio_value,
                    'cash': self.cash,
                    'position_value': position_value,
                    'num_positions': len(open_positions)
                })
                last_recorded_ts = timestamp

        # Close any remaining positions at end of backtest
        for symbol, pos in list(open_positions.items()):
            if pos is None:
                continue

            direction = pos['direction']
            entry_price = pos['entry_price']
            shares = pos['shares']

            # Get last price for this symbol
            df = signals_data[symbol]
            last_row = df.iloc[-1]
            close_price = last_row['close']

            if direction == 'LONG':
                exit_price = close_price * (1 - self.EXIT_SLIPPAGE)
                pnl = (exit_price - entry_price) * shares
            else:
                exit_price = close_price * (1 + self.EXIT_SLIPPAGE)
                pnl = (entry_price - exit_price) * shares

            pnl_pct = (pnl / (entry_price * shares)) * 100 if entry_price > 0 else 0

            # Calculate MFE/MAE for end_of_backtest positions
            if direction == 'LONG':
                mfe = pos.get('highest_price', entry_price) - entry_price
                mae = entry_price - pos.get('lowest_price', entry_price)
            else:
                mfe = entry_price - pos.get('lowest_price', entry_price)
                mae = pos.get('highest_price', entry_price) - entry_price
            mfe_pct = (mfe / entry_price) * 100 if entry_price > 0 else 0
            mae_pct = (mae / entry_price) * 100 if entry_price > 0 else 0

            all_trades.append({
                'symbol': symbol,
                'direction': direction,
                'entry_date': pos['entry_time'],
                'exit_date': last_row.get('timestamp', len(df) - 1),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': 'end_of_backtest',
                'strategy': pos.get('strategy', 'Unknown'),
                'reasoning': pos.get('reasoning', ''),
                'bars_held': len(df) - 1 - pos['entry_bar'],
                'mfe': mfe,
                'mae': mae,
                'mfe_pct': mfe_pct,
                'mae_pct': mae_pct
            })

            # Update realized P&L on end-of-backtest exit
            self.cash += pnl

        return all_trades

    def calculate_metrics(self, trades: List[Dict]) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Args:
            trades: List of trade dicts

        Returns:
            Dict with performance metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }

        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]

        total_wins = sum(t['pnl'] for t in wins) if wins else 0
        total_losses = abs(sum(t['pnl'] for t in losses)) if losses else 0

        if total_losses > 0:
            profit_factor = total_wins / total_losses
        else:
            profit_factor = 999.99 if total_wins > 0 else 0

        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_pnl = np.mean([t['pnl'] for t in trades]) if trades else 0
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0

        # Sharpe ratio
        sharpe_ratio = 0
        if len(self.equity_curve) > 1:
            equity_df = pd.DataFrame(self.equity_curve)
            returns = equity_df['portfolio_value'].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                daily_rf = 0.05 / 252
                excess_returns = returns - daily_rf
                sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(252)

        # Calculate worst daily drops from actual trade P&L (not equity curve)
        # This avoids the bug where equity curve mixes per-symbol data
        worst_daily_drops = []
        if trades:
            from collections import defaultdict
            daily_pnl = defaultdict(float)
            for t in trades:
                exit_date = t.get('exit_date', '')
                if hasattr(exit_date, 'strftime'):
                    date_str = exit_date.strftime('%Y-%m-%d')
                elif hasattr(exit_date, 'isoformat'):
                    date_str = str(exit_date)[:10]
                else:
                    date_str = str(exit_date)[:10]
                daily_pnl[date_str] += t.get('pnl', 0)

            # Convert to list and calculate % of initial capital
            daily_list = []
            for date_str, pnl in daily_pnl.items():
                pct = (pnl / self.initial_capital) * 100
                daily_list.append({
                    'date': date_str,
                    'open': 0,  # Not tracked per-day
                    'close': 0,
                    'high': 0,
                    'low': 0,
                    'change_pct': pct,
                    'change_dollars': pnl
                })

            # Sort by change_pct ascending (worst first) and take top 5
            daily_list.sort(key=lambda x: x['change_pct'])
            worst_daily_drops = daily_list[:5]

        # Calculate drawdown peak/trough - find the peak before max drawdown and the trough during it
        drawdown_peak_date = None
        drawdown_peak_value = self.initial_capital
        drawdown_trough_date = None
        drawdown_trough_value = self.initial_capital
        if self.equity_curve:
            # Track running peak and find max drawdown point
            running_peak = 0
            running_peak_date = None
            max_dd = 0
            max_dd_peak = self.initial_capital
            max_dd_peak_date = None
            max_dd_trough = self.initial_capital
            max_dd_trough_date = None

            for entry in self.equity_curve:
                val = entry.get('portfolio_value', 0)
                ts = entry.get('timestamp', '')

                # Update running peak
                if val > running_peak:
                    running_peak = val
                    running_peak_date = str(ts)

                # Calculate current drawdown from running peak
                if running_peak > 0:
                    current_dd = (running_peak - val) / running_peak
                    if current_dd > max_dd:
                        max_dd = current_dd
                        max_dd_peak = running_peak
                        max_dd_peak_date = running_peak_date
                        max_dd_trough = val
                        max_dd_trough_date = str(ts)

            drawdown_peak_value = max_dd_peak
            drawdown_peak_date = max_dd_peak_date
            drawdown_trough_value = max_dd_trough
            drawdown_trough_date = max_dd_trough_date
            # Update self.max_drawdown with correctly calculated value from equity curve
            self.max_drawdown = max_dd

        # Get risk analytics if available
        risk_analytics = getattr(self, '_risk_analytics', {})

        return {
            'initial_capital': self.initial_capital,
            'final_value': self.cash,
            'total_return_pct': (self.cash - self.initial_capital) / self.initial_capital * 100,
            'total_pnl': self.total_pnl,
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': self.max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'best_trade': max(t['pnl'] for t in trades) if trades else 0,
            'worst_trade': min(t['pnl'] for t in trades) if trades else 0,
            'avg_bars_held': np.mean([t['bars_held'] for t in trades]) if trades else 0,
            'worst_daily_drops': worst_daily_drops,
            'drawdown_peak_date': drawdown_peak_date,
            'drawdown_peak_value': drawdown_peak_value,
            'drawdown_trough_date': drawdown_trough_date,
            'drawdown_trough_value': drawdown_trough_value,
            # Risk Analytics
            'max_positions_open': risk_analytics.get('max_positions_open', 0),
            'max_risk_exposure_pct': risk_analytics.get('max_risk_exposure_pct', 0),
            'max_risk_exposure_dollars': risk_analytics.get('max_risk_exposure_dollars', 0),
            'entries_blocked_by_limit': risk_analytics.get('entries_blocked_by_limit', 0),
            'daily_max_exposure': risk_analytics.get('daily_max_exposure', {}),
        }

    def run(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """
        Run full backtest on multiple symbols.

        Args:
            symbols: List of symbols to backtest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Complete backtest results
        """
        self._reset_state()

        logger.info(f"Running Backtest")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Scanner enabled: {self.scanner_enabled}")

        # Set backtest start date for equity curve filtering
        self._backtest_start_date = pd.to_datetime(start_date).tz_localize('UTC')

        # Record initial equity curve entry
        pre_start = self._backtest_start_date - pd.Timedelta(days=1)
        self.equity_curve.append({
            'timestamp': pre_start,
            'portfolio_value': self.initial_capital,
            'cash': self.initial_capital,
            'position_value': 0,
            'direction': None
        })

        all_trades = []

        # FIX (Jan 7, 2026): Use parallel data fetching for 5-10x speedup
        # Old: sequential fetch = 100 symbols x 0.5s = 50+ seconds
        # New: parallel fetch = 100 symbols / 10 workers = ~5-10 seconds
        logger.info("Fetching data for all symbols in parallel...")
        all_data = self.fetch_data_parallel(symbols, start_date, end_date, max_workers=10)

        if self.scanner_enabled:
            logger.info(f"Scanner mode: building daily scan results from {len(all_data)} symbols...")
            self._daily_scanned_symbols = self._build_daily_scan_results(
                all_data, start_date, end_date
            )

        # FIX (Jan 7, 2026): Use parallel signal generation for 3-5x speedup
        # Signal generation is CPU-bound and independent per symbol
        logger.info("Generating signals for all symbols in parallel...")
        signals_data = self.generate_signals_parallel(all_data, max_workers=8)

        # Simulate trades with position limit enforcement across all symbols
        if signals_data:
            logger.info(f"Running interleaved simulation with max_open_positions={self.max_open_positions}")
            all_trades = self.simulate_trades_interleaved(signals_data)

            # Log per-symbol summary
            trades_by_symbol = {}
            for t in all_trades:
                sym = t['symbol']
                if sym not in trades_by_symbol:
                    trades_by_symbol[sym] = []
                trades_by_symbol[sym].append(t)

            for sym, sym_trades in trades_by_symbol.items():
                logger.info(f"{sym}: {len(sym_trades)} trades, P&L: ${sum(t['pnl'] for t in sym_trades):,.2f}")

            # Update instance state from trades (simulate_trades_interleaved doesn't update these)
            self.total_pnl = sum(t['pnl'] for t in all_trades)
            self.cash = self.initial_capital + self.total_pnl
            self.portfolio_value = self.cash

            # Note: max_drawdown is already calculated correctly from equity curve
            # in simulate_trades_interleaved() using actual portfolio values.
            # Do NOT overwrite it here with trade-based P&L calculation.

        # Calculate metrics
        metrics = self.calculate_metrics(all_trades)

        results = {
            'timeframe': '1Hour',
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital,
            'final_capital': self.cash,
            'trades': all_trades,
            'total_trades': len(all_trades),
            'metrics': metrics,
            'equity_curve': pd.DataFrame(self.equity_curve) if self.equity_curve else pd.DataFrame(),
            'max_drawdown': self.max_drawdown,
        }

        logger.info(f"Backtest complete: {len(all_trades)} trades, Total P&L: ${self.total_pnl:,.2f}")

        return results


def run_backtest(
    symbols: list = None,
    start_date: str = None,
    end_date: str = None,
    initial_capital: float = 100000.0,
    longs_only: bool = False,
    shorts_only: bool = False,
) -> Dict:
    """
    Run backtest with configuration from config.yaml.

    Args:
        symbols: List of symbols to backtest (uses universe.yaml if not provided)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting capital
        longs_only: Only take LONG positions
        shorts_only: Only take SHORT positions

    Returns:
        dict: Backtest results including P&L, trades, metrics
    """
    bot_dir = Path(__file__).parent

    # Load universe if no symbols provided
    if symbols is None:
        universe_path = bot_dir / 'universe.yaml'
        if universe_path.exists():
            with open(universe_path, 'r') as f:
                universe = yaml.safe_load(f)

            # Use scanner_universe (400 symbols) for full scanner benefit
            scanner_universe = universe.get('scanner_universe', {})
            symbols = []
            for category, syms in scanner_universe.items():
                if isinstance(syms, list):
                    for s in syms:
                        if s not in symbols:
                            symbols.append(s)

            # Fallback to proven_symbols if scanner_universe empty
            if not symbols:
                symbols = universe.get('proven_symbols', [])
            if not symbols:
                symbols = universe.get('candidates', ['SPY', 'AAPL', 'MSFT'])
        else:
            symbols = ['SPY', 'AAPL', 'MSFT']

    if not symbols:
        logger.error("No symbols to backtest!")
        return None

    # Set date range
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

    # Load config and run backtest
    config_path = bot_dir / 'config.yaml'
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    backtester = Backtest1Hour(
        initial_capital=initial_capital,
        config=config,
        longs_only=longs_only,
        shorts_only=shorts_only,
    )

    return backtester.run(symbols, start_date, end_date)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Run backtest')
    parser.add_argument('--symbols', nargs='+', help='Symbols to backtest')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--longs-only', action='store_true', help='Only take LONG positions')
    parser.add_argument('--shorts-only', action='store_true', help='Only take SHORT positions')

    args = parser.parse_args()

    results = run_backtest(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        longs_only=args.longs_only,
        shorts_only=args.shorts_only,
    )

    if results:
        metrics = results['metrics']

        print(f"\n{'='*60}")
        print(f"  BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"  Period: {results['start_date']} to {results['end_date']}")
        print(f"  Symbols: {', '.join(results['symbols'])}")
        print(f"{'='*60}")
        print(f"  Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"  Final Capital: ${results['final_capital']:,.2f}")
        print(f"  Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"  Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
        print(f"{'='*60}")
        print(f"  Total Trades: {metrics.get('total_trades', 0)}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.1f}%")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"{'='*60}")
        print(f"  RISK ANALYTICS (Funded Account Check)")
        print(f"{'='*60}")
        print(f"  Max Positions Open: {metrics.get('max_positions_open', 0)}")
        print(f"  Max Risk Exposure: {metrics.get('max_risk_exposure_pct', 0):.1f}% (${metrics.get('max_risk_exposure_dollars', 0):,.2f})")
        print(f"  Entries Blocked by Limit: {metrics.get('entries_blocked_by_limit', 0)}")

        # Show worst daily exposure days
        daily_exposure = metrics.get('daily_max_exposure', {})
        if daily_exposure:
            sorted_days = sorted(daily_exposure.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top 5 Highest Exposure Days:")
            for date, exp in sorted_days:
                print(f"    {date}: {exp:.1f}%")

        # Funded account warning
        max_exp = metrics.get('max_risk_exposure_pct', 0)
        if max_exp > 50:
            print(f"  ⚠️  WARNING: Max exposure {max_exp:.1f}% exceeds 50% - risky for funded account")
        elif max_exp > 30:
            print(f"  ⚠️  CAUTION: Max exposure {max_exp:.1f}% is moderate")
        else:
            print(f"  ✓ Exposure within safe limits for funded account")

        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
