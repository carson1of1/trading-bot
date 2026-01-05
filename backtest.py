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
    TradeLogger,
    VolatilityScanner,
    YFinanceDataFetcher,
)
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
        initial_capital: float = 10000.0,
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

        self.risk_manager = RiskManager(risk_config)
        self.entry_gate = EntryGate(entry_config)

        # Build exit manager settings
        exit_settings = {
            'hard_stop_pct': abs(exit_config.get('tier_0_hard_stop', -0.02)) * 100,
            'profit_floor_pct': exit_config.get('tier_1_profit_floor', 0.02) * 100,
            'trailing_activation_pct': exit_config.get('tier_2_atr_trailing', 0.03) * 100,
            'partial_tp_pct': exit_config.get('tier_3_partial_take', 0.04) * 100,
        }
        bot_settings = {'risk': exit_settings}
        self.exit_manager = ExitManager(bot_settings)
        self.use_tiered_exits = exit_config.get('enabled', True)

        # Strategy manager
        self.strategy_manager = StrategyManager(self.config)

        # Default risk settings (defaults match config.yaml expectations)
        self.default_stop_loss_pct = risk_config.get('stop_loss_pct', 5.0) / 100
        self.default_take_profit_pct = risk_config.get('take_profit_pct', 8.0) / 100

        # Max hold hours from config
        self.max_hold_hours = exit_config.get('max_hold_hours', self.DEFAULT_MAX_HOLD_BARS)

        # BUG FIX (Jan 4, 2026): Enforce max concurrent positions
        # Previously no limit was enforced, causing 62%+ daily losses from over-allocation
        self.max_open_positions = risk_config.get('max_open_positions', 5)

        # Daily loss kill switch
        self.max_daily_loss_pct = risk_config.get('max_daily_loss_pct', 3.0) / 100
        self.daily_loss_kill_switch_enabled = True

        # Emergency stop - force close ALL positions if unrealized loss exceeds threshold
        # FIX (Jan 2026): Added after $4K loss on single SHORT with no stop
        self.emergency_stop_pct = risk_config.get('emergency_stop_pct', 5.0) / 100
        self.emergency_stop_enabled = True

        # EOD close simulation - Respects config setting to match live trading
        # When disabled (default), positions are NOT force-closed at EOD
        self.eod_close_bar_hour = 15  # Close on bars starting at 3 PM or later
        self.eod_close_enabled = exit_config.get('eod_close', False)  # Read from config

        # Trailing stop configuration (defaults match config.yaml expectations)
        trailing_config = self.config.get('trailing_stop', {})
        self.trailing_stop_enabled = trailing_config.get('enabled', True)
        self.trailing_activation_pct = trailing_config.get('activation_pct', 0.25) / 100  # 0.25%
        self.trailing_trail_pct = trailing_config.get('trail_pct', 0.25) / 100  # 0.25%
        self.trailing_move_to_breakeven = trailing_config.get('move_to_breakeven', True)

        # Kill switch trace logging (for debugging)
        self.kill_switch_trace = kill_switch_trace

        # State tracking
        self._reset_state()

        # Setup dedicated backtest analytics log file
        self._analytics_log_file = Path('logs/backtest_analytics.log')
        self._analytics_log_file.parent.mkdir(parents=True, exist_ok=True)

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
        self._backtest_start_date = None

        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0

        # Drawdown window tracking
        self.drawdown_peak_value = self.initial_capital
        self.drawdown_peak_date = None
        self.drawdown_trough_value = self.initial_capital
        self.drawdown_trough_date = None

        # Daily equity tracking for worst days analysis
        self.daily_equity_snapshots = {}  # {date_str: {'open': val, 'close': val, 'high': val, 'low': val}}
        self._current_day_equity = {'high': self.initial_capital, 'low': self.initial_capital}

        # Daily P&L tracking
        self.daily_pnl = 0.0
        self.daily_starting_capital = self.initial_capital
        self.current_trading_day = None
        self.kill_switch_triggered = False

        # Kill switch trace logging (for debugging) - reset trace log
        self._kill_switch_trace_log = []  # Stores trace events

        # Reset entry gate
        if self.entry_gate:
            self.entry_gate.reset()

        # Diagnostics
        self._diag_exit_reasons = defaultdict(lambda: defaultdict(int))
        self._diag_entry_blocks = defaultdict(lambda: defaultdict(int))
        self._diag_kill_switch_blocks_per_day = defaultdict(int)

        # Scanner state
        self._daily_scanned_symbols = {}

    def _build_daily_scan_results(
        self,
        historical_data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str
    ) -> Dict[str, List[str]]:
        """
        Build a dict of date -> scanned symbols for the backtest period.

        PERFORMANCE OPTIMIZED (Jan 2026):
        - Pre-computes volatility metrics once per symbol using vectorized operations
        - Avoids repeated DataFrame copies and recalculations
        - Reduces O(symbols × days) to O(symbols) for score calculation

        NO LOOK-AHEAD BIAS (Jan 2026 FIX):
        - For each trading day N, scanner uses PREVIOUS day (N-1) data
        - This ensures we only use information available at market open

        Args:
            historical_data: Dict mapping symbol -> DataFrame with OHLCV
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Dict mapping date_str -> list of symbols to trade that day
        """
        if not self.scanner:
            return {}

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

        logger.info(f"Building daily scan results for {len(trading_days)} trading days (optimized)...")

        # PERFORMANCE FIX: Pre-compute rolling volatility scores for all symbols
        # This avoids recalculating the same metrics for each day
        symbol_daily_scores = {}  # {symbol: {date_str: score_dict}}

        for symbol, df in historical_data.items():
            if df is None or df.empty or 'timestamp' not in df.columns:
                continue

            try:
                df = df.copy()
                df['date'] = pd.to_datetime(df['timestamp']).dt.date

                # Pre-compute True Range and rolling ATR (vectorized)
                df['prev_close'] = df['close'].shift(1)
                df['tr'] = np.maximum(
                    df['high'] - df['low'],
                    np.maximum(
                        abs(df['high'] - df['prev_close']),
                        abs(df['low'] - df['prev_close'])
                    )
                )

                # Rolling ATR (14 periods)
                lookback = self.scanner.lookback_days if self.scanner else 14
                min_bars = lookback * 7  # For hourly data
                df['atr'] = df['tr'].rolling(lookback).mean()

                # Rolling average volume (20 periods)
                df['avg_vol_20'] = df['volume'].rolling(20).mean()

                # Daily range %
                df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100

                # Group by date and get last bar of each day
                daily_data = df.groupby('date').agg({
                    'close': 'last',
                    'atr': 'last',
                    'volume': 'mean',
                    'avg_vol_20': 'last',
                    'range_pct': 'mean'
                }).reset_index()

                symbol_daily_scores[symbol] = {}
                min_price = self.scanner.min_price if self.scanner else 5
                max_price = self.scanner.max_price if self.scanner else 1000
                min_volume = self.scanner.min_volume if self.scanner else 500000
                # FIX (Jan 2026): Use scanner's configured weights for backtest/live alignment
                weights = self.scanner.weights if self.scanner else {'atr_pct': 0.5, 'daily_range_pct': 0.3, 'volume_ratio': 0.2}

                for _, row in daily_data.iterrows():
                    date = row['date']
                    if date < start_dt or date > end_dt:
                        continue

                    price = row['close']
                    atr = row['atr']
                    avg_vol = row['avg_vol_20']
                    vol = row['volume']
                    range_pct = row['range_pct']

                    # Apply filters
                    if pd.isna(price) or price < min_price or price > max_price:
                        continue
                    if pd.isna(avg_vol) or avg_vol < min_volume:
                        continue
                    if pd.isna(atr):
                        continue

                    # Calculate volatility score (using scanner weights for alignment)
                    atr_pct = (atr / price * 100) if price > 0 else 0
                    vol_ratio = min((vol / avg_vol) if avg_vol > 0 else 1.0, 5.0)

                    score = (atr_pct * weights.get('atr_pct', 0.5) +
                             range_pct * weights.get('daily_range_pct', 0.3) +
                             vol_ratio * weights.get('volume_ratio', 0.2))

                    symbol_daily_scores[symbol][date.strftime('%Y-%m-%d')] = {
                        'score': score,
                        'price': price,
                        'volume': avg_vol
                    }

            except Exception as e:
                logger.debug(f"Error pre-computing scores for {symbol}: {e}")
                continue

        # Build daily scans by ranking pre-computed scores
        # FIX (Jan 2026): Use PREVIOUS day's scores to avoid look-ahead bias
        # For trading day N, we use day N-1's data (what we'd know at market open)
        daily_scans = {}
        top_n = self.scanner.top_n if self.scanner else 10

        # Build a mapping from each date to the previous trading day
        prev_day_map = {}
        for i, date in enumerate(trading_days):
            if i > 0:
                prev_day_map[date.strftime('%Y-%m-%d')] = trading_days[i - 1].strftime('%Y-%m-%d')

        for date in trading_days:
            date_str = date.strftime('%Y-%m-%d')

            # FIX: Use PREVIOUS day's scores, not current day
            # This ensures no look-ahead bias - we only know yesterday's data at market open
            lookup_date = prev_day_map.get(date_str)
            if lookup_date is None:
                # First trading day - no previous data, skip or use empty
                daily_scans[date_str] = []
                continue

            # Collect scores for the PREVIOUS day
            candidates = []
            for symbol, date_scores in symbol_daily_scores.items():
                if lookup_date in date_scores:
                    candidates.append({
                        'symbol': symbol,
                        'score': date_scores[lookup_date]['score']
                    })

            # Sort by score descending and take top N
            candidates.sort(key=lambda x: x['score'], reverse=True)
            daily_scans[date_str] = [c['symbol'] for c in candidates[:top_n]]

        # Log summary
        unique_symbols = set()
        for syms in daily_scans.values():
            unique_symbols.update(syms)
        logger.info(f"Scanner selected {len(unique_symbols)} unique symbols across all days")

        # NEW (Jan 2026): Log scan results for each day for live/backtest comparison
        self._log_daily_scans(daily_scans, symbol_daily_scores, top_n)

        return daily_scans

    def _log_daily_scans(self, daily_scans: Dict[str, List[str]],
                         symbol_daily_scores: Dict[str, Dict[str, Dict]],
                         top_n: int):
        """
        Log daily scan results to database for live/backtest comparison.

        NEW (Jan 2026): Enables verification of scanner output alignment.
        """
        try:
            trade_logger = TradeLogger()
            scanner_config = self.scanner.get_config() if self.scanner else {}

            for date_str, selected_symbols in daily_scans.items():
                # Rebuild scored list for this date
                all_scores = []
                for symbol, date_scores in symbol_daily_scores.items():
                    if date_str in date_scores:
                        score_data = date_scores[date_str]
                        all_scores.append({
                            'symbol': symbol,
                            'vol_score': score_data['score'],
                            'price': score_data['price'],
                            'avg_volume': score_data['volume']
                        })

                # Sort by score for consistent logging
                all_scores.sort(key=lambda x: x['vol_score'], reverse=True)

                trade_logger.log_scan_result({
                    'scan_date': date_str,
                    'mode': 'BACKTEST',
                    'symbols_scanned': len(symbol_daily_scores),
                    'selected_symbols': selected_symbols,
                    'all_scores': all_scores[:50],  # Top 50 to keep size reasonable
                    'config': scanner_config
                })

        except Exception as e:
            logger.debug(f"Could not log backtest scan results: {e}")

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
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

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
            # PERF FIX: Don't copy DataFrame - strategies only read, never modify
            historical_data = data.iloc[:i]
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

                # TRACE POINT A: Start of trading day
                if self.kill_switch_trace:
                    self._kill_switch_trace_log.append({
                        'event': 'DAY_START',
                        'timestamp': str(timestamp),
                        'day': str(bar_date),
                        'day_start_equity': self.daily_starting_capital,
                        'daily_pnl': 0.0,
                        'kill_switch_triggered': False
                    })

            # ============ EXIT LOGIC (runs first, every bar) ============
            # FIX (Jan 2026): Exit precedence order: hard stop > trailing > time exit > EOD
            # This ensures catastrophic losses are prevented even if trailing stop is active
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

                # FIX (Jan 2026): Increment bars held for ExitManager time-based logic
                if self.use_tiered_exits and self.exit_manager and symbol in self.exit_manager.positions:
                    self.exit_manager.increment_bars_held(symbol)

                # ============ EMERGENCY STOP (UNREALIZED LOSS) ============
                # FIX (Jan 2026): Force close if unrealized daily loss exceeds threshold
                # This catches catastrophic losses before they wipe the account
                if self.emergency_stop_enabled and not exit_triggered:
                    if position_direction == 'LONG':
                        unrealized_pnl = (current_price - entry_price) * shares
                    else:  # SHORT
                        unrealized_pnl = (entry_price - current_price) * shares

                    total_daily_loss = self.daily_pnl + unrealized_pnl
                    if self.daily_starting_capital > 0:
                        total_daily_loss_pct = -total_daily_loss / self.daily_starting_capital
                        if total_daily_loss_pct >= self.emergency_stop_pct:
                            exit_triggered = True
                            if position_direction == 'LONG':
                                exit_price = current_price * (1 - self.STOP_SLIPPAGE - self.BID_ASK_SPREAD)
                            else:
                                exit_price = current_price * (1 + self.STOP_SLIPPAGE + self.BID_ASK_SPREAD)
                            exit_reason = 'emergency_stop'
                            logger.warning(f"EMERGENCY STOP: {symbol} - Daily loss {total_daily_loss_pct*100:.1f}% >= {self.emergency_stop_pct*100:.1f}%")

                # ============ TRAILING STOP LOGIC ============
                # Trailing stop runs FIRST (before hard stop) to match live/API behavior
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

                # ============ HARD STOP CHECK (after trailing, to match live/API) ============
                if not exit_triggered and not self.use_tiered_exits:
                    # Legacy hard stop for non-tiered mode
                    if position_direction == 'LONG' and bar_low <= stop_loss_price:
                        exit_triggered = True
                        exit_price = stop_loss_price * (1 - self.STOP_SLIPPAGE - self.BID_ASK_SPREAD)
                        exit_reason = 'stop_loss'
                    elif position_direction == 'SHORT' and bar_high >= stop_loss_price:
                        exit_triggered = True
                        exit_price = stop_loss_price * (1 + self.STOP_SLIPPAGE + self.BID_ASK_SPREAD)
                        exit_reason = 'stop_loss'

                # For tiered exits, ExitManager handles all exit logic (LONG and SHORT - Jan 2026)
                if not exit_triggered and self.use_tiered_exits and self.exit_manager:
                    current_atr = self._calculate_atr(data, i, period=14)
                    # Pass bar_high and bar_low for proper stop checking
                    exit_action = self.exit_manager.evaluate_exit(
                        symbol, current_price, current_atr,
                        bar_high=bar_high, bar_low=bar_low
                    )

                    if exit_action:
                        exit_triggered = True
                        exit_reason = exit_action['reason']
                        # Normalize 'hard_stop' to 'stop_loss' to match live/API
                        if exit_reason == 'hard_stop':
                            exit_reason = 'stop_loss'
                        exit_qty = exit_action.get('qty', shares)

                        # Apply slippage based on direction
                        if position_direction == 'LONG':
                            if exit_reason in ['stop_loss', 'profit_floor', 'atr_trailing']:
                                exit_price = exit_action.get('stop_price', current_price) * (1 - self.STOP_SLIPPAGE - self.BID_ASK_SPREAD)
                            else:
                                exit_price = current_price * (1 - self.EXIT_SLIPPAGE - self.BID_ASK_SPREAD)
                        else:  # SHORT
                            if exit_reason in ['stop_loss', 'profit_floor', 'atr_trailing']:
                                exit_price = exit_action.get('stop_price', current_price) * (1 + self.STOP_SLIPPAGE + self.BID_ASK_SPREAD)
                            else:
                                exit_price = current_price * (1 + self.EXIT_SLIPPAGE + self.BID_ASK_SPREAD)

                # Legacy exit logic for LONG (take profit and signal exit only)
                if not exit_triggered and position_direction == 'LONG' and not self.use_tiered_exits:
                    if bar_high >= take_profit_price:
                        exit_triggered = True
                        exit_price = take_profit_price * (1 - self.EXIT_SLIPPAGE - self.BID_ASK_SPREAD)
                        exit_reason = 'take_profit'

                    elif row['signal'] == -1 and current_price <= entry_price:
                        exit_triggered = True
                        exit_price = current_price * (1 - self.EXIT_SLIPPAGE - self.BID_ASK_SPREAD)
                        exit_reason = 'sell_signal'

                # Legacy exit logic for SHORT (take profit only - when NOT using tiered exits)
                # When tiered exits enabled, ExitManager handles profit floor, ATR trailing, partial TP
                if not exit_triggered and position_direction == 'SHORT' and not self.use_tiered_exits:
                    if bar_low <= take_profit_price:
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
                    kill_switch_before = self.kill_switch_triggered
                    self.daily_pnl += pnl

                    # Check kill switch after each trade
                    daily_loss_pct = 0.0
                    if self.daily_loss_kill_switch_enabled and self.daily_starting_capital > 0:
                        daily_loss_pct = -self.daily_pnl / self.daily_starting_capital
                        if daily_loss_pct >= self.max_daily_loss_pct:
                            if not self.kill_switch_triggered:
                                self.kill_switch_triggered = True
                                logger.info(f"KILL SWITCH TRIGGERED: Daily loss {daily_loss_pct*100:.2f}% >= {self.max_daily_loss_pct*100:.1f}%")
                                self._diag_entry_blocks[symbol]['kill_switch'] += 1

                    # TRACE POINT B: Trade close
                    if self.kill_switch_trace:
                        self._kill_switch_trace_log.append({
                            'event': 'TRADE_CLOSE',
                            'timestamp': str(timestamp),
                            'symbol': symbol,
                            'realized_pnl': pnl,
                            'daily_pnl_after': self.daily_pnl,
                            'day_start_equity': self.daily_starting_capital,
                            'daily_loss_pct': daily_loss_pct * 100,
                            'kill_switch_before': kill_switch_before,
                            'kill_switch_after': self.kill_switch_triggered,
                            'kill_switch_threshold_pct': self.max_daily_loss_pct * 100
                        })

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

            # Track drawdown with dates
            if self.portfolio_value > self.peak_value:
                self.peak_value = self.portfolio_value
                # New peak means new drawdown window starts
                self.drawdown_peak_value = self.portfolio_value
                self.drawdown_peak_date = timestamp

            drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
                self.drawdown_trough_value = self.portfolio_value
                self.drawdown_trough_date = timestamp

            # Track daily equity high/low
            bar_date_str = pd.to_datetime(timestamp).strftime('%Y-%m-%d') if timestamp else None
            if bar_date_str:
                if bar_date_str not in self.daily_equity_snapshots:
                    # New day - save previous day if exists
                    self.daily_equity_snapshots[bar_date_str] = {
                        'open': self.portfolio_value,
                        'high': self.portfolio_value,
                        'low': self.portfolio_value,
                        'close': self.portfolio_value
                    }
                else:
                    day = self.daily_equity_snapshots[bar_date_str]
                    day['high'] = max(day['high'], self.portfolio_value)
                    day['low'] = min(day['low'], self.portfolio_value)
                    day['close'] = self.portfolio_value

                # Log when transitioning to new day
                if bar_date_str and hasattr(self, '_prev_bar_date') and self._prev_bar_date != bar_date_str:
                    self._log_daily_summary(self._prev_bar_date)
                self._prev_bar_date = bar_date_str

            # ============ PROCESS PENDING ENTRY ============
            # FIX (Jan 2026): Check kill switch BEFORE processing pending entry
            # Bug: Pending entries from previous bar would execute even after kill switch triggered
            # This violated the rule: "if live would be blocked, backtest must also be blocked"
            if pending_entry is not None and position_direction is None:
                # Kill switch check - block pending entry if triggered
                if self.kill_switch_triggered:
                    self._diag_entry_blocks[symbol]['kill_switch'] += 1
                    if bar_date is not None:
                        self._diag_kill_switch_blocks_per_day[str(bar_date)] += 1

                    # TRACE POINT C: Pending entry BLOCKED by kill switch
                    if self.kill_switch_trace:
                        self._kill_switch_trace_log.append({
                            'event': 'ENTRY_BLOCKED',
                            'timestamp': str(timestamp),
                            'symbol': symbol,
                            'entry_type': 'pending_fill',
                            'kill_switch_triggered': True,
                            'block_reason': 'kill_switch',
                            'direction': pending_entry.get('direction', 'UNKNOWN')
                        })

                    pending_entry = None  # Discard the pending entry
                else:
                    # Process the pending entry
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

                        # TRACE POINT C: Entry EXECUTED (pending fill succeeded)
                        if self.kill_switch_trace:
                            self._kill_switch_trace_log.append({
                                'event': 'ENTRY_EXECUTED',
                                'timestamp': str(timestamp),
                                'symbol': symbol,
                                'entry_type': 'pending_fill',
                                'kill_switch_triggered': self.kill_switch_triggered,
                                'direction': direction,
                                'entry_price': entry_price,
                                'shares': shares
                            })

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

                            # TRACE POINT C: New signal BLOCKED by kill switch
                            if self.kill_switch_trace:
                                self._kill_switch_trace_log.append({
                                    'event': 'ENTRY_BLOCKED',
                                    'timestamp': str(timestamp),
                                    'symbol': symbol,
                                    'entry_type': 'new_signal',
                                    'kill_switch_triggered': True,
                                    'block_reason': 'kill_switch',
                                    'direction': 'LONG' if signal == 1 else 'SHORT'
                                })

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

                                # TRACE POINT C: New signal BLOCKED by entry gate
                                if self.kill_switch_trace:
                                    self._kill_switch_trace_log.append({
                                        'event': 'ENTRY_BLOCKED',
                                        'timestamp': str(timestamp),
                                        'symbol': symbol,
                                        'entry_type': 'new_signal',
                                        'kill_switch_triggered': self.kill_switch_triggered,
                                        'block_reason': reason,
                                        'direction': 'LONG' if signal == 1 else 'SHORT'
                                    })

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
        Simulate trading across ALL symbols interleaved by timestamp.

        This ensures portfolio-wide daily P&L tracking and kill switch
        works across all symbols, not per-symbol.

        Args:
            signals_data: Dict mapping symbol -> DataFrame with signals

        Returns:
            List of all trades across all symbols
        """
        all_trades = []

        # Per-symbol state tracking
        symbol_state = {}
        for symbol in signals_data:
            symbol_state[symbol] = {
                'position_direction': None,
                'entry_price': 0.0,
                'entry_index': 0,
                'entry_time': None,
                'shares': 0,
                'stop_loss_price': 0.0,
                'take_profit_price': 0.0,
                'highest_price': 0.0,
                'lowest_price': float('inf'),
                'last_trade_bar': -999,
                'pending_entry': None,
                'entry_strategy': '',
                'entry_reasoning': '',
                'trailing_activated': False,
                'trailing_stop_price': 0.0,
                'bar_count': 0,
            }

        # Build unified timeline: list of (timestamp, symbol, row_index, row)
        timeline = []
        for symbol, data in signals_data.items():
            for i in range(len(data)):
                row = data.iloc[i]
                ts = row.get('timestamp', i)
                # Normalize timestamp for sorting
                if isinstance(ts, str):
                    ts = pd.to_datetime(ts)
                timeline.append((ts, symbol, i, row, data))

        # Sort by timestamp
        timeline.sort(key=lambda x: x[0])

        # Process each bar in chronological order
        for ts, symbol, i, row, data in timeline:
            state = symbol_state[symbol]
            state['bar_count'] += 1

            current_price = row['close']
            timestamp = row.get('timestamp', i)

            # Skip NaN prices
            if pd.isna(current_price) or pd.isna(row.get('open')) or pd.isna(row.get('high')) or pd.isna(row.get('low')):
                continue

            bar_high = row.get('high', current_price)
            bar_low = row.get('low', current_price)

            # ============ DAILY RESET (PORTFOLIO-WIDE) ============
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

            # Reset daily P&L on new trading day (portfolio-wide)
            if bar_date is not None and bar_date != self.current_trading_day:
                if self.current_trading_day is not None:
                    logger.debug(f"New trading day: {bar_date}, resetting daily P&L (was ${self.daily_pnl:.2f})")
                self.current_trading_day = bar_date
                self.daily_pnl = 0.0
                self.daily_starting_capital = self.portfolio_value
                self.kill_switch_triggered = False

                # TRACE POINT A: Start of trading day
                if self.kill_switch_trace:
                    self._kill_switch_trace_log.append({
                        'event': 'DAY_START',
                        'timestamp': str(timestamp),
                        'day': str(bar_date),
                        'day_start_equity': self.daily_starting_capital,
                        'daily_pnl': 0.0,
                        'kill_switch_triggered': False
                    })

            position_direction = state['position_direction']
            entry_price = state['entry_price']
            shares = state['shares']
            highest_price = state['highest_price']
            lowest_price = state['lowest_price']
            trailing_activated = state['trailing_activated']
            trailing_stop_price = state['trailing_stop_price']
            stop_loss_price = state['stop_loss_price']

            # ============ EXIT LOGIC ============
            if position_direction is not None:
                exit_triggered = False
                exit_reason = ''
                exit_price = current_price
                exit_qty = shares

                # Update price tracking
                if not pd.isna(bar_high) and bar_high > highest_price:
                    highest_price = bar_high
                    state['highest_price'] = highest_price
                if not pd.isna(bar_low) and bar_low < lowest_price:
                    lowest_price = bar_low
                    state['lowest_price'] = lowest_price

                # FIX (Jan 2026): Increment bars held for ExitManager time-based logic
                if self.use_tiered_exits and self.exit_manager and symbol in self.exit_manager.positions:
                    self.exit_manager.increment_bars_held(symbol)

                # ============ EMERGENCY STOP (UNREALIZED LOSS) ============
                # FIX (Jan 2026): Force close if unrealized daily loss exceeds threshold
                if self.emergency_stop_enabled and not exit_triggered:
                    if position_direction == 'LONG':
                        unrealized_pnl = (current_price - entry_price) * shares
                    else:  # SHORT
                        unrealized_pnl = (entry_price - current_price) * shares

                    total_daily_loss = self.daily_pnl + unrealized_pnl
                    if self.daily_starting_capital > 0:
                        total_daily_loss_pct = -total_daily_loss / self.daily_starting_capital
                        if total_daily_loss_pct >= self.emergency_stop_pct:
                            exit_triggered = True
                            if position_direction == 'LONG':
                                exit_price = current_price * (1 - self.STOP_SLIPPAGE - self.BID_ASK_SPREAD)
                            else:
                                exit_price = current_price * (1 + self.STOP_SLIPPAGE + self.BID_ASK_SPREAD)
                            exit_reason = 'emergency_stop'
                            logger.warning(f"EMERGENCY STOP: {symbol} - Daily loss {total_daily_loss_pct*100:.1f}% >= {self.emergency_stop_pct*100:.1f}%")

                # Trailing stop logic
                if self.trailing_stop_enabled and not exit_triggered:
                    if position_direction == 'LONG':
                        current_profit_pct = (highest_price - entry_price) / entry_price
                        if not trailing_activated and current_profit_pct >= self.trailing_activation_pct:
                            trailing_activated = True
                            state['trailing_activated'] = True
                            if self.trailing_move_to_breakeven:
                                trailing_stop_price = entry_price
                            else:
                                trailing_stop_price = highest_price * (1 - self.trailing_trail_pct)
                            state['trailing_stop_price'] = trailing_stop_price

                        if trailing_activated:
                            new_trail_price = highest_price * (1 - self.trailing_trail_pct)
                            if new_trail_price > trailing_stop_price:
                                trailing_stop_price = new_trail_price
                                state['trailing_stop_price'] = trailing_stop_price

                            if bar_low <= trailing_stop_price:
                                exit_triggered = True
                                exit_price = trailing_stop_price * (1 - self.STOP_SLIPPAGE - self.BID_ASK_SPREAD)
                                exit_reason = 'trailing_stop'

                    elif position_direction == 'SHORT':
                        current_profit_pct = (entry_price - lowest_price) / entry_price
                        if not trailing_activated and current_profit_pct >= self.trailing_activation_pct:
                            trailing_activated = True
                            state['trailing_activated'] = True
                            if self.trailing_move_to_breakeven:
                                trailing_stop_price = entry_price
                            else:
                                trailing_stop_price = lowest_price * (1 + self.trailing_trail_pct)
                            state['trailing_stop_price'] = trailing_stop_price

                        if trailing_activated:
                            new_trail_price = lowest_price * (1 + self.trailing_trail_pct)
                            if new_trail_price < trailing_stop_price or trailing_stop_price == 0:
                                trailing_stop_price = new_trail_price
                                state['trailing_stop_price'] = trailing_stop_price

                            if bar_high >= trailing_stop_price:
                                exit_triggered = True
                                exit_price = trailing_stop_price * (1 + self.STOP_SLIPPAGE + self.BID_ASK_SPREAD)
                                exit_reason = 'trailing_stop'

                # Hard stop check
                if not exit_triggered and not self.use_tiered_exits:
                    if position_direction == 'LONG' and bar_low <= stop_loss_price:
                        exit_triggered = True
                        exit_price = stop_loss_price * (1 - self.STOP_SLIPPAGE - self.BID_ASK_SPREAD)
                        exit_reason = 'stop_loss'
                    elif position_direction == 'SHORT' and bar_high >= stop_loss_price:
                        exit_triggered = True
                        exit_price = stop_loss_price * (1 + self.STOP_SLIPPAGE + self.BID_ASK_SPREAD)
                        exit_reason = 'stop_loss'

                # Tiered exits via ExitManager (LONG and SHORT - Jan 2026)
                if not exit_triggered and self.use_tiered_exits and self.exit_manager:
                    current_atr = self._calculate_atr(data, i, period=14)
                    # Pass bar_high and bar_low for proper stop checking
                    exit_action = self.exit_manager.evaluate_exit(
                        symbol, current_price, current_atr,
                        bar_high=bar_high, bar_low=bar_low
                    )
                    if exit_action:
                        exit_triggered = True
                        exit_reason = exit_action['reason']
                        if exit_reason == 'hard_stop':
                            exit_reason = 'stop_loss'
                        exit_qty = exit_action.get('qty', shares)

                        # Apply slippage based on direction
                        if position_direction == 'LONG':
                            if exit_reason in ['stop_loss', 'profit_floor', 'atr_trailing']:
                                exit_price = exit_action.get('stop_price', current_price) * (1 - self.STOP_SLIPPAGE - self.BID_ASK_SPREAD)
                            else:
                                exit_price = current_price * (1 - self.EXIT_SLIPPAGE - self.BID_ASK_SPREAD)
                        else:  # SHORT
                            if exit_reason in ['stop_loss', 'profit_floor', 'atr_trailing']:
                                exit_price = exit_action.get('stop_price', current_price) * (1 + self.STOP_SLIPPAGE + self.BID_ASK_SPREAD)
                            else:
                                exit_price = current_price * (1 + self.EXIT_SLIPPAGE + self.BID_ASK_SPREAD)

                # Max hold time
                if not exit_triggered:
                    entry_time_val = state['entry_time']
                    if entry_time_val is not None and bar_datetime is not None:
                        elapsed = bar_datetime - entry_time_val
                        elapsed_hours = elapsed.total_seconds() / 3600
                    else:
                        elapsed_hours = state['bar_count'] - state['entry_index']

                    if elapsed_hours >= self.max_hold_hours:
                        exit_triggered = True
                        if position_direction == 'LONG':
                            exit_price = current_price * (1 - self.EXIT_SLIPPAGE - self.BID_ASK_SPREAD)
                        else:
                            exit_price = current_price * (1 + self.EXIT_SLIPPAGE + self.BID_ASK_SPREAD)
                        exit_reason = 'max_hold'

                bars_held = state['bar_count'] - state['entry_index']

                # Execute exit
                if exit_triggered:
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

                    state['last_trade_bar'] = state['bar_count']
                    pnl_pct = (pnl / entry_cost) * 100 if entry_cost > 0 else 0

                    self.total_pnl += pnl
                    self.total_trades += 1

                    if pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                        if self.entry_gate:
                            self.entry_gate.record_loss(timestamp)

                    # Track PORTFOLIO-WIDE daily P&L for kill switch
                    kill_switch_before = self.kill_switch_triggered
                    self.daily_pnl += pnl

                    # Check kill switch after each trade
                    daily_loss_pct = 0.0
                    if self.daily_loss_kill_switch_enabled and self.daily_starting_capital > 0:
                        daily_loss_pct = -self.daily_pnl / self.daily_starting_capital
                        if daily_loss_pct >= self.max_daily_loss_pct:
                            if not self.kill_switch_triggered:
                                self.kill_switch_triggered = True
                                logger.info(f"KILL SWITCH TRIGGERED: Portfolio daily loss {daily_loss_pct*100:.2f}% >= {self.max_daily_loss_pct*100:.1f}%")

                    # TRACE POINT B: Trade close
                    if self.kill_switch_trace:
                        self._kill_switch_trace_log.append({
                            'event': 'TRADE_CLOSE',
                            'timestamp': str(timestamp),
                            'symbol': symbol,
                            'realized_pnl': pnl,
                            'daily_pnl_after': self.daily_pnl,
                            'day_start_equity': self.daily_starting_capital,
                            'daily_loss_pct': daily_loss_pct * 100,
                            'kill_switch_before': kill_switch_before,
                            'kill_switch_after': self.kill_switch_triggered,
                            'kill_switch_threshold_pct': self.max_daily_loss_pct * 100
                        })

                    # MFE/MAE calculation
                    if position_direction == 'LONG':
                        mfe = highest_price - entry_price
                        mae = entry_price - lowest_price
                    else:
                        mfe = entry_price - lowest_price
                        mae = highest_price - entry_price
                    mfe_pct = (mfe / entry_price) * 100 if entry_price > 0 else 0
                    mae_pct = (mae / entry_price) * 100 if entry_price > 0 else 0

                    all_trades.append({
                        'symbol': symbol,
                        'direction': position_direction,
                        'entry_date': state['entry_time'],
                        'exit_date': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'strategy': state['entry_strategy'],
                        'reasoning': state['entry_reasoning'],
                        'bars_held': bars_held,
                        'mfe': mfe,
                        'mae': mae,
                        'mfe_pct': mfe_pct,
                        'mae_pct': mae_pct
                    })

                    if self.use_tiered_exits and self.exit_manager:
                        self.exit_manager.unregister_position(symbol)

                    # Reset state
                    state['trailing_activated'] = False
                    state['trailing_stop_price'] = 0.0
                    state['position_direction'] = None
                    state['highest_price'] = 0.0
                    state['lowest_price'] = float('inf')
                    position_direction = None

            # ============ UPDATE PORTFOLIO VALUE ============
            self.portfolio_value = self.cash
            for sym, st in symbol_state.items():
                if st['position_direction'] == 'LONG':
                    # Get current price for this symbol
                    sym_price = signals_data[sym].iloc[min(st['bar_count']-1, len(signals_data[sym])-1)]['close']
                    self.portfolio_value += st['shares'] * sym_price
                elif st['position_direction'] == 'SHORT':
                    sym_price = signals_data[sym].iloc[min(st['bar_count']-1, len(signals_data[sym])-1)]['close']
                    unrealized_pnl = (st['entry_price'] - sym_price) * st['shares']
                    self.portfolio_value += unrealized_pnl

            if self.portfolio_value > self.peak_value:
                self.peak_value = self.portfolio_value
                # New peak means new drawdown window starts
                self.drawdown_peak_value = self.portfolio_value
                self.drawdown_peak_date = timestamp

            drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
                self.drawdown_trough_value = self.portfolio_value
                self.drawdown_trough_date = timestamp

            # Track daily equity high/low
            bar_date_str = pd.to_datetime(timestamp).strftime('%Y-%m-%d') if timestamp else None
            if bar_date_str:
                if bar_date_str not in self.daily_equity_snapshots:
                    # New day - save previous day if exists
                    self.daily_equity_snapshots[bar_date_str] = {
                        'open': self.portfolio_value,
                        'high': self.portfolio_value,
                        'low': self.portfolio_value,
                        'close': self.portfolio_value
                    }
                else:
                    day = self.daily_equity_snapshots[bar_date_str]
                    day['high'] = max(day['high'], self.portfolio_value)
                    day['low'] = min(day['low'], self.portfolio_value)
                    day['close'] = self.portfolio_value

                # Log when transitioning to new day
                if bar_date_str and hasattr(self, '_prev_bar_date') and self._prev_bar_date != bar_date_str:
                    self._log_daily_summary(self._prev_bar_date)
                self._prev_bar_date = bar_date_str

            # ============ PROCESS PENDING ENTRY ============
            pending_entry = state['pending_entry']
            if pending_entry is not None and state['position_direction'] is None:
                if self.kill_switch_triggered:
                    # TRACE POINT C: Pending entry BLOCKED
                    if self.kill_switch_trace:
                        self._kill_switch_trace_log.append({
                            'event': 'ENTRY_BLOCKED',
                            'timestamp': str(timestamp),
                            'symbol': symbol,
                            'entry_type': 'pending_fill',
                            'kill_switch_triggered': True,
                            'block_reason': 'kill_switch',
                            'direction': pending_entry.get('direction', 'UNKNOWN')
                        })
                    state['pending_entry'] = None
                else:
                    # BUG FIX (Jan 4, 2026): Check max concurrent positions for pending entries too
                    open_position_count = sum(
                        1 for s in symbol_state.values()
                        if s['position_direction'] is not None
                    )
                    if open_position_count >= self.max_open_positions:
                        if self.kill_switch_trace:
                            self._kill_switch_trace_log.append({
                                'event': 'ENTRY_BLOCKED',
                                'timestamp': str(timestamp),
                                'symbol': symbol,
                                'entry_type': 'pending_fill',
                                'kill_switch_triggered': False,
                                'block_reason': f'max_positions ({open_position_count}/{self.max_open_positions})',
                                'direction': pending_entry.get('direction', 'UNKNOWN')
                            })
                        state['pending_entry'] = None
                        continue
                    # Process the pending entry
                    open_price = row.get('open', current_price)
                    direction = pending_entry.get('direction', 'LONG')

                    if direction == 'LONG':
                        realistic_entry_price = open_price * (1 + self.ENTRY_SLIPPAGE + self.BID_ASK_SPREAD)
                        new_stop_loss_price = realistic_entry_price * (1 - self.default_stop_loss_pct)
                    else:
                        realistic_entry_price = open_price * (1 - self.ENTRY_SLIPPAGE - self.BID_ASK_SPREAD)
                        new_stop_loss_price = realistic_entry_price * (1 + self.default_stop_loss_pct)

                    new_shares = self.risk_manager.calculate_position_size(
                        self.portfolio_value, realistic_entry_price, new_stop_loss_price
                    )

                    can_enter = False
                    if direction == 'LONG':
                        cost = new_shares * realistic_entry_price * (1 + self.COMMISSION)
                        if cost <= self.cash and new_shares > 0:
                            self.cash -= cost
                            can_enter = True
                    else:
                        margin_required = new_shares * realistic_entry_price * 0.5
                        if margin_required <= self.cash and new_shares > 0:
                            self.cash += new_shares * realistic_entry_price * (1 - self.COMMISSION)
                            can_enter = True

                    if can_enter:
                        state['position_direction'] = direction
                        state['entry_price'] = realistic_entry_price
                        state['entry_index'] = state['bar_count']
                        state['entry_time'] = timestamp
                        state['shares'] = new_shares
                        state['stop_loss_price'] = new_stop_loss_price
                        state['highest_price'] = realistic_entry_price
                        state['lowest_price'] = realistic_entry_price
                        state['entry_strategy'] = pending_entry.get('strategy', 'Unknown')
                        state['entry_reasoning'] = pending_entry.get('reasoning', '')
                        state['trailing_activated'] = False
                        state['trailing_stop_price'] = 0.0

                        # Register with exit manager (LONG and SHORT - Jan 2026)
                        if self.use_tiered_exits and self.exit_manager:
                            self.exit_manager.register_position(
                                symbol=symbol,
                                entry_price=realistic_entry_price,
                                quantity=new_shares,
                                entry_time=timestamp if isinstance(timestamp, datetime) else None,
                                direction=direction
                            )

                        if self.entry_gate:
                            self.entry_gate.record_entry(symbol, timestamp)

                        # TRACE POINT C: Entry EXECUTED
                        if self.kill_switch_trace:
                            self._kill_switch_trace_log.append({
                                'event': 'ENTRY_EXECUTED',
                                'timestamp': str(timestamp),
                                'symbol': symbol,
                                'entry_type': 'pending_fill',
                                'kill_switch_triggered': self.kill_switch_triggered,
                                'direction': direction,
                                'entry_price': realistic_entry_price,
                                'shares': new_shares
                            })

                    state['pending_entry'] = None

            # ============ CHECK FOR NEW ENTRY ============
            if state['position_direction'] is None and state['pending_entry'] is None:
                signal = row['signal']
                if signal in [1, -1]:
                    bars_since_last = state['bar_count'] - state['last_trade_bar']
                    if bars_since_last >= self.COOLDOWN_BARS:
                        # Kill switch check (PORTFOLIO-WIDE)
                        if self.kill_switch_triggered:
                            # TRACE POINT C: New signal BLOCKED
                            if self.kill_switch_trace:
                                self._kill_switch_trace_log.append({
                                    'event': 'ENTRY_BLOCKED',
                                    'timestamp': str(timestamp),
                                    'symbol': symbol,
                                    'entry_type': 'new_signal',
                                    'kill_switch_triggered': True,
                                    'block_reason': 'kill_switch',
                                    'direction': 'LONG' if signal == 1 else 'SHORT'
                                })
                            continue

                        # BUG FIX (Jan 4, 2026): Check max concurrent positions
                        # Previously no limit was enforced, causing 62%+ daily losses
                        open_position_count = sum(
                            1 for s in symbol_state.values()
                            if s['position_direction'] is not None
                        )
                        if open_position_count >= self.max_open_positions:
                            if self.kill_switch_trace:
                                self._kill_switch_trace_log.append({
                                    'event': 'ENTRY_BLOCKED',
                                    'timestamp': str(timestamp),
                                    'symbol': symbol,
                                    'entry_type': 'new_signal',
                                    'kill_switch_triggered': False,
                                    'block_reason': f'max_positions ({open_position_count}/{self.max_open_positions})',
                                    'direction': 'LONG' if signal == 1 else 'SHORT'
                                })
                            continue

                        # Check entry gate
                        entry_allowed = True
                        if self.entry_gate:
                            entry_allowed, reason = self.entry_gate.check_entry_allowed(symbol, timestamp)

                            if not entry_allowed:
                                if self.kill_switch_trace:
                                    self._kill_switch_trace_log.append({
                                        'event': 'ENTRY_BLOCKED',
                                        'timestamp': str(timestamp),
                                        'symbol': symbol,
                                        'entry_type': 'new_signal',
                                        'kill_switch_triggered': self.kill_switch_triggered,
                                        'block_reason': reason,
                                        'direction': 'LONG' if signal == 1 else 'SHORT'
                                    })

                        if entry_allowed:
                            if signal == 1:  # BUY -> LONG
                                if self.shorts_only:
                                    continue
                                state['pending_entry'] = {
                                    'direction': 'LONG',
                                    'signal_price': current_price,
                                    'strategy': row.get('strategy', 'Unknown'),
                                    'reasoning': row.get('reasoning', '')
                                }
                            elif signal == -1:  # SELL -> SHORT
                                if self.longs_only:
                                    continue
                                state['pending_entry'] = {
                                    'direction': 'SHORT',
                                    'signal_price': current_price,
                                    'strategy': row.get('strategy', 'Unknown') + '_SHORT',
                                    'reasoning': row.get('reasoning', '')
                                }

        # Close any remaining positions at end of backtest
        for symbol, state in symbol_state.items():
            if state['position_direction'] is not None:
                data = signals_data[symbol]
                close_price = data.iloc[-1]['close']
                if state['position_direction'] == 'LONG':
                    exit_price = close_price * (1 - self.EXIT_SLIPPAGE - self.BID_ASK_SPREAD)
                    proceeds = state['shares'] * exit_price * (1 - self.COMMISSION)
                    self.cash += proceeds
                    entry_cost = state['shares'] * state['entry_price'] * (1 + self.COMMISSION)
                    pnl = proceeds - entry_cost
                else:
                    exit_price = close_price * (1 + self.EXIT_SLIPPAGE + self.BID_ASK_SPREAD)
                    cover_cost = state['shares'] * exit_price * (1 + self.COMMISSION)
                    self.cash -= cover_cost
                    pnl = (state['entry_price'] - exit_price) * state['shares']
                    entry_cost = state['shares'] * state['entry_price']

                pnl_pct = (pnl / entry_cost) * 100 if entry_cost > 0 else 0
                bars_held = state['bar_count'] - state['entry_index']

                all_trades.append({
                    'symbol': symbol,
                    'direction': state['position_direction'],
                    'entry_date': state['entry_time'],
                    'exit_date': data.iloc[-1].get('timestamp', 'end'),
                    'entry_price': state['entry_price'],
                    'exit_price': exit_price,
                    'shares': state['shares'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'end_of_backtest',
                    'strategy': state['entry_strategy'],
                    'reasoning': state['entry_reasoning'],
                    'bars_held': bars_held,
                    'mfe': 0,
                    'mae': 0,
                    'mfe_pct': 0,
                    'mae_pct': 0
                })

                if self.use_tiered_exits and self.exit_manager:
                    self.exit_manager.unregister_position(symbol)

        return all_trades

    def _get_worst_daily_drops(self, top_n: int = 5) -> List[Dict]:
        """
        Get the top N worst daily equity drops.

        Returns list of dicts with date, open, close, drop_pct, drop_dollars.
        """
        if not self.daily_equity_snapshots:
            return []

        daily_drops = []
        for date_str, day in self.daily_equity_snapshots.items():
            if day['open'] > 0:
                drop_pct = (day['close'] - day['open']) / day['open'] * 100
                drop_dollars = day['close'] - day['open']
                daily_drops.append({
                    'date': date_str,
                    'open': round(day['open'], 2),
                    'close': round(day['close'], 2),
                    'high': round(day['high'], 2),
                    'low': round(day['low'], 2),
                    'change_pct': round(drop_pct, 2),
                    'change_dollars': round(drop_dollars, 2)
                })

        # Sort by change_pct ascending (most negative first)
        daily_drops.sort(key=lambda x: x['change_pct'])
        return daily_drops[:top_n]

    def _write_analytics_log(self, results: Dict):
        """Write backtest analytics to persistent log file."""
        import json
        from datetime import datetime as dt

        log_entry = {
            'timestamp': dt.now().isoformat(),
            'initial_capital': self.initial_capital,
            'final_value': self.cash,
            'total_return_pct': (self.cash - self.initial_capital) / self.initial_capital * 100,
            'max_drawdown_pct': self.max_drawdown * 100,
            'drawdown_peak_date': str(self.drawdown_peak_date) if self.drawdown_peak_date else None,
            'drawdown_peak_value': self.drawdown_peak_value,
            'drawdown_trough_date': str(self.drawdown_trough_date) if self.drawdown_trough_date else None,
            'drawdown_trough_value': self.drawdown_trough_value,
            'worst_days': self._get_worst_daily_drops(10),
            'total_trades': len(results.get('trades', [])),
            'symbols': results.get('symbols', [])
        }

        # Append to log file (keep last 50 entries)
        existing = []
        if self._analytics_log_file.exists():
            try:
                with open(self._analytics_log_file, 'r') as f:
                    existing = [json.loads(line) for line in f if line.strip()]
            except Exception:
                existing = []

        existing.append(log_entry)
        existing = existing[-50:]  # Keep last 50

        try:
            with open(self._analytics_log_file, 'w') as f:
                for entry in existing:
                    f.write(json.dumps(entry) + '\n')
            logger.info(f"Analytics saved to {self._analytics_log_file}")
        except Exception as e:
            logger.warning(f"Failed to save analytics log: {e}")

    def _log_daily_summary(self, date_str: str):
        """Log daily equity summary to console."""
        if date_str not in self.daily_equity_snapshots:
            return

        day = self.daily_equity_snapshots[date_str]
        change_pct = (day['close'] - day['open']) / day['open'] * 100 if day['open'] > 0 else 0
        change_dollars = day['close'] - day['open']

        # Calculate current drawdown from peak
        current_dd = (self.peak_value - day['close']) / self.peak_value * 100 if self.peak_value > 0 else 0

        logger.info(
            f"[DAILY] {date_str} | "
            f"Open: ${day['open']:,.2f} | Close: ${day['close']:,.2f} | "
            f"Change: {change_pct:+.2f}% (${change_dollars:+,.2f}) | "
            f"DD from peak: {current_dd:.1f}%"
        )

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
            # Drawdown window analytics
            'drawdown_peak_date': str(self.drawdown_peak_date) if self.drawdown_peak_date else None,
            'drawdown_peak_value': self.drawdown_peak_value,
            'drawdown_trough_date': str(self.drawdown_trough_date) if self.drawdown_trough_date else None,
            'drawdown_trough_value': self.drawdown_trough_value,
            'worst_daily_drops': self._get_worst_daily_drops(5),
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

        # Pre-fetch all data for scanner mode (parallel loading)
        all_data = {}
        if self.scanner_enabled:
            logger.info("Scanner mode: pre-fetching data for all symbols in parallel...")

            def fetch_symbol_data(sym):
                return sym, self.fetch_data(sym, start_date, end_date)

            with ThreadPoolExecutor(max_workers=15) as executor:
                futures = {executor.submit(fetch_symbol_data, sym): sym for sym in symbols}
                for future in as_completed(futures):
                    symbol_result, data = future.result()
                    if data is not None and len(data) >= 30:
                        all_data[symbol_result] = data

            logger.info(f"Pre-fetched data for {len(all_data)} symbols")

            self._daily_scanned_symbols = self._build_daily_scan_results(
                all_data, start_date, end_date
            )

        # Phase 1: Parallel data fetching and signal generation
        logger.info("Generating signals for all symbols in parallel...")
        signals_data = {}

        def process_symbol(sym):
            """Fetch data and generate signals for a symbol."""
            if self.scanner_enabled and sym in all_data:
                data = all_data[sym]
            else:
                data = self.fetch_data(sym, start_date, end_date)

            if data is None or len(data) < 30:
                return sym, None

            # Generate signals (stateless per-symbol)
            data = self.generate_signals(sym, data)
            return sym, data

        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(process_symbol, sym): sym for sym in symbols}
            for future in as_completed(futures):
                symbol_result, data = future.result()
                if data is not None:
                    signals_data[symbol_result] = data

        logger.info(f"Generated signals for {len(signals_data)} symbols")

        # Phase 2: Interleaved trade simulation (portfolio-wide daily P&L tracking)
        # Process all symbols together, bar by bar, to ensure kill switch works across portfolio
        all_trades = self.simulate_trades_interleaved(signals_data)

        # Log per-symbol summary
        for symbol in symbols:
            symbol_trades = [t for t in all_trades if t['symbol'] == symbol]
            if symbol_trades:
                logger.info(f"{symbol}: {len(symbol_trades)} trades, P&L: ${sum(t['pnl'] for t in symbol_trades):,.2f}")

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
            'kill_switch_trace': self._kill_switch_trace_log if self.kill_switch_trace else None,
        }

        logger.info(f"Backtest complete: {len(all_trades)} trades, Total P&L: ${self.total_pnl:,.2f}")

        # Log final daily summary
        if hasattr(self, '_prev_bar_date') and self._prev_bar_date:
            self._log_daily_summary(self._prev_bar_date)

        # Log drawdown analysis
        logger.info("=" * 60)
        logger.info("DRAWDOWN ANALYSIS")
        logger.info("=" * 60)
        if self.drawdown_peak_date:
            logger.info(f"Peak: ${self.drawdown_peak_value:,.2f} on {self.drawdown_peak_date}")
        if self.drawdown_trough_date:
            logger.info(f"Trough: ${self.drawdown_trough_value:,.2f} on {self.drawdown_trough_date}")
        logger.info(f"Max Drawdown: {self.max_drawdown * 100:.1f}%")

        worst_days = self._get_worst_daily_drops(10)
        if worst_days:
            logger.info("\nTop 10 Worst Days:")
            for i, day in enumerate(worst_days, 1):
                logger.info(
                    f"  {i}. {day['date']}: {day['change_pct']:+.2f}% "
                    f"(${day['change_dollars']:+,.2f}) | "
                    f"${day['open']:,.2f} -> ${day['close']:,.2f}"
                )
        logger.info("=" * 60)

        # Log scanner diagnostic summary
        if self.scanner_enabled:
            # Get unique scanned symbols
            all_scanned = set()
            for syms in self._daily_scanned_symbols.values():
                all_scanned.update(syms)

            # Get symbols that actually traded
            traded_symbols = set(t['symbol'] for t in all_trades)

            # Get scanner filter stats
            total_scanner_filtered = sum(
                self._diag_entry_blocks[sym].get('scanner_filtered', 0)
                for sym in self._diag_entry_blocks
            )
            total_confidence_blocked = sum(
                self._diag_entry_blocks[sym].get('confidence', 0)
                for sym in self._diag_entry_blocks
            )

            logger.info("=" * 60)
            logger.info("SCANNER DIAGNOSTIC SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total symbols in universe: {len(symbols)}")
            logger.info(f"Unique scanner-selected symbols: {len(all_scanned)}")
            logger.info(f"Symbols that actually traded: {len(traded_symbols)}")
            logger.info(f"Scanner-selected: {sorted(all_scanned)}")
            logger.info(f"Actually traded: {sorted(traded_symbols)}")
            logger.info(f"Bars blocked by scanner filter: {total_scanner_filtered:,}")
            logger.info(f"Signals blocked by confidence: {total_confidence_blocked:,}")

            # Show scanned symbols that didn't trade
            scanned_but_no_trades = all_scanned - traded_symbols
            if scanned_but_no_trades:
                logger.info(f"Scanned but no trades (no signals or blocked): {sorted(scanned_but_no_trades)}")

            # Show if any non-scanned symbols traded (would be a BUG)
            non_scanned_trades = traded_symbols - all_scanned
            if non_scanned_trades:
                logger.warning(f"BUG: Non-scanned symbols traded: {sorted(non_scanned_trades)}")
            logger.info("=" * 60)

        self._write_analytics_log(results)

        return results


def run_backtest(
    symbols: list = None,
    start_date: str = None,
    end_date: str = None,
    initial_capital: float = 10000.0,
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

    # Load config first to check if scanner is enabled
    config_path = bot_dir / 'config.yaml'
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    scanner_enabled = config.get('volatility_scanner', {}).get('enabled', False)

    # Load universe if no symbols provided
    if symbols is None:
        universe_path = bot_dir / 'universe.yaml'
        if universe_path.exists():
            with open(universe_path, 'r') as f:
                universe = yaml.safe_load(f)

            if scanner_enabled:
                # Use full scanner_universe when scanner is enabled
                scanner_universe = universe.get('scanner_universe', {})
                symbols = []
                for category, syms in scanner_universe.items():
                    if isinstance(syms, list):
                        symbols.extend(syms)
                # Remove duplicates while preserving order
                seen = set()
                symbols = [s for s in symbols if not (s in seen or seen.add(s))]
                logger.info(f"Scanner mode: loaded {len(symbols)} symbols from scanner_universe")
            else:
                # Use proven_symbols when scanner is disabled
                symbols = universe.get('proven_symbols', [])
                if not symbols:
                    symbols = universe.get('candidates', ['SPY', 'AAPL', 'MSFT'])
        else:
            symbols = ['SPY', 'AAPL', 'MSFT']

    if not symbols:
        logger.error("No symbols to backtest!")
        return None

    # Filter out non-string entries (e.g., YAML parsing "ON" as True)
    original_count = len(symbols)
    symbols = [s for s in symbols if isinstance(s, str)]
    if len(symbols) < original_count:
        logger.warning(f"Filtered {original_count - len(symbols)} non-string entries from symbols list")

    # Set date range
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

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
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
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
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
