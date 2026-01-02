"""
Entry Gate - Trade frequency controls to reduce overtrading on 1-minute ML signals.

Created: December 18, 2025
Updated: December 19, 2025 - Added daily loss safety guard

This module provides shared entry gating logic for both live trading (trading_bot_ml.py)
and backtesting (backtester.py). It implements:

1. Per-Symbol Daily Trade Cap: Limits entries per symbol per day
2. Time-of-Day Filter: Restricts entries to specific trading sessions
3. Daily Loss Safety Guard: Blocks entries after N losing trades in a day

CRITICAL DESIGN NOTES:
- These controls ONLY gate NEW ENTRIES
- Exits (stop-loss, take-profit, trailing stops, time-based) are NEVER blocked
- The same gating logic must be used in both live and backtest for parity
"""

import logging
from datetime import datetime, date
from typing import Dict, Tuple, Optional
import pytz


class EntryGate:
    """
    Entry gate for controlling trade frequency.

    Implements per-symbol daily trade caps, time-of-day filtering, and daily loss guard.
    Used by both live trading and backtesting for consistent behavior.
    """

    # Time-of-day session definitions (in market time: US/Eastern)
    SESSION_OPEN_START = (9, 30)   # 9:30 AM ET
    SESSION_OPEN_END = (11, 0)     # 11:00 AM ET (first 90 minutes)
    SESSION_POWER_START = (14, 30) # 2:30 PM ET
    SESSION_POWER_END = (16, 0)    # 4:00 PM ET (last 90 minutes)

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the entry gate.

        Args:
            config: Configuration dict with trade_frequency settings.
                   If None, loads from global_config.
        """
        self.logger = logging.getLogger(__name__)

        # Load config
        if config is None:
            try:
                from .config import get_global_config
                full_config = get_global_config()
                config = full_config.config.get('trade_frequency', {})
            except Exception as e:
                self.logger.warning(f"Could not load config: {e}, using defaults")
                config = {}

        # Per-symbol daily trade cap settings
        self.max_trades_per_symbol_per_day = config.get('max_trades_per_symbol_per_day', 2)

        # Time-of-day filter settings
        self.enable_time_filter = config.get('enable_time_filter', False)
        self.allowed_sessions = config.get('allowed_sessions', 'OPEN_AND_POWER')
        self.timezone_str = config.get('timezone', 'America/New_York')
        self.timezone = pytz.timezone(self.timezone_str)

        # Daily loss safety guard settings (Dec 19, 2025)
        daily_loss_guard_config = config.get('daily_loss_guard', {})
        self.daily_loss_guard_enabled = daily_loss_guard_config.get('enabled', True)
        self.max_losing_trades_per_day = daily_loss_guard_config.get('max_losing_trades_per_day', 2)

        # Track trades per symbol per day: {(symbol, 'YYYY-MM-DD'): count}
        self.trades_today: Dict[Tuple[str, str], int] = {}

        # Track losing trades per day: {'YYYY-MM-DD': count}
        self.losses_today: Dict[str, int] = {}

        # Track current trading day for reset detection
        self._current_trading_day: Optional[str] = None

        # Log initialization
        loss_guard_status = (
            f"enabled (max_losses={self.max_losing_trades_per_day})"
            if self.daily_loss_guard_enabled
            else "disabled"
        )
        self.logger.info(
            f"EntryGate initialized: max_trades_per_symbol={self.max_trades_per_symbol_per_day}, "
            f"time_filter={'enabled (' + self.allowed_sessions + ')' if self.enable_time_filter else 'disabled'}, "
            f"daily_loss_guard={loss_guard_status}"
        )

    def _get_market_date(self, timestamp: Optional[datetime] = None) -> str:
        """
        Get the current market date as a string 'YYYY-MM-DD'.

        Uses the provided timestamp or current time, converted to market timezone.

        Args:
            timestamp: Optional datetime. If None, uses current time.

        Returns:
            Date string in 'YYYY-MM-DD' format.
        """
        if timestamp is None:
            timestamp = datetime.now(self.timezone)
        elif timestamp.tzinfo is None:
            # Naive datetime - assume it's in market timezone
            timestamp = self.timezone.localize(timestamp)
        else:
            # Convert to market timezone
            timestamp = timestamp.astimezone(self.timezone)

        return timestamp.strftime('%Y-%m-%d')

    def _reset_daily_counts_if_needed(self, market_date: str) -> None:
        """
        Reset daily trade counts and loss counts if we've moved to a new trading day.

        Args:
            market_date: Current market date string 'YYYY-MM-DD'.
        """
        if self._current_trading_day != market_date:
            # New trading day - reset all counts
            old_day = self._current_trading_day
            self._current_trading_day = market_date

            # Clear old trade entries (keep memory clean)
            old_keys = [k for k in self.trades_today.keys() if k[1] != market_date]
            for key in old_keys:
                del self.trades_today[key]

            # Clear old loss entries (keep memory clean)
            old_loss_keys = [k for k in self.losses_today.keys() if k != market_date]
            for key in old_loss_keys:
                del self.losses_today[key]

            if old_day:
                self.logger.info(f"EntryGate: Reset daily trade and loss counts for new day {market_date}")

    def _is_within_session(self, timestamp: datetime, session_start: Tuple[int, int], session_end: Tuple[int, int]) -> bool:
        """
        Check if timestamp is within a trading session.

        Args:
            timestamp: Datetime to check (should be in market timezone).
            session_start: (hour, minute) tuple for session start.
            session_end: (hour, minute) tuple for session end.

        Returns:
            True if within session, False otherwise.
        """
        hour = timestamp.hour
        minute = timestamp.minute

        start_minutes = session_start[0] * 60 + session_start[1]
        end_minutes = session_end[0] * 60 + session_end[1]
        current_minutes = hour * 60 + minute

        return start_minutes <= current_minutes < end_minutes

    def check_entry_allowed(self, symbol: str, timestamp: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Check if a new entry is allowed for the given symbol.

        This is the main entry point for gating logic. Call this BEFORE
        attempting to open a new position.

        Args:
            symbol: Stock symbol to check.
            timestamp: Optional datetime for the check. If None, uses current time.
                      For backtesting, pass the bar's timestamp.

        Returns:
            Tuple of (allowed: bool, reason: str).
            If allowed is False, reason contains a log-friendly explanation.
        """
        # Convert timestamp to market timezone
        if timestamp is None:
            market_time = datetime.now(self.timezone)
        elif timestamp.tzinfo is None:
            market_time = self.timezone.localize(timestamp)
        else:
            market_time = timestamp.astimezone(self.timezone)

        market_date = market_time.strftime('%Y-%m-%d')

        # Reset counts if new day
        self._reset_daily_counts_if_needed(market_date)

        # Check 1: Daily loss safety guard (Dec 19, 2025)
        # Blocks ALL entries if max losing trades reached for the day
        if self.daily_loss_guard_enabled:
            losses_today = self.losses_today.get(market_date, 0)
            if losses_today >= self.max_losing_trades_per_day:
                reason = (
                    f"ENTRY_BLOCKED | reason=daily_loss_limit | "
                    f"losses_today={losses_today}"
                )
                return False, reason

        # Check 2: Per-symbol daily trade cap
        key = (symbol, market_date)
        trades_today = self.trades_today.get(key, 0)

        if trades_today >= self.max_trades_per_symbol_per_day:
            reason = (
                f"ENTRY_BLOCKED | reason=max_trades_per_day | "
                f"symbol={symbol} | trades_today={trades_today}"
            )
            return False, reason

        # Check 3: Time-of-day filter (if enabled)
        if self.enable_time_filter:
            in_open = self._is_within_session(
                market_time, self.SESSION_OPEN_START, self.SESSION_OPEN_END
            )
            in_power = self._is_within_session(
                market_time, self.SESSION_POWER_START, self.SESSION_POWER_END
            )

            allowed_by_session = False

            if self.allowed_sessions == 'OPEN_ONLY':
                allowed_by_session = in_open
            elif self.allowed_sessions == 'POWER_HOUR_ONLY':
                allowed_by_session = in_power
            elif self.allowed_sessions == 'OPEN_AND_POWER':
                allowed_by_session = in_open or in_power

            if not allowed_by_session:
                current_time_str = market_time.strftime('%H:%M')
                reason = (
                    f"ENTRY_BLOCKED | reason=time_filter | "
                    f"symbol={symbol} | time={current_time_str} | "
                    f"allowed_sessions={self.allowed_sessions}"
                )
                return False, reason

        # All checks passed
        return True, ""

    def record_entry(self, symbol: str, timestamp: Optional[datetime] = None) -> None:
        """
        Record that an entry was made for the given symbol.

        Call this AFTER an entry order is successfully placed/filled.

        Args:
            symbol: Stock symbol that was entered.
            timestamp: Optional datetime of the entry. If None, uses current time.
        """
        market_date = self._get_market_date(timestamp)

        # Reset counts if new day
        self._reset_daily_counts_if_needed(market_date)

        # Increment count
        key = (symbol, market_date)
        self.trades_today[key] = self.trades_today.get(key, 0) + 1

        new_count = self.trades_today[key]
        remaining = self.max_trades_per_symbol_per_day - new_count

        self.logger.debug(
            f"EntryGate: Recorded entry for {symbol} on {market_date} "
            f"(count: {new_count}/{self.max_trades_per_symbol_per_day}, remaining: {remaining})"
        )

    def record_loss(self, timestamp: Optional[datetime] = None) -> None:
        """
        Record a losing trade for the daily loss guard.

        Call this AFTER a trade is closed with a loss (pnl < 0).
        This is portfolio-wide, not per-symbol.

        Args:
            timestamp: Optional datetime of the loss. If None, uses current time.
        """
        market_date = self._get_market_date(timestamp)

        # Reset counts if new day
        self._reset_daily_counts_if_needed(market_date)

        # Increment loss count
        self.losses_today[market_date] = self.losses_today.get(market_date, 0) + 1

        new_count = self.losses_today[market_date]
        remaining = self.max_losing_trades_per_day - new_count

        if remaining <= 0:
            self.logger.warning(
                f"EntryGate: Daily loss limit reached ({new_count}/{self.max_losing_trades_per_day}). "
                f"All new entries BLOCKED for rest of day. Exits still allowed."
            )
        else:
            self.logger.info(
                f"EntryGate: Recorded losing trade on {market_date} "
                f"(losses: {new_count}/{self.max_losing_trades_per_day}, remaining: {remaining})"
            )

    def get_trades_today(self, symbol: str, timestamp: Optional[datetime] = None) -> int:
        """
        Get the number of trades made today for a symbol.

        Args:
            symbol: Stock symbol to check.
            timestamp: Optional datetime. If None, uses current time.

        Returns:
            Number of entries made today for the symbol.
        """
        market_date = self._get_market_date(timestamp)
        key = (symbol, market_date)
        return self.trades_today.get(key, 0)

    def get_losses_today(self, timestamp: Optional[datetime] = None) -> int:
        """
        Get the number of losing trades today (portfolio-wide).

        Args:
            timestamp: Optional datetime. If None, uses current time.

        Returns:
            Number of losing trades today.
        """
        market_date = self._get_market_date(timestamp)
        return self.losses_today.get(market_date, 0)

    def get_remaining_trades(self, symbol: str, timestamp: Optional[datetime] = None) -> int:
        """
        Get the number of remaining trades allowed today for a symbol.

        Args:
            symbol: Stock symbol to check.
            timestamp: Optional datetime. If None, uses current time.

        Returns:
            Number of entries still allowed today for the symbol.
        """
        trades_made = self.get_trades_today(symbol, timestamp)
        return max(0, self.max_trades_per_symbol_per_day - trades_made)

    def is_daily_loss_limit_reached(self, timestamp: Optional[datetime] = None) -> bool:
        """
        Check if the daily loss limit has been reached.

        Args:
            timestamp: Optional datetime. If None, uses current time.

        Returns:
            True if loss limit reached and entries should be blocked.
        """
        if not self.daily_loss_guard_enabled:
            return False

        market_date = self._get_market_date(timestamp)
        losses = self.losses_today.get(market_date, 0)
        return losses >= self.max_losing_trades_per_day

    def reset(self) -> None:
        """
        Reset all trade and loss counts. Useful for testing or manual reset.
        """
        self.trades_today.clear()
        self.losses_today.clear()
        self._current_trading_day = None
        self.logger.info("EntryGate: All trade and loss counts reset")

    def get_status(self) -> Dict:
        """
        Get current status of the entry gate.

        Returns:
            Dict with configuration and current state.
        """
        market_date = self._get_market_date()
        return {
            'max_trades_per_symbol_per_day': self.max_trades_per_symbol_per_day,
            'enable_time_filter': self.enable_time_filter,
            'allowed_sessions': self.allowed_sessions,
            'timezone': self.timezone_str,
            'current_trading_day': self._current_trading_day,
            'active_counts': dict(self.trades_today),
            # Daily loss guard status
            'daily_loss_guard_enabled': self.daily_loss_guard_enabled,
            'max_losing_trades_per_day': self.max_losing_trades_per_day,
            'losses_today': self.losses_today.get(market_date, 0),
            'daily_loss_limit_reached': self.is_daily_loss_limit_reached()
        }
