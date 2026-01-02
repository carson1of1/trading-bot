"""Tests for EntryGate module - Trade frequency control"""
import pytest
import os
import sys
from datetime import datetime, timedelta
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.entry_gate import EntryGate


class TestEntryGateInitialization:
    """Test EntryGate initialization with default and custom config"""

    def test_default_initialization(self):
        """Should initialize with default values when no config provided"""
        gate = EntryGate(config={})
        assert gate.max_trades_per_symbol_per_day == 2
        assert gate.enable_time_filter is False
        assert gate.allowed_sessions == 'OPEN_AND_POWER'
        assert gate.timezone_str == 'America/New_York'
        assert gate.daily_loss_guard_enabled is True
        assert gate.max_losing_trades_per_day == 2

    def test_custom_config_initialization(self):
        """Should use custom config values when provided"""
        config = {
            'max_trades_per_symbol_per_day': 5,
            'enable_time_filter': True,
            'allowed_sessions': 'POWER_HOUR_ONLY',
            'timezone': 'US/Pacific',
            'daily_loss_guard': {
                'enabled': False,
                'max_losing_trades_per_day': 10
            }
        }
        gate = EntryGate(config=config)
        assert gate.max_trades_per_symbol_per_day == 5
        assert gate.enable_time_filter is True
        assert gate.allowed_sessions == 'POWER_HOUR_ONLY'
        assert gate.timezone_str == 'US/Pacific'
        assert gate.daily_loss_guard_enabled is False
        assert gate.max_losing_trades_per_day == 10

    def test_empty_trades_and_losses_on_init(self):
        """Should start with empty trade and loss counters"""
        gate = EntryGate(config={})
        assert len(gate.trades_today) == 0
        assert len(gate.losses_today) == 0
        assert gate._current_trading_day is None


class TestCheckEntryAllowed:
    """Test check_entry_allowed() method"""

    def test_allows_entry_when_under_limits(self):
        """Should allow entry when no limits are reached"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 2,
            'enable_time_filter': False,
            'daily_loss_guard': {'enabled': False}
        })
        allowed, reason = gate.check_entry_allowed('AAPL')
        assert allowed is True
        assert reason == ""

    def test_allows_multiple_symbols_independently(self):
        """Should track trades per symbol independently"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 2,
            'enable_time_filter': False,
            'daily_loss_guard': {'enabled': False}
        })
        # Record max trades for AAPL
        gate.record_entry('AAPL')
        gate.record_entry('AAPL')

        # AAPL should be blocked
        allowed_aapl, _ = gate.check_entry_allowed('AAPL')
        assert allowed_aapl is False

        # NVDA should still be allowed
        allowed_nvda, _ = gate.check_entry_allowed('NVDA')
        assert allowed_nvda is True

    def test_blocks_after_max_trades_per_symbol(self):
        """Should block entry after max trades for a symbol"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 2,
            'enable_time_filter': False,
            'daily_loss_guard': {'enabled': False}
        })
        # Record max trades
        gate.record_entry('AAPL')
        gate.record_entry('AAPL')

        allowed, reason = gate.check_entry_allowed('AAPL')
        assert allowed is False
        assert 'max_trades_per_day' in reason
        assert 'AAPL' in reason

    def test_blocks_entry_with_timezone_aware_timestamp(self):
        """Should handle timezone-aware timestamps correctly"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 1,
            'enable_time_filter': False,
            'daily_loss_guard': {'enabled': False}
        })
        tz = pytz.timezone('America/New_York')
        ts = tz.localize(datetime(2025, 12, 20, 10, 0, 0))

        gate.record_entry('AAPL', timestamp=ts)

        allowed, _ = gate.check_entry_allowed('AAPL', timestamp=ts)
        assert allowed is False

    def test_blocks_entry_with_naive_timestamp(self):
        """Should handle naive timestamps correctly"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 1,
            'enable_time_filter': False,
            'daily_loss_guard': {'enabled': False}
        })
        ts = datetime(2025, 12, 20, 10, 0, 0)  # Naive datetime

        gate.record_entry('AAPL', timestamp=ts)

        allowed, _ = gate.check_entry_allowed('AAPL', timestamp=ts)
        assert allowed is False


class TestRecordEntry:
    """Test record_entry() method"""

    def test_record_entry_increments_count(self):
        """Should increment trade count when recording entry"""
        gate = EntryGate(config={})
        assert gate.get_trades_today('AAPL') == 0

        gate.record_entry('AAPL')
        assert gate.get_trades_today('AAPL') == 1

        gate.record_entry('AAPL')
        assert gate.get_trades_today('AAPL') == 2

    def test_record_entry_with_timestamp(self):
        """Should record entry with specific timestamp"""
        gate = EntryGate(config={})
        ts = datetime(2025, 12, 20, 10, 0, 0)

        gate.record_entry('AAPL', timestamp=ts)
        assert gate.get_trades_today('AAPL', timestamp=ts) == 1

    def test_record_entry_separate_symbols(self):
        """Should track entries separately per symbol"""
        gate = EntryGate(config={})

        gate.record_entry('AAPL')
        gate.record_entry('NVDA')
        gate.record_entry('AAPL')

        assert gate.get_trades_today('AAPL') == 2
        assert gate.get_trades_today('NVDA') == 1


class TestRecordLoss:
    """Test record_loss() method"""

    def test_record_loss_increments_count(self):
        """Should increment loss count when recording loss"""
        gate = EntryGate(config={})
        assert gate.get_losses_today() == 0

        gate.record_loss()
        assert gate.get_losses_today() == 1

        gate.record_loss()
        assert gate.get_losses_today() == 2

    def test_record_loss_with_timestamp(self):
        """Should record loss with specific timestamp"""
        gate = EntryGate(config={})
        ts = datetime(2025, 12, 20, 10, 0, 0)

        gate.record_loss(timestamp=ts)
        assert gate.get_losses_today(timestamp=ts) == 1


class TestDailyLossGuard:
    """Test daily loss guard functionality"""

    def test_blocks_entries_after_max_losses(self):
        """Should block all entries after max losing trades"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 10,
            'enable_time_filter': False,
            'daily_loss_guard': {
                'enabled': True,
                'max_losing_trades_per_day': 2
            }
        })

        # First entry should be allowed
        allowed, _ = gate.check_entry_allowed('AAPL')
        assert allowed is True

        # Record max losses
        gate.record_loss()
        gate.record_loss()

        # Entry should now be blocked for all symbols
        allowed, reason = gate.check_entry_allowed('AAPL')
        assert allowed is False
        assert 'daily_loss_limit' in reason

        allowed_nvda, _ = gate.check_entry_allowed('NVDA')
        assert allowed_nvda is False

    def test_daily_loss_guard_disabled(self):
        """Should not block entries when loss guard is disabled"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 10,
            'enable_time_filter': False,
            'daily_loss_guard': {
                'enabled': False,
                'max_losing_trades_per_day': 1
            }
        })

        # Record many losses
        for _ in range(5):
            gate.record_loss()

        # Entry should still be allowed (guard disabled)
        allowed, _ = gate.check_entry_allowed('AAPL')
        assert allowed is True

    def test_is_daily_loss_limit_reached(self):
        """Should report when daily loss limit is reached"""
        gate = EntryGate(config={
            'daily_loss_guard': {
                'enabled': True,
                'max_losing_trades_per_day': 2
            }
        })

        assert gate.is_daily_loss_limit_reached() is False

        gate.record_loss()
        assert gate.is_daily_loss_limit_reached() is False

        gate.record_loss()
        assert gate.is_daily_loss_limit_reached() is True

    def test_is_daily_loss_limit_reached_disabled(self):
        """Should always return False when loss guard disabled"""
        gate = EntryGate(config={
            'daily_loss_guard': {
                'enabled': False,
                'max_losing_trades_per_day': 1
            }
        })

        gate.record_loss()
        gate.record_loss()
        gate.record_loss()

        assert gate.is_daily_loss_limit_reached() is False


class TestTimeOfDayFilter:
    """Test time-of-day filtering"""

    def test_time_filter_blocks_outside_sessions(self):
        """Should block entries outside allowed sessions when filter enabled"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 10,
            'enable_time_filter': True,
            'allowed_sessions': 'OPEN_AND_POWER',
            'daily_loss_guard': {'enabled': False}
        })

        # Midday - outside both open and power hour sessions
        tz = pytz.timezone('America/New_York')
        midday = tz.localize(datetime(2025, 12, 20, 12, 30, 0))

        allowed, reason = gate.check_entry_allowed('AAPL', timestamp=midday)
        assert allowed is False
        assert 'time_filter' in reason

    def test_time_filter_allows_during_open_session(self):
        """Should allow entries during open session (9:30-11:00 AM)"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 10,
            'enable_time_filter': True,
            'allowed_sessions': 'OPEN_AND_POWER',
            'daily_loss_guard': {'enabled': False}
        })

        tz = pytz.timezone('America/New_York')
        open_time = tz.localize(datetime(2025, 12, 20, 10, 0, 0))

        allowed, _ = gate.check_entry_allowed('AAPL', timestamp=open_time)
        assert allowed is True

    def test_time_filter_allows_during_power_hour(self):
        """Should allow entries during power hour (2:30-4:00 PM)"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 10,
            'enable_time_filter': True,
            'allowed_sessions': 'OPEN_AND_POWER',
            'daily_loss_guard': {'enabled': False}
        })

        tz = pytz.timezone('America/New_York')
        power_time = tz.localize(datetime(2025, 12, 20, 15, 0, 0))

        allowed, _ = gate.check_entry_allowed('AAPL', timestamp=power_time)
        assert allowed is True

    def test_time_filter_open_only(self):
        """Should only allow during open session when OPEN_ONLY"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 10,
            'enable_time_filter': True,
            'allowed_sessions': 'OPEN_ONLY',
            'daily_loss_guard': {'enabled': False}
        })

        tz = pytz.timezone('America/New_York')
        open_time = tz.localize(datetime(2025, 12, 20, 10, 0, 0))
        power_time = tz.localize(datetime(2025, 12, 20, 15, 0, 0))

        allowed_open, _ = gate.check_entry_allowed('AAPL', timestamp=open_time)
        allowed_power, _ = gate.check_entry_allowed('AAPL', timestamp=power_time)

        assert allowed_open is True
        assert allowed_power is False

    def test_time_filter_power_hour_only(self):
        """Should only allow during power hour when POWER_HOUR_ONLY"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 10,
            'enable_time_filter': True,
            'allowed_sessions': 'POWER_HOUR_ONLY',
            'daily_loss_guard': {'enabled': False}
        })

        tz = pytz.timezone('America/New_York')
        open_time = tz.localize(datetime(2025, 12, 20, 10, 0, 0))
        power_time = tz.localize(datetime(2025, 12, 20, 15, 0, 0))

        allowed_open, _ = gate.check_entry_allowed('AAPL', timestamp=open_time)
        allowed_power, _ = gate.check_entry_allowed('AAPL', timestamp=power_time)

        assert allowed_open is False
        assert allowed_power is True

    def test_time_filter_disabled(self):
        """Should allow entries at any time when filter disabled"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 10,
            'enable_time_filter': False,
            'daily_loss_guard': {'enabled': False}
        })

        tz = pytz.timezone('America/New_York')
        midday = tz.localize(datetime(2025, 12, 20, 12, 30, 0))

        allowed, _ = gate.check_entry_allowed('AAPL', timestamp=midday)
        assert allowed is True


class TestGetTradesToday:
    """Test get_trades_today() method"""

    def test_returns_zero_for_no_trades(self):
        """Should return 0 when no trades made"""
        gate = EntryGate(config={})
        assert gate.get_trades_today('AAPL') == 0

    def test_returns_correct_count(self):
        """Should return correct trade count"""
        gate = EntryGate(config={})
        gate.record_entry('AAPL')
        gate.record_entry('AAPL')
        gate.record_entry('NVDA')

        assert gate.get_trades_today('AAPL') == 2
        assert gate.get_trades_today('NVDA') == 1
        assert gate.get_trades_today('TSLA') == 0


class TestGetLossesToday:
    """Test get_losses_today() method"""

    def test_returns_zero_for_no_losses(self):
        """Should return 0 when no losses recorded"""
        gate = EntryGate(config={})
        assert gate.get_losses_today() == 0

    def test_returns_correct_count(self):
        """Should return correct loss count"""
        gate = EntryGate(config={})
        gate.record_loss()
        gate.record_loss()
        gate.record_loss()

        assert gate.get_losses_today() == 3


class TestGetRemainingTrades:
    """Test get_remaining_trades() method"""

    def test_returns_max_when_no_trades(self):
        """Should return max trades when none made"""
        gate = EntryGate(config={'max_trades_per_symbol_per_day': 5})
        assert gate.get_remaining_trades('AAPL') == 5

    def test_returns_correct_remaining(self):
        """Should return correct remaining trades"""
        gate = EntryGate(config={'max_trades_per_symbol_per_day': 5})
        gate.record_entry('AAPL')
        gate.record_entry('AAPL')

        assert gate.get_remaining_trades('AAPL') == 3

    def test_returns_zero_when_limit_reached(self):
        """Should return 0 when limit reached"""
        gate = EntryGate(config={'max_trades_per_symbol_per_day': 2})
        gate.record_entry('AAPL')
        gate.record_entry('AAPL')

        assert gate.get_remaining_trades('AAPL') == 0


class TestReset:
    """Test reset() method"""

    def test_reset_clears_all_counts(self):
        """Should clear all trade and loss counts"""
        gate = EntryGate(config={})

        # Record some activity
        gate.record_entry('AAPL')
        gate.record_entry('NVDA')
        gate.record_loss()
        gate.record_loss()

        # Verify activity recorded
        assert gate.get_trades_today('AAPL') == 1
        assert gate.get_losses_today() == 2

        # Reset
        gate.reset()

        # Verify all cleared
        assert gate.get_trades_today('AAPL') == 0
        assert gate.get_trades_today('NVDA') == 0
        assert gate.get_losses_today() == 0
        assert len(gate.trades_today) == 0
        assert len(gate.losses_today) == 0
        assert gate._current_trading_day is None


class TestGetStatus:
    """Test get_status() method"""

    def test_get_status_returns_complete_state(self):
        """Should return complete status dict"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 3,
            'enable_time_filter': True,
            'allowed_sessions': 'POWER_HOUR_ONLY',
            'timezone': 'America/New_York',
            'daily_loss_guard': {
                'enabled': True,
                'max_losing_trades_per_day': 5
            }
        })

        # Record some activity
        gate.record_entry('AAPL')
        gate.record_loss()

        status = gate.get_status()

        assert status['max_trades_per_symbol_per_day'] == 3
        assert status['enable_time_filter'] is True
        assert status['allowed_sessions'] == 'POWER_HOUR_ONLY'
        assert status['timezone'] == 'America/New_York'
        assert status['daily_loss_guard_enabled'] is True
        assert status['max_losing_trades_per_day'] == 5
        assert status['losses_today'] == 1
        assert status['daily_loss_limit_reached'] is False
        assert 'active_counts' in status
        assert 'current_trading_day' in status


class TestDayRollover:
    """Test day rollover resets counts"""

    def test_day_rollover_resets_trade_counts(self):
        """Should reset trade counts when day changes"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 2,
            'enable_time_filter': False,
            'daily_loss_guard': {'enabled': False}
        })

        tz = pytz.timezone('America/New_York')
        day1 = tz.localize(datetime(2025, 12, 20, 10, 0, 0))
        day2 = tz.localize(datetime(2025, 12, 21, 10, 0, 0))

        # Record max trades on day 1
        gate.record_entry('AAPL', timestamp=day1)
        gate.record_entry('AAPL', timestamp=day1)

        # Should be blocked on day 1
        allowed_day1, _ = gate.check_entry_allowed('AAPL', timestamp=day1)
        assert allowed_day1 is False

        # Should be allowed on day 2
        allowed_day2, _ = gate.check_entry_allowed('AAPL', timestamp=day2)
        assert allowed_day2 is True

    def test_day_rollover_resets_loss_counts(self):
        """Should reset loss counts when day changes"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 10,
            'enable_time_filter': False,
            'daily_loss_guard': {
                'enabled': True,
                'max_losing_trades_per_day': 2
            }
        })

        tz = pytz.timezone('America/New_York')
        day1 = tz.localize(datetime(2025, 12, 20, 10, 0, 0))
        day2 = tz.localize(datetime(2025, 12, 21, 10, 0, 0))

        # Record max losses on day 1
        gate.record_loss(timestamp=day1)
        gate.record_loss(timestamp=day1)

        # Should be blocked on day 1
        allowed_day1, _ = gate.check_entry_allowed('AAPL', timestamp=day1)
        assert allowed_day1 is False

        # Should be allowed on day 2
        allowed_day2, _ = gate.check_entry_allowed('AAPL', timestamp=day2)
        assert allowed_day2 is True

    def test_day_rollover_clears_old_entries(self):
        """Should clear old day entries from memory on day change"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 2,
            'enable_time_filter': False,
            'daily_loss_guard': {'enabled': True, 'max_losing_trades_per_day': 2}
        })

        tz = pytz.timezone('America/New_York')
        day1 = tz.localize(datetime(2025, 12, 20, 10, 0, 0))
        day2 = tz.localize(datetime(2025, 12, 21, 10, 0, 0))

        # Record on day 1
        gate.record_entry('AAPL', timestamp=day1)
        gate.record_loss(timestamp=day1)

        # Trigger day rollover by checking day 2
        gate.check_entry_allowed('AAPL', timestamp=day2)

        # Old entries should be cleared
        assert gate.get_trades_today('AAPL', timestamp=day1) == 0
        assert gate.get_losses_today(timestamp=day1) == 0


class TestSessionBoundaries:
    """Test session boundary handling"""

    def test_open_session_boundaries(self):
        """Should correctly handle open session boundaries (9:30-11:00)"""
        gate = EntryGate(config={
            'enable_time_filter': True,
            'allowed_sessions': 'OPEN_ONLY',
            'daily_loss_guard': {'enabled': False}
        })

        tz = pytz.timezone('America/New_York')

        # Just before 9:30 - blocked
        before_open = tz.localize(datetime(2025, 12, 20, 9, 29, 0))
        allowed, _ = gate.check_entry_allowed('AAPL', timestamp=before_open)
        assert allowed is False

        # Exactly 9:30 - allowed
        at_open = tz.localize(datetime(2025, 12, 20, 9, 30, 0))
        allowed, _ = gate.check_entry_allowed('AAPL', timestamp=at_open)
        assert allowed is True

        # 10:59 - allowed
        before_close = tz.localize(datetime(2025, 12, 20, 10, 59, 0))
        allowed, _ = gate.check_entry_allowed('AAPL', timestamp=before_close)
        assert allowed is True

        # Exactly 11:00 - blocked (end is exclusive)
        at_close = tz.localize(datetime(2025, 12, 20, 11, 0, 0))
        allowed, _ = gate.check_entry_allowed('AAPL', timestamp=at_close)
        assert allowed is False

    def test_power_session_boundaries(self):
        """Should correctly handle power session boundaries (14:30-16:00)"""
        gate = EntryGate(config={
            'enable_time_filter': True,
            'allowed_sessions': 'POWER_HOUR_ONLY',
            'daily_loss_guard': {'enabled': False}
        })

        tz = pytz.timezone('America/New_York')

        # Just before 14:30 - blocked
        before_power = tz.localize(datetime(2025, 12, 20, 14, 29, 0))
        allowed, _ = gate.check_entry_allowed('AAPL', timestamp=before_power)
        assert allowed is False

        # Exactly 14:30 - allowed
        at_power = tz.localize(datetime(2025, 12, 20, 14, 30, 0))
        allowed, _ = gate.check_entry_allowed('AAPL', timestamp=at_power)
        assert allowed is True

        # 15:59 - allowed
        before_close = tz.localize(datetime(2025, 12, 20, 15, 59, 0))
        allowed, _ = gate.check_entry_allowed('AAPL', timestamp=before_close)
        assert allowed is True

        # Exactly 16:00 - blocked (end is exclusive)
        at_close = tz.localize(datetime(2025, 12, 20, 16, 0, 0))
        allowed, _ = gate.check_entry_allowed('AAPL', timestamp=at_close)
        assert allowed is False


class TestPriorityOfChecks:
    """Test that checks are applied in correct priority order"""

    def test_loss_guard_checked_before_trade_limit(self):
        """Loss guard should block before per-symbol limit is checked"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 5,
            'enable_time_filter': False,
            'daily_loss_guard': {
                'enabled': True,
                'max_losing_trades_per_day': 1
            }
        })

        # Record loss (blocks all entries)
        gate.record_loss()

        # Entry should be blocked due to loss limit, not trade limit
        allowed, reason = gate.check_entry_allowed('AAPL')
        assert allowed is False
        assert 'daily_loss_limit' in reason
        assert 'max_trades_per_day' not in reason

    def test_trade_limit_checked_before_time_filter(self):
        """Per-symbol limit should block before time filter is checked"""
        gate = EntryGate(config={
            'max_trades_per_symbol_per_day': 1,
            'enable_time_filter': True,
            'allowed_sessions': 'OPEN_ONLY',
            'daily_loss_guard': {'enabled': False}
        })

        tz = pytz.timezone('America/New_York')
        open_time = tz.localize(datetime(2025, 12, 20, 10, 0, 0))

        # Record max trades during open session
        gate.record_entry('AAPL', timestamp=open_time)

        # Entry should be blocked due to trade limit, not time filter
        allowed, reason = gate.check_entry_allowed('AAPL', timestamp=open_time)
        assert allowed is False
        assert 'max_trades_per_day' in reason
        assert 'time_filter' not in reason


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
