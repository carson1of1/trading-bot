"""
Tests for LosingStreakGuard

Tests the multi-day losing streak protection system:
- Trigger: 2+ losing trades (≤-0.5R) within 3 days
- Reset: Green day (net positive realized P&L)
- Effect: Position sizes reduced to 50%
"""

import pytest
from datetime import datetime, date, timedelta

from core import LosingStreakGuard, TradeResult, create_losing_streak_guard


class TestTradeResult:
    """Test TradeResult dataclass."""

    def test_trade_result_is_loser_true(self):
        """Test that trade with pnl <= -0.5R is marked as loser."""
        result = TradeResult(
            symbol='AAPL',
            close_time=datetime.now(),
            realized_pnl=-30.0,  # Lost $30
            risk_amount=50.0     # Risked $50
        )
        # -30 <= -25 (0.5 * 50), so is_losing_trade should be True
        assert result.is_losing_trade is True

    def test_trade_result_is_loser_false(self):
        """Test that scratch trade (pnl > -0.5R) is not a loser."""
        result = TradeResult(
            symbol='AAPL',
            close_time=datetime.now(),
            realized_pnl=-10.0,  # Lost only $10
            risk_amount=50.0     # Risked $50
        )
        # -10 > -25 (0.5 * 50), so is_losing_trade should be False
        assert result.is_losing_trade is False

    def test_trade_result_winner_not_loser(self):
        """Test that winning trade is not a loser."""
        result = TradeResult(
            symbol='AAPL',
            close_time=datetime.now(),
            realized_pnl=25.0,   # Won $25
            risk_amount=50.0     # Risked $50
        )
        assert result.is_losing_trade is False

    def test_trade_result_exact_threshold(self):
        """Test trade at exactly -0.5R threshold."""
        result = TradeResult(
            symbol='AAPL',
            close_time=datetime.now(),
            realized_pnl=-25.0,  # Exactly -0.5R
            risk_amount=50.0
        )
        # -25 <= -25, so is_losing_trade should be True
        assert result.is_losing_trade is True


class TestLosingStreakGuardInit:
    """Test LosingStreakGuard initialization."""

    def test_default_config(self):
        """Test initialization with default config."""
        guard = LosingStreakGuard()

        assert guard.enabled is True
        assert guard.losing_threshold_r == 0.5
        assert guard.lookback_days == 3
        assert guard.min_losing_trades == 2
        assert guard.throttle_multiplier == 0.5
        assert guard.position_size_multiplier == 1.0
        assert guard.is_throttled is False

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = {
            'losing_streak_guard': {
                'enabled': True,
                'losing_threshold_r': 0.75,
                'lookback_days': 5,
                'min_losing_trades': 3,
                'throttle_multiplier': 0.25
            }
        }
        guard = LosingStreakGuard(config)

        assert guard.losing_threshold_r == 0.75
        assert guard.lookback_days == 5
        assert guard.min_losing_trades == 3
        assert guard.throttle_multiplier == 0.25

    def test_disabled_config(self):
        """Test initialization with guard disabled."""
        config = {'losing_streak_guard': {'enabled': False}}
        guard = LosingStreakGuard(config)

        assert guard.enabled is False
        assert guard.position_size_multiplier == 1.0


class TestRecordTrade:
    """Test record_trade method."""

    def test_record_single_trade(self):
        """Test recording a single trade."""
        guard = LosingStreakGuard()

        guard.record_trade(
            symbol='AAPL',
            realized_pnl=-30.0,
            risk_amount=50.0,
            close_time=datetime.now()
        )

        assert len(guard._trade_history) == 1
        assert guard._trade_history[0].symbol == 'AAPL'
        assert guard._trade_history[0].is_losing_trade is True

    def test_one_loser_stays_normal(self):
        """Test that single losing trade doesn't trigger throttle."""
        guard = LosingStreakGuard()

        guard.record_trade('AAPL', -30.0, 50.0, datetime.now())

        assert guard.is_throttled is False
        assert guard.position_size_multiplier == 1.0

    def test_two_losers_triggers_throttle(self):
        """Test that 2 losing trades in 3 days triggers throttle."""
        guard = LosingStreakGuard()
        now = datetime.now()

        # Two losing trades same day
        guard.record_trade('AAPL', -30.0, 50.0, now)
        guard.record_trade('MSFT', -40.0, 50.0, now)

        assert guard.is_throttled is True
        assert guard.position_size_multiplier == 0.5

    def test_scratch_trade_not_counted(self):
        """Test that scratch trades don't count toward streak."""
        guard = LosingStreakGuard()
        now = datetime.now()

        # One real loser, one scratch
        guard.record_trade('AAPL', -30.0, 50.0, now)  # Real loser (-30 <= -25)
        guard.record_trade('MSFT', -10.0, 50.0, now)  # Scratch (-10 > -25)

        assert guard.is_throttled is False  # Only 1 real loser

    def test_losers_outside_window_ignored(self):
        """Test that old losing trades are not counted."""
        guard = LosingStreakGuard()
        now = datetime.now()
        old = now - timedelta(days=5)  # Outside 3-day window

        guard.record_trade('AAPL', -30.0, 50.0, old)  # Old loser
        guard.record_trade('MSFT', -30.0, 50.0, now)  # Recent loser

        assert guard.is_throttled is False  # Only 1 in window

    def test_disabled_guard_ignores_trades(self):
        """Test that disabled guard doesn't track trades."""
        config = {'losing_streak_guard': {'enabled': False}}
        guard = LosingStreakGuard(config)

        guard.record_trade('AAPL', -30.0, 50.0, datetime.now())
        guard.record_trade('MSFT', -30.0, 50.0, datetime.now())

        assert guard.is_throttled is False
        assert len(guard._trade_history) == 0


class TestEndOfDay:
    """Test end_of_day reset logic."""

    def test_green_day_resets_throttle(self):
        """Test that green day resets throttle to normal."""
        guard = LosingStreakGuard()
        now = datetime.now()
        today = now.date()

        # Trigger throttle
        guard.record_trade('AAPL', -30.0, 50.0, now)
        guard.record_trade('MSFT', -30.0, 50.0, now)
        assert guard.is_throttled is True

        # Simulate winning trade that makes day green
        guard.record_trade('GOOGL', 100.0, 50.0, now)

        # End of day - should reset (net P&L = -30 -30 +100 = +40)
        guard.end_of_day(today)

        assert guard.is_throttled is False
        assert guard.position_size_multiplier == 1.0

    def test_red_day_stays_throttled(self):
        """Test that red day keeps throttle active."""
        guard = LosingStreakGuard()
        now = datetime.now()
        today = now.date()

        # Trigger throttle
        guard.record_trade('AAPL', -30.0, 50.0, now)
        guard.record_trade('MSFT', -30.0, 50.0, now)
        assert guard.is_throttled is True

        # End of day - still red (net P&L = -60)
        guard.end_of_day(today)

        assert guard.is_throttled is True

    def test_no_trades_today_no_reset(self):
        """Test that day with no trades doesn't reset."""
        guard = LosingStreakGuard()
        now = datetime.now()
        today = now.date()
        tomorrow = today + timedelta(days=1)

        # Trigger throttle today
        guard.record_trade('AAPL', -30.0, 50.0, now)
        guard.record_trade('MSFT', -30.0, 50.0, now)
        assert guard.is_throttled is True

        # End tomorrow with no trades
        guard.end_of_day(tomorrow)

        assert guard.is_throttled is True  # No trades = no green day

    def test_prunes_old_trades(self):
        """Test that old trades are pruned."""
        guard = LosingStreakGuard()
        now = datetime.now()
        old = now - timedelta(days=10)

        # Add old trade
        guard.record_trade('AAPL', -30.0, 50.0, old)
        assert len(guard._trade_history) == 1

        # End of day triggers pruning
        guard.end_of_day(now.date())

        assert len(guard._trade_history) == 0  # Old trade pruned


class TestGetStatus:
    """Test get_status method."""

    def test_status_contains_all_fields(self):
        """Test that get_status returns all expected fields."""
        guard = LosingStreakGuard()
        guard.record_trade('AAPL', -30.0, 50.0, datetime.now())

        status = guard.get_status()

        assert 'enabled' in status
        assert 'is_throttled' in status
        assert 'position_size_multiplier' in status
        assert 'recent_losers' in status
        assert 'lookback_days' in status
        assert 'min_losing_trades' in status

    def test_status_values_correct(self):
        """Test that get_status returns correct values."""
        guard = LosingStreakGuard()
        now = datetime.now()

        guard.record_trade('AAPL', -30.0, 50.0, now)
        guard.record_trade('MSFT', -30.0, 50.0, now)

        status = guard.get_status()

        assert status['is_throttled'] is True
        assert status['position_size_multiplier'] == 0.5
        assert status['recent_losers'] == 2


class TestFactoryFunction:
    """Test create_losing_streak_guard factory."""

    def test_factory_creates_guard(self):
        """Test factory function creates configured guard."""
        config = {
            'losing_streak_guard': {
                'enabled': True,
                'throttle_multiplier': 0.25
            }
        }
        guard = create_losing_streak_guard(config)

        assert isinstance(guard, LosingStreakGuard)
        assert guard.throttle_multiplier == 0.25


class TestStackingWithDrawdownGuard:
    """Test that guards can stack multipliers."""

    def test_both_guards_multiply(self):
        """Test that both guards reduce position size."""
        # LosingStreakGuard at 0.5
        streak_guard = LosingStreakGuard()
        now = datetime.now()
        streak_guard.record_trade('AAPL', -30.0, 50.0, now)
        streak_guard.record_trade('MSFT', -30.0, 50.0, now)

        assert streak_guard.position_size_multiplier == 0.5

        # In real usage, bot would multiply:
        # final_multiplier = drawdown_guard.multiplier * streak_guard.multiplier
        # e.g., 0.5 * 0.5 = 0.25
