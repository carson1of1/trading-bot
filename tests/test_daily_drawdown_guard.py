"""
Tests for DailyDrawdownGuard

Tests the tiered daily drawdown protection system:
- Threshold detection at each tier (NORMAL, WARNING, SOFT_LIMIT, HARD_LIMIT)
- Hysteresis (thresholds stay triggered for the day, no recovery)
- Liquidation triggers at -4%
- Position size reduction at warning level
- Entry blocking at soft limit
"""

import pytest
from datetime import datetime, date
from unittest.mock import MagicMock, patch

from core.risk import DailyDrawdownGuard, DrawdownTier


class MockAccount:
    """Mock broker account for testing."""

    def __init__(self, equity: float):
        self.equity = equity
        self.portfolio_value = equity


class TestDailyDrawdownGuardInit:
    """Test DailyDrawdownGuard initialization."""

    def test_default_config(self):
        """Test initialization with default config.

        ODE-90: Updated defaults for funded account protection (3%/4%/5%)
        """
        guard = DailyDrawdownGuard()

        assert guard.enabled is True
        assert guard.warning_pct == 0.025  # 2.5%
        assert guard.soft_limit_pct == 0.03  # 3.0% (ODE-90: was 3.5%)
        assert guard.medium_limit_pct == 0.04  # 4.0% (ODE-90: new tier)
        assert guard.hard_limit_pct == 0.05  # 5.0% (ODE-90: was 4.0%)
        assert guard.warning_size_multiplier == 0.5
        assert guard.entries_allowed is True
        assert guard.position_size_multiplier == 1.0
        assert guard.day_halted is False

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = {
            'daily_drawdown_guard': {
                'enabled': True,
                'warning_pct': 2.0,
                'soft_limit_pct': 3.0,
                'hard_limit_pct': 3.5,
                'warning_size_multiplier': 0.75
            }
        }
        guard = DailyDrawdownGuard(config)

        assert guard.warning_pct == 0.02  # 2.0%
        assert guard.soft_limit_pct == 0.03  # 3.0%
        assert guard.hard_limit_pct == 0.035  # 3.5%
        assert guard.warning_size_multiplier == 0.75

    def test_disabled_config(self):
        """Test initialization with guard disabled."""
        config = {
            'daily_drawdown_guard': {
                'enabled': False
            }
        }
        guard = DailyDrawdownGuard(config)

        assert guard.enabled is False


class TestThresholdDetection:
    """Test threshold detection at each tier."""

    def test_normal_tier(self):
        """Test NORMAL tier when no drawdown."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # No drawdown
        tier = guard.update_equity(MockAccount(10000.0))

        assert tier == DrawdownTier.NORMAL
        assert guard.entries_allowed is True
        assert guard.position_size_multiplier == 1.0
        assert guard.day_halted is False

    def test_normal_with_small_loss(self):
        """Test NORMAL tier with small loss below warning."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # 2% loss (below 2.5% warning)
        tier = guard.update_equity(MockAccount(9800.0))

        assert tier == DrawdownTier.NORMAL
        assert guard.drawdown_pct == 0.02
        assert guard.entries_allowed is True
        assert guard.position_size_multiplier == 1.0

    def test_warning_tier(self):
        """Test WARNING tier at -2.5% drawdown."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # 2.5% loss (at warning threshold)
        tier = guard.update_equity(MockAccount(9750.0))

        assert tier == DrawdownTier.WARNING
        assert guard.drawdown_pct == 0.025
        assert guard.entries_allowed is True
        assert guard.position_size_multiplier == 0.5
        assert guard.day_halted is False

    def test_warning_tier_deep(self):
        """Test WARNING tier at -2.8% drawdown (above warning, below soft).

        ODE-90: Updated for new thresholds (3%/4%/5%).
        2.8% is above 2.5% warning but below 3% soft limit.
        """
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # 2.8% loss (above warning, below soft limit)
        tier = guard.update_equity(MockAccount(9720.0))

        assert tier == DrawdownTier.WARNING
        assert guard.entries_allowed is True
        assert guard.position_size_multiplier == 0.5

    def test_soft_limit_tier(self):
        """Test SOFT_LIMIT tier at -3% drawdown.

        ODE-90: Updated for funded account thresholds (3%/4%/5%).
        """
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # 3% loss (at soft limit)
        tier = guard.update_equity(MockAccount(9700.0))

        assert tier == DrawdownTier.SOFT_LIMIT
        assert guard.drawdown_pct == 0.03
        assert guard.entries_allowed is False
        assert guard.position_size_multiplier == 0.0
        assert guard.day_halted is False

    def test_hard_limit_tier(self):
        """Test HARD_LIMIT tier at -5% drawdown.

        ODE-90: Updated for funded account thresholds (3%/4%/5%).
        4% now triggers MEDIUM tier, 5% triggers HARD_LIMIT.
        """
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # 5% loss (at hard limit)
        tier = guard.update_equity(MockAccount(9500.0))

        assert tier == DrawdownTier.HARD_LIMIT
        assert guard.drawdown_pct == 0.05
        assert guard.entries_allowed is False
        assert guard.position_size_multiplier == 0.0
        assert guard.day_halted is True

    def test_hard_limit_extreme(self):
        """Test HARD_LIMIT tier at extreme drawdown (-6%).

        ODE-90: Updated threshold to 6% (beyond 5% hard limit).
        """
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # 6% loss (beyond hard limit)
        tier = guard.update_equity(MockAccount(9400.0))

        assert tier == DrawdownTier.HARD_LIMIT
        assert guard.day_halted is True


class TestHysteresis:
    """Test hysteresis - thresholds stay triggered for the day."""

    def test_warning_hysteresis(self):
        """Test that WARNING tier persists even if equity recovers."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # Drop to warning level
        tier1 = guard.update_equity(MockAccount(9750.0))
        assert tier1 == DrawdownTier.WARNING
        assert guard.position_size_multiplier == 0.5

        # Recover above warning (but hysteresis keeps warning active)
        tier2 = guard.update_equity(MockAccount(9900.0))
        assert tier2 == DrawdownTier.WARNING  # Still WARNING due to hysteresis
        assert guard.position_size_multiplier == 0.5

        # Even full recovery keeps WARNING
        tier3 = guard.update_equity(MockAccount(10200.0))
        assert tier3 == DrawdownTier.WARNING  # Still WARNING
        assert guard.position_size_multiplier == 0.5

    def test_soft_limit_hysteresis(self):
        """Test that SOFT_LIMIT tier persists even if equity recovers.

        ODE-90: Updated for funded account thresholds (3% soft limit).
        """
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # Drop to soft limit (3%)
        tier1 = guard.update_equity(MockAccount(9700.0))
        assert tier1 == DrawdownTier.SOFT_LIMIT
        assert guard.entries_allowed is False

        # Recover to warning level (hysteresis keeps soft limit)
        tier2 = guard.update_equity(MockAccount(9750.0))
        assert tier2 == DrawdownTier.SOFT_LIMIT  # Still SOFT due to hysteresis
        assert guard.entries_allowed is False

        # Full recovery still keeps SOFT_LIMIT
        tier3 = guard.update_equity(MockAccount(10500.0))
        assert tier3 == DrawdownTier.SOFT_LIMIT
        assert guard.entries_allowed is False

    def test_hard_limit_hysteresis(self):
        """Test that HARD_LIMIT tier persists - no recovery possible.

        ODE-90: Updated for funded account thresholds (5% hard limit).
        """
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # Drop to hard limit (5%)
        tier1 = guard.update_equity(MockAccount(9500.0))
        assert tier1 == DrawdownTier.HARD_LIMIT
        assert guard.day_halted is True

        # Any recovery is irrelevant - day is done
        tier2 = guard.update_equity(MockAccount(11000.0))
        assert tier2 == DrawdownTier.HARD_LIMIT
        assert guard.day_halted is True

    def test_hysteresis_resets_on_new_day(self):
        """Test that hysteresis resets on new trading day via reset_day call.

        ODE-90: Updated for funded account thresholds (3% soft limit).
        """
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # Drop to soft limit (3%)
        tier1 = guard.update_equity(MockAccount(9700.0))
        assert tier1 == DrawdownTier.SOFT_LIMIT

        # Manually reset for new day (simulating bot's day detection)
        guard.reset_day(9700.0)

        # After reset, guard should be NORMAL until new drawdown
        assert guard.tier == DrawdownTier.NORMAL
        assert guard.entries_allowed is True

        # Update with same equity - no drawdown from new starting point
        tier2 = guard.update_equity(MockAccount(9700.0))
        assert tier2 == DrawdownTier.NORMAL
        assert guard.entries_allowed is True


class TestLiquidation:
    """Test liquidation triggers at -5% (ODE-90: updated from -4%)."""

    def test_liquidation_at_hard_limit(self):
        """Test that liquidation is triggered at hard limit.

        ODE-90: Updated for funded account thresholds (5% hard limit).
        """
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # Drop to hard limit (5%)
        tier = guard.update_equity(MockAccount(9500.0))

        assert tier == DrawdownTier.HARD_LIMIT
        assert guard.day_halted is True
        # In real usage, bot would call force_liquidate_all()

    def test_force_liquidate_all(self):
        """Test force_liquidate_all method."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # Create mock broker
        mock_broker = MagicMock()
        mock_order = MagicMock()
        mock_order.status = 'filled'
        mock_order.filled_avg_price = 100.0
        mock_broker.submit_order.return_value = mock_order

        # Create mock positions
        positions = [
            {'symbol': 'AAPL', 'qty': 10, 'direction': 'LONG'},
            {'symbol': 'MSFT', 'qty': 5, 'direction': 'SHORT'}
        ]

        # Trigger liquidation
        result = guard.force_liquidate_all(mock_broker, positions)

        assert result['success'] is True
        assert len(result['liquidated']) == 2
        assert mock_broker.submit_order.call_count == 2

    def test_double_liquidation_prevention(self):
        """Test that double liquidation is prevented."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # Simulate liquidation in progress
        guard._liquidation_in_progress = True

        mock_broker = MagicMock()
        positions = [{'symbol': 'AAPL', 'qty': 10, 'direction': 'LONG'}]

        result = guard.force_liquidate_all(mock_broker, positions)

        assert result['success'] is False
        assert result['reason'] == 'already_in_progress'
        mock_broker.submit_order.assert_not_called()


class TestPositionSizeReduction:
    """Test position size reduction at warning level."""

    def test_normal_position_size(self):
        """Test normal position size multiplier at NORMAL tier."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        guard.update_equity(MockAccount(10000.0))

        assert guard.position_size_multiplier == 1.0

    def test_warning_position_size(self):
        """Test reduced position size at WARNING tier."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # Drop to warning level
        guard.update_equity(MockAccount(9750.0))

        assert guard.position_size_multiplier == 0.5  # Default 50% reduction

    def test_custom_warning_size_multiplier(self):
        """Test custom warning size multiplier."""
        config = {
            'daily_drawdown_guard': {
                'warning_size_multiplier': 0.25
            }
        }
        guard = DailyDrawdownGuard(config)
        guard.reset_day(10000.0)

        # Drop to warning level
        guard.update_equity(MockAccount(9750.0))

        assert guard.position_size_multiplier == 0.25

    def test_soft_limit_blocks_positions(self):
        """Test that soft limit blocks new positions entirely.

        ODE-90: Updated for funded account thresholds (3% soft limit).
        """
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # Drop to soft limit (3%)
        guard.update_equity(MockAccount(9700.0))

        assert guard.position_size_multiplier == 0.0  # No new positions


class TestEntryBlocking:
    """Test entry blocking at soft limit."""

    def test_entries_allowed_at_normal(self):
        """Test that entries are allowed at NORMAL tier."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        guard.update_equity(MockAccount(10000.0))

        assert guard.entries_allowed is True

    def test_entries_allowed_at_warning(self):
        """Test that entries are allowed at WARNING tier (with reduced size)."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        guard.update_equity(MockAccount(9750.0))

        assert guard.entries_allowed is True
        assert guard.position_size_multiplier == 0.5

    def test_entries_blocked_at_soft_limit(self):
        """Test that entries are blocked at SOFT_LIMIT tier.

        ODE-90: Updated for funded account thresholds (3% soft limit).
        """
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        guard.update_equity(MockAccount(9700.0))

        assert guard.entries_allowed is False
        assert guard.tier == DrawdownTier.SOFT_LIMIT

    def test_entries_blocked_at_hard_limit(self):
        """Test that entries are blocked at HARD_LIMIT tier.

        ODE-90: Updated for funded account thresholds (5% hard limit).
        """
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        guard.update_equity(MockAccount(9500.0))

        assert guard.entries_allowed is False
        assert guard.tier == DrawdownTier.HARD_LIMIT


class TestDayReset:
    """Test day reset functionality."""

    def test_reset_day_clears_state(self):
        """Test that reset_day clears all state.

        ODE-90: Updated for funded account thresholds (3% soft limit).
        """
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # Trigger soft limit (3%)
        guard.update_equity(MockAccount(9700.0))
        assert guard.tier == DrawdownTier.SOFT_LIMIT
        assert guard.entries_allowed is False

        # Reset for new day
        guard.reset_day(9700.0, date(2026, 1, 4))

        assert guard.day_start_equity == 9700.0
        assert guard.tier == DrawdownTier.NORMAL
        assert guard.entries_allowed is True
        assert guard.position_size_multiplier == 1.0
        assert guard.day_halted is False

    def test_realized_pnl_tracking(self):
        """Test realized P&L tracking."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        guard.record_realized_pnl(-50.0)
        guard.record_realized_pnl(-100.0)
        guard.record_realized_pnl(25.0)

        assert guard.realized_pnl_today == -125.0

    def test_realized_pnl_resets_on_new_day(self):
        """Test that realized P&L resets on new day."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0, date(2026, 1, 3))

        guard.record_realized_pnl(-100.0)
        assert guard.realized_pnl_today == -100.0

        guard.reset_day(9900.0, date(2026, 1, 4))
        assert guard.realized_pnl_today == 0.0


class TestDisabledGuard:
    """Test behavior when guard is disabled."""

    def test_disabled_always_returns_normal(self):
        """Test that disabled guard always returns NORMAL."""
        config = {'daily_drawdown_guard': {'enabled': False}}
        guard = DailyDrawdownGuard(config)
        guard.reset_day(10000.0)

        # Even with catastrophic loss, returns NORMAL when disabled
        tier = guard.update_equity(MockAccount(5000.0))

        assert tier == DrawdownTier.NORMAL
        assert guard.entries_allowed is True
        assert guard.position_size_multiplier == 1.0

    def test_disabled_never_halts_day(self):
        """Test that disabled guard never halts the day."""
        config = {'daily_drawdown_guard': {'enabled': False}}
        guard = DailyDrawdownGuard(config)
        guard.reset_day(10000.0)

        guard.update_equity(MockAccount(5000.0))

        assert guard.day_halted is False


class TestGetStatus:
    """Test get_status method."""

    def test_status_contains_all_fields(self):
        """Test that get_status returns all expected fields."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)
        guard.update_equity(MockAccount(9750.0))

        status = guard.get_status()

        assert 'enabled' in status
        assert 'tier' in status
        assert 'day_start_equity' in status
        assert 'current_equity' in status
        assert 'drawdown_pct' in status
        assert 'realized_pnl_today' in status
        assert 'entries_allowed' in status
        assert 'position_size_multiplier' in status
        assert 'day_halted' in status
        assert 'thresholds' in status

    def test_status_values_are_correct(self):
        """Test that get_status returns correct values."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)
        guard.update_equity(MockAccount(9750.0))

        status = guard.get_status()

        assert status['tier'] == 'WARNING'
        assert status['day_start_equity'] == 10000.0
        assert status['current_equity'] == 9750.0
        assert status['drawdown_pct'] == 2.5  # Percentage, not decimal
        assert status['entries_allowed'] is True
        assert status['position_size_multiplier'] == 0.5


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_starting_equity(self):
        """Test behavior with zero starting equity."""
        guard = DailyDrawdownGuard()
        guard.reset_day(0.0)

        tier = guard.update_equity(MockAccount(0.0))

        assert tier == DrawdownTier.NORMAL  # No division by zero

    def test_exact_threshold_values(self):
        """Test behavior at exact threshold boundaries.

        ODE-90: Updated for funded account thresholds (3%/4%/5%).
        """
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # Exactly at 2.5% (warning)
        tier = guard.update_equity(MockAccount(9750.0))
        assert tier == DrawdownTier.WARNING

        # Reset and test exactly at 3.0% (soft limit)
        guard.reset_day(10000.0)
        tier = guard.update_equity(MockAccount(9700.0))
        assert tier == DrawdownTier.SOFT_LIMIT

        # Reset and test exactly at 4.0% (medium - partial liquidation)
        guard.reset_day(10000.0)
        tier = guard.update_equity(MockAccount(9600.0))
        assert tier == DrawdownTier.MEDIUM

        # Reset and test exactly at 5.0% (hard limit)
        guard.reset_day(10000.0)
        tier = guard.update_equity(MockAccount(9500.0))
        assert tier == DrawdownTier.HARD_LIMIT

    def test_profit_does_not_trigger_tiers(self):
        """Test that profit (negative drawdown) stays NORMAL."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # Profit scenario
        tier = guard.update_equity(MockAccount(11000.0))

        assert tier == DrawdownTier.NORMAL
        assert guard.drawdown_pct < 0  # Negative drawdown = profit
        assert guard.entries_allowed is True


class TestFundedAccountProtection:
    """Test funded account protection features (ODE-90).

    Funded accounts (Apex, FTMO) have strict 5% daily drawdown limits.
    This tests the tiered protection:
    - 3% DD: Block new entries
    - 4% DD: Partial liquidation of existing positions (50%)
    - 5% DD: Full liquidation + halt
    """

    def test_funded_account_thresholds(self):
        """Test funded account specific thresholds (3%/4%/5%)."""
        config = {
            'daily_drawdown_guard': {
                'enabled': True,
                'soft_limit_pct': 3.0,    # Block entries at 3%
                'medium_limit_pct': 4.0,  # Partial liquidation at 4%
                'hard_limit_pct': 5.0,    # Full liquidation at 5%
            }
        }
        guard = DailyDrawdownGuard(config)
        guard.reset_day(10000.0)

        # At 3% - should block entries
        tier = guard.update_equity(MockAccount(9700.0))
        assert tier == DrawdownTier.SOFT_LIMIT
        assert guard.entries_allowed is False

    def test_medium_tier_exists(self):
        """Test that MEDIUM tier exists between SOFT_LIMIT and HARD_LIMIT."""
        # MEDIUM tier should exist in DrawdownTier enum
        assert hasattr(DrawdownTier, 'MEDIUM')
        assert DrawdownTier.SOFT_LIMIT.value < DrawdownTier.MEDIUM.value < DrawdownTier.HARD_LIMIT.value

    def test_medium_tier_triggers_partial_liquidation(self):
        """Test that MEDIUM tier triggers partial liquidation flag."""
        config = {
            'daily_drawdown_guard': {
                'enabled': True,
                'soft_limit_pct': 3.0,
                'medium_limit_pct': 4.0,
                'hard_limit_pct': 5.0,
            }
        }
        guard = DailyDrawdownGuard(config)
        guard.reset_day(10000.0)

        # At 4% - should trigger partial liquidation
        tier = guard.update_equity(MockAccount(9600.0))
        assert tier == DrawdownTier.MEDIUM
        assert guard.entries_allowed is False
        assert guard.partial_liquidation_triggered is True
        assert guard.day_halted is False  # Not halted yet

    def test_hard_limit_at_5_percent(self):
        """Test that hard limit triggers at 5% for funded accounts."""
        config = {
            'daily_drawdown_guard': {
                'enabled': True,
                'hard_limit_pct': 5.0,
            }
        }
        guard = DailyDrawdownGuard(config)
        guard.reset_day(10000.0)

        # At 5% - should trigger full liquidation
        tier = guard.update_equity(MockAccount(9500.0))
        assert tier == DrawdownTier.HARD_LIMIT
        assert guard.day_halted is True

    def test_partial_liquidation_method(self):
        """Test force_partial_liquidate closes 50% of positions."""
        config = {
            'daily_drawdown_guard': {
                'partial_liquidation_pct': 50,
            }
        }
        guard = DailyDrawdownGuard(config)
        guard.reset_day(10000.0)

        # Create mock broker
        mock_broker = MagicMock()
        mock_order = MagicMock()
        mock_order.status = 'filled'
        mock_order.filled_avg_price = 100.0
        mock_broker.submit_order.return_value = mock_order

        # Position with 100 shares
        positions = [
            {'symbol': 'AAPL', 'qty': 100, 'direction': 'LONG'},
        ]

        result = guard.force_partial_liquidate(mock_broker, positions)

        assert result['success'] is True
        assert len(result['reduced']) == 1
        # Should close 50 shares (50% of 100)
        mock_broker.submit_order.assert_called_once()
        call_args = mock_broker.submit_order.call_args
        assert call_args.kwargs.get('qty') == 50 or call_args[1].get('qty') == 50

    def test_partial_liquidation_custom_percentage(self):
        """Test partial liquidation with custom percentage."""
        config = {
            'daily_drawdown_guard': {
                'partial_liquidation_pct': 75,  # Close 75%
            }
        }
        guard = DailyDrawdownGuard(config)
        guard.reset_day(10000.0)

        mock_broker = MagicMock()
        mock_order = MagicMock()
        mock_order.status = 'filled'
        mock_order.filled_avg_price = 100.0
        mock_broker.submit_order.return_value = mock_order

        positions = [{'symbol': 'AAPL', 'qty': 100, 'direction': 'LONG'}]
        result = guard.force_partial_liquidate(mock_broker, positions)

        # Should close 75 shares
        call_args = mock_broker.submit_order.call_args
        assert call_args.kwargs.get('qty') == 75 or call_args[1].get('qty') == 75


class TestAPIErrorFailsafe:
    """Test API error failsafe (ODE-90).

    If we can't calculate DD (API error), block all trading.
    """

    def test_api_error_blocks_trading(self):
        """Test that API errors block all trading."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # Simulate API error - account with no equity attribute
        bad_account = MagicMock(spec=[])  # No equity or portfolio_value

        tier = guard.update_equity(bad_account)

        # Should set API error flag and block entries
        assert guard.api_error_occurred is True
        assert guard.entries_allowed is False

    def test_api_error_flag_persists(self):
        """Test that API error flag persists until reset."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # Trigger API error
        bad_account = MagicMock(spec=[])
        guard.update_equity(bad_account)
        assert guard.api_error_occurred is True

        # Even with good account, error flag should persist
        good_account = MockAccount(10000.0)
        guard.update_equity(good_account)
        assert guard.api_error_occurred is True

    def test_api_error_clears_on_day_reset(self):
        """Test that API error flag clears on new day."""
        guard = DailyDrawdownGuard()
        guard.reset_day(10000.0)

        # Trigger API error
        bad_account = MagicMock(spec=[])
        guard.update_equity(bad_account)
        assert guard.api_error_occurred is True

        # Reset day should clear the flag
        guard.reset_day(10000.0)
        assert guard.api_error_occurred is False
        assert guard.entries_allowed is True
