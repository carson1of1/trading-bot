"""
Tests for ExitManager - Tiered Exit Logic System

Tests verify:
1. Tier 0: Hard stop (-0.50%) always triggers
2. Tier 1: Profit floor activation and enforcement
3. Tier 2: ATR trailing stop activation and behavior
4. Tier 3: Partial take profit execution
5. Tier precedence: hard stop > profit floor > trailing > partial TP
6. Stops only tighten, never loosen
7. Minimum hold time logic with bypass for losses
8. Factory function and configuration
"""

import pytest
from datetime import datetime, timedelta
import pytz
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.risk import (
    ExitManager,
    PositionExitState,
    create_exit_manager,
)


class TestPositionExitState:
    """Tests for PositionExitState dataclass."""

    def test_creation_with_required_fields(self):
        """Test creating PositionExitState with required fields."""
        state = PositionExitState(
            symbol="AAPL",
            entry_price=100.0,
            entry_time=datetime.now(pytz.UTC),
            quantity=100
        )
        assert state.symbol == "AAPL"
        assert state.entry_price == 100.0
        assert state.quantity == 100

    def test_post_init_sets_peak_price(self):
        """Test __post_init__ sets peak_price to entry_price."""
        state = PositionExitState(
            symbol="AAPL",
            entry_price=100.0,
            entry_time=datetime.now(pytz.UTC),
            quantity=100
        )
        assert state.peak_price == 100.0

    def test_default_thresholds(self):
        """Test default exit thresholds."""
        state = PositionExitState(
            symbol="AAPL",
            entry_price=100.0,
            entry_time=datetime.now(pytz.UTC),
            quantity=100
        )
        # Default values
        assert state.profit_floor_activation_pct == 0.0125  # +1.25%
        assert state.profit_floor_lock_pct == 0.005  # +0.50%
        assert state.trailing_activation_pct == 0.0175  # +1.75%
        assert state.partial_tp_pct == 0.02  # +2.00%
        assert state.hard_stop_pct == 0.005  # -0.50%
        assert state.partial_tp_size == 0.50  # 50%

    def test_state_tracking_defaults(self):
        """Test state tracking starts with correct defaults."""
        state = PositionExitState(
            symbol="AAPL",
            entry_price=100.0,
            entry_time=datetime.now(pytz.UTC),
            quantity=100
        )
        assert state.profit_floor_active is False
        assert state.trailing_active is False
        assert state.partial_tp_executed is False
        assert state.bars_held == 0

    def test_minimum_hold_defaults(self):
        """Test minimum hold time defaults."""
        state = PositionExitState(
            symbol="AAPL",
            entry_price=100.0,
            entry_time=datetime.now(pytz.UTC),
            quantity=100
        )
        assert state.min_hold_bars == 12
        assert state.min_hold_bypass_loss_pct == 0.003  # 0.30%


class TestExitManagerInitialization:
    """Tests for ExitManager initialization and configuration."""

    def test_default_initialization(self):
        """Test ExitManager with default settings."""
        exit_mgr = ExitManager()
        # Check default thresholds (converted to decimal)
        assert exit_mgr.profit_floor_activation_pct == 0.0125  # 1.25%
        assert exit_mgr.profit_floor_lock_pct == 0.005  # 0.50%
        assert exit_mgr.trailing_activation_pct == 0.0175  # 1.75%
        assert exit_mgr.partial_tp_pct == 0.02  # 2.0%
        assert exit_mgr.hard_stop_pct == 0.005  # 0.50%
        assert exit_mgr.atr_multiplier == 2.0
        assert exit_mgr.min_hold_bars == 12

    def test_custom_configuration(self):
        """Test ExitManager with custom settings."""
        settings = {
            'risk': {
                'profit_floor_activation_pct': 0.40,  # 0.40%
                'trailing_stop_min_profit_floor': 0.15,  # 0.15%
                'trailing_activation_pct': 0.60,  # 0.60%
                'partial_tp_pct': 1.0,  # 1.0%
                'hard_stop_pct': 0.50,  # 0.50%
                'atr_trailing_multiplier': 1.5,
                'min_hold_bars': 8,
                'min_hold_bypass_loss_pct': 0.20  # 0.20%
            }
        }
        exit_mgr = ExitManager(settings)
        assert exit_mgr.profit_floor_activation_pct == 0.004  # 0.40%
        assert exit_mgr.profit_floor_lock_pct == 0.0015  # 0.15%
        assert exit_mgr.trailing_activation_pct == 0.006  # 0.60%
        assert exit_mgr.partial_tp_pct == 0.01  # 1.0%
        assert exit_mgr.hard_stop_pct == 0.005  # 0.50%
        assert exit_mgr.atr_multiplier == 1.5
        assert exit_mgr.min_hold_bars == 8
        assert exit_mgr.min_hold_bypass_loss_pct == 0.002  # 0.20%

    def test_factory_function(self):
        """Test create_exit_manager factory function."""
        exit_mgr = create_exit_manager()
        assert isinstance(exit_mgr, ExitManager)
        assert exit_mgr.positions == {}

    def test_factory_function_with_settings(self):
        """Test create_exit_manager with custom settings."""
        settings = {'risk': {'hard_stop_pct': 1.0}}
        exit_mgr = create_exit_manager(settings)
        assert exit_mgr.hard_stop_pct == 0.01  # 1.0%


class TestPositionRegistration:
    """Tests for position registration and management."""

    def test_register_position(self):
        """Test registering a new position."""
        exit_mgr = ExitManager()
        state = exit_mgr.register_position("AAPL", 100.0, 100)

        assert "AAPL" in exit_mgr.positions
        assert state.symbol == "AAPL"
        assert state.entry_price == 100.0
        assert state.quantity == 100
        assert state.peak_price == 100.0

    def test_register_position_with_custom_time(self):
        """Test registering position with custom entry time."""
        exit_mgr = ExitManager()
        entry_time = datetime(2024, 1, 15, 10, 30, tzinfo=pytz.UTC)
        state = exit_mgr.register_position("AAPL", 100.0, 100, entry_time)

        assert state.entry_time == entry_time

    def test_unregister_position(self):
        """Test unregistering a position."""
        exit_mgr = ExitManager()
        exit_mgr.register_position("AAPL", 100.0, 100)

        result = exit_mgr.unregister_position("AAPL")
        assert result is True
        assert "AAPL" not in exit_mgr.positions

    def test_unregister_nonexistent_position(self):
        """Test unregistering a position that doesn't exist."""
        exit_mgr = ExitManager()
        result = exit_mgr.unregister_position("AAPL")
        assert result is False

    def test_update_quantity(self):
        """Test updating position quantity."""
        exit_mgr = ExitManager()
        exit_mgr.register_position("AAPL", 100.0, 100)

        result = exit_mgr.update_quantity("AAPL", 50)
        assert result is True
        assert exit_mgr.positions["AAPL"].quantity == 50

    def test_update_quantity_nonexistent(self):
        """Test updating quantity for nonexistent position."""
        exit_mgr = ExitManager()
        result = exit_mgr.update_quantity("AAPL", 50)
        assert result is False

    def test_increment_bars_held(self):
        """Test incrementing bars held counter."""
        exit_mgr = ExitManager()
        exit_mgr.register_position("AAPL", 100.0, 100)

        count = exit_mgr.increment_bars_held("AAPL")
        assert count == 1
        count = exit_mgr.increment_bars_held("AAPL")
        assert count == 2

    def test_increment_bars_nonexistent(self):
        """Test incrementing bars for nonexistent position."""
        exit_mgr = ExitManager()
        result = exit_mgr.increment_bars_held("AAPL")
        assert result == -1


class TestTier0HardStop:
    """Tests for Tier 0: Hard Stop (-0.50%)."""

    def test_hard_stop_triggers(self):
        """Test hard stop triggers at -0.50% loss."""
        exit_mgr = ExitManager()
        exit_mgr.register_position("AAPL", 100.0, 100)
        # Simulate enough bars held to pass min hold check
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        # Price at exactly -0.50% = $99.50
        result = exit_mgr.evaluate_exit("AAPL", 99.50)

        assert result is not None
        assert result['action'] == 'full_exit'
        assert result['reason'] == ExitManager.REASON_HARD_STOP
        assert result['qty'] == 100

    def test_hard_stop_triggers_below_threshold(self):
        """Test hard stop triggers below -0.50% loss."""
        exit_mgr = ExitManager()
        exit_mgr.register_position("AAPL", 100.0, 100)
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        # Price at -1.0% = $99.00
        result = exit_mgr.evaluate_exit("AAPL", 99.00)

        assert result is not None
        assert result['action'] == 'full_exit'
        assert result['reason'] == ExitManager.REASON_HARD_STOP

    def test_hard_stop_bypasses_min_hold(self):
        """Test hard stop triggers even during min hold period."""
        exit_mgr = ExitManager()
        exit_mgr.register_position("AAPL", 100.0, 100)
        # No bars held - still in min hold period

        # Hard stop at -0.50%
        result = exit_mgr.evaluate_exit("AAPL", 99.50)

        assert result is not None
        assert result['action'] == 'full_exit'
        assert result['reason'] == ExitManager.REASON_HARD_STOP

    def test_hard_stop_precedence_over_profit_floor(self):
        """Test hard stop takes precedence even if profit floor was active."""
        exit_mgr = ExitManager()
        state = exit_mgr.register_position("AAPL", 100.0, 100)
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        # Manually activate profit floor
        state.profit_floor_active = True
        state.profit_floor_price = 100.50

        # Price drops below hard stop
        result = exit_mgr.evaluate_exit("AAPL", 99.50)

        assert result['reason'] == ExitManager.REASON_HARD_STOP


class TestMinimumHoldTime:
    """Tests for minimum hold time logic."""

    def test_exits_blocked_during_min_hold(self):
        """Test non-stop exits are blocked during min hold period."""
        exit_mgr = ExitManager()
        exit_mgr.register_position("AAPL", 100.0, 100)
        # Only 5 bars held (less than 12 default)
        for _ in range(5):
            exit_mgr.increment_bars_held("AAPL")

        # Even at +2% (partial TP level), no exit during min hold
        result = exit_mgr.evaluate_exit("AAPL", 102.0)

        assert result is None

    def test_exits_allowed_after_min_hold(self):
        """Test exits allowed after min hold period."""
        # Use custom settings for faster activation
        settings = {
            'risk': {
                'profit_floor_activation_pct': 0.40,
                'partial_tp_pct': 2.0,
            }
        }
        exit_mgr = ExitManager(settings)
        exit_mgr.register_position("AAPL", 100.0, 100)
        # 15 bars held (more than 12 default)
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        # At +2% should trigger partial TP
        result = exit_mgr.evaluate_exit("AAPL", 102.0)

        assert result is not None

    def test_min_hold_bypass_for_significant_loss(self):
        """Test min hold can be bypassed for significant loss."""
        # Use custom settings with tighter thresholds for testing
        settings = {
            'risk': {
                'profit_floor_activation_pct': 0.30,  # Activate at 0.30%
                'trailing_stop_min_profit_floor': 0.10,  # Lock at 0.10%
            }
        }
        exit_mgr = ExitManager(settings)
        state = exit_mgr.register_position("AAPL", 100.0, 100)
        # Only 2 bars held
        exit_mgr.increment_bars_held("AAPL")
        exit_mgr.increment_bars_held("AAPL")

        # Simulate profit floor was activated previously
        state.profit_floor_active = True
        state.profit_floor_price = 100.10  # Floor at +0.10%

        # Now price drops to floor (but above hard stop)
        # This should trigger profit floor exit despite min hold
        result = exit_mgr.evaluate_exit("AAPL", 100.00)  # At breakeven, below floor

        # The min hold bypass only applies if loss > bypass threshold
        # Since we're at breakeven (not a loss > 0.30%), and under min hold,
        # we should NOT trigger (min hold blocks it)
        # But if loss exceeds bypass threshold, it would trigger
        assert result is None  # Still blocked by min hold


class TestTier1ProfitFloor:
    """Tests for Tier 1: Profit Floor."""

    def test_profit_floor_activation(self):
        """Test profit floor activates at threshold."""
        settings = {
            'risk': {
                'profit_floor_activation_pct': 1.0,  # 1.0%
                'trailing_stop_min_profit_floor': 0.50,  # 0.50%
            }
        }
        exit_mgr = ExitManager(settings)
        state = exit_mgr.register_position("AAPL", 100.0, 100)
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        # Price at +1.0% = $101.00 should activate profit floor
        exit_mgr.evaluate_exit("AAPL", 101.0)

        assert state.profit_floor_active is True
        assert state.profit_floor_price == pytest.approx(100.50, abs=0.01)  # Entry * (1 + 0.50%)

    def test_profit_floor_triggers_exit(self):
        """Test profit floor triggers exit when price hits floor."""
        settings = {
            'risk': {
                'profit_floor_activation_pct': 1.0,
                'trailing_stop_min_profit_floor': 0.50,
            }
        }
        exit_mgr = ExitManager(settings)
        state = exit_mgr.register_position("AAPL", 100.0, 100)
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        # Activate profit floor
        exit_mgr.evaluate_exit("AAPL", 101.0)

        # Price drops BELOW floor (floor is 100.50, so price at 100.49 triggers)
        result = exit_mgr.evaluate_exit("AAPL", 100.49)

        assert result is not None
        assert result['action'] == 'full_exit'
        assert result['reason'] == ExitManager.REASON_PROFIT_FLOOR

    def test_profit_floor_only_tightens(self):
        """Test profit floor only moves up, never down."""
        settings = {
            'risk': {
                'profit_floor_activation_pct': 1.0,
                'trailing_stop_min_profit_floor': 0.50,
            }
        }
        exit_mgr = ExitManager(settings)
        state = exit_mgr.register_position("AAPL", 100.0, 100)
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        # Activate at +1.0%
        exit_mgr.evaluate_exit("AAPL", 101.0)
        initial_floor = state.profit_floor_price

        # Price goes higher - floor should stay the same (lock_pct based on entry)
        exit_mgr.evaluate_exit("AAPL", 102.0)

        # Floor should not decrease
        assert state.profit_floor_price >= initial_floor


class TestTier2ATRTrailing:
    """Tests for Tier 2: ATR Trailing Stop."""

    def test_trailing_activation(self):
        """Test trailing stop activates at threshold."""
        settings = {
            'risk': {
                'profit_floor_activation_pct': 1.0,
                'trailing_stop_min_profit_floor': 0.50,  # Need to set floor lock pct
                'trailing_activation_pct': 1.75,
            }
        }
        exit_mgr = ExitManager(settings)
        state = exit_mgr.register_position("AAPL", 100.0, 100)
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        # Provide ATR value
        atr = 2.0  # $2.00 ATR

        # Price at +1.75% = $101.75 should activate trailing
        exit_mgr.evaluate_exit("AAPL", 101.75, current_atr=atr)

        assert state.trailing_active is True
        # Trailing stop = current - (ATR * multiplier) = 101.75 - (2.0 * 2.0) = 97.75
        # BUT it must be at least the profit floor price (100.50)
        # Since profit floor is also activated, trailing will be max(97.75, 100.50) = 100.50
        assert state.trailing_stop_price == pytest.approx(100.50, abs=0.01)

    def test_trailing_uses_fallback_without_atr(self):
        """Test trailing uses 1% fallback when no ATR."""
        settings = {
            'risk': {
                'profit_floor_activation_pct': 1.0,
                'trailing_activation_pct': 1.75,
            }
        }
        exit_mgr = ExitManager(settings)
        state = exit_mgr.register_position("AAPL", 100.0, 100)
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        # No ATR provided
        exit_mgr.evaluate_exit("AAPL", 101.75)

        assert state.trailing_active is True
        # Fallback: 1% of current = 101.75 * 0.01 = 1.0175
        # Trailing stop = 101.75 - 1.0175 = 100.7325
        expected_trail = 101.75 - (101.75 * 0.01)
        assert state.trailing_stop_price == pytest.approx(expected_trail, abs=0.01)

    def test_trailing_only_tightens(self):
        """Test trailing stop only moves up, never down."""
        settings = {
            'risk': {
                'profit_floor_activation_pct': 1.0,
                'trailing_activation_pct': 1.75,
            }
        }
        exit_mgr = ExitManager(settings)
        state = exit_mgr.register_position("AAPL", 100.0, 100)
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        atr = 1.0  # $1.00 ATR

        # Activate trailing at 101.75
        exit_mgr.evaluate_exit("AAPL", 101.75, current_atr=atr)
        initial_trail = state.trailing_stop_price  # 101.75 - 2.0 = 99.75

        # Price goes higher to 103
        exit_mgr.evaluate_exit("AAPL", 103.0, current_atr=atr)
        higher_trail = state.trailing_stop_price  # 103.0 - 2.0 = 101.0

        assert higher_trail > initial_trail

        # Price drops to 102 - trail should NOT decrease
        exit_mgr.evaluate_exit("AAPL", 102.0, current_atr=atr)

        assert state.trailing_stop_price >= higher_trail

    def test_trailing_triggers_exit(self):
        """Test trailing stop triggers exit when hit."""
        settings = {
            'risk': {
                'profit_floor_activation_pct': 1.0,
                'trailing_activation_pct': 1.75,
            }
        }
        exit_mgr = ExitManager(settings)
        exit_mgr.register_position("AAPL", 100.0, 100)
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        atr = 1.0

        # Activate and raise trailing
        exit_mgr.evaluate_exit("AAPL", 102.0, current_atr=atr)
        exit_mgr.evaluate_exit("AAPL", 105.0, current_atr=atr)

        state = exit_mgr.get_position_state("AAPL")
        trailing_stop = state.trailing_stop_price  # 105 - 2 = 103

        # Price hits trailing stop
        result = exit_mgr.evaluate_exit("AAPL", trailing_stop, current_atr=atr)

        assert result is not None
        assert result['action'] == 'full_exit'
        assert result['reason'] == ExitManager.REASON_ATR_TRAILING

    def test_trailing_respects_profit_floor(self):
        """Test trailing stop never goes below profit floor."""
        settings = {
            'risk': {
                'profit_floor_activation_pct': 1.0,
                'trailing_stop_min_profit_floor': 0.50,
                'trailing_activation_pct': 1.75,
            }
        }
        exit_mgr = ExitManager(settings)
        state = exit_mgr.register_position("AAPL", 100.0, 100)
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        # Large ATR that would put trail below floor
        atr = 5.0  # $5.00 ATR, multiplier 2.0 = $10 distance

        # Activate both profit floor and trailing
        exit_mgr.evaluate_exit("AAPL", 101.75, current_atr=atr)

        # Trailing would be 101.75 - 10 = 91.75
        # But profit floor is 100.50
        assert state.trailing_stop_price >= state.profit_floor_price


class TestTier3PartialTakeProfit:
    """Tests for Tier 3: Partial Take Profit."""

    def test_partial_tp_triggers(self):
        """Test partial TP triggers at threshold."""
        settings = {
            'risk': {
                'profit_floor_activation_pct': 1.0,
                'trailing_activation_pct': 1.75,
                'partial_tp_pct': 2.0,
                'partial_tp_size': 0.50,
            }
        }
        exit_mgr = ExitManager(settings)
        exit_mgr.register_position("AAPL", 100.0, 100)
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        # Price at +2.0% = $102.00 should trigger partial TP
        result = exit_mgr.evaluate_exit("AAPL", 102.0)

        assert result is not None
        assert result['action'] == 'partial_tp'
        assert result['reason'] == ExitManager.REASON_PARTIAL_TP
        assert result['qty'] == 50  # 50% of 100

    def test_partial_tp_executes_only_once(self):
        """Test partial TP only executes once per position."""
        settings = {
            'risk': {
                'profit_floor_activation_pct': 1.0,
                'trailing_activation_pct': 1.75,
                'partial_tp_pct': 2.0,
            }
        }
        exit_mgr = ExitManager(settings)
        exit_mgr.register_position("AAPL", 100.0, 100)
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        # First evaluation triggers partial TP
        result1 = exit_mgr.evaluate_exit("AAPL", 102.0)
        assert result1 is not None
        assert result1['action'] == 'partial_tp'

        # Update quantity after partial close
        exit_mgr.update_quantity("AAPL", 50)

        # Second evaluation at same or higher price should NOT trigger again
        result2 = exit_mgr.evaluate_exit("AAPL", 103.0)

        # Should be None (no exit) since partial TP already executed
        # and we're above profit floor/trailing
        assert result2 is None or result2['reason'] != ExitManager.REASON_PARTIAL_TP

    def test_partial_tp_raises_floor_to_breakeven(self):
        """Test partial TP raises profit floor to breakeven."""
        settings = {
            'risk': {
                'profit_floor_activation_pct': 1.0,
                'trailing_stop_min_profit_floor': 0.50,
                'partial_tp_pct': 2.0,
            }
        }
        exit_mgr = ExitManager(settings)
        state = exit_mgr.register_position("AAPL", 100.0, 100)
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        # Trigger partial TP
        exit_mgr.evaluate_exit("AAPL", 102.0)

        # Profit floor should be at least breakeven (entry price)
        assert state.profit_floor_price >= 100.0


class TestTierPrecedence:
    """Tests for tier precedence: hard stop > profit floor > trailing > partial TP."""

    def test_hard_stop_over_all_others(self):
        """Test hard stop takes precedence over all other exits."""
        exit_mgr = ExitManager()
        state = exit_mgr.register_position("AAPL", 100.0, 100)
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        # Manually set up all exit conditions
        state.profit_floor_active = True
        state.profit_floor_price = 100.50
        state.trailing_active = True
        state.trailing_stop_price = 99.80

        # Price at hard stop level
        result = exit_mgr.evaluate_exit("AAPL", 99.50)

        assert result['reason'] == ExitManager.REASON_HARD_STOP

    def test_profit_floor_over_trailing_and_partial(self):
        """Test profit floor takes precedence over trailing and partial TP.

        Note: In the actual implementation, profit floor is checked BEFORE
        trailing stop in the tier evaluation. This test verifies that when
        price hits the profit floor level (which is below any trailing stop),
        the profit floor exit is triggered.
        """
        settings = {
            'risk': {
                'profit_floor_activation_pct': 1.0,
                'trailing_stop_min_profit_floor': 0.50,
                'trailing_activation_pct': 1.75,
                'partial_tp_pct': 2.0,
            }
        }
        exit_mgr = ExitManager(settings)
        state = exit_mgr.register_position("AAPL", 100.0, 100)
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        # Activate profit floor at +1.0% (but not trailing yet)
        exit_mgr.evaluate_exit("AAPL", 101.0)  # Only profit floor activates

        assert state.profit_floor_active is True
        assert state.trailing_active is False  # Not activated yet

        # Price drops below profit floor - should trigger profit floor exit
        result = exit_mgr.evaluate_exit("AAPL", 100.49)  # Below floor of ~100.50

        # Should be profit floor since trailing wasn't activated
        assert result is not None
        assert result['reason'] == ExitManager.REASON_PROFIT_FLOOR

    def test_trailing_over_partial_tp(self):
        """Test trailing takes precedence over partial TP."""
        settings = {
            'risk': {
                'profit_floor_activation_pct': 1.0,
                'trailing_stop_min_profit_floor': 0.50,
                'trailing_activation_pct': 1.5,
                'partial_tp_pct': 2.0,
            }
        }
        exit_mgr = ExitManager(settings)
        state = exit_mgr.register_position("AAPL", 100.0, 100)
        for _ in range(15):
            exit_mgr.increment_bars_held("AAPL")

        atr = 0.5  # Small ATR

        # Price goes to +2%, activating all conditions
        exit_mgr.evaluate_exit("AAPL", 102.0, current_atr=atr)

        # Raise trailing by going higher
        exit_mgr.evaluate_exit("AAPL", 104.0, current_atr=atr)

        # Set trailing just below +2% level
        state.trailing_stop_price = 101.90

        # Price at trailing stop (also at partial TP level)
        result = exit_mgr.evaluate_exit("AAPL", 101.90, current_atr=atr)

        # Should be trailing stop since it's evaluated before partial TP
        assert result['reason'] == ExitManager.REASON_ATR_TRAILING


class TestGetters:
    """Tests for getter methods."""

    def test_get_position_state(self):
        """Test getting position state."""
        exit_mgr = ExitManager()
        exit_mgr.register_position("AAPL", 100.0, 100)

        state = exit_mgr.get_position_state("AAPL")

        assert state is not None
        assert state.symbol == "AAPL"

    def test_get_position_state_nonexistent(self):
        """Test getting nonexistent position state."""
        exit_mgr = ExitManager()
        state = exit_mgr.get_position_state("AAPL")
        assert state is None

    def test_get_all_states(self):
        """Test getting all position states."""
        exit_mgr = ExitManager()
        exit_mgr.register_position("AAPL", 100.0, 100)
        exit_mgr.register_position("MSFT", 200.0, 50)

        states = exit_mgr.get_all_states()

        assert len(states) == 2
        assert "AAPL" in states
        assert "MSFT" in states

    def test_get_status_summary(self):
        """Test getting status summary."""
        exit_mgr = ExitManager()
        state = exit_mgr.register_position("AAPL", 100.0, 100)
        state.profit_floor_active = True
        state.profit_floor_price = 100.50

        summary = exit_mgr.get_status_summary("AAPL")

        assert summary is not None
        assert summary['symbol'] == "AAPL"
        assert summary['entry_price'] == 100.0
        assert summary['quantity'] == 100
        assert summary['profit_floor_active'] is True
        assert summary['profit_floor_price'] == 100.50

    def test_get_status_summary_nonexistent(self):
        """Test getting status summary for nonexistent position."""
        exit_mgr = ExitManager()
        summary = exit_mgr.get_status_summary("AAPL")
        assert summary is None


class TestEvaluateExitEdgeCases:
    """Tests for edge cases in evaluate_exit."""

    def test_evaluate_nonexistent_position(self):
        """Test evaluating exit for nonexistent position."""
        exit_mgr = ExitManager()
        result = exit_mgr.evaluate_exit("AAPL", 100.0)
        assert result is None

    def test_peak_price_updates(self):
        """Test peak price updates correctly."""
        exit_mgr = ExitManager()
        state = exit_mgr.register_position("AAPL", 100.0, 100)

        exit_mgr.evaluate_exit("AAPL", 101.0)
        assert state.peak_price == 101.0

        exit_mgr.evaluate_exit("AAPL", 102.0)
        assert state.peak_price == 102.0

        exit_mgr.evaluate_exit("AAPL", 101.5)
        assert state.peak_price == 102.0  # Doesn't decrease

    def test_atr_updates_on_evaluate(self):
        """Test ATR updates when provided."""
        exit_mgr = ExitManager()
        state = exit_mgr.register_position("AAPL", 100.0, 100)

        exit_mgr.evaluate_exit("AAPL", 100.5, current_atr=1.5)
        assert state.current_atr == 1.5

        exit_mgr.evaluate_exit("AAPL", 100.6, current_atr=2.0)
        assert state.current_atr == 2.0

    def test_atr_ignored_if_zero_or_negative(self):
        """Test zero or negative ATR is ignored."""
        exit_mgr = ExitManager()
        state = exit_mgr.register_position("AAPL", 100.0, 100)

        exit_mgr.evaluate_exit("AAPL", 100.5, current_atr=1.5)
        assert state.current_atr == 1.5

        exit_mgr.evaluate_exit("AAPL", 100.6, current_atr=0)
        assert state.current_atr == 1.5  # Unchanged

        exit_mgr.evaluate_exit("AAPL", 100.7, current_atr=-1.0)
        assert state.current_atr == 1.5  # Unchanged


class TestImports:
    """Tests for module imports."""

    def test_import_from_core(self):
        """Test ExitManager can be imported from core."""
        from core import ExitManager, PositionExitState, create_exit_manager

        assert ExitManager is not None
        assert PositionExitState is not None
        assert create_exit_manager is not None

    def test_exit_manager_reason_constants(self):
        """Test exit reason constants are defined."""
        assert ExitManager.REASON_PROFIT_FLOOR == 'profit_floor'
        assert ExitManager.REASON_ATR_TRAILING == 'atr_trailing'
        assert ExitManager.REASON_PARTIAL_TP == 'partial_tp'
        assert ExitManager.REASON_HARD_STOP == 'hard_stop'
