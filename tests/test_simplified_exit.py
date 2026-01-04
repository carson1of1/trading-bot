"""Tests for SimplifiedExitManager - R-based exit logic"""
import pytest
import sys
import os
from datetime import datetime
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simplified_exit import SimplifiedExitManager, RBasedPosition


class TestSimplifiedExitManagerInitialization:
    """Test SimplifiedExitManager initialization"""

    def test_default_initialization(self):
        """Should initialize with R = ATR × 2.0 stop logic"""
        mgr = SimplifiedExitManager()
        assert mgr.atr_multiplier == 2.0
        assert mgr.profit_floor_r == 2.0  # Activate floor at +2R
        assert mgr.floor_stop_r == -0.25  # Move stop to -0.25R when floor activates
        assert mgr.partial_exit_r == 3.0  # Partial exit at +3R
        assert mgr.partial_exit_pct == 0.50  # Close 50% at partial

    def test_custom_settings(self):
        """Should accept custom R-based settings"""
        settings = {
            'atr_multiplier': 2.5,
            'profit_floor_r': 2.5,
            'floor_stop_r': -0.5,
            'partial_exit_r': 4.0,
            'partial_exit_pct': 0.40
        }
        mgr = SimplifiedExitManager(settings)
        assert mgr.atr_multiplier == 2.5
        assert mgr.profit_floor_r == 2.5


class TestPositionRegistration:
    """Test position registration with ATR-based R calculation"""

    def test_register_position_calculates_r_value(self):
        """Should calculate R = ATR × multiplier"""
        mgr = SimplifiedExitManager({'atr_multiplier': 2.0})
        pos = mgr.register_position(
            symbol='AAPL',
            entry_price=100.0,
            quantity=10,
            atr=2.5  # ATR = $2.50
        )
        # R = 2.5 × 2.0 = $5.00
        assert pos.r_value == 5.0
        assert pos.symbol == 'AAPL'
        assert pos.quantity == 10

    def test_register_position_sets_initial_stop(self):
        """Should set initial stop at -1R from entry"""
        mgr = SimplifiedExitManager({'atr_multiplier': 2.0})
        pos = mgr.register_position(
            symbol='AAPL',
            entry_price=100.0,
            quantity=10,
            atr=2.5  # R = $5.00
        )
        # Stop at entry - 1R = 100 - 5 = $95
        assert pos.stop_price == 95.0

    def test_register_position_stores_in_positions(self):
        """Should store position for later evaluation"""
        mgr = SimplifiedExitManager()
        mgr.register_position('AAPL', 100.0, 10, atr=2.0)
        assert 'AAPL' in mgr.positions

    def test_register_position_rejects_zero_atr(self):
        """Should reject position with zero or negative ATR"""
        mgr = SimplifiedExitManager()
        with pytest.raises(ValueError, match="ATR must be positive"):
            mgr.register_position('AAPL', 100.0, 10, atr=0.0)

    def test_register_position_rejects_nan_atr(self):
        """Should reject position with NaN ATR"""
        mgr = SimplifiedExitManager()
        with pytest.raises(ValueError, match="ATR must be positive"):
            mgr.register_position('AAPL', 100.0, 10, atr=float('nan'))


class TestATRStopEvaluation:
    """Test ATR-based stop (Phase 1: Entry -> +2R)"""

    def test_triggers_exit_when_price_hits_stop(self):
        """Should trigger full exit when price <= stop"""
        mgr = SimplifiedExitManager({'atr_multiplier': 2.0})
        mgr.register_position('AAPL', entry_price=100.0, quantity=10, atr=2.5)
        # R = $5, stop at $95

        # Price drops to stop level
        result = mgr.evaluate_exit('AAPL', current_price=95.0)

        assert result is not None
        assert result['action'] == 'full_exit'
        assert result['reason'] == 'atr_stop'
        assert result['qty'] == 10

    def test_triggers_exit_when_price_below_stop(self):
        """Should trigger exit when price goes through stop (gap down)"""
        mgr = SimplifiedExitManager({'atr_multiplier': 2.0})
        mgr.register_position('AAPL', entry_price=100.0, quantity=10, atr=2.5)
        # Stop at $95

        result = mgr.evaluate_exit('AAPL', current_price=93.0)  # Below stop

        assert result is not None
        assert result['action'] == 'full_exit'
        assert result['reason'] == 'atr_stop'

    def test_no_exit_when_price_above_stop(self):
        """Should not exit when price is above stop"""
        mgr = SimplifiedExitManager({'atr_multiplier': 2.0})
        mgr.register_position('AAPL', entry_price=100.0, quantity=10, atr=2.5)
        # Stop at $95

        result = mgr.evaluate_exit('AAPL', current_price=96.0)

        assert result is None

    def test_no_exit_when_price_at_entry(self):
        """Should not exit at entry price"""
        mgr = SimplifiedExitManager()
        mgr.register_position('AAPL', entry_price=100.0, quantity=10, atr=2.0)

        result = mgr.evaluate_exit('AAPL', current_price=100.0)

        assert result is None

    def test_no_exit_when_in_profit(self):
        """Should not exit when price is in profit (before +2R)"""
        mgr = SimplifiedExitManager({'atr_multiplier': 2.0})
        mgr.register_position('AAPL', entry_price=100.0, quantity=10, atr=2.5)
        # +2R = $110

        result = mgr.evaluate_exit('AAPL', current_price=108.0)  # +1.6R

        assert result is None

    def test_returns_none_for_unknown_symbol(self):
        """Should return None for unknown symbol"""
        mgr = SimplifiedExitManager()

        result = mgr.evaluate_exit('UNKNOWN', current_price=100.0)

        assert result is None

    def test_returns_r_multiple_in_result(self):
        """Should include current R-multiple in exit result"""
        mgr = SimplifiedExitManager({'atr_multiplier': 2.0})
        mgr.register_position('AAPL', entry_price=100.0, quantity=10, atr=2.5)
        # R = $5, stop at $95 = -1R

        result = mgr.evaluate_exit('AAPL', current_price=95.0)

        assert 'r_multiple' in result
        assert result['r_multiple'] == -1.0


class TestProfitFloorActivation:
    """Test profit floor at +2R (Phase 2: +2R → +3R)"""

    def test_activates_floor_at_2r(self):
        """Should activate profit floor when price reaches +2R"""
        mgr = SimplifiedExitManager({'atr_multiplier': 2.0, 'profit_floor_r': 2.0})
        mgr.register_position('AAPL', entry_price=100.0, quantity=10, atr=2.5)
        # R = $5, +2R = $110

        # Price reaches +2R
        mgr.evaluate_exit('AAPL', current_price=110.0)

        pos = mgr.get_position('AAPL')
        assert pos.profit_floor_active is True

    def test_moves_stop_to_floor_level_on_activation(self):
        """Should move stop to -0.25R when floor activates"""
        mgr = SimplifiedExitManager({
            'atr_multiplier': 2.0,
            'profit_floor_r': 2.0,
            'floor_stop_r': -0.25
        })
        mgr.register_position('AAPL', entry_price=100.0, quantity=10, atr=2.5)
        # R = $5, initial stop at $95 (-1R)
        # Floor stop = entry + (-0.25 × $5) = $98.75

        # Price reaches +2R
        mgr.evaluate_exit('AAPL', current_price=110.0)

        pos = mgr.get_position('AAPL')
        assert pos.stop_price == 98.75  # -0.25R from entry

    def test_stop_does_not_trail_after_floor(self):
        """Stop should NOT trail price after floor activation"""
        mgr = SimplifiedExitManager({
            'atr_multiplier': 2.0,
            'profit_floor_r': 2.0,
            'floor_stop_r': -0.25
        })
        mgr.register_position('AAPL', entry_price=100.0, quantity=10, atr=2.5)
        # R = $5, floor stop = $98.75

        # Activate floor at +2R
        mgr.evaluate_exit('AAPL', current_price=110.0)

        # Price goes higher to +3R
        mgr.evaluate_exit('AAPL', current_price=115.0)

        # Stop should NOT have moved - it stays at -0.25R
        pos = mgr.get_position('AAPL')
        assert pos.stop_price == 98.75

    def test_floor_triggers_exit_at_floor_stop(self):
        """Should exit when price falls to floor stop"""
        mgr = SimplifiedExitManager({
            'atr_multiplier': 2.0,
            'profit_floor_r': 2.0,
            'floor_stop_r': -0.25
        })
        mgr.register_position('AAPL', entry_price=100.0, quantity=10, atr=2.5)

        # Activate floor at +2R
        mgr.evaluate_exit('AAPL', current_price=110.0)

        # Price falls to floor stop ($98.75)
        result = mgr.evaluate_exit('AAPL', current_price=98.75)

        assert result is not None
        assert result['action'] == 'full_exit'
        assert result['reason'] == 'profit_floor'

    def test_floor_not_activated_before_2r(self):
        """Should not activate floor before +2R"""
        mgr = SimplifiedExitManager({'atr_multiplier': 2.0, 'profit_floor_r': 2.0})
        mgr.register_position('AAPL', entry_price=100.0, quantity=10, atr=2.5)
        # +2R = $110

        # Price at +1.5R = $107.50
        mgr.evaluate_exit('AAPL', current_price=107.50)

        pos = mgr.get_position('AAPL')
        assert pos.profit_floor_active is False
        assert pos.stop_price == 95.0  # Still at original -1R
