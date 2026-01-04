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
