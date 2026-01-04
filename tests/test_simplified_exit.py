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
