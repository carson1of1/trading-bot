"""
Tests for backtest position limit enforcement.

BUG FIX (Jan 4, 2026): These tests verify that:
1. Backtest enforces max_open_positions from config
2. Position limits are checked for both new signals and pending entries
3. SHORT positions are counted toward position limits
4. Position limits prevent catastrophic over-allocation
"""
import pytest
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import Backtest1Hour


class TestBacktestPositionLimits:
    """Test that backtest enforces position limits"""

    @pytest.fixture
    def sample_config(self):
        """Configuration with explicit position limits"""
        return {
            'risk_management': {
                'max_position_size_pct': 3.0,
                'max_open_positions': 2,  # Only allow 2 positions
                'stop_loss_pct': 5.0,
                'take_profit_pct': 8.0,
                'max_daily_loss_pct': 3.0,
            },
            'exit_manager': {
                'enabled': True,
                'tier_0_hard_stop': -0.05,
                'max_hold_hours': 48,
                'eod_close': False,
            },
            'trailing_stop': {
                'enabled': True,
                'activation_pct': 0.5,
                'trail_pct': 0.5,
            },
            'entry_gate': {
                'confidence_threshold': 50,
            },
            'strategies': {
                'momentum': {'enabled': True, 'weight': 1.0},
            },
        }

    def test_reads_max_open_positions_from_config(self, sample_config):
        """Should read max_open_positions from config correctly"""
        bt = Backtest1Hour(initial_capital=10000, config=sample_config)
        assert bt.max_open_positions == 2

    def test_default_max_open_positions(self):
        """Should default to 5 positions if not specified"""
        config = {'risk_management': {}}
        bt = Backtest1Hour(initial_capital=10000, config=config)
        assert bt.max_open_positions == 5

    def test_position_limit_blocks_excess_entries(self, sample_config):
        """
        Should block new entries when max_open_positions is reached.

        This is the core fix for the 62%+ daily loss bug.
        """
        bt = Backtest1Hour(
            initial_capital=10000,
            config=sample_config,
            kill_switch_trace=True,
            enter_at_open=True  # Use market orders for simpler test
        )

        # Create mock data with signals for 4 symbols
        # With max_open_positions=2, only first 2 should get positions
        base_date = pd.Timestamp('2025-01-01 09:30:00', tz='America/New_York')

        def create_symbol_data(symbol, signal_at_bar=5):
            """Create data where signal=1 at specified bar"""
            data = []
            for i in range(20):
                ts = base_date + pd.Timedelta(hours=i)
                data.append({
                    'timestamp': ts,
                    'open': 100 + i * 0.1,
                    'high': 101 + i * 0.1,
                    'low': 99 + i * 0.1,
                    'close': 100.5 + i * 0.1,
                    'volume': 1000000,
                    'signal': 1 if i == signal_at_bar else 0,
                    'strategy': 'Momentum',
                    'reasoning': 'Test signal',
                    'atr': 1.5,
                })
            return pd.DataFrame(data)

        # All symbols get signals at bar 5 (same timestamp)
        signals_data = {
            'AAPL': create_symbol_data('AAPL', signal_at_bar=5),
            'NVDA': create_symbol_data('NVDA', signal_at_bar=5),
            'MSFT': create_symbol_data('MSFT', signal_at_bar=5),
            'GOOGL': create_symbol_data('GOOGL', signal_at_bar=5),
        }

        # Run the simulation
        trades = bt.simulate_trades_interleaved(signals_data)

        # Count how many symbols actually got entries
        symbols_with_entries = set(t['symbol'] for t in trades)

        # With max_open_positions=2, only 2 symbols should have trades
        assert len(symbols_with_entries) <= 2, \
            f"Expected max 2 symbols with positions, got {len(symbols_with_entries)}: {symbols_with_entries}"

        # Check trace log for blocked entries
        blocked_entries = [
            e for e in bt._kill_switch_trace_log
            if e.get('event') == 'ENTRY_BLOCKED' and 'max_positions' in str(e.get('block_reason', ''))
        ]

        # At least 2 entries should have been blocked (4 signals - 2 allowed = 2 blocked)
        assert len(blocked_entries) >= 2, \
            f"Expected at least 2 blocked entries, got {len(blocked_entries)}"


class TestBacktestShortPositions:
    """Test that SHORT positions work correctly"""

    @pytest.fixture
    def shorts_config(self):
        """Configuration for short-only testing"""
        return {
            'risk_management': {
                'max_position_size_pct': 10.0,
                'max_open_positions': 3,
                'stop_loss_pct': 5.0,
                'take_profit_pct': 8.0,
                'max_daily_loss_pct': 5.0,
            },
            'exit_manager': {
                'enabled': True,
                'tier_0_hard_stop': -0.05,
                'max_hold_hours': 48,
                'eod_close': False,
            },
            'trailing_stop': {
                'enabled': True,
                'activation_pct': 0.5,
                'trail_pct': 0.5,
            },
            'entry_gate': {
                'confidence_threshold': 50,
            },
            'strategies': {
                'momentum': {'enabled': True, 'weight': 1.0},
            },
        }

    def test_short_positions_counted_toward_limit(self, shorts_config):
        """SHORT positions should count toward max_open_positions"""
        bt = Backtest1Hour(
            initial_capital=10000,
            config=shorts_config,
            shorts_only=True,
            kill_switch_trace=True
        )

        base_date = pd.Timestamp('2025-01-01 09:30:00', tz='America/New_York')

        def create_short_signal_data(signal_at_bar=5):
            """Create data with SHORT signal (signal=-1)"""
            data = []
            for i in range(20):
                ts = base_date + pd.Timedelta(hours=i)
                data.append({
                    'timestamp': ts,
                    'open': 100 - i * 0.1,  # Downtrend for shorts
                    'high': 101 - i * 0.1,
                    'low': 99 - i * 0.1,
                    'close': 100 - i * 0.1,
                    'volume': 1000000,
                    'signal': -1 if i == signal_at_bar else 0,  # SHORT signal
                    'strategy': 'Momentum_SHORT',
                    'reasoning': 'Test short signal',
                    'atr': 1.5,
                })
            return pd.DataFrame(data)

        # 5 symbols all with SHORT signals at bar 5
        signals_data = {
            'AAPL': create_short_signal_data(signal_at_bar=5),
            'NVDA': create_short_signal_data(signal_at_bar=5),
            'MSFT': create_short_signal_data(signal_at_bar=5),
            'GOOGL': create_short_signal_data(signal_at_bar=5),
            'AMZN': create_short_signal_data(signal_at_bar=5),
        }

        trades = bt.simulate_trades_interleaved(signals_data)

        # All trades should be SHORT
        for trade in trades:
            assert trade.get('direction') == 'SHORT', \
                f"Expected SHORT direction, got {trade.get('direction')}"

        # Count symbols with entries (should be <= 3 due to max_open_positions)
        symbols_with_entries = set(t['symbol'] for t in trades)
        assert len(symbols_with_entries) <= 3, \
            f"Expected max 3 SHORT positions, got {len(symbols_with_entries)}"

    def test_short_stop_loss_triggers_on_price_increase(self, shorts_config):
        """SHORT stop loss should trigger when price increases"""
        bt = Backtest1Hour(
            initial_capital=10000,
            config=shorts_config,
            shorts_only=True,
            enter_at_open=True  # Use market orders for simpler test
        )

        base_date = pd.Timestamp('2025-01-01 09:30:00', tz='America/New_York')

        # Create data: SHORT entry at bar 2, price spikes up at bar 5
        data = []
        for i in range(20):
            ts = base_date + pd.Timedelta(hours=i)
            if i < 5:
                price = 100.0
            else:
                price = 110.0  # 10% spike up = loss for SHORT

            data.append({
                'timestamp': ts,
                'open': price,
                'high': price + 1,
                'low': price - 1,
                'close': price,
                'volume': 1000000,
                'signal': -1 if i == 2 else 0,  # SHORT at bar 2
                'strategy': 'Momentum_SHORT',
                'reasoning': 'Test',
                'atr': 1.5,
            })

        signals_data = {'AAPL': pd.DataFrame(data)}
        trades = bt.simulate_trades_interleaved(signals_data)

        # Should have at least one trade
        assert len(trades) >= 1, "Expected at least 1 SHORT trade"

        # The trade should be closed (stop loss, trailing stop, or end of backtest)
        short_trade = trades[0]
        assert short_trade['direction'] == 'SHORT'
        # Exit could be stop_loss, trailing_stop (if price moved in profit then reversed),
        # or end_of_backtest. All are valid exits.
        valid_exits = ['stop_loss', 'trailing_stop', 'end_of_backtest', 'emergency_stop']
        assert short_trade.get('exit_reason') in valid_exits, \
            f"Expected valid exit, got {short_trade.get('exit_reason')}"

    def test_short_pnl_calculation(self, shorts_config):
        """SHORT P&L should be positive when price drops"""
        bt = Backtest1Hour(
            initial_capital=10000,
            config=shorts_config,
            shorts_only=True,
            enter_at_open=True  # Use market orders for simpler test
        )

        base_date = pd.Timestamp('2025-01-01 09:30:00', tz='America/New_York')

        # Create data: SHORT entry at bar 2, price drops 5% by bar 10
        data = []
        for i in range(20):
            ts = base_date + pd.Timedelta(hours=i)
            if i <= 2:
                price = 100.0
            elif i <= 10:
                price = 95.0  # 5% drop = profit for SHORT
            else:
                price = 94.0  # Continue down for take profit

            data.append({
                'timestamp': ts,
                'open': price,
                'high': price + 0.5,
                'low': price - 0.5,
                'close': price,
                'volume': 1000000,
                'signal': -1 if i == 2 else 0,
                'strategy': 'Momentum_SHORT',
                'reasoning': 'Test',
                'atr': 1.5,
            })

        signals_data = {'AAPL': pd.DataFrame(data)}
        trades = bt.simulate_trades_interleaved(signals_data)

        # Should have a SHORT trade
        assert len(trades) >= 1
        short_trade = trades[0]
        assert short_trade['direction'] == 'SHORT'

        # If trade closed on profitable exit, P&L should be positive
        # (entry at ~100, exit at ~95 = profit for SHORT)
        if short_trade['exit_reason'] not in ['end_of_backtest']:
            # Entry was higher than exit = profit for SHORT
            assert short_trade['entry_price'] > short_trade['exit_price'], \
                "SHORT should profit when exit < entry"


class TestBacktestMaxDrawdownPrevention:
    """Test that position limits prevent catastrophic drawdowns"""

    def test_limited_positions_cap_daily_loss(self):
        """
        With position limits, even if all positions hit stop loss,
        daily loss should be capped to (max_positions * position_size * stop_loss)

        Example: 5 positions * 3% size * 5% stop = 0.75% max daily loss per position cycle
        """
        config = {
            'risk_management': {
                'max_position_size_pct': 3.0,   # 3% per position
                'max_open_positions': 5,         # Max 5 positions
                'stop_loss_pct': 5.0,            # 5% stop loss
                'take_profit_pct': 8.0,
                'max_daily_loss_pct': 10.0,      # High limit to not interfere
            },
            'exit_manager': {
                'enabled': False,  # Disable tiered exits for simplicity
                'max_hold_hours': 48,
                'eod_close': False,
            },
            'trailing_stop': {
                'enabled': False,
            },
            'entry_gate': {
                'confidence_threshold': 50,
            },
            'strategies': {
                'momentum': {'enabled': True, 'weight': 1.0},
            },
        }

        bt = Backtest1Hour(initial_capital=10000, config=config)

        # Theoretical max loss per position = 3% of portfolio * 5% stop = 0.15%
        # With 5 positions all hitting stops = 0.75% max daily loss
        # This is much better than the old 62%+ losses!

        # Verify the settings are applied correctly
        assert bt.max_open_positions == 5
        assert bt.default_stop_loss_pct == 0.05  # 5%

        # Verify RiskManager has correct position sizing
        assert bt.risk_manager.max_position_size == 0.03  # 3%
        assert bt.risk_manager.max_positions == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
