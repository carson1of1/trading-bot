"""Tests for Bracket Order functionality (ODE-117)

Tests cover:
- BrokerInterface.submit_bracket_order abstract method
- AlpacaBroker bracket order submission (mocked)
- FakeBroker bracket order simulation
- Stop order tracking and cancellation
- Stop trigger logic in FakeBroker
"""
import pytest
import os
import sys
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.broker import (
    Position, Order, Account,
    BrokerInterface, FakeBroker, AlpacaBroker,
    BrokerAPIError
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def allow_alpaca_in_tests():
    """Fixture to temporarily allow AlpacaBroker in test environment.

    This is used for tests that intentionally test AlpacaBroker with mocked API.
    The fixture ensures the flag is reset after the test.
    """
    original = AlpacaBroker._allow_in_tests
    AlpacaBroker._allow_in_tests = True
    yield
    AlpacaBroker._allow_in_tests = original


# =============================================================================
# FAKEBROKER BRACKET ORDER TESTS
# =============================================================================

class TestFakeBrokerBracketOrderSubmission:
    """Test FakeBroker bracket order submission"""

    def test_submit_bracket_order_long_creates_position_and_stop(self):
        """LONG bracket order should create position and stop order"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        order = broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            stop_loss_percent=0.05,
            price=100.0
        )

        # Entry should be filled
        assert order.status == 'filled'
        assert order.symbol == 'AAPL'
        assert order.qty == 10

        # Position should exist
        pos = broker.get_position('AAPL')
        assert pos is not None
        assert pos.qty == 10
        assert pos.side == 'long'

        # Stop order should be tracked
        assert 'AAPL' in broker.stop_orders
        stop = broker.stop_orders['AAPL']
        assert stop['stop_price'] == pytest.approx(95.0, rel=0.001)  # 100 * (1 - 0.05)
        assert stop['side'] == 'sell'
        assert stop['qty'] == 10

    def test_submit_bracket_order_short_creates_position_and_stop(self):
        """SHORT bracket order should create position and stop order"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        order = broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='sell',
            stop_loss_percent=0.05,
            price=100.0
        )

        # Entry should be filled
        assert order.status == 'filled'

        # Position should be SHORT
        pos = broker.get_position('AAPL')
        assert pos is not None
        assert pos.side == 'short'

        # Stop order should be BUY at higher price (to cover short)
        assert 'AAPL' in broker.stop_orders
        stop = broker.stop_orders['AAPL']
        assert stop['stop_price'] == pytest.approx(105.0, rel=0.001)  # 100 * (1 + 0.05)
        assert stop['side'] == 'buy'

    def test_submit_bracket_order_default_stop_loss_percent(self):
        """Default stop_loss_percent should be 5%"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        order = broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            price=100.0
        )

        stop = broker.stop_orders['AAPL']
        assert stop['stop_price'] == pytest.approx(95.0, rel=0.001)  # 5% below

    def test_submit_bracket_order_custom_stop_loss_percent(self):
        """Custom stop_loss_percent should be respected"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        order = broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            stop_loss_percent=0.10,  # 10%
            price=100.0
        )

        stop = broker.stop_orders['AAPL']
        assert stop['stop_price'] == pytest.approx(90.0, rel=0.001)  # 10% below

    def test_submit_bracket_order_returns_stop_order_id(self):
        """Bracket order should return stop_order_id attribute"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        order = broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            price=100.0
        )

        assert hasattr(order, 'stop_order_id')
        assert order.stop_order_id is not None
        assert order.stop_order_id == broker.stop_orders['AAPL']['order_id']

    def test_submit_bracket_order_rejected_insufficient_funds(self):
        """Bracket order should be rejected if insufficient funds"""
        broker = FakeBroker(initial_cash=1000, slippage=0)
        order = broker.submit_bracket_order(
            symbol='AAPL',
            qty=100,  # $10,000 > $1,000 cash
            side='buy',
            price=100.0
        )

        assert order.status == 'rejected'
        assert 'AAPL' not in broker.stop_orders

    def test_submit_bracket_order_with_slippage(self):
        """Bracket order should apply slippage to entry but not stop calculation"""
        broker = FakeBroker(initial_cash=100000, slippage=0.01)  # 1% slippage
        order = broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            stop_loss_percent=0.05,
            price=100.0
        )

        # Entry filled with slippage
        assert order.filled_avg_price == pytest.approx(101.0, rel=0.001)

        # Stop should be 5% below ORIGINAL price, not slipped price
        # This ensures consistent stop placement
        stop = broker.stop_orders['AAPL']
        assert stop['stop_price'] == pytest.approx(95.0, rel=0.001)


class TestFakeBrokerStopTrigger:
    """Test FakeBroker stop order triggering"""

    def test_stop_triggers_on_price_drop_long(self):
        """LONG position stop should trigger when price drops to stop level"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            stop_loss_percent=0.05,
            price=100.0
        )

        # Position exists
        assert broker.get_position('AAPL') is not None

        # Price drops to stop level
        broker.update_price('AAPL', 95.0)

        # Stop should have triggered - position closed
        assert broker.get_position('AAPL') is None
        assert 'AAPL' not in broker.stop_orders

    def test_stop_triggers_on_price_rise_short(self):
        """SHORT position stop should trigger when price rises to stop level"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='sell',
            stop_loss_percent=0.05,
            price=100.0
        )

        # Position exists
        assert broker.get_position('AAPL') is not None

        # Price rises to stop level
        broker.update_price('AAPL', 105.0)

        # Stop should have triggered - position closed
        assert broker.get_position('AAPL') is None
        assert 'AAPL' not in broker.stop_orders

    def test_stop_does_not_trigger_above_stop_price_long(self):
        """LONG stop should NOT trigger if price stays above stop"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            stop_loss_percent=0.05,
            price=100.0
        )

        # Price drops but stays above stop (95)
        broker.update_price('AAPL', 96.0)

        # Position should still exist
        assert broker.get_position('AAPL') is not None
        assert 'AAPL' in broker.stop_orders

    def test_stop_does_not_trigger_below_stop_price_short(self):
        """SHORT stop should NOT trigger if price stays below stop"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='sell',
            stop_loss_percent=0.05,
            price=100.0
        )

        # Price rises but stays below stop (105)
        broker.update_price('AAPL', 104.0)

        # Position should still exist
        assert broker.get_position('AAPL') is not None
        assert 'AAPL' in broker.stop_orders

    def test_stop_triggers_on_gap_through_long(self):
        """LONG stop should trigger on gap through stop price"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            stop_loss_percent=0.05,
            price=100.0
        )

        # Price gaps below stop (95)
        broker.update_price('AAPL', 90.0)

        # Stop should have triggered
        assert broker.get_position('AAPL') is None

    def test_stop_triggers_on_gap_through_short(self):
        """SHORT stop should trigger on gap through stop price"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='sell',
            stop_loss_percent=0.05,
            price=100.0
        )

        # Price gaps above stop (105)
        broker.update_price('AAPL', 110.0)

        # Stop should have triggered
        assert broker.get_position('AAPL') is None


class TestFakeBrokerStopCancellation:
    """Test FakeBroker stop order cancellation"""

    def test_cancel_stop_order_removes_from_tracking(self):
        """Cancelling stop order should remove it from tracking"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        order = broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            price=100.0
        )

        stop_order_id = order.stop_order_id
        assert 'AAPL' in broker.stop_orders

        # Cancel the stop order
        result = broker.cancel_order(stop_order_id)
        assert result is True
        assert 'AAPL' not in broker.stop_orders

    def test_close_position_cancels_stop_order(self):
        """Closing position should cancel associated stop order"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            price=100.0
        )

        assert 'AAPL' in broker.stop_orders

        # Close the position
        broker.close_position('AAPL')

        # Stop order should be cancelled
        assert 'AAPL' not in broker.stop_orders

    def test_manual_exit_clears_stop_order(self):
        """Manual sell order should clear stop order"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            price=100.0
        )

        assert 'AAPL' in broker.stop_orders

        # Manually sell to exit
        broker.submit_order('AAPL', 10, 'sell', 'market', price=110.0)

        # Position closed, stop should be cleared
        assert broker.get_position('AAPL') is None
        assert 'AAPL' not in broker.stop_orders


class TestFakeBrokerStopOrdersList:
    """Test FakeBroker stop orders in list_orders"""

    def test_stop_orders_appear_in_open_orders(self):
        """Stop orders should appear in get_open_orders"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            price=100.0
        )

        open_orders = broker.get_open_orders()
        stop_orders = [o for o in open_orders if o.type == 'stop']
        assert len(stop_orders) == 1
        assert stop_orders[0].symbol == 'AAPL'
        assert stop_orders[0].side == 'sell'

    def test_multiple_bracket_orders_track_separate_stops(self):
        """Multiple positions should have separate stop orders"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        broker.submit_bracket_order('AAPL', 10, 'buy', price=100.0)
        broker.submit_bracket_order('NVDA', 5, 'buy', price=200.0)

        assert 'AAPL' in broker.stop_orders
        assert 'NVDA' in broker.stop_orders
        assert broker.stop_orders['AAPL']['stop_price'] == pytest.approx(95.0)
        assert broker.stop_orders['NVDA']['stop_price'] == pytest.approx(190.0)


# =============================================================================
# ALPACABROKER BRACKET ORDER TESTS (MOCKED)
# =============================================================================

@pytest.mark.usefixtures('allow_alpaca_in_tests')
class TestAlpacaBrokerBracketOrder:
    """Test AlpacaBroker bracket order submission with mocked API"""

    @patch('alpaca_trade_api.REST')
    def test_submit_bracket_order_calls_api_correctly(self, mock_rest_class):
        """Bracket order should call Alpaca API with correct params"""
        # Setup mock
        mock_api = MagicMock()
        mock_rest_class.return_value = mock_api

        # Mock order response
        mock_order = MagicMock()
        mock_order.id = 'ORDER_123'
        mock_order.symbol = 'AAPL'
        mock_order.qty = '10'
        mock_order.side = 'buy'
        mock_order.type = 'market'
        mock_order.status = 'accepted'
        mock_order.order_class = 'bracket'
        mock_order.limit_price = None
        mock_order.stop_price = None
        mock_order.legs = [
            MagicMock(id='STOP_123', type='stop', stop_price='95.00'),
            MagicMock(id='TP_123', type='limit', limit_price='200.00')
        ]
        mock_order.submitted_at = datetime.now(pytz.UTC)
        mock_api.submit_order.return_value = mock_order

        broker = AlpacaBroker('key', 'secret', 'https://paper-api.alpaca.markets')
        order = broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            stop_loss_percent=0.05,
            price=100.0
        )

        # Verify API called with bracket order params
        mock_api.submit_order.assert_called_once()
        call_kwargs = mock_api.submit_order.call_args.kwargs
        assert call_kwargs['symbol'] == 'AAPL'
        assert call_kwargs['qty'] == 10
        assert call_kwargs['side'] == 'buy'
        assert call_kwargs['type'] == 'market'
        assert call_kwargs['order_class'] == 'bracket'
        assert 'stop_loss' in call_kwargs
        assert call_kwargs['stop_loss']['stop_price'] == pytest.approx(95.0, rel=0.01)

    @patch('alpaca_trade_api.REST')
    def test_submit_bracket_order_short_calculates_stop_correctly(self, mock_rest_class):
        """SHORT bracket order should calculate stop above entry"""
        mock_api = MagicMock()
        mock_rest_class.return_value = mock_api

        mock_order = MagicMock()
        mock_order.id = 'ORDER_123'
        mock_order.symbol = 'AAPL'
        mock_order.qty = '10'
        mock_order.side = 'sell'
        mock_order.type = 'market'
        mock_order.status = 'accepted'
        mock_order.order_class = 'bracket'
        mock_order.limit_price = None
        mock_order.stop_price = None
        mock_order.legs = []
        mock_order.submitted_at = datetime.now(pytz.UTC)
        mock_api.submit_order.return_value = mock_order

        broker = AlpacaBroker('key', 'secret', 'https://paper-api.alpaca.markets')
        broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='sell',
            stop_loss_percent=0.05,
            price=100.0
        )

        call_kwargs = mock_api.submit_order.call_args.kwargs
        assert call_kwargs['stop_loss']['stop_price'] == pytest.approx(105.0, rel=0.01)

    @patch('alpaca_trade_api.REST')
    def test_submit_bracket_order_uses_gtc_time_in_force(self, mock_rest_class):
        """Bracket order should use GTC time-in-force for stop persistence"""
        mock_api = MagicMock()
        mock_rest_class.return_value = mock_api

        mock_order = MagicMock()
        mock_order.id = 'ORDER_123'
        mock_order.symbol = 'AAPL'
        mock_order.qty = '10'
        mock_order.side = 'buy'
        mock_order.type = 'market'
        mock_order.status = 'accepted'
        mock_order.order_class = 'bracket'
        mock_order.limit_price = None
        mock_order.stop_price = None
        mock_order.legs = []
        mock_order.submitted_at = datetime.now(pytz.UTC)
        mock_api.submit_order.return_value = mock_order

        broker = AlpacaBroker('key', 'secret', 'https://paper-api.alpaca.markets')
        broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            price=100.0
        )

        call_kwargs = mock_api.submit_order.call_args.kwargs
        assert call_kwargs['time_in_force'] == 'gtc'

    @patch('alpaca_trade_api.REST')
    def test_submit_bracket_order_returns_stop_order_id(self, mock_rest_class):
        """Bracket order should return stop_order_id from legs"""
        mock_api = MagicMock()
        mock_rest_class.return_value = mock_api

        mock_order = MagicMock()
        mock_order.id = 'ORDER_123'
        mock_order.symbol = 'AAPL'
        mock_order.qty = '10'
        mock_order.side = 'buy'
        mock_order.type = 'market'
        mock_order.status = 'accepted'
        mock_order.order_class = 'bracket'
        mock_order.limit_price = None
        mock_order.stop_price = None
        mock_order.legs = [
            MagicMock(id='STOP_456', type='stop', stop_price='95.00'),
            MagicMock(id='TP_789', type='limit', limit_price='200.00')
        ]
        mock_order.submitted_at = datetime.now(pytz.UTC)
        mock_api.submit_order.return_value = mock_order

        broker = AlpacaBroker('key', 'secret', 'https://paper-api.alpaca.markets')
        order = broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            price=100.0
        )

        assert hasattr(order, 'stop_order_id')
        assert order.stop_order_id == 'STOP_456'


# =============================================================================
# BROKERINTERFACE ABSTRACT METHOD TESTS
# =============================================================================

class TestBrokerInterfaceBracketOrder:
    """Test that BrokerInterface requires submit_bracket_order"""

    def test_submit_bracket_order_is_abstract(self):
        """submit_bracket_order should be abstract method"""
        # Create a concrete class that doesn't implement submit_bracket_order
        class IncompleteBroker(BrokerInterface):
            def get_account(self): pass
            def get_positions(self): pass
            def list_positions(self): pass
            def get_position(self, symbol): pass
            def get_open_orders(self): pass
            def list_orders(self, status): pass
            def submit_order(self, symbol, qty, side, type, time_in_force, limit_price, stop_price, **kwargs): pass
            def cancel_order(self, order_id): pass
            def cancel_all_orders(self): pass
            def close_position(self, symbol): pass
            def close_all_positions(self): pass
            def get_broker_name(self): pass
            def get_portfolio_history(self, period): pass
            # Missing: submit_bracket_order

        # Should raise TypeError on instantiation
        with pytest.raises(TypeError):
            IncompleteBroker()


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestBracketOrderEdgeCases:
    """Test edge cases for bracket orders"""

    def test_bracket_order_with_zero_stop_loss_percent(self):
        """0% stop_loss should still work (stop at entry price)"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        order = broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            stop_loss_percent=0.0,
            price=100.0
        )

        stop = broker.stop_orders['AAPL']
        assert stop['stop_price'] == pytest.approx(100.0)

    def test_bracket_order_very_small_stop_loss_percent(self):
        """Very small stop_loss_percent should work"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        order = broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            stop_loss_percent=0.001,  # 0.1%
            price=100.0
        )

        stop = broker.stop_orders['AAPL']
        assert stop['stop_price'] == pytest.approx(99.9)

    def test_replacing_existing_position_updates_stop(self):
        """Opening new bracket position for existing symbol should update stop"""
        broker = FakeBroker(initial_cash=100000, slippage=0)

        # First entry
        broker.submit_bracket_order('AAPL', 10, 'buy', price=100.0)
        assert broker.stop_orders['AAPL']['stop_price'] == pytest.approx(95.0)

        # Close position
        broker.submit_order('AAPL', 10, 'sell', 'market', price=110.0)
        assert 'AAPL' not in broker.stop_orders

        # Re-enter at different price
        broker.submit_bracket_order('AAPL', 10, 'buy', price=120.0)
        assert broker.stop_orders['AAPL']['stop_price'] == pytest.approx(114.0)

    def test_partial_fill_stop_quantity(self):
        """If entry partially fills, stop should match filled qty"""
        # This is an edge case for real broker - FakeBroker always fully fills
        # Just verify the stop qty matches order qty
        broker = FakeBroker(initial_cash=100000, slippage=0)
        order = broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            price=100.0
        )

        assert broker.stop_orders['AAPL']['qty'] == order.filled_qty


# =============================================================================
# INTEGRATION TESTS - Full Entry/Exit Cycle
# =============================================================================

class TestBracketOrderIntegration:
    """Integration tests for full entry/exit lifecycle with bracket orders"""

    def test_full_long_cycle_with_bracket_order(self):
        """Test complete LONG position lifecycle: entry -> monitor -> exit"""
        broker = FakeBroker(initial_cash=100000, slippage=0)

        # 1. Entry with bracket order
        entry_order = broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            stop_loss_percent=0.05,
            price=100.0
        )

        assert entry_order.status == 'filled'
        assert entry_order.stop_order_id is not None
        assert broker.get_position('AAPL') is not None
        assert 'AAPL' in broker.stop_orders

        # 2. Price moves up (no stop trigger)
        broker.update_price('AAPL', 105.0)
        assert broker.get_position('AAPL') is not None  # Still open

        # 3. Manual exit at profit
        exit_order = broker.submit_order('AAPL', 10, 'sell', 'market', price=105.0)
        assert exit_order.status == 'filled'

        # 4. Verify position closed and stop order cleared
        assert broker.get_position('AAPL') is None
        assert 'AAPL' not in broker.stop_orders

    def test_full_short_cycle_with_bracket_order(self):
        """Test complete SHORT position lifecycle: entry -> monitor -> exit"""
        broker = FakeBroker(initial_cash=100000, slippage=0)

        # 1. Entry with bracket order (short)
        entry_order = broker.submit_bracket_order(
            symbol='AAPL',
            qty=10,
            side='sell',
            stop_loss_percent=0.05,
            price=100.0
        )

        assert entry_order.status == 'filled'
        assert entry_order.stop_order_id is not None
        pos = broker.get_position('AAPL')
        assert pos is not None
        assert pos.side == 'short'
        assert 'AAPL' in broker.stop_orders
        assert broker.stop_orders['AAPL']['side'] == 'buy'  # Stop is to cover

        # 2. Price moves down (favorable, no stop)
        broker.update_price('AAPL', 95.0)
        assert broker.get_position('AAPL') is not None

        # 3. Cover at profit
        exit_order = broker.submit_order('AAPL', 10, 'buy', 'market', price=95.0)
        assert exit_order.status == 'filled'

        # 4. Verify cleaned up
        assert broker.get_position('AAPL') is None
        assert 'AAPL' not in broker.stop_orders

    def test_stop_triggers_during_crash_simulation(self):
        """Test that broker stop catches disaster when 'bot is down'"""
        broker = FakeBroker(initial_cash=100000, slippage=0)

        # Entry with bracket order
        broker.submit_bracket_order('AAPL', 10, 'buy', price=100.0)
        initial_cash_after_entry = broker.cash

        # Simulate crash: bot goes down, price plummets
        # In real scenario, broker would execute the stop
        broker.update_price('AAPL', 90.0)  # Below 5% stop

        # Stop should have triggered, position closed
        assert broker.get_position('AAPL') is None
        assert 'AAPL' not in broker.stop_orders

        # Cash recovered (minus loss)
        assert broker.cash > initial_cash_after_entry  # Got proceeds from stop exit

    def test_multiple_positions_independent_stops(self):
        """Test that multiple positions have independent stop tracking"""
        broker = FakeBroker(initial_cash=100000, slippage=0)

        # Open two positions
        broker.submit_bracket_order('AAPL', 10, 'buy', price=100.0)
        broker.submit_bracket_order('NVDA', 5, 'buy', price=200.0)

        assert 'AAPL' in broker.stop_orders
        assert 'NVDA' in broker.stop_orders

        # AAPL stop triggers
        broker.update_price('AAPL', 95.0)

        # AAPL closed, NVDA still open
        assert broker.get_position('AAPL') is None
        assert broker.get_position('NVDA') is not None
        assert 'AAPL' not in broker.stop_orders
        assert 'NVDA' in broker.stop_orders

    def test_cancel_stop_before_exit(self):
        """Test manual stop cancellation before exit works correctly"""
        broker = FakeBroker(initial_cash=100000, slippage=0)

        entry_order = broker.submit_bracket_order('AAPL', 10, 'buy', price=100.0)
        stop_order_id = entry_order.stop_order_id

        # Cancel stop manually (simulating execute_exit flow)
        broker.cancel_order(stop_order_id)
        assert 'AAPL' not in broker.stop_orders

        # Exit manually
        broker.submit_order('AAPL', 10, 'sell', 'market', price=105.0)

        assert broker.get_position('AAPL') is None

    def test_close_position_auto_cancels_stop(self):
        """Test that close_position cancels associated stop order"""
        broker = FakeBroker(initial_cash=100000, slippage=0)

        broker.submit_bracket_order('AAPL', 10, 'buy', price=100.0)
        assert 'AAPL' in broker.stop_orders

        # Use close_position helper
        broker.close_position('AAPL')

        assert broker.get_position('AAPL') is None
        assert 'AAPL' not in broker.stop_orders


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
