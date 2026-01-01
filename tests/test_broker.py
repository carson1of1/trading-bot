"""Tests for Broker module (FakeBroker, dataclasses, BrokerFactory)

NOTE: AlpacaBroker is NOT tested here - it requires real API keys.
These tests focus on:
- Position, Order, Account dataclasses
- FakeBroker simulation functionality
- BrokerFactory with mocked config
"""
import pytest
import os
import sys
import tempfile
import yaml
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.broker import (
    Position, Order, Account,
    BrokerInterface, FakeBroker, BrokerFactory,
    BrokerAPIError, create_broker
)


# =============================================================================
# DATACLASS TESTS
# =============================================================================

class TestPositionDataclass:
    """Test Position dataclass"""

    def test_position_creation(self):
        """Should create Position with all required fields"""
        pos = Position(
            symbol='AAPL',
            qty=100,
            side='long',
            avg_entry_price=150.0,
            current_price=155.0,
            market_value=15500.0,
            unrealized_pl=500.0,
            unrealized_plpc=0.0333
        )
        assert pos.symbol == 'AAPL'
        assert pos.qty == 100
        assert pos.side == 'long'
        assert pos.avg_entry_price == 150.0
        assert pos.current_price == 155.0
        assert pos.unrealized_pl == 500.0

    def test_position_repr(self):
        """Position repr should show symbol, qty, and P&L"""
        pos = Position(
            symbol='NVDA', qty=50, side='long',
            avg_entry_price=100.0, current_price=110.0,
            market_value=5500.0, unrealized_pl=500.0, unrealized_plpc=0.1
        )
        repr_str = repr(pos)
        assert 'NVDA' in repr_str
        assert '50' in repr_str
        assert '500' in repr_str


class TestOrderDataclass:
    """Test Order dataclass"""

    def test_order_creation(self):
        """Should create Order with required and optional fields"""
        order = Order(
            id='ORDER_001',
            symbol='TSLA',
            qty=10,
            side='buy',
            type='market',
            status='new'
        )
        assert order.id == 'ORDER_001'
        assert order.symbol == 'TSLA'
        assert order.qty == 10
        assert order.side == 'buy'
        assert order.type == 'market'
        assert order.status == 'new'
        assert order.limit_price is None
        assert order.filled_qty == 0

    def test_order_with_limit_price(self):
        """Should create limit order with limit_price"""
        order = Order(
            id='ORDER_002', symbol='AAPL', qty=20,
            side='sell', type='limit', status='new',
            limit_price=175.50
        )
        assert order.limit_price == 175.50

    def test_order_repr(self):
        """Order repr should show id, symbol, side, qty, status"""
        order = Order(
            id='TEST123', symbol='SPY', qty=5,
            side='buy', type='market', status='filled'
        )
        repr_str = repr(order)
        assert 'TEST123' in repr_str
        assert 'SPY' in repr_str
        assert 'buy' in repr_str
        assert 'filled' in repr_str


class TestAccountDataclass:
    """Test Account dataclass"""

    def test_account_creation(self):
        """Should create Account with all fields"""
        account = Account(
            equity=105000.0,
            cash=50000.0,
            buying_power=100000.0,
            portfolio_value=105000.0,
            last_equity=100000.0
        )
        assert account.equity == 105000.0
        assert account.cash == 50000.0
        assert account.last_equity == 100000.0

    def test_daily_pnl_property(self):
        """daily_pnl should calculate equity - last_equity"""
        account = Account(
            equity=105000.0, cash=50000.0,
            buying_power=100000.0, portfolio_value=105000.0,
            last_equity=100000.0
        )
        assert account.daily_pnl == 5000.0

    def test_daily_pnl_percent_property(self):
        """daily_pnl_percent should calculate percentage gain/loss"""
        account = Account(
            equity=110000.0, cash=50000.0,
            buying_power=100000.0, portfolio_value=110000.0,
            last_equity=100000.0
        )
        assert account.daily_pnl_percent == 0.1  # 10% gain

    def test_daily_pnl_percent_zero_equity(self):
        """daily_pnl_percent should return 0 if last_equity is 0"""
        account = Account(
            equity=100.0, cash=100.0,
            buying_power=100.0, portfolio_value=100.0,
            last_equity=0.0
        )
        assert account.daily_pnl_percent == 0.0

    def test_account_repr(self):
        """Account repr should show equity and daily_pnl"""
        account = Account(
            equity=100000.0, cash=50000.0,
            buying_power=100000.0, portfolio_value=100000.0,
            last_equity=95000.0
        )
        repr_str = repr(account)
        assert '100000' in repr_str
        assert '5000' in repr_str  # daily_pnl


# =============================================================================
# FAKEBROKER TESTS
# =============================================================================

class TestFakeBrokerInitialization:
    """Test FakeBroker initialization"""

    def test_default_initialization(self):
        """Should initialize with default values"""
        broker = FakeBroker()
        assert broker.initial_cash == 100000
        assert broker.cash == 100000
        assert broker.commission == 0
        assert broker.slippage == 0.0005
        assert len(broker.positions) == 0
        assert len(broker.orders) == 0

    def test_custom_initialization(self):
        """Should initialize with custom values"""
        broker = FakeBroker(
            initial_cash=50000,
            commission=1.0,
            slippage=0.001
        )
        assert broker.initial_cash == 50000
        assert broker.cash == 50000
        assert broker.commission == 1.0
        assert broker.slippage == 0.001

    def test_get_broker_name(self):
        """Should return 'FakeBroker'"""
        broker = FakeBroker()
        assert broker.get_broker_name() == 'FakeBroker'


class TestFakeBrokerAccount:
    """Test FakeBroker account operations"""

    def test_get_account_initial(self):
        """Should return correct account info initially"""
        broker = FakeBroker(initial_cash=100000)
        account = broker.get_account()

        assert account.equity == 100000
        assert account.cash == 100000
        assert account.buying_power == 100000
        assert account.portfolio_value == 100000
        assert account.last_equity == 100000
        assert account.daily_pnl == 0

    def test_get_account_with_position(self):
        """Account equity should reflect position value"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        order = broker.submit_order('AAPL', 10, 'buy', 'market', price=150.0)

        account = broker.get_account()
        # Cash reduced by 10 * 150 = 1500
        assert account.cash == 100000 - 1500
        # Portfolio includes position value
        assert account.portfolio_value == 100000  # Entry price = current price initially


class TestFakeBrokerPositions:
    """Test FakeBroker position operations"""

    def test_get_positions_empty(self):
        """Should return empty list when no positions"""
        broker = FakeBroker()
        assert broker.get_positions() == []

    def test_list_positions_alias(self):
        """list_positions should be alias for get_positions"""
        broker = FakeBroker()
        assert broker.list_positions() == broker.get_positions()

    def test_get_position_none(self):
        """Should return None for non-existent position"""
        broker = FakeBroker()
        assert broker.get_position('AAPL') is None

    def test_get_position_after_buy(self):
        """Should return position after buying"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        broker.submit_order('AAPL', 10, 'buy', 'market', price=150.0)

        pos = broker.get_position('AAPL')
        assert pos is not None
        assert pos.symbol == 'AAPL'
        assert pos.qty == 10
        assert pos.side == 'long'
        assert pos.avg_entry_price == 150.0


class TestFakeBrokerOrders:
    """Test FakeBroker order operations"""

    def test_submit_buy_order_creates_position(self):
        """BUY order should create LONG position"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        order = broker.submit_order('AAPL', 10, 'buy', 'market', price=150.0)

        assert order.status == 'filled'
        assert order.symbol == 'AAPL'
        assert order.qty == 10
        assert order.side == 'buy'
        assert order.filled_qty == 10
        assert order.filled_avg_price == 150.0

        # Verify position created
        pos = broker.get_position('AAPL')
        assert pos.side == 'long'
        assert pos.qty == 10

    def test_submit_sell_order_closes_position(self):
        """SELL order should close LONG position"""
        broker = FakeBroker(initial_cash=100000, slippage=0)

        # Buy first
        broker.submit_order('AAPL', 10, 'buy', 'market', price=150.0)
        assert broker.get_position('AAPL') is not None

        # Sell to close
        sell_order = broker.submit_order('AAPL', 10, 'sell', 'market', price=160.0)
        assert sell_order.status == 'filled'

        # Position should be closed
        assert broker.get_position('AAPL') is None

    def test_order_with_slippage(self):
        """Orders should apply slippage"""
        broker = FakeBroker(initial_cash=100000, slippage=0.01)  # 1% slippage

        # BUY with slippage: price * (1 + slippage) = 150 * 1.01 = 151.5
        order = broker.submit_order('AAPL', 10, 'buy', 'market', price=150.0)
        assert order.filled_avg_price == pytest.approx(151.5, rel=0.001)

    def test_order_id_generation(self):
        """Orders should have unique IDs"""
        broker = FakeBroker()
        order1 = broker.submit_order('AAPL', 1, 'buy', 'market', price=100.0)
        order2 = broker.submit_order('NVDA', 1, 'buy', 'market', price=100.0)

        assert order1.id != order2.id
        assert order1.id.startswith('FAKE_')
        assert order2.id.startswith('FAKE_')

    def test_get_open_orders(self):
        """Should return only 'new' status orders"""
        broker = FakeBroker()
        # Market orders execute immediately (status=filled), not 'new'
        broker.submit_order('AAPL', 1, 'buy', 'market', price=100.0)
        open_orders = broker.get_open_orders()
        assert len(open_orders) == 0  # Market orders are filled immediately

    def test_list_orders_by_status(self):
        """Should filter orders by status"""
        broker = FakeBroker()
        broker.submit_order('AAPL', 1, 'buy', 'market', price=100.0)

        filled_orders = broker.list_orders(status='filled')
        assert len(filled_orders) == 1
        assert filled_orders[0].status == 'filled'


class TestFakeBrokerOrderValidation:
    """Test FakeBroker order validation"""

    def test_invalid_symbol_rejected(self):
        """Order with invalid symbol should be rejected"""
        broker = FakeBroker()

        # Empty string
        order = broker.submit_order('', 10, 'buy', 'market', price=100.0)
        assert order.status == 'rejected'

        # None (becomes INVALID)
        order = broker.submit_order(None, 10, 'buy', 'market', price=100.0)
        assert order.status == 'rejected'

    def test_invalid_side_rejected(self):
        """Order with invalid side should be rejected"""
        broker = FakeBroker()
        order = broker.submit_order('AAPL', 10, 'hold', 'market', price=100.0)
        assert order.status == 'rejected'

    def test_zero_quantity_rejected(self):
        """Order with fractional qty that rounds to 0 should be rejected"""
        broker = FakeBroker()

        # Fractional qty that rounds to 0
        order = broker.submit_order('AAPL', 0.1, 'buy', 'market', price=100.0)
        assert order.status == 'rejected'

        # Another fractional that rounds down to 0
        order = broker.submit_order('AAPL', 0.9, 'buy', 'market', price=100.0)
        assert order.status == 'rejected'

    def test_insufficient_funds_rejected(self):
        """Order exceeding cash should be rejected"""
        broker = FakeBroker(initial_cash=1000, slippage=0)

        # Try to buy 100 shares at $100 = $10,000 (exceeds $1,000 cash)
        order = broker.submit_order('AAPL', 100, 'buy', 'market', price=100.0)
        assert order.status == 'rejected'

    def test_no_price_rejected(self):
        """Market order without price should be rejected"""
        broker = FakeBroker()
        order = broker.submit_order('AAPL', 10, 'buy', 'market')
        assert order.status == 'rejected'


class TestFakeBrokerCancelOrder:
    """Test FakeBroker cancel operations"""

    def test_cancel_nonexistent_order(self):
        """Cancelling non-existent order should return False"""
        broker = FakeBroker()
        result = broker.cancel_order('FAKE_999999')
        assert result is False

    def test_cancel_filled_order(self):
        """Cancelling filled order should return False"""
        broker = FakeBroker()
        order = broker.submit_order('AAPL', 1, 'buy', 'market', price=100.0)
        assert order.status == 'filled'

        result = broker.cancel_order(order.id)
        assert result is False

    def test_cancel_all_orders(self):
        """cancel_all_orders should cancel all 'new' orders"""
        broker = FakeBroker()
        # Create some limit orders that stay 'new' (not executed)
        # Actually, limit orders also need special handling - market orders are filled
        # For this test, manually add new orders
        order1 = Order(id='TEST1', symbol='AAPL', qty=1, side='buy', type='limit', status='new')
        order2 = Order(id='TEST2', symbol='NVDA', qty=1, side='buy', type='limit', status='new')
        broker.orders['TEST1'] = order1
        broker.orders['TEST2'] = order2

        count = broker.cancel_all_orders()
        assert count == 2
        assert broker.orders['TEST1'].status == 'cancelled'
        assert broker.orders['TEST2'].status == 'cancelled'


class TestFakeBrokerClosePosition:
    """Test FakeBroker close position operations"""

    def test_close_position(self):
        """close_position should sell all shares"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        broker.submit_order('AAPL', 10, 'buy', 'market', price=150.0)

        result = broker.close_position('AAPL')
        assert result is True
        assert broker.get_position('AAPL') is None

    def test_close_nonexistent_position(self):
        """Closing non-existent position should return False"""
        broker = FakeBroker()
        result = broker.close_position('AAPL')
        assert result is False

    def test_close_all_positions(self):
        """close_all_positions should close all open positions"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        broker.submit_order('AAPL', 10, 'buy', 'market', price=150.0)
        broker.submit_order('NVDA', 5, 'buy', 'market', price=200.0)

        assert len(broker.positions) == 2

        count = broker.close_all_positions()
        assert count == 2
        assert len(broker.positions) == 0


class TestFakeBrokerReset:
    """Test FakeBroker reset functionality"""

    def test_reset_clears_state(self):
        """reset() should restore initial state"""
        broker = FakeBroker(initial_cash=100000, slippage=0)

        # Make some trades
        broker.submit_order('AAPL', 10, 'buy', 'market', price=150.0)
        broker.submit_order('NVDA', 5, 'buy', 'market', price=200.0)

        assert len(broker.positions) == 2
        assert broker.cash < 100000
        assert len(broker.orders) > 0

        # Reset
        broker.reset()

        assert broker.cash == 100000
        assert len(broker.positions) == 0
        assert len(broker.orders) == 0
        assert len(broker.execution_log) == 0


class TestFakeBrokerExecutionLog:
    """Test FakeBroker execution logging"""

    def test_execution_log_recorded(self):
        """Executions should be recorded in execution_log"""
        broker = FakeBroker(initial_cash=100000, slippage=0)
        broker.submit_order('AAPL', 10, 'buy', 'market', price=150.0)

        log = broker.get_execution_log()
        assert len(log) == 1
        assert log[0]['symbol'] == 'AAPL'
        assert log[0]['side'] == 'buy'
        assert log[0]['qty'] == 10


# =============================================================================
# BROKERFACTORY TESTS
# =============================================================================

class TestBrokerFactory:
    """Test BrokerFactory"""

    def test_backtest_mode_returns_fakebroker(self):
        """BACKTEST mode should return FakeBroker"""
        config_data = {
            'mode': 'BACKTEST',
            'trading': {},
            'risk_management': {},
            'strategies': {},
            'broker': {
                'fake_broker': {
                    'initial_cash': 50000,
                    'commission_per_trade': 1.0,
                    'slippage_percent': 0.001
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()

            # Mock get_global_config to return our test config
            from core.config import GlobalConfig
            test_config = GlobalConfig(f.name)

            with patch('core.broker.get_global_config', return_value=test_config):
                broker = BrokerFactory.create_broker()

                assert isinstance(broker, FakeBroker)
                assert broker.get_broker_name() == 'FakeBroker'
                assert broker.initial_cash == 50000
                assert broker.commission == 1.0

            os.unlink(f.name)

    def test_dryrun_mode_returns_fakebroker(self):
        """DRY_RUN mode should return FakeBroker"""
        config_data = {
            'mode': 'DRY_RUN',
            'trading': {},
            'risk_management': {},
            'strategies': {}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()

            from core.config import GlobalConfig
            test_config = GlobalConfig(f.name)

            with patch('core.broker.get_global_config', return_value=test_config):
                broker = BrokerFactory.create_broker()
                assert isinstance(broker, FakeBroker)

            os.unlink(f.name)

    def test_create_broker_convenience_function(self):
        """create_broker() should be alias for BrokerFactory.create_broker()"""
        config_data = {
            'mode': 'BACKTEST',
            'trading': {},
            'risk_management': {},
            'strategies': {}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()

            from core.config import GlobalConfig
            test_config = GlobalConfig(f.name)

            with patch('core.broker.get_global_config', return_value=test_config):
                broker = create_broker()
                assert isinstance(broker, FakeBroker)

            os.unlink(f.name)


# =============================================================================
# BROKERINTERFACE TESTS
# =============================================================================

class TestBrokerInterfaceAbstract:
    """Test that BrokerInterface is abstract"""

    def test_cannot_instantiate_directly(self):
        """BrokerInterface should not be instantiatable"""
        with pytest.raises(TypeError):
            BrokerInterface()


# =============================================================================
# BROKERAPI ERROR TESTS
# =============================================================================

class TestBrokerAPIError:
    """Test BrokerAPIError exception"""

    def test_error_with_message(self):
        """Should create error with message"""
        err = BrokerAPIError("Test error message")
        assert str(err) == "Test error message"

    def test_error_with_original_exception(self):
        """Should store original exception"""
        original = ValueError("Original error")
        err = BrokerAPIError("Wrapper message", original_exception=original)
        assert err.original_exception == original


# =============================================================================
# IMPORT TESTS
# =============================================================================

class TestBrokerImports:
    """Test that all broker exports are importable"""

    def test_import_from_core(self):
        """All broker classes should be importable from core"""
        from core import (
            BrokerInterface,
            AlpacaBroker,
            FakeBroker,
            BrokerFactory,
            Position,
            Order,
            Account,
            BrokerAPIError,
            create_broker
        )

        assert BrokerInterface is not None
        assert AlpacaBroker is not None
        assert FakeBroker is not None
        assert BrokerFactory is not None
        assert Position is not None
        assert Order is not None
        assert Account is not None
        assert BrokerAPIError is not None
        assert create_broker is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
