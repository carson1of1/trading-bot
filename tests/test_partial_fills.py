"""
Tests for partial order fill handling.

Verifies:
- Entry partial fills are logged correctly
- Exit partial fills update remaining position quantity
- ExitManager quantity is updated on partial exit fills
- Position is NOT cleaned up when exit only partially fills
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from bot import TradingBot


class TestPartialFillEntry:
    """Test partial fill handling for entry orders."""

    @pytest.fixture
    def bot_with_mocks(self, tmp_path):
        """Create a bot with mocked components."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
risk_management:
  max_position_size_pct: 10.0
  stop_loss_pct: 5.0
logging:
  database: "logs/trades.db"
"""
        universe = """
proven_symbols:
  - AAPL
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config)
        universe_path = tmp_path / "universe.yaml"
        universe_path.write_text(universe)

        with patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):
            mock_broker_instance = MagicMock()
            mock_broker.return_value = mock_broker_instance
            bot = TradingBot(config_path=str(config_path))
            bot.portfolio_value = 100000.0
            return bot

    def test_entry_partial_fill_logs_warning(self, bot_with_mocks, caplog):
        """Entry partial fill should log a warning with filled/requested quantities."""
        bot = bot_with_mocks

        # Mock order with partial fill: requested qty calculated by risk_manager,
        # but only 75 shares filled
        mock_order = MagicMock()
        mock_order.status = 'filled'
        mock_order.id = 'order123'
        mock_order.filled_avg_price = 150.0
        mock_order.filled_qty = 75  # Only 75 of requested filled
        bot.broker.submit_bracket_order.return_value = mock_order

        # Mock risk_manager to return 100 shares
        bot.risk_manager.calculate_position_size = MagicMock(return_value=100)

        result = bot.execute_entry('AAPL', 'LONG', 150.0, 'Momentum', 'Test')

        assert result['filled'] is True
        assert result['qty'] == 75  # Should use filled qty
        assert bot.open_positions['AAPL']['qty'] == 75
        # Check that partial fill was logged
        assert 'PARTIAL_FILL' in caplog.text or 'partial' in caplog.text.lower()

    def test_entry_full_fill_no_warning(self, bot_with_mocks, caplog):
        """Full fill should not log partial fill warning."""
        bot = bot_with_mocks

        mock_order = MagicMock()
        mock_order.status = 'filled'
        mock_order.id = 'order123'
        mock_order.filled_avg_price = 150.0
        mock_order.filled_qty = 100  # Full fill
        bot.broker.submit_bracket_order.return_value = mock_order

        bot.risk_manager.calculate_position_size = MagicMock(return_value=100)

        bot.execute_entry('AAPL', 'LONG', 150.0, 'Momentum', 'Test')

        # Should not have partial fill warning
        assert 'PARTIAL_FILL' not in caplog.text


class TestPartialFillExit:
    """Test partial fill handling for exit orders."""

    @pytest.fixture
    def bot_with_mocks(self, tmp_path):
        """Create a bot with mocked components and an open position."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
logging:
  database: "logs/trades.db"
exit_manager:
  enabled: true
"""
        universe = """
proven_symbols:
  - AAPL
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config)
        universe_path = tmp_path / "universe.yaml"
        universe_path.write_text(universe)

        with patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):
            mock_broker_instance = MagicMock()
            mock_broker.return_value = mock_broker_instance
            bot = TradingBot(config_path=str(config_path))
            bot.portfolio_value = 100000.0

            # Set up an existing position
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 100,
                    'entry_price': 150.0,
                    'direction': 'LONG',
                    'strategy': 'Momentum',
                    'entry_time': datetime.now(),
                }
            }
            bot.highest_prices['AAPL'] = 150.0
            bot.lowest_prices['AAPL'] = 150.0
            bot.trailing_stops['AAPL'] = {'activated': False, 'price': 0.0}

            return bot

    def test_exit_partial_fill_preserves_remaining_position(self, bot_with_mocks):
        """Exit partial fill should preserve remaining position quantity."""
        bot = bot_with_mocks

        # Mock order with partial fill: only 75 of 100 shares filled
        mock_order = MagicMock()
        mock_order.status = 'filled'
        mock_order.id = 'order456'
        mock_order.filled_avg_price = 155.0
        mock_order.filled_qty = 75  # Only 75 filled
        bot.broker.submit_order.return_value = mock_order

        exit_signal = {'exit': True, 'reason': 'stop_loss', 'price': 155.0, 'qty': 100}
        result = bot.execute_exit('AAPL', exit_signal)

        assert result['filled'] is True
        # Position should still exist with remaining 25 shares
        assert 'AAPL' in bot.open_positions
        assert bot.open_positions['AAPL']['qty'] == 25

    def test_exit_partial_fill_updates_exit_manager(self, bot_with_mocks):
        """Exit partial fill should update ExitManager quantity."""
        bot = bot_with_mocks

        # Register position with exit manager
        bot.exit_manager.register_position(
            symbol='AAPL',
            entry_price=150.0,
            quantity=100,
            direction='LONG'
        )

        mock_order = MagicMock()
        mock_order.status = 'filled'
        mock_order.id = 'order456'
        mock_order.filled_avg_price = 155.0
        mock_order.filled_qty = 75  # Only 75 filled
        bot.broker.submit_order.return_value = mock_order

        exit_signal = {'exit': True, 'reason': 'stop_loss', 'price': 155.0, 'qty': 100}
        bot.execute_exit('AAPL', exit_signal)

        # ExitManager should have updated quantity
        assert 'AAPL' in bot.exit_manager.positions
        assert bot.exit_manager.positions['AAPL'].quantity == 25

    def test_exit_full_fill_removes_position(self, bot_with_mocks):
        """Full exit fill should remove position entirely."""
        bot = bot_with_mocks

        mock_order = MagicMock()
        mock_order.status = 'filled'
        mock_order.id = 'order456'
        mock_order.filled_avg_price = 155.0
        mock_order.filled_qty = 100  # Full fill
        bot.broker.submit_order.return_value = mock_order

        exit_signal = {'exit': True, 'reason': 'stop_loss', 'price': 155.0, 'qty': 100}
        result = bot.execute_exit('AAPL', exit_signal)

        assert result['filled'] is True
        # Position should be completely removed
        assert 'AAPL' not in bot.open_positions

    def test_exit_partial_fill_logs_correctly(self, bot_with_mocks, caplog):
        """Exit partial fill should log with filled/remaining quantities."""
        bot = bot_with_mocks

        mock_order = MagicMock()
        mock_order.status = 'filled'
        mock_order.id = 'order456'
        mock_order.filled_avg_price = 155.0
        mock_order.filled_qty = 75  # Partial
        bot.broker.submit_order.return_value = mock_order

        exit_signal = {'exit': True, 'reason': 'stop_loss', 'price': 155.0, 'qty': 100}
        bot.execute_exit('AAPL', exit_signal)

        # Should log partial fill info
        assert 'PARTIAL_FILL' in caplog.text or 'partial' in caplog.text.lower()

    def test_exit_zero_fill_returns_not_filled(self, bot_with_mocks, caplog):
        """Zero fill should return filled=False, not be treated as full fill."""
        bot = bot_with_mocks

        mock_order = MagicMock()
        mock_order.status = 'accepted'  # Order accepted but not filled
        mock_order.id = 'order456'
        mock_order.filled_avg_price = None
        mock_order.filled_qty = 0  # Zero shares filled
        bot.broker.submit_order.return_value = mock_order

        exit_signal = {'exit': True, 'reason': 'stop_loss', 'price': 155.0, 'qty': 100}
        result = bot.execute_exit('AAPL', exit_signal)

        # Should return filled=False
        assert result['filled'] is False
        # Position should still exist unchanged
        assert 'AAPL' in bot.open_positions
        assert bot.open_positions['AAPL']['qty'] == 100
        # Should log the failure
        assert 'EXIT_FAILED' in caplog.text or '0 shares' in caplog.text


class TestExitManagerUpdateQuantity:
    """Test ExitManager.update_quantity() integration."""

    def test_update_quantity_reduces_position(self):
        """update_quantity should reduce tracked position quantity."""
        from core.risk import ExitManager

        exit_mgr = ExitManager()
        exit_mgr.register_position('AAPL', 150.0, 100, direction='LONG')

        result = exit_mgr.update_quantity('AAPL', 25)

        assert result is True
        assert exit_mgr.positions['AAPL'].quantity == 25

    def test_update_quantity_nonexistent_position(self):
        """update_quantity returns False for non-existent position."""
        from core.risk import ExitManager

        exit_mgr = ExitManager()

        result = exit_mgr.update_quantity('NONEXISTENT', 25)

        assert result is False
