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


class TestDustCleanup:
    """Test automatic cleanup of small 'dust' positions after partial fills."""

    @pytest.fixture
    def bot_with_dust_config(self, tmp_path):
        """Create a bot with dust cleanup enabled."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
risk_management:
  min_position_value: 50
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
             patch('bot.TradeLogger') as mock_logger, \
             patch('bot.YFinanceDataFetcher'):
            mock_broker_instance = MagicMock()
            mock_broker.return_value = mock_broker_instance
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance
            bot = TradingBot(config_path=str(config_path))
            bot.portfolio_value = 100000.0

            # Set up an existing position: 100 shares @ $150
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

    def test_dust_cleanup_triggers_on_small_remaining(self, bot_with_dust_config, caplog):
        """Partial fill leaving < $50 should trigger dust cleanup."""
        bot = bot_with_dust_config

        # First order: partial fill leaves 2 shares at $24 = $48 < $50 threshold
        mock_order_partial = MagicMock()
        mock_order_partial.status = 'filled'
        mock_order_partial.id = 'order_partial'
        mock_order_partial.filled_avg_price = 24.0
        mock_order_partial.filled_qty = 98  # 98 of 100 filled, 2 remaining

        # Second order: dust cleanup order
        mock_order_dust = MagicMock()
        mock_order_dust.status = 'filled'
        mock_order_dust.id = 'order_dust'
        mock_order_dust.filled_avg_price = 24.0
        mock_order_dust.filled_qty = 2

        bot.broker.submit_order.side_effect = [mock_order_partial, mock_order_dust]

        exit_signal = {'exit': True, 'reason': 'stop_loss', 'price': 24.0, 'qty': 100}
        result = bot.execute_exit('AAPL', exit_signal)

        assert result['filled'] is True
        # Position should be fully cleaned up (not left with 2 shares)
        assert 'AAPL' not in bot.open_positions
        # Should have logged dust cleanup
        assert 'DUST_CLEANUP' in caplog.text

    def test_dust_cleanup_skipped_when_above_threshold(self, bot_with_dust_config):
        """Partial fill leaving >= $50 should NOT trigger dust cleanup."""
        bot = bot_with_dust_config

        # Partial fill leaves 10 shares at $150 = $1500, well above threshold
        mock_order = MagicMock()
        mock_order.status = 'filled'
        mock_order.id = 'order456'
        mock_order.filled_avg_price = 150.0
        mock_order.filled_qty = 90  # 90 of 100 filled, 10 remaining
        bot.broker.submit_order.return_value = mock_order

        exit_signal = {'exit': True, 'reason': 'stop_loss', 'price': 150.0, 'qty': 100}
        bot.execute_exit('AAPL', exit_signal)

        # Position should still exist with 10 shares
        assert 'AAPL' in bot.open_positions
        assert bot.open_positions['AAPL']['qty'] == 10
        # Only one order should have been submitted
        assert bot.broker.submit_order.call_count == 1

    def test_dust_cleanup_disabled_when_threshold_zero(self, tmp_path):
        """Dust cleanup should not trigger when min_position_value is 0 or missing."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
risk_management:
  min_position_value: 0
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

            # Partial fill leaving tiny amount
            mock_order = MagicMock()
            mock_order.status = 'filled'
            mock_order.id = 'order456'
            mock_order.filled_avg_price = 10.0
            mock_order.filled_qty = 98  # Leaves 2 shares @ $10 = $20
            bot.broker.submit_order.return_value = mock_order

            exit_signal = {'exit': True, 'reason': 'stop_loss', 'price': 10.0, 'qty': 100}
            bot.execute_exit('AAPL', exit_signal)

            # Position should still exist (no cleanup)
            assert 'AAPL' in bot.open_positions
            assert bot.open_positions['AAPL']['qty'] == 2

    def test_dust_cleanup_logs_pnl_correctly(self, bot_with_dust_config):
        """Dust cleanup should log P&L for the dust portion."""
        bot = bot_with_dust_config

        mock_order_partial = MagicMock()
        mock_order_partial.status = 'filled'
        mock_order_partial.id = 'order_partial'
        mock_order_partial.filled_avg_price = 24.0
        mock_order_partial.filled_qty = 98

        mock_order_dust = MagicMock()
        mock_order_dust.status = 'filled'
        mock_order_dust.id = 'order_dust'
        mock_order_dust.filled_avg_price = 24.0
        mock_order_dust.filled_qty = 2

        bot.broker.submit_order.side_effect = [mock_order_partial, mock_order_dust]

        exit_signal = {'exit': True, 'reason': 'stop_loss', 'price': 24.0, 'qty': 100}
        bot.execute_exit('AAPL', exit_signal)

        # Trade logger should have been called twice (main exit + dust)
        assert bot.trade_logger.log_trade.call_count == 2

        # Check dust trade was logged with correct P&L
        dust_call = bot.trade_logger.log_trade.call_args_list[1]
        dust_trade = dust_call[0][0]
        assert dust_trade['exit_reason'] == 'dust_cleanup'
        assert dust_trade['quantity'] == 2
        # P&L: (24 - 150) * 2 = -252
        assert dust_trade['pnl'] == pytest.approx(-252.0)

    def test_dust_cleanup_handles_failed_order(self, bot_with_dust_config, caplog):
        """Dust cleanup should handle broker errors gracefully."""
        bot = bot_with_dust_config

        mock_order_partial = MagicMock()
        mock_order_partial.status = 'filled'
        mock_order_partial.id = 'order_partial'
        mock_order_partial.filled_avg_price = 24.0
        mock_order_partial.filled_qty = 98

        # Dust cleanup order fails
        bot.broker.submit_order.side_effect = [mock_order_partial, Exception("Broker error")]

        exit_signal = {'exit': True, 'reason': 'stop_loss', 'price': 24.0, 'qty': 100}
        result = bot.execute_exit('AAPL', exit_signal)

        # Original exit should still succeed
        assert result['filled'] is True
        # Error should be logged
        assert 'DUST_CLEANUP' in caplog.text and 'Failed' in caplog.text
        # Position should be cleaned up anyway to avoid stuck state
        assert 'AAPL' not in bot.open_positions
