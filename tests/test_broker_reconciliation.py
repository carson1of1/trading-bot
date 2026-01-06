"""Tests for broker state reconciliation."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime


class TestBrokerReconciliation:
    """Test _reconcile_broker_state() divergence detection."""

    @pytest.fixture
    def bot_with_mocks(self, tmp_path):
        """Create a bot with mocked components."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
risk_management:
  max_daily_loss_pct: 3.0
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
            from bot import TradingBot
            bot = TradingBot(config_path=str(config_path))
            return bot

    def test_ghost_position_logs_warning(self, bot_with_mocks, caplog):
        """Ghost position (internal exists, broker doesn't) logs warning."""
        bot = bot_with_mocks

        # Internal state has AAPL position
        bot.open_positions = {
            'AAPL': {
                'symbol': 'AAPL',
                'qty': 100,
                'entry_price': 150.00,
                'direction': 'LONG',
                'entry_time': datetime.now()
            }
        }

        # Broker returns empty (no positions)
        bot.broker.get_positions.return_value = []

        bot._reconcile_broker_state()

        assert 'RECONCILE | GHOST | AAPL' in caplog.text
        assert 'Internal: 100 shares @ $150.00' in caplog.text
        assert 'Broker: NOT FOUND' in caplog.text

    def test_orphan_position_logs_warning(self, bot_with_mocks, caplog):
        """Orphan position (broker exists, internal doesn't) logs warning."""
        bot = bot_with_mocks

        # Internal state is empty
        bot.open_positions = {}

        # Broker has MSFT position
        mock_position = MagicMock()
        mock_position.symbol = 'MSFT'
        mock_position.qty = 50
        mock_position.avg_entry_price = 400.00
        bot.broker.get_positions.return_value = [mock_position]

        bot._reconcile_broker_state()

        assert 'RECONCILE | ORPHAN | MSFT' in caplog.text
        assert 'Internal: NOT TRACKED' in caplog.text
        assert 'Broker: 50 shares @ $400.00' in caplog.text

    def test_quantity_mismatch_logs_warning(self, bot_with_mocks, caplog):
        """Quantity mismatch logs warning."""
        bot = bot_with_mocks

        # Internal has 100 shares
        bot.open_positions = {
            'AAPL': {
                'symbol': 'AAPL',
                'qty': 100,
                'entry_price': 150.00,
                'direction': 'LONG',
                'entry_time': datetime.now()
            }
        }

        # Broker has 75 shares (partial fill or manual trade)
        mock_position = MagicMock()
        mock_position.symbol = 'AAPL'
        mock_position.qty = 75
        mock_position.avg_entry_price = 150.00
        bot.broker.get_positions.return_value = [mock_position]

        bot._reconcile_broker_state()

        assert 'RECONCILE | QTY_MISMATCH | AAPL' in caplog.text
        assert 'Internal: 100 shares' in caplog.text
        assert 'Broker: 75 shares' in caplog.text

    def test_price_mismatch_above_tolerance_logs_warning(self, bot_with_mocks, caplog):
        """Price mismatch >1% logs warning."""
        bot = bot_with_mocks

        # Internal has entry at $100
        bot.open_positions = {
            'AAPL': {
                'symbol': 'AAPL',
                'qty': 100,
                'entry_price': 100.00,
                'direction': 'LONG',
                'entry_time': datetime.now()
            }
        }

        # Broker has entry at $102 (2% difference)
        mock_position = MagicMock()
        mock_position.symbol = 'AAPL'
        mock_position.qty = 100
        mock_position.avg_entry_price = 102.00
        bot.broker.get_positions.return_value = [mock_position]

        bot._reconcile_broker_state()

        assert 'RECONCILE | PRICE_MISMATCH | AAPL' in caplog.text
        assert 'Internal: $100.00' in caplog.text
        assert 'Broker: $102.00' in caplog.text
        assert 'Diff: 2.00%' in caplog.text

    def test_price_mismatch_within_tolerance_no_warning(self, bot_with_mocks, caplog):
        """Price mismatch <1% does NOT log warning."""
        bot = bot_with_mocks

        # Internal has entry at $100
        bot.open_positions = {
            'AAPL': {
                'symbol': 'AAPL',
                'qty': 100,
                'entry_price': 100.00,
                'direction': 'LONG',
                'entry_time': datetime.now()
            }
        }

        # Broker has entry at $100.50 (0.5% difference - within tolerance)
        mock_position = MagicMock()
        mock_position.symbol = 'AAPL'
        mock_position.qty = 100
        mock_position.avg_entry_price = 100.50
        bot.broker.get_positions.return_value = [mock_position]

        bot._reconcile_broker_state()

        assert 'PRICE_MISMATCH' not in caplog.text

    def test_broker_api_failure_logs_error_and_returns(self, bot_with_mocks, caplog):
        """Broker API failure logs error and returns gracefully."""
        bot = bot_with_mocks

        # Broker raises exception
        bot.broker.get_positions.side_effect = Exception("API timeout")

        # Should not raise, should log error
        bot._reconcile_broker_state()

        assert 'RECONCILE | Failed to fetch broker positions' in caplog.text
        assert 'API timeout' in caplog.text

    def test_no_divergence_no_warnings(self, bot_with_mocks, caplog):
        """When internal matches broker exactly, no warnings logged."""
        bot = bot_with_mocks

        # Internal state
        bot.open_positions = {
            'AAPL': {
                'symbol': 'AAPL',
                'qty': 100,
                'entry_price': 150.00,
                'direction': 'LONG',
                'entry_time': datetime.now()
            }
        }

        # Broker matches exactly
        mock_position = MagicMock()
        mock_position.symbol = 'AAPL'
        mock_position.qty = 100
        mock_position.avg_entry_price = 150.00
        bot.broker.get_positions.return_value = [mock_position]

        bot._reconcile_broker_state()

        assert 'RECONCILE |' not in caplog.text

    def test_empty_positions_no_warnings(self, bot_with_mocks, caplog):
        """When both internal and broker are empty, no warnings logged."""
        bot = bot_with_mocks

        bot.open_positions = {}
        bot.broker.get_positions.return_value = []

        bot._reconcile_broker_state()

        assert 'RECONCILE |' not in caplog.text
