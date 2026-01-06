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
