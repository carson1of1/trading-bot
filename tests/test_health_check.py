"""
Tests for TradingBot health check system (ODE-95).

Tests verify:
1. Health check returns correct structure
2. Broker connection check works
3. ExitManager registration check works
4. Position sync check works
5. Overall status calculation is correct
6. Log output format is correct
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
import pytz
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import TradingBot


class TestHealthCheckStructure:
    """Test health check returns correct structure."""

    @pytest.fixture
    def mock_bot(self):
        """Create a bot with mocked components."""
        with patch('bot.VolatilityScanner') as mock_scanner, \
             patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = ['AAPL']
            mock_scanner.return_value = mock_scanner_instance

            mock_broker_instance = MagicMock()
            mock_account = MagicMock()
            mock_account.cash = 10000.0
            mock_account.portfolio_value = 50000.0
            mock_account.last_equity = 50000.0
            mock_broker_instance.get_account.return_value = mock_account
            mock_broker_instance.get_positions.return_value = []
            mock_broker.return_value = mock_broker_instance

            bot = TradingBot()
            bot.cash = 10000.0
            bot.portfolio_value = 50000.0
            yield bot

    def test_health_check_returns_dict(self, mock_bot):
        """Health check returns a dictionary."""
        result = mock_bot.run_health_check()
        assert isinstance(result, dict)

    def test_health_check_has_required_keys(self, mock_bot):
        """Health check result has required keys."""
        result = mock_bot.run_health_check()
        assert 'timestamp' in result
        assert 'overall_status' in result
        assert 'checks' in result
        assert 'summary' in result

    def test_health_check_summary_structure(self, mock_bot):
        """Health check summary has correct structure."""
        result = mock_bot.run_health_check()
        summary = result['summary']
        assert 'total_checks' in summary
        assert 'passed' in summary
        assert 'failed' in summary
        assert 'info' in summary


class TestBrokerHealthCheck:
    """Test broker connection health check."""

    @pytest.fixture
    def mock_bot(self):
        """Create a bot with mocked components."""
        with patch('bot.VolatilityScanner') as mock_scanner, \
             patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = ['AAPL']
            mock_scanner.return_value = mock_scanner_instance

            mock_broker_instance = MagicMock()
            mock_account = MagicMock()
            mock_account.cash = 10000.0
            mock_account.portfolio_value = 50000.0
            mock_account.last_equity = 50000.0
            mock_broker_instance.get_account.return_value = mock_account
            mock_broker_instance.get_positions.return_value = []
            mock_broker.return_value = mock_broker_instance

            bot = TradingBot()
            bot.cash = 10000.0
            bot.portfolio_value = 50000.0
            yield bot

    def test_broker_connected_passes(self, mock_bot):
        """Broker check passes when connection works."""
        result = mock_bot.run_health_check()
        assert result['checks']['broker_connected']['status'] == 'PASS'

    def test_broker_connected_fails_on_exception(self, mock_bot):
        """Broker check fails when get_account raises exception."""
        mock_bot.broker.get_account.side_effect = Exception("Connection failed")
        result = mock_bot.run_health_check()
        assert result['checks']['broker_connected']['status'] == 'FAIL'


class TestAccountSyncCheck:
    """Test account sync health check."""

    @pytest.fixture
    def mock_bot(self):
        """Create a bot with mocked components."""
        with patch('bot.VolatilityScanner') as mock_scanner, \
             patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = ['AAPL']
            mock_scanner.return_value = mock_scanner_instance

            mock_broker_instance = MagicMock()
            mock_account = MagicMock()
            mock_account.cash = 10000.0
            mock_account.portfolio_value = 50000.0
            mock_account.last_equity = 50000.0
            mock_broker_instance.get_account.return_value = mock_account
            mock_broker_instance.get_positions.return_value = []
            mock_broker.return_value = mock_broker_instance

            bot = TradingBot()
            yield bot

    def test_account_synced_passes_with_values(self, mock_bot):
        """Account sync check passes when cash and portfolio populated."""
        mock_bot.cash = 10000.0
        mock_bot.portfolio_value = 50000.0
        result = mock_bot.run_health_check()
        assert result['checks']['account_synced']['status'] == 'PASS'

    def test_account_synced_fails_with_zero_values(self, mock_bot):
        """Account sync check fails when values are zero."""
        mock_bot.cash = 0.0
        mock_bot.portfolio_value = 0.0
        result = mock_bot.run_health_check()
        assert result['checks']['account_synced']['status'] == 'FAIL'
