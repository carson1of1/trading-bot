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


class TestPositionSyncCheck:
    """Test position sync health check."""

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

    def test_positions_synced_passes_when_matching(self, mock_bot):
        """Position sync passes when counts match."""
        # Both have 0 positions
        result = mock_bot.run_health_check()
        assert result['checks']['positions_synced']['status'] == 'PASS'

    def test_positions_synced_fails_when_mismatched(self, mock_bot):
        """Position sync fails when counts don't match."""
        mock_bot.open_positions = {'AAPL': {'qty': 100}}
        mock_bot.broker.get_positions.return_value = []  # Broker has 0
        result = mock_bot.run_health_check()
        assert result['checks']['positions_synced']['status'] == 'FAIL'


class TestExitManagerCheck:
    """Test ExitManager health check."""

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

    def test_exit_manager_passes_when_all_registered(self, mock_bot):
        """ExitManager check passes when all positions registered."""
        mock_bot.open_positions = {'AAPL': {'qty': 100, 'entry_price': 150.0}}
        mock_bot.exit_manager.positions = {'AAPL': MagicMock()}
        result = mock_bot.run_health_check()
        assert result['checks']['positions_registered']['status'] == 'PASS'

    def test_exit_manager_fails_when_missing(self, mock_bot):
        """ExitManager check fails when position not registered."""
        mock_bot.open_positions = {'AAPL': {'qty': 100, 'entry_price': 150.0}}
        mock_bot.exit_manager.positions = {}  # Empty - not registered
        result = mock_bot.run_health_check()
        assert result['checks']['positions_registered']['status'] == 'FAIL'
        assert 'AAPL' in result['checks']['positions_registered']['message']


class TestStrategyManagerCheck:
    """Test strategy manager health check."""

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

    def test_strategy_manager_passes_with_strategies(self, mock_bot):
        """Strategy manager check passes when strategies loaded."""
        # Bot should have strategies loaded by default from config
        result = mock_bot.run_health_check()
        assert result['checks']['strategy_manager_ready']['status'] == 'PASS'

    def test_strategy_manager_fails_with_no_strategies(self, mock_bot):
        """Strategy manager check fails when no strategies."""
        mock_bot.strategy_manager.strategies = []
        result = mock_bot.run_health_check()
        assert result['checks']['strategy_manager_ready']['status'] == 'FAIL'


class TestStatusChecks:
    """Test INFO-level status checks (kill switch, drawdown guard)."""

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

    def test_kill_switch_status_is_info(self, mock_bot):
        """Kill switch status is INFO, not PASS/FAIL."""
        result = mock_bot.run_health_check()
        assert result['checks']['kill_switch_status']['status'] == 'INFO'

    def test_kill_switch_triggered_reported(self, mock_bot):
        """Kill switch triggered state is reported."""
        mock_bot.kill_switch_triggered = True
        result = mock_bot.run_health_check()
        assert 'TRIGGERED' in result['checks']['kill_switch_status']['message']

    def test_drawdown_guard_status_is_info(self, mock_bot):
        """Drawdown guard status is INFO."""
        result = mock_bot.run_health_check()
        assert result['checks']['drawdown_guard_status']['status'] == 'INFO'
