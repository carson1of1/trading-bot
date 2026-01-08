"""Tests for preflight checklist."""
import os
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestCheckResult:
    """Test CheckResult namedtuple."""

    def test_check_result_fields(self):
        """CheckResult has name, passed, and message fields."""
        from core.preflight import CheckResult

        result = CheckResult(name="test_check", passed=True, message="All good")

        assert result.name == "test_check"
        assert result.passed is True
        assert result.message == "All good"

    def test_check_result_failed(self):
        """CheckResult can represent failed check."""
        from core.preflight import CheckResult

        result = CheckResult(name="api_keys", passed=False, message="Missing API key")

        assert result.passed is False


class TestCheckApiKeys:
    """Test API key validation."""

    def test_check_api_keys_present(self):
        """Passes when both API keys are set."""
        from core.preflight import PreflightChecklist

        with patch.dict(os.environ, {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret'
        }):
            checklist = PreflightChecklist({}, MagicMock())
            result = checklist.check_api_keys()

            assert result.passed is True
            assert result.name == "api_keys"
            assert "loaded" in result.message.lower()

    def test_check_api_keys_missing_key(self):
        """Fails when API key is missing."""
        from core.preflight import PreflightChecklist

        with patch.dict(os.environ, {'ALPACA_SECRET_KEY': 'test_secret'}, clear=True):
            # Ensure ALPACA_API_KEY is not set
            os.environ.pop('ALPACA_API_KEY', None)
            checklist = PreflightChecklist({}, MagicMock())
            result = checklist.check_api_keys()

            assert result.passed is False
            assert "ALPACA_API_KEY" in result.message

    def test_check_api_keys_missing_secret(self):
        """Fails when secret key is missing."""
        from core.preflight import PreflightChecklist

        with patch.dict(os.environ, {'ALPACA_API_KEY': 'test_key'}, clear=True):
            os.environ.pop('ALPACA_SECRET_KEY', None)
            checklist = PreflightChecklist({}, MagicMock())
            result = checklist.check_api_keys()

            assert result.passed is False
            assert "ALPACA_SECRET_KEY" in result.message

    def test_check_api_keys_empty_value(self):
        """Fails when API key is empty string."""
        from core.preflight import PreflightChecklist

        with patch.dict(os.environ, {
            'ALPACA_API_KEY': '',
            'ALPACA_SECRET_KEY': 'test_secret'
        }):
            checklist = PreflightChecklist({}, MagicMock())
            result = checklist.check_api_keys()

            assert result.passed is False


class TestCheckNoDuplicateProcess:
    """Test duplicate process detection."""

    def test_no_pid_file(self, tmp_path):
        """Passes when no PID file exists."""
        from core.preflight import PreflightChecklist

        pid_file = tmp_path / "bot.pid"
        checklist = PreflightChecklist({}, MagicMock())

        result = checklist.check_no_duplicate_process(pid_file)

        assert result.passed is True
        assert "no duplicate" in result.message.lower()

    def test_stale_pid_file(self, tmp_path):
        """Passes when PID file exists but process is dead."""
        from core.preflight import PreflightChecklist

        pid_file = tmp_path / "bot.pid"
        pid_file.write_text("999999")  # Non-existent PID

        checklist = PreflightChecklist({}, MagicMock())
        result = checklist.check_no_duplicate_process(pid_file)

        assert result.passed is True
        # Should also clean up stale PID file
        assert not pid_file.exists()

    def test_running_process(self, tmp_path):
        """Fails when another bot process is running."""
        from core.preflight import PreflightChecklist

        pid_file = tmp_path / "bot.pid"
        # Use current process PID (known to be running)
        pid_file.write_text(str(os.getpid()))

        checklist = PreflightChecklist({}, MagicMock())
        result = checklist.check_no_duplicate_process(pid_file)

        assert result.passed is False
        assert "already running" in result.message.lower()

    def test_invalid_pid_file(self, tmp_path):
        """Passes when PID file contains invalid data."""
        from core.preflight import PreflightChecklist

        pid_file = tmp_path / "bot.pid"
        pid_file.write_text("not_a_number")

        checklist = PreflightChecklist({}, MagicMock())
        result = checklist.check_no_duplicate_process(pid_file)

        assert result.passed is True


class TestCheckUniverseLoaded:
    """Test universe file loading."""

    def test_universe_loaded(self, tmp_path):
        """Passes when universe file has symbols."""
        from core.preflight import PreflightChecklist

        universe_file = tmp_path / "universe.yaml"
        universe_file.write_text(yaml.dump({
            'scanner_universe': {
                'tech': ['AAPL', 'MSFT', 'GOOGL']
            }
        }))

        config = {'trading': {'watchlist_file': str(universe_file)}}
        checklist = PreflightChecklist(config, MagicMock())
        checklist.bot_dir = tmp_path

        result = checklist.check_universe_loaded()

        assert result.passed is True
        assert "3 symbols" in result.message

    def test_universe_empty(self, tmp_path):
        """Fails when universe file has no symbols."""
        from core.preflight import PreflightChecklist

        universe_file = tmp_path / "universe.yaml"
        universe_file.write_text(yaml.dump({'scanner_universe': {}}))

        config = {'trading': {'watchlist_file': str(universe_file)}}
        checklist = PreflightChecklist(config, MagicMock())
        checklist.bot_dir = tmp_path

        result = checklist.check_universe_loaded()

        assert result.passed is False
        assert "empty" in result.message.lower()

    def test_universe_missing(self, tmp_path):
        """Fails when universe file doesn't exist."""
        from core.preflight import PreflightChecklist

        config = {'trading': {'watchlist_file': 'nonexistent.yaml'}}
        checklist = PreflightChecklist(config, MagicMock())
        checklist.bot_dir = tmp_path

        result = checklist.check_universe_loaded()

        assert result.passed is False
        assert "not found" in result.message.lower() or "error" in result.message.lower()


class TestCheckAccountBalance:
    """Test account balance validation."""

    def test_account_balance_success(self):
        """Passes when balance is fetched and positive."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_account = MagicMock()
        mock_account.cash = "10000.00"
        mock_account.portfolio_value = "25000.00"
        mock_broker.get_account.return_value = mock_account

        checklist = PreflightChecklist({}, mock_broker)
        result = checklist.check_account_balance()

        assert result.passed is True
        assert "$25,000" in result.message or "25000" in result.message

    def test_account_balance_zero(self):
        """Fails when balance is zero."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_account = MagicMock()
        mock_account.cash = "0"
        mock_account.portfolio_value = "0"
        mock_broker.get_account.return_value = mock_account

        checklist = PreflightChecklist({}, mock_broker)
        result = checklist.check_account_balance()

        assert result.passed is False
        assert "zero" in result.message.lower() or "0" in result.message

    def test_account_balance_api_error(self):
        """Fails when API call fails."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_broker.get_account.side_effect = Exception("API connection failed")

        checklist = PreflightChecklist({}, mock_broker)
        result = checklist.check_account_balance()

        assert result.passed is False
        assert "error" in result.message.lower() or "failed" in result.message.lower()


class TestCheckMarketStatus:
    """Test market status validation."""

    def test_market_open(self):
        """Passes when market is open."""
        from core.preflight import PreflightChecklist

        with patch('core.market_hours.MarketHours') as MockMarketHours:
            mock_mh = MagicMock()
            mock_mh.is_market_open.return_value = True
            MockMarketHours.return_value = mock_mh

            checklist = PreflightChecklist({}, MagicMock())
            result = checklist.check_market_status()

            assert result.passed is True
            assert "open" in result.message.lower()

    def test_market_opens_soon(self):
        """Passes when market opens within 10 minutes."""
        from core.preflight import PreflightChecklist

        with patch('core.market_hours.MarketHours') as MockMarketHours:
            mock_mh = MagicMock()
            mock_mh.is_market_open.return_value = False
            mock_mh.time_until_market_open.return_value = 5  # 5 minutes
            MockMarketHours.return_value = mock_mh

            checklist = PreflightChecklist({}, MagicMock())
            result = checklist.check_market_status()

            assert result.passed is True
            assert "5" in result.message

    def test_market_closed(self):
        """Fails when market is closed and not opening soon."""
        from core.preflight import PreflightChecklist

        with patch('core.market_hours.MarketHours') as MockMarketHours:
            mock_mh = MagicMock()
            mock_mh.is_market_open.return_value = False
            mock_mh.time_until_market_open.return_value = 120  # 2 hours
            MockMarketHours.return_value = mock_mh

            checklist = PreflightChecklist({}, MagicMock())
            result = checklist.check_market_status()

            assert result.passed is False
            assert "closed" in result.message.lower()

    def test_market_opens_in_10_minutes_edge(self):
        """Passes at exactly 10 minutes until open."""
        from core.preflight import PreflightChecklist

        with patch('core.market_hours.MarketHours') as MockMarketHours:
            mock_mh = MagicMock()
            mock_mh.is_market_open.return_value = False
            mock_mh.time_until_market_open.return_value = 10
            MockMarketHours.return_value = mock_mh

            checklist = PreflightChecklist({}, MagicMock())
            result = checklist.check_market_status()

            assert result.passed is True


class TestCheckDailyLossReset:
    """Test daily loss reset validation."""

    def test_daily_loss_reset_success(self):
        """Passes when last_equity is accessible."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_account = MagicMock()
        mock_account.last_equity = "25000.00"
        mock_broker.get_account.return_value = mock_account

        checklist = PreflightChecklist({}, mock_broker)
        result = checklist.check_daily_loss_reset()

        assert result.passed is True
        assert "25,000" in result.message or "25000" in result.message

    def test_daily_loss_reset_no_last_equity(self):
        """Fails when last_equity is not available."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_account = MagicMock(spec=[])  # No last_equity attribute
        mock_broker.get_account.return_value = mock_account

        checklist = PreflightChecklist({}, mock_broker)
        result = checklist.check_daily_loss_reset()

        assert result.passed is False

    def test_daily_loss_reset_api_error(self):
        """Fails when API call fails."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_broker.get_account.side_effect = Exception("Connection refused")

        checklist = PreflightChecklist({}, mock_broker)
        result = checklist.check_daily_loss_reset()

        assert result.passed is False


class TestCheckPositionsAccounted:
    """Test position accounting validation."""

    def test_no_positions(self):
        """Passes when no positions are open."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = []

        checklist = PreflightChecklist({}, mock_broker)
        checklist.watchlist = ['AAPL', 'MSFT']

        result = checklist.check_positions_accounted()

        assert result.passed is True
        assert "0" in result.message

    def test_positions_in_watchlist(self):
        """Passes when all positions are in watchlist."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_pos = MagicMock()
        mock_pos.symbol = 'AAPL'
        mock_broker.get_positions.return_value = [mock_pos]

        checklist = PreflightChecklist({}, mock_broker)
        checklist.watchlist = ['AAPL', 'MSFT', 'GOOGL']

        result = checklist.check_positions_accounted()

        assert result.passed is True
        assert "1" in result.message

    def test_orphaned_positions(self):
        """Fails when positions exist that aren't in watchlist."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_pos = MagicMock()
        mock_pos.symbol = 'TSLA'  # Not in watchlist
        mock_broker.get_positions.return_value = [mock_pos]

        checklist = PreflightChecklist({}, mock_broker)
        checklist.watchlist = ['AAPL', 'MSFT']

        result = checklist.check_positions_accounted()

        assert result.passed is False
        assert "TSLA" in result.message

    def test_positions_api_error(self):
        """Fails when API call fails."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_broker.get_positions.side_effect = Exception("API error")

        checklist = PreflightChecklist({}, mock_broker)
        checklist.watchlist = ['AAPL']

        result = checklist.check_positions_accounted()

        assert result.passed is False


class TestRunAllChecks:
    """Test run_all_checks aggregation."""

    def test_all_checks_pass(self, tmp_path):
        """Returns True when all checks pass."""
        from core.preflight import PreflightChecklist

        # Setup mocks for all checks to pass
        mock_broker = MagicMock()
        mock_account = MagicMock()
        mock_account.portfolio_value = "25000.00"
        mock_account.last_equity = "25000.00"
        mock_broker.get_account.return_value = mock_account
        mock_broker.get_positions.return_value = []

        universe_file = tmp_path / "universe.yaml"
        universe_file.write_text(yaml.dump({
            'scanner_universe': {'tech': ['AAPL', 'MSFT']}
        }))

        config = {'trading': {'watchlist_file': str(universe_file)}}

        with patch.dict(os.environ, {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret'
        }), patch('core.market_hours.MarketHours') as MockMH:
            mock_mh = MagicMock()
            mock_mh.is_market_open.return_value = True
            MockMH.return_value = mock_mh

            checklist = PreflightChecklist(config, mock_broker)
            checklist.bot_dir = tmp_path
            checklist.watchlist = ['AAPL', 'MSFT']

            # Use a non-existent PID file
            pid_file = tmp_path / "bot.pid"

            all_passed, results = checklist.run_all_checks(pid_file=pid_file)

            assert all_passed is True
            assert len(results) == 7
            assert all(r.passed for r in results)

    def test_one_check_fails(self, tmp_path):
        """Returns False when any check fails."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_account = MagicMock()
        mock_account.portfolio_value = "25000.00"
        mock_account.last_equity = "25000.00"
        mock_broker.get_account.return_value = mock_account
        mock_broker.get_positions.return_value = []

        universe_file = tmp_path / "universe.yaml"
        universe_file.write_text(yaml.dump({
            'scanner_universe': {'tech': ['AAPL']}
        }))

        config = {'trading': {'watchlist_file': str(universe_file)}}

        # Missing API key - this will cause one check to fail
        with patch.dict(os.environ, {'ALPACA_SECRET_KEY': 'test_secret'}, clear=True), \
             patch('core.market_hours.MarketHours') as MockMH:
            os.environ.pop('ALPACA_API_KEY', None)
            mock_mh = MagicMock()
            mock_mh.is_market_open.return_value = True
            MockMH.return_value = mock_mh

            checklist = PreflightChecklist(config, mock_broker)
            checklist.bot_dir = tmp_path
            checklist.watchlist = ['AAPL']

            pid_file = tmp_path / "bot.pid"
            all_passed, results = checklist.run_all_checks(pid_file=pid_file)

            assert all_passed is False
            failed = [r for r in results if not r.passed]
            assert len(failed) >= 1

    def test_runs_all_checks_even_if_one_fails(self, tmp_path):
        """Runs all checks even when early ones fail (no short-circuit)."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_broker.get_account.side_effect = Exception("API down")
        mock_broker.get_positions.side_effect = Exception("API down")

        universe_file = tmp_path / "universe.yaml"
        universe_file.write_text(yaml.dump({
            'scanner_universe': {'tech': ['AAPL']}
        }))

        config = {'trading': {'watchlist_file': str(universe_file)}}

        with patch.dict(os.environ, {}, clear=True), \
             patch('core.market_hours.MarketHours') as MockMH:
            mock_mh = MagicMock()
            mock_mh.is_market_open.return_value = False
            mock_mh.time_until_market_open.return_value = 120
            MockMH.return_value = mock_mh

            checklist = PreflightChecklist(config, mock_broker)
            checklist.bot_dir = tmp_path
            checklist.watchlist = ['AAPL']

            pid_file = tmp_path / "bot.pid"
            all_passed, results = checklist.run_all_checks(pid_file=pid_file)

            # Should have run all 7 checks even though many failed
            assert len(results) == 7
            assert all_passed is False


class TestBotIntegration:
    """Test integration with bot.py."""

    def test_run_preflight_method_exists(self):
        """TradingBot has run_preflight method."""
        # Import to verify method exists (don't actually run)
        from bot import TradingBot
        assert hasattr(TradingBot, 'run_preflight')
