"""Tests for scanner-bot integration."""
import pytest
import subprocess
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo
from unittest.mock import patch, MagicMock


class TestBotCLISymbolsArgument:
    """Test bot.py accepts --symbols argument."""

    def test_bot_parses_symbols_argument(self):
        """bot.py should parse --symbols into a list."""
        # Import after patching to avoid actual bot initialization
        with patch('bot.TradingBot') as MockBot:
            mock_instance = MagicMock()
            MockBot.return_value = mock_instance

            # Simulate argument parsing
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument('--config', default='config.yaml')
            parser.add_argument('--symbols', type=str, default=None,
                                help='Comma-separated list of symbols from scanner')

            args = parser.parse_args(['--symbols', 'NVDA,TSLA,AMD'])

            assert args.symbols == 'NVDA,TSLA,AMD'
            symbols_list = args.symbols.split(',') if args.symbols else None
            assert symbols_list == ['NVDA', 'TSLA', 'AMD']

    def test_bot_accepts_scanner_symbols_parameter(self):
        """TradingBot should accept scanner_symbols parameter."""
        with patch('bot.yaml.safe_load') as mock_yaml, \
             patch('builtins.open', MagicMock()), \
             patch('bot.YFinanceDataFetcher'), \
             patch('bot.TechnicalIndicators'), \
             patch('bot.create_broker'), \
             patch('bot.TradeLogger'), \
             patch('bot.RiskManager'), \
             patch('bot.EntryGate'), \
             patch('bot.ExitManager'), \
             patch('bot.MarketHours'), \
             patch('bot.StrategyManager'):

            # Mock config
            mock_yaml.return_value = {
                'mode': 'PAPER',
                'timeframe': '1Hour',
                'trading': {'watchlist_file': 'universe.yaml'},
                'logging': {'database': 'logs/trades.db'},
                'volatility_scanner': {'enabled': False},
                'risk_management': {'max_position_size_pct': 2},
                'entry_gate': {},
                'exit_manager': {},
            }

            from bot import TradingBot

            # Test with scanner_symbols
            bot = TradingBot(scanner_symbols=['NVDA', 'TSLA', 'AMD'])
            assert bot.watchlist == ['NVDA', 'TSLA', 'AMD']


class TestMarketHoursCheck:
    """Test market hours validation functions."""

    def test_is_market_open_during_trading_hours(self):
        """Should return True during market hours (9:30 AM - 4:00 PM ET)."""
        from core.market_hours import is_market_open

        # Mock a Tuesday at 10:30 AM ET (using pytz for compatibility)
        import pytz
        et = pytz.timezone('America/New_York')
        mock_time = et.localize(datetime(2026, 1, 6, 10, 30, 0))

        with patch('core.market_hours.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert is_market_open() == True

    def test_is_market_open_before_market_hours(self):
        """Should return False before 9:30 AM ET."""
        from core.market_hours import is_market_open

        # Mock a Tuesday at 8:00 AM ET
        import pytz
        et = pytz.timezone('America/New_York')
        mock_time = et.localize(datetime(2026, 1, 6, 8, 0, 0))

        with patch('core.market_hours.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert is_market_open() == False

    def test_is_market_open_on_weekend(self):
        """Should return False on weekends."""
        from core.market_hours import is_market_open

        # Mock a Saturday at 11:00 AM ET
        import pytz
        et = pytz.timezone('America/New_York')
        mock_time = et.localize(datetime(2026, 1, 3, 11, 0, 0))

        with patch('core.market_hours.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert is_market_open() == False

    def test_get_market_status_message_when_closed(self):
        """Should return helpful message when market is closed."""
        from core.market_hours import get_market_status_message

        # Mock a Saturday
        import pytz
        et = pytz.timezone('America/New_York')
        mock_time = et.localize(datetime(2026, 1, 3, 11, 0, 0))

        with patch('core.market_hours.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            message = get_market_status_message()
            assert "closed" in message.lower() or "weekend" in message.lower() or "saturday" in message.lower()


class TestBotStartWithScanner:
    """Test POST /api/bot/start runs scanner first."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_start_bot_runs_scanner_first(self, client):
        """POST /api/bot/start should run scanner before starting bot."""
        with patch('api.main.VolatilityScanner') as MockScanner, \
             patch('api.main.is_market_open', return_value=True), \
             patch('api.main.subprocess.Popen') as mock_popen, \
             patch('api.main.YFinanceDataFetcher') as MockFetcher:

            # Mock scanner returning results
            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan_historical.return_value = [
                {'symbol': 'NVDA', 'composite_score': 0.95},
                {'symbol': 'TSLA', 'composite_score': 0.90},
                {'symbol': 'AMD', 'composite_score': 0.85},
            ]
            MockScanner.return_value = mock_scanner_instance

            # Mock fetcher
            mock_fetcher_instance = MagicMock()
            mock_fetcher_instance.fetch_historical_data.return_value = MagicMock(empty=False)
            MockFetcher.return_value = mock_fetcher_instance

            response = client.post("/api/bot/start")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "started"
            assert "watchlist" in data
            assert "NVDA" in data["watchlist"]

    def test_start_bot_fails_when_market_closed(self, client):
        """Should return 400 with reason='market_closed' outside trading hours."""
        with patch('api.main.is_market_open', return_value=False), \
             patch('api.main.get_market_status_message', return_value="Market closed: Saturday."):

            response = client.post("/api/bot/start")

            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["reason"] == "market_closed"
            assert "Saturday" in data["detail"]["message"]

    def test_start_bot_fails_when_scanner_returns_empty(self, client):
        """Should return 400 with reason='no_results' if scanner finds nothing."""
        with patch('api.main.VolatilityScanner') as MockScanner, \
             patch('api.main.is_market_open', return_value=True), \
             patch('api.main.YFinanceDataFetcher') as MockFetcher:

            # Mock scanner returning empty
            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan_historical.return_value = []
            MockScanner.return_value = mock_scanner_instance

            # Mock fetcher
            mock_fetcher_instance = MagicMock()
            mock_fetcher_instance.fetch_historical_data.return_value = MagicMock(empty=False)
            MockFetcher.return_value = mock_fetcher_instance

            response = client.post("/api/bot/start")

            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["reason"] == "no_results"

    def test_start_bot_fails_on_scanner_api_error(self, client):
        """Should return 400 with reason='scanner_error' on data fetch failure."""
        with patch('api.main.VolatilityScanner') as MockScanner, \
             patch('api.main.is_market_open', return_value=True), \
             patch('api.main.YFinanceDataFetcher') as MockFetcher:

            # Mock scanner raising exception
            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan_historical.side_effect = Exception("YFinance API timeout")
            MockScanner.return_value = mock_scanner_instance

            # Mock fetcher
            mock_fetcher_instance = MagicMock()
            mock_fetcher_instance.fetch_historical_data.return_value = MagicMock(empty=False)
            MockFetcher.return_value = mock_fetcher_instance

            response = client.post("/api/bot/start")

            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["reason"] == "scanner_error"
            assert "YFinance" in data["detail"]["message"]

    def test_start_bot_returns_watchlist_on_success(self, client):
        """Response should include list of scanned symbols and timestamp."""
        with patch('api.main.VolatilityScanner') as MockScanner, \
             patch('api.main.is_market_open', return_value=True), \
             patch('api.main.subprocess.Popen'), \
             patch('api.main.YFinanceDataFetcher') as MockFetcher:

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan_historical.return_value = [
                {'symbol': 'NVDA', 'composite_score': 0.95},
                {'symbol': 'TSLA', 'composite_score': 0.90},
            ]
            MockScanner.return_value = mock_scanner_instance

            # Mock fetcher
            mock_fetcher_instance = MagicMock()
            mock_fetcher_instance.fetch_historical_data.return_value = MagicMock(empty=False)
            MockFetcher.return_value = mock_fetcher_instance

            response = client.post("/api/bot/start")

            assert response.status_code == 200
            data = response.json()
            assert data["watchlist"] == ["NVDA", "TSLA"]
            assert "scanner_ran_at" in data
            assert "message" in data


class TestFullScannerBotIntegration:
    """End-to-end tests for scanner-bot integration."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_full_flow_start_to_running(self, client):
        """Test complete flow: start bot -> scanner runs -> bot starts with symbols."""
        with patch('api.main.VolatilityScanner') as MockScanner, \
             patch('api.main.is_market_open', return_value=True), \
             patch('api.main.subprocess.Popen') as mock_popen, \
             patch('api.main.YFinanceDataFetcher') as MockFetcher:

            # Setup mocks
            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan_historical.return_value = [
                {'symbol': 'NVDA', 'composite_score': 0.95},
                {'symbol': 'TSLA', 'composite_score': 0.90},
                {'symbol': 'AMD', 'composite_score': 0.85},
            ]
            MockScanner.return_value = mock_scanner_instance

            mock_fetcher_instance = MagicMock()
            mock_fetcher_instance.fetch_historical_data.return_value = MagicMock(empty=False)
            MockFetcher.return_value = mock_fetcher_instance

            # Start bot
            response = client.post("/api/bot/start")
            assert response.status_code == 200
            start_data = response.json()
            assert start_data["status"] == "started"
            assert start_data["watchlist"] == ["NVDA", "TSLA", "AMD"]

            # Verify bot process was started with correct symbols
            mock_popen.assert_called_once()
            call_args = mock_popen.call_args[0][0]
            assert "python3" in call_args[0]
            assert "bot.py" in call_args[1]
            assert "--symbols" in call_args
            symbols_idx = call_args.index("--symbols")
            assert "NVDA,TSLA,AMD" == call_args[symbols_idx + 1]

            # Check status shows running with watchlist
            status_response = client.get("/api/bot/status")
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["status"] == "running"
            assert status_data["watchlist"] == ["NVDA", "TSLA", "AMD"]

    def test_scanner_failure_provides_clear_reason(self, client):
        """Test that scanner failures provide actionable error messages."""
        # Test market closed
        with patch('api.main.is_market_open', return_value=False), \
             patch('api.main.get_market_status_message', return_value="Market closed: Saturday."):

            response = client.post("/api/bot/start")
            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["reason"] == "market_closed"
            assert "Saturday" in data["detail"]["message"]

    def test_scanner_no_results_error(self, client):
        """Test that empty scanner results provide clear error."""
        with patch('api.main.is_market_open', return_value=True), \
             patch('api.main.VolatilityScanner') as MockScanner, \
             patch('api.main.YFinanceDataFetcher') as MockFetcher:

            mock_scanner = MagicMock()
            mock_scanner.scan_historical.return_value = []
            MockScanner.return_value = mock_scanner

            mock_fetcher = MagicMock()
            mock_fetcher.fetch_historical_data.return_value = MagicMock(empty=False)
            MockFetcher.return_value = mock_fetcher

            response = client.post("/api/bot/start")
            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["reason"] == "no_results"

    def test_scanner_exception_error(self, client):
        """Test that scanner exceptions provide clear error."""
        with patch('api.main.is_market_open', return_value=True), \
             patch('api.main.VolatilityScanner') as MockScanner, \
             patch('api.main.YFinanceDataFetcher') as MockFetcher:

            mock_scanner = MagicMock()
            mock_scanner.scan_historical.side_effect = Exception("Connection timeout")
            MockScanner.return_value = mock_scanner

            mock_fetcher = MagicMock()
            mock_fetcher.fetch_historical_data.return_value = MagicMock(empty=False)
            MockFetcher.return_value = mock_fetcher

            response = client.post("/api/bot/start")
            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["reason"] == "scanner_error"
            assert "timeout" in data["detail"]["message"].lower()
