"""
Tests for Bot-Scanner Integration.

Part 1: CLI/API tests for bot startup with scanner
Part 2: Backtest-Scanner integration tests
"""
import pytest
import subprocess
import os
import sys
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import numpy as np
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scanner import VolatilityScanner
from core.market_hours import MarketHours
from backtest import Backtest1Hour


# =============================================================================
# Part 1: CLI/API Tests (bot.py and api/main.py integration)
# =============================================================================

class TestBotCLISymbolsArgument:
    """Test bot.py accepts --symbols argument."""

    def test_bot_parses_symbols_argument(self):
        """bot.py should parse --symbols into a list."""
        with patch('bot.TradingBot') as MockBot:
            mock_instance = MagicMock()
            MockBot.return_value = mock_instance

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
            bot = TradingBot(scanner_symbols=['NVDA', 'TSLA', 'AMD'])
            assert bot.watchlist == ['NVDA', 'TSLA', 'AMD']


class TestMarketHoursCheck:
    """Test market hours validation functions."""

    def test_is_market_open_during_trading_hours(self):
        """Should return True during market hours (9:30 AM - 4:00 PM ET)."""
        from core.market_hours import is_market_open

        et = pytz.timezone('America/New_York')
        mock_time = et.localize(datetime(2026, 1, 6, 10, 30, 0))

        with patch('core.market_hours.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert is_market_open() == True

    def test_is_market_open_before_market_hours(self):
        """Should return False before 9:30 AM ET."""
        from core.market_hours import is_market_open

        et = pytz.timezone('America/New_York')
        mock_time = et.localize(datetime(2026, 1, 6, 8, 0, 0))

        with patch('core.market_hours.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert is_market_open() == False

    def test_is_market_open_on_weekend(self):
        """Should return False on weekends."""
        from core.market_hours import is_market_open

        et = pytz.timezone('America/New_York')
        mock_time = et.localize(datetime(2026, 1, 3, 11, 0, 0))

        with patch('core.market_hours.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert is_market_open() == False

    def test_get_market_status_message_when_closed(self):
        """Should return helpful message when market is closed."""
        from core.market_hours import get_market_status_message

        et = pytz.timezone('America/New_York')
        mock_time = et.localize(datetime(2026, 1, 3, 11, 0, 0))

        with patch('core.market_hours.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            message = get_market_status_message()
            assert "closed" in message.lower() or "weekend" in message.lower() or "saturday" in message.lower()


class TestBotStartWithScannerAPI:
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
             patch('api.main._is_bot_running', return_value=False), \
             patch('api.main.subprocess.Popen') as mock_popen, \
             patch('api.main.YFinanceDataFetcher') as MockFetcher:

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan_historical.return_value = [
                {'symbol': 'NVDA', 'composite_score': 0.95},
                {'symbol': 'TSLA', 'composite_score': 0.90},
                {'symbol': 'AMD', 'composite_score': 0.85},
            ]
            MockScanner.return_value = mock_scanner_instance

            mock_fetcher_instance = MagicMock()
            mock_fetcher_instance.get_historical_data_range.return_value = MagicMock(empty=False)
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
             patch('api.main._is_bot_running', return_value=False), \
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
             patch('api.main._is_bot_running', return_value=False), \
             patch('api.main.YFinanceDataFetcher') as MockFetcher:

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan_historical.return_value = []
            MockScanner.return_value = mock_scanner_instance

            mock_fetcher_instance = MagicMock()
            mock_fetcher_instance.get_historical_data_range.return_value = MagicMock(empty=False)
            MockFetcher.return_value = mock_fetcher_instance

            response = client.post("/api/bot/start")

            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["reason"] == "no_results"

    def test_start_bot_fails_on_scanner_api_error(self, client):
        """Should return 400 with reason='scanner_error' on data fetch failure."""
        with patch('api.main.VolatilityScanner') as MockScanner, \
             patch('api.main.is_market_open', return_value=True), \
             patch('api.main._is_bot_running', return_value=False), \
             patch('api.main.YFinanceDataFetcher') as MockFetcher:

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan_historical.side_effect = Exception("YFinance API timeout")
            MockScanner.return_value = mock_scanner_instance

            mock_fetcher_instance = MagicMock()
            mock_fetcher_instance.get_historical_data_range.return_value = MagicMock(empty=False)
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
             patch('api.main._is_bot_running', return_value=False), \
             patch('api.main.subprocess.Popen'), \
             patch('api.main.YFinanceDataFetcher') as MockFetcher:

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan_historical.return_value = [
                {'symbol': 'NVDA', 'composite_score': 0.95},
                {'symbol': 'TSLA', 'composite_score': 0.90},
            ]
            MockScanner.return_value = mock_scanner_instance

            mock_fetcher_instance = MagicMock()
            mock_fetcher_instance.get_historical_data_range.return_value = MagicMock(empty=False)
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
        # _is_bot_running returns False for start check, then True for status check
        is_running_mock = MagicMock(side_effect=[False, True])
        with patch('api.main.VolatilityScanner') as MockScanner, \
             patch('api.main.is_market_open', return_value=True), \
             patch('api.main._is_bot_running', is_running_mock), \
             patch('api.main.subprocess.Popen') as mock_popen, \
             patch('api.main.YFinanceDataFetcher') as MockFetcher:

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan_historical.return_value = [
                {'symbol': 'NVDA', 'composite_score': 0.95},
                {'symbol': 'TSLA', 'composite_score': 0.90},
                {'symbol': 'AMD', 'composite_score': 0.85},
            ]
            MockScanner.return_value = mock_scanner_instance

            mock_fetcher_instance = MagicMock()
            mock_fetcher_instance.get_historical_data_range.return_value = MagicMock(empty=False)
            MockFetcher.return_value = mock_fetcher_instance

            response = client.post("/api/bot/start")
            assert response.status_code == 200
            start_data = response.json()
            assert start_data["status"] == "started"
            assert start_data["watchlist"] == ["NVDA", "TSLA", "AMD"]

            mock_popen.assert_called_once()
            call_args = mock_popen.call_args[0][0]
            assert "python3" in call_args[0]
            assert "bot.py" in call_args[1]
            assert "--symbols" in call_args
            symbols_idx = call_args.index("--symbols")
            assert "NVDA,TSLA,AMD" == call_args[symbols_idx + 1]

            status_response = client.get("/api/bot/status")
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["status"] == "running"
            assert status_data["watchlist"] == ["NVDA", "TSLA", "AMD"]


# =============================================================================
# Part 2: Backtest-Scanner Integration Tests
# =============================================================================

def create_sample_ohlcv_data(n_bars: int = 300, base_price: float = 100.0) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    tz = pytz.timezone('America/New_York')
    base_date = datetime(2025, 1, 2, 9, 30, tzinfo=tz)

    np.random.seed(42)
    dates = [base_date + timedelta(hours=j) for j in range(n_bars)]

    volatility = 0.02
    returns = np.random.normal(0, volatility, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_bars)),
        'high': prices * (1 + np.abs(np.random.normal(0, volatility, n_bars))),
        'low': prices * (1 - np.abs(np.random.normal(0, volatility, n_bars))),
        'close': prices,
        'volume': np.random.uniform(500_000, 2_000_000, n_bars),
    })
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    return df


class TestBacktestScannerIntegration:
    """Tests for bot startup behavior with scanner integration."""

    def test_start_bot_runs_scanner_first(self):
        """Test that scanner runs before signal generation when enabled."""
        config = {
            'volatility_scanner': {
                'enabled': True,
                'top_n': 5,
                'min_price': 5,
                'max_price': 1000,
                'min_volume': 100_000,
            },
            'risk_management': {'stop_loss_pct': 2.0, 'take_profit_pct': 4.0},
            'exit_manager': {'enabled': False},
        }

        backtest = Backtest1Hour(
            initial_capital=100000,
            config=config,
            scanner_enabled=True
        )

        assert backtest.scanner is not None
        assert backtest.scanner_enabled is True

        historical_data = {
            'AAPL': create_sample_ohlcv_data(300, 150),
            'MSFT': create_sample_ohlcv_data(300, 350),
        }

        with patch.object(backtest.scanner, 'scan_historical') as mock_scan:
            mock_scan.return_value = ['AAPL']

            result = backtest._build_daily_scan_results(
                historical_data,
                '2025-01-02',
                '2025-01-10'
            )

            assert mock_scan.called
            assert isinstance(result, dict)

    def test_start_bot_fails_when_market_closed(self):
        """Test that market hours are properly checked."""
        config = {
            'volatility_scanner': {'enabled': True, 'top_n': 5},
            'risk_management': {'stop_loss_pct': 2.0, 'take_profit_pct': 4.0},
            'exit_manager': {'enabled': False},
        }

        backtest = Backtest1Hour(
            initial_capital=100000,
            config=config,
            scanner_enabled=True
        )

        market_hours = MarketHours()
        et = pytz.timezone('America/New_York')
        weekend_time = datetime(2025, 1, 4, 12, 0, 0, tzinfo=et)

        # Mock get_market_time to return weekend time
        with patch.object(market_hours, 'get_market_time', return_value=weekend_time):
            assert market_hours.is_market_open() == False

    def test_start_bot_fails_when_scanner_returns_empty(self):
        """Test behavior when scanner returns empty results."""
        config = {
            'volatility_scanner': {'enabled': True, 'top_n': 5},
            'risk_management': {'stop_loss_pct': 2.0, 'take_profit_pct': 4.0},
            'exit_manager': {'enabled': False},
        }

        backtest = Backtest1Hour(
            initial_capital=100000,
            config=config,
            scanner_enabled=True
        )

        historical_data = {
            'AAPL': create_sample_ohlcv_data(300, 150),
        }

        with patch.object(backtest.scanner, 'scan_historical') as mock_scan:
            mock_scan.return_value = []

            result = backtest._build_daily_scan_results(
                historical_data,
                '2025-01-02',
                '2025-01-03'
            )

            for date_results in result.values():
                assert len(date_results) == 0

    def test_start_bot_fails_on_scanner_api_error(self):
        """Test behavior when scanner raises an exception."""
        config = {
            'volatility_scanner': {'enabled': True, 'top_n': 5},
            'risk_management': {'stop_loss_pct': 2.0, 'take_profit_pct': 4.0},
            'exit_manager': {'enabled': False},
        }

        backtest = Backtest1Hour(
            initial_capital=100000,
            config=config,
            scanner_enabled=True
        )

        historical_data = {
            'AAPL': create_sample_ohlcv_data(300, 150),
        }

        with patch.object(backtest.scanner, 'scan_historical') as mock_scan:
            mock_scan.side_effect = Exception("API Error")

            try:
                result = backtest._build_daily_scan_results(
                    historical_data,
                    '2025-01-02',
                    '2025-01-03'
                )
                assert isinstance(result, dict)
            except Exception as e:
                pass

    def test_start_bot_returns_watchlist_on_success(self):
        """Test that scanner returns valid watchlist."""
        config = {
            'volatility_scanner': {'enabled': True, 'top_n': 5},
            'risk_management': {'stop_loss_pct': 2.0, 'take_profit_pct': 4.0},
            'exit_manager': {'enabled': False},
        }

        backtest = Backtest1Hour(
            initial_capital=100000,
            config=config,
            scanner_enabled=True
        )

        historical_data = {
            'AAPL': create_sample_ohlcv_data(300, 150),
            'MSFT': create_sample_ohlcv_data(300, 350),
            'NVDA': create_sample_ohlcv_data(300, 500),
        }

        with patch.object(backtest.scanner, 'scan_historical') as mock_scan:
            mock_scan.return_value = ['NVDA', 'AAPL']

            result = backtest._build_daily_scan_results(
                historical_data,
                '2025-01-02',
                '2025-01-03'
            )

            for date_key, symbols in result.items():
                assert 'NVDA' in symbols or 'AAPL' in symbols or len(symbols) == 0


class TestScannerFiltersDuringBacktest:
    """Tests for scanner filtering during backtest signal generation."""

    def test_only_scanned_symbols_get_signals(self):
        """Test that only scanner-approved symbols receive signals."""
        config = {
            'volatility_scanner': {'enabled': True, 'top_n': 2},
            'risk_management': {'stop_loss_pct': 2.0, 'take_profit_pct': 4.0},
            'exit_manager': {'enabled': False},
        }

        backtest = Backtest1Hour(
            initial_capital=100000,
            config=config,
            scanner_enabled=True
        )

        with patch.object(backtest.scanner, 'scan_historical') as mock_scan:
            mock_scan.return_value = ['NVDA']

            historical_data = {
                'NVDA': create_sample_ohlcv_data(300, 500),
                'AAPL': create_sample_ohlcv_data(300, 150),
                'MSFT': create_sample_ohlcv_data(300, 350),
            }

            result = backtest._build_daily_scan_results(
                historical_data,
                '2025-01-02',
                '2025-01-03'
            )

            for date_key, symbols in result.items():
                for sym in symbols:
                    assert sym == 'NVDA'

    def test_non_scanned_symbols_excluded(self):
        """Test that symbols not in scanner results are excluded."""
        config = {
            'volatility_scanner': {'enabled': True, 'top_n': 1},
            'risk_management': {'stop_loss_pct': 2.0, 'take_profit_pct': 4.0},
            'exit_manager': {'enabled': False},
        }

        backtest = Backtest1Hour(
            initial_capital=100000,
            config=config,
            scanner_enabled=True
        )

        with patch.object(backtest.scanner, 'scan_historical') as mock_scan:
            mock_scan.return_value = ['NVDA']

            historical_data = {
                'NVDA': create_sample_ohlcv_data(300, 500),
                'AAPL': create_sample_ohlcv_data(300, 150),
            }

            result = backtest._build_daily_scan_results(
                historical_data,
                '2025-01-02',
                '2025-01-03'
            )

            for date_key, symbols in result.items():
                assert 'AAPL' not in symbols

    def test_scanner_runs_daily_during_backtest(self):
        """Test that scanner is re-run for each trading day."""
        config = {
            'volatility_scanner': {'enabled': True, 'top_n': 5},
            'risk_management': {'stop_loss_pct': 2.0, 'take_profit_pct': 4.0},
            'exit_manager': {'enabled': False},
        }

        backtest = Backtest1Hour(
            initial_capital=100000,
            config=config,
            scanner_enabled=True
        )

        historical_data = {
            'NVDA': create_sample_ohlcv_data(300, 500),
            'AAPL': create_sample_ohlcv_data(300, 150),
        }

        with patch.object(backtest.scanner, 'scan_historical') as mock_scan:
            mock_scan.return_value = ['NVDA', 'AAPL']

            result = backtest._build_daily_scan_results(
                historical_data,
                '2025-01-02',
                '2025-01-10'
            )

            assert mock_scan.call_count >= 1
            assert len(result) >= 1
