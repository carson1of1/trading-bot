"""Tests for portfolio history / equity curve functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from core.broker import (
    PortfolioHistory,
    AlpacaBroker,
    FakeBroker,
)


class TestPortfolioHistoryDataclass:
    """Test the PortfolioHistory dataclass."""

    def test_portfolio_history_creation(self):
        """Test creating a PortfolioHistory instance."""
        timestamps = [datetime.now() - timedelta(days=i) for i in range(7)]
        equity = [10000.0 + i * 100 for i in range(7)]

        history = PortfolioHistory(
            timestamps=timestamps,
            equity=equity,
            timeframe="1D",
            base_value=10000.0
        )

        assert len(history.timestamps) == 7
        assert len(history.equity) == 7
        assert history.timeframe == "1D"
        assert history.base_value == 10000.0

    def test_portfolio_history_empty(self):
        """Test creating empty PortfolioHistory."""
        history = PortfolioHistory(
            timestamps=[],
            equity=[],
            timeframe="1D",
            base_value=10000.0
        )

        assert len(history.timestamps) == 0
        assert len(history.equity) == 0


class TestFakeBrokerPortfolioHistory:
    """Test FakeBroker.get_portfolio_history()."""

    def test_get_portfolio_history_returns_data(self):
        """Test that FakeBroker returns mock portfolio history."""
        broker = FakeBroker(initial_cash=10000)

        history = broker.get_portfolio_history(period="7D")

        assert history is not None
        assert isinstance(history, PortfolioHistory)
        assert len(history.timestamps) > 0
        assert len(history.equity) > 0
        assert len(history.timestamps) == len(history.equity)

    def test_get_portfolio_history_different_periods(self):
        """Test different period options."""
        broker = FakeBroker(initial_cash=10000)

        for period in ["7D", "30D", "90D", "1Y"]:
            history = broker.get_portfolio_history(period=period)
            assert history is not None
            assert len(history.timestamps) > 0

    def test_get_portfolio_history_base_value_matches_initial_cash(self):
        """Test that base value matches broker's initial cash."""
        broker = FakeBroker(initial_cash=25000)

        history = broker.get_portfolio_history(period="7D")

        assert history.base_value == 25000.0


class TestAlpacaBrokerPortfolioHistory:
    """Test AlpacaBroker.get_portfolio_history()."""

    @patch('alpaca_trade_api.REST')
    def test_get_portfolio_history_calls_alpaca_api(self, mock_rest):
        """Test that AlpacaBroker calls Alpaca's get_portfolio_history."""
        # Setup mock
        mock_api = MagicMock()
        mock_rest.return_value = mock_api

        mock_history = MagicMock()
        mock_history.timestamp = [1704067200, 1704153600, 1704240000]  # Unix timestamps
        mock_history.equity = [10000.0, 10100.0, 10200.0]
        mock_history.base_value = 10000.0
        mock_history.timeframe = "1D"
        mock_api.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker("test_key", "test_secret", "https://paper-api.alpaca.markets")

        history = broker.get_portfolio_history(period="7D")

        mock_api.get_portfolio_history.assert_called_once()
        assert history is not None
        assert isinstance(history, PortfolioHistory)

    @patch('alpaca_trade_api.REST')
    def test_get_portfolio_history_handles_api_error(self, mock_rest):
        """Test that API errors are handled gracefully."""
        mock_api = MagicMock()
        mock_rest.return_value = mock_api
        mock_api.get_portfolio_history.side_effect = Exception("API Error")

        broker = AlpacaBroker("test_key", "test_secret", "https://paper-api.alpaca.markets")

        # Should not raise, return empty history
        history = broker.get_portfolio_history(period="7D")

        assert history is not None
        assert len(history.timestamps) == 0
        assert len(history.equity) == 0


class TestEquityHistoryAPI:
    """Test the /api/equity-history endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked broker."""
        from api.main import app, get_broker
        from core.broker import FakeBroker

        # Create a test broker
        test_broker = FakeBroker(initial_cash=10000)

        # Override the get_broker function
        def override_get_broker():
            return test_broker

        # Temporarily replace
        import api.main as api_main
        original_get_broker = api_main.get_broker
        api_main.get_broker = override_get_broker

        client = TestClient(app)
        yield client

        # Restore
        api_main.get_broker = original_get_broker

    def test_equity_history_default_period(self, client):
        """Test getting equity history with default period."""
        response = client.get("/api/equity-history")

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "period" in data
        assert "base_value" in data
        assert data["period"] == "30D"
        assert len(data["data"]) > 0

    def test_equity_history_7d_period(self, client):
        """Test getting 7-day equity history."""
        response = client.get("/api/equity-history?period=7D")

        assert response.status_code == 200
        data = response.json()
        assert data["period"] == "7D"
        assert len(data["data"]) == 7

    def test_equity_history_invalid_period(self, client):
        """Test that invalid period returns 400 error."""
        response = client.get("/api/equity-history?period=INVALID")

        assert response.status_code == 400
        assert "Invalid period" in response.json()["detail"]

    def test_equity_history_data_structure(self, client):
        """Test that returned data has correct structure."""
        response = client.get("/api/equity-history?period=7D")

        assert response.status_code == 200
        data = response.json()

        # Check each data point has required fields
        for point in data["data"]:
            assert "timestamp" in point
            assert "equity" in point
            assert isinstance(point["equity"], (int, float))
