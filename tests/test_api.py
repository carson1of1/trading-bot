"""Tests for FastAPI endpoints in api/main.py

Tests for:
- /api/account
- /api/positions
- /api/bot/status
- /api/orders

Uses FakeBroker to avoid needing real Alpaca credentials.
"""
import pytest
import os
import sys
import tempfile
import yaml
from unittest.mock import patch, MagicMock
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from core.broker import FakeBroker, Account, Position, Order


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_config():
    """Create a temporary config file for testing."""
    config_data = {
        'mode': 'DRY_RUN',
        'trading': {},
        'risk': {
            'risk_per_trade': 0.02,
            'max_position_size_pct': 0.20,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'max_open_positions': 5
        },
        'exit_rules': {
            'hard_stop_loss': 0.02,
            'partial_take_profit': {
                'threshold': 0.02
            }
        },
        'strategies': [
            {'name': 'Momentum', 'enabled': True},
            {'name': 'MeanReversion', 'enabled': True}
        ],
        'broker': {
            'fake_broker': {
                'initial_cash': 100000,
                'commission_per_trade': 0,
                'slippage_percent': 0
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    yield config_path

    os.unlink(config_path)


@pytest.fixture
def fake_broker():
    """Create a FakeBroker instance for testing."""
    return FakeBroker(initial_cash=100000, commission=0, slippage=0)


@pytest.fixture
def client(mock_config, fake_broker):
    """Create a test client with mocked dependencies."""
    from core.config import GlobalConfig

    # Create config from temp file
    test_config = GlobalConfig(mock_config)

    # Import and reset the API module's broker singleton
    import api.main as api_module
    api_module._broker = None

    # Reset bot state to defaults for test isolation
    api_module._bot_state = {
        "status": "stopped",
        "last_action": None,
        "last_action_time": None,
        "kill_switch_triggered": False,
        "watchlist": None
    }

    # Patch get_broker, get_global_config, and _is_bot_running for test isolation
    with patch.object(api_module, 'get_broker', return_value=fake_broker):
        with patch.object(api_module, 'get_global_config', return_value=test_config):
            with patch.object(api_module, '_is_bot_running', return_value=False):
                yield TestClient(api_module.app)


# =============================================================================
# ACCOUNT ENDPOINT TESTS
# =============================================================================

class TestAccountEndpoint:
    """Tests for /api/account endpoint."""

    def test_get_account_success(self, client, fake_broker):
        """Should return account info with correct fields."""
        response = client.get("/api/account")

        assert response.status_code == 200
        data = response.json()

        assert "equity" in data
        assert "cash" in data
        assert "buying_power" in data
        assert "portfolio_value" in data
        assert "daily_pnl" in data
        assert "daily_pnl_percent" in data

        # Default values for fresh FakeBroker
        assert data["equity"] == 100000.0
        assert data["cash"] == 100000.0
        assert data["portfolio_value"] == 100000.0

    def test_get_account_with_position(self, client, fake_broker):
        """Account should reflect position value."""
        # Buy some shares
        fake_broker.submit_order('AAPL', 10, 'buy', 'market', price=150.0)

        response = client.get("/api/account")

        assert response.status_code == 200
        data = response.json()

        # Cash should be reduced
        assert data["cash"] == 100000 - (10 * 150.0)


# =============================================================================
# POSITIONS ENDPOINT TESTS
# =============================================================================

class TestPositionsEndpoint:
    """Tests for /api/positions endpoint."""

    def test_get_positions_empty(self, client, fake_broker):
        """Should return empty positions list."""
        response = client.get("/api/positions")

        assert response.status_code == 200
        data = response.json()

        assert "positions" in data
        assert "total_unrealized_pl" in data
        assert data["positions"] == []
        assert data["total_unrealized_pl"] == 0.0

    def test_get_positions_with_position(self, client, fake_broker):
        """Should return position details."""
        # Create a position
        fake_broker.submit_order('AAPL', 10, 'buy', 'market', price=150.0)

        response = client.get("/api/positions")

        assert response.status_code == 200
        data = response.json()

        assert len(data["positions"]) == 1

        pos = data["positions"][0]
        assert pos["symbol"] == "AAPL"
        assert pos["qty"] == 10
        assert pos["side"] == "long"
        assert pos["avg_entry_price"] == 150.0

    def test_get_positions_multiple(self, client, fake_broker):
        """Should return multiple positions."""
        fake_broker.submit_order('AAPL', 10, 'buy', 'market', price=150.0)
        fake_broker.submit_order('NVDA', 5, 'buy', 'market', price=500.0)

        response = client.get("/api/positions")

        assert response.status_code == 200
        data = response.json()

        assert len(data["positions"]) == 2
        symbols = [p["symbol"] for p in data["positions"]]
        assert "AAPL" in symbols
        assert "NVDA" in symbols


# =============================================================================
# BOT STATUS ENDPOINT TESTS
# =============================================================================

class TestBotStatusEndpoint:
    """Tests for /api/bot/status endpoint."""

    def test_get_bot_status_default(self, client):
        """Should return default bot status."""
        response = client.get("/api/bot/status")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "mode" in data
        assert "last_action" in data
        assert "last_action_time" in data
        assert "kill_switch_triggered" in data

        # Default values
        assert data["status"] == "stopped"
        assert data["mode"] == "DRY_RUN"
        assert data["kill_switch_triggered"] is False

    def test_get_bot_status_updated(self, mock_config, fake_broker):
        """Should return updated bot status."""
        from core.config import GlobalConfig
        import api.main as api_module
        from fastapi.testclient import TestClient

        test_config = GlobalConfig(mock_config)
        api_module._broker = None

        # Set bot state to running
        api_module._bot_state = {
            "status": "running",
            "last_action": "Checked positions",
            "last_action_time": "2026-01-05T12:00:00",
            "kill_switch_triggered": False,
            "watchlist": None
        }

        # Mock _is_bot_running to return True so the running state is preserved
        with patch.object(api_module, 'get_broker', return_value=fake_broker):
            with patch.object(api_module, 'get_global_config', return_value=test_config):
                with patch.object(api_module, '_is_bot_running', return_value=True):
                    client = TestClient(api_module.app)
                    response = client.get("/api/bot/status")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "running"
        assert data["last_action"] == "Checked positions"
        assert data["last_action_time"] is not None


# =============================================================================
# ORDERS ENDPOINT TESTS
# =============================================================================

class TestOrdersEndpoint:
    """Tests for /api/orders endpoint."""

    def test_get_orders_empty(self, client, fake_broker):
        """Should return empty orders list when no open orders."""
        response = client.get("/api/orders")

        assert response.status_code == 200
        data = response.json()

        assert "orders" in data
        assert data["orders"] == []

    def test_get_orders_filled(self, client, fake_broker):
        """Should return filled orders when status=filled."""
        # Submit market order (executes immediately)
        fake_broker.submit_order('AAPL', 10, 'buy', 'market', price=150.0)

        response = client.get("/api/orders?status=filled")

        assert response.status_code == 200
        data = response.json()

        assert len(data["orders"]) == 1
        order = data["orders"][0]
        assert order["symbol"] == "AAPL"
        assert order["qty"] == 10
        assert order["side"] == "buy"
        assert order["status"] == "filled"

    def test_get_orders_multiple(self, client, fake_broker):
        """Should return multiple orders."""
        fake_broker.submit_order('AAPL', 10, 'buy', 'market', price=150.0)
        fake_broker.submit_order('NVDA', 5, 'buy', 'market', price=500.0)

        response = client.get("/api/orders?status=filled")

        assert response.status_code == 200
        data = response.json()

        assert len(data["orders"]) == 2


# =============================================================================
# HEALTH CHECK TEST
# =============================================================================

class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Should return healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "timestamp" in data


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for API error handling."""

    def test_account_broker_error(self, mock_config):
        """Should return 503 on broker API error."""
        from core.config import GlobalConfig
        from core.broker import BrokerAPIError
        import api.main as api_module

        test_config = GlobalConfig(mock_config)
        api_module._broker = None

        # Create mock broker that raises error
        mock_broker = MagicMock()
        mock_broker.get_account.side_effect = BrokerAPIError("API Error")

        with patch.object(api_module, 'get_broker', return_value=mock_broker):
            with patch.object(api_module, 'get_global_config', return_value=test_config):
                client = TestClient(api_module.app)
                response = client.get("/api/account")

                assert response.status_code == 503

    def test_positions_broker_error(self, mock_config):
        """Should return 503 on broker API error for positions."""
        from core.config import GlobalConfig
        from core.broker import BrokerAPIError
        import api.main as api_module

        test_config = GlobalConfig(mock_config)
        api_module._broker = None

        mock_broker = MagicMock()
        mock_broker.get_positions.side_effect = BrokerAPIError("API Error")

        with patch.object(api_module, 'get_broker', return_value=mock_broker):
            with patch.object(api_module, 'get_global_config', return_value=test_config):
                client = TestClient(api_module.app)
                response = client.get("/api/positions")

                assert response.status_code == 503


# =============================================================================
# SETTINGS ENDPOINT TESTS
# =============================================================================

class TestSettingsEndpoint:
    """Tests for /api/settings endpoint."""

    def test_get_settings_success(self, mock_config):
        """Should return settings with correct fields."""
        import api.main as api_module
        from fastapi.testclient import TestClient

        # Mock load_config to return test config
        test_config = {
            'mode': 'DRY_RUN',
            'risk': {'risk_per_trade': 0.02, 'max_open_positions': 5},
            'exit_rules': {
                'hard_stop_loss': 0.02,
                'partial_take_profit': {'threshold': 0.02}
            },
            'strategies': [{'name': 'Momentum', 'enabled': True}]
        }

        with patch.object(api_module, 'load_config', return_value=test_config):
            client = TestClient(api_module.app)
            response = client.get("/api/settings")

            assert response.status_code == 200
            data = response.json()

            assert "mode" in data
            assert "risk_per_trade" in data
            assert "max_positions" in data
            assert "stop_loss_pct" in data
            assert "take_profit_pct" in data
            assert "strategies_enabled" in data
            assert data["mode"] == "DRY_RUN"


# =============================================================================
# SCANNER ENDPOINT TESTS
# =============================================================================

class TestScannerEndpoint:
    """Tests for /api/scanner/scan endpoint."""

    @pytest.mark.skip(reason="Scanner requires external data - tested manually")
    def test_scanner_returns_results(self, client):
        """Should return scanner results with correct structure."""
        response = client.get("/api/scanner/scan?top_n=5")

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert "scanned_at" in data
        assert isinstance(data["results"], list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
