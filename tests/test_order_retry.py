"""Tests for order submission retry logic (ODE-92)

Tests the retry behavior for transient API/network failures during order submission.
"""
import pytest
import time
from unittest.mock import patch, MagicMock, call
import requests

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.broker import (
    AlpacaBroker, BrokerAPIError, Order,
    RetryableOrderError, RETRYABLE_ORDER_EXCEPTIONS
)


@pytest.fixture
def allow_alpaca_in_tests():
    """Fixture to temporarily allow AlpacaBroker in test environment."""
    original = AlpacaBroker._allow_in_tests
    AlpacaBroker._allow_in_tests = True
    yield
    AlpacaBroker._allow_in_tests = original


class TestRetryableExceptions:
    """Test that retryable exception types are properly defined"""

    def test_retryable_exceptions_tuple_exists(self):
        """RETRYABLE_ORDER_EXCEPTIONS should be defined"""
        assert RETRYABLE_ORDER_EXCEPTIONS is not None
        assert isinstance(RETRYABLE_ORDER_EXCEPTIONS, tuple)

    def test_includes_connection_error(self):
        """Should include ConnectionError"""
        assert ConnectionError in RETRYABLE_ORDER_EXCEPTIONS

    def test_includes_timeout_error(self):
        """Should include TimeoutError"""
        assert TimeoutError in RETRYABLE_ORDER_EXCEPTIONS

    def test_includes_retryable_order_error(self):
        """Should include RetryableOrderError"""
        assert RetryableOrderError in RETRYABLE_ORDER_EXCEPTIONS


class TestRetryableOrderError:
    """Test RetryableOrderError exception class"""

    def test_is_exception(self):
        """RetryableOrderError should be an Exception"""
        err = RetryableOrderError("Test error")
        assert isinstance(err, Exception)

    def test_stores_message(self):
        """Should store error message"""
        err = RetryableOrderError("Network timeout")
        assert str(err) == "Network timeout"

    def test_stores_original_exception(self):
        """Should store original exception"""
        original = ConnectionError("Connection refused")
        err = RetryableOrderError("Wrapped error", original_exception=original)
        assert err.original_exception == original


@pytest.mark.usefixtures('allow_alpaca_in_tests')
class TestSubmitOrderRetry:
    """Test that submit_order retries on transient failures"""

    @pytest.fixture
    def mock_alpaca_broker(self):
        """Create AlpacaBroker with mocked API"""
        with patch.dict('sys.modules', {'alpaca_trade_api': MagicMock()}):
            import sys
            mock_tradeapi = sys.modules['alpaca_trade_api']
            mock_api = MagicMock()
            mock_tradeapi.REST.return_value = mock_api
            broker = AlpacaBroker('test_key', 'test_secret', 'https://paper-api.alpaca.markets')
            broker.api = mock_api
            yield broker, mock_api

    def test_successful_order_no_retry(self, mock_alpaca_broker):
        """Successful order should not trigger retries"""
        broker, mock_api = mock_alpaca_broker

        # Mock successful order
        mock_order = MagicMock()
        mock_order.id = 'ORDER123'
        mock_order.symbol = 'AAPL'
        mock_order.qty = 10
        mock_order.side = 'buy'
        mock_order.type = 'market'
        mock_order.status = 'filled'
        mock_order.limit_price = None
        mock_order.stop_price = None
        mock_order.submitted_at = None
        mock_api.submit_order.return_value = mock_order

        order = broker.submit_order('AAPL', 10, 'buy', 'market')

        assert order.id == 'ORDER123'
        assert mock_api.submit_order.call_count == 1

    def test_retry_on_connection_error(self, mock_alpaca_broker):
        """Should retry on ConnectionError"""
        broker, mock_api = mock_alpaca_broker

        # First two calls fail with ConnectionError, third succeeds
        mock_order = MagicMock()
        mock_order.id = 'ORDER123'
        mock_order.symbol = 'AAPL'
        mock_order.qty = 10
        mock_order.side = 'buy'
        mock_order.type = 'market'
        mock_order.status = 'filled'
        mock_order.limit_price = None
        mock_order.stop_price = None
        mock_order.submitted_at = None

        mock_api.submit_order.side_effect = [
            ConnectionError("Connection refused"),
            ConnectionError("Connection refused"),
            mock_order
        ]

        with patch('time.sleep'):  # Skip actual sleeping
            order = broker.submit_order('AAPL', 10, 'buy', 'market')

        assert order.id == 'ORDER123'
        assert mock_api.submit_order.call_count == 3

    def test_retry_on_timeout_error(self, mock_alpaca_broker):
        """Should retry on TimeoutError"""
        broker, mock_api = mock_alpaca_broker

        mock_order = MagicMock()
        mock_order.id = 'ORDER456'
        mock_order.symbol = 'NVDA'
        mock_order.qty = 5
        mock_order.side = 'sell'
        mock_order.type = 'market'
        mock_order.status = 'filled'
        mock_order.limit_price = None
        mock_order.stop_price = None
        mock_order.submitted_at = None

        mock_api.submit_order.side_effect = [
            TimeoutError("Request timed out"),
            mock_order
        ]

        with patch('time.sleep'):
            order = broker.submit_order('NVDA', 5, 'sell', 'market')

        assert order.id == 'ORDER456'
        assert mock_api.submit_order.call_count == 2

    def test_retry_on_rate_limit_429(self, mock_alpaca_broker):
        """Should retry on rate limit (429) errors"""
        broker, mock_api = mock_alpaca_broker

        # Simulate Alpaca rate limit error (raises exception with 429 status)
        rate_limit_error = Exception("rate limit exceeded")
        rate_limit_error.status_code = 429

        mock_order = MagicMock()
        mock_order.id = 'ORDER789'
        mock_order.symbol = 'SPY'
        mock_order.qty = 1
        mock_order.side = 'buy'
        mock_order.type = 'market'
        mock_order.status = 'filled'
        mock_order.limit_price = None
        mock_order.stop_price = None
        mock_order.submitted_at = None

        mock_api.submit_order.side_effect = [
            rate_limit_error,
            mock_order
        ]

        with patch('time.sleep'):
            order = broker.submit_order('SPY', 1, 'buy', 'market')

        assert order.id == 'ORDER789'
        assert mock_api.submit_order.call_count == 2

    def test_no_retry_on_business_error(self, mock_alpaca_broker):
        """Should NOT retry on business logic errors (insufficient funds)"""
        broker, mock_api = mock_alpaca_broker

        # Simulate insufficient funds error
        insufficient_error = Exception("insufficient buying power")
        insufficient_error.status_code = 403

        mock_api.submit_order.side_effect = insufficient_error

        with pytest.raises(BrokerAPIError) as exc_info:
            broker.submit_order('AAPL', 1000, 'buy', 'market')

        # Should only try once - no retries for business errors
        assert mock_api.submit_order.call_count == 1
        assert "insufficient" in str(exc_info.value).lower()

    def test_no_retry_on_invalid_symbol(self, mock_alpaca_broker):
        """Should NOT retry on invalid symbol errors"""
        broker, mock_api = mock_alpaca_broker

        invalid_symbol_error = Exception("symbol not found")
        invalid_symbol_error.status_code = 404

        mock_api.submit_order.side_effect = invalid_symbol_error

        with pytest.raises(BrokerAPIError):
            broker.submit_order('INVALID_SYMBOL', 10, 'buy', 'market')

        assert mock_api.submit_order.call_count == 1

    def test_max_retries_exhausted(self, mock_alpaca_broker):
        """Should raise after max retries exhausted"""
        broker, mock_api = mock_alpaca_broker

        # Always fail with retryable error
        mock_api.submit_order.side_effect = ConnectionError("Connection refused")

        with patch('time.sleep'):
            with pytest.raises((ConnectionError, BrokerAPIError)):
                broker.submit_order('AAPL', 10, 'buy', 'market')

        # Should try 4 times (initial + 3 retries)
        assert mock_api.submit_order.call_count == 4

    def test_exponential_backoff_timing(self, mock_alpaca_broker):
        """Should use exponential backoff between retries"""
        broker, mock_api = mock_alpaca_broker

        mock_api.submit_order.side_effect = ConnectionError("Connection refused")

        sleep_times = []
        original_sleep = time.sleep

        def mock_sleep(seconds):
            sleep_times.append(seconds)

        with patch('time.sleep', side_effect=mock_sleep):
            with pytest.raises((ConnectionError, BrokerAPIError)):
                broker.submit_order('AAPL', 10, 'buy', 'market')

        # Verify exponential backoff: 1.0, 2.0, 4.0 seconds
        assert len(sleep_times) == 3
        assert sleep_times[0] == pytest.approx(1.0, rel=0.1)
        assert sleep_times[1] == pytest.approx(2.0, rel=0.1)
        assert sleep_times[2] == pytest.approx(4.0, rel=0.1)

    def test_retry_on_requests_connection_error(self, mock_alpaca_broker):
        """Should retry on requests.exceptions.ConnectionError"""
        broker, mock_api = mock_alpaca_broker

        mock_order = MagicMock()
        mock_order.id = 'ORDER_RETRY'
        mock_order.symbol = 'TSLA'
        mock_order.qty = 2
        mock_order.side = 'buy'
        mock_order.type = 'market'
        mock_order.status = 'filled'
        mock_order.limit_price = None
        mock_order.stop_price = None
        mock_order.submitted_at = None

        mock_api.submit_order.side_effect = [
            requests.exceptions.ConnectionError("Failed to connect"),
            mock_order
        ]

        with patch('time.sleep'):
            order = broker.submit_order('TSLA', 2, 'buy', 'market')

        assert order.id == 'ORDER_RETRY'
        assert mock_api.submit_order.call_count == 2

    def test_retry_on_requests_timeout(self, mock_alpaca_broker):
        """Should retry on requests.exceptions.Timeout"""
        broker, mock_api = mock_alpaca_broker

        mock_order = MagicMock()
        mock_order.id = 'ORDER_TIMEOUT'
        mock_order.symbol = 'AMD'
        mock_order.qty = 3
        mock_order.side = 'buy'
        mock_order.type = 'market'
        mock_order.status = 'filled'
        mock_order.limit_price = None
        mock_order.stop_price = None
        mock_order.submitted_at = None

        mock_api.submit_order.side_effect = [
            requests.exceptions.Timeout("Read timed out"),
            mock_order
        ]

        with patch('time.sleep'):
            order = broker.submit_order('AMD', 3, 'buy', 'market')

        assert order.id == 'ORDER_TIMEOUT'
        assert mock_api.submit_order.call_count == 2


@pytest.mark.usefixtures('allow_alpaca_in_tests')
class TestSubmitOrderRetryLogging:
    """Test that retry attempts are properly logged"""

    @pytest.fixture
    def mock_alpaca_broker(self):
        """Create AlpacaBroker with mocked API"""
        with patch.dict('sys.modules', {'alpaca_trade_api': MagicMock()}):
            import sys
            mock_tradeapi = sys.modules['alpaca_trade_api']
            mock_api = MagicMock()
            mock_tradeapi.REST.return_value = mock_api
            broker = AlpacaBroker('test_key', 'test_secret', 'https://paper-api.alpaca.markets')
            broker.api = mock_api
            yield broker, mock_api

    def test_logs_retry_attempts(self, mock_alpaca_broker, caplog):
        """Should log retry attempts with attempt number"""
        broker, mock_api = mock_alpaca_broker

        mock_order = MagicMock()
        mock_order.id = 'ORDER123'
        mock_order.symbol = 'AAPL'
        mock_order.qty = 10
        mock_order.side = 'buy'
        mock_order.type = 'market'
        mock_order.status = 'filled'
        mock_order.limit_price = None
        mock_order.stop_price = None
        mock_order.submitted_at = None

        mock_api.submit_order.side_effect = [
            ConnectionError("Connection refused"),
            mock_order
        ]

        import logging
        with caplog.at_level(logging.WARNING):
            with patch('time.sleep'):
                broker.submit_order('AAPL', 10, 'buy', 'market')

        # Should have logged the retry attempt
        assert any('failed' in record.message.lower() and 'retry' in record.message.lower()
                   for record in caplog.records)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
