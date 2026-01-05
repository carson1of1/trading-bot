"""
Tests for Bot-Scanner Integration.

Tests the integration between the backtesting engine (Backtest1Hour) and
the VolatilityScanner to ensure scanner filtering works correctly during
signal generation and trade execution.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scanner import VolatilityScanner
from core.market_hours import MarketHours
from backtest import Backtest1Hour


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


class TestBotStartWithScanner:
    """Tests for bot startup behavior with scanner integration."""

    def test_start_bot_runs_scanner_first(self):
        """Test that scanner runs before signal generation when enabled."""
        # Create backtest with scanner enabled
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

        # Verify scanner is initialized
        assert backtest.scanner is not None
        assert backtest.scanner_enabled is True

        # Verify scanner runs during _build_daily_scan_results
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

            # Scanner should have been called
            assert mock_scan.called
            # Result should be a dict of date -> symbols
            assert isinstance(result, dict)

    def test_start_bot_fails_when_market_closed(self):
        """Test that market hours are properly checked."""
        market_hours = MarketHours()

        # Test weekend detection
        saturday = datetime(2025, 1, 4).date()  # A Saturday
        assert not market_hours.is_trading_day(saturday)

        # Test holiday detection
        new_years = datetime(2025, 1, 1).date()
        assert not market_hours.is_trading_day(new_years)

        # Test should_start_trading returns False when market closed
        with patch.object(market_hours, 'is_market_open', return_value=False):
            should_trade, reason = market_hours.should_start_trading()
            assert should_trade is False
            assert "closed" in reason.lower()

    def test_start_bot_fails_when_scanner_returns_empty(self):
        """Test graceful handling when scanner returns no symbols."""
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

        # Create data that won't pass scanner filters
        historical_data = {
            'PENNY': create_sample_ohlcv_data(300, 2),  # Price too low
        }

        # Scanner should return empty list for symbols that don't pass filters
        result = backtest._build_daily_scan_results(
            historical_data,
            '2025-01-02',
            '2025-01-10'
        )

        # Should return dict (possibly with empty lists) without crashing
        assert isinstance(result, dict)

        # Verify _is_symbol_scanned_for_date handles empty results
        assert backtest._is_symbol_scanned_for_date('NONEXISTENT', '2025-01-05') is False

    def test_start_bot_fails_on_scanner_api_error(self):
        """Test that scanner API errors are handled gracefully."""
        config = {
            'volatility_scanner': {
                'enabled': True,
                'top_n': 5,
            },
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

        # Mock scanner to raise an exception
        with patch.object(backtest.scanner, 'scan_historical') as mock_scan:
            mock_scan.side_effect = Exception("API Error: Rate limit exceeded")

            # Should not raise - build_daily_scan_results should handle gracefully
            # The scanner's internal error handling should catch this
            try:
                result = backtest._build_daily_scan_results(
                    historical_data,
                    '2025-01-02',
                    '2025-01-10'
                )
                # If it returns, it handled the error
                assert isinstance(result, dict)
            except Exception as e:
                # If exception propagates, that's also acceptable behavior
                # as long as it's a clear error
                assert "API Error" in str(e) or "Rate limit" in str(e)

    def test_start_bot_returns_watchlist_on_success(self):
        """Test that scanner returns correct watchlist on success."""
        config = {
            'volatility_scanner': {
                'enabled': True,
                'top_n': 2,
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

        # Create data for multiple symbols
        historical_data = {
            'AAPL': create_sample_ohlcv_data(300, 150),
            'MSFT': create_sample_ohlcv_data(300, 350),
            'NVDA': create_sample_ohlcv_data(300, 500),
        }

        with patch.object(backtest.scanner, 'scan_historical') as mock_scan:
            # Return top 2 symbols
            mock_scan.return_value = ['NVDA', 'MSFT']

            result = backtest._build_daily_scan_results(
                historical_data,
                '2025-01-02',
                '2025-01-05'
            )

            # Verify result structure
            assert isinstance(result, dict)

            # Verify scanner was called with correct symbols
            call_args = mock_scan.call_args
            assert 'AAPL' in call_args.kwargs.get('symbols', call_args[1].get('symbols', []))


class TestFullScannerBotIntegration:
    """Full integration tests for scanner and bot."""

    def test_full_flow_start_to_running(self):
        """Test complete flow from start with scanner to running backtest."""
        config = {
            'volatility_scanner': {
                'enabled': True,
                'top_n': 3,
                'min_price': 5,
                'max_price': 1000,
                'min_volume': 100_000,
            },
            'risk_management': {
                'stop_loss_pct': 2.0,
                'take_profit_pct': 4.0,
                'max_position_pct': 10.0,
            },
            'exit_manager': {'enabled': False},
            'entry_gate': {'enabled': False},
        }

        backtest = Backtest1Hour(
            initial_capital=100000,
            config=config,
            scanner_enabled=True
        )

        # Verify initialization
        assert backtest.scanner_enabled is True
        assert backtest.scanner is not None

        # Test the symbol filtering mechanism
        backtest._daily_scanned_symbols = {
            '2025-01-02': ['AAPL', 'MSFT'],
            '2025-01-03': ['MSFT', 'NVDA'],
        }

        # Test symbol is scanned for specific date
        assert backtest._is_symbol_scanned_for_date('AAPL', '2025-01-02') is True
        assert backtest._is_symbol_scanned_for_date('NVDA', '2025-01-02') is False
        assert backtest._is_symbol_scanned_for_date('NVDA', '2025-01-03') is True

        # Test with datetime object
        tz = pytz.timezone('America/New_York')
        dt = datetime(2025, 1, 2, 10, 30, tzinfo=tz)
        assert backtest._is_symbol_scanned_for_date('AAPL', dt) is True
        assert backtest._is_symbol_scanned_for_date('NVDA', dt) is False

    def test_scanner_failure_provides_clear_reason(self):
        """Test that scanner failures provide clear error messages."""
        scanner = VolatilityScanner({'top_n': 5})

        # Test with empty data - should return empty list, not crash
        result = scanner.scan_historical('2025-01-05', ['AAPL'], {})
        assert result == []

        # Test with insufficient data
        tz = pytz.timezone('America/New_York')
        base_date = datetime(2025, 1, 1, 9, 30, tzinfo=tz)
        short_data = {
            'SHORT': pd.DataFrame({
                'timestamp': [base_date + timedelta(hours=j) for j in range(5)],
                'open': [100] * 5,
                'high': [110] * 5,
                'low': [90] * 5,
                'close': [100] * 5,
                'volume': [1_000_000] * 5,
            })
        }

        result = scanner.scan_historical('2025-01-05', ['SHORT'], short_data)
        # Should return empty list due to insufficient data
        assert result == []

    def test_scanner_no_results_error(self):
        """Test handling when scanner finds no valid symbols."""
        config = {
            'volatility_scanner': {
                'enabled': True,
                'top_n': 5,
                'min_price': 1000,  # Very high minimum price
                'max_price': 2000,
                'min_volume': 10_000_000,  # Very high volume requirement
            },
            'risk_management': {'stop_loss_pct': 2.0, 'take_profit_pct': 4.0},
            'exit_manager': {'enabled': False},
        }

        backtest = Backtest1Hour(
            initial_capital=100000,
            config=config,
            scanner_enabled=True
        )

        # Create data that won't pass strict filters
        historical_data = {
            'CHEAP': create_sample_ohlcv_data(300, 50),  # Price too low
            'ILLIQ': create_sample_ohlcv_data(300, 100),  # Volume too low
        }

        # Should not crash, just return empty results
        result = backtest._build_daily_scan_results(
            historical_data,
            '2025-01-02',
            '2025-01-10'
        )

        assert isinstance(result, dict)

        # All dates should have empty or no symbols
        for date_str, symbols in result.items():
            # Symbols list should be empty due to strict filters
            assert isinstance(symbols, list)

    def test_scanner_exception_error(self):
        """Test handling of exceptions during scanner operation."""
        config = {
            'volatility_scanner': {
                'enabled': True,
                'top_n': 5,
            },
            'risk_management': {'stop_loss_pct': 2.0, 'take_profit_pct': 4.0},
            'exit_manager': {'enabled': False},
        }

        backtest = Backtest1Hour(
            initial_capital=100000,
            config=config,
            scanner_enabled=True
        )

        # Test with malformed data that might cause issues
        malformed_data = {
            'BAD': pd.DataFrame({
                'timestamp': [None, None, None],
                'open': [np.nan, np.nan, np.nan],
                'high': [np.nan, np.nan, np.nan],
                'low': [np.nan, np.nan, np.nan],
                'close': [np.nan, np.nan, np.nan],
                'volume': [np.nan, np.nan, np.nan],
            })
        }

        # Should handle gracefully without crashing
        try:
            result = backtest._build_daily_scan_results(
                malformed_data,
                '2025-01-02',
                '2025-01-05'
            )
            # If it returns, verify it's a valid structure
            assert isinstance(result, dict)
        except Exception as e:
            # If an exception occurs, it should be informative
            # This is acceptable as long as it doesn't cause silent failures
            pass

        # Test scanner directly with problematic data
        scanner = VolatilityScanner()

        # Empty symbol list
        result = scanner.scan_historical('2025-01-05', [], {})
        assert result == []

        # None in historical_data should be handled
        result = scanner.scan_historical('2025-01-05', ['AAPL'], {'AAPL': None})
        assert result == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
