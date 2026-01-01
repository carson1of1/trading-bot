"""Tests for YFinanceDataFetcher module"""
import pytest
import os
import sys
import tempfile
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import YFinanceDataFetcher, DataFetcher


class TestDataFetcherInitialization:
    """Test YFinanceDataFetcher initialization"""

    def test_initialization_with_default_config(self):
        """Should initialize with default config when no config file exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal config file
            config_path = os.path.join(tmpdir, "config.yaml")
            config_data = {
                'mode': 'PAPER',
                'trading': {},
                'risk_management': {
                    'max_position_size_pct': 5.0,
                    'stop_loss_pct': 2.0,
                    'take_profit_pct': 4.0,
                    'max_open_positions': 3
                },
                'strategies': {}
            }
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)

            # Reset singleton before test
            import core.config as config_module
            config_module._config_instance = None

            with patch.object(config_module, '_config_instance', None):
                with patch('core.data.get_global_config') as mock_config:
                    mock_cfg = Mock()
                    mock_cfg.config = {'data_quality': {}}
                    mock_config.return_value = mock_cfg

                    fetcher = YFinanceDataFetcher()

                    assert fetcher.cache == {}
                    assert fetcher.cache_ttl == 120
                    assert fetcher.min_call_interval == 0.3
                    assert fetcher.stale_threshold_1min == 10
                    assert fetcher.stale_threshold_5min == 30
                    assert fetcher.stale_threshold_daily == 360
                    assert fetcher.stale_threshold_outside_hours == 960

    def test_initialization_with_custom_thresholds(self):
        """Should use custom stale thresholds from config"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {
                'data_quality': {
                    'stale_threshold_1min': 5,
                    'stale_threshold_5min': 15,
                    'stale_threshold_daily': 180,
                    'stale_threshold_outside_hours': 480
                }
            }
            mock_config.return_value = mock_cfg

            fetcher = YFinanceDataFetcher()

            assert fetcher.stale_threshold_1min == 5
            assert fetcher.stale_threshold_5min == 15
            assert fetcher.stale_threshold_daily == 180
            assert fetcher.stale_threshold_outside_hours == 480

    def test_dataFetcher_alias(self):
        """DataFetcher should be an alias for YFinanceDataFetcher"""
        assert DataFetcher is YFinanceDataFetcher


class TestCacheMechanism:
    """Test cache validity and TTL"""

    def test_cache_valid_within_ttl(self):
        """Cache should be valid within TTL window"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            fetcher = YFinanceDataFetcher()
            market_tz = pytz.timezone('America/New_York')

            # Add item to cache
            cache_key = "TEST_1Min_100"
            fetcher.cache[cache_key] = {
                'data': pd.DataFrame(),
                'timestamp': datetime.now(market_tz)
            }

            assert fetcher._is_cache_valid(cache_key) is True

    def test_cache_invalid_after_ttl(self):
        """Cache should be invalid after TTL expires"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            fetcher = YFinanceDataFetcher()
            market_tz = pytz.timezone('America/New_York')

            # Add expired item to cache
            cache_key = "TEST_1Min_100"
            fetcher.cache[cache_key] = {
                'data': pd.DataFrame(),
                'timestamp': datetime.now(market_tz) - timedelta(seconds=150)  # 150s > 120s TTL
            }

            assert fetcher._is_cache_valid(cache_key) is False

    def test_cache_invalid_for_missing_key(self):
        """Cache should be invalid for non-existent keys"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            fetcher = YFinanceDataFetcher()

            assert fetcher._is_cache_valid("NONEXISTENT_KEY") is False


class TestColumnNormalization:
    """Test column normalization and timestamp handling"""

    def test_datetime_column_normalized_to_timestamp(self):
        """'Datetime' column should be normalized to 'timestamp'"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            # Create mock DataFrame with 'Datetime' column
            market_tz = pytz.timezone('America/New_York')
            df = pd.DataFrame({
                'Datetime': pd.date_range('2025-01-01 09:30', periods=5, freq='1min', tz='UTC'),
                'Open': [100.0] * 5,
                'High': [101.0] * 5,
                'Low': [99.0] * 5,
                'Close': [100.5] * 5,
                'Volume': [1000] * 5
            })

            with patch('yfinance.Ticker') as mock_ticker:
                mock_ticker_instance = Mock()
                mock_ticker_instance.history.return_value = df.set_index('Datetime')
                mock_ticker.return_value = mock_ticker_instance

                fetcher = YFinanceDataFetcher()
                fetcher.allow_stale_data = True  # Allow stale data for testing

                with patch.object(fetcher, 'market_tz', market_tz):
                    result = fetcher.get_historical_data('TEST', '1Min', 5)

                    assert result is not None
                    assert 'timestamp' in result.columns
                    assert 'datetime' not in result.columns.str.lower()

    def test_date_column_normalized_to_timestamp(self):
        """'Date' column should be normalized to 'timestamp' for daily data"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            # Create mock DataFrame with 'Date' column
            market_tz = pytz.timezone('America/New_York')
            df = pd.DataFrame({
                'Date': pd.date_range('2025-01-01', periods=5, freq='1D', tz='UTC'),
                'Open': [100.0] * 5,
                'High': [101.0] * 5,
                'Low': [99.0] * 5,
                'Close': [100.5] * 5,
                'Volume': [1000000] * 5
            })

            with patch('yfinance.Ticker') as mock_ticker:
                mock_ticker_instance = Mock()
                mock_ticker_instance.history.return_value = df.set_index('Date')
                mock_ticker.return_value = mock_ticker_instance

                fetcher = YFinanceDataFetcher()
                fetcher.allow_stale_data = True

                result = fetcher.get_historical_data('TEST', '1Day', 5)

                assert result is not None
                assert 'timestamp' in result.columns


class TestOHLCValidation:
    """Test OHLC data validation (NaN, invalid prices, high<low)"""

    def test_nan_rows_are_dropped(self):
        """Rows with NaN in OHLC columns should be dropped"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            market_tz = pytz.timezone('America/New_York')
            df = pd.DataFrame({
                'Datetime': pd.date_range('2025-01-01 09:30', periods=5, freq='1min', tz='UTC'),
                'Open': [100.0, np.nan, 100.0, 100.0, 100.0],
                'High': [101.0, 101.0, 101.0, 101.0, 101.0],
                'Low': [99.0, 99.0, 99.0, 99.0, 99.0],
                'Close': [100.5, 100.5, np.nan, 100.5, 100.5],
                'Volume': [1000] * 5
            })

            with patch('yfinance.Ticker') as mock_ticker:
                mock_ticker_instance = Mock()
                mock_ticker_instance.history.return_value = df.set_index('Datetime')
                mock_ticker.return_value = mock_ticker_instance

                fetcher = YFinanceDataFetcher()
                fetcher.allow_stale_data = True

                result = fetcher.get_historical_data('TEST', '1Min', 5)

                assert result is not None
                # 2 rows with NaN should be dropped
                assert len(result) == 3
                assert result['open'].isna().sum() == 0
                assert result['close'].isna().sum() == 0

    def test_zero_or_negative_prices_dropped(self):
        """Rows with zero or negative prices should be dropped"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            market_tz = pytz.timezone('America/New_York')
            df = pd.DataFrame({
                'Datetime': pd.date_range('2025-01-01 09:30', periods=5, freq='1min', tz='UTC'),
                'Open': [100.0, 0, 100.0, -5.0, 100.0],
                'High': [101.0, 101.0, 101.0, 101.0, 101.0],
                'Low': [99.0, 99.0, 99.0, 99.0, 99.0],
                'Close': [100.5, 100.5, 100.5, 100.5, 100.5],
                'Volume': [1000] * 5
            })

            with patch('yfinance.Ticker') as mock_ticker:
                mock_ticker_instance = Mock()
                mock_ticker_instance.history.return_value = df.set_index('Datetime')
                mock_ticker.return_value = mock_ticker_instance

                fetcher = YFinanceDataFetcher()
                fetcher.allow_stale_data = True

                result = fetcher.get_historical_data('TEST', '1Min', 5)

                assert result is not None
                # 2 rows with invalid open should be dropped
                assert len(result) == 3
                assert (result['open'] > 0).all()

    def test_high_less_than_low_is_fixed(self):
        """Rows where high < low should be fixed by swapping"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            market_tz = pytz.timezone('America/New_York')
            df = pd.DataFrame({
                'Datetime': pd.date_range('2025-01-01 09:30', periods=3, freq='1min', tz='UTC'),
                'Open': [100.0, 100.0, 100.0],
                'High': [99.0, 101.0, 101.0],   # First row: high < low (invalid)
                'Low': [101.0, 99.0, 99.0],     # First row: low > high (will be swapped)
                'Close': [100.5, 100.5, 100.5],
                'Volume': [1000] * 3
            })

            with patch('yfinance.Ticker') as mock_ticker:
                mock_ticker_instance = Mock()
                mock_ticker_instance.history.return_value = df.set_index('Datetime')
                mock_ticker.return_value = mock_ticker_instance

                fetcher = YFinanceDataFetcher()
                fetcher.allow_stale_data = True

                result = fetcher.get_historical_data('TEST', '1Min', 3)

                assert result is not None
                # All rows should have high >= low after fix
                assert (result['high'] >= result['low']).all()

    def test_negative_volume_clipped_to_zero(self):
        """Negative volume should be clipped to zero"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            market_tz = pytz.timezone('America/New_York')
            df = pd.DataFrame({
                'Datetime': pd.date_range('2025-01-01 09:30', periods=3, freq='1min', tz='UTC'),
                'Open': [100.0, 100.0, 100.0],
                'High': [101.0, 101.0, 101.0],
                'Low': [99.0, 99.0, 99.0],
                'Close': [100.5, 100.5, 100.5],
                'Volume': [1000, -500, 2000]
            })

            with patch('yfinance.Ticker') as mock_ticker:
                mock_ticker_instance = Mock()
                mock_ticker_instance.history.return_value = df.set_index('Datetime')
                mock_ticker.return_value = mock_ticker_instance

                fetcher = YFinanceDataFetcher()
                fetcher.allow_stale_data = True

                result = fetcher.get_historical_data('TEST', '1Min', 3)

                assert result is not None
                assert (result['volume'] >= 0).all()


class TestTimeframeMapping:
    """Test timeframe to yfinance interval mapping"""

    @pytest.mark.parametrize("timeframe,expected_period", [
        ('1Min', '7d'),
        ('5Min', '1mo'),
        ('1Hour', '3mo'),
        ('1Day', '1y'),
    ])
    def test_timeframe_to_period_mapping(self, timeframe, expected_period):
        """Test that each timeframe maps to correct yfinance period"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            market_tz = pytz.timezone('America/New_York')
            df = pd.DataFrame({
                'Datetime': pd.date_range('2025-01-01 09:30', periods=5, freq='1min', tz='UTC'),
                'Open': [100.0] * 5,
                'High': [101.0] * 5,
                'Low': [99.0] * 5,
                'Close': [100.5] * 5,
                'Volume': [1000] * 5
            })

            with patch('yfinance.Ticker') as mock_ticker:
                mock_ticker_instance = Mock()
                mock_ticker_instance.history.return_value = df.set_index('Datetime')
                mock_ticker.return_value = mock_ticker_instance

                fetcher = YFinanceDataFetcher()
                fetcher.allow_stale_data = True

                result = fetcher.get_historical_data('TEST', timeframe, 5)

                # Verify history was called with correct period
                call_kwargs = mock_ticker_instance.history.call_args[1]
                assert call_kwargs['period'] == expected_period

    def test_invalid_timeframe_returns_none(self):
        """Invalid timeframe should return None"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            fetcher = YFinanceDataFetcher()
            result = fetcher.get_historical_data('TEST', 'INVALID', 5)

            assert result is None


class TestRateLimiting:
    """Test rate limiting logic"""

    def test_rate_limiting_sleeps_when_called_too_fast(self):
        """Should sleep when called faster than min_call_interval"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            with patch('time.sleep') as mock_sleep:
                with patch('time.time') as mock_time:
                    # Simulate rapid successive calls
                    mock_time.side_effect = [0.0, 0.1, 0.1, 0.5]  # 0.1s between calls

                    market_tz = pytz.timezone('America/New_York')
                    df = pd.DataFrame({
                        'Datetime': pd.date_range('2025-01-01 09:30', periods=5, freq='1min', tz='UTC'),
                        'Open': [100.0] * 5,
                        'High': [101.0] * 5,
                        'Low': [99.0] * 5,
                        'Close': [100.5] * 5,
                        'Volume': [1000] * 5
                    })

                    with patch('yfinance.Ticker') as mock_ticker:
                        mock_ticker_instance = Mock()
                        mock_ticker_instance.history.return_value = df.set_index('Datetime')
                        mock_ticker.return_value = mock_ticker_instance

                        fetcher = YFinanceDataFetcher()
                        fetcher.allow_stale_data = True
                        fetcher.cache = {}  # Clear cache

                        # First call
                        fetcher.last_api_call['TEST'] = 0.0  # Simulate previous call

                        # Second call should trigger rate limiting
                        fetcher.get_historical_data('TEST', '1Min', 5)

                        # Verify sleep was called (0.3 - 0.1 = 0.2s)
                        mock_sleep.assert_called()


class TestGetLatestBars:
    """Test get_latest_bars wrapper method"""

    def test_get_latest_bars_calls_get_historical_data(self):
        """get_latest_bars should call get_historical_data with same params"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            fetcher = YFinanceDataFetcher()

            with patch.object(fetcher, 'get_historical_data', return_value=pd.DataFrame()) as mock_get:
                fetcher.get_latest_bars('AAPL', '1Hour', 10)

                mock_get.assert_called_once_with('AAPL', '1Hour', 10)

    def test_get_latest_bars_default_params(self):
        """get_latest_bars should use default params when not specified"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            fetcher = YFinanceDataFetcher()

            with patch.object(fetcher, 'get_historical_data', return_value=pd.DataFrame()) as mock_get:
                fetcher.get_latest_bars('AAPL')

                mock_get.assert_called_once_with('AAPL', '1Min', 5)


class TestGetLatestQuote:
    """Test get_latest_quote method"""

    def test_get_latest_quote_returns_quote_object(self):
        """get_latest_quote should return Quote object with price attributes"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            fetcher = YFinanceDataFetcher()

            mock_df = pd.DataFrame({
                'timestamp': [datetime.now(pytz.timezone('America/New_York'))],
                'open': [100.0],
                'high': [101.0],
                'low': [99.0],
                'close': [100.5],
                'volume': [1000]
            })

            with patch.object(fetcher, 'get_historical_data', return_value=mock_df):
                quote = fetcher.get_latest_quote('AAPL')

                assert quote is not None
                assert quote.ask_price == 100.5
                assert quote.bid_price == 100.5
                assert quote.last_price == 100.5

    def test_get_latest_quote_returns_none_for_invalid_price(self):
        """get_latest_quote should return None for zero/negative prices"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            fetcher = YFinanceDataFetcher()

            mock_df = pd.DataFrame({
                'timestamp': [datetime.now(pytz.timezone('America/New_York'))],
                'open': [100.0],
                'high': [101.0],
                'low': [99.0],
                'close': [0],  # Invalid price
                'volume': [1000]
            })

            with patch.object(fetcher, 'get_historical_data', return_value=mock_df):
                quote = fetcher.get_latest_quote('AAPL')

                assert quote is None

    def test_get_latest_quote_returns_none_for_nan_price(self):
        """get_latest_quote should return None for NaN prices"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            fetcher = YFinanceDataFetcher()

            mock_df = pd.DataFrame({
                'timestamp': [datetime.now(pytz.timezone('America/New_York'))],
                'open': [100.0],
                'high': [101.0],
                'low': [99.0],
                'close': [np.nan],
                'volume': [1000]
            })

            with patch.object(fetcher, 'get_historical_data', return_value=mock_df):
                quote = fetcher.get_latest_quote('AAPL')

                assert quote is None

    def test_get_latest_quote_caches_result(self):
        """get_latest_quote should cache result for 15 seconds"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            fetcher = YFinanceDataFetcher()

            mock_df = pd.DataFrame({
                'timestamp': [datetime.now(pytz.timezone('America/New_York'))],
                'open': [100.0],
                'high': [101.0],
                'low': [99.0],
                'close': [100.5],
                'volume': [1000]
            })

            with patch.object(fetcher, 'get_historical_data', return_value=mock_df) as mock_get:
                # First call
                quote1 = fetcher.get_latest_quote('AAPL')
                # Second call (should use cache)
                quote2 = fetcher.get_latest_quote('AAPL')

                # get_historical_data should only be called once due to caching
                assert mock_get.call_count == 1
                assert quote1.last_price == quote2.last_price


class TestMockedYFinanceCalls:
    """Test that yfinance is properly mocked in all tests"""

    def test_no_real_api_calls(self):
        """Verify no real API calls are made during testing"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            with patch('yfinance.Ticker') as mock_ticker:
                mock_ticker_instance = Mock()
                mock_ticker_instance.history.return_value = pd.DataFrame()
                mock_ticker.return_value = mock_ticker_instance

                fetcher = YFinanceDataFetcher()
                result = fetcher.get_historical_data('AAPL', '1Min', 5)

                # Verify Ticker was called with correct symbol
                mock_ticker.assert_called_once_with('AAPL')
                # Verify history was called
                mock_ticker_instance.history.assert_called_once()


class TestHistoricalDataRange:
    """Test get_historical_data_range method"""

    def test_historical_data_range_with_string_dates(self):
        """Should accept string dates in YYYY-MM-DD format"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            market_tz = pytz.timezone('America/New_York')
            df = pd.DataFrame({
                'Datetime': pd.date_range('2025-01-01 09:30', periods=5, freq='1h', tz='UTC'),
                'Open': [100.0] * 5,
                'High': [101.0] * 5,
                'Low': [99.0] * 5,
                'Close': [100.5] * 5,
                'Volume': [1000] * 5
            })

            with patch('yfinance.Ticker') as mock_ticker:
                mock_ticker_instance = Mock()
                mock_ticker_instance.history.return_value = df.set_index('Datetime')
                mock_ticker.return_value = mock_ticker_instance

                fetcher = YFinanceDataFetcher()

                result = fetcher.get_historical_data_range(
                    'TEST',
                    '1Hour',
                    start_date='2025-01-01',
                    end_date='2025-01-02'
                )

                assert result is not None
                assert 'timestamp' in result.columns

    def test_historical_data_range_invalid_timeframe(self):
        """Should return None for invalid timeframe"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            fetcher = YFinanceDataFetcher()
            result = fetcher.get_historical_data_range(
                'TEST',
                'INVALID',
                start_date='2025-01-01',
                end_date='2025-01-02'
            )

            assert result is None


class TestBatchFetching:
    """Test get_historical_data_batch method"""

    def test_batch_fetch_multiple_symbols(self):
        """Should fetch data for multiple symbols in parallel"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            fetcher = YFinanceDataFetcher()

            mock_df = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01 09:30', periods=5, freq='1min'),
                'open': [100.0] * 5,
                'high': [101.0] * 5,
                'low': [99.0] * 5,
                'close': [100.5] * 5,
                'volume': [1000] * 5
            })

            with patch.object(fetcher, 'get_historical_data', return_value=mock_df) as mock_get:
                symbols = ['AAPL', 'NVDA', 'TSLA']
                results = fetcher.get_historical_data_batch(symbols, '1Min', 100, max_workers=3)

                # Verify all symbols were fetched
                assert len(results) == 3
                assert 'AAPL' in results
                assert 'NVDA' in results
                assert 'TSLA' in results

                # Verify get_historical_data was called for each symbol
                assert mock_get.call_count == 3

    def test_batch_fetch_handles_failures(self):
        """Should handle failures gracefully and continue with other symbols"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            fetcher = YFinanceDataFetcher()

            mock_df = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01 09:30', periods=5, freq='1min'),
                'open': [100.0] * 5,
                'high': [101.0] * 5,
                'low': [99.0] * 5,
                'close': [100.5] * 5,
                'volume': [1000] * 5
            })

            def mock_get_historical(symbol, timeframe, limit):
                if symbol == 'FAIL':
                    return None
                return mock_df

            with patch.object(fetcher, 'get_historical_data', side_effect=mock_get_historical):
                symbols = ['AAPL', 'FAIL', 'TSLA']
                results = fetcher.get_historical_data_batch(symbols, '1Min', 100, max_workers=3)

                # Should have 2 successful, 1 failed
                assert len(results) == 2
                assert 'AAPL' in results
                assert 'FAIL' not in results
                assert 'TSLA' in results


class TestFallbackOnError:
    """Test fallback to cached data on API errors"""

    def test_returns_none_on_yfinance_error(self):
        """Should return None when yfinance download fails (inner try block)"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            fetcher = YFinanceDataFetcher()
            market_tz = pytz.timezone('America/New_York')

            # Pre-populate cache with stale data
            cache_key = "AAPL_1Min_5"
            cached_df = pd.DataFrame({'close': [100.0]})
            fetcher.cache[cache_key] = {
                'data': cached_df,
                'timestamp': datetime.now(market_tz) - timedelta(hours=1)  # Stale
            }

            with patch('yfinance.Ticker') as mock_ticker:
                mock_ticker_instance = Mock()
                # Inner exception in ticker.history() returns None (no fallback)
                mock_ticker_instance.history.side_effect = Exception("API Error")
                mock_ticker.return_value = mock_ticker_instance

                result = fetcher.get_historical_data('AAPL', '1Min', 5)

                # Returns None for inner exception (no fallback for yfinance errors)
                assert result is None

    def test_returns_stale_cache_on_outer_error(self):
        """Should return stale cached data when outer exception occurs"""
        with patch('core.data.get_global_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.config = {'data_quality': {}}
            mock_config.return_value = mock_cfg

            fetcher = YFinanceDataFetcher()
            market_tz = pytz.timezone('America/New_York')

            # Pre-populate cache with stale data
            cache_key = "AAPL_1Min_5"
            cached_df = pd.DataFrame({'close': [100.0]})
            fetcher.cache[cache_key] = {
                'data': cached_df,
                'timestamp': datetime.now(market_tz) - timedelta(hours=1)  # Stale
            }

            # Simulate an outer exception by patching yf.Ticker to raise
            with patch('yfinance.Ticker', side_effect=Exception("Outer error")):
                result = fetcher.get_historical_data('AAPL', '1Min', 5)

                # Should return stale cached data for outer exceptions
                assert result is not None
                assert result.equals(cached_df)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
