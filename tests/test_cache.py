"""Tests for DataCache module - disk-based caching for historical market data."""
import pytest
import os
import sys
import tempfile
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cache import DataCache, get_cache


class TestDataCacheInitialization:
    """Test DataCache initialization"""

    def test_default_cache_directory(self):
        """Should create default cache directory when None is passed"""
        # When cache_dir is None, it defaults to data/cache/ relative to the module
        # We test this by passing None and checking the result is a valid path
        cache = DataCache(cache_dir=None)

        assert cache.cache_dir.exists()
        assert cache.cache_dir.name == "cache"
        assert cache.freshness_hours == 24

    def test_custom_cache_directory(self):
        """Should use custom cache directory when provided"""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = os.path.join(tmpdir, "my_cache")

            cache = DataCache(cache_dir=custom_dir)

            assert cache.cache_dir == Path(custom_dir)
            assert cache.cache_dir.exists()

    def test_creates_directory_if_not_exists(self):
        """Should create cache directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "level1", "level2", "cache")

            cache = DataCache(cache_dir=nested_dir)

            assert cache.cache_dir.exists()

    def test_market_timezone_set(self):
        """Should set market timezone to America/New_York"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            assert str(cache.market_tz) == 'America/New_York'


class TestGetCachePath:
    """Test _get_cache_path method"""

    def test_returns_parquet_path(self):
        """Should return path with .parquet extension"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            path = cache._get_cache_path("AAPL")

            assert path.suffix == ".parquet"
            assert path.name == "AAPL.parquet"

    def test_uppercase_symbol(self):
        """Should uppercase the symbol in filename"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            path = cache._get_cache_path("aapl")

            assert path.name == "AAPL.parquet"

    def test_path_in_cache_directory(self):
        """Should return path within cache directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            path = cache._get_cache_path("TSLA")

            assert path.parent == cache.cache_dir


class TestHasFreshData:
    """Test has_fresh_data method"""

    def test_returns_false_for_missing_file(self):
        """Should return False if cache file doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            result = cache.has_fresh_data("NONEXISTENT")

            assert result is False

    def test_returns_false_for_empty_dataframe(self):
        """Should return False if cached DataFrame is empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            # Save empty DataFrame
            empty_df = pd.DataFrame()
            cache_path = cache._get_cache_path("EMPTY")
            empty_df.to_parquet(cache_path)

            result = cache.has_fresh_data("EMPTY")

            assert result is False

    def test_returns_true_for_fresh_data(self):
        """Should return True if data is recent (within 4 days)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Create DataFrame with recent timestamp
            df = pd.DataFrame({
                'timestamp': [datetime.now(market_tz) - timedelta(days=1)],
                'open': [100.0],
                'high': [101.0],
                'low': [99.0],
                'close': [100.5],
                'volume': [1000]
            })
            cache.save("FRESH", df)

            result = cache.has_fresh_data("FRESH")

            assert result is True

    def test_returns_false_for_stale_data(self):
        """Should return False if data is older than 4 days"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Create DataFrame with old timestamp
            df = pd.DataFrame({
                'timestamp': [datetime.now(market_tz) - timedelta(days=10)],
                'open': [100.0],
                'high': [101.0],
                'low': [99.0],
                'close': [100.5],
                'volume': [1000]
            })
            cache.save("STALE", df)

            result = cache.has_fresh_data("STALE")

            assert result is False

    def test_checks_against_end_date(self):
        """Should check freshness against provided end_date"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Create DataFrame with data from 2025-01-01
            df = pd.DataFrame({
                'timestamp': [market_tz.localize(datetime(2025, 1, 1, 10, 0))],
                'open': [100.0],
                'high': [101.0],
                'low': [99.0],
                'close': [100.5],
                'volume': [1000]
            })
            cache.save("DATED", df)

            # Fresh relative to 2025-01-02
            assert cache.has_fresh_data("DATED", "2025-01-02") is True

            # Stale relative to 2025-01-15
            assert cache.has_fresh_data("DATED", "2025-01-15") is False

    def test_handles_timezone_naive_timestamps(self):
        """Should handle timezone-naive timestamps in cache"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            # Create DataFrame with naive timestamp
            df = pd.DataFrame({
                'timestamp': [datetime.now() - timedelta(days=1)],
                'open': [100.0],
                'high': [101.0],
                'low': [99.0],
                'close': [100.5],
                'volume': [1000]
            })
            cache.save("NAIVE", df)

            result = cache.has_fresh_data("NAIVE")

            assert result is True

    def test_handles_dataframe_with_index_timestamps(self):
        """Should check timestamp from index if no timestamp column"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Create DataFrame with timestamp as index
            timestamps = [datetime.now(market_tz) - timedelta(days=1)]
            df = pd.DataFrame({
                'open': [100.0],
                'high': [101.0],
                'low': [99.0],
                'close': [100.5],
                'volume': [1000]
            }, index=pd.DatetimeIndex(timestamps))

            # Save directly to parquet (bypass save() which adds timestamp column)
            cache_path = cache._get_cache_path("INDEXED")
            df.to_parquet(cache_path)

            result = cache.has_fresh_data("INDEXED")

            assert result is True


class TestLoad:
    """Test load method"""

    def test_returns_none_for_missing_file(self):
        """Should return None if cache file doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            result = cache.load("NONEXISTENT")

            assert result is None

    def test_returns_none_for_empty_dataframe(self):
        """Should return None if cached DataFrame is empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            # Save empty DataFrame directly
            empty_df = pd.DataFrame()
            cache_path = cache._get_cache_path("EMPTY")
            empty_df.to_parquet(cache_path)

            result = cache.load("EMPTY")

            assert result is None

    def test_loads_cached_data(self):
        """Should load and return cached DataFrame"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Save test data
            df = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01 09:30', periods=5, freq='1h', tz=market_tz),
                'open': [100.0, 101.0, 102.0, 103.0, 104.0],
                'high': [101.0, 102.0, 103.0, 104.0, 105.0],
                'low': [99.0, 100.0, 101.0, 102.0, 103.0],
                'close': [100.5, 101.5, 102.5, 103.5, 104.5],
                'volume': [1000, 1100, 1200, 1300, 1400]
            })
            cache.save("TEST", df)

            result = cache.load("TEST")

            assert result is not None
            assert len(result) == 5
            assert 'timestamp' in result.columns

    def test_filters_by_start_date(self):
        """Should filter data by start_date"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Save data spanning multiple days
            df = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01 09:30', periods=48, freq='1h', tz=market_tz),
                'open': [100.0] * 48,
                'high': [101.0] * 48,
                'low': [99.0] * 48,
                'close': [100.5] * 48,
                'volume': [1000] * 48
            })
            cache.save("RANGE", df)

            result = cache.load("RANGE", start_date="2025-01-02")

            assert result is not None
            # Should only have data from Jan 2 onwards
            assert result['timestamp'].min().date() >= datetime(2025, 1, 2).date()

    def test_filters_by_end_date(self):
        """Should filter data by end_date"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Save data spanning multiple days
            df = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01 09:30', periods=72, freq='1h', tz=market_tz),
                'open': [100.0] * 72,
                'high': [101.0] * 72,
                'low': [99.0] * 72,
                'close': [100.5] * 72,
                'volume': [1000] * 72
            })
            cache.save("RANGE2", df)

            result = cache.load("RANGE2", end_date="2025-01-02")

            assert result is not None
            # Should only have data up to and including Jan 2
            assert result['timestamp'].max().date() <= datetime(2025, 1, 2).date()

    def test_filters_by_date_range(self):
        """Should filter data by both start and end date"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Save data spanning a week
            df = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01 09:30', periods=168, freq='1h', tz=market_tz),
                'open': [100.0] * 168,
                'high': [101.0] * 168,
                'low': [99.0] * 168,
                'close': [100.5] * 168,
                'volume': [1000] * 168
            })
            cache.save("RANGE3", df)

            result = cache.load("RANGE3", start_date="2025-01-03", end_date="2025-01-05")

            assert result is not None
            assert result['timestamp'].min().date() >= datetime(2025, 1, 3).date()
            assert result['timestamp'].max().date() <= datetime(2025, 1, 5).date()

    def test_returns_none_if_filtered_empty(self):
        """Should return None if date filter results in empty DataFrame"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            df = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01 09:30', periods=5, freq='1h', tz=market_tz),
                'open': [100.0] * 5,
                'high': [101.0] * 5,
                'low': [99.0] * 5,
                'close': [100.5] * 5,
                'volume': [1000] * 5
            })
            cache.save("NARROW", df)

            # Filter for dates not in cache
            result = cache.load("NARROW", start_date="2025-02-01")

            assert result is None


class TestSave:
    """Test save method"""

    def test_returns_false_for_none(self):
        """Should return False when saving None"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            result = cache.save("TEST", None)

            assert result is False

    def test_returns_false_for_empty_dataframe(self):
        """Should return False when saving empty DataFrame"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            result = cache.save("TEST", pd.DataFrame())

            assert result is False

    def test_saves_dataframe_to_parquet(self):
        """Should save DataFrame to parquet file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            df = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01 09:30', periods=5, freq='1h', tz=market_tz),
                'open': [100.0] * 5,
                'high': [101.0] * 5,
                'low': [99.0] * 5,
                'close': [100.5] * 5,
                'volume': [1000] * 5
            })

            result = cache.save("SAVED", df)

            assert result is True
            assert cache._get_cache_path("SAVED").exists()

    def test_merges_with_existing_data(self):
        """Should merge new data with existing cached data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Save initial data
            df1 = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01 09:30', periods=5, freq='1h', tz=market_tz),
                'open': [100.0] * 5,
                'high': [101.0] * 5,
                'low': [99.0] * 5,
                'close': [100.5] * 5,
                'volume': [1000] * 5
            })
            cache.save("MERGE", df1)

            # Save additional data
            df2 = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01 14:30', periods=5, freq='1h', tz=market_tz),
                'open': [105.0] * 5,
                'high': [106.0] * 5,
                'low': [104.0] * 5,
                'close': [105.5] * 5,
                'volume': [2000] * 5
            })
            cache.save("MERGE", df2)

            # Load and verify merged data
            result = cache.load("MERGE")

            assert result is not None
            # Should have 10 unique rows (5 + 5, no overlap)
            assert len(result) == 10

    def test_deduplicates_by_timestamp(self):
        """Should deduplicate data by timestamp when merging"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Save initial data
            df1 = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01 09:30', periods=5, freq='1h', tz=market_tz),
                'open': [100.0] * 5,
                'high': [101.0] * 5,
                'low': [99.0] * 5,
                'close': [100.5] * 5,
                'volume': [1000] * 5
            })
            cache.save("DEDUP", df1)

            # Save overlapping data (same timestamps, different values)
            df2 = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01 09:30', periods=5, freq='1h', tz=market_tz),
                'open': [200.0] * 5,  # Different values
                'high': [201.0] * 5,
                'low': [199.0] * 5,
                'close': [200.5] * 5,
                'volume': [2000] * 5
            })
            cache.save("DEDUP", df2)

            result = cache.load("DEDUP")

            assert result is not None
            # Should still have 5 rows (deduplicated)
            assert len(result) == 5
            # Should keep the newer values (from df2)
            assert result['open'].iloc[0] == 200.0

    def test_sorts_by_timestamp(self):
        """Should sort data by timestamp after merging"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Save data in reverse order
            df1 = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-02 09:30', periods=3, freq='1h', tz=market_tz),
                'open': [102.0] * 3,
                'high': [103.0] * 3,
                'low': [101.0] * 3,
                'close': [102.5] * 3,
                'volume': [1000] * 3
            })
            cache.save("SORT", df1)

            df2 = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01 09:30', periods=3, freq='1h', tz=market_tz),
                'open': [100.0] * 3,
                'high': [101.0] * 3,
                'low': [99.0] * 3,
                'close': [100.5] * 3,
                'volume': [1000] * 3
            })
            cache.save("SORT", df2)

            result = cache.load("SORT")

            assert result is not None
            # Should be sorted by timestamp
            timestamps = result['timestamp'].tolist()
            assert timestamps == sorted(timestamps)


class TestLoadBatch:
    """Test load_batch method"""

    def test_loads_multiple_symbols(self):
        """Should load data for multiple symbols"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Save data for multiple symbols
            for symbol in ['AAPL', 'NVDA', 'TSLA']:
                df = pd.DataFrame({
                    'timestamp': pd.date_range('2025-01-01 09:30', periods=5, freq='1h', tz=market_tz),
                    'open': [100.0] * 5,
                    'high': [101.0] * 5,
                    'low': [99.0] * 5,
                    'close': [100.5] * 5,
                    'volume': [1000] * 5
                })
                cache.save(symbol, df)

            result = cache.load_batch(['AAPL', 'NVDA', 'TSLA'])

            assert len(result) == 3
            assert 'AAPL' in result
            assert 'NVDA' in result
            assert 'TSLA' in result

    def test_skips_missing_symbols(self):
        """Should skip symbols without cached data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Save data for only one symbol
            df = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01 09:30', periods=5, freq='1h', tz=market_tz),
                'open': [100.0] * 5,
                'high': [101.0] * 5,
                'low': [99.0] * 5,
                'close': [100.5] * 5,
                'volume': [1000] * 5
            })
            cache.save('AAPL', df)

            result = cache.load_batch(['AAPL', 'MISSING1', 'MISSING2'])

            assert len(result) == 1
            assert 'AAPL' in result
            assert 'MISSING1' not in result

    def test_applies_date_filters(self):
        """Should apply date filters to batch load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Save data spanning multiple days
            for symbol in ['AAPL', 'NVDA']:
                df = pd.DataFrame({
                    'timestamp': pd.date_range('2025-01-01 09:30', periods=48, freq='1h', tz=market_tz),
                    'open': [100.0] * 48,
                    'high': [101.0] * 48,
                    'low': [99.0] * 48,
                    'close': [100.5] * 48,
                    'volume': [1000] * 48
                })
                cache.save(symbol, df)

            result = cache.load_batch(['AAPL', 'NVDA'], start_date="2025-01-02")

            for symbol, df in result.items():
                assert df['timestamp'].min().date() >= datetime(2025, 1, 2).date()

    def test_returns_empty_dict_for_no_data(self):
        """Should return empty dict if no symbols have data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            result = cache.load_batch(['MISSING1', 'MISSING2'])

            assert result == {}


class TestGetMissingSymbols:
    """Test get_missing_symbols method"""

    def test_returns_all_for_empty_cache(self):
        """Should return all symbols if cache is empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            result = cache.get_missing_symbols(['AAPL', 'NVDA', 'TSLA'])

            assert result == ['AAPL', 'NVDA', 'TSLA']

    def test_returns_missing_only(self):
        """Should return only symbols without fresh data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Save fresh data for AAPL only
            df = pd.DataFrame({
                'timestamp': [datetime.now(market_tz) - timedelta(hours=1)],
                'open': [100.0],
                'high': [101.0],
                'low': [99.0],
                'close': [100.5],
                'volume': [1000]
            })
            cache.save('AAPL', df)

            result = cache.get_missing_symbols(['AAPL', 'NVDA', 'TSLA'])

            assert 'AAPL' not in result
            assert 'NVDA' in result
            assert 'TSLA' in result

    def test_includes_stale_symbols(self):
        """Should include symbols with stale data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Save stale data
            df = pd.DataFrame({
                'timestamp': [datetime.now(market_tz) - timedelta(days=10)],
                'open': [100.0],
                'high': [101.0],
                'low': [99.0],
                'close': [100.5],
                'volume': [1000]
            })
            cache.save('STALE', df)

            result = cache.get_missing_symbols(['STALE', 'MISSING'])

            assert 'STALE' in result
            assert 'MISSING' in result

    def test_respects_end_date(self):
        """Should check freshness against provided end_date"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Save data from Jan 1
            df = pd.DataFrame({
                'timestamp': [market_tz.localize(datetime(2025, 1, 1, 10, 0))],
                'open': [100.0],
                'high': [101.0],
                'low': [99.0],
                'close': [100.5],
                'volume': [1000]
            })
            cache.save('DATED', df)

            # Fresh for Jan 2
            result1 = cache.get_missing_symbols(['DATED'], end_date="2025-01-02")
            assert 'DATED' not in result1

            # Stale for Jan 15
            result2 = cache.get_missing_symbols(['DATED'], end_date="2025-01-15")
            assert 'DATED' in result2


class TestClear:
    """Test clear method"""

    def test_clears_specific_symbol(self):
        """Should clear cache for specific symbol"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Save data for multiple symbols
            for symbol in ['AAPL', 'NVDA']:
                df = pd.DataFrame({
                    'timestamp': [datetime.now(market_tz)],
                    'open': [100.0],
                    'high': [101.0],
                    'low': [99.0],
                    'close': [100.5],
                    'volume': [1000]
                })
                cache.save(symbol, df)

            # Clear only AAPL
            cache.clear('AAPL')

            assert not cache._get_cache_path('AAPL').exists()
            assert cache._get_cache_path('NVDA').exists()

    def test_clears_all_symbols(self):
        """Should clear all cache files when symbol is None"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Save data for multiple symbols
            for symbol in ['AAPL', 'NVDA', 'TSLA']:
                df = pd.DataFrame({
                    'timestamp': [datetime.now(market_tz)],
                    'open': [100.0],
                    'high': [101.0],
                    'low': [99.0],
                    'close': [100.5],
                    'volume': [1000]
                })
                cache.save(symbol, df)

            # Clear all
            cache.clear()

            assert len(list(cache.cache_dir.glob("*.parquet"))) == 0

    def test_handles_nonexistent_symbol(self):
        """Should handle clearing nonexistent symbol gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            # Should not raise
            cache.clear('NONEXISTENT')


class TestGetCacheStats:
    """Test get_cache_stats method"""

    def test_returns_stats_for_empty_cache(self):
        """Should return stats for empty cache"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            stats = cache.get_cache_stats()

            assert stats['symbols_cached'] == 0
            assert stats['total_size_mb'] == 0
            assert stats['cache_dir'] == str(cache.cache_dir)

    def test_returns_accurate_stats(self):
        """Should return accurate statistics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            # Save data for multiple symbols
            for symbol in ['AAPL', 'NVDA', 'TSLA']:
                df = pd.DataFrame({
                    'timestamp': pd.date_range('2025-01-01 09:30', periods=100, freq='1h', tz=market_tz),
                    'open': [100.0] * 100,
                    'high': [101.0] * 100,
                    'low': [99.0] * 100,
                    'close': [100.5] * 100,
                    'volume': [1000] * 100
                })
                cache.save(symbol, df)

            stats = cache.get_cache_stats()

            assert stats['symbols_cached'] == 3
            assert stats['total_size_mb'] > 0


class TestGetCacheSingleton:
    """Test get_cache singleton function"""

    def test_returns_datacache_instance(self):
        """Should return a DataCache instance"""
        # Reset singleton
        import core.cache as cache_module
        cache_module._cache_instance = None

        cache = get_cache()

        assert isinstance(cache, DataCache)

    def test_returns_same_instance(self):
        """Should return the same instance on subsequent calls"""
        # Reset singleton
        import core.cache as cache_module
        cache_module._cache_instance = None

        cache1 = get_cache()
        cache2 = get_cache()

        assert cache1 is cache2


class TestErrorHandling:
    """Test error handling in cache operations"""

    def test_load_handles_corrupted_file(self):
        """Should handle corrupted parquet file gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            # Create corrupted file
            cache_path = cache._get_cache_path("CORRUPTED")
            with open(cache_path, 'w') as f:
                f.write("not a valid parquet file")

            result = cache.load("CORRUPTED")

            assert result is None

    def test_has_fresh_data_handles_corrupted_file(self):
        """Should return False for corrupted file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            # Create corrupted file
            cache_path = cache._get_cache_path("CORRUPTED")
            with open(cache_path, 'w') as f:
                f.write("not a valid parquet file")

            result = cache.has_fresh_data("CORRUPTED")

            assert result is False

    def test_save_handles_write_error(self):
        """Should handle write errors gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            market_tz = pytz.timezone('America/New_York')

            df = pd.DataFrame({
                'timestamp': [datetime.now(market_tz)],
                'open': [100.0],
                'high': [101.0],
                'low': [99.0],
                'close': [100.5],
                'volume': [1000]
            })

            # Make cache directory read-only
            os.chmod(tmpdir, 0o444)

            try:
                result = cache.save("TEST", df)
                # Should return False on error
                assert result is False
            finally:
                # Restore permissions for cleanup
                os.chmod(tmpdir, 0o755)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
