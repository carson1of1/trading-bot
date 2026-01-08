"""
Tests for HotStocksFeed - verifies hot stocks fetching and filtering.
"""

import pytest
import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hot_stocks import HotStocksFeed


class TestHotStocksFeedInit:
    """Test HotStocksFeed initialization and configuration."""

    def test_default_config(self):
        """Test feed initializes with default config."""
        feed = HotStocksFeed()

        assert feed.enabled is True
        assert feed.top_n == 50
        assert feed.min_price == 5
        assert feed.max_price == 1000
        assert feed.min_volume == 500_000
        assert feed.cache_hours == 20

    def test_custom_config(self):
        """Test feed accepts custom configuration."""
        config = {
            'enabled': False,
            'top_n': 30,
            'min_price': 10,
            'max_price': 500,
            'min_volume': 1_000_000,
            'cache_hours': 12,
        }
        feed = HotStocksFeed(config)

        assert feed.enabled is False
        assert feed.top_n == 30
        assert feed.min_price == 10
        assert feed.max_price == 500
        assert feed.min_volume == 1_000_000
        assert feed.cache_hours == 12

    def test_disabled_returns_empty(self):
        """Test disabled feed returns empty list."""
        feed = HotStocksFeed({'enabled': False})
        result = feed.fetch()

        assert result == []


class TestHotStocksFiltering:
    """Test symbol filtering logic."""

    def test_filter_by_price(self):
        """Test price filtering works correctly."""
        feed = HotStocksFeed({'min_price': 5, 'max_price': 100})

        stocks = [
            {'symbol': 'CHEAP', 'price': 2, 'avg_volume': 1_000_000},
            {'symbol': 'GOOD', 'price': 50, 'avg_volume': 1_000_000},
            {'symbol': 'EXPENSIVE', 'price': 200, 'avg_volume': 1_000_000},
        ]
        result = feed._filter_symbols(stocks)

        assert 'GOOD' in result
        assert 'CHEAP' not in result
        assert 'EXPENSIVE' not in result

    def test_filter_by_volume(self):
        """Test volume filtering works correctly."""
        feed = HotStocksFeed({'min_volume': 500_000})

        stocks = [
            {'symbol': 'LOWVOL', 'price': 50, 'avg_volume': 100_000},
            {'symbol': 'HIGHVOL', 'price': 50, 'avg_volume': 1_000_000},
        ]
        result = feed._filter_symbols(stocks)

        assert 'HIGHVOL' in result
        assert 'LOWVOL' not in result

    def test_top_n_limit(self):
        """Test only top_n symbols are returned."""
        feed = HotStocksFeed({'top_n': 3})

        stocks = [
            {'symbol': f'SYM{i}', 'price': 50, 'avg_volume': 1_000_000}
            for i in range(10)
        ]
        result = feed._filter_symbols(stocks)

        assert len(result) == 3


class TestHotStocksCaching:
    """Test caching functionality."""

    def test_cache_save_and_load(self, tmp_path):
        """Test cache is saved and loaded correctly."""
        feed = HotStocksFeed()
        feed.cache_file = tmp_path / 'hot_stocks.json'

        # Save cache
        symbols = ['AAPL', 'NVDA', 'TSLA']
        feed._save_cache(symbols)

        # Verify cache file exists
        assert feed.cache_file.exists()

        # Load cache
        loaded = feed._load_cache()
        assert loaded == symbols

    def test_expired_cache_returns_none(self, tmp_path):
        """Test expired cache is not used."""
        feed = HotStocksFeed({'cache_hours': 1})
        feed.cache_file = tmp_path / 'hot_stocks.json'

        # Create expired cache
        market_tz = pytz.timezone('America/New_York')
        expired_time = datetime.now(market_tz) - timedelta(hours=2)

        cache = {
            'fetched_at': expired_time.isoformat(),
            'expires_at': expired_time.isoformat(),  # Already expired
            'symbols': ['AAPL', 'NVDA'],
        }
        with open(feed.cache_file, 'w') as f:
            json.dump(cache, f)

        # Should return None for expired cache
        result = feed._load_cache()
        assert result is None

    def test_fresh_cache_is_used(self, tmp_path):
        """Test fresh cache is used."""
        feed = HotStocksFeed()
        feed.cache_file = tmp_path / 'hot_stocks.json'

        # Create fresh cache
        market_tz = pytz.timezone('America/New_York')
        now = datetime.now(market_tz)
        future = now + timedelta(hours=10)

        cache = {
            'fetched_at': now.isoformat(),
            'expires_at': future.isoformat(),
            'symbols': ['AAPL', 'NVDA'],
        }
        with open(feed.cache_file, 'w') as f:
            json.dump(cache, f)

        # Should return cached symbols
        result = feed._load_cache()
        assert result == ['AAPL', 'NVDA']


class TestHotStocksYahooFetch:
    """Test Yahoo Finance fetching with mocked responses."""

    @patch('core.hot_stocks.requests.get')
    def test_fetch_from_yahoo_success(self, mock_get):
        """Test successful Yahoo Finance fetch."""
        feed = HotStocksFeed()

        # Mock Yahoo response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'finance': {
                'result': [{
                    'quotes': [
                        {
                            'symbol': 'NVDA',
                            'regularMarketPrice': 150,
                            'regularMarketVolume': 5_000_000,
                            'averageDailyVolume3Month': 4_000_000,
                            'regularMarketChangePercent': 8.5,
                        },
                        {
                            'symbol': 'TSLA',
                            'regularMarketPrice': 250,
                            'regularMarketVolume': 3_000_000,
                            'averageDailyVolume3Month': 2_500_000,
                            'regularMarketChangePercent': 6.2,
                        },
                    ]
                }]
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = feed._fetch_from_yahoo()

        assert len(result) == 2
        assert result[0]['symbol'] == 'NVDA'
        assert result[0]['price'] == 150
        assert result[1]['symbol'] == 'TSLA'

    @patch('core.hot_stocks.requests.get')
    def test_fetch_filters_bad_symbols(self, mock_get):
        """Test Yahoo fetch filters out ADRs and unusual symbols."""
        feed = HotStocksFeed()

        # Mock response with bad symbols
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'finance': {
                'result': [{
                    'quotes': [
                        {'symbol': 'NVDA', 'regularMarketPrice': 150, 'regularMarketVolume': 5_000_000},
                        {'symbol': 'BRK.A', 'regularMarketPrice': 500000, 'regularMarketVolume': 1000},  # Has dot
                        {'symbol': 'AAPL-USD', 'regularMarketPrice': 150, 'regularMarketVolume': 1000},  # Has dash
                        {'symbol': 'VERYLONGNAME', 'regularMarketPrice': 50, 'regularMarketVolume': 1000},  # Too long
                    ]
                }]
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = feed._fetch_from_yahoo()

        assert len(result) == 1
        assert result[0]['symbol'] == 'NVDA'

    @patch('core.hot_stocks.requests.get')
    def test_fetch_handles_error_gracefully(self, mock_get):
        """Test fetch handles network errors gracefully."""
        feed = HotStocksFeed()

        mock_get.side_effect = Exception("Network error")

        result = feed.fetch()
        assert result == []  # Should return empty on error


class TestHotStocksUniverseExclusion:
    """Test exclusion of symbols already in universe."""

    def test_excludes_universe_symbols(self, tmp_path):
        """Test symbols in universe.yaml are excluded."""
        feed = HotStocksFeed()

        # Simulate universe symbols
        feed._universe_symbols = {'AAPL', 'NVDA', 'TSLA'}

        # Mock filtered results
        with patch.object(feed, '_fetch_from_yahoo') as mock_fetch:
            mock_fetch.return_value = [
                {'symbol': 'AAPL', 'price': 150, 'avg_volume': 5_000_000},
                {'symbol': 'NEWSTOCK', 'price': 50, 'avg_volume': 1_000_000},
                {'symbol': 'NVDA', 'price': 140, 'avg_volume': 4_000_000},
            ]

            with patch.object(feed, '_load_cache', return_value=None):
                with patch.object(feed, '_save_cache'):
                    result = feed.fetch()

        # Only NEWSTOCK should be returned (not in universe)
        assert 'NEWSTOCK' in result
        assert 'AAPL' not in result
        assert 'NVDA' not in result


class TestScannerTemporarySymbols:
    """Test scanner integration with temporary symbols."""

    def test_add_temporary_symbols(self):
        """Test scanner accepts temporary symbols."""
        from core.scanner import VolatilityScanner

        scanner = VolatilityScanner()
        assert scanner._temporary_symbols == []

        scanner.add_temporary_symbols(['NEW1', 'NEW2'])
        assert 'NEW1' in scanner._temporary_symbols
        assert 'NEW2' in scanner._temporary_symbols

    def test_clear_temporary_symbols(self):
        """Test scanner clears temporary symbols."""
        from core.scanner import VolatilityScanner

        scanner = VolatilityScanner()
        scanner.add_temporary_symbols(['NEW1', 'NEW2'])
        assert len(scanner._temporary_symbols) == 2

        scanner.clear_temporary_symbols()
        assert scanner._temporary_symbols == []

    def test_no_duplicates_in_temporary(self):
        """Test duplicate symbols are not added."""
        from core.scanner import VolatilityScanner

        scanner = VolatilityScanner()
        scanner.add_temporary_symbols(['NEW1', 'NEW2'])
        scanner.add_temporary_symbols(['NEW1', 'NEW3'])

        # Should have 3 unique symbols
        assert len(scanner._temporary_symbols) == 3
        assert scanner._temporary_symbols.count('NEW1') == 1
