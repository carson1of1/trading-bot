"""
Tests for VolatilityScanner - verifies no look-ahead bias and volatility scoring.
"""

import pytest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scanner import VolatilityScanner


class TestVolatilityScannerInit:
    """Test scanner initialization and configuration."""

    def test_default_config(self):
        """Test scanner initializes with default config."""
        scanner = VolatilityScanner()

        assert scanner.top_n == 10
        assert scanner.min_price == 5
        assert scanner.max_price == 1000
        assert scanner.min_volume == 500_000
        assert scanner.lookback_days == 14
        assert scanner.weights['atr_pct'] == 0.5
        assert scanner.weights['daily_range_pct'] == 0.3
        assert scanner.weights['volume_ratio'] == 0.2

    def test_custom_config(self):
        """Test scanner accepts custom configuration."""
        config = {
            'top_n': 5,
            'min_price': 10,
            'max_price': 500,
            'min_volume': 1_000_000,
            'lookback_days': 20,
            'weights': {
                'atr_pct': 0.6,
                'daily_range_pct': 0.2,
                'volume_ratio': 0.2
            }
        }
        scanner = VolatilityScanner(config)

        assert scanner.top_n == 5
        assert scanner.min_price == 10
        assert scanner.max_price == 500
        assert scanner.min_volume == 1_000_000
        assert scanner.lookback_days == 20
        assert scanner.weights['atr_pct'] == 0.6

    def test_get_config(self):
        """Test get_config returns current configuration."""
        scanner = VolatilityScanner({'top_n': 7})
        config = scanner.get_config()

        assert config['top_n'] == 7
        assert 'min_price' in config
        assert 'max_price' in config
        assert 'min_volume' in config
        assert 'weights' in config
        assert 'lookback_days' in config


class TestValidateSymbols:
    """Test symbol validation."""

    def test_valid_symbols(self):
        """Test valid symbols pass through."""
        scanner = VolatilityScanner()
        symbols = ['AAPL', 'MSFT', 'NVDA', 'SPY', 'QQQ']
        result = scanner.validate_symbols(symbols)

        assert result == symbols

    def test_filters_empty_symbols(self):
        """Test empty symbols are filtered out."""
        scanner = VolatilityScanner()
        symbols = ['AAPL', '', 'MSFT', None]
        result = scanner.validate_symbols(symbols)

        assert 'AAPL' in result
        assert 'MSFT' in result
        assert '' not in result

    def test_filters_long_symbols(self):
        """Test symbols longer than 5 chars are filtered."""
        scanner = VolatilityScanner()
        symbols = ['AAPL', 'GOOGL', 'TOOLONG', 'SPY']
        result = scanner.validate_symbols(symbols)

        assert 'AAPL' in result
        assert 'GOOGL' in result
        assert 'SPY' in result
        assert 'TOOLONG' not in result

    def test_filters_non_alphanumeric(self):
        """Test non-alphanumeric symbols are filtered."""
        scanner = VolatilityScanner()
        symbols = ['AAPL', 'A-B', 'MSFT', 'X.Y']
        result = scanner.validate_symbols(symbols)

        assert 'AAPL' in result
        assert 'MSFT' in result
        assert 'A-B' not in result
        assert 'X.Y' not in result


class TestApplyFilters:
    """Test candidate filtering."""

    def test_filters_low_price(self):
        """Test candidates with low price are filtered."""
        scanner = VolatilityScanner({'min_price': 10})
        candidates = [
            {'symbol': 'AAPL', 'price': 150, 'volume': 1_000_000},
            {'symbol': 'PENNY', 'price': 5, 'volume': 1_000_000},
        ]
        result = scanner._apply_filters(candidates)

        assert len(result) == 1
        assert result[0]['symbol'] == 'AAPL'

    def test_filters_high_price(self):
        """Test candidates with high price are filtered."""
        scanner = VolatilityScanner({'max_price': 100})
        candidates = [
            {'symbol': 'CHEAP', 'price': 50, 'volume': 1_000_000},
            {'symbol': 'EXPEN', 'price': 500, 'volume': 1_000_000},
        ]
        result = scanner._apply_filters(candidates)

        assert len(result) == 1
        assert result[0]['symbol'] == 'CHEAP'

    def test_filters_low_volume(self):
        """Test candidates with low volume are filtered."""
        scanner = VolatilityScanner({'min_volume': 1_000_000})
        candidates = [
            {'symbol': 'AAPL', 'price': 100, 'volume': 5_000_000},
            {'symbol': 'ILLIQ', 'price': 100, 'volume': 100_000},
        ]
        result = scanner._apply_filters(candidates)

        assert len(result) == 1
        assert result[0]['symbol'] == 'AAPL'

    def test_filters_invalid_symbols(self):
        """Test invalid symbols are filtered."""
        scanner = VolatilityScanner()
        candidates = [
            {'symbol': 'AAPL', 'price': 150, 'volume': 1_000_000},
            {'symbol': 'TOOLONG', 'price': 150, 'volume': 1_000_000},
            {'symbol': '', 'price': 150, 'volume': 1_000_000},
        ]
        result = scanner._apply_filters(candidates)

        assert len(result) == 1
        assert result[0]['symbol'] == 'AAPL'


class TestVolatilityScore:
    """Test volatility score calculation."""

    def _create_sample_data(self, n_bars: int = 200, volatility: float = 0.02) -> pd.DataFrame:
        """Create sample OHLCV data with specified volatility."""
        np.random.seed(42)

        dates = pd.date_range(start='2025-01-01', periods=n_bars, freq='h')
        base_price = 100

        # Generate price path
        returns = np.random.normal(0, volatility, n_bars)
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        data = {
            'timestamp': dates,
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_bars)),
            'high': prices * (1 + np.abs(np.random.normal(0, volatility, n_bars))),
            'low': prices * (1 - np.abs(np.random.normal(0, volatility, n_bars))),
            'close': prices,
            'volume': np.random.uniform(500_000, 2_000_000, n_bars),
        }

        df = pd.DataFrame(data)
        # Ensure high >= open/close and low <= open/close
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)

        return df

    def test_score_calculation_basic(self):
        """Test basic volatility score calculation."""
        scanner = VolatilityScanner()
        df = self._create_sample_data()

        score = scanner._calculate_volatility_score(df)

        assert score > 0
        assert isinstance(score, float)

    def test_higher_volatility_higher_score(self):
        """Test that higher volatility data produces higher scores."""
        scanner = VolatilityScanner()

        df_low_vol = self._create_sample_data(volatility=0.01)
        df_high_vol = self._create_sample_data(volatility=0.05)

        score_low = scanner._calculate_volatility_score(df_low_vol)
        score_high = scanner._calculate_volatility_score(df_high_vol)

        assert score_high > score_low

    def test_score_uses_weights(self):
        """Test that custom weights affect the score."""
        df = self._create_sample_data()

        scanner_default = VolatilityScanner()
        scanner_atr_heavy = VolatilityScanner({
            'weights': {'atr_pct': 0.9, 'daily_range_pct': 0.05, 'volume_ratio': 0.05}
        })

        score_default = scanner_default._calculate_volatility_score(df)
        score_atr = scanner_atr_heavy._calculate_volatility_score(df)

        # Scores should be different due to different weights
        assert score_default != score_atr

    def test_score_handles_edge_cases(self):
        """Test score calculation handles edge cases gracefully."""
        scanner = VolatilityScanner()

        # Minimal data
        df = self._create_sample_data(n_bars=20)
        score = scanner._calculate_volatility_score(df)
        assert score >= 0

    def test_volume_ratio_capped(self):
        """Test that volume ratio is capped at 5."""
        scanner = VolatilityScanner()

        # Create data with very high recent volume
        df = self._create_sample_data()
        df.loc[df.index[-5:], 'volume'] = 100_000_000  # Spike last 5 bars

        score = scanner._calculate_volatility_score(df)

        # Score should be finite (volume ratio capped)
        assert np.isfinite(score)
        assert score < 1000  # Reasonable upper bound


class TestScanHistorical:
    """Test historical scanning with NO LOOK-AHEAD BIAS."""

    def _create_historical_data(self, symbols: list, n_bars: int = 300) -> dict:
        """Create historical data for multiple symbols."""
        data = {}
        tz = pytz.timezone('America/New_York')
        base_date = datetime(2025, 1, 1, 9, 30, tzinfo=tz)

        for i, symbol in enumerate(symbols):
            np.random.seed(42 + i)
            dates = [base_date + timedelta(hours=j) for j in range(n_bars)]
            volatility = 0.02 + i * 0.01  # Different volatility per symbol
            base_price = 50 + i * 50

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

            data[symbol] = df

        return data

    def test_scan_returns_top_n(self):
        """Test scan returns top_n symbols."""
        scanner = VolatilityScanner({'top_n': 3})
        symbols = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'AMD']
        data = self._create_historical_data(symbols)

        result = scanner.scan_historical('2025-01-10', symbols, data)

        assert len(result) <= 3
        assert all(s in symbols for s in result)

    def test_scan_filters_by_price(self):
        """Test scan applies price filters."""
        scanner = VolatilityScanner({'min_price': 100, 'top_n': 5})
        symbols = ['CHEAP', 'EXPEN']

        tz = pytz.timezone('America/New_York')
        base_date = datetime(2025, 1, 1, 9, 30, tzinfo=tz)

        # Create cheap stock (price ~25)
        np.random.seed(42)
        dates = [base_date + timedelta(hours=j) for j in range(300)]
        data = {
            'CHEAP': pd.DataFrame({
                'timestamp': dates,
                'open': [25] * 300,
                'high': [26] * 300,
                'low': [24] * 300,
                'close': [25] * 300,
                'volume': [1_000_000] * 300,
            }),
            'EXPEN': pd.DataFrame({
                'timestamp': dates,
                'open': [200] * 300,
                'high': [210] * 300,
                'low': [190] * 300,
                'close': [200] * 300,
                'volume': [1_000_000] * 300,
            }),
        }

        result = scanner.scan_historical('2025-01-10', symbols, data)

        assert 'EXPEN' in result
        assert 'CHEAP' not in result

    def test_no_lookahead_bias(self):
        """CRITICAL TEST: Verify no look-ahead bias in historical scan."""
        scanner = VolatilityScanner({'top_n': 2})
        tz = pytz.timezone('America/New_York')

        # Create data where FUTURE data would change ranking
        base_date = datetime(2025, 1, 1, 9, 30, tzinfo=tz)
        scan_date = '2025-01-05'  # Scan as of day 5
        scan_datetime = datetime(2025, 1, 5, 9, 30, tzinfo=tz)

        # STOCK_A: Low volatility before scan_date, HIGH volatility after
        # STOCK_B: High volatility before scan_date, low volatility after
        # If there's look-ahead bias, STOCK_A would rank higher

        np.random.seed(42)
        n_bars = 300
        dates = [base_date + timedelta(hours=j) for j in range(n_bars)]

        # Find split point (approximately day 5)
        split_idx = int(n_bars * 0.3)  # ~30% through data

        # STOCK_A: low vol before, high vol after
        prices_a = np.ones(n_bars) * 100
        for i in range(n_bars):
            if i < split_idx:
                prices_a[i] = 100 + np.random.uniform(-1, 1)  # Low vol
            else:
                prices_a[i] = 100 + np.random.uniform(-10, 10)  # High vol (FUTURE)

        # STOCK_B: high vol before, low vol after
        prices_b = np.ones(n_bars) * 100
        for i in range(n_bars):
            if i < split_idx:
                prices_b[i] = 100 + np.random.uniform(-10, 10)  # High vol
            else:
                prices_b[i] = 100 + np.random.uniform(-1, 1)  # Low vol (FUTURE)

        data = {
            'STOCK_A': pd.DataFrame({
                'timestamp': dates,
                'open': prices_a,
                'high': prices_a + np.abs(np.diff(np.append(prices_a, prices_a[-1]))) + 0.5,
                'low': prices_a - np.abs(np.diff(np.append(prices_a, prices_a[-1]))) - 0.5,
                'close': prices_a,
                'volume': [1_000_000] * n_bars,
            }),
            'STOCK_B': pd.DataFrame({
                'timestamp': dates,
                'open': prices_b,
                'high': prices_b + np.abs(np.diff(np.append(prices_b, prices_b[-1]))) + 0.5,
                'low': prices_b - np.abs(np.diff(np.append(prices_b, prices_b[-1]))) - 0.5,
                'close': prices_b,
                'volume': [1_000_000] * n_bars,
            }),
        }

        result = scanner.scan_historical(scan_date, ['STOCK_A', 'STOCK_B'], data)

        # STOCK_B should rank higher because it was MORE volatile BEFORE scan_date
        # If look-ahead bias exists, STOCK_A would incorrectly rank higher
        if len(result) >= 2:
            assert result[0] == 'STOCK_B', \
                f"Look-ahead bias detected! STOCK_B should rank first but got {result}"

    def test_only_uses_available_data(self):
        """Test scanner only uses data available up to target date."""
        scanner = VolatilityScanner({'top_n': 5})
        tz = pytz.timezone('America/New_York')

        # Create data that starts AFTER the scan date
        base_date = datetime(2025, 2, 1, 9, 30, tzinfo=tz)
        n_bars = 100
        dates = [base_date + timedelta(hours=j) for j in range(n_bars)]

        data = {
            'FUTURE': pd.DataFrame({
                'timestamp': dates,
                'open': [100] * n_bars,
                'high': [110] * n_bars,
                'low': [90] * n_bars,
                'close': [100] * n_bars,
                'volume': [1_000_000] * n_bars,
            })
        }

        # Scan for January (before data exists)
        result = scanner.scan_historical('2025-01-15', ['FUTURE'], data)

        # Should return empty because no data available before scan date
        assert len(result) == 0

    def test_handles_missing_symbols(self):
        """Test scanner handles missing symbols gracefully."""
        scanner = VolatilityScanner({'top_n': 5})
        symbols = ['EXISTS', 'MISSING']
        data = self._create_historical_data(['EXISTS'])

        result = scanner.scan_historical('2025-01-10', symbols, data)

        # Should only return symbols with data
        assert 'EXISTS' in result or len(result) == 0  # May be empty if filtered
        assert 'MISSING' not in result

    def test_handles_empty_data(self):
        """Test scanner handles empty historical data."""
        scanner = VolatilityScanner({'top_n': 5})
        symbols = ['AAPL', 'MSFT']

        result = scanner.scan_historical('2025-01-10', symbols, {})

        assert result == []

    def test_handles_insufficient_data(self):
        """Test scanner handles symbols with insufficient data."""
        scanner = VolatilityScanner({'top_n': 5, 'lookback_days': 14})
        tz = pytz.timezone('America/New_York')
        base_date = datetime(2025, 1, 1, 9, 30, tzinfo=tz)

        # Only 10 bars of data (insufficient for 14-day lookback)
        dates = [base_date + timedelta(hours=j) for j in range(10)]
        data = {
            'SHORT': pd.DataFrame({
                'timestamp': dates,
                'open': [100] * 10,
                'high': [110] * 10,
                'low': [90] * 10,
                'close': [100] * 10,
                'volume': [1_000_000] * 10,
            })
        }

        result = scanner.scan_historical('2025-01-10', ['SHORT'], data)

        # Should return empty because insufficient data
        assert len(result) == 0

    def test_uses_datetime_index_fallback(self):
        """Test scanner works with datetime index instead of timestamp column."""
        scanner = VolatilityScanner({'top_n': 5})
        tz = pytz.timezone('America/New_York')
        base_date = datetime(2025, 1, 1, 9, 30, tzinfo=tz)

        n_bars = 300
        dates = [base_date + timedelta(hours=j) for j in range(n_bars)]

        # Create DataFrame with datetime index (no timestamp column)
        df = pd.DataFrame({
            'open': [100] * n_bars,
            'high': [110] * n_bars,
            'low': [90] * n_bars,
            'close': [100] * n_bars,
            'volume': [1_000_000] * n_bars,
        }, index=pd.DatetimeIndex(dates))

        data = {'AAPL': df}

        result = scanner.scan_historical('2025-01-10', ['AAPL'], data)

        # Should work with datetime index
        assert 'AAPL' in result

    def test_volume_filter_historical(self):
        """Test volume filter is applied in historical scan."""
        scanner = VolatilityScanner({'min_volume': 1_000_000, 'top_n': 5})
        tz = pytz.timezone('America/New_York')
        base_date = datetime(2025, 1, 1, 9, 30, tzinfo=tz)
        n_bars = 300
        dates = [base_date + timedelta(hours=j) for j in range(n_bars)]

        data = {
            'LIQUID': pd.DataFrame({
                'timestamp': dates,
                'open': [100] * n_bars,
                'high': [110] * n_bars,
                'low': [90] * n_bars,
                'close': [100] * n_bars,
                'volume': [2_000_000] * n_bars,
            }),
            'ILLIQ': pd.DataFrame({
                'timestamp': dates,
                'open': [100] * n_bars,
                'high': [110] * n_bars,
                'low': [90] * n_bars,
                'close': [100] * n_bars,
                'volume': [100_000] * n_bars,  # Below threshold
            }),
        }

        result = scanner.scan_historical('2025-01-10', ['LIQUID', 'ILLIQ'], data)

        assert 'LIQUID' in result
        assert 'ILLIQ' not in result


class TestNoOpenBBDependency:
    """Test that scanner has no OpenBB dependencies."""

    def test_no_scan_method(self):
        """Test that live scan() method is not present."""
        scanner = VolatilityScanner()

        # Verify scan_historical exists
        assert hasattr(scanner, 'scan_historical')

        # The live scan() should not exist in minimal extraction
        assert not hasattr(scanner, 'scan')

    def test_no_universe_manager(self):
        """Test that UniverseManager is not used."""
        scanner = VolatilityScanner()

        assert not hasattr(scanner, 'universe_manager')

    def test_no_data_cache(self):
        """Test that DataCache is not used."""
        scanner = VolatilityScanner()

        assert not hasattr(scanner, 'cache')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
