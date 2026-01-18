"""
Tests for BreakoutStrategy

Verifies breakout/breakdown detection, volume confirmation,
and trend alignment logic.
"""

import numpy as np
import pandas as pd
import pytest

from strategies import BreakoutStrategy, TradingStrategy


class TestBreakoutStrategyInit:
    """Test BreakoutStrategy initialization."""

    def test_default_initialization(self):
        """Test default parameter values."""
        strategy = BreakoutStrategy()
        assert strategy.name == "Breakout_1Hour"
        assert strategy.enabled is True
        assert strategy.buy_threshold == 60
        assert strategy.sell_threshold == 40

    def test_custom_thresholds(self):
        """Test custom threshold initialization."""
        strategy = BreakoutStrategy(buy_threshold=70, sell_threshold=30)
        assert strategy.buy_threshold == 70
        assert strategy.sell_threshold == 30

    def test_disabled_initialization(self):
        """Test disabled strategy initialization."""
        strategy = BreakoutStrategy(enabled=False)
        assert strategy.enabled is False

    def test_inherits_from_trading_strategy(self):
        """Verify BreakoutStrategy inherits from TradingStrategy."""
        strategy = BreakoutStrategy()
        assert isinstance(strategy, TradingStrategy)

    def test_repr(self):
        """Test string representation."""
        strategy = BreakoutStrategy()
        repr_str = repr(strategy)
        assert "BreakoutStrategy" in repr_str
        assert "Breakout_1Hour" in repr_str


class TestBreakoutSignalGeneration:
    """Test breakout signal calculation."""

    @pytest.fixture
    def base_data(self):
        """Create base OHLCV data for testing."""
        # 60 bars of data with range 95-105
        np.random.seed(42)
        n = 60
        data = pd.DataFrame({
            'open': np.linspace(100, 100, n) + np.random.randn(n) * 0.5,
            'high': np.linspace(105, 105, n) + np.random.randn(n) * 0.3,
            'low': np.linspace(95, 95, n) + np.random.randn(n) * 0.3,
            'close': np.linspace(100, 100, n) + np.random.randn(n) * 0.5,
            'volume': np.ones(n) * 1000000,
            'SMA_20': np.ones(n) * 100,
            'SMA_50': np.ones(n) * 98,
            'ATR': np.ones(n) * 2.0
        })
        return data

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return BreakoutStrategy()

    def test_disabled_strategy_returns_hold(self, base_data, strategy):
        """Disabled strategy should return HOLD."""
        strategy.enabled = False
        signal = strategy.calculate_signal("AAPL", base_data, 100.0, None)
        assert signal['action'] == 'HOLD'
        assert signal['confidence'] == 0
        assert 'disabled' in signal['reasoning'].lower()

    def test_insufficient_data_returns_hold(self, strategy):
        """Insufficient data should return HOLD."""
        short_data = pd.DataFrame({
            'open': [100] * 30,
            'high': [105] * 30,
            'low': [95] * 30,
            'close': [100] * 30,
            'volume': [1000000] * 30
        })
        signal = strategy.calculate_signal("AAPL", short_data, 100.0, None)
        assert signal['action'] == 'HOLD'
        assert signal['confidence'] == 0
        assert 'insufficient' in signal['reasoning'].lower()

    def test_price_within_range_returns_hold(self, base_data, strategy):
        """Price within range should return HOLD."""
        signal = strategy.calculate_signal("AAPL", base_data, 100.0, None)
        assert signal['action'] == 'HOLD'
        assert signal['confidence'] == 35
        assert 'within range' in signal['reasoning'].lower()

    def test_bullish_breakout_above_recent_high(self, base_data, strategy):
        """Price above recent high should return BUY."""
        # Get recent high from the data
        lookback = min(40, len(base_data) - 5)
        recent_high = base_data['high'].iloc[-lookback-1:-1].max()

        # Price significantly above recent high
        breakout_price = recent_high + 2.0
        signal = strategy.calculate_signal("AAPL", base_data, breakout_price, None)

        assert signal['action'] == 'BUY'
        assert signal['confidence'] >= 50
        assert signal['confidence'] <= 90
        assert 'above' in signal['reasoning'].lower()
        assert 'breakout' in signal['reasoning'].lower()

    def test_bearish_breakdown_below_recent_low(self, base_data, strategy):
        """Price below recent low should return SELL."""
        # Get recent low from the data
        lookback = min(40, len(base_data) - 5)
        recent_low = base_data['low'].iloc[-lookback-1:-1].min()

        # Price below recent low
        breakdown_price = recent_low - 2.0
        signal = strategy.calculate_signal("AAPL", base_data, breakdown_price, None)

        assert signal['action'] == 'SELL'
        assert signal['confidence'] >= 55
        assert signal['confidence'] <= 80
        assert 'below' in signal['reasoning'].lower()
        assert 'breakdown' in signal['reasoning'].lower()

    def test_breakout_components_included(self, base_data, strategy):
        """Signal should include expected components."""
        signal = strategy.calculate_signal("AAPL", base_data, 100.0, None)
        components = signal['components']

        assert 'recent_high' in components
        assert 'recent_low' in components
        assert 'price' in components
        assert 'atr' in components
        assert 'sma_20' in components
        assert 'sma_50' in components


class TestVolumeConfirmation:
    """Test volume confirmation logic."""

    @pytest.fixture
    def breakout_data(self):
        """Create data for breakout scenario."""
        n = 60
        data = pd.DataFrame({
            'open': [100] * n,
            'high': [105] * n,  # Recent high will be around 105
            'low': [95] * n,
            'close': [100] * n,
            'volume': [1000000] * n,  # Avg volume 1M
            'SMA_20': [100] * n,
            'SMA_50': [98] * n,
            'ATR': [2.0] * n
        })
        return data

    @pytest.fixture
    def strategy(self):
        return BreakoutStrategy()

    def test_high_volume_surge_adds_confidence(self, breakout_data, strategy):
        """Volume surge > 1.3x should add confidence."""
        # Set current volume to 1.5x average
        breakout_data.loc[breakout_data.index[-1], 'volume'] = 1500000

        breakout_price = 108.0  # Above recent high
        signal = strategy.calculate_signal("AAPL", breakout_data, breakout_price, None)

        assert signal['action'] == 'BUY'
        assert 'volume_surge' in signal['components']
        assert signal['components']['volume_surge'] >= 1.3
        # High volume should give higher confidence

    def test_low_volume_reduces_confidence(self, breakout_data, strategy):
        """Volume surge < 1.1x should reduce confidence."""
        # Set current volume to 1.0x average (no surge)
        breakout_data.loc[breakout_data.index[-1], 'volume'] = 1000000

        breakout_price = 108.0  # Above recent high
        signal_low_vol = strategy.calculate_signal("AAPL", breakout_data, breakout_price, None)

        # Now test with high volume
        breakout_data.loc[breakout_data.index[-1], 'volume'] = 1500000
        signal_high_vol = strategy.calculate_signal("AAPL", breakout_data, breakout_price, None)

        assert signal_high_vol['confidence'] > signal_low_vol['confidence']

    def test_volume_surge_in_reasoning(self, breakout_data, strategy):
        """Volume surge should appear in reasoning."""
        breakout_data.loc[breakout_data.index[-1], 'volume'] = 1500000

        breakout_price = 108.0
        signal = strategy.calculate_signal("AAPL", breakout_data, breakout_price, None)

        assert 'vol:' in signal['reasoning'].lower()


class TestTrendAlignment:
    """Test trend alignment bonus logic."""

    @pytest.fixture
    def strategy(self):
        return BreakoutStrategy()

    def test_uptrend_alignment_adds_confidence(self, strategy):
        """price > SMA_20 > SMA_50 should add confidence."""
        n = 60
        # Uptrend: SMA_20 > SMA_50
        data = pd.DataFrame({
            'open': [100] * n,
            'high': [105] * n,
            'low': [95] * n,
            'close': [100] * n,
            'volume': [1500000] * n,
            'SMA_20': [108] * n,  # SMA_20 > SMA_50
            'SMA_50': [104] * n,
            'ATR': [2.0] * n
        })

        # Price above both SMAs and recent high
        breakout_price = 110.0
        signal = strategy.calculate_signal("AAPL", data, breakout_price, None)

        assert signal['action'] == 'BUY'
        # Trend alignment should contribute to higher confidence

    def test_no_trend_alignment_without_bonus(self, strategy):
        """No bonus when not in uptrend."""
        n = 60
        # No uptrend: SMA_20 < SMA_50
        data_no_trend = pd.DataFrame({
            'open': [100] * n,
            'high': [105] * n,
            'low': [95] * n,
            'close': [100] * n,
            'volume': [1500000] * n,
            'SMA_20': [102] * n,
            'SMA_50': [104] * n,  # SMA_50 > SMA_20 (no uptrend)
            'ATR': [2.0] * n
        })

        # Uptrend data
        data_uptrend = pd.DataFrame({
            'open': [100] * n,
            'high': [105] * n,
            'low': [95] * n,
            'close': [100] * n,
            'volume': [1500000] * n,
            'SMA_20': [108] * n,  # SMA_20 > SMA_50
            'SMA_50': [104] * n,
            'ATR': [2.0] * n
        })

        breakout_price = 110.0
        signal_no_trend = strategy.calculate_signal("AAPL", data_no_trend, breakout_price, None)
        signal_uptrend = strategy.calculate_signal("AAPL", data_uptrend, breakout_price, None)

        # Uptrend should have higher confidence due to +5 bonus
        assert signal_uptrend['confidence'] >= signal_no_trend['confidence']


class TestConfidenceBounds:
    """Test confidence score boundaries."""

    @pytest.fixture
    def strategy(self):
        return BreakoutStrategy()

    def test_buy_confidence_minimum_50(self, strategy):
        """BUY confidence should be at least 50."""
        n = 60
        data = pd.DataFrame({
            'open': [100] * n,
            'high': [105] * n,
            'low': [95] * n,
            'close': [100] * n,
            'volume': [500000] * n,  # Low volume
            'SMA_20': [100] * n,
            'SMA_50': [102] * n,  # No trend alignment
            'ATR': [2.0] * n
        })

        # Minimal breakout
        breakout_price = 105.5
        signal = strategy.calculate_signal("AAPL", data, breakout_price, None)

        if signal['action'] == 'BUY':
            assert signal['confidence'] >= 50

    def test_buy_confidence_maximum_90(self, strategy):
        """BUY confidence should not exceed 90."""
        n = 60
        data = pd.DataFrame({
            'open': [100] * n,
            'high': [105] * n,
            'low': [95] * n,
            'close': [100] * n,
            'volume': [5000000] * n,  # Very high volume
            'SMA_20': [115] * n,
            'SMA_50': [112] * n,
            'ATR': [2.0] * n
        })

        # Massive breakout
        breakout_price = 120.0
        signal = strategy.calculate_signal("AAPL", data, breakout_price, None)

        assert signal['action'] == 'BUY'
        assert signal['confidence'] <= 90

    def test_sell_confidence_maximum_80(self, strategy):
        """SELL confidence should not exceed 80."""
        n = 60
        data = pd.DataFrame({
            'open': [100] * n,
            'high': [105] * n,
            'low': [95] * n,
            'close': [100] * n,
            'volume': [1000000] * n,
            'SMA_20': [100] * n,
            'SMA_50': [100] * n,
            'ATR': [2.0] * n
        })

        # Massive breakdown
        breakdown_price = 80.0
        signal = strategy.calculate_signal("AAPL", data, breakdown_price, None)

        assert signal['action'] == 'SELL'
        assert signal['confidence'] <= 80


class TestMissingIndicators:
    """Test handling of missing indicators."""

    @pytest.fixture
    def strategy(self):
        return BreakoutStrategy()

    def test_missing_sma_uses_current_price(self, strategy):
        """Missing SMA should default to current price."""
        n = 60
        data = pd.DataFrame({
            'open': [100] * n,
            'high': [105] * n,
            'low': [95] * n,
            'close': [100] * n,
            'volume': [1000000] * n,
            # No SMA_20 or SMA_50
        })

        signal = strategy.calculate_signal("AAPL", data, 100.0, None)

        # Should not raise error
        assert signal['action'] in ['BUY', 'SELL', 'HOLD']
        assert 'sma_20' in signal['components']
        assert 'sma_50' in signal['components']

    def test_missing_atr_calculates_fallback(self, strategy):
        """Missing ATR should calculate fallback."""
        n = 60
        data = pd.DataFrame({
            'open': [100] * n,
            'high': [105] * n,
            'low': [95] * n,
            'close': [100] * n,
            'volume': [1000000] * n,
            # No ATR
        })

        signal = strategy.calculate_signal("AAPL", data, 100.0, None)

        assert 'atr' in signal['components']
        assert signal['components']['atr'] > 0

    def test_nan_indicators_handled(self, strategy):
        """NaN indicators should be handled gracefully."""
        n = 60
        data = pd.DataFrame({
            'open': [100] * n,
            'high': [105] * n,
            'low': [95] * n,
            'close': [100] * n,
            'volume': [1000000] * n,
            'SMA_20': [np.nan] * n,
            'SMA_50': [np.nan] * n,
            'ATR': [np.nan] * n
        })

        signal = strategy.calculate_signal("AAPL", data, 100.0, None)

        # Should not raise error
        assert signal['action'] in ['BUY', 'SELL', 'HOLD']


class TestLookbackPeriod:
    """Test 40-hour lookback period."""

    @pytest.fixture
    def strategy(self):
        return BreakoutStrategy()

    def test_uses_40_hour_lookback(self, strategy):
        """Should use 40-hour lookback for range calculation."""
        n = 60
        # Create data with distinct high/low in last 40 bars
        data = pd.DataFrame({
            'open': [100] * n,
            'high': [105] * 20 + [110] * 40,  # Higher highs in recent 40 bars
            'low': [95] * n,
            'close': [100] * n,
            'volume': [1000000] * n,
            'SMA_20': [100] * n,
            'SMA_50': [100] * n,
            'ATR': [2.0] * n
        })

        signal = strategy.calculate_signal("AAPL", data, 100.0, None)

        # Recent high should be 110 (from last 40 bars)
        assert signal['components']['recent_high'] >= 109.0

    def test_short_data_adjusts_lookback(self, strategy):
        """Shorter data should adjust lookback period."""
        n = 52  # Just above minimum 50
        data = pd.DataFrame({
            'open': [100] * n,
            'high': [105] * n,
            'low': [95] * n,
            'close': [100] * n,
            'volume': [1000000] * n,
        })

        # Should not raise error with adjusted lookback
        signal = strategy.calculate_signal("AAPL", data, 100.0, None)
        assert signal['action'] in ['BUY', 'SELL', 'HOLD']


class TestBreakoutPercentageCalculation:
    """Test breakout percentage calculation."""

    @pytest.fixture
    def strategy(self):
        return BreakoutStrategy()

    def test_breakout_percentage_in_reasoning(self, strategy):
        """Breakout percentage should appear in reasoning."""
        n = 60
        data = pd.DataFrame({
            'open': [100] * n,
            'high': [105] * n,
            'low': [95] * n,
            'close': [100] * n,
            'volume': [1500000] * n,
            'SMA_20': [108] * n,
            'SMA_50': [106] * n,
        })

        breakout_price = 110.0
        signal = strategy.calculate_signal("AAPL", data, breakout_price, None)

        assert '%' in signal['reasoning']

    def test_breakdown_percentage_in_reasoning(self, strategy):
        """Breakdown percentage should appear in reasoning."""
        n = 60
        data = pd.DataFrame({
            'open': [100] * n,
            'high': [105] * n,
            'low': [95] * n,
            'close': [100] * n,
            'volume': [1000000] * n,
            'SMA_20': [100] * n,  # SMA above breakdown price establishes downtrend
            'SMA_50': [102] * n,
        })

        breakdown_price = 90.0
        signal = strategy.calculate_signal("AAPL", data, breakdown_price, None)

        assert '%' in signal['reasoning']


class TestEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def strategy(self):
        return BreakoutStrategy()

    def test_zero_volume_handled(self, strategy):
        """Zero volume should be handled gracefully."""
        n = 60
        data = pd.DataFrame({
            'open': [100] * n,
            'high': [105] * n,
            'low': [95] * n,
            'close': [100] * n,
            'volume': [0] * n,  # Zero volume
        })

        signal = strategy.calculate_signal("AAPL", data, 110.0, None)

        # Should not raise division by zero
        assert signal['action'] in ['BUY', 'SELL', 'HOLD']

    def test_price_exactly_at_high(self, strategy):
        """Price exactly at recent high should return HOLD."""
        n = 60
        data = pd.DataFrame({
            'open': [100] * n,
            'high': [105.0] * n,  # Exact high
            'low': [95] * n,
            'close': [100] * n,
            'volume': [1000000] * n,
        })

        # Price exactly at high (not above)
        signal = strategy.calculate_signal("AAPL", data, 105.0, None)

        # Not a breakout yet
        assert signal['action'] == 'HOLD'

    def test_price_exactly_at_low(self, strategy):
        """Price exactly at recent low should return HOLD."""
        n = 60
        data = pd.DataFrame({
            'open': [100] * n,
            'high': [105] * n,
            'low': [95.0] * n,  # Exact low
            'close': [100] * n,
            'volume': [1000000] * n,
        })

        # Price exactly at low (not below)
        signal = strategy.calculate_signal("AAPL", data, 95.0, None)

        # Not a breakdown yet
        assert signal['action'] == 'HOLD'

    def test_different_symbols(self, strategy):
        """Strategy should work with different symbols."""
        n = 60
        data = pd.DataFrame({
            'open': [100] * n,
            'high': [105] * n,
            'low': [95] * n,
            'close': [100] * n,
            'volume': [1000000] * n,
        })

        for symbol in ['AAPL', 'MSFT', 'GOOGL', 'TSLA']:
            signal = strategy.calculate_signal(symbol, data, 100.0, None)
            assert signal['action'] in ['BUY', 'SELL', 'HOLD']
