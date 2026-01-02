"""
Tests for MeanReversionStrategy

Tests cover:
- Strategy initialization with defaults and custom parameters
- Volatility filter (ATR% > max blocks signals)
- BUY conditions: RSI < 32 AND BB position < 0.15 AND in uptrend
- SELL conditions: RSI > 72 AND BB position > 0.85
- HOLD when oversold in downtrend (don't catch falling knife)
- HOLD in neutral zone
- Edge cases: disabled strategy, insufficient data, missing indicators
"""

import pandas as pd
import pytest

from strategies.mean_reversion import MeanReversionStrategy


def make_data(rows: int = 30, **kwargs) -> pd.DataFrame:
    """Helper to create test DataFrames with proper length.

    Args:
        rows: Number of rows (default 30)
        **kwargs: Column name -> value (scalar or list)

    Returns:
        DataFrame with all columns having 'rows' length
    """
    data = {'Close': [100.0] * rows}
    for col, val in kwargs.items():
        if isinstance(val, list):
            data[col] = val
        else:
            data[col] = [val] * rows
    return pd.DataFrame(data)


class TestMeanReversionStrategyInit:
    """Tests for strategy initialization."""

    def test_default_initialization(self):
        """Strategy initializes with correct defaults."""
        strategy = MeanReversionStrategy()

        assert strategy.name == "MeanReversion_1Hour"
        assert strategy.enabled is True
        assert strategy.buy_threshold == 55
        assert strategy.sell_threshold == 70
        assert strategy.max_atr_pct == 2.5

    def test_custom_initialization(self):
        """Strategy accepts custom parameters."""
        strategy = MeanReversionStrategy(
            buy_threshold=60,
            sell_threshold=75,
            enabled=False,
            max_atr_pct=3.0
        )

        assert strategy.buy_threshold == 60
        assert strategy.sell_threshold == 75
        assert strategy.enabled is False
        assert strategy.max_atr_pct == 3.0

    def test_inherits_from_trading_strategy(self):
        """Strategy inherits from TradingStrategy base class."""
        from strategies.base import TradingStrategy
        strategy = MeanReversionStrategy()
        assert isinstance(strategy, TradingStrategy)


class TestVolatilityFilter:
    """Tests for volatility filter (key feature of mean reversion)."""

    def test_high_volatility_blocks_signal(self):
        """High ATR% should block any signal (return HOLD)."""
        strategy = MeanReversionStrategy(max_atr_pct=2.5)

        # Create data with high ATR (3% > 2.5% threshold)
        data = make_data(
            ATR=3.0,  # 3% of $100
            RSI=25.0,  # Very oversold (would normally trigger BUY)
            BB_LOWER=98.0,
            BB_UPPER=102.0,
            SMA_50=99.0  # Uptrend
        )

        signal = strategy.calculate_signal('TSLA', data, 100.0, None)

        assert signal['action'] == 'HOLD'
        assert signal['confidence'] == 0
        assert 'Volatility too high' in signal['reasoning']
        assert signal['components']['atr_pct'] == 3.0
        assert signal['components']['max_atr_pct'] == 2.5

    def test_low_volatility_allows_signal(self):
        """Low ATR% should allow normal signal processing."""
        strategy = MeanReversionStrategy(max_atr_pct=2.5)

        # Create buy condition data with low volatility
        data = make_data(
            ATR=1.5,  # 1.5% < 2.5% threshold
            RSI=25.0,  # Oversold
            BB_LOWER=99.5,  # Price at 100, lower BB at 99.5
            BB_UPPER=103.5,  # Upper BB at 103.5
            SMA_50=99.0  # Uptrend (price > SMA)
        )

        signal = strategy.calculate_signal('MSFT', data, 100.0, None)

        # Should process normally (BUY signal with this data)
        assert signal['action'] == 'BUY'
        assert signal['confidence'] > 0

    def test_volatility_filter_with_missing_atr(self):
        """Missing ATR should not block signal."""
        strategy = MeanReversionStrategy()

        data = make_data(
            # No ATR column
            RSI=25.0,
            BB_LOWER=99.5,
            BB_UPPER=103.5,
            SMA_50=99.0
        )

        signal = strategy.calculate_signal('AAPL', data, 100.0, None)

        # Should process normally without volatility filter
        assert signal['action'] == 'BUY'

    def test_volatility_filter_with_nan_atr(self):
        """NaN ATR should not block signal."""
        strategy = MeanReversionStrategy()

        data = make_data(
            ATR=float('nan'),
            RSI=25.0,
            BB_LOWER=99.5,
            BB_UPPER=103.5,
            SMA_50=99.0
        )

        signal = strategy.calculate_signal('AAPL', data, 100.0, None)

        assert signal['action'] == 'BUY'

    def test_volatility_filter_custom_threshold(self):
        """Custom max_atr_pct threshold works correctly."""
        strategy = MeanReversionStrategy(max_atr_pct=5.0)  # Higher threshold

        # 4% ATR should pass with 5% threshold
        data = make_data(
            ATR=4.0,  # 4% < 5% threshold
            RSI=25.0,
            BB_LOWER=99.5,
            BB_UPPER=103.5,
            SMA_50=99.0
        )

        signal = strategy.calculate_signal('TEST', data, 100.0, None)

        assert signal['action'] == 'BUY'  # Not blocked


class TestBuySignal:
    """Tests for BUY signal conditions: RSI < 32 AND BB position < 0.15 AND uptrend."""

    def test_buy_signal_all_conditions_met(self):
        """BUY when RSI < 32, BB position < 0.15, and in uptrend."""
        strategy = MeanReversionStrategy()

        # RSI=25, price at 99.5, BB range 99-104
        # position = (99.5-99)/(104-99) = 0.5/5 = 0.1 < 0.15
        data = make_data(
            ATR=1.0,  # Low volatility
            RSI=25.0,  # < 32
            BB_LOWER=99.0,
            BB_UPPER=104.0,
            SMA_50=98.0  # price > SMA = uptrend
        )

        signal = strategy.calculate_signal('AAPL', data, 99.5, None)

        assert signal['action'] == 'BUY'
        assert signal['confidence'] >= 75
        assert signal['confidence'] <= 95
        assert 'Oversold' in signal['reasoning']
        assert 'uptrend' in signal['reasoning']
        assert signal['components']['in_uptrend'] == True

    def test_buy_confidence_calculation(self):
        """Buy confidence increases as RSI decreases."""
        strategy = MeanReversionStrategy()

        # Test with RSI=20 (very oversold)
        data = make_data(
            ATR=1.0,
            RSI=20.0,  # Very oversold
            BB_LOWER=99.0,
            BB_UPPER=104.0,
            SMA_50=98.0
        )

        signal = strategy.calculate_signal('AAPL', data, 99.5, None)

        # Confidence = 75 + (32 - 20) / 2 = 75 + 6 = 81
        assert signal['action'] == 'BUY'
        assert signal['confidence'] == 81

    def test_buy_confidence_capped_at_95(self):
        """Buy confidence is capped at 95."""
        strategy = MeanReversionStrategy()

        # RSI=-50 would give 75 + 41 = 116, capped to 95
        data = make_data(
            ATR=1.0,
            RSI=-50.0,  # Extreme (unrealistic but tests cap)
            BB_LOWER=99.0,
            BB_UPPER=104.0,
            SMA_50=98.0
        )

        signal = strategy.calculate_signal('AAPL', data, 99.5, None)

        assert signal['confidence'] == 95

    def test_no_buy_when_rsi_too_high(self):
        """No BUY when RSI >= 32."""
        strategy = MeanReversionStrategy()

        data = make_data(
            ATR=1.0,
            RSI=35.0,  # >= 32
            BB_LOWER=99.0,
            BB_UPPER=104.0,
            SMA_50=98.0
        )

        signal = strategy.calculate_signal('AAPL', data, 99.5, None)

        assert signal['action'] != 'BUY'

    def test_no_buy_when_bb_position_too_high(self):
        """No BUY when BB position >= 0.15."""
        strategy = MeanReversionStrategy()

        data = make_data(
            ATR=1.0,
            RSI=25.0,  # Oversold
            BB_LOWER=95.0,
            BB_UPPER=105.0,  # Position = (100-95)/(105-95) = 0.5 >= 0.15
            SMA_50=98.0
        )

        signal = strategy.calculate_signal('AAPL', data, 100.0, None)

        assert signal['action'] != 'BUY'

    def test_no_buy_in_downtrend(self):
        """No BUY when in downtrend (price < SMA_50)."""
        strategy = MeanReversionStrategy()

        data = make_data(
            ATR=1.0,
            RSI=25.0,  # Oversold
            BB_LOWER=99.0,
            BB_UPPER=104.0,
            SMA_50=102.0  # price < SMA = downtrend
        )

        signal = strategy.calculate_signal('AAPL', data, 99.5, None)

        assert signal['action'] == 'HOLD'
        assert signal['components']['in_uptrend'] == False


class TestSellSignal:
    """Tests for SELL signal conditions: RSI > 72 AND BB position > 0.85."""

    def test_sell_signal_all_conditions_met(self):
        """SELL when RSI > 72 and BB position > 0.85."""
        strategy = MeanReversionStrategy()

        # BB position = (104 - 95) / (105 - 95) = 9/10 = 0.9 > 0.85
        data = make_data(
            ATR=1.0,
            RSI=78.0,  # > 72
            BB_LOWER=95.0,
            BB_UPPER=105.0,
            SMA_50=100.0
        )

        signal = strategy.calculate_signal('AAPL', data, 104.0, None)

        assert signal['action'] == 'SELL'
        assert signal['confidence'] >= 65
        assert signal['confidence'] <= 85
        assert 'Overbought' in signal['reasoning']

    def test_sell_confidence_calculation(self):
        """Sell confidence increases as RSI increases."""
        strategy = MeanReversionStrategy()

        data = make_data(
            ATR=1.0,
            RSI=80.0,  # > 72
            BB_LOWER=95.0,
            BB_UPPER=105.0,
            SMA_50=100.0
        )

        signal = strategy.calculate_signal('AAPL', data, 104.0, None)

        # Confidence = 65 + (80 - 72) / 2 = 65 + 4 = 69
        assert signal['action'] == 'SELL'
        assert signal['confidence'] == 69

    def test_sell_confidence_capped_at_85(self):
        """Sell confidence is capped at 85."""
        strategy = MeanReversionStrategy()

        data = make_data(
            ATR=1.0,
            RSI=150.0,  # Extreme (unrealistic but tests cap)
            BB_LOWER=95.0,
            BB_UPPER=105.0,
            SMA_50=100.0
        )

        signal = strategy.calculate_signal('AAPL', data, 104.0, None)

        assert signal['confidence'] == 85

    def test_no_sell_when_rsi_too_low(self):
        """No SELL when RSI <= 72."""
        strategy = MeanReversionStrategy()

        data = make_data(
            ATR=1.0,
            RSI=70.0,  # <= 72
            BB_LOWER=95.0,
            BB_UPPER=105.0,
            SMA_50=100.0
        )

        signal = strategy.calculate_signal('AAPL', data, 104.0, None)

        assert signal['action'] != 'SELL'

    def test_no_sell_when_bb_position_too_low(self):
        """No SELL when BB position <= 0.85."""
        strategy = MeanReversionStrategy()

        data = make_data(
            ATR=1.0,
            RSI=78.0,  # Overbought
            BB_LOWER=95.0,
            BB_UPPER=105.0,  # Position = (100-95)/(105-95) = 0.5 <= 0.85
            SMA_50=100.0
        )

        signal = strategy.calculate_signal('AAPL', data, 100.0, None)

        assert signal['action'] != 'SELL'


class TestDowntrendProtection:
    """Tests for protection against buying in downtrend."""

    def test_hold_when_oversold_in_downtrend(self):
        """HOLD with specific message when oversold in downtrend."""
        strategy = MeanReversionStrategy()

        data = make_data(
            ATR=1.0,
            RSI=30.0,  # < 35, oversold
            BB_LOWER=98.0,
            BB_UPPER=102.0,
            SMA_50=105.0  # Price (100) < SMA (105) = downtrend
        )

        signal = strategy.calculate_signal('AAPL', data, 100.0, None)

        assert signal['action'] == 'HOLD'
        assert signal['confidence'] == 30
        assert 'downtrend' in signal['reasoning']
        assert signal['components']['in_uptrend'] == False

    def test_downtrend_threshold_is_35(self):
        """Downtrend protection triggers when RSI < 35."""
        strategy = MeanReversionStrategy()

        # RSI = 36 should NOT trigger downtrend protection
        data = make_data(
            ATR=1.0,
            RSI=36.0,  # >= 35
            BB_LOWER=98.0,
            BB_UPPER=102.0,
            SMA_50=105.0  # Downtrend
        )

        signal = strategy.calculate_signal('AAPL', data, 100.0, None)

        # Should be neutral HOLD, not downtrend HOLD
        assert signal['action'] == 'HOLD'
        assert 'Neutral zone' in signal['reasoning']


class TestNeutralZone:
    """Tests for neutral zone behavior."""

    def test_neutral_zone_hold(self):
        """HOLD with neutral reasoning when no conditions met."""
        strategy = MeanReversionStrategy()

        data = make_data(
            ATR=1.0,
            RSI=50.0,  # Neutral
            BB_LOWER=95.0,
            BB_UPPER=105.0,
            SMA_50=100.0
        )

        signal = strategy.calculate_signal('AAPL', data, 100.0, None)

        assert signal['action'] == 'HOLD'
        assert signal['confidence'] == 40
        assert 'Neutral zone' in signal['reasoning']
        assert 'RSI 50' in signal['reasoning']


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_strategy_disabled(self):
        """Disabled strategy returns HOLD with 0 confidence."""
        strategy = MeanReversionStrategy(enabled=False)

        data = make_data(
            RSI=25.0,
            BB_LOWER=99.0,
            BB_UPPER=104.0,
            SMA_50=98.0
        )

        signal = strategy.calculate_signal('AAPL', data, 99.5, None)

        assert signal['action'] == 'HOLD'
        assert signal['confidence'] == 0
        assert 'disabled' in signal['reasoning']
        assert signal['components'] == {}

    def test_insufficient_data(self):
        """Returns HOLD when less than 30 bars of data."""
        strategy = MeanReversionStrategy()

        data = make_data(
            rows=29,  # Only 29 rows
            RSI=25.0,
            BB_LOWER=99.0,
            BB_UPPER=104.0,
            SMA_50=98.0
        )

        signal = strategy.calculate_signal('AAPL', data, 100.0, None)

        assert signal['action'] == 'HOLD'
        assert signal['confidence'] == 0
        assert 'Insufficient data' in signal['reasoning']

    def test_missing_rsi_defaults_to_50(self):
        """Missing RSI defaults to 50 (neutral)."""
        strategy = MeanReversionStrategy()

        data = make_data(
            ATR=1.0,
            # No RSI column
            BB_LOWER=95.0,
            BB_UPPER=105.0,
            SMA_50=100.0
        )

        signal = strategy.calculate_signal('AAPL', data, 100.0, None)

        assert signal['components']['rsi'] == 50.0

    def test_nan_rsi_defaults_to_50(self):
        """NaN RSI defaults to 50."""
        strategy = MeanReversionStrategy()

        data = make_data(
            ATR=1.0,
            RSI=float('nan'),
            BB_LOWER=95.0,
            BB_UPPER=105.0,
            SMA_50=100.0
        )

        signal = strategy.calculate_signal('AAPL', data, 100.0, None)

        assert signal['components']['rsi'] == 50.0

    def test_missing_bollinger_bands_use_defaults(self):
        """Missing BB uses 2% range around price."""
        strategy = MeanReversionStrategy()

        data = make_data(
            ATR=1.0,
            RSI=50.0,
            # No BB columns
            SMA_50=100.0
        )

        signal = strategy.calculate_signal('AAPL', data, 100.0, None)

        # Default: bb_lower = price * 0.98 = 98
        # Default: bb_upper = price * 1.02 = 102
        assert signal['components']['bb_lower'] == 98.0
        assert signal['components']['bb_upper'] == 102.0

    def test_bb_lower_column_case_sensitivity(self):
        """Handles both BB_LOWER and BB_Lower column names."""
        strategy = MeanReversionStrategy()

        # Test lowercase version
        data = make_data(
            ATR=1.0,
            RSI=25.0,
            BB_Lower=99.0,  # Lowercase
            BB_Upper=104.0,  # Lowercase
            SMA_50=98.0
        )

        signal = strategy.calculate_signal('AAPL', data, 99.5, None)

        assert signal['components']['bb_lower'] == 99.0
        assert signal['components']['bb_upper'] == 104.0

    def test_zero_bb_range_defaults_to_half(self):
        """BB position defaults to 0.5 when BB range is zero."""
        strategy = MeanReversionStrategy()

        data = make_data(
            ATR=1.0,
            RSI=50.0,
            BB_LOWER=100.0,  # Same as upper
            BB_UPPER=100.0,  # Creates zero range
            SMA_50=100.0
        )

        signal = strategy.calculate_signal('AAPL', data, 100.0, None)

        assert signal['components']['bb_position'] == 0.5

    def test_zero_price_avoids_division_error(self):
        """Zero price handles ATR calculation safely."""
        strategy = MeanReversionStrategy()

        data = make_data(
            ATR=1.0,
            RSI=50.0,
            BB_LOWER=95.0,
            BB_UPPER=105.0,
            SMA_50=100.0
        )

        # Zero price - ATR % calculation should skip
        signal = strategy.calculate_signal('AAPL', data, 0.0, None)

        # Should not crash, should return neutral
        assert signal['action'] == 'HOLD'


class TestComponentsOutput:
    """Tests for signal components output."""

    def test_components_include_all_fields(self):
        """Components dict includes all expected fields."""
        strategy = MeanReversionStrategy()

        data = make_data(
            ATR=1.0,
            RSI=50.0,
            BB_LOWER=95.0,
            BB_UPPER=105.0,
            SMA_50=100.0
        )

        signal = strategy.calculate_signal('AAPL', data, 100.0, None)

        components = signal['components']
        assert 'rsi' in components
        assert 'bb_position' in components
        assert 'price' in components
        assert 'bb_lower' in components
        assert 'bb_upper' in components
        assert 'sma_50' in components
        assert 'in_uptrend' in components

    def test_components_are_rounded(self):
        """Numeric components are properly rounded."""
        strategy = MeanReversionStrategy()

        data = make_data(
            ATR=1.0,
            RSI=50.12345,
            BB_LOWER=95.6789,
            BB_UPPER=105.1234,
            SMA_50=100.5678
        )

        signal = strategy.calculate_signal('AAPL', data, 100.0, None)

        components = signal['components']
        # RSI rounded to 1 decimal
        assert components['rsi'] == 50.1
        # BB values rounded to 2 decimals
        assert components['bb_lower'] == 95.68
        assert components['bb_upper'] == 105.12
        assert components['sma_50'] == 100.57


class TestImportability:
    """Tests for module imports."""

    def test_import_from_strategies_package(self):
        """MeanReversionStrategy can be imported from strategies package."""
        from strategies import MeanReversionStrategy as MRS
        assert MRS is not None

    def test_import_from_module(self):
        """MeanReversionStrategy can be imported from module."""
        from strategies.mean_reversion import MeanReversionStrategy as MRS
        assert MRS is not None
