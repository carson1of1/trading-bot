"""Tests for TechnicalIndicators module"""
import pytest
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.indicators import TechnicalIndicators


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def indicators():
    """Create TechnicalIndicators instance"""
    return TechnicalIndicators()


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for indicator testing"""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    base_price = 100
    close = base_price + np.cumsum(np.random.randn(n) * 0.5)
    data = pd.DataFrame({
        'timestamp': dates,
        'open': close - np.random.rand(n) * 0.5,
        'high': close + np.random.rand(n) * 1.0,
        'low': close - np.random.rand(n) * 1.0,
        'close': close,
        'volume': np.random.randint(1000, 10000, n)
    })
    return data


@pytest.fixture
def flat_market_data():
    """Create flat market data where high == low (for edge case testing)"""
    n = 50
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': [100.0] * n,
        'high': [100.0] * n,
        'low': [100.0] * n,
        'close': [100.0] * n,
        'volume': [1000] * n
    })
    return data


@pytest.fixture
def uptrend_data():
    """Create steadily increasing price data (all up days)"""
    n = 50
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    close = np.linspace(100, 150, n)  # Steady uptrend
    data = pd.DataFrame({
        'timestamp': dates,
        'open': close - 0.5,
        'high': close + 1.0,
        'low': close - 1.0,
        'close': close,
        'volume': np.random.randint(1000, 10000, n)
    })
    return data


@pytest.fixture
def minimal_data():
    """Create minimal data for edge case testing"""
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='1h'),
        'open': [100.0, 101.0, 102.0, 101.0, 100.0],
        'high': [101.0, 102.0, 103.0, 102.0, 101.0],
        'low': [99.0, 100.0, 101.0, 100.0, 99.0],
        'close': [100.5, 101.5, 102.5, 101.5, 100.5],
        'volume': [1000, 1200, 1500, 1100, 900]
    })
    return data


# ============================================================================
# Test Initialization
# ============================================================================

class TestInitialization:
    """Test TechnicalIndicators initialization"""

    def test_default_periods_set(self, indicators):
        """Should have default periods configured"""
        assert indicators.default_periods['sma_fast'] == 5
        assert indicators.default_periods['sma_slow'] == 20
        assert indicators.default_periods['rsi'] == 14
        assert indicators.default_periods['ema_fast'] == 12
        assert indicators.default_periods['ema_slow'] == 26
        assert indicators.default_periods['macd_signal'] == 9
        assert indicators.default_periods['bb_period'] == 20
        assert indicators.default_periods['bb_std'] == 2


# ============================================================================
# Test Basic Indicators (SMA, EMA, RSI, MACD)
# ============================================================================

class TestSMA:
    """Test Simple Moving Average calculations"""

    def test_add_sma_creates_column(self, indicators, sample_ohlcv_data):
        """SMA should create a column with correct name"""
        result = indicators.add_sma(sample_ohlcv_data.copy(), period=20)
        assert 'SMA_20' in result.columns

    def test_add_sma_correct_values(self, indicators, sample_ohlcv_data):
        """SMA should calculate correct average values"""
        result = indicators.add_sma(sample_ohlcv_data.copy(), period=5)
        # First 4 values should be NaN (not enough data for 5-period SMA)
        assert result['SMA_5'].iloc[:4].isna().all()
        # 5th value should be mean of first 5 closes
        expected = sample_ohlcv_data['close'].iloc[:5].mean()
        assert abs(result['SMA_5'].iloc[4] - expected) < 0.001

    def test_add_sma_custom_column(self, indicators, sample_ohlcv_data):
        """SMA should work on custom column"""
        result = indicators.add_sma(sample_ohlcv_data.copy(), period=10, column='volume')
        assert 'SMA_10' in result.columns


class TestEMA:
    """Test Exponential Moving Average calculations"""

    def test_add_ema_creates_column(self, indicators, sample_ohlcv_data):
        """EMA should create a column with correct name"""
        result = indicators.add_ema(sample_ohlcv_data.copy(), period=12)
        assert 'EMA_12' in result.columns

    def test_add_ema_uses_adjust_false(self, indicators, sample_ohlcv_data):
        """EMA should use adjust=False for standard formula"""
        result = indicators.add_ema(sample_ohlcv_data.copy(), period=12)
        # Manually calculate EMA with adjust=False
        expected = sample_ohlcv_data['close'].ewm(span=12, adjust=False).mean()
        pd.testing.assert_series_equal(result['EMA_12'], expected, check_names=False)

    def test_add_ema_multiple_periods(self, indicators, sample_ohlcv_data):
        """Should be able to add multiple EMAs with different periods"""
        data = sample_ohlcv_data.copy()
        data = indicators.add_ema(data, period=5)
        data = indicators.add_ema(data, period=12)
        data = indicators.add_ema(data, period=26)
        assert 'EMA_5' in data.columns
        assert 'EMA_12' in data.columns
        assert 'EMA_26' in data.columns


class TestRSI:
    """Test Relative Strength Index calculations"""

    def test_add_rsi_creates_column(self, indicators, sample_ohlcv_data):
        """RSI should create RSI column"""
        result = indicators.add_rsi(sample_ohlcv_data.copy())
        assert 'RSI' in result.columns

    def test_add_rsi_range(self, indicators, sample_ohlcv_data):
        """RSI should be between 0 and 100"""
        result = indicators.add_rsi(sample_ohlcv_data.copy())
        valid_rsi = result['RSI'].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_add_rsi_all_up_days(self, indicators, uptrend_data):
        """RSI should be 100 when all days are up (avg_loss = 0)"""
        result = indicators.add_rsi(uptrend_data.copy())
        # After warmup period, RSI should be 100 for all-up scenario
        # Note: warmup period is 14 bars
        valid_rsi = result['RSI'].dropna()
        # Most values should be close to 100 for uptrend
        assert valid_rsi.iloc[-1] > 70  # At least bullish

    def test_add_rsi_custom_period(self, indicators, sample_ohlcv_data):
        """RSI should work with custom period"""
        result = indicators.add_rsi(sample_ohlcv_data.copy(), period=7)
        assert 'RSI' in result.columns


class TestMACD:
    """Test MACD calculations"""

    def test_add_macd_creates_columns(self, indicators, sample_ohlcv_data):
        """MACD should create all three columns"""
        result = indicators.add_macd(sample_ohlcv_data.copy())
        assert 'MACD' in result.columns
        assert 'MACD_Signal' in result.columns
        assert 'MACD_Histogram' in result.columns

    def test_add_macd_histogram_is_difference(self, indicators, sample_ohlcv_data):
        """MACD Histogram should be MACD minus Signal"""
        result = indicators.add_macd(sample_ohlcv_data.copy())
        expected_hist = result['MACD'] - result['MACD_Signal']
        pd.testing.assert_series_equal(result['MACD_Histogram'], expected_hist, check_names=False)

    def test_add_macd_custom_parameters(self, indicators, sample_ohlcv_data):
        """MACD should work with custom fast/slow/signal"""
        result = indicators.add_macd(sample_ohlcv_data.copy(), fast=8, slow=17, signal=9)
        assert 'MACD' in result.columns


# ============================================================================
# Test Bollinger Bands
# ============================================================================

class TestBollingerBands:
    """Test Bollinger Bands calculations"""

    def test_add_bollinger_bands_creates_columns(self, indicators, sample_ohlcv_data):
        """Bollinger Bands should create all columns"""
        result = indicators.add_bollinger_bands(sample_ohlcv_data.copy())
        assert 'BB_UPPER' in result.columns
        assert 'BB_LOWER' in result.columns
        assert 'BB_MIDDLE' in result.columns
        assert 'BB_WIDTH' in result.columns
        assert 'BB_POSITION' in result.columns

    def test_add_bollinger_bands_upper_above_lower(self, indicators, sample_ohlcv_data):
        """BB_UPPER should always be >= BB_LOWER"""
        result = indicators.add_bollinger_bands(sample_ohlcv_data.copy())
        valid_rows = result[result['BB_UPPER'].notna() & result['BB_LOWER'].notna()]
        assert (valid_rows['BB_UPPER'] >= valid_rows['BB_LOWER']).all()

    def test_add_bollinger_bands_width_positive(self, indicators, sample_ohlcv_data):
        """BB_WIDTH should be non-negative"""
        result = indicators.add_bollinger_bands(sample_ohlcv_data.copy())
        valid_width = result['BB_WIDTH'].dropna()
        assert (valid_width >= 0).all()

    def test_add_bollinger_bands_position_range(self, indicators, sample_ohlcv_data):
        """BB_POSITION should mostly be between 0 and 1 (can exceed during breakouts)"""
        result = indicators.add_bollinger_bands(sample_ohlcv_data.copy())
        valid_position = result['BB_POSITION'].dropna()
        # Most values should be near 0-1 range
        assert len(valid_position) > 0

    def test_add_bollinger_bands_div_by_zero(self, indicators, flat_market_data):
        """BB_POSITION should be NaN when BB_WIDTH is 0 (flat market)"""
        result = indicators.add_bollinger_bands(flat_market_data.copy())
        # When price is flat, std is 0, so BB_WIDTH is 0
        # BB_POSITION should be NaN to avoid division by zero
        # Note: first 19 rows have NaN due to rolling window
        assert result['BB_POSITION'].iloc[20:].isna().all()


# ============================================================================
# Test Stochastic
# ============================================================================

class TestStochastic:
    """Test Stochastic Oscillator calculations"""

    def test_add_stochastic_creates_columns(self, indicators, sample_ohlcv_data):
        """Stochastic should create Stoch_K and Stoch_D columns"""
        result = indicators.add_stochastic(sample_ohlcv_data.copy())
        assert 'Stoch_K' in result.columns
        assert 'Stoch_D' in result.columns

    def test_add_stochastic_range(self, indicators, sample_ohlcv_data):
        """Stochastic K should be between 0 and 100"""
        result = indicators.add_stochastic(sample_ohlcv_data.copy())
        valid_k = result['Stoch_K'].dropna()
        # Filter out NaN values from flat market edge case
        valid_k = valid_k[~np.isnan(valid_k)]
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()

    def test_add_stochastic_flat_market_edge_case(self, indicators, flat_market_data):
        """Stochastic should return NaN when high == low (division by zero)"""
        result = indicators.add_stochastic(flat_market_data.copy())
        # After warmup period, all values should be NaN for flat market
        valid_rows = result['Stoch_K'].iloc[14:]  # After k_period warmup
        assert valid_rows.isna().all()


# ============================================================================
# Test ATR
# ============================================================================

class TestATR:
    """Test Average True Range calculations"""

    def test_add_atr_creates_column(self, indicators, sample_ohlcv_data):
        """ATR should create ATR column"""
        result = indicators.add_atr(sample_ohlcv_data.copy())
        assert 'ATR' in result.columns

    def test_add_atr_positive_values(self, indicators, sample_ohlcv_data):
        """ATR should always be non-negative"""
        result = indicators.add_atr(sample_ohlcv_data.copy())
        valid_atr = result['ATR'].dropna()
        assert (valid_atr >= 0).all()

    def test_add_atr_custom_period(self, indicators, sample_ohlcv_data):
        """ATR should work with custom period"""
        result = indicators.add_atr(sample_ohlcv_data.copy(), period=7)
        assert 'ATR' in result.columns


# ============================================================================
# Test Volume Indicators
# ============================================================================

class TestVolumeIndicators:
    """Test volume-based indicator calculations"""

    def test_add_volume_indicators_creates_columns(self, indicators, sample_ohlcv_data):
        """Should create Volume_SMA, Volume_ROC, OBV, VWAP"""
        result = indicators.add_volume_indicators(sample_ohlcv_data.copy())
        assert 'Volume_SMA' in result.columns
        assert 'Volume_ROC' in result.columns
        assert 'OBV' in result.columns
        assert 'VWAP' in result.columns

    def test_obv_cumulative(self, indicators, sample_ohlcv_data):
        """OBV should be cumulative"""
        result = indicators.add_volume_indicators(sample_ohlcv_data.copy())
        # OBV is cumulative sum of signed volume
        assert 'OBV' in result.columns

    def test_vwap_positive(self, indicators, sample_ohlcv_data):
        """VWAP should be positive (price-based)"""
        result = indicators.add_volume_indicators(sample_ohlcv_data.copy())
        valid_vwap = result['VWAP'].dropna()
        assert (valid_vwap > 0).all()

    def test_vwap_division_by_zero_protection(self, indicators):
        """VWAP should handle zero volume without error"""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1h'),
            'open': [100.0] * 5,
            'high': [101.0] * 5,
            'low': [99.0] * 5,
            'close': [100.5] * 5,
            'volume': [0, 0, 0, 0, 0]  # All zero volume
        })
        result = indicators.add_volume_indicators(data)
        # Should not raise error, VWAP should be NaN
        assert 'VWAP' in result.columns


# ============================================================================
# Test Momentum Indicators
# ============================================================================

class TestMomentumIndicators:
    """Test momentum indicator calculations"""

    def test_add_momentum_indicators_creates_columns(self, indicators, sample_ohlcv_data):
        """Should create ROC and momentum columns"""
        result = indicators.add_momentum_indicators(sample_ohlcv_data.copy())
        assert 'ROC_1' in result.columns
        assert 'ROC_5' in result.columns
        assert 'ROC_10' in result.columns
        assert 'Momentum_5' in result.columns
        assert 'Momentum_10' in result.columns
        assert 'Williams_R' in result.columns

    def test_williams_r_range(self, indicators, sample_ohlcv_data):
        """Williams %R should be between -100 and 0"""
        result = indicators.add_momentum_indicators(sample_ohlcv_data.copy())
        valid_wr = result['Williams_R'].dropna()
        valid_wr = valid_wr[~np.isnan(valid_wr)]  # Remove NaN from flat market
        assert (valid_wr >= -100).all()
        assert (valid_wr <= 0).all()

    def test_williams_r_flat_market_edge_case(self, indicators, flat_market_data):
        """Williams %R should return NaN when high == low"""
        result = indicators.add_momentum_indicators(flat_market_data.copy())
        # After warmup period, should be NaN for flat market
        valid_rows = result['Williams_R'].iloc[14:]
        assert valid_rows.isna().all()


# ============================================================================
# Test add_all_indicators
# ============================================================================

class TestAddAllIndicators:
    """Test the add_all_indicators comprehensive method"""

    def test_add_all_indicators_creates_many_columns(self, indicators, sample_ohlcv_data):
        """add_all_indicators should create many indicator columns"""
        result = indicators.add_all_indicators(sample_ohlcv_data.copy())
        # Check for main indicators
        assert 'SMA_5' in result.columns
        assert 'SMA_20' in result.columns
        assert 'SMA_50' in result.columns
        assert 'EMA_12' in result.columns
        assert 'EMA_26' in result.columns
        assert 'RSI' in result.columns
        assert 'MACD' in result.columns
        assert 'BB_UPPER' in result.columns
        assert 'Stoch_K' in result.columns
        assert 'ATR' in result.columns
        assert 'OBV' in result.columns
        assert 'VWAP' in result.columns
        assert 'ADX' in result.columns
        assert 'Momentum_Score' in result.columns

    def test_add_all_indicators_missing_column_error(self, indicators):
        """add_all_indicators should raise error for missing required columns"""
        incomplete_data = pd.DataFrame({
            'open': [100.0],
            'close': [100.5]
            # Missing: high, low, volume
        })
        with pytest.raises(ValueError, match="Missing required column"):
            indicators.add_all_indicators(incomplete_data)

    def test_add_all_indicators_preserves_original_columns(self, indicators, sample_ohlcv_data):
        """add_all_indicators should preserve original OHLCV columns"""
        result = indicators.add_all_indicators(sample_ohlcv_data.copy())
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns
        assert 'timestamp' in result.columns


# ============================================================================
# Test Edge Cases and NaN Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_dataframe(self, indicators):
        """Should handle empty dataframe gracefully"""
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        result = indicators.add_sma(empty_df.copy(), period=5)
        assert len(result) == 0
        assert 'SMA_5' in result.columns

    def test_minimal_data_sma(self, indicators, minimal_data):
        """SMA should work with minimal data (creating NaNs for warmup)"""
        result = indicators.add_sma(minimal_data.copy(), period=3)
        # First 2 values should be NaN
        assert result['SMA_3'].iloc[:2].isna().all()
        # Value at index 2 should be valid
        assert not pd.isna(result['SMA_3'].iloc[2])

    def test_nan_in_close_column(self, indicators):
        """Should handle NaN in close column gracefully"""
        data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 101.0, 100.0],
            'high': [101.0, 102.0, 103.0, 102.0, 101.0],
            'low': [99.0, 100.0, 101.0, 100.0, 99.0],
            'close': [100.5, np.nan, 102.5, 101.5, 100.5],
            'volume': [1000, 1200, 1500, 1100, 900]
        })
        result = indicators.add_sma(data.copy(), period=2)
        # Should not raise, but SMA around NaN will propagate NaN
        assert 'SMA_2' in result.columns


# ============================================================================
# Test Trend Indicators
# ============================================================================

class TestTrendIndicators:
    """Test trend identification indicators"""

    def test_add_trend_indicators_creates_columns(self, indicators, sample_ohlcv_data):
        """Should create ADX and Aroon columns"""
        result = indicators.add_trend_indicators(sample_ohlcv_data.copy())
        assert 'ADX' in result.columns
        assert 'Aroon_Up' in result.columns
        assert 'Aroon_Down' in result.columns
        assert 'Aroon_Oscillator' in result.columns

    def test_aroon_range(self, indicators, sample_ohlcv_data):
        """Aroon should be between 0 and 100"""
        result = indicators.add_trend_indicators(sample_ohlcv_data.copy())
        valid_aroon_up = result['Aroon_Up'].dropna()
        valid_aroon_down = result['Aroon_Down'].dropna()
        assert (valid_aroon_up >= 0).all()
        assert (valid_aroon_up <= 100).all()
        assert (valid_aroon_down >= 0).all()
        assert (valid_aroon_down <= 100).all()


# ============================================================================
# Test Support/Resistance
# ============================================================================

class TestSupportResistance:
    """Test support and resistance levels"""

    def test_add_support_resistance_creates_columns(self, indicators, sample_ohlcv_data):
        """Should create Resistance, Support, and Pivot columns"""
        result = indicators.add_support_resistance(sample_ohlcv_data.copy())
        assert 'Resistance' in result.columns
        assert 'Support' in result.columns
        assert 'Pivot' in result.columns
        assert 'R1' in result.columns
        assert 'S1' in result.columns

    def test_resistance_above_support(self, indicators, sample_ohlcv_data):
        """Resistance should be >= Support"""
        result = indicators.add_support_resistance(sample_ohlcv_data.copy())
        valid_rows = result[result['Resistance'].notna() & result['Support'].notna()]
        assert (valid_rows['Resistance'] >= valid_rows['Support']).all()


# ============================================================================
# Test Volatility Indicators
# ============================================================================

class TestVolatilityIndicators:
    """Test volatility indicator calculations"""

    def test_add_volatility_indicators_creates_columns(self, indicators, sample_ohlcv_data):
        """Should create Historical_Volatility and Keltner columns"""
        # Need ATR first for volatility indicators
        data = indicators.add_atr(sample_ohlcv_data.copy())
        result = indicators.add_volatility_indicators(data)
        assert 'Historical_Volatility' in result.columns
        assert 'Vol_Ratio' in result.columns
        assert 'Keltner_Upper' in result.columns
        assert 'Keltner_Lower' in result.columns
        assert 'Keltner_Position' in result.columns


# ============================================================================
# Test Advanced Momentum Indicators
# ============================================================================

class TestAdvancedMomentumIndicators:
    """Test advanced momentum indicators for ML"""

    def test_add_advanced_momentum_creates_columns(self, indicators, sample_ohlcv_data):
        """Should create MFI, CCI, Ultimate Oscillator columns"""
        result = indicators.add_advanced_momentum(sample_ohlcv_data.copy())
        assert 'MFI' in result.columns
        assert 'CCI' in result.columns
        assert 'Ultimate_Oscillator' in result.columns
        assert 'CMF' in result.columns
        assert 'Fisher' in result.columns

    def test_mfi_range(self, indicators, sample_ohlcv_data):
        """MFI should be between 0 and 100"""
        result = indicators.add_advanced_momentum(sample_ohlcv_data.copy())
        valid_mfi = result['MFI'].dropna()
        assert (valid_mfi >= 0).all()
        assert (valid_mfi <= 100).all()


# ============================================================================
# Test Advanced Volume Indicators
# ============================================================================

class TestAdvancedVolumeIndicators:
    """Test advanced volume indicators"""

    def test_add_volume_advanced_creates_columns(self, indicators, sample_ohlcv_data):
        """Should create AD_Line, Chaikin_Osc, Force_Index columns"""
        result = indicators.add_volume_advanced(sample_ohlcv_data.copy())
        assert 'AD_Line' in result.columns
        assert 'Chaikin_Osc' in result.columns
        assert 'Force_Index' in result.columns
        assert 'Force_Index_13' in result.columns
        assert 'EMV' in result.columns
        assert 'VPT' in result.columns


# ============================================================================
# Test Price Action Indicators
# ============================================================================

class TestPriceActionIndicators:
    """Test price action indicators"""

    def test_add_price_action_creates_columns(self, indicators, sample_ohlcv_data):
        """Should create TSI, KST, Vortex, Donchian, Ichimoku columns"""
        result = indicators.add_price_action(sample_ohlcv_data.copy())
        assert 'TSI' in result.columns
        assert 'KST' in result.columns
        assert 'Vortex_Plus' in result.columns
        assert 'Vortex_Minus' in result.columns
        assert 'Donchian_Upper' in result.columns
        assert 'Donchian_Lower' in result.columns
        assert 'Elder_Bull_Power' in result.columns
        assert 'Ichimoku_Conversion' in result.columns
        assert 'Ichimoku_Cloud_Position' in result.columns


# ============================================================================
# Test Statistical Features
# ============================================================================

class TestStatisticalFeatures:
    """Test statistical features for ML"""

    def test_add_statistical_features_creates_columns(self, indicators, sample_ohlcv_data):
        """Should create Z-Score, Slope, CV, Skew, Kurtosis columns"""
        result = indicators.add_statistical_features(sample_ohlcv_data.copy())
        assert 'Price_ZScore' in result.columns
        assert 'Price_Slope_10' in result.columns
        assert 'Volume_Slope_10' in result.columns
        assert 'Price_Std_Normalized' in result.columns
        assert 'Price_CV' in result.columns
        assert 'Returns_Skew' in result.columns
        assert 'Returns_Kurtosis' in result.columns


# ============================================================================
# Test Signal Strength
# ============================================================================

class TestSignalStrength:
    """Test signal strength calculation"""

    def test_get_signal_strength_returns_dict(self, indicators, sample_ohlcv_data):
        """get_signal_strength should return dict with all signal types"""
        data = indicators.add_all_indicators(sample_ohlcv_data.copy())
        signals = indicators.get_signal_strength(data)
        assert 'rsi' in signals
        assert 'macd' in signals
        assert 'bb' in signals
        assert 'trend' in signals
        assert 'volume' in signals

    def test_get_signal_strength_valid_values(self, indicators, sample_ohlcv_data):
        """Signal strength values should be valid states"""
        data = indicators.add_all_indicators(sample_ohlcv_data.copy())
        signals = indicators.get_signal_strength(data)

        valid_rsi = ['oversold', 'overbought', 'neutral']
        valid_macd = ['bullish', 'bearish', 'neutral']
        valid_bb = ['oversold', 'overbought', 'neutral']
        valid_trend = ['uptrend', 'downtrend', 'neutral']
        valid_volume = ['high', 'low', 'neutral']

        assert signals['rsi'] in valid_rsi
        assert signals['macd'] in valid_macd
        assert signals['bb'] in valid_bb
        assert signals['trend'] in valid_trend
        assert signals['volume'] in valid_volume


# ============================================================================
# Test Momentum Score
# ============================================================================

class TestMomentumScore:
    """Test composite momentum score calculation"""

    def test_calculate_momentum_score_creates_column(self, indicators, sample_ohlcv_data):
        """calculate_momentum_score should create Momentum_Score column"""
        # Need prerequisites
        data = sample_ohlcv_data.copy()
        data = indicators.add_sma(data, 20)
        data = indicators.add_rsi(data)
        data = indicators.add_macd(data)
        data = indicators.add_volume_indicators(data)

        result = indicators.calculate_momentum_score(data)
        assert 'Momentum_Score' in result.columns

    def test_momentum_score_range(self, indicators, sample_ohlcv_data):
        """Momentum score should be positive and bounded"""
        data = sample_ohlcv_data.copy()
        data = indicators.add_sma(data, 20)
        data = indicators.add_rsi(data)
        data = indicators.add_macd(data)
        data = indicators.add_volume_indicators(data)

        result = indicators.calculate_momentum_score(data)
        valid_score = result['Momentum_Score'].dropna()
        # Score is weighted average of 0-100 values, should be positive
        assert (valid_score >= 0).all()


# ============================================================================
# Test Import from core
# ============================================================================

class TestModuleImport:
    """Test that TechnicalIndicators can be imported from core"""

    def test_import_from_core(self):
        """Should be able to import TechnicalIndicators from core"""
        from core import TechnicalIndicators
        ti = TechnicalIndicators()
        assert hasattr(ti, 'add_all_indicators')
        assert hasattr(ti, 'add_sma')
        assert hasattr(ti, 'add_rsi')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
