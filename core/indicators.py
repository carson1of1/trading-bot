import pandas as pd
import numpy as np
# import pandas_ta as ta  # Temporarily disabled - Windows DLL issue

class TechnicalIndicators:
    """Calculate various technical indicators for trading analysis"""

    def __init__(self):
        self.default_periods = {
            'sma_fast': 5,
            'sma_slow': 20,
            'rsi': 14,
            'ema_fast': 12,
            'ema_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2
        }

    def add_sma(self, data, period=20, column='close'):
        """Add Simple Moving Average"""
        data[f'SMA_{period}'] = data[column].rolling(window=period).mean()
        return data

    def add_ema(self, data, period=12, column='close'):
        """Add Exponential Moving Average"""
        # BUG FIX (Dec 4, 2025): Added adjust=False for proper EMA calculation
        # Without adjust=False, pandas uses a weighted average that doesn't match
        # the standard EMA formula: EMA = Price * k + EMA(prev) * (1-k) where k = 2/(period+1)
        data[f'EMA_{period}'] = data[column].ewm(span=period, adjust=False).mean()
        return data

    def add_rsi(self, data, period=14, column='close'):
        """Add Relative Strength Index"""
        delta = data[column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # BUG FIX (Dec 9, 2025): Prevent division by zero when avg_loss = 0 (all up days)
        # RSI should be 100 when avg_loss=0 (all gains), not NaN
        # When avg_loss=0, RS=infinity, so RSI = 100 - (100 / (1 + inf)) = 100 - 0 = 100
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
        data['RSI'] = 100 - (100 / (1 + rs))
        return data

    def add_macd(self, data, fast=12, slow=26, signal=9, column='close'):
        """Add MACD (Moving Average Convergence Divergence)

        BUG FIX (Dec 5, 2025): Added adjust=False for consistency with add_ema().
        This ensures MACD uses the same EMA formula as standalone EMA calculations.
        """
        ema_fast = data[column].ewm(span=fast, adjust=False).mean()
        ema_slow = data[column].ewm(span=slow, adjust=False).mean()

        data['MACD'] = ema_fast - ema_slow
        data['MACD_Signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        return data

    def add_bollinger_bands(self, data, period=20, std_dev=2, column='close'):
        """Add Bollinger Bands"""
        rolling_mean = data[column].rolling(window=period).mean()
        rolling_std = data[column].rolling(window=period).std()

        data['BB_UPPER'] = rolling_mean + (rolling_std * std_dev)
        data['BB_LOWER'] = rolling_mean - (rolling_std * std_dev)
        data['BB_MIDDLE'] = rolling_mean
        data['BB_WIDTH'] = data['BB_UPPER'] - data['BB_LOWER']

        # BUG FIX (Dec 9, 2025): Prevent division by zero when BB_WIDTH = 0 (low volatility)
        # When bands collapse (zero width), position cannot be calculated - return NaN
        bb_width = data['BB_UPPER'] - data['BB_LOWER']
        data['BB_POSITION'] = np.where(bb_width != 0, (data[column] - data['BB_LOWER']) / bb_width, np.nan)
        return data

    def add_stochastic(self, data, k_period=14, d_period=3):
        """Add Stochastic Oscillator"""
        lowest_low = data['low'].rolling(window=k_period).min()
        highest_high = data['high'].rolling(window=k_period).max()

        # BUG FIX (Dec 9, 2025): Prevent division by zero when highest_high == lowest_low (flat market)
        # When price doesn't move in the period, range is zero - return NaN for stochastic
        denom = highest_high - lowest_low
        data['Stoch_K'] = np.where(denom != 0, 100 * (data['close'] - lowest_low) / denom, np.nan)
        data['Stoch_D'] = data['Stoch_K'].rolling(window=d_period).mean()
        return data

    def add_atr(self, data, period=14):
        """Add Average True Range

        BUG FIX (Dec 2025): ATR calculation now properly handles first bar.
        True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        On first bar, shift() returns NaN, so we use fillna(0) to default to high-low only.
        """
        high_low = data['high'] - data['low']
        high_close_prev = np.abs(data['high'] - data['close'].shift(1))
        low_close_prev = np.abs(data['low'] - data['close'].shift(1))

        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        data['ATR'] = true_range.rolling(window=period).mean()
        return data

    def add_volume_indicators(self, data):
        """Add volume-based indicators"""
        # Volume Moving Average
        data['Volume_SMA'] = data['volume'].rolling(window=20).mean()

        # Volume Rate of Change
        # BUG FIX (Dec 3, 2025): Add fill_method=None to prevent FutureWarning in pandas
        data['Volume_ROC'] = data['volume'].pct_change(periods=1, fill_method=None)

        # On-Balance Volume (OBV)
        # BUG FIX (Dec 3, 2025): Use vectorized operation instead of chained assignment
        # The old loop caused pandas FutureWarning and will break in pandas 3.0
        # BUG FIX (Dec 9, 2025): Optimized OBV calculation - use np.where for volume signs
        # Old: obv_direction * volume then cumsum | New: directly compute signed volume then cumsum
        # BUG FIX (Dec 10, 2025): OBV should be 0 when close equals previous close, not -volume
        # OBV formula: +volume if close > prev_close, -volume if close < prev_close, 0 if equal
        obv = np.where(data['close'] > data['close'].shift(), data['volume'],
                       np.where(data['close'] < data['close'].shift(), -data['volume'], 0))
        # BUG FIX (Dec 9, 2025): Convert numpy array to Series for fillna() - numpy arrays don't have fillna
        data['OBV'] = pd.Series(obv, index=data.index).fillna(0).cumsum()

        # Volume Weighted Average Price (VWAP)
        # BUG FIX #5 (Nov 18, 2025): Prevent division by zero when cumulative volume = 0 (first row)
        # BUG FIX (Dec 2025): VWAP must reset daily for intraday trading
        # VWAP = Sum(Price * Volume) / Sum(Volume) where sum is from market open to current bar
        # BUG FIX (Dec 9, 2025): Enhanced division by zero protection - replace 0 with np.nan BEFORE division
        # This prevents division by zero warnings and handles first bar correctly

        # Check if data has date index to detect day changes
        if isinstance(data.index, pd.DatetimeIndex):
            # Group by date to reset VWAP daily
            data['trading_date'] = data.index.date
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            data['cum_tp_volume'] = (typical_price * data['volume']).groupby(data['trading_date']).cumsum()
            data['cum_volume'] = data['volume'].groupby(data['trading_date']).cumsum()
            data['cum_volume'] = data['cum_volume'].replace(0, np.nan)  # BUG FIX (Dec 9, 2025): Prevent division by zero
            data['VWAP'] = data['cum_tp_volume'] / data['cum_volume']
            # Clean up temporary columns
            data.drop(columns=['trading_date', 'cum_tp_volume', 'cum_volume'], inplace=True)
        else:
            # Fallback: no date reset (original behavior for non-datetime index)
            cum_volume = data['volume'].cumsum()
            cum_volume = cum_volume.replace(0, np.nan)  # BUG FIX (Dec 9, 2025): Prevent division by zero
            data['VWAP'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / cum_volume

        return data

    def add_momentum_indicators(self, data):
        """Add momentum-based indicators"""
        # Price Rate of Change
        # BUG FIX (Dec 3, 2025): Add fill_method=None to prevent FutureWarning in pandas
        # BUG FIX (Dec 3, 2025): Replace inf values with NaN (occurs when previous price is 0)
        data['ROC_1'] = data['close'].pct_change(periods=1, fill_method=None)
        data['ROC_5'] = data['close'].pct_change(periods=5, fill_method=None)
        data['ROC_10'] = data['close'].pct_change(periods=10, fill_method=None)

        # Replace inf values with NaN to prevent downstream errors
        for col in ['ROC_1', 'ROC_5', 'ROC_10']:
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)

        # Momentum
        data['Momentum_5'] = data['close'] - data['close'].shift(5)
        data['Momentum_10'] = data['close'] - data['close'].shift(10)

        # Williams %R
        # BUG FIX (Dec 9, 2025): Prevent division by zero when highest_high == lowest_low (flat market)
        # When price range is zero over the period, Williams %R is undefined - return NaN
        highest_high = data['high'].rolling(window=14).max()
        lowest_low = data['low'].rolling(window=14).min()
        denom = highest_high - lowest_low
        data['Williams_R'] = np.where(denom != 0, -100 * (highest_high - data['close']) / denom, np.nan)

        return data

    def add_trend_indicators(self, data):
        """Add trend identification indicators"""
        # Average Directional Index (ADX)
        # BUG FIX (Dec 10, 2025): Fixed ADX directional movement calculation
        # +DM = current_high - previous_high (if positive and > -DM)
        # -DM = previous_low - current_low (if positive and > +DM)
        high_diff = data['high'].diff()  # Current high - previous high
        low_diff_raw = data['low'].shift(1) - data['low']  # Previous low - current low (positive = down move)

        # +DM: positive high movement that exceeds any downward low movement
        plus_dm = high_diff.where((high_diff > low_diff_raw) & (high_diff > 0), 0)
        # -DM: positive low movement (price moving down) that exceeds any upward high movement
        minus_dm = low_diff_raw.where((low_diff_raw > high_diff) & (low_diff_raw > 0), 0)

        # True Range
        tr1 = data['high'] - data['low']
        tr2 = (data['high'] - data['close'].shift()).abs()
        tr3 = (data['low'] - data['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Indicators
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())

        # BUG FIX #5 (Nov 18, 2025): Prevent division by zero when plus_di + minus_di = 0
        # BUG FIX #8 (Nov 18, 2025): Convert np.where result to Series before calling .rolling()
        di_sum = plus_di + minus_di
        dx = np.where(di_sum != 0, 100 * (plus_di - minus_di).abs() / di_sum, np.nan)
        dx_series = pd.Series(dx, index=data.index)  # Convert numpy array to Series
        data['ADX'] = dx_series.rolling(window=14).mean()

        # Aroon Indicator
        # BUG FIX (Dec 4, 2025): Fixed Aroon calculation to find MOST RECENT high/low, not first
        # np.argmax() returns the first occurrence when there are ties, but Aroon should
        # measure periods since the MOST RECENT high/low. We reverse the array to get last occurrence.
        aroon_period = 25

        def periods_since_high(x):
            """Find periods since most recent high (reverse to get last occurrence)"""
            reversed_idx = len(x) - 1 - np.argmax(x[::-1])
            return len(x) - 1 - reversed_idx

        def periods_since_low(x):
            """Find periods since most recent low (reverse to get last occurrence)"""
            reversed_idx = len(x) - 1 - np.argmin(x[::-1])
            return len(x) - 1 - reversed_idx

        data['Aroon_Up'] = 100 * (aroon_period - data['high'].rolling(aroon_period).apply(periods_since_high, raw=True)) / aroon_period
        data['Aroon_Down'] = 100 * (aroon_period - data['low'].rolling(aroon_period).apply(periods_since_low, raw=True)) / aroon_period
        data['Aroon_Oscillator'] = data['Aroon_Up'] - data['Aroon_Down']

        return data

    def add_support_resistance(self, data, window=20):
        """Add dynamic support and resistance levels"""
        # Rolling max/min as dynamic resistance/support
        data['Resistance'] = data['high'].rolling(window=window).max()
        data['Support'] = data['low'].rolling(window=window).min()

        # Pivot points (simplified)
        data['Pivot'] = (data['high'].shift() + data['low'].shift() + data['close'].shift()) / 3
        data['R1'] = 2 * data['Pivot'] - data['low'].shift()
        data['S1'] = 2 * data['Pivot'] - data['high'].shift()

        return data

    def add_volatility_indicators(self, data):
        """Add volatility measurements"""
        # Historical Volatility (20-day)
        # BUG FIX (Dec 3, 2025): Add fill_method=None to prevent FutureWarning in pandas
        returns = data['close'].pct_change(fill_method=None)
        data['Historical_Volatility'] = returns.rolling(window=20).std() * np.sqrt(252) * 100

        # Volatility Ratio
        data['Vol_Ratio'] = data['Historical_Volatility'] / data['Historical_Volatility'].rolling(window=50).mean()

        # Keltner Channels
        # BUG FIX (Dec 5, 2025): Added adjust=False for EMA consistency
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        keltner_ema = typical_price.ewm(span=20, adjust=False).mean()
        keltner_atr = data['ATR'] if 'ATR' in data.columns else self.add_atr(data.copy(), 14)['ATR']
        data['Keltner_Upper'] = keltner_ema + (2 * keltner_atr)
        data['Keltner_Lower'] = keltner_ema - (2 * keltner_atr)
        data['Keltner_Middle'] = keltner_ema
        keltner_width = data['Keltner_Upper'] - data['Keltner_Lower']
        data['Keltner_Position'] = np.where(keltner_width != 0,
                                            (data['close'] - data['Keltner_Lower']) / keltner_width,
                                            np.nan)

        return data

    def add_advanced_momentum(self, data):
        """Add advanced momentum indicators for ML"""
        # Money Flow Index (MFI) - Volume-weighted RSI
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']

        # Positive and negative money flow
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)

        positive_mf = pd.Series(positive_flow, index=data.index).rolling(window=14).sum()
        negative_mf = pd.Series(negative_flow, index=data.index).rolling(window=14).sum()

        # BUG FIX (Dec 10, 2025): MFI should handle edge cases and stay in 0-100 range
        # When negative_mf=0 (all buying), MFI should be 100 (use np.inf for ratio)
        mfi_ratio = np.where(negative_mf != 0, positive_mf / negative_mf, np.inf)
        data['MFI'] = 100 - (100 / (1 + mfi_ratio))
        # Ensure MFI stays in valid 0-100 range (like RSI)
        data['MFI'] = data['MFI'].clip(0, 100)

        # Commodity Channel Index (CCI)
        tp_sma = typical_price.rolling(window=20).mean()
        mean_deviation = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        data['CCI'] = np.where(mean_deviation != 0, (typical_price - tp_sma) / (0.015 * mean_deviation), np.nan)

        # Ultimate Oscillator (combines 3 timeframes)
        bp = data['close'] - pd.concat([data['low'], data['close'].shift()], axis=1).min(axis=1)
        tr = pd.concat([data['high'] - data['low'],
                       (data['high'] - data['close'].shift()).abs(),
                       (data['low'] - data['close'].shift()).abs()], axis=1).max(axis=1)

        # BUG FIX (Dec 10, 2025): Prevent division by zero when true range sum is 0
        tr_sum7 = tr.rolling(7).sum().replace(0, np.nan)
        tr_sum14 = tr.rolling(14).sum().replace(0, np.nan)
        tr_sum28 = tr.rolling(28).sum().replace(0, np.nan)
        avg7 = bp.rolling(7).sum() / tr_sum7
        avg14 = bp.rolling(14).sum() / tr_sum14
        avg28 = bp.rolling(28).sum() / tr_sum28

        data['Ultimate_Oscillator'] = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7

        # Chaikin Money Flow
        mfv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        mfv = np.where(data['high'] != data['low'], mfv, 0)
        mfv_volume = mfv * data['volume']
        data['CMF'] = mfv_volume.rolling(window=20).sum() / data['volume'].rolling(window=20).sum()

        # Fisher Transform (converts prices to Gaussian distribution)
        hl2 = (data['high'] + data['low']) / 2
        hl2_min = hl2.rolling(window=10).min()
        hl2_max = hl2.rolling(window=10).max()
        hl2_range = hl2_max - hl2_min
        value = np.where(hl2_range != 0, 2 * ((hl2 - hl2_min) / hl2_range - 0.5), 0)
        value = np.clip(value, -0.999, 0.999)  # Prevent log(0)
        data['Fisher'] = 0.5 * np.log((1 + value) / (1 - value))
        data['Fisher_Signal'] = data['Fisher'].shift(1)

        return data

    def add_volume_advanced(self, data):
        """Add advanced volume indicators"""
        # Accumulation/Distribution Line
        mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        mfm = np.where(data['high'] != data['low'], mfm, 0)
        mfv = mfm * data['volume']
        data['AD_Line'] = mfv.cumsum()

        # Chaikin Oscillator (AD Line momentum)
        # BUG FIX (Dec 5, 2025): Added adjust=False for EMA consistency
        ad_ema3 = data['AD_Line'].ewm(span=3, adjust=False).mean()
        ad_ema10 = data['AD_Line'].ewm(span=10, adjust=False).mean()
        data['Chaikin_Osc'] = ad_ema3 - ad_ema10

        # Force Index
        data['Force_Index'] = (data['close'] - data['close'].shift(1)) * data['volume']
        data['Force_Index_13'] = data['Force_Index'].ewm(span=13, adjust=False).mean()

        # Ease of Movement
        # BUG FIX (Dec 10, 2025): Prevent division by zero when high-low range is 0
        distance = ((data['high'] + data['low']) / 2) - ((data['high'].shift(1) + data['low'].shift(1)) / 2)
        hl_range = (data['high'] - data['low']).replace(0, np.nan)
        box_ratio = (data['volume'] / 100000000) / hl_range
        data['EMV'] = np.where(pd.notna(box_ratio) & (box_ratio != 0), distance / box_ratio, 0)
        data['EMV_14'] = data['EMV'].rolling(window=14).mean()

        # Volume Price Trend
        # BUG FIX (Dec 3, 2025): Handle division by zero when previous close is 0
        prev_close = data['close'].shift(1)
        price_change_ratio = (data['close'] - prev_close) / prev_close
        price_change_ratio = price_change_ratio.replace([np.inf, -np.inf], np.nan)
        data['VPT'] = (data['volume'] * price_change_ratio).cumsum()

        return data

    def add_price_action(self, data):
        """Add price action indicators"""
        # True Strength Index (TSI)
        # BUG FIX (Dec 5, 2025): Added adjust=False for EMA consistency
        price_change = data['close'].diff()
        double_smoothed_pc = price_change.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
        double_smoothed_abs_pc = price_change.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
        data['TSI'] = np.where(double_smoothed_abs_pc != 0,
                               100 * (double_smoothed_pc / double_smoothed_abs_pc),
                               0)

        # Know Sure Thing (KST)
        # BUG FIX (Dec 3, 2025): Add fill_method=None to prevent FutureWarning in pandas
        rcma1 = data['close'].pct_change(10, fill_method=None).rolling(10).mean()
        rcma2 = data['close'].pct_change(15, fill_method=None).rolling(10).mean()
        rcma3 = data['close'].pct_change(20, fill_method=None).rolling(10).mean()
        rcma4 = data['close'].pct_change(30, fill_method=None).rolling(15).mean()
        data['KST'] = (rcma1 * 1) + (rcma2 * 2) + (rcma3 * 3) + (rcma4 * 4)
        data['KST_Signal'] = data['KST'].rolling(window=9).mean()

        # Vortex Indicator
        vm_plus = (data['high'] - data['low'].shift(1)).abs()
        vm_minus = (data['low'] - data['high'].shift(1)).abs()
        tr = pd.concat([data['high'] - data['low'],
                       (data['high'] - data['close'].shift(1)).abs(),
                       (data['low'] - data['close'].shift(1)).abs()], axis=1).max(axis=1)

        vi_plus = vm_plus.rolling(14).sum() / tr.rolling(14).sum()
        vi_minus = vm_minus.rolling(14).sum() / tr.rolling(14).sum()
        data['Vortex_Plus'] = vi_plus
        data['Vortex_Minus'] = vi_minus
        data['Vortex_Diff'] = vi_plus - vi_minus

        # Donchian Channels
        data['Donchian_Upper'] = data['high'].rolling(window=20).max()
        data['Donchian_Lower'] = data['low'].rolling(window=20).min()
        data['Donchian_Middle'] = (data['Donchian_Upper'] + data['Donchian_Lower']) / 2
        donchian_width = data['Donchian_Upper'] - data['Donchian_Lower']
        data['Donchian_Position'] = np.where(donchian_width != 0,
                                             (data['close'] - data['Donchian_Lower']) / donchian_width,
                                             np.nan)

        # Elder Ray Index
        # BUG FIX (Dec 5, 2025): Added adjust=False for EMA consistency
        ema13 = data['close'].ewm(span=13, adjust=False).mean()
        data['Elder_Bull_Power'] = data['high'] - ema13
        data['Elder_Bear_Power'] = data['low'] - ema13

        # Ichimoku Cloud components
        nine_period_high = data['high'].rolling(window=9).max()
        nine_period_low = data['low'].rolling(window=9).min()
        data['Ichimoku_Conversion'] = (nine_period_high + nine_period_low) / 2

        twenty_six_period_high = data['high'].rolling(window=26).max()
        twenty_six_period_low = data['low'].rolling(window=26).min()
        data['Ichimoku_Base'] = (twenty_six_period_high + twenty_six_period_low) / 2

        data['Ichimoku_LeadingA'] = ((data['Ichimoku_Conversion'] + data['Ichimoku_Base']) / 2).shift(26)

        fifty_two_period_high = data['high'].rolling(window=52).max()
        fifty_two_period_low = data['low'].rolling(window=52).min()
        data['Ichimoku_LeadingB'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)

        data['Ichimoku_Lagging'] = data['close'].shift(-26)

        # Ichimoku cloud position
        cloud_top = pd.concat([data['Ichimoku_LeadingA'], data['Ichimoku_LeadingB']], axis=1).max(axis=1)
        cloud_bottom = pd.concat([data['Ichimoku_LeadingA'], data['Ichimoku_LeadingB']], axis=1).min(axis=1)
        data['Ichimoku_Cloud_Position'] = np.where(
            data['close'] > cloud_top, 1,  # Above cloud
            np.where(data['close'] < cloud_bottom, -1, 0)  # Below cloud or inside
        )

        return data

    def add_statistical_features(self, data):
        """Add statistical features for ML"""
        # Z-Score of price
        price_mean = data['close'].rolling(window=20).mean()
        price_std = data['close'].rolling(window=20).std()
        data['Price_ZScore'] = np.where(price_std != 0, (data['close'] - price_mean) / price_std, 0)

        # Linear Regression Slope
        def calc_slope(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            y = series.values
            if len(x) == 0 or len(y) == 0:
                return 0
            slope = np.polyfit(x, y, 1)[0]
            return slope

        data['Price_Slope_10'] = data['close'].rolling(window=10).apply(calc_slope, raw=False)
        data['Volume_Slope_10'] = data['volume'].rolling(window=10).apply(calc_slope, raw=False)

        # Standard Deviation (normalized)
        data['Price_Std_Normalized'] = data['close'].rolling(window=20).std() / data['close']

        # Coefficient of Variation
        data['Price_CV'] = np.where(price_mean != 0, price_std / price_mean, 0)

        # Skewness (3rd moment)
        # BUG FIX (Dec 3, 2025): Add fill_method=None to prevent FutureWarning in pandas
        data['Returns_Skew'] = data['close'].pct_change(fill_method=None).rolling(window=20).skew()

        # Kurtosis (4th moment - tail risk)
        # BUG FIX (Dec 3, 2025): Add fill_method=None to prevent FutureWarning in pandas
        data['Returns_Kurtosis'] = data['close'].pct_change(fill_method=None).rolling(window=20).kurt()

        return data

    def calculate_momentum_score(self, data):
        """Calculate a composite momentum score"""
        # Normalize indicators to 0-100 scale
        rsi_score = data['RSI']

        # MACD momentum (1 if positive, 0 if negative)
        macd_score = np.where(data['MACD'] > data['MACD_Signal'], 75, 25)

        # Price vs moving averages
        sma_score = np.where(data['close'] > data['SMA_20'], 75, 25)

        # Volume momentum
        vol_score = np.where(data['volume'] > data['Volume_SMA'], 75, 25)

        # Composite score
        data['Momentum_Score'] = (rsi_score * 0.3 + macd_score * 0.25 + sma_score * 0.25 + vol_score * 0.2)

        return data

    def add_all_indicators(self, data):
        """Add all technical indicators to the dataframe"""
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add all indicators
        data = self.add_sma(data, 5)
        data = self.add_sma(data, 20)
        data = self.add_sma(data, 50)

        data = self.add_ema(data, 5)   # BUG FIX (Dec 2, 2025): For MA Crossover strategy
        data = self.add_ema(data, 9)   # For Trend Continuation strategy
        data = self.add_ema(data, 12)
        data = self.add_ema(data, 13)  # BUG FIX (Dec 2, 2025): For MA Crossover strategy
        data = self.add_ema(data, 21)  # For Trend Continuation strategy
        data = self.add_ema(data, 26)

        data = self.add_rsi(data)
        data = self.add_macd(data)
        data = self.add_bollinger_bands(data)
        data = self.add_stochastic(data)
        data = self.add_atr(data)

        data = self.add_volume_indicators(data)
        data = self.add_momentum_indicators(data)
        data = self.add_trend_indicators(data)
        data = self.add_support_resistance(data)
        data = self.add_volatility_indicators(data)

        # NEW (Nov 20, 2025): Advanced indicators for ML strategy generation
        data = self.add_advanced_momentum(data)
        data = self.add_volume_advanced(data)
        data = self.add_price_action(data)
        data = self.add_statistical_features(data)

        data = self.calculate_momentum_score(data)

        return data

    def get_signal_strength(self, data, row_index=-1):
        """Calculate overall signal strength for a given row (default: latest)"""
        row = data.iloc[row_index]

        signals = {
            'rsi': 'neutral',
            'macd': 'neutral',
            'bb': 'neutral',
            'trend': 'neutral',
            'volume': 'neutral'
        }

        # RSI signals
        if row['RSI'] < 30:
            signals['rsi'] = 'oversold'
        elif row['RSI'] > 70:
            signals['rsi'] = 'overbought'
        elif 40 < row['RSI'] < 60:
            signals['rsi'] = 'neutral'

        # MACD signals
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Histogram'] > 0:
            signals['macd'] = 'bullish'
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Histogram'] < 0:
            signals['macd'] = 'bearish'

        # Bollinger Bands
        if row['close'] < row['BB_LOWER']:
            signals['bb'] = 'oversold'
        elif row['close'] > row['BB_UPPER']:
            signals['bb'] = 'overbought'

        # Trend signals
        if row['SMA_5'] > row['SMA_20'] > row['SMA_50']:
            signals['trend'] = 'uptrend'
        elif row['SMA_5'] < row['SMA_20'] < row['SMA_50']:
            signals['trend'] = 'downtrend'

        # Volume signals
        if row['volume'] > row['Volume_SMA'] * 1.5:
            signals['volume'] = 'high'
        elif row['volume'] < row['Volume_SMA'] * 0.5:
            signals['volume'] = 'low'

        return signals
