"""
Filtered Mean Reversion Strategy

Walk-forward validated strategy that buys oversold conditions in strong uptrends
and shorts overbought conditions in strong downtrends.

Validation Results (12-month walk-forward, 147 symbols):
- 10/12 positive windows (83% consistency)
- Average Profit Factor: 2.81
- Total Expectancy: +15.37%

Economic Rationale:
Mean reversion works best in established trends where pullbacks are buying
opportunities (uptrend) or shorting opportunities (downtrend). Combining
strong trend filter with extreme RSI and Bollinger Band conditions improves odds.

Entry Logic:
- LONG: Price > SMA_100 (strong uptrend) + RSI < 30 (oversold) + Price at lower BB
- SHORT: Price < SMA_100 (strong downtrend) + RSI > 70 (overbought) + Price at upper BB
"""

import logging
from typing import Any, Dict

import pandas as pd

from .base import TradingStrategy

logger = logging.getLogger(__name__)


class FilteredMeanReversionStrategy(TradingStrategy):
    """
    Filtered Mean Reversion Strategy

    Walk-forward validated mean reversion with strong trend filter.
    Only takes mean reversion trades in the direction of the larger trend.

    Research validated:
    - 10/12 walk-forward windows positive
    - Average PF: 2.81
    - Expectancy: +0.41% per trade
    """

    def __init__(
        self,
        buy_threshold: int = 70,
        sell_threshold: int = 70,
        trend_period: int = 100,
        oversold_rsi: float = 30,
        overbought_rsi: float = 70,
        bb_period: int = 20,
        enabled: bool = True
    ):
        """
        Initialize the filtered mean reversion strategy.

        Args:
            buy_threshold: Minimum confidence for BUY signal (default: 70)
            sell_threshold: Minimum confidence for SELL signal (default: 70)
            trend_period: SMA period for trend determination (default: 100)
            oversold_rsi: RSI threshold for oversold (default: 30)
            overbought_rsi: RSI threshold for overbought (default: 70)
            bb_period: Bollinger Band period (default: 20)
            enabled: Whether the strategy is active (default: True)
        """
        super().__init__("FilteredMeanReversion_1Hour", enabled)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.trend_period = trend_period
        self.oversold_rsi = oversold_rsi
        self.overbought_rsi = overbought_rsi
        self.bb_period = bb_period

    def calculate_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_price: float,
        indicators: Any,
        **kwargs
    ) -> Dict:
        """
        Calculate filtered mean reversion trading signal.

        Requires confluence of:
        1. Strong trend (price vs SMA_100)
        2. RSI at extreme (oversold in uptrend, overbought in downtrend)
        3. Price at Bollinger Band extreme

        Args:
            symbol: Stock ticker symbol
            data: DataFrame with OHLCV data and indicators
            current_price: Current market price
            indicators: TechnicalIndicators instance
            **kwargs: Additional parameters (unused)

        Returns:
            Dict with action, confidence, reasoning, and components
        """
        if not self.enabled:
            return {'action': 'HOLD', 'confidence': 0, 'reasoning': 'Strategy disabled', 'components': {}}

        if len(data) < self.trend_period + 10:
            return {'action': 'HOLD', 'confidence': 0, 'reasoning': 'Insufficient data for trend calculation', 'components': {}}

        latest = data.iloc[-1]

        # Calculate SMA_100 for trend
        sma_100 = data['close'].rolling(self.trend_period).mean().iloc[-1]
        if pd.isna(sma_100):
            return {'action': 'HOLD', 'confidence': 0, 'reasoning': 'Unable to calculate trend', 'components': {}}

        # Determine trend
        in_uptrend = current_price > sma_100
        in_downtrend = current_price < sma_100

        # Get RSI
        rsi = latest.get('RSI', 50)
        if pd.isna(rsi):
            # Calculate RSI if not in data
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            rsi = rsi_series.iloc[-1]
            if pd.isna(rsi):
                rsi = 50

        # Check RSI conditions
        is_oversold = rsi < self.oversold_rsi
        is_overbought = rsi > self.overbought_rsi

        # Calculate Bollinger Bands
        bb_mid = data['close'].rolling(self.bb_period).mean()
        bb_std = data['close'].rolling(self.bb_period).std()
        bb_lower = (bb_mid - 2 * bb_std).iloc[-1]
        bb_upper = (bb_mid + 2 * bb_std).iloc[-1]

        if pd.isna(bb_lower) or pd.isna(bb_upper):
            return {'action': 'HOLD', 'confidence': 0, 'reasoning': 'Unable to calculate Bollinger Bands', 'components': {}}

        at_lower_bb = current_price <= bb_lower
        at_upper_bb = current_price >= bb_upper

        # Build components for logging
        components = {
            'price': round(current_price, 2),
            'sma_100': round(sma_100, 2),
            'rsi': round(rsi, 1),
            'bb_lower': round(bb_lower, 2),
            'bb_upper': round(bb_upper, 2),
            'trend': 'uptrend' if in_uptrend else 'downtrend',
            'at_bb_extreme': 'lower' if at_lower_bb else ('upper' if at_upper_bb else 'none')
        }

        # LONG SIGNAL: Uptrend + Oversold + At Lower BB
        if in_uptrend and is_oversold and at_lower_bb:
            confidence = self._calculate_long_confidence(rsi, current_price, sma_100, bb_lower)

            if confidence >= self.buy_threshold:
                return {
                    'action': 'BUY',
                    'confidence': confidence,
                    'reasoning': f"FilteredMR: Oversold bounce in uptrend (RSI: {rsi:.1f}, at lower BB)",
                    'components': components
                }

        # SHORT SIGNAL: Downtrend + Overbought + At Upper BB
        if in_downtrend and is_overbought and at_upper_bb:
            confidence = self._calculate_short_confidence(rsi, current_price, sma_100, bb_upper)

            if confidence >= self.sell_threshold:
                return {
                    'action': 'SELL',
                    'confidence': confidence,
                    'reasoning': f"FilteredMR: Overbought reversal in downtrend (RSI: {rsi:.1f}, at upper BB)",
                    'components': components
                }

        # Default: HOLD
        if in_uptrend:
            reasoning = f"FilteredMR: Uptrend but not oversold (RSI: {rsi:.1f})"
        elif in_downtrend:
            reasoning = f"FilteredMR: Downtrend but not overbought (RSI: {rsi:.1f})"
        else:
            reasoning = f"FilteredMR: No clear trend"

        return {
            'action': 'HOLD',
            'confidence': 0,
            'reasoning': reasoning,
            'components': components
        }

    def _calculate_long_confidence(
        self,
        rsi: float,
        price: float,
        sma_100: float,
        bb_lower: float
    ) -> float:
        """
        Calculate confidence for long entry.

        Higher confidence when:
        - RSI is more oversold (closer to 20)
        - Price is well above SMA_100 (strong uptrend)
        - Price is at or below lower BB

        Args:
            rsi: Current RSI value
            price: Current price
            sma_100: 100-period SMA
            bb_lower: Lower Bollinger Band

        Returns:
            Confidence score (0-100)
        """
        # RSI score: more oversold = better (20-30 range)
        if rsi < 20:
            rsi_score = 100
        elif rsi < 25:
            rsi_score = 90
        else:
            rsi_score = 75

        # Trend strength score: how far above SMA_100
        trend_strength = (price - sma_100) / sma_100 * 100
        if trend_strength > 10:
            trend_score = 90
        elif trend_strength > 5:
            trend_score = 80
        else:
            trend_score = 70

        # BB score: how far below lower BB
        bb_distance = (bb_lower - price) / price * 100
        if bb_distance > 2:
            bb_score = 95
        elif bb_distance > 0:
            bb_score = 85
        else:
            bb_score = 75

        # Weighted combination
        return (rsi_score * 0.4 + trend_score * 0.3 + bb_score * 0.3)

    def _calculate_short_confidence(
        self,
        rsi: float,
        price: float,
        sma_100: float,
        bb_upper: float
    ) -> float:
        """
        Calculate confidence for short entry.

        Higher confidence when:
        - RSI is more overbought (closer to 80)
        - Price is well below SMA_100 (strong downtrend)
        - Price is at or above upper BB

        Args:
            rsi: Current RSI value
            price: Current price
            sma_100: 100-period SMA
            bb_upper: Upper Bollinger Band

        Returns:
            Confidence score (0-100)
        """
        # RSI score: more overbought = better
        if rsi > 80:
            rsi_score = 100
        elif rsi > 75:
            rsi_score = 90
        else:
            rsi_score = 75

        # Trend strength score: how far below SMA_100
        trend_strength = (sma_100 - price) / sma_100 * 100
        if trend_strength > 10:
            trend_score = 90
        elif trend_strength > 5:
            trend_score = 80
        else:
            trend_score = 70

        # BB score: how far above upper BB
        bb_distance = (price - bb_upper) / price * 100
        if bb_distance > 2:
            bb_score = 95
        elif bb_distance > 0:
            bb_score = 85
        else:
            bb_score = 75

        # Weighted combination
        return (rsi_score * 0.4 + trend_score * 0.3 + bb_score * 0.3)
