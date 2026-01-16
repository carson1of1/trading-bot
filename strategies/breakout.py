"""
Breakout Strategy - Trend-Aware Breakout/Breakdown Detection

Detects price breakouts above/below recent trading ranges.
TREND-AWARE: Only generates signals aligned with the current trend.
- LONGS: Only when price is in uptrend (above SMA_20)
- SHORTS: Only when price is in downtrend (below SMA_20)

This prevents false signals from counter-trend trades.
"""

import logging
from typing import Any, Dict

import pandas as pd

from .base import TradingStrategy

logger = logging.getLogger(__name__)


class BreakoutStrategy(TradingStrategy):
    """
    Breakout Strategy - Trend-Aware

    Key principle: Trade WITH the trend, not against it.
    - Bullish breakouts require uptrend context (price > SMA_20)
    - Bearish breakdowns require downtrend context (price < SMA_20)

    Additional filters:
    - Volume confirmation required (1.5x avg for strong signals)
    - 40-hour range for breakout detection
    """

    def __init__(self, buy_threshold: int = 60, sell_threshold: int = 40, enabled: bool = True):
        """
        Initialize Breakout Strategy.

        Args:
            buy_threshold: Minimum confidence for BUY signals (default: 60)
            sell_threshold: Minimum confidence for SELL signals (default: 40)
            enabled: Whether the strategy is active (default: True)
        """
        super().__init__("Breakout_1Hour", enabled)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        # Quality filters
        self.strong_volume_threshold = 1.5  # Strong volume confirmation
        self.weak_volume_threshold = 1.1    # Minimum acceptable volume

    def calculate_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_price: float,
        indicators: Any,
        **kwargs
    ) -> Dict:
        """
        Calculate breakout signal for a symbol.

        TREND-AWARE LOGIC:
        - BUY signals only when price > SMA_20 (uptrend)
        - SELL signals only when price < SMA_20 (downtrend)

        Args:
            symbol: Stock ticker symbol
            data: DataFrame with OHLCV data and indicators
            current_price: Current market price
            indicators: TechnicalIndicators instance
            **kwargs: Additional parameters

        Returns:
            Dict with action, confidence, reasoning, and components
        """
        if not self.enabled:
            return {'action': 'HOLD', 'confidence': 0, 'reasoning': 'Strategy disabled', 'components': {}}

        if len(data) < 50:
            return {'action': 'HOLD', 'confidence': 0, 'reasoning': 'Insufficient data', 'components': {}}

        # 40-hour range (~5 trading days)
        lookback = min(40, len(data) - 5)
        recent_high = data['high'].iloc[-lookback-1:-1].max()
        recent_low = data['low'].iloc[-lookback-1:-1].min()

        latest = data.iloc[-1]
        atr = latest.get('ATR', (recent_high - recent_low) * 0.05)
        if pd.isna(atr):
            atr = (recent_high - recent_low) * 0.05

        sma_20 = latest.get('SMA_20', current_price)
        sma_50 = latest.get('SMA_50', current_price)
        if pd.isna(sma_20):
            sma_20 = current_price
        if pd.isna(sma_50):
            sma_50 = current_price

        # Calculate volume surge
        avg_volume = data['volume'].iloc[-21:-1].mean() if len(data) > 21 else data['volume'].mean()
        current_volume = data['volume'].iloc[-1]
        volume_surge = current_volume / avg_volume if avg_volume > 0 else 1

        # Determine trend context
        in_uptrend = current_price > sma_20
        strong_uptrend = current_price > sma_20 > sma_50
        in_downtrend = current_price < sma_20
        strong_downtrend = current_price < sma_20 < sma_50

        components = {
            'recent_high': round(recent_high, 2),
            'recent_low': round(recent_low, 2),
            'price': current_price,
            'atr': round(atr, 2),
            'sma_20': round(sma_20, 2),
            'sma_50': round(sma_50, 2),
            'volume_surge': round(volume_surge, 2),
            'in_uptrend': in_uptrend,
            'in_downtrend': in_downtrend
        }

        # ===== BULLISH BREAKOUT (LONG) =====
        if current_price > recent_high:
            breakout_pct = (current_price - recent_high) / recent_high * 100

            # TREND FILTER: Only take longs in uptrend
            if not in_uptrend:
                return {
                    'action': 'HOLD',
                    'confidence': 40,
                    'reasoning': f'Breakout: Price below SMA_20 - not in uptrend for LONG',
                    'components': components
                }

            # Build confidence score
            confidence = 60 + min(25, breakout_pct * 8)

            # Volume confirmation
            if volume_surge > self.strong_volume_threshold:
                confidence += 15
            elif volume_surge > self.weak_volume_threshold:
                confidence += 5
            elif volume_surge < 1.0:
                confidence -= 10

            # Strong trend bonus
            if strong_uptrend:
                confidence += 5

            return {
                'action': 'BUY',
                'confidence': min(95, max(55, confidence)),
                'reasoning': f'Breakout: Above {recent_high:.2f} by {breakout_pct:.2f}% in uptrend (vol: {volume_surge:.1f}x)',
                'components': components
            }

        # ===== BEARISH BREAKDOWN (SHORT) =====
        if current_price < recent_low:
            breakdown_pct = (recent_low - current_price) / recent_low * 100

            # TREND FILTER: Only take shorts in downtrend
            if not in_downtrend:
                return {
                    'action': 'HOLD',
                    'confidence': 40,
                    'reasoning': f'Breakdown: Price above SMA_20 - not in downtrend for SHORT',
                    'components': components
                }

            # Build confidence score
            confidence = 55 + min(20, breakdown_pct * 8)

            # Volume confirmation
            if volume_surge > self.strong_volume_threshold:
                confidence += 15
            elif volume_surge > self.weak_volume_threshold:
                confidence += 5
            elif volume_surge < 1.0:
                confidence -= 10

            # Strong downtrend bonus
            if strong_downtrend:
                confidence += 5

            return {
                'action': 'SELL',
                'confidence': min(90, max(50, confidence)),
                'reasoning': f'Breakdown: Below {recent_low:.2f} by {breakdown_pct:.2f}% in downtrend (vol: {volume_surge:.1f}x)',
                'components': components
            }

        return {
            'action': 'HOLD',
            'confidence': 35,
            'reasoning': f'Breakout: Price within range [{recent_low:.2f}, {recent_high:.2f}]',
            'components': components
        }
