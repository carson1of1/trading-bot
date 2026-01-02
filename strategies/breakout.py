"""
Breakout Strategy - Range Breakout Detection

Detects price breakouts above/below recent trading ranges.
Uses volume confirmation and trend alignment for signal quality.
"""

import logging
from typing import Any, Dict

import pandas as pd

from .base import TradingStrategy

logger = logging.getLogger(__name__)


class BreakoutStrategy(TradingStrategy):
    """
    Breakout Strategy

    Conservative approach:
    - 40-hour range (~5 trading days)
    - Volume confirmation required
    - Trend alignment bonus
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

        components = {
            'recent_high': round(recent_high, 2),
            'recent_low': round(recent_low, 2),
            'price': current_price,
            'atr': round(atr, 2),
            'sma_20': round(sma_20, 2),
            'sma_50': round(sma_50, 2)
        }

        # Bullish breakout
        if current_price > recent_high:
            breakout_pct = (current_price - recent_high) / recent_high * 100
            confidence = 60 + min(25, breakout_pct * 8)

            # Volume confirmation required
            avg_volume = data['volume'].iloc[-21:-1].mean() if len(data) > 21 else data['volume'].mean()
            current_volume = data['volume'].iloc[-1]
            volume_surge = current_volume / avg_volume if avg_volume > 0 else 1

            if volume_surge > 1.3:
                confidence += 10
            elif volume_surge < 1.1:
                confidence -= 10

            # Trend alignment bonus
            if current_price > sma_20 > sma_50:
                confidence += 5

            return {
                'action': 'BUY',
                'confidence': min(90, max(50, confidence)),
                'reasoning': f'Breakout: Above {recent_high:.2f} by {breakout_pct:.2f}% (vol: {volume_surge:.1f}x)',
                'components': {**components, 'volume_surge': round(volume_surge, 2)}
            }

        # Bearish breakdown
        if current_price < recent_low:
            breakdown_pct = (recent_low - current_price) / recent_low * 100
            confidence = 55 + min(20, breakdown_pct * 8)
            return {
                'action': 'SELL',
                'confidence': min(80, confidence),
                'reasoning': f'Breakout: Below {recent_low:.2f} by {breakdown_pct:.2f}%',
                'components': components
            }

        return {
            'action': 'HOLD',
            'confidence': 35,
            'reasoning': f'Breakout: Price within range [{recent_low:.2f}, {recent_high:.2f}]',
            'components': components
        }
