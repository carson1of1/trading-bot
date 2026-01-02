"""
Momentum Trading Strategy

Conservative momentum strategy for capital preservation.
Requires strong trend alignment and filtered entry conditions.

Filters:
1. Strong trend: Price > SMA_20 > SMA_50
2. RSI in 45-72 range (not overbought/oversold)
3. Volume surge > 1.15x average

Uses 10-bar momentum calculation with weighted confidence scoring.
"""

import logging
from typing import Any, Dict

import pandas as pd

from .base import TradingStrategy

logger = logging.getLogger(__name__)


class MomentumStrategy(TradingStrategy):
    """
    Momentum Strategy

    Conservative filters for capital preservation:
    1. Price > SMA_20 > SMA_50 (strong trend alignment)
    2. RSI in 45-72 range (tighter overbought limit)
    3. Volume surge > 1.15x

    Lower thresholds for proven setups.
    """

    def __init__(self, buy_threshold: int = 55, sell_threshold: int = 35, enabled: bool = True):
        """
        Initialize the momentum strategy.

        Args:
            buy_threshold: Minimum confidence for BUY signal (default: 55)
            sell_threshold: Threshold below which to SELL (default: 35)
            enabled: Whether the strategy is active (default: True)
        """
        super().__init__("Momentum_1Hour", enabled)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_volume_surge = 1.15
        self.rsi_min = 45
        self.rsi_max = 72

    def calculate_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_price: float,
        indicators: Any,
        **kwargs
    ) -> Dict:
        """
        Calculate momentum trading signal.

        Args:
            symbol: Stock ticker symbol
            data: DataFrame with OHLCV data and indicators (SMA_20, SMA_50, RSI)
            current_price: Current market price
            indicators: TechnicalIndicators instance (not used directly)
            **kwargs: Additional parameters (unused)

        Returns:
            Dict with action, confidence, reasoning, and components
        """
        if not self.enabled:
            return {'action': 'HOLD', 'confidence': 0, 'reasoning': 'Strategy disabled', 'components': {}}

        if len(data) < 30:
            return {'action': 'HOLD', 'confidence': 0, 'reasoning': 'Insufficient data', 'components': {}}

        latest = data.iloc[-1]

        # Strong trend filter: Price > SMA_20 > SMA_50
        sma_20 = latest.get('SMA_20', current_price)
        sma_50 = latest.get('SMA_50', current_price)
        if pd.isna(sma_20):
            sma_20 = current_price
        if pd.isna(sma_50):
            sma_50 = current_price

        if not (current_price > sma_20 > sma_50):
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': f'Weak trend alignment (Price: ${current_price:.2f}, SMA20: ${sma_20:.2f}, SMA50: ${sma_50:.2f})',
                'components': {'price': current_price, 'sma_20': sma_20, 'sma_50': sma_50}
            }

        # RSI filter
        rsi = latest.get('RSI', 50)
        if pd.isna(rsi):
            rsi = 50
        if rsi < self.rsi_min or rsi > self.rsi_max:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': f'RSI {rsi:.1f} outside {self.rsi_min}-{self.rsi_max}',
                'components': {'rsi': round(rsi, 1)}
            }

        # Volume filter
        avg_volume = data['volume'].iloc[-21:-1].mean() if len(data) > 21 else data['volume'].mean()
        current_volume = data['volume'].iloc[-1]
        volume_surge = current_volume / avg_volume if avg_volume > 0 else 1

        if volume_surge < self.min_volume_surge:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': f'Low volume ({volume_surge:.2f}x < {self.min_volume_surge:.1f}x)',
                'components': {'volume_surge': round(volume_surge, 2)}
            }

        # Calculate momentum
        momentum_10bar = 0
        if len(data) >= 11:
            close_10 = data['close'].iloc[-10]
            if close_10 > 0:
                momentum_10bar = (current_price - close_10) / close_10

        confidence = self._calculate_confidence(volume_surge, momentum_10bar, rsi, current_price, sma_20, sma_50)

        if confidence >= self.buy_threshold:
            action = 'BUY'
            reasoning = f"Momentum: Strong hourly setup ({confidence:.0f}/100) - Trend aligned"
        elif confidence < self.sell_threshold or rsi > 70:
            action = 'SELL'
            reasoning = f"Momentum: Exit signal ({confidence:.0f}/100)"
        else:
            action = 'HOLD'
            reasoning = f"Momentum: Moderate ({confidence:.0f}/100)"

        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'components': {
                'rsi': round(rsi, 1),
                'volume_surge': round(volume_surge, 2),
                'momentum_10bar': round(momentum_10bar * 100, 2),
                'sma_20': round(sma_20, 2),
                'sma_50': round(sma_50, 2)
            }
        }

    def _calculate_confidence(
        self,
        volume_surge: float,
        momentum_10bar: float,
        rsi: float,
        price: float,
        sma_20: float,
        sma_50: float
    ) -> float:
        """
        Calculate confidence score with weighted components.

        Weights:
        - Momentum score: 30%
        - Volume score: 20%
        - RSI score: 25%
        - Trend score: 25%

        Args:
            volume_surge: Current volume / 20-day average volume
            momentum_10bar: 10-bar price change as decimal
            rsi: Current RSI value
            price: Current price
            sma_20: 20-period SMA
            sma_50: 50-period SMA

        Returns:
            Weighted confidence score (0-100)
        """
        momentum_pct = momentum_10bar * 100

        if momentum_pct > 5:
            momentum_score = 100
        elif momentum_pct > 3:
            momentum_score = 90
        elif momentum_pct > 2:
            momentum_score = 80
        elif momentum_pct > 1:
            momentum_score = 70
        else:
            momentum_score = 60

        if volume_surge >= 1.5:
            volume_score = 90
        elif volume_surge >= 1.3:
            volume_score = 80
        else:
            volume_score = 65

        if 50 <= rsi <= 65:
            rsi_score = 90
        elif 45 <= rsi <= 70:
            rsi_score = 75
        else:
            rsi_score = 60

        trend_gap = (price - sma_50) / sma_50 * 100 if sma_50 > 0 else 0
        if trend_gap > 5:
            trend_score = 90
        elif trend_gap > 2:
            trend_score = 80
        else:
            trend_score = 70

        return (momentum_score * 0.3 + volume_score * 0.2 + rsi_score * 0.25 + trend_score * 0.25)
