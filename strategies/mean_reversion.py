"""
Mean Reversion Trading Strategy

Conservative mean reversion for proven symbols.
Requires stronger oversold conditions and uptrend context.

Key Features:
- VOLATILITY FILTER: Skip high-volatility stocks (ATR% > threshold)
  Mean reversion fails on volatile stocks but works on stable ones
- Only buy oversold in uptrend (price > SMA_50)
- Stricter RSI thresholds with BB position confirmation
"""

import logging
from typing import Any, Dict

import pandas as pd

from .base import TradingStrategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(TradingStrategy):
    """
    Mean Reversion Strategy

    Conservative approach for capital preservation:
    - Only buy oversold in uptrend (price > SMA_50)
    - Stricter RSI thresholds
    - BB position confirmation
    - VOLATILITY FILTER: Skip high-volatility stocks (ATR% > threshold)
      Mean reversion fails on volatile stocks (TSLA, SPY) but works on stable ones (MSFT, AAPL)
    """

    def __init__(
        self,
        buy_threshold: int = 55,
        sell_threshold: int = 70,
        enabled: bool = True,
        max_atr_pct: float = 2.5
    ):
        """
        Initialize Mean Reversion Strategy.

        Args:
            buy_threshold: Buy signal threshold (default: 55) - currently unused
            sell_threshold: Sell signal threshold (default: 70) - currently unused
            enabled: Whether strategy is active (default: True)
            max_atr_pct: Max ATR as % of price (default: 2.5% = low volatility)
        """
        super().__init__("MeanReversion_1Hour", enabled)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.max_atr_pct = max_atr_pct

    def calculate_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_price: float,
        indicators: Any,
        **kwargs
    ) -> Dict:
        """
        Calculate mean reversion trading signal.

        Signal Logic:
        - BUY: RSI < 32 AND BB position < 0.15 AND in uptrend
        - SELL: RSI > 72 AND BB position > 0.85
        - HOLD: Neutral zone or oversold in downtrend

        Args:
            symbol: Stock ticker symbol
            data: DataFrame with OHLCV and indicators
            current_price: Current market price
            indicators: TechnicalIndicators instance
            **kwargs: Additional parameters

        Returns:
            Signal dict with action, confidence, reasoning, components
        """
        if not self.enabled:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': 'Strategy disabled',
                'components': {}
            }

        if len(data) < 30:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': 'Insufficient data',
                'components': {}
            }

        latest = data.iloc[-1]

        # VOLATILITY FILTER: Skip high-volatility stocks
        atr = latest.get('ATR', None)
        if atr is not None and not pd.isna(atr) and current_price > 0:
            atr_pct = (atr / current_price) * 100
            if atr_pct > self.max_atr_pct:
                return {
                    'action': 'HOLD',
                    'confidence': 0,
                    'reasoning': f'Mean Reversion: Volatility too high (ATR {atr_pct:.1f}% > {self.max_atr_pct}%)',
                    'components': {
                        'atr_pct': round(atr_pct, 2),
                        'max_atr_pct': self.max_atr_pct
                    }
                }

        rsi = latest.get('RSI', 50)
        # NOTE: Indicators use uppercase BB_UPPER, BB_LOWER
        bb_lower = latest.get('BB_LOWER', latest.get('BB_Lower', current_price * 0.98))
        bb_upper = latest.get('BB_UPPER', latest.get('BB_Upper', current_price * 1.02))
        sma_50 = latest.get('SMA_50', current_price)

        if pd.isna(rsi):
            rsi = 50
        if pd.isna(bb_lower):
            bb_lower = current_price * 0.98
        if pd.isna(bb_upper):
            bb_upper = current_price * 1.02
        if pd.isna(sma_50):
            sma_50 = current_price

        bb_position = (
            (current_price - bb_lower) / (bb_upper - bb_lower)
            if (bb_upper - bb_lower) > 0
            else 0.5
        )
        in_uptrend = current_price > sma_50

        components = {
            'rsi': round(rsi, 1),
            'bb_position': round(bb_position, 2),
            'price': current_price,
            'bb_lower': round(bb_lower, 2),
            'bb_upper': round(bb_upper, 2),
            'sma_50': round(sma_50, 2),
            'in_uptrend': in_uptrend
        }

        # Buy: Oversold in uptrend only
        if rsi < 32 and bb_position < 0.15 and in_uptrend:
            confidence = 75 + (32 - rsi) / 2
            return {
                'action': 'BUY',
                'confidence': min(95, confidence),
                'reasoning': f'Mean Reversion: Oversold RSI {rsi:.0f} in uptrend, near lower BB',
                'components': components
            }

        # Sell: Overbought
        if rsi > 72 and bb_position > 0.85:
            confidence = 65 + (rsi - 72) / 2
            return {
                'action': 'SELL',
                'confidence': min(85, confidence),
                'reasoning': f'Mean Reversion: Overbought RSI {rsi:.0f}, near upper BB',
                'components': components
            }

        # Don't buy oversold in downtrend
        if rsi < 35 and not in_uptrend:
            return {
                'action': 'HOLD',
                'confidence': 30,
                'reasoning': f'Mean Reversion: Oversold but in downtrend (price < SMA50)',
                'components': components
            }

        return {
            'action': 'HOLD',
            'confidence': 40,
            'reasoning': f'Mean Reversion: Neutral zone RSI {rsi:.0f}',
            'components': components
        }
