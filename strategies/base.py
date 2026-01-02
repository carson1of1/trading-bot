"""
Base Trading Strategy Interface

Defines the abstract base class for all trading strategies.
All concrete strategy implementations must inherit from TradingStrategy
and implement the calculate_signal method.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class TradingStrategy(ABC):
    """
    Base class for trading strategies.

    All trading strategies must inherit from this class and implement
    the calculate_signal method to generate trading signals.

    Attributes:
        name: Strategy identifier name
        enabled: Whether the strategy is active for signal generation
    """

    def __init__(self, name: str, enabled: bool = True):
        """
        Initialize the trading strategy.

        Args:
            name: Strategy identifier name
            enabled: Whether the strategy is active (default: True)
        """
        self.name = name
        self.enabled = enabled

    @abstractmethod
    def calculate_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_price: float,
        indicators: Any,
        **kwargs
    ) -> Dict:
        """
        Calculate trading signal for a symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            data: DataFrame with OHLCV data and calculated indicators
            current_price: Current market price
            indicators: TechnicalIndicators instance for additional calculations
            **kwargs: Additional strategy-specific parameters

        Returns:
            Dict containing:
                - action: 'BUY', 'SELL', or 'HOLD'
                - confidence: Signal strength (0-100)
                - reasoning: Human-readable explanation
                - components: Dict of indicator values used in decision
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.enabled})"
