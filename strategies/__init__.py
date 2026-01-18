"""
Strategies module - Trading strategy implementations

This module provides the base TradingStrategy interface and concrete strategy implementations.
"""

from strategies.base import TradingStrategy
from strategies.breakout import BreakoutStrategy
from strategies.filtered_mean_reversion import FilteredMeanReversionStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.manager import StrategyManager

__all__ = [
    'TradingStrategy',
    'BreakoutStrategy',
    'FilteredMeanReversionStrategy',
    'MeanReversionStrategy',
    'MomentumStrategy',
    'StrategyManager',
]
