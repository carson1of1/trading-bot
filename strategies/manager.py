"""
Strategy Manager - Coordinates multiple trading strategies

Aggregates signals from individual strategies and returns the best one.
Simplified from the source bot_1hour/strategy/strategy_manager.py.
"""

import logging
from typing import Dict, List

import pandas as pd

from .base import TradingStrategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy

logger = logging.getLogger(__name__)


class StrategyManager:
    """
    Strategy Manager - Coordinates trading strategies.

    Evaluates all enabled strategies and returns the best signal
    based on confidence scores.

    Attributes:
        strategies: List of TradingStrategy instances
        confidence_threshold: Minimum confidence for BUY signals
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the strategy manager.

        Args:
            config: Configuration dict with strategy settings
        """
        self.config = config or {}
        self.strategies: List[TradingStrategy] = []

        # Get confidence threshold from entry_gate config
        self.confidence_threshold = self.config.get('entry_gate', {}).get(
            'confidence_threshold', 60
        )

        self._initialize_strategies()

        logger.info(f"StrategyManager initialized with {len(self.strategies)} strategies")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")

    def _initialize_strategies(self):
        """Initialize trading strategies based on config."""
        strat_config = self.config.get('strategies', {})

        # Momentum strategy
        mom_config = strat_config.get('momentum', {})
        self.strategies.append(MomentumStrategy(
            buy_threshold=55,
            sell_threshold=35,
            enabled=mom_config.get('enabled', True)
        ))

        # Mean Reversion strategy
        mr_config = strat_config.get('mean_reversion', {})
        self.strategies.append(MeanReversionStrategy(
            buy_threshold=55,
            sell_threshold=70,
            enabled=mr_config.get('enabled', True)
        ))

        # Breakout strategy
        breakout_config = strat_config.get('breakout', {})
        self.strategies.append(BreakoutStrategy(
            buy_threshold=60,
            sell_threshold=40,
            enabled=breakout_config.get('enabled', True)
        ))

    def evaluate_all_strategies(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_price: float,
        indicators,
        **kwargs
    ) -> List[Dict]:
        """
        Evaluate all enabled strategies for a symbol.

        Args:
            symbol: Stock ticker symbol
            data: DataFrame with OHLCV data and indicators
            current_price: Current market price
            indicators: TechnicalIndicators instance
            **kwargs: Additional strategy parameters

        Returns:
            List of signal dicts sorted by confidence (highest first)
        """
        results = []

        for strategy in self.strategies:
            if not strategy.enabled:
                continue

            try:
                result = strategy.calculate_signal(
                    symbol=symbol,
                    data=data,
                    current_price=current_price,
                    indicators=indicators,
                    **kwargs
                )
                result['strategy'] = strategy.name
                results.append(result)

            except Exception as e:
                logger.error(f"Error in {strategy.name} for {symbol}: {e}")

        # Sort by confidence descending
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results

    def get_best_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_price: float,
        indicators,
        **kwargs
    ) -> Dict:
        """
        Get the best trading signal across all strategies.

        Returns the highest confidence signal if it meets the threshold.

        Args:
            symbol: Stock ticker symbol
            data: DataFrame with OHLCV data and indicators
            current_price: Current market price
            indicators: TechnicalIndicators instance
            **kwargs: Additional strategy parameters

        Returns:
            Best signal dict with action, confidence, reasoning, strategy, components
        """
        results = self.evaluate_all_strategies(
            symbol, data, current_price, indicators, **kwargs
        )

        if not results:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': 'No strategies evaluated',
                'strategy': 'None',
                'components': {},
                'all_strategies': []
            }

        best = results[0]

        # Apply confidence threshold for BUY signals
        if best['action'] == 'BUY' and best['confidence'] < self.confidence_threshold:
            logger.debug(
                f"[{symbol}] {best['strategy']} BUY blocked: "
                f"{best['confidence']:.0f} < {self.confidence_threshold}"
            )
            best = {
                'action': 'HOLD',
                'confidence': best['confidence'],
                'reasoning': f"Below threshold ({best['confidence']:.0f} < {self.confidence_threshold})",
                'strategy': best['strategy'],
                'components': best.get('components', {}),
                'all_strategies': results
            }
        else:
            best['all_strategies'] = results

        return best

    def get_strategy_weights(self) -> Dict[str, float]:
        """
        Get the weight for each strategy.

        Returns:
            Dict mapping strategy name to weight
        """
        strat_config = self.config.get('strategies', {})
        return {
            'Momentum_1Hour': strat_config.get('momentum', {}).get('weight', 0.35),
            'MeanReversion_1Hour': strat_config.get('mean_reversion', {}).get('weight', 0.25),
            'Breakout_1Hour': strat_config.get('breakout', {}).get('weight', 0.20),
        }
