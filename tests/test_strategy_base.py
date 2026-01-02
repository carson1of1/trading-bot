"""
Tests for TradingStrategy base class.

Verifies:
- ABC behavior (cannot instantiate directly)
- Subclass requirements (must implement calculate_signal)
- Initialization and attributes
- Method signatures
"""

import pandas as pd
import pytest
from typing import Dict, Any

from strategies.base import TradingStrategy


class TestTradingStrategyABC:
    """Test that TradingStrategy is a proper ABC."""

    def test_cannot_instantiate_directly(self):
        """TradingStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            TradingStrategy("TestStrategy")

        assert "abstract" in str(exc_info.value).lower()
        assert "calculate_signal" in str(exc_info.value)

    def test_subclass_without_implementation_fails(self):
        """Subclass without calculate_signal implementation cannot be instantiated."""

        class IncompleteStrategy(TradingStrategy):
            pass

        with pytest.raises(TypeError) as exc_info:
            IncompleteStrategy("Incomplete")

        assert "abstract" in str(exc_info.value).lower()

    def test_subclass_with_implementation_succeeds(self):
        """Subclass with calculate_signal can be instantiated."""

        class CompleteStrategy(TradingStrategy):
            def calculate_signal(
                self,
                symbol: str,
                data: pd.DataFrame,
                current_price: float,
                indicators: Any,
                **kwargs
            ) -> Dict:
                return {
                    'action': 'HOLD',
                    'confidence': 0,
                    'reasoning': 'Test strategy',
                    'components': {}
                }

        strategy = CompleteStrategy("TestStrategy")
        assert strategy.name == "TestStrategy"
        assert strategy.enabled is True


class TestTradingStrategyInitialization:
    """Test TradingStrategy initialization."""

    def _create_concrete_strategy(self, name: str, enabled: bool = True):
        """Helper to create a concrete strategy for testing."""

        class ConcreteStrategy(TradingStrategy):
            def calculate_signal(
                self,
                symbol: str,
                data: pd.DataFrame,
                current_price: float,
                indicators: Any,
                **kwargs
            ) -> Dict:
                return {
                    'action': 'HOLD',
                    'confidence': 0,
                    'reasoning': 'Concrete test',
                    'components': {}
                }

        return ConcreteStrategy(name, enabled)

    def test_default_enabled_true(self):
        """Strategy is enabled by default."""
        strategy = self._create_concrete_strategy("MyStrategy")
        assert strategy.enabled is True

    def test_enabled_false(self):
        """Strategy can be disabled at init."""
        strategy = self._create_concrete_strategy("MyStrategy", enabled=False)
        assert strategy.enabled is False

    def test_name_stored(self):
        """Strategy name is stored correctly."""
        strategy = self._create_concrete_strategy("MomentumStrategy")
        assert strategy.name == "MomentumStrategy"

    def test_repr(self):
        """Strategy has useful repr."""
        strategy = self._create_concrete_strategy("TestStrategy", enabled=True)
        repr_str = repr(strategy)
        assert "ConcreteStrategy" in repr_str
        assert "TestStrategy" in repr_str
        assert "enabled=True" in repr_str


class TestCalculateSignalContract:
    """Test calculate_signal method contract."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV DataFrame."""
        return pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0],
            'volume': [1000, 1100, 1200],
        })

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock strategy that returns configurable signals."""

        class MockStrategy(TradingStrategy):
            def __init__(self, name: str, signal_to_return: Dict, enabled: bool = True):
                super().__init__(name, enabled)
                self._signal = signal_to_return

            def calculate_signal(
                self,
                symbol: str,
                data: pd.DataFrame,
                current_price: float,
                indicators: Any,
                **kwargs
            ) -> Dict:
                return self._signal

        return MockStrategy

    def test_returns_dict(self, mock_strategy, sample_data):
        """calculate_signal returns a dictionary."""
        signal = {
            'action': 'BUY',
            'confidence': 75,
            'reasoning': 'Strong momentum',
            'components': {'rsi': 55}
        }
        strategy = mock_strategy("Test", signal)
        result = strategy.calculate_signal("AAPL", sample_data, 103.0, None)

        assert isinstance(result, dict)

    def test_buy_signal(self, mock_strategy, sample_data):
        """BUY signal is valid."""
        signal = {
            'action': 'BUY',
            'confidence': 80,
            'reasoning': 'Bullish setup',
            'components': {}
        }
        strategy = mock_strategy("Test", signal)
        result = strategy.calculate_signal("AAPL", sample_data, 103.0, None)

        assert result['action'] == 'BUY'
        assert result['confidence'] == 80

    def test_sell_signal(self, mock_strategy, sample_data):
        """SELL signal is valid."""
        signal = {
            'action': 'SELL',
            'confidence': 70,
            'reasoning': 'Exit conditions met',
            'components': {}
        }
        strategy = mock_strategy("Test", signal)
        result = strategy.calculate_signal("AAPL", sample_data, 103.0, None)

        assert result['action'] == 'SELL'

    def test_hold_signal(self, mock_strategy, sample_data):
        """HOLD signal is valid."""
        signal = {
            'action': 'HOLD',
            'confidence': 50,
            'reasoning': 'No clear signal',
            'components': {}
        }
        strategy = mock_strategy("Test", signal)
        result = strategy.calculate_signal("AAPL", sample_data, 103.0, None)

        assert result['action'] == 'HOLD'

    def test_accepts_kwargs(self, sample_data):
        """calculate_signal accepts additional kwargs."""

        class KwargsStrategy(TradingStrategy):
            def calculate_signal(
                self,
                symbol: str,
                data: pd.DataFrame,
                current_price: float,
                indicators: Any,
                **kwargs
            ) -> Dict:
                # Use kwargs in the response
                extra_param = kwargs.get('extra_param', 'default')
                return {
                    'action': 'HOLD',
                    'confidence': 0,
                    'reasoning': f'Extra: {extra_param}',
                    'components': {'extra': extra_param}
                }

        strategy = KwargsStrategy("KwargsTest")
        result = strategy.calculate_signal(
            "AAPL", sample_data, 103.0, None,
            extra_param='custom_value'
        )

        assert result['components']['extra'] == 'custom_value'
        assert 'custom_value' in result['reasoning']


class TestStrategyEnabledBehavior:
    """Test enabled/disabled strategy behavior patterns."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV DataFrame."""
        return pd.DataFrame({
            'open': [100.0],
            'high': [102.0],
            'low': [99.0],
            'close': [101.0],
            'volume': [1000],
        })

    def test_disabled_strategy_pattern(self, sample_data):
        """Demonstrate disabled strategy pattern."""

        class RespectfulStrategy(TradingStrategy):
            def calculate_signal(
                self,
                symbol: str,
                data: pd.DataFrame,
                current_price: float,
                indicators: Any,
                **kwargs
            ) -> Dict:
                if not self.enabled:
                    return {
                        'action': 'HOLD',
                        'confidence': 0,
                        'reasoning': 'Strategy disabled',
                        'components': {}
                    }
                return {
                    'action': 'BUY',
                    'confidence': 80,
                    'reasoning': 'Active signal',
                    'components': {}
                }

        # Enabled strategy generates signals
        enabled_strategy = RespectfulStrategy("Test", enabled=True)
        result = enabled_strategy.calculate_signal("AAPL", sample_data, 101.0, None)
        assert result['action'] == 'BUY'

        # Disabled strategy holds
        disabled_strategy = RespectfulStrategy("Test", enabled=False)
        result = disabled_strategy.calculate_signal("AAPL", sample_data, 101.0, None)
        assert result['action'] == 'HOLD'
        assert result['confidence'] == 0
        assert 'disabled' in result['reasoning'].lower()


class TestModuleExports:
    """Test module exports."""

    def test_import_from_strategies(self):
        """TradingStrategy can be imported from strategies package."""
        from strategies import TradingStrategy as ImportedStrategy
        assert ImportedStrategy is TradingStrategy

    def test_trading_strategy_in_all(self):
        """TradingStrategy is in strategies.__all__."""
        import strategies
        assert 'TradingStrategy' in strategies.__all__
