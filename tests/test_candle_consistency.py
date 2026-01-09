"""
Tests for candle data consistency between live bot and backtest.

Verifies:
- Live bot passes historical data (excluding current bar) to strategies
- This matches backtest behavior where data.iloc[:i] is passed
- Trading flow still works correctly with the fix

FIX (Jan 2026): Addresses discrepancy where live bot passed ALL data
including current bar, while backtest excluded it. This caused different
indicator values and signal timing between live and backtest.
"""

import os
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call
import pytz


class TestCandleDataConsistency:
    """Test that live bot and backtest see the same candle data."""

    def _create_sample_data(self, num_bars: int = 50) -> pd.DataFrame:
        """Create sample OHLCV data with indicators."""
        np.random.seed(42)

        # Generate realistic price data
        base_price = 100.0
        prices = [base_price]
        for _ in range(num_bars - 1):
            change = np.random.normal(0, 0.5)
            prices.append(prices[-1] * (1 + change / 100))

        # Create timestamps (hourly bars)
        et = pytz.timezone('America/New_York')
        base_time = datetime(2026, 1, 7, 9, 30, tzinfo=et)
        timestamps = [base_time + timedelta(hours=i) for i in range(num_bars)]

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': [p * (1 + np.random.uniform(-0.002, 0.002)) for p in prices],
            'volume': [1000000 + np.random.randint(-100000, 100000) for _ in range(num_bars)]
        })

        # Add indicators (simplified)
        df['SMA_20'] = df['close'].rolling(20).mean()
        df['SMA_50'] = df['close'].rolling(50).mean()
        df['RSI'] = 50 + np.random.uniform(-10, 10, num_bars)  # Simplified RSI

        return df

    @pytest.fixture
    def bot_with_mocks(self, tmp_path):
        """Create a bot with mocked components."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
entry_gate:
  confidence_threshold: 60
  min_time_between_trades_minutes: 60
logging:
  database: "logs/trades.db"
"""
        universe = """
proven_symbols:
  - AAPL
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config)
        universe_path = tmp_path / "universe.yaml"
        universe_path.write_text(universe)

        with patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):
            mock_broker_instance = MagicMock()
            mock_broker.return_value = mock_broker_instance

            from bot import TradingBot
            bot = TradingBot(config_path=str(config_path))
            return bot

    def test_check_entry_receives_historical_data_only(self, bot_with_mocks):
        """
        Verify check_entry passes historical data (excluding current bar).

        The strategy should see data.iloc[:-1] (bars 0 to N-1) and
        current_price should be from the latest bar (bar N).

        This matches backtest behavior in generate_signals():
            historical_data = data.iloc[:i].copy()  # excludes bar i
            current_price = data.iloc[i]['close']   # bar i's close
        """
        bot = bot_with_mocks
        data = self._create_sample_data(50)

        # Mock strategy manager to capture what data it receives
        captured_data = []
        captured_prices = []

        original_get_best_signal = bot.strategy_manager.get_best_signal
        def capture_signal(symbol, data, current_price, indicators, **kwargs):
            captured_data.append(len(data))
            captured_prices.append(current_price)
            return {'action': 'HOLD', 'confidence': 0, 'reasoning': 'test'}

        bot.strategy_manager.get_best_signal = capture_signal

        # Call check_entry with 50 bars of data
        current_price = data['close'].iloc[-1]
        result = bot.check_entry('AAPL', data, current_price)

        # The strategy should receive N-1 bars (49), not N bars (50)
        # because we exclude the current bar to match backtest
        assert len(captured_data) == 1, "get_best_signal should be called once"
        assert captured_data[0] == 49, f"Strategy should receive 49 bars (N-1), got {captured_data[0]}"
        assert captured_prices[0] == current_price, "current_price should be from latest bar"

    def test_indicator_values_match_between_live_and_backtest(self, bot_with_mocks):
        """
        Verify that indicator values seen by strategy match between live and backtest.

        When strategy reads data.iloc[-1]['SMA_20'], it should get the SMA
        calculated from bars 0 to N-2 (same as backtest at bar N-1).
        """
        bot = bot_with_mocks
        data = self._create_sample_data(50)

        # Capture what the strategy sees
        captured_sma20 = []

        def capture_signal(symbol, data, current_price, indicators, **kwargs):
            latest = data.iloc[-1]
            captured_sma20.append(latest.get('SMA_20', None))
            return {'action': 'HOLD', 'confidence': 0, 'reasoning': 'test'}

        bot.strategy_manager.get_best_signal = capture_signal

        current_price = data['close'].iloc[-1]
        bot.check_entry('AAPL', data, current_price)

        # The SMA_20 should be from bar N-2 (index -2 of original data)
        # because we pass data.iloc[:-1] and strategy reads iloc[-1]
        expected_sma20 = data['SMA_20'].iloc[-2]  # Bar before current

        assert len(captured_sma20) == 1
        if not pd.isna(expected_sma20) and not pd.isna(captured_sma20[0]):
            assert abs(captured_sma20[0] - expected_sma20) < 0.01, \
                f"SMA_20 mismatch: got {captured_sma20[0]}, expected {expected_sma20}"

    def test_backtest_generate_signals_data_slicing(self):
        """
        Verify backtest generate_signals passes data.iloc[:i] (excludes current bar).

        This is the reference behavior that live bot should match.

        Note: This test verifies the CODE LOGIC, not the runtime behavior,
        because backtest.generate_signals has scanner filtering that may skip calls.
        """
        from backtest import Backtest1Hour

        # Verify the backtest code does the right thing by inspecting the source
        import inspect
        source = inspect.getsource(Backtest1Hour.generate_signals)

        # The key pattern we're looking for:
        # historical_data = data.iloc[:i].copy()  # excludes bar i
        assert 'data.iloc[:i]' in source, \
            "Backtest should use data.iloc[:i] to exclude current bar"

        # And current_price from bar i
        assert "data.iloc[i]['close']" in source or "data.iloc[i]" in source, \
            "Backtest should get current_price from data.iloc[i]"

        # Verify the strategy receives historical_data, not full data
        assert 'data=historical_data' in source, \
            "Backtest should pass historical_data (not data) to strategy"

    def test_trading_flow_works_with_fix(self, bot_with_mocks):
        """
        Verify that trading still works correctly after the fix.

        A BUY signal should still result in a trade being added to
        qualifying_signals when confidence meets threshold.
        """
        bot = bot_with_mocks
        data = self._create_sample_data(50)

        # Mock strategy to return a BUY signal
        bot.strategy_manager.get_best_signal = MagicMock(return_value={
            'action': 'BUY',
            'confidence': 75,
            'strategy': 'TestStrategy',
            'reasoning': 'Test buy signal',
            'components': {}
        })

        current_price = data['close'].iloc[-1]
        result = bot.check_entry('AAPL', data, current_price)

        assert result['action'] == 'BUY', f"Expected BUY, got {result['action']}"
        assert result['confidence'] == 75
        assert result['direction'] == 'LONG'

    def test_check_entry_minimum_data_requirement(self, bot_with_mocks):
        """
        Verify check_entry requires enough bars after excluding current bar.

        With 30 bars minimum required and excluding current bar,
        we need at least 31 bars of input data.
        """
        bot = bot_with_mocks

        # 30 bars should fail (need 31 minimum)
        data_30 = self._create_sample_data(30)
        result = bot.check_entry('AAPL', data_30, 100.0)
        assert result['action'] == 'HOLD'
        assert 'Insufficient data' in result['reasoning']

        # 31 bars should work (30 after excluding current bar)
        data_31 = self._create_sample_data(31)

        # Mock strategy to avoid actual strategy execution
        bot.strategy_manager.get_best_signal = MagicMock(return_value={
            'action': 'HOLD',
            'confidence': 0,
            'reasoning': 'test'
        })

        result = bot.check_entry('AAPL', data_31, 100.0)
        # Should not fail with "Insufficient data" - strategy was called
        assert bot.strategy_manager.get_best_signal.called


@pytest.mark.skipif(
    not os.getenv('ALPACA_API_KEY'),
    reason="Requires Alpaca credentials (integration test)"
)
class TestFakeTradeSimulation:
    """Test that a fake trade can go through end-to-end."""

    @pytest.fixture
    def bot_with_fake_broker(self, tmp_path):
        """Create a bot with FakeBroker for trade simulation."""
        config = """
mode: DRY_RUN
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
entry_gate:
  confidence_threshold: 60
  min_time_between_trades_minutes: 0
risk_management:
  max_position_size_pct: 20
  stop_loss_pct: 2.0
  take_profit_pct: 4.0
exit_manager:
  enabled: true
  tier_0_hard_stop: -0.02
logging:
  database: "logs/trades.db"
"""
        universe = """
proven_symbols:
  - AAPL
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config)
        universe_path = tmp_path / "universe.yaml"
        universe_path.write_text(universe)

        with patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):
            from bot import TradingBot
            bot = TradingBot(config_path=str(config_path))

            # Set up fake account state
            bot.cash = 100000.0
            bot.portfolio_value = 100000.0

            return bot

    def _create_sample_data(self, num_bars: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data with indicators."""
        np.random.seed(42)

        base_price = 150.0
        prices = [base_price]
        for _ in range(num_bars - 1):
            change = np.random.normal(0.1, 0.3)  # Slight uptrend
            prices.append(prices[-1] * (1 + change / 100))

        et = pytz.timezone('America/New_York')
        base_time = datetime(2026, 1, 7, 9, 30, tzinfo=et)
        timestamps = [base_time + timedelta(hours=i) for i in range(num_bars)]

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': [p * (1 + np.random.uniform(-0.001, 0.001)) for p in prices],
            'volume': [1000000 + np.random.randint(-100000, 100000) for _ in range(num_bars)]
        })

        # Add indicators
        df['SMA_20'] = df['close'].rolling(20).mean()
        df['SMA_50'] = df['close'].rolling(50).mean()
        df['RSI'] = 55 + np.random.uniform(-5, 5, num_bars)
        df['ATR'] = df['high'] - df['low']

        return df

    def test_fake_trade_entry_flow(self, bot_with_fake_broker):
        """
        Test that a fake trade can be entered end-to-end.

        Simulates:
        1. Getting candle data
        2. Checking entry signal
        3. Executing entry (with FakeBroker)
        """
        bot = bot_with_fake_broker
        data = self._create_sample_data(100)

        # Mock the strategy to return a strong BUY signal
        bot.strategy_manager.get_best_signal = MagicMock(return_value={
            'action': 'BUY',
            'confidence': 80,
            'strategy': 'Momentum_1Hour',
            'reasoning': 'Strong momentum setup',
            'components': {'rsi': 55, 'volume_surge': 1.3}
        })

        current_price = data['close'].iloc[-1]

        # Step 1: Check entry
        signal = bot.check_entry('AAPL', data, current_price)

        assert signal['action'] == 'BUY', f"Expected BUY signal, got {signal}"
        assert signal['confidence'] >= 60

        # Step 2: Calculate position size
        stop_loss_price = current_price * 0.98  # 2% stop
        position_size = bot.risk_manager.calculate_position_size(
            bot.portfolio_value, current_price, stop_loss_price
        )

        assert position_size > 0, "Position size should be positive"

        # Step 3: Execute entry (mock the broker order)
        bot.broker.submit_order = MagicMock(return_value=MagicMock(
            id='test-order-123',
            status='filled',
            filled_qty=position_size,
            filled_avg_price=current_price
        ))

        # Call execute_entry with correct signature
        result = bot.execute_entry(
            symbol='AAPL',
            direction='LONG',
            price=current_price,
            strategy=signal['strategy'],
            reasoning=signal['reasoning']
        )

        assert result['filled'] is True, f"Order should be filled, got {result}"

    def test_candle_then_trade_integration(self, bot_with_fake_broker):
        """
        Integration test: Get candle -> Generate signal -> Execute trade.

        Verifies the full flow works with the data consistency fix.
        """
        bot = bot_with_fake_broker
        data = self._create_sample_data(100)

        # Verify data has correct structure
        assert 'timestamp' in data.columns
        assert 'close' in data.columns
        assert len(data) == 100

        # The strategy receives data.iloc[:-1] (99 bars) after the fix
        # Verify this doesn't break the flow

        captured_data_len = None
        def capture_and_signal(symbol, data, current_price, indicators, **kwargs):
            nonlocal captured_data_len
            captured_data_len = len(data)
            return {
                'action': 'BUY',
                'confidence': 75,
                'strategy': 'TestStrategy',
                'reasoning': 'Test signal',
                'components': {}
            }

        bot.strategy_manager.get_best_signal = capture_and_signal

        current_price = data['close'].iloc[-1]
        signal = bot.check_entry('AAPL', data, current_price)

        # Verify strategy received N-1 bars
        assert captured_data_len == 99, f"Strategy should receive 99 bars, got {captured_data_len}"

        # Verify signal is valid
        assert signal['action'] == 'BUY'
        assert signal['direction'] == 'LONG'

        # Mock broker for execution
        bot.broker.submit_order = MagicMock(return_value=MagicMock(
            id='test-order-456',
            status='filled',
            filled_qty=10,
            filled_avg_price=current_price
        ))

        # Execute trade with correct signature
        result = bot.execute_entry(
            symbol='AAPL',
            direction='LONG',
            price=current_price,
            strategy=signal['strategy'],
            reasoning=signal['reasoning']
        )
        assert result['filled'] is True
