"""
Defensive tests for failure scenarios.

Verifies the bot handles edge cases gracefully:
- Data fetch failures
- Malformed position dicts
- Max position violations
- Broker state divergence
- API failures with retry logic
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch

from bot import TradingBot


class TestDataFetchFailures:
    """Test that exit checks continue when data fetch fails for some symbols."""

    @pytest.fixture
    def bot_with_positions(self, tmp_path):
        """Create a bot with multiple positions for testing."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
risk_management:
  max_open_positions: 5
  max_position_dollars: 50000  # High limit to prevent position size guard triggering in tests
exit_manager:
  enabled: false
logging:
  database: "logs/trades.db"
"""
        universe = """
proven_symbols:
  - AAPL
  - MSFT
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
            bot = TradingBot(config_path=str(config_path))

            # Pre-populate positions
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 100,
                    'entry_price': 150.0,
                    'direction': 'LONG',
                    'entry_time': datetime.now()
                },
                'MSFT': {
                    'symbol': 'MSFT',
                    'qty': 50,
                    'entry_price': 400.0,
                    'direction': 'LONG',
                    'entry_time': datetime.now()
                }
            }
            bot.highest_prices = {'AAPL': 155.0, 'MSFT': 410.0}
            bot.lowest_prices = {'AAPL': 148.0, 'MSFT': 395.0}
            bot.trailing_stops = {
                'AAPL': {'activated': False, 'price': 0.0},
                'MSFT': {'activated': False, 'price': 0.0}
            }

            return bot

    def test_exit_check_continues_on_data_fetch_failure(self, bot_with_positions):
        """Exit checks should continue for other positions when one symbol's data fetch fails."""
        bot = bot_with_positions

        # Track which symbols had check_exit called
        checked_symbols = []
        original_check_exit = bot.check_exit

        def tracking_check_exit(symbol, *args, **kwargs):
            checked_symbols.append(symbol)
            return None  # No exit triggered

        bot.check_exit = tracking_check_exit

        # Mock fetch_data: AAPL returns None, MSFT returns valid data
        def mock_fetch_data(symbol, bars=200):
            if symbol == 'AAPL':
                return None  # Simulate data fetch failure
            else:
                return pd.DataFrame({
                    'open': [400.0] * 50,
                    'high': [405.0] * 50,
                    'low': [395.0] * 50,
                    'close': [402.0] * 50,
                    'volume': [1000000] * 50
                })

        bot.fetch_data = mock_fetch_data

        # Mock account sync
        mock_account = MagicMock()
        mock_account.cash = 100000.0
        mock_account.portfolio_value = 100000.0
        mock_account.last_equity = 100000.0
        bot.broker.get_account.return_value = mock_account
        bot.broker.get_positions.return_value = []
        bot.sync_account = MagicMock()
        bot.sync_positions = MagicMock()

        # Run trading cycle - should not crash
        bot.run_trading_cycle()

        # MSFT should still have had exit check called
        assert 'MSFT' in checked_symbols
        # AAPL should NOT be in checked_symbols because data fetch returned None
        assert 'AAPL' not in checked_symbols

    def test_exit_check_handles_empty_dataframe(self, bot_with_positions):
        """Exit checks should handle empty DataFrame gracefully."""
        bot = bot_with_positions
        initial_position_count = len(bot.open_positions)

        # Mock fetch_data to return empty DataFrame
        def mock_fetch_data(symbol, bars=200):
            return pd.DataFrame()  # Empty DataFrame

        bot.fetch_data = mock_fetch_data

        # Mock account sync
        mock_account = MagicMock()
        mock_account.cash = 100000.0
        mock_account.portfolio_value = 100000.0
        mock_account.last_equity = 100000.0
        bot.broker.get_account.return_value = mock_account
        bot.broker.get_positions.return_value = []
        bot.sync_account = MagicMock()
        bot.sync_positions = MagicMock()

        # Should not crash and positions should remain unchanged
        bot.run_trading_cycle()

        # Positions should not be erroneously modified
        assert len(bot.open_positions) == initial_position_count


class TestMalformedPositions:
    """Test handling of malformed position dictionaries."""

    @pytest.fixture
    def bot_with_mocks(self, tmp_path):
        """Create a bot with mocked components."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
exit_manager:
  enabled: false
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
            bot = TradingBot(config_path=str(config_path))
            # Disable tiered exits for simpler testing
            bot.use_tiered_exits = False
            return bot

    def test_exit_check_handles_missing_entry_price(self, bot_with_mocks):
        """check_exit should return None when position is missing entry_price."""
        bot = bot_with_mocks

        # Malformed position dict - missing entry_price
        position = {
            'qty': 100,
            'direction': 'LONG',
            'entry_time': datetime.now()
        }

        # Should return None, not raise KeyError
        result = bot.check_exit('AAPL', position, 100.0, bar_high=101.0, bar_low=99.0)

        assert result is None

    def test_exit_check_handles_missing_direction(self, bot_with_mocks):
        """check_exit should default to LONG when direction is missing."""
        bot = bot_with_mocks

        # Disable trailing stop for this test - we're testing direction defaulting
        bot.config['trailing_stop'] = {'enabled': False}

        # Position without direction field
        position = {
            'entry_price': 100.0,
            'qty': 100,
            'entry_time': datetime.now()
        }

        # Initialize tracking
        bot.highest_prices['AAPL'] = 100.0
        bot.lowest_prices['AAPL'] = 100.0
        bot.trailing_stops['AAPL'] = {'activated': False, 'price': 0.0}

        # Should not crash - defaults to LONG
        # Use tight bar range to avoid triggering any exits
        result = bot.check_exit('AAPL', position, 100.0, bar_high=100.5, bar_low=99.5)

        # No exit should be triggered at entry price
        assert result is None

    def test_run_trading_cycle_skips_malformed_positions(self, bot_with_mocks):
        """run_trading_cycle should skip positions with missing entry_price."""
        bot = bot_with_mocks

        # Add malformed position
        bot.open_positions = {
            'AAPL': {
                'symbol': 'AAPL',
                'qty': 100,
                'direction': 'LONG'
                # Missing entry_price!
            }
        }

        # Mock data fetch
        bot.fetch_data = MagicMock(return_value=pd.DataFrame({
            'open': [100.0] * 50,
            'high': [101.0] * 50,
            'low': [99.0] * 50,
            'close': [100.5] * 50,
            'volume': [1000000] * 50
        }))

        # Mock account sync
        mock_account = MagicMock()
        mock_account.cash = 100000.0
        mock_account.portfolio_value = 100000.0
        mock_account.last_equity = 100000.0
        bot.broker.get_account.return_value = mock_account
        bot.broker.get_positions.return_value = []
        bot.sync_account = MagicMock()
        bot.sync_positions = MagicMock()

        # Should not crash
        bot.run_trading_cycle()


class TestMaxPositionViolations:
    """Test handling of position limit violations."""

    @pytest.fixture
    def bot_at_max_positions(self, tmp_path):
        """Create a bot at max position limit."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
risk_management:
  max_open_positions: 2
  max_position_dollars: 50000  # High limit to prevent position size guard triggering
entry_gate:
  confidence_threshold: 60
logging:
  database: "logs/trades.db"
"""
        universe = """
proven_symbols:
  - AAPL
  - MSFT
  - GOOGL
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
            bot = TradingBot(config_path=str(config_path))

            # Pre-populate at max positions (2)
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 100,
                    'entry_price': 150.0,
                    'direction': 'LONG',
                    'entry_time': datetime.now()
                },
                'MSFT': {
                    'symbol': 'MSFT',
                    'qty': 50,
                    'entry_price': 400.0,
                    'direction': 'LONG',
                    'entry_time': datetime.now()
                }
            }

            return bot

    def test_entry_blocked_at_max_positions(self, bot_at_max_positions):
        """No new entries should be allowed when at max positions."""
        bot = bot_at_max_positions

        # Track if execute_entry was called
        bot.execute_entry = MagicMock()

        # Mock strong BUY signal for GOOGL
        def mock_check_entry(symbol, data, price):
            if symbol == 'GOOGL':
                return {
                    'action': 'BUY',
                    'confidence': 90,
                    'strategy': 'Momentum',
                    'reasoning': 'Strong signal',
                    'direction': 'LONG'
                }
            return {'action': 'HOLD', 'confidence': 0, 'strategy': '', 'reasoning': ''}

        bot.check_entry = mock_check_entry

        # Mock data fetch
        bot.fetch_data = MagicMock(return_value=pd.DataFrame({
            'open': [100.0] * 50,
            'high': [101.0] * 50,
            'low': [99.0] * 50,
            'close': [100.5] * 50,
            'volume': [1000000] * 50
        }))

        # Mock account sync
        mock_account = MagicMock()
        mock_account.cash = 100000.0
        mock_account.portfolio_value = 100000.0
        mock_account.last_equity = 100000.0
        bot.broker.get_account.return_value = mock_account
        bot.broker.get_positions.return_value = []
        bot.sync_account = MagicMock()
        bot.sync_positions = MagicMock()

        # Run trading cycle
        bot.run_trading_cycle()

        # execute_entry should NOT have been called - we're at max positions
        bot.execute_entry.assert_not_called()

    def test_position_count_matches_broker_after_sync(self, bot_at_max_positions):
        """Internal position count should match broker after sync."""
        bot = bot_at_max_positions

        # Clear internal positions
        bot.open_positions = {}

        # Mock broker returning 3 positions
        mock_positions = []
        for symbol in ['AAPL', 'MSFT', 'GOOGL']:
            mock_pos = MagicMock()
            mock_pos.symbol = symbol
            mock_pos.qty = 100
            mock_pos.avg_entry_price = 150.0
            mock_pos.current_price = 155.0
            mock_pos.unrealized_pl = 500.0
            mock_pos.side = 'long'
            mock_positions.append(mock_pos)

        bot.broker.get_positions.return_value = mock_positions

        # Sync positions
        bot.sync_positions()

        # Internal count should now match broker (3 positions)
        assert len(bot.open_positions) == 3
        assert 'AAPL' in bot.open_positions
        assert 'MSFT' in bot.open_positions
        assert 'GOOGL' in bot.open_positions


class TestBrokerStateDivergence:
    """Test reconciliation between internal state and broker state."""

    @pytest.fixture
    def bot_with_mocks(self, tmp_path):
        """Create a bot with mocked components."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
exit_manager:
  enabled: false
logging:
  database: "logs/trades.db"
"""
        universe = """
proven_symbols:
  - AAPL
  - MSFT
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
            bot = TradingBot(config_path=str(config_path))
            return bot

    def test_reconciliation_removes_ghost_positions(self, bot_with_mocks):
        """Ghost positions (internal only) should be removed during sync."""
        bot = bot_with_mocks

        # Internal state has AAPL position
        bot.open_positions = {
            'AAPL': {
                'symbol': 'AAPL',
                'qty': 100,
                'entry_price': 150.0,
                'direction': 'LONG'
            }
        }
        bot.highest_prices = {'AAPL': 155.0}
        bot.lowest_prices = {'AAPL': 148.0}
        bot.trailing_stops = {'AAPL': {'activated': True, 'price': 150.0}}

        # Broker returns empty (position was closed externally)
        bot.broker.get_positions.return_value = []

        # Sync should remove ghost position
        bot.sync_positions()

        # AAPL should be removed from all tracking
        assert 'AAPL' not in bot.open_positions
        assert 'AAPL' not in bot.highest_prices
        assert 'AAPL' not in bot.lowest_prices
        assert 'AAPL' not in bot.trailing_stops

    def test_reconciliation_adds_missing_positions(self, bot_with_mocks):
        """Positions from broker should be added to internal state."""
        bot = bot_with_mocks

        # Internal state is empty
        bot.open_positions = {}

        # Broker has AAPL position
        mock_pos = MagicMock()
        mock_pos.symbol = 'AAPL'
        mock_pos.qty = 100
        mock_pos.avg_entry_price = 150.0
        mock_pos.current_price = 155.0
        mock_pos.unrealized_pl = 500.0
        mock_pos.side = 'long'

        bot.broker.get_positions.return_value = [mock_pos]

        # Sync should add position
        bot.sync_positions()

        # AAPL should now be tracked
        assert 'AAPL' in bot.open_positions
        assert bot.open_positions['AAPL']['qty'] == 100
        assert bot.open_positions['AAPL']['entry_price'] == 150.0
        assert bot.open_positions['AAPL']['direction'] == 'LONG'

        # Tracking should be initialized
        assert 'AAPL' in bot.highest_prices
        assert 'AAPL' in bot.lowest_prices
        assert 'AAPL' in bot.trailing_stops

    def test_handles_broker_returning_none(self, bot_with_mocks):
        """Should handle broker returning None for positions gracefully."""
        bot = bot_with_mocks

        # Internal state has position
        bot.open_positions = {
            'AAPL': {
                'symbol': 'AAPL',
                'qty': 100,
                'entry_price': 150.0,
                'direction': 'LONG'
            }
        }

        # Broker returns None (API error case)
        bot.broker.get_positions.return_value = None

        # Should not crash - exception is caught in sync_positions
        try:
            bot.sync_positions()
        except TypeError:
            # This is expected if the code doesn't handle None
            pytest.fail("sync_positions should handle None from broker.get_positions()")


class TestAPIRetry:
    """Test API retry logic for transient failures."""

    def test_retry_decorator_retries_on_failure(self):
        """retry_on_failure decorator should retry specified number of times."""
        from core.broker import retry_on_failure

        call_count = 0

        class MockService:
            def __init__(self):
                self.logger = MagicMock()

            @retry_on_failure(max_retries=2, delay=0.01, backoff=1.0)
            def flaky_method(self):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("Network error")
                return "success"

        service = MockService()
        result = service.flaky_method()

        assert result == "success"
        assert call_count == 3  # Initial + 2 retries

    def test_retry_decorator_raises_after_max_retries(self):
        """retry_on_failure should raise after exhausting retries."""
        from core.broker import retry_on_failure

        call_count = 0

        class MockService:
            def __init__(self):
                self.logger = MagicMock()

            @retry_on_failure(max_retries=2, delay=0.01, backoff=1.0)
            def always_fails(self):
                nonlocal call_count
                call_count += 1
                raise ConnectionError("Persistent network error")

        service = MockService()

        with pytest.raises(ConnectionError):
            service.always_fails()

        # Should have tried 3 times (initial + 2 retries)
        assert call_count == 3

    def test_alpaca_broker_get_account_has_retry(self):
        """AlpacaBroker.get_account should have retry logic."""
        from core.broker import AlpacaBroker

        # Check that the method has the retry decorator
        # by inspecting its __wrapped__ attribute
        method = AlpacaBroker.get_account
        assert hasattr(method, '__wrapped__'), "get_account should have retry decorator"

    def test_alpaca_broker_get_positions_has_retry(self):
        """AlpacaBroker.get_positions should have retry logic."""
        from core.broker import AlpacaBroker

        method = AlpacaBroker.get_positions
        assert hasattr(method, '__wrapped__'), "get_positions should have retry decorator"

    def test_execute_entry_handles_broker_exception(self, tmp_path):
        """execute_entry should handle broker exceptions gracefully."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
risk_management:
  max_position_size_pct: 5.0
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
            bot = TradingBot(config_path=str(config_path))
            bot.portfolio_value = 100000.0

            # Mock broker to raise exception on submit_bracket_order
            bot.broker.submit_bracket_order.side_effect = Exception("API connection failed")

            # execute_entry should catch exception and return filled=False
            result = bot.execute_entry('AAPL', 'LONG', 150.0, 'Momentum', 'Test')

            assert result['filled'] is False
            assert 'API connection failed' in result['reason']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
