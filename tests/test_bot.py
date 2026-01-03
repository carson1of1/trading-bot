"""
Tests for the Trading Bot module.

Verifies:
- TradingBot initialization and configuration
- Account and position syncing
- Entry signal checking
- Exit condition checking (trailing stop, hard stop, take profit, max hold)
- Order execution
- Kill switch functionality
- Trading cycle orchestration
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

from bot import TradingBot, main


class TestTradingBotInitialization:
    """Test TradingBot initialization and configuration loading."""

    @patch('bot.VolatilityScanner')
    @patch('bot.create_broker')
    @patch('bot.TradeLogger')
    @patch('bot.YFinanceDataFetcher')
    def test_initialization_with_config(self, mock_fetcher, mock_logger, mock_broker, mock_scanner):
        """TradingBot initializes with config file."""
        mock_broker.return_value = MagicMock()
        mock_scanner_instance = MagicMock()
        mock_scanner_instance.scan.return_value = ['AAPL', 'MSFT']
        mock_scanner.return_value = mock_scanner_instance

        # Use the actual config.yaml in the trading-bot directory
        bot = TradingBot()

        assert bot.mode == 'PAPER'
        assert bot.timeframe == '1Hour'
        assert bot.running is False
        # Scanner returns 2 symbols
        assert len(bot.watchlist) >= 1

    @patch('bot.VolatilityScanner')
    @patch('bot.create_broker')
    @patch('bot.TradeLogger')
    @patch('bot.YFinanceDataFetcher')
    def test_initialization_components_created(self, mock_fetcher, mock_logger, mock_broker, mock_scanner):
        """TradingBot creates all required components."""
        mock_broker.return_value = MagicMock()
        mock_scanner_instance = MagicMock()
        mock_scanner_instance.scan.return_value = ['AAPL']
        mock_scanner.return_value = mock_scanner_instance

        bot = TradingBot()

        assert bot.data_fetcher is not None
        assert bot.indicators is not None
        assert bot.broker is not None
        assert bot.trade_logger is not None
        assert bot.risk_manager is not None
        assert bot.entry_gate is not None
        assert bot.exit_manager is not None
        assert bot.market_hours is not None
        assert bot.strategy_manager is not None

    @patch('bot.VolatilityScanner')
    @patch('bot.create_broker')
    @patch('bot.TradeLogger')
    @patch('bot.YFinanceDataFetcher')
    def test_initial_state(self, mock_fetcher, mock_logger, mock_broker, mock_scanner):
        """TradingBot initializes with correct default state."""
        mock_broker.return_value = MagicMock()
        mock_scanner_instance = MagicMock()
        mock_scanner_instance.scan.return_value = ['AAPL']
        mock_scanner.return_value = mock_scanner_instance

        bot = TradingBot()

        assert bot.cash == 0.0
        assert bot.portfolio_value == 0.0
        assert bot.peak_value == 0.0
        assert bot.daily_pnl == 0.0
        assert bot.kill_switch_triggered is False
        assert len(bot.open_positions) == 0
        assert len(bot.pending_entries) == 0
        assert len(bot.trailing_stops) == 0


class TestAccountSync:
    """Test account synchronization."""

    @pytest.fixture
    def bot_with_mocks(self, tmp_path):
        """Create a bot with mocked components."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
risk_management:
  max_daily_loss_pct: 3.0
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
            return bot

    def test_sync_account_updates_cash(self, bot_with_mocks):
        """sync_account updates cash from broker."""
        bot = bot_with_mocks
        mock_account = MagicMock()
        mock_account.cash = 50000.0
        mock_account.portfolio_value = 55000.0
        bot.broker.get_account.return_value = mock_account

        bot.sync_account()

        assert bot.cash == 50000.0
        assert bot.portfolio_value == 55000.0

    def test_sync_account_tracks_peak(self, bot_with_mocks):
        """sync_account tracks peak portfolio value."""
        bot = bot_with_mocks
        bot.peak_value = 50000.0

        mock_account = MagicMock()
        mock_account.cash = 50000.0
        mock_account.portfolio_value = 60000.0
        bot.broker.get_account.return_value = mock_account

        bot.sync_account()

        assert bot.peak_value == 60000.0

    def test_sync_account_triggers_kill_switch(self, bot_with_mocks):
        """sync_account triggers kill switch on daily loss limit."""
        bot = bot_with_mocks
        bot.current_trading_day = datetime.now().date()
        bot.daily_starting_capital = 100000.0

        mock_account = MagicMock()
        mock_account.cash = 96000.0
        mock_account.portfolio_value = 96000.0  # 4% loss
        bot.broker.get_account.return_value = mock_account

        bot.sync_account()

        assert bot.kill_switch_triggered is True

    def test_sync_account_resets_on_new_day(self, bot_with_mocks):
        """sync_account resets daily tracking on new day."""
        bot = bot_with_mocks
        bot.current_trading_day = datetime.now().date() - timedelta(days=1)
        bot.daily_pnl = 500.0
        bot.kill_switch_triggered = True

        mock_account = MagicMock()
        mock_account.cash = 100000.0
        mock_account.portfolio_value = 100000.0
        bot.broker.get_account.return_value = mock_account

        bot.sync_account()

        assert bot.current_trading_day == datetime.now().date()
        assert bot.daily_pnl == 0.0
        assert bot.kill_switch_triggered is False


class TestPositionSync:
    """Test position synchronization."""

    @patch('bot.VolatilityScanner')
    @patch('bot.create_broker')
    @patch('bot.TradeLogger')
    @patch('bot.YFinanceDataFetcher')
    def test_sync_positions_adds_new_positions(self, mock_fetcher, mock_logger, mock_broker, mock_scanner):
        """sync_positions adds new broker positions."""
        mock_broker_instance = MagicMock()
        mock_broker.return_value = mock_broker_instance
        mock_scanner_instance = MagicMock()
        mock_scanner_instance.scan.return_value = ['TSLA', 'AAPL']
        mock_scanner.return_value = mock_scanner_instance

        bot = TradingBot()
        test_symbol = bot.watchlist[0]

        mock_pos = MagicMock()
        mock_pos.symbol = test_symbol
        mock_pos.qty = 100
        mock_pos.avg_entry_price = 150.0
        mock_pos.current_price = 155.0
        mock_pos.unrealized_pl = 500.0
        mock_pos.side = 'long'

        bot.broker.get_positions.return_value = [mock_pos]

        bot.sync_positions()

        assert test_symbol in bot.open_positions
        assert bot.open_positions[test_symbol]['qty'] == 100
        assert bot.open_positions[test_symbol]['entry_price'] == 150.0
        assert bot.open_positions[test_symbol]['direction'] == 'LONG'

    @patch('bot.VolatilityScanner')
    @patch('bot.create_broker')
    @patch('bot.TradeLogger')
    @patch('bot.YFinanceDataFetcher')
    def test_sync_positions_initializes_tracking(self, mock_fetcher, mock_logger, mock_broker, mock_scanner):
        """sync_positions initializes price tracking for new positions."""
        mock_broker_instance = MagicMock()
        mock_broker.return_value = mock_broker_instance
        mock_scanner_instance = MagicMock()
        mock_scanner_instance.scan.return_value = ['TSLA', 'AAPL']
        mock_scanner.return_value = mock_scanner_instance

        bot = TradingBot()
        test_symbol = bot.watchlist[0]

        mock_pos = MagicMock()
        mock_pos.symbol = test_symbol
        mock_pos.qty = 100
        mock_pos.avg_entry_price = 150.0
        mock_pos.current_price = 155.0
        mock_pos.unrealized_pl = 500.0
        mock_pos.side = 'long'

        bot.broker.get_positions.return_value = [mock_pos]

        bot.sync_positions()

        assert test_symbol in bot.highest_prices
        assert test_symbol in bot.lowest_prices
        assert test_symbol in bot.trailing_stops

    @patch('bot.VolatilityScanner')
    @patch('bot.create_broker')
    @patch('bot.TradeLogger')
    @patch('bot.YFinanceDataFetcher')
    def test_sync_positions_cleans_up_closed(self, mock_fetcher, mock_logger, mock_broker, mock_scanner):
        """sync_positions cleans up externally closed positions."""
        mock_broker_instance = MagicMock()
        mock_broker.return_value = mock_broker_instance
        mock_scanner_instance = MagicMock()
        mock_scanner_instance.scan.return_value = ['TSLA']
        mock_scanner.return_value = mock_scanner_instance

        bot = TradingBot()
        # Set up a tracked position that is then removed
        bot.open_positions = {'AAPL': {'symbol': 'AAPL', 'qty': 100}}
        bot.highest_prices = {'AAPL': 155.0}
        bot.lowest_prices = {'AAPL': 145.0}
        bot.trailing_stops = {'AAPL': {'activated': True, 'price': 150.0}}

        bot.broker.get_positions.return_value = []  # No positions

        bot.sync_positions()

        assert 'AAPL' not in bot.open_positions
        assert 'AAPL' not in bot.highest_prices
        assert 'AAPL' not in bot.lowest_prices
        assert 'AAPL' not in bot.trailing_stops


class TestCheckEntry:
    """Test entry signal checking."""

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
            bot = TradingBot(config_path=str(config_path))
            return bot

    def test_check_entry_insufficient_data(self, bot_with_mocks):
        """check_entry returns HOLD for insufficient data."""
        bot = bot_with_mocks
        data = pd.DataFrame({'close': [100.0] * 10})

        result = bot.check_entry('AAPL', data, 100.0)

        assert result['action'] == 'HOLD'
        assert 'Insufficient data' in result['reasoning']

    def test_check_entry_kill_switch_blocks(self, bot_with_mocks):
        """check_entry returns HOLD when kill switch is active."""
        bot = bot_with_mocks
        bot.kill_switch_triggered = True
        data = pd.DataFrame({'close': [100.0] * 50})

        result = bot.check_entry('AAPL', data, 100.0)

        assert result['action'] == 'HOLD'
        assert 'Kill switch' in result['reasoning']

    def test_check_entry_existing_position_blocks(self, bot_with_mocks):
        """check_entry returns HOLD when position already exists."""
        bot = bot_with_mocks
        bot.open_positions = {'AAPL': {'symbol': 'AAPL'}}
        data = pd.DataFrame({'close': [100.0] * 50})

        result = bot.check_entry('AAPL', data, 100.0)

        assert result['action'] == 'HOLD'
        assert 'Already have position' in result['reasoning']

    def test_check_entry_cooldown_blocks(self, bot_with_mocks):
        """check_entry returns HOLD during cooldown period."""
        bot = bot_with_mocks
        bot.last_trade_time = {'AAPL': datetime.now() - timedelta(minutes=30)}
        data = pd.DataFrame({'close': [100.0] * 50})

        result = bot.check_entry('AAPL', data, 100.0)

        assert result['action'] == 'HOLD'
        assert 'Cooldown' in result['reasoning']


class TestCheckExit:
    """Test exit condition checking."""

    @pytest.fixture
    def bot_with_mocks(self, tmp_path):
        """Create a bot with mocked components."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
exit_manager:
  tier_0_hard_stop: -0.02
  tier_1_profit_floor: 0.02
  max_hold_hours: 48
trailing_stop:
  enabled: true
  activation_pct: 0.5
  trail_pct: 0.5
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
            # Disable exit manager for simpler testing of legacy exit paths
            bot.exit_manager = None
            # FIX (Jan 2026): Disable tiered exits to test legacy hard_stop/take_profit
            bot.use_tiered_exits = False
            return bot

    def test_check_exit_hard_stop_long(self, bot_with_mocks):
        """check_exit triggers hard stop for LONG position."""
        bot = bot_with_mocks
        position = {
            'entry_price': 100.0,
            'qty': 100,
            'direction': 'LONG',
            'entry_time': datetime.now()
        }

        # Price drops 3% below entry (below 2% stop)
        result = bot.check_exit('AAPL', position, 97.0, bar_high=98.0, bar_low=96.0)

        assert result is not None
        assert result['exit'] is True
        assert result['reason'] == 'hard_stop'

    def test_check_exit_take_profit_long(self, bot_with_mocks):
        """check_exit triggers take profit for LONG position."""
        bot = bot_with_mocks
        # Disable trailing stop for this test
        bot.config['trailing_stop'] = {'enabled': False}
        position = {
            'entry_price': 100.0,
            'qty': 100,
            'direction': 'LONG',
            'entry_time': datetime.now()
        }

        # Price rises 9% above entry (above 8% take profit from config)
        result = bot.check_exit('AAPL', position, 109.0, bar_high=110.0, bar_low=108.0)

        assert result is not None
        assert result['exit'] is True
        assert result['reason'] == 'take_profit'

    def test_check_exit_max_hold_time(self, bot_with_mocks):
        """check_exit triggers on max hold time."""
        bot = bot_with_mocks
        # Disable trailing stop for this test
        bot.config['trailing_stop'] = {'enabled': False}
        position = {
            'entry_price': 100.0,
            'qty': 100,
            'direction': 'LONG',
            'entry_time': datetime.now() - timedelta(hours=170)  # Exceed 168 hours (1 week)
        }

        result = bot.check_exit('AAPL', position, 101.0)

        assert result is not None
        assert result['exit'] is True
        assert result['reason'] == 'max_hold'

    def test_check_exit_no_trigger(self, bot_with_mocks):
        """check_exit returns None when no exit conditions met."""
        bot = bot_with_mocks
        # Disable trailing stop for this test to check no trigger scenario
        bot.config['trailing_stop'] = {'enabled': False}
        position = {
            'entry_price': 100.0,
            'qty': 100,
            'direction': 'LONG',
            'entry_time': datetime.now()
        }

        # Price at entry level - no exit conditions met
        result = bot.check_exit('AAPL', position, 100.5, bar_high=101.0, bar_low=99.5)

        assert result is None

    def test_check_exit_trailing_stop_activation(self, bot_with_mocks):
        """check_exit activates trailing stop at threshold."""
        bot = bot_with_mocks
        position = {
            'entry_price': 100.0,
            'qty': 100,
            'direction': 'LONG',
            'entry_time': datetime.now()
        }

        # First call at +1% (above 0.5% activation) with bar_low above trailing stop
        bot.check_exit('AAPL', position, 101.0, bar_high=101.5, bar_low=101.0)

        assert bot.trailing_stops['AAPL']['activated'] is True
        # Price is 100.0 entry, highest is 101.5, trailing is highest * (1 - 0.005) = 101.0
        # But move_to_breakeven starts at entry price, then trail updates
        assert bot.trailing_stops['AAPL']['price'] >= 100.0  # At least breakeven

    def test_check_exit_trailing_stop_triggers(self, bot_with_mocks):
        """check_exit triggers trailing stop on pullback."""
        bot = bot_with_mocks
        bot.trailing_stops['AAPL'] = {'activated': True, 'price': 101.0}
        bot.highest_prices['AAPL'] = 102.0
        bot.lowest_prices['AAPL'] = 100.0

        position = {
            'entry_price': 100.0,
            'qty': 100,
            'direction': 'LONG',
            'entry_time': datetime.now()
        }

        # Price drops to trailing stop
        result = bot.check_exit('AAPL', position, 100.5, bar_high=101.0, bar_low=100.8)

        assert result is not None
        assert result['exit'] is True
        assert result['reason'] == 'trailing_stop'


class TestExecuteEntry:
    """Test entry order execution."""

    @pytest.fixture
    def bot_with_mocks(self, tmp_path):
        """Create a bot with mocked components."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
risk_management:
  max_position_size_pct: 5.0
  stop_loss_pct: 2.0
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
            return bot

    def test_execute_entry_successful(self, bot_with_mocks):
        """execute_entry successfully executes order."""
        bot = bot_with_mocks

        mock_order = MagicMock()
        mock_order.status = 'filled'
        mock_order.id = 'order123'
        mock_order.filled_avg_price = 150.0
        mock_order.filled_qty = 100
        bot.broker.submit_order.return_value = mock_order

        result = bot.execute_entry('AAPL', 'LONG', 150.0, 'Momentum', 'Strong uptrend')

        assert result['filled'] is True
        assert result['order_id'] == 'order123'
        assert 'AAPL' in bot.open_positions
        assert bot.open_positions['AAPL']['direction'] == 'LONG'

    def test_execute_entry_updates_tracking(self, bot_with_mocks):
        """execute_entry updates price tracking."""
        bot = bot_with_mocks

        mock_order = MagicMock()
        mock_order.status = 'filled'
        mock_order.id = 'order123'
        mock_order.filled_avg_price = 150.0
        mock_order.filled_qty = 100
        bot.broker.submit_order.return_value = mock_order

        bot.execute_entry('AAPL', 'LONG', 150.0, 'Momentum', 'Strong uptrend')

        assert bot.highest_prices['AAPL'] == 150.0
        assert bot.lowest_prices['AAPL'] == 150.0
        assert bot.trailing_stops['AAPL'] == {'activated': False, 'price': 0.0}
        assert 'AAPL' in bot.last_trade_time

    def test_execute_entry_zero_position_size(self, bot_with_mocks):
        """execute_entry handles zero position size."""
        bot = bot_with_mocks
        bot.risk_manager.calculate_position_size = MagicMock(return_value=0)

        result = bot.execute_entry('AAPL', 'LONG', 150.0, 'Momentum', 'Strong uptrend')

        assert result['filled'] is False
        assert 'Position size too small' in result['reason']


class TestExecuteExit:
    """Test exit order execution."""

    @pytest.fixture
    def bot_with_mocks(self, tmp_path):
        """Create a bot with mocked components."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
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
             patch('bot.TradeLogger') as mock_logger, \
             patch('bot.YFinanceDataFetcher'):
            mock_broker_instance = MagicMock()
            mock_broker.return_value = mock_broker_instance
            bot = TradingBot(config_path=str(config_path))
            return bot

    def test_execute_exit_successful(self, bot_with_mocks):
        """execute_exit successfully executes order."""
        bot = bot_with_mocks
        bot.open_positions = {
            'AAPL': {
                'symbol': 'AAPL',
                'qty': 100,
                'entry_price': 150.0,
                'direction': 'LONG',
                'strategy': 'Momentum',
            }
        }

        mock_order = MagicMock()
        mock_order.status = 'filled'
        mock_order.id = 'order456'
        mock_order.filled_avg_price = 155.0
        bot.broker.submit_order.return_value = mock_order

        exit_signal = {'exit': True, 'reason': 'take_profit', 'price': 155.0, 'qty': 100}
        result = bot.execute_exit('AAPL', exit_signal)

        assert result['filled'] is True
        assert result['pnl'] == 500.0  # (155 - 150) * 100
        assert 'AAPL' not in bot.open_positions

    def test_execute_exit_no_position(self, bot_with_mocks):
        """execute_exit handles missing position."""
        bot = bot_with_mocks

        exit_signal = {'exit': True, 'reason': 'take_profit', 'price': 155.0, 'qty': 100}
        result = bot.execute_exit('AAPL', exit_signal)

        assert result['filled'] is False
        assert 'No position' in result['reason']

    def test_execute_exit_cleans_up_tracking(self, bot_with_mocks):
        """execute_exit cleans up price tracking."""
        bot = bot_with_mocks
        bot.open_positions = {
            'AAPL': {
                'symbol': 'AAPL',
                'qty': 100,
                'entry_price': 150.0,
                'direction': 'LONG',
                'strategy': 'Momentum',
            }
        }
        bot.highest_prices = {'AAPL': 155.0}
        bot.lowest_prices = {'AAPL': 148.0}
        bot.trailing_stops = {'AAPL': {'activated': True, 'price': 152.0}}

        mock_order = MagicMock()
        mock_order.status = 'filled'
        mock_order.id = 'order456'
        mock_order.filled_avg_price = 155.0
        bot.broker.submit_order.return_value = mock_order

        exit_signal = {'exit': True, 'reason': 'take_profit', 'price': 155.0, 'qty': 100}
        bot.execute_exit('AAPL', exit_signal)

        assert 'AAPL' not in bot.highest_prices
        assert 'AAPL' not in bot.lowest_prices
        assert 'AAPL' not in bot.trailing_stops


class TestFetchData:
    """Test data fetching."""

    @pytest.fixture
    def bot_with_mocks(self, tmp_path):
        """Create a bot with mocked components."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
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
             patch('bot.YFinanceDataFetcher') as mock_fetcher:
            mock_broker_instance = MagicMock()
            mock_broker.return_value = mock_broker_instance
            mock_fetcher_instance = MagicMock()
            mock_fetcher.return_value = mock_fetcher_instance
            bot = TradingBot(config_path=str(config_path))
            return bot

    def test_fetch_data_returns_dataframe(self, bot_with_mocks):
        """fetch_data returns DataFrame with indicators."""
        bot = bot_with_mocks

        mock_df = pd.DataFrame({
            'open': [100.0] * 50,
            'high': [101.0] * 50,
            'low': [99.0] * 50,
            'close': [100.5] * 50,
            'volume': [1000000] * 50,
        })
        bot.data_fetcher.get_historical_data_range.return_value = mock_df

        result = bot.fetch_data('AAPL', bars=50)

        assert result is not None
        assert len(result) == 50

    def test_fetch_data_handles_empty_response(self, bot_with_mocks):
        """fetch_data handles empty data."""
        bot = bot_with_mocks
        bot.data_fetcher.get_historical_data_range.return_value = None

        result = bot.fetch_data('AAPL')

        assert result is None


class TestTradingCycle:
    """Test trading cycle orchestration."""

    @pytest.fixture
    def bot_with_mocks(self, tmp_path):
        """Create a bot with mocked components."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
risk_management:
  max_open_positions: 2
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

    def test_run_trading_cycle_syncs_state(self, bot_with_mocks):
        """run_trading_cycle syncs account and positions."""
        bot = bot_with_mocks
        bot.sync_account = MagicMock()
        bot.sync_positions = MagicMock()
        bot.fetch_data = MagicMock(return_value=None)

        mock_account = MagicMock()
        mock_account.cash = 100000.0
        mock_account.portfolio_value = 100000.0
        bot.broker.get_account.return_value = mock_account
        bot.broker.get_positions.return_value = []

        bot.run_trading_cycle()

        bot.sync_account.assert_called_once()
        bot.sync_positions.assert_called_once()

    def test_run_trading_cycle_respects_position_limit(self, bot_with_mocks):
        """run_trading_cycle respects max positions limit."""
        bot = bot_with_mocks
        bot.open_positions = {
            'AAPL': {'symbol': 'AAPL'},
            'MSFT': {'symbol': 'MSFT'},
        }  # At max (2)
        bot.sync_account = MagicMock()
        bot.sync_positions = MagicMock()
        bot.fetch_data = MagicMock()
        bot.check_entry = MagicMock()

        mock_account = MagicMock()
        mock_account.cash = 100000.0
        mock_account.portfolio_value = 100000.0
        bot.broker.get_account.return_value = mock_account
        bot.broker.get_positions.return_value = []

        bot.run_trading_cycle()

        # check_entry should not be called because we're at max positions
        # But check_exit would run for existing positions
        bot.check_entry.assert_not_called()


class TestStartStop:
    """Test bot lifecycle."""

    @pytest.fixture
    def bot_with_mocks(self, tmp_path):
        """Create a bot with mocked components."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
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
            return bot

    def test_start_with_watchlist(self, bot_with_mocks):
        """start succeeds with symbols in watchlist."""
        bot = bot_with_mocks

        mock_account = MagicMock()
        mock_account.cash = 100000.0
        mock_account.portfolio_value = 100000.0
        bot.broker.get_account.return_value = mock_account
        bot.broker.get_positions.return_value = []

        result = bot.start()

        assert result is True
        assert bot.running is True

    def test_start_without_watchlist(self, bot_with_mocks):
        """start fails without symbols in watchlist."""
        bot = bot_with_mocks
        bot.watchlist = []

        result = bot.start()

        assert result is False
        assert bot.running is False

    def test_stop_sets_running_false(self, bot_with_mocks):
        """stop sets running to False."""
        bot = bot_with_mocks
        bot.running = True

        bot.stop()

        assert bot.running is False


class TestCleanupPosition:
    """Test position cleanup."""

    @pytest.fixture
    def bot_with_mocks(self, tmp_path):
        """Create a bot with mocked components."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
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
            return bot

    def test_cleanup_position_removes_tracking(self, bot_with_mocks):
        """_cleanup_position removes all tracking state."""
        bot = bot_with_mocks
        bot.highest_prices = {'AAPL': 155.0}
        bot.lowest_prices = {'AAPL': 145.0}
        bot.trailing_stops = {'AAPL': {'activated': True, 'price': 150.0}}
        bot.pending_entries = {'AAPL': {'direction': 'LONG'}}

        bot._cleanup_position('AAPL')

        assert 'AAPL' not in bot.highest_prices
        assert 'AAPL' not in bot.lowest_prices
        assert 'AAPL' not in bot.trailing_stops
        assert 'AAPL' not in bot.pending_entries

    def test_cleanup_position_handles_missing(self, bot_with_mocks):
        """_cleanup_position handles missing keys gracefully."""
        bot = bot_with_mocks

        # Should not raise
        bot._cleanup_position('AAPL')


class TestMain:
    """Test CLI entry point."""

    @patch('bot.argparse.ArgumentParser')
    @patch('bot.TradingBot')
    @patch('bot.time.sleep', side_effect=KeyboardInterrupt)
    def test_main_keyboard_interrupt(self, mock_sleep, mock_bot_class, mock_parser):
        """main handles KeyboardInterrupt gracefully."""
        # Mock argument parser
        mock_args = MagicMock()
        mock_args.config = None
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        # Mock bot
        mock_bot = MagicMock()
        mock_bot.start.return_value = True
        mock_bot.running = True
        mock_bot_class.return_value = mock_bot

        # Should not raise
        main()

        mock_bot.stop.assert_called_once()

    @patch('bot.argparse.ArgumentParser')
    @patch('bot.TradingBot')
    def test_main_start_fails(self, mock_bot_class, mock_parser):
        """main handles start failure gracefully."""
        # Mock argument parser
        mock_args = MagicMock()
        mock_args.config = None
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        # Mock bot
        mock_bot = MagicMock()
        mock_bot.start.return_value = False
        mock_bot_class.return_value = mock_bot

        # Should not raise
        main()

        # Stop should still be called in finally block
        mock_bot.stop.assert_called_once()
