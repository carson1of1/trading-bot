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
        mock_account.last_equity = 55000.0  # Alpaca's official start-of-day value
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
        mock_account.last_equity = 60000.0  # Alpaca's official start-of-day value
        bot.broker.get_account.return_value = mock_account

        bot.sync_account()

        assert bot.peak_value == 60000.0

    def test_sync_account_triggers_kill_switch(self, bot_with_mocks):
        """sync_account triggers kill switch on daily loss limit."""
        bot = bot_with_mocks
        bot.current_trading_day = datetime.now().date()

        mock_account = MagicMock()
        mock_account.cash = 94000.0
        mock_account.portfolio_value = 94000.0  # 6% loss from last_equity
        mock_account.last_equity = 100000.0  # Alpaca's official start-of-day value
        bot.broker.get_account.return_value = mock_account

        bot.sync_account()

        assert bot.kill_switch_triggered is True

    def test_sync_account_resets_on_new_day(self, bot_with_mocks):
        """sync_account resets kill switch on new day and uses last_equity for P&L."""
        bot = bot_with_mocks
        bot.current_trading_day = datetime.now().date() - timedelta(days=1)
        bot.kill_switch_triggered = True

        mock_account = MagicMock()
        mock_account.cash = 100500.0
        mock_account.portfolio_value = 100500.0  # Up $500 from last_equity
        mock_account.last_equity = 100000.0  # Alpaca's official start-of-day value
        bot.broker.get_account.return_value = mock_account

        bot.sync_account()

        assert bot.current_trading_day == datetime.now().date()
        assert bot.daily_pnl == 500.0  # portfolio_value - last_equity
        assert bot.daily_starting_capital == 100000.0  # Uses last_equity
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
        # FIX (Jan 2026): Changed from 'hard_stop' to 'stop_loss' to match backtest.py naming
        assert result['reason'] == 'stop_loss'

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

    def test_check_exit_missing_entry_price(self, bot_with_mocks):
        """check_exit returns None when position is missing entry_price (defensive)."""
        bot = bot_with_mocks

        # Malformed position dict missing entry_price
        position = {
            'qty': 100,
            'direction': 'LONG',
            'entry_time': datetime.now()
        }

        # Should return None, not crash with KeyError
        result = bot.check_exit('AAPL', position, 100.0, bar_high=101.0, bar_low=99.0)

        assert result is None


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
        mock_order.filled_qty = 100  # Full fill
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
        mock_order.filled_qty = 100  # Full fill
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
        mock_account.last_equity = 100000.0  # Alpaca's official start-of-day value
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
        mock_args.symbols = None
        mock_args.candle_delay = 2
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
        mock_args.symbols = None
        mock_args.candle_delay = 2
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


class TestValidateCandleTimestamp:
    """Test incomplete bar detection to ensure backtest/live alignment."""

    def test_incomplete_bar_rejected(self):
        """Bars still forming should be rejected."""
        from bot import validate_candle_timestamp
        import pytz

        eastern = pytz.timezone('America/New_York')
        # Simulate: it's 10:29 EST and we see a 9:30 bar (completes at 10:30)
        now = datetime(2026, 1, 5, 10, 29, 0, tzinfo=eastern)

        # Bar starts at 9:30, so it's incomplete until 10:30
        bar_time = datetime(2026, 1, 5, 9, 30, 0, tzinfo=eastern)
        data = pd.DataFrame({
            'timestamp': [bar_time],
            'close': [100.0]
        })

        with patch('bot.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            result = validate_candle_timestamp(data, expected_hour=9)

        # Should be rejected - bar not complete yet
        assert result is False

    def test_complete_bar_accepted(self):
        """Completed bars should be accepted."""
        from bot import validate_candle_timestamp
        import pytz

        eastern = pytz.timezone('America/New_York')
        # Simulate: it's 10:32 EST and we see a 9:30 bar (completed at 10:30)
        now = datetime(2026, 1, 5, 10, 32, 0, tzinfo=eastern)

        # Bar starts at 9:30, completed at 10:30 - we're at 10:32, so it's complete
        bar_time = datetime(2026, 1, 5, 9, 30, 0, tzinfo=eastern)
        data = pd.DataFrame({
            'timestamp': [bar_time],
            'close': [100.0]
        })

        with patch('bot.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            result = validate_candle_timestamp(data, expected_hour=9)

        # Should be accepted - bar is complete
        assert result is True

    def test_market_open_930_bar_expected_hour_9(self):
        """9:30 bar should be accepted when expecting hour 9 (market open case)."""
        from bot import validate_candle_timestamp
        import pytz

        eastern = pytz.timezone('America/New_York')
        # It's 10:32 EST (after 9:30 bar completes)
        now = datetime(2026, 1, 5, 10, 32, 0, tzinfo=eastern)

        # First bar of day is 9:30 (market opens at 9:30)
        bar_time = datetime(2026, 1, 5, 9, 30, 0, tzinfo=eastern)
        data = pd.DataFrame({
            'timestamp': [bar_time],
            'close': [100.0]
        })

        with patch('bot.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            # Expected hour 9 should match 9:30 bar (market open special case)
            result = validate_candle_timestamp(data, expected_hour=9)

        assert result is True

    def test_normal_hourly_bar_accepted(self):
        """Normal hourly bars (10:00, 11:00, etc.) should be accepted when complete."""
        from bot import validate_candle_timestamp
        import pytz

        eastern = pytz.timezone('America/New_York')
        # It's 11:02 EST (after 10:00 bar completes at 11:00)
        now = eastern.localize(datetime(2026, 1, 5, 11, 2, 0))

        # 10:00 bar completes at 11:00
        bar_time = eastern.localize(datetime(2026, 1, 5, 10, 0, 0))
        data = pd.DataFrame({
            'timestamp': [bar_time],
            'close': [100.0]
        })

        with patch('bot.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            # Allow timedelta to work normally
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            result = validate_candle_timestamp(data, expected_hour=10)

        assert result is True


class TestExitLoopIsolation:
    """Test that exit loop continues when one position throws an exception."""

    @pytest.fixture
    def bot_with_mocks(self, tmp_path):
        """Create a bot with mocked components."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
risk_management:
  max_open_positions: 5
  max_position_dollars: 100000
  max_position_size_pct: 100.0
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
            return bot

    def test_exit_loop_continues_after_exception(self, bot_with_mocks):
        """One bad position should not crash exit checks for other positions.

        Bug: ODE-86 - One bad position crashes entire exit loop, preventing
        stop losses from executing on ALL positions.
        """
        bot = bot_with_mocks

        # Set up 3 positions
        positions = {
            'AAPL': {
                'symbol': 'AAPL',
                'qty': 100,
                'entry_price': 150.0,
                'direction': 'LONG',
                'entry_time': datetime.now(),
            },
            'MSFT': {
                'symbol': 'MSFT',
                'qty': 50,
                'entry_price': 300.0,
                'direction': 'LONG',
                'entry_time': datetime.now(),
            },
            'GOOGL': {
                'symbol': 'GOOGL',
                'qty': 25,
                'entry_price': 140.0,
                'direction': 'LONG',
                'entry_time': datetime.now(),
            },
        }
        bot.open_positions = positions

        # Mock account sync
        mock_account = MagicMock()
        mock_account.cash = 100000.0
        mock_account.equity = 100000.0  # ODE-90: Required for drawdown guard
        mock_account.portfolio_value = 100000.0
        mock_account.last_equity = 100000.0
        bot.broker.get_account.return_value = mock_account
        bot.broker.get_positions.return_value = []

        # Mock sync_positions to preserve our test positions
        def mock_sync_positions():
            bot.open_positions = positions
        bot.sync_positions = mock_sync_positions

        # Track which positions had exit checks called
        exit_checks_called = []

        # Mock fetch_data to throw exception for MSFT, return valid data for others
        def mock_fetch_data(symbol, bars=200):
            if symbol == 'MSFT':
                raise Exception("Simulated API error for MSFT")
            # Return valid data for other symbols
            return pd.DataFrame({
                'open': [100.0] * 100,
                'high': [101.0] * 100,
                'low': [99.0] * 100,
                'close': [100.5] * 100,
                'volume': [1000000] * 100,
                'timestamp': [datetime.now()] * 100,
            })

        # Mock check_exit to track calls
        def mock_check_exit(symbol, position, current_price, bar_high=None, bar_low=None, data=None):
            exit_checks_called.append(symbol)
            return None  # No exit signal

        bot.fetch_data = mock_fetch_data
        bot.check_exit = mock_check_exit

        # Run trading cycle
        bot.run_trading_cycle()

        # AAPL and GOOGL should have their exit checks called despite MSFT throwing
        assert 'AAPL' in exit_checks_called, "AAPL exit check should have been called"
        assert 'GOOGL' in exit_checks_called, "GOOGL exit check should have been called"
        # MSFT should NOT be in exit_checks_called because fetch_data threw
        assert 'MSFT' not in exit_checks_called, "MSFT exit check should not have been called (data fetch failed)"

    def test_exit_loop_logs_error_on_exception(self, bot_with_mocks):
        """Exception during exit check should be logged with full traceback."""
        bot = bot_with_mocks

        positions = {
            'AAPL': {
                'symbol': 'AAPL',
                'qty': 100,
                'entry_price': 150.0,
                'direction': 'LONG',
                'entry_time': datetime.now(),
            },
        }
        bot.open_positions = positions

        mock_account = MagicMock()
        mock_account.cash = 100000.0
        mock_account.equity = 100000.0  # ODE-90: Required for drawdown guard
        mock_account.portfolio_value = 100000.0
        mock_account.last_equity = 100000.0
        bot.broker.get_account.return_value = mock_account
        bot.broker.get_positions.return_value = []

        # Mock sync_positions to preserve our test positions
        def mock_sync_positions():
            bot.open_positions = positions
        bot.sync_positions = mock_sync_positions

        # Mock fetch_data to throw exception
        bot.fetch_data = MagicMock(side_effect=Exception("Simulated error"))

        # Run trading cycle and capture logs
        with patch('bot.logger') as mock_logger:
            bot.run_trading_cycle()

            # Should have logged an error with exc_info=True
            error_calls = [call for call in mock_logger.error.call_args_list
                          if 'AAPL' in str(call) and 'FAILED' in str(call)]
            assert len(error_calls) > 0, "Should log error for failed position exit check"


class TestEmergencyPositionLimitCheck:
    """Tests for _emergency_position_limit_check() method (ODE-88)."""

    def test_no_violation_returns_false(self):
        """When position count <= max, returns False and takes no action."""
        bot = TradingBot.__new__(TradingBot)
        bot.config = {'risk_management': {'max_open_positions': 5}}
        bot.open_positions = {
            'AAPL': {'qty': 10, 'direction': 'LONG', 'entry_time': datetime.now()},
            'MSFT': {'qty': 10, 'direction': 'LONG', 'entry_time': datetime.now()},
            'GOOG': {'qty': 10, 'direction': 'LONG', 'entry_time': datetime.now()},
        }
        bot.kill_switch_triggered = False

        result = bot._emergency_position_limit_check()

        assert result is False
        assert bot.kill_switch_triggered is False
        assert len(bot.open_positions) == 3  # No positions liquidated

    @patch('bot.TradingBot._cleanup_position')
    def test_violation_liquidates_oldest_positions(self, mock_cleanup):
        """When position count > max, liquidates oldest excess positions."""
        bot = TradingBot.__new__(TradingBot)
        bot.config = {'risk_management': {'max_open_positions': 2}}
        bot.kill_switch_triggered = False
        bot.use_tiered_exits = False
        bot.exit_manager = None

        # Create mock broker
        mock_broker = MagicMock()
        mock_order = MagicMock()
        mock_order.filled_avg_price = 150.0
        mock_order.id = 'order123'
        mock_broker.submit_order.return_value = mock_order
        bot.broker = mock_broker

        # Create mock trade logger
        mock_trade_logger = MagicMock()
        bot.trade_logger = mock_trade_logger

        # 4 positions, max 2 = 2 excess (liquidate oldest 2)
        base_time = datetime(2026, 1, 6, 10, 0, 0)
        bot.open_positions = {
            'OLD1': {'qty': 10, 'direction': 'LONG', 'entry_price': 100.0,
                     'entry_time': base_time, 'strategy': 'Test'},  # Oldest - liquidate
            'OLD2': {'qty': 20, 'direction': 'SHORT', 'entry_price': 200.0,
                     'entry_time': base_time + timedelta(hours=1), 'strategy': 'Test'},  # 2nd oldest - liquidate
            'NEW1': {'qty': 15, 'direction': 'LONG', 'entry_price': 150.0,
                     'entry_time': base_time + timedelta(hours=2), 'strategy': 'Test'},  # Keep
            'NEW2': {'qty': 25, 'direction': 'LONG', 'entry_price': 250.0,
                     'entry_time': base_time + timedelta(hours=3), 'strategy': 'Test'},  # Keep
        }

        result = bot._emergency_position_limit_check()

        assert result is True
        assert bot.kill_switch_triggered is True
        assert len(bot.open_positions) == 2
        assert 'OLD1' not in bot.open_positions
        assert 'OLD2' not in bot.open_positions
        assert 'NEW1' in bot.open_positions
        assert 'NEW2' in bot.open_positions

        # Verify broker calls - OLD1 is LONG (sell), OLD2 is SHORT (buy)
        calls = mock_broker.submit_order.call_args_list
        assert len(calls) == 2

    @patch('bot.TradingBot._cleanup_position')
    def test_correct_sides_for_long_and_short(self, mock_cleanup):
        """LONG positions sell to close, SHORT positions buy to close."""
        bot = TradingBot.__new__(TradingBot)
        bot.config = {'risk_management': {'max_open_positions': 1}}
        bot.kill_switch_triggered = False
        bot.use_tiered_exits = False
        bot.exit_manager = None

        mock_broker = MagicMock()
        mock_order = MagicMock()
        mock_order.filled_avg_price = 100.0
        mock_broker.submit_order.return_value = mock_order
        bot.broker = mock_broker
        bot.trade_logger = MagicMock()

        base_time = datetime(2026, 1, 6, 10, 0, 0)
        bot.open_positions = {
            'LONG_POS': {'qty': 10, 'direction': 'LONG', 'entry_price': 100.0,
                         'entry_time': base_time, 'strategy': 'Test'},
            'SHORT_POS': {'qty': 20, 'direction': 'SHORT', 'entry_price': 200.0,
                          'entry_time': base_time + timedelta(hours=1), 'strategy': 'Test'},
            'KEEP': {'qty': 5, 'direction': 'LONG', 'entry_price': 50.0,
                     'entry_time': base_time + timedelta(hours=2), 'strategy': 'Test'},
        }

        bot._emergency_position_limit_check()

        calls = mock_broker.submit_order.call_args_list

        # First call: LONG_POS (oldest) - should sell
        assert calls[0][1]['symbol'] == 'LONG_POS'
        assert calls[0][1]['side'] == 'sell'
        assert calls[0][1]['qty'] == 10

        # Second call: SHORT_POS (2nd oldest) - should buy
        assert calls[1][1]['symbol'] == 'SHORT_POS'
        assert calls[1][1]['side'] == 'buy'
        assert calls[1][1]['qty'] == 20
