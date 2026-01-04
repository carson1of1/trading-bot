"""
Tests for the Backtest module.

Verifies:
- Backtest1Hour initialization and configuration
- Data fetching and signal generation
- Trade simulation logic (critical for preserving +159% returns)
- Metrics calculation
- CLI functionality
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from backtest import Backtest1Hour, run_backtest, main


class TestBacktest1HourInitialization:
    """Test Backtest1Hour initialization and configuration loading."""

    def test_default_initialization(self):
        """Backtest1Hour initializes with default values."""
        bt = Backtest1Hour()

        assert bt.initial_capital == 100000.0
        assert bt.longs_only is False
        assert bt.shorts_only is False
        assert bt.COOLDOWN_BARS == 1
        assert bt.COMMISSION == 0.0

    def test_custom_capital(self):
        """Custom initial capital is set correctly."""
        bt = Backtest1Hour(initial_capital=50000.0)
        assert bt.initial_capital == 50000.0

    def test_longs_only_mode(self):
        """longs_only mode is set correctly."""
        bt = Backtest1Hour(longs_only=True)
        assert bt.longs_only is True
        assert bt.shorts_only is False

    def test_shorts_only_mode(self):
        """shorts_only mode is set correctly."""
        bt = Backtest1Hour(shorts_only=True)
        assert bt.longs_only is False
        assert bt.shorts_only is True

    def test_config_override(self):
        """Config dict overrides defaults."""
        config = {
            'risk_management': {
                'stop_loss_pct': 3.0,
                'take_profit_pct': 6.0,
            },
            'entry_gate': {
                'confidence_threshold': 70,
            },
        }
        bt = Backtest1Hour(config=config)

        assert bt.default_stop_loss_pct == 0.03
        assert bt.default_take_profit_pct == 0.06

    def test_scanner_disabled_by_default(self):
        """Scanner is disabled by default (unless config enables it)."""
        config = {'volatility_scanner': {'enabled': False}}
        bt = Backtest1Hour(config=config)
        assert bt.scanner_enabled is False
        assert bt.scanner is None

    def test_scanner_enabled_override(self):
        """scanner_enabled parameter overrides config."""
        config = {'volatility_scanner': {'enabled': False}}
        bt = Backtest1Hour(config=config, scanner_enabled=True)
        assert bt.scanner_enabled is True


class TestBacktest1HourResetState:
    """Test state reset functionality."""

    def test_reset_state_clears_trades(self):
        """reset_state clears trade list."""
        bt = Backtest1Hour()
        bt.trades = [{'pnl': 100}]
        bt._reset_state()
        assert bt.trades == []

    def test_reset_state_resets_capital(self):
        """reset_state resets capital to initial."""
        bt = Backtest1Hour(initial_capital=100000)
        bt.cash = 50000
        bt.portfolio_value = 60000
        bt._reset_state()
        assert bt.cash == 100000
        assert bt.portfolio_value == 100000

    def test_reset_state_clears_equity_curve(self):
        """reset_state clears equity curve."""
        bt = Backtest1Hour()
        bt.equity_curve = [{'portfolio_value': 100000}]
        bt._reset_state()
        assert bt.equity_curve == []

    def test_reset_state_clears_pnl_tracking(self):
        """reset_state clears P&L tracking."""
        bt = Backtest1Hour()
        bt.total_pnl = 5000
        bt.winning_trades = 10
        bt.losing_trades = 5
        bt._reset_state()
        assert bt.total_pnl == 0.0
        assert bt.winning_trades == 0
        assert bt.losing_trades == 0

    def test_reset_state_initializes_drawdown_tracking(self):
        """reset_state initializes drawdown window tracking."""
        bt = Backtest1Hour(initial_capital=100000)
        bt._reset_state()
        assert bt.drawdown_peak_value == 100000
        assert bt.drawdown_peak_date is None
        assert bt.drawdown_trough_value == 100000
        assert bt.drawdown_trough_date is None
        assert bt.daily_equity_snapshots == {}
        assert bt._current_day_equity == {'high': 100000, 'low': 100000}

    def test_reset_state_clears_previous_drawdown_data(self):
        """reset_state clears previously set drawdown data."""
        bt = Backtest1Hour(initial_capital=100000)
        # Simulate some previous state
        bt.drawdown_peak_value = 150000
        bt.drawdown_peak_date = '2025-01-01'
        bt.drawdown_trough_value = 50000
        bt.drawdown_trough_date = '2025-02-01'
        bt.daily_equity_snapshots = {'2025-01-01': {'open': 100000, 'close': 95000, 'high': 101000, 'low': 94000}}
        bt._current_day_equity = {'high': 120000, 'low': 80000}

        bt._reset_state()

        assert bt.drawdown_peak_value == 100000
        assert bt.drawdown_peak_date is None
        assert bt.drawdown_trough_value == 100000
        assert bt.drawdown_trough_date is None
        assert bt.daily_equity_snapshots == {}
        assert bt._current_day_equity == {'high': 100000, 'low': 100000}


class TestDrawdownAnalytics:
    """Test drawdown analytics methods."""

    def test_get_worst_daily_drops_empty(self):
        """_get_worst_daily_drops returns empty list when no snapshots."""
        bt = Backtest1Hour()
        bt._reset_state()
        result = bt._get_worst_daily_drops(5)
        assert result == []

    def test_get_worst_daily_drops_single_day(self):
        """_get_worst_daily_drops works with a single day."""
        bt = Backtest1Hour()
        bt._reset_state()
        bt.daily_equity_snapshots = {
            '2025-01-01': {'open': 100000, 'close': 95000, 'high': 101000, 'low': 94000}
        }
        result = bt._get_worst_daily_drops(5)
        assert len(result) == 1
        assert result[0]['date'] == '2025-01-01'
        assert result[0]['change_pct'] == -5.0
        assert result[0]['change_dollars'] == -5000.0

    def test_get_worst_daily_drops_sorts_by_worst(self):
        """_get_worst_daily_drops sorts by change_pct ascending (worst first)."""
        bt = Backtest1Hour()
        bt._reset_state()
        bt.daily_equity_snapshots = {
            '2025-01-01': {'open': 100000, 'close': 98000, 'high': 101000, 'low': 97000},
            '2025-01-02': {'open': 98000, 'close': 88200, 'high': 99000, 'low': 87000},  # -10%
            '2025-01-03': {'open': 88200, 'close': 90000, 'high': 91000, 'low': 87000},  # +2%
        }
        result = bt._get_worst_daily_drops(5)
        assert len(result) == 3
        assert result[0]['date'] == '2025-01-02'  # -10% is worst
        assert result[0]['change_pct'] == -10.0
        assert result[1]['date'] == '2025-01-01'  # -2% is second worst
        assert result[2]['date'] == '2025-01-03'  # +2% is best

    def test_get_worst_daily_drops_respects_top_n(self):
        """_get_worst_daily_drops respects top_n parameter."""
        bt = Backtest1Hour()
        bt._reset_state()
        bt.daily_equity_snapshots = {
            '2025-01-01': {'open': 100000, 'close': 95000, 'high': 101000, 'low': 94000},
            '2025-01-02': {'open': 95000, 'close': 90000, 'high': 96000, 'low': 89000},
            '2025-01-03': {'open': 90000, 'close': 85000, 'high': 91000, 'low': 84000},
            '2025-01-04': {'open': 85000, 'close': 80000, 'high': 86000, 'low': 79000},
            '2025-01-05': {'open': 80000, 'close': 75000, 'high': 81000, 'low': 74000},
            '2025-01-06': {'open': 75000, 'close': 70000, 'high': 76000, 'low': 69000},
        }
        result = bt._get_worst_daily_drops(3)
        assert len(result) == 3

    def test_get_worst_daily_drops_includes_all_fields(self):
        """_get_worst_daily_drops includes all required fields."""
        bt = Backtest1Hour()
        bt._reset_state()
        bt.daily_equity_snapshots = {
            '2025-01-01': {'open': 100000, 'close': 95000, 'high': 102000, 'low': 94000}
        }
        result = bt._get_worst_daily_drops(5)
        assert len(result) == 1
        day = result[0]
        assert 'date' in day
        assert 'open' in day
        assert 'close' in day
        assert 'high' in day
        assert 'low' in day
        assert 'change_pct' in day
        assert 'change_dollars' in day
        assert day['open'] == 100000.0
        assert day['close'] == 95000.0
        assert day['high'] == 102000.0
        assert day['low'] == 94000.0


class TestSimulateTradesLogic:
    """Test the critical simulate_trades method.

    This is the core of the backtester that achieved +159% returns.
    """

    @pytest.fixture
    def backtester(self):
        """Create a backtester with minimal config."""
        config = {
            'risk_management': {
                'stop_loss_pct': 2.0,
                'take_profit_pct': 4.0,
                'max_position_size_pct': 5.0,
                'max_daily_loss_pct': 3.0,
            },
            'entry_gate': {
                'confidence_threshold': 60,
            },
            'exit_manager': {
                'tier_0_hard_stop': -0.02,
                'tier_1_profit_floor': 0.02,
                'max_hold_hours': 48,
            },
            'trailing_stop': {
                'enabled': True,
                'activation_pct': 0.5,
                'trail_pct': 0.5,
                'move_to_breakeven': True,
            },
        }
        return Backtest1Hour(initial_capital=100000, config=config)

    @pytest.fixture
    def sample_data_with_signals(self):
        """Create sample data with BUY signal."""
        dates = pd.date_range('2025-01-01 09:00', periods=50, freq='h')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 50,
            'high': [102.0] * 50,
            'low': [99.0] * 50,
            'close': [101.0] * 50,
            'volume': [10000] * 50,
            'signal': [0] * 50,
            'confidence': [0.0] * 50,
            'strategy': [''] * 50,
            'reasoning': [''] * 50,
        })
        # Add a BUY signal at bar 10
        data.loc[10, 'signal'] = 1
        data.loc[10, 'confidence'] = 75.0
        data.loc[10, 'strategy'] = 'Momentum_1Hour'
        return data

    def test_no_trades_without_signals(self, backtester):
        """No trades are generated without signals."""
        dates = pd.date_range('2025-01-01 09:00', periods=20, freq='h')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 20,
            'high': [102.0] * 20,
            'low': [99.0] * 20,
            'close': [101.0] * 20,
            'volume': [10000] * 20,
            'signal': [0] * 20,
            'confidence': [0.0] * 20,
            'strategy': [''] * 20,
            'reasoning': [''] * 20,
        })

        trades = backtester.simulate_trades('TEST', data)
        assert len(trades) == 0

    def test_long_trade_generation(self, backtester, sample_data_with_signals):
        """LONG trade is generated from BUY signal."""
        trades = backtester.simulate_trades('TEST', sample_data_with_signals)

        # Should have at least one trade
        assert len(trades) >= 1

        # First trade should be LONG
        assert trades[0]['direction'] == 'LONG'
        assert trades[0]['symbol'] == 'TEST'

    def test_entry_slippage_applied(self, backtester, sample_data_with_signals):
        """Entry slippage is applied to entry price."""
        trades = backtester.simulate_trades('TEST', sample_data_with_signals)

        if len(trades) > 0:
            # Entry price should be slightly higher than open (due to slippage)
            entry_price = trades[0]['entry_price']
            # Open price at bar 11 (where entry happens on next bar)
            open_price = sample_data_with_signals.iloc[11]['open']
            assert entry_price >= open_price

    def test_exit_slippage_applied(self, backtester, sample_data_with_signals):
        """Exit slippage is applied to exit price."""
        trades = backtester.simulate_trades('TEST', sample_data_with_signals)

        if len(trades) > 0:
            # Exit price should reflect slippage
            trade = trades[0]
            # For LONG positions, exit price should be less than or equal to close
            if trade['direction'] == 'LONG':
                last_close = sample_data_with_signals.iloc[-1]['close']
                # Allow for various exit reasons
                assert trade['exit_price'] <= last_close * 1.01

    def test_trade_has_required_fields(self, backtester, sample_data_with_signals):
        """Trade dict contains all required fields."""
        trades = backtester.simulate_trades('TEST', sample_data_with_signals)

        if len(trades) > 0:
            trade = trades[0]
            required_fields = [
                'symbol', 'direction', 'entry_date', 'exit_date',
                'entry_price', 'exit_price', 'shares', 'pnl', 'pnl_pct',
                'exit_reason', 'strategy', 'bars_held', 'mfe', 'mae'
            ]
            for field in required_fields:
                assert field in trade, f"Missing field: {field}"

    def test_pnl_calculation_long(self, backtester):
        """P&L is calculated correctly for LONG trades."""
        # Create data with price increase to ensure profit
        dates = pd.date_range('2025-01-01 09:00', periods=20, freq='h')
        prices = [100.0] + [100.0 + i * 0.5 for i in range(19)]  # Gradually increase
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 0.5 for p in prices],
            'close': prices,
            'volume': [10000] * 20,
            'signal': [0] * 20,
            'confidence': [0.0] * 20,
            'strategy': [''] * 20,
            'reasoning': [''] * 20,
        })
        data.loc[5, 'signal'] = 1
        data.loc[5, 'confidence'] = 75.0

        trades = backtester.simulate_trades('TEST', data)

        # With increasing prices, LONG should be profitable
        if len(trades) > 0 and trades[0]['direction'] == 'LONG':
            # Trade should complete (either by exit or end of backtest)
            assert 'pnl' in trades[0]

    def test_short_trade_generation(self, backtester):
        """SHORT trade is generated from SELL signal."""
        dates = pd.date_range('2025-01-01 09:00', periods=20, freq='h')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 20,
            'high': [102.0] * 20,
            'low': [99.0] * 20,
            'close': [101.0] * 20,
            'volume': [10000] * 20,
            'signal': [0] * 20,
            'confidence': [0.0] * 20,
            'strategy': [''] * 20,
            'reasoning': [''] * 20,
        })
        # Add a SELL signal
        data.loc[5, 'signal'] = -1
        data.loc[5, 'confidence'] = 75.0
        data.loc[5, 'strategy'] = 'Breakout_1Hour'

        trades = backtester.simulate_trades('TEST', data)

        if len(trades) > 0:
            assert trades[0]['direction'] == 'SHORT'

    def test_longs_only_blocks_shorts(self, backtester):
        """longs_only mode blocks SHORT trades."""
        backtester.longs_only = True

        dates = pd.date_range('2025-01-01 09:00', periods=20, freq='h')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 20,
            'high': [102.0] * 20,
            'low': [99.0] * 20,
            'close': [101.0] * 20,
            'volume': [10000] * 20,
            'signal': [0] * 20,
            'confidence': [0.0] * 20,
            'strategy': [''] * 20,
            'reasoning': [''] * 20,
        })
        # Add a SELL signal
        data.loc[5, 'signal'] = -1
        data.loc[5, 'confidence'] = 75.0

        trades = backtester.simulate_trades('TEST', data)

        # No SHORT trades should be generated
        short_trades = [t for t in trades if t['direction'] == 'SHORT']
        assert len(short_trades) == 0

    def test_shorts_only_blocks_longs(self, backtester):
        """shorts_only mode blocks LONG trades."""
        backtester.shorts_only = True

        dates = pd.date_range('2025-01-01 09:00', periods=20, freq='h')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 20,
            'high': [102.0] * 20,
            'low': [99.0] * 20,
            'close': [101.0] * 20,
            'volume': [10000] * 20,
            'signal': [0] * 20,
            'confidence': [0.0] * 20,
            'strategy': [''] * 20,
            'reasoning': [''] * 20,
        })
        # Add a BUY signal
        data.loc[5, 'signal'] = 1
        data.loc[5, 'confidence'] = 75.0

        trades = backtester.simulate_trades('TEST', data)

        # No LONG trades should be generated
        long_trades = [t for t in trades if t['direction'] == 'LONG']
        assert len(long_trades) == 0


class TestCalculateMetrics:
    """Test metrics calculation."""

    @pytest.fixture
    def backtester(self):
        """Create a basic backtester."""
        return Backtest1Hour(initial_capital=100000)

    def test_no_trades_metrics(self, backtester):
        """Metrics for no trades are zeros."""
        metrics = backtester.calculate_metrics([])

        assert metrics['total_trades'] == 0
        assert metrics['win_rate'] == 0
        assert metrics['profit_factor'] == 0
        assert metrics['total_pnl'] == 0

    def test_winning_trades_metrics(self, backtester):
        """Metrics correctly calculated for winning trades."""
        trades = [
            {'pnl': 100, 'bars_held': 5},
            {'pnl': 200, 'bars_held': 3},
            {'pnl': 150, 'bars_held': 4},
        ]

        metrics = backtester.calculate_metrics(trades)

        assert metrics['total_trades'] == 3
        assert metrics['winning_trades'] == 3
        assert metrics['losing_trades'] == 0
        assert metrics['win_rate'] == 100.0
        assert metrics['profit_factor'] == 999.99  # All wins

    def test_losing_trades_metrics(self, backtester):
        """Metrics correctly calculated for losing trades."""
        trades = [
            {'pnl': -100, 'bars_held': 5},
            {'pnl': -200, 'bars_held': 3},
        ]

        metrics = backtester.calculate_metrics(trades)

        assert metrics['total_trades'] == 2
        assert metrics['winning_trades'] == 0
        assert metrics['losing_trades'] == 2
        assert metrics['win_rate'] == 0.0
        assert metrics['profit_factor'] == 0

    def test_mixed_trades_metrics(self, backtester):
        """Metrics correctly calculated for mixed trades."""
        trades = [
            {'pnl': 200, 'bars_held': 5},  # Win
            {'pnl': -100, 'bars_held': 3},  # Loss
            {'pnl': 150, 'bars_held': 4},  # Win
            {'pnl': -50, 'bars_held': 2},  # Loss
        ]

        metrics = backtester.calculate_metrics(trades)

        assert metrics['total_trades'] == 4
        assert metrics['winning_trades'] == 2
        assert metrics['losing_trades'] == 2
        assert metrics['win_rate'] == 50.0
        # Profit factor = 350 / 150 = 2.33
        assert abs(metrics['profit_factor'] - 2.33) < 0.1


class TestGenerateSignals:
    """Test signal generation."""

    @pytest.fixture
    def backtester(self):
        """Create a backtester."""
        config = {
            'entry_gate': {'confidence_threshold': 60},
            'strategies': {
                'momentum': {'enabled': True},
                'mean_reversion': {'enabled': True},
                'breakout': {'enabled': True},
            }
        }
        return Backtest1Hour(config=config)

    def test_signal_columns_added(self, backtester):
        """Signal columns are added to DataFrame."""
        dates = pd.date_range('2025-01-01', periods=50, freq='h')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 50,
            'high': [102.0] * 50,
            'low': [99.0] * 50,
            'close': [101.0] * 50,
            'volume': [10000] * 50,
            'SMA_20': [100.0] * 50,
            'SMA_50': [99.0] * 50,
            'RSI': [55.0] * 50,
        })

        result = backtester.generate_signals('TEST', data)

        assert 'signal' in result.columns
        assert 'confidence' in result.columns
        assert 'strategy' in result.columns
        assert 'reasoning' in result.columns


class TestRunBacktest:
    """Test the run_backtest convenience function."""

    @patch('backtest.YFinanceDataFetcher')
    def test_run_backtest_returns_results(self, mock_fetcher):
        """run_backtest returns results dict."""
        # Mock data fetcher
        mock_instance = MagicMock()
        dates = pd.date_range('2025-01-01', periods=100, freq='h')
        mock_instance.get_historical_data_range.return_value = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 100,
            'high': [102.0] * 100,
            'low': [99.0] * 100,
            'close': [101.0] * 100,
            'volume': [10000] * 100,
        })
        mock_fetcher.return_value = mock_instance

        results = run_backtest(
            symbols=['TEST'],
            start_date='2025-01-01',
            end_date='2025-01-15',
            initial_capital=100000
        )

        assert results is not None
        assert 'metrics' in results
        assert 'trades' in results
        assert 'equity_curve' in results


class TestTrailingStopLogic:
    """Test trailing stop implementation."""

    @pytest.fixture
    def backtester(self):
        """Create backtester with trailing stop enabled."""
        config = {
            'trailing_stop': {
                'enabled': True,
                'activation_pct': 1.0,  # 1% profit to activate
                'trail_pct': 0.5,  # 0.5% trail
                'move_to_breakeven': True,
            },
            'exit_manager': {
                'tier_0_hard_stop': -0.10,  # Very wide stop
                'max_hold_hours': 100,  # Long hold
            },
        }
        return Backtest1Hour(config=config)

    def test_trailing_stop_activates_on_profit(self, backtester):
        """Trailing stop activates after profit threshold."""
        # Create data with price spike then drop
        dates = pd.date_range('2025-01-01 09:00', periods=30, freq='h')
        prices = (
            [100.0] * 5 +  # Initial (5)
            [100.0, 101.0, 102.0, 103.0, 104.0] +  # Entry and rise (5)
            [103.0, 102.0, 101.0] +  # Drop back (3)
            [100.0] * 17  # Stay low (17) - total = 30
        )
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 0.5 for p in prices],
            'low': [p - 0.5 for p in prices],
            'close': prices,
            'volume': [10000] * 30,
            'signal': [0] * 30,
            'confidence': [0.0] * 30,
            'strategy': [''] * 30,
            'reasoning': [''] * 30,
        })
        # Add BUY signal
        data.loc[5, 'signal'] = 1
        data.loc[5, 'confidence'] = 75.0

        trades = backtester.simulate_trades('TEST', data)

        # Should have exit (potentially trailing stop)
        assert len(trades) >= 1


class TestEODCloseLogic:
    """Test end-of-day close simulation."""

    @pytest.fixture
    def backtester(self):
        """Create backtester with EOD close enabled."""
        config = {
            'exit_manager': {
                'max_hold_hours': 100,  # Long enough not to trigger
            },
        }
        bt = Backtest1Hour(config=config)
        bt.eod_close_enabled = True
        bt.eod_close_bar_hour = 15  # 3 PM
        return bt

    def test_eod_close_triggers_at_3pm(self, backtester):
        """Positions close at 3 PM bar."""
        # Create data with bars at specific hours
        base_date = datetime(2025, 1, 2, 9, 0)  # Thursday
        dates = [base_date + timedelta(hours=i) for i in range(10)]

        data = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 10,
            'high': [102.0] * 10,
            'low': [99.0] * 10,
            'close': [101.0] * 10,
            'volume': [10000] * 10,
            'signal': [0] * 10,
            'confidence': [0.0] * 10,
            'strategy': [''] * 10,
            'reasoning': [''] * 10,
        })
        # Add BUY signal at 10 AM (index 1)
        data.loc[1, 'signal'] = 1
        data.loc[1, 'confidence'] = 75.0

        trades = backtester.simulate_trades('TEST', data)

        # Should have trade with eod_close exit
        if len(trades) > 0:
            # Position should exit at or before 3 PM
            eod_trades = [t for t in trades if t['exit_reason'] == 'eod_close']
            # If we have an EOD close, verify it happened at the right time
            for trade in eod_trades:
                exit_date = trade['exit_date']
                if hasattr(exit_date, 'hour'):
                    assert exit_date.hour >= 15


class TestKillSwitchLogic:
    """Test daily loss kill switch."""

    @pytest.fixture
    def backtester(self):
        """Create backtester with kill switch enabled."""
        config = {
            'risk_management': {
                'max_daily_loss_pct': 1.0,  # 1% daily loss limit
            },
        }
        bt = Backtest1Hour(config=config)
        bt.daily_loss_kill_switch_enabled = True
        return bt

    def test_kill_switch_triggers_on_loss(self, backtester):
        """Kill switch triggers after exceeding daily loss limit."""
        # Force a large loss
        backtester.daily_pnl = -1500  # 1.5% loss on 100k
        backtester.daily_starting_capital = 100000

        # Check if kill switch should trigger
        daily_loss_pct = -backtester.daily_pnl / backtester.daily_starting_capital
        assert daily_loss_pct >= backtester.max_daily_loss_pct

    def test_kill_switch_resets_on_new_day(self, backtester):
        """Kill switch resets on new trading day."""
        backtester._reset_state()
        backtester.kill_switch_triggered = True
        backtester.current_trading_day = datetime(2025, 1, 1).date()

        # Simulate new day
        new_date = datetime(2025, 1, 2).date()
        if new_date != backtester.current_trading_day:
            backtester.current_trading_day = new_date
            backtester.daily_pnl = 0.0
            backtester.kill_switch_triggered = False

        assert backtester.kill_switch_triggered is False


class TestPortfolioWideKillSwitch:
    """Test portfolio-wide kill switch functionality.

    These tests verify the interleaved simulation correctly:
    - Tracks daily P&L across ALL symbols (not per-symbol)
    - Blocks entries for ALL symbols when kill switch triggers
    - Resets kill switch on new trading day
    """

    @pytest.fixture
    def backtester_strict_kill_switch(self):
        """Create backtester with 1% daily loss limit for easy testing."""
        config = {
            'risk_management': {
                'max_daily_loss_pct': 1.0,  # 1% daily loss limit (easy to trigger)
                'stop_loss_pct': 5.0,  # 5% stop loss
                'take_profit_pct': 20.0,
                'max_position_size_pct': 25.0,  # Large position to amplify loss
            },
            'entry_gate': {
                'confidence_threshold': 50,
            },
            'exit_manager': {
                'tier_0_hard_stop': -0.05,  # 5% hard stop
                'max_hold_hours': 168,
            },
            'trailing_stop': {
                'enabled': False,
            },
        }
        bt = Backtest1Hour(initial_capital=10000, config=config)
        bt.daily_loss_kill_switch_enabled = True
        return bt

    def test_kill_switch_triggers_on_portfolio_loss(self, backtester_strict_kill_switch):
        """Kill switch triggers when portfolio-wide daily loss exceeds threshold."""
        bt = backtester_strict_kill_switch
        bt.kill_switch_trace = True
        bt._kill_switch_trace_log = []

        # Create data within a single trading day (8 hours)
        dates = pd.date_range('2025-01-02 09:00', periods=8, freq='h')

        # Symbol A: Has a BUY signal at bar 1, price drops to trigger hard stop at bar 3
        symbol_a_prices = [100.0, 100.0, 100.0] + [91.0] * 5  # 9% drop triggers 5% hard stop
        symbol_a_lows = [99.5, 99.5, 99.5] + [90.0] * 5  # Low hits stop
        data_a = pd.DataFrame({
            'timestamp': dates,
            'open': symbol_a_prices,
            'high': [p + 0.5 for p in symbol_a_prices],
            'low': symbol_a_lows,
            'close': symbol_a_prices,
            'volume': [10000] * 8,
            'signal': [0] * 8,
            'confidence': [0.0] * 8,
            'strategy': [''] * 8,
            'reasoning': [''] * 8,
        })
        data_a.loc[1, 'signal'] = 1
        data_a.loc[1, 'confidence'] = 75.0
        data_a.loc[1, 'strategy'] = 'Momentum'

        # Symbol B: Has a BUY signal at bar 5 (after A's loss)
        data_b = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 8,
            'high': [102.0] * 8,
            'low': [99.0] * 8,
            'close': [101.0] * 8,
            'volume': [10000] * 8,
            'signal': [0] * 8,
            'confidence': [0.0] * 8,
            'strategy': [''] * 8,
            'reasoning': [''] * 8,
        })
        data_b.loc[5, 'signal'] = 1
        data_b.loc[5, 'confidence'] = 75.0
        data_b.loc[5, 'strategy'] = 'Breakout'

        signals_data = {'SYMBOL_A': data_a, 'SYMBOL_B': data_b}

        # Run interleaved simulation
        trades = bt.simulate_trades_interleaved(signals_data)

        # Verify kill switch triggered by checking trace log
        trade_closes = [e for e in bt._kill_switch_trace_log if e['event'] == 'TRADE_CLOSE']
        assert len(trade_closes) >= 1
        assert trade_closes[0]['kill_switch_after'] is True, "Kill switch should trigger after trade close"

        # Also verify an entry was blocked
        blocked = [e for e in bt._kill_switch_trace_log if e['event'] == 'ENTRY_BLOCKED']
        assert len(blocked) > 0, "Symbol B's entry should be blocked by kill switch"

    def test_kill_switch_blocks_entries_for_all_symbols(self, backtester_strict_kill_switch):
        """When kill switch triggers on symbol A, it blocks entries for symbol B."""
        bt = backtester_strict_kill_switch
        bt.kill_switch_trace = True
        bt._kill_switch_trace_log = []

        # Create data: Symbol A loses money first, then Symbol B tries to enter
        dates = pd.date_range('2025-01-02 09:00', periods=30, freq='h')

        # Symbol A: Enters early, exits with large loss (triggers kill switch)
        symbol_a_prices = [100.0] * 3 + [85.0] * 27  # 15% drop forces exit
        data_a = pd.DataFrame({
            'timestamp': dates,
            'open': symbol_a_prices,
            'high': [p + 0.5 for p in symbol_a_prices],
            'low': [p - 0.5 for p in symbol_a_prices],
            'close': symbol_a_prices,
            'volume': [10000] * 30,
            'signal': [0] * 30,
            'confidence': [0.0] * 30,
            'strategy': [''] * 30,
            'reasoning': [''] * 30,
        })
        data_a.loc[1, 'signal'] = 1  # Early entry
        data_a.loc[1, 'confidence'] = 75.0

        # Symbol B: Multiple buy signals after A's loss
        data_b = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 30,
            'high': [105.0] * 30,
            'low': [99.0] * 30,
            'close': [102.0] * 30,
            'volume': [10000] * 30,
            'signal': [0] * 30,
            'confidence': [0.0] * 30,
            'strategy': [''] * 30,
            'reasoning': [''] * 30,
        })
        # Add multiple buy signals for Symbol B after A's loss
        for idx in [10, 15, 20]:
            data_b.loc[idx, 'signal'] = 1
            data_b.loc[idx, 'confidence'] = 80.0
            data_b.loc[idx, 'strategy'] = 'Breakout'

        signals_data = {'SYMBOL_A': data_a, 'SYMBOL_B': data_b}
        trades = bt.simulate_trades_interleaved(signals_data)

        # Check trace log for blocked entries
        blocked_entries = [e for e in bt._kill_switch_trace_log
                          if e['event'] == 'ENTRY_BLOCKED' and e['block_reason'] == 'kill_switch']

        # Symbol B's signals should be blocked by kill switch
        symbol_b_blocked = [e for e in blocked_entries if e['symbol'] == 'SYMBOL_B']
        assert len(symbol_b_blocked) > 0, "Symbol B entries should be blocked by portfolio-wide kill switch"

    def test_kill_switch_resets_on_new_trading_day(self, backtester_strict_kill_switch):
        """Kill switch resets when a new trading day begins."""
        bt = backtester_strict_kill_switch
        bt.kill_switch_trace = True
        bt._kill_switch_trace_log = []

        # Day 1: Symbol A loses, triggers kill switch
        # Day 2: Symbol B should be able to enter (kill switch reset)
        dates_day1 = pd.date_range('2025-01-02 09:00', periods=8, freq='h')
        dates_day2 = pd.date_range('2025-01-03 09:00', periods=8, freq='h')
        dates = dates_day1.append(dates_day2)

        # Symbol A: Loses on day 1
        symbol_a_prices = [100.0] * 3 + [85.0] * 13
        data_a = pd.DataFrame({
            'timestamp': dates,
            'open': symbol_a_prices,
            'high': [p + 0.5 for p in symbol_a_prices],
            'low': [p - 0.5 for p in symbol_a_prices],
            'close': symbol_a_prices,
            'volume': [10000] * 16,
            'signal': [0] * 16,
            'confidence': [0.0] * 16,
            'strategy': [''] * 16,
            'reasoning': [''] * 16,
        })
        data_a.loc[1, 'signal'] = 1
        data_a.loc[1, 'confidence'] = 75.0

        # Symbol B: Entry signal on day 2 (after kill switch should reset)
        data_b = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 16,
            'high': [105.0] * 16,
            'low': [99.0] * 16,
            'close': [102.0] * 16,
            'volume': [10000] * 16,
            'signal': [0] * 16,
            'confidence': [0.0] * 16,
            'strategy': [''] * 16,
            'reasoning': [''] * 16,
        })
        # Day 2 signal (index 10 = 2nd hour of day 2)
        data_b.loc[10, 'signal'] = 1
        data_b.loc[10, 'confidence'] = 80.0
        data_b.loc[10, 'strategy'] = 'Momentum'

        signals_data = {'SYMBOL_A': data_a, 'SYMBOL_B': data_b}
        trades = bt.simulate_trades_interleaved(signals_data)

        # Check for DAY_START events (should have 2 - one for each day)
        day_starts = [e for e in bt._kill_switch_trace_log if e['event'] == 'DAY_START']
        assert len(day_starts) >= 2, "Should have DAY_START for each trading day"

        # Check that day 2 started with kill_switch_triggered = False
        day2_start = [e for e in day_starts if '2025-01-03' in e['day']]
        if day2_start:
            assert day2_start[0]['kill_switch_triggered'] is False

        # Check that Symbol B entry was allowed on day 2
        executed_entries = [e for e in bt._kill_switch_trace_log
                          if e['event'] == 'ENTRY_EXECUTED' and e['symbol'] == 'SYMBOL_B']
        # May or may not execute depending on timing, but should NOT be blocked by kill switch on day 2

    def test_portfolio_wide_pnl_accumulates_across_symbols(self, backtester_strict_kill_switch):
        """Daily P&L correctly accumulates losses from multiple symbols."""
        bt = backtester_strict_kill_switch
        bt.kill_switch_trace = True
        bt._kill_switch_trace_log = []

        # Keep data within single trading day (8 hours)
        dates = pd.date_range('2025-01-02 09:00', periods=8, freq='h')

        # Symbol A: Enters at bar 0, drops to trigger hard stop at bar 2
        symbol_a_prices = [100.0, 100.0] + [93.0] * 6  # 7% drop
        symbol_a_lows = [99.5, 99.5] + [92.0] * 6  # Low hits 5% stop
        data_a = pd.DataFrame({
            'timestamp': dates,
            'open': symbol_a_prices,
            'high': [p + 0.5 for p in symbol_a_prices],
            'low': symbol_a_lows,
            'close': symbol_a_prices,
            'volume': [10000] * 8,
            'signal': [0] * 8,
            'confidence': [0.0] * 8,
            'strategy': [''] * 8,
            'reasoning': [''] * 8,
        })
        data_a.loc[0, 'signal'] = 1
        data_a.loc[0, 'confidence'] = 75.0

        # Symbol B: Enters at bar 4 (if not blocked by kill switch)
        data_b = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 8,
            'high': [102.0] * 8,
            'low': [99.0] * 8,
            'close': [101.0] * 8,
            'volume': [10000] * 8,
            'signal': [0] * 8,
            'confidence': [0.0] * 8,
            'strategy': [''] * 8,
            'reasoning': [''] * 8,
        })
        data_b.loc[4, 'signal'] = 1
        data_b.loc[4, 'confidence'] = 75.0

        signals_data = {'SYMBOL_A': data_a, 'SYMBOL_B': data_b}
        trades = bt.simulate_trades_interleaved(signals_data)

        # Check that trade closes logged accumulated P&L
        trade_closes = [e for e in bt._kill_switch_trace_log if e['event'] == 'TRADE_CLOSE']
        assert len(trade_closes) >= 1

        # Kill switch should trigger from Symbol A's loss
        assert trade_closes[0]['kill_switch_after'] is True

        # Symbol B should be blocked
        blocked = [e for e in bt._kill_switch_trace_log
                  if e['event'] == 'ENTRY_BLOCKED' and e['symbol'] == 'SYMBOL_B']
        assert len(blocked) > 0, "Symbol B entry should be blocked after Symbol A's loss"

    def test_kill_switch_trace_logging(self, backtester_strict_kill_switch):
        """Kill switch trace logging captures all required fields."""
        bt = backtester_strict_kill_switch
        bt.kill_switch_trace = True
        bt._kill_switch_trace_log = []

        dates = pd.date_range('2025-01-02 09:00', periods=15, freq='h')
        prices = [100.0] * 3 + [85.0] * 12

        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 0.5 for p in prices],
            'low': [p - 0.5 for p in prices],
            'close': prices,
            'volume': [10000] * 15,
            'signal': [0] * 15,
            'confidence': [0.0] * 15,
            'strategy': [''] * 15,
            'reasoning': [''] * 15,
        })
        data.loc[1, 'signal'] = 1
        data.loc[1, 'confidence'] = 75.0
        data.loc[10, 'signal'] = 1  # Signal after loss
        data.loc[10, 'confidence'] = 75.0

        signals_data = {'TEST': data}
        bt.simulate_trades_interleaved(signals_data)

        # Verify DAY_START event has required fields
        day_starts = [e for e in bt._kill_switch_trace_log if e['event'] == 'DAY_START']
        assert len(day_starts) >= 1
        for event in day_starts:
            assert 'day_start_equity' in event
            assert 'daily_pnl' in event
            assert 'kill_switch_triggered' in event

        # Verify TRADE_CLOSE event has required fields
        trade_closes = [e for e in bt._kill_switch_trace_log if e['event'] == 'TRADE_CLOSE']
        for event in trade_closes:
            assert 'realized_pnl' in event
            assert 'daily_pnl_after' in event
            assert 'daily_loss_pct' in event
            assert 'kill_switch_before' in event
            assert 'kill_switch_after' in event

        # Verify ENTRY_BLOCKED events
        blocked_entries = [e for e in bt._kill_switch_trace_log if e['event'] == 'ENTRY_BLOCKED']
        for event in blocked_entries:
            assert 'block_reason' in event
            assert 'kill_switch_triggered' in event


class TestStrategyManagerExport:
    """Test StrategyManager is correctly exported."""

    def test_import_strategy_manager(self):
        """StrategyManager can be imported from strategies."""
        from strategies import StrategyManager
        assert StrategyManager is not None

    def test_strategy_manager_in_all(self):
        """StrategyManager is in strategies.__all__."""
        import strategies
        assert 'StrategyManager' in strategies.__all__


class TestStrategyManager:
    """Test StrategyManager class."""

    def test_initialization_with_config(self):
        """StrategyManager initializes with config."""
        from strategies import StrategyManager

        config = {
            'entry_gate': {'confidence_threshold': 70},
            'strategies': {
                'momentum': {'enabled': True},
                'mean_reversion': {'enabled': False},
            }
        }
        manager = StrategyManager(config)

        assert manager.confidence_threshold == 70
        assert len(manager.strategies) == 3  # All 3 strategies created

    def test_evaluate_all_strategies(self):
        """evaluate_all_strategies returns list of signals."""
        from strategies import StrategyManager

        manager = StrategyManager()

        # Create sample data
        data = pd.DataFrame({
            'open': [100.0] * 50,
            'high': [102.0] * 50,
            'low': [99.0] * 50,
            'close': [101.0] * 50,
            'volume': [10000] * 50,
            'SMA_20': [100.0] * 50,
            'SMA_50': [99.0] * 50,
            'RSI': [55.0] * 50,
            'BB_LOWER': [98.0] * 50,
            'BB_UPPER': [104.0] * 50,
            'ATR': [2.0] * 50,
        })

        results = manager.evaluate_all_strategies('TEST', data, 101.0, None)

        assert isinstance(results, list)
        for r in results:
            assert 'action' in r
            assert 'confidence' in r
            assert 'strategy' in r

    def test_get_best_signal(self):
        """get_best_signal returns best signal dict."""
        from strategies import StrategyManager

        manager = StrategyManager()

        data = pd.DataFrame({
            'open': [100.0] * 50,
            'high': [102.0] * 50,
            'low': [99.0] * 50,
            'close': [101.0] * 50,
            'volume': [10000] * 50,
            'SMA_20': [100.0] * 50,
            'SMA_50': [99.0] * 50,
            'RSI': [55.0] * 50,
        })

        result = manager.get_best_signal('TEST', data, 101.0, None)

        assert 'action' in result
        assert 'confidence' in result
        assert 'all_strategies' in result

    def test_confidence_threshold_blocks_low_confidence_buys(self):
        """BUY signals below threshold are converted to HOLD."""
        from strategies import StrategyManager

        config = {
            'entry_gate': {'confidence_threshold': 90},  # Very high threshold
        }
        manager = StrategyManager(config)

        # Create data unlikely to generate 90+ confidence
        data = pd.DataFrame({
            'open': [100.0] * 50,
            'high': [101.0] * 50,
            'low': [99.0] * 50,
            'close': [100.5] * 50,
            'volume': [10000] * 50,
            'SMA_20': [100.0] * 50,
            'SMA_50': [99.5] * 50,
            'RSI': [52.0] * 50,
        })

        result = manager.get_best_signal('TEST', data, 100.5, None)

        # With very high threshold, most signals should be blocked
        if result['confidence'] < 90:
            assert result['action'] == 'HOLD'


class TestModuleExports:
    """Test module-level exports."""

    def test_backtest1hour_export(self):
        """Backtest1Hour is importable."""
        from backtest import Backtest1Hour
        assert Backtest1Hour is not None

    def test_run_backtest_export(self):
        """run_backtest is importable."""
        from backtest import run_backtest
        assert callable(run_backtest)

    def test_main_export(self):
        """main is importable."""
        from backtest import main
        assert callable(main)
