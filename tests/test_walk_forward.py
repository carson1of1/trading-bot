"""Tests for walk-forward validation framework."""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from validation.walk_forward import WalkForwardTest, WalkForwardConfig, PeriodResults


class TestWalkForwardConfig:
    """Test walk-forward configuration."""

    def test_default_periods(self):
        """Test default walk-forward periods are set correctly."""
        config = WalkForwardConfig()

        assert config.train_start == '2024-01-05'
        assert config.train_end == '2024-09-30'
        assert config.validation_start == '2024-10-01'
        assert config.validation_end == '2024-12-31'
        assert config.test_start == '2025-01-01'
        assert config.test_end == '2025-12-31'

    def test_no_overlap(self):
        """Test periods don't overlap."""
        config = WalkForwardConfig()

        train_end = datetime.strptime(config.train_end, '%Y-%m-%d')
        val_start = datetime.strptime(config.validation_start, '%Y-%m-%d')
        val_end = datetime.strptime(config.validation_end, '%Y-%m-%d')
        test_start = datetime.strptime(config.test_start, '%Y-%m-%d')

        assert val_start > train_end, "Validation must start after train ends"
        assert test_start > val_end, "Test must start after validation ends"

    def test_custom_periods(self):
        """Test custom period configuration."""
        config = WalkForwardConfig(
            train_start='2023-01-01',
            train_end='2023-06-30',
            validation_start='2023-07-01',
            validation_end='2023-09-30',
            test_start='2023-10-01',
            test_end='2023-12-31',
        )

        assert config.train_start == '2023-01-01'
        assert config.test_end == '2023-12-31'


class TestPeriodResults:
    """Test PeriodResults dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        results = PeriodResults(
            period_name='train',
            start_date='2024-01-01',
            end_date='2024-06-30',
            total_return=15.5,
            max_drawdown=5.0,
            profit_factor=2.1,
            total_trades=50,
            win_rate=60.0,
        )

        d = results.to_dict()

        assert d['period_name'] == 'train'
        assert d['total_return'] == 15.5
        assert d['profit_factor'] == 2.1
        assert d['total_trades'] == 50

    def test_default_values(self):
        """Test default values are set."""
        results = PeriodResults(
            period_name='test',
            start_date='2024-01-01',
            end_date='2024-06-30',
        )

        assert results.total_return == 0.0
        assert results.avg_hold_bars == 0.0
        assert results.symbol_pnl == {}


class TestWalkForwardTest:
    """Test walk-forward test runner."""

    @patch('validation.walk_forward.Backtest1Hour')
    def test_run_returns_metrics(self, mock_backtest_class):
        """Test that run() returns required metrics."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.run.return_value = {
            'trades': [
                {'symbol': 'SPY', 'pnl': 100, 'entry_price': 100, 'shares': 10, 'bars_held': 5},
                {'symbol': 'SPY', 'pnl': -50, 'entry_price': 100, 'shares': 5, 'bars_held': 3},
            ],
            'metrics': {
                'total_return_pct': 5.0,
                'max_drawdown': 2.0,
                'profit_factor': 2.0,
                'win_rate': 50.0,
                'sharpe_ratio': 1.5,
            }
        }
        mock_backtest_class.return_value = mock_instance

        wf = WalkForwardTest(symbols=['SPY'], quick_mode=True)
        results = wf.run_period('train')

        assert 'total_return' in results
        assert 'max_drawdown' in results
        assert 'profit_factor' in results
        assert 'total_trades' in results
        assert 'avg_hold_bars' in results

    @patch('validation.walk_forward.Backtest1Hour')
    def test_concentration_metrics(self, mock_backtest_class):
        """Test per-symbol concentration is calculated."""
        # Setup mock with multiple symbols
        mock_instance = MagicMock()
        mock_instance.run.return_value = {
            'trades': [
                {'symbol': 'SPY', 'pnl': 100, 'entry_price': 100, 'shares': 10, 'bars_held': 5},
                {'symbol': 'QQQ', 'pnl': 50, 'entry_price': 100, 'shares': 5, 'bars_held': 3},
                {'symbol': 'AAPL', 'pnl': 25, 'entry_price': 100, 'shares': 5, 'bars_held': 3},
            ],
            'metrics': {
                'total_return_pct': 5.0,
                'max_drawdown': 2.0,
                'profit_factor': 2.0,
            }
        }
        mock_backtest_class.return_value = mock_instance

        wf = WalkForwardTest(symbols=['SPY', 'QQQ', 'AAPL'], quick_mode=True)
        results = wf.run_period('train')

        assert 'top1_contribution' in results
        assert 'top3_contribution' in results
        # Top 1 should be SPY at ~57% (100/175)
        assert results['top1_contribution'] > 50

    @patch('validation.walk_forward.Backtest1Hour')
    def test_invalid_period_raises(self, mock_backtest_class):
        """Test invalid period raises ValueError."""
        wf = WalkForwardTest(symbols=['SPY'], quick_mode=True)

        with pytest.raises(ValueError, match="Unknown period"):
            wf.run_period('invalid')

    def test_quick_mode_limits_symbols(self):
        """Test quick mode limits number of symbols."""
        # Create with many symbols
        many_symbols = [f'SYM{i}' for i in range(50)]
        wf = WalkForwardTest(symbols=many_symbols, quick_mode=True)

        assert len(wf.symbols) <= 10

    def test_custom_config(self):
        """Test custom configuration is used."""
        config = WalkForwardConfig(
            train_start='2023-01-01',
            train_end='2023-06-30',
        )
        wf = WalkForwardTest(symbols=['SPY'], config=config)

        assert wf.config.train_start == '2023-01-01'
        assert wf.config.train_end == '2023-06-30'


class TestWalkForwardIntegration:
    """Integration tests for walk-forward framework."""

    @patch('validation.walk_forward.Backtest1Hour')
    def test_print_summary_with_results(self, mock_backtest_class, capsys):
        """Test summary printing works with results."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = {
            'trades': [
                {'symbol': 'SPY', 'pnl': 100, 'entry_price': 100, 'shares': 10, 'bars_held': 5},
            ],
            'metrics': {
                'total_return_pct': 5.0,
                'max_drawdown': 2.0,
                'profit_factor': 2.0,
            }
        }
        mock_backtest_class.return_value = mock_instance

        wf = WalkForwardTest(symbols=['SPY'], quick_mode=True)
        wf.run_period('train')
        wf.print_summary()

        captured = capsys.readouterr()
        assert 'WALK-FORWARD TEST SUMMARY' in captured.out

    def test_print_summary_without_results(self, capsys):
        """Test summary printing handles no results."""
        wf = WalkForwardTest(symbols=['SPY'], quick_mode=True)
        wf.print_summary()

        captured = capsys.readouterr()
        assert 'No results to display' in captured.out
