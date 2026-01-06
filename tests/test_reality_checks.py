"""Tests for reality check analyzers."""

import pytest
from unittest.mock import patch, MagicMock
from validation.reality_checks import (
    CostSensitivityAnalyzer,
    DelaySensitivityAnalyzer,
    RegimeSplitAnalyzer,
    ShortDisableAnalyzer,
    RealityCheckSuite,
)


def create_mock_backtest_result(total_return=5.0, profit_factor=2.0, trades=None):
    """Create a mock backtest result."""
    if trades is None:
        trades = [
            {'symbol': 'SPY', 'pnl': 100, 'entry_price': 100, 'shares': 10, 'bars_held': 5},
        ]
    return {
        'trades': trades,
        'metrics': {
            'total_return_pct': total_return,
            'max_drawdown': 2.0,
            'profit_factor': profit_factor,
            'win_rate': 60.0,
            'total_trades': len(trades),
        }
    }


class TestCostSensitivity:
    """Test cost sensitivity analysis."""

    @patch('validation.reality_checks.Backtest1Hour')
    def test_runs_multiple_friction_levels(self, mock_backtest_class):
        """Test analyzer runs with different friction levels."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = create_mock_backtest_result()
        mock_backtest_class.return_value = mock_instance

        analyzer = CostSensitivityAnalyzer(symbols=['SPY'], quick_mode=True)
        results = analyzer.run()

        assert '0_bps' in results
        assert '5_bps' in results
        assert '15_bps' in results

    @patch('validation.reality_checks.Backtest1Hour')
    def test_detects_fragility(self, mock_backtest_class):
        """Test fragility detection."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = create_mock_backtest_result()
        mock_backtest_class.return_value = mock_instance

        analyzer = CostSensitivityAnalyzer(symbols=['SPY'], quick_mode=True)
        results = analyzer.run()

        assert 'is_fragile' in results
        assert isinstance(results['is_fragile'], bool)

    @patch('validation.reality_checks.Backtest1Hour')
    def test_fragile_when_high_decay(self, mock_backtest_class):
        """Test fragility detection with high decay."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            # First call (0 bps): high return
            if call_count[0] == 1:
                return create_mock_backtest_result(total_return=10.0)
            # Third call (15 bps): very low return
            return create_mock_backtest_result(total_return=1.0)

        mock_instance = MagicMock()
        mock_instance.run.side_effect = side_effect
        mock_backtest_class.return_value = mock_instance

        analyzer = CostSensitivityAnalyzer(symbols=['SPY'], quick_mode=True)
        results = analyzer.run()

        # 90% decay should be fragile
        assert results['is_fragile'] is True

    @patch('validation.reality_checks.Backtest1Hour')
    def test_not_fragile_when_low_decay(self, mock_backtest_class):
        """Test not fragile with low decay."""
        mock_instance = MagicMock()
        # Consistent returns across friction levels
        mock_instance.run.return_value = create_mock_backtest_result(total_return=10.0)
        mock_backtest_class.return_value = mock_instance

        analyzer = CostSensitivityAnalyzer(symbols=['SPY'], quick_mode=True)
        results = analyzer.run()

        # 0% decay should not be fragile
        assert results['is_fragile'] is False


class TestDelaySensitivity:
    """Test delay sensitivity analysis."""

    @patch('validation.reality_checks.Backtest1Hour')
    def test_runs_multiple_delay_modes(self, mock_backtest_class):
        """Test analyzer runs with different delay modes."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = create_mock_backtest_result()
        mock_backtest_class.return_value = mock_instance

        analyzer = DelaySensitivityAnalyzer(symbols=['SPY'], quick_mode=True)
        results = analyzer.run()

        assert 'same_bar' in results
        assert 'next_open' in results
        assert 'next_open_plus_1' in results

    @patch('validation.reality_checks.Backtest1Hour')
    def test_next_open_has_metrics(self, mock_backtest_class):
        """Test next_open mode has proper metrics."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = create_mock_backtest_result(total_return=5.0)
        mock_backtest_class.return_value = mock_instance

        analyzer = DelaySensitivityAnalyzer(symbols=['SPY'], quick_mode=True)
        results = analyzer.run()

        assert results['next_open']['total_return'] == 5.0
        assert 'description' in results['next_open']


class TestRegimeSplitAnalyzer:
    """Test regime split analysis."""

    @patch('validation.reality_checks.Backtest1Hour')
    def test_runs_both_regimes(self, mock_backtest_class):
        """Test analyzer runs for both high and low vol regimes."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = create_mock_backtest_result()
        mock_backtest_class.return_value = mock_instance

        analyzer = RegimeSplitAnalyzer(symbols=['SPY'], quick_mode=True)
        results = analyzer.run()

        assert 'high_vol' in results
        assert 'low_vol' in results
        assert 'regime_dependency' in results

    @patch('validation.reality_checks.Backtest1Hour')
    def test_regime_months_included(self, mock_backtest_class):
        """Test regime results include month lists."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = create_mock_backtest_result()
        mock_backtest_class.return_value = mock_instance

        analyzer = RegimeSplitAnalyzer(symbols=['SPY'], quick_mode=True)
        results = analyzer.run()

        assert 'months' in results['high_vol']
        assert 'months' in results['low_vol']


class TestShortDisableAnalyzer:
    """Test short disable analysis."""

    @patch('validation.reality_checks.Backtest1Hour')
    def test_runs_all_modes(self, mock_backtest_class):
        """Test analyzer runs both/longs/shorts modes."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = create_mock_backtest_result()
        mock_backtest_class.return_value = mock_instance

        analyzer = ShortDisableAnalyzer(symbols=['SPY'], quick_mode=True)
        results = analyzer.run()

        assert 'both' in results
        assert 'longs_only' in results
        assert 'shorts_only' in results

    @patch('validation.reality_checks.Backtest1Hour')
    def test_shorts_helping_calculation(self, mock_backtest_class):
        """Test shorts_helping is calculated correctly."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = create_mock_backtest_result()
        mock_backtest_class.return_value = mock_instance

        analyzer = ShortDisableAnalyzer(symbols=['SPY'], quick_mode=True)
        results = analyzer.run()

        assert 'shorts_helping' in results
        assert isinstance(results['shorts_helping'], bool)


class TestRealityCheckSuite:
    """Test full reality check suite."""

    @patch('validation.reality_checks.Backtest1Hour')
    def test_run_all_returns_summary(self, mock_backtest_class):
        """Test suite runs all checks and returns summary."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = create_mock_backtest_result()
        mock_backtest_class.return_value = mock_instance

        suite = RealityCheckSuite(symbols=['SPY'], quick_mode=True)
        results = suite.run_all()

        assert 'cost_sensitivity' in results
        assert 'delay_sensitivity' in results
        assert 'regime_split' in results
        assert 'short_disable' in results
        assert 'warnings' in results

    @patch('validation.reality_checks.Backtest1Hour')
    def test_warnings_list_exists(self, mock_backtest_class):
        """Test warnings is a list."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = create_mock_backtest_result()
        mock_backtest_class.return_value = mock_instance

        suite = RealityCheckSuite(symbols=['SPY'], quick_mode=True)
        results = suite.run_all()

        assert isinstance(results['warnings'], list)

    @patch('validation.reality_checks.Backtest1Hour')
    def test_generated_at_timestamp(self, mock_backtest_class):
        """Test results include timestamp."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = create_mock_backtest_result()
        mock_backtest_class.return_value = mock_instance

        suite = RealityCheckSuite(symbols=['SPY'], quick_mode=True)
        results = suite.run_all()

        assert 'generated_at' in results


class TestQuickMode:
    """Test quick mode behavior across analyzers."""

    def test_cost_analyzer_limits_symbols(self):
        """Test cost analyzer limits symbols in quick mode."""
        many_symbols = [f'SYM{i}' for i in range(50)]
        analyzer = CostSensitivityAnalyzer(symbols=many_symbols, quick_mode=True)

        assert len(analyzer.symbols) <= 20

    def test_delay_analyzer_limits_symbols(self):
        """Test delay analyzer limits symbols in quick mode."""
        many_symbols = [f'SYM{i}' for i in range(50)]
        analyzer = DelaySensitivityAnalyzer(symbols=many_symbols, quick_mode=True)

        assert len(analyzer.symbols) <= 20

    def test_regime_analyzer_limits_symbols(self):
        """Test regime analyzer limits symbols in quick mode."""
        many_symbols = [f'SYM{i}' for i in range(50)]
        analyzer = RegimeSplitAnalyzer(symbols=many_symbols, quick_mode=True)

        assert len(analyzer.symbols) <= 20

    def test_short_analyzer_limits_symbols(self):
        """Test short analyzer limits symbols in quick mode."""
        many_symbols = [f'SYM{i}' for i in range(50)]
        analyzer = ShortDisableAnalyzer(symbols=many_symbols, quick_mode=True)

        assert len(analyzer.symbols) <= 20
