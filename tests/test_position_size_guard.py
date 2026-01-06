"""
Tests for position size violation detection and auto-liquidation.

Tests the position size guard that:
- Detects positions exceeding max_position_dollars
- Detects positions exceeding max_position_size_pct of portfolio
- Auto-liquidates violating positions
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class MockOrder:
    """Mock order response."""
    def __init__(self, status='filled', filled_price=100.0):
        self.id = 'test-order-123'
        self.status = status
        self.filled_avg_price = filled_price
        self.filled_qty = 10


class TestPositionSizeViolationDetection:
    """Test detection of position size violations."""

    def test_no_violation_under_dollar_limit(self):
        """Position under max_position_dollars is not flagged."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_broker.get_account.return_value = MagicMock(
                cash=50000,
                portfolio_value=100000,
                last_equity=100000
            )
            mock_broker.get_positions.return_value = []
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 100000
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 50,
                    'current_price': 150.0,  # Value: $7,500 < $10,000
                    'entry_price': 145.0,
                    'direction': 'LONG'
                }
            }

            violations = bot._check_position_size_violations()

            assert violations == []
            mock_broker.submit_order.assert_not_called()

    def test_violation_exceeds_dollar_limit(self):
        """Position exceeding max_position_dollars triggers liquidation."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_broker.get_account.return_value = MagicMock(
                cash=50000,
                portfolio_value=100000,
                last_equity=100000
            )
            mock_broker.get_positions.return_value = []
            mock_broker.submit_order.return_value = MockOrder()
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 100000
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 100,
                    'current_price': 150.0,  # Value: $15,000 > $10,000
                    'entry_price': 145.0,
                    'direction': 'LONG'
                }
            }

            violations = bot._check_position_size_violations()

            assert len(violations) == 1
            assert violations[0]['symbol'] == 'AAPL'
            assert violations[0]['reason'] == 'exceeds_max_position_dollars'
            mock_broker.submit_order.assert_called_once()

    def test_violation_exceeds_portfolio_pct(self):
        """Position exceeding max_position_size_pct triggers liquidation."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_broker.get_account.return_value = MagicMock(
                cash=50000,
                portfolio_value=50000,
                last_equity=50000
            )
            mock_broker.get_positions.return_value = []
            mock_broker.submit_order.return_value = MockOrder()
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 50000  # Small portfolio
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 50,
                    'current_price': 150.0,  # Value: $7,500 = 15% > 10%
                    'entry_price': 145.0,
                    'direction': 'LONG'
                }
            }

            violations = bot._check_position_size_violations()

            assert len(violations) == 1
            assert violations[0]['symbol'] == 'AAPL'
            assert violations[0]['reason'] == 'exceeds_max_position_pct'

    def test_no_violation_when_under_both_limits(self):
        """Position under both limits is not flagged."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_broker.get_account.return_value = MagicMock(
                cash=50000,
                portfolio_value=200000,
                last_equity=200000
            )
            mock_broker.get_positions.return_value = []
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 200000
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 50,
                    'current_price': 150.0,  # Value: $7,500 < $10K and < 10% of $200K
                    'entry_price': 145.0,
                    'direction': 'LONG'
                }
            }

            violations = bot._check_position_size_violations()

            assert violations == []


class TestPositionSizeLiquidation:
    """Test auto-liquidation of violating positions."""

    def test_long_position_liquidation(self):
        """LONG position is sold when violating."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_broker.get_account.return_value = MagicMock(
                cash=50000,
                portfolio_value=100000,
                last_equity=100000
            )
            mock_broker.get_positions.return_value = []
            mock_broker.submit_order.return_value = MockOrder()
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 100000
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 100,
                    'current_price': 150.0,
                    'entry_price': 145.0,
                    'direction': 'LONG'
                }
            }

            bot._check_position_size_violations()

            mock_broker.submit_order.assert_called_once_with(
                symbol='AAPL',
                qty=100,
                side='sell',
                type='market',
                time_in_force='day'
            )

    def test_short_position_liquidation(self):
        """SHORT position is bought back when violating."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_broker.get_account.return_value = MagicMock(
                cash=50000,
                portfolio_value=100000,
                last_equity=100000
            )
            mock_broker.get_positions.return_value = []
            mock_broker.submit_order.return_value = MockOrder()
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 100000
            bot.open_positions = {
                'TSLA': {
                    'symbol': 'TSLA',
                    'qty': 50,
                    'current_price': 250.0,  # Value: $12,500 > $10,000
                    'entry_price': 260.0,
                    'direction': 'SHORT'
                }
            }

            bot._check_position_size_violations()

            mock_broker.submit_order.assert_called_once_with(
                symbol='TSLA',
                qty=50,
                side='buy',
                type='market',
                time_in_force='day'
            )

    def test_multiple_violations_all_liquidated(self):
        """Multiple violating positions are all liquidated."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_broker.get_account.return_value = MagicMock(
                cash=50000,
                portfolio_value=100000,
                last_equity=100000
            )
            mock_broker.get_positions.return_value = []
            mock_broker.submit_order.return_value = MockOrder()
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 100000
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 100,
                    'current_price': 150.0,  # Value: $15,000
                    'entry_price': 145.0,
                    'direction': 'LONG'
                },
                'TSLA': {
                    'symbol': 'TSLA',
                    'qty': 50,
                    'current_price': 250.0,  # Value: $12,500
                    'entry_price': 260.0,
                    'direction': 'SHORT'
                }
            }

            violations = bot._check_position_size_violations()

            assert len(violations) == 2
            assert mock_broker.submit_order.call_count == 2


class TestPositionCleanupAfterLiquidation:
    """Test that positions are cleaned up after liquidation."""

    def test_position_removed_after_liquidation(self):
        """Position is removed from tracking after liquidation."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_broker.get_account.return_value = MagicMock(
                cash=50000,
                portfolio_value=100000,
                last_equity=100000
            )
            mock_broker.get_positions.return_value = []
            mock_broker.submit_order.return_value = MockOrder()
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 100000
            bot.highest_prices = {'AAPL': 150.0}
            bot.lowest_prices = {'AAPL': 145.0}
            bot.trailing_stops = {'AAPL': {'activated': False, 'price': 0.0}}
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 100,
                    'current_price': 150.0,
                    'entry_price': 145.0,
                    'direction': 'LONG'
                }
            }

            bot._check_position_size_violations()

            assert 'AAPL' not in bot.open_positions
            assert 'AAPL' not in bot.highest_prices
            assert 'AAPL' not in bot.lowest_prices
            assert 'AAPL' not in bot.trailing_stops


class TestPositionSizeGuardIntegration:
    """Test position size guard integration with trading cycle."""

    def test_guard_runs_in_trading_cycle(self):
        """Verify _check_position_size_violations is called in run_trading_cycle."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_account = MagicMock(
                cash=50000,
                portfolio_value=100000,
                last_equity=100000,
                equity=100000
            )
            mock_broker.get_account.return_value = mock_account
            mock_broker.get_positions.return_value = []
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 100000
            bot.daily_starting_capital = 100000

            with patch.object(bot, '_check_position_size_violations') as mock_check:
                mock_check.return_value = []
                bot.run_trading_cycle()
                mock_check.assert_called_once()
