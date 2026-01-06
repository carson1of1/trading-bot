"""Tests for execute_entry position tracking fix.

Tests for the Jan 2026 fix where:
- Orders were submitted successfully to Alpaca
- But positions weren't tracked because order.status wasn't in ['filled', 'new', 'accepted']
- This caused max_open_positions to not be enforced (7 orders went through instead of 1)

The fix changed:
    if order and order.status in ['filled', 'new', 'accepted']:
to:
    if order:

This ensures positions are tracked optimistically regardless of order status.
"""
import pytest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.broker import Order


class TestExecuteEntryTracking:
    """Test that execute_entry tracks positions regardless of order status."""

    @pytest.fixture
    def mock_bot(self):
        """Create a minimally mocked TradingBot for testing execute_entry."""
        with patch('bot.VolatilityScanner'), \
             patch('bot.create_broker') as mock_create_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):

            mock_broker = MagicMock()
            mock_create_broker.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()

            # Reset state
            bot.open_positions = {}
            bot.highest_prices = {}
            bot.lowest_prices = {}
            bot.trailing_stops = {}
            bot.last_trade_time = {}
            bot.portfolio_value = 100000

            yield bot, mock_broker

    def test_position_tracked_with_filled_status(self, mock_bot):
        """Position tracked when order status is 'filled'."""
        bot, mock_broker = mock_bot

        # Create order with 'filled' status
        order = Order(
            id='order-123',
            symbol='AAPL',
            qty=100,
            side='buy',
            type='market',
            status='filled',
            filled_qty=100,
            filled_avg_price=150.0
        )
        mock_broker.submit_order.return_value = order

        result = bot.execute_entry(
            symbol='AAPL',
            direction='LONG',
            price=150.0,
            strategy='Momentum',
            reasoning='Test'
        )

        assert result['filled'] is True
        assert 'AAPL' in bot.open_positions
        assert bot.open_positions['AAPL']['qty'] == 100

    def test_position_tracked_with_new_status(self, mock_bot):
        """Position tracked when order status is 'new'."""
        bot, mock_broker = mock_bot

        order = Order(
            id='order-123',
            symbol='AAPL',
            qty=100,
            side='buy',
            type='market',
            status='new',
            filled_qty=0,
            filled_avg_price=None
        )
        mock_broker.submit_order.return_value = order

        result = bot.execute_entry(
            symbol='AAPL',
            direction='LONG',
            price=150.0,
            strategy='Momentum',
            reasoning='Test'
        )

        assert result['filled'] is True
        assert 'AAPL' in bot.open_positions
        # Should use price fallback when filled_avg_price is None
        assert bot.open_positions['AAPL']['entry_price'] == 150.0

    def test_position_tracked_with_pending_new_status(self, mock_bot):
        """Position tracked when order status is 'pending_new' (the bug case)."""
        bot, mock_broker = mock_bot

        # This was the bug - 'pending_new' wasn't in ['filled', 'new', 'accepted']
        order = Order(
            id='order-123',
            symbol='AAPL',
            qty=100,
            side='buy',
            type='market',
            status='pending_new',  # This status caused the bug
            filled_qty=0,
            filled_avg_price=None
        )
        mock_broker.submit_order.return_value = order

        result = bot.execute_entry(
            symbol='AAPL',
            direction='LONG',
            price=150.0,
            strategy='Momentum',
            reasoning='Test'
        )

        # With the fix, this should now work
        assert result['filled'] is True
        assert 'AAPL' in bot.open_positions

    def test_position_tracked_with_accepted_status(self, mock_bot):
        """Position tracked when order status is 'accepted'."""
        bot, mock_broker = mock_bot

        order = Order(
            id='order-123',
            symbol='AAPL',
            qty=100,
            side='buy',
            type='market',
            status='accepted',
            filled_qty=0,
            filled_avg_price=None
        )
        mock_broker.submit_order.return_value = order

        result = bot.execute_entry(
            symbol='AAPL',
            direction='LONG',
            price=150.0,
            strategy='Momentum',
            reasoning='Test'
        )

        assert result['filled'] is True
        assert 'AAPL' in bot.open_positions

    def test_position_tracked_with_any_order_object(self, mock_bot):
        """Position tracked as long as order object exists (the fix)."""
        bot, mock_broker = mock_bot

        # Test with various non-standard statuses
        for status in ['pending_cancel', 'pending_replace', 'accepted_for_bidding', 'calculated']:
            bot.open_positions = {}  # Reset

            order = Order(
                id='order-123',
                symbol='AAPL',
                qty=100,
                side='buy',
                type='market',
                status=status,
                filled_qty=0,
                filled_avg_price=None
            )
            mock_broker.submit_order.return_value = order

            result = bot.execute_entry(
                symbol='AAPL',
                direction='LONG',
                price=150.0,
                strategy='Momentum',
                reasoning='Test'
            )

            assert result['filled'] is True, f"Failed for status: {status}"
            assert 'AAPL' in bot.open_positions, f"Position not tracked for status: {status}"

    def test_position_not_tracked_when_order_none(self, mock_bot):
        """Position NOT tracked when broker returns None."""
        bot, mock_broker = mock_bot

        mock_broker.submit_order.return_value = None

        result = bot.execute_entry(
            symbol='AAPL',
            direction='LONG',
            price=150.0,
            strategy='Momentum',
            reasoning='Test'
        )

        assert result['filled'] is False
        assert 'AAPL' not in bot.open_positions

    def test_exit_manager_registered_with_any_status(self, mock_bot):
        """ExitManager registers position regardless of order status."""
        bot, mock_broker = mock_bot

        mock_exit_manager = MagicMock()
        bot.exit_manager = mock_exit_manager
        bot.use_tiered_exits = True

        order = Order(
            id='order-123',
            symbol='AAPL',
            qty=100,
            side='buy',
            type='market',
            status='pending_new',  # Non-standard status
            filled_qty=0,
            filled_avg_price=None
        )
        mock_broker.submit_order.return_value = order

        bot.execute_entry(
            symbol='AAPL',
            direction='LONG',
            price=150.0,
            strategy='Momentum',
            reasoning='Test'
        )

        # ExitManager should be called
        mock_exit_manager.register_position.assert_called_once()


class TestMaxOpenPositionsEnforcement:
    """Test that max_open_positions is enforced correctly after fix."""

    @pytest.fixture
    def mock_bot_with_signals(self):
        """Create bot with multiple qualifying signals."""
        with patch('bot.VolatilityScanner'), \
             patch('bot.create_broker') as mock_create_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):

            mock_broker = MagicMock()
            mock_create_broker.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()

            # Setup for testing
            bot.open_positions = {}
            bot.highest_prices = {}
            bot.lowest_prices = {}
            bot.trailing_stops = {}
            bot.last_trade_time = {}
            bot.portfolio_value = 100000

            # Make execute_entry add to open_positions
            def mock_execute_entry(symbol, direction, price, strategy, reasoning):
                # Simulate order with pending_new status
                order = Order(
                    id=f'order-{symbol}',
                    symbol=symbol,
                    qty=100,
                    side='buy',
                    type='market',
                    status='pending_new',
                    filled_qty=0,
                    filled_avg_price=None
                )
                mock_broker.submit_order.return_value = order

                # Call real execute_entry
                return TradingBot.execute_entry(bot, symbol, direction, price, strategy, reasoning)

            yield bot, mock_broker, mock_execute_entry

    def test_max_positions_enforced_with_pending_status(self, mock_bot_with_signals):
        """Only 1 position opened when max_open_positions=1, even with pending_new status."""
        bot, mock_broker, mock_execute_entry = mock_bot_with_signals

        # Patch execute_entry to use our mock
        with patch.object(bot, 'execute_entry', mock_execute_entry):
            # Simulate what run_trading_cycle does
            max_positions = 1
            qualifying_signals = [
                {'symbol': 'AAPL', 'signal': {'action': 'BUY', 'confidence': 90, 'direction': 'LONG'}, 'price': 150.0},
                {'symbol': 'MSFT', 'signal': {'action': 'BUY', 'confidence': 85, 'direction': 'LONG'}, 'price': 350.0},
                {'symbol': 'GOOGL', 'signal': {'action': 'BUY', 'confidence': 80, 'direction': 'LONG'}, 'price': 140.0},
            ]

            for entry in qualifying_signals:
                if len(bot.open_positions) >= max_positions:
                    break

                bot.execute_entry(
                    symbol=entry['symbol'],
                    direction=entry['signal'].get('direction', 'LONG'),
                    price=entry['price'],
                    strategy='Test',
                    reasoning='Test'
                )

            # Should only have 1 position
            assert len(bot.open_positions) == 1
            assert 'AAPL' in bot.open_positions  # First one (highest confidence)
            assert 'MSFT' not in bot.open_positions
            assert 'GOOGL' not in bot.open_positions


class TestShortPositionTracking:
    """Test that SHORT positions are also tracked correctly."""

    @pytest.fixture
    def mock_bot(self):
        """Create a minimally mocked TradingBot."""
        with patch('bot.VolatilityScanner'), \
             patch('bot.create_broker') as mock_create_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):

            mock_broker = MagicMock()
            mock_create_broker.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()

            bot.open_positions = {}
            bot.highest_prices = {}
            bot.lowest_prices = {}
            bot.trailing_stops = {}
            bot.last_trade_time = {}
            bot.portfolio_value = 100000

            yield bot, mock_broker

    def test_short_position_tracked(self, mock_bot):
        """SHORT position tracked correctly."""
        bot, mock_broker = mock_bot

        order = Order(
            id='order-123',
            symbol='AAPL',
            qty=100,
            side='sell',
            type='market',
            status='pending_new',
            filled_qty=0,
            filled_avg_price=None
        )
        mock_broker.submit_order.return_value = order

        result = bot.execute_entry(
            symbol='AAPL',
            direction='SHORT',
            price=150.0,
            strategy='MeanReversion',
            reasoning='Test'
        )

        assert result['filled'] is True
        assert 'AAPL' in bot.open_positions
        assert bot.open_positions['AAPL']['direction'] == 'SHORT'
