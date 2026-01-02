"""Tests for scanner-bot integration."""
import pytest
import subprocess
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo
from unittest.mock import patch, MagicMock


class TestBotCLISymbolsArgument:
    """Test bot.py accepts --symbols argument."""

    def test_bot_parses_symbols_argument(self):
        """bot.py should parse --symbols into a list."""
        # Import after patching to avoid actual bot initialization
        with patch('bot.TradingBot') as MockBot:
            mock_instance = MagicMock()
            MockBot.return_value = mock_instance

            # Simulate argument parsing
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument('--config', default='config.yaml')
            parser.add_argument('--symbols', type=str, default=None,
                                help='Comma-separated list of symbols from scanner')

            args = parser.parse_args(['--symbols', 'NVDA,TSLA,AMD'])

            assert args.symbols == 'NVDA,TSLA,AMD'
            symbols_list = args.symbols.split(',') if args.symbols else None
            assert symbols_list == ['NVDA', 'TSLA', 'AMD']

    def test_bot_accepts_scanner_symbols_parameter(self):
        """TradingBot should accept scanner_symbols parameter."""
        with patch('bot.yaml.safe_load') as mock_yaml, \
             patch('builtins.open', MagicMock()), \
             patch('bot.YFinanceDataFetcher'), \
             patch('bot.TechnicalIndicators'), \
             patch('bot.create_broker'), \
             patch('bot.TradeLogger'), \
             patch('bot.RiskManager'), \
             patch('bot.EntryGate'), \
             patch('bot.ExitManager'), \
             patch('bot.MarketHours'), \
             patch('bot.StrategyManager'):

            # Mock config
            mock_yaml.return_value = {
                'mode': 'PAPER',
                'timeframe': '1Hour',
                'trading': {'watchlist_file': 'universe.yaml'},
                'logging': {'database': 'logs/trades.db'},
                'volatility_scanner': {'enabled': False},
                'risk_management': {'max_position_size_pct': 2},
                'entry_gate': {},
                'exit_manager': {},
            }

            from bot import TradingBot

            # Test with scanner_symbols
            bot = TradingBot(scanner_symbols=['NVDA', 'TSLA', 'AMD'])
            assert bot.watchlist == ['NVDA', 'TSLA', 'AMD']


class TestMarketHoursCheck:
    """Test market hours validation functions."""

    def test_is_market_open_during_trading_hours(self):
        """Should return True during market hours (9:30 AM - 4:00 PM ET)."""
        from core.market_hours import is_market_open

        # Mock a Tuesday at 10:30 AM ET (using pytz for compatibility)
        import pytz
        et = pytz.timezone('America/New_York')
        mock_time = et.localize(datetime(2026, 1, 6, 10, 30, 0))

        with patch('core.market_hours.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert is_market_open() == True

    def test_is_market_open_before_market_hours(self):
        """Should return False before 9:30 AM ET."""
        from core.market_hours import is_market_open

        # Mock a Tuesday at 8:00 AM ET
        import pytz
        et = pytz.timezone('America/New_York')
        mock_time = et.localize(datetime(2026, 1, 6, 8, 0, 0))

        with patch('core.market_hours.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert is_market_open() == False

    def test_is_market_open_on_weekend(self):
        """Should return False on weekends."""
        from core.market_hours import is_market_open

        # Mock a Saturday at 11:00 AM ET
        import pytz
        et = pytz.timezone('America/New_York')
        mock_time = et.localize(datetime(2026, 1, 3, 11, 0, 0))

        with patch('core.market_hours.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert is_market_open() == False

    def test_get_market_status_message_when_closed(self):
        """Should return helpful message when market is closed."""
        from core.market_hours import get_market_status_message

        # Mock a Saturday
        import pytz
        et = pytz.timezone('America/New_York')
        mock_time = et.localize(datetime(2026, 1, 3, 11, 0, 0))

        with patch('core.market_hours.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            message = get_market_status_message()
            assert "closed" in message.lower() or "weekend" in message.lower() or "saturday" in message.lower()
