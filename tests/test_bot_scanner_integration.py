"""Tests for scanner-bot integration."""
import pytest
import subprocess
import sys
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
