#!/usr/bin/env python3
"""
Signal Verification Script (ODE-97)

Standalone script to test signal generation without running the full bot.
Useful for debugging why signals aren't being generated or why they're
not meeting threshold.

Usage:
    python verify_signals.py --symbol AAPL
    python verify_signals.py --symbol AAPL --show-components
    python verify_signals.py --symbols AAPL,NVDA,TSLA
    python verify_signals.py --universe  # Use universe.yaml
"""

import argparse
import logging
import sys
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core import YFinanceDataFetcher, TechnicalIndicators
from strategies import StrategyManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}


def load_universe(universe_path: str = "universe.yaml") -> List[str]:
    """Load symbols from universe file (scanner_universe for full coverage)."""
    try:
        with open(universe_path, 'r') as f:
            universe = yaml.safe_load(f)
            # Use scanner_universe (400 symbols) for full scanner benefit
            scanner_universe = universe.get('scanner_universe', {})
            symbols = []
            for category, syms in scanner_universe.items():
                if isinstance(syms, list):
                    for s in syms:
                        if s not in symbols:
                            symbols.append(s)
            # Fallback to proven_symbols if scanner_universe empty
            if not symbols:
                symbols = universe.get('proven_symbols', [])
            return symbols
    except FileNotFoundError:
        logger.error(f"Universe file not found: {universe_path}")
        return []


def fetch_data(symbol: str, fetcher: YFinanceDataFetcher,
               indicators: TechnicalIndicators, bars: int = 200) -> Optional[pd.DataFrame]:
    """Fetch and prepare data for a symbol."""
    end_date = datetime.now()
    warmup_days = int(bars / 6.5) + 10
    start_date = end_date - timedelta(days=warmup_days)

    df = fetcher.get_historical_data_range(
        symbol=symbol,
        timeframe='1Hour',
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        use_cache=False
    )

    if df is None or df.empty:
        return None

    # Add indicators
    df = indicators.add_all_indicators(df)
    return df


def verify_signal(symbol: str, config: dict, show_components: bool = False) -> dict:
    """
    Verify signal generation for a single symbol.

    Returns:
        dict with signal details
    """
    fetcher = YFinanceDataFetcher()
    indicators = TechnicalIndicators()
    strategy_manager = StrategyManager(config)

    # Fetch data
    data = fetch_data(symbol, fetcher, indicators)
    if data is None or len(data) < 30:
        return {
            'symbol': symbol,
            'status': 'ERROR',
            'error': 'Insufficient data',
            'signal': None
        }

    current_price = data['close'].iloc[-1]
    latest_bar = data.iloc[-1]

    # Get signal
    signal = strategy_manager.get_best_signal(
        symbol=symbol,
        data=data,
        current_price=current_price,
        indicators=indicators
    )

    result = {
        'symbol': symbol,
        'status': 'OK',
        'current_price': current_price,
        'timestamp': latest_bar.get('timestamp', 'N/A') if 'timestamp' in data.columns else 'N/A',
        'signal': signal
    }

    # Add all strategy results if available
    if 'all_strategies' in signal:
        result['all_strategies'] = signal['all_strategies']

    return result


def print_signal_result(result: dict, show_components: bool = False, verbose: bool = False):
    """Print signal result in a readable format."""
    symbol = result['symbol']

    if result['status'] == 'ERROR':
        print(f"\n❌ {symbol}: {result['error']}")
        return

    signal = result['signal']
    price = result['current_price']
    timestamp = result['timestamp']

    # Signal header
    action = signal.get('action', 'HOLD')
    confidence = signal.get('confidence', 0)
    strategy = signal.get('strategy', 'Unknown')
    reasoning = signal.get('reasoning', '')

    # Color coding for action
    if action == 'BUY':
        action_str = f"🟢 {action}"
    elif action == 'SELL':
        action_str = f"🔴 {action}"
    else:
        action_str = f"⚪ {action}"

    print(f"\n{'='*60}")
    print(f"📊 {symbol} @ ${price:.2f}")
    print(f"{'='*60}")
    print(f"Latest Bar: {timestamp}")
    print(f"Signal:     {action_str}")
    print(f"Confidence: {confidence:.1f}")
    print(f"Strategy:   {strategy}")
    print(f"Reasoning:  {reasoning}")

    # Show components if requested
    if show_components:
        components = signal.get('components', {})
        if components:
            print(f"\n📈 Signal Components:")
            for key, value in components.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")

    # Show all strategies if verbose
    if verbose and 'all_strategies' in result:
        print(f"\n📋 All Strategy Results:")
        for strat in result.get('all_strategies', []):
            strat_name = strat.get('strategy', 'Unknown')
            strat_action = strat.get('action', 'HOLD')
            strat_conf = strat.get('confidence', 0)
            print(f"   {strat_name}: {strat_action} ({strat_conf:.1f})")


def main():
    parser = argparse.ArgumentParser(
        description='Verify trading signal generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python verify_signals.py --symbol AAPL
    python verify_signals.py --symbol AAPL --show-components
    python verify_signals.py --symbols AAPL,NVDA,TSLA --verbose
    python verify_signals.py --universe
        """
    )
    parser.add_argument('--symbol', type=str, help='Single symbol to verify')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    parser.add_argument('--universe', action='store_true', help='Use all symbols from universe.yaml')
    parser.add_argument('--show-components', action='store_true', help='Show indicator components')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all strategy results')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--summary-only', action='store_true', help='Only show summary counts')

    args = parser.parse_args()

    # Determine symbols to verify
    symbols = []
    if args.symbol:
        symbols = [args.symbol.upper()]
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    elif args.universe:
        symbols = load_universe()
    else:
        parser.print_help()
        print("\n❌ Error: Must specify --symbol, --symbols, or --universe")
        sys.exit(1)

    if not symbols:
        print("❌ Error: No symbols to verify")
        sys.exit(1)

    # Load config
    config = load_config(args.config)
    threshold = config.get('entry_gate', {}).get('confidence_threshold', 60)

    print(f"🔍 Signal Verification Tool (ODE-97)")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Symbols:    {len(symbols)}")
    print(f"Threshold:  {threshold}")
    print(f"Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Verify each symbol
    results = []
    for symbol in symbols:
        try:
            result = verify_signal(symbol, config, args.show_components)
            results.append(result)

            if not args.summary_only:
                print_signal_result(result, args.show_components, args.verbose)

        except Exception as e:
            logger.error(f"Error verifying {symbol}: {e}")
            results.append({
                'symbol': symbol,
                'status': 'ERROR',
                'error': str(e),
                'signal': None
            })

    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")

    total = len(results)
    errors = sum(1 for r in results if r['status'] == 'ERROR')
    buy_signals = sum(1 for r in results if r.get('signal', {}).get('action') == 'BUY')
    sell_signals = sum(1 for r in results if r.get('signal', {}).get('action') == 'SELL')
    hold_signals = total - errors - buy_signals - sell_signals

    above_threshold = sum(
        1 for r in results
        if r.get('signal', {}).get('action') in ['BUY', 'SELL']
        and r.get('signal', {}).get('confidence', 0) >= threshold
    )

    print(f"Total:          {total}")
    print(f"Errors:         {errors}")
    print(f"BUY signals:    {buy_signals}")
    print(f"SELL signals:   {sell_signals}")
    print(f"HOLD:           {hold_signals}")
    print(f"Above threshold ({threshold}): {above_threshold}")

    # List symbols with actionable signals
    if buy_signals > 0 or sell_signals > 0:
        print(f"\n🎯 Actionable Signals:")
        for r in results:
            signal = r.get('signal', {})
            if signal.get('action') in ['BUY', 'SELL']:
                conf = signal.get('confidence', 0)
                status = "✅" if conf >= threshold else "⚠️"
                print(f"   {status} {r['symbol']}: {signal['action']} ({conf:.1f})")


if __name__ == '__main__':
    main()
