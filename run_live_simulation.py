#!/usr/bin/env python3
"""
Live Trading Simulation - Exact Match

This script simulates EXACTLY how live trading works:
1. Loads all symbols from universe.yaml (scanner_universe)
2. Uses volatility scanner to select top N stocks per day
3. Applies all config.yaml settings (stops, exits, trailing)
4. Runs the full backtest engine with realistic execution

Usage:
    python run_live_simulation.py                    # Default: 1 year, top 10
    python run_live_simulation.py --days 365        # Custom days
    python run_live_simulation.py --top-n 15        # Custom scanner top N
    python run_live_simulation.py --start 2024-01-01 --end 2024-12-31
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backtest import Backtest1Hour


def load_universe() -> list:
    """Load all symbols from universe.yaml scanner_universe."""
    universe_path = Path(__file__).parent / 'universe.yaml'
    with open(universe_path, 'r') as f:
        universe = yaml.safe_load(f)

    scanner_universe = universe.get('scanner_universe', {})
    all_symbols = []

    for category, symbols in scanner_universe.items():
        if isinstance(symbols, list):
            all_symbols.extend(symbols)

    # Deduplicate while preserving order
    unique = list(dict.fromkeys(all_symbols))
    return unique


def load_config() -> dict:
    """Load config.yaml."""
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_simulation(
    start_date: str = None,
    end_date: str = None,
    days: int = 365,
    top_n: int = None,
    longs_only: bool = False,
    shorts_only: bool = False,
    initial_capital: float = 100_000.0,
) -> dict:
    """
    Run live trading simulation.

    Args:
        start_date: Start date (YYYY-MM-DD), defaults to (today - days)
        end_date: End date (YYYY-MM-DD), defaults to today
        days: Number of days if start_date not provided
        top_n: Scanner top N (uses config default if None)
        longs_only: Only long positions
        shorts_only: Only short positions
        initial_capital: Starting capital

    Returns:
        Dict with full backtest results
    """
    # Load symbols and config
    symbols = load_universe()
    config = load_config()

    # Calculate dates
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days)
        start_date = start_dt.strftime('%Y-%m-%d')

    # Override scanner settings if specified
    if top_n is not None:
        config['volatility_scanner']['top_n'] = top_n

    scanner_top_n = config.get('volatility_scanner', {}).get('top_n', 10)

    # Print configuration
    print("\n" + "=" * 70)
    print("LIVE TRADING SIMULATION")
    print("=" * 70)
    print(f"\nDate Range: {start_date} to {end_date}")
    print(f"Total Symbols in Universe: {len(symbols)}")
    print(f"Scanner Top N per Day: {scanner_top_n}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Position Mode: {'Longs Only' if longs_only else 'Shorts Only' if shorts_only else 'Both Longs & Shorts'}")

    # Print key config settings
    risk_config = config.get('risk_management', {})
    exit_config = config.get('exit_manager', {})
    trailing_config = config.get('trailing_stop', {})

    print(f"\nRisk Settings:")
    print(f"  Stop Loss: {risk_config.get('stop_loss_pct', 5)}%")
    print(f"  Take Profit: {risk_config.get('take_profit_pct', 8)}%")
    print(f"  Max Position Size: {risk_config.get('max_position_size_pct', 3)}%")
    print(f"  Max Open Positions: {risk_config.get('max_open_positions', 5)}")

    print(f"\nExit Settings:")
    print(f"  Hard Stop: {exit_config.get('tier_0_hard_stop', -0.05) * 100:.1f}%")
    print(f"  Max Hold: {exit_config.get('max_hold_hours', 168)} hours")
    print(f"  EOD Close: {exit_config.get('eod_close', False)}")

    print(f"\nTrailing Stop:")
    print(f"  Activation: {trailing_config.get('activation_pct', 0.25)}%")
    print(f"  Trail Distance: {trailing_config.get('trail_pct', 0.25)}%")

    print("\n" + "=" * 70)
    print("Running simulation... (this may take several minutes)")
    print("=" * 70 + "\n")

    # Create backtester with live-like settings
    backtester = Backtest1Hour(
        initial_capital=initial_capital,
        config=config,
        longs_only=longs_only,
        shorts_only=shorts_only,
        scanner_enabled=True,  # Critical: enables daily scanner selection
    )

    # Run backtest
    results = backtester.run(symbols, start_date, end_date)

    # Extract metrics
    metrics = results.get('metrics', {})
    trades = results.get('trades', [])

    # Print results
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)

    print(f"\n{'PERFORMANCE':-^50}")
    print(f"  Initial Capital:    ${initial_capital:>15,.2f}")
    print(f"  Final Value:        ${metrics.get('final_value', 0):>15,.2f}")
    print(f"  Total P&L:          ${metrics.get('total_pnl', 0):>15,.2f}")
    print(f"  Total Return:       {metrics.get('total_return_pct', 0):>15.2f}%")
    print(f"  Max Drawdown:       {metrics.get('max_drawdown', 0):>15.2f}%")
    print(f"  Sharpe Ratio:       {metrics.get('sharpe_ratio', 0):>15.2f}")

    print(f"\n{'TRADE STATISTICS':-^50}")
    print(f"  Total Trades:       {metrics.get('total_trades', 0):>15}")
    print(f"  Winning Trades:     {metrics.get('winning_trades', 0):>15}")
    print(f"  Losing Trades:      {metrics.get('losing_trades', 0):>15}")
    print(f"  Win Rate:           {metrics.get('win_rate', 0):>15.1f}%")
    print(f"  Profit Factor:      {metrics.get('profit_factor', 0):>15.2f}")
    print(f"  Avg P&L per Trade:  ${metrics.get('avg_pnl', 0):>15.2f}")
    print(f"  Avg Win:            ${metrics.get('avg_win', 0):>15.2f}")
    print(f"  Avg Loss:           ${metrics.get('avg_loss', 0):>15.2f}")
    print(f"  Best Trade:         ${metrics.get('best_trade', 0):>15.2f}")
    print(f"  Worst Trade:        ${metrics.get('worst_trade', 0):>15.2f}")
    print(f"  Avg Hold (bars):    {metrics.get('avg_bars_held', 0):>15.1f}")

    # Calculate win/loss ratio
    avg_win = metrics.get('avg_win', 0)
    avg_loss = abs(metrics.get('avg_loss', 1))
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    print(f"  Win/Loss Ratio:     {win_loss_ratio:>15.2f}")

    # Symbol breakdown
    if trades:
        from collections import defaultdict
        symbol_pnl = defaultdict(float)
        symbol_trades = defaultdict(int)

        for trade in trades:
            sym = trade['symbol']
            symbol_pnl[sym] += trade['pnl']
            symbol_trades[sym] += 1

        # Sort by P&L
        sorted_symbols = sorted(symbol_pnl.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{'TOP 10 PERFORMERS':-^50}")
        for sym, pnl in sorted_symbols[:10]:
            print(f"  {sym:<8} {symbol_trades[sym]:>3} trades  ${pnl:>10,.2f}")

        print(f"\n{'BOTTOM 10 PERFORMERS':-^50}")
        for sym, pnl in sorted_symbols[-10:]:
            print(f"  {sym:<8} {symbol_trades[sym]:>3} trades  ${pnl:>10,.2f}")

        # Exit reason breakdown
        exit_reasons = defaultdict(lambda: {'count': 0, 'pnl': 0})
        for trade in trades:
            reason = trade.get('exit_reason', 'unknown')
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['pnl'] += trade['pnl']

        print(f"\n{'EXIT REASON BREAKDOWN':-^50}")
        for reason, data in sorted(exit_reasons.items(), key=lambda x: x[1]['count'], reverse=True):
            avg = data['pnl'] / data['count'] if data['count'] > 0 else 0
            print(f"  {reason:<20} {data['count']:>5} trades  ${data['pnl']:>10,.2f}  (avg ${avg:.2f})")

        # Strategy breakdown
        strategy_stats = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0})
        for trade in trades:
            strat = trade.get('strategy', 'unknown')
            if strat.endswith('_SHORT'):
                strat = strat[:-6]
            strategy_stats[strat]['count'] += 1
            strategy_stats[strat]['pnl'] += trade['pnl']
            if trade['pnl'] > 0:
                strategy_stats[strat]['wins'] += 1

        print(f"\n{'STRATEGY BREAKDOWN':-^50}")
        for strat, data in sorted(strategy_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
            wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
            print(f"  {strat:<15} {data['count']:>5} trades  {wr:>5.1f}% WR  ${data['pnl']:>10,.2f}")

    print("\n" + "=" * 70)

    # Save results to JSON
    output_path = Path(__file__).parent / 'logs' / 'live_simulation_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'config': {
            'start_date': start_date,
            'end_date': end_date,
            'symbols_count': len(symbols),
            'scanner_top_n': scanner_top_n,
            'initial_capital': initial_capital,
            'longs_only': longs_only,
            'shorts_only': shorts_only,
        },
        'metrics': metrics,
        'trade_count': len(trades),
        'generated_at': datetime.now().isoformat(),
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run live trading simulation with scanner-selected stocks'
    )
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=365, help='Days to simulate (default: 365)')
    parser.add_argument('--top-n', type=int, help='Scanner top N (default: from config)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--longs-only', action='store_true', help='Only long positions')
    parser.add_argument('--shorts-only', action='store_true', help='Only short positions')

    args = parser.parse_args()

    run_simulation(
        start_date=args.start,
        end_date=args.end,
        days=args.days,
        top_n=args.top_n,
        longs_only=args.longs_only,
        shorts_only=args.shorts_only,
        initial_capital=args.capital,
    )


if __name__ == '__main__':
    main()
