#!/usr/bin/env python3
"""
Backtest Verification Tool

Compares your backtest results against a simple replay simulation.
This isolates whether discrepancies come from signal generation vs P&L calculation.

Approach:
1. Run your backtest to get actual trades (with timestamps, prices, exits)
2. Replay those exact trades with simple math (no complex exit logic)
3. Compare results - if they match, your backtest is trustworthy
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backtest import Backtest1Hour


def load_symbols(limit=None):
    """Load symbols from universe.yaml"""
    universe_path = Path(__file__).parent / 'universe.yaml'
    with open(universe_path, 'r') as f:
        universe = yaml.safe_load(f)

    scanner = universe.get('scanner_universe', {})
    symbols = []
    for category, syms in scanner.items():
        if isinstance(syms, list):
            symbols.extend(syms)

    symbols = list(set(symbols))
    if limit:
        symbols = symbols[:limit]
    return symbols


def run_backtest_and_get_trades(symbols, start_date, end_date):
    """Run your actual backtest and return trades"""
    print(f"\n{'='*70}")
    print("  STEP 1: Running Your Backtest")
    print(f"{'='*70}")
    print(f"  Symbols: {len(symbols)}")
    print(f"  Period: {start_date} to {end_date}")

    bt = Backtest1Hour(initial_capital=10000)
    results = bt.run(symbols, start_date, end_date)

    trades = results.get('trades', [])
    print(f"\n  Your backtest returned {len(trades)} trades")
    print(f"  Final capital: ${results.get('final_capital', 0):,.2f}")
    print(f"  Total return: {((results.get('final_capital', 10000) / 10000) - 1) * 100:.2f}%")

    return trades, results


def simple_replay_trades(trades, initial_capital=10000):
    """
    Replay trades with simple P&L math.

    This mimics what VectorBT does - just entry price, exit price, shares.
    No complex exit logic, no slippage adjustments.
    """
    print(f"\n{'='*70}")
    print("  STEP 2: Simple Replay (VectorBT-style)")
    print(f"{'='*70}")

    total_pnl = 0
    winning = 0
    losing = 0

    replay_trades = []

    for trade in trades:
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        shares = trade.get('shares', 0)
        direction = trade.get('direction', 'LONG')

        # Simple P&L calculation
        if direction == 'LONG':
            pnl = (exit_price - entry_price) * shares
        else:  # SHORT
            pnl = (entry_price - exit_price) * shares

        pnl_pct = ((exit_price / entry_price) - 1) * 100 if direction == 'LONG' else ((entry_price / exit_price) - 1) * 100

        replay_trades.append({
            'symbol': trade.get('symbol'),
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': shares,
            'backtest_pnl': trade.get('pnl', 0),
            'replay_pnl': pnl,
            'pnl_diff': pnl - trade.get('pnl', 0),
            'exit_reason': trade.get('exit_reason', '')
        })

        total_pnl += pnl
        if pnl > 0:
            winning += 1
        else:
            losing += 1

    final_capital = initial_capital + total_pnl
    total_return = ((final_capital / initial_capital) - 1) * 100
    win_rate = (winning / len(trades) * 100) if trades else 0

    print(f"  Replayed {len(trades)} trades")
    print(f"  Final capital: ${final_capital:,.2f}")
    print(f"  Total return: {total_return:.2f}%")
    print(f"  Win rate: {win_rate:.1f}%")

    return {
        'final_capital': final_capital,
        'total_return': total_return,
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'trades': replay_trades
    }


def compare_results(backtest_results, replay_results, trades):
    """Compare backtest vs replay results"""
    print(f"\n{'='*70}")
    print("  STEP 3: Comparison")
    print(f"{'='*70}")

    bt_final = backtest_results.get('final_capital', 10000)
    bt_return = ((bt_final / 10000) - 1) * 100

    rp_final = replay_results['final_capital']
    rp_return = replay_results['total_return']

    diff_dollars = bt_final - rp_final
    diff_pct = bt_return - rp_return

    print(f"\n  {'Metric':<25} {'Backtest':>15} {'Replay':>15} {'Diff':>15}")
    print(f"  {'-'*70}")
    print(f"  {'Final Capital':<25} ${bt_final:>14,.2f} ${rp_final:>14,.2f} ${diff_dollars:>14,.2f}")
    print(f"  {'Total Return':<25} {bt_return:>14.2f}% {rp_return:>14.2f}% {diff_pct:>14.2f}%")

    # Check for P&L discrepancies per trade
    replay_trades = replay_results['trades']
    discrepancies = [t for t in replay_trades if abs(t['pnl_diff']) > 0.01]

    print(f"\n  Trade-level P&L discrepancies: {len(discrepancies)} / {len(trades)}")

    if discrepancies:
        print(f"\n  Sample discrepancies (first 10):")
        print(f"  {'Symbol':<8} {'Direction':<6} {'Entry':>10} {'Exit':>10} {'BT P&L':>12} {'Replay P&L':>12} {'Diff':>10}")
        print(f"  {'-'*80}")
        for t in discrepancies[:10]:
            print(f"  {t['symbol']:<8} {t['direction']:<6} ${t['entry_price']:>9.2f} ${t['exit_price']:>9.2f} ${t['backtest_pnl']:>11.2f} ${t['replay_pnl']:>11.2f} ${t['pnl_diff']:>9.2f}")

    # Verdict
    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")

    if abs(diff_pct) < 1.0:
        print(f"\n  MATCH - Backtest and replay within 1% ({diff_pct:.2f}% difference)")
        print("  Your backtest P&L calculation is accurate.")
    elif abs(diff_pct) < 5.0:
        print(f"\n  CLOSE - Backtest and replay within 5% ({diff_pct:.2f}% difference)")
        print("  Small discrepancies likely from slippage/commission modeling.")
    else:
        print(f"\n  MISMATCH - Significant difference ({diff_pct:.2f}%)")
        print("  Investigate trade-level discrepancies above.")

    return diff_pct


def analyze_exit_reasons(trades):
    """Analyze exit reason distribution"""
    print(f"\n{'='*70}")
    print("  Exit Reason Analysis")
    print(f"{'='*70}")

    exit_reasons = {}
    exit_pnl = {}

    for trade in trades:
        reason = trade.get('exit_reason', 'unknown')
        pnl = trade.get('pnl', 0)

        if reason not in exit_reasons:
            exit_reasons[reason] = 0
            exit_pnl[reason] = 0

        exit_reasons[reason] += 1
        exit_pnl[reason] += pnl

    print(f"\n  {'Exit Reason':<20} {'Count':>10} {'Total P&L':>15} {'Avg P&L':>12}")
    print(f"  {'-'*60}")

    for reason in sorted(exit_reasons.keys()):
        count = exit_reasons[reason]
        total = exit_pnl[reason]
        avg = total / count if count > 0 else 0
        print(f"  {reason:<20} {count:>10} ${total:>14,.2f} ${avg:>11,.2f}")


def analyze_trade_details(trades):
    """Show detailed trade analysis"""
    print(f"\n{'='*70}")
    print("  Trade Details Analysis")
    print(f"{'='*70}")

    if not trades:
        print("  No trades to analyze")
        return

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(trades)

    print(f"\n  Total trades: {len(df)}")
    print(f"  Winning trades: {len(df[df['pnl'] > 0])}")
    print(f"  Losing trades: {len(df[df['pnl'] <= 0])}")

    if 'pnl' in df.columns:
        print(f"\n  P&L Statistics:")
        print(f"    Total P&L: ${df['pnl'].sum():,.2f}")
        print(f"    Avg P&L: ${df['pnl'].mean():,.2f}")
        print(f"    Max Win: ${df['pnl'].max():,.2f}")
        print(f"    Max Loss: ${df['pnl'].min():,.2f}")

    if 'bars_held' in df.columns:
        print(f"\n  Holding Period:")
        print(f"    Avg bars held: {df['bars_held'].mean():.1f}")
        print(f"    Max bars held: {df['bars_held'].max()}")
        print(f"    Min bars held: {df['bars_held'].min()}")

    # Show sample trades
    print(f"\n  Sample Trades (first 10):")
    print(f"  {'Symbol':<8} {'Dir':<5} {'Entry':>10} {'Exit':>10} {'P&L':>12} {'Reason':<15} {'Bars':>5}")
    print(f"  {'-'*75}")

    for _, t in df.head(10).iterrows():
        print(f"  {t.get('symbol', 'N/A'):<8} {t.get('direction', 'N/A'):<5} ${t.get('entry_price', 0):>9.2f} ${t.get('exit_price', 0):>9.2f} ${t.get('pnl', 0):>11.2f} {t.get('exit_reason', 'N/A'):<15} {t.get('bars_held', 0):>5}")


def main():
    print("\n" + "="*70)
    print("  BACKTEST VERIFICATION TOOL")
    print("="*70)

    # Setup - match your typical backtest parameters
    symbols = load_symbols(limit=100)  # Use 100 symbols for faster testing

    # 6 month period
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

    print(f"\n  Configuration:")
    print(f"    Symbols: {len(symbols)}")
    print(f"    Period: {start_date} to {end_date}")
    print(f"    Initial Capital: $10,000")

    # Step 1: Run your backtest
    trades, backtest_results = run_backtest_and_get_trades(symbols, start_date, end_date)

    if not trades:
        print("\n  ERROR: No trades generated. Check your backtest.")
        return

    # Step 2: Simple replay
    replay_results = simple_replay_trades(trades)

    # Step 3: Compare
    diff = compare_results(backtest_results, replay_results, trades)

    # Additional analysis
    analyze_exit_reasons(trades)
    analyze_trade_details(trades)

    print("\n" + "="*70)
    print("  VERIFICATION COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
