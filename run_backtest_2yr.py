#!/usr/bin/env python3
"""
2-Year Backtest Script - Mirrors UI Functionality

Runs a comprehensive backtest with all features matching the UI:
- Volatility scanner (top N daily)
- DailyDrawdownGuard protection
- Full strategy ensemble
- Detailed metrics and analysis
"""

from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import yaml

from backtest import Backtest1Hour


def load_config() -> dict:
    """Load config from config.yaml."""
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_universe() -> dict:
    """Load universe from universe.yaml."""
    universe_path = Path(__file__).parent / 'universe.yaml'
    with open(universe_path, 'r') as f:
        return yaml.safe_load(f)


def collect_scanner_symbols(universe: dict) -> list:
    """Flatten scanner_universe dict into list of symbols."""
    scanner_universe = universe.get('scanner_universe', {})
    if isinstance(scanner_universe, dict):
        symbols = []
        for category_symbols in scanner_universe.values():
            if isinstance(category_symbols, list):
                symbols.extend(category_symbols)
        return symbols
    elif isinstance(scanner_universe, list):
        return scanner_universe
    return []


def main():
    # Configuration - matches UI defaults
    INITIAL_CAPITAL = 10000.0
    DAYS = 730  # 2 years
    TOP_N = 10  # Scanner picks top 10 volatile stocks daily
    LONGS_ONLY = False
    SHORTS_ONLY = False

    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=DAYS)).strftime('%Y-%m-%d')

    print("=" * 70)
    print("2-YEAR BACKTEST - Full UI Configuration")
    print("=" * 70)
    print(f"Period:          {start_date} to {end_date} ({DAYS} days)")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Scanner:         Top {TOP_N} volatile stocks daily")
    print(f"Position Filter: {'Longs Only' if LONGS_ONLY else 'Shorts Only' if SHORTS_ONLY else 'Both Long & Short'}")
    print("=" * 70)

    # Load config and universe
    config = load_config()
    universe = load_universe()

    # Configure scanner
    if "volatility_scanner" not in config:
        config["volatility_scanner"] = {}
    config["volatility_scanner"]["enabled"] = True
    config["volatility_scanner"]["top_n"] = TOP_N

    # Collect symbols
    symbols = collect_scanner_symbols(universe)
    print(f"Universe:        {len(symbols)} symbols across all categories")

    if not symbols:
        print("ERROR: No symbols found in scanner_universe")
        return

    # Create backtest instance
    bt = Backtest1Hour(
        initial_capital=INITIAL_CAPITAL,
        config=config,
        longs_only=LONGS_ONLY,
        shorts_only=SHORTS_ONLY
    )

    # Show active settings
    print(f"\nStrategy Settings:")
    for strat in bt.strategy_manager.strategies:
        status = "ENABLED" if strat.enabled else "disabled"
        print(f"  - {strat.name}: {status}")

    print(f"\nRisk Settings:")
    print(f"  Max Positions:     {bt.max_open_positions}")
    print(f"  Stop Loss:         {bt.default_stop_loss_pct * 100:.1f}%")
    print(f"  Take Profit:       {bt.default_take_profit_pct * 100:.1f}%")
    print(f"  Max Hold Hours:    {bt.max_hold_hours}")

    print(f"\nDrawdown Guard:")
    print(f"  Warning:           {bt.drawdown_guard.warning_pct * 100:.1f}%")
    print(f"  Soft Limit:        {bt.drawdown_guard.soft_limit_pct * 100:.1f}%")
    print(f"  Hard Limit:        {bt.drawdown_guard.hard_limit_pct * 100:.1f}%")

    print(f"\nExit Manager:")
    print(f"  Hard Stop:         -{bt.exit_manager.hard_stop_pct * 100:.2f}%")
    print(f"  Profit Floor:      +{bt.exit_manager.profit_floor_activation_pct * 100:.2f}% -> +{bt.exit_manager.profit_floor_lock_pct * 100:.2f}%")
    print(f"  Trailing Activation: +{bt.exit_manager.trailing_activation_pct * 100:.2f}%")

    print(f"\n{'=' * 70}")
    print("Running backtest... (this may take several minutes)")
    print("=" * 70)

    # Run backtest
    results = bt.run(symbols, start_date, end_date)

    if not results:
        print("\nERROR: Backtest returned no results")
        return

    # Extract results
    metrics = results.get('metrics', {})
    trades = results.get('trades', [])
    equity_curve = results.get('equity_curve', [])

    # Print main metrics
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print("=" * 70)
    print(f"Initial Capital: ${metrics.get('initial_capital', INITIAL_CAPITAL):,.2f}")
    print(f"Final Capital:   ${metrics.get('final_value', 0):,.2f}")
    print(f"Total Return:    {metrics.get('total_return_pct', 0):.2f}%")
    print(f"Total P&L:       ${metrics.get('total_pnl', 0):,.2f}")
    print("=" * 70)
    print(f"Total Trades:    {metrics.get('total_trades', 0)}")
    print(f"Winning Trades:  {metrics.get('winning_trades', 0)}")
    print(f"Losing Trades:   {metrics.get('losing_trades', 0)}")
    print(f"Win Rate:        {metrics.get('win_rate', 0):.1f}%")
    print(f"Profit Factor:   {metrics.get('profit_factor', 0):.2f}")
    print("=" * 70)
    print(f"Avg P&L:         ${metrics.get('avg_pnl', 0):.2f}")
    print(f"Avg Win:         ${metrics.get('avg_win', 0):.2f}")
    print(f"Avg Loss:        ${metrics.get('avg_loss', 0):.2f}")
    print(f"Max Drawdown:    {metrics.get('max_drawdown', 0):.2f}%")
    print(f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.2f}")
    print("=" * 70)

    if not trades:
        print("\nNo trades executed.")
        return

    # Analyze by exit reason
    exit_reasons = defaultdict(lambda: {'count': 0, 'pnl': 0})
    for t in trades:
        reason = t.get('exit_reason', 'unknown')
        exit_reasons[reason]['count'] += 1
        exit_reasons[reason]['pnl'] += t.get('pnl', 0)

    print(f"\nExit Reason Breakdown:")
    print("-" * 50)
    for reason, data in sorted(exit_reasons.items(), key=lambda x: -x[1]['count']):
        avg_pnl = data['pnl'] / data['count'] if data['count'] > 0 else 0
        print(f"  {reason:25s}: {data['count']:4d} trades, ${data['pnl']:>10,.2f} (avg ${avg_pnl:>7,.2f})")

    # Analyze by symbol (top 10)
    symbol_stats = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0})
    for t in trades:
        sym = t.get('symbol', 'UNKNOWN')
        symbol_stats[sym]['count'] += 1
        symbol_stats[sym]['pnl'] += t.get('pnl', 0)
        if t.get('pnl', 0) > 0:
            symbol_stats[sym]['wins'] += 1

    print(f"\nTop 10 Symbols by P&L:")
    print("-" * 60)
    sorted_symbols = sorted(symbol_stats.items(), key=lambda x: -x[1]['pnl'])[:10]
    for sym, data in sorted_symbols:
        win_rate = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
        print(f"  {sym:6s}: {data['count']:3d} trades, ${data['pnl']:>10,.2f}, Win Rate: {win_rate:.1f}%")

    print(f"\nBottom 10 Symbols by P&L:")
    print("-" * 60)
    bottom_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'])[:10]
    for sym, data in bottom_symbols:
        win_rate = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
        print(f"  {sym:6s}: {data['count']:3d} trades, ${data['pnl']:>10,.2f}, Win Rate: {win_rate:.1f}%")

    # Analyze by strategy
    strategy_stats = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0})
    for t in trades:
        strat = t.get('strategy', 'Unknown')
        strategy_stats[strat]['count'] += 1
        strategy_stats[strat]['pnl'] += t.get('pnl', 0)
        if t.get('pnl', 0) > 0:
            strategy_stats[strat]['wins'] += 1

    print(f"\nStrategy Performance:")
    print("-" * 60)
    for strat, data in sorted(strategy_stats.items(), key=lambda x: -x[1]['pnl']):
        win_rate = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
        avg_pnl = data['pnl'] / data['count'] if data['count'] > 0 else 0
        print(f"  {strat:25s}: {data['count']:4d} trades, ${data['pnl']:>10,.2f}, Win: {win_rate:.1f}%, Avg: ${avg_pnl:.2f}")

    # Daily P&L analysis
    daily_pnl = defaultdict(float)
    for t in trades:
        day = str(t.get('exit_date', ''))[:10]
        daily_pnl[day] += t.get('pnl', 0)

    if daily_pnl:
        worst_day = min(daily_pnl.items(), key=lambda x: x[1])
        best_day = max(daily_pnl.items(), key=lambda x: x[1])
        avg_daily_pnl = sum(daily_pnl.values()) / len(daily_pnl)

        print(f"\nDaily P&L Analysis:")
        print("-" * 50)
        print(f"  Trading Days:  {len(daily_pnl)}")
        print(f"  Best Day:      {best_day[0]} (${best_day[1]:,.2f})")
        print(f"  Worst Day:     {worst_day[0]} (${worst_day[1]:,.2f})")
        print(f"  Avg Daily P&L: ${avg_daily_pnl:,.2f}")

        # Days exceeding loss thresholds
        days_over_1pct = sum(1 for pnl in daily_pnl.values() if pnl < -INITIAL_CAPITAL * 0.01)
        days_over_2pct = sum(1 for pnl in daily_pnl.values() if pnl < -INITIAL_CAPITAL * 0.02)
        days_over_3pct = sum(1 for pnl in daily_pnl.values() if pnl < -INITIAL_CAPITAL * 0.03)
        days_over_4pct = sum(1 for pnl in daily_pnl.values() if pnl < -INITIAL_CAPITAL * 0.04)

        print(f"\n  Days with >1% loss: {days_over_1pct}")
        print(f"  Days with >2% loss: {days_over_2pct}")
        print(f"  Days with >3% loss: {days_over_3pct}")
        print(f"  Days with >4% loss: {days_over_4pct}")

    # Drawdown guard triggers
    guard_exits = [t for t in trades if 'liquidation' in t.get('exit_reason', '').lower()]
    if guard_exits:
        print(f"\nDrawdown Guard Triggers: {len(guard_exits)}")
        print("-" * 50)
        for t in guard_exits[:10]:
            print(f"  {t['exit_date']}: {t['symbol']} - {t['exit_reason']} (${t['pnl']:.2f})")
        if len(guard_exits) > 10:
            print(f"  ... and {len(guard_exits) - 10} more")
    else:
        print(f"\nDrawdown Guard: No forced liquidations triggered")

    # Monthly breakdown
    monthly_pnl = defaultdict(float)
    monthly_trades = defaultdict(int)
    for t in trades:
        month = str(t.get('exit_date', ''))[:7]  # YYYY-MM
        monthly_pnl[month] += t.get('pnl', 0)
        monthly_trades[month] += 1

    if monthly_pnl:
        print(f"\nMonthly Performance:")
        print("-" * 60)
        for month in sorted(monthly_pnl.keys()):
            pnl = monthly_pnl[month]
            count = monthly_trades[month]
            print(f"  {month}: {count:3d} trades, ${pnl:>10,.2f}")

    print(f"\n{'=' * 70}")
    print("Backtest complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
