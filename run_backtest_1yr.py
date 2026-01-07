#!/usr/bin/env python3
"""
1-Year Backtest with Scanner
Tests DailyDrawdownGuard integration with $100k capital.
"""

from datetime import datetime, timedelta
from backtest import Backtest1Hour
import yaml
from pathlib import Path


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def main():
    # Date range: 1 year ending today
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    print(f"=" * 60)
    print(f"1-Year Backtest with Scanner")
    print(f"=" * 60)
    print(f"Period: {start_date} to {end_date}")
    print(f"Capital: $100,000")
    print(f"Scanner: Enabled")
    print(f"=" * 60)

    # Load symbols from universe
    universe_path = Path(__file__).parent / 'universe.yaml'
    with open(universe_path, 'r') as f:
        universe = yaml.safe_load(f)

    # Use scanner_universe for volatility scanner simulation
    # scanner_universe is nested dict with categories as keys
    scanner_universe = universe.get('scanner_universe', {})
    if isinstance(scanner_universe, dict):
        symbols = []
        for category_symbols in scanner_universe.values():
            if isinstance(category_symbols, list):
                symbols.extend(category_symbols)
    else:
        symbols = scanner_universe if isinstance(scanner_universe, list) else []

    # Fallback to proven_symbols if empty
    if not symbols:
        symbols = universe.get('proven_symbols', [])

    print(f"Universe: {len(symbols)} symbols")

    # Load base config from config.yaml
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override specific settings for this test
    overrides = {
        'risk_management': {
            'max_position_size_pct': 10,
            'max_open_positions': 3,
            'stop_loss_pct': 5.0,
            'max_daily_loss_pct': 5.0,
        },
        'daily_drawdown_guard': {
            'enabled': True,
            'warning_pct': 2.0,
            'soft_limit_pct': 2.5,
            'hard_limit_pct': 4.0,
            'warning_size_multiplier': 0.5,
        },
        'volatility_scanner': {
            'enabled': True,
            'top_n': 10,
        }
    }
    config = deep_merge(config, overrides)

    bt = Backtest1Hour(initial_capital=100000, config=config)
    
    print(f"\nDailyDrawdownGuard Settings:")
    print(f"  Warning (reduce size): {bt.drawdown_guard.warning_pct*100:.1f}%")
    print(f"  Soft Limit (block entries): {bt.drawdown_guard.soft_limit_pct*100:.1f}%")
    print(f"  Hard Limit (liquidate): {bt.drawdown_guard.hard_limit_pct*100:.1f}%")
    print(f"\nRunning backtest...\n")
    
    # Run backtest
    results = bt.run(symbols, start_date, end_date)
    
    if results:
        metrics = results.get('metrics', {})
        trades = results.get('trades', [])
        
        print(f"\n{'=' * 60}")
        print(f"RESULTS")
        print(f"{'=' * 60}")
        print(f"Final Value:     ${metrics.get('final_value', 0):,.2f}")
        print(f"Total Return:    {metrics.get('total_return_pct', 0):.2f}%")
        print(f"Total P&L:       ${metrics.get('total_pnl', 0):,.2f}")
        print(f"{'=' * 60}")
        print(f"Total Trades:    {metrics.get('total_trades', 0)}")
        print(f"Win Rate:        {metrics.get('win_rate', 0):.1f}%")
        print(f"Profit Factor:   {metrics.get('profit_factor', 0):.2f}")
        print(f"Max Drawdown:    {metrics.get('max_drawdown', 0):.2f}%")
        print(f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"{'=' * 60}")
        
        # Check for drawdown guard triggers
        guard_exits = [t for t in trades if 'liquidation' in t.get('exit_reason', '')]
        if guard_exits:
            print(f"\nDrawdown Guard Triggers: {len(guard_exits)}")
            for t in guard_exits[:5]:
                print(f"  {t['exit_date']}: {t['symbol']} - {t['exit_reason']}")
        else:
            print(f"\nDrawdown Guard: No forced liquidations triggered")
        
        # Daily loss analysis
        if trades:
            from collections import defaultdict
            daily_pnl = defaultdict(float)
            for t in trades:
                day = str(t['exit_date'])[:10]
                daily_pnl[day] += t['pnl']
            
            worst_day = min(daily_pnl.items(), key=lambda x: x[1])
            best_day = max(daily_pnl.items(), key=lambda x: x[1])
            
            print(f"\nDaily P&L Analysis:")
            print(f"  Worst Day: {worst_day[0]} (${worst_day[1]:,.2f})")
            print(f"  Best Day:  {best_day[0]} (${best_day[1]:,.2f})")
            
            # Count days exceeding thresholds
            days_over_2pct = sum(1 for d, pnl in daily_pnl.items() if pnl < -2000)
            days_over_3pct = sum(1 for d, pnl in daily_pnl.items() if pnl < -3000)
            days_over_4pct = sum(1 for d, pnl in daily_pnl.items() if pnl < -4000)
            
            print(f"\n  Days with >2% loss: {days_over_2pct}")
            print(f"  Days with >3% loss: {days_over_3pct}")
            print(f"  Days with >4% loss: {days_over_4pct}")
    else:
        print("Backtest returned no results")

if __name__ == '__main__':
    main()
