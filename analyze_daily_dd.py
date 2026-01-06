"""
Calculate ACTUAL daily drawdowns from trade P&L, not equity curve.
The equity curve has per-symbol data which causes false DD readings.
"""
import sys
from pathlib import Path
import pandas as pd
import yaml
from datetime import datetime
from backtest import Backtest1Hour

# Load symbols from scanner_universe (400 symbols)
bot_dir = Path(__file__).parent
with open(bot_dir / 'universe.yaml', 'r') as f:
    universe = yaml.safe_load(f)
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

def analyze_actual_daily_dd(capital=100000, year='2025'):
    print(f"\n{'=' * 60}")
    print(f"ACTUAL DAILY DRAWDOWN ANALYSIS: ${capital:,} ({year})")
    print(f"{'=' * 60}")

    bt = Backtest1Hour(initial_capital=float(capital))
    results = bt.run(symbols=symbols, start_date=f'{year}-01-01', end_date=f'{year}-12-31')

    trades = results.get('trades', [])
    if not trades:
        print("ERROR: No trades")
        return

    # Convert to DataFrame
    df = pd.DataFrame(trades)
    print(f"\nTotal trades: {len(df)}")

    # Extract exit date - backtest already has exit_date column
    if 'exit_date' not in df.columns:
        if 'exit_time' in df.columns:
            df['exit_date'] = pd.to_datetime(df['exit_time']).dt.date
        elif 'exit_timestamp' in df.columns:
            df['exit_date'] = pd.to_datetime(df['exit_timestamp']).dt.date
        else:
            print("ERROR: No exit date column found")
            print(f"Columns: {df.columns.tolist()}")
            return
    else:
        # Convert to date if it's a string
        df['exit_date'] = pd.to_datetime(df['exit_date']).dt.date

    # Get P&L column
    pnl_col = None
    for col in ['pnl', 'realized_pnl', 'profit']:
        if col in df.columns:
            pnl_col = col
            break

    if not pnl_col:
        print("ERROR: No P&L column found")
        print(f"Columns: {df.columns.tolist()}")
        return

    # Sum daily P&L from actual trades
    daily_pnl = df.groupby('exit_date')[pnl_col].sum().reset_index()
    daily_pnl.columns = ['date', 'daily_pnl']

    # Calculate daily DD as % of capital
    # Note: For a running account, we'd track cumulative.
    # For simplicity, calculate as % of starting capital.
    daily_pnl['daily_dd_pct'] = (daily_pnl['daily_pnl'] / capital) * 100

    # Track cumulative equity and daily DD from peak
    daily_pnl = daily_pnl.sort_values('date')
    cumulative = capital
    peak = capital
    daily_dd_from_peak = []

    for _, row in daily_pnl.iterrows():
        cumulative += row['daily_pnl']
        if cumulative > peak:
            peak = cumulative
        dd_from_peak = ((peak - cumulative) / peak) * 100
        daily_dd_from_peak.append(dd_from_peak)

    daily_pnl['dd_from_peak_pct'] = daily_dd_from_peak

    # Results
    print(f"\n--- Daily P&L Summary ---")
    print(f"Trading days: {len(daily_pnl)}")

    # Worst single-day losses
    worst_days = daily_pnl.nsmallest(5, 'daily_pnl')
    print(f"\nWorst Single-Day Losses:")
    for _, row in worst_days.iterrows():
        print(f"  {row['date']}: ${row['daily_pnl']:+,.2f} ({row['daily_dd_pct']:+.2f}%)")

    # Best single-day gains
    best_days = daily_pnl.nlargest(5, 'daily_pnl')
    print(f"\nBest Single-Day Gains:")
    for _, row in best_days.iterrows():
        print(f"  {row['date']}: ${row['daily_pnl']:+,.2f} ({row['daily_dd_pct']:+.2f}%)")

    # Daily loss distribution
    print(f"\n--- Daily Loss Distribution ---")
    for thresh in [1, 2, 3, 4, 5]:
        # Days where we lost more than X% of capital
        count = len(daily_pnl[daily_pnl['daily_dd_pct'] <= -thresh])
        print(f"  Days losing >= {thresh}% of capital: {count}")

    # Funded account check
    death_days = daily_pnl[daily_pnl['daily_dd_pct'] <= -5]
    if len(death_days) > 0:
        print(f"\n⚠️  FUNDED ACCOUNT DANGER: {len(death_days)} days with 5%+ loss")
        for _, row in death_days.iterrows():
            print(f"    {row['date']}: ${row['daily_pnl']:+,.2f} ({row['daily_dd_pct']:.2f}%)")
    else:
        worst_daily = daily_pnl['daily_dd_pct'].min()
        print(f"\n✓ FUNDED ACCOUNT SAFE: Worst daily loss was {worst_daily:.2f}%")

    # Final results
    total_pnl = daily_pnl['daily_pnl'].sum()
    final_equity = capital + total_pnl
    max_dd_from_peak = daily_pnl['dd_from_peak_pct'].max()

    print(f"\n--- Overall Results ---")
    print(f"Starting capital: ${capital:,.2f}")
    print(f"Final equity:     ${final_equity:,.2f}")
    print(f"Total P&L:        ${total_pnl:+,.2f} ({(total_pnl/capital)*100:+.2f}%)")
    print(f"Max DD from peak: {max_dd_from_peak:.2f}%")

    return daily_pnl

# Run analysis
print("Calculating ACTUAL daily drawdowns from trades...")
analyze_actual_daily_dd(10000, '2025')
