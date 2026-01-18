#!/usr/bin/env python3
"""
Extended Validation for Promising Strategies

Tests H11 (Trend Structure) and H12 (Filtered Mean Reversion) with:
- More symbols (100+)
- Longer history (12 months)
- 12 walk-forward windows
- Stricter pass criteria
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')


def load_symbols(limit=100):
    """Load more symbols for robust testing"""
    universe_path = Path(__file__).parent / 'universe_original.yaml'
    with open(universe_path, 'r') as f:
        universe = yaml.safe_load(f)

    scanner = universe.get('scanner_universe', {})
    blacklist = universe.get('blacklist', [])

    symbols = []
    for cat, syms in scanner.items():
        if isinstance(syms, list):
            symbols.extend(syms)

    # Remove crypto and blacklisted
    symbols = [s for s in symbols if '/' not in s and s not in blacklist]
    symbols = list(dict.fromkeys(symbols))

    return symbols[:limit]


def fetch_data(symbols, start_date, end_date, interval='1h'):
    """Fetch OHLCV data"""
    print(f"  Fetching {interval} data for {len(symbols)} symbols...")

    all_data = {}
    successful = 0

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            if df is not None and len(df) >= 100:
                df = df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Volume': 'volume'
                })
                all_data[symbol] = df
                successful += 1
        except Exception:
            pass

    print(f"  Successfully loaded {successful}/{len(symbols)} symbols")
    return all_data


# =============================================================================
# H11: TREND STRUCTURE
# =============================================================================

def h11_trend_structure(df, swing_period=10):
    """Trend Structure Continuation - enter on pullbacks in established trends"""
    df = df.copy()

    df['swing_high'] = df['high'].rolling(swing_period, center=True).max()
    df['swing_low'] = df['low'].rolling(swing_period, center=True).min()

    df['is_swing_high'] = df['high'] == df['swing_high']
    df['is_swing_low'] = df['low'] == df['swing_low']

    df['prev_swing_high'] = df['high'].where(df['is_swing_high']).ffill().shift(1)
    df['prev_swing_low'] = df['low'].where(df['is_swing_low']).ffill().shift(1)

    df['higher_high'] = df['high'] > df['prev_swing_high']
    df['higher_low'] = df['low'] > df['prev_swing_low']
    df['uptrend_structure'] = df['higher_high'].rolling(swing_period).sum() > 0

    df['lower_high'] = df['high'] < df['prev_swing_high']
    df['lower_low'] = df['low'] < df['prev_swing_low']
    df['downtrend_structure'] = df['lower_low'].rolling(swing_period).sum() > 0

    df['pullback_in_uptrend'] = df['uptrend_structure'] & (df['close'] < df['close'].rolling(5).mean())
    df['bounce_in_downtrend'] = df['downtrend_structure'] & (df['close'] > df['close'].rolling(5).mean())

    df['long_entry'] = df['pullback_in_uptrend'] & (df['close'] > df['open'])
    df['short_entry'] = df['bounce_in_downtrend'] & (df['close'] < df['open'])

    return df


# =============================================================================
# H12: FILTERED MEAN REVERSION
# =============================================================================

def h12_filtered_mean_reversion(df, trend_period=100, oversold_rsi=30, overbought_rsi=70, bb_period=20):
    """Mean Reversion with Strong Trend Filter"""
    df = df.copy()

    df['sma_100'] = df['close'].rolling(trend_period).mean()
    df['strong_uptrend'] = df['close'] > df['sma_100']
    df['strong_downtrend'] = df['close'] < df['sma_100']

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['oversold'] = df['rsi'] < oversold_rsi
    df['overbought'] = df['rsi'] > overbought_rsi

    df['bb_mid'] = df['close'].rolling(bb_period).mean()
    df['bb_std'] = df['close'].rolling(bb_period).std()
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']

    df['at_lower_bb'] = df['close'] <= df['bb_lower']
    df['at_upper_bb'] = df['close'] >= df['bb_upper']

    df['long_entry'] = df['strong_uptrend'] & df['oversold'] & df['at_lower_bb']
    df['short_entry'] = df['strong_downtrend'] & df['overbought'] & df['at_upper_bb']

    return df


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(all_data, signal_func, hypothesis_name,
                 stop_loss_pct=0.05, take_profit_pct=0.05, max_hold_bars=48):
    """Run backtest for a hypothesis"""
    all_trades = []

    for symbol, df in all_data.items():
        if len(df) < 100:
            continue

        df = signal_func(df)
        position = None

        for i in range(50, len(df)):
            row = df.iloc[i]

            if position is not None:
                bars_held = i - position['entry_bar']
                exit_triggered = False
                exit_price = row['close']
                exit_reason = None

                if position['direction'] == 'long':
                    if row['low'] <= position['stop']:
                        exit_triggered = True
                        exit_price = position['stop']
                        exit_reason = 'stop_loss'
                    elif row['high'] >= position['take']:
                        exit_triggered = True
                        exit_price = position['take']
                        exit_reason = 'take_profit'
                else:
                    if row['high'] >= position['stop']:
                        exit_triggered = True
                        exit_price = position['stop']
                        exit_reason = 'stop_loss'
                    elif row['low'] <= position['take']:
                        exit_triggered = True
                        exit_price = position['take']
                        exit_reason = 'take_profit'

                if not exit_triggered and bars_held >= max_hold_bars:
                    exit_triggered = True
                    exit_reason = 'max_hold'

                if exit_triggered:
                    if position['direction'] == 'long':
                        pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']

                    all_trades.append({
                        'symbol': symbol,
                        'direction': position['direction'],
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'entry_time': df.index[position['entry_bar']]
                    })
                    position = None

            if position is None:
                entry_price = row['close']

                if row['long_entry']:
                    position = {
                        'direction': 'long',
                        'entry_price': entry_price,
                        'stop': entry_price * (1 - stop_loss_pct),
                        'take': entry_price * (1 + take_profit_pct),
                        'entry_bar': i
                    }
                elif row['short_entry']:
                    position = {
                        'direction': 'short',
                        'entry_price': entry_price,
                        'stop': entry_price * (1 + stop_loss_pct),
                        'take': entry_price * (1 - take_profit_pct),
                        'entry_bar': i
                    }

    return all_trades


def calculate_metrics(trades):
    """Calculate performance metrics"""
    if not trades:
        return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0, 'expectancy': 0}

    df = pd.DataFrame(trades)
    winners = df[df['pnl_pct'] > 0]
    losers = df[df['pnl_pct'] <= 0]

    total_trades = len(df)
    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

    gross_profit = winners['pnl_pct'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['pnl_pct'].sum()) if len(losers) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    expectancy = df['pnl_pct'].mean() * 100

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy
    }


def run_walk_forward(all_data, signal_func, n_windows=12):
    """Run walk-forward validation with N windows"""
    all_dates = []
    for symbol, df in all_data.items():
        all_dates.extend(df.index.tolist())

    min_date = min(all_dates)
    max_date = max(all_dates)
    total_days = (max_date - min_date).days
    window_days = total_days // n_windows

    window_results = []

    for i in range(n_windows):
        window_start = min_date + timedelta(days=i * window_days)
        window_end = min_date + timedelta(days=(i + 1) * window_days)

        window_data = {}
        for symbol, df in all_data.items():
            mask = (df.index >= window_start) & (df.index < window_end)
            window_df = df[mask]
            if len(window_df) >= 30:
                window_data[symbol] = window_df

        if window_data:
            trades = run_backtest(window_data, signal_func, "")
            metrics = calculate_metrics(trades)
            metrics['window'] = i + 1
            metrics['start'] = window_start.strftime('%Y-%m-%d')
            metrics['end'] = window_end.strftime('%Y-%m-%d')
            window_results.append(metrics)

    return window_results


def main():
    print("\n" + "="*70)
    print("  EXTENDED VALIDATION")
    print("  H11: Trend Structure + H12: Filtered Mean Reversion")
    print("="*70)

    # Configuration
    symbols = load_symbols(limit=150)

    # 12 months of hourly data (max available from Yahoo)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print(f"\n  Configuration:")
    print(f"  - Symbols: {len(symbols)} (extended universe)")
    print(f"  - Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  - Walk-forward: 12 monthly windows")
    print(f"  - Pass criteria: 8/12 positive windows, PF > 1.05")

    # Fetch data
    print(f"\n  Loading data...")
    all_data = fetch_data(symbols, start_date, end_date, interval='1h')

    if not all_data:
        print("  ERROR: No data loaded!")
        return

    strategies = [
        ('H11: Trend Structure', h11_trend_structure),
        ('H12: Filtered Mean Reversion', h12_filtered_mean_reversion),
    ]

    for name, func in strategies:
        print(f"\n\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

        # Full period test
        trades = run_backtest(all_data, func, name)
        metrics = calculate_metrics(trades)

        print(f"\n  Full Period Results:")
        print(f"    Trades: {metrics['total_trades']}")
        print(f"    Win Rate: {metrics['win_rate']:.1f}%")
        print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"    Expectancy: {metrics['expectancy']:.2f}%")

        # Walk-forward validation
        print(f"\n  Walk-Forward Validation (12 windows):")
        window_results = run_walk_forward(all_data, func, n_windows=12)

        positive_count = 0
        total_exp = 0
        all_pf = []

        for w in window_results:
            status = "+" if w['expectancy'] > 0 else "-"
            if w['expectancy'] > 0:
                positive_count += 1
            total_exp += w['expectancy']
            if w['profit_factor'] != float('inf') and w['total_trades'] > 0:
                all_pf.append(w['profit_factor'])

            print(f"    [{status}] W{w['window']:>2}: {w['start']} - {w['end']} | "
                  f"{w['total_trades']:>4} trades, WR: {w['win_rate']:>5.1f}%, "
                  f"PF: {w['profit_factor']:>5.2f}, Exp: {w['expectancy']:>6.2f}%")

        avg_pf = np.mean(all_pf) if all_pf else 0

        print(f"\n  Summary:")
        print(f"    Positive Windows: {positive_count}/12")
        print(f"    Average PF: {avg_pf:.2f}")
        print(f"    Total Expectancy: {total_exp:.2f}%")

        # Verdict
        if positive_count >= 8 and avg_pf > 1.05:
            verdict = "PROCEED TO LIVE PAPER TESTING"
        elif positive_count >= 6 and avg_pf > 1.0:
            verdict = "PROCEED WITH CAUTION - needs monitoring"
        else:
            verdict = "ARCHIVE"

        print(f"\n  VERDICT: {verdict}")

    print("\n" + "="*70)
    print("  EXTENDED VALIDATION COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
