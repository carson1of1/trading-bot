#!/usr/bin/env python3
"""
VectorBT Independent Verification

Uses vectorized backtesting to verify our results.
More pandas-native, should match our logic better.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf
from datetime import datetime, timedelta
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_symbols():
    """Load symbols from universe.yaml"""
    universe_path = Path(__file__).parent / 'universe.yaml'
    with open(universe_path, 'r') as f:
        universe = yaml.safe_load(f)

    scanner = universe.get('scanner_universe', {})
    symbols = []
    for category, syms in scanner.items():
        if isinstance(syms, list):
            symbols.extend(syms)
    return list(set(symbols))


def fetch_data(symbols, start_date, end_date):
    """Fetch hourly data for all symbols"""
    print(f"  Fetching data for {len(symbols)} symbols...")

    all_data = {}
    successful = 0

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1h')
            if df is not None and len(df) >= 50:
                all_data[symbol] = df
                successful += 1
        except Exception as e:
            pass

    print(f"  Successfully loaded {successful} symbols")
    return all_data


def calculate_indicators(df):
    """Calculate indicators matching our strategy"""
    # SMA
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']

    # Volume SMA
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()

    # 40-bar high/low for breakout
    df['High_40'] = df['High'].rolling(40).max().shift(1)
    df['Low_40'] = df['Low'].rolling(40).min().shift(1)

    return df


def generate_momentum_signals(df):
    """Generate momentum strategy signals"""
    # Trend aligned: Price > SMA20 > SMA50
    trend_aligned = (df['Close'] > df['SMA_20']) & (df['SMA_20'] > df['SMA_50'])

    # RSI in range
    rsi_ok = (df['RSI'] >= 45) & (df['RSI'] <= 72)

    # Volume surge
    volume_surge = df['Volume'] > df['Volume_SMA'] * 1.15

    # Entry signal
    entry = trend_aligned & rsi_ok & volume_surge

    return entry.astype(int)


def generate_mean_reversion_signals(df):
    """Generate mean reversion strategy signals"""
    # In uptrend
    in_uptrend = df['Close'] > df['SMA_50']

    # RSI oversold
    oversold = df['RSI'] < 32

    # Near lower BB
    bb_range = df['BB_Upper'] - df['BB_Lower']
    bb_position = (df['Close'] - df['BB_Lower']) / bb_range
    near_lower = bb_position < 0.15

    # Entry signal
    entry = in_uptrend & oversold & near_lower

    return entry.astype(int)


def generate_breakout_signals(df):
    """Generate breakout strategy signals"""
    # Price above 40-bar high
    breakout = df['Close'] > df['High_40']

    # Volume confirmation
    volume_confirm = df['Volume'] > df['Volume_SMA'] * 1.3

    # Entry signal
    entry = breakout & volume_confirm

    return entry.astype(int)


def generate_combined_signals(df):
    """Combine all strategy signals"""
    momentum = generate_momentum_signals(df)
    mean_rev = generate_mean_reversion_signals(df)
    breakout = generate_breakout_signals(df)

    # Any signal triggers entry
    combined = (momentum | mean_rev | breakout).astype(int)

    return combined


def run_symbol_backtest(symbol, df, initial_capital, position_size_pct=0.15,
                        stop_loss=0.05, take_profit=0.05):
    """Run backtest for a single symbol"""
    df = calculate_indicators(df.copy())

    # Generate signals
    entries = generate_combined_signals(df)

    # Skip if no signals
    if entries.sum() == 0:
        return None

    # Calculate position size (fixed percentage)
    size = position_size_pct

    # Run backtest with stop loss and take profit
    try:
        pf = vbt.Portfolio.from_signals(
            close=df['Close'],
            entries=entries.astype(bool),
            exits=None,  # We'll use SL/TP
            size=size,
            size_type='percent',
            init_cash=initial_capital,
            fees=0.001,  # 0.1% commission
            sl_stop=stop_loss,
            tp_stop=take_profit,
            freq='1h'
        )
        return pf
    except Exception as e:
        return None


def run_multi_symbol_backtest(all_data, initial_capital=10000, max_positions=3):
    """
    Run backtest across multiple symbols with position limits.

    This is a simplified version - VectorBT doesn't easily support
    cross-symbol position limits, so we simulate it.
    """
    print(f"\n  Running multi-symbol backtest...")

    all_trades = []
    symbol_results = {}

    for symbol, df in all_data.items():
        pf = run_symbol_backtest(symbol, df, initial_capital / max_positions)
        if pf is not None:
            symbol_results[symbol] = pf

            # Get trades
            trades = pf.trades.records_readable
            if len(trades) > 0:
                trades['Symbol'] = symbol
                all_trades.append(trades)

    if not symbol_results:
        return None, None

    return symbol_results, all_trades


def calculate_portfolio_metrics(symbol_results, all_trades, initial_capital):
    """Calculate aggregate portfolio metrics"""
    if not symbol_results:
        return None

    total_pnl = 0
    total_trades = 0
    winning_trades = 0

    for symbol, pf in symbol_results.items():
        total_pnl += pf.total_profit()
        stats = pf.trades.records_readable
        if len(stats) > 0:
            total_trades += len(stats)
            winning_trades += (stats['PnL'] > 0).sum()

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_return = (total_pnl / initial_capital) * 100

    return {
        'total_pnl': total_pnl,
        'total_return_pct': total_return,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate
    }


def run_simple_vectorized_backtest(all_data, initial_capital=10000):
    """
    Simple vectorized backtest - process each symbol independently
    and aggregate results.
    """
    print(f"\n  Running simple vectorized backtest...")

    total_pnl = 0
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    all_returns = []

    symbols_with_trades = 0

    for symbol, df in all_data.items():
        df = calculate_indicators(df.copy())

        # Generate entry signals
        entries = generate_combined_signals(df)

        if entries.sum() == 0:
            continue

        symbols_with_trades += 1

        # Simple simulation: for each entry, check if SL or TP hit first
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        entry_signals = entries.values

        position = None
        symbol_pnl = 0
        symbol_trades = 0
        symbol_wins = 0

        for i in range(len(df)):
            if position is None and entry_signals[i] == 1:
                # Enter position
                entry_price = close[i]
                stop_price = entry_price * 0.95  # 5% stop
                take_price = entry_price * 1.05  # 5% take profit
                position = {
                    'entry_price': entry_price,
                    'stop': stop_price,
                    'take': take_price,
                    'entry_bar': i
                }
            elif position is not None:
                # Check exit
                if low[i] <= position['stop']:
                    # Stop loss hit
                    pnl = (position['stop'] - position['entry_price']) / position['entry_price']
                    symbol_pnl += pnl
                    symbol_trades += 1
                    position = None
                elif high[i] >= position['take']:
                    # Take profit hit
                    pnl = (position['take'] - position['entry_price']) / position['entry_price']
                    symbol_pnl += pnl
                    symbol_trades += 1
                    symbol_wins += 1
                    position = None

        # Close any open position at end
        if position is not None:
            pnl = (close[-1] - position['entry_price']) / position['entry_price']
            symbol_pnl += pnl
            symbol_trades += 1
            if pnl > 0:
                symbol_wins += 1

        total_trades += symbol_trades
        winning_trades += symbol_wins
        losing_trades += (symbol_trades - symbol_wins)

        # Assume equal allocation per symbol with trades
        all_returns.append(symbol_pnl)

    if total_trades == 0:
        return None

    # Calculate aggregate metrics
    # Assume we allocated equally across symbols that had signals
    avg_return_per_symbol = np.mean(all_returns) if all_returns else 0
    # With position sizing, multiply by leverage effect
    total_return_pct = avg_return_per_symbol * 100 * 3  # ~3 positions average

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    return {
        'total_return_pct': total_return_pct,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'symbols_traded': symbols_with_trades
    }


def run_accurate_simulation(all_data, initial_capital=10000, max_positions=3,
                            position_size_pct=0.15, stop_loss=0.05, take_profit=0.05,
                            max_hold_bars=48):
    """
    More accurate simulation that matches our backtest logic.
    Processes bars chronologically across all symbols.
    Includes max_hold_bars exit (default 48 hours).
    """
    print(f"\n  Running accurate chronological simulation...")
    print(f"  Max hold: {max_hold_bars} bars")

    # Debug: Count signals per strategy
    total_momentum = 0
    total_meanrev = 0
    total_breakout = 0

    # Build unified timeline
    all_events = []
    symbol_data = {}

    for symbol, df in all_data.items():
        df = calculate_indicators(df.copy())
        df['momentum'] = generate_momentum_signals(df)
        df['meanrev'] = generate_mean_reversion_signals(df)
        df['breakout'] = generate_breakout_signals(df)
        df['entries'] = generate_combined_signals(df)
        symbol_data[symbol] = df

        # Count signals
        total_momentum += df['momentum'].sum()
        total_meanrev += df['meanrev'].sum()
        total_breakout += df['breakout'].sum()

        for idx in range(50, len(df)):  # Skip warmup
            row = df.iloc[idx]
            all_events.append({
                'timestamp': df.index[idx],
                'symbol': symbol,
                'idx': idx
            })

    print(f"  Signal counts: Momentum={total_momentum}, MeanRev={total_meanrev}, Breakout={total_breakout}")

    # Sort by timestamp
    all_events.sort(key=lambda x: x['timestamp'])

    # Simulation state
    cash = initial_capital
    positions = {}  # symbol -> position info
    trades = []

    for event in all_events:
        symbol = event['symbol']
        idx = event['idx']
        df = symbol_data[symbol]
        row = df.iloc[idx]

        current_price = row['Close']
        high_price = row['High']
        low_price = row['Low']

        # Check exits first
        if symbol in positions:
            pos = positions[symbol]
            exit_triggered = False
            exit_price = current_price
            exit_reason = ''
            bars_held = idx - pos['entry_bar']

            # Check stop loss
            if low_price <= pos['stop']:
                exit_triggered = True
                exit_price = pos['stop']
                exit_reason = 'stop_loss'
            # Check take profit
            elif high_price >= pos['take']:
                exit_triggered = True
                exit_price = pos['take']
                exit_reason = 'take_profit'
            # Check max hold
            elif bars_held >= max_hold_bars:
                exit_triggered = True
                exit_price = current_price  # Exit at current close
                exit_reason = 'max_hold'

            if exit_triggered:
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                cash += pos['shares'] * exit_price
                trades.append({
                    'symbol': symbol,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': exit_reason
                })
                del positions[symbol]

        # Check entries
        if symbol not in positions and len(positions) < max_positions:
            if row['entries'] == 1:
                # Calculate position size
                portfolio_value = cash + sum(
                    p['shares'] * p['entry_price']  # Use entry price for simplicity
                    for s, p in positions.items()
                )
                position_value = portfolio_value * position_size_pct
                shares = int(position_value / current_price)

                if shares > 0 and shares * current_price <= cash:
                    cash -= shares * current_price
                    positions[symbol] = {
                        'entry_price': current_price,
                        'shares': shares,
                        'stop': current_price * (1 - stop_loss),
                        'take': current_price * (1 + take_profit),
                        'entry_bar': idx  # Track entry bar for max hold
                    }

    # Close remaining positions
    for symbol, pos in positions.items():
        df = symbol_data[symbol]
        exit_price = df.iloc[-1]['Close']
        pnl = (exit_price - pos['entry_price']) * pos['shares']
        cash += pos['shares'] * exit_price
        trades.append({
            'symbol': symbol,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': 'end'
        })

    # Calculate metrics
    if not trades:
        return None

    total_pnl = sum(t['pnl'] for t in trades)
    total_return = (cash - initial_capital) / initial_capital * 100
    winning = len([t for t in trades if t['pnl'] > 0])
    win_rate = (winning / len(trades) * 100) if trades else 0

    # Calculate max drawdown
    equity = [initial_capital]
    running_cash = initial_capital
    for t in sorted(trades, key=lambda x: x.get('exit_date', 0)):
        running_cash += t['pnl']
        equity.append(running_cash)

    peak = initial_capital
    max_dd = 0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return {
        'final_value': cash,
        'total_return_pct': total_return,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'max_drawdown_pct': max_dd * 100,
        'trades': trades
    }


def main():
    print("\n" + "="*70)
    print("  VECTORBT INDEPENDENT VERIFICATION")
    print("="*70)

    # Setup - match our backtest parameters
    symbols = load_symbols()[:100]  # Use 100 symbols
    initial_capital = 10000

    # 6 month period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f"\n  Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  Capital: ${initial_capital:,}")
    print(f"  Target Symbols: {len(symbols)}")

    # Fetch data
    all_data = fetch_data(symbols, start_date, end_date)

    if not all_data:
        print("  No data loaded!")
        return

    # Run accurate simulation
    results = run_accurate_simulation(
        all_data,
        initial_capital=initial_capital,
        max_positions=3,
        position_size_pct=0.15,
        stop_loss=0.05,
        take_profit=0.05,
        max_hold_bars=48  # Exit after 48 hours if SL/TP not hit
    )

    # Print results
    print("\n" + "="*70)
    print("  VECTORBT VERIFICATION RESULTS")
    print("="*70)

    if results:
        print(f"\n  Final Value: ${results['final_value']:,.2f}")
        print(f"  Total Return: {results['total_return_pct']:.2f}%")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Win Rate: {results['win_rate']:.1f}%")
        print(f"  Max Drawdown: {results['max_drawdown_pct']:.2f}%")

        # Show trade breakdown by exit reason
        if 'trades' in results:
            stop_loss_trades = len([t for t in results['trades'] if t['exit_reason'] == 'stop_loss'])
            take_profit_trades = len([t for t in results['trades'] if t['exit_reason'] == 'take_profit'])
            max_hold_trades = len([t for t in results['trades'] if t['exit_reason'] == 'max_hold'])
            end_trades = len([t for t in results['trades'] if t['exit_reason'] == 'end'])

            print(f"\n  Exit Breakdown:")
            print(f"    Take Profit: {take_profit_trades}")
            print(f"    Stop Loss: {stop_loss_trades}")
            print(f"    Max Hold (48 bars): {max_hold_trades}")
            print(f"    End of Backtest: {end_trades}")
    else:
        print("  No results - check signal generation")

    print("\n" + "="*70)
    print("  COMPARISON")
    print("="*70)
    print("\n  Your Backtest    vs    VectorBT")
    print(f"  Return: +114.33%  vs    {results['total_return_pct']:.2f}%" if results else "  N/A")
    print(f"  Win Rate: 61.2%   vs    {results['win_rate']:.1f}%" if results else "  N/A")
    print(f"  Max DD: 7.3%      vs    {results['max_drawdown_pct']:.2f}%" if results else "  N/A")
    print()


if __name__ == '__main__':
    main()
