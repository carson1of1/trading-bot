# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python trading bot for live and paper trading using the Alpaca API. It operates on 1-hour bars and uses multiple trading strategies (Momentum, Mean Reversion, Breakout) to generate signals. The bot supports backtesting, paper trading, and live trading modes.

## Commands

### Running the Bot
```bash
python3 bot.py                           # Run with default config.yaml
python3 bot.py --config custom.yaml      # Run with custom config
```

### Running Backtests
```bash
python3 backtest.py                                    # Uses universe.yaml symbols
python3 backtest.py --symbols AAPL MSFT SPY            # Specific symbols
python3 backtest.py --symbols SPY --start 2025-11-01 --end 2025-12-15
python3 backtest.py --longs-only                       # Only LONG positions
python3 backtest.py --shorts-only                      # Only SHORT positions
```

### Running Tests
```bash
python3 -m pytest                        # Run all tests
python3 -m pytest tests/test_bot.py      # Run single test file
python3 -m pytest tests/test_bot.py::TestCheckEntry -v   # Run single test class
python3 -m pytest -k "test_check_exit"   # Run tests matching pattern
```

## Architecture

### Entry Points
- **bot.py**: `TradingBot` class - Live/paper trading engine. Runs hourly cycles: sync account → check exits → check entries
- **backtest.py**: `Backtest1Hour` class - Historical trade simulation with realistic slippage, commission, and tiered exits

### Core Module (`core/`)
| File | Purpose |
|------|---------|
| `broker.py` | `BrokerInterface` ABC with `AlpacaBroker` (real) and `FakeBroker` (simulation) implementations. `BrokerFactory.create_broker()` selects based on mode |
| `config.py` | `GlobalConfig` singleton loads config.yaml. Access via `get_global_config()` |
| `risk.py` | `RiskManager` for position sizing, stop/take-profit calculation. `ExitManager` for tiered exit logic (profit floor, ATR trailing, partial take-profit) |
| `entry_gate.py` | `EntryGate` filters entries by confidence threshold, cooldowns, daily loss limits |
| `data.py` | `YFinanceDataFetcher` for historical OHLCV data |
| `indicators.py` | `TechnicalIndicators` adds SMA, RSI, MACD, ATR, Bollinger Bands to DataFrames |
| `scanner.py` | `VolatilityScanner` dynamically selects stocks by ATR/volume metrics |

### Strategies Module (`strategies/`)
- **base.py**: `TradingStrategy` ABC - all strategies implement `calculate_signal(symbol, data, current_price, indicators) -> dict`
- **manager.py**: `StrategyManager` aggregates signals from all enabled strategies, returns highest confidence
- **momentum.py**, **mean_reversion.py**, **breakout.py**: Concrete strategy implementations

### Key Data Flow
1. `TradingBot.run_trading_cycle()` fetches data via `YFinanceDataFetcher`
2. `TechnicalIndicators.add_all_indicators()` enriches DataFrame
3. `StrategyManager.get_best_signal()` evaluates all strategies
4. `EntryGate.check_entry_allowed()` filters low-quality entries
5. `RiskManager.calculate_position_size()` determines shares to buy
6. `BrokerInterface.submit_order()` executes trade
7. `ExitManager.evaluate_exit()` checks exit conditions each cycle

### Configuration
- **config.yaml**: Main config (mode, risk params, strategy weights, exit thresholds)
- **universe.yaml**: Stock watchlist (`proven_symbols` list)
- **.env**: Alpaca API keys (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`)

### Trading Modes (set in config.yaml `mode:`)
- `PAPER`: Paper trading via Alpaca paper API
- `LIVE`: Real money trading via Alpaca live API
- `BACKTEST`: Historical simulation with `FakeBroker`
- `DRY_RUN`: Simulation with fake broker, no API calls

## Key Patterns

### Signal Format
Strategies return: `{'action': 'BUY'|'SELL'|'HOLD', 'confidence': 0-100, 'strategy': str, 'reasoning': str, 'components': dict}`

### Exit Tiers (ExitManager)
1. **Hard Stop**: -2% from entry (always active)
2. **Profit Floor**: Locks +0.5% profit after reaching +1.25%
3. **ATR Trailing**: Activates at +1.75%, trails by ATR × 2.0
4. **Partial Take-Profit**: Closes 50% at +2%

### Broker Abstraction
Tests mock `create_broker()` to inject `FakeBroker`. Real code uses `BrokerFactory.create_broker()` which reads mode from config.
