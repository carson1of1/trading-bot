# Multi-Timeframe Backtest System Design

**Date:** 2026-01-13
**Status:** Approved
**Goal:** Compare strategy performance across different timeframes (5min, 15min, 1hour, 4hour, 1day) using TradeLocker's 3-month historical data.

---

## Overview

Build a configurable backtest system that runs the same trading strategy across multiple timeframes to identify optimal settings. TradeLocker provides 3 months of intraday data (vs yfinance's 7-30 days), enabling meaningful backtests on faster timeframes.

### TradeLocker Data Availability

| Timeframe | Bars  | Days | 3-Month Backtest |
|-----------|-------|------|------------------|
| 5 min     | 4,556 | 90   | Yes              |
| 15 min    | 1,560 | 90   | Yes              |
| 1 hour    | 422   | 90   | Yes              |
| 4 hour    | 168   | 89   | Yes              |
| 1 day     | 61    | 89   | Yes              |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    backtest_multi.py                        │
│              (CLI entry point)                              │
│  python backtest_multi.py --timeframes 5min 15min 1hour     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 MultiTimeframeBacktest                      │
│  - Orchestrates backtests across timeframes                 │
│  - Collects results and generates comparison                │
│  - Exports to CSV/JSON                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   BacktestEngine                            │
│  (Refactored from Backtest1Hour)                            │
│  - Timeframe-agnostic simulation                            │
│  - Accepts timeframe parameter                              │
│  - Auto-scales strategy parameters                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 core/tradelocker.py                         │
│  - TradeLockerDataFetcher class                             │
│  - 3-month intraday data (5min, 15min, 1hour, 4hour, 1day)  │
│  - Falls back to YFinance if TradeLocker unavailable        │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. TradeLocker Data Fetcher (`core/tradelocker.py`)

**Authentication (.env):**
```bash
TRADELOCKER_EMAIL=your@email.com
TRADELOCKER_PASSWORD=your_password
TRADELOCKER_SERVER=demo  # or 'live'
TRADELOCKER_ACCOUNT_ID=your_account_id
```

**Configuration (config.yaml):**
```yaml
data_source:
  backtest: tradelocker  # 'tradelocker' or 'yfinance'
  live: yfinance         # Keep yfinance for live trading

tradelocker:
  cache_enabled: true
  cache_dir: data/cache/tradelocker
  rate_limit_ms: 100
```

**Timeframe mapping:**

| Backtest Input | TradeLocker Resolution |
|----------------|------------------------|
| 5min           | D5 (5 minutes)         |
| 15min          | D15 (15 minutes)       |
| 1hour          | H1 (1 hour)            |
| 4hour          | H4 (4 hours)           |
| 1day           | D1 (1 day)             |

**Class interface:**
```python
class TradeLockerDataFetcher:
    def __init__(self):
        # Load credentials from .env
        # Authenticate with TradeLocker API
        # Initialize disk cache

    def get_historical_data_range(self, symbol, timeframe, start_date, end_date):
        # Map timeframe to TradeLocker resolution
        # Check disk cache first
        # Fetch from API if not cached
        # Return DataFrame: timestamp, open, high, low, close, volume

    def get_available_symbols(self):
        # Return list of tradeable symbols
```

### 2. Timeframe Scaler (`core/timeframe_scaler.py`)

Auto-scales indicator parameters based on timeframe ratio relative to 1-hour baseline.

**Scaling factors:**

| Timeframe | Bars per Hour | Scale Factor |
|-----------|---------------|--------------|
| 5min      | 12            | 12x          |
| 15min     | 4             | 4x           |
| 1hour     | 1             | 1x (baseline)|
| 4hour     | 0.25          | 0.25x        |
| 1day      | ~0.15         | 0.15x        |

**What gets scaled:**
```python
class TimeframeScaler:
    def __init__(self, base_timeframe='1hour'):
        self.scale_factors = {
            '5min': 12, '15min': 4, '1hour': 1, '4hour': 0.25, '1day': 0.15
        }

    def scale_config(self, config: dict, timeframe: str) -> dict:
        factor = self.scale_factors[timeframe]

        # Indicator periods (scale UP for faster timeframes)
        config['sma_period'] = int(20 * factor)
        config['rsi_period'] = int(14 * factor)
        config['atr_period'] = int(14 * factor)

        # Max hold time (scale UP for faster timeframes)
        config['max_hold_bars'] = int(48 * factor)

        # Cooldown (scale UP)
        config['cooldown_bars'] = int(1 * factor)

        return config
```

**What stays fixed (from config.yaml):**
- Stop loss: 5% (from `risk_management.stop_loss_pct`)
- Take profit: 5% (from `risk_management.take_profit_pct`)
- Position sizing (% of portfolio)
- Daily loss limits

### 3. Backtest Engine (`core/backtest_engine.py`)

Refactored from `Backtest1Hour` to be timeframe-agnostic.

**Key changes from Backtest1Hour:**
- Accept `timeframe` parameter in constructor
- Use `TimeframeScaler` to adjust parameters
- Support both TradeLocker and YFinance data sources
- Remove hard-coded "1Hour" assumptions

### 4. CLI Interface (`backtest_multi.py`)

**Usage:**
```bash
# Compare all timeframes
python backtest_multi.py --timeframes 5min 15min 1hour 4hour 1day

# Specific timeframes with date range
python backtest_multi.py --timeframes 5min 1hour --start 2025-10-15 --end 2026-01-13

# Single timeframe
python backtest_multi.py --timeframes 1hour --symbols AAPL NVDA TSLA
```

**Console output:**
```
================================================================================
  MULTI-TIMEFRAME BACKTEST COMPARISON
  Period: 2025-10-15 to 2026-01-13 (90 days)
  Symbols: 50 | Capital: $100,000
================================================================================
  Timeframe │ Return % │ Win Rate │ Sharpe │ Max DD │ Trades │ Profit Factor
  ──────────┼──────────┼──────────┼────────┼────────┼────────┼──────────────
  5min      │   +12.4% │   52.1%  │  1.82  │  8.2%  │   847  │    1.45
  15min     │   +9.8%  │   54.3%  │  1.65  │  6.9%  │   312  │    1.52
  1hour     │   +7.2%  │   48.7%  │  1.41  │  5.4%  │    89  │    1.38
  4hour     │   +4.1%  │   51.2%  │  1.12  │  4.8%  │    34  │    1.21
  1day      │   +2.3%  │   46.5%  │  0.89  │  3.9%  │    12  │    1.08
================================================================================
  BEST TIMEFRAME: 5min (highest return: +12.4%)
================================================================================
```

**Export files:**
```
results/
├── backtest_comparison_2026-01-13.csv
├── backtest_comparison_2026-01-13.json
├── backtest_5min_2026-01-13.json
├── backtest_15min_2026-01-13.json
└── ...
```

---

## File Structure

```
trading-bot/
├── backtest_multi.py          # NEW - Multi-timeframe CLI
├── backtest.py                # UNCHANGED
├── config.yaml                # MODIFY - Add data_source, tradelocker sections
├── .env                       # MODIFY - Add TradeLocker credentials
│
├── core/
│   ├── __init__.py            # MODIFY - Export new classes
│   ├── tradelocker.py         # NEW - TradeLockerDataFetcher
│   ├── data.py                # UNCHANGED
│   ├── backtest_engine.py     # NEW - Timeframe-agnostic engine
│   └── timeframe_scaler.py    # NEW - Auto-scaling logic
│
└── results/                   # NEW - Output directory
    └── .gitkeep
```

---

## Implementation Order

1. **TradeLocker data fetcher** (`core/tradelocker.py`)
   - Authentication, API calls, disk caching
   - Test with single symbol fetch

2. **Timeframe scaler** (`core/timeframe_scaler.py`)
   - Parameter scaling logic
   - Unit tests for scaling factors

3. **Backtest engine** (`core/backtest_engine.py`)
   - Refactor Backtest1Hour to accept timeframe parameter
   - Integrate scaler and TradeLocker data

4. **Multi-timeframe CLI** (`backtest_multi.py`)
   - Orchestration, comparison table, exports
   - Integration tests

---

## Dependencies

- `tradelocker` Python package (pip install tradelocker)
- Existing: pandas, numpy, yfinance, yaml

---

## Notes

- Existing `backtest.py` remains unchanged for backwards compatibility
- TradeLocker credentials stored in `.env` (not committed to git)
- Disk caching prevents redundant API calls during development
- Falls back to YFinance if TradeLocker is unavailable
