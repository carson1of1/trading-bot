# Daily Drawdown Guard Design

**Date:** 2026-01-04
**Status:** Approved
**Problem:** Worst backtest days show -9.5% losses, exceeding funded account 5% daily limit
**Goal:** Cap daily drawdown to 4.0% with tiered protection

---

## Overview

Implement a `DailyDrawdownGuard` that tracks real-time equity (unrealized + realized P&L) and enforces tiered daily drawdown limits to protect funded account capital.

## Requirements

- **Calculation:** Equity-based (real-time unrealized + realized P&L)
- **Hard Limit Action:** Full liquidation via market orders + day halt
- **Buffer:** 4.0% hard limit provides 1% buffer before 5% funded account fail

## Tiered Thresholds

| Threshold | Level | Action |
|-----------|-------|--------|
| Warning | -2.5% | Alert + reduce new position sizes to 50% |
| Soft Limit | -3.5% | Block all new entries, tighten stops to breakeven |
| Hard Limit | -4.0% | Full liquidation via market orders + day halt |

## Architecture

### New Component: `DailyDrawdownGuard`

Location: `core/risk.py`

```
DailyDrawdownGuard
├── State Tracking
│   ├── day_start_equity: float      # Snapshot at market open
│   ├── realized_pnl_today: float    # Sum of closed trade P&L
│   ├── current_equity: float        # Updated on every price tick
│   └── drawdown_pct: float          # (day_start - current) / day_start
│
├── Thresholds (configurable)
│   ├── warning_pct: 2.5%
│   ├── soft_limit_pct: 3.5%
│   └── hard_limit_pct: 4.0%
│
└── Actions
    ├── check_drawdown() → returns current tier (NORMAL/WARNING/SOFT/HARD)
    ├── on_warning() → logs alert, sets position_size_multiplier = 0.5
    ├── on_soft_limit() → blocks entries, tightens stops to breakeven
    └── on_hard_limit() → triggers full_liquidation(), halts_day()
```

### Integration Points

- `TradingBot.run_trading_cycle()` calls `guard.update_equity()` at cycle start
- `RiskManager.calculate_position_size()` checks `guard.position_size_multiplier`
- `EntryGate.check_entry_allowed()` checks `guard.entries_allowed`
- New `guard.force_liquidate_all()` method for emergency exits

## Equity Calculation

```python
def calculate_current_drawdown(self):
    # Get current equity from broker
    unrealized_pnl = sum(position.unrealized_pnl for position in open_positions)
    current_equity = day_start_equity + realized_pnl_today + unrealized_pnl

    # Calculate drawdown percentage
    drawdown_pct = (day_start_equity - current_equity) / day_start_equity
    return drawdown_pct  # Positive number means loss (e.g., 0.035 = -3.5%)
```

## Daily Reset Logic

| Event | Action |
|-------|--------|
| Market open (9:30 ET) | Snapshot `day_start_equity` from broker account value |
| Each trading cycle | Update `current_equity`, check thresholds |
| Trade closed | Add P&L to `realized_pnl_today` |
| Market close (4:00 ET) | Log final daily drawdown, reset state |
| New trading day | Clear all counters, reset `entries_allowed = True` |

## Threshold Hysteresis

Once a threshold is crossed, it stays active for the day:
- Hit -2.5% warning → position sizing stays at 50% even if equity recovers
- Hit -3.5% soft limit → entries stay blocked even if you recover to -2%
- Hit -4.0% hard limit → day is done, no recovery possible

This prevents "bouncing" where recovery leads to new positions that then lose again.

## Liquidation Mechanism

### Hard Limit Flow

```
on_hard_limit() triggered at -4.0%
    │
    ├─► Log: "EMERGENCY: Daily drawdown limit hit (-4.0%)"
    │
    ├─► Set state: entries_allowed = False, day_halted = True
    │
    ├─► Get all open positions from broker
    │
    ├─► For each position:
    │   ├─► Cancel any pending orders (stops, limits)
    │   ├─► Submit market order to close
    │   │   ├─► LONG: submit SELL market order
    │   │   └─► SHORT: submit BUY market order
    │   └─► Log: "LIQUIDATED {symbol} {qty} shares @ market"
    │
    ├─► Wait for all fills (max 30 seconds timeout)
    │
    ├─► Unregister all positions from ExitManager
    │
    └─► Log final equity and total realized loss for the day
```

### Safety Guards

| Guard | Purpose |
|-------|---------|
| Double-submission prevention | Track `liquidation_in_progress` flag to prevent duplicate orders |
| Order verification | Confirm each position is actually closed via broker API |
| Timeout handling | If order doesn't fill in 30s, log error and retry |
| Partial fill handling | If partially filled, submit another market order for remainder |

## Configuration

### config.yaml Additions

```yaml
daily_drawdown_guard:
  enabled: true

  # Thresholds (percentage of day's starting equity)
  warning_pct: 2.5          # Reduce position sizes to 50%
  soft_limit_pct: 3.5       # Block new entries
  hard_limit_pct: 4.0       # Full liquidation + day halt

  # Behavior
  warning_size_multiplier: 0.5    # Position size when in warning zone
  tighten_stops_at_soft: true     # Move stops to breakeven at soft limit

  # Execution
  liquidation_timeout_sec: 30
  use_market_orders: true
```

## Backtest Integration

The guard must work identically in backtests to validate effectiveness:

| Component | Live | Backtest |
|-----------|------|----------|
| Equity source | `broker.get_account().equity` | `FakeBroker.portfolio_value + unrealized` |
| Day detection | Real clock (9:30 ET) | Bar timestamp date change |
| Liquidation | Real market orders | Simulated fills at bar close price |
| Logging | `logs/trading_*.log` | `logs/backtest_analytics.log` |

## Files to Modify

| File | Changes |
|------|---------|
| `core/risk.py` | Add `DailyDrawdownGuard` class (~150 lines) |
| `core/entry_gate.py` | Add `set_drawdown_guard()` method, check `guard.entries_allowed` |
| `bot.py` | Initialize guard at startup, call `guard.update_equity()` each cycle, handle liquidation |
| `backtest.py` | Same integration as bot.py for parity |
| `config.yaml` | Add `daily_drawdown_guard` section |

## Execution Order in Trading Cycle

```python
run_trading_cycle():
    1. guard.update_equity(broker.get_account())     # Check drawdown FIRST
    2. if guard.day_halted: return                   # Skip if halted
    3. if guard.tier == HARD: guard.liquidate_all() # Emergency exit

    4. sync_positions()                              # Existing
    5. check_exits()                                 # Existing
    6. if guard.entries_allowed:                     # Gate check
           check_entries()
```

## Expected Results

With this design, backtest worst days should change from:

| Before | After |
|--------|-------|
| -9.56% | ≤ -4.0% (liquidated) |
| -9.52% | ≤ -4.0% (liquidated) |
| -9.50% | ≤ -4.0% (liquidated) |

Slippage may cause actual exits at -4.1% to -4.3%, but well within the 5% fail threshold.

## Validation

After implementation, run backtests and verify no days exceed -4.0%:

```sql
SELECT date, MIN(drawdown_pct) as worst_dd
FROM daily_snapshots
GROUP BY date
HAVING worst_dd < -0.04  -- Should return 0 rows
```
