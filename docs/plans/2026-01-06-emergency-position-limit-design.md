# ODE-88: Emergency Shutdown - Position Count Exceeds Max

**Date:** 2026-01-06
**Status:** Approved
**Issue:** [ODE-88](https://linear.app/odell/issue/ODE-88)

## Problem

The bot may encounter situations where the broker shows more positions than `max_open_positions` allows. This can happen due to:
- Manual trades placed outside the bot
- Race conditions
- Bugs in the system

Currently, the bot only checks position limits before **new entries** (line 1352). There's no protection when we **already have** too many positions.

## Solution

Add `_emergency_position_limit_check()` method that runs immediately after `sync_positions()` in `run_trading_cycle()`.

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Liquidation order | Oldest first (FIFO) | Simple, deterministic, preserves newer positions |
| Kill switch | Yes, trigger it | If this fires, something unexpected happened - force manual investigation |
| Liquidation amount | All excess at once | Get to safe state immediately |
| Location | After `sync_positions()` | Earliest detection after positions are known |

## Implementation

### Location in `run_trading_cycle()`

```python
def run_trading_cycle(self):
    # 0. Reconcile broker state
    self._reconcile_broker_state()

    # 0.5 Health check
    self.run_health_check()

    # 1. Sync state
    self.sync_account()
    self.sync_positions()

    # NEW: 1.5 Emergency position limit check
    if self._emergency_position_limit_check():
        return  # Halt cycle - kill switch triggered

    # 2. Drawdown guard (continues as before)
    ...
```

### New Method

```python
def _emergency_position_limit_check(self) -> bool:
    """
    Check if position count exceeds max and liquidate excess (oldest first).

    Triggers when broker shows more positions than max_open_positions.
    This indicates a bug, race condition, or manual intervention.

    Returns:
        True if emergency triggered (kill switch set), False otherwise
    """
```

### Logic Flow

1. Get `max_open_positions` from config (default 5)
2. Compare against `len(self.open_positions)`
3. If not exceeded, return `False` (no action)
4. If exceeded:
   - Log CRITICAL with position count vs limit
   - Sort positions by `entry_time` ascending (oldest first)
   - Calculate excess count: `len(positions) - max_positions`
   - Liquidate the oldest N positions via market orders
   - Cleanup tracking for each liquidated position
   - Log trade to database with `exit_reason='emergency_position_limit'`
   - Set `kill_switch_triggered = True`
   - Return `True`

### Error Handling

- Wrap each liquidation in try/except
- If one position fails, continue with others
- Log failures at ERROR level
- Still set kill switch even if some liquidations fail

### Logging

```python
# When triggered
logger.critical(
    f"EMERGENCY: Position count {current_count} exceeds max {max_positions} - "
    f"LIQUIDATING {excess_count} oldest positions"
)

# Per position
logger.critical(
    f"EMERGENCY_LIQUIDATE | {symbol} | {direction} {qty} shares | "
    f"P&L: ${pnl:+.2f} | Reason: position_limit_exceeded"
)

# After completion
logger.critical(
    f"EMERGENCY: Kill switch triggered - liquidated {liquidated}/{excess_count} positions"
)
```

## Files Changed

- `bot.py` - Add `_emergency_position_limit_check()` method, call from `run_trading_cycle()`
- `tests/test_bot.py` - Add `TestEmergencyPositionLimitCheck` test class

## Test Cases

1. **No violation** - 3 positions, max 5 → returns `False`, no action
2. **Violation triggers liquidation** - 7 positions, max 5 → liquidates 2 oldest
3. **FIFO order verified** - Oldest positions liquidated first
4. **Mixed LONG/SHORT** - Correct side used for each direction
5. **Partial failure** - One fails, others still liquidated, kill switch set
6. **Kill switch blocks entries** - No new entries after emergency
