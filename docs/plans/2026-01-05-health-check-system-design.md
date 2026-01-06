# Health Check System Design (ODE-95)

## Overview

A health check system that verifies the ExitManager and live bot are functioning correctly. Runs automatically at the start of each trading cycle, logs status at INFO level, and alerts on failures.

## Design Decisions

1. **Trigger**: Inside `run_trading_cycle()` - runs automatically each hourly cycle
2. **On failure**: Log and continue - health check is observability, not a gate
3. **Output**: Structured dict with INFO-level summary, WARNING/ERROR for failures

## Architecture

### Single Method in TradingBot

```python
def run_health_check(self) -> dict:
    """Run comprehensive health check on bot systems."""
```

- Called at START of `run_trading_cycle()` (after sync, before exits/entries)
- Returns structured dict with all check results
- Logs summary at INFO level each cycle
- Does NOT block trading - purely observational

## Health Checks

### ExitManager Health (5 checks)

| Check | Description | Pass Condition |
|-------|-------------|----------------|
| `positions_registered` | All open positions are in ExitManager | `open_positions.keys() == exit_manager.positions.keys()` |
| `state_persistence` | State file exists and is recent | `exit_manager_state.json` modified within 2 hours |
| `hard_stops_valid` | Hard stops calculated correctly | For each position: stop = entry × (1 ± hard_stop_pct) |
| `trailing_thresholds` | Trailing config matches | ExitManager thresholds match bot config |
| `partial_tp_tracking` | Partial TP tracking accurate | Flag state consistent with position quantity |

### Live Bot Health (7 checks)

| Check | Description | Pass Condition |
|-------|-------------|----------------|
| `broker_connected` | Broker connection active | `broker.get_account()` succeeds |
| `account_synced` | Cash/portfolio populated | `cash > 0 AND portfolio_value > 0` |
| `positions_synced` | Positions match broker | Count matches `broker.get_positions()` |
| `data_fetcher_valid` | Data fetcher returns data | Fetch SPY returns non-empty DataFrame |
| `strategy_manager_ready` | Strategies initialized | `len(strategies) > 0` |
| `kill_switch_status` | Kill switch state (INFO) | Reports current state, always passes |
| `drawdown_guard_status` | Drawdown guard state (INFO) | Reports tier and entries_allowed |

## Output Format

### Return Value

```python
{
    'timestamp': '2026-01-05T10:02:15',
    'overall_status': 'HEALTHY',  # or 'DEGRADED' or 'UNHEALTHY'
    'checks': {
        'broker_connected': {'status': 'PASS', 'message': 'Connected to Alpaca paper'},
        'positions_registered': {'status': 'PASS', 'message': '3/3 positions registered'},
        'kill_switch_status': {'status': 'INFO', 'message': 'Not triggered'},
        # ...
    },
    'summary': {
        'total_checks': 12,
        'passed': 10,
        'failed': 0,
        'info': 2
    }
}
```

### Status Levels

- **PASS**: Check succeeded
- **FAIL**: Check failed (logs at WARNING/ERROR)
- **INFO**: Informational only (kill switch, drawdown guard status)

### Overall Status

- **HEALTHY**: All checks PASS (INFO checks don't affect this)
- **DEGRADED**: 1-2 non-critical checks failed
- **UNHEALTHY**: 3+ checks failed or critical check failed

Critical checks: `broker_connected`, `account_synced`, `positions_synced`

## Log Output

### Normal (each cycle)

```
INFO - HEALTH_CHECK | HEALTHY | 10/12 PASS | 2 INFO | Positions: 3 | ExitMgr: 3 | KillSwitch: OFF
```

### On Failure

```
WARNING - HEALTH_CHECK | positions_registered FAIL: 3 bot positions but only 2 in ExitManager (missing: AAPL)
INFO - HEALTH_CHECK | DEGRADED | 9/12 PASS | 1 FAIL | 2 INFO
```

## Implementation Plan

### Files to Modify

1. **bot.py** - Add `run_health_check()` method and call it in `run_trading_cycle()`

### Implementation Steps

1. Add `_check_broker_health()` helper method
2. Add `_check_exit_manager_health()` helper method
3. Add `_check_position_sync()` helper method
4. Add `_check_data_fetcher()` helper method
5. Add `_check_strategy_manager()` helper method
6. Add `run_health_check()` method that calls all helpers
7. Call `run_health_check()` at start of `run_trading_cycle()`
8. Add tests for health check functionality

### Test Plan

1. Test with healthy bot state - should return HEALTHY
2. Test with missing ExitManager registration - should detect and report
3. Test with broker connection failure - should catch and report UNHEALTHY
4. Test with kill switch triggered - should report INFO status correctly
5. Test log output format matches specification
