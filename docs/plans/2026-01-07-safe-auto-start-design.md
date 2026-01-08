# Safe Auto-Start at Market Open

**Issue:** ODE-124
**Date:** 2026-01-07
**Status:** Approved

## Overview

Implement automated startup at market open with safety checks. The bot auto-starts but trading is only enabled after all pre-market checks pass.

## Architecture

### Startup Sequence

1. Systemd timer triggers at 9:25 AM ET (weekdays)
2. `bot.py` starts, runs `PreflightChecklist`
3. If all checks pass → enable trading at 9:30
4. If any check fails → exit with code 1

### Components

```
core/preflight.py
├── PreflightChecklist
│   ├── __init__(config, broker)
│   ├── run_all_checks() -> (bool, list[CheckResult])
│   └── Individual checks:
│       ├── check_api_keys()
│       ├── check_account_balance()
│       ├── check_market_status()
│       ├── check_daily_loss_reset()
│       ├── check_positions_accounted()
│       ├── check_universe_loaded()
│       └── check_no_duplicate_process()
└── CheckResult(namedtuple)
    ├── name: str
    ├── passed: bool
    └── message: str
```

## Preflight Checks

| Check | Implementation | Failure Condition |
|-------|---------------|-------------------|
| API keys | Check `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` env vars | Missing or empty |
| Account balance | Call `broker.get_account()` | API error or balance ≤ 0 |
| Market status | Use `MarketHours.is_market_open()` | Market closed and >10 min until open |
| Daily loss reset | Verify `account.last_equity` accessible | Can't fetch last_equity |
| Positions accounted | Fetch positions, check against watchlist | Orphaned positions not in universe |
| Universe loaded | Load `universe.yaml` | Empty or missing file |
| No duplicate process | Check PID file at `logs/bot.pid` | Another bot instance running |

### Market Status Check Detail

Since we start at 9:25 but market opens at 9:30:

```python
def check_market_status(self) -> CheckResult:
    """Check market is open or opens within 10 minutes."""
    market_hours = MarketHours()

    if market_hours.is_market_open():
        return CheckResult("market_status", True, "Market is open")

    minutes_until_open = market_hours.time_until_market_open()
    if 0 < minutes_until_open <= 10:
        return CheckResult("market_status", True,
            f"Market opens in {minutes_until_open} minutes")

    return CheckResult("market_status", False,
        f"Market closed ({minutes_until_open} min until open)")
```

## Integration with bot.py

### New Method

```python
def run_preflight(self) -> bool:
    """Run preflight checks before enabling trading."""
    from core.preflight import PreflightChecklist

    checklist = PreflightChecklist(self.config, self.broker)
    all_passed, results = checklist.run_all_checks()

    if not all_passed:
        failed = [r for r in results if not r.passed]
        logger.error(f"PREFLIGHT FAILED: {len(failed)} check(s) failed")
        return False

    logger.info("PREFLIGHT PASSED: All checks passed, trading enabled")
    return True
```

### Modified main()

```python
def main():
    bot = TradingBot(config_path=args.config, scanner_symbols=scanner_symbols)

    # Run preflight checks
    if not bot.run_preflight():
        logger.error("Exiting due to preflight failure")
        sys.exit(1)

    # Continue to trading loop...
    if bot.start():
        # ... existing logic ...
```

## Systemd Configuration

### trading-bot.service

```ini
[Unit]
Description=Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=carsonodell
WorkingDirectory=/home/carsonodell/trading-bot
ExecStart=/usr/bin/python3 bot.py
Restart=no
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

### trading-bot.timer

```ini
[Unit]
Description=Start Trading Bot before market open

[Timer]
OnCalendar=Mon..Fri 09:25:00 America/New_York
Persistent=false

[Install]
WantedBy=timers.target
```

### Installation

```bash
sudo cp trading-bot.service /etc/systemd/system/
sudo cp trading-bot.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable trading-bot.timer
```

## Example Output

```
[PREFLIGHT] ✓ API keys loaded
[PREFLIGHT] ✓ Account balance: $25,432.50
[PREFLIGHT] ✓ Market opens in 5 minutes
[PREFLIGHT] ✓ Daily loss reset (starting equity: $25,432.50)
[PREFLIGHT] ✓ No unexpected positions (0 open)
[PREFLIGHT] ✓ Universe loaded: 10 symbols
[PREFLIGHT] ✓ No duplicate process
[PREFLIGHT] PASSED - 7 of 7 checks passed
```

## Testing Strategy

### Unit Tests (tests/test_preflight.py)

- `test_check_api_keys_present`
- `test_check_api_keys_missing`
- `test_check_account_balance_success`
- `test_check_account_balance_api_error`
- `test_check_market_open`
- `test_check_market_opens_soon`
- `test_check_market_closed`
- `test_check_no_duplicate_no_pid_file`
- `test_check_no_duplicate_stale_pid`
- `test_check_no_duplicate_running_process`
- `test_check_positions_none_open`
- `test_check_positions_all_in_watchlist`
- `test_check_positions_orphaned`
- `test_check_universe_loaded`
- `test_check_universe_empty`
- `test_run_all_checks_aggregates_results`

### Testing Patterns

- Mock `broker.get_account()` and `broker.get_positions()`
- Mock environment variables for API key tests
- Mock `MarketHours` for market status tests
- Use temp PID files (like existing `test_api_bot_management.py`)

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scheduler | Systemd timer | Most robust for unattended operation, integrates with existing PID tracking |
| Location | Standalone `core/preflight.py` | Clean separation, easy to test, follows existing pattern |
| Failure handling | Exit with code 1 | Fail-fast is safer for trading, systemd handles retries |
| Config hash | Skipped | Adds complexity without clear benefit for single-operator setup |
