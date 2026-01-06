# Broker State Reconciliation Design

**Issue:** ODE-91
**Date:** 2026-01-05
**Status:** Approved

## Overview

Detect and alert when internal bot state diverges from broker state. Alert-only mode - logs warnings but takes no corrective action.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Behavior | Alert-only | Safe for initial deployment, observe before auto-correcting |
| Placement | Before `sync_positions()` | Captures divergence before internal state is overwritten |
| Log detail | Detailed with action hints | Provides enough context to diagnose issues |
| Price tolerance | 1% | Small slippage is normal with market orders |
| Frequency | Hourly (each trading cycle) | Matches trading cycle, catches issues within an hour |

## Divergence Scenarios

1. **Ghost positions** - Bot thinks position exists, broker doesn't have it
2. **Orphan positions** - Broker has position, bot doesn't track it
3. **Quantity mismatch** - Share count differs (partial fills, manual trades)
4. **Entry price mismatch** - Price differs by >1% (significant slippage)

## Implementation

### New Method: `_reconcile_broker_state()`

```python
def _reconcile_broker_state(self):
    """
    Detect divergence between internal state and broker state.

    Runs BEFORE sync_positions() to capture mismatches before overwriting.
    Alert-only mode - logs warnings but takes no corrective action.
    """
    try:
        broker_positions = {p.symbol: p for p in self.broker.get_positions()}
    except Exception as e:
        logger.error(f"RECONCILE | Failed to fetch broker positions: {e}", exc_info=True)
        return

    # 1. Ghost positions (internal but not broker)
    for symbol, pos in self.open_positions.items():
        if symbol not in broker_positions:
            logger.warning(
                f"RECONCILE | GHOST | {symbol} | "
                f"Internal: {pos['qty']} shares @ ${pos['entry_price']:.2f} | "
                f"Broker: NOT FOUND | Action: Position may have been closed externally"
            )

    # 2. Orphan positions (broker but not internal)
    for symbol, bp in broker_positions.items():
        if symbol not in self.open_positions:
            logger.warning(
                f"RECONCILE | ORPHAN | {symbol} | "
                f"Internal: NOT TRACKED | "
                f"Broker: {int(bp.qty)} shares @ ${float(bp.avg_entry_price):.2f} | "
                f"Action: Position opened externally or bot restarted mid-trade"
            )
            continue

        pos = self.open_positions[symbol]

        # 3. Quantity mismatch
        if int(bp.qty) != pos['qty']:
            logger.warning(
                f"RECONCILE | QTY_MISMATCH | {symbol} | "
                f"Internal: {pos['qty']} shares | Broker: {int(bp.qty)} shares | "
                f"Action: Partial fill or manual trade occurred"
            )

        # 4. Entry price mismatch (>1% tolerance)
        broker_price = float(bp.avg_entry_price)
        if pos['entry_price'] > 0:
            price_diff_pct = abs(broker_price - pos['entry_price']) / pos['entry_price']
            if price_diff_pct > 0.01:
                logger.warning(
                    f"RECONCILE | PRICE_MISMATCH | {symbol} | "
                    f"Internal: ${pos['entry_price']:.2f} | Broker: ${broker_price:.2f} | "
                    f"Diff: {price_diff_pct*100:.2f}% | Action: Significant slippage or averaging occurred"
                )
```

### Call Site

In `run_trading_cycle()`, before sync methods:

```python
def run_trading_cycle(self):
    try:
        # ... logging ...

        # 0. Reconcile state BEFORE syncing (detect divergence)
        self._reconcile_broker_state()

        # 1. Sync state
        self.sync_account()
        self.sync_positions()
        # ... rest of method
```

## Files Changed

- `bot.py` - Add `_reconcile_broker_state()` method, call at start of `run_trading_cycle()`
- `tests/test_broker_reconciliation.py` - New test file

## Test Cases

| Test Case | Setup | Expected |
|-----------|-------|----------|
| Ghost position | Internal has AAPL, broker empty | Log `RECONCILE \| GHOST \| AAPL` |
| Orphan position | Internal empty, broker has MSFT | Log `RECONCILE \| ORPHAN \| MSFT` |
| Quantity mismatch | Internal: 100, broker: 75 | Log `RECONCILE \| QTY_MISMATCH` |
| Price mismatch >1% | Internal: $100, broker: $102 | Log `RECONCILE \| PRICE_MISMATCH` |
| Price mismatch <1% | Internal: $100, broker: $100.50 | No warning |
| Broker API failure | `get_positions()` raises | Log error, return gracefully |
| No positions | Both empty | No warnings |
| All in sync | Internal matches broker | No warnings |
