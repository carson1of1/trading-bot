# Broker-Level Stop-Loss Orders Design

**Issue:** ODE-117
**Date:** 2026-01-07
**Status:** Approved

## Problem

Currently, entry orders are submitted as market orders without attached stop-loss orders at the broker level. The bot relies on periodic exit checks to enforce stops.

**Risk**: If the bot crashes, restarts, or loses connectivity during a violent market move, positions have NO broker-level protection. A flash crash could result in unlimited losses.

## Solution

Submit bracket orders to Alpaca with attached 5% stop-loss on every entry. This stop acts as a **backup safety net** that only triggers if the bot is down - normal operation uses tighter software stops.

## Design

### 1. New BrokerInterface Method

Add `submit_bracket_order()` to the abstract base class:

```python
@abstractmethod
def submit_bracket_order(
    self,
    symbol: str,
    qty: int,
    side: str,  # 'buy' or 'sell'
    stop_loss_percent: float = 0.05,  # 5% default
    time_in_force: str = 'gtc',
    **kwargs
) -> Order:
    """Submit market entry with attached stop-loss order for crash protection."""
    pass
```

### 2. AlpacaBroker Implementation

Use Alpaca's bracket order API:

```python
def submit_bracket_order(self, symbol, qty, side, stop_loss_percent=0.05, time_in_force='gtc', **kwargs):
    price = kwargs.get('price')  # Expected entry price for stop calculation

    if side == 'buy':
        stop_price = price * (1 - stop_loss_percent)
        take_profit_price = price * 2  # Set high, won't trigger
    else:
        stop_price = price * (1 + stop_loss_percent)
        take_profit_price = price * 0.5  # Set low, won't trigger

    alpaca_order = self.api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type='market',
        time_in_force=time_in_force,
        order_class='bracket',
        stop_loss={'stop_price': round(stop_price, 2)},
        take_profit={'limit_price': round(take_profit_price, 2)}
    )

    return self._convert_bracket_order(alpaca_order)
```

### 3. FakeBroker Implementation

Simulate bracket orders for backtest/dry-run:

```python
def submit_bracket_order(self, symbol, qty, side, stop_loss_percent=0.05, **kwargs):
    # Execute market entry
    entry_order = self.submit_order(symbol, qty, side, type='market', **kwargs)

    if entry_order.status == 'filled':
        # Create simulated stop order
        entry_price = entry_order.filled_avg_price
        if side == 'buy':
            stop_price = entry_price * (1 - stop_loss_percent)
            stop_side = 'sell'
        else:
            stop_price = entry_price * (1 + stop_loss_percent)
            stop_side = 'buy'

        stop_order_id = self._generate_order_id()
        self.stop_orders[symbol] = {
            'order_id': stop_order_id,
            'stop_price': stop_price,
            'qty': qty,
            'side': stop_side
        }
        entry_order.stop_order_id = stop_order_id

    return entry_order
```

### 4. Stop Order Tracking

Store stop order ID in position tracking:

```python
self.open_positions[symbol] = {
    'symbol': symbol,
    'qty': fill_qty,
    'entry_price': fill_price,
    'direction': direction,
    'stop_order_id': order.stop_order_id,  # NEW: Track broker stop
    ...
}
```

### 5. Cancel Stop on Exit

When bot closes position, cancel the broker stop first:

```python
def execute_exit(self, symbol, ...):
    position = self.open_positions.get(symbol)

    # Cancel broker-level stop order before closing
    if position and position.get('stop_order_id'):
        try:
            self.broker.cancel_order(position['stop_order_id'])
        except Exception as e:
            logger.warning(f"Failed to cancel stop order for {symbol}: {e}")

    # Close position as normal
    ...
```

### 6. Startup Reconciliation

On bot startup, clean up orphaned stop orders:

```python
def _reconcile_stop_orders(self):
    """Cancel any orphaned stop orders from previous sessions."""
    open_orders = self.broker.list_orders(status='open')
    positions = {p.symbol for p in self.broker.get_positions()}

    for order in open_orders:
        if order.type in ['stop', 'stop_limit'] and order.symbol not in positions:
            logger.info(f"Cancelling orphaned stop order: {order.id} for {order.symbol}")
            self.broker.cancel_order(order.id)
```

## Bot.py Changes

Update `check_entry()` to use bracket orders:

```python
# OLD:
order = self.broker.submit_order(
    symbol=symbol,
    qty=qty,
    side=side,
    type='market',
    time_in_force='day'
)

# NEW:
order = self.broker.submit_bracket_order(
    symbol=symbol,
    qty=qty,
    side=side,
    stop_loss_percent=0.05,
    time_in_force='gtc',
    price=price  # For stop calculation
)
```

## Test Plan

1. **Unit Tests - AlpacaBroker**
   - `test_submit_bracket_order_long_calculates_correct_stop`
   - `test_submit_bracket_order_short_calculates_correct_stop`
   - `test_bracket_order_uses_gtc_time_in_force`

2. **Unit Tests - FakeBroker**
   - `test_fake_bracket_order_creates_stop_order`
   - `test_fake_stop_order_triggers_on_price_drop`
   - `test_fake_stop_order_cancelled_on_position_close`

3. **Unit Tests - Bot**
   - `test_entry_uses_bracket_order`
   - `test_exit_cancels_stop_order`
   - `test_startup_reconciles_orphaned_stops`

4. **Integration Tests**
   - `test_full_entry_exit_cycle_with_bracket_orders`
   - `test_bracket_order_survives_bot_restart` (manual)

## Acceptance Criteria

- [x] Design approved
- [ ] Every entry has a corresponding stop-loss order at the broker level
- [ ] Stop orders survive bot restarts (GTC time-in-force)
- [ ] Stop orders are cancelled when position is closed
- [ ] Test coverage for bracket order submission
- [ ] FakeBroker simulates bracket orders for backtesting
