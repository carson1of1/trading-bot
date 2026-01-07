# Emergency Position Limit Check Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add emergency shutdown when broker shows more positions than max_open_positions allows.

**Architecture:** New method `_emergency_position_limit_check()` in TradingBot, called immediately after `sync_positions()`. Liquidates oldest excess positions (FIFO), triggers kill switch, returns early from cycle.

**Tech Stack:** Python, pytest, existing broker/logging infrastructure

---

### Task 1: Write failing test for no-violation case

**Files:**
- Test: `tests/test_bot.py`

**Step 1: Write the failing test**

Add to `tests/test_bot.py`:

```python
class TestEmergencyPositionLimitCheck:
    """Tests for _emergency_position_limit_check() method."""

    def test_no_violation_returns_false(self):
        """When position count <= max, returns False and takes no action."""
        bot = TradingBot.__new__(TradingBot)
        bot.config = {'risk_management': {'max_open_positions': 5}}
        bot.open_positions = {
            'AAPL': {'qty': 10, 'direction': 'LONG', 'entry_time': datetime.now()},
            'MSFT': {'qty': 10, 'direction': 'LONG', 'entry_time': datetime.now()},
            'GOOG': {'qty': 10, 'direction': 'LONG', 'entry_time': datetime.now()},
        }
        bot.kill_switch_triggered = False

        result = bot._emergency_position_limit_check()

        assert result is False
        assert bot.kill_switch_triggered is False
        assert len(bot.open_positions) == 3  # No positions liquidated
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_bot.py::TestEmergencyPositionLimitCheck::test_no_violation_returns_false -v`

Expected: FAIL with `AttributeError: 'TradingBot' object has no attribute '_emergency_position_limit_check'`

**Step 3: Write minimal implementation**

Add to `bot.py` in TradingBot class (after `_check_position_size_violations` method around line 1201):

```python
def _emergency_position_limit_check(self) -> bool:
    """
    Check if position count exceeds max and liquidate excess (oldest first).

    Triggers when broker shows more positions than max_open_positions.
    This indicates a bug, race condition, or manual intervention.

    Returns:
        True if emergency triggered (kill switch set), False otherwise
    """
    max_positions = self.config.get('risk_management', {}).get('max_open_positions', 5)
    current_count = len(self.open_positions)

    if current_count <= max_positions:
        return False

    # TODO: Implement liquidation in next task
    return True
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_bot.py::TestEmergencyPositionLimitCheck::test_no_violation_returns_false -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_bot.py bot.py
git commit -m "feat(ODE-88): Add emergency position limit check stub"
```

---

### Task 2: Write failing test for violation triggers liquidation

**Files:**
- Test: `tests/test_bot.py`

**Step 1: Write the failing test**

Add to `TestEmergencyPositionLimitCheck` class:

```python
    @patch('bot.TradingBot._cleanup_position')
    def test_violation_liquidates_oldest_positions(self, mock_cleanup):
        """When position count > max, liquidates oldest excess positions."""
        bot = TradingBot.__new__(TradingBot)
        bot.config = {'risk_management': {'max_open_positions': 2}}
        bot.kill_switch_triggered = False
        bot.use_tiered_exits = False
        bot.exit_manager = None

        # Create mock broker
        mock_broker = MagicMock()
        mock_order = MagicMock()
        mock_order.filled_avg_price = 150.0
        mock_order.id = 'order123'
        mock_broker.submit_order.return_value = mock_order
        bot.broker = mock_broker

        # Create mock trade logger
        mock_trade_logger = MagicMock()
        bot.trade_logger = mock_trade_logger

        # 4 positions, max 2 = 2 excess (liquidate oldest 2)
        base_time = datetime(2026, 1, 6, 10, 0, 0)
        bot.open_positions = {
            'OLD1': {'qty': 10, 'direction': 'LONG', 'entry_price': 100.0,
                     'entry_time': base_time, 'strategy': 'Test'},  # Oldest - liquidate
            'OLD2': {'qty': 20, 'direction': 'SHORT', 'entry_price': 200.0,
                     'entry_time': base_time + timedelta(hours=1), 'strategy': 'Test'},  # 2nd oldest - liquidate
            'NEW1': {'qty': 15, 'direction': 'LONG', 'entry_price': 150.0,
                     'entry_time': base_time + timedelta(hours=2), 'strategy': 'Test'},  # Keep
            'NEW2': {'qty': 25, 'direction': 'LONG', 'entry_price': 250.0,
                     'entry_time': base_time + timedelta(hours=3), 'strategy': 'Test'},  # Keep
        }

        result = bot._emergency_position_limit_check()

        assert result is True
        assert bot.kill_switch_triggered is True
        assert len(bot.open_positions) == 2
        assert 'OLD1' not in bot.open_positions
        assert 'OLD2' not in bot.open_positions
        assert 'NEW1' in bot.open_positions
        assert 'NEW2' in bot.open_positions

        # Verify broker calls - OLD1 is LONG (sell), OLD2 is SHORT (buy)
        calls = mock_broker.submit_order.call_args_list
        assert len(calls) == 2
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_bot.py::TestEmergencyPositionLimitCheck::test_violation_liquidates_oldest_positions -v`

Expected: FAIL (method returns True but doesn't liquidate)

**Step 3: Write full implementation**

Replace the stub in `bot.py` with full implementation:

```python
def _emergency_position_limit_check(self) -> bool:
    """
    Check if position count exceeds max and liquidate excess (oldest first).

    Triggers when broker shows more positions than max_open_positions.
    This indicates a bug, race condition, or manual intervention.

    Returns:
        True if emergency triggered (kill switch set), False otherwise
    """
    max_positions = self.config.get('risk_management', {}).get('max_open_positions', 5)
    current_count = len(self.open_positions)

    if current_count <= max_positions:
        return False

    excess_count = current_count - max_positions

    logger.critical(
        f"EMERGENCY: Position count {current_count} exceeds max {max_positions} - "
        f"LIQUIDATING {excess_count} oldest positions"
    )

    # Sort by entry_time (oldest first)
    sorted_positions = sorted(
        self.open_positions.items(),
        key=lambda x: x[1].get('entry_time', datetime.now())
    )

    # Liquidate oldest excess positions
    liquidated = 0
    for symbol, pos in sorted_positions[:excess_count]:
        try:
            direction = pos.get('direction', 'LONG')
            qty = pos['qty']
            entry_price = pos.get('entry_price', 0)

            side = 'sell' if direction == 'LONG' else 'buy'
            order = self.broker.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )

            if order:
                exit_price = float(order.filled_avg_price) if hasattr(order, 'filled_avg_price') and order.filled_avg_price else entry_price

                # Calculate P&L
                if direction == 'LONG':
                    pnl = (exit_price - entry_price) * qty
                else:
                    pnl = (entry_price - exit_price) * qty

                # Log trade
                self.trade_logger.log_trade(
                    symbol=symbol,
                    action='SELL' if direction == 'LONG' else 'BUY',
                    quantity=qty,
                    price=exit_price,
                    strategy=pos.get('strategy', 'Unknown'),
                    pnl=pnl,
                    exit_reason='emergency_position_limit'
                )

                # Cleanup
                if self.use_tiered_exits and self.exit_manager:
                    self.exit_manager.unregister_position(symbol)
                self._cleanup_position(symbol)
                del self.open_positions[symbol]

                logger.critical(
                    f"EMERGENCY_LIQUIDATE | {symbol} | {direction} {qty} shares | "
                    f"P&L: ${pnl:+.2f} | Reason: position_limit_exceeded"
                )
                liquidated += 1

        except Exception as e:
            logger.error(f"EMERGENCY_LIQUIDATE | {symbol} | FAILED: {e}", exc_info=True)

    # Set kill switch
    self.kill_switch_triggered = True
    logger.critical(
        f"EMERGENCY: Kill switch triggered - liquidated {liquidated}/{excess_count} positions"
    )

    return True
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_bot.py::TestEmergencyPositionLimitCheck::test_violation_liquidates_oldest_positions -v`

Expected: PASS

**Step 5: Commit**

```bash
git add bot.py tests/test_bot.py
git commit -m "feat(ODE-88): Implement emergency position limit liquidation"
```

---

### Task 3: Write test for correct order sides (LONG sells, SHORT buys)

**Files:**
- Test: `tests/test_bot.py`

**Step 1: Write the test**

Add to `TestEmergencyPositionLimitCheck` class:

```python
    @patch('bot.TradingBot._cleanup_position')
    def test_correct_sides_for_long_and_short(self, mock_cleanup):
        """LONG positions sell to close, SHORT positions buy to close."""
        bot = TradingBot.__new__(TradingBot)
        bot.config = {'risk_management': {'max_open_positions': 1}}
        bot.kill_switch_triggered = False
        bot.use_tiered_exits = False
        bot.exit_manager = None

        mock_broker = MagicMock()
        mock_order = MagicMock()
        mock_order.filled_avg_price = 100.0
        mock_broker.submit_order.return_value = mock_order
        bot.broker = mock_broker
        bot.trade_logger = MagicMock()

        base_time = datetime(2026, 1, 6, 10, 0, 0)
        bot.open_positions = {
            'LONG_POS': {'qty': 10, 'direction': 'LONG', 'entry_price': 100.0,
                         'entry_time': base_time, 'strategy': 'Test'},
            'SHORT_POS': {'qty': 20, 'direction': 'SHORT', 'entry_price': 200.0,
                          'entry_time': base_time + timedelta(hours=1), 'strategy': 'Test'},
            'KEEP': {'qty': 5, 'direction': 'LONG', 'entry_price': 50.0,
                     'entry_time': base_time + timedelta(hours=2), 'strategy': 'Test'},
        }

        bot._emergency_position_limit_check()

        calls = mock_broker.submit_order.call_args_list

        # First call: LONG_POS (oldest) - should sell
        assert calls[0][1]['symbol'] == 'LONG_POS'
        assert calls[0][1]['side'] == 'sell'
        assert calls[0][1]['qty'] == 10

        # Second call: SHORT_POS (2nd oldest) - should buy
        assert calls[1][1]['symbol'] == 'SHORT_POS'
        assert calls[1][1]['side'] == 'buy'
        assert calls[1][1]['qty'] == 20
```

**Step 2: Run test**

Run: `python3 -m pytest tests/test_bot.py::TestEmergencyPositionLimitCheck::test_correct_sides_for_long_and_short -v`

Expected: PASS (implementation already handles this)

**Step 3: Commit**

```bash
git add tests/test_bot.py
git commit -m "test(ODE-88): Add test for LONG/SHORT side handling"
```

---

### Task 4: Write test for partial liquidation failure

**Files:**
- Test: `tests/test_bot.py`

**Step 1: Write the test**

Add to `TestEmergencyPositionLimitCheck` class:

```python
    @patch('bot.TradingBot._cleanup_position')
    def test_partial_failure_continues_and_sets_kill_switch(self, mock_cleanup):
        """If one liquidation fails, others continue and kill switch still set."""
        bot = TradingBot.__new__(TradingBot)
        bot.config = {'risk_management': {'max_open_positions': 1}}
        bot.kill_switch_triggered = False
        bot.use_tiered_exits = False
        bot.exit_manager = None

        # First call fails, second succeeds
        mock_broker = MagicMock()
        mock_order = MagicMock()
        mock_order.filled_avg_price = 100.0
        mock_broker.submit_order.side_effect = [
            Exception("Network error"),  # First fails
            mock_order,  # Second succeeds
        ]
        bot.broker = mock_broker
        bot.trade_logger = MagicMock()

        base_time = datetime(2026, 1, 6, 10, 0, 0)
        bot.open_positions = {
            'FAIL': {'qty': 10, 'direction': 'LONG', 'entry_price': 100.0,
                     'entry_time': base_time, 'strategy': 'Test'},
            'SUCCESS': {'qty': 20, 'direction': 'LONG', 'entry_price': 200.0,
                        'entry_time': base_time + timedelta(hours=1), 'strategy': 'Test'},
            'KEEP': {'qty': 5, 'direction': 'LONG', 'entry_price': 50.0,
                     'entry_time': base_time + timedelta(hours=2), 'strategy': 'Test'},
        }

        result = bot._emergency_position_limit_check()

        assert result is True
        assert bot.kill_switch_triggered is True
        # FAIL stays (couldn't liquidate), SUCCESS removed, KEEP stays
        assert 'FAIL' in bot.open_positions  # Failed to liquidate
        assert 'SUCCESS' not in bot.open_positions  # Successfully liquidated
        assert 'KEEP' in bot.open_positions  # Not targeted
```

**Step 2: Run test**

Run: `python3 -m pytest tests/test_bot.py::TestEmergencyPositionLimitCheck::test_partial_failure_continues_and_sets_kill_switch -v`

Expected: PASS (implementation already handles this with try/except)

**Step 3: Commit**

```bash
git add tests/test_bot.py
git commit -m "test(ODE-88): Add test for partial liquidation failure"
```

---

### Task 5: Integrate into run_trading_cycle

**Files:**
- Modify: `bot.py` - `run_trading_cycle()` method around line 1229

**Step 1: Write integration test**

Add to `TestEmergencyPositionLimitCheck` class:

```python
    @patch('bot.TradingBot._emergency_position_limit_check')
    @patch('bot.TradingBot.sync_positions')
    @patch('bot.TradingBot.sync_account')
    @patch('bot.TradingBot._reconcile_broker_state')
    @patch('bot.TradingBot.run_health_check')
    def test_called_in_trading_cycle_after_sync(self, mock_health, mock_reconcile,
                                                  mock_sync_account, mock_sync_pos,
                                                  mock_emergency):
        """Emergency check is called after sync_positions in trading cycle."""
        mock_emergency.return_value = True  # Trigger emergency

        bot = TradingBot.__new__(TradingBot)
        bot.config = {'risk_management': {}}
        bot.drawdown_guard = MagicMock()
        bot.drawdown_guard.enabled = False
        bot.open_positions = {}

        bot.run_trading_cycle()

        # Verify call order
        mock_sync_pos.assert_called_once()
        mock_emergency.assert_called_once()

    @patch('bot.TradingBot._emergency_position_limit_check')
    @patch('bot.TradingBot.sync_positions')
    @patch('bot.TradingBot.sync_account')
    @patch('bot.TradingBot._reconcile_broker_state')
    @patch('bot.TradingBot.run_health_check')
    def test_cycle_returns_early_when_emergency_triggered(self, mock_health, mock_reconcile,
                                                           mock_sync_account, mock_sync_pos,
                                                           mock_emergency):
        """When emergency triggers, cycle returns early without checking exits/entries."""
        mock_emergency.return_value = True

        bot = TradingBot.__new__(TradingBot)
        bot.config = {'risk_management': {}}
        bot.drawdown_guard = MagicMock()
        bot.drawdown_guard.enabled = False
        bot.open_positions = {'AAPL': {}}
        bot.fetch_data = MagicMock()

        bot.run_trading_cycle()

        # fetch_data should NOT be called (cycle returned early)
        bot.fetch_data.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_bot.py::TestEmergencyPositionLimitCheck::test_called_in_trading_cycle_after_sync -v`

Expected: FAIL (emergency check not called yet)

**Step 3: Add call to run_trading_cycle**

In `bot.py`, modify `run_trading_cycle()` - add after `sync_positions()` call (around line 1230):

```python
        # 1. Sync state
        self.sync_account()
        self.sync_positions()

        # 1.5 Emergency position limit check (ODE-88)
        # Must run immediately after sync to detect violations before any other logic
        if self._emergency_position_limit_check():
            return  # Halt cycle - kill switch triggered

        # 2. Update drawdown guard (must be done after sync)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_bot.py::TestEmergencyPositionLimitCheck -v`

Expected: All PASS

**Step 5: Commit**

```bash
git add bot.py tests/test_bot.py
git commit -m "feat(ODE-88): Integrate emergency check into trading cycle"
```

---

### Task 6: Run full test suite

**Step 1: Run all tests**

Run: `python3 -m pytest tests/test_bot.py -v`

Expected: All tests PASS

**Step 2: If any failures, fix and re-run**

**Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix(ODE-88): Test fixes"
```

---

### Task 7: Add required imports to test file

**Files:**
- Modify: `tests/test_bot.py` - imports section

**Step 1: Verify imports exist**

Ensure these imports are at top of `tests/test_bot.py`:

```python
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
```

**Step 2: Run tests**

Run: `python3 -m pytest tests/test_bot.py::TestEmergencyPositionLimitCheck -v`

Expected: PASS

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Test no-violation case + stub | bot.py, tests/test_bot.py |
| 2 | Test liquidation + full implementation | bot.py, tests/test_bot.py |
| 3 | Test LONG/SHORT side handling | tests/test_bot.py |
| 4 | Test partial failure handling | tests/test_bot.py |
| 5 | Integrate into run_trading_cycle | bot.py, tests/test_bot.py |
| 6 | Run full test suite | - |
| 7 | Verify imports | tests/test_bot.py |
