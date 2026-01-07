# Position Size Guard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Auto-liquidate positions that exceed value limits to protect against concentration risk.

**Architecture:** Add `_check_position_size_violations()` method to `TradingBot` that runs after sync but before exit checks. Uses existing broker interface for liquidation, consistent with DailyDrawdownGuard pattern.

**Tech Stack:** Python, pytest, existing broker abstraction

---

### Task 1: Add config parameter

**Files:**
- Modify: `config.yaml:13-21` (risk_management section)

**Step 1: Add max_position_dollars to config**

Add under `risk_management:` section after `max_open_positions`:

```yaml
risk_management:
  max_position_size_pct: 10
  max_portfolio_risk_pct: 50.0
  stop_loss_pct: 5.0
  take_profit_pct: 5.0
  max_daily_loss_pct: 5.0
  emergency_stop_pct: 8.0
  max_open_positions: 1
  max_position_dollars: 10000  # NEW: Maximum single position value in dollars
```

**Step 2: Commit**

```bash
git add config.yaml
git commit -m "config: add max_position_dollars parameter for position size guard"
```

---

### Task 2: Write failing tests for position size violations

**Files:**
- Create: `tests/test_position_size_guard.py`

**Step 1: Write the failing tests**

```python
"""
Tests for position size violation detection and auto-liquidation.

Tests the position size guard that:
- Detects positions exceeding max_position_dollars
- Detects positions exceeding max_position_size_pct of portfolio
- Auto-liquidates violating positions
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class MockOrder:
    """Mock order response."""
    def __init__(self, status='filled', filled_price=100.0):
        self.id = 'test-order-123'
        self.status = status
        self.filled_avg_price = filled_price
        self.filled_qty = 10


class TestPositionSizeViolationDetection:
    """Test detection of position size violations."""

    def test_no_violation_under_dollar_limit(self):
        """Position under max_position_dollars is not flagged."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_broker.get_account.return_value = MagicMock(
                cash=50000,
                portfolio_value=100000,
                last_equity=100000
            )
            mock_broker.get_positions.return_value = []
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 100000
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 50,
                    'current_price': 150.0,  # Value: $7,500 < $10,000
                    'entry_price': 145.0,
                    'direction': 'LONG'
                }
            }

            violations = bot._check_position_size_violations()

            assert violations == []
            mock_broker.submit_order.assert_not_called()

    def test_violation_exceeds_dollar_limit(self):
        """Position exceeding max_position_dollars triggers liquidation."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_broker.get_account.return_value = MagicMock(
                cash=50000,
                portfolio_value=100000,
                last_equity=100000
            )
            mock_broker.get_positions.return_value = []
            mock_broker.submit_order.return_value = MockOrder()
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 100000
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 100,
                    'current_price': 150.0,  # Value: $15,000 > $10,000
                    'entry_price': 145.0,
                    'direction': 'LONG'
                }
            }

            violations = bot._check_position_size_violations()

            assert len(violations) == 1
            assert violations[0]['symbol'] == 'AAPL'
            assert violations[0]['reason'] == 'exceeds_max_position_dollars'
            mock_broker.submit_order.assert_called_once()

    def test_violation_exceeds_portfolio_pct(self):
        """Position exceeding max_position_size_pct triggers liquidation."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_broker.get_account.return_value = MagicMock(
                cash=50000,
                portfolio_value=50000,
                last_equity=50000
            )
            mock_broker.get_positions.return_value = []
            mock_broker.submit_order.return_value = MockOrder()
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 50000  # Small portfolio
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 50,
                    'current_price': 150.0,  # Value: $7,500 = 15% > 10%
                    'entry_price': 145.0,
                    'direction': 'LONG'
                }
            }

            violations = bot._check_position_size_violations()

            assert len(violations) == 1
            assert violations[0]['symbol'] == 'AAPL'
            assert violations[0]['reason'] == 'exceeds_max_position_pct'

    def test_no_violation_when_under_both_limits(self):
        """Position under both limits is not flagged."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_broker.get_account.return_value = MagicMock(
                cash=50000,
                portfolio_value=200000,
                last_equity=200000
            )
            mock_broker.get_positions.return_value = []
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 200000
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 50,
                    'current_price': 150.0,  # Value: $7,500 < $10K and < 10% of $200K
                    'entry_price': 145.0,
                    'direction': 'LONG'
                }
            }

            violations = bot._check_position_size_violations()

            assert violations == []


class TestPositionSizeLiquidation:
    """Test auto-liquidation of violating positions."""

    def test_long_position_liquidation(self):
        """LONG position is sold when violating."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_broker.get_account.return_value = MagicMock(
                cash=50000,
                portfolio_value=100000,
                last_equity=100000
            )
            mock_broker.get_positions.return_value = []
            mock_broker.submit_order.return_value = MockOrder()
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 100000
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 100,
                    'current_price': 150.0,
                    'entry_price': 145.0,
                    'direction': 'LONG'
                }
            }

            bot._check_position_size_violations()

            mock_broker.submit_order.assert_called_once_with(
                symbol='AAPL',
                qty=100,
                side='sell',
                type='market',
                time_in_force='day'
            )

    def test_short_position_liquidation(self):
        """SHORT position is bought back when violating."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_broker.get_account.return_value = MagicMock(
                cash=50000,
                portfolio_value=100000,
                last_equity=100000
            )
            mock_broker.get_positions.return_value = []
            mock_broker.submit_order.return_value = MockOrder()
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 100000
            bot.open_positions = {
                'TSLA': {
                    'symbol': 'TSLA',
                    'qty': 50,
                    'current_price': 250.0,  # Value: $12,500 > $10,000
                    'entry_price': 260.0,
                    'direction': 'SHORT'
                }
            }

            bot._check_position_size_violations()

            mock_broker.submit_order.assert_called_once_with(
                symbol='TSLA',
                qty=50,
                side='buy',
                type='market',
                time_in_force='day'
            )

    def test_multiple_violations_all_liquidated(self):
        """Multiple violating positions are all liquidated."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_broker.get_account.return_value = MagicMock(
                cash=50000,
                portfolio_value=100000,
                last_equity=100000
            )
            mock_broker.get_positions.return_value = []
            mock_broker.submit_order.return_value = MockOrder()
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 100000
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 100,
                    'current_price': 150.0,  # Value: $15,000
                    'entry_price': 145.0,
                    'direction': 'LONG'
                },
                'TSLA': {
                    'symbol': 'TSLA',
                    'qty': 50,
                    'current_price': 250.0,  # Value: $12,500
                    'entry_price': 260.0,
                    'direction': 'SHORT'
                }
            }

            violations = bot._check_position_size_violations()

            assert len(violations) == 2
            assert mock_broker.submit_order.call_count == 2


class TestPositionCleanupAfterLiquidation:
    """Test that positions are cleaned up after liquidation."""

    def test_position_removed_after_liquidation(self):
        """Position is removed from tracking after liquidation."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_broker.get_account.return_value = MagicMock(
                cash=50000,
                portfolio_value=100000,
                last_equity=100000
            )
            mock_broker.get_positions.return_value = []
            mock_broker.submit_order.return_value = MockOrder()
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 100000
            bot.highest_prices = {'AAPL': 150.0}
            bot.lowest_prices = {'AAPL': 145.0}
            bot.trailing_stops = {'AAPL': {'activated': False, 'price': 0.0}}
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 100,
                    'current_price': 150.0,
                    'entry_price': 145.0,
                    'direction': 'LONG'
                }
            }

            bot._check_position_size_violations()

            assert 'AAPL' not in bot.open_positions
            assert 'AAPL' not in bot.highest_prices
            assert 'AAPL' not in bot.lowest_prices
            assert 'AAPL' not in bot.trailing_stops
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_position_size_guard.py -v`
Expected: FAIL with "AttributeError: 'TradingBot' object has no attribute '_check_position_size_violations'"

**Step 3: Commit the failing tests**

```bash
git add tests/test_position_size_guard.py
git commit -m "test: add failing tests for position size guard"
```

---

### Task 3: Implement _check_position_size_violations method

**Files:**
- Modify: `bot.py:1078` (add method before run_trading_cycle)

**Step 1: Add the implementation**

Add this method after `fetch_data()` (around line 1078), before `run_trading_cycle()`:

```python
    def _check_position_size_violations(self) -> list:
        """
        Check for and liquidate positions exceeding size limits.

        Checks each position against:
        - max_position_dollars: Absolute dollar limit (default $10,000)
        - max_position_size_pct: Percentage of portfolio (from config)

        Returns:
            List of violation dicts with symbol, value, reason, liquidated
        """
        violations = []

        risk_config = self.config.get('risk_management', {})
        max_dollars = risk_config.get('max_position_dollars', 10000)
        max_pct = risk_config.get('max_position_size_pct', 10.0) / 100

        for symbol, pos in list(self.open_positions.items()):
            qty = pos['qty']
            current_price = pos.get('current_price', pos.get('entry_price', 0))
            direction = pos.get('direction', 'LONG')
            value = qty * current_price

            violation = None

            # Check absolute dollar limit
            if value > max_dollars:
                violation = {
                    'symbol': symbol,
                    'value': value,
                    'limit': max_dollars,
                    'reason': 'exceeds_max_position_dollars'
                }
            # Check percentage of portfolio limit
            elif self.portfolio_value > 0 and value > self.portfolio_value * max_pct:
                violation = {
                    'symbol': symbol,
                    'value': value,
                    'limit': self.portfolio_value * max_pct,
                    'reason': 'exceeds_max_position_pct'
                }

            if violation:
                logger.critical(
                    f"POSITION_VIOLATION | {symbol} | "
                    f"Value: ${value:,.2f} | Limit: ${violation['limit']:,.2f} | "
                    f"Reason: {violation['reason']} | ACTION: LIQUIDATING"
                )

                # Force liquidate
                try:
                    side = 'sell' if direction == 'LONG' else 'buy'
                    order = self.broker.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        type='market',
                        time_in_force='day'
                    )

                    if order:
                        violation['liquidated'] = True
                        violation['order_id'] = order.id

                        # Cleanup position tracking
                        self._cleanup_position(symbol)
                        del self.open_positions[symbol]

                        logger.critical(
                            f"POSITION_VIOLATION | {symbol} | LIQUIDATED | Order: {order.id}"
                        )
                    else:
                        violation['liquidated'] = False
                        logger.error(f"POSITION_VIOLATION | {symbol} | LIQUIDATION FAILED - no order")

                except Exception as e:
                    violation['liquidated'] = False
                    violation['error'] = str(e)
                    logger.error(f"POSITION_VIOLATION | {symbol} | LIQUIDATION ERROR: {e}", exc_info=True)

                violations.append(violation)

        return violations
```

**Step 2: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_position_size_guard.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add bot.py
git commit -m "feat(ODE-89): add _check_position_size_violations method"
```

---

### Task 4: Integrate into run_trading_cycle

**Files:**
- Modify: `bot.py:1079-1141` (run_trading_cycle method)

**Step 1: Add call to _check_position_size_violations**

In `run_trading_cycle()`, after the sync calls and drawdown guard update (around line 1140), add:

```python
            # 3. Check position size violations (before exit checks)
            # FIX (Jan 2026): ODE-89 - Auto-liquidate oversized positions
            violations = self._check_position_size_violations()
            if violations:
                logger.warning(f"POSITION_SIZE_GUARD | Liquidated {len(violations)} oversized positions")
```

The section should look like:

```python
            # 2. Update drawdown guard (must be done after sync)
            if self.drawdown_guard.enabled:
                # ... existing drawdown guard code ...

            # 3. Check position size violations (before exit checks)
            # FIX (Jan 2026): ODE-89 - Auto-liquidate oversized positions
            violations = self._check_position_size_violations()
            if violations:
                logger.warning(f"POSITION_SIZE_GUARD | Liquidated {len(violations)} oversized positions")

            # 4. Check exits for all positions (renumber from 2)
            logger.info(f"Checking exits for {len(self.open_positions)} positions")
```

**Step 2: Run all tests**

Run: `python3 -m pytest tests/test_position_size_guard.py tests/test_bot.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add bot.py
git commit -m "feat(ODE-89): integrate position size guard into trading cycle"
```

---

### Task 5: Add integration test

**Files:**
- Modify: `tests/test_position_size_guard.py`

**Step 1: Add integration test**

Add to the test file:

```python
class TestPositionSizeGuardIntegration:
    """Test position size guard integration with trading cycle."""

    def test_guard_runs_in_trading_cycle(self):
        """Verify _check_position_size_violations is called in run_trading_cycle."""
        with patch('bot.create_broker') as mock_broker_factory:
            mock_broker = MagicMock()
            mock_account = MagicMock(
                cash=50000,
                portfolio_value=100000,
                last_equity=100000,
                equity=100000
            )
            mock_broker.get_account.return_value = mock_account
            mock_broker.get_positions.return_value = []
            mock_broker_factory.return_value = mock_broker

            from bot import TradingBot
            bot = TradingBot()
            bot.portfolio_value = 100000
            bot.daily_starting_capital = 100000

            with patch.object(bot, '_check_position_size_violations') as mock_check:
                mock_check.return_value = []
                bot.run_trading_cycle()
                mock_check.assert_called_once()
```

**Step 2: Run tests**

Run: `python3 -m pytest tests/test_position_size_guard.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_position_size_guard.py
git commit -m "test(ODE-89): add integration test for position size guard"
```

---

### Task 6: Final verification and cleanup

**Step 1: Run full test suite**

Run: `python3 -m pytest -v`
Expected: All tests PASS

**Step 2: Verify no lint errors**

Run: `python3 -m py_compile bot.py && echo "Syntax OK"`
Expected: "Syntax OK"

**Step 3: Final commit if any cleanup needed**

```bash
git status
# If any uncommitted changes:
git add -A
git commit -m "chore(ODE-89): final cleanup"
```
