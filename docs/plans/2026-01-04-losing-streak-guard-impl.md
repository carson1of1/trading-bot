# Losing Streak Guard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce position sizes by 50% after 2 losing trades (≤-0.5R) in 3 days, resume after a green day.

**Architecture:** New `LosingStreakGuard` class in `core/risk.py` mirrors `DailyDrawdownGuard` pattern. Tracks trade results in SQLite, applies multiplier through `RiskManager`.

**Tech Stack:** Python, SQLite, pytest

---

## Task 1: Create TradeResult Dataclass

**Files:**
- Modify: `core/risk.py:1532` (before DailyDrawdownGuard class)

**Step 1: Write the failing test**

Create file `tests/test_losing_streak_guard.py`:

```python
"""
Tests for LosingStreakGuard

Tests the multi-day losing streak protection system:
- Trigger: 2+ losing trades (≤-0.5R) within 3 days
- Reset: Green day (net positive realized P&L)
- Effect: Position sizes reduced to 50%
"""

import pytest
from datetime import datetime, date, timedelta
from core.risk import LosingStreakGuard, TradeResult


class TestTradeResult:
    """Test TradeResult dataclass."""

    def test_trade_result_is_loser_true(self):
        """Test that trade with pnl <= -0.5R is marked as loser."""
        result = TradeResult(
            symbol='AAPL',
            close_time=datetime.now(),
            realized_pnl=-30.0,  # Lost $30
            risk_amount=50.0     # Risked $50
        )
        # -30 <= -25 (0.5 * 50), so is_losing_trade should be True
        assert result.is_losing_trade is True

    def test_trade_result_is_loser_false(self):
        """Test that scratch trade (pnl > -0.5R) is not a loser."""
        result = TradeResult(
            symbol='AAPL',
            close_time=datetime.now(),
            realized_pnl=-10.0,  # Lost only $10
            risk_amount=50.0     # Risked $50
        )
        # -10 > -25 (0.5 * 50), so is_losing_trade should be False
        assert result.is_losing_trade is False

    def test_trade_result_winner_not_loser(self):
        """Test that winning trade is not a loser."""
        result = TradeResult(
            symbol='AAPL',
            close_time=datetime.now(),
            realized_pnl=25.0,   # Won $25
            risk_amount=50.0     # Risked $50
        )
        assert result.is_losing_trade is False
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_losing_streak_guard.py::TestTradeResult -v`
Expected: FAIL with "cannot import name 'LosingStreakGuard'"

**Step 3: Write minimal implementation**

Add to `core/risk.py` before the `DailyDrawdownGuard` class (around line 1532):

```python
# =============================================================================
# LOSING STREAK GUARD (Jan 4, 2026)
# =============================================================================
#
# Protects capital after consecutive losses by reducing position sizes.
# Trigger: 2+ losing trades (≤-0.5R) within a rolling 3-day window
# Reset: Green day (net positive realized P&L)
# Effect: Position sizes multiplied by 0.5
# =============================================================================


@dataclass
class TradeResult:
    """
    Record of a closed trade for streak tracking.

    A trade is considered a "loser" if realized_pnl <= -0.5 * risk_amount.
    This filters out scratches and noise, only counting meaningful losses.
    """
    symbol: str
    close_time: datetime
    realized_pnl: float      # Actual P&L in dollars
    risk_amount: float       # The R value (what was risked)

    @property
    def is_losing_trade(self) -> bool:
        """True if this trade lost 0.5R or more."""
        threshold = -0.5 * self.risk_amount
        return self.realized_pnl <= threshold
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_losing_streak_guard.py::TestTradeResult -v`
Expected: PASS

**Step 5: Commit**

```bash
git add core/risk.py tests/test_losing_streak_guard.py
git commit -m "feat(risk): add TradeResult dataclass for losing streak tracking"
```

---

## Task 2: Create LosingStreakGuard Class - Init and Properties

**Files:**
- Modify: `core/risk.py` (after TradeResult)
- Modify: `tests/test_losing_streak_guard.py`

**Step 1: Write the failing test**

Add to `tests/test_losing_streak_guard.py`:

```python
class TestLosingStreakGuardInit:
    """Test LosingStreakGuard initialization."""

    def test_default_config(self):
        """Test initialization with default config."""
        guard = LosingStreakGuard()

        assert guard.enabled is True
        assert guard.losing_threshold_r == 0.5
        assert guard.lookback_days == 3
        assert guard.min_losing_trades == 2
        assert guard.throttle_multiplier == 0.5
        assert guard.position_size_multiplier == 1.0
        assert guard.is_throttled is False

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = {
            'losing_streak_guard': {
                'enabled': True,
                'losing_threshold_r': 0.75,
                'lookback_days': 5,
                'min_losing_trades': 3,
                'throttle_multiplier': 0.25
            }
        }
        guard = LosingStreakGuard(config)

        assert guard.losing_threshold_r == 0.75
        assert guard.lookback_days == 5
        assert guard.min_losing_trades == 3
        assert guard.throttle_multiplier == 0.25

    def test_disabled_config(self):
        """Test initialization with guard disabled."""
        config = {'losing_streak_guard': {'enabled': False}}
        guard = LosingStreakGuard(config)

        assert guard.enabled is False
        assert guard.position_size_multiplier == 1.0
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_losing_streak_guard.py::TestLosingStreakGuardInit -v`
Expected: FAIL with "LosingStreakGuard not defined" or similar

**Step 3: Write minimal implementation**

Add to `core/risk.py` after TradeResult:

```python
class LosingStreakGuard:
    """
    Multi-day losing streak protection.

    Reduces position sizes after consecutive losses to prevent
    compounding drawdowns during strategy-market mismatch periods.

    TRIGGER: 2+ trades with P&L ≤ -0.5R within 3 days
    RESET: Green day (net positive realized P&L)
    EFFECT: Position sizes multiplied by 0.5

    USAGE:
        guard = LosingStreakGuard(config)

        # On trade close:
        guard.record_trade(symbol, realized_pnl, risk_amount, close_time)

        # Get position size multiplier:
        size = base_size * guard.position_size_multiplier
    """

    def __init__(self, config: dict = None):
        """
        Initialize the losing streak guard.

        Args:
            config: Configuration dict with losing_streak_guard section.
        """
        self.logger = logging.getLogger(__name__)

        # Load config
        config = config or {}
        guard_config = config.get('losing_streak_guard', {})

        # Enable/disable
        self.enabled = guard_config.get('enabled', True)

        # Thresholds
        self.losing_threshold_r = guard_config.get('losing_threshold_r', 0.5)
        self.lookback_days = guard_config.get('lookback_days', 3)
        self.min_losing_trades = guard_config.get('min_losing_trades', 2)
        self.throttle_multiplier = guard_config.get('throttle_multiplier', 0.5)

        # State
        self._throttled = False
        self._throttle_start_date: Optional[date] = None
        self._trade_history: List[TradeResult] = []
        self._daily_pnl: Dict[date, float] = {}

        if self.enabled:
            self.logger.info(
                f"LosingStreakGuard initialized: "
                f"threshold=-{self.losing_threshold_r}R, "
                f"lookback={self.lookback_days} days, "
                f"min_losers={self.min_losing_trades}, "
                f"throttle={self.throttle_multiplier}"
            )
        else:
            self.logger.info("LosingStreakGuard: DISABLED")

    @property
    def is_throttled(self) -> bool:
        """Whether position sizes are currently reduced."""
        return self._throttled and self.enabled

    @property
    def position_size_multiplier(self) -> float:
        """Current position size multiplier (1.0 normal, 0.5 throttled)."""
        if not self.enabled:
            return 1.0
        return self.throttle_multiplier if self._throttled else 1.0
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_losing_streak_guard.py::TestLosingStreakGuardInit -v`
Expected: PASS

**Step 5: Commit**

```bash
git add core/risk.py tests/test_losing_streak_guard.py
git commit -m "feat(risk): add LosingStreakGuard class with init and properties"
```

---

## Task 3: Implement record_trade Method

**Files:**
- Modify: `core/risk.py`
- Modify: `tests/test_losing_streak_guard.py`

**Step 1: Write the failing test**

Add to `tests/test_losing_streak_guard.py`:

```python
class TestRecordTrade:
    """Test record_trade method."""

    def test_record_single_trade(self):
        """Test recording a single trade."""
        guard = LosingStreakGuard()

        guard.record_trade(
            symbol='AAPL',
            realized_pnl=-30.0,
            risk_amount=50.0,
            close_time=datetime.now()
        )

        assert len(guard._trade_history) == 1
        assert guard._trade_history[0].symbol == 'AAPL'
        assert guard._trade_history[0].is_losing_trade is True

    def test_one_loser_stays_normal(self):
        """Test that single losing trade doesn't trigger throttle."""
        guard = LosingStreakGuard()

        guard.record_trade('AAPL', -30.0, 50.0, datetime.now())

        assert guard.is_throttled is False
        assert guard.position_size_multiplier == 1.0

    def test_two_losers_triggers_throttle(self):
        """Test that 2 losing trades in 3 days triggers throttle."""
        guard = LosingStreakGuard()
        now = datetime.now()

        # Two losing trades same day
        guard.record_trade('AAPL', -30.0, 50.0, now)
        guard.record_trade('MSFT', -40.0, 50.0, now)

        assert guard.is_throttled is True
        assert guard.position_size_multiplier == 0.5

    def test_scratch_trade_not_counted(self):
        """Test that scratch trades don't count toward streak."""
        guard = LosingStreakGuard()
        now = datetime.now()

        # One real loser, one scratch
        guard.record_trade('AAPL', -30.0, 50.0, now)  # Real loser (-30 <= -25)
        guard.record_trade('MSFT', -10.0, 50.0, now)  # Scratch (-10 > -25)

        assert guard.is_throttled is False  # Only 1 real loser

    def test_losers_outside_window_ignored(self):
        """Test that old losing trades are not counted."""
        guard = LosingStreakGuard()
        now = datetime.now()
        old = now - timedelta(days=5)  # Outside 3-day window

        guard.record_trade('AAPL', -30.0, 50.0, old)  # Old loser
        guard.record_trade('MSFT', -30.0, 50.0, now)  # Recent loser

        assert guard.is_throttled is False  # Only 1 in window
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_losing_streak_guard.py::TestRecordTrade -v`
Expected: FAIL with "record_trade not defined" or assertion errors

**Step 3: Write minimal implementation**

Add to `LosingStreakGuard` class in `core/risk.py`:

```python
    def record_trade(self, symbol: str, realized_pnl: float,
                     risk_amount: float, close_time: datetime = None):
        """
        Record a closed trade and check streak condition.

        Args:
            symbol: Stock symbol
            realized_pnl: Realized P&L in dollars
            risk_amount: The R value (risk amount for this trade)
            close_time: Trade close time (defaults to now)
        """
        if not self.enabled:
            return

        if close_time is None:
            close_time = datetime.now()

        # Create trade result
        trade = TradeResult(
            symbol=symbol,
            close_time=close_time,
            realized_pnl=realized_pnl,
            risk_amount=risk_amount
        )

        self._trade_history.append(trade)

        # Update daily P&L
        trade_date = close_time.date()
        if trade_date not in self._daily_pnl:
            self._daily_pnl[trade_date] = 0.0
        self._daily_pnl[trade_date] += realized_pnl

        # Log
        self.logger.info(
            f"STREAK_GUARD | TRADE_RECORDED | {symbol} | "
            f"PnL: ${realized_pnl:+.2f}, Risk: ${risk_amount:.2f}, "
            f"IsLoser: {trade.is_losing_trade}"
        )

        # Check streak condition
        self._check_streak_condition()

    def _count_recent_losers(self) -> int:
        """Count losing trades within the lookback window."""
        cutoff = datetime.now() - timedelta(days=self.lookback_days)

        count = 0
        for trade in self._trade_history:
            if trade.close_time >= cutoff and trade.is_losing_trade:
                count += 1

        return count

    def _check_streak_condition(self):
        """Check if streak condition is met and update throttle state."""
        if self._throttled:
            # Already throttled, don't re-trigger
            return

        loser_count = self._count_recent_losers()

        if loser_count >= self.min_losing_trades:
            self._throttled = True
            self._throttle_start_date = datetime.now().date()

            self.logger.warning(
                f"STREAK_GUARD | THROTTLED | "
                f"{loser_count} losing trades in {self.lookback_days} days | "
                f"Multiplier: {self.throttle_multiplier}"
            )
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_losing_streak_guard.py::TestRecordTrade -v`
Expected: PASS

**Step 5: Commit**

```bash
git add core/risk.py tests/test_losing_streak_guard.py
git commit -m "feat(risk): add LosingStreakGuard.record_trade with streak detection"
```

---

## Task 4: Implement end_of_day Reset Logic

**Files:**
- Modify: `core/risk.py`
- Modify: `tests/test_losing_streak_guard.py`

**Step 1: Write the failing test**

Add to `tests/test_losing_streak_guard.py`:

```python
class TestEndOfDay:
    """Test end_of_day reset logic."""

    def test_green_day_resets_throttle(self):
        """Test that green day resets throttle to normal."""
        guard = LosingStreakGuard()
        now = datetime.now()
        today = now.date()

        # Trigger throttle
        guard.record_trade('AAPL', -30.0, 50.0, now)
        guard.record_trade('MSFT', -30.0, 50.0, now)
        assert guard.is_throttled is True

        # Simulate winning trade that makes day green
        guard.record_trade('GOOGL', 100.0, 50.0, now)

        # End of day - should reset (net P&L = -30 -30 +100 = +40)
        guard.end_of_day(today)

        assert guard.is_throttled is False
        assert guard.position_size_multiplier == 1.0

    def test_red_day_stays_throttled(self):
        """Test that red day keeps throttle active."""
        guard = LosingStreakGuard()
        now = datetime.now()
        today = now.date()

        # Trigger throttle
        guard.record_trade('AAPL', -30.0, 50.0, now)
        guard.record_trade('MSFT', -30.0, 50.0, now)
        assert guard.is_throttled is True

        # End of day - still red (net P&L = -60)
        guard.end_of_day(today)

        assert guard.is_throttled is True

    def test_green_day_must_be_after_throttle(self):
        """Test that green day before throttle doesn't prevent throttle."""
        guard = LosingStreakGuard()
        now = datetime.now()
        yesterday = (now - timedelta(days=1)).date()
        today = now.date()

        # Yesterday was green
        guard._daily_pnl[yesterday] = 50.0
        guard.end_of_day(yesterday)  # No throttle, so no-op

        # Today: two losers trigger throttle
        guard.record_trade('AAPL', -30.0, 50.0, now)
        guard.record_trade('MSFT', -30.0, 50.0, now)

        assert guard.is_throttled is True  # Yesterday's green doesn't help

    def test_no_trades_today_no_reset(self):
        """Test that day with no trades doesn't reset."""
        guard = LosingStreakGuard()
        now = datetime.now()
        today = now.date()
        tomorrow = today + timedelta(days=1)

        # Trigger throttle today
        guard.record_trade('AAPL', -30.0, 50.0, now)
        guard.record_trade('MSFT', -30.0, 50.0, now)
        assert guard.is_throttled is True

        # End tomorrow with no trades
        guard.end_of_day(tomorrow)

        assert guard.is_throttled is True  # No trades = no green day
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_losing_streak_guard.py::TestEndOfDay -v`
Expected: FAIL with "end_of_day not defined"

**Step 3: Write minimal implementation**

Add to `LosingStreakGuard` class in `core/risk.py`:

```python
    def end_of_day(self, trading_date: date = None):
        """
        Evaluate end-of-day condition and potentially reset throttle.

        Call this at market close (4 PM ET).

        Args:
            trading_date: The trading date to evaluate (defaults to today)
        """
        if not self.enabled:
            return

        if trading_date is None:
            trading_date = datetime.now().date()

        # Get daily P&L
        daily_pnl = self._daily_pnl.get(trading_date, 0.0)

        # Check for green day reset
        if self._throttled and daily_pnl > 0:
            # Only reset if green day is AFTER throttle started
            if self._throttle_start_date and trading_date >= self._throttle_start_date:
                self._throttled = False
                self._throttle_start_date = None

                self.logger.info(
                    f"STREAK_GUARD | GREEN_DAY | "
                    f"Net PnL: ${daily_pnl:+.2f} | Resuming normal risk"
                )

        # Prune old trade history (keep only lookback_days + buffer)
        self._prune_old_trades()

    def _prune_old_trades(self):
        """Remove trades older than lookback window + 1 day buffer."""
        cutoff = datetime.now() - timedelta(days=self.lookback_days + 1)

        self._trade_history = [
            t for t in self._trade_history
            if t.close_time >= cutoff
        ]

        # Also prune daily P&L
        cutoff_date = cutoff.date()
        self._daily_pnl = {
            d: pnl for d, pnl in self._daily_pnl.items()
            if d >= cutoff_date
        }
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_losing_streak_guard.py::TestEndOfDay -v`
Expected: PASS

**Step 5: Commit**

```bash
git add core/risk.py tests/test_losing_streak_guard.py
git commit -m "feat(risk): add LosingStreakGuard.end_of_day with green day reset"
```

---

## Task 5: Add get_status and Factory Function

**Files:**
- Modify: `core/risk.py`
- Modify: `tests/test_losing_streak_guard.py`

**Step 1: Write the failing test**

Add to `tests/test_losing_streak_guard.py`:

```python
from core.risk import create_losing_streak_guard


class TestGetStatus:
    """Test get_status method."""

    def test_status_contains_all_fields(self):
        """Test that get_status returns all expected fields."""
        guard = LosingStreakGuard()
        guard.record_trade('AAPL', -30.0, 50.0, datetime.now())

        status = guard.get_status()

        assert 'enabled' in status
        assert 'is_throttled' in status
        assert 'position_size_multiplier' in status
        assert 'recent_losers' in status
        assert 'lookback_days' in status
        assert 'min_losing_trades' in status

    def test_status_values_correct(self):
        """Test that get_status returns correct values."""
        guard = LosingStreakGuard()
        now = datetime.now()

        guard.record_trade('AAPL', -30.0, 50.0, now)
        guard.record_trade('MSFT', -30.0, 50.0, now)

        status = guard.get_status()

        assert status['is_throttled'] is True
        assert status['position_size_multiplier'] == 0.5
        assert status['recent_losers'] == 2


class TestFactoryFunction:
    """Test create_losing_streak_guard factory."""

    def test_factory_creates_guard(self):
        """Test factory function creates configured guard."""
        config = {
            'losing_streak_guard': {
                'enabled': True,
                'throttle_multiplier': 0.25
            }
        }
        guard = create_losing_streak_guard(config)

        assert isinstance(guard, LosingStreakGuard)
        assert guard.throttle_multiplier == 0.25
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_losing_streak_guard.py::TestGetStatus tests/test_losing_streak_guard.py::TestFactoryFunction -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `LosingStreakGuard` class in `core/risk.py`:

```python
    def get_status(self) -> dict:
        """
        Get current guard status.

        Returns:
            Dict with current state information
        """
        return {
            'enabled': self.enabled,
            'is_throttled': self.is_throttled,
            'position_size_multiplier': self.position_size_multiplier,
            'recent_losers': self._count_recent_losers(),
            'lookback_days': self.lookback_days,
            'min_losing_trades': self.min_losing_trades,
            'throttle_multiplier': self.throttle_multiplier,
            'throttle_start_date': str(self._throttle_start_date) if self._throttle_start_date else None,
            'trade_count': len(self._trade_history)
        }
```

Add after the `LosingStreakGuard` class:

```python
def create_losing_streak_guard(config: dict = None) -> LosingStreakGuard:
    """
    Factory function to create LosingStreakGuard with proper configuration.

    Args:
        config: Configuration dict with losing_streak_guard section

    Returns:
        Configured LosingStreakGuard instance
    """
    return LosingStreakGuard(config)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_losing_streak_guard.py::TestGetStatus tests/test_losing_streak_guard.py::TestFactoryFunction -v`
Expected: PASS

**Step 5: Commit**

```bash
git add core/risk.py tests/test_losing_streak_guard.py
git commit -m "feat(risk): add LosingStreakGuard.get_status and factory function"
```

---

## Task 6: Export from core/__init__.py

**Files:**
- Modify: `core/__init__.py`

**Step 1: Write the failing test**

```python
# In tests/test_losing_streak_guard.py, change import at top:
from core import LosingStreakGuard, TradeResult, create_losing_streak_guard
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_losing_streak_guard.py -v`
Expected: FAIL with import error

**Step 3: Write minimal implementation**

Modify `core/__init__.py` to add exports:

```python
from .risk import (
    RiskManager,
    ExitManager,
    PositionExitState,
    create_exit_manager,
    DailyDrawdownGuard,
    DrawdownTier,
    create_drawdown_guard,
    LosingStreakGuard,
    TradeResult,
    create_losing_streak_guard,
)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_losing_streak_guard.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add core/__init__.py tests/test_losing_streak_guard.py
git commit -m "feat(core): export LosingStreakGuard from core module"
```

---

## Task 7: Add Config Section

**Files:**
- Modify: `config.yaml`

**Step 1: Add config section**

Add to `config.yaml` after `daily_drawdown_guard`:

```yaml
losing_streak_guard:
  enabled: true
  losing_threshold_r: 0.5      # Trade counts as loss if pnl <= -0.5R
  lookback_days: 3             # Rolling window
  min_losing_trades: 2         # Trigger after this many losers
  throttle_multiplier: 0.5     # Reduce size to 50%
```

**Step 2: Commit**

```bash
git add config.yaml
git commit -m "config: add losing_streak_guard settings"
```

---

## Task 8: Integrate with bot.py - Store risk_amount

**Files:**
- Modify: `bot.py:731-739` (position tracking)

**Step 1: Calculate and store risk_amount**

In `bot.py`, modify `execute_entry` to calculate and store risk_amount.

Find line ~688 where position size is calculated:
```python
qty = self.risk_manager.calculate_position_size(
    self.portfolio_value, realistic_entry_price, stop_price
)
```

After this, add:
```python
# Calculate risk amount for losing streak tracking
risk_amount = abs(realistic_entry_price - stop_price) * qty
```

Then update the position dict at line ~731:
```python
self.open_positions[symbol] = {
    'symbol': symbol,
    'qty': fill_qty,
    'entry_price': fill_price,
    'direction': direction,
    'entry_time': datetime.now(),
    'strategy': strategy,
    'reasoning': reasoning,
    'risk_amount': risk_amount,  # ADD THIS LINE
}
```

**Step 2: Commit**

```bash
git add bot.py
git commit -m "feat(bot): store risk_amount in position for streak tracking"
```

---

## Task 9: Integrate with bot.py - Record Trade

**Files:**
- Modify: `bot.py` (around line 830-840)

**Step 1: Initialize LosingStreakGuard in TradingBot.__init__**

Find the `__init__` method (around line 70) where other guards are initialized.
After `self.drawdown_guard = DailyDrawdownGuard(self.config)`, add:

```python
# Initialize losing streak guard
from core import LosingStreakGuard
self.losing_streak_guard = LosingStreakGuard(self.config)
```

**Step 2: Record trade on exit**

In `execute_exit` method (around line 833), after:
```python
if self.drawdown_guard.enabled:
    self.drawdown_guard.record_realized_pnl(pnl)
```

Add:
```python
# Record trade for losing streak guard
if self.losing_streak_guard.enabled:
    risk_amount = position.get('risk_amount', abs(pnl))  # Fallback to pnl if not stored
    self.losing_streak_guard.record_trade(
        symbol=symbol,
        realized_pnl=pnl,
        risk_amount=risk_amount,
        close_time=datetime.now()
    )
```

**Step 3: Commit**

```bash
git add bot.py
git commit -m "feat(bot): integrate LosingStreakGuard on trade close"
```

---

## Task 10: Apply Position Size Multiplier

**Files:**
- Modify: `bot.py` (around line 692-699)

**Step 1: Apply multiplier after drawdown guard**

In `execute_entry`, after the drawdown guard multiplier (around line 699):
```python
if self.drawdown_guard.enabled and self.drawdown_guard.position_size_multiplier < 1.0:
    ...
```

Add:
```python
# Apply losing streak guard multiplier
if self.losing_streak_guard.enabled and self.losing_streak_guard.position_size_multiplier < 1.0:
    original_qty = qty
    qty = int(qty * self.losing_streak_guard.position_size_multiplier)
    if qty != original_qty:
        logger.info(
            f"STREAK_GUARD | Position size reduced: {original_qty} -> {qty} shares "
            f"({self.losing_streak_guard.position_size_multiplier*100:.0f}% multiplier)"
        )
```

**Step 2: Commit**

```bash
git add bot.py
git commit -m "feat(bot): apply LosingStreakGuard position size multiplier"
```

---

## Task 11: Run All Tests

**Step 1: Run full test suite**

```bash
python3 -m pytest tests/test_losing_streak_guard.py -v
python3 -m pytest tests/ -v --ignore=tests/test_walk_forward.py
```

Expected: All tests pass

**Step 2: Final commit if needed**

```bash
git status
# If any uncommitted changes, commit them
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | TradeResult dataclass | core/risk.py, tests/test_losing_streak_guard.py |
| 2 | LosingStreakGuard init | core/risk.py, tests/test_losing_streak_guard.py |
| 3 | record_trade method | core/risk.py, tests/test_losing_streak_guard.py |
| 4 | end_of_day reset | core/risk.py, tests/test_losing_streak_guard.py |
| 5 | get_status + factory | core/risk.py, tests/test_losing_streak_guard.py |
| 6 | Export from core | core/__init__.py |
| 7 | Config section | config.yaml |
| 8 | Store risk_amount | bot.py |
| 9 | Record trade on exit | bot.py |
| 10 | Apply multiplier | bot.py |
| 11 | Run all tests | - |
