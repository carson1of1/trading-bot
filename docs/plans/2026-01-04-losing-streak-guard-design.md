# Losing Streak Guard Design

**Date**: 2026-01-04
**Status**: Approved
**Purpose**: Protect capital after consecutive losses by reducing position sizes until a profitable day

---

## Overview

After 2 losing trades in 3 days, cut risk by 50%. Resume normal risk after a green day.

This guards against:
- Behavioral tilt after losses
- Strategy-market mismatch periods
- Compounding losses during drawdowns

---

## Definitions

### Losing Trade
A trade whose realized P&L ≤ -0.5R, where R = the per-trade risk amount.

**Examples** (assuming $50 risk per trade):
- -$47 loss → **counts** (-$47 < -$25 threshold)
- -$8 scratch → **does not count** (-$8 > -$25 threshold)
- -$25 loss → **counts** (-$25 = -$25 threshold)

**Rationale**: Small scratches are noise, not behavioral failures. Funded firms care about risk events, not pennies.

### Green Day
Net positive REALIZED P&L for the complete trading day.

**Rationale**:
- Funded firms evaluate realized equity
- Unrealized P&L is irrelevant to compliance
- One big loser + one tiny winner is not a green day

---

## Data Model

### TradeResult (per closed trade)
```python
@dataclass
class TradeResult:
    symbol: str
    close_time: datetime
    realized_pnl: float      # Actual P&L in dollars
    risk_amount: float       # The R value (what was risked)
    is_losing_trade: bool    # True if pnl <= -0.5 * risk_amount
```

### DailySummary (computed at EOD)
```python
@dataclass
class DailySummary:
    date: date
    net_realized_pnl: float  # Sum of all realized P&L
    is_green_day: bool       # net_realized_pnl > 0
    losing_trade_count: int  # Trades where pnl <= -0.5R
```

### Persistence
- SQLite table: `losing_streak_state` in `logs/trades.db`
- Columns: date, net_pnl, losing_count, is_green
- Survives bot restarts

---

## State Machine

```
NORMAL (multiplier = 1.0)
    │
    ▼ [2+ losing trades in 3 days]
    │
THROTTLED (multiplier = 0.5)
    │
    ▼ [green day at market close]
    │
NORMAL
```

### Hysteresis
- Once THROTTLED, stays throttled until green day
- Green day must be after entering throttled state
- If green day occurs, then 2 losers next day → immediately throttled again

---

## Configuration

```yaml
losing_streak_guard:
  enabled: true
  losing_threshold_r: 0.5      # Trade counts as loss if pnl <= -0.5R
  lookback_days: 3             # Rolling window
  min_losing_trades: 2         # Trigger after this many losers
  throttle_multiplier: 0.5     # Reduce size to 50%
```

---

## Integration Points

### 1. New Class: LosingStreakGuard (core/risk.py)

```python
class LosingStreakGuard:
    def __init__(self, config: dict = None)

    # Called when a trade closes
    def record_trade(self, symbol, realized_pnl, risk_amount, close_time)

    # Called at market close (4 PM ET)
    def end_of_day(self, date)

    # Get current multiplier (1.0 or 0.5)
    @property
    def position_size_multiplier(self) -> float

    # Check if guard is active
    @property
    def is_throttled(self) -> bool
```

### 2. RiskManager.calculate_position_size()

Add multiplier after existing VIX adjustment (~line 413):

```python
if self.losing_streak_guard:
    streak_multiplier = self.losing_streak_guard.position_size_multiplier
    if streak_multiplier < 1.0:
        self.logger.info(f"Losing streak guard: size {position_size} -> {int(position_size * streak_multiplier)}")
        position_size = int(position_size * streak_multiplier)
```

### 3. bot.py Trade Close Handler

After recording trade in database:

```python
if hasattr(self, 'losing_streak_guard'):
    self.losing_streak_guard.record_trade(
        symbol=symbol,
        realized_pnl=realized_pnl,
        risk_amount=position_risk,
        close_time=datetime.now()
    )
```

### 4. Stacking with DailyDrawdownGuard

Both guards multiply together:
- DailyDrawdownGuard at WARNING (0.5) + LosingStreakGuard throttled (0.5) = 0.25 multiplier
- This is intentional: multiple risk signals = more aggressive de-risking

---

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| Bot restarts mid-day | Load state from SQLite, continue tracking |
| No trades for 2+ days | Rolling window clears naturally, stays NORMAL |
| Partial day (early close) | EOD evaluation at actual close time |
| First day of trading | No history = NORMAL, start accumulating |
| Multiple losses same day | Each counts separately toward 2-trade threshold |
| Scratch trade (-$2 on $50 risk) | Does NOT count (-$2 > -$25 threshold) |
| Green day but still 2 losers in window | Resets to NORMAL (green day is the reset trigger) |

---

## Logging

```
STREAK_GUARD | TRADE_RECORDED | AAPL | PnL: -$47.23, Risk: $50.00, IsLoser: True
STREAK_GUARD | THROTTLED | 2 losing trades in 3 days | Multiplier: 0.5
STREAK_GUARD | GREEN_DAY | Net PnL: +$23.45 | Resuming normal risk
```

---

## Testing Strategy

### Unit Tests (tests/test_losing_streak_guard.py)

| Test | Description |
|------|-------------|
| `test_no_trades_stays_normal` | Empty state returns multiplier 1.0 |
| `test_one_loser_stays_normal` | Single losing trade doesn't trigger |
| `test_two_losers_triggers_throttle` | 2 losers in 3 days → multiplier 0.5 |
| `test_scratch_trade_not_counted` | -$10 on $50 risk doesn't count as loser |
| `test_green_day_resets` | Throttled + green day → back to normal |
| `test_losers_outside_window_ignored` | 4-day-old loser doesn't count |
| `test_persistence_survives_restart` | Save/load from SQLite works |
| `test_stacks_with_drawdown_guard` | Both guards active → 0.25 multiplier |

### Backtest Validation
- Run existing backtest with guard enabled
- Compare: trades taken, position sizes, final P&L
- Verify guard activates during losing streaks in historical data

---

## File Locations

| Component | Location |
|-----------|----------|
| `LosingStreakGuard` class | `core/risk.py` |
| `TradeResult` dataclass | `core/risk.py` |
| Config section | `config.yaml` |
| SQLite table | `logs/trades.db` |
| Integration hook | `bot.py` trade close handler |
| Position size multiplier | `RiskManager.calculate_position_size()` |
| Tests | `tests/test_losing_streak_guard.py` |
