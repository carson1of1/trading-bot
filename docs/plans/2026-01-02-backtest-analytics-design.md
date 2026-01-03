# Backtest Analytics Design

## Summary

Add detailed analytics breakdowns to the backtest UI to identify performance drags. Display inline on the backtest page with collapsible panels.

## Requirements

1. Performance breakdown by strategy (Momentum, Mean Reversion, Breakout)
2. Performance breakdown by exit reason (stop_loss, take_profit, trailing_stop, eod_close, max_hold)
3. Performance breakdown by symbol (sorted worst to best to surface drags)
4. Include avg MAE% and MFE% to identify if trades are leaving money on table or not cutting losses fast

## API Changes

### New Models (api/main.py)

```python
class StrategyBreakdown(BaseModel):
    strategy: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    avg_mfe_pct: float  # Average max favorable excursion
    avg_mae_pct: float  # Average max adverse excursion

class ExitReasonBreakdown(BaseModel):
    exit_reason: str
    count: int
    total_pnl: float
    avg_pnl: float
    pct_of_trades: float

class SymbolBreakdown(BaseModel):
    symbol: str
    trades: int
    total_pnl: float
    win_rate: float
    avg_pnl: float
```

### BacktestResponse Changes

Add three new fields:
- `by_strategy: List[StrategyBreakdown]`
- `by_exit_reason: List[ExitReasonBreakdown]`
- `by_symbol: List[SymbolBreakdown]` (sorted by total_pnl ascending - worst first)

### TradeResult Changes

Add fields already tracked in backtest.py:
- `mfe_pct: float`
- `mae_pct: float`

## Frontend Changes

### Backtest Page (frontend/src/app/backtest/page.tsx)

Add three collapsible panels after the Trade List section:

1. **Strategy Performance Panel** - Table showing each strategy's trades, win rate, P&L, avg MFE/MAE
2. **Exit Reason Panel** - Table showing breakdown by exit_reason
3. **Symbol Performance Panel** - Table sorted worst to best, highlighting drags

### API Types (frontend/src/lib/api.ts)

Add TypeScript interfaces for new breakdown types.

## Performance Notes

- Analytics aggregation happens AFTER all trades are simulated
- Uses existing trade data - no additional calculations during simulation
- Aggregation is O(n) where n is number of trades - negligible overhead

## Implementation Order

1. Add new Pydantic models to api/main.py
2. Add analytics computation in format_backtest_results()
3. Add mfe_pct/mae_pct to TradeResult model
4. Update frontend TypeScript types
5. Add collapsible analytics panels to backtest page
