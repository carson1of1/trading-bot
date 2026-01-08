# Hot Stocks Feed Design

**Date:** 2026-01-08
**Status:** Approved

## Summary

Add a "hot stocks" feed that fetches top weekly movers and merges them into the scanner pool at bot startup. This expands exposure to capture new volatile stocks (like the next RGTI) without manually updating `universe.yaml`.

## Problem

The scanner picks from a static 400-stock universe curated based on 2025 performance. If a new stock becomes highly volatile in 2026, we'd miss it entirely because it's not in the pool.

## Solution

Fetch top 50 weekly gainers from Yahoo Finance at bot startup. Filter them, then temporarily add new symbols to the scanner pool. The scanner's existing volatility scoring decides if they're worth trading.

**Key behaviors:**
- New stocks are added for **that day only** (temporary)
- If a stock stays hot, it naturally reappears the next day
- If it cools off, it drops out - no manual cleanup needed
- Failures never block trading - just continue with static universe

## Flow

```
Bot Starts (any time)
       │
       ▼
┌─────────────────────┐
│ Preflight Checks    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Fetch Hot Stocks    │  ◀── NEW STEP
│ (merge into pool)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Scanner Runs        │
│ (picks top 10)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Trading Cycle       │
└─────────────────────┘
```

## Filters

| Filter | Value | Why |
|--------|-------|-----|
| Top weekly % gainers | 50 stocks | Cast wide net, scanner filters |
| Price | $5 - $1000 | Match existing scanner filters |
| Volume | 500k+ avg | Enough liquidity to trade |
| Exclude | Already in universe | Only add new discoveries |

## Code Structure

### New file: `core/hot_stocks.py`

```python
class HotStocksFeed:
    """Fetches top weekly movers to expand scanner pool."""

    def __init__(self, config: dict):
        self.min_price = config.get('min_price', 5)
        self.max_price = config.get('max_price', 1000)
        self.min_volume = config.get('min_volume', 500_000)
        self.top_n = config.get('top_n', 50)
        self.cache_file = 'data/cache/hot_stocks.json'

    def fetch(self) -> List[str]:
        """Fetch hot stocks, using cache if fresh."""

    def _fetch_from_yahoo(self) -> List[str]:
        """Scrape Yahoo Finance top gainers."""

    def _filter_symbols(self, symbols: List[dict]) -> List[str]:
        """Apply price/volume filters."""
```

### Integration in `bot.py`

```python
def startup(self):
    self.run_preflight_checks()

    # Expand scanner pool with hot stocks
    hot_feed = HotStocksFeed(self.config.get('hot_stocks', {}))
    hot_symbols = hot_feed.fetch()
    self.scanner.add_temporary_symbols(hot_symbols)

    self.run_scanner()
```

### Config addition (`config.yaml`)

```yaml
hot_stocks:
  enabled: true
  top_n: 50
  min_price: 5
  min_volume: 500000
```

## Error Handling

| Failure | Behavior |
|---------|----------|
| Yahoo rate limited | Use cache if available, else skip |
| Yahoo down | Log warning, continue with static universe |
| Bad data | Skip, continue with static universe |
| Network timeout | 10 second timeout, then skip |

**Principle:** Hot stocks are a bonus, never block trading.

## Logging

```
INFO: Hot stocks feed: fetched 47 gainers, 12 new symbols added to pool
INFO: Added hot stocks: ['NEWAI', 'XYZ', 'BOOM', ...]
INFO: Scanner pool: 400 static + 12 hot = 412 total
```

## Testing

1. Unit test: Mock Yahoo response, verify filtering
2. Integration test: Verify hot stocks merge into scanner pool
3. Manual: Run once, confirm new symbols appear in scanner ranking
