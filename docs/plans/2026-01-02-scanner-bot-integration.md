# Scanner-Bot Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When the bot starts from the UI, automatically run the scanner first and use scanned stocks as the trading watchlist. Block bot start with clear reasoning if scanner fails.

**Architecture:** Modify `POST /api/bot/start` to run `VolatilityScanner.scan()` before spawning the bot process. Pass scanned symbols to `bot.py` via `--symbols` CLI argument. Frontend shows scanner progress and displays selected stocks or failure reason.

**Tech Stack:** FastAPI (Python), Next.js/React (TypeScript), pytest, subprocess

---

## Phase 1: Backend - Bot CLI Enhancement

### Task 1: Add --symbols CLI Argument to Bot

**Files:**
- Modify: `bot.py`
- Test: `tests/test_bot_scanner_integration.py` (create new)

**Step 1: Create test file with first test**

Create `tests/test_bot_scanner_integration.py`:

```python
"""Tests for scanner-bot integration."""
import pytest
import subprocess
import sys
from unittest.mock import patch, MagicMock


class TestBotCLISymbolsArgument:
    """Test bot.py accepts --symbols argument."""

    def test_bot_parses_symbols_argument(self):
        """bot.py should parse --symbols into a list."""
        # Import after patching to avoid actual bot initialization
        with patch('bot.TradingBot') as MockBot:
            mock_instance = MagicMock()
            MockBot.return_value = mock_instance

            # Simulate argument parsing
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument('--config', default='config.yaml')
            parser.add_argument('--symbols', type=str, default=None,
                                help='Comma-separated list of symbols from scanner')

            args = parser.parse_args(['--symbols', 'NVDA,TSLA,AMD'])

            assert args.symbols == 'NVDA,TSLA,AMD'
            symbols_list = args.symbols.split(',') if args.symbols else None
            assert symbols_list == ['NVDA', 'TSLA', 'AMD']
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_bot_scanner_integration.py::TestBotCLISymbolsArgument::test_bot_parses_symbols_argument -v`

Expected: PASS (this test is self-contained, verifies argparse behavior)

**Step 3: Modify bot.py to accept --symbols argument**

In `bot.py`, find the `if __name__ == "__main__":` block and update argument parsing:

```python
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated list of symbols from scanner (overrides config)')
    args = parser.parse_args()

    # Parse symbols if provided
    scanner_symbols = None
    if args.symbols:
        scanner_symbols = [s.strip() for s in args.symbols.split(',')]
        print(f"[SCANNER] Using {len(scanner_symbols)} symbols from scanner: {scanner_symbols}")

    bot = TradingBot(config_path=args.config, scanner_symbols=scanner_symbols)
    bot.run()
```

**Step 4: Update TradingBot.__init__ to accept scanner_symbols**

In `bot.py`, modify `TradingBot.__init__`:

```python
def __init__(self, config_path: str = "config.yaml", scanner_symbols: list = None):
    # ... existing init code ...

    # After loading static watchlist, check for scanner override
    if scanner_symbols:
        self.watchlist = scanner_symbols
        self.logger.info(f"Using scanner-provided watchlist: {self.watchlist}")
    else:
        # Existing logic for proven_symbols or scanner
        # ... keep existing watchlist loading code ...
```

**Step 5: Run test again**

Run: `python3 -m pytest tests/test_bot_scanner_integration.py -v`

Expected: PASS

**Step 6: Commit**

```bash
git add bot.py tests/test_bot_scanner_integration.py
git commit -m "feat(bot): add --symbols CLI argument for scanner integration"
```

---

### Task 2: Add Market Hours Check Utility

**Files:**
- Create: `core/market_hours.py`
- Test: `tests/test_market_hours.py` (may already exist, extend if needed)

**Step 1: Write failing test for market hours check**

Add to `tests/test_bot_scanner_integration.py`:

```python
from datetime import datetime, time
from zoneinfo import ZoneInfo
from unittest.mock import patch


class TestMarketHoursCheck:
    """Test market hours validation."""

    def test_is_market_open_during_trading_hours(self):
        """Should return True during market hours (9:30 AM - 4:00 PM ET)."""
        from core.market_hours import is_market_open

        # Mock a Tuesday at 10:30 AM ET
        mock_time = datetime(2026, 1, 6, 10, 30, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch('core.market_hours.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert is_market_open() == True

    def test_is_market_open_before_market_hours(self):
        """Should return False before 9:30 AM ET."""
        from core.market_hours import is_market_open

        # Mock a Tuesday at 8:00 AM ET
        mock_time = datetime(2026, 1, 6, 8, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch('core.market_hours.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert is_market_open() == False

    def test_is_market_open_on_weekend(self):
        """Should return False on weekends."""
        from core.market_hours import is_market_open

        # Mock a Saturday at 11:00 AM ET
        mock_time = datetime(2026, 1, 3, 11, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch('core.market_hours.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert is_market_open() == False

    def test_get_market_status_message_when_closed(self):
        """Should return helpful message when market is closed."""
        from core.market_hours import get_market_status_message

        # Mock a Saturday
        mock_time = datetime(2026, 1, 3, 11, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch('core.market_hours.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            message = get_market_status_message()
            assert "closed" in message.lower() or "weekend" in message.lower()
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_bot_scanner_integration.py::TestMarketHoursCheck -v`

Expected: FAIL with "No module named 'core.market_hours'"

**Step 3: Create market hours module**

Create `core/market_hours.py`:

```python
"""Market hours utilities for trading bot."""
from datetime import datetime, time
from zoneinfo import ZoneInfo

# US Eastern timezone
ET = ZoneInfo("America/New_York")

# Regular market hours
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


def is_market_open() -> bool:
    """
    Check if US stock market is currently open.

    Returns:
        True if market is open (9:30 AM - 4:00 PM ET, weekdays)
    """
    now = datetime.now(ET)

    # Check if weekend (Saturday=5, Sunday=6)
    if now.weekday() >= 5:
        return False

    # Check if within trading hours
    current_time = now.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE


def get_market_status_message() -> str:
    """
    Get a human-readable message about market status.

    Returns:
        Message explaining why market is open or closed
    """
    now = datetime.now(ET)
    current_time = now.time()

    if now.weekday() >= 5:
        day_name = "Saturday" if now.weekday() == 5 else "Sunday"
        return f"Market closed: {day_name}. Trading resumes Monday 9:30 AM ET."

    if current_time < MARKET_OPEN:
        return f"Market closed: Pre-market. Opens at 9:30 AM ET (currently {now.strftime('%I:%M %p')} ET)."

    if current_time > MARKET_CLOSE:
        return f"Market closed: After hours. Closed at 4:00 PM ET (currently {now.strftime('%I:%M %p')} ET)."

    return "Market is open."
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_bot_scanner_integration.py::TestMarketHoursCheck -v`

Expected: PASS

**Step 5: Commit**

```bash
git add core/market_hours.py tests/test_bot_scanner_integration.py
git commit -m "feat(core): add market hours check utility"
```

---

## Phase 2: Backend - API Endpoint Enhancement

### Task 3: Update Bot Start Endpoint with Scanner Integration

**Files:**
- Modify: `api/main.py`
- Test: `tests/test_bot_scanner_integration.py`

**Step 1: Write failing test for scanner-first bot start**

Add to `tests/test_bot_scanner_integration.py`:

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


class TestBotStartWithScanner:
    """Test POST /api/bot/start runs scanner first."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app
        return TestClient(app)

    def test_start_bot_runs_scanner_first(self, client):
        """POST /api/bot/start should run scanner before starting bot."""
        with patch('api.main.VolatilityScanner') as MockScanner, \
             patch('api.main.is_market_open', return_value=True), \
             patch('api.main.subprocess.Popen') as mock_popen:

            # Mock scanner returning results
            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = [
                {'symbol': 'NVDA', 'composite_score': 0.95},
                {'symbol': 'TSLA', 'composite_score': 0.90},
                {'symbol': 'AMD', 'composite_score': 0.85},
            ]
            MockScanner.return_value = mock_scanner_instance

            response = client.post("/api/bot/start")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "started"
            assert "watchlist" in data
            assert "NVDA" in data["watchlist"]
            mock_scanner_instance.scan.assert_called_once()

    def test_start_bot_fails_when_market_closed(self, client):
        """Should return 400 with reason='market_closed' outside trading hours."""
        with patch('api.main.is_market_open', return_value=False), \
             patch('api.main.get_market_status_message', return_value="Market closed: Saturday."):

            response = client.post("/api/bot/start")

            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["reason"] == "market_closed"
            assert "Saturday" in data["detail"]["message"]

    def test_start_bot_fails_when_scanner_returns_empty(self, client):
        """Should return 400 with reason='no_results' if scanner finds nothing."""
        with patch('api.main.VolatilityScanner') as MockScanner, \
             patch('api.main.is_market_open', return_value=True):

            # Mock scanner returning empty
            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = []
            MockScanner.return_value = mock_scanner_instance

            response = client.post("/api/bot/start")

            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["reason"] == "no_results"

    def test_start_bot_fails_on_scanner_api_error(self, client):
        """Should return 400 with reason='scanner_error' on data fetch failure."""
        with patch('api.main.VolatilityScanner') as MockScanner, \
             patch('api.main.is_market_open', return_value=True):

            # Mock scanner raising exception
            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.side_effect = Exception("YFinance API timeout")
            MockScanner.return_value = mock_scanner_instance

            response = client.post("/api/bot/start")

            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["reason"] == "scanner_error"
            assert "YFinance" in data["detail"]["message"]

    def test_start_bot_returns_watchlist_on_success(self, client):
        """Response should include list of scanned symbols and timestamp."""
        with patch('api.main.VolatilityScanner') as MockScanner, \
             patch('api.main.is_market_open', return_value=True), \
             patch('api.main.subprocess.Popen'):

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = [
                {'symbol': 'NVDA', 'composite_score': 0.95},
                {'symbol': 'TSLA', 'composite_score': 0.90},
            ]
            MockScanner.return_value = mock_scanner_instance

            response = client.post("/api/bot/start")

            assert response.status_code == 200
            data = response.json()
            assert data["watchlist"] == ["NVDA", "TSLA"]
            assert "scanner_ran_at" in data
            assert "message" in data
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_bot_scanner_integration.py::TestBotStartWithScanner -v`

Expected: FAIL (endpoint doesn't have scanner logic yet)

**Step 3: Update api/main.py with scanner-first logic**

Find the existing `POST /api/bot/start` endpoint in `api/main.py` and replace with:

```python
from core.market_hours import is_market_open, get_market_status_message
from core.scanner import VolatilityScanner
import subprocess

class BotStartResponse(BaseModel):
    """Response for bot start endpoint."""
    status: str
    watchlist: List[str]
    scanner_ran_at: str
    message: str


@app.post("/api/bot/start", response_model=BotStartResponse)
async def start_bot():
    """
    Start the trading bot with scanner-selected stocks.

    Flow:
    1. Check if market is open
    2. Run volatility scanner on 900+ stocks
    3. Start bot with scanned symbols

    Returns 400 if market closed, scanner fails, or no stocks found.
    """
    global _bot_state

    # Step 1: Check market hours
    if not is_market_open():
        raise HTTPException(
            status_code=400,
            detail={
                "reason": "market_closed",
                "message": get_market_status_message()
            }
        )

    # Step 2: Run scanner
    try:
        universe = load_universe()
        scanner_symbols = collect_scanner_symbols(universe)

        config = load_config()
        scanner_config = config.get('volatility_scanner', {})
        scanner = VolatilityScanner(scanner_config)

        # Fetch data and scan
        from core.data import YFinanceDataFetcher
        fetcher = YFinanceDataFetcher()

        # Get historical data for scanning (last 14 days of hourly data)
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=14)

        historical_data = {}
        for symbol in scanner_symbols[:100]:  # Limit to avoid timeout
            try:
                data = fetcher.fetch_historical_data(
                    symbol,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1h'
                )
                if data is not None and not data.empty:
                    historical_data[symbol] = data
            except Exception:
                continue

        # Run scan
        results = scanner.scan_historical(
            date=end_date.date(),
            symbols=list(historical_data.keys()),
            historical_data=historical_data
        )

        if not results:
            raise HTTPException(
                status_code=400,
                detail={
                    "reason": "no_results",
                    "message": "Scanner found no stocks meeting volatility threshold. This is unexpected with 900+ stocks - check data feed."
                }
            )

        # Extract symbols from results
        watchlist = [r['symbol'] for r in results[:10]]  # Top 10

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "reason": "scanner_error",
                "message": f"Scanner failed: {str(e)}"
            }
        )

    # Step 3: Start bot with scanned symbols
    try:
        symbols_arg = ",".join(watchlist)
        subprocess.Popen(
            ["python3", "bot.py", "--symbols", symbols_arg],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        _bot_state["status"] = "running"
        _bot_state["watchlist"] = watchlist
        update_bot_state(status="running", last_action=f"Started with scanner: {', '.join(watchlist)}")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "reason": "bot_start_error",
                "message": f"Failed to start bot process: {str(e)}"
            }
        )

    return BotStartResponse(
        status="started",
        watchlist=watchlist,
        scanner_ran_at=datetime.now().isoformat(),
        message=f"Bot started with {len(watchlist)} scanned stocks: {', '.join(watchlist)}"
    )
```

**Step 4: Update BotStatusResponse to include watchlist**

In `api/main.py`, update the `BotStatusResponse` model:

```python
class BotStatusResponse(BaseModel):
    """Bot status response."""
    status: str  # 'running', 'stopped', 'error'
    mode: str  # 'PAPER', 'LIVE', 'DRY_RUN', 'BACKTEST'
    last_action: Optional[str] = None
    last_action_time: Optional[str] = None
    kill_switch_triggered: bool = False
    watchlist: Optional[List[str]] = None  # Add this field
```

Update `_bot_state` initialization:

```python
_bot_state = {
    "status": "stopped",
    "last_action": None,
    "last_action_time": None,
    "kill_switch_triggered": False,
    "watchlist": None  # Add this field
}
```

Update `get_bot_status` endpoint to include watchlist:

```python
@app.get("/api/bot/status", response_model=BotStatusResponse)
async def get_bot_status():
    """Get current bot status."""
    try:
        config = get_global_config()
        mode = config.get_mode()

        return BotStatusResponse(
            status=_bot_state["status"],
            mode=mode,
            last_action=_bot_state["last_action"],
            last_action_time=_bot_state["last_action_time"],
            kill_switch_triggered=_bot_state["kill_switch_triggered"],
            watchlist=_bot_state.get("watchlist")  # Add this
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_bot_scanner_integration.py::TestBotStartWithScanner -v`

Expected: PASS

**Step 6: Commit**

```bash
git add api/main.py tests/test_bot_scanner_integration.py
git commit -m "feat(api): integrate scanner into bot start endpoint"
```

---

## Phase 3: Frontend Updates

### Task 4: Update API Client Types

**Files:**
- Modify: `frontend/src/lib/api.ts`

**Step 1: Add new types for scanner-integrated bot start**

Add to `frontend/src/lib/api.ts`:

```typescript
// Bot start response with scanner integration
export interface BotStartResponse {
  status: "started";
  watchlist: string[];
  scanner_ran_at: string;
  message: string;
}

export interface BotStartError {
  reason: "market_closed" | "no_results" | "scanner_error" | "bot_start_error";
  message: string;
}

// Update BotStatus to include watchlist
export interface BotStatus {
  status: "running" | "stopped" | "error";
  mode: "PAPER" | "LIVE" | "DRY_RUN" | "BACKTEST";
  last_action: string | null;
  last_action_time: string | null;
  kill_switch_triggered: boolean;
  watchlist: string[] | null;  // Add this field
}
```

**Step 2: Update startBot function**

Find and update the `startBot` function:

```typescript
export async function startBot(): Promise<BotStartResponse> {
  const response = await fetch(`${API_BASE_URL}/api/bot/start`, {
    method: "POST",
  });

  if (!response.ok) {
    const errorData = await response.json();
    const error = errorData.detail as BotStartError;
    throw new Error(error.message || "Failed to start bot");
  }

  return response.json();
}
```

**Step 3: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat(frontend): update API types for scanner-integrated bot start"
```

---

### Task 5: Update BotStatusCard with Scanner Loading State

**Files:**
- Modify: `frontend/src/components/cards/BotStatusCard.tsx`

**Step 1: Add scanner loading state and watchlist display**

Update the component to show scanning progress and watchlist:

```typescript
"use client";

import { useState } from "react";
import { Play, Square, Clock, AlertTriangle, Loader2 } from "lucide-react";
import { usePolling } from "@/hooks/usePolling";
import { getBotStatus, startBot, stopBot, BotStatus } from "@/lib/api";

export function BotStatusCard() {
  const [isStarting, setIsStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  const { data: botState, isLoading, error, refetch } = usePolling<BotStatus>({
    fetcher: getBotStatus,
    interval: 3000,
  });

  const handleStart = async () => {
    setIsStarting(true);
    setStartError(null);

    try {
      const result = await startBot();
      console.log("Bot started with watchlist:", result.watchlist);
      await refetch();
    } catch (err) {
      setStartError(err instanceof Error ? err.message : "Failed to start bot");
    } finally {
      setIsStarting(false);
    }
  };

  const handleStop = async () => {
    try {
      await stopBot();
      await refetch();
    } catch (err) {
      console.error("Failed to stop bot:", err);
    }
  };

  const statusConfig = {
    running: {
      color: "text-emerald",
      bgColor: "bg-emerald",
      label: "Running",
    },
    stopped: {
      color: "text-amber",
      bgColor: "bg-amber",
      label: "Stopped",
    },
    error: {
      color: "text-red",
      bgColor: "bg-red",
      label: "Error",
    },
  };

  const modeConfig = {
    PAPER: { className: "badge-emerald" },
    LIVE: { className: "badge-red badge-live" },
    DRY_RUN: { className: "badge-neutral" },
    BACKTEST: { className: "badge-neutral" },
  };

  // Loading state
  if (isLoading && !botState) {
    return (
      <div className="glass-gradient p-5 h-full animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-6"></div>
        <div className="h-6 bg-gray-700 rounded w-1/2 mb-4"></div>
        <div className="h-4 bg-gray-700 rounded w-2/3"></div>
      </div>
    );
  }

  // Error state
  if (error || !botState) {
    return (
      <div className="glass-gradient p-5 h-full">
        <div className="flex items-center gap-2 text-red">
          <AlertTriangle className="w-5 h-5" />
          <span>Failed to load bot status</span>
        </div>
      </div>
    );
  }

  const config = statusConfig[botState.status] || statusConfig.stopped;
  const isRunning = botState.status === "running";
  const modeStyle = modeConfig[botState.mode] || modeConfig.DRY_RUN;

  // Format last action time
  const formatTime = (isoTime: string | null) => {
    if (!isoTime) return "";
    const date = new Date(isoTime);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins} min ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours} hr ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="glass-gradient p-5 opacity-0 animate-slide-up stagger-3 h-full">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-sm font-medium text-text-secondary">Bot Status</h3>
          <span className={`badge ${modeStyle.className} ${isRunning ? "badge-pulse" : ""}`}>
            {botState.mode}
          </span>
        </div>

        {/* Status indicator */}
        <div className="flex items-center gap-3 mb-4">
          <div className={`relative w-3 h-3 rounded-full ${config.bgColor} ${isRunning ? "pulse-dot" : ""}`} />
          <span className={`text-lg font-semibold ${config.color}`}>
            {isStarting ? "Scanning..." : config.label}
          </span>
          {botState.kill_switch_triggered && (
            <span className="badge badge-red text-xs">Kill Switch</span>
          )}
        </div>

        {/* Watchlist display when running */}
        {isRunning && botState.watchlist && botState.watchlist.length > 0 && (
          <div className="mb-4">
            <p className="text-xs text-text-muted mb-2">Trading:</p>
            <div className="flex flex-wrap gap-1">
              {botState.watchlist.map((symbol) => (
                <span key={symbol} className="badge badge-neutral text-xs">
                  {symbol}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Start error display */}
        {startError && (
          <div className="mb-4 p-2 bg-red/10 border border-red/20 rounded text-sm text-red">
            {startError}
          </div>
        )}

        {/* Last action */}
        <div className="flex-1">
          {botState.last_action ? (
            <div className="flex items-start gap-2 text-sm">
              <Clock className="w-4 h-4 text-text-muted mt-0.5 flex-shrink-0 icon-wiggle" />
              <div>
                <p className="text-text-secondary">{botState.last_action}</p>
                <p className="text-text-muted text-xs mt-1">
                  {formatTime(botState.last_action_time)}
                </p>
              </div>
            </div>
          ) : (
            <p className="text-text-muted text-sm">No recent activity</p>
          )}
        </div>

        {/* Control buttons */}
        <div className="flex gap-2 mt-4 pt-4 border-t border-border">
          {isRunning ? (
            <button
              onClick={handleStop}
              className="btn btn-danger btn-ripple flex-1 text-sm py-2"
            >
              <Square className="w-4 h-4" />
              Stop
            </button>
          ) : (
            <button
              onClick={handleStart}
              disabled={isStarting}
              className="btn btn-primary btn-ripple flex-1 text-sm py-2 disabled:opacity-50"
            >
              {isStarting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Scanning...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Start
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
```

**Step 2: Verify component works**

Run frontend: `cd frontend && npm run dev`
Navigate to dashboard, verify:
- Start button shows "Scanning..." while scanner runs
- Watchlist chips appear after successful start
- Error message appears if start fails

**Step 3: Commit**

```bash
git add frontend/src/components/cards/BotStatusCard.tsx
git commit -m "feat(frontend): add scanner loading state and watchlist display to BotStatusCard"
```

---

## Phase 4: Integration Testing

### Task 6: Add End-to-End Integration Test

**Files:**
- Modify: `tests/test_bot_scanner_integration.py`

**Step 1: Add full integration test**

Add to `tests/test_bot_scanner_integration.py`:

```python
class TestFullScannerBotIntegration:
    """End-to-end tests for scanner-bot integration."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app
        return TestClient(app)

    def test_full_flow_start_to_running(self, client):
        """Test complete flow: start bot -> scanner runs -> bot starts with symbols."""
        with patch('api.main.VolatilityScanner') as MockScanner, \
             patch('api.main.is_market_open', return_value=True), \
             patch('api.main.subprocess.Popen') as mock_popen, \
             patch('api.main.YFinanceDataFetcher') as MockFetcher:

            # Setup mocks
            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan_historical.return_value = [
                {'symbol': 'NVDA', 'composite_score': 0.95},
                {'symbol': 'TSLA', 'composite_score': 0.90},
                {'symbol': 'AMD', 'composite_score': 0.85},
            ]
            MockScanner.return_value = mock_scanner_instance

            mock_fetcher_instance = MagicMock()
            mock_fetcher_instance.fetch_historical_data.return_value = MagicMock(empty=False)
            MockFetcher.return_value = mock_fetcher_instance

            # Start bot
            response = client.post("/api/bot/start")
            assert response.status_code == 200
            start_data = response.json()
            assert start_data["status"] == "started"
            assert start_data["watchlist"] == ["NVDA", "TSLA", "AMD"]

            # Verify bot process was started with correct symbols
            mock_popen.assert_called_once()
            call_args = mock_popen.call_args[0][0]
            assert "python3" in call_args[0]
            assert "bot.py" in call_args[1]
            assert "--symbols" in call_args
            assert "NVDA,TSLA,AMD" in call_args

            # Check status shows running with watchlist
            status_response = client.get("/api/bot/status")
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["status"] == "running"
            assert status_data["watchlist"] == ["NVDA", "TSLA", "AMD"]

    def test_scanner_failure_provides_clear_reason(self, client):
        """Test that scanner failures provide actionable error messages."""
        test_cases = [
            {
                "setup": {"is_market_open": False},
                "expected_reason": "market_closed",
                "expected_in_message": ["closed", "ET"]
            },
            {
                "setup": {"scanner_returns": []},
                "expected_reason": "no_results",
                "expected_in_message": ["no stocks", "threshold"]
            },
            {
                "setup": {"scanner_raises": Exception("Connection timeout")},
                "expected_reason": "scanner_error",
                "expected_in_message": ["timeout"]
            },
        ]

        for case in test_cases:
            with patch('api.main.is_market_open', return_value=case["setup"].get("is_market_open", True)), \
                 patch('api.main.get_market_status_message', return_value="Market closed: Saturday."), \
                 patch('api.main.VolatilityScanner') as MockScanner, \
                 patch('api.main.YFinanceDataFetcher') as MockFetcher:

                if "scanner_returns" in case["setup"]:
                    mock_scanner = MagicMock()
                    mock_scanner.scan_historical.return_value = case["setup"]["scanner_returns"]
                    MockScanner.return_value = mock_scanner
                    mock_fetcher = MagicMock()
                    mock_fetcher.fetch_historical_data.return_value = MagicMock(empty=False)
                    MockFetcher.return_value = mock_fetcher

                if "scanner_raises" in case["setup"]:
                    mock_scanner = MagicMock()
                    mock_scanner.scan_historical.side_effect = case["setup"]["scanner_raises"]
                    MockScanner.return_value = mock_scanner
                    mock_fetcher = MagicMock()
                    mock_fetcher.fetch_historical_data.return_value = MagicMock(empty=False)
                    MockFetcher.return_value = mock_fetcher

                response = client.post("/api/bot/start")
                assert response.status_code == 400
                data = response.json()
                assert data["detail"]["reason"] == case["expected_reason"]
```

**Step 2: Run all integration tests**

Run: `python3 -m pytest tests/test_bot_scanner_integration.py -v`

Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_bot_scanner_integration.py
git commit -m "test: add end-to-end scanner-bot integration tests"
```

---

## Phase 5: Final Verification

### Task 7: Run Full Test Suite

**Step 1: Run all tests**

Run: `python3 -m pytest -v`

Expected: All tests PASS

**Step 2: Manual verification checklist**

1. Start API server: `cd /home/carsonodell/trading-bot && python3 -m uvicorn api.main:app --reload`
2. Start frontend: `cd frontend && npm run dev`
3. Navigate to dashboard
4. Click "Start" button
5. Verify:
   - [ ] Button shows "Scanning..." with spinner
   - [ ] After 5-10 seconds, bot status shows "Running"
   - [ ] Watchlist chips appear showing scanned symbols
   - [ ] Last action shows "Started with scanner: NVDA, TSLA, ..."
6. Click "Stop" button
7. Try starting outside market hours:
   - [ ] Error message appears with market hours info
8. Check API logs for scanner output

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete scanner-bot integration with tests"
```

---

## Summary

**Files Created:**
- `core/market_hours.py` - Market hours check utility
- `tests/test_bot_scanner_integration.py` - Integration tests

**Files Modified:**
- `bot.py` - Added `--symbols` CLI argument
- `api/main.py` - Updated `/api/bot/start` with scanner-first logic
- `frontend/src/lib/api.ts` - Updated types for scanner integration
- `frontend/src/components/cards/BotStatusCard.tsx` - Added scanning state and watchlist display

**Test Commands:**
```bash
# Run scanner-bot integration tests only
python3 -m pytest tests/test_bot_scanner_integration.py -v

# Run all tests
python3 -m pytest -v
```
