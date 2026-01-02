# Live Trading UI Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace all mock/dummy data in the frontend UI with real live trading data from the Python backend.

**Architecture:** Add REST API endpoints to FastAPI (`api/main.py`) that expose broker data (account, positions, orders) and bot state. Frontend pages fetch data via `lib/api.ts` using React hooks with polling for real-time updates.

**Tech Stack:** FastAPI (Python), Next.js/React (TypeScript), Pydantic models, fetch API with polling

---

## Phase 1: Core API Endpoints

### Task 1: Add Account Endpoint

**Files:**
- Modify: `api/main.py`
- Modify: `frontend/src/lib/api.ts`
- Test: Manual curl test

**Step 1: Add Pydantic models for Account**

In `api/main.py`, add after line 84 (after `BacktestResponse`):

```python
class AccountResponse(BaseModel):
    """Account information response."""
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    daily_pnl: float
    daily_pnl_percent: float


class PositionResponse(BaseModel):
    """Single position response."""
    symbol: str
    qty: float
    side: str
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float


class PositionsResponse(BaseModel):
    """List of positions response."""
    positions: List[PositionResponse]
    total_unrealized_pl: float


class BotStatusResponse(BaseModel):
    """Bot status response."""
    status: str  # 'running', 'stopped', 'error'
    mode: str  # 'PAPER', 'LIVE', 'DRY_RUN', 'BACKTEST'
    last_action: Optional[str] = None
    last_action_time: Optional[str] = None
    kill_switch_triggered: bool = False
```

**Step 2: Add broker import and singleton**

Add after line 21 (after `from core.cache import get_cache`):

```python
from core.broker import create_broker, BrokerInterface, BrokerAPIError
from core.config import get_global_config

# Broker singleton for API
_broker: Optional[BrokerInterface] = None

def get_broker() -> BrokerInterface:
    """Get or create broker singleton."""
    global _broker
    if _broker is None:
        _broker = create_broker()
    return _broker
```

**Step 3: Add /api/account endpoint**

Add after the `/api/cache/clear` endpoint:

```python
@app.get("/api/account", response_model=AccountResponse)
async def get_account():
    """Get current account information."""
    try:
        broker = get_broker()
        account = broker.get_account()
        return AccountResponse(
            equity=round(account.equity, 2),
            cash=round(account.cash, 2),
            buying_power=round(account.buying_power, 2),
            portfolio_value=round(account.portfolio_value, 2),
            daily_pnl=round(account.daily_pnl, 2),
            daily_pnl_percent=round(account.daily_pnl_percent * 100, 2)
        )
    except BrokerAPIError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 4: Verify endpoint works**

Run: `curl http://localhost:8000/api/account`
Expected: JSON with equity, cash, buying_power, etc.

**Step 5: Commit**

```bash
git add api/main.py
git commit -m "feat(api): add /api/account endpoint for live account data"
```

---

### Task 2: Add Positions Endpoint

**Files:**
- Modify: `api/main.py`

**Step 1: Add /api/positions endpoint**

Add after the `/api/account` endpoint:

```python
@app.get("/api/positions", response_model=PositionsResponse)
async def get_positions():
    """Get all open positions."""
    try:
        broker = get_broker()
        positions = broker.get_positions()

        position_list = [
            PositionResponse(
                symbol=pos.symbol,
                qty=pos.qty,
                side=pos.side,
                avg_entry_price=round(pos.avg_entry_price, 2),
                current_price=round(pos.current_price, 2),
                market_value=round(pos.market_value, 2),
                unrealized_pl=round(pos.unrealized_pl, 2),
                unrealized_plpc=round(pos.unrealized_plpc * 100, 2)
            )
            for pos in positions
        ]

        total_pl = sum(pos.unrealized_pl for pos in positions)

        return PositionsResponse(
            positions=position_list,
            total_unrealized_pl=round(total_pl, 2)
        )
    except BrokerAPIError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 2: Verify endpoint works**

Run: `curl http://localhost:8000/api/positions`
Expected: JSON with positions array and total_unrealized_pl

**Step 3: Commit**

```bash
git add api/main.py
git commit -m "feat(api): add /api/positions endpoint for open positions"
```

---

### Task 3: Add Bot Status Endpoint

**Files:**
- Modify: `api/main.py`

**Step 1: Add bot state tracking**

Add after the broker singleton code:

```python
# Bot state tracking (simple in-memory for now)
_bot_state = {
    "status": "stopped",
    "last_action": None,
    "last_action_time": None,
    "kill_switch_triggered": False
}

def update_bot_state(status: str = None, last_action: str = None):
    """Update bot state."""
    if status:
        _bot_state["status"] = status
    if last_action:
        _bot_state["last_action"] = last_action
        _bot_state["last_action_time"] = datetime.now().isoformat()
```

**Step 2: Add /api/bot/status endpoint**

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
            kill_switch_triggered=_bot_state["kill_switch_triggered"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 3: Verify endpoint works**

Run: `curl http://localhost:8000/api/bot/status`
Expected: JSON with status, mode, last_action fields

**Step 4: Commit**

```bash
git add api/main.py
git commit -m "feat(api): add /api/bot/status endpoint"
```

---

### Task 4: Add Orders Endpoint

**Files:**
- Modify: `api/main.py`

**Step 1: Add Pydantic models for Orders**

Add after `BotStatusResponse`:

```python
class OrderResponse(BaseModel):
    """Single order response."""
    id: str
    symbol: str
    qty: float
    side: str
    type: str
    status: str
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_qty: float = 0
    filled_avg_price: Optional[float] = None
    submitted_at: Optional[str] = None
    filled_at: Optional[str] = None


class OrdersResponse(BaseModel):
    """List of orders response."""
    orders: List[OrderResponse]
```

**Step 2: Add /api/orders endpoint**

```python
@app.get("/api/orders", response_model=OrdersResponse)
async def get_orders(status: str = "open"):
    """Get orders with optional status filter."""
    try:
        broker = get_broker()
        orders = broker.list_orders(status=status)

        order_list = [
            OrderResponse(
                id=order.id,
                symbol=order.symbol,
                qty=order.qty,
                side=order.side,
                type=order.type,
                status=order.status,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                filled_qty=order.filled_qty,
                filled_avg_price=order.filled_avg_price,
                submitted_at=order.submitted_at.isoformat() if order.submitted_at else None,
                filled_at=order.filled_at.isoformat() if order.filled_at else None
            )
            for order in orders
        ]

        return OrdersResponse(orders=order_list)
    except BrokerAPIError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 3: Verify endpoint works**

Run: `curl http://localhost:8000/api/orders`
Expected: JSON with orders array

**Step 4: Commit**

```bash
git add api/main.py
git commit -m "feat(api): add /api/orders endpoint"
```

---

## Phase 2: Frontend API Client

### Task 5: Add TypeScript Types for Live Data

**Files:**
- Modify: `frontend/src/lib/api.ts`

**Step 1: Add types for account, positions, orders, bot status**

Add after the existing backtest types:

```typescript
// Live Trading Types
export interface AccountData {
  equity: number;
  cash: number;
  buying_power: number;
  portfolio_value: number;
  daily_pnl: number;
  daily_pnl_percent: number;
}

export interface Position {
  symbol: string;
  qty: number;
  side: string;
  avg_entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pl: number;
  unrealized_plpc: number;
}

export interface PositionsData {
  positions: Position[];
  total_unrealized_pl: number;
}

export interface BotStatus {
  status: "running" | "stopped" | "error";
  mode: "PAPER" | "LIVE" | "DRY_RUN" | "BACKTEST";
  last_action: string | null;
  last_action_time: string | null;
  kill_switch_triggered: boolean;
}

export interface Order {
  id: string;
  symbol: string;
  qty: number;
  side: string;
  type: string;
  status: string;
  limit_price: number | null;
  stop_price: number | null;
  filled_qty: number;
  filled_avg_price: number | null;
  submitted_at: string | null;
  filled_at: string | null;
}

export interface OrdersData {
  orders: Order[];
}
```

**Step 2: Add fetch functions**

```typescript
export async function getAccount(): Promise<AccountData> {
  const response = await fetch(`${API_BASE_URL}/api/account`);
  if (!response.ok) {
    throw new Error(`Failed to fetch account: ${response.status}`);
  }
  return response.json();
}

export async function getPositions(): Promise<PositionsData> {
  const response = await fetch(`${API_BASE_URL}/api/positions`);
  if (!response.ok) {
    throw new Error(`Failed to fetch positions: ${response.status}`);
  }
  return response.json();
}

export async function getBotStatus(): Promise<BotStatus> {
  const response = await fetch(`${API_BASE_URL}/api/bot/status`);
  if (!response.ok) {
    throw new Error(`Failed to fetch bot status: ${response.status}`);
  }
  return response.json();
}

export async function getOrders(status: string = "open"): Promise<OrdersData> {
  const response = await fetch(`${API_BASE_URL}/api/orders?status=${status}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch orders: ${response.status}`);
  }
  return response.json();
}
```

**Step 3: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat(frontend): add API client functions for live trading data"
```

---

### Task 6: Create usePolling Hook

**Files:**
- Create: `frontend/src/hooks/usePolling.ts`

**Step 1: Create the hook file**

```typescript
"use client";

import { useState, useEffect, useCallback, useRef } from "react";

interface UsePollingOptions<T> {
  fetcher: () => Promise<T>;
  interval?: number;
  enabled?: boolean;
  onError?: (error: Error) => void;
}

interface UsePollingResult<T> {
  data: T | null;
  error: Error | null;
  isLoading: boolean;
  refetch: () => Promise<void>;
}

export function usePolling<T>({
  fetcher,
  interval = 5000,
  enabled = true,
  onError,
}: UsePollingOptions<T>): UsePollingResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const result = await fetcher();
      setData(result);
      setError(null);
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error);
      onError?.(error);
    } finally {
      setIsLoading(false);
    }
  }, [fetcher, onError]);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    // Initial fetch
    fetchData();

    // Set up polling
    intervalRef.current = setInterval(fetchData, interval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [enabled, interval, fetchData]);

  return { data, error, isLoading, refetch: fetchData };
}
```

**Step 2: Commit**

```bash
git add frontend/src/hooks/usePolling.ts
git commit -m "feat(frontend): add usePolling hook for real-time data fetching"
```

---

## Phase 3: Dashboard Integration

### Task 7: Update BotStatusCard with Real Data

**Files:**
- Modify: `frontend/src/components/cards/BotStatusCard.tsx`

**Step 1: Replace mock data with API call**

Replace the entire file content:

```typescript
"use client";

import { Play, Square, Clock, AlertTriangle } from "lucide-react";
import { usePolling } from "@/hooks/usePolling";
import { getBotStatus, BotStatus } from "@/lib/api";

export function BotStatusCard() {
  const { data: botState, isLoading, error } = usePolling<BotStatus>({
    fetcher: getBotStatus,
    interval: 3000,
  });

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
        <div className="flex items-center gap-3 mb-6">
          <div className={`relative w-3 h-3 rounded-full ${config.bgColor} ${isRunning ? "pulse-dot" : ""}`} />
          <span className={`text-lg font-semibold ${config.color}`}>
            {config.label}
          </span>
          {botState.kill_switch_triggered && (
            <span className="badge badge-red text-xs">Kill Switch</span>
          )}
        </div>

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
            <button className="btn btn-danger btn-ripple flex-1 text-sm py-2">
              <Square className="w-4 h-4" />
              Stop
            </button>
          ) : (
            <button className="btn btn-primary btn-ripple flex-1 text-sm py-2">
              <Play className="w-4 h-4" />
              Start
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
```

**Step 2: Verify component renders with API data**

Run frontend and check that BotStatusCard shows data from API (or loading/error states).

**Step 3: Commit**

```bash
git add frontend/src/components/cards/BotStatusCard.tsx
git commit -m "feat(frontend): connect BotStatusCard to live API data"
```

---

### Task 8: Update Dashboard Page with Real Data

**Files:**
- Modify: `frontend/src/app/page.tsx`

**Step 1: Replace mock data with API calls**

Replace the entire file content:

```typescript
"use client";

import { PageWrapper } from "@/components/layout/PageWrapper";
import { MetricCard } from "@/components/cards/MetricCard";
import { BotStatusCard } from "@/components/cards/BotStatusCard";
import { EquityCurveChart } from "@/components/charts/EquityCurveChart";
import { MiniPositionsTable } from "@/components/cards/MiniPositionsTable";
import { DollarSign, TrendingUp, Briefcase, Target, AlertTriangle } from "lucide-react";
import { usePolling } from "@/hooks/usePolling";
import { getAccount, getPositions, AccountData, PositionsData } from "@/lib/api";

export default function DashboardPage() {
  const { data: account, isLoading: accountLoading, error: accountError } = usePolling<AccountData>({
    fetcher: getAccount,
    interval: 5000,
  });

  const { data: positionsData, isLoading: positionsLoading } = usePolling<PositionsData>({
    fetcher: getPositions,
    interval: 5000,
  });

  // Transform positions for MiniPositionsTable
  const positions = positionsData?.positions.map(pos => ({
    symbol: pos.symbol,
    side: pos.side.toUpperCase() as "LONG" | "SHORT",
    pnlPercent: pos.unrealized_plpc,
  })) || [];

  // Mock equity data for now (will add equity curve endpoint later)
  const equityData = account ? [
    { date: "Now", value: account.portfolio_value },
  ] : [];

  // Loading state
  if (accountLoading && !account) {
    return (
      <PageWrapper>
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald"></div>
        </div>
      </PageWrapper>
    );
  }

  // Error state
  if (accountError) {
    return (
      <PageWrapper>
        <div className="flex flex-col items-center justify-center h-64 text-red">
          <AlertTriangle className="w-12 h-12 mb-4" />
          <p className="text-lg">Failed to load account data</p>
          <p className="text-sm text-text-muted mt-2">{accountError.message}</p>
        </div>
      </PageWrapper>
    );
  }

  return (
    <PageWrapper>
      <div className="space-y-6 pt-6">
        {/* Top Row - Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Account Balance"
            value=""
            numericValue={account?.portfolio_value || 0}
            prefix="$"
            decimals={2}
            change={account?.daily_pnl_percent || 0}
            icon={<DollarSign className="w-5 h-5" />}
            delay={1}
          />
          <MetricCard
            title="Today's P&L"
            value=""
            numericValue={account?.daily_pnl || 0}
            prefix="$"
            decimals={2}
            change={account?.daily_pnl_percent || 0}
            icon={<TrendingUp className="w-5 h-5" />}
            delay={2}
          />
          <MetricCard
            title="Open Positions"
            value=""
            numericValue={positions.length}
            subtitle={positions.map(p => p.symbol).join(", ") || "None"}
            icon={<Briefcase className="w-5 h-5" />}
            delay={3}
          />
          <MetricCard
            title="Buying Power"
            value=""
            numericValue={account?.buying_power || 0}
            prefix="$"
            decimals={2}
            icon={<Target className="w-5 h-5" />}
            delay={4}
          />
        </div>

        {/* Middle Row - Bot Status + Equity Curve */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <BotStatusCard />
          <div className="lg:col-span-2">
            <EquityCurveChart data={equityData} />
          </div>
        </div>

        {/* Bottom Row - Mini Positions Table */}
        <MiniPositionsTable positions={positions} />
      </div>
    </PageWrapper>
  );
}
```

**Step 2: Verify dashboard loads with real data**

Run frontend and verify all metrics show real account/position data from API.

**Step 3: Commit**

```bash
git add frontend/src/app/page.tsx
git commit -m "feat(frontend): connect dashboard to live account and positions API"
```

---

## Phase 4: Positions Page Integration

### Task 9: Update Positions Page with Real Data

**Files:**
- Modify: `frontend/src/app/positions/page.tsx`

**Step 1: Replace mock data with API call**

Replace the file with real API integration:

```typescript
"use client";

import { PageWrapper } from "@/components/layout/PageWrapper";
import { usePolling } from "@/hooks/usePolling";
import { getPositions, PositionsData } from "@/lib/api";
import { TrendingUp, TrendingDown, AlertTriangle } from "lucide-react";

export default function PositionsPage() {
  const { data: positionsData, isLoading, error } = usePolling<PositionsData>({
    fetcher: getPositions,
    interval: 3000,
  });

  if (isLoading && !positionsData) {
    return (
      <PageWrapper>
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald"></div>
        </div>
      </PageWrapper>
    );
  }

  if (error) {
    return (
      <PageWrapper>
        <div className="flex flex-col items-center justify-center h-64 text-red">
          <AlertTriangle className="w-12 h-12 mb-4" />
          <p>Failed to load positions</p>
        </div>
      </PageWrapper>
    );
  }

  const positions = positionsData?.positions || [];

  return (
    <PageWrapper>
      <div className="space-y-6 pt-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold text-text-primary">Open Positions</h1>
          <div className="text-sm text-text-muted">
            Total Unrealized P&L:{" "}
            <span className={positionsData?.total_unrealized_pl && positionsData.total_unrealized_pl >= 0 ? "text-emerald" : "text-red"}>
              ${positionsData?.total_unrealized_pl?.toFixed(2) || "0.00"}
            </span>
          </div>
        </div>

        {positions.length === 0 ? (
          <div className="glass-gradient p-8 text-center text-text-muted">
            No open positions
          </div>
        ) : (
          <div className="glass-gradient overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-4 text-sm font-medium text-text-secondary">Symbol</th>
                  <th className="text-left p-4 text-sm font-medium text-text-secondary">Side</th>
                  <th className="text-right p-4 text-sm font-medium text-text-secondary">Qty</th>
                  <th className="text-right p-4 text-sm font-medium text-text-secondary">Entry</th>
                  <th className="text-right p-4 text-sm font-medium text-text-secondary">Current</th>
                  <th className="text-right p-4 text-sm font-medium text-text-secondary">Value</th>
                  <th className="text-right p-4 text-sm font-medium text-text-secondary">P&L</th>
                  <th className="text-right p-4 text-sm font-medium text-text-secondary">P&L %</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((position) => {
                  const isProfit = position.unrealized_pl >= 0;
                  return (
                    <tr key={position.symbol} className="border-b border-border/50 hover:bg-white/5">
                      <td className="p-4 font-medium text-text-primary">{position.symbol}</td>
                      <td className="p-4">
                        <span className={`badge ${position.side === "long" ? "badge-emerald" : "badge-red"}`}>
                          {position.side.toUpperCase()}
                        </span>
                      </td>
                      <td className="p-4 text-right text-text-secondary">{position.qty}</td>
                      <td className="p-4 text-right text-text-secondary">${position.avg_entry_price.toFixed(2)}</td>
                      <td className="p-4 text-right text-text-secondary">${position.current_price.toFixed(2)}</td>
                      <td className="p-4 text-right text-text-secondary">${position.market_value.toFixed(2)}</td>
                      <td className={`p-4 text-right font-medium ${isProfit ? "text-emerald" : "text-red"}`}>
                        <div className="flex items-center justify-end gap-1">
                          {isProfit ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                          ${Math.abs(position.unrealized_pl).toFixed(2)}
                        </div>
                      </td>
                      <td className={`p-4 text-right font-medium ${isProfit ? "text-emerald" : "text-red"}`}>
                        {isProfit ? "+" : ""}{position.unrealized_plpc.toFixed(2)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </PageWrapper>
  );
}
```

**Step 2: Verify positions page works**

Navigate to /positions and verify table shows real data.

**Step 3: Commit**

```bash
git add frontend/src/app/positions/page.tsx
git commit -m "feat(frontend): connect positions page to live API"
```

---

## Phase 5: Additional API Endpoints

### Task 10: Add Config/Settings Endpoint

**Files:**
- Modify: `api/main.py`

**Step 1: Add settings response model**

```python
class SettingsResponse(BaseModel):
    """Bot settings/configuration response."""
    mode: str
    risk_per_trade: float
    max_positions: int
    stop_loss_pct: float
    take_profit_pct: float
    strategies_enabled: List[str]
```

**Step 2: Add /api/settings endpoint**

```python
@app.get("/api/settings", response_model=SettingsResponse)
async def get_settings():
    """Get current bot settings."""
    try:
        config = load_config()

        return SettingsResponse(
            mode=config.get("mode", "DRY_RUN"),
            risk_per_trade=config.get("risk", {}).get("risk_per_trade", 0.02),
            max_positions=config.get("risk", {}).get("max_open_positions", 5),
            stop_loss_pct=config.get("exit_rules", {}).get("hard_stop_loss", 0.02),
            take_profit_pct=config.get("exit_rules", {}).get("partial_take_profit", {}).get("threshold", 0.02),
            strategies_enabled=[s["name"] for s in config.get("strategies", []) if s.get("enabled", False)]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 3: Commit**

```bash
git add api/main.py
git commit -m "feat(api): add /api/settings endpoint for configuration"
```

---

### Task 11: Add Scanner Results Endpoint

**Files:**
- Modify: `api/main.py`

**Step 1: Add scanner response models**

```python
class ScannerResult(BaseModel):
    """Single scanner result."""
    symbol: str
    atr_ratio: float
    volume_ratio: float
    composite_score: float
    current_price: float


class ScannerResponse(BaseModel):
    """Scanner results response."""
    results: List[ScannerResult]
    scanned_at: str
```

**Step 2: Add /api/scanner/scan endpoint**

```python
from core.scanner import VolatilityScanner

@app.get("/api/scanner/scan", response_model=ScannerResponse)
async def run_scanner(top_n: int = 10):
    """Run volatility scanner and return top N symbols."""
    try:
        config = load_config()
        universe = load_universe()
        symbols = collect_scanner_symbols(universe)

        scanner = VolatilityScanner(config.get("volatility_scanner", {}))
        results = scanner.scan(symbols, top_n=top_n)

        scanner_results = [
            ScannerResult(
                symbol=r["symbol"],
                atr_ratio=round(r.get("atr_ratio", 0), 4),
                volume_ratio=round(r.get("volume_ratio", 0), 2),
                composite_score=round(r.get("composite_score", 0), 4),
                current_price=round(r.get("current_price", 0), 2)
            )
            for r in results
        ]

        return ScannerResponse(
            results=scanner_results,
            scanned_at=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 3: Commit**

```bash
git add api/main.py
git commit -m "feat(api): add /api/scanner/scan endpoint"
```

---

## Phase 6: Remaining Frontend Pages

### Task 12: Update Settings Page

**Files:**
- Modify: `frontend/src/app/settings/page.tsx`
- Modify: `frontend/src/lib/api.ts`

**Step 1: Add settings types and fetch function to api.ts**

```typescript
export interface Settings {
  mode: string;
  risk_per_trade: number;
  max_positions: number;
  stop_loss_pct: number;
  take_profit_pct: number;
  strategies_enabled: string[];
}

export async function getSettings(): Promise<Settings> {
  const response = await fetch(`${API_BASE_URL}/api/settings`);
  if (!response.ok) {
    throw new Error(`Failed to fetch settings: ${response.status}`);
  }
  return response.json();
}
```

**Step 2: Update settings page with API call**

(Implementation similar to other pages - fetch settings and display in form)

**Step 3: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/app/settings/page.tsx
git commit -m "feat(frontend): connect settings page to live API"
```

---

### Task 13: Update Scanner Page

**Files:**
- Modify: `frontend/src/app/scanner/page.tsx`
- Modify: `frontend/src/lib/api.ts`

**Step 1: Add scanner types and fetch to api.ts**

```typescript
export interface ScannerResult {
  symbol: string;
  atr_ratio: number;
  volume_ratio: number;
  composite_score: number;
  current_price: number;
}

export interface ScannerData {
  results: ScannerResult[];
  scanned_at: string;
}

export async function runScanner(topN: number = 10): Promise<ScannerData> {
  const response = await fetch(`${API_BASE_URL}/api/scanner/scan?top_n=${topN}`);
  if (!response.ok) {
    throw new Error(`Failed to run scanner: ${response.status}`);
  }
  return response.json();
}
```

**Step 2: Update scanner page with API integration**

(Implementation fetches scanner results and displays in table)

**Step 3: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/app/scanner/page.tsx
git commit -m "feat(frontend): connect scanner page to live API"
```

---

## Summary

**API Endpoints to Add:**
1. `GET /api/account` - Account balance, equity, P&L
2. `GET /api/positions` - Open positions with P&L
3. `GET /api/bot/status` - Bot running state and mode
4. `GET /api/orders` - Open/filled orders
5. `GET /api/settings` - Current configuration
6. `GET /api/scanner/scan` - Run volatility scanner

**Frontend Pages to Update:**
1. Dashboard (`page.tsx`) - Account metrics, positions summary
2. BotStatusCard - Real bot status
3. Positions page - Live positions table
4. Settings page - Current config display
5. Scanner page - Live scanner results

**Key Patterns:**
- Use `usePolling` hook for real-time updates (3-5 second intervals)
- Handle loading/error states gracefully
- Keep existing UI styling, just replace data source
