# Backtest API Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect the frontend backtest page to the Python backtest engine via a FastAPI backend, using the volatility scanner for stock selection.

**Architecture:** Create a FastAPI server with `/api/backtest` endpoint. Frontend selects "top N" stocks and "days to run" - scanner picks the symbols. Warmup period data is fetched but excluded from P&L calculations.

**Tech Stack:** FastAPI (Python backend), Next.js fetch API (frontend), CORS middleware

---

## Task 1: Create FastAPI Backend Server

**Files:**
- Create: `api/main.py`
- Create: `api/__init__.py`

**Step 1: Create the API directory and init file**

```bash
mkdir -p api
touch api/__init__.py
```

**Step 2: Create main.py with FastAPI app and backtest endpoint**

Create `api/main.py`:
```python
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta

from backtest import Backtest1Hour
from core.config import load_config
import yaml

app = FastAPI(title="Trading Bot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BacktestRequest(BaseModel):
    top_n: int = 10  # Number of stocks from scanner
    days: int = 30  # Days to run backtest
    longs_only: bool = False
    shorts_only: bool = False
    initial_capital: float = 100000.0


class TradeResult(BaseModel):
    date: str
    symbol: str
    side: str
    pnl: float
    pnl_percent: float
    entry_price: float
    exit_price: float
    shares: int
    exit_reason: str
    strategy: str


class BacktestMetrics(BaseModel):
    total_pnl: float
    total_return_pct: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_pnl: float
    sharpe_ratio: Optional[float]


class EquityPoint(BaseModel):
    date: str
    value: float


class BacktestResponse(BaseModel):
    success: bool
    metrics: Optional[BacktestMetrics] = None
    equity_curve: list[EquityPoint] = []
    trades: list[TradeResult] = []
    symbols_scanned: list[str] = []
    error: Optional[str] = None


def format_backtest_results(results: dict) -> BacktestResponse:
    """Transform backtest.py output to API response format."""
    if not results or "error" in results:
        return BacktestResponse(
            success=False,
            error=results.get("error", "Unknown error") if results else "No results",
        )

    metrics = results.get("metrics", {})

    # Format equity curve
    equity_curve = []
    if "equity_curve" in results and results["equity_curve"] is not None:
        df = results["equity_curve"]
        for _, row in df.iterrows():
            ts = row.get("timestamp")
            if ts is not None:
                date_str = ts.strftime("%b %d") if hasattr(ts, "strftime") else str(ts)[:10]
                equity_curve.append(EquityPoint(
                    date=date_str,
                    value=round(row.get("portfolio_value", 0), 2),
                ))

    # Sample equity curve to max 50 points for chart performance
    if len(equity_curve) > 50:
        step = len(equity_curve) // 50
        equity_curve = equity_curve[::step]

    # Format trades
    trades = []
    for trade in results.get("trades", []):
        exit_date = trade.get("exit_date")
        date_str = exit_date.strftime("%Y-%m-%d") if hasattr(exit_date, "strftime") else str(exit_date)[:10]
        trades.append(TradeResult(
            date=date_str,
            symbol=trade.get("symbol", ""),
            side=trade.get("direction", "LONG"),
            pnl=round(trade.get("pnl", 0), 2),
            pnl_percent=round(trade.get("pnl_pct", 0), 2),
            entry_price=round(trade.get("entry_price", 0), 2),
            exit_price=round(trade.get("exit_price", 0), 2),
            shares=trade.get("shares", 0),
            exit_reason=trade.get("exit_reason", ""),
            strategy=trade.get("strategy", ""),
        ))

    # Sort trades by date descending
    trades.sort(key=lambda t: t.date, reverse=True)

    return BacktestResponse(
        success=True,
        symbols_scanned=results.get("symbols", []),
        metrics=BacktestMetrics(
            total_pnl=round(metrics.get("total_pnl", 0), 2),
            total_return_pct=round(metrics.get("total_return_pct", 0), 2),
            win_rate=round(metrics.get("win_rate", 0), 1),
            profit_factor=round(metrics.get("profit_factor", 0), 2),
            max_drawdown=round(metrics.get("max_drawdown", 0), 2),
            total_trades=metrics.get("total_trades", 0),
            winning_trades=metrics.get("winning_trades", 0),
            losing_trades=metrics.get("losing_trades", 0),
            avg_pnl=round(metrics.get("avg_pnl", 0), 2),
            sharpe_ratio=round(metrics.get("sharpe_ratio", 0), 2) if metrics.get("sharpe_ratio") else None,
        ),
        equity_curve=equity_curve,
        trades=trades,
    )


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/backtest", response_model=BacktestResponse)
def run_backtest_endpoint(request: BacktestRequest):
    """Run a backtest with scanner-selected symbols."""
    try:
        bot_dir = Path(__file__).parent.parent

        # Load config
        config_path = bot_dir / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Override scanner top_n
        if "volatility_scanner" not in config:
            config["volatility_scanner"] = {}
        config["volatility_scanner"]["enabled"] = True
        config["volatility_scanner"]["top_n"] = request.top_n

        # Calculate dates (warmup is handled internally by backtest)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=request.days)).strftime("%Y-%m-%d")

        # Load universe symbols for scanner
        universe_path = bot_dir / "universe.yaml"
        with open(universe_path, "r") as f:
            universe = yaml.safe_load(f)

        # Get all symbols from scanner_universe
        all_symbols = []
        scanner_universe = universe.get("scanner_universe", {})
        for category_symbols in scanner_universe.values():
            if isinstance(category_symbols, list):
                all_symbols.extend(category_symbols)
        all_symbols = list(set(all_symbols))  # Dedupe

        # Create backtest with scanner enabled
        backtest = Backtest1Hour(
            initial_capital=request.initial_capital,
            config=config,
            longs_only=request.longs_only,
            shorts_only=request.shorts_only,
            scanner_enabled=True,
        )

        # Run backtest - scanner will pick top_n symbols
        results = backtest.run(
            symbols=all_symbols,
            start_date=start_date,
            end_date=end_date,
        )

        return format_backtest_results(results)

    except Exception as e:
        import traceback
        return BacktestResponse(
            success=False,
            error=f"{str(e)}\n{traceback.format_exc()}",
        )
```

**Step 3: Test the server starts**

Run: `cd /home/carsonodell/trading-bot && python3 -m uvicorn api.main:app --reload --port 8000`

Expected: Server starts on http://localhost:8000

**Step 4: Commit**

```bash
git add api/
git commit -m "feat: add FastAPI backend with scanner-based backtest endpoint"
```

---

## Task 2: Create Frontend API Client

**Files:**
- Create: `frontend/src/lib/api.ts`

**Step 1: Create API client with types**

Create `frontend/src/lib/api.ts`:
```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface BacktestRequest {
  top_n: number;
  days: number;
  longs_only?: boolean;
  shorts_only?: boolean;
  initial_capital?: number;
}

export interface TradeResult {
  date: string;
  symbol: string;
  side: string;
  pnl: number;
  pnl_percent: number;
  entry_price: number;
  exit_price: number;
  shares: number;
  exit_reason: string;
  strategy: string;
}

export interface BacktestMetrics {
  total_pnl: number;
  total_return_pct: number;
  win_rate: number;
  profit_factor: number;
  max_drawdown: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  avg_pnl: number;
  sharpe_ratio: number | null;
}

export interface EquityPoint {
  date: string;
  value: number;
}

export interface BacktestResponse {
  success: boolean;
  metrics: BacktestMetrics | null;
  equity_curve: EquityPoint[];
  trades: TradeResult[];
  symbols_scanned: string[];
  error?: string;
}

export async function runBacktest(request: BacktestRequest): Promise<BacktestResponse> {
  const response = await fetch(`${API_BASE_URL}/api/backtest`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}
```

**Step 2: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat: add frontend API client for backtest"
```

---

## Task 3: Update Backtest Page UI

**Files:**
- Modify: `frontend/src/app/backtest/page.tsx`

**Step 1: Replace with scanner-based UI**

Replace entire `frontend/src/app/backtest/page.tsx`:
```tsx
"use client";

import { useState } from "react";
import { PageWrapper } from "@/components/layout/PageWrapper";
import { TestTube, Play, Calendar, Loader2, AlertCircle, ScanSearch } from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useMounted } from "@/lib/utils";
import { runBacktest, BacktestResponse } from "@/lib/api";

const TOP_N_OPTIONS = [5, 10, 15, 20, 25];
const DAYS_OPTIONS = [7, 14, 30, 60, 90];

export default function BacktestPage() {
  const [topN, setTopN] = useState(10);
  const [days, setDays] = useState(30);
  const [sideFilter, setSideFilter] = useState<"both" | "longs" | "shorts">("both");

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<BacktestResponse | null>(null);

  const mounted = useMounted();

  const handleRunBacktest = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await runBacktest({
        top_n: topN,
        days: days,
        longs_only: sideFilter === "longs",
        shorts_only: sideFilter === "shorts",
      });

      if (!response.success) {
        setError(response.error || "Backtest failed");
        setResults(null);
      } else {
        setResults(response);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run backtest");
      setResults(null);
    } finally {
      setIsLoading(false);
    }
  };

  const metrics = results?.metrics;
  const equityCurve = results?.equity_curve || [];
  const trades = results?.trades || [];
  const symbolsScanned = results?.symbols_scanned || [];

  return (
    <PageWrapper title="Backtesting" subtitle="Test strategies on historical data">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Config Panel */}
        <div className="glass p-5 opacity-0 animate-slide-up stagger-1 h-fit">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-lg bg-emerald-glow">
              <ScanSearch className="w-5 h-5 text-emerald" />
            </div>
            <h3 className="font-semibold text-white">Scanner Settings</h3>
          </div>

          {/* Top N Selector */}
          <div className="mb-6">
            <label className="block text-sm text-text-secondary mb-2">
              Top Stocks (by volatility)
            </label>
            <div className="flex gap-1 p-0.5 bg-surface-1 rounded-lg">
              {TOP_N_OPTIONS.map((n) => (
                <button
                  key={n}
                  onClick={() => setTopN(n)}
                  disabled={isLoading}
                  className={`flex-1 px-3 py-2 text-sm font-medium rounded-md transition-all ${
                    topN === n
                      ? "bg-emerald text-black"
                      : "text-text-muted hover:text-white"
                  } ${isLoading ? "opacity-50 cursor-not-allowed" : ""}`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          {/* Days Selector */}
          <div className="mb-6">
            <label className="block text-sm text-text-secondary mb-2">
              Backtest Period (days)
            </label>
            <div className="flex gap-1 p-0.5 bg-surface-1 rounded-lg">
              {DAYS_OPTIONS.map((d) => (
                <button
                  key={d}
                  onClick={() => setDays(d)}
                  disabled={isLoading}
                  className={`flex-1 px-3 py-2 text-sm font-medium rounded-md transition-all ${
                    days === d
                      ? "bg-emerald text-black"
                      : "text-text-muted hover:text-white"
                  } ${isLoading ? "opacity-50 cursor-not-allowed" : ""}`}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>

          {/* Side Filter */}
          <div className="mb-6">
            <label className="block text-sm text-text-secondary mb-2">
              Position Type
            </label>
            <div className="flex gap-1 p-0.5 bg-surface-1 rounded-lg">
              {(["both", "longs", "shorts"] as const).map((side) => (
                <button
                  key={side}
                  onClick={() => setSideFilter(side)}
                  disabled={isLoading}
                  className={`flex-1 px-3 py-2 text-sm font-medium rounded-md capitalize transition-all ${
                    sideFilter === side
                      ? "bg-emerald text-black"
                      : "text-text-muted hover:text-white"
                  } ${isLoading ? "opacity-50 cursor-not-allowed" : ""}`}
                >
                  {side === "both" ? "Both" : side === "longs" ? "Long" : "Short"}
                </button>
              ))}
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-4 p-3 bg-red/10 border border-red/20 rounded-lg flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-red mt-0.5 flex-shrink-0" />
              <p className="text-sm text-red break-all">{error}</p>
            </div>
          )}

          <button
            onClick={handleRunBacktest}
            disabled={isLoading}
            className="btn btn-primary w-full"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Running Scanner...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run Backtest
              </>
            )}
          </button>

          {/* Scanned Symbols */}
          {symbolsScanned.length > 0 && (
            <div className="mt-4 pt-4 border-t border-border">
              <p className="text-xs text-text-muted mb-2">Scanner Selected:</p>
              <div className="flex flex-wrap gap-1">
                {symbolsScanned.map((sym) => (
                  <span key={sym} className="px-2 py-0.5 text-xs bg-surface-2 rounded text-text-secondary">
                    {sym}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-6">
          {results && metrics ? (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 opacity-0 animate-slide-up stagger-2">
                <div className="glass p-4">
                  <p className="text-xs text-text-muted uppercase mb-1">Total P&L</p>
                  <p className={`text-xl font-bold mono ${metrics.total_pnl >= 0 ? "text-emerald" : "text-red"}`}>
                    ${metrics.total_pnl.toLocaleString()}
                  </p>
                </div>
                <div className="glass p-4">
                  <p className="text-xs text-text-muted uppercase mb-1">Win Rate</p>
                  <p className="text-xl font-bold mono text-white">
                    {metrics.win_rate}%
                  </p>
                </div>
                <div className="glass p-4">
                  <p className="text-xs text-text-muted uppercase mb-1">Profit Factor</p>
                  <p className={`text-xl font-bold mono ${metrics.profit_factor >= 1 ? "text-emerald" : "text-red"}`}>
                    {metrics.profit_factor}
                  </p>
                </div>
                <div className="glass p-4">
                  <p className="text-xs text-text-muted uppercase mb-1">Max Drawdown</p>
                  <p className="text-xl font-bold mono text-red">
                    -{metrics.max_drawdown}%
                  </p>
                </div>
              </div>

              {/* Equity Curve */}
              <div className="glass p-5 opacity-0 animate-slide-up stagger-3">
                <h3 className="text-sm font-medium text-text-secondary mb-4">
                  Equity Curve
                </h3>
                <div className="h-48">
                  {mounted && equityCurve.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={equityCurve}>
                        <defs>
                          <linearGradient id="backtestGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                            <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <XAxis
                          dataKey="date"
                          axisLine={false}
                          tickLine={false}
                          tick={{ fill: "#71717a", fontSize: 11 }}
                        />
                        <YAxis
                          axisLine={false}
                          tickLine={false}
                          tick={{ fill: "#71717a", fontSize: 11 }}
                          tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                          width={55}
                        />
                        <Tooltip
                          content={({ active, payload, label }) => {
                            if (active && payload && payload.length) {
                              return (
                                <div className="glass px-3 py-2 border border-border">
                                  <p className="text-xs text-text-muted">{label}</p>
                                  <p className="text-sm font-semibold text-emerald mono">
                                    ${payload[0].value?.toLocaleString()}
                                  </p>
                                </div>
                              );
                            }
                            return null;
                          }}
                        />
                        <Area
                          type="monotone"
                          dataKey="value"
                          stroke="#10b981"
                          strokeWidth={2}
                          fill="url(#backtestGradient)"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center">
                      <p className="text-text-muted">No equity data</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Trade List */}
              <div className="glass p-5 opacity-0 animate-slide-up stagger-4">
                <h3 className="text-sm font-medium text-text-secondary mb-4">
                  Trades ({metrics.total_trades} total)
                </h3>
                {trades.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Date</th>
                          <th>Symbol</th>
                          <th>Side</th>
                          <th>P&L</th>
                          <th>P&L %</th>
                        </tr>
                      </thead>
                      <tbody>
                        {trades.slice(0, 20).map((trade, i) => (
                          <tr key={i}>
                            <td className="text-text-secondary text-sm">{trade.date}</td>
                            <td className="font-semibold text-white mono">{trade.symbol}</td>
                            <td>
                              <span className={`badge ${trade.side === "LONG" ? "badge-emerald" : "badge-red"}`}>
                                {trade.side}
                              </span>
                            </td>
                            <td className={`mono font-medium ${trade.pnl >= 0 ? "pnl-positive" : "pnl-negative"}`}>
                              {trade.pnl >= 0 ? "+" : ""}${trade.pnl}
                            </td>
                            <td className={`mono font-medium ${trade.pnl_percent >= 0 ? "pnl-positive" : "pnl-negative"}`}>
                              {trade.pnl_percent >= 0 ? "+" : ""}{trade.pnl_percent}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-text-muted text-center py-4">No trades executed</p>
                )}
              </div>
            </>
          ) : (
            <div className="glass p-12 text-center opacity-0 animate-slide-up stagger-2">
              <TestTube className="w-12 h-12 text-text-muted mx-auto mb-4" />
              <p className="text-text-secondary">Configure scanner settings and run a backtest</p>
            </div>
          )}
        </div>
      </div>
    </PageWrapper>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/app/backtest/page.tsx
git commit -m "feat: update backtest page with scanner-based UI"
```

---

## Task 4: Add Environment Config

**Files:**
- Create: `frontend/.env.local`

**Step 1: Create env file**

```bash
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > frontend/.env.local
```

**Step 2: Commit example**

```bash
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > frontend/.env.local.example
git add frontend/.env.local.example
git commit -m "docs: add env example for API URL"
```

---

## Task 5: Test End-to-End

**Step 1: Install FastAPI**

```bash
pip install fastapi uvicorn[standard]
```

**Step 2: Start API (terminal 1)**

```bash
python3 -m uvicorn api.main:app --reload --port 8000
```

**Step 3: Start frontend (terminal 2)**

```bash
cd frontend && npm run dev
```

**Step 4: Test in browser**

1. Open http://localhost:3000/backtest
2. Select "10" for top stocks
3. Select "30" for days
4. Click "Run Backtest"
5. Verify scanner-selected symbols appear
6. Verify results display correctly

---

## Summary

**What changed from original plan:**
- Removed manual symbol selection
- Added "Top N" selector (5, 10, 15, 20, 25)
- Added "Days" selector (7, 14, 30, 60, 90)
- Scanner automatically picks most volatile stocks
- Shows which symbols the scanner selected
- Warmup handled internally by backtest (40+ days fetched, only requested period counted)
