"use client";

import { useState } from "react";
import { PageWrapper } from "@/components/layout/PageWrapper";
import { TestTube, Play, ScanSearch, Loader2 } from "lucide-react";
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
        days,
        longs_only: sideFilter === "longs",
        shorts_only: sideFilter === "shorts",
      });
      setResults(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  // Transform equity curve data for the chart
  const equityCurveData = results?.equity_curve.map((point) => ({
    date: point.timestamp,
    value: point.portfolio_value,
  })) || [];

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

          {/* Top N Selection */}
          <div className="mb-6">
            <label className="block text-sm text-text-secondary mb-2">
              Top N Symbols
            </label>
            <div className="flex flex-wrap gap-2">
              {TOP_N_OPTIONS.map((n) => (
                <button
                  key={n}
                  onClick={() => setTopN(n)}
                  className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-all ${
                    topN === n
                      ? "bg-emerald text-black"
                      : "bg-surface-2 text-text-secondary hover:text-white"
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          {/* Days Selection */}
          <div className="mb-6">
            <label className="block text-sm text-text-secondary mb-2">
              Days
            </label>
            <div className="flex flex-wrap gap-2">
              {DAYS_OPTIONS.map((d) => (
                <button
                  key={d}
                  onClick={() => setDays(d)}
                  className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-all ${
                    days === d
                      ? "bg-emerald text-black"
                      : "bg-surface-2 text-text-secondary hover:text-white"
                  }`}
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
                  className={`flex-1 px-3 py-2 text-sm font-medium rounded-md capitalize transition-all ${
                    sideFilter === side
                      ? "bg-emerald text-black"
                      : "text-text-muted hover:text-white"
                  }`}
                >
                  {side === "both" ? "Both" : side === "longs" ? "Long" : "Short"}
                </button>
              ))}
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-4 p-3 bg-red/10 border border-red/30 rounded-lg">
              <p className="text-sm text-red">{error}</p>
            </div>
          )}

          <button
            className="btn btn-primary w-full"
            onClick={handleRunBacktest}
            disabled={isLoading}
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
          {results && results.symbols_scanned.length > 0 && (
            <div className="mt-6">
              <label className="block text-sm text-text-secondary mb-2">
                Scanner Selected:
              </label>
              <div className="flex flex-wrap gap-2">
                {results.symbols_scanned.map((symbol) => (
                  <span
                    key={symbol}
                    className="px-2 py-1 text-xs font-medium bg-surface-2 text-text-secondary rounded"
                  >
                    {symbol}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-6">
          {results ? (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 opacity-0 animate-slide-up stagger-2">
                <div className="glass p-4">
                  <p className="text-xs text-text-muted uppercase mb-1">Total P&L</p>
                  <p className={`text-xl font-bold mono ${(results.metrics?.total_pnl ?? 0) >= 0 ? "text-emerald" : "text-red"}`}>
                    ${(results.metrics?.total_pnl ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </p>
                </div>
                <div className="glass p-4">
                  <p className="text-xs text-text-muted uppercase mb-1">Win Rate</p>
                  <p className="text-xl font-bold mono text-white">
                    {(results.metrics?.win_rate ?? 0).toFixed(1)}%
                  </p>
                </div>
                <div className="glass p-4">
                  <p className="text-xs text-text-muted uppercase mb-1">Profit Factor</p>
                  <p className="text-xl font-bold mono text-emerald">
                    {(results.metrics?.profit_factor ?? 0).toFixed(2)}
                  </p>
                </div>
                <div className="glass p-4">
                  <p className="text-xs text-text-muted uppercase mb-1">Max Drawdown</p>
                  <p className="text-xl font-bold mono text-red">
                    {(results.metrics?.max_drawdown ?? 0).toFixed(1)}%
                  </p>
                </div>
              </div>

              {/* Equity Curve */}
              <div className="glass p-5 opacity-0 animate-slide-up stagger-3">
                <h3 className="text-sm font-medium text-text-secondary mb-4">
                  Backtest Equity Curve
                </h3>
                <div className="h-48">
                  {mounted ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={equityCurveData}>
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
                      <div className="w-6 h-6 border-2 border-emerald border-t-transparent rounded-full animate-spin" />
                    </div>
                  )}
                </div>
              </div>

              {/* Trade List */}
              <div className="glass p-5 opacity-0 animate-slide-up stagger-4">
                <h3 className="text-sm font-medium text-text-secondary mb-4">
                  Recent Trades ({results.trades.length} total)
                </h3>
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
                    {results.trades.slice(0, 20).map((trade, i) => (
                      <tr key={i}>
                        <td className="text-text-secondary text-sm">{trade.exit_date}</td>
                        <td className="font-semibold text-white mono">{trade.symbol}</td>
                        <td>
                          <span className={`badge ${trade.direction === "LONG" ? "badge-emerald" : "badge-red"}`}>
                            {trade.direction}
                          </span>
                        </td>
                        <td className={`mono font-medium ${trade.pnl >= 0 ? "pnl-positive" : "pnl-negative"}`}>
                          {trade.pnl >= 0 ? "+" : ""}${trade.pnl.toFixed(2)}
                        </td>
                        <td className={`mono font-medium ${trade.pnl_pct >= 0 ? "pnl-positive" : "pnl-negative"}`}>
                          {trade.pnl_pct >= 0 ? "+" : ""}{trade.pnl_pct.toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div className="glass p-12 text-center opacity-0 animate-slide-up stagger-2">
              <TestTube className="w-12 h-12 text-text-muted mx-auto mb-4" />
              <p className="text-text-secondary">Configure and run a backtest to see results</p>
            </div>
          )}
        </div>
      </div>
    </PageWrapper>
  );
}
