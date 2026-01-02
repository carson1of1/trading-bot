"use client";

import { useState } from "react";
import { PageWrapper } from "@/components/layout/PageWrapper";
import { TestTube, Play, Calendar } from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useMounted } from "@/lib/utils";

const mockBacktestResults = {
  totalPnl: 8542.35,
  winRate: 62.5,
  profitFactor: 1.85,
  maxDrawdown: -3.2,
  totalTrades: 48,
  equityCurve: [
    { date: "Nov 01", value: 100000 },
    { date: "Nov 08", value: 102500 },
    { date: "Nov 15", value: 101800 },
    { date: "Nov 22", value: 105200 },
    { date: "Nov 29", value: 104500 },
    { date: "Dec 06", value: 107800 },
    { date: "Dec 13", value: 108542 },
  ],
  trades: [
    { date: "2024-12-13", symbol: "AAPL", side: "LONG", pnl: 245, pnlPercent: 1.2 },
    { date: "2024-12-12", symbol: "MSFT", side: "LONG", pnl: 180, pnlPercent: 0.9 },
    { date: "2024-12-11", symbol: "NVDA", side: "SHORT", pnl: -120, pnlPercent: -0.6 },
    { date: "2024-12-10", symbol: "TSLA", side: "LONG", pnl: 320, pnlPercent: 1.5 },
    { date: "2024-12-09", symbol: "META", side: "LONG", pnl: 280, pnlPercent: 1.3 },
  ],
};

const watchlistSymbols = ["AAPL", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "AMZN", "SPY", "QQQ"];

export default function BacktestPage() {
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(["AAPL", "MSFT", "NVDA"]);
  const [startDate, setStartDate] = useState("2024-11-01");
  const [endDate, setEndDate] = useState("2024-12-13");
  const [sideFilter, setSideFilter] = useState<"both" | "longs" | "shorts">("both");
  const [hasResults, setHasResults] = useState(true);
  const mounted = useMounted();

  const toggleSymbol = (symbol: string) => {
    setSelectedSymbols((prev) =>
      prev.includes(symbol)
        ? prev.filter((s) => s !== symbol)
        : [...prev, symbol]
    );
  };

  return (
    <PageWrapper title="Backtesting" subtitle="Test strategies on historical data">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Config Panel */}
        <div className="glass p-5 opacity-0 animate-slide-up stagger-1 h-fit">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-lg bg-emerald-glow">
              <TestTube className="w-5 h-5 text-emerald" />
            </div>
            <h3 className="font-semibold text-white">Configuration</h3>
          </div>

          {/* Symbol Selection */}
          <div className="mb-6">
            <label className="block text-sm text-text-secondary mb-2">
              Symbols
            </label>
            <div className="flex flex-wrap gap-2">
              {watchlistSymbols.map((symbol) => (
                <button
                  key={symbol}
                  onClick={() => toggleSymbol(symbol)}
                  className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-all ${
                    selectedSymbols.includes(symbol)
                      ? "bg-emerald text-black"
                      : "bg-surface-2 text-text-secondary hover:text-white"
                  }`}
                >
                  {symbol}
                </button>
              ))}
            </div>
          </div>

          {/* Date Range */}
          <div className="grid grid-cols-2 gap-3 mb-6">
            <div>
              <label className="block text-sm text-text-secondary mb-2">
                Start Date
              </label>
              <div className="relative">
                <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="input !pl-10"
                />
              </div>
            </div>
            <div>
              <label className="block text-sm text-text-secondary mb-2">
                End Date
              </label>
              <div className="relative">
                <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="input !pl-10"
                />
              </div>
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
                  {side === "both" ? "Both" : side === "longs" ? "Long Only" : "Short Only"}
                </button>
              ))}
            </div>
          </div>

          <button className="btn btn-primary w-full">
            <Play className="w-4 h-4" />
            Run Backtest
          </button>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-6">
          {hasResults ? (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 opacity-0 animate-slide-up stagger-2">
                <div className="glass p-4">
                  <p className="text-xs text-text-muted uppercase mb-1">Total P&L</p>
                  <p className={`text-xl font-bold mono ${mockBacktestResults.totalPnl >= 0 ? "text-emerald" : "text-red"}`}>
                    ${mockBacktestResults.totalPnl.toLocaleString()}
                  </p>
                </div>
                <div className="glass p-4">
                  <p className="text-xs text-text-muted uppercase mb-1">Win Rate</p>
                  <p className="text-xl font-bold mono text-white">
                    {mockBacktestResults.winRate}%
                  </p>
                </div>
                <div className="glass p-4">
                  <p className="text-xs text-text-muted uppercase mb-1">Profit Factor</p>
                  <p className="text-xl font-bold mono text-emerald">
                    {mockBacktestResults.profitFactor}
                  </p>
                </div>
                <div className="glass p-4">
                  <p className="text-xs text-text-muted uppercase mb-1">Max Drawdown</p>
                  <p className="text-xl font-bold mono text-red">
                    {mockBacktestResults.maxDrawdown}%
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
                      <AreaChart data={mockBacktestResults.equityCurve}>
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
                  Recent Trades ({mockBacktestResults.totalTrades} total)
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
                    {mockBacktestResults.trades.map((trade, i) => (
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
                        <td className={`mono font-medium ${trade.pnlPercent >= 0 ? "pnl-positive" : "pnl-negative"}`}>
                          {trade.pnlPercent >= 0 ? "+" : ""}{trade.pnlPercent}%
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
