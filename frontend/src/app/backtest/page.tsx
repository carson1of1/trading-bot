"use client";

import { useState } from "react";
import { PageWrapper } from "@/components/layout/PageWrapper";
import { TestTube, Play, ScanSearch, Loader2, ChevronDown, ChevronRight, TrendingUp, LogOut, BarChart3, Calendar } from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useMounted } from "@/lib/utils";
import { runBacktest, BacktestResponse, PeriodBreakdown } from "@/lib/api";

const TOP_N_OPTIONS = [5, 10, 15, 20, 25];
const DAYS_OPTIONS = [7, 14, 30, 60, 90, 180, 365];
const CAPITAL_OPTIONS = [5000, 10000, 25000, 50000, 100000];
const TRAILING_ACTIVATION_OPTIONS = [0.1, 0.15, 0.25, 0.5, 1.0];
const TRAILING_TRAIL_OPTIONS = [0.1, 0.15, 0.25, 0.5];

export default function BacktestPage() {
  const [topN, setTopN] = useState(10);
  const [days, setDays] = useState(30);
  const [initialCapital, setInitialCapital] = useState(10000);
  const [sideFilter, setSideFilter] = useState<"both" | "longs" | "shorts">("both");
  // Trailing stop state (defaults match live trading config.yaml)
  const [trailingEnabled, setTrailingEnabled] = useState(true);
  const [trailingActivation, setTrailingActivation] = useState(0.15);
  const [trailingTrail, setTrailingTrail] = useState(0.15);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<BacktestResponse | null>(null);
  const mounted = useMounted();

  // Analytics panel expansion state
  const [strategyExpanded, setStrategyExpanded] = useState(false);
  const [exitReasonExpanded, setExitReasonExpanded] = useState(false);
  const [symbolExpanded, setSymbolExpanded] = useState(false);
  const [periodExpanded, setPeriodExpanded] = useState(false);
  const [periodView, setPeriodView] = useState<"daily" | "monthly">("daily");

  const handleRunBacktest = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await runBacktest({
        top_n: topN,
        days,
        longs_only: sideFilter === "longs",
        shorts_only: sideFilter === "shorts",
        initial_capital: initialCapital,
        trailing_stop_enabled: trailingEnabled,
        trailing_activation_pct: trailingActivation,
        trailing_trail_pct: trailingTrail,
      });
      setResults(response);
      // Default to monthly view for 90+ day backtests
      setPeriodView(days >= 90 ? "monthly" : "daily");
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  // Transform equity curve data for the chart
  const equityCurveData = results?.equity_curve.map((point) => {
    // Parse timestamp and format as short date
    const date = new Date(point.timestamp);
    const formattedDate = date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric'
    });
    return {
      date: formattedDate,
      value: point.portfolio_value,
      fullTimestamp: point.timestamp, // Keep for tooltip
    };
  }) || [];

  // Group daily data into monthly for the period breakdown
  const getMonthlyData = (dailyData: PeriodBreakdown[]): PeriodBreakdown[] => {
    const monthlyMap = new Map<string, { trades: number; wins: number; losses: number; total_pnl: number }>();

    for (const day of dailyData) {
      // Extract YYYY-MM from date
      const monthKey = day.date.substring(0, 7);
      const existing = monthlyMap.get(monthKey) || { trades: 0, wins: 0, losses: 0, total_pnl: 0 };
      existing.trades += day.trades;
      existing.wins += day.wins;
      existing.losses += day.losses;
      existing.total_pnl += day.total_pnl;
      monthlyMap.set(monthKey, existing);
    }

    return Array.from(monthlyMap.entries())
      .map(([month, data]) => ({
        date: month,
        trades: data.trades,
        wins: data.wins,
        losses: data.losses,
        win_rate: data.trades > 0 ? Math.round((data.wins / data.trades) * 1000) / 10 : 0,
        total_pnl: Math.round(data.total_pnl * 100) / 100,
        avg_pnl: data.trades > 0 ? Math.round((data.total_pnl / data.trades) * 100) / 100 : 0,
      }))
      .sort((a, b) => a.date.localeCompare(b.date));
  };

  // Get period data based on current view mode
  const periodData = results?.by_period
    ? (periodView === "monthly" ? getMonthlyData(results.by_period) : results.by_period)
    : [];

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

          {/* Starting Capital Selection */}
          <div className="mb-6">
            <label className="block text-sm text-text-secondary mb-2">
              Starting Capital
            </label>
            <div className="flex flex-wrap gap-2">
              {CAPITAL_OPTIONS.map((c) => (
                <button
                  key={c}
                  onClick={() => setInitialCapital(c)}
                  className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-all ${
                    initialCapital === c
                      ? "bg-emerald text-black"
                      : "bg-surface-2 text-text-secondary hover:text-white"
                  }`}
                >
                  ${(c / 1000).toFixed(0)}k
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

          {/* Trailing Stop Settings */}
          <div className="mb-6 p-4 bg-surface-1 rounded-lg">
            <div className="flex items-center justify-between mb-4">
              <label className="text-sm text-text-secondary">Trailing Stop</label>
              <button
                onClick={() => setTrailingEnabled(!trailingEnabled)}
                className={`relative w-10 h-5 rounded-full transition-colors ${
                  trailingEnabled ? "bg-emerald" : "bg-surface-2"
                }`}
              >
                <span
                  className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform ${
                    trailingEnabled ? "left-5" : "left-0.5"
                  }`}
                />
              </button>
            </div>

            {trailingEnabled && (
              <>
                <div className="mb-4">
                  <label className="block text-xs text-text-muted mb-2">
                    Activation (% profit)
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {TRAILING_ACTIVATION_OPTIONS.map((pct) => (
                      <button
                        key={pct}
                        onClick={() => setTrailingActivation(pct)}
                        className={`px-2 py-1 text-xs font-medium rounded-md transition-all ${
                          trailingActivation === pct
                            ? "bg-emerald text-black"
                            : "bg-surface-2 text-text-secondary hover:text-white"
                        }`}
                      >
                        {pct}%
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-xs text-text-muted mb-2">
                    Trail Distance (%)
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {TRAILING_TRAIL_OPTIONS.map((pct) => (
                      <button
                        key={pct}
                        onClick={() => setTrailingTrail(pct)}
                        className={`px-2 py-1 text-xs font-medium rounded-md transition-all ${
                          trailingTrail === pct
                            ? "bg-emerald text-black"
                            : "bg-surface-2 text-text-secondary hover:text-white"
                        }`}
                      >
                        {pct}%
                      </button>
                    ))}
                  </div>
                </div>
              </>
            )}
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
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4 opacity-0 animate-slide-up stagger-2">
                <div className="glass p-4">
                  <p className="text-xs text-text-muted uppercase mb-1">Total Return</p>
                  <p className={`text-xl font-bold mono ${(results.metrics?.total_return_pct ?? 0) >= 0 ? "text-emerald" : "text-red"}`}>
                    {(results.metrics?.total_return_pct ?? 0) >= 0 ? "+" : ""}{(results.metrics?.total_return_pct ?? 0).toFixed(2)}%
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
                <div className="glass p-4">
                  <p className="text-xs text-text-muted uppercase mb-1">Days Traded</p>
                  <p className="text-xl font-bold mono text-white">
                    {results.metrics?.days_traded ?? 0}
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
                      <AreaChart data={equityCurveData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
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
                          tick={{ fill: "#71717a", fontSize: 10 }}
                          interval="preserveStartEnd"
                        />
                        <YAxis
                          axisLine={false}
                          tickLine={false}
                          tick={{ fill: "#71717a", fontSize: 11 }}
                          tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                          width={55}
                          domain={['dataMin - 1000', 'dataMax + 1000']}
                        />
                        <Tooltip
                          content={({ active, payload }) => {
                            if (active && payload && payload.length) {
                              const data = payload[0].payload;
                              return (
                                <div className="glass px-3 py-2 border border-border">
                                  <p className="text-xs text-text-muted">{data.date}</p>
                                  <p className="text-sm font-semibold text-emerald mono">
                                    ${Number(payload[0].value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
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
                          dot={false}
                          activeDot={{ r: 4, fill: "#10b981", stroke: "#fff", strokeWidth: 2 }}
                          isAnimationActive={false}
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

              {/* Drawdown Analysis */}
              {(results.metrics?.max_drawdown ?? 0) > 0 && (
                <div className="glass p-5 opacity-0 animate-slide-up stagger-3">
                  <h3 className="text-sm font-medium text-text-secondary mb-4">
                    Drawdown Analysis
                  </h3>
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <p className="text-xs text-text-muted">Peak</p>
                      <p className="text-lg font-bold text-emerald">
                        ${results.metrics?.drawdown_peak_value?.toLocaleString()}
                      </p>
                      <p className="text-xs text-text-muted">
                        {results.metrics?.drawdown_peak_date?.split('T')[0]}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-text-muted">Trough</p>
                      <p className="text-lg font-bold text-red">
                        ${results.metrics?.drawdown_trough_value?.toLocaleString()}
                      </p>
                      <p className="text-xs text-text-muted">
                        {results.metrics?.drawdown_trough_date?.split('T')[0]}
                      </p>
                    </div>
                  </div>

                  {/* Worst Days Table */}
                  {results.worst_daily_drops?.length > 0 && (
                    <>
                      <h4 className="text-xs text-text-muted uppercase mt-4 mb-2">Worst Days</h4>
                      <div className="space-y-1">
                        {results.worst_daily_drops.slice(0, 5).map((day) => (
                          <div key={day.date} className="flex justify-between text-sm">
                            <span className="text-text-secondary">{day.date}</span>
                            <span className="text-red mono">
                              {day.change_pct.toFixed(2)}% (${day.change_dollars.toLocaleString()})
                            </span>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}

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

              {/* Analytics Panels */}
              {/* Strategy Performance */}
              {results.by_strategy && results.by_strategy.length > 0 && (
                <div className="glass p-5 opacity-0 animate-slide-up stagger-5">
                  <button
                    onClick={() => setStrategyExpanded(!strategyExpanded)}
                    className="flex items-center gap-3 w-full text-left"
                  >
                    {strategyExpanded ? (
                      <ChevronDown className="w-4 h-4 text-text-muted" />
                    ) : (
                      <ChevronRight className="w-4 h-4 text-text-muted" />
                    )}
                    <div className="p-2 rounded-lg bg-emerald-glow">
                      <TrendingUp className="w-4 h-4 text-emerald" />
                    </div>
                    <h3 className="text-sm font-medium text-text-secondary">
                      Performance by Strategy
                    </h3>
                  </button>
                  {strategyExpanded && (
                    <div className="mt-4 overflow-x-auto">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Strategy</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                            <th>Total P&L</th>
                            <th>Avg P&L</th>
                            <th>Avg MFE%</th>
                            <th>Avg MAE%</th>
                          </tr>
                        </thead>
                        <tbody>
                          {results.by_strategy.map((s, i) => (
                            <tr key={i}>
                              <td className="font-semibold text-white">{s.strategy}</td>
                              <td className="text-text-secondary">{s.trades}</td>
                              <td className={`mono ${s.win_rate >= 50 ? "text-emerald" : "text-red"}`}>
                                {s.win_rate.toFixed(1)}%
                              </td>
                              <td className={`mono font-medium ${s.total_pnl >= 0 ? "pnl-positive" : "pnl-negative"}`}>
                                {s.total_pnl >= 0 ? "+" : ""}${s.total_pnl.toLocaleString()}
                              </td>
                              <td className={`mono ${s.avg_pnl >= 0 ? "text-emerald" : "text-red"}`}>
                                {s.avg_pnl >= 0 ? "+" : ""}${s.avg_pnl.toFixed(2)}
                              </td>
                              <td className="mono text-emerald">{s.avg_mfe_pct.toFixed(2)}%</td>
                              <td className="mono text-red">{s.avg_mae_pct.toFixed(2)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* Exit Reason Breakdown */}
              {results.by_exit_reason && results.by_exit_reason.length > 0 && (
                <div className="glass p-5 opacity-0 animate-slide-up stagger-5">
                  <button
                    onClick={() => setExitReasonExpanded(!exitReasonExpanded)}
                    className="flex items-center gap-3 w-full text-left"
                  >
                    {exitReasonExpanded ? (
                      <ChevronDown className="w-4 h-4 text-text-muted" />
                    ) : (
                      <ChevronRight className="w-4 h-4 text-text-muted" />
                    )}
                    <div className="p-2 rounded-lg bg-amber-500/10">
                      <LogOut className="w-4 h-4 text-amber-500" />
                    </div>
                    <h3 className="text-sm font-medium text-text-secondary">
                      Exit Reason Breakdown
                    </h3>
                  </button>
                  {exitReasonExpanded && (
                    <div className="mt-4 overflow-x-auto">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Exit Reason</th>
                            <th>Count</th>
                            <th>% of Trades</th>
                            <th>Total P&L</th>
                            <th>Avg P&L</th>
                          </tr>
                        </thead>
                        <tbody>
                          {results.by_exit_reason.map((e, i) => (
                            <tr key={i}>
                              <td className="font-semibold text-white">{e.exit_reason}</td>
                              <td className="text-text-secondary">{e.count}</td>
                              <td className="mono text-text-secondary">{e.pct_of_trades.toFixed(1)}%</td>
                              <td className={`mono font-medium ${e.total_pnl >= 0 ? "pnl-positive" : "pnl-negative"}`}>
                                {e.total_pnl >= 0 ? "+" : ""}${e.total_pnl.toLocaleString()}
                              </td>
                              <td className={`mono ${e.avg_pnl >= 0 ? "text-emerald" : "text-red"}`}>
                                {e.avg_pnl >= 0 ? "+" : ""}${e.avg_pnl.toFixed(2)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* Symbol Performance */}
              {results.by_symbol && results.by_symbol.length > 0 && (
                <div className="glass p-5 opacity-0 animate-slide-up stagger-5">
                  <button
                    onClick={() => setSymbolExpanded(!symbolExpanded)}
                    className="flex items-center gap-3 w-full text-left"
                  >
                    {symbolExpanded ? (
                      <ChevronDown className="w-4 h-4 text-text-muted" />
                    ) : (
                      <ChevronRight className="w-4 h-4 text-text-muted" />
                    )}
                    <div className="p-2 rounded-lg bg-blue-500/10">
                      <BarChart3 className="w-4 h-4 text-blue-500" />
                    </div>
                    <h3 className="text-sm font-medium text-text-secondary">
                      Performance by Symbol (worst to best)
                    </h3>
                  </button>
                  {symbolExpanded && (
                    <div className="mt-4 overflow-x-auto max-h-80 overflow-y-auto">
                      <table className="data-table">
                        <thead className="sticky top-0 bg-surface-1">
                          <tr>
                            <th>Symbol</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                            <th>Total P&L</th>
                            <th>Avg P&L</th>
                          </tr>
                        </thead>
                        <tbody>
                          {results.by_symbol.map((s, i) => (
                            <tr key={i} className={s.total_pnl < -100 ? "bg-red/5" : s.total_pnl > 500 ? "bg-emerald/5" : ""}>
                              <td className="font-semibold text-white mono">{s.symbol}</td>
                              <td className="text-text-secondary">{s.trades}</td>
                              <td className={`mono ${s.win_rate >= 50 ? "text-emerald" : "text-red"}`}>
                                {s.win_rate.toFixed(1)}%
                              </td>
                              <td className={`mono font-medium ${s.total_pnl >= 0 ? "pnl-positive" : "pnl-negative"}`}>
                                {s.total_pnl >= 0 ? "+" : ""}${s.total_pnl.toLocaleString()}
                              </td>
                              <td className={`mono ${s.avg_pnl >= 0 ? "text-emerald" : "text-red"}`}>
                                {s.avg_pnl >= 0 ? "+" : ""}${s.avg_pnl.toFixed(2)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* Performance by Period */}
              {results.by_period && results.by_period.length > 0 && (
                <div className="glass p-5 opacity-0 animate-slide-up stagger-5">
                  <div className="flex items-center justify-between">
                    <button
                      onClick={() => setPeriodExpanded(!periodExpanded)}
                      className="flex items-center gap-3 text-left"
                    >
                      {periodExpanded ? (
                        <ChevronDown className="w-4 h-4 text-text-muted" />
                      ) : (
                        <ChevronRight className="w-4 h-4 text-text-muted" />
                      )}
                      <div className="p-2 rounded-lg bg-purple-500/10">
                        <Calendar className="w-4 h-4 text-purple-500" />
                      </div>
                      <h3 className="text-sm font-medium text-text-secondary">
                        Performance by Period
                      </h3>
                    </button>
                    {/* Toggle for Daily/Monthly */}
                    <div className="flex gap-1 p-0.5 bg-surface-1 rounded-lg">
                      <button
                        onClick={() => setPeriodView("daily")}
                        className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                          periodView === "daily"
                            ? "bg-purple-500 text-white"
                            : "text-text-muted hover:text-white"
                        }`}
                      >
                        Daily
                      </button>
                      <button
                        onClick={() => setPeriodView("monthly")}
                        className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                          periodView === "monthly"
                            ? "bg-purple-500 text-white"
                            : "text-text-muted hover:text-white"
                        }`}
                      >
                        Monthly
                      </button>
                    </div>
                  </div>
                  {periodExpanded && (
                    <div className="mt-4 overflow-x-auto max-h-80 overflow-y-auto">
                      <table className="data-table">
                        <thead className="sticky top-0 bg-surface-1">
                          <tr>
                            <th>{periodView === "monthly" ? "Month" : "Date"}</th>
                            <th>Trades</th>
                            <th>Wins</th>
                            <th>Losses</th>
                            <th>Win Rate</th>
                            <th>Total P&L</th>
                            <th>Avg P&L</th>
                          </tr>
                        </thead>
                        <tbody>
                          {periodData.map((p, i) => (
                            <tr key={i} className={p.total_pnl < 0 ? "bg-red/5" : p.total_pnl > 100 ? "bg-emerald/5" : ""}>
                              <td className="font-semibold text-white mono">
                                {periodView === "monthly"
                                  ? new Date(p.date + "-01").toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
                                  : new Date(p.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
                                }
                              </td>
                              <td className="text-text-secondary">{p.trades}</td>
                              <td className="text-emerald">{p.wins}</td>
                              <td className="text-red">{p.losses}</td>
                              <td className={`mono ${p.win_rate >= 50 ? "text-emerald" : "text-red"}`}>
                                {p.win_rate.toFixed(1)}%
                              </td>
                              <td className={`mono font-medium ${p.total_pnl >= 0 ? "pnl-positive" : "pnl-negative"}`}>
                                {p.total_pnl >= 0 ? "+" : ""}${p.total_pnl.toLocaleString()}
                              </td>
                              <td className={`mono ${p.avg_pnl >= 0 ? "text-emerald" : "text-red"}`}>
                                {p.avg_pnl >= 0 ? "+" : ""}${p.avg_pnl.toFixed(2)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
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
