"use client";

import { useState } from "react";
import { PageWrapper } from "@/components/layout/PageWrapper";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { useMounted } from "@/lib/utils";

type TimeRange = "7D" | "30D" | "90D" | "YTD" | "ALL";
const timeRanges: TimeRange[] = ["7D", "30D", "90D", "YTD", "ALL"];

const equityData = [
  { date: "Nov 01", value: 100000 },
  { date: "Nov 15", value: 105200 },
  { date: "Dec 01", value: 108500 },
  { date: "Dec 15", value: 115800 },
  { date: "Dec 30", value: 125432 },
];

const hourlyData = Array.from({ length: 24 }, (_, i) => ({
  hour: `${i.toString().padStart(2, "0")}:00`,
  trades: Math.floor(Math.random() * 20) + 5,
  winRate: Math.random() * 40 + 40,
}));

const drawdownData = [
  { date: "Nov 01", drawdown: 0 },
  { date: "Nov 08", drawdown: -1.2 },
  { date: "Nov 15", drawdown: -0.5 },
  { date: "Nov 22", drawdown: -2.8 },
  { date: "Nov 29", drawdown: -1.5 },
  { date: "Dec 06", drawdown: -0.8 },
  { date: "Dec 13", drawdown: -1.1 },
  { date: "Dec 20", drawdown: -0.3 },
  { date: "Dec 27", drawdown: 0 },
];

const maxDrawdown = Math.min(...drawdownData.map((d) => d.drawdown));

export default function AnalyticsPage() {
  const [selectedRange, setSelectedRange] = useState<TimeRange>("30D");
  const mounted = useMounted();

  const CustomTooltip = ({
    active,
    payload,
    label,
  }: {
    active?: boolean;
    payload?: Array<{ value: number }>;
    label?: string;
  }) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass px-3 py-2 border border-border">
          <p className="text-xs text-text-muted">{label}</p>
          <p className="text-sm font-semibold text-emerald mono">
            ${payload[0].value.toLocaleString()}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <PageWrapper title="Analytics" subtitle="Analyze your trading performance">
      <div className="space-y-6">
        {/* Equity Curve */}
        <div className="glass p-5 opacity-0 animate-slide-up stagger-1">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-text-secondary">
              Equity Curve
            </h3>
            <div className="flex gap-1 p-0.5 bg-surface-1 rounded-lg">
              {timeRanges.map((range) => (
                <button
                  key={range}
                  onClick={() => setSelectedRange(range)}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                    selectedRange === range
                      ? "bg-emerald text-black"
                      : "text-text-muted hover:text-white"
                  }`}
                >
                  {range}
                </button>
              ))}
            </div>
          </div>
          <div className="h-64">
            {mounted ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={equityData}>
                  <defs>
                    <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
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
                    width={60}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="value"
                    stroke="#10b981"
                    strokeWidth={2}
                    fill="url(#equityFill)"
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

        {/* Trade Distribution by Hour */}
        <div className="glass p-5 opacity-0 animate-slide-up stagger-2">
          <h3 className="text-sm font-medium text-text-secondary mb-4">
            Trade Distribution by Hour
          </h3>
          <div className="h-48">
            {mounted ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={hourlyData}>
                  <XAxis
                    dataKey="hour"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "#71717a", fontSize: 9 }}
                    interval={2}
                  />
                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "#71717a", fontSize: 11 }}
                    width={30}
                  />
                  <Tooltip
                    content={({ active, payload, label }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="glass px-3 py-2 border border-border">
                            <p className="text-xs text-text-muted">{label}</p>
                            <p className="text-sm text-white">
                              {data.trades} trades
                            </p>
                            <p className="text-xs text-emerald">
                              {data.winRate.toFixed(1)}% win rate
                            </p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar dataKey="trades" radius={[4, 4, 0, 0]}>
                    {hourlyData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={
                          entry.winRate >= 60
                            ? "#10b981"
                            : entry.winRate >= 50
                            ? "#f59e0b"
                            : "#ef4444"
                        }
                        fillOpacity={0.7}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center">
                <div className="w-6 h-6 border-2 border-emerald border-t-transparent rounded-full animate-spin" />
              </div>
            )}
          </div>
          <div className="flex items-center justify-center gap-6 mt-4 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-emerald" />
              <span className="text-text-muted">Win rate {">"}60%</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-amber" />
              <span className="text-text-muted">Win rate 50-60%</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-red" />
              <span className="text-text-muted">Win rate {"<"}50%</span>
            </div>
          </div>
        </div>

        {/* Drawdown Chart */}
        <div className="glass p-5 opacity-0 animate-slide-up stagger-3">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-text-secondary">
              Drawdown
            </h3>
            <span className="text-sm text-red mono font-medium">
              Max: {maxDrawdown.toFixed(2)}%
            </span>
          </div>
          <div className="h-40">
            {mounted ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={drawdownData}>
                  <defs>
                    <linearGradient id="drawdownFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ef4444" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#ef4444" stopOpacity={0.1} />
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
                    tickFormatter={(v) => `${v}%`}
                    domain={["dataMin", 0]}
                    width={45}
                  />
                  <Tooltip
                    content={({ active, payload, label }) => {
                      if (active && payload && payload.length) {
                        return (
                          <div className="glass px-3 py-2 border border-border">
                            <p className="text-xs text-text-muted">{label}</p>
                            <p className="text-sm font-semibold text-red mono">
                              {payload[0].value}%
                            </p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="drawdown"
                    stroke="#ef4444"
                    strokeWidth={2}
                    fill="url(#drawdownFill)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center">
                <div className="w-6 h-6 border-2 border-red border-t-transparent rounded-full animate-spin" />
              </div>
            )}
          </div>
        </div>
      </div>
    </PageWrapper>
  );
}
