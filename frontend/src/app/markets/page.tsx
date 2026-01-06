"use client";

import { PageWrapper } from "@/components/layout/PageWrapper";
import { Globe, TrendingUp, TrendingDown, AlertTriangle } from "lucide-react";
import {
  AreaChart,
  Area,
  ResponsiveContainer,
} from "recharts";
import { useMounted } from "@/lib/utils";

const indexData = [
  {
    symbol: "SPY",
    name: "S&P 500",
    price: 477.42,
    change: 1.25,
    sparkline: [470, 472, 471, 474, 476, 475, 477],
  },
  {
    symbol: "QQQ",
    name: "Nasdaq 100",
    price: 412.35,
    change: 1.82,
    sparkline: [405, 408, 407, 410, 411, 409, 412],
  },
  {
    symbol: "VIX",
    name: "Volatility Index",
    price: 14.25,
    change: -5.32,
    sparkline: [16, 15.5, 15, 14.8, 14.5, 14.3, 14.25],
    isElevated: false,
  },
];

const sectorData = [
  { name: "Technology", change: 2.15 },
  { name: "Healthcare", change: 0.85 },
  { name: "Financials", change: -0.45 },
  { name: "Energy", change: 1.32 },
  { name: "Consumer", change: 0.65 },
  { name: "Industrial", change: -0.22 },
  { name: "Materials", change: 0.95 },
  { name: "Utilities", change: -0.15 },
  { name: "Real Estate", change: -0.85 },
  { name: "Telecom", change: 0.42 },
];

const topMovers = [
  { symbol: "NVDA", change: 4.52 },
  { symbol: "TSLA", change: 3.21 },
  { symbol: "AMD", change: 2.85 },
  { symbol: "META", change: -1.92 },
  { symbol: "NFLX", change: -1.45 },
];

const activeSignals = [
  { action: "BUY", symbol: "AAPL", strategy: "Momentum", confidence: 78 },
  { action: "SELL", symbol: "TSLA", strategy: "Mean Reversion", confidence: 72 },
  { action: "BUY", symbol: "MSFT", strategy: "Breakout", confidence: 68 },
];

function MiniSparkline({ data, isPositive }: { data: number[]; isPositive: boolean }) {
  const chartData = data.map((value, i) => ({ value, index: i }));
  const mounted = useMounted();

  return (
    <div className="w-20 h-8">
      {mounted ? (
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id={`spark-${isPositive ? 'green' : 'red'}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={isPositive ? "#10b981" : "#ef4444"} stopOpacity={0.3} />
                <stop offset="100%" stopColor={isPositive ? "#10b981" : "#ef4444"} stopOpacity={0} />
              </linearGradient>
            </defs>
            <Area
              type="monotone"
              dataKey="value"
              stroke={isPositive ? "#10b981" : "#ef4444"}
              strokeWidth={1.5}
              fill={`url(#spark-${isPositive ? 'green' : 'red'})`}
            />
          </AreaChart>
        </ResponsiveContainer>
      ) : null}
    </div>
  );
}

function SectorHeatmapCell({ name, change }: { name: string; change: number }) {
  const intensity = Math.min(Math.abs(change) / 3, 1);
  const bgColor =
    change >= 0
      ? `rgba(16, 185, 129, ${intensity * 0.4})`
      : `rgba(239, 68, 68, ${intensity * 0.4})`;
  const borderColor =
    change >= 0
      ? `rgba(16, 185, 129, ${intensity * 0.6})`
      : `rgba(239, 68, 68, ${intensity * 0.6})`;

  return (
    <div
      className="p-3 rounded-lg border cursor-pointer transition-transform hover:scale-105"
      style={{ backgroundColor: bgColor, borderColor }}
    >
      <div className="text-xs text-text-secondary mb-1">{name}</div>
      <div
        className={`text-sm font-semibold mono ${
          change >= 0 ? "text-emerald" : "text-red"
        }`}
      >
        {change >= 0 ? "+" : ""}
        {change.toFixed(2)}%
      </div>
    </div>
  );
}

export default function MarketOverviewPage() {
  return (
    <PageWrapper title="Market Overview" subtitle="Track market indices and sector performance">
      <div className="space-y-6">
        {/* Index Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 opacity-0 animate-slide-up stagger-1">
          {indexData.map((index) => (
            <div
              key={index.symbol}
              className={`glass glass-hover p-5 ${
                index.symbol === "VIX" && index.price > 20 ? "border-red/30" : ""
              }`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-white">{index.symbol}</span>
                    {index.symbol === "VIX" && index.price > 20 && (
                      <AlertTriangle className="w-4 h-4 text-red" />
                    )}
                  </div>
                  <span className="text-xs text-text-muted">{index.name}</span>
                </div>
                <MiniSparkline data={index.sparkline} isPositive={index.change >= 0} />
              </div>
              <div className="mt-3 flex items-end justify-between">
                <span className="text-2xl font-bold text-white mono">
                  ${index.price.toFixed(2)}
                </span>
                <span
                  className={`flex items-center gap-1 text-sm font-medium ${
                    index.change >= 0 ? "text-emerald" : "text-red"
                  }`}
                >
                  {index.change >= 0 ? (
                    <TrendingUp className="w-4 h-4" />
                  ) : (
                    <TrendingDown className="w-4 h-4" />
                  )}
                  {index.change >= 0 ? "+" : ""}
                  {index.change.toFixed(2)}%
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* Sector Heatmap */}
        <div className="glass p-5 opacity-0 animate-slide-up stagger-2">
          <h3 className="text-sm font-medium text-text-secondary mb-4">
            Sector Performance
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {sectorData.map((sector) => (
              <SectorHeatmapCell
                key={sector.name}
                name={sector.name}
                change={sector.change}
              />
            ))}
          </div>
        </div>

        {/* Bottom columns */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Top Movers */}
          <div className="glass p-5 opacity-0 animate-slide-up stagger-3">
            <h3 className="text-sm font-medium text-text-secondary mb-4">
              Top Movers (Watchlist)
            </h3>
            <div className="space-y-3">
              {topMovers.map((stock) => (
                <div
                  key={stock.symbol}
                  className="flex items-center justify-between py-2 border-b border-border last:border-0"
                >
                  <span className="font-semibold text-white mono">
                    {stock.symbol}
                  </span>
                  <span
                    className={`font-medium mono ${
                      stock.change >= 0 ? "text-emerald" : "text-red"
                    }`}
                  >
                    {stock.change >= 0 ? "+" : ""}
                    {stock.change.toFixed(2)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Active Signals */}
          <div className="glass p-5 opacity-0 animate-slide-up stagger-4">
            <h3 className="text-sm font-medium text-text-secondary mb-4">
              Active Signals
            </h3>
            <div className="space-y-3">
              {activeSignals.map((signal, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between py-2 border-b border-border last:border-0"
                >
                  <div className="flex items-center gap-3">
                    <span
                      className={`badge ${
                        signal.action === "BUY" ? "badge-emerald" : "badge-red"
                      }`}
                    >
                      {signal.action}
                    </span>
                    <span className="font-semibold text-white mono">
                      {signal.symbol}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-text-secondary">
                      {signal.strategy}
                    </div>
                    <div className="text-xs text-emerald mono">
                      {signal.confidence}% confidence
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </PageWrapper>
  );
}
