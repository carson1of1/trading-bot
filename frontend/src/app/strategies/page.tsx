"use client";

import { useState } from "react";
import { PageWrapper } from "@/components/layout/PageWrapper";
import { Layers, TrendingUp, TrendingDown, Activity } from "lucide-react";

interface Strategy {
  id: string;
  name: string;
  enabled: boolean;
  weight: number;
  winRate: number;
  pnl: number;
  tradeCount: number;
  lastSignal: {
    action: "BUY" | "SELL" | "HOLD";
    symbol: string;
    time: string;
  } | null;
}

const mockStrategies: Strategy[] = [
  {
    id: "momentum",
    name: "Momentum",
    enabled: true,
    weight: 40,
    winRate: 72.5,
    pnl: 4250.0,
    tradeCount: 28,
    lastSignal: { action: "BUY", symbol: "AAPL", time: "5 min ago" },
  },
  {
    id: "mean_reversion",
    name: "Mean Reversion",
    enabled: true,
    weight: 35,
    winRate: 65.2,
    pnl: 2180.0,
    tradeCount: 23,
    lastSignal: { action: "SELL", symbol: "TSLA", time: "12 min ago" },
  },
  {
    id: "breakout",
    name: "Breakout",
    enabled: true,
    weight: 25,
    winRate: 58.3,
    pnl: 980.0,
    tradeCount: 12,
    lastSignal: { action: "HOLD", symbol: "NVDA", time: "1 hr ago" },
  },
];

export default function StrategiesPage() {
  const [strategies, setStrategies] = useState(mockStrategies);

  const toggleStrategy = (id: string) => {
    setStrategies((prev) =>
      prev.map((s) => (s.id === id ? { ...s, enabled: !s.enabled } : s))
    );
  };

  const updateWeight = (id: string, weight: number) => {
    setStrategies((prev) =>
      prev.map((s) => (s.id === id ? { ...s, weight } : s))
    );
  };

  return (
    <PageWrapper title="Strategies" subtitle="Configure and monitor trading strategies">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {strategies.map((strategy, index) => (
          <div
            key={strategy.id}
            className={`glass p-5 opacity-0 animate-slide-up stagger-${index + 1}`}
          >
            <div className={`${!strategy.enabled ? "opacity-60" : ""}`}>
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-emerald-glow">
                  <Layers className="w-5 h-5 text-emerald" />
                </div>
                <h3 className="font-semibold text-white">{strategy.name}</h3>
              </div>
              <button
                onClick={() => toggleStrategy(strategy.id)}
                className={`relative w-12 h-6 rounded-full transition-colors ${
                  strategy.enabled ? "bg-emerald" : "bg-surface-3"
                }`}
              >
                <span
                  className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
                    strategy.enabled ? "left-7" : "left-1"
                  }`}
                />
              </button>
            </div>

            {/* Weight Slider */}
            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-text-secondary">Weight</span>
                <span className="text-sm font-medium text-emerald mono">
                  {strategy.weight}%
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={strategy.weight}
                onChange={(e) => updateWeight(strategy.id, parseInt(e.target.value))}
                className="w-full h-1.5 bg-surface-2 rounded-full appearance-none cursor-pointer
                  [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4
                  [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full
                  [&::-webkit-slider-thumb]:bg-emerald [&::-webkit-slider-thumb]:cursor-pointer
                  [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(16,185,129,0.5)]"
                disabled={!strategy.enabled}
              />
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div>
                <p className="text-xs text-text-muted uppercase mb-1">Win Rate</p>
                <p className="text-lg font-bold text-white mono">
                  {strategy.winRate}%
                </p>
              </div>
              <div>
                <p className="text-xs text-text-muted uppercase mb-1">P&L</p>
                <p
                  className={`text-lg font-bold mono ${
                    strategy.pnl >= 0 ? "text-emerald" : "text-red"
                  }`}
                >
                  ${strategy.pnl.toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-xs text-text-muted uppercase mb-1">Trades</p>
                <p className="text-lg font-bold text-white mono">
                  {strategy.tradeCount}
                </p>
              </div>
            </div>

            {/* Last Signal */}
            {strategy.lastSignal && (
              <div className="pt-4 border-t border-border">
                <p className="text-xs text-text-muted uppercase mb-2">Last Signal</p>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span
                      className={`badge ${
                        strategy.lastSignal.action === "BUY"
                          ? "badge-emerald"
                          : strategy.lastSignal.action === "SELL"
                          ? "badge-red"
                          : "badge-neutral"
                      }`}
                    >
                      {strategy.lastSignal.action}
                    </span>
                    <span className="font-medium text-white mono">
                      {strategy.lastSignal.symbol}
                    </span>
                  </div>
                  <span className="text-xs text-text-muted">
                    {strategy.lastSignal.time}
                  </span>
                </div>
              </div>
            )}
            </div>
          </div>
        ))}
      </div>
    </PageWrapper>
  );
}
