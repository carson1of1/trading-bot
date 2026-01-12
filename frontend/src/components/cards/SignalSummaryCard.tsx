"use client";

import { Radio, TrendingUp, TrendingDown, Minus, Clock, Zap } from "lucide-react";
import { usePolling } from "@/hooks/usePolling";
import { getLatestSignals, LatestSignalsData } from "@/lib/api";

export function SignalSummaryCard() {
  const { data: signalsData, isLoading, error } = usePolling<LatestSignalsData>({
    fetcher: getLatestSignals,
    interval: 10000, // Refresh every 10 seconds
  });

  // Format timestamp - show actual candle time (e.g., "10:00")
  const formatTime = (isoTime: string | null) => {
    if (!isoTime) return "No data";
    const date = new Date(isoTime);
    // Show the actual candle time in HH:MM format
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  };

  // Loading state
  if (isLoading && !signalsData) {
    return (
      <div className="glass-gradient p-5 h-full animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="h-20 bg-gray-700 rounded mb-4"></div>
        <div className="space-y-2">
          <div className="h-4 bg-gray-700 rounded w-full"></div>
          <div className="h-4 bg-gray-700 rounded w-3/4"></div>
        </div>
      </div>
    );
  }

  // Error or no data state
  if (error || !signalsData?.has_data) {
    return (
      <div className="glass-gradient p-5 h-full opacity-0 animate-slide-up stagger-5">
        <div className="flex items-center gap-2 mb-4">
          <div className="p-2 rounded-lg bg-surface-2">
            <Radio className="w-4 h-4 text-text-muted" />
          </div>
          <h3 className="text-sm font-medium text-text-secondary">Signal Scanner</h3>
        </div>
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <div className="w-12 h-12 rounded-full bg-surface-2 flex items-center justify-center mb-3">
            <Radio className="w-6 h-6 text-text-muted" />
          </div>
          <p className="text-text-muted text-sm">Waiting for signals</p>
          <p className="text-text-muted text-xs mt-1">Start the bot to scan</p>
        </div>
      </div>
    );
  }

  const { summary, top_signals, timestamp } = signalsData;
  const total = summary?.total_scanned || 0;
  const buyCount = summary?.buy_signals || 0;
  const sellCount = summary?.sell_signals || 0;
  const holdCount = summary?.hold_signals || 0;

  // Calculate percentages for the bar
  const buyPct = total > 0 ? (buyCount / total) * 100 : 0;
  const sellPct = total > 0 ? (sellCount / total) * 100 : 0;
  const holdPct = total > 0 ? (holdCount / total) * 100 : 0;

  return (
    <div className="glass-gradient p-5 h-full opacity-0 animate-slide-up stagger-5">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-emerald-glow">
            <Radio className="w-4 h-4 text-emerald" />
          </div>
          <h3 className="text-sm font-medium text-text-secondary">Signal Scanner</h3>
        </div>
        <div className="flex items-center gap-1.5 text-text-muted text-xs">
          <Clock className="w-3 h-3" />
          <span>{formatTime(timestamp)}</span>
        </div>
      </div>

      {/* Signal Distribution Bar */}
      <div className="mb-5">
        <div className="flex items-center justify-between text-xs text-text-muted mb-2">
          <span>Scanned {total} symbols</span>
          <span className="mono">{summary?.above_threshold || 0} actionable</span>
        </div>

        {/* Segmented bar */}
        <div className="relative h-3 rounded-full overflow-hidden bg-surface-2">
          <div className="absolute inset-0 flex">
            {buyPct > 0 && (
              <div
                className="h-full bg-emerald transition-all duration-500"
                style={{ width: `${buyPct}%` }}
              />
            )}
            {sellPct > 0 && (
              <div
                className="h-full bg-red transition-all duration-500"
                style={{ width: `${sellPct}%` }}
              />
            )}
            {holdPct > 0 && (
              <div
                className="h-full bg-text-muted/30 transition-all duration-500"
                style={{ width: `${holdPct}%` }}
              />
            )}
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 mt-2.5 text-xs">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-emerald" />
            <span className="text-emerald font-medium">{buyCount} BUY</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-red" />
            <span className="text-red font-medium">{sellCount} SELL</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-text-muted/50" />
            <span className="text-text-muted">{holdCount} HOLD</span>
          </div>
        </div>
      </div>

      {/* Execution Stats */}
      <div className="grid grid-cols-2 gap-3 mb-5">
        <div className="p-3 rounded-lg bg-surface-1 border border-border">
          <div className="flex items-center gap-2 mb-1">
            <Zap className="w-3.5 h-3.5 text-emerald" />
            <span className="text-xs text-text-muted">Executed</span>
          </div>
          <span className="text-lg font-semibold mono text-emerald">
            {summary?.executed || 0}
          </span>
        </div>
        <div className="p-3 rounded-lg bg-surface-1 border border-border">
          <div className="flex items-center gap-2 mb-1">
            <Minus className="w-3.5 h-3.5 text-amber" />
            <span className="text-xs text-text-muted">Blocked</span>
          </div>
          <span className="text-lg font-semibold mono text-amber">
            {summary?.blocked || 0}
          </span>
        </div>
      </div>

      {/* Top Signals */}
      {top_signals && top_signals.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
            Top Signals
          </h4>
          <div className="space-y-2 max-h-[140px] overflow-y-auto">
            {top_signals.slice(0, 5).map((signal, idx) => (
              <div
                key={`${signal.symbol}-${idx}`}
                className="flex items-center gap-3 p-2.5 rounded-lg bg-surface-1 border border-border hover:bg-surface-2 transition-colors"
              >
                {/* Direction Icon */}
                <div className={`p-1.5 rounded-md ${
                  signal.action === "BUY"
                    ? "bg-emerald-glow text-emerald"
                    : signal.action === "SELL"
                    ? "bg-red-glow text-red"
                    : "bg-surface-2 text-text-muted"
                }`}>
                  {signal.action === "BUY" ? (
                    <TrendingUp className="w-3.5 h-3.5" />
                  ) : signal.action === "SELL" ? (
                    <TrendingDown className="w-3.5 h-3.5" />
                  ) : (
                    <Minus className="w-3.5 h-3.5" />
                  )}
                </div>

                {/* Symbol & Strategy */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-sm text-white">
                      {signal.symbol}
                    </span>
                    <span className={`text-xs px-1.5 py-0.5 rounded ${
                      signal.direction === "LONG"
                        ? "bg-emerald-glow text-emerald"
                        : "bg-red-glow text-red"
                    }`}>
                      {signal.direction}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="text-xs text-text-muted truncate">
                      {signal.strategy}
                    </span>
                    <span className="text-xs text-text-muted mono">
                      ${signal.price.toFixed(2)}
                    </span>
                  </div>
                </div>

                {/* Confidence Meter */}
                <div className="flex flex-col items-end gap-1">
                  <span className={`text-xs font-semibold mono ${
                    signal.confidence >= 70 ? "text-emerald" :
                    signal.confidence >= 50 ? "text-amber" : "text-text-muted"
                  }`}>
                    {signal.confidence.toFixed(0)}%
                  </span>
                  <div className="w-12 h-1.5 rounded-full bg-surface-2 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${
                        signal.confidence >= 70 ? "bg-emerald" :
                        signal.confidence >= 50 ? "bg-amber" : "bg-text-muted"
                      }`}
                      style={{ width: `${Math.min(signal.confidence, 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No signals state */}
      {(!top_signals || top_signals.length === 0) && (
        <div className="text-center py-4">
          <p className="text-text-muted text-sm">No actionable signals</p>
          <p className="text-text-muted text-xs mt-1">All symbols below threshold</p>
        </div>
      )}
    </div>
  );
}
