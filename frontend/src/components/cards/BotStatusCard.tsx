"use client";

import { useState } from "react";
import { Play, Square, Clock, AlertTriangle, Loader2 } from "lucide-react";
import { usePolling } from "@/hooks/usePolling";
import { getBotStatus, startBot, stopBot, BotStatus } from "@/lib/api";

export function BotStatusCard() {
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
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
    setIsStopping(true);
    try {
      await stopBot();
      await refetch();
    } catch (err) {
      console.error("Failed to stop bot:", err);
    } finally {
      setIsStopping(false);
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
              disabled={isStopping}
              className="btn btn-danger btn-ripple flex-1 text-sm py-2"
            >
              {isStopping ? <Loader2 className="w-4 h-4 animate-spin" /> : <Square className="w-4 h-4" />}
              {isStopping ? "Stopping..." : "Stop"}
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
