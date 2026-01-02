"use client";

import { useState } from "react";
import { Play, Square, Clock } from "lucide-react";

type BotStatus = "running" | "paused" | "error";
type BotMode = "PAPER" | "LIVE" | "DRY_RUN";

interface BotState {
  status: BotStatus;
  mode: BotMode;
  lastAction: string;
  lastActionTime: string;
}

// Mock data - will be replaced with API
const mockBotState: BotState = {
  status: "running",
  mode: "PAPER",
  lastAction: "Bought 10 AAPL @ $189.45",
  lastActionTime: "2 min ago",
};

export function BotStatusCard() {
  const [botState] = useState<BotState>(mockBotState);

  const statusConfig = {
    running: {
      color: "text-emerald",
      bgColor: "bg-emerald",
      label: "Running",
    },
    paused: {
      color: "text-amber",
      bgColor: "bg-amber",
      label: "Paused",
    },
    error: {
      color: "text-red",
      bgColor: "bg-red",
      label: "Error",
    },
  };

  const modeConfig = {
    PAPER: { className: "badge-emerald" },
    LIVE: { className: "badge-red" },
    DRY_RUN: { className: "badge-neutral" },
  };

  const config = statusConfig[botState.status];

  return (
    <div className="glass glass-hover p-5 opacity-0 animate-slide-up stagger-3 h-full">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-sm font-medium text-text-secondary">Bot Status</h3>
          <span className={`badge ${modeConfig[botState.mode].className}`}>
            {botState.mode}
          </span>
        </div>

        {/* Status indicator */}
        <div className="flex items-center gap-3 mb-6">
          <div className={`relative w-3 h-3 rounded-full ${config.bgColor} pulse-dot`} />
          <span className={`text-lg font-semibold ${config.color}`}>
            {config.label}
          </span>
        </div>

        {/* Last action */}
        <div className="flex-1">
          <div className="flex items-start gap-2 text-sm">
            <Clock className="w-4 h-4 text-text-muted mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-text-secondary">{botState.lastAction}</p>
              <p className="text-text-muted text-xs mt-1">{botState.lastActionTime}</p>
            </div>
          </div>
        </div>

        {/* Control buttons */}
        <div className="flex gap-2 mt-4 pt-4 border-t border-border">
          {botState.status === "running" ? (
            <button className="btn btn-danger flex-1 text-sm py-2">
              <Square className="w-4 h-4" />
              Stop
            </button>
          ) : (
            <button className="btn btn-primary flex-1 text-sm py-2">
              <Play className="w-4 h-4" />
              Start
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
