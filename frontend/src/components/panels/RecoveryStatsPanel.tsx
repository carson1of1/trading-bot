"use client";

import { useEffect, useState, useCallback } from "react";
import { X, TrendingDown, AlertTriangle, Target, Clock, BarChart3, Zap } from "lucide-react";
import { getRecoveryStats, RecoveryStats, Position } from "@/lib/api";

interface RecoveryStatsPanelProps {
  position: Position;
  isOpen: boolean;
  onClose: () => void;
}

export function RecoveryStatsPanel({ position, isOpen, onClose }: RecoveryStatsPanelProps) {
  const [stats, setStats] = useState<RecoveryStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = useCallback(async (symbol: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getRecoveryStats(symbol);
      setStats(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch stats");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen && position) {
      fetchStats(position.symbol);
    }
  }, [isOpen, position, fetchStats]);

  if (!isOpen) return null;

  const getRecommendationColor = (rec: string) => {
    switch (rec) {
      case "CUT":
        return "text-red";
      case "HOLD_TO_BE":
        return "text-amber-400";
      case "HOLD_FOR_TP":
        return "text-emerald";
      default:
        return "text-text-secondary";
    }
  };

  const getRecommendationLabel = (rec: string) => {
    switch (rec) {
      case "CUT":
        return "Cut Losses";
      case "HOLD_TO_BE":
        return "Hold to Breakeven";
      case "HOLD_FOR_TP":
        return "Hold for Take Profit";
      default:
        return rec;
    }
  };

  const getEVColor = (ev: number) => {
    if (ev > 0.5) return "text-emerald";
    if (ev > 0) return "text-amber-400";
    return "text-red";
  };

  const getBestEV = (stats: RecoveryStats) => {
    const evs = [
      { action: "CUT", ev: stats.ev_cut_now },
      { action: "HOLD_TO_BE", ev: stats.ev_hold_to_breakeven },
      { action: "HOLD_FOR_TP", ev: stats.ev_hold_for_tp },
    ];
    return evs.reduce((best, current) => (current.ev > best.ev ? current : best));
  };

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40 transition-opacity"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="fixed right-0 top-0 h-full w-[420px] bg-[#0a0a0a] border-l border-border z-50 overflow-y-auto animate-panel-slide-in">
        {/* Header */}
        <div className="sticky top-0 bg-surface border-b border-border p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-emerald-glow">
              <BarChart3 className="w-5 h-5 text-emerald" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">{position.symbol}</h2>
              <p className="text-sm text-text-muted">Recovery Analysis</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-text-muted" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {isLoading && (
            <div className="flex items-center justify-center h-48">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald" />
            </div>
          )}

          {error && (
            <div className="bg-red/10 border border-red/20 rounded-lg p-4 text-red">
              <AlertTriangle className="w-5 h-5 inline mr-2" />
              {error}
            </div>
          )}

          {stats && (
            <>
              {/* Current Position Status */}
              <div className="glass p-4 rounded-lg">
                <h3 className="text-sm font-medium text-text-muted mb-3">Current Position</h3>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <p className="text-xs text-text-muted">P&L</p>
                    <p className={`text-lg font-semibold mono ${stats.current_drawdown_pct >= 0 ? "text-emerald" : "text-red"}`}>
                      {stats.current_drawdown_pct >= 0 ? "+" : ""}{stats.current_drawdown_pct.toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-text-muted">Distance to Stop</p>
                    <p className="text-lg font-semibold mono text-amber-400">
                      {stats.distance_to_stop_pct.toFixed(2)}%
                    </p>
                  </div>
                </div>
              </div>

              {/* Recommendation Box */}
              <div className={`border-2 rounded-lg p-4 ${
                stats.recommendation === "CUT" ? "border-red bg-red/10" :
                stats.recommendation === "HOLD_TO_BE" ? "border-amber-400 bg-amber-400/10" :
                "border-emerald bg-emerald/10"
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  <Zap className={`w-5 h-5 ${getRecommendationColor(stats.recommendation)}`} />
                  <span className={`text-lg font-bold ${getRecommendationColor(stats.recommendation)}`}>
                    {getRecommendationLabel(stats.recommendation)}
                  </span>
                </div>
                <p className="text-sm text-text-secondary">{stats.recommendation_reason}</p>
                {stats.risk_warning && (
                  <div className="mt-2 flex items-center gap-2 text-amber-400 text-sm">
                    <AlertTriangle className="w-4 h-4" />
                    {stats.risk_warning}
                  </div>
                )}
              </div>

              {/* Probabilities */}
              <div className="glass p-4 rounded-lg">
                <h3 className="text-sm font-medium text-text-muted mb-3">Probabilities</h3>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-text-secondary">Recovery to Breakeven</span>
                      <span className="font-semibold text-emerald">{stats.recovery_probability.toFixed(0)}%</span>
                    </div>
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-emerald rounded-full transition-all"
                        style={{ width: `${stats.recovery_probability}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-text-secondary">Hit Take Profit (+5%)</span>
                      <span className="font-semibold text-blue-400">{stats.take_profit_probability.toFixed(0)}%</span>
                    </div>
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-400 rounded-full transition-all"
                        style={{ width: `${stats.take_profit_probability}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-text-secondary">Hit Stop Loss</span>
                      <span className="font-semibold text-red">{stats.stop_loss_probability.toFixed(0)}%</span>
                    </div>
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-red rounded-full transition-all"
                        style={{ width: `${stats.stop_loss_probability}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Expected Value Analysis */}
              <div className="glass p-4 rounded-lg">
                <h3 className="text-sm font-medium text-text-muted mb-3">Expected Value Analysis</h3>
                <div className="space-y-2">
                  {[
                    { label: "Cut Now", ev: stats.ev_cut_now, action: "CUT" },
                    { label: "Hold to Breakeven", ev: stats.ev_hold_to_breakeven, action: "HOLD_TO_BE" },
                    { label: "Hold for TP", ev: stats.ev_hold_for_tp, action: "HOLD_FOR_TP" },
                  ].map((item) => {
                    const isBest = getBestEV(stats).action === item.action;
                    return (
                      <div
                        key={item.action}
                        className={`flex items-center justify-between p-2 rounded ${isBest ? "bg-emerald/20 border border-emerald/30" : ""}`}
                      >
                        <div className="flex items-center gap-2">
                          {isBest && <Target className="w-4 h-4 text-emerald" />}
                          <span className={`text-sm ${isBest ? "text-white font-medium" : "text-text-secondary"}`}>
                            {item.label}
                          </span>
                        </div>
                        <span className={`mono font-semibold ${getEVColor(item.ev)}`}>
                          {item.ev >= 0 ? "+" : ""}{item.ev.toFixed(2)}% EV
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Historical Context */}
              <div className="glass p-4 rounded-lg">
                <h3 className="text-sm font-medium text-text-muted mb-3">Historical Context</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-center gap-2">
                    <TrendingDown className="w-4 h-4 text-red" />
                    <div>
                      <p className="text-xs text-text-muted">Avg Max Drawdown</p>
                      <p className="text-sm font-semibold text-red">{stats.avg_max_drawdown.toFixed(1)}%</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-blue-400" />
                    <div>
                      <p className="text-xs text-text-muted">Avg Time to Recovery</p>
                      <p className="text-sm font-semibold text-blue-400">{stats.avg_bars_to_recovery.toFixed(0)} bars</p>
                    </div>
                  </div>
                </div>

                {stats.is_gap_entry && (
                  <div className="mt-3 p-2 bg-amber-400/10 border border-amber-400/20 rounded">
                    <div className="flex items-center gap-2 text-amber-400 text-sm">
                      <AlertTriangle className="w-4 h-4" />
                      <span>Gap-up entry detected</span>
                    </div>
                    <div className="text-xs text-text-muted mt-1">
                      Gap: {stats.gap_percentage?.toFixed(1)}% | Historical win rate: {stats.gap_win_rate?.toFixed(0)}%
                    </div>
                  </div>
                )}
              </div>

              {/* Disclaimer */}
              <p className="text-xs text-text-muted text-center px-4">
                Based on 1-year historical data for {position.symbol}. Past performance does not guarantee future results.
              </p>
            </>
          )}
        </div>
      </div>
    </>
  );
}
