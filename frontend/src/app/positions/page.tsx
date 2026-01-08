"use client";

import { useState } from "react";
import { PageWrapper } from "@/components/layout/PageWrapper";
import { usePolling } from "@/hooks/usePolling";
import { getPositions, PositionsData, Position } from "@/lib/api";
import { Briefcase, TrendingUp, TrendingDown, AlertTriangle, BarChart3 } from "lucide-react";
import { RecoveryStatsPanel } from "@/components/panels/RecoveryStatsPanel";

export default function PositionsPage() {
  const { data: positionsData, isLoading, error } = usePolling<PositionsData>({
    fetcher: getPositions,
    interval: 3000,
  });

  const [selectedPosition, setSelectedPosition] = useState<Position | null>(null);
  const [isPanelOpen, setIsPanelOpen] = useState(false);

  const handlePositionClick = (position: Position) => {
    setSelectedPosition(position);
    setIsPanelOpen(true);
  };

  const handleClosePanel = () => {
    setIsPanelOpen(false);
  };

  if (isLoading && !positionsData) {
    return (
      <PageWrapper title="Positions" subtitle="Manage your open positions">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald"></div>
        </div>
      </PageWrapper>
    );
  }

  if (error) {
    return (
      <PageWrapper title="Positions" subtitle="Manage your open positions">
        <div className="flex flex-col items-center justify-center h-64 text-red">
          <AlertTriangle className="w-12 h-12 mb-4" />
          <p>Failed to load positions</p>
        </div>
      </PageWrapper>
    );
  }

  const positions = positionsData?.positions || [];

  return (
    <PageWrapper title="Positions" subtitle="Manage your open positions">
      <div className="glass p-5 opacity-0 animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-emerald-glow">
              <Briefcase className="w-5 h-5 text-emerald" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">Open Positions</h2>
              <p className="text-sm text-text-muted">
                {positions.length} active position
                {positions.length !== 1 ? "s" : ""}
              </p>
            </div>
          </div>
          <div className="text-sm text-text-muted">
            Total Unrealized P&L:{" "}
            <span className={positionsData?.total_unrealized_pl && positionsData.total_unrealized_pl >= 0 ? "text-emerald" : "text-red"}>
              ${positionsData?.total_unrealized_pl?.toFixed(2) || "0.00"}
            </span>
          </div>
        </div>

        {positions.length === 0 ? (
          <div className="text-center py-12 text-text-muted">
            No open positions
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Side</th>
                  <th>Qty</th>
                  <th>Entry Price</th>
                  <th>Current Price</th>
                  <th>Market Value</th>
                  <th>P&L $</th>
                  <th>P&L %</th>
                  <th>Analyze</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((position) => {
                  const isProfit = position.unrealized_pl >= 0;
                  return (
                    <tr
                      key={position.symbol}
                      onClick={() => handlePositionClick(position)}
                      className={`border-l-2 ${isProfit ? "border-l-emerald" : "border-l-red"} cursor-pointer hover:bg-surface-2`}
                    >
                      <td>
                        <span className="font-semibold text-white mono">
                          {position.symbol}
                        </span>
                      </td>
                      <td>
                        <span className={`badge ${position.side === "long" ? "badge-emerald" : "badge-red"}`}>
                          {position.side.toUpperCase()}
                        </span>
                      </td>
                      <td className="mono text-text-secondary">{position.qty}</td>
                      <td className="mono text-text-secondary">
                        ${position.avg_entry_price.toFixed(2)}
                      </td>
                      <td className="mono text-white">
                        ${position.current_price.toFixed(2)}
                      </td>
                      <td className="mono text-text-secondary">
                        ${position.market_value.toFixed(2)}
                      </td>
                      <td className={`mono font-medium ${isProfit ? "pnl-positive" : "pnl-negative"}`}>
                        <div className="flex items-center gap-1">
                          {isProfit ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                          {isProfit ? "+" : ""}${position.unrealized_pl.toFixed(2)}
                        </div>
                      </td>
                      <td className={`mono font-medium ${isProfit ? "pnl-positive" : "pnl-negative"}`}>
                        {isProfit ? "+" : ""}{position.unrealized_plpc.toFixed(2)}%
                      </td>
                      <td>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handlePositionClick(position);
                          }}
                          className="flex items-center gap-1 px-2 py-1 text-xs bg-blue-glow text-blue-light border border-blue-strong rounded hover:bg-blue/20 transition-colors"
                        >
                          <BarChart3 className="w-3 h-3" />
                          Stats
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Recovery Stats Panel */}
      {selectedPosition && (
        <RecoveryStatsPanel
          position={selectedPosition}
          isOpen={isPanelOpen}
          onClose={handleClosePanel}
        />
      )}
    </PageWrapper>
  );
}
