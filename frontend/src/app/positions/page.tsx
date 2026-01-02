"use client";

import { PageWrapper } from "@/components/layout/PageWrapper";
import { usePolling } from "@/hooks/usePolling";
import { getPositions, PositionsData } from "@/lib/api";
import { Briefcase, TrendingUp, TrendingDown, AlertTriangle } from "lucide-react";

export default function PositionsPage() {
  const { data: positionsData, isLoading, error } = usePolling<PositionsData>({
    fetcher: getPositions,
    interval: 3000,
  });

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
                </tr>
              </thead>
              <tbody>
                {positions.map((position) => {
                  const isProfit = position.unrealized_pl >= 0;
                  return (
                    <tr
                      key={position.symbol}
                      className={`border-l-2 ${isProfit ? "border-l-emerald" : "border-l-red"}`}
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
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </PageWrapper>
  );
}
