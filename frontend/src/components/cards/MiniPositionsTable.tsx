"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";

interface Position {
  symbol: string;
  side: "LONG" | "SHORT";
  pnlPercent: number;
}

interface MiniPositionsTableProps {
  positions: Position[];
}

export function MiniPositionsTable({ positions }: MiniPositionsTableProps) {
  const displayPositions = positions.slice(0, 5);

  return (
    <div className="glass glass-hover p-5 opacity-0 animate-slide-up stagger-5">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-text-secondary">
          Open Positions
        </h3>
        <Link
          href="/positions"
          className="flex items-center gap-1 text-xs text-emerald hover:text-emerald/80 transition-colors"
        >
          View all
          <ArrowRight className="w-3 h-3" />
        </Link>
      </div>

      {/* Table */}
      {displayPositions.length > 0 ? (
        <table className="data-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Side</th>
              <th className="text-right">P&L %</th>
            </tr>
          </thead>
          <tbody>
            {displayPositions.map((position) => (
              <tr key={position.symbol}>
                <td>
                  <span className="font-medium text-white mono">
                    {position.symbol}
                  </span>
                </td>
                <td>
                  <span
                    className={`badge ${
                      position.side === "LONG" ? "badge-emerald" : "badge-red"
                    }`}
                  >
                    {position.side}
                  </span>
                </td>
                <td className="text-right">
                  <span
                    className={`mono font-medium ${
                      position.pnlPercent >= 0 ? "pnl-positive" : "pnl-negative"
                    }`}
                  >
                    {position.pnlPercent >= 0 ? "+" : ""}
                    {position.pnlPercent.toFixed(2)}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="text-center py-8">
          <p className="text-text-muted text-sm">No open positions</p>
        </div>
      )}
    </div>
  );
}
