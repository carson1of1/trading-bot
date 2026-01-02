"use client";

import { PageWrapper } from "@/components/layout/PageWrapper";
import { Briefcase } from "lucide-react";

const mockPositions = [
  {
    symbol: "AAPL",
    side: "LONG",
    entryPrice: 189.45,
    currentPrice: 194.12,
    pnlDollar: 467.0,
    pnlPercent: 2.45,
    stopLoss: 185.0,
    takeProfit: 200.0,
    holdTime: "2d 4h",
    strategy: "Momentum",
  },
  {
    symbol: "MSFT",
    side: "LONG",
    entryPrice: 378.9,
    currentPrice: 383.56,
    pnlDollar: 233.0,
    pnlPercent: 1.23,
    stopLoss: 370.0,
    takeProfit: 395.0,
    holdTime: "1d 2h",
    strategy: "Breakout",
  },
  {
    symbol: "NVDA",
    side: "SHORT",
    entryPrice: 495.0,
    currentPrice: 499.31,
    pnlDollar: -215.5,
    pnlPercent: -0.87,
    stopLoss: 510.0,
    takeProfit: 475.0,
    holdTime: "6h",
    strategy: "Mean Reversion",
  },
  {
    symbol: "SPY",
    side: "LONG",
    entryPrice: 475.2,
    currentPrice: 477.86,
    pnlDollar: 133.0,
    pnlPercent: 0.56,
    stopLoss: 470.0,
    takeProfit: 485.0,
    holdTime: "4h",
    strategy: "Momentum",
  },
];

export default function PositionsPage() {
  return (
    <PageWrapper title="Positions" subtitle="Manage your open positions">
      <div className="glass p-5 opacity-0 animate-slide-up">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-emerald-glow">
            <Briefcase className="w-5 h-5 text-emerald" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">Open Positions</h2>
            <p className="text-sm text-text-muted">
              {mockPositions.length} active position
              {mockPositions.length !== 1 ? "s" : ""}
            </p>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Side</th>
                <th>Entry Price</th>
                <th>Current Price</th>
                <th>P&L $</th>
                <th>P&L %</th>
                <th>Stop Loss</th>
                <th>Take Profit</th>
                <th>Hold Time</th>
                <th>Strategy</th>
              </tr>
            </thead>
            <tbody>
              {mockPositions.map((position) => (
                <tr
                  key={position.symbol}
                  className={`border-l-2 ${
                    position.pnlPercent >= 0
                      ? "border-l-emerald"
                      : "border-l-red"
                  }`}
                >
                  <td>
                    <span className="font-semibold text-white mono">
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
                  <td className="mono text-text-secondary">
                    ${position.entryPrice.toFixed(2)}
                  </td>
                  <td className="mono text-white">
                    ${position.currentPrice.toFixed(2)}
                  </td>
                  <td
                    className={`mono font-medium ${
                      position.pnlDollar >= 0 ? "pnl-positive" : "pnl-negative"
                    }`}
                  >
                    {position.pnlDollar >= 0 ? "+" : ""}$
                    {position.pnlDollar.toFixed(2)}
                  </td>
                  <td
                    className={`mono font-medium ${
                      position.pnlPercent >= 0 ? "pnl-positive" : "pnl-negative"
                    }`}
                  >
                    {position.pnlPercent >= 0 ? "+" : ""}
                    {position.pnlPercent.toFixed(2)}%
                  </td>
                  <td className="mono text-red">${position.stopLoss.toFixed(2)}</td>
                  <td className="mono text-emerald">
                    ${position.takeProfit.toFixed(2)}
                  </td>
                  <td className="text-text-secondary">{position.holdTime}</td>
                  <td>
                    <span className="badge badge-neutral">{position.strategy}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </PageWrapper>
  );
}
