"use client";

import { useState } from "react";
import { PageWrapper } from "@/components/layout/PageWrapper";
import { History, Download, Search } from "lucide-react";

const mockTrades = [
  {
    id: 1,
    date: "2024-12-30 14:32",
    symbol: "AAPL",
    side: "LONG",
    entryPrice: 185.2,
    exitPrice: 189.45,
    pnlDollar: 425.0,
    pnlPercent: 2.29,
    holdDuration: "2d 4h",
    strategy: "Momentum",
  },
  {
    id: 2,
    date: "2024-12-29 10:15",
    symbol: "TSLA",
    side: "SHORT",
    entryPrice: 252.8,
    exitPrice: 248.3,
    pnlDollar: 450.0,
    pnlPercent: 1.78,
    holdDuration: "1d 6h",
    strategy: "Mean Reversion",
  },
  {
    id: 3,
    date: "2024-12-28 09:30",
    symbol: "NVDA",
    side: "LONG",
    entryPrice: 485.0,
    exitPrice: 478.5,
    pnlDollar: -325.0,
    pnlPercent: -1.34,
    holdDuration: "4h",
    strategy: "Breakout",
  },
  {
    id: 4,
    date: "2024-12-27 11:45",
    symbol: "META",
    side: "LONG",
    entryPrice: 510.2,
    exitPrice: 525.8,
    pnlDollar: 780.0,
    pnlPercent: 3.06,
    holdDuration: "3d 2h",
    strategy: "Momentum",
  },
  {
    id: 5,
    date: "2024-12-26 15:20",
    symbol: "GOOGL",
    side: "SHORT",
    entryPrice: 142.5,
    exitPrice: 145.2,
    pnlDollar: -270.0,
    pnlPercent: -1.89,
    holdDuration: "8h",
    strategy: "Mean Reversion",
  },
];

export default function TradeHistoryPage() {
  const [searchTerm, setSearchTerm] = useState("");
  const [sideFilter, setSideFilter] = useState<"all" | "long" | "short">("all");

  const filteredTrades = mockTrades.filter((trade) => {
    const matchesSearch = trade.symbol
      .toLowerCase()
      .includes(searchTerm.toLowerCase());
    const matchesSide =
      sideFilter === "all" || trade.side.toLowerCase() === sideFilter;
    return matchesSearch && matchesSide;
  });

  return (
    <PageWrapper title="Trade History" subtitle="View your past trades">
      <div className="glass p-5 opacity-0 animate-slide-up">
        {/* Filters */}
        <div className="flex flex-wrap items-center gap-4 mb-6">
          <div className="relative flex-1 min-w-[200px] max-w-[300px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
            <input
              type="text"
              placeholder="Search symbol..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="input !pl-10"
            />
          </div>

          <div className="flex gap-1 p-0.5 bg-surface-1 rounded-lg">
            {["all", "long", "short"].map((side) => (
              <button
                key={side}
                onClick={() => setSideFilter(side as typeof sideFilter)}
                className={`px-4 py-1.5 text-sm font-medium rounded-md capitalize transition-all ${
                  sideFilter === side
                    ? "bg-emerald text-black"
                    : "text-text-muted hover:text-white"
                }`}
              >
                {side}
              </button>
            ))}
          </div>

          <button className="btn btn-secondary ml-auto">
            <Download className="w-4 h-4" />
            Export CSV
          </button>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Date/Time</th>
                <th>Symbol</th>
                <th>Side</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>P&L $</th>
                <th>P&L %</th>
                <th>Duration</th>
                <th>Strategy</th>
              </tr>
            </thead>
            <tbody>
              {filteredTrades.map((trade) => (
                <tr key={trade.id}>
                  <td className="text-text-secondary text-sm">{trade.date}</td>
                  <td>
                    <span className="font-semibold text-white mono">
                      {trade.symbol}
                    </span>
                  </td>
                  <td>
                    <span
                      className={`badge ${
                        trade.side === "LONG" ? "badge-emerald" : "badge-red"
                      }`}
                    >
                      {trade.side}
                    </span>
                  </td>
                  <td className="mono text-text-secondary">
                    ${trade.entryPrice.toFixed(2)}
                  </td>
                  <td className="mono text-white">
                    ${trade.exitPrice.toFixed(2)}
                  </td>
                  <td
                    className={`mono font-medium ${
                      trade.pnlDollar >= 0 ? "pnl-positive" : "pnl-negative"
                    }`}
                  >
                    {trade.pnlDollar >= 0 ? "+" : ""}${trade.pnlDollar.toFixed(2)}
                  </td>
                  <td
                    className={`mono font-medium ${
                      trade.pnlPercent >= 0 ? "pnl-positive" : "pnl-negative"
                    }`}
                  >
                    {trade.pnlPercent >= 0 ? "+" : ""}
                    {trade.pnlPercent.toFixed(2)}%
                  </td>
                  <td className="text-text-secondary">{trade.holdDuration}</td>
                  <td>
                    <span className="badge badge-neutral">{trade.strategy}</span>
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
