"use client";

import { useState, useEffect } from "react";
import { PageWrapper } from "@/components/layout/PageWrapper";
import { Download, Search, RefreshCw, AlertCircle } from "lucide-react";
import { getTradeHistory, TradeHistoryItem } from "@/lib/api";

export default function TradeHistoryPage() {
  const [trades, setTrades] = useState<TradeHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [sideFilter, setSideFilter] = useState<"all" | "long" | "short">("all");

  const fetchTrades = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getTradeHistory(90); // Last 90 days
      setTrades(data.trades);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch trade history");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrades();
  }, []);

  const filteredTrades = trades.filter((trade) => {
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

          <button
            onClick={fetchTrades}
            disabled={loading}
            className="btn btn-secondary"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </button>

          <button className="btn btn-secondary ml-auto">
            <Download className="w-4 h-4" />
            Export CSV
          </button>
        </div>

        {/* Error State */}
        {error && (
          <div className="flex items-center gap-3 p-4 mb-6 bg-red-500/10 border border-red-500/20 rounded-lg">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <span className="text-red-400">{error}</span>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-6 h-6 text-emerald animate-spin" />
            <span className="ml-3 text-text-secondary">Loading trades...</span>
          </div>
        )}

        {/* Empty State */}
        {!loading && !error && filteredTrades.length === 0 && (
          <div className="text-center py-12 text-text-muted">
            {trades.length === 0
              ? "No trades found. Start trading to see your history here."
              : "No trades match your filters."}
          </div>
        )}

        {/* Table */}
        {!loading && filteredTrades.length > 0 && (
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
        )}
      </div>
    </PageWrapper>
  );
}
