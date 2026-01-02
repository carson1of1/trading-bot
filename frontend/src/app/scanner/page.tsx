"use client";

import { useState } from "react";
import { PageWrapper } from "@/components/layout/PageWrapper";
import { ScanLine, RefreshCw, Clock } from "lucide-react";

const mockScannerData = [
  {
    rank: 1,
    symbol: "NVDA",
    price: 495.32,
    atrPercent: 3.45,
    dailyRangePercent: 4.12,
    volumeRatio: 2.8,
    compositeScore: 92,
  },
  {
    rank: 2,
    symbol: "TSLA",
    price: 248.65,
    atrPercent: 4.21,
    dailyRangePercent: 3.85,
    volumeRatio: 2.1,
    compositeScore: 88,
  },
  {
    rank: 3,
    symbol: "AMD",
    price: 142.8,
    atrPercent: 3.12,
    dailyRangePercent: 3.45,
    volumeRatio: 1.9,
    compositeScore: 82,
  },
  {
    rank: 4,
    symbol: "META",
    price: 525.42,
    atrPercent: 2.85,
    dailyRangePercent: 2.95,
    volumeRatio: 1.7,
    compositeScore: 76,
  },
  {
    rank: 5,
    symbol: "AAPL",
    price: 194.27,
    atrPercent: 1.82,
    dailyRangePercent: 2.15,
    volumeRatio: 1.5,
    compositeScore: 68,
  },
  {
    rank: 6,
    symbol: "GOOGL",
    price: 141.85,
    atrPercent: 2.15,
    dailyRangePercent: 2.35,
    volumeRatio: 1.4,
    compositeScore: 65,
  },
  {
    rank: 7,
    symbol: "MSFT",
    price: 383.92,
    atrPercent: 1.65,
    dailyRangePercent: 1.82,
    volumeRatio: 1.3,
    compositeScore: 58,
  },
  {
    rank: 8,
    symbol: "AMZN",
    price: 185.67,
    atrPercent: 2.08,
    dailyRangePercent: 2.25,
    volumeRatio: 1.2,
    compositeScore: 55,
  },
  {
    rank: 9,
    symbol: "SPY",
    price: 477.42,
    atrPercent: 0.95,
    dailyRangePercent: 1.15,
    volumeRatio: 1.1,
    compositeScore: 42,
  },
  {
    rank: 10,
    symbol: "QQQ",
    price: 412.35,
    atrPercent: 1.12,
    dailyRangePercent: 1.28,
    volumeRatio: 1.0,
    compositeScore: 38,
  },
];

export default function ScannerPage() {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const lastRefresh = "2 min ago";

  const handleRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => setIsRefreshing(false), 1000);
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-emerald";
    if (score >= 60) return "text-amber";
    return "text-text-secondary";
  };

  return (
    <PageWrapper title="Volatility Scanner" subtitle="Find high-volatility trading opportunities">
      <div className="glass p-5 opacity-0 animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-emerald-glow">
              <ScanLine className="w-5 h-5 text-emerald" />
            </div>
            <div className="flex items-center gap-2 text-sm text-text-muted">
              <Clock className="w-4 h-4" />
              Last refresh: {lastRefresh}
            </div>
          </div>
          <button
            onClick={handleRefresh}
            className="btn btn-secondary"
            disabled={isRefreshing}
          >
            <RefreshCw
              className={`w-4 h-4 ${isRefreshing ? "animate-spin" : ""}`}
            />
            Refresh
          </button>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Symbol</th>
                <th>Price</th>
                <th>ATR %</th>
                <th>Daily Range %</th>
                <th>Vol vs Avg</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {mockScannerData.map((stock) => (
                <tr key={stock.symbol} className="cursor-pointer">
                  <td>
                    <span
                      className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${
                        stock.rank <= 3
                          ? "bg-emerald-glow text-emerald"
                          : "bg-surface-2 text-text-muted"
                      }`}
                    >
                      {stock.rank}
                    </span>
                  </td>
                  <td>
                    <span className="font-semibold text-white mono">
                      {stock.symbol}
                    </span>
                  </td>
                  <td className="mono text-white">
                    ${stock.price.toFixed(2)}
                  </td>
                  <td className="mono text-amber">
                    {stock.atrPercent.toFixed(2)}%
                  </td>
                  <td className="mono text-text-secondary">
                    {stock.dailyRangePercent.toFixed(2)}%
                  </td>
                  <td className="mono">
                    <span
                      className={
                        stock.volumeRatio >= 1.5 ? "text-emerald" : "text-text-secondary"
                      }
                    >
                      {stock.volumeRatio.toFixed(1)}x
                    </span>
                  </td>
                  <td>
                    <span
                      className={`mono font-bold ${getScoreColor(
                        stock.compositeScore
                      )}`}
                    >
                      {stock.compositeScore}
                    </span>
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
