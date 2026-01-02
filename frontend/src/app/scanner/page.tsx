"use client";

import { useState, useCallback } from "react";
import { PageWrapper } from "@/components/layout/PageWrapper";
import { ScanLine, RefreshCw, Clock, AlertTriangle } from "lucide-react";
import { runScanner, ScannerData } from "@/lib/api";

export default function ScannerPage() {
  const [scannerData, setScannerData] = useState<ScannerData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleScan = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await runScanner(10);
      setScannerData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run scanner");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getScoreColor = (score: number) => {
    if (score >= 0.008) return "text-emerald";
    if (score >= 0.004) return "text-amber";
    return "text-text-secondary";
  };

  const formatTime = (isoTime: string) => {
    const date = new Date(isoTime);
    return date.toLocaleTimeString();
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
            {scannerData && (
              <div className="flex items-center gap-2 text-sm text-text-muted">
                <Clock className="w-4 h-4" />
                Last scan: {formatTime(scannerData.scanned_at)}
              </div>
            )}
          </div>
          <button
            onClick={handleScan}
            className="btn btn-primary"
            disabled={isLoading}
          >
            <RefreshCw
              className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`}
            />
            {isLoading ? "Scanning..." : scannerData ? "Rescan" : "Run Scanner"}
          </button>
        </div>

        {/* Error state */}
        {error && (
          <div className="flex items-center gap-2 p-4 rounded-lg bg-red-glow border border-red/30 mb-6">
            <AlertTriangle className="w-5 h-5 text-red" />
            <span className="text-red">{error}</span>
          </div>
        )}

        {/* Initial state - no scan run yet */}
        {!scannerData && !isLoading && !error && (
          <div className="text-center py-12 text-text-muted">
            <ScanLine className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>Click &quot;Run Scanner&quot; to scan for volatile stocks</p>
          </div>
        )}

        {/* Loading state */}
        {isLoading && !scannerData && (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald"></div>
          </div>
        )}

        {/* Results table */}
        {scannerData && scannerData.results.length > 0 && (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Symbol</th>
                  <th>Price</th>
                  <th>ATR Ratio</th>
                  <th>Volume Ratio</th>
                  <th>Composite Score</th>
                </tr>
              </thead>
              <tbody>
                {scannerData.results.map((stock, index) => (
                  <tr key={stock.symbol} className="cursor-pointer">
                    <td>
                      <span
                        className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${
                          index < 3
                            ? "bg-emerald-glow text-emerald"
                            : "bg-surface-2 text-text-muted"
                        }`}
                      >
                        {index + 1}
                      </span>
                    </td>
                    <td>
                      <span className="font-semibold text-white mono">
                        {stock.symbol}
                      </span>
                    </td>
                    <td className="mono text-white">
                      ${stock.current_price.toFixed(2)}
                    </td>
                    <td className="mono text-amber">
                      {(stock.atr_ratio * 100).toFixed(2)}%
                    </td>
                    <td className="mono">
                      <span
                        className={
                          stock.volume_ratio >= 1.5 ? "text-emerald" : "text-text-secondary"
                        }
                      >
                        {stock.volume_ratio.toFixed(1)}x
                      </span>
                    </td>
                    <td>
                      <span
                        className={`mono font-bold ${getScoreColor(stock.composite_score)}`}
                      >
                        {(stock.composite_score * 100).toFixed(2)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Empty results */}
        {scannerData && scannerData.results.length === 0 && (
          <div className="text-center py-12 text-text-muted">
            No results found
          </div>
        )}
      </div>
    </PageWrapper>
  );
}
