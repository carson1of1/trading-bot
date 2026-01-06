"use client";

import { useState, useEffect } from "react";
import { PageWrapper } from "@/components/layout/PageWrapper";
import { Shield, AlertTriangle, TrendingDown, Briefcase, RefreshCw, AlertCircle } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell } from "recharts";
import { useMounted } from "@/lib/utils";
import { getRiskMetrics, RiskMetrics } from "@/lib/api";

const maxPositionSize = 20;

function RiskCard({
  title,
  icon,
  value,
  subtitle,
  status,
}: {
  title: string;
  icon: React.ReactNode;
  value: string;
  subtitle: string;
  status: "safe" | "warning" | "danger";
}) {
  const statusColors = {
    safe: "text-emerald",
    warning: "text-amber",
    danger: "text-red",
  };

  return (
    <div className="glass glass-hover p-5">
      <div className="flex items-start justify-between mb-4">
        <p className="text-sm text-text-secondary">{title}</p>
        <div className="p-2 rounded-lg bg-surface-2">{icon}</div>
      </div>
      <p className={`text-2xl font-bold mono ${statusColors[status]}`}>{value}</p>
      <p className="text-sm text-text-muted mt-1">{subtitle}</p>
    </div>
  );
}

function DrawdownGauge({ value }: { value: number }) {
  const normalizedValue = Math.min(value / 10, 1);
  const angle = normalizedValue * 180;

  const getZoneColor = () => {
    if (value <= 2) return "#10b981";
    if (value <= 5) return "#f59e0b";
    return "#ef4444";
  };

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-48 h-24 overflow-hidden">
        {/* Background arc */}
        <svg className="w-full h-full" viewBox="0 0 100 50">
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#10b981" />
              <stop offset="50%" stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>
          </defs>
          {/* Background track */}
          <path
            d="M 5 45 A 45 45 0 0 1 95 45"
            fill="none"
            stroke="rgba(255,255,255,0.1)"
            strokeWidth="8"
            strokeLinecap="round"
          />
          {/* Colored arc */}
          <path
            d="M 5 45 A 45 45 0 0 1 95 45"
            fill="none"
            stroke="url(#gaugeGradient)"
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${normalizedValue * 141.37} 141.37`}
          />
          {/* Needle */}
          <line
            x1="50"
            y1="45"
            x2={50 + 35 * Math.cos((Math.PI * (180 - angle)) / 180)}
            y2={45 - 35 * Math.sin((Math.PI * (180 - angle)) / 180)}
            stroke={getZoneColor()}
            strokeWidth="3"
            strokeLinecap="round"
          />
          <circle cx="50" cy="45" r="4" fill={getZoneColor()} />
        </svg>
      </div>
      <div className="text-center mt-2">
        <p className="text-2xl font-bold mono" style={{ color: getZoneColor() }}>
          {value.toFixed(1)}%
        </p>
        <p className="text-sm text-text-muted">Current Drawdown</p>
      </div>
      <div className="flex justify-between w-full mt-4 text-xs text-text-muted">
        <span>0%</span>
        <span>5%</span>
        <span>10%</span>
      </div>
    </div>
  );
}

export default function RiskMonitorPage() {
  const mounted = useMounted();
  const [riskData, setRiskData] = useState<RiskMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchRiskMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getRiskMetrics();
      setRiskData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch risk metrics");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRiskMetrics();
    // Auto-refresh every 60 seconds
    const interval = setInterval(fetchRiskMetrics, 60000);
    return () => clearInterval(interval);
  }, []);

  // Calculate percentages for status indicators
  const dailyLossPercent = riskData
    ? (riskData.daily_loss / riskData.daily_loss_limit) * 100
    : 0;
  const losingTradesPercent = riskData
    ? (riskData.losing_trades_today / riskData.losing_trades_limit) * 100
    : 0;

  return (
    <PageWrapper title="Risk Monitor" subtitle="Monitor portfolio risk and exposure">
      <div className="space-y-6">
        {/* Refresh Button */}
        <div className="flex justify-end">
          <button
            onClick={fetchRiskMetrics}
            disabled={loading}
            className="btn btn-secondary"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </button>
        </div>

        {/* Error State */}
        {error && (
          <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <span className="text-red-400">{error}</span>
          </div>
        )}

        {/* Loading State */}
        {loading && !riskData && (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-6 h-6 text-emerald animate-spin" />
            <span className="ml-3 text-text-secondary">Loading risk metrics...</span>
          </div>
        )}

        {/* Top Risk Cards */}
        {riskData && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 opacity-0 animate-slide-up stagger-1">
              <RiskCard
                title="Daily Loss"
                icon={<TrendingDown className="w-5 h-5 text-text-muted" />}
                value={`$${riskData.daily_loss.toFixed(0)}`}
                subtitle={`Limit: $${riskData.daily_loss_limit}`}
                status={dailyLossPercent < 50 ? "safe" : dailyLossPercent < 80 ? "warning" : "danger"}
              />
              <RiskCard
                title="Open Risk"
                icon={<Shield className="w-5 h-5 text-text-muted" />}
                value={`${riskData.open_risk.toFixed(1)}%`}
                subtitle="Portfolio exposure"
                status={riskData.open_risk < 5 ? "safe" : riskData.open_risk < 8 ? "warning" : "danger"}
              />
              <RiskCard
                title="Losing Trades Today"
                icon={<AlertTriangle className="w-5 h-5 text-text-muted" />}
                value={riskData.losing_trades_today.toString()}
                subtitle={`Lockout at ${riskData.losing_trades_limit}`}
                status={losingTradesPercent < 60 ? "safe" : losingTradesPercent < 80 ? "warning" : "danger"}
              />
              <RiskCard
                title="Largest Position"
                icon={<Briefcase className="w-5 h-5 text-text-muted" />}
                value={riskData.largest_position_symbol}
                subtitle={`${riskData.largest_position_percent.toFixed(1)}% of portfolio`}
                status={riskData.largest_position_percent < 15 ? "safe" : riskData.largest_position_percent < 25 ? "warning" : "danger"}
              />
            </div>

            {/* Position Sizing + Drawdown Gauge */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Position Sizing Visualizer */}
              <div className="glass p-5 opacity-0 animate-slide-up stagger-2">
                <h3 className="text-sm font-medium text-text-secondary mb-4">
                  Position Sizes (% of Portfolio)
                </h3>
                <div className="h-48">
                  {mounted && riskData.position_sizes.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={riskData.position_sizes} layout="vertical">
                        <XAxis
                          type="number"
                          domain={[0, 25]}
                          axisLine={false}
                          tickLine={false}
                          tick={{ fill: "#71717a", fontSize: 11 }}
                          tickFormatter={(v) => `${v}%`}
                        />
                        <YAxis
                          type="category"
                          dataKey="symbol"
                          axisLine={false}
                          tickLine={false}
                          tick={{ fill: "#ffffff", fontSize: 12, fontFamily: "Space Mono" }}
                          width={50}
                        />
                        <Bar dataKey="size" radius={[0, 4, 4, 0]} barSize={20}>
                          {riskData.position_sizes.map((entry, index) => (
                            <Cell
                              key={`cell-${index}`}
                              fill={
                                entry.size > maxPositionSize
                                  ? "#ef4444"
                                  : entry.size > maxPositionSize * 0.8
                                  ? "#f59e0b"
                                  : "#10b981"
                              }
                              fillOpacity={0.8}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  ) : mounted && riskData.position_sizes.length === 0 ? (
                    <div className="h-full flex items-center justify-center text-text-muted">
                      No open positions
                    </div>
                  ) : (
                    <div className="h-full flex items-center justify-center">
                      <div className="w-6 h-6 border-2 border-emerald border-t-transparent rounded-full animate-spin" />
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-2 mt-2 text-xs text-text-muted">
                  <div className="w-3 h-0.5 bg-red" />
                  <span>Max position limit: {maxPositionSize}%</span>
                </div>
              </div>

              {/* Drawdown Gauge */}
              <div className="glass p-5 opacity-0 animate-slide-up stagger-3">
                <h3 className="text-sm font-medium text-text-secondary mb-6 text-center">
                  Drawdown Gauge
                </h3>
                <DrawdownGauge value={riskData.current_drawdown} />
                <div className="flex justify-center gap-6 mt-6 text-xs">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-emerald" />
                    <span className="text-text-muted">Safe (0-2%)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-amber" />
                    <span className="text-text-muted">Warning (2-5%)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-red" />
                    <span className="text-text-muted">Danger (5%+)</span>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </PageWrapper>
  );
}
