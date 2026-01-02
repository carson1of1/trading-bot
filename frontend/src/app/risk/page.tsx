"use client";

import { PageWrapper } from "@/components/layout/PageWrapper";
import { Shield, AlertTriangle, TrendingDown, Briefcase } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell } from "recharts";
import { useMounted } from "@/lib/utils";

const mockRiskData = {
  dailyLoss: { current: 320, limit: 1000 },
  openRisk: 4.2,
  losingTradesToday: { count: 2, limit: 5 },
  largestPosition: { symbol: "AAPL", percent: 18.5 },
  currentDrawdown: 1.8,
};

const positionSizes = [
  { symbol: "AAPL", size: 18.5 },
  { symbol: "MSFT", size: 15.2 },
  { symbol: "NVDA", size: 12.8 },
  { symbol: "SPY", size: 8.5 },
];

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
  const dailyLossPercent = (mockRiskData.dailyLoss.current / mockRiskData.dailyLoss.limit) * 100;
  const losingTradesPercent =
    (mockRiskData.losingTradesToday.count / mockRiskData.losingTradesToday.limit) * 100;

  return (
    <PageWrapper title="Risk Monitor" subtitle="Monitor portfolio risk and exposure">
      <div className="space-y-6">
        {/* Top Risk Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 opacity-0 animate-slide-up stagger-1">
          <RiskCard
            title="Daily Loss"
            icon={<TrendingDown className="w-5 h-5 text-text-muted" />}
            value={`$${mockRiskData.dailyLoss.current}`}
            subtitle={`Limit: $${mockRiskData.dailyLoss.limit}`}
            status={dailyLossPercent < 50 ? "safe" : dailyLossPercent < 80 ? "warning" : "danger"}
          />
          <RiskCard
            title="Open Risk"
            icon={<Shield className="w-5 h-5 text-text-muted" />}
            value={`${mockRiskData.openRisk}%`}
            subtitle="Portfolio exposure"
            status={mockRiskData.openRisk < 5 ? "safe" : mockRiskData.openRisk < 8 ? "warning" : "danger"}
          />
          <RiskCard
            title="Losing Trades Today"
            icon={<AlertTriangle className="w-5 h-5 text-text-muted" />}
            value={mockRiskData.losingTradesToday.count.toString()}
            subtitle={`Lockout at ${mockRiskData.losingTradesToday.limit}`}
            status={losingTradesPercent < 60 ? "safe" : losingTradesPercent < 80 ? "warning" : "danger"}
          />
          <RiskCard
            title="Largest Position"
            icon={<Briefcase className="w-5 h-5 text-text-muted" />}
            value={mockRiskData.largestPosition.symbol}
            subtitle={`${mockRiskData.largestPosition.percent}% of portfolio`}
            status={mockRiskData.largestPosition.percent < 15 ? "safe" : mockRiskData.largestPosition.percent < 25 ? "warning" : "danger"}
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
              {mounted ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={positionSizes} layout="vertical">
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
                      {positionSizes.map((entry, index) => (
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
            <DrawdownGauge value={mockRiskData.currentDrawdown} />
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
      </div>
    </PageWrapper>
  );
}
