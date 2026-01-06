"use client";

import { useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useMounted } from "@/lib/utils";

interface DataPoint {
  date: string;
  value: number;
}

interface EquityCurveChartProps {
  data: DataPoint[];
}

type TimeRange = "7D" | "30D" | "90D" | "ALL";

const timeRanges: TimeRange[] = ["7D", "30D", "90D", "ALL"];

export function EquityCurveChart({ data }: EquityCurveChartProps) {
  const [selectedRange, setSelectedRange] = useState<TimeRange>("30D");
  const mounted = useMounted();

  const formatValue = (value: number) => {
    return `$${value.toLocaleString()}`;
  };

  const CustomTooltip = ({
    active,
    payload,
    label,
  }: {
    active?: boolean;
    payload?: Array<{ value: number }>;
    label?: string;
  }) => {
    if (active && payload && payload.length) {
      return (
        <div className="chart-tooltip glass px-3 py-2 border border-border">
          <p className="text-xs text-text-muted">{label}</p>
          <p className="text-sm font-semibold text-emerald mono">
            {formatValue(payload[0].value)}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="glass-gradient p-5 opacity-0 animate-slide-up stagger-4 h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-text-secondary">Equity Curve</h3>
        <div className="flex gap-1 p-0.5 bg-surface-1 rounded-lg">
          {timeRanges.map((range) => (
            <button
              key={range}
              onClick={() => setSelectedRange(range)}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-all hover:scale-105 ${
                selectedRange === range
                  ? "bg-blue text-white"
                  : "text-text-muted hover:text-white hover:bg-surface-2"
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div className="h-48 chart-container cursor-crosshair">
        {mounted ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={data}
              margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
            >
              <defs>
                <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="date"
                axisLine={false}
                tickLine={false}
                tick={{ fill: "#71717a", fontSize: 11 }}
                dy={10}
              />
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fill: "#71717a", fontSize: 11 }}
                tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                dx={-10}
                width={50}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="value"
                stroke="#10b981"
                strokeWidth={2}
                fill="url(#equityGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full flex items-center justify-center">
            <div className="w-6 h-6 border-2 border-emerald border-t-transparent rounded-full animate-spin" />
          </div>
        )}
      </div>
    </div>
  );
}
