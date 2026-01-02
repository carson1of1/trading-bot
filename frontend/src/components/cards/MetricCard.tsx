"use client";

import { TrendingUp, TrendingDown } from "lucide-react";

interface MetricCardProps {
  title: string;
  value: string;
  change?: number;
  subtitle?: string;
  icon: React.ReactNode;
  delay?: number;
}

export function MetricCard({
  title,
  value,
  change,
  subtitle,
  icon,
  delay = 1,
}: MetricCardProps) {
  const isPositive = change !== undefined && change >= 0;
  const changeColor = isPositive ? "text-emerald" : "text-red";

  return (
    <div
      className={`glass glass-hover p-5 opacity-0 animate-slide-up stagger-${delay}`}
    >
      <div className="flex items-start justify-between">
        <div className="space-y-3">
          <p className="text-sm text-text-secondary font-medium">{title}</p>
          <p className="metric-value text-2xl text-white">{value}</p>

          {change !== undefined && (
            <div className={`flex items-center gap-1 ${changeColor}`}>
              {isPositive ? (
                <TrendingUp className="w-4 h-4" />
              ) : (
                <TrendingDown className="w-4 h-4" />
              )}
              <span className="text-sm font-medium mono">
                {isPositive ? "+" : ""}
                {change.toFixed(2)}%
              </span>
            </div>
          )}

          {subtitle && !change && (
            <p className="text-xs text-text-muted font-medium truncate max-w-[180px]">
              {subtitle}
            </p>
          )}
        </div>

        <div className="p-2.5 rounded-xl bg-surface-2 text-text-secondary">
          {icon}
        </div>
      </div>
    </div>
  );
}
