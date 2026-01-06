"use client";

import { TrendingUp, TrendingDown } from "lucide-react";
import { CountUp } from "@/components/ui/CountUp";

interface MetricCardProps {
  title: string;
  value: string;
  numericValue?: number;
  change?: number;
  subtitle?: string;
  icon: React.ReactNode;
  delay?: number;
  prefix?: string;
  suffix?: string;
  decimals?: number;
}

export function MetricCard({
  title,
  value,
  numericValue,
  change,
  subtitle,
  icon,
  delay = 1,
  prefix = "",
  suffix = "",
  decimals = 0,
}: MetricCardProps) {
  const isPositive = change !== undefined && change >= 0;
  const changeColor = isPositive ? "text-emerald" : "text-red";
  const shadowClass = change !== undefined
    ? isPositive ? "card-pnl-positive" : "card-pnl-negative"
    : "";

  return (
    <div
      className={`glass-gradient p-5 opacity-0 animate-slide-up stagger-${delay} ${shadowClass}`}
    >
      <div className="flex items-start justify-between">
        <div className="space-y-3">
          <p className="text-sm text-text-secondary font-medium">{title}</p>
          <p className="metric-value text-2xl text-white">
            {numericValue !== undefined ? (
              <CountUp
                end={numericValue}
                prefix={prefix}
                suffix={suffix}
                decimals={decimals}
                duration={1.2}
              />
            ) : (
              value
            )}
          </p>

          {change !== undefined && (
            <div className={`flex items-center gap-1 ${changeColor}`}>
              {isPositive ? (
                <TrendingUp className="w-4 h-4" />
              ) : (
                <TrendingDown className="w-4 h-4" />
              )}
              <span className="text-sm font-medium mono">
                {isPositive ? "+" : ""}
                <CountUp end={change} decimals={2} duration={1} />%
              </span>
            </div>
          )}

          {subtitle && !change && (
            <p className="text-xs text-text-muted font-medium truncate max-w-[180px]">
              {subtitle}
            </p>
          )}
        </div>

        <div className="p-2.5 rounded-xl bg-surface-2 text-text-secondary hover-pop">
          {icon}
        </div>
      </div>
    </div>
  );
}
