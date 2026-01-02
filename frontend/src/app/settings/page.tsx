"use client";

import { useState, useEffect } from "react";
import { PageWrapper } from "@/components/layout/PageWrapper";
import { Settings as SettingsIcon, Shield, Bell, Save, AlertTriangle, CheckCircle } from "lucide-react";
import { usePolling } from "@/hooks/usePolling";
import { getSettings, Settings } from "@/lib/api";

export default function SettingsPage() {
  const { data: apiSettings, isLoading, error } = usePolling<Settings>({
    fetcher: getSettings,
    interval: 10000, // Less frequent polling for settings
  });

  if (isLoading && !apiSettings) {
    return (
      <PageWrapper title="Settings" subtitle="Configure bot parameters and preferences">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald"></div>
        </div>
      </PageWrapper>
    );
  }

  if (error) {
    return (
      <PageWrapper title="Settings" subtitle="Configure bot parameters and preferences">
        <div className="flex flex-col items-center justify-center h-64 text-red">
          <AlertTriangle className="w-12 h-12 mb-4" />
          <p>Failed to load settings</p>
        </div>
      </PageWrapper>
    );
  }

  const modeConfig = {
    PAPER: { color: "text-emerald", bg: "bg-emerald-glow", border: "border-emerald" },
    LIVE: { color: "text-red", bg: "bg-red-glow", border: "border-red" },
    DRY_RUN: { color: "text-amber", bg: "bg-amber-glow", border: "border-amber" },
    BACKTEST: { color: "text-blue", bg: "bg-blue-glow", border: "border-blue" },
  };

  const currentMode = apiSettings?.mode || "DRY_RUN";
  const modeStyle = modeConfig[currentMode as keyof typeof modeConfig] || modeConfig.DRY_RUN;

  return (
    <PageWrapper title="Settings" subtitle="Configure bot parameters and preferences">
      <div className="space-y-4 pt-2">
        {/* Top Row - Trading Mode + Risk Parameters */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Trading Mode */}
          <div className="glass-gradient p-4 opacity-0 animate-slide-up stagger-1">
            <div className="flex items-center gap-2 mb-4">
              <div className="p-2 rounded-lg bg-blue-glow">
                <SettingsIcon className="w-4 h-4 text-blue icon-spin-hover" />
              </div>
              <h3 className="font-semibold text-white text-sm">Current Trading Mode</h3>
            </div>

            <div className={`p-4 rounded-lg border ${modeStyle.border} ${modeStyle.bg}`}>
              <p className={`text-xl font-bold ${modeStyle.color}`}>
                {currentMode.replace("_", " ")}
              </p>
              <p className="text-xs text-text-muted mt-1">
                {currentMode === "LIVE" && "Using real money - be careful!"}
                {currentMode === "PAPER" && "Paper trading - no real money"}
                {currentMode === "DRY_RUN" && "Simulation mode - no API calls"}
                {currentMode === "BACKTEST" && "Historical backtesting mode"}
              </p>
            </div>

            {currentMode === "LIVE" && (
              <div className="mt-3 p-2 rounded-lg bg-red-glow border border-red/30 flex items-center gap-2">
                <Shield className="w-4 h-4 text-red flex-shrink-0" />
                <p className="text-xs text-red">Live mode uses real money!</p>
              </div>
            )}
          </div>

          {/* Risk Parameters */}
          <div className="glass-gradient p-4 opacity-0 animate-slide-up stagger-2">
            <div className="flex items-center gap-2 mb-4">
              <div className="p-2 rounded-lg bg-emerald-glow">
                <Shield className="w-4 h-4 text-emerald icon-wiggle" />
              </div>
              <h3 className="font-semibold text-white text-sm">Risk Parameters</h3>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-surface-1 border border-border">
                <p className="text-xs text-text-muted">Risk Per Trade</p>
                <p className="text-lg font-semibold text-white">
                  {((apiSettings?.risk_per_trade || 0) * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-3 rounded-lg bg-surface-1 border border-border">
                <p className="text-xs text-text-muted">Max Positions</p>
                <p className="text-lg font-semibold text-white">
                  {apiSettings?.max_positions || 0}
                </p>
              </div>
              <div className="p-3 rounded-lg bg-surface-1 border border-border">
                <p className="text-xs text-text-muted">Stop Loss</p>
                <p className="text-lg font-semibold text-red">
                  {((apiSettings?.stop_loss_pct || 0) * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-3 rounded-lg bg-surface-1 border border-border">
                <p className="text-xs text-text-muted">Take Profit</p>
                <p className="text-lg font-semibold text-emerald">
                  {((apiSettings?.take_profit_pct || 0) * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Row - Strategies */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Enabled Strategies */}
          <div className="glass-gradient p-4 opacity-0 animate-slide-up stagger-3">
            <div className="flex items-center gap-2 mb-4">
              <div className="p-2 rounded-lg bg-cyan-glow">
                <CheckCircle className="w-4 h-4 text-cyan" />
              </div>
              <h3 className="font-semibold text-white text-sm">Enabled Strategies</h3>
            </div>

            <div className="space-y-2">
              {apiSettings?.strategies_enabled && apiSettings.strategies_enabled.length > 0 ? (
                apiSettings.strategies_enabled.map((strategy) => (
                  <div
                    key={strategy}
                    className="flex items-center justify-between p-3 rounded-lg bg-surface-1 border border-border"
                  >
                    <span className="text-text-secondary">{strategy}</span>
                    <span className="badge badge-emerald">Active</span>
                  </div>
                ))
              ) : (
                <p className="text-text-muted text-sm">No strategies enabled</p>
              )}
            </div>
          </div>

          {/* Config File Notice */}
          <div className="glass-gradient p-4 opacity-0 animate-slide-up stagger-4">
            <div className="flex items-center gap-2 mb-4">
              <div className="p-2 rounded-lg bg-amber-glow">
                <Bell className="w-4 h-4 text-amber" />
              </div>
              <h3 className="font-semibold text-white text-sm">Configuration</h3>
            </div>

            <div className="p-4 rounded-lg bg-surface-1 border border-border">
              <p className="text-sm text-text-secondary mb-2">
                Settings are loaded from <code className="text-cyan">config.yaml</code>
              </p>
              <p className="text-xs text-text-muted">
                To modify settings, edit the config file and restart the bot.
                A web-based settings editor is coming soon.
              </p>
            </div>
          </div>
        </div>
      </div>
    </PageWrapper>
  );
}
