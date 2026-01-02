"use client";

import { useState } from "react";
import { PageWrapper } from "@/components/layout/PageWrapper";
import { Settings, Key, Shield, Bell, Save, Eye, EyeOff } from "lucide-react";

type TradingMode = "PAPER" | "LIVE" | "DRY_RUN";

interface SettingsState {
  mode: TradingMode;
  riskParams: {
    maxPositionSize: number;
    dailyLossLimit: number;
    maxOpenPositions: number;
    stopLossPercent: number;
  };
  apiKeys: {
    alpacaKey: string;
    alpacaSecret: string;
  };
  notifications: {
    tradeAlerts: boolean;
    dailySummary: boolean;
    errorAlerts: boolean;
    riskWarnings: boolean;
  };
}

const initialSettings: SettingsState = {
  mode: "PAPER",
  riskParams: {
    maxPositionSize: 20,
    dailyLossLimit: 1000,
    maxOpenPositions: 5,
    stopLossPercent: 2,
  },
  apiKeys: {
    alpacaKey: "AKXXXXXXXXXXXXX",
    alpacaSecret: "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
  },
  notifications: {
    tradeAlerts: true,
    dailySummary: true,
    errorAlerts: true,
    riskWarnings: true,
  },
};

export default function SettingsPage() {
  const [settings, setSettings] = useState<SettingsState>(initialSettings);
  const [showApiKey, setShowApiKey] = useState(false);
  const [showApiSecret, setShowApiSecret] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  const updateSettings = <K extends keyof SettingsState>(
    section: K,
    value: SettingsState[K]
  ) => {
    setSettings((prev) => ({ ...prev, [section]: value }));
    setHasChanges(true);
  };

  const updateRiskParam = (key: keyof SettingsState["riskParams"], value: number) => {
    setSettings((prev) => ({
      ...prev,
      riskParams: { ...prev.riskParams, [key]: value },
    }));
    setHasChanges(true);
  };

  const updateNotification = (key: keyof SettingsState["notifications"], value: boolean) => {
    setSettings((prev) => ({
      ...prev,
      notifications: { ...prev.notifications, [key]: value },
    }));
    setHasChanges(true);
  };

  const handleSave = () => {
    // TODO: Save to API
    setHasChanges(false);
  };

  return (
    <PageWrapper title="Settings" subtitle="Configure bot parameters and preferences">
      <div className="max-w-3xl space-y-6">
        {/* Trading Mode */}
        <div className="glass p-5 opacity-0 animate-slide-up stagger-1">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-lg bg-emerald-glow">
              <Settings className="w-5 h-5 text-emerald" />
            </div>
            <div>
              <h3 className="font-semibold text-white">Trading Mode</h3>
              <p className="text-sm text-text-muted">Select the trading environment</p>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3">
            {(["PAPER", "LIVE", "DRY_RUN"] as TradingMode[]).map((mode) => (
              <button
                key={mode}
                onClick={() => updateSettings("mode", mode)}
                className={`p-4 rounded-lg border transition-all ${
                  settings.mode === mode
                    ? mode === "LIVE"
                      ? "border-red bg-red-glow"
                      : "border-emerald bg-emerald-glow"
                    : "border-border hover:border-border-hover bg-surface-1"
                }`}
              >
                <p
                  className={`font-semibold ${
                    settings.mode === mode
                      ? mode === "LIVE"
                        ? "text-red"
                        : "text-emerald"
                      : "text-white"
                  }`}
                >
                  {mode.replace("_", " ")}
                </p>
                <p className="text-xs text-text-muted mt-1">
                  {mode === "PAPER" && "Simulated trading"}
                  {mode === "LIVE" && "Real money"}
                  {mode === "DRY_RUN" && "No API calls"}
                </p>
              </button>
            ))}
          </div>

          {settings.mode === "LIVE" && (
            <div className="mt-4 p-3 rounded-lg bg-red-glow border border-red/30 flex items-start gap-3">
              <Shield className="w-5 h-5 text-red flex-shrink-0 mt-0.5" />
              <p className="text-sm text-red">
                Live mode uses real money. Ensure you understand the risks before enabling.
              </p>
            </div>
          )}
        </div>

        {/* Risk Parameters */}
        <div className="glass p-5 opacity-0 animate-slide-up stagger-2">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-lg bg-emerald-glow">
              <Shield className="w-5 h-5 text-emerald" />
            </div>
            <div>
              <h3 className="font-semibold text-white">Risk Parameters</h3>
              <p className="text-sm text-text-muted">Configure risk management limits</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-text-secondary mb-2">
                Max Position Size (%)
              </label>
              <input
                type="number"
                value={settings.riskParams.maxPositionSize}
                onChange={(e) => updateRiskParam("maxPositionSize", parseFloat(e.target.value))}
                className="input"
              />
            </div>
            <div>
              <label className="block text-sm text-text-secondary mb-2">
                Daily Loss Limit ($)
              </label>
              <input
                type="number"
                value={settings.riskParams.dailyLossLimit}
                onChange={(e) => updateRiskParam("dailyLossLimit", parseFloat(e.target.value))}
                className="input"
              />
            </div>
            <div>
              <label className="block text-sm text-text-secondary mb-2">
                Max Open Positions
              </label>
              <input
                type="number"
                value={settings.riskParams.maxOpenPositions}
                onChange={(e) => updateRiskParam("maxOpenPositions", parseInt(e.target.value))}
                className="input"
              />
            </div>
            <div>
              <label className="block text-sm text-text-secondary mb-2">
                Stop Loss (%)
              </label>
              <input
                type="number"
                step="0.1"
                value={settings.riskParams.stopLossPercent}
                onChange={(e) => updateRiskParam("stopLossPercent", parseFloat(e.target.value))}
                className="input"
              />
            </div>
          </div>
        </div>

        {/* API Keys */}
        <div className="glass p-5 opacity-0 animate-slide-up stagger-3">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-lg bg-emerald-glow">
              <Key className="w-5 h-5 text-emerald" />
            </div>
            <div>
              <h3 className="font-semibold text-white">API Keys</h3>
              <p className="text-sm text-text-muted">Alpaca API credentials</p>
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm text-text-secondary mb-2">API Key</label>
              <div className="relative">
                <input
                  type={showApiKey ? "text" : "password"}
                  value={settings.apiKeys.alpacaKey}
                  onChange={(e) => {
                    setSettings((prev) => ({
                      ...prev,
                      apiKeys: { ...prev.apiKeys, alpacaKey: e.target.value },
                    }));
                    setHasChanges(true);
                  }}
                  className="input pr-10"
                />
                <button
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-white"
                >
                  {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>
            <div>
              <label className="block text-sm text-text-secondary mb-2">Secret Key</label>
              <div className="relative">
                <input
                  type={showApiSecret ? "text" : "password"}
                  value={settings.apiKeys.alpacaSecret}
                  onChange={(e) => {
                    setSettings((prev) => ({
                      ...prev,
                      apiKeys: { ...prev.apiKeys, alpacaSecret: e.target.value },
                    }));
                    setHasChanges(true);
                  }}
                  className="input pr-10"
                />
                <button
                  onClick={() => setShowApiSecret(!showApiSecret)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-white"
                >
                  {showApiSecret ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Notifications */}
        <div className="glass p-5 opacity-0 animate-slide-up stagger-4">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-lg bg-emerald-glow">
              <Bell className="w-5 h-5 text-emerald" />
            </div>
            <div>
              <h3 className="font-semibold text-white">Notifications</h3>
              <p className="text-sm text-text-muted">Configure alert preferences</p>
            </div>
          </div>

          <div className="space-y-4">
            {[
              { key: "tradeAlerts" as const, label: "Trade Alerts", desc: "Get notified on trade entries and exits" },
              { key: "dailySummary" as const, label: "Daily Summary", desc: "Receive end-of-day performance summary" },
              { key: "errorAlerts" as const, label: "Error Alerts", desc: "Get notified on bot errors" },
              { key: "riskWarnings" as const, label: "Risk Warnings", desc: "Alerts when approaching risk limits" },
            ].map(({ key, label, desc }) => (
              <div key={key} className="flex items-center justify-between py-2">
                <div>
                  <p className="font-medium text-white">{label}</p>
                  <p className="text-sm text-text-muted">{desc}</p>
                </div>
                <button
                  onClick={() => updateNotification(key, !settings.notifications[key])}
                  className={`relative w-12 h-6 rounded-full transition-colors ${
                    settings.notifications[key] ? "bg-emerald" : "bg-surface-3"
                  }`}
                >
                  <span
                    className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
                      settings.notifications[key] ? "left-7" : "left-1"
                    }`}
                  />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Save Button */}
        {hasChanges && (
          <div className="sticky bottom-24 flex justify-end opacity-0 animate-slide-up">
            <button onClick={handleSave} className="btn btn-primary">
              <Save className="w-4 h-4" />
              Save Changes
            </button>
          </div>
        )}
      </div>
    </PageWrapper>
  );
}
