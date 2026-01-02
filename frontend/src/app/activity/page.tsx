"use client";

import { useState } from "react";
import { PageWrapper } from "@/components/layout/PageWrapper";
import { Activity, Search, ArrowUpCircle, ArrowDownCircle, XCircle, AlertCircle, Info } from "lucide-react";

type ActivityType = "entry" | "exit" | "skipped" | "error" | "system";

interface ActivityItem {
  id: number;
  type: ActivityType;
  message: string;
  details?: string;
  timestamp: string;
}

const mockActivities: ActivityItem[] = [
  {
    id: 1,
    type: "entry",
    message: "Opened LONG position: AAPL",
    details: "10 shares @ $189.45 | Momentum strategy | 78% confidence",
    timestamp: "14:32:15",
  },
  {
    id: 2,
    type: "skipped",
    message: "Signal skipped: TSLA",
    details: "Daily loss limit approaching (80% of limit)",
    timestamp: "14:28:42",
  },
  {
    id: 3,
    type: "exit",
    message: "Closed LONG position: MSFT",
    details: "Profit floor triggered | +$245 (+1.2%)",
    timestamp: "14:15:08",
  },
  {
    id: 4,
    type: "system",
    message: "Trading cycle completed",
    details: "Checked 15 symbols, 3 signals generated",
    timestamp: "14:00:02",
  },
  {
    id: 5,
    type: "error",
    message: "Order rejected: NVDA",
    details: "Insufficient buying power for requested position size",
    timestamp: "13:45:33",
  },
  {
    id: 6,
    type: "entry",
    message: "Opened SHORT position: META",
    details: "5 shares @ $528.50 | Mean Reversion strategy | 72% confidence",
    timestamp: "13:32:18",
  },
  {
    id: 7,
    type: "exit",
    message: "Closed SHORT position: AMD",
    details: "Stop loss triggered | -$85 (-0.8%)",
    timestamp: "13:15:45",
  },
  {
    id: 8,
    type: "skipped",
    message: "Signal skipped: GOOGL",
    details: "Cooldown active (45 min remaining)",
    timestamp: "13:00:12",
  },
  {
    id: 9,
    type: "system",
    message: "Scanner refresh completed",
    details: "Top 3: NVDA (92), TSLA (88), AMD (82)",
    timestamp: "12:55:00",
  },
  {
    id: 10,
    type: "entry",
    message: "Opened LONG position: SPY",
    details: "20 shares @ $475.20 | Breakout strategy | 68% confidence",
    timestamp: "12:32:55",
  },
];

const typeConfig: Record<ActivityType, { icon: React.ReactNode; color: string; bgColor: string }> = {
  entry: {
    icon: <ArrowUpCircle className="w-4 h-4" />,
    color: "text-emerald",
    bgColor: "bg-emerald-glow",
  },
  exit: {
    icon: <ArrowDownCircle className="w-4 h-4" />,
    color: "text-red",
    bgColor: "bg-red-glow",
  },
  skipped: {
    icon: <XCircle className="w-4 h-4" />,
    color: "text-amber",
    bgColor: "bg-amber-glow",
  },
  error: {
    icon: <AlertCircle className="w-4 h-4" />,
    color: "text-red",
    bgColor: "bg-red-glow",
  },
  system: {
    icon: <Info className="w-4 h-4" />,
    color: "text-text-secondary",
    bgColor: "bg-surface-2",
  },
};

const filterOptions = [
  { id: "all", label: "All" },
  { id: "trades", label: "Trades" },
  { id: "skipped", label: "Skipped" },
  { id: "errors", label: "Errors" },
];

export default function ActivityFeedPage() {
  const [filter, setFilter] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");

  const filteredActivities = mockActivities.filter((activity) => {
    const matchesFilter =
      filter === "all" ||
      (filter === "trades" && (activity.type === "entry" || activity.type === "exit")) ||
      (filter === "skipped" && activity.type === "skipped") ||
      (filter === "errors" && activity.type === "error");

    const matchesSearch =
      activity.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      activity.details?.toLowerCase().includes(searchTerm.toLowerCase());

    return matchesFilter && matchesSearch;
  });

  return (
    <PageWrapper title="Activity Feed" subtitle="Real-time bot activity log">
      <div className="glass p-5 opacity-0 animate-slide-up">
        {/* Header Controls */}
        <div className="flex flex-wrap items-center gap-4 mb-6">
          <div className="relative flex-1 min-w-[200px] max-w-[300px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
            <input
              type="text"
              placeholder="Search activities..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="input !pl-10"
            />
          </div>

          <div className="flex gap-1 p-0.5 bg-surface-1 rounded-lg">
            {filterOptions.map((option) => (
              <button
                key={option.id}
                onClick={() => setFilter(option.id)}
                className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${
                  filter === option.id
                    ? "bg-emerald text-black"
                    : "text-text-muted hover:text-white"
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>

          <div className="ml-auto flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald animate-pulse" />
            <span className="text-sm text-text-muted">Live</span>
          </div>
        </div>

        {/* Activity List */}
        <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2">
          {filteredActivities.map((activity, index) => {
            const config = typeConfig[activity.type];
            return (
              <div
                key={activity.id}
                className="flex items-start gap-3 p-3 rounded-lg bg-surface-1 hover:bg-surface-2 transition-colors opacity-0 animate-slide-in-right"
                style={{ animationDelay: `${index * 0.05}s`, animationFillMode: "forwards" }}
              >
                <div className={`p-2 rounded-lg ${config.bgColor} ${config.color}`}>
                  {config.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <p className={`font-medium ${config.color}`}>{activity.message}</p>
                  {activity.details && (
                    <p className="text-sm text-text-muted mt-0.5 truncate">
                      {activity.details}
                    </p>
                  )}
                </div>
                <span className="text-xs text-text-muted mono flex-shrink-0">
                  {activity.timestamp}
                </span>
              </div>
            );
          })}
        </div>

        {filteredActivities.length === 0 && (
          <div className="text-center py-12">
            <Activity className="w-12 h-12 text-text-muted mx-auto mb-4" />
            <p className="text-text-secondary">No activities found</p>
          </div>
        )}
      </div>
    </PageWrapper>
  );
}
