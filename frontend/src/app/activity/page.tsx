"use client";

import { useState, useEffect } from "react";
import { PageWrapper } from "@/components/layout/PageWrapper";
import { Activity, Search, ArrowUpCircle, ArrowDownCircle, XCircle, AlertCircle, Info, RefreshCw } from "lucide-react";
import { getActivity, ActivityItem } from "@/lib/api";

type ActivityType = "entry" | "exit" | "skipped" | "error" | "system";

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
  const [activities, setActivities] = useState<ActivityItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");

  const fetchActivities = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getActivity(100);
      setActivities(data.activities);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch activity");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchActivities();
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchActivities, 30000);
    return () => clearInterval(interval);
  }, []);

  const filteredActivities = activities.filter((activity) => {
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

          <button
            onClick={fetchActivities}
            disabled={loading}
            className="btn btn-secondary"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </button>

          <div className="ml-auto flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald animate-pulse" />
            <span className="text-sm text-text-muted">Live</span>
          </div>
        </div>

        {/* Error State */}
        {error && (
          <div className="flex items-center gap-3 p-4 mb-6 bg-red-500/10 border border-red-500/20 rounded-lg">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <span className="text-red-400">{error}</span>
          </div>
        )}

        {/* Loading State */}
        {loading && activities.length === 0 && (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-6 h-6 text-emerald animate-spin" />
            <span className="ml-3 text-text-secondary">Loading activity...</span>
          </div>
        )}

        {/* Activity List */}
        {!loading || activities.length > 0 ? (
          <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2">
            {filteredActivities.map((activity, index) => {
              const config = typeConfig[activity.type as ActivityType] || typeConfig.system;
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
        ) : null}

        {!loading && filteredActivities.length === 0 && (
          <div className="text-center py-12">
            <Activity className="w-12 h-12 text-text-muted mx-auto mb-4" />
            <p className="text-text-secondary">
              {activities.length === 0
                ? "No activity yet. Start trading to see your activity here."
                : "No activities match your filters."}
            </p>
          </div>
        )}
      </div>
    </PageWrapper>
  );
}
