"use client";

import { PageWrapper } from "@/components/layout/PageWrapper";
import { MetricCard } from "@/components/cards/MetricCard";
import { BotStatusCard } from "@/components/cards/BotStatusCard";
import { EquityCurveChart } from "@/components/charts/EquityCurveChart";
import { MiniPositionsTable } from "@/components/cards/MiniPositionsTable";
import { DollarSign, TrendingUp, Briefcase, Target, AlertTriangle } from "lucide-react";
import { usePolling } from "@/hooks/usePolling";
import { getAccount, getPositions, AccountData, PositionsData } from "@/lib/api";

export default function DashboardPage() {
  const { data: account, isLoading: accountLoading, error: accountError } = usePolling<AccountData>({
    fetcher: getAccount,
    interval: 5000,
  });

  const { data: positionsData, isLoading: positionsLoading } = usePolling<PositionsData>({
    fetcher: getPositions,
    interval: 5000,
  });

  // Transform positions for MiniPositionsTable
  const positions = positionsData?.positions.map(pos => ({
    symbol: pos.symbol,
    side: pos.side.toUpperCase() as "LONG" | "SHORT",
    pnlPercent: pos.unrealized_plpc,
  })) || [];

  // Mock equity data for now (will add equity curve endpoint later)
  const equityData = account ? [
    { date: "Now", value: account.portfolio_value },
  ] : [];

  // Loading state
  if (accountLoading && !account) {
    return (
      <PageWrapper>
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald"></div>
        </div>
      </PageWrapper>
    );
  }

  // Error state
  if (accountError) {
    return (
      <PageWrapper>
        <div className="flex flex-col items-center justify-center h-64 text-red">
          <AlertTriangle className="w-12 h-12 mb-4" />
          <p className="text-lg">Failed to load account data</p>
          <p className="text-sm text-text-muted mt-2">{accountError.message}</p>
        </div>
      </PageWrapper>
    );
  }

  return (
    <PageWrapper>
      <div className="space-y-6 pt-6">
        {/* Top Row - Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Account Balance"
            value=""
            numericValue={account?.portfolio_value || 0}
            prefix="$"
            decimals={2}
            change={account?.daily_pnl_percent || 0}
            icon={<DollarSign className="w-5 h-5" />}
            delay={1}
          />
          <MetricCard
            title="Today's P&L"
            value=""
            numericValue={account?.daily_pnl || 0}
            prefix="$"
            decimals={2}
            change={account?.daily_pnl_percent || 0}
            icon={<TrendingUp className="w-5 h-5" />}
            delay={2}
          />
          <MetricCard
            title="Open Positions"
            value=""
            numericValue={positions.length}
            subtitle={positions.map(p => p.symbol).join(", ") || "None"}
            icon={<Briefcase className="w-5 h-5" />}
            delay={3}
          />
          <MetricCard
            title="Buying Power"
            value=""
            numericValue={account?.buying_power || 0}
            prefix="$"
            decimals={2}
            icon={<Target className="w-5 h-5" />}
            delay={4}
          />
        </div>

        {/* Middle Row - Bot Status + Equity Curve */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <BotStatusCard />
          <div className="lg:col-span-2">
            <EquityCurveChart data={equityData} />
          </div>
        </div>

        {/* Bottom Row - Mini Positions Table */}
        <MiniPositionsTable positions={positions} />
      </div>
    </PageWrapper>
  );
}
