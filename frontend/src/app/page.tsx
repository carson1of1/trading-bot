"use client";

import { PageWrapper } from "@/components/layout/PageWrapper";
import { MetricCard } from "@/components/cards/MetricCard";
import { BotStatusCard } from "@/components/cards/BotStatusCard";
import { EquityCurveChart } from "@/components/charts/EquityCurveChart";
import { MiniPositionsTable } from "@/components/cards/MiniPositionsTable";
import { DollarSign, TrendingUp, Briefcase, Target } from "lucide-react";

// Mock data - will be replaced with API calls
const mockMetrics = {
  accountBalance: 125432.56,
  balanceChange: 2.34,
  todayPnl: 1245.67,
  todayPnlPercent: 1.02,
  openPositions: 4,
  positionSymbols: ["AAPL", "MSFT", "NVDA", "SPY"],
  winRate: 68.5,
  winLossCount: "41/19",
};

const mockEquityData = [
  { date: "Dec 01", value: 120000 },
  { date: "Dec 05", value: 118500 },
  { date: "Dec 10", value: 121200 },
  { date: "Dec 15", value: 119800 },
  { date: "Dec 20", value: 123400 },
  { date: "Dec 25", value: 122100 },
  { date: "Dec 30", value: 125432 },
];

const mockPositions = [
  { symbol: "AAPL", side: "LONG" as const, pnlPercent: 2.45 },
  { symbol: "MSFT", side: "LONG" as const, pnlPercent: 1.23 },
  { symbol: "NVDA", side: "SHORT" as const, pnlPercent: -0.87 },
  { symbol: "SPY", side: "LONG" as const, pnlPercent: 0.56 },
];

export default function DashboardPage() {
  return (
    <PageWrapper>
      <div className="space-y-6 pt-6">
        {/* Top Row - Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Account Balance"
            value={`$${mockMetrics.accountBalance.toLocaleString()}`}
            change={mockMetrics.balanceChange}
            icon={<DollarSign className="w-5 h-5" />}
            delay={1}
          />
          <MetricCard
            title="Today's P&L"
            value={`$${mockMetrics.todayPnl.toLocaleString()}`}
            change={mockMetrics.todayPnlPercent}
            icon={<TrendingUp className="w-5 h-5" />}
            delay={2}
          />
          <MetricCard
            title="Open Positions"
            value={mockMetrics.openPositions.toString()}
            subtitle={mockMetrics.positionSymbols.join(", ")}
            icon={<Briefcase className="w-5 h-5" />}
            delay={3}
          />
          <MetricCard
            title="Win Rate"
            value={`${mockMetrics.winRate}%`}
            subtitle={`${mockMetrics.winLossCount} trades`}
            icon={<Target className="w-5 h-5" />}
            delay={4}
          />
        </div>

        {/* Middle Row - Bot Status + Equity Curve */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <BotStatusCard />
          <div className="lg:col-span-2">
            <EquityCurveChart data={mockEquityData} />
          </div>
        </div>

        {/* Bottom Row - Mini Positions Table */}
        <MiniPositionsTable positions={mockPositions} />
      </div>
    </PageWrapper>
  );
}
