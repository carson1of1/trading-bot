/**
 * Frontend API client for Trading Bot backtest endpoint
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Request types
export interface BacktestRequest {
  top_n: number;
  days: number;
  longs_only?: boolean;
  shorts_only?: boolean;
  initial_capital?: number;
}

// Response types
export interface TradeResult {
  symbol: string;
  direction: string;
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  shares: number;
  pnl: number;
  pnl_pct: number;
  exit_reason: string;
  strategy: string;
  bars_held: number;
  mfe_pct: number;  // Maximum Favorable Excursion %
  mae_pct: number;  // Maximum Adverse Excursion %
}

export interface BacktestMetrics {
  initial_capital: number;
  final_value: number;
  total_return_pct: number;
  total_pnl: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  profit_factor: number;
  avg_pnl: number;
  avg_win: number;
  avg_loss: number;
  max_drawdown: number;
  sharpe_ratio: number;
  best_trade: number;
  worst_trade: number;
  avg_bars_held: number;
  days_traded: number;
  drawdown_peak_date: string | null;
  drawdown_peak_value: number;
  drawdown_trough_date: string | null;
  drawdown_trough_value: number;
}

export interface EquityPoint {
  timestamp: string;
  portfolio_value: number;
}

// Analytics breakdown types
export interface StrategyBreakdown {
  strategy: string;
  trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
  avg_mfe_pct: number;
  avg_mae_pct: number;
}

export interface ExitReasonBreakdown {
  exit_reason: string;
  count: number;
  total_pnl: number;
  avg_pnl: number;
  pct_of_trades: number;
}

export interface SymbolBreakdown {
  symbol: string;
  trades: number;
  total_pnl: number;
  win_rate: number;
  avg_pnl: number;
}

export interface PeriodBreakdown {
  date: string;
  trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
}

export interface DailyDrop {
  date: string;
  open: number;
  close: number;
  high: number;
  low: number;
  change_pct: number;
  change_dollars: number;
}

export interface BacktestResponse {
  success: boolean;
  metrics: BacktestMetrics | null;
  equity_curve: EquityPoint[];
  trades: TradeResult[];
  symbols_scanned: string[];
  by_strategy: StrategyBreakdown[];
  by_exit_reason: ExitReasonBreakdown[];
  by_symbol: SymbolBreakdown[];
  by_period: PeriodBreakdown[];
  worst_daily_drops: DailyDrop[];
  error?: string;
}

/**
 * Run a backtest with the specified parameters
 *
 * @param request - Backtest configuration parameters
 * @returns Promise resolving to the backtest results
 * @throws Error if the request fails or returns non-OK status
 */
export async function runBacktest(
  request: BacktestRequest
): Promise<BacktestResponse> {
  const response = await fetch(`${API_BASE_URL}/api/backtest`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`Backtest request failed: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

// =============================================================================
// Live Trading Types
// =============================================================================

export interface AccountData {
  equity: number;
  cash: number;
  buying_power: number;
  portfolio_value: number;
  daily_pnl: number;
  daily_pnl_percent: number;
}

export interface Position {
  symbol: string;
  qty: number;
  side: string;
  avg_entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pl: number;
  unrealized_plpc: number;
}

export interface PositionsData {
  positions: Position[];
  total_unrealized_pl: number;
}

export interface BotStatus {
  status: "running" | "stopped" | "error";
  mode: "PAPER" | "LIVE" | "DRY_RUN" | "BACKTEST";
  last_action: string | null;
  last_action_time: string | null;
  kill_switch_triggered: boolean;
  watchlist: string[] | null;
}

// Bot start response with scanner integration
export interface BotStartResponse {
  status: "started";
  watchlist: string[];
  scanner_ran_at: string;
  message: string;
}

export interface BotStartError {
  reason: "market_closed" | "no_results" | "scanner_error" | "bot_start_error";
  message: string;
}

export interface Order {
  id: string;
  symbol: string;
  qty: number;
  side: string;
  type: string;
  status: string;
  limit_price: number | null;
  stop_price: number | null;
  filled_qty: number;
  filled_avg_price: number | null;
  submitted_at: string | null;
  filled_at: string | null;
}

export interface OrdersData {
  orders: Order[];
}

// =============================================================================
// Live Trading API Functions
// =============================================================================

export async function getAccount(): Promise<AccountData> {
  const response = await fetch(`${API_BASE_URL}/api/account`);
  if (!response.ok) {
    throw new Error(`Failed to fetch account: ${response.status}`);
  }
  return response.json();
}

export async function getPositions(): Promise<PositionsData> {
  const response = await fetch(`${API_BASE_URL}/api/positions`);
  if (!response.ok) {
    throw new Error(`Failed to fetch positions: ${response.status}`);
  }
  return response.json();
}

export async function getBotStatus(): Promise<BotStatus> {
  const response = await fetch(`${API_BASE_URL}/api/bot/status`);
  if (!response.ok) {
    throw new Error(`Failed to fetch bot status: ${response.status}`);
  }
  return response.json();
}

export async function startBot(): Promise<BotStartResponse> {
  const response = await fetch(`${API_BASE_URL}/api/bot/start`, {
    method: "POST",
  });

  if (!response.ok) {
    const errorData = await response.json();
    const error = errorData.detail as BotStartError;
    throw new Error(error.message || "Failed to start bot");
  }

  return response.json();
}

export async function stopBot(): Promise<{ success: boolean; status: string }> {
  const response = await fetch(`${API_BASE_URL}/api/bot/stop`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to stop bot: ${response.status}`);
  }
  return response.json();
}

export async function getOrders(status: string = "open"): Promise<OrdersData> {
  const response = await fetch(`${API_BASE_URL}/api/orders?status=${status}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch orders: ${response.status}`);
  }
  return response.json();
}

// =============================================================================
// Settings Types and API
// =============================================================================

export interface Settings {
  mode: string;
  risk_per_trade: number;
  max_positions: number;
  stop_loss_pct: number;
  take_profit_pct: number;
  strategies_enabled: string[];
}

export async function getSettings(): Promise<Settings> {
  const response = await fetch(`${API_BASE_URL}/api/settings`);
  if (!response.ok) {
    throw new Error(`Failed to fetch settings: ${response.status}`);
  }
  return response.json();
}

// =============================================================================
// Scanner Types and API
// =============================================================================

export interface ScannerResult {
  symbol: string;
  atr_ratio: number;
  volume_ratio: number;
  composite_score: number;
  current_price: number;
}

export interface ScannerData {
  results: ScannerResult[];
  scanned_at: string;
}

export async function runScanner(topN: number = 10): Promise<ScannerData> {
  const response = await fetch(`${API_BASE_URL}/api/scanner/scan?top_n=${topN}`);
  if (!response.ok) {
    throw new Error(`Failed to run scanner: ${response.status}`);
  }
  return response.json();
}

// =============================================================================
// Trade History Types and API
// =============================================================================

export interface TradeHistoryItem {
  id: number;
  date: string;
  symbol: string;
  side: "LONG" | "SHORT";
  entryPrice: number;
  exitPrice: number;
  pnlDollar: number;
  pnlPercent: number;
  holdDuration: string;
  strategy: string;
}

export interface TradeHistoryData {
  trades: TradeHistoryItem[];
  total_count: number;
}

export async function getTradeHistory(days: number = 30): Promise<TradeHistoryData> {
  const response = await fetch(`${API_BASE_URL}/api/trades/history?days=${days}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch trade history: ${response.status}`);
  }
  return response.json();
}

// =============================================================================
// Activity Feed Types and API
// =============================================================================

export interface ActivityItem {
  id: number;
  type: "entry" | "exit" | "system" | "skipped" | "error";
  message: string;
  details?: string;
  timestamp: string;
}

export interface ActivityData {
  activities: ActivityItem[];
  total_count: number;
}

export async function getActivity(limit: number = 50): Promise<ActivityData> {
  const response = await fetch(`${API_BASE_URL}/api/activity?limit=${limit}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch activity: ${response.status}`);
  }
  return response.json();
}

// =============================================================================
// Risk Metrics Types and API
// =============================================================================

export interface RiskMetrics {
  daily_loss: number;
  daily_loss_limit: number;
  open_risk: number;
  losing_trades_today: number;
  losing_trades_limit: number;
  largest_position_symbol: string;
  largest_position_percent: number;
  current_drawdown: number;
  position_sizes: { symbol: string; size: number }[];
}

export async function getRiskMetrics(): Promise<RiskMetrics> {
  const response = await fetch(`${API_BASE_URL}/api/risk`);
  if (!response.ok) {
    throw new Error(`Failed to fetch risk metrics: ${response.status}`);
  }
  return response.json();
}
