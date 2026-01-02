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
}

export interface EquityPoint {
  timestamp: string;
  portfolio_value: number;
}

export interface BacktestResponse {
  success: boolean;
  metrics: BacktestMetrics | null;
  equity_curve: EquityPoint[];
  trades: TradeResult[];
  symbols_scanned: string[];
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

export async function getOrders(status: string = "open"): Promise<OrdersData> {
  const response = await fetch(`${API_BASE_URL}/api/orders?status=${status}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch orders: ${response.status}`);
  }
  return response.json();
}
