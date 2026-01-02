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
