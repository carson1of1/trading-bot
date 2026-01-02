"""
FastAPI Backend Server for Trading Bot

Provides REST API endpoints for running backtests with scanner-selected symbols.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yaml

from backtest import Backtest1Hour
from core.cache import get_cache
from core.broker import create_broker, BrokerInterface, BrokerAPIError
from core.config import get_global_config

# Broker singleton for API
_broker: Optional[BrokerInterface] = None


def get_broker() -> BrokerInterface:
    """Get or create broker singleton."""
    global _broker
    if _broker is None:
        _broker = create_broker()
    return _broker


# Bot state tracking (simple in-memory for now)
_bot_state = {
    "status": "stopped",
    "last_action": None,
    "last_action_time": None,
    "kill_switch_triggered": False
}


def update_bot_state(status: str = None, last_action: str = None):
    """Update bot state."""
    if status:
        _bot_state["status"] = status
    if last_action:
        _bot_state["last_action"] = last_action
        _bot_state["last_action_time"] = datetime.now().isoformat()


# Pydantic models for request/response
class BacktestRequest(BaseModel):
    """Request model for backtest endpoint."""
    top_n: int = Field(default=10, ge=1, le=50, description="Number of top volatile stocks to scan")
    days: int = Field(default=60, ge=7, le=365, description="Number of days to backtest")
    longs_only: bool = Field(default=False, description="Only take LONG positions")
    shorts_only: bool = Field(default=False, description="Only take SHORT positions")
    initial_capital: float = Field(default=100000.0, ge=1000, le=10000000, description="Starting capital")


class TradeResult(BaseModel):
    """Individual trade result."""
    symbol: str
    direction: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    strategy: str
    bars_held: int


class BacktestMetrics(BaseModel):
    """Backtest performance metrics."""
    initial_capital: float
    final_value: float
    total_return_pct: float
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    best_trade: float
    worst_trade: float
    avg_bars_held: float


class EquityCurvePoint(BaseModel):
    """Single point on equity curve."""
    timestamp: str
    portfolio_value: float


class BacktestResponse(BaseModel):
    """Response model for backtest endpoint."""
    success: bool
    metrics: Optional[BacktestMetrics] = None
    equity_curve: List[EquityCurvePoint] = []
    trades: List[TradeResult] = []
    symbols_scanned: List[str] = []
    error: Optional[str] = None


class AccountResponse(BaseModel):
    """Account information response."""
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    daily_pnl: float
    daily_pnl_percent: float


class PositionResponse(BaseModel):
    """Single position response."""
    symbol: str
    qty: float
    side: str
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float


class PositionsResponse(BaseModel):
    """List of positions response."""
    positions: List[PositionResponse]
    total_unrealized_pl: float


class BotStatusResponse(BaseModel):
    """Bot status response."""
    status: str  # 'running', 'stopped', 'error'
    mode: str  # 'PAPER', 'LIVE', 'DRY_RUN', 'BACKTEST'
    last_action: Optional[str] = None
    last_action_time: Optional[str] = None
    kill_switch_triggered: bool = False


class OrderResponse(BaseModel):
    """Single order response."""
    id: str
    symbol: str
    qty: float
    side: str
    type: str
    status: str
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_qty: float = 0
    filled_avg_price: Optional[float] = None
    submitted_at: Optional[str] = None
    filled_at: Optional[str] = None


class OrdersResponse(BaseModel):
    """List of orders response."""
    orders: List[OrderResponse]


class SettingsResponse(BaseModel):
    """Bot settings/configuration response."""
    mode: str
    risk_per_trade: float
    max_positions: int
    stop_loss_pct: float
    take_profit_pct: float
    strategies_enabled: List[str]


class ScannerResult(BaseModel):
    """Single scanner result."""
    symbol: str
    atr_ratio: float
    volume_ratio: float
    composite_score: float
    current_price: float


class ScannerResponse(BaseModel):
    """Scanner results response."""
    results: List[ScannerResult]
    scanned_at: str


# Initialize FastAPI app
app = FastAPI(
    title="Trading Bot API",
    description="REST API for running backtests with scanner-selected symbols",
    version="1.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_config() -> Dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def load_universe() -> Dict:
    """Load universe from universe.yaml."""
    universe_path = Path(__file__).parent.parent / "universe.yaml"
    if universe_path.exists():
        with open(universe_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def collect_scanner_symbols(universe: Dict) -> List[str]:
    """Collect all symbols from scanner_universe in universe.yaml."""
    scanner_universe = universe.get("scanner_universe", {})
    all_symbols = []

    for category, symbols in scanner_universe.items():
        if isinstance(symbols, list):
            all_symbols.extend(symbols)

    # Remove duplicates while preserving order
    seen = set()
    unique_symbols = []
    for symbol in all_symbols:
        if symbol not in seen:
            seen.add(symbol)
            unique_symbols.append(symbol)

    return unique_symbols


def format_backtest_results(results: Dict, symbols_scanned: List[str]) -> BacktestResponse:
    """
    Transform backtest output dict into API response format.

    Args:
        results: Raw backtest results from Backtest1Hour.run()
        symbols_scanned: List of symbols that were included in the scan

    Returns:
        Formatted BacktestResponse
    """
    if not results:
        return BacktestResponse(
            success=False,
            error="Backtest returned no results"
        )

    # Format metrics
    raw_metrics = results.get("metrics", {})
    metrics = BacktestMetrics(
        initial_capital=round(raw_metrics.get("initial_capital", 0), 2),
        final_value=round(raw_metrics.get("final_value", 0), 2),
        total_return_pct=round(raw_metrics.get("total_return_pct", 0), 2),
        total_pnl=round(raw_metrics.get("total_pnl", 0), 2),
        total_trades=raw_metrics.get("total_trades", 0),
        winning_trades=raw_metrics.get("winning_trades", 0),
        losing_trades=raw_metrics.get("losing_trades", 0),
        win_rate=round(raw_metrics.get("win_rate", 0), 2),
        profit_factor=round(raw_metrics.get("profit_factor", 0), 2),
        avg_pnl=round(raw_metrics.get("avg_pnl", 0), 2),
        avg_win=round(raw_metrics.get("avg_win", 0), 2),
        avg_loss=round(raw_metrics.get("avg_loss", 0), 2),
        max_drawdown=round(raw_metrics.get("max_drawdown", 0), 2),
        sharpe_ratio=round(raw_metrics.get("sharpe_ratio", 0), 2),
        best_trade=round(raw_metrics.get("best_trade", 0), 2),
        worst_trade=round(raw_metrics.get("worst_trade", 0), 2),
        avg_bars_held=round(raw_metrics.get("avg_bars_held", 0), 1)
    )

    # Format equity curve - sample to max 50 points
    equity_curve = []
    raw_equity = results.get("equity_curve")

    if raw_equity is not None and len(raw_equity) > 0:
        # Convert DataFrame to list of dicts if needed
        if hasattr(raw_equity, "to_dict"):
            equity_records = raw_equity.to_dict("records")
        else:
            equity_records = raw_equity

        # Sample to max 50 points
        max_points = 50
        if len(equity_records) > max_points:
            step = len(equity_records) / max_points
            sampled_indices = [int(i * step) for i in range(max_points)]
            # Always include the last point
            if sampled_indices[-1] != len(equity_records) - 1:
                sampled_indices[-1] = len(equity_records) - 1
            equity_records = [equity_records[i] for i in sampled_indices]

        for record in equity_records:
            timestamp = record.get("timestamp", "")
            if hasattr(timestamp, "isoformat"):
                timestamp = timestamp.isoformat()
            elif hasattr(timestamp, "strftime"):
                timestamp = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                timestamp = str(timestamp)

            equity_curve.append(EquityCurvePoint(
                timestamp=timestamp,
                portfolio_value=round(record.get("portfolio_value", 0), 2)
            ))

    # Format trades - sort by date descending
    trades = []
    raw_trades = results.get("trades", [])

    # Sort by exit_date descending
    sorted_trades = sorted(
        raw_trades,
        key=lambda t: str(t.get("exit_date", "")),
        reverse=True
    )

    for trade in sorted_trades:
        entry_date = trade.get("entry_date", "")
        exit_date = trade.get("exit_date", "")

        # Convert timestamps to strings
        if hasattr(entry_date, "isoformat"):
            entry_date = entry_date.isoformat()
        elif hasattr(entry_date, "strftime"):
            entry_date = entry_date.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            entry_date = str(entry_date)

        if hasattr(exit_date, "isoformat"):
            exit_date = exit_date.isoformat()
        elif hasattr(exit_date, "strftime"):
            exit_date = exit_date.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            exit_date = str(exit_date)

        trades.append(TradeResult(
            symbol=trade.get("symbol", ""),
            direction=trade.get("direction", "LONG"),
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=round(trade.get("entry_price", 0), 2),
            exit_price=round(trade.get("exit_price", 0), 2),
            shares=int(trade.get("shares", 0)),
            pnl=round(trade.get("pnl", 0), 2),
            pnl_pct=round(trade.get("pnl_pct", 0), 2),
            exit_reason=trade.get("exit_reason", ""),
            strategy=trade.get("strategy", ""),
            bars_held=int(trade.get("bars_held", 0))
        ))

    return BacktestResponse(
        success=True,
        metrics=metrics,
        equity_curve=equity_curve,
        trades=trades,
        symbols_scanned=symbols_scanned
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    cache = get_cache()
    stats = cache.get_cache_stats()
    return stats


@app.post("/api/cache/clear")
async def clear_cache():
    """Clear all cached data."""
    cache = get_cache()
    cache.clear()
    return {"success": True, "message": "Cache cleared"}


@app.get("/api/account", response_model=AccountResponse)
async def get_account():
    """Get current account information."""
    try:
        broker = get_broker()
        account = broker.get_account()
        return AccountResponse(
            equity=round(account.equity, 2),
            cash=round(account.cash, 2),
            buying_power=round(account.buying_power, 2),
            portfolio_value=round(account.portfolio_value, 2),
            daily_pnl=round(account.daily_pnl, 2),
            daily_pnl_percent=round(account.daily_pnl_percent * 100, 2)
        )
    except BrokerAPIError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions", response_model=PositionsResponse)
async def get_positions():
    """Get all open positions."""
    try:
        broker = get_broker()
        positions = broker.get_positions()

        position_list = [
            PositionResponse(
                symbol=pos.symbol,
                qty=pos.qty,
                side=pos.side,
                avg_entry_price=round(pos.avg_entry_price, 2),
                current_price=round(pos.current_price, 2),
                market_value=round(pos.market_value, 2),
                unrealized_pl=round(pos.unrealized_pl, 2),
                unrealized_plpc=round(pos.unrealized_plpc * 100, 2)
            )
            for pos in positions
        ]

        total_pl = sum(pos.unrealized_pl for pos in positions)

        return PositionsResponse(
            positions=position_list,
            total_unrealized_pl=round(total_pl, 2)
        )
    except BrokerAPIError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bot/status", response_model=BotStatusResponse)
async def get_bot_status():
    """Get current bot status."""
    try:
        config = get_global_config()
        mode = config.get_mode()

        return BotStatusResponse(
            status=_bot_state["status"],
            mode=mode,
            last_action=_bot_state["last_action"],
            last_action_time=_bot_state["last_action_time"],
            kill_switch_triggered=_bot_state["kill_switch_triggered"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orders", response_model=OrdersResponse)
async def get_orders(status: str = "open"):
    """Get orders with optional status filter."""
    try:
        broker = get_broker()
        orders = broker.list_orders(status=status)

        order_list = [
            OrderResponse(
                id=order.id,
                symbol=order.symbol,
                qty=order.qty,
                side=order.side,
                type=order.type,
                status=order.status,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                filled_qty=order.filled_qty,
                filled_avg_price=order.filled_avg_price,
                submitted_at=order.submitted_at.isoformat() if order.submitted_at else None,
                filled_at=order.filled_at.isoformat() if order.filled_at else None
            )
            for order in orders
        ]

        return OrdersResponse(orders=order_list)
    except BrokerAPIError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/settings", response_model=SettingsResponse)
async def get_settings():
    """Get current bot settings."""
    try:
        config = load_config()

        return SettingsResponse(
            mode=config.get("mode", "DRY_RUN"),
            risk_per_trade=config.get("risk", {}).get("risk_per_trade", 0.02),
            max_positions=config.get("risk", {}).get("max_open_positions", 5),
            stop_loss_pct=config.get("exit_rules", {}).get("hard_stop_loss", 0.02),
            take_profit_pct=config.get("exit_rules", {}).get("partial_take_profit", {}).get("threshold", 0.02),
            strategies_enabled=[s["name"] for s in config.get("strategies", []) if s.get("enabled", False)]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scanner/scan", response_model=ScannerResponse)
async def run_scanner(top_n: int = 10):
    """Run volatility scanner and return top N symbols."""
    try:
        from core.scanner import VolatilityScanner

        config = load_config()
        universe = load_universe()
        symbols = collect_scanner_symbols(universe)

        scanner = VolatilityScanner(config.get("volatility_scanner", {}))
        results = scanner.scan(symbols, top_n=top_n)

        scanner_results = [
            ScannerResult(
                symbol=r["symbol"],
                atr_ratio=round(r.get("atr_ratio", 0), 4),
                volume_ratio=round(r.get("volume_ratio", 0), 2),
                composite_score=round(r.get("composite_score", 0), 4),
                current_price=round(r.get("current_price", 0), 2)
            )
            for r in results
        ]

        return ScannerResponse(
            results=scanner_results,
            scanned_at=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """
    Run a backtest with scanner-selected symbols.

    The scanner will select the top N volatile stocks from the scanner_universe
    each day during the backtest period.
    """
    try:
        # Load config and universe
        config = load_config()
        universe = load_universe()

        # Set scanner enabled with top_n from request
        if "volatility_scanner" not in config:
            config["volatility_scanner"] = {}
        config["volatility_scanner"]["enabled"] = True
        config["volatility_scanner"]["top_n"] = request.top_n

        # Calculate date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=request.days)).strftime("%Y-%m-%d")

        # Collect all symbols from scanner_universe
        symbols = collect_scanner_symbols(universe)

        if not symbols:
            return BacktestResponse(
                success=False,
                error="No symbols found in scanner_universe"
            )

        # Create and run backtest
        backtester = Backtest1Hour(
            initial_capital=request.initial_capital,
            config=config,
            longs_only=request.longs_only,
            shorts_only=request.shorts_only,
            scanner_enabled=True
        )

        results = backtester.run(symbols, start_date, end_date)

        # Format and return results
        return format_backtest_results(results, symbols)

    except Exception as e:
        return BacktestResponse(
            success=False,
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
