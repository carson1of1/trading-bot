"""
FastAPI Backend Server for Trading Bot

Provides REST API endpoints for running backtests with scanner-selected symbols.
"""

import sys
import os
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yaml
import pandas as pd

from backtest import Backtest1Hour
from core.cache import get_cache
from core.broker import create_broker, BrokerInterface, BrokerAPIError
from core.config import get_global_config
from core.market_hours import is_market_open, get_market_status_message
from core.scanner import VolatilityScanner
from core.data import YFinanceDataFetcher
from core.logger import TradeLogger
import subprocess

# PID file for tracking bot process
BOT_PID_FILE = Path(__file__).parent.parent / "logs" / "bot.pid"


def _read_bot_pid() -> Optional[int]:
    """Read bot PID from file."""
    try:
        if BOT_PID_FILE.exists():
            pid = int(BOT_PID_FILE.read_text().strip())
            return pid
    except (ValueError, IOError):
        pass
    return None


def _write_bot_pid(pid: int):
    """Write bot PID to file."""
    BOT_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    BOT_PID_FILE.write_text(str(pid))


def _clear_bot_pid():
    """Clear bot PID file."""
    try:
        if BOT_PID_FILE.exists():
            BOT_PID_FILE.unlink()
    except IOError:
        pass


def _is_bot_running() -> bool:
    """Check if bot process is actually running."""
    pid = _read_bot_pid()
    if pid is None:
        return False
    try:
        # Check if process exists (signal 0 doesn't kill, just checks)
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        # Process doesn't exist, clean up stale PID file
        _clear_bot_pid()
        return False


def _kill_bot_process() -> bool:
    """Kill the bot process if running."""
    pid = _read_bot_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        # Give it a moment to terminate gracefully
        import time
        time.sleep(0.5)
        # Check if still running, force kill if needed
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        _clear_bot_pid()
        return True
    except (OSError, ProcessLookupError):
        _clear_bot_pid()
        return False

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
    "kill_switch_triggered": False,
    "watchlist": None
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
    initial_capital: float = Field(default=10000.0, ge=1000, le=10000000, description="Starting capital")
    # Trailing stop parameters (defaults match live trading config.yaml)
    trailing_stop_enabled: bool = Field(default=True, description="Enable trailing stop loss")
    trailing_activation_pct: float = Field(default=0.15, ge=0.1, le=10.0, description="% profit to activate trailing stop")
    trailing_trail_pct: float = Field(default=0.15, ge=0.1, le=10.0, description="% to trail below peak")


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
    mfe_pct: float = 0.0  # Maximum Favorable Excursion %
    mae_pct: float = 0.0  # Maximum Adverse Excursion %


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
    days_traded: int = 0
    drawdown_peak_date: Optional[str] = None
    drawdown_peak_value: float = 0.0
    drawdown_trough_date: Optional[str] = None
    drawdown_trough_value: float = 0.0


class DailyDrop(BaseModel):
    """Single day equity change."""
    date: str
    open: float
    close: float
    high: float
    low: float
    change_pct: float
    change_dollars: float


class EquityCurvePoint(BaseModel):
    """Single point on equity curve."""
    timestamp: str
    portfolio_value: float


class StrategyBreakdown(BaseModel):
    """Performance breakdown by strategy."""
    strategy: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    avg_mfe_pct: float  # Average max favorable excursion
    avg_mae_pct: float  # Average max adverse excursion


class ExitReasonBreakdown(BaseModel):
    """Performance breakdown by exit reason."""
    exit_reason: str
    count: int
    total_pnl: float
    avg_pnl: float
    pct_of_trades: float


class SymbolBreakdown(BaseModel):
    """Performance breakdown by symbol."""
    symbol: str
    trades: int
    total_pnl: float
    win_rate: float
    avg_pnl: float


class PeriodBreakdown(BaseModel):
    """Performance breakdown by date period (daily)."""
    date: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float


class BacktestResponse(BaseModel):
    """Response model for backtest endpoint."""
    success: bool
    metrics: Optional[BacktestMetrics] = None
    equity_curve: List[EquityCurvePoint] = []
    trades: List[TradeResult] = []
    symbols_scanned: List[str] = []
    by_strategy: List[StrategyBreakdown] = []
    by_exit_reason: List[ExitReasonBreakdown] = []
    by_symbol: List[SymbolBreakdown] = []  # Sorted worst to best
    by_period: List[PeriodBreakdown] = []  # Daily breakdown, sorted by date
    worst_daily_drops: List[DailyDrop] = []
    error: Optional[str] = None


class AccountResponse(BaseModel):
    """Account information response."""
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    daily_pnl: float
    daily_pnl_percent: float


class EquityHistoryPoint(BaseModel):
    """Single point on equity history curve."""
    timestamp: str
    equity: float


class EquityHistoryResponse(BaseModel):
    """Response model for equity history endpoint."""
    data: List[EquityHistoryPoint]
    period: str
    base_value: float


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
    watchlist: Optional[List[str]] = None


class BotStartResponse(BaseModel):
    """Response for bot start endpoint."""
    status: str
    watchlist: List[str]
    scanner_ran_at: str
    message: str


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


class TradeHistoryItem(BaseModel):
    """Single trade history item."""
    id: int
    date: str
    symbol: str
    side: str  # LONG or SHORT
    entryPrice: float
    exitPrice: float
    pnlDollar: float
    pnlPercent: float
    holdDuration: str
    strategy: str


class TradeHistoryResponse(BaseModel):
    """Trade history response."""
    trades: List[TradeHistoryItem]
    total_count: int


class ActivityItem(BaseModel):
    """Single activity item for activity feed."""
    id: int
    type: str  # 'entry', 'exit', 'system'
    message: str
    details: Optional[str] = None
    timestamp: str


class ActivityResponse(BaseModel):
    """Activity feed response."""
    activities: List[ActivityItem]
    total_count: int


class RiskMetrics(BaseModel):
    """Risk monitoring metrics."""
    daily_loss: float
    daily_loss_limit: float
    open_risk: float
    losing_trades_today: int
    losing_trades_limit: int
    largest_position_symbol: str
    largest_position_percent: float
    current_drawdown: float
    position_sizes: List[dict]


# Initialize FastAPI app
app = FastAPI(
    title="Trading Bot API",
    description="REST API for running backtests with scanner-selected symbols",
    version="1.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
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


def collect_scanner_symbols(universe: Dict, prioritize_volatile: bool = True) -> List[str]:
    """Collect all symbols from scanner_universe in universe.yaml.

    Args:
        universe: Loaded universe.yaml dict
        prioritize_volatile: If True, put high_volatility symbols first
    """
    scanner_universe = universe.get("scanner_universe", {})

    # Priority order: volatile stocks first (these are our money makers)
    priority_categories = [
        'high_volatility',  # BTC miners, meme stocks, crypto
        'clean_energy',     # EVs, solar, hydrogen
        'spacs_ipos',       # Recent IPOs, space stocks
        'cannabis',         # Volatile sector
        'biotech',          # High volatility pharma
    ]

    all_symbols = []
    seen = set()

    if prioritize_volatile:
        # Add priority categories first
        for category in priority_categories:
            symbols = scanner_universe.get(category, [])
            if isinstance(symbols, list):
                for s in symbols:
                    if s not in seen:
                        seen.add(s)
                        all_symbols.append(s)

        # Then add remaining categories
        for category, symbols in scanner_universe.items():
            if category not in priority_categories and isinstance(symbols, list):
                for s in symbols:
                    if s not in seen:
                        seen.add(s)
                        all_symbols.append(s)
    else:
        # Original order
        for category, symbols in scanner_universe.items():
            if isinstance(symbols, list):
                for s in symbols:
                    if s not in seen:
                        seen.add(s)
                        all_symbols.append(s)

    return all_symbols


def _compute_strategy_breakdown(trades: List[Dict]) -> List[StrategyBreakdown]:
    """Compute performance breakdown by strategy."""
    if not trades:
        return []

    from collections import defaultdict

    strategy_data = defaultdict(lambda: {
        'trades': 0, 'wins': 0, 'losses': 0,
        'total_pnl': 0.0, 'mfe_pcts': [], 'mae_pcts': []
    })

    for trade in trades:
        strategy = trade.get('strategy', 'Unknown') or 'Unknown'
        # Normalize strategy name (remove _SHORT suffix for grouping)
        if strategy.endswith('_SHORT'):
            strategy = strategy[:-6]

        pnl = trade.get('pnl', 0) or 0
        strategy_data[strategy]['trades'] += 1
        strategy_data[strategy]['total_pnl'] += pnl
        if pnl > 0:
            strategy_data[strategy]['wins'] += 1
        elif pnl < 0:
            strategy_data[strategy]['losses'] += 1

        mfe_pct = trade.get('mfe_pct', 0) or 0
        mae_pct = trade.get('mae_pct', 0) or 0
        strategy_data[strategy]['mfe_pcts'].append(mfe_pct)
        strategy_data[strategy]['mae_pcts'].append(mae_pct)

    result = []
    for strategy, data in strategy_data.items():
        trade_count = data['trades']
        win_rate = (data['wins'] / trade_count * 100) if trade_count > 0 else 0
        avg_pnl = data['total_pnl'] / trade_count if trade_count > 0 else 0
        avg_mfe = sum(data['mfe_pcts']) / len(data['mfe_pcts']) if data['mfe_pcts'] else 0
        avg_mae = sum(data['mae_pcts']) / len(data['mae_pcts']) if data['mae_pcts'] else 0

        result.append(StrategyBreakdown(
            strategy=strategy,
            trades=trade_count,
            wins=data['wins'],
            losses=data['losses'],
            win_rate=round(win_rate, 1),
            total_pnl=round(data['total_pnl'], 2),
            avg_pnl=round(avg_pnl, 2),
            avg_mfe_pct=round(avg_mfe, 2),
            avg_mae_pct=round(avg_mae, 2)
        ))

    # Sort by total_pnl descending (best first)
    result.sort(key=lambda x: x.total_pnl, reverse=True)
    return result


def _compute_exit_reason_breakdown(trades: List[Dict]) -> List[ExitReasonBreakdown]:
    """Compute performance breakdown by exit reason."""
    if not trades:
        return []

    from collections import defaultdict

    exit_data = defaultdict(lambda: {'count': 0, 'total_pnl': 0.0})
    total_trades = len(trades)

    for trade in trades:
        reason = trade.get('exit_reason', 'unknown') or 'unknown'
        pnl = trade.get('pnl', 0) or 0
        exit_data[reason]['count'] += 1
        exit_data[reason]['total_pnl'] += pnl

    result = []
    for reason, data in exit_data.items():
        count = data['count']
        avg_pnl = data['total_pnl'] / count if count > 0 else 0
        pct_of_trades = (count / total_trades * 100) if total_trades > 0 else 0

        result.append(ExitReasonBreakdown(
            exit_reason=reason,
            count=count,
            total_pnl=round(data['total_pnl'], 2),
            avg_pnl=round(avg_pnl, 2),
            pct_of_trades=round(pct_of_trades, 1)
        ))

    # Sort by count descending (most common first)
    result.sort(key=lambda x: x.count, reverse=True)
    return result


def _compute_symbol_breakdown(trades: List[Dict]) -> List[SymbolBreakdown]:
    """Compute performance breakdown by symbol, sorted worst to best."""
    if not trades:
        return []

    from collections import defaultdict

    symbol_data = defaultdict(lambda: {'trades': 0, 'wins': 0, 'total_pnl': 0.0})

    for trade in trades:
        symbol = trade.get('symbol', 'Unknown') or 'Unknown'
        pnl = trade.get('pnl', 0) or 0
        symbol_data[symbol]['trades'] += 1
        symbol_data[symbol]['total_pnl'] += pnl
        if pnl > 0:
            symbol_data[symbol]['wins'] += 1

    result = []
    for symbol, data in symbol_data.items():
        trade_count = data['trades']
        win_rate = (data['wins'] / trade_count * 100) if trade_count > 0 else 0
        avg_pnl = data['total_pnl'] / trade_count if trade_count > 0 else 0

        result.append(SymbolBreakdown(
            symbol=symbol,
            trades=trade_count,
            total_pnl=round(data['total_pnl'], 2),
            win_rate=round(win_rate, 1),
            avg_pnl=round(avg_pnl, 2)
        ))

    # Sort by total_pnl ascending (worst first to highlight drags)
    result.sort(key=lambda x: x.total_pnl)
    return result


def _compute_period_breakdown(trades: List[Dict]) -> List[PeriodBreakdown]:
    """Compute performance breakdown by date (daily)."""
    if not trades:
        return []

    from collections import defaultdict

    period_data = defaultdict(lambda: {'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0.0})

    for trade in trades:
        # Use exit_date for the period
        exit_date = trade.get('exit_date', '')
        if not exit_date:
            continue

        # Extract just the date portion (YYYY-MM-DD)
        if hasattr(exit_date, 'strftime'):
            date_str = exit_date.strftime('%Y-%m-%d')
        elif hasattr(exit_date, 'isoformat'):
            date_str = exit_date.isoformat()[:10]
        else:
            date_str = str(exit_date)[:10]

        pnl = trade.get('pnl', 0) or 0
        period_data[date_str]['trades'] += 1
        period_data[date_str]['total_pnl'] += pnl
        if pnl > 0:
            period_data[date_str]['wins'] += 1
        elif pnl < 0:
            period_data[date_str]['losses'] += 1

    result = []
    for date_str, data in period_data.items():
        trade_count = data['trades']
        win_rate = (data['wins'] / trade_count * 100) if trade_count > 0 else 0
        avg_pnl = data['total_pnl'] / trade_count if trade_count > 0 else 0

        result.append(PeriodBreakdown(
            date=date_str,
            trades=trade_count,
            wins=data['wins'],
            losses=data['losses'],
            win_rate=round(win_rate, 1),
            total_pnl=round(data['total_pnl'], 2),
            avg_pnl=round(avg_pnl, 2)
        ))

    # Sort by date ascending
    result.sort(key=lambda x: x.date)
    return result


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
        avg_bars_held=round(raw_metrics.get("avg_bars_held", 0), 1),
        drawdown_peak_date=raw_metrics.get("drawdown_peak_date"),
        drawdown_peak_value=round(raw_metrics.get("drawdown_peak_value", 0), 2),
        drawdown_trough_date=raw_metrics.get("drawdown_trough_date"),
        drawdown_trough_value=round(raw_metrics.get("drawdown_trough_value", 0), 2)
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
            bars_held=int(trade.get("bars_held", 0)),
            mfe_pct=round(trade.get("mfe_pct", 0), 2),
            mae_pct=round(trade.get("mae_pct", 0), 2)
        ))

    # Compute analytics breakdowns
    by_strategy = _compute_strategy_breakdown(raw_trades)
    by_exit_reason = _compute_exit_reason_breakdown(raw_trades)
    by_symbol = _compute_symbol_breakdown(raw_trades)
    by_period = _compute_period_breakdown(raw_trades)

    # Update metrics with days_traded
    metrics.days_traded = len(by_period)

    # Format worst daily drops
    worst_drops = []
    for drop in raw_metrics.get("worst_daily_drops", []):
        worst_drops.append(DailyDrop(
            date=drop.get("date", ""),
            open=round(drop.get("open", 0), 2),
            close=round(drop.get("close", 0), 2),
            high=round(drop.get("high", 0), 2),
            low=round(drop.get("low", 0), 2),
            change_pct=round(drop.get("change_pct", 0), 2),
            change_dollars=round(drop.get("change_dollars", 0), 2)
        ))

    return BacktestResponse(
        success=True,
        metrics=metrics,
        equity_curve=equity_curve,
        trades=trades,
        symbols_scanned=symbols_scanned,
        by_strategy=by_strategy,
        by_exit_reason=by_exit_reason,
        by_symbol=by_symbol,
        by_period=by_period,
        worst_daily_drops=worst_drops
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


@app.get("/api/equity-history", response_model=EquityHistoryResponse)
async def get_equity_history(period: str = "30D"):
    """Get historical portfolio equity values.

    Args:
        period: Time period for history. Options: "7D", "30D", "90D", "1Y", "ALL"

    Returns:
        EquityHistoryResponse with timestamps and equity values
    """
    # Validate period
    valid_periods = ["7D", "30D", "90D", "1Y", "ALL"]
    if period not in valid_periods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid period '{period}'. Must be one of: {valid_periods}"
        )

    try:
        broker = get_broker()
        history = broker.get_portfolio_history(period=period)

        # Convert to response format
        data = []
        for ts, eq in zip(history.timestamps, history.equity):
            # Format timestamp as ISO string
            if hasattr(ts, 'isoformat'):
                timestamp_str = ts.isoformat()
            else:
                timestamp_str = str(ts)

            data.append(EquityHistoryPoint(
                timestamp=timestamp_str,
                equity=round(eq, 2)
            ))

        return EquityHistoryResponse(
            data=data,
            period=period,
            base_value=round(history.base_value, 2)
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

        # FIX (Jan 2026): Detect actual process state instead of relying on in-memory state
        # This prevents duplicate bots from being started when API restarts
        actual_running = _is_bot_running()
        if actual_running and _bot_state["status"] == "stopped":
            # Process is running but state says stopped (API restarted)
            _bot_state["status"] = "running"
            _bot_state["last_action"] = "Bot detected running (recovered after API restart)"
        elif not actual_running and _bot_state["status"] == "running":
            # State says running but process died
            _bot_state["status"] = "stopped"
            _bot_state["last_action"] = "Bot stopped (process terminated)"

        return BotStatusResponse(
            status=_bot_state["status"],
            mode=mode,
            last_action=_bot_state["last_action"],
            last_action_time=_bot_state["last_action_time"],
            kill_switch_triggered=_bot_state["kill_switch_triggered"],
            watchlist=_bot_state.get("watchlist")
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


@app.post("/api/bot/start", response_model=BotStartResponse)
async def start_bot():
    """
    Start the trading bot with scanner-selected stocks.

    Flow:
    1. Check if bot is already running
    2. Check if market is open
    3. Run volatility scanner on available stocks
    4. Start bot with scanned symbols

    Returns 400 if market closed, scanner fails, or no stocks found.
    Returns 409 if bot is already running.
    """
    global _bot_state

    # FIX (Jan 2026): Check if bot is already running to prevent duplicates
    if _is_bot_running():
        raise HTTPException(
            status_code=409,
            detail={
                "reason": "already_running",
                "message": "Bot is already running. Stop it first before starting a new instance."
            }
        )

    # Step 1: Check market hours
    if not is_market_open():
        raise HTTPException(
            status_code=400,
            detail={
                "reason": "market_closed",
                "message": get_market_status_message()
            }
        )

    # Step 2: Run scanner
    try:
        universe = load_universe()
        scanner_symbols = collect_scanner_symbols(universe)

        config = load_config()
        scanner_config = config.get('volatility_scanner', {})
        scanner = VolatilityScanner(scanner_config)

        # Fetch data and scan
        fetcher = YFinanceDataFetcher()
        end_date = datetime.now()

        # Use get_historical_data with limit=200 (same as scanner/scan endpoint)
        # This ensures enough bars for ATR calculation (scanner needs ~98 bars minimum)
        historical_data = {}
        for symbol in scanner_symbols[:100]:  # Limit to avoid timeout
            try:
                data = fetcher.get_historical_data(
                    symbol,
                    timeframe='1Hour',
                    limit=200
                )
                if data is not None and not data.empty:
                    historical_data[symbol] = data
            except Exception:
                continue

        # Run scan
        results = scanner.scan_historical(
            date=end_date.strftime('%Y-%m-%d'),
            symbols=list(historical_data.keys()),
            historical_data=historical_data
        )

        if not results:
            raise HTTPException(
                status_code=400,
                detail={
                    "reason": "no_results",
                    "message": "Scanner found no stocks meeting volatility threshold. This is unexpected with 900+ stocks - check data feed."
                }
            )

        # Extract symbols from results (results is a list of symbol strings)
        if isinstance(results[0], dict):
            watchlist = [r['symbol'] for r in results[:10]]  # Top 10
        else:
            watchlist = results[:10]  # Already a list of symbols

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "reason": "scanner_error",
                "message": f"Scanner failed: {str(e)}"
            }
        )

    # Step 3: Start bot with scanned symbols
    try:
        symbols_arg = ",".join(watchlist)
        process = subprocess.Popen(
            ["python3", "bot.py", "--symbols", symbols_arg],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # FIX (Jan 2026): Save PID to prevent duplicate bots
        _write_bot_pid(process.pid)

        _bot_state["status"] = "running"
        _bot_state["watchlist"] = watchlist
        update_bot_state(status="running", last_action=f"Started with scanner: {', '.join(watchlist)} (PID: {process.pid})")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "reason": "bot_start_error",
                "message": f"Failed to start bot process: {str(e)}"
            }
        )

    return BotStartResponse(
        status="started",
        watchlist=watchlist,
        scanner_ran_at=datetime.now().isoformat(),
        message=f"Bot started with {len(watchlist)} scanned stocks: {', '.join(watchlist)}"
    )


@app.post("/api/bot/stop")
async def stop_bot():
    """Stop the trading bot."""
    global _bot_state

    # FIX (Jan 2026): Actually kill the bot process instead of just updating state
    pid = _read_bot_pid()
    was_running = _is_bot_running()

    if was_running:
        killed = _kill_bot_process()
        if killed:
            _bot_state["status"] = "stopped"
            _bot_state["last_action"] = f"Bot stopped (killed PID {pid})"
            _bot_state["last_action_time"] = datetime.now().isoformat()
            return {"success": True, "status": "stopped", "killed_pid": pid}
        else:
            raise HTTPException(
                status_code=500,
                detail={"reason": "kill_failed", "message": f"Failed to kill bot process (PID {pid})"}
            )
    else:
        # No process running, just update state
        _bot_state["status"] = "stopped"
        _bot_state["last_action"] = "Bot stopped (was not running)"
        _bot_state["last_action_time"] = datetime.now().isoformat()
        return {"success": True, "status": "stopped", "message": "Bot was not running"}


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
        from core.data import YFinanceDataFetcher

        config = load_config()
        universe = load_universe()
        symbols = collect_scanner_symbols(universe)

        # Fetch historical data for all symbols
        fetcher = YFinanceDataFetcher()
        historical_data = {}

        for symbol in symbols[:100]:  # Scan top 100 volatile symbols
            try:
                df = fetcher.get_historical_data(symbol, timeframe="1Hour", limit=200)
                if df is not None and not df.empty:
                    historical_data[symbol] = df
            except Exception:
                continue

        if not historical_data:
            return ScannerResponse(results=[], scanned_at=datetime.now().isoformat())

        # Run scanner
        scanner_config = config.get("volatility_scanner", {})
        scanner_config["top_n"] = top_n
        scanner = VolatilityScanner(scanner_config)

        today = datetime.now().strftime("%Y-%m-%d")
        top_symbols = scanner.scan_historical(today, list(historical_data.keys()), historical_data)

        # Build results with current prices
        scanner_results = []
        for symbol in top_symbols:
            df = historical_data.get(symbol)
            if df is not None and not df.empty:
                current_price = float(df['close'].iloc[-1])
                # Calculate ATR ratio
                high = df['high'].tail(14)
                low = df['low'].tail(14)
                close = df['close'].tail(14)
                tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
                atr = tr.mean()
                atr_ratio = atr / current_price if current_price > 0 else 0

                # Volume ratio
                vol_avg = df['volume'].tail(20).mean()
                vol_recent = df['volume'].tail(5).mean()
                vol_ratio = vol_recent / vol_avg if vol_avg > 0 else 1

                scanner_results.append(ScannerResult(
                    symbol=symbol,
                    atr_ratio=round(atr_ratio, 4),
                    volume_ratio=round(vol_ratio, 2),
                    composite_score=round(atr_ratio * vol_ratio, 4),
                    current_price=round(current_price, 2)
                ))

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

        # Override trailing stop settings from request
        if "trailing_stop" not in config:
            config["trailing_stop"] = {}
        config["trailing_stop"]["enabled"] = request.trailing_stop_enabled
        config["trailing_stop"]["activation_pct"] = request.trailing_activation_pct / 100.0  # Convert % to decimal
        config["trailing_stop"]["trail_pct"] = request.trailing_trail_pct / 100.0  # Convert % to decimal

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


def _format_hold_duration(entry_ts, exit_ts) -> str:
    """Format hold duration as human-readable string."""
    if not entry_ts or not exit_ts:
        return "N/A"

    try:
        # Parse timestamps if they're strings
        if isinstance(entry_ts, str):
            entry_ts = datetime.fromisoformat(entry_ts.replace('Z', '+00:00'))
        if isinstance(exit_ts, str):
            exit_ts = datetime.fromisoformat(exit_ts.replace('Z', '+00:00'))

        delta = exit_ts - entry_ts
        total_hours = delta.total_seconds() / 3600

        if total_hours < 1:
            return f"{int(delta.total_seconds() / 60)}m"
        elif total_hours < 24:
            return f"{int(total_hours)}h"
        else:
            days = int(total_hours // 24)
            hours = int(total_hours % 24)
            if hours > 0:
                return f"{days}d {hours}h"
            return f"{days}d"
    except Exception:
        return "N/A"


@app.get("/api/trades/history", response_model=TradeHistoryResponse)
async def get_trade_history(days: int = 30):
    """
    Get trade history for the specified number of days.

    Returns closed trades with P&L information.
    """
    try:
        logger = TradeLogger()
        df = logger.get_trade_history(days=days)

        if df.empty:
            return TradeHistoryResponse(trades=[], total_count=0)

        # Filter to only closed trades (have exit_price)
        closed_trades = df[df['status'] == 'closed'].copy()

        if closed_trades.empty:
            return TradeHistoryResponse(trades=[], total_count=0)

        trades = []
        for _, row in closed_trades.iterrows():
            # Determine side from action
            action = row.get('action', 'BUY')
            side = 'SHORT' if action == 'SHORT' else 'LONG'

            # Get prices
            entry_price = float(row.get('price', 0) or 0)
            exit_price = float(row.get('exit_price', 0) or 0)
            pnl = float(row.get('pnl', 0) or 0)

            # Calculate P&L percent
            value = float(row.get('value', 0) or 0)
            if value > 0:
                pnl_percent = (pnl / value) * 100
            elif entry_price > 0:
                qty = float(row.get('quantity', 0) or 0)
                if qty > 0:
                    pnl_percent = (pnl / (entry_price * qty)) * 100
                else:
                    pnl_percent = 0
            else:
                pnl_percent = 0

            # Format date
            timestamp = row.get('exit_timestamp') or row.get('timestamp')
            if hasattr(timestamp, 'strftime'):
                date_str = timestamp.strftime('%Y-%m-%d %H:%M')
            else:
                date_str = str(timestamp)[:16] if timestamp else 'N/A'

            # Calculate hold duration
            hold_duration = _format_hold_duration(
                row.get('timestamp'),
                row.get('exit_timestamp')
            )

            trades.append(TradeHistoryItem(
                id=int(row.get('id', 0)),
                date=date_str,
                symbol=str(row.get('symbol', '')),
                side=side,
                entryPrice=round(entry_price, 2),
                exitPrice=round(exit_price, 2),
                pnlDollar=round(pnl, 2),
                pnlPercent=round(pnl_percent, 2),
                holdDuration=hold_duration,
                strategy=str(row.get('strategy', 'Unknown') or 'Unknown')
            ))

        return TradeHistoryResponse(
            trades=trades,
            total_count=len(trades)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/activity", response_model=ActivityResponse)
async def get_activity(limit: int = 50):
    """
    Get recent activity feed combining trades and system events.
    """
    try:
        logger = TradeLogger()
        df = logger.get_trade_history(days=7)

        activities = []
        activity_id = 1

        if not df.empty:
            for _, row in df.iterrows():
                action = row.get('action', '')
                symbol = row.get('symbol', '')
                status = row.get('status', '')

                # Entry activity
                if action in ('BUY', 'SHORT'):
                    side = 'LONG' if action == 'BUY' else 'SHORT'
                    qty = int(row.get('quantity', 0) or 0)
                    price = float(row.get('price', 0) or 0)
                    strategy = row.get('strategy', 'Unknown') or 'Unknown'
                    confidence = row.get('confidence', 0) or 0

                    timestamp = row.get('timestamp')
                    if hasattr(timestamp, 'strftime'):
                        ts_str = timestamp.strftime('%H:%M:%S')
                    else:
                        ts_str = str(timestamp)[11:19] if timestamp else 'N/A'

                    activities.append(ActivityItem(
                        id=activity_id,
                        type='entry',
                        message=f"Opened {side} position: {symbol}",
                        details=f"{qty} shares @ ${price:.2f} | {strategy} strategy | {confidence}% confidence",
                        timestamp=ts_str
                    ))
                    activity_id += 1

                # Exit activity (for closed trades)
                if status == 'closed' and row.get('exit_price'):
                    pnl = float(row.get('pnl', 0) or 0)
                    exit_reason = row.get('exit_reason', 'Unknown') or 'Unknown'
                    pnl_sign = '+' if pnl >= 0 else ''

                    # Calculate P&L percent
                    value = float(row.get('value', 0) or 0)
                    if value > 0:
                        pnl_pct = (pnl / value) * 100
                    else:
                        pnl_pct = 0

                    exit_ts = row.get('exit_timestamp')
                    if hasattr(exit_ts, 'strftime'):
                        ts_str = exit_ts.strftime('%H:%M:%S')
                    else:
                        ts_str = str(exit_ts)[11:19] if exit_ts else 'N/A'

                    side = 'LONG' if action == 'BUY' else 'SHORT'
                    activities.append(ActivityItem(
                        id=activity_id,
                        type='exit',
                        message=f"Closed {side} position: {symbol}",
                        details=f"{exit_reason} | {pnl_sign}${pnl:.2f} ({pnl_sign}{pnl_pct:.1f}%)",
                        timestamp=ts_str
                    ))
                    activity_id += 1

        # Sort by timestamp descending and limit
        activities = activities[:limit]

        return ActivityResponse(
            activities=activities,
            total_count=len(activities)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk", response_model=RiskMetrics)
async def get_risk_metrics():
    """
    Get current risk monitoring metrics.
    """
    try:
        broker = get_broker()
        config = load_config()
        logger = TradeLogger()

        # Get account info
        account = broker.get_account()
        positions = broker.get_positions()

        # Get today's trades
        df = logger.get_trade_history(days=1)

        # Calculate daily loss from today's closed trades
        daily_loss = 0.0
        losing_trades_today = 0
        if not df.empty:
            closed_today = df[df['status'] == 'closed']
            if not closed_today.empty:
                pnl_values = closed_today['pnl'].fillna(0)
                daily_loss = abs(min(0, pnl_values.sum()))
                losing_trades_today = len(closed_today[closed_today['pnl'] < 0])

        # Risk config
        risk_config = config.get('risk', {})
        daily_loss_limit = risk_config.get('daily_loss_limit', 1000)
        losing_trades_limit = risk_config.get('max_losing_trades', 5)

        # Calculate position sizes
        portfolio_value = account.portfolio_value
        position_sizes = []
        largest_symbol = "None"
        largest_percent = 0.0

        for pos in positions:
            pct = (pos.market_value / portfolio_value * 100) if portfolio_value > 0 else 0
            position_sizes.append({
                "symbol": pos.symbol,
                "size": round(pct, 1)
            })
            if pct > largest_percent:
                largest_percent = pct
                largest_symbol = pos.symbol

        # Sort by size descending
        position_sizes.sort(key=lambda x: x['size'], reverse=True)

        # Calculate open risk (total unrealized loss exposure)
        total_unrealized = sum(pos.unrealized_pl for pos in positions)
        open_risk = abs(min(0, total_unrealized)) / portfolio_value * 100 if portfolio_value > 0 else 0

        # Calculate drawdown from daily P&L
        current_drawdown = abs(account.daily_pnl_percent * 100) if account.daily_pnl < 0 else 0

        return RiskMetrics(
            daily_loss=round(daily_loss, 2),
            daily_loss_limit=daily_loss_limit,
            open_risk=round(open_risk, 2),
            losing_trades_today=losing_trades_today,
            losing_trades_limit=losing_trades_limit,
            largest_position_symbol=largest_symbol,
            largest_position_percent=round(largest_percent, 1),
            current_drawdown=round(current_drawdown, 2),
            position_sizes=position_sizes[:10]  # Top 10
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
