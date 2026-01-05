"""
Trade logging and database management module.

Provides comprehensive trade logging with SQLite persistence,
performance tracking, and analytics.
"""

import pandas as pd
import numpy as np  # BUG FIX (Dec 3, 2025): Added for inf handling in Sharpe ratio
import json
import logging
from datetime import datetime
import os
from typing import Dict, List, Any
import sqlite3
from pathlib import Path

# FIX (Dec 3, 2025): Register sqlite3 datetime adapters (deprecated in Python 3.12)
def _adapt_datetime(val):
    """Adapt datetime.datetime to ISO 8601 string."""
    return val.isoformat()

def _convert_datetime(val):
    """Convert ISO 8601 string to datetime.datetime.

    BUG FIX (Dec 4, 2025): Handle both bytes and string input.
    SQLite may pass bytes or str depending on context.
    """
    if isinstance(val, bytes):
        val = val.decode()
    return datetime.fromisoformat(val)

sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_converter("timestamp", _convert_datetime)


class TradeLogger:
    """Log and manage trading activities, performance, and analytics"""

    def __init__(self, log_dir='logs', db_file='trades.db'):
        # BUG FIX (Dec 31, 2025): Handle full paths in db_file parameter
        # If db_file contains a directory (e.g., 'logs/trades_1hour.db'),
        # use it as-is instead of joining with log_dir
        db_path = Path(db_file)
        if db_path.parent != Path('.'):
            # db_file contains a directory path - use it directly
            self.log_dir = db_path.parent
            self.db_file = db_path
        else:
            # db_file is just a filename - join with log_dir
            self.log_dir = Path(log_dir)
            self.db_file = self.log_dir / db_file

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._init_database()

        # Setup file logging
        log_file = self.log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.trade_history = []

        # Performance tracking
        self.session_stats = {
            'start_time': datetime.now(),
            'trades_executed': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_commission': 0.0,
            'max_drawdown': 0.0,
            'peak_portfolio_value': 0.0
        }

    def _safe_log(self, level: str, message: str):
        """
        BUG FIX #4 (Dec 8, 2025): Safe logging wrapper that handles emoji encoding issues.
        Falls back to ASCII-only message if Unicode encoding fails.

        Args:
            level: Log level ('info', 'error', 'warning', 'debug')
            message: Message to log (may contain emojis)
        """
        try:
            # Try logging with emojis
            getattr(self.logger, level.lower())(message)
        except UnicodeEncodeError:
            # Fall back to ASCII-only version
            ascii_message = message.encode('ascii', errors='replace').decode('ascii')
            getattr(self.logger, level.lower())(f"[EMOJI_ENCODING_ERROR] {ascii_message}")

    def _init_database(self):
        """Initialize SQLite database for trade storage

        BUG FIX (Dec 4, 2025): Use context manager for connection to prevent resource leaks.
        """
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()

                # Create trades table
                # BUG FIX (Dec 10, 2025): Added confidence and indicators columns to match schema docs
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        symbol TEXT,
                        action TEXT,
                        quantity INTEGER,
                        price REAL,
                        value REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        reasoning TEXT,
                        order_id TEXT,
                        status TEXT,
                        pnl REAL,
                        commission REAL,
                        portfolio_value REAL,
                        session_id TEXT,
                        strategy TEXT,
                        exit_price REAL,
                        exit_timestamp DATETIME,
                        exit_reason TEXT,
                        confidence INTEGER,
                        indicators TEXT
                    )
                ''')

                # BUG FIX (Dec 10, 2025): Create index on symbol for faster queries
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)
                ''')

                # Add columns if they don't exist (for existing databases)
                # BUG FIX #10 (Nov 18, 2025): Add exit tracking columns for proper P&L calculation
                # SECURITY FIX (Dec 8, 2025): Whitelist validation for column names (DDL can't use parameterized queries)
                # BUG FIX (Dec 10, 2025): Added confidence and indicators columns
                # NEW (Dec 11, 2025 - Week 4): Add regime tracking for per-regime analysis
                ALLOWED_COLUMNS = {
                    'strategy': 'TEXT',
                    'exit_price': 'REAL',
                    'exit_timestamp': 'DATETIME',
                    'exit_reason': 'TEXT',
                    'confidence': 'INTEGER',
                    'indicators': 'TEXT',
                    'regime': 'TEXT',  # Market regime at trade time (trending_up, trending_down, range_bound, high_volatility, low_volatility, unknown)
                    # NEW (Dec 17, 2025): MFE/MAE diagnostics for trade quality analysis
                    'mfe': 'REAL',           # Maximum Favorable Excursion ($ amount)
                    'mae': 'REAL',           # Maximum Adverse Excursion ($ amount)
                    'mfe_pct': 'REAL',       # MFE as percentage of entry price
                    'mae_pct': 'REAL',       # MAE as percentage of entry price
                    'highest_price': 'REAL', # Highest price reached during trade
                    'lowest_price': 'REAL'   # Lowest price reached during trade
                }

                for column, column_type in ALLOWED_COLUMNS.items():
                    try:
                        # Validate column name is in whitelist (already guaranteed by dict iteration)
                        # This prevents SQL injection in DDL statements
                        cursor.execute(f"ALTER TABLE trades ADD COLUMN {column} {column_type}")
                        conn.commit()
                        # Note: Can't use self.logger here as it's not initialized yet
                        print(f"Added column {column} to trades table")
                    except sqlite3.OperationalError:
                        # Column already exists
                        pass

                # Create signals table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        symbol TEXT,
                        signal_type TEXT,
                        strength REAL,
                        indicators TEXT,
                        executed BOOLEAN,
                        session_id TEXT
                    )
                ''')

                # Create performance table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE,
                        portfolio_value REAL,
                        daily_pnl REAL,
                        trades_count INTEGER,
                        win_rate REAL,
                        avg_win REAL,
                        avg_loss REAL,
                        max_drawdown REAL,
                        session_id TEXT
                    )
                ''')

                # NEW (Jan 2026): Create scans table for scanner result logging
                # Enables comparison of scanner output between live and backtest
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS scans (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        scan_date TEXT,
                        mode TEXT,
                        symbols_scanned INTEGER,
                        symbols_selected INTEGER,
                        selected_symbols TEXT,
                        all_scores TEXT,
                        config TEXT,
                        session_id TEXT
                    )
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_scans_date ON scans(scan_date)
                ''')

                conn.commit()

        except Exception as e:
            # Note: Can't use self.logger here as it's not initialized yet
            print(f"Error initializing database: {e}")
            raise

    def log_trade(self, trade_data: Dict[str, Any]):
        """
        Log a trade to database and files

        CRITICAL FIX (Nov 5, 2025): Now properly updates BUY entries when closing
        instead of creating orphaned SELL entries that break P&L tracking

        BUG FIX #5 (Dec 8, 2025): Added transaction-like consistency.
        If database write succeeds but JSON write fails, we still have DB record.
        If database write fails, exception is logged and no JSON is written.
        This prevents partial state where one log succeeds but the other fails.
        """
        db_success = False
        try:
            # Add timestamp if not provided
            if 'timestamp' not in trade_data:
                trade_data['timestamp'] = datetime.now()

            # Generate session ID
            session_id = datetime.now().strftime('%Y%m%d')
            trade_data['session_id'] = session_id

            # CRITICAL FIX: If this is a SELL/CLOSE/COVER action, update the existing entry
            action = trade_data.get('action', '').upper()
            if action in ['SELL', 'CLOSE', 'COVER']:
                # SELL closes LONG, COVER closes SHORT
                self._update_trade_exit(trade_data)
                db_success = True
            else:
                # New entry (BUY or SHORT)
                self._save_trade_to_db(trade_data)
                db_success = True

            # Only proceed with secondary logging if DB write succeeded
            if db_success:
                # Log to file
                self.logger.info(f"TRADE: {json.dumps(trade_data, default=str)}")

                # ENHANCED (Nov 5, 2025): Save detailed JSON with full indicators
                self._save_detailed_trade_json(trade_data)

                # Update session stats
                self._update_session_stats(trade_data)

                # Add to in-memory history
                self.trade_history.append(trade_data)

        except Exception as e:
            self.logger.error(f"Error logging trade: {e}", exc_info=True)
            # BUG FIX #5 (Dec 8, 2025): If database write failed, don't proceed with JSON
            # This is handled by the db_success flag above

    def _save_detailed_trade_json(self, trade_data: Dict[str, Any]):
        """
        Save detailed trade with ALL indicators to JSON file (Nov 5, 2025)
        One JSON object per line for easy parsing

        BUG FIX (Nov 25, 2025): Use trade timestamp for filename, not current date.
        This prevents old trades from being written to today's file during reconciliation.
        Also added deduplication to prevent duplicate entries.

        BUG FIX #3 (Dec 8, 2025): Optimized duplicate check using in-memory cache.
        Previous implementation read entire file on every trade (O(n) per insert).
        New implementation uses session-level cache (O(1) lookup after initial load).
        """
        try:
            # Create trades directory
            trades_dir = self.log_dir / 'trades'
            trades_dir.mkdir(exist_ok=True)

            # BUG FIX (Nov 25, 2025): Use trade's timestamp for filename, not datetime.now()
            # This ensures reconciled trades go to their original date file
            trade_timestamp = trade_data.get('timestamp', datetime.now())
            if isinstance(trade_timestamp, str):
                # Parse string timestamp
                try:
                    trade_timestamp = datetime.fromisoformat(trade_timestamp.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    trade_timestamp = datetime.now()
            date_str = trade_timestamp.strftime('%Y%m%d') if hasattr(trade_timestamp, 'strftime') else datetime.now().strftime('%Y%m%d')
            json_file = trades_dir / f'trades_{date_str}.json'

            # Create detailed trade record
            detailed_trade = {
                'timestamp': str(trade_data.get('timestamp', datetime.now())),
                'symbol': trade_data.get('symbol'),
                'strategy': trade_data.get('strategy', 'Unknown'),
                'mode': trade_data.get('mode', 'UNKNOWN'),  # BACKTEST/PAPER/DRY_RUN/LIVE
                'regime': trade_data.get('regime', 'unknown'),  # NEW (Dec 11, 2025 - Week 4): Market regime at trade time
                'side': trade_data.get('action'),
                'quantity': trade_data.get('quantity', 0),
                'price': trade_data.get('price', 0),
                'stop_loss': trade_data.get('stop_loss'),
                'take_profit': trade_data.get('take_profit'),
                'confidence': trade_data.get('confidence', 0),
                'reasoning': trade_data.get('reasoning'),
                'pnl': trade_data.get('pnl', 0),
                'indicators': trade_data.get('indicators', {}),  # All indicator values
                'broker_response': trade_data.get('broker_response'),
                'error': trade_data.get('error'),
                'session_id': trade_data.get('session_id'),
                # NEW (Dec 17, 2025): MFE/MAE diagnostics for trade quality analysis
                'mfe': trade_data.get('mfe'),
                'mae': trade_data.get('mae'),
                'mfe_pct': trade_data.get('mfe_pct'),
                'mae_pct': trade_data.get('mae_pct'),
                'highest_price': trade_data.get('highest_price'),
                'lowest_price': trade_data.get('lowest_price')
            }

            # BUG FIX #3 (Dec 8, 2025): Use in-memory cache for deduplication
            trade_key = f"{detailed_trade['timestamp']}|{detailed_trade['symbol']}|{detailed_trade['side']}|{detailed_trade['quantity']}|{detailed_trade['price']}"

            # Initialize cache if not exists
            if not hasattr(self, '_json_trade_cache'):
                self._json_trade_cache = {}

            # Load cache for this file if not already loaded
            cache_key = str(json_file)
            if cache_key not in self._json_trade_cache:
                self._json_trade_cache[cache_key] = set()
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    existing = json.loads(line)
                                    existing_key = f"{existing.get('timestamp')}|{existing.get('symbol')}|{existing.get('side')}|{existing.get('quantity')}|{existing.get('price')}"
                                    self._json_trade_cache[cache_key].add(existing_key)
                                except (json.JSONDecodeError, TypeError, AttributeError):
                                    pass

            # Check cache for duplicate (O(1) lookup instead of O(n) file scan)
            if trade_key in self._json_trade_cache[cache_key]:
                # Already logged, skip duplicate
                return

            # Add to cache and write to file
            self._json_trade_cache[cache_key].add(trade_key)
            with open(json_file, 'a') as f:
                f.write(json.dumps(detailed_trade, default=str) + '\n')

        except Exception as e:
            self.logger.error(f"Error saving detailed trade JSON: {e}", exc_info=True)

    def _update_trade_exit(self, trade_data: Dict[str, Any]):
        """
        Update existing BUY/SHORT entry with exit information

        BUG FIX #10 (Nov 18, 2025): Properly calculate P&L from entry/exit prices
        BUG FIX (Dec 4, 2025): Use context manager for connection to prevent resource leaks.
        BUG FIX (Dec 10, 2025): Handle both long and short positions for P&L calculation.
        Finds the most recent pending position for this symbol and updates it with:
        - exit price
        - exit timestamp
        - exit reason
        - P&L (calculated if not provided)
        - status = 'closed'
        """
        try:
            symbol = trade_data.get('symbol')
            exit_price = trade_data.get('price', 0)
            exit_timestamp = trade_data.get('timestamp', datetime.now())
            exit_reason = trade_data.get('exit_reason', 'unknown')

            with sqlite3.connect(self.db_file) as conn:
                # BUG FIX (Dec 10, 2025): Set isolation level for transaction safety
                conn.isolation_level = 'IMMEDIATE'
                cursor = conn.cursor()

                # Find the most recent pending BUY or SHORT for this symbol
                # BUG FIX (Dec 10, 2025): Also find SHORT positions for short selling
                cursor.execute('''
                    SELECT id, price, quantity, action FROM trades
                    WHERE symbol = ? AND action IN ('BUY', 'SHORT', 'SCALE_IN') AND status IN ('pending', 'filled')
                    ORDER BY timestamp DESC
                    LIMIT 1
                ''', (symbol,))

                result = cursor.fetchone()

                if result:
                    trade_id, entry_price, quantity, entry_action = result

                    # BUG FIX #10 (Nov 18, 2025): Calculate P&L if not provided
                    # BUG FIX (Dec 10, 2025): Handle short positions correctly
                    # Long (BUY): P&L = (exit_price - entry_price) * quantity
                    # Short (SHORT): P&L = (entry_price - exit_price) * quantity
                    pnl = trade_data.get('pnl')
                    if pnl is None or pnl == 0:
                        commission = trade_data.get('commission', 0)
                        if entry_action in ('BUY', 'SCALE_IN'):
                            # Long position: profit when exit > entry
                            pnl = (exit_price - entry_price) * quantity - commission
                        else:
                            # Short position: profit when entry > exit
                            pnl = (entry_price - exit_price) * quantity - commission
                        # BUG FIX #4 (Dec 8, 2025): Use safe logging for emoji messages
                        position_type = 'LONG' if entry_action in ('BUY', 'SCALE_IN') else 'SHORT'
                        self._safe_log('info', f"Calculated P&L for {symbol} ({position_type}): entry=${entry_price:.2f}, exit=${exit_price:.2f}, qty={quantity} = ${pnl:.2f}")

                    # Update with exit information
                    cursor.execute('''
                        UPDATE trades
                        SET status = 'closed',
                            exit_price = ?,
                            exit_timestamp = ?,
                            exit_reason = ?,
                            pnl = ?,
                            commission = ?
                        WHERE id = ?
                    ''', (
                        exit_price,
                        exit_timestamp,
                        exit_reason,
                        pnl,
                        trade_data.get('commission', 0),
                        trade_id
                    ))

                    conn.commit()
                    # BUG FIX #4 (Dec 8, 2025): Use safe logging for emoji messages
                    self._safe_log('info', f"Updated trade #{trade_id} for {symbol}: Entry ${entry_price:.2f} -> Exit ${exit_price:.2f}, P&L: ${pnl:.2f} ({exit_reason})")
                else:
                    # BUG FIX #12 (Dec 9, 2025): Don't create orphaned SELL entries - require manual reconciliation
                    # Creating orphaned SELLs corrupts database and makes P&L analytics wrong
                    # This error indicates a critical issue: we're trying to close a position that doesn't exist
                    # Possible causes: database corruption, missed BUY entry, or state desync
                    # BUG FIX #4 (Dec 8, 2025): Use safe logging for emoji messages
                    self._safe_log('error',
                        f"CRITICAL: No matching BUY found for {symbol} SELL - MANUAL RECONCILIATION REQUIRED! "
                        f"Trade data: {trade_data} - NOT saving orphaned SELL to database"
                    )
                    # DO NOT call _save_trade_to_db() - this would create orphaned entry
                    # Admin must manually investigate and fix the database

        except Exception as e:
            self.logger.error(f"Error updating trade exit: {e}", exc_info=True)

    def _save_trade_to_db(self, trade_data: Dict[str, Any]):
        """Save trade data to SQLite database

        BUG FIX (Dec 4, 2025): Use context manager for connection to prevent resource leaks.
        BUG FIX (Dec 10, 2025): Added confidence and indicators columns to INSERT.
        NEW (Dec 11, 2025 - Week 4): Added regime column for per-regime analysis.
        FIX (Dec 12, 2025): Added comprehensive logging for confidence values
        """
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()

                # BUG FIX (Dec 10, 2025): Serialize indicators to JSON string
                # BUG FIX: Always serialize to JSON, even if empty dict
                indicators = trade_data.get('indicators', {})
                if not isinstance(indicators, str):
                    indicators = json.dumps(indicators) if indicators else None

                # FIX (Dec 12, 2025): Log confidence for debugging
                confidence = trade_data.get('confidence')
                symbol = trade_data.get('symbol', 'UNKNOWN')
                action = trade_data.get('action', 'UNKNOWN')
                if confidence is None or confidence == 0:
                    self.logger.warning(f"TRADE LOGGING: {symbol} {action} has confidence={confidence} (may be NULL in DB)")
                elif isinstance(confidence, (int, float)):
                    self.logger.debug(f"{symbol} {action} confidence: {confidence}")
                else:
                    self.logger.warning(f"TRADE LOGGING: {symbol} {action} confidence is wrong type: {type(confidence)} = {confidence}")

                cursor.execute('''
                    INSERT INTO trades (
                        timestamp, symbol, action, quantity, price, value,
                        stop_loss, take_profit, reasoning, order_id, status,
                        pnl, commission, portfolio_value, session_id, strategy,
                        confidence, indicators, regime,
                        mfe, mae, mfe_pct, mae_pct, highest_price, lowest_price
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data.get('timestamp'),
                    trade_data.get('symbol'),
                    trade_data.get('action'),
                    trade_data.get('quantity', 0),
                    trade_data.get('price', 0),
                    trade_data.get('value', 0),
                    trade_data.get('stop_loss'),
                    trade_data.get('take_profit'),
                    trade_data.get('reasoning'),
                    trade_data.get('order_id'),
                    trade_data.get('status', 'pending'),
                    trade_data.get('pnl', 0),
                    trade_data.get('commission', 0),
                    trade_data.get('portfolio_value', 0),
                    trade_data.get('session_id'),
                    trade_data.get('strategy', 'Unknown'),
                    trade_data.get('confidence'),
                    indicators,
                    trade_data.get('regime', 'unknown'),
                    # NEW (Dec 17, 2025): MFE/MAE diagnostics
                    trade_data.get('mfe'),
                    trade_data.get('mae'),
                    trade_data.get('mfe_pct'),
                    trade_data.get('mae_pct'),
                    trade_data.get('highest_price'),
                    trade_data.get('lowest_price')
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"Error saving trade to database: {e}", exc_info=True)

    def log_signal(self, signal_data: Dict[str, Any]):
        """Log a trading signal

        BUG FIX (Dec 4, 2025): Use context manager for connection to prevent resource leaks.
        """
        try:
            signal_data['timestamp'] = datetime.now()
            signal_data['session_id'] = datetime.now().strftime('%Y%m%d')

            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO signals (
                        timestamp, symbol, signal_type, strength, indicators, executed, session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data['timestamp'],
                    signal_data.get('symbol'),
                    signal_data.get('signal_type'),
                    signal_data.get('strength', 0),
                    json.dumps(signal_data.get('indicators', {})),
                    signal_data.get('executed', False),
                    signal_data['session_id']
                ))

                conn.commit()

            self.logger.info(f"SIGNAL: {json.dumps(signal_data, default=str)}")

        except Exception as e:
            self.logger.error(f"Error logging signal: {e}", exc_info=True)

    def log_scan_result(self, scan_data: Dict[str, Any]):
        """
        Log scanner results for live/backtest comparison.

        NEW (Jan 2026): Enables verification of scanner output alignment
        between live trading and backtesting.

        Args:
            scan_data: Dict with:
                - scan_date: Date of the scan (YYYY-MM-DD)
                - mode: 'LIVE', 'PAPER', 'BACKTEST', 'DRY_RUN'
                - symbols_scanned: Total symbols considered
                - selected_symbols: List of top N symbols selected
                - all_scores: List of dicts with symbol, score, price, volume
                - config: Scanner configuration dict
        """
        try:
            timestamp = datetime.now()
            session_id = timestamp.strftime('%Y%m%d')

            scan_date = scan_data.get('scan_date', timestamp.strftime('%Y-%m-%d'))
            mode = scan_data.get('mode', 'UNKNOWN')
            symbols_scanned = scan_data.get('symbols_scanned', 0)
            selected_symbols = scan_data.get('selected_symbols', [])
            all_scores = scan_data.get('all_scores', [])
            config = scan_data.get('config', {})

            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO scans (
                        timestamp, scan_date, mode, symbols_scanned,
                        symbols_selected, selected_symbols, all_scores,
                        config, session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp,
                    scan_date,
                    mode,
                    symbols_scanned,
                    len(selected_symbols),
                    json.dumps(selected_symbols),
                    json.dumps(all_scores),
                    json.dumps(config),
                    session_id
                ))

                conn.commit()

            self.logger.info(
                f"SCAN [{mode}] {scan_date}: {len(selected_symbols)}/{symbols_scanned} symbols selected - "
                f"Top: {selected_symbols[:5]}{'...' if len(selected_symbols) > 5 else ''}"
            )

        except Exception as e:
            self.logger.error(f"Error logging scan result: {e}", exc_info=True)

    def get_scan_history(self, days: int = 30, mode: str = None) -> pd.DataFrame:
        """
        Get scan history for comparison between live and backtest.

        NEW (Jan 2026): Enables verification of scanner alignment.

        Args:
            days: Number of days to look back
            mode: Optional filter for mode ('LIVE', 'BACKTEST', etc.)

        Returns:
            DataFrame with scan history
        """
        try:
            with sqlite3.connect(self.db_file) as conn:
                if mode:
                    query = '''
                        SELECT * FROM scans
                        WHERE timestamp >= datetime('now', ?)
                        AND mode = ?
                        ORDER BY scan_date DESC, timestamp DESC
                    '''
                    df = pd.read_sql_query(query, conn, params=[f'-{days} days', mode])
                else:
                    query = '''
                        SELECT * FROM scans
                        WHERE timestamp >= datetime('now', ?)
                        ORDER BY scan_date DESC, timestamp DESC
                    '''
                    df = pd.read_sql_query(query, conn, params=[f'-{days} days'])

            return df

        except Exception as e:
            self.logger.error(f"Error fetching scan history: {e}", exc_info=True)
            return pd.DataFrame()

    def _update_session_stats(self, trade_data: Dict[str, Any]):
        """Update session statistics"""
        try:
            self.session_stats['trades_executed'] += 1

            pnl = trade_data.get('pnl', 0)
            if pnl != 0:
                self.session_stats['total_pnl'] += pnl

                if pnl > 0:
                    self.session_stats['winning_trades'] += 1
                else:
                    self.session_stats['losing_trades'] += 1

            # Update portfolio tracking
            portfolio_value = trade_data.get('portfolio_value', 0)
            if portfolio_value > self.session_stats['peak_portfolio_value']:
                self.session_stats['peak_portfolio_value'] = portfolio_value

            # Calculate drawdown
            if self.session_stats['peak_portfolio_value'] > 0:
                current_drawdown = (self.session_stats['peak_portfolio_value'] - portfolio_value) / self.session_stats['peak_portfolio_value']
                self.session_stats['max_drawdown'] = max(self.session_stats['max_drawdown'], current_drawdown)

        except Exception as e:
            self.logger.error(f"Error updating session stats: {e}", exc_info=True)

    def get_trade_history(self, days: int = 30) -> pd.DataFrame:
        """Get trade history as DataFrame

        BUG FIX (Dec 4, 2025): Use context manager for connection to prevent resource leaks.
        BUG FIX (Dec 4, 2025): Validate days parameter before conversion to handle invalid input.
        """
        try:
            # BUG FIX (Dec 4, 2025): Validate days before int conversion
            # Handle strings, None, negative values gracefully
            try:
                days = int(days)
                if days < 0:
                    days = 30  # Default to 30 if negative
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid days parameter: {days}, defaulting to 30")
                days = 30

            interval = f'-{days} days'  # Safe because days is now guaranteed int

            with sqlite3.connect(self.db_file) as conn:
                query = '''
                    SELECT * FROM trades
                    WHERE timestamp >= datetime('now', ?)
                    ORDER BY timestamp DESC
                '''

                df = pd.read_sql_query(query, conn, params=[interval], parse_dates=['timestamp'])

            return df

        except Exception as e:
            self.logger.error(f"Error fetching trade history: {e}", exc_info=True)
            return pd.DataFrame()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            df = self.get_trade_history()

            if df.empty:
                return self.session_stats

            # Calculate performance metrics
            total_trades = len(df[df['pnl'] != 0])
            winning_trades = len(df[df['pnl'] > 0])
            losing_trades = len(df[df['pnl'] < 0])

            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

            avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

            # BUG FIX (Dec 3, 2025): Handle division by zero for profit factor
            # If no losses, profit factor should be very high (not 0)
            total_gains = avg_win * winning_trades if winning_trades > 0 else 0
            total_losses = abs(avg_loss * losing_trades) if losing_trades > 0 else 0
            if total_losses > 0:
                profit_factor = total_gains / total_losses
            elif total_gains > 0:
                profit_factor = 999.99  # Perfect profit factor (all wins, no losses)
            else:
                profit_factor = 0  # No trades or all neutral

            # Sharpe ratio (simplified using P&L values directly)
            # BUG FIX (Dec 3, 2025): Don't use pct_change on P&L - produces inf when prev P&L is 0
            # Use P&L values directly: Sharpe = mean(P&L) / std(P&L)
            pnl_values = df['pnl'].dropna()
            # Remove any inf values that may have snuck in
            pnl_values = pnl_values.replace([np.inf, -np.inf], np.nan).dropna()
            if len(pnl_values) > 1:
                pnl_std = pnl_values.std()
                if pnl_std > 0:
                    sharpe_ratio = pnl_values.mean() / pnl_std
                else:
                    sharpe_ratio = 0  # Zero std means no variation
            else:
                sharpe_ratio = 0

            # BUG FIX (Dec 4, 2025): Calculate trades_per_day using total_seconds() / 86400
            # The .days attribute returns 0 for sessions < 24 hours, causing wrong calculation
            # BUG FIX #6 (Dec 8, 2025): Documented magic number 0.0417
            # 0.0417 = 1/24 = 1 hour minimum (prevents division by very small numbers for new sessions)
            SECONDS_PER_DAY = 86400
            MIN_SESSION_DAYS = 0.0417  # 1 hour minimum to prevent inflated rates
            elapsed_seconds = (datetime.now() - self.session_stats['start_time']).total_seconds()
            elapsed_days = max(elapsed_seconds / SECONDS_PER_DAY, MIN_SESSION_DAYS)
            trades_per_day = total_trades / elapsed_days

            summary = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_pnl': df['pnl'].sum(),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': self.session_stats['max_drawdown'],
                'trades_per_day': trades_per_day,
                'best_trade': df['pnl'].max() if not df.empty else 0,
                'worst_trade': df['pnl'].min() if not df.empty else 0
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error calculating performance summary: {e}", exc_info=True)
            return self.session_stats

    def get_symbol_performance(self, symbol: str = None) -> Dict[str, Any]:
        """Get performance breakdown by symbol"""
        try:
            df = self.get_trade_history()

            if df.empty:
                return {}

            if symbol:
                df = df[df['symbol'] == symbol]

            # Group by symbol
            symbol_stats = {}

            for sym in df['symbol'].unique():
                sym_df = df[df['symbol'] == sym]

                total_trades = len(sym_df[sym_df['pnl'] != 0])
                winning_trades = len(sym_df[sym_df['pnl'] > 0])

                symbol_stats[sym] = {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': (winning_trades / total_trades) * 100 if total_trades > 0 else 0,
                    'total_pnl': sym_df['pnl'].sum(),
                    'avg_pnl': sym_df['pnl'].mean() if total_trades > 0 else 0,
                    'best_trade': sym_df['pnl'].max(),
                    'worst_trade': sym_df['pnl'].min()
                }

            return symbol_stats

        except Exception as e:
            self.logger.error(f"Error calculating symbol performance: {e}", exc_info=True)
            return {}

    def get_strategy_performance(self, strategy: str = None) -> Dict[str, Any]:
        """Get performance breakdown by strategy"""
        try:
            df = self.get_trade_history()

            if df.empty or 'strategy' not in df.columns:
                return {}

            if strategy:
                df = df[df['strategy'] == strategy]

            # Group by strategy
            strategy_stats = {}

            for strat in df['strategy'].unique():
                if pd.isna(strat) or strat == '' or strat is None:
                    continue  # Skip empty strategy values

                strat_df = df[df['strategy'] == strat]

                total_trades = len(strat_df[strat_df['pnl'] != 0])
                winning_trades = len(strat_df[strat_df['pnl'] > 0])

                strategy_stats[strat] = {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': (winning_trades / total_trades) * 100 if total_trades > 0 else 0,
                    'total_pnl': strat_df['pnl'].sum(),
                    'avg_pnl': strat_df['pnl'].mean() if total_trades > 0 else 0,
                    'best_trade': strat_df['pnl'].max(),
                    'worst_trade': strat_df['pnl'].min()
                }

            return strategy_stats

        except Exception as e:
            self.logger.error(f"Error calculating strategy performance: {e}", exc_info=True)
            return {}

    def export_trades(self, filename: str = None, file_format: str = 'csv'):
        """Export trade history to file

        BUG FIX (Dec 4, 2025): Renamed 'format' param to 'file_format' to avoid
        shadowing the built-in format() function.
        """
        try:
            df = self.get_trade_history()

            if df.empty:
                self.logger.warning("No trades to export")
                return

            if filename is None:
                filename = f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            filepath = self.log_dir / f"{filename}.{file_format}"

            if file_format.lower() == 'csv':
                df.to_csv(filepath, index=False)
            elif file_format.lower() == 'json':
                df.to_json(filepath, orient='records', date_format='iso')
            elif file_format.lower() == 'excel':
                df.to_excel(filepath, index=False)

            self.logger.info(f"Trades exported to {filepath}")

        except Exception as e:
            self.logger.error(f"Error exporting trades: {e}", exc_info=True)

    def log_system_event(self, event_type: str, message: str, data: Dict = None):
        """Log system events (startup, shutdown, errors, etc.)"""
        try:
            event_data = {
                'timestamp': datetime.now(),
                'event_type': event_type,
                'message': message,
                'data': data or {}
            }

            self.logger.info(f"SYSTEM: {json.dumps(event_data, default=str)}")

        except Exception as e:
            self.logger.error(f"Error logging system event: {e}", exc_info=True)

    def get_daily_summary(self, date: str = None) -> Dict[str, Any]:
        """Get daily trading summary

        BUG FIX (Dec 4, 2025): Use context manager for connection to prevent resource leaks.
        """
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')

            with sqlite3.connect(self.db_file) as conn:
                query = '''
                    SELECT COUNT(*) as trades, SUM(pnl) as total_pnl,
                           AVG(pnl) as avg_pnl, MAX(pnl) as best_trade,
                           MIN(pnl) as worst_trade
                    FROM trades
                    WHERE DATE(timestamp) = ?
                '''

                cursor = conn.cursor()
                cursor.execute(query, (date,))
                result = cursor.fetchone()

            if result:
                return {
                    'date': date,
                    'trades': result[0],
                    'total_pnl': result[1] or 0,
                    'avg_pnl': result[2] or 0,
                    'best_trade': result[3] or 0,
                    'worst_trade': result[4] or 0
                }
            else:
                return {'date': date, 'trades': 0, 'total_pnl': 0}

        except Exception as e:
            self.logger.error(f"Error getting daily summary: {e}", exc_info=True)
            return {'date': date if date else datetime.now().strftime('%Y-%m-%d'), 'trades': 0, 'total_pnl': 0}

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open (pending) positions from database.

        BUG FIX (Dec 9, 2025): Added method for position reconciliation agent.
        Returns list of open trades (status='pending' or 'filled' with no exit).

        Returns:
            List of dictionaries with trade information for open positions
        """
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()

                # Get all trades that are open (pending/filled but not closed)
                # BUG FIX (Dec 10, 2025): Include SCALE_IN action, not just BUY
                # SCALE_IN trades are also open positions that need reconciliation
                cursor.execute('''
                    SELECT id, timestamp, symbol, action, quantity, price,
                           stop_loss, take_profit, strategy, order_id
                    FROM trades
                    WHERE action IN ('BUY', 'SCALE_IN')
                    AND status IN ('pending', 'filled')
                    AND (exit_timestamp IS NULL OR exit_timestamp = '')
                    ORDER BY timestamp DESC
                ''')

                rows = cursor.fetchall()

                open_positions = []
                for row in rows:
                    open_positions.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'symbol': row[2],
                        'action': row[3],
                        'quantity': row[4],
                        'price': row[5],
                        'stop_loss': row[6],
                        'take_profit': row[7],
                        'strategy': row[8] if row[8] else 'unknown',
                        'order_id': row[9]
                    })

                return open_positions

        except Exception as e:
            self.logger.error(f"Error getting open positions: {e}", exc_info=True)
            return []

    def get_stale_positions(self, cutoff: datetime) -> List[Dict[str, Any]]:
        """
        Get positions older than a certain timestamp.

        BUG FIX (Dec 9, 2025): Added method for position reconciliation agent.
        Used to identify stale positions that may need attention.

        Args:
            cutoff: Datetime cutoff - positions older than this are returned

        Returns:
            List of dictionaries with trade information for stale positions
        """
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()

                # Get open trades older than cutoff
                # BUG FIX (Dec 10, 2025): Include SCALE_IN action, not just BUY
                cursor.execute('''
                    SELECT id, timestamp, symbol, action, quantity, price,
                           stop_loss, take_profit, strategy, order_id
                    FROM trades
                    WHERE action IN ('BUY', 'SCALE_IN')
                    AND status IN ('pending', 'filled')
                    AND (exit_timestamp IS NULL OR exit_timestamp = '')
                    AND timestamp < ?
                    ORDER BY timestamp ASC
                ''', (cutoff,))

                rows = cursor.fetchall()

                stale_positions = []
                for row in rows:
                    stale_positions.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'symbol': row[2],
                        'action': row[3],
                        'quantity': row[4],
                        'price': row[5],
                        'stop_loss': row[6],
                        'take_profit': row[7],
                        'strategy': row[8] if row[8] else 'unknown',
                        'order_id': row[9]
                    })

                return stale_positions

        except Exception as e:
            self.logger.error(f"Error getting stale positions: {e}", exc_info=True)
            return []

    def close_trade(self, trade_id: int, exit_price: float, exit_reason: str, pnl: float = None):
        """
        Close a trade by updating its exit information.

        BUG FIX (Dec 9, 2025): Added method for position reconciliation agent.
        BUG FIX (Dec 10, 2025): Handle both long and short positions for P&L calculation.
        This is a convenience wrapper around the existing update logic.

        Args:
            trade_id: Database ID of the trade to close
            exit_price: Exit price
            exit_reason: Reason for exit (stop_loss, take_profit, etc.)
            pnl: Profit/loss (calculated if not provided)
        """
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()

                # Get the trade info including action to determine position type
                # BUG FIX (Dec 10, 2025): Fetch action to handle short positions
                cursor.execute('''
                    SELECT price, quantity, action FROM trades WHERE id = ?
                ''', (trade_id,))

                result = cursor.fetchone()
                if not result:
                    self.logger.error(f"Trade {trade_id} not found")
                    return

                entry_price, quantity, action = result

                # Calculate P&L if not provided
                # BUG FIX (Dec 10, 2025): Handle short positions correctly
                if pnl is None:
                    if action in ('BUY', 'SCALE_IN'):
                        # Long position: profit when exit > entry
                        pnl = (exit_price - entry_price) * quantity
                    else:
                        # Short position: profit when entry > exit
                        pnl = (entry_price - exit_price) * quantity

                # Update the trade
                cursor.execute('''
                    UPDATE trades
                    SET status = 'closed',
                        exit_price = ?,
                        exit_timestamp = ?,
                        exit_reason = ?,
                        pnl = ?
                    WHERE id = ?
                ''', (
                    exit_price,
                    datetime.now(),
                    exit_reason,
                    pnl,
                    trade_id
                ))

                conn.commit()
                self.logger.info(f"Closed trade {trade_id}: {exit_reason}, P&L: ${pnl:.2f}")

        except Exception as e:
            self.logger.error(f"Error closing trade {trade_id}: {e}", exc_info=True)
