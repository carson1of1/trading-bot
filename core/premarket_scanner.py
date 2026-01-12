"""
Pre-market Scanner - Identifies gap-ups before market open.

Runs at 9:25 AM to catch stocks gapping up in pre-market.
These stocks are added to the hot stocks list for the 9:30 candle.
"""

import logging
import os
from datetime import datetime, time
from typing import Dict, List, Optional, Set

import pytz
import yaml
from pathlib import Path

try:
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockSnapshotRequest
    ALPACA_DATA_SDK = True
except ImportError:
    ALPACA_DATA_SDK = False


class PremarketScanner:
    """
    Scans for pre-market gap-ups before market open.

    Key behaviors:
    - Runs at configurable time (default 9:25 AM ET)
    - Gets pre-market snapshots for universe symbols
    - Identifies stocks gapping up by X% with volume
    - Returns symbols to add to hot stocks list
    """

    DEFAULT_CONFIG = {
        'enabled': True,
        'scan_time': '09:25',  # Time to run scan (ET)
        'min_gap_pct': 3.0,    # Minimum gap up percentage
        'max_gap_pct': 30.0,   # Maximum gap (avoid penny stock pumps)
        'min_price': 5.0,      # Minimum price filter
        'max_price': 500.0,    # Maximum price filter
        'min_premarket_volume': 50000,  # Minimum pre-market volume
        'min_relative_volume': 2.0,     # Pre-market vol vs avg pre-market
        'max_symbols': 20,     # Max symbols to return
    }

    def __init__(self, config: Dict = None):
        """
        Initialize the pre-market scanner.

        Args:
            config: Configuration dict overrides
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.logger = logging.getLogger(__name__)
        self.market_tz = pytz.timezone('America/New_York')

        self.enabled = self.config['enabled']
        self.min_gap_pct = self.config['min_gap_pct']
        self.max_gap_pct = self.config['max_gap_pct']
        self.min_price = self.config['min_price']
        self.max_price = self.config['max_price']
        self.min_premarket_volume = self.config['min_premarket_volume']
        self.min_relative_volume = self.config['min_relative_volume']
        self.max_symbols = self.config['max_symbols']

        # Parse scan time
        scan_time_str = self.config['scan_time']
        hour, minute = map(int, scan_time_str.split(':'))
        self.scan_time = time(hour, minute)

        # Load universe symbols
        self._universe_symbols = self._load_universe_symbols()

        # Initialize Alpaca data client
        self._data_client = None
        self._init_alpaca_client()

        self.logger.info(
            f"PremarketScanner initialized: enabled={self.enabled}, "
            f"scan_time={self.scan_time}, min_gap={self.min_gap_pct}%"
        )

    def _init_alpaca_client(self):
        """Initialize Alpaca data client for snapshots."""
        if not ALPACA_DATA_SDK:
            self.logger.warning(
                "Alpaca data SDK not installed. "
                "Install with: pip install alpaca-py"
            )
            return

        api_key = os.environ.get('ALPACA_API_KEY')
        secret_key = os.environ.get('ALPACA_SECRET_KEY')

        if not api_key or not secret_key:
            self.logger.warning("Alpaca API keys not found in environment")
            return

        try:
            self._data_client = StockHistoricalDataClient(api_key, secret_key)
            self.logger.info("Alpaca data client initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca data client: {e}")

    def _load_universe_symbols(self) -> List[str]:
        """Load symbols from universe.yaml."""
        universe_path = Path(__file__).parent.parent / 'universe.yaml'

        if not universe_path.exists():
            self.logger.warning("universe.yaml not found")
            return []

        try:
            with open(universe_path, 'r') as f:
                universe = yaml.safe_load(f)

            symbols = []
            scanner_universe = universe.get('scanner_universe', {})
            for category, syms in scanner_universe.items():
                if isinstance(syms, list):
                    symbols.extend(syms)

            # Remove duplicates while preserving order
            seen = set()
            unique_symbols = []
            for s in symbols:
                if s not in seen:
                    seen.add(s)
                    unique_symbols.append(s)

            self.logger.info(f"Loaded {len(unique_symbols)} symbols from universe")
            return unique_symbols

        except Exception as e:
            self.logger.warning(f"Failed to load universe.yaml: {e}")
            return []

    def should_scan_now(self) -> bool:
        """Check if it's time to run the pre-market scan."""
        if not self.enabled:
            return False

        now = datetime.now(self.market_tz)
        current_time = now.time()

        # Check if within 2 minutes of scan time
        scan_dt = datetime.combine(now.date(), self.scan_time)
        scan_dt = self.market_tz.localize(scan_dt)

        diff_seconds = abs((now - scan_dt).total_seconds())
        return diff_seconds <= 120  # Within 2 minutes

    def scan(self, symbols: List[str] = None) -> List[Dict]:
        """
        Scan for pre-market gap-ups.

        Args:
            symbols: Optional list of symbols to scan (defaults to universe)

        Returns:
            List of dicts with gap-up candidates, sorted by gap %
        """
        if not self.enabled:
            self.logger.info("PremarketScanner disabled")
            return []

        if not self._data_client:
            self.logger.warning("No Alpaca data client, cannot scan pre-market")
            return []

        symbols_to_scan = symbols or self._universe_symbols
        if not symbols_to_scan:
            self.logger.warning("No symbols to scan")
            return []

        self.logger.info(f"Scanning {len(symbols_to_scan)} symbols for pre-market gaps...")

        try:
            # Get snapshots in batches (Alpaca limit)
            all_snapshots = {}
            batch_size = 100

            for i in range(0, len(symbols_to_scan), batch_size):
                batch = symbols_to_scan[i:i + batch_size]
                request = StockSnapshotRequest(symbol_or_symbols=batch)
                snapshots = self._data_client.get_stock_snapshot(request)
                all_snapshots.update(snapshots)

            self.logger.info(f"Got snapshots for {len(all_snapshots)} symbols")

            # Find gap-ups
            candidates = []
            for symbol, snapshot in all_snapshots.items():
                result = self._evaluate_gap(symbol, snapshot)
                if result:
                    candidates.append(result)

            # Sort by gap percentage descending
            candidates.sort(key=lambda x: x['gap_pct'], reverse=True)

            # Limit results
            candidates = candidates[:self.max_symbols]

            self.logger.info(
                f"PremarketScanner found {len(candidates)} gap-ups: "
                f"{[c['symbol'] for c in candidates[:5]]}..."
            )

            return candidates

        except Exception as e:
            self.logger.error(f"Pre-market scan failed: {e}")
            return []

    def _evaluate_gap(self, symbol: str, snapshot) -> Optional[Dict]:
        """
        Evaluate if a snapshot represents a valid gap-up.

        Args:
            symbol: Stock symbol
            snapshot: Alpaca snapshot object

        Returns:
            Dict with gap info if valid, None otherwise
        """
        try:
            # Get previous close and current pre-market price
            if not snapshot.daily_bar or not snapshot.previous_daily_bar:
                return None

            prev_close = snapshot.previous_daily_bar.close

            # Use pre-market price if available, otherwise latest trade
            if snapshot.latest_quote:
                current_price = (snapshot.latest_quote.ask_price +
                               snapshot.latest_quote.bid_price) / 2
            elif snapshot.latest_trade:
                current_price = snapshot.latest_trade.price
            else:
                return None

            # Calculate gap percentage
            gap_pct = ((current_price - prev_close) / prev_close) * 100

            # Apply filters
            if gap_pct < self.min_gap_pct or gap_pct > self.max_gap_pct:
                return None

            if current_price < self.min_price or current_price > self.max_price:
                return None

            # Check pre-market volume if available
            premarket_volume = 0
            if snapshot.minute_bar:
                premarket_volume = snapshot.minute_bar.volume or 0

            # For now, skip strict volume filter in pre-market
            # (pre-market volume data can be spotty)

            return {
                'symbol': symbol,
                'prev_close': prev_close,
                'current_price': current_price,
                'gap_pct': gap_pct,
                'premarket_volume': premarket_volume,
            }

        except Exception as e:
            self.logger.debug(f"Error evaluating {symbol}: {e}")
            return None

    def get_gap_symbols(self, symbols: List[str] = None) -> List[str]:
        """
        Get just the symbols of gap-up candidates.

        Convenience method for adding to hot stocks list.

        Args:
            symbols: Optional list to scan

        Returns:
            List of symbol strings
        """
        candidates = self.scan(symbols)
        return [c['symbol'] for c in candidates]
