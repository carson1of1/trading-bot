"""
Hot Stocks Feed - Fetches top weekly movers to expand scanner pool.

Runs at bot startup to catch new volatile stocks that aren't in the
static universe.yaml. Symbols are added temporarily (that day only).
If they stay hot, they naturally reappear the next day.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set

import pytz
import requests
import yaml


class HotStocksFeed:
    """
    Fetches top weekly movers from Yahoo Finance to expand the scanner pool.

    Key behaviors:
    - Fetches top N weekly gainers
    - Filters by price, volume
    - Excludes symbols already in universe.yaml
    - Caches results for 24 hours
    - Never blocks trading on failure
    """

    DEFAULT_CONFIG = {
        'enabled': True,
        'top_n': 50,
        'min_price': 5,
        'max_price': 1000,
        'min_volume': 500_000,
        'cache_hours': 20,  # Cache expires before next market open
    }

    def __init__(self, config: Dict = None):
        """
        Initialize the hot stocks feed.

        Args:
            config: Configuration dict with:
                - enabled: Whether to fetch hot stocks (default: True)
                - top_n: Number of gainers to fetch (default: 50)
                - min_price: Minimum stock price (default: $5)
                - max_price: Maximum stock price (default: $1000)
                - min_volume: Minimum average volume (default: 500,000)
                - cache_hours: Hours to cache results (default: 20)
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.logger = logging.getLogger(__name__)

        self.enabled = self.config['enabled']
        self.top_n = self.config['top_n']
        self.min_price = self.config['min_price']
        self.max_price = self.config['max_price']
        self.min_volume = self.config['min_volume']
        self.cache_hours = self.config['cache_hours']

        self.cache_file = Path(__file__).parent.parent / 'data' / 'cache' / 'hot_stocks.json'
        self.market_tz = pytz.timezone('America/New_York')

        # Load universe symbols for exclusion
        self._universe_symbols = self._load_universe_symbols()

        self.logger.info(
            f"HotStocksFeed initialized: enabled={self.enabled}, "
            f"top_n={self.top_n}, min_price=${self.min_price}"
        )

    def fetch(self) -> List[str]:
        """
        Fetch hot stocks, using cache if fresh.

        Returns:
            List of symbols not already in universe (may be empty on failure)
        """
        if not self.enabled:
            self.logger.info("Hot stocks feed disabled")
            return []

        try:
            # Check cache first
            cached = self._load_cache()
            if cached is not None:
                self.logger.info(f"Hot stocks: using cached {len(cached)} symbols")
                return cached

            # Fetch fresh data
            hot_symbols = self._fetch_from_yahoo()

            # Filter and dedupe against universe
            filtered = self._filter_symbols(hot_symbols)
            new_symbols = [s for s in filtered if s not in self._universe_symbols]

            # Cache results
            self._save_cache(new_symbols)

            self.logger.info(
                f"Hot stocks feed: fetched {len(hot_symbols)} gainers, "
                f"{len(new_symbols)} new symbols added to pool"
            )
            if new_symbols:
                self.logger.info(f"Added hot stocks: {new_symbols[:20]}...")

            return new_symbols

        except Exception as e:
            self.logger.warning(f"Hot stocks fetch failed: {e}, using static universe only")
            return []

    def _fetch_from_yahoo(self) -> List[Dict]:
        """
        Fetch top gainers from Yahoo Finance screener.

        Returns:
            List of dicts with symbol, price, volume, change_pct
        """
        # Yahoo Finance screener API endpoint
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        params = {
            'scrIds': 'day_gainers',
            'count': self.top_n * 2,  # Fetch extra to account for filtering
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        self.logger.debug(f"Fetching top gainers from Yahoo Finance...")

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        quotes = data.get('finance', {}).get('result', [{}])[0].get('quotes', [])

        results = []
        for quote in quotes:
            symbol = quote.get('symbol', '')
            # Skip non-standard symbols (ADRs, preferred shares, etc.)
            if not symbol or '.' in symbol or '-' in symbol or len(symbol) > 5:
                continue

            results.append({
                'symbol': symbol,
                'price': quote.get('regularMarketPrice', 0),
                'volume': quote.get('regularMarketVolume', 0),
                'avg_volume': quote.get('averageDailyVolume3Month', 0),
                'change_pct': quote.get('regularMarketChangePercent', 0),
            })

        self.logger.debug(f"Yahoo returned {len(results)} valid symbols")
        return results

    def _filter_symbols(self, stocks: List[Dict]) -> List[str]:
        """
        Apply price and volume filters.

        Args:
            stocks: List of stock dicts from Yahoo

        Returns:
            List of symbols passing filters
        """
        filtered = []

        for stock in stocks:
            symbol = stock.get('symbol', '')
            price = stock.get('price', 0)
            avg_volume = stock.get('avg_volume', 0) or stock.get('volume', 0)

            # Price filter
            if price < self.min_price or price > self.max_price:
                self.logger.debug(f"Filtered {symbol}: price ${price:.2f}")
                continue

            # Volume filter
            if avg_volume < self.min_volume:
                self.logger.debug(f"Filtered {symbol}: volume {avg_volume:,.0f}")
                continue

            filtered.append(symbol)

        return filtered[:self.top_n]

    def _load_universe_symbols(self) -> Set[str]:
        """Load symbols from universe.yaml for exclusion."""
        universe_path = Path(__file__).parent.parent / 'universe.yaml'

        if not universe_path.exists():
            self.logger.warning("universe.yaml not found")
            return set()

        try:
            with open(universe_path, 'r') as f:
                universe = yaml.safe_load(f)

            symbols = set()
            scanner_universe = universe.get('scanner_universe', {})
            for category, syms in scanner_universe.items():
                if isinstance(syms, list):
                    symbols.update(syms)

            self.logger.debug(f"Loaded {len(symbols)} symbols from universe.yaml")
            return symbols

        except Exception as e:
            self.logger.warning(f"Failed to load universe.yaml: {e}")
            return set()

    def _load_cache(self) -> Optional[List[str]]:
        """Load cached hot stocks if fresh."""
        if not self.cache_file.exists():
            return None

        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)

            # Check if cache is fresh
            expires_at = datetime.fromisoformat(cache.get('expires_at', '2000-01-01'))
            if datetime.now(self.market_tz) > expires_at.replace(tzinfo=self.market_tz):
                self.logger.debug("Cache expired")
                return None

            return cache.get('symbols', [])

        except Exception as e:
            self.logger.debug(f"Cache load failed: {e}")
            return None

    def _save_cache(self, symbols: List[str]) -> None:
        """Save hot stocks to cache."""
        try:
            # Ensure cache directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

            now = datetime.now(self.market_tz)
            expires_at = now + timedelta(hours=self.cache_hours)

            cache = {
                'fetched_at': now.isoformat(),
                'expires_at': expires_at.isoformat(),
                'symbols': symbols,
            }

            with open(self.cache_file, 'w') as f:
                json.dump(cache, f, indent=2)

            self.logger.debug(f"Cached {len(symbols)} hot stocks until {expires_at}")

        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    def clear_cache(self) -> None:
        """Clear the cache file (for testing)."""
        if self.cache_file.exists():
            self.cache_file.unlink()
            self.logger.debug("Cache cleared")
