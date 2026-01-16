"""
Continuous Breakout Scanner - Finds stocks approaching breakout levels.

Runs continuously during market hours to identify stocks that are:
1. Approaching key resistance levels
2. Building volume as they approach
3. Likely to trigger breakout signals at candle close

These candidates are fed to the bot for evaluation at each candle close.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pytz
import yaml
from pathlib import Path

from .indicators import TechnicalIndicators


class BreakoutScanner:
    """
    Continuously scans for stocks approaching breakout levels.

    Identifies stocks that are:
    - Within X% of resistance (20-period high)
    - Showing increasing volume
    - Have sufficient ATR for profit potential

    These are fed to the bot as candidates BEFORE they break out,
    allowing entry at the start of the move.
    """

    DEFAULT_CONFIG = {
        'enabled': True,
        'scan_interval_minutes': 15,  # How often to scan
        'proximity_pct': 2.0,         # Within X% of resistance
        'min_price': 5.0,
        'max_price': 500.0,
        'min_volume_ratio': 1.5,      # Current vs avg volume
        'min_atr_pct': 2.0,           # Minimum ATR as % of price
        'max_candidates': 30,         # Max candidates to track
        'lookback_bars': 20,          # Bars for resistance calc
    }

    def __init__(self, config: Dict = None, data_fetcher=None):
        """
        Initialize the breakout scanner.

        Args:
            config: Configuration dict overrides
            data_fetcher: YFinanceDataFetcher instance for getting data
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.logger = logging.getLogger(__name__)
        self.market_tz = pytz.timezone('America/New_York')

        self.enabled = self.config['enabled']
        self.scan_interval = self.config['scan_interval_minutes']
        self.proximity_pct = self.config['proximity_pct']
        self.min_price = self.config['min_price']
        self.max_price = self.config['max_price']
        self.min_volume_ratio = self.config['min_volume_ratio']
        self.min_atr_pct = self.config['min_atr_pct']
        self.max_candidates = self.config['max_candidates']
        self.lookback_bars = self.config['lookback_bars']

        self.data_fetcher = data_fetcher
        self.indicators = TechnicalIndicators()

        # Track candidates and last scan time
        self._candidates: List[Dict] = []
        self._last_scan: Optional[datetime] = None
        self._candidate_symbols: Set[str] = set()

        # Load universe symbols
        self._universe_symbols = self._load_universe_symbols()

        self.logger.info(
            f"BreakoutScanner initialized: enabled={self.enabled}, "
            f"interval={self.scan_interval}min, proximity={self.proximity_pct}%"
        )

    def _load_universe_symbols(self) -> List[str]:
        """Load symbols from universe.yaml."""
        universe_path = Path(__file__).parent.parent / 'universe.yaml'

        if not universe_path.exists():
            return []

        try:
            with open(universe_path, 'r') as f:
                universe = yaml.safe_load(f)

            symbols = []
            scanner_universe = universe.get('scanner_universe', {})
            for category, syms in scanner_universe.items():
                if isinstance(syms, list):
                    symbols.extend(syms)

            return list(set(symbols))

        except Exception as e:
            self.logger.warning(f"Failed to load universe: {e}")
            return []

    def should_scan_now(self) -> bool:
        """Check if it's time to run another scan."""
        if not self.enabled:
            return False

        if self._last_scan is None:
            return True

        now = datetime.now(self.market_tz)
        elapsed = (now - self._last_scan).total_seconds() / 60

        return elapsed >= self.scan_interval

    def scan(
        self,
        symbols: List[str] = None,
        data_cache: Dict[str, pd.DataFrame] = None
    ) -> List[Dict]:
        """
        Scan for breakout candidates.

        Args:
            symbols: Optional list of symbols (defaults to universe)
            data_cache: Optional pre-fetched data cache

        Returns:
            List of candidate dicts sorted by proximity to breakout
        """
        if not self.enabled:
            return []

        symbols_to_scan = symbols or self._universe_symbols
        if not symbols_to_scan:
            self.logger.warning("No symbols to scan")
            return []

        self.logger.info(f"Scanning {len(symbols_to_scan)} symbols for breakout setups...")

        candidates = []
        scanned = 0
        errors = 0

        for symbol in symbols_to_scan:
            try:
                # FIX (Jan 15, 2026): Convert symbol to yfinance format for data fetch
                # Universe uses Alpaca format (BTC/USD), yfinance needs BTC-USD
                from core.symbol_mapping import to_yfinance
                data_symbol = to_yfinance(symbol)

                # Get data from cache or fetch
                if data_cache and symbol in data_cache:
                    df = data_cache[symbol]
                elif self.data_fetcher:
                    # Use get_historical_data with 1Hour timeframe, ~200 bars for lookback
                    df = self.data_fetcher.get_historical_data(data_symbol, timeframe='1Hour', limit=200)
                else:
                    continue

                if df is None or len(df) < self.lookback_bars + 10:
                    continue

                result = self._evaluate_breakout_setup(symbol, df)
                if result:
                    candidates.append(result)

                scanned += 1

            except Exception as e:
                self.logger.debug(f"Error scanning {symbol}: {e}")
                errors += 1

        # Sort by proximity to resistance (closest first)
        candidates.sort(key=lambda x: x['proximity_pct'])

        # Limit results
        candidates = candidates[:self.max_candidates]

        # Update tracking
        self._candidates = candidates
        self._candidate_symbols = {c['symbol'] for c in candidates}
        self._last_scan = datetime.now(self.market_tz)

        self.logger.info(
            f"BreakoutScanner found {len(candidates)} setups "
            f"(scanned {scanned}, errors {errors}): "
            f"{[c['symbol'] for c in candidates[:5]]}..."
        )

        return candidates

    def _evaluate_breakout_setup(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Evaluate if a symbol is setting up for a breakout.

        Criteria:
        1. Price within X% of N-period high (resistance)
        2. Volume trending up (current > avg)
        3. Sufficient ATR for profit potential
        4. Price within acceptable range

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data

        Returns:
            Dict with setup info if valid, None otherwise
        """
        try:
            # Normalize column names
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]

            if 'close' not in df.columns:
                return None

            # Get recent data
            recent = df.tail(self.lookback_bars + 10)
            if len(recent) < self.lookback_bars:
                return None

            current_price = recent['close'].iloc[-1]

            # Price filter
            if current_price < self.min_price or current_price > self.max_price:
                return None

            # Calculate resistance (N-period high)
            lookback_data = recent.tail(self.lookback_bars)
            resistance = lookback_data['high'].max()

            # Calculate proximity to resistance
            proximity_pct = ((resistance - current_price) / current_price) * 100

            # Must be approaching but not already above resistance
            if proximity_pct < 0 or proximity_pct > self.proximity_pct:
                return None

            # Calculate ATR
            df_with_ind = self.indicators.add_all_indicators(recent.copy())
            # ATR column is uppercase from TechnicalIndicators
            if 'ATR' not in df_with_ind.columns:
                return None

            atr = df_with_ind['ATR'].iloc[-1]
            atr_pct = (atr / current_price) * 100

            if atr_pct < self.min_atr_pct:
                return None

            # Check volume trend
            avg_volume = recent['volume'].rolling(10).mean().iloc[-1]
            current_volume = recent['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

            # Relaxed volume filter for setup detection
            # (volume often spikes ON the breakout, not before)
            # Only filter if volume is extremely low (dead stock)
            # Current bar may be incomplete, so use 0.1 threshold
            if volume_ratio < 0.1:  # Very low volume = dead stock
                return None

            return {
                'symbol': symbol,
                'current_price': current_price,
                'resistance': resistance,
                'proximity_pct': proximity_pct,
                'atr': atr,
                'atr_pct': atr_pct,
                'volume_ratio': volume_ratio,
                'setup_quality': self._calculate_setup_quality(
                    proximity_pct, atr_pct, volume_ratio
                ),
            }

        except Exception as e:
            self.logger.debug(f"Error evaluating {symbol}: {e}")
            return None

    def _calculate_setup_quality(
        self,
        proximity_pct: float,
        atr_pct: float,
        volume_ratio: float
    ) -> float:
        """
        Calculate overall setup quality score (0-100).

        Components:
        - Proximity: Closer to resistance = better (40%)
        - ATR: Higher = better potential (30%)
        - Volume: Higher ratio = better (30%)
        """
        # Proximity score (0-40): closer is better
        # At 0% proximity = 40, at 2% = 0
        proximity_score = max(0, 40 * (1 - proximity_pct / self.proximity_pct))

        # ATR score (0-30): higher is better, capped at 5%
        atr_score = min(30, 30 * (atr_pct / 5.0))

        # Volume score (0-30): higher is better, capped at 3x
        volume_score = min(30, 30 * (volume_ratio / 3.0))

        return proximity_score + atr_score + volume_score

    def get_candidate_symbols(self) -> List[str]:
        """Get list of current breakout candidate symbols."""
        return list(self._candidate_symbols)

    def get_candidates(self) -> List[Dict]:
        """Get full candidate details."""
        return self._candidates.copy()

    def is_candidate(self, symbol: str) -> bool:
        """Check if a symbol is a current breakout candidate."""
        return symbol in self._candidate_symbols

    def merge_with_hot_stocks(
        self,
        hot_stocks: List[str],
        max_total: int = 50
    ) -> List[str]:
        """
        Merge breakout candidates with hot stocks list.

        Prioritizes:
        1. Stocks that are both hot AND breakout candidates
        2. Breakout candidates (approaching resistance)
        3. Hot stocks (already moving)

        Args:
            hot_stocks: List from HotStocksFeed
            max_total: Maximum combined symbols

        Returns:
            Combined list prioritized for breakout potential
        """
        candidate_symbols = set(self._candidate_symbols)
        hot_set = set(hot_stocks)

        # Priority 1: Both hot and candidate
        both = list(candidate_symbols & hot_set)

        # Priority 2: Candidates not in hot stocks
        candidates_only = [s for s in self._candidate_symbols if s not in hot_set]

        # Priority 3: Hot stocks not candidates
        hot_only = [s for s in hot_stocks if s not in candidate_symbols]

        # Combine with priority order
        combined = both + candidates_only + hot_only

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for s in combined:
            if s not in seen:
                seen.add(s)
                result.append(s)

        return result[:max_total]
