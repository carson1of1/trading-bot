"""
Disk-based cache for historical market data using Parquet files.

Speeds up backtests by caching fetched data to disk.
Data is stored per-symbol in data/cache/<symbol>.parquet
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pandas as pd
import pytz


class DataCache:
    """
    Persistent disk cache for historical OHLCV data.

    Uses Parquet files for fast read/write of pandas DataFrames.
    One file per symbol, containing all cached hourly data.
    """

    def __init__(self, cache_dir: str = None):
        """
        Initialize the data cache.

        Args:
            cache_dir: Directory for cache files. Defaults to data/cache/
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "data" / "cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.market_tz = pytz.timezone('America/New_York')

        # Cache freshness: consider data fresh if last bar is within this many hours
        self.freshness_hours = 24

        self.logger.info(f"DataCache initialized: {self.cache_dir}")

    def _get_cache_path(self, symbol: str) -> Path:
        """Get the cache file path for a symbol."""
        return self.cache_dir / f"{symbol.upper()}.parquet"

    def has_fresh_data(self, symbol: str, end_date: str = None) -> bool:
        """
        Check if we have fresh cached data for a symbol.

        Args:
            symbol: Stock ticker
            end_date: End date we need data for (YYYY-MM-DD). Defaults to today.

        Returns:
            True if cache exists and has data up to end_date
        """
        cache_path = self._get_cache_path(symbol)

        if not cache_path.exists():
            return False

        try:
            # Read just the last few rows to check freshness
            df = pd.read_parquet(cache_path)
            if df.empty:
                return False

            # Parse end_date
            if end_date is None:
                target_date = datetime.now(self.market_tz).date()
            else:
                target_date = pd.to_datetime(end_date).date()

            # Get the latest timestamp in cache
            if 'timestamp' in df.columns:
                last_ts = pd.to_datetime(df['timestamp'].max())
            else:
                last_ts = pd.to_datetime(df.index.max())

            if last_ts.tzinfo is None:
                last_ts = self.market_tz.localize(last_ts)

            last_date = last_ts.date()

            # Consider fresh if we have data from recent trading days
            # Allow up to 4 days to account for:
            # - Today (market not yet open)
            # - Weekends (2 days)
            # - Holidays (1-2 days, e.g., New Year's)
            # This is safe for backtesting where we need historical data
            days_behind = (target_date - last_date).days
            return days_behind <= 4

        except Exception as e:
            self.logger.warning(f"Error checking cache for {symbol}: {e}")
            return False

    def load(self, symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Load cached data for a symbol.

        Args:
            symbol: Stock ticker
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            DataFrame with cached data, or None if no cache exists
        """
        cache_path = self._get_cache_path(symbol)

        if not cache_path.exists():
            return None

        try:
            df = pd.read_parquet(cache_path)

            if df.empty:
                return None

            # Ensure timestamp column exists and is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Filter by date range if specified
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    if start_dt.tzinfo is None:
                        start_dt = self.market_tz.localize(start_dt)
                    df = df[df['timestamp'] >= start_dt]

                if end_date:
                    end_dt = pd.to_datetime(end_date) + timedelta(days=1)  # Include end date
                    if end_dt.tzinfo is None:
                        end_dt = self.market_tz.localize(end_dt)
                    df = df[df['timestamp'] < end_dt]

            return df if not df.empty else None

        except Exception as e:
            self.logger.warning(f"Error loading cache for {symbol}: {e}")
            return None

    def save(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        Save data to cache, merging with existing data.

        Args:
            symbol: Stock ticker
            df: DataFrame with OHLCV data (must have 'timestamp' column)

        Returns:
            True if save was successful
        """
        if df is None or df.empty:
            return False

        cache_path = self._get_cache_path(symbol)

        try:
            # Load existing data and merge
            existing = self.load(symbol)

            if existing is not None and not existing.empty:
                # Combine and deduplicate by timestamp
                combined = pd.concat([existing, df], ignore_index=True)
                if 'timestamp' in combined.columns:
                    combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
                    combined = combined.sort_values('timestamp').reset_index(drop=True)
                df = combined

            # Save to parquet
            df.to_parquet(cache_path, index=False)
            self.logger.debug(f"Cached {len(df)} bars for {symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving cache for {symbol}: {e}", exc_info=True)
            return False

    def load_batch(self, symbols: List[str], start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load cached data for multiple symbols.

        Args:
            symbols: List of stock tickers
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dict mapping symbol -> DataFrame (only includes symbols with data)
        """
        result = {}
        for symbol in symbols:
            df = self.load(symbol, start_date, end_date)
            if df is not None:
                result[symbol] = df
        return result

    def get_missing_symbols(self, symbols: List[str], end_date: str = None) -> List[str]:
        """
        Get list of symbols that need fresh data.

        Args:
            symbols: List of symbols to check
            end_date: Date we need data for

        Returns:
            List of symbols missing from cache or with stale data
        """
        return [s for s in symbols if not self.has_fresh_data(s, end_date)]

    def clear(self, symbol: str = None):
        """
        Clear cache for a symbol or all symbols.

        Args:
            symbol: Specific symbol to clear, or None for all
        """
        if symbol:
            cache_path = self._get_cache_path(symbol)
            if cache_path.exists():
                cache_path.unlink()
                self.logger.info(f"Cleared cache for {symbol}")
        else:
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()
            self.logger.info("Cleared all cache files")

    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache."""
        files = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files)

        return {
            "symbols_cached": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }


# Global cache instance
_cache_instance = None


def get_cache() -> DataCache:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DataCache()
    return _cache_instance
