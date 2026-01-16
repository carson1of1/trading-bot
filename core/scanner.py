"""
Volatility Scanner - Dynamic stock selection based on volatility metrics.

Scans a list of symbols for the most volatile stocks and returns the top N
for trading. Used by backtesting with NO LOOK-AHEAD BIAS.

Volatility Score Formula:
    vol_score = (
        (ATR_14 / price * 100) * 0.5 +      # ATR as % of price (50%)
        (daily_range_pct) * 0.3 +            # (High-Low)/Close (30%)
        (volume / avg_volume_20) * 0.2       # Volume ratio (20%)
    )
"""

import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import pytz


class VolatilityScanner:
    """
    Scans for the most volatile stocks in the market.

    Uses historical data for backtesting with NO LOOK-AHEAD BIAS.
    Applies price, volume, and symbol filters before ranking.
    """

    # Default configuration
    DEFAULT_CONFIG = {
        'top_n': 10,
        'min_price': 5,
        'max_price': 1000,
        'min_volume': 500_000,
        'weights': {
            'atr_pct': 0.5,
            'daily_range_pct': 0.3,
            'volume_ratio': 0.2
        },
        'lookback_days': 14,  # Days for ATR calculation
        'trend_filter': True,  # Only trade stocks in favorable trend
        'trend_sma_fast': 20,  # Fast SMA period (hourly bars)
        'trend_sma_slow': 50,  # Slow SMA period (hourly bars)
    }

    def __init__(self, config: Dict = None):
        """
        Initialize the volatility scanner.

        Args:
            config: Configuration dict with:
                - top_n: Number of stocks to return (default: 10)
                - min_price: Minimum stock price (default: $5)
                - max_price: Maximum stock price (default: $1000)
                - min_volume: Minimum average volume (default: 500,000)
                - weights: Dict of scoring weights
                    - atr_pct: Weight for ATR % (default: 0.5)
                    - daily_range_pct: Weight for daily range % (default: 0.3)
                    - volume_ratio: Weight for volume ratio (default: 0.2)
                - lookback_days: Days for ATR calculation (default: 14)
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.logger = logging.getLogger(__name__)

        # Extract configuration
        self.top_n = self.config['top_n']
        self.min_price = self.config['min_price']
        self.max_price = self.config['max_price']
        self.min_volume = self.config['min_volume']
        self.weights = self.config['weights']
        self.lookback_days = self.config['lookback_days']

        self.market_tz = pytz.timezone('America/New_York')

        # Trend filter settings
        self.trend_filter = self.config.get('trend_filter', True)
        self.trend_sma_fast = self.config.get('trend_sma_fast', 20)
        self.trend_sma_slow = self.config.get('trend_sma_slow', 50)

        # Temporary symbols from hot stocks feed (cleared daily)
        self._temporary_symbols: List[str] = []

        self.logger.info(
            f"VolatilityScanner initialized: "
            f"top_n={self.top_n}, min_price=${self.min_price}, "
            f"min_volume={self.min_volume:,}, trend_filter={self.trend_filter}"
        )

    def scan_historical(self, date: str, symbols: List[str],
                        historical_data: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Scan for most volatile stocks on a historical date (for backtesting).

        CRITICAL: NO LOOK-AHEAD BIAS
        - Only uses data available up to 'date'
        - Uses 14-day rolling ATR% to rank stocks
        - Filters applied based on data as of that date

        Args:
            date: Date string in YYYY-MM-DD format (target date)
            symbols: List of all symbols to consider
            historical_data: Dict mapping symbol -> DataFrame with OHLCV data
                           Must have columns: timestamp, open, high, low, close, volume

        Returns:
            List of top N most volatile symbols as of that date
        """
        target_date = pd.to_datetime(date)
        if target_date.tzinfo is None:
            target_date = self.market_tz.localize(target_date)

        self.logger.debug(f"Historical scan for {date} across {len(symbols)} symbols")

        scored_symbols = []

        for symbol in symbols:
            df = historical_data.get(symbol)
            if df is None or df.empty:
                continue

            try:
                # Filter to data available before target date (NO LOOK-AHEAD)
                # With hourly bars, <= midnight excludes all intraday data
                if 'timestamp' in df.columns:
                    df_available = df[df['timestamp'] <= target_date].copy()
                else:
                    # Assume index is datetime
                    df_available = df[df.index <= target_date].copy()

                # Need sufficient data for ATR calculation
                min_bars = self.lookback_days * 7  # ~7 bars per day for hourly data
                if len(df_available) < min_bars:
                    continue

                # Get lookback window
                df_lookback = df_available.tail(min_bars * 2)  # Extra buffer

                if len(df_lookback) < 10:
                    continue

                # Calculate volatility score
                score = self._calculate_volatility_score(df_lookback)

                # Get last price (as of target date)
                last_price = df_lookback['close'].iloc[-1]

                # Apply price filter
                if last_price < self.min_price or last_price > self.max_price:
                    continue

                # Apply volume filter (average of available data)
                avg_volume = df_lookback['volume'].mean()
                if avg_volume < self.min_volume:
                    continue

                # Apply trend filter - only trade stocks in uptrend for longs
                trend = self._get_trend_direction(df_lookback)
                if self.trend_filter:
                    if trend == 'down':
                        self.logger.debug(f"Filtered {symbol}: downtrend (20 SMA < 50 SMA)")
                        continue

                scored_symbols.append({
                    'symbol': symbol,
                    'vol_score': score,
                    'price': last_price,
                    'avg_volume': avg_volume,
                    'trend': trend
                })

            except Exception as e:
                self.logger.debug(f"Error scoring {symbol} for {date}: {e}")
                continue

        # Sort by volatility score (highest first)
        scored_symbols.sort(key=lambda x: x['vol_score'], reverse=True)

        # Return top N
        result = [s['symbol'] for s in scored_symbols[:self.top_n]]

        self.logger.debug(
            f"Historical scan for {date}: found {len(scored_symbols)} valid symbols, "
            f"returning top {len(result)}: {result}"
        )

        return result

    def _calculate_volatility_score(self, df: pd.DataFrame) -> float:
        """
        Calculate volatility score from OHLCV data.

        Formula:
            vol_score = (
                (ATR_14 / price * 100) * 0.5 +      # ATR as % of price (50%)
                (daily_range_pct) * 0.3 +            # (High-Low)/Close (30%)
                (volume / avg_volume_20) * 0.2       # Volume ratio (20%)
            )

        Args:
            df: DataFrame with OHLCV columns (assumes data is sorted by time)

        Returns:
            Volatility score (higher = more volatile)
        """
        try:
            df = df.copy()

            # Calculate True Range components
            df['prev_close'] = df['close'].shift(1)
            df['hl'] = df['high'] - df['low']
            df['hc'] = abs(df['high'] - df['prev_close'])
            df['lc'] = abs(df['low'] - df['prev_close'])
            df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)

            # ATR (14-period rolling)
            atr = df['tr'].rolling(self.lookback_days).mean().iloc[-1]
            last_price = df['close'].iloc[-1]

            # ATR as percentage of price
            atr_pct = (atr / last_price * 100) if last_price > 0 else 0

            # Daily range percentage (average)
            df['range_pct'] = ((df['high'] - df['low']) / df['close'] * 100)
            avg_range_pct = df['range_pct'].mean()

            # Volume ratio (recent 5 bars vs 20-bar average)
            if len(df) >= 20:
                avg_volume_20 = df['volume'].tail(20).mean()
            else:
                avg_volume_20 = df['volume'].mean()

            recent_volume = df['volume'].tail(5).mean()
            volume_ratio = (recent_volume / avg_volume_20) if avg_volume_20 > 0 else 1.0

            # Cap volume ratio at 5 to prevent outliers from dominating
            volume_ratio = min(volume_ratio, 5.0)

            # Composite score using weights
            score = (
                atr_pct * self.weights.get('atr_pct', 0.5) +
                avg_range_pct * self.weights.get('daily_range_pct', 0.3) +
                volume_ratio * self.weights.get('volume_ratio', 0.2)
            )

            return score

        except Exception as e:
            self.logger.debug(f"Error calculating volatility score: {e}")
            return 0.0

    def _get_trend_direction(self, df: pd.DataFrame) -> str:
        """
        Determine trend direction using SMA crossover.

        Returns:
            'up': 20 SMA > 50 SMA (uptrend - good for longs)
            'down': 20 SMA < 50 SMA (downtrend - good for shorts)
            'neutral': SMAs are within 0.5% of each other
        """
        try:
            if len(df) < self.trend_sma_slow:
                return 'neutral'

            close = df['close']
            sma_fast = close.rolling(self.trend_sma_fast).mean().iloc[-1]
            sma_slow = close.rolling(self.trend_sma_slow).mean().iloc[-1]

            if pd.isna(sma_fast) or pd.isna(sma_slow) or sma_slow == 0:
                return 'neutral'

            # Calculate difference as percentage
            diff_pct = (sma_fast - sma_slow) / sma_slow * 100

            if diff_pct > 0.5:
                return 'up'
            elif diff_pct < -0.5:
                return 'down'
            else:
                return 'neutral'

        except Exception as e:
            self.logger.debug(f"Error calculating trend: {e}")
            return 'neutral'

    def _apply_filters(self, candidates: List[Dict]) -> List[Dict]:
        """
        Apply price, volume, and other filters to candidates.

        Args:
            candidates: List of candidate dicts with price, volume, symbol

        Returns:
            Filtered list of candidates
        """
        filtered = []

        for c in candidates:
            symbol = c.get('symbol', '')
            price = c.get('price', 0)
            volume = c.get('volume', 0)

            # Symbol validation
            if not symbol or len(symbol) > 5:
                continue

            # Price filter
            if price < self.min_price:
                self.logger.debug(f"Filtered {symbol}: price ${price:.2f} < ${self.min_price}")
                continue

            if price > self.max_price:
                self.logger.debug(f"Filtered {symbol}: price ${price:.2f} > ${self.max_price}")
                continue

            # Volume filter
            if volume < self.min_volume:
                self.logger.debug(f"Filtered {symbol}: volume {volume:,} < {self.min_volume:,}")
                continue

            filtered.append(c)

        return filtered

    def get_config(self) -> Dict:
        """
        Return current scanner configuration.

        Returns:
            Dict with all configuration values
        """
        return {
            'top_n': self.top_n,
            'min_price': self.min_price,
            'max_price': self.max_price,
            'min_volume': self.min_volume,
            'weights': self.weights,
            'lookback_days': self.lookback_days,
            'trend_filter': self.trend_filter,
            'trend_sma_fast': self.trend_sma_fast,
            'trend_sma_slow': self.trend_sma_slow
        }

    def add_temporary_symbols(self, symbols: List[str]) -> None:
        """
        Add temporary symbols to the scanner pool (e.g., from hot stocks feed).

        These symbols are included in the next scan() call but are not persisted.
        Call clear_temporary_symbols() to remove them.

        Args:
            symbols: List of symbols to add temporarily
        """
        if not symbols:
            return

        # Filter out duplicates
        new_symbols = [s for s in symbols if s not in self._temporary_symbols]
        self._temporary_symbols.extend(new_symbols)

        self.logger.info(
            f"Added {len(new_symbols)} temporary symbols to scanner pool "
            f"(total temporary: {len(self._temporary_symbols)})"
        )

    def clear_temporary_symbols(self) -> None:
        """Clear all temporary symbols from the scanner pool."""
        count = len(self._temporary_symbols)
        self._temporary_symbols = []
        if count > 0:
            self.logger.debug(f"Cleared {count} temporary symbols")

    def scan(self) -> List[str]:
        """
        Scan for most volatile stocks using current market data (for live trading).

        This method fetches fresh data and calculates volatility scores,
        returning the top N most volatile symbols.

        Uses the same scoring logic as scan_historical() for backtest/live alignment.

        Returns:
            List of top N most volatile symbols
        """
        from pathlib import Path
        import yaml

        # Load scanner universe from universe.yaml
        bot_dir = Path(__file__).parent.parent
        universe_path = bot_dir / 'universe.yaml'

        if not universe_path.exists():
            self.logger.error("universe.yaml not found for scanner")
            return []

        with open(universe_path, 'r') as f:
            universe = yaml.safe_load(f)

        # Get scanner universe symbols
        scanner_universe = universe.get('scanner_universe', {})
        symbols = []
        for category, syms in scanner_universe.items():
            if isinstance(syms, list):
                symbols.extend(syms)

        # Remove duplicates while preserving order
        seen = set()
        symbols = [s for s in symbols if not (s in seen or seen.add(s))]

        if not symbols:
            self.logger.warning("No symbols in scanner_universe")
            return []

        # Add temporary symbols (from hot stocks feed)
        static_count = len(symbols)
        for sym in self._temporary_symbols:
            if sym not in seen:
                symbols.append(sym)
                seen.add(sym)

        if self._temporary_symbols:
            self.logger.info(
                f"Scanner pool: {static_count} static + "
                f"{len(symbols) - static_count} hot = {len(symbols)} total"
            )

        self.logger.info(f"Scanning {len(symbols)} symbols for volatility...")

        # Fetch data and score each symbol
        from .data import YFinanceDataFetcher
        from datetime import datetime, timedelta

        fetcher = YFinanceDataFetcher()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days + 5)

        scored_symbols = []

        for symbol in symbols:
            try:
                # FIX (Jan 15, 2026): Convert symbol to yfinance format for data fetch
                from core.symbol_mapping import to_yfinance
                data_symbol = to_yfinance(symbol)

                df = fetcher.get_historical_data_range(
                    symbol=data_symbol,
                    timeframe='1Hour',
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )

                if df is None or len(df) < 20:
                    continue

                # Calculate volatility score (same as scan_historical)
                score = self._calculate_volatility_score(df)

                # Get last price and volume
                last_price = df['close'].iloc[-1]
                avg_volume = df['volume'].mean()

                # Apply filters
                if last_price < self.min_price or last_price > self.max_price:
                    continue
                if avg_volume < self.min_volume:
                    continue

                # Apply trend filter - only trade stocks in uptrend for longs
                trend = self._get_trend_direction(df)
                if self.trend_filter:
                    if trend == 'down':
                        self.logger.debug(f"Filtered {symbol}: downtrend (20 SMA < 50 SMA)")
                        continue

                scored_symbols.append({
                    'symbol': symbol,
                    'vol_score': score,
                    'price': last_price,
                    'avg_volume': avg_volume,
                    'trend': trend
                })

            except Exception as e:
                self.logger.debug(f"Error scanning {symbol}: {e}")
                continue

        # Sort by volatility score (highest first)
        scored_symbols.sort(key=lambda x: x['vol_score'], reverse=True)

        # Return top N
        result = [s['symbol'] for s in scored_symbols[:self.top_n]]

        self.logger.info(
            f"Scanner selected top {len(result)} symbols: {result}"
        )

        # NEW (Jan 2026): Log scan results for live/backtest comparison
        self._log_scan_result(
            scan_date=datetime.now().strftime('%Y-%m-%d'),
            mode='LIVE',  # Will be overridden if PAPER mode detected
            symbols_scanned=len(symbols),
            selected_symbols=result,
            all_scores=scored_symbols
        )

        return result

    def _log_scan_result(self, scan_date: str, mode: str, symbols_scanned: int,
                         selected_symbols: List[str], all_scores: List[Dict]):
        """
        Log scan results to database for live/backtest comparison.

        NEW (Jan 2026): Enables verification of scanner output alignment.
        """
        try:
            from .logger import TradeLogger

            logger = TradeLogger()
            logger.log_scan_result({
                'scan_date': scan_date,
                'mode': mode,
                'symbols_scanned': symbols_scanned,
                'selected_symbols': selected_symbols,
                'all_scores': all_scores[:50],  # Top 50 to keep size reasonable
                'config': self.get_config()
            })
        except Exception as e:
            self.logger.debug(f"Could not log scan result: {e}")

    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """
        Validate a list of symbols against scanner filters.

        Useful for checking if manually specified symbols meet criteria.

        Args:
            symbols: List of symbols to validate

        Returns:
            List of symbols that pass filters
        """
        valid = []

        for symbol in symbols:
            # Basic validation
            if not symbol or len(symbol) > 5:
                continue
            if not all(c.isalnum() for c in symbol):
                continue
            valid.append(symbol)

        return valid
