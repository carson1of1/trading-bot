import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from .market_hours import MarketHours  # FIX (Dec 9, 2025): Check market hours for stale threshold
from .config import get_global_config  # FIX (Dec 9, 2025): Load configurable stale thresholds
from .cache import get_cache  # Disk cache for backtest data
from .symbol_mapping import to_yfinance  # FIX (Jan 16, 2026): Convert crypto symbols for yfinance

# FIX (Jan 7, 2026): Timeout for individual yfinance API calls to prevent hanging
YFINANCE_TIMEOUT_SECONDS = 30

class YFinanceDataFetcher:
    """
    Handle historical data fetching from Yahoo Finance for live trading

    CRITICAL FIX (Nov 5, 2025):
    - Alpaca free tier: Only 4 bars with IEX feed (requires paid subscription for SIP)
    - Polygon free tier: No access to 1-minute data (requires paid upgrade)
    - Yahoo Finance: FREE unlimited 1-minute data (~390 bars for 7 days)

    This is the PRIMARY data source for live trading until user upgrades paid APIs.
    """

    def __init__(self):
        """Initialize Yahoo Finance data fetcher"""
        self.cache = {}
        self.cache_ttl = 120  # FIX (Dec 9, 2025): Increased from 60 to 120 sec for faster UI
        self.logger = logging.getLogger(__name__)
        self.market_tz = pytz.timezone('America/New_York')

        # FIX (Dec 9, 2025): Reduced rate limiting - 2 sec was way too slow (5+ min for 50 symbols)
        # Yahoo Finance limit is 2000 req/hour = 33/min, we typically do 50-100 symbols
        # 0.3 sec interval = ~200 req/min max, well under limit
        self.last_api_call = {}  # Track last call time per symbol
        self.min_call_interval = 0.3  # 300ms between calls (was 2.0 - too slow!)

        # FIX (Dec 9, 2025): Load configurable stale data thresholds from config.yaml
        # CRITICAL for day trading: 120-minute threshold was way too permissive
        config = get_global_config()
        data_quality = config.config.get('data_quality', {})
        self.stale_threshold_1min = data_quality.get('stale_threshold_1min', 10)  # Default 10 min (was 120!)
        self.stale_threshold_5min = data_quality.get('stale_threshold_5min', 30)  # Default 30 min (was 180!)
        self.stale_threshold_daily = data_quality.get('stale_threshold_daily', 360)  # Default 6 hours
        self.stale_threshold_outside_hours = data_quality.get('stale_threshold_outside_hours', 960)  # Default 16 hours

        self.logger.info("[OK] YFinanceDataFetcher initialized - using FREE Yahoo Finance data")
        self.logger.info(f"Stale thresholds: 1min={self.stale_threshold_1min}m, 5min={self.stale_threshold_5min}m, daily={self.stale_threshold_daily}m")

    def _is_cache_valid(self, key):
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False

        cached_time = self.cache[key]['timestamp']
        # BUG FIX (Dec 4, 2025): Use total_seconds() instead of .seconds
        # FIX (Dec 9, 2025): Use timezone-aware now() to match cached_time
        current_time = datetime.now(self.market_tz)
        return (current_time - cached_time).total_seconds() < self.cache_ttl

    def get_historical_data(self, symbol, timeframe='1Min', limit=100):
        """
        Get historical OHLCV data for a symbol from Yahoo Finance

        Args:
            symbol: Stock ticker (e.g., 'TSLA')
            timeframe: '1Min', '5Min', '1Hour', or '1Day'
            limit: Max number of bars to return

        Returns:
            pandas DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            cache_key = f"{symbol}_{timeframe}_{limit}"

            # Check cache first
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]['data']

            # BUG FIX (Dec 2025): Rate limiting - prevent exceeding Yahoo Finance limits
            if symbol in self.last_api_call:
                time_since_last = time.time() - self.last_api_call[symbol]
                if time_since_last < self.min_call_interval:
                    sleep_time = self.min_call_interval - time_since_last
                    self.logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s for {symbol}")
                    time.sleep(sleep_time)

            # Map timeframe to yfinance interval
            interval_map = {
                '1Min': '1m',
                '5Min': '5m',
                '15Min': '15m',
                '1Hour': '1h',
                '1Day': '1d'
            }

            if timeframe not in interval_map:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return None

            yf_interval = interval_map[timeframe]

            # Calculate period based on limit
            # yfinance allows: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            if timeframe == '1Min':
                # 1-minute data: max 7 days back (~390 bars/day = ~2730 bars total)
                period = '7d'
            elif timeframe == '5Min':
                period = '1mo'  # 1 month for 5-minute bars
            elif timeframe == '1Hour':
                period = '3mo'  # 3 months for hourly
            elif timeframe == '1Day':
                period = '1y'  # 1 year for daily
            else:
                period = '7d'  # Default

            # Fetch data from Yahoo Finance
            # FIX (Jan 16, 2026): Convert symbol to yfinance format (BTC/USD -> BTC-USD)
            yf_symbol = to_yfinance(symbol)
            ticker = yf.Ticker(yf_symbol)

            # BUG FIX (Dec 2025): Record API call time for rate limiting
            self.last_api_call[symbol] = time.time()

            # FIX (Jan 7, 2026): Add timeout to yfinance API call to prevent hanging
            # yfinance doesn't have built-in timeout, so use ThreadPoolExecutor
            def _fetch_history():
                return ticker.history(
                    period=period,
                    interval=yf_interval,
                    actions=False,  # Don't include dividends/splits
                    auto_adjust=False,  # Don't auto-adjust prices
                    prepost=False  # No pre/post market data
                )

            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_fetch_history)
                    df = future.result(timeout=YFINANCE_TIMEOUT_SECONDS)
            except FuturesTimeoutError:
                self.logger.warning(f"yfinance timeout for {symbol} after {YFINANCE_TIMEOUT_SECONDS}s")
                return None
            except Exception as e:
                self.logger.error(f"yfinance download error for {symbol}: {e}", exc_info=True)
                return None

            if df is None or df.empty:
                self.logger.warning(f"No data received from Yahoo Finance for {symbol}")
                return None

            # Reset index to make timestamp a column
            df = df.reset_index()

            # BUG FIX (Dec 2025): Enhanced column normalization - handle all variations
            # yfinance can return 'Datetime', 'Date', 'datetime', 'date' depending on version
            # BUG FIX (Dec 9, 2025): Simplified column renaming logic - convert to lowercase first, then rename
            # Old logic had inconsistent handling - 'Datetime' would become 'timestamp' but 'datetime' wouldn't
            # BUG FIX (Dec 10, 2025): Ensure we don't have duplicate timestamp columns after renaming
            df.columns = [col.lower() for col in df.columns]

            # Check for existing timestamp column before renaming
            columns_to_rename = {}
            if 'timestamp' not in df.columns:
                if 'datetime' in df.columns:
                    columns_to_rename['datetime'] = 'timestamp'
                elif 'date' in df.columns:
                    columns_to_rename['date'] = 'timestamp'
                elif 'index' in df.columns:
                    columns_to_rename['index'] = 'timestamp'

            if columns_to_rename:
                df.rename(columns=columns_to_rename, inplace=True)

            # Convert timezone to Eastern Time
            if 'timestamp' in df.columns:
                if df['timestamp'].dt.tz is None:
                    # If no timezone, assume UTC
                    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')

                # Convert to Eastern Time
                df['timestamp'] = df['timestamp'].dt.tz_convert(self.market_tz)

            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Missing column {col} in data for {symbol}")
                    return None

            # BUG FIX (Dec 10, 2025): Validate OHLC data integrity
            # CRITICAL: Detect and handle corrupted or invalid price data
            # 1. Check for NaN values in OHLC columns
            ohlc_columns = ['open', 'high', 'low', 'close']
            nan_mask = df[ohlc_columns].isna().any(axis=1)
            if nan_mask.any():
                nan_count = nan_mask.sum()
                self.logger.warning(f"Found {nan_count} rows with NaN in OHLC data for {symbol} - dropping")
                df = df[~nan_mask].reset_index(drop=True)
                if df.empty:
                    self.logger.error(f"All data was NaN for {symbol}")
                    return None

            # 2. Check for zero or negative prices (invalid)
            for col in ohlc_columns:
                invalid_prices = df[df[col] <= 0]
                if len(invalid_prices) > 0:
                    self.logger.warning(f"Found {len(invalid_prices)} rows with invalid {col} <= 0 for {symbol} - dropping")
                    df = df[df[col] > 0].reset_index(drop=True)

            if df.empty:
                self.logger.error(f"No valid price data for {symbol} after filtering")
                return None

            # 3. Validate high >= low (data integrity)
            invalid_hl = df[df['high'] < df['low']]
            if len(invalid_hl) > 0:
                self.logger.warning(f"Found {len(invalid_hl)} rows where high < low for {symbol} - fixing")
                # Fix by swapping high and low where invalid
                swap_mask = df['high'] < df['low']
                df.loc[swap_mask, ['high', 'low']] = df.loc[swap_mask, ['low', 'high']].values

            # 4. Validate open/close within high/low range
            df['open'] = df['open'].clip(lower=df['low'], upper=df['high'])
            df['close'] = df['close'].clip(lower=df['low'], upper=df['high'])

            # 5. Validate volume is non-negative
            df['volume'] = df['volume'].clip(lower=0)

            # Apply limit - take the most recent N bars
            if limit and len(df) > limit:
                df = df.tail(limit).reset_index(drop=True)

            # BUG FIX (Dec 2025): Validate data for gaps (missing bars)
            if 'timestamp' in df.columns and len(df) > 1:
                time_diffs = df['timestamp'].diff()
                if timeframe == '1Min':
                    max_gap = timedelta(minutes=5)
                elif timeframe == '5Min':
                    max_gap = timedelta(minutes=15)
                else:
                    max_gap = None

                if max_gap:
                    large_gaps = time_diffs[time_diffs > max_gap]
                    if len(large_gaps) > 0:
                        self.logger.warning(
                            f"Data gaps detected for {symbol}: {len(large_gaps)} gaps > {max_gap}"
                        )

            # BUG FIX (Dec 9, 2025): Validate timestamp for stale data detection
            # CRITICAL: Day trading requires fresh data - validate latest bar timestamp
            # Tightened stale data thresholds (default: 10 min for 1-minute bars vs old 120 min)
            if 'timestamp' in df.columns and len(df) > 0:
                latest_timestamp = df['timestamp'].iloc[-1]
                current_time = datetime.now(self.market_tz)

                # FIX (Dec 9, 2025): Handle timezone-naive vs timezone-aware datetime comparison
                # Yahoo Finance may return timezone-naive or timezone-aware timestamps
                if latest_timestamp.tzinfo is None:
                    # Naive timestamp - assume it's in market timezone
                    latest_timestamp = self.market_tz.localize(latest_timestamp)
                elif str(latest_timestamp.tzinfo) != str(self.market_tz):
                    # Different timezone - convert to market timezone
                    latest_timestamp = latest_timestamp.astimezone(self.market_tz)

                # Calculate age of data in minutes
                time_diff_minutes = (current_time - latest_timestamp).total_seconds() / 60

                # FIX (Dec 9, 2025): Different thresholds for market hours vs outside
                # Outside market hours, data from previous close is expected and valid
                try:
                    market_hours = MarketHours()
                    is_market_open = market_hours.is_market_open()
                except Exception as e:
                    self.logger.warning(f"Failed to check market hours: {e} - defaulting to market CLOSED")
                    is_market_open = False  # Safer default: assume closed = use permissive threshold

                # FIX (Dec 10, 2025): Daily bars ALWAYS use generous threshold regardless of market hours
                # Daily bars only update ONCE per day after market close, so they're always "old" during the day
                if timeframe == '1Day':
                    max_age_minutes = max(self.stale_threshold_daily, 1440)  # At least 24 hours for daily
                elif is_market_open:
                    # CRITICAL (Dec 9, 2025): Use MUCH tighter thresholds from config.yaml
                    # Day trading requires fresh data - old thresholds were killing accuracy
                    if timeframe == '1Min':
                        max_age_minutes = self.stale_threshold_1min  # 10 min (was 120!)
                    elif timeframe == '5Min':
                        max_age_minutes = self.stale_threshold_5min  # 30 min (was 180!)
                    else:
                        max_age_minutes = self.stale_threshold_daily  # Default to daily threshold
                else:
                    # Outside market hours: accept previous day's data
                    # Last bar could be 16+ hours old (4 PM yesterday to 8 AM today)
                    max_age_minutes = self.stale_threshold_outside_hours  # 1440 min (24 hours)

                # DEBUG (Dec 5, 2025): Allow stale data for testing outside market hours
                allow_stale = getattr(self, 'allow_stale_data', False)
                if not allow_stale and time_diff_minutes > max_age_minutes:
                    self.logger.warning(
                        f"Stale data for {symbol}: Latest bar is {time_diff_minutes:.1f} min old "
                        f"(threshold: {max_age_minutes} min, timeframe: {timeframe}) - rejecting"
                    )
                    return None
                elif allow_stale and time_diff_minutes > max_age_minutes:
                    self.logger.warning(
                        f"DEBUG MODE: Using stale data for {symbol} ({time_diff_minutes:.1f} min old)"
                    )
                elif time_diff_minutes > max_age_minutes * 0.7:
                    # WARN when approaching threshold (70%)
                    self.logger.info(
                        f"Data aging for {symbol}: {time_diff_minutes:.1f} min old "
                        f"(threshold: {max_age_minutes} min, {time_diff_minutes/max_age_minutes*100:.0f}%)"
                    )

            # BUG FIX (Dec 2025): Use timezone-aware timestamp for cache consistency
            self.cache[cache_key] = {
                'data': df,
                'timestamp': datetime.now(self.market_tz)
            }

            self.logger.info(f"Fetched {len(df)} bars for {symbol} from Yahoo Finance")

            return df

        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}", exc_info=True)

            # BUG FIX (Dec 2025): Implement fallback - return stale cache on error
            if cache_key in self.cache:
                self.logger.warning(f"Returning stale cached data for {symbol} due to API error")
                return self.cache[cache_key]['data']

            return None

    def get_historical_data_range(self, symbol, timeframe='1Min', start_date=None, end_date=None, use_cache=True):
        """
        Get historical data for a specific date range (for backtesting)

        Args:
            symbol: Stock ticker
            timeframe: '1Min', '5Min', '15Min', '1Hour', '1Day'
            start_date: Start date (string YYYY-MM-DD or datetime)
            end_date: End date (string YYYY-MM-DD or datetime)
            use_cache: Whether to use disk cache (default True for 1Hour timeframe)

        Returns:
            pandas DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            # Map timeframe to yfinance interval
            interval_map = {
                '1Min': '1m',
                '5Min': '5m',
                '15Min': '15m',
                '1Hour': '1h',
                '1Day': '1d'
            }

            if timeframe not in interval_map:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return None

            yf_interval = interval_map[timeframe]

            # Format dates
            start_date_str = start_date if isinstance(start_date, str) else start_date.strftime('%Y-%m-%d') if start_date else None
            end_date_str = end_date if isinstance(end_date, str) else end_date.strftime('%Y-%m-%d') if end_date else None

            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                # FIX (Jan 2026): Add 1 day because yfinance end parameter is EXCLUSIVE
                # Without this, end_date='2026-01-06' returns only data up to Jan 5
                end_date = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)

            # Use disk cache for hourly data (most common for backtesting)
            if use_cache and timeframe == '1Hour':
                cache = get_cache()

                # Check if we have fresh cached data
                if cache.has_fresh_data(symbol, end_date_str):
                    cached_df = cache.load(symbol, start_date_str, end_date_str)
                    if cached_df is not None and len(cached_df) > 0:
                        self.logger.info(f"[CACHE HIT] {symbol}: {len(cached_df)} bars from disk cache")
                        return cached_df

            # FIX (Dec 21, 2025): yfinance only allows 8 days of 1-minute data per request
            # For longer ranges, we need to chunk into 7-day windows and combine
            if timeframe == '1Min':
                total_days = (end_date - start_date).days
                if total_days > 7:
                    self.logger.info(f"Chunking {total_days}-day request into 7-day windows for {symbol}")
                    return self._fetch_chunked_1min_data(symbol, start_date, end_date)

            # Fetch data from Yahoo Finance
            # FIX (Jan 16, 2026): Convert symbol to yfinance format (BTC/USD -> BTC-USD)
            yf_symbol = to_yfinance(symbol)
            ticker = yf.Ticker(yf_symbol)

            try:
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=yf_interval,
                    actions=False,
                    auto_adjust=False,
                    prepost=False
                )
            except Exception as e:
                self.logger.error(f"yfinance download error for {symbol}: {e}", exc_info=True)
                return None

            if df is None or df.empty:
                self.logger.warning(f"No data received from Yahoo Finance for {symbol} ({start_date} to {end_date})")
                return None

            # Reset index to make timestamp a column
            df = df.reset_index()

            # BUG FIX (Dec 4, 2025): Handle both 'Datetime' and 'Date' column names
            # yfinance returns 'Datetime' for intraday data but 'Date' for daily data
            def normalize_column(col):
                if col in ('Datetime', 'Date'):
                    return 'timestamp'
                return col.lower()

            df.columns = [normalize_column(col) for col in df.columns]

            # Convert timezone to Eastern Time
            if 'timestamp' in df.columns:
                if df['timestamp'].dt.tz is None:
                    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
                df['timestamp'] = df['timestamp'].dt.tz_convert(self.market_tz)

            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Missing column {col} in data for {symbol}")
                    return None

            self.logger.info(f"[OK] Fetched {len(df)} bars for {symbol} ({start_date} to {end_date})")

            # Save to disk cache for hourly data
            if use_cache and timeframe == '1Hour':
                cache = get_cache()
                cache.save(symbol, df)
                self.logger.debug(f"[CACHE SAVE] {symbol}: {len(df)} bars saved to disk cache")

            return df

        except Exception as e:
            self.logger.error(f"Error fetching historical data range for {symbol}: {e}", exc_info=True)
            return None

    def _fetch_chunked_1min_data(self, symbol, start_date, end_date):
        """
        Fetch 1-minute data in 7-day chunks and combine.

        FIX (Dec 21, 2025): yfinance only allows 8 days of 1-minute data per request.
        This method splits long date ranges into 7-day windows, fetches each,
        and combines them into a single DataFrame.

        Args:
            symbol: Stock ticker
            start_date: Start datetime
            end_date: End datetime

        Returns:
            Combined pandas DataFrame with all 1-minute bars
        """
        all_chunks = []
        current_start = start_date
        chunk_size = timedelta(days=7)  # 7 days per chunk (within 8-day limit)

        # FIX (Jan 16, 2026): Convert symbol to yfinance format (BTC/USD -> BTC-USD)
        yf_symbol = to_yfinance(symbol)
        ticker = yf.Ticker(yf_symbol)

        while current_start < end_date:
            current_end = min(current_start + chunk_size, end_date)

            try:
                # Rate limiting between chunks
                if symbol in self.last_api_call:
                    time_since_last = time.time() - self.last_api_call[symbol]
                    if time_since_last < self.min_call_interval:
                        time.sleep(self.min_call_interval - time_since_last)

                self.last_api_call[symbol] = time.time()

                df_chunk = ticker.history(
                    start=current_start,
                    end=current_end,
                    interval='1m',
                    actions=False,
                    auto_adjust=False,
                    prepost=False
                )

                if df_chunk is not None and not df_chunk.empty:
                    df_chunk = df_chunk.reset_index()
                    all_chunks.append(df_chunk)
                    self.logger.debug(f"Chunk {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}: {len(df_chunk)} bars")

            except Exception as e:
                self.logger.warning(f"Error fetching chunk for {symbol} ({current_start} to {current_end}): {e}")

            current_start = current_end

        if not all_chunks:
            self.logger.warning(f"No data received from any chunk for {symbol}")
            return None

        # Combine all chunks
        df = pd.concat(all_chunks, ignore_index=True)

        # Remove duplicates (overlapping chunks)
        if 'Datetime' in df.columns:
            df = df.drop_duplicates(subset=['Datetime'], keep='first')
        elif 'datetime' in df.columns:
            df = df.drop_duplicates(subset=['datetime'], keep='first')

        # Normalize column names
        def normalize_column(col):
            if col in ('Datetime', 'Date'):
                return 'timestamp'
            return col.lower()

        df.columns = [normalize_column(col) for col in df.columns]

        # Convert timezone to Eastern Time
        if 'timestamp' in df.columns:
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
            df['timestamp'] = df['timestamp'].dt.tz_convert(self.market_tz)

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                self.logger.error(f"Missing column {col} in combined data for {symbol}")
                return None

        self.logger.info(f"[OK] Fetched {len(df)} bars for {symbol} in {len(all_chunks)} chunks")

        return df

    def get_latest_bars(self, symbol, timeframe='1Min', limit=5):
        """
        Get the latest N bars for a symbol (API compatibility wrapper)

        BUG FIX (Dec 9, 2025): Add get_latest_bars() method for API compatibility
        trading_bot_ml.py expects this method, but YFinanceDataFetcher only had get_historical_data()
        This is a simple wrapper that calls get_historical_data() with the same parameters.

        Args:
            symbol: Stock ticker (e.g., 'TSLA')
            timeframe: '1Min', '5Min', '1Hour', or '1Day'
            limit: Number of bars to return

        Returns:
            pandas DataFrame with columns: timestamp, open, high, low, close, volume
        """
        return self.get_historical_data(symbol, timeframe, limit)

    def get_latest_quote(self, symbol):
        """
        Get the latest quote for a symbol

        NOTE: Yahoo Finance doesn't have a real-time quote API like Alpaca.
        This method returns the most recent bar from 1-minute data.
        For true real-time quotes, consider falling back to Alpaca's get_latest_quote().
        """
        try:
            cache_key = f"quote_{symbol}"

            # BUG FIX (Dec 2025): Increase quote cache to 15 seconds - more efficient
            # 5 seconds was causing too many unnecessary API calls
            if cache_key in self.cache:
                cached_time = self.cache[cache_key]['timestamp']
                # BUG FIX (Dec 9, 2025): Use timezone-aware datetime for cache comparison
                # datetime.now() is naive, cached_time is aware - must match
                if (datetime.now(self.market_tz) - cached_time).total_seconds() < 15:  # 15 second cache for quotes
                    return self.cache[cache_key]['data']

            # Get most recent 1-minute bar
            df = self.get_historical_data(symbol, '1Min', limit=1)

            if df is None or df.empty:
                self.logger.warning(f"No quote data for {symbol}")
                return None

            # Extract latest bar data
            latest = df.iloc[-1]

            # BUG FIX (Dec 9, 2025): Validate close price before creating quote
            # CRITICAL: Trading at $0 or negative price would cause catastrophic position sizing errors
            # Must validate price is positive, numeric, and not NaN before creating quote object
            try:
                close_price = float(latest['close'])

                # Validate price is positive and not NaN
                if close_price <= 0 or pd.isna(close_price):
                    self.logger.error(f"Invalid close price for {symbol}: ${close_price}")
                    return None

            except (ValueError, TypeError) as e:
                self.logger.error(f"Could not parse close price for {symbol}: {e}", exc_info=True)
                return None

            # Create a quote-like object
            class Quote:
                def __init__(self, close_price):
                    self.ask_price = close_price
                    self.bid_price = close_price
                    self.last_price = close_price

            quote = Quote(close_price)

            # BUG FIX (Dec 2025): Use timezone-aware timestamp for cache
            self.cache[cache_key] = {
                'data': quote,
                'timestamp': datetime.now(self.market_tz)
            }

            return quote

        except Exception as e:
            self.logger.error(f"Error fetching quote for {symbol}: {e}", exc_info=True)
            return None

    def get_historical_data_batch(self, symbols: list, timeframe: str = '1Min', limit: int = 100, max_workers: int = 10):
        """
        Fetch historical data for multiple symbols in parallel.

        PERFORMANCE UPGRADE (Dec 9, 2025): Parallel data fetching for market scans
        Problem: Sequential fetching = 50 symbols x 0.3s = 15+ seconds per scan (too slow)
        Solution: ThreadPoolExecutor with 10 concurrent threads = ~2-3 seconds for 50 symbols

        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'NVDA', 'TSLA'])
            timeframe: Bar timeframe ('1Min', '5Min', '1Day', etc.)
            limit: Number of bars per symbol
            max_workers: Maximum parallel threads (default 10 to respect rate limits)
                        Yahoo Finance limit: ~2000 req/hour = 33/min
                        10 parallel = ~200 req/min max, well under limit

        Returns:
            Dict[symbol, DataFrame] - Data for each symbol that succeeded
        """
        results = {}
        failed_symbols = []

        self.logger.info(f"Batch fetching {len(symbols)} symbols with {max_workers} parallel workers...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fetch tasks
            future_to_symbol = {
                executor.submit(self.get_historical_data, symbol, timeframe, limit): symbol
                for symbol in symbols
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1

                try:
                    data = future.result()
                    if data is not None and len(data) > 0:
                        results[symbol] = data
                        self.logger.debug(f"[{completed}/{len(symbols)}] {symbol}: {len(data)} bars")
                    else:
                        failed_symbols.append(symbol)
                        self.logger.debug(f"[{completed}/{len(symbols)}] {symbol}: No data")
                except Exception as e:
                    failed_symbols.append(symbol)
                    self.logger.warning(f"[{completed}/{len(symbols)}] {symbol}: {e}")

        elapsed_time = time.time() - start_time
        success_rate = len(results) / len(symbols) * 100 if symbols else 0

        self.logger.info(
            f"Batch fetch complete: {len(results)}/{len(symbols)} succeeded "
            f"({success_rate:.1f}%) in {elapsed_time:.2f}s "
            f"({elapsed_time/len(symbols):.3f}s per symbol)"
        )

        if failed_symbols:
            self.logger.warning(f"Failed symbols ({len(failed_symbols)}): {', '.join(failed_symbols[:10])}"
                              f"{'...' if len(failed_symbols) > 10 else ''}")

        return results


# Alias for compatibility
DataFetcher = YFinanceDataFetcher
