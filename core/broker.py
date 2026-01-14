"""
Broker Abstraction Layer for Trading Bot
Provides unified interface for real (Alpaca) and simulated (Fake) brokers

Combined from:
- broker_interface.py (BrokerInterface ABC, AlpacaBroker, FakeBroker)
- broker_factory.py (BrokerFactory)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pytz
import logging
import math
import time
import os
from dataclasses import dataclass
from functools import wraps

from .config import get_global_config

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(override=True)

# Import requests exceptions for retry handling
try:
    import requests.exceptions
except ImportError:
    requests = None


class RetryableOrderError(Exception):
    """Exception for transient order failures that should be retried.

    Used to wrap rate limit (429), temporary outages (5xx), and other
    transient API errors that are likely to succeed on retry.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception


# Tuple of exception types that should trigger order retry
# These are transient failures that may succeed on a subsequent attempt
RETRYABLE_ORDER_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    RetryableOrderError,
)

# Add requests exceptions if available
if requests is not None:
    RETRYABLE_ORDER_EXCEPTIONS = RETRYABLE_ORDER_EXCEPTIONS + (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectTimeout,
    )


def _is_retryable_api_error(exception: Exception) -> bool:
    """Check if an API exception is retryable based on status code or message.

    Args:
        exception: The exception to check

    Returns:
        True if the error is transient and should be retried
    """
    # Check for status_code attribute (Alpaca APIError)
    status_code = getattr(exception, 'status_code', None)
    if status_code is not None:
        # Rate limit (429) - definitely retryable
        if status_code == 429:
            return True
        # Server errors (5xx) - likely transient
        if 500 <= status_code < 600:
            return True

    # Check error message for common transient patterns
    error_msg = str(exception).lower()
    retryable_patterns = [
        'rate limit',
        'too many requests',
        'service unavailable',
        'gateway timeout',
        'connection reset',
        'temporary',
    ]
    return any(pattern in error_msg for pattern in retryable_patterns)


# Retry decorator for transient API failures
def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0,
                     exceptions: tuple = (Exception,)):
    """
    Decorator to retry a function on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier applied to delay for each retry
        exceptions: Tuple of exception types to catch and retry on
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(self, *args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        self.logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        self.logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}",
                            exc_info=True
                        )

            # Re-raise the last exception after all retries exhausted
            raise last_exception

        return wrapper
    return decorator


class BrokerAPIError(Exception):
    """Custom exception for broker API errors"""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    qty: float
    side: str  # 'long' or 'short'
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float  # Unrealized P&L percentage

    def __repr__(self) -> str:
        return f"Position({self.symbol}, qty={self.qty}, P&L=${self.unrealized_pl:.2f})"


@dataclass
class Order:
    """Represents a trading order"""
    id: str
    symbol: str
    qty: float
    side: str  # 'buy' or 'sell'
    type: str  # 'market', 'limit', 'stop', 'stop_limit'
    status: str  # 'new', 'filled', 'partially_filled', 'cancelled', 'rejected'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_qty: float = 0
    filled_avg_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    def __repr__(self) -> str:
        return f"Order({self.id}, {self.symbol}, {self.side} {self.qty}, status={self.status})"


@dataclass
class PortfolioHistory:
    """Represents historical portfolio equity values."""
    timestamps: List[datetime]
    equity: List[float]
    timeframe: str  # e.g., "1D", "1H"
    base_value: float


@dataclass
class Account:
    """Represents trading account information"""
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    last_equity: float  # Previous day's equity

    @property
    def daily_pnl(self) -> float:
        """Calculate daily P&L"""
        return self.equity - self.last_equity

    @property
    def daily_pnl_percent(self) -> float:
        """Calculate daily P&L percentage"""
        if self.last_equity > 0:
            return (self.equity - self.last_equity) / self.last_equity
        return 0.0

    def __repr__(self) -> str:
        return f"Account(equity=${self.equity:.2f}, daily_pnl=${self.daily_pnl:.2f})"


class BrokerInterface(ABC):
    """Abstract base class for broker implementations"""

    @abstractmethod
    def get_account(self) -> Account:
        """Get account information"""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get all open positions"""
        pass

    @abstractmethod
    def list_positions(self) -> List[Position]:
        """Alias for get_positions (for Alpaca compatibility)"""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        pass

    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        """Get all open orders"""
        pass

    @abstractmethod
    def list_orders(self, status: str = 'open') -> List[Order]:
        """List orders with optional status filter"""
        pass

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        type: str = 'market',
        time_in_force: str = 'day',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """Submit a new order"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID"""
        pass

    @abstractmethod
    def cancel_all_orders(self) -> int:
        """Cancel all open orders, returns number of orders cancelled"""
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> bool:
        """Close position for symbol"""
        pass

    @abstractmethod
    def close_all_positions(self) -> int:
        """Close all positions, returns number of positions closed"""
        pass

    @abstractmethod
    def get_broker_name(self) -> str:
        """Get broker implementation name"""
        pass

    @abstractmethod
    def get_portfolio_history(self, period: str = "30D") -> PortfolioHistory:
        """Get historical portfolio equity values.

        Args:
            period: Time period for history. Options: "7D", "30D", "90D", "1Y", "ALL"

        Returns:
            PortfolioHistory with timestamps and equity values
        """
        pass

    @abstractmethod
    def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        stop_loss_percent: float = 0.05,
        time_in_force: str = 'gtc',
        **kwargs
    ) -> Order:
        """Submit market entry with attached stop-loss order for crash protection.

        Creates a bracket order with:
        - Market entry order (executes immediately)
        - Stop-loss order at stop_loss_percent below/above entry (for long/short)
        - Take-profit order set far away (won't trigger, just for bracket structure)

        Args:
            symbol: Stock ticker symbol
            qty: Number of shares
            side: 'buy' for long entry, 'sell' for short entry
            stop_loss_percent: Percentage for stop-loss (default 5%)
            time_in_force: Order duration ('gtc' recommended for stop persistence)
            **kwargs: Additional params, especially 'price' for stop calculation

        Returns:
            Order with stop_order_id attribute for tracking the stop leg
        """
        pass


class AlpacaBroker(BrokerInterface):
    """Real broker implementation using Alpaca API"""

    # Safety flag: Set to True ONLY for integration tests that intentionally use real API
    _allow_in_tests = False

    def __init__(self, api_key: str, secret_key: str, base_url: str = None):
        """
        Initialize Alpaca broker

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            base_url: Base URL (paper or live endpoint)

        Raises:
            RuntimeError: If called from a test environment without explicit permission
        """
        import sys
        import alpaca_trade_api as tradeapi

        self.logger = logging.getLogger(__name__)

        # SAFETY: Block real broker in test environments
        # This prevents tests from accidentally placing real orders
        in_pytest = 'pytest' in sys.modules
        in_unittest = 'unittest' in sys.modules and any(
            'test_' in arg or 'tests/' in arg for arg in sys.argv
        )
        testing_env = os.environ.get('TESTING', '').lower() in ('1', 'true', 'yes')

        if (in_pytest or in_unittest or testing_env) and not AlpacaBroker._allow_in_tests:
            raise RuntimeError(
                "SAFETY BLOCK: AlpacaBroker cannot be used in test environment! "
                "Tests must use FakeBroker to prevent accidental real orders. "
                "If this is an intentional integration test, set AlpacaBroker._allow_in_tests = True"
            )

        # Initialize Alpaca API
        if base_url:
            self.api = tradeapi.REST(api_key, secret_key, base_url)
        else:
            self.api = tradeapi.REST(api_key, secret_key)

        # Track rate limiting to avoid API throttling
        # Alpaca has 200 requests/minute limit
        self._request_times: List[float] = []
        self._rate_limit_window = 60.0  # seconds
        self._rate_limit_max = 180  # Conservative: 180/min instead of 200/min

        self.logger.info(f"AlpacaBroker initialized with endpoint: {base_url or 'default'}")

    def _check_rate_limit(self):
        """
        Check if we're approaching rate limit and sleep if necessary.
        """
        now = time.time()

        # Remove requests older than the window
        self._request_times = [t for t in self._request_times if now - t < self._rate_limit_window]

        # Check if we're at the limit
        if len(self._request_times) >= self._rate_limit_max:
            # Calculate how long until the oldest request expires
            oldest = self._request_times[0]
            sleep_time = self._rate_limit_window - (now - oldest) + 0.1  # Add 100ms buffer
            if sleep_time > 0:
                self.logger.warning(f"Rate limit approaching ({len(self._request_times)} requests), sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                # Clean up old requests after sleep
                now = time.time()
                self._request_times = [t for t in self._request_times if now - t < self._rate_limit_window]

        # Record this request
        self._request_times.append(now)

    @retry_on_failure(max_retries=3, delay=0.5, backoff=2.0)
    def get_account(self) -> Account:
        """Get account information from Alpaca."""
        self._check_rate_limit()
        try:
            alpaca_account = self.api.get_account()
            return Account(
                equity=float(alpaca_account.equity),
                cash=float(alpaca_account.cash),
                buying_power=float(alpaca_account.buying_power),
                portfolio_value=float(alpaca_account.portfolio_value),
                last_equity=float(alpaca_account.last_equity)
            )
        except Exception as e:
            self.logger.error(f"Failed to get account: {e}", exc_info=True)
            raise BrokerAPIError(f"Failed to get account: {e}", original_exception=e)

    @retry_on_failure(max_retries=3, delay=0.5, backoff=2.0)
    def get_positions(self) -> List[Position]:
        """Get all positions from Alpaca."""
        self._check_rate_limit()
        try:
            alpaca_positions = self.api.list_positions()
            if alpaca_positions is None:
                self.logger.warning("Alpaca returned None for positions, returning empty list")
                return []

            positions = []
            for pos in alpaca_positions:
                positions.append(Position(
                    symbol=pos.symbol,
                    qty=abs(float(pos.qty)),
                    side='long' if float(pos.qty) > 0 else 'short',
                    avg_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    market_value=float(pos.market_value),
                    unrealized_pl=float(pos.unrealized_pl),
                    unrealized_plpc=float(pos.unrealized_plpc)
                ))

            return positions
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}", exc_info=True)
            raise BrokerAPIError(f"Failed to get positions: {e}", original_exception=e)

    def list_positions(self) -> List[Position]:
        """Alias for get_positions"""
        return self.get_positions()

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        # Validate symbol input
        if not symbol or not isinstance(symbol, str) or not symbol.strip():
            self.logger.warning(f"Invalid symbol provided to get_position: '{symbol}'")
            return None
        symbol = symbol.strip().upper()

        self._check_rate_limit()
        try:
            pos = self.api.get_position(symbol)
            return Position(
                symbol=pos.symbol,
                qty=abs(float(pos.qty)),
                side='long' if float(pos.qty) > 0 else 'short',
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                unrealized_pl=float(pos.unrealized_pl),
                unrealized_plpc=float(pos.unrealized_plpc)
            )
        except Exception as e:
            # Position not found is expected - log at debug level
            error_str = str(e).lower()
            if 'position does not exist' in error_str or '404' in error_str:
                self.logger.debug(f"Position not found for {symbol}: {e}")
            else:
                # Other errors should be logged as warnings
                self.logger.warning(f"Error getting position for {symbol}: {e}")
            return None

    @retry_on_failure(max_retries=2, delay=0.5, backoff=2.0)
    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        self._check_rate_limit()
        try:
            alpaca_orders = self.api.list_orders(status='open')
            return self._convert_orders(alpaca_orders)
        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}", exc_info=True)
            raise BrokerAPIError(f"Failed to get open orders: {e}", original_exception=e)

    @retry_on_failure(max_retries=2, delay=0.5, backoff=2.0)
    def list_orders(self, status: str = 'open', **kwargs) -> List[Order]:
        """List orders with status filter."""
        self._check_rate_limit()
        try:
            alpaca_orders = self.api.list_orders(status=status, **kwargs)
            return self._convert_orders(alpaca_orders)
        except Exception as e:
            self.logger.error(f"Failed to list orders (status={status}): {e}", exc_info=True)
            raise BrokerAPIError(f"Failed to list orders: {e}", original_exception=e)

    def _convert_orders(self, alpaca_orders) -> List[Order]:
        """Convert Alpaca orders to our Order objects"""
        orders = []
        for ord in alpaca_orders:
            orders.append(Order(
                id=ord.id,
                symbol=ord.symbol,
                qty=float(ord.qty),
                side=ord.side,
                type=ord.type,
                status=ord.status,
                limit_price=float(ord.limit_price) if ord.limit_price is not None else None,
                stop_price=float(ord.stop_price) if ord.stop_price is not None else None,
                filled_qty=float(ord.filled_qty) if ord.filled_qty is not None else 0,
                filled_avg_price=float(ord.filled_avg_price) if ord.filled_avg_price is not None else None,
                submitted_at=ord.submitted_at,
                filled_at=ord.filled_at
            ))
        return orders

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        type: str = 'market',
        time_in_force: str = 'day',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """Submit order to Alpaca with retry logic for transient failures.

        Retries on:
        - ConnectionError, TimeoutError (network issues)
        - Rate limit errors (429)
        - Temporary API outages (5xx)

        Does NOT retry on:
        - Insufficient buying power
        - Invalid symbol
        - Order rejected by exchange
        - Other business logic errors
        """
        # Validate symbol input
        if not symbol or not isinstance(symbol, str) or not symbol.strip():
            raise BrokerAPIError(f"Invalid symbol: '{symbol}'. Symbol must be a non-empty string.")
        symbol = symbol.strip().upper()

        # Validate side
        if side not in ['buy', 'sell']:
            raise BrokerAPIError(f"Invalid side '{side}'. Must be 'buy' or 'sell'.")

        # Validate and convert quantity to integer for stocks
        if not isinstance(qty, int):
            original_qty = qty
            qty = int(qty)
            if qty <= 0:
                raise BrokerAPIError(
                    f"Invalid quantity: {original_qty} rounds to {qty} shares. Minimum is 1 share."
                )
            if abs(original_qty - qty) > 0.01:
                self.logger.info(f"Rounded quantity from {original_qty} to {qty} shares for {symbol}")

        # Validate order type
        valid_types = ['market', 'limit', 'stop', 'stop_limit', 'trailing_stop']
        if type not in valid_types:
            raise BrokerAPIError(f"Invalid order type '{type}'. Must be one of: {valid_types}")

        # Validate limit_price for limit orders
        if type in ['limit', 'stop_limit'] and limit_price is None:
            raise BrokerAPIError(f"limit_price required for {type} orders")

        # Validate stop_price for stop orders
        if type in ['stop', 'stop_limit'] and stop_price is None:
            raise BrokerAPIError(f"stop_price required for {type} orders")

        # Retry configuration for order submission
        max_retries = 3
        delay = 1.0
        backoff = 2.0
        last_exception = None
        current_delay = delay

        for attempt in range(max_retries + 1):
            self._check_rate_limit()

            try:
                self.logger.info(
                    f"Submitting order: {side.upper()} {qty} {symbol} @ {type} "
                    f"(limit={limit_price}, stop={stop_price})"
                )

                alpaca_order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type=type,
                    time_in_force=time_in_force,
                    limit_price=limit_price,
                    stop_price=stop_price
                )

                self.logger.info(f"Order submitted successfully: {alpaca_order.id}")

                return Order(
                    id=alpaca_order.id,
                    symbol=alpaca_order.symbol,
                    qty=float(alpaca_order.qty),
                    side=alpaca_order.side,
                    type=alpaca_order.type,
                    status=alpaca_order.status,
                    limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                    stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                    submitted_at=alpaca_order.submitted_at
                )

            except RETRYABLE_ORDER_EXCEPTIONS as e:
                # Known retryable exceptions - retry with backoff
                last_exception = e
                if attempt < max_retries:
                    self.logger.warning(
                        f"submit_order failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
                else:
                    self.logger.error(
                        f"submit_order failed after {max_retries + 1} attempts: {e}",
                        exc_info=True
                    )

            except Exception as e:
                # Check if this is a retryable API error (rate limit, 5xx)
                if _is_retryable_api_error(e):
                    last_exception = e
                    if attempt < max_retries:
                        self.logger.warning(
                            f"submit_order failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        self.logger.error(
                            f"submit_order failed after {max_retries + 1} attempts: {e}",
                            exc_info=True
                        )
                else:
                    # Non-retryable error (business logic failure)
                    self.logger.error(
                        f"Failed to submit order: {side.upper()} {qty} {symbol} @ {type} - {e}",
                        exc_info=True
                    )
                    raise BrokerAPIError(
                        f"Order submission failed for {symbol}: {e}",
                        original_exception=e
                    )

        # All retries exhausted
        if last_exception:
            raise last_exception

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order by ID."""
        self._check_rate_limit()
        try:
            # First check if order exists and is cancellable
            try:
                order = self.api.get_order(order_id)
                cancellable_states = ['new', 'partially_filled', 'accepted', 'pending_new']
                if order.status not in cancellable_states:
                    self.logger.debug(
                        f"Order {order_id} cannot be cancelled - status is '{order.status}'"
                    )
                    return False
            except Exception as e:
                self.logger.warning(f"Cannot verify order {order_id} status: {e}")

            # Attempt cancellation
            self.api.cancel_order(order_id)
            self.logger.info(f"Successfully cancelled order {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}", exc_info=True)
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders"""
        self._check_rate_limit()
        try:
            result = self.api.cancel_all_orders()
            count = len(result) if result else 0
            self.logger.info(f"Cancelled {count} orders")
            return count
        except Exception as e:
            self.logger.error(f"Failed to cancel all orders: {e}", exc_info=True)
            return 0

    def close_position(self, symbol: str) -> bool:
        """Close position for symbol"""
        self._check_rate_limit()
        try:
            self.api.close_position(symbol)
            self.logger.info(f"Closed position: {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to close position {symbol}: {e}", exc_info=True)
            return False

    def close_all_positions(self) -> int:
        """Close all positions"""
        self._check_rate_limit()
        try:
            result = self.api.close_all_positions()
            count = len(result) if result else 0
            self.logger.info(f"Closed {count} positions")
            return count
        except Exception as e:
            self.logger.error(f"Failed to close all positions: {e}", exc_info=True)
            return 0

    def get_broker_name(self) -> str:
        return "AlpacaBroker"

    @retry_on_failure(max_retries=2, delay=0.5, backoff=2.0)
    def get_portfolio_history(self, period: str = "30D") -> PortfolioHistory:
        """Get historical portfolio equity values from Alpaca.

        Args:
            period: Time period. Options: "7D", "30D", "90D", "1Y", "ALL"

        Returns:
            PortfolioHistory with timestamps and equity values
        """
        self._check_rate_limit()
        try:
            # Map our periods to Alpaca's period format
            period_map = {
                "7D": "1W",
                "30D": "1M",
                "90D": "3M",
                "1Y": "1A",
                "ALL": "all",
            }
            alpaca_period = period_map.get(period, "1M")

            # Use 1D timeframe for daily granularity
            alpaca_history = self.api.get_portfolio_history(
                period=alpaca_period,
                timeframe="1D"
            )

            # Convert timestamps from Unix to datetime
            timestamps = []
            if alpaca_history.timestamp:
                for ts in alpaca_history.timestamp:
                    if isinstance(ts, (int, float)):
                        timestamps.append(datetime.fromtimestamp(ts, tz=pytz.UTC))
                    else:
                        timestamps.append(ts)

            equity = list(alpaca_history.equity) if alpaca_history.equity else []
            base_value = float(alpaca_history.base_value) if alpaca_history.base_value else 0.0

            return PortfolioHistory(
                timestamps=timestamps,
                equity=equity,
                timeframe="1D",
                base_value=base_value
            )

        except Exception as e:
            self.logger.error(f"Failed to get portfolio history: {e}", exc_info=True)
            # Return empty history on error
            return PortfolioHistory(
                timestamps=[],
                equity=[],
                timeframe="1D",
                base_value=0.0
            )

    def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        stop_loss_percent: float = 0.05,
        time_in_force: str = 'gtc',
        **kwargs
    ) -> Order:
        """Submit bracket order to Alpaca with stop-loss for crash protection.

        Args:
            symbol: Stock ticker
            qty: Number of shares
            side: 'buy' or 'sell'
            stop_loss_percent: Stop-loss percentage (default 5%)
            time_in_force: Order duration (default 'gtc')
            **kwargs: Must include 'price' for stop calculation

        Returns:
            Order with stop_order_id attribute
        """
        price = kwargs.get('price')
        if price is None:
            raise BrokerAPIError("price is required for bracket orders")

        # Calculate stop and take-profit prices
        if side == 'buy':
            # LONG: stop below entry, take-profit above
            stop_price = round(price * (1 - stop_loss_percent), 2)
            take_profit_price = round(price * 2, 2)  # Set high, won't trigger
        else:
            # SHORT: stop above entry, take-profit below
            stop_price = round(price * (1 + stop_loss_percent), 2)
            take_profit_price = round(price * 0.5, 2)  # Set low, won't trigger

        self._check_rate_limit()

        try:
            self.logger.info(
                f"Submitting bracket order: {side.upper()} {qty} {symbol} "
                f"(stop={stop_price}, tp={take_profit_price})"
            )

            alpaca_order = self.api.submit_order(
                symbol=symbol,
                qty=int(qty),
                side=side,
                type='market',
                time_in_force=time_in_force,
                order_class='bracket',
                stop_loss={'stop_price': stop_price},
                take_profit={'limit_price': take_profit_price}
            )

            self.logger.info(f"Bracket order submitted: {alpaca_order.id}")

            # Extract stop and take-profit order IDs from legs
            # Both must be cancelled before any exit to free locked shares
            stop_order_id = None
            take_profit_order_id = None
            if hasattr(alpaca_order, 'legs') and alpaca_order.legs:
                for leg in alpaca_order.legs:
                    if hasattr(leg, 'type'):
                        if leg.type == 'stop':
                            stop_order_id = leg.id
                        elif leg.type == 'limit':
                            take_profit_order_id = leg.id

            order = Order(
                id=alpaca_order.id,
                symbol=alpaca_order.symbol,
                qty=float(alpaca_order.qty),
                side=alpaca_order.side,
                type=alpaca_order.type,
                status=alpaca_order.status,
                limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                submitted_at=alpaca_order.submitted_at
            )

            # Attach bracket leg order IDs for tracking
            order.stop_order_id = stop_order_id
            order.take_profit_order_id = take_profit_order_id

            return order

        except Exception as e:
            self.logger.error(f"Failed to submit bracket order: {e}", exc_info=True)
            raise BrokerAPIError(f"Bracket order failed for {symbol}: {e}", original_exception=e)


class FakeBroker(BrokerInterface):
    """Simulated broker for BACKTEST and DRY_RUN modes"""

    def __init__(self, initial_cash: float = 100000, commission: float = 0, slippage: float = 0.0005):
        """
        Initialize fake broker

        Args:
            initial_cash: Starting cash
            commission: Commission per trade
            slippage: Slippage percentage (0.0005 = 0.05%)
        """
        self.logger = logging.getLogger(__name__)

        # Account state
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission = commission
        self.slippage = slippage

        # Positions: {symbol: {qty, avg_entry_price, side}}
        self.positions: Dict[str, Dict[str, Any]] = {}

        # Track current prices separately from entry prices
        self.current_prices: Dict[str, float] = {}

        # Orders: {order_id: Order}
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0

        # Stop orders for bracket order simulation: {symbol: {order_id, stop_price, qty, side}}
        self.stop_orders: Dict[str, Dict[str, Any]] = {}

        # Execution log
        self.execution_log: List[Dict[str, Any]] = []

        self.logger.info(f"FakeBroker initialized with ${initial_cash:,.2f}")

    def update_price(self, symbol: str, price: float):
        """
        Update current price for a symbol (used for P&L calculations).
        Also checks if any stop orders should be triggered.

        Args:
            symbol: Stock ticker
            price: Current market price
        """
        if price > 0:
            if symbol in self.positions:
                self.current_prices[symbol] = price

                # Check if stop order should be triggered
                if symbol in self.stop_orders:
                    stop = self.stop_orders[symbol]
                    triggered = False

                    if stop['side'] == 'sell' and price <= stop['stop_price']:
                        # LONG position stop triggered (price dropped to stop)
                        triggered = True
                    elif stop['side'] == 'buy' and price >= stop['stop_price']:
                        # SHORT position stop triggered (price rose to stop)
                        triggered = True

                    if triggered:
                        self.logger.warning(
                            f"STOP TRIGGERED | {symbol} | Price ${price:.2f} hit stop ${stop['stop_price']:.2f}"
                        )
                        self._execute_stop_order(symbol, price)
            else:
                self.logger.debug(
                    f"Price update for {symbol} (${price:.2f}) ignored - no position"
                )

    def _execute_stop_order(self, symbol: str, trigger_price: float):
        """Execute a triggered stop order.

        Args:
            symbol: Stock ticker
            trigger_price: Price that triggered the stop
        """
        if symbol not in self.stop_orders:
            return

        stop = self.stop_orders[symbol]

        # Execute the stop order (this will close position and clear stop_orders via _execute_order)
        self.submit_order(
            symbol=symbol,
            qty=stop['qty'],
            side=stop['side'],
            type='market',
            price=trigger_price
        )

        # Note: stop_orders[symbol] is already removed by _execute_order when position closes
        self.logger.info(f"Stop order executed for {symbol}")

    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self.order_counter += 1
        return f"FAKE_{self.order_counter:06d}"

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to price"""
        if side == 'buy':
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)

    def get_account(self) -> Account:
        """Get simulated account information"""
        # Calculate portfolio value
        portfolio_value = self.cash
        for symbol, pos_data in self.positions.items():
            current_price = self.current_prices.get(symbol, pos_data['avg_entry_price'])
            portfolio_value += pos_data['qty'] * current_price

        return Account(
            equity=portfolio_value,
            cash=self.cash,
            buying_power=self.cash,  # Simplified
            portfolio_value=portfolio_value,
            last_equity=self.initial_cash
        )

    def get_positions(self) -> List[Position]:
        """Get all simulated positions (LONG and SHORT)"""
        positions = []
        for symbol, pos_data in self.positions.items():
            current_price = self.current_prices.get(symbol, pos_data['avg_entry_price'])
            qty = pos_data['qty']
            side = pos_data.get('side', 'long')
            market_value = qty * current_price
            cost_basis = qty * pos_data['avg_entry_price']

            # Calculate unrealized P&L based on position side
            if side == 'long':
                # LONG: Profit when current price > entry price
                unrealized_pl = market_value - cost_basis
            else:
                # SHORT: Profit when current price < entry price
                unrealized_pl = cost_basis - market_value

            unrealized_plpc = (unrealized_pl / cost_basis) if cost_basis > 0 else 0

            positions.append(Position(
                symbol=symbol,
                qty=qty,
                side=side,
                avg_entry_price=pos_data['avg_entry_price'],
                current_price=current_price,
                market_value=market_value,
                unrealized_pl=unrealized_pl,
                unrealized_plpc=unrealized_plpc
            ))

        return positions

    def list_positions(self) -> List[Position]:
        """Alias for get_positions"""
        return self.get_positions()

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol (LONG or SHORT)"""
        if symbol not in self.positions:
            return None

        pos_data = self.positions[symbol]
        current_price = self.current_prices.get(symbol, pos_data['avg_entry_price'])
        qty = pos_data['qty']
        side = pos_data.get('side', 'long')
        market_value = qty * current_price
        cost_basis = qty * pos_data['avg_entry_price']

        # Calculate unrealized P&L based on position side
        if side == 'long':
            unrealized_pl = market_value - cost_basis
        else:
            unrealized_pl = cost_basis - market_value

        unrealized_plpc = (unrealized_pl / cost_basis) if cost_basis > 0 else 0

        return Position(
            symbol=symbol,
            qty=qty,
            side=side,
            avg_entry_price=pos_data['avg_entry_price'],
            current_price=current_price,
            market_value=market_value,
            unrealized_pl=unrealized_pl,
            unrealized_plpc=unrealized_plpc
        )

    def get_open_orders(self) -> List[Order]:
        """Get all open orders"""
        return [order for order in self.orders.values() if order.status == 'new']

    def list_orders(self, status: str = 'open') -> List[Order]:
        """List orders with status filter"""
        if status == 'open':
            return self.get_open_orders()
        return [order for order in self.orders.values() if order.status == status]

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        type: str = 'market',
        time_in_force: str = 'day',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """Simulate order submission."""
        # Validate symbol input
        if not symbol or not isinstance(symbol, str) or not symbol.strip():
            self.logger.error(f"FakeBroker: Invalid symbol '{symbol}'")
            order = Order(
                id=self._generate_order_id(),
                symbol=str(symbol) if symbol else 'INVALID',
                qty=qty,
                side=side,
                type=type,
                status='rejected',
                submitted_at=datetime.now(pytz.UTC)
            )
            self.orders[order.id] = order
            return order
        symbol = symbol.strip().upper()

        # Validate side
        if side not in ['buy', 'sell']:
            self.logger.error(f"FakeBroker: Invalid side '{side}' - must be 'buy' or 'sell'")
            order = Order(
                id=self._generate_order_id(),
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                status='rejected',
                submitted_at=datetime.now(pytz.UTC)
            )
            self.orders[order.id] = order
            return order

        # Validate and convert quantity to integer for consistency
        if not isinstance(qty, int):
            original_qty = qty
            qty = int(qty)
            if qty <= 0:
                self.logger.error(
                    f"FakeBroker: Invalid quantity {original_qty} rounds to {qty} - rejecting order"
                )
                order = Order(
                    id=self._generate_order_id(),
                    symbol=symbol,
                    qty=original_qty,
                    side=side,
                    type=type,
                    status='rejected',
                    submitted_at=datetime.now(pytz.UTC)
                )
                self.orders[order.id] = order
                return order
            if abs(original_qty - qty) > 0.01:
                self.logger.info(f"FakeBroker: Rounded quantity from {original_qty} to {qty} shares")

        order_id = self._generate_order_id()
        order = Order(
            id=order_id,
            symbol=symbol,
            qty=qty,
            side=side,
            type=type,
            status='new',
            limit_price=limit_price,
            stop_price=stop_price,
            submitted_at=datetime.now(pytz.UTC)
        )

        self.orders[order_id] = order

        # For market orders, execute immediately
        if type == 'market':
            price = kwargs.get('price')
            if price is None:
                # Try to get from tracked current prices
                price = self.current_prices.get(symbol)
            if price is None:
                # Last resort: reject order
                self.logger.error(f"FakeBroker: No price provided for {symbol} market order - rejecting")
                order.status = 'rejected'
                return order
            self._execute_order(order, price)

        self.logger.info(f"FakeBroker: Submitted {side.upper()} order for {qty} {symbol} (Order ID: {order_id})")

        return order

    def _execute_order(self, order: Order, price: float):
        """Execute a simulated order

        Supports both LONG and SHORT positions:
        - BUY with no position or LONG position: Opens/adds to LONG
        - BUY with SHORT position: Covers (closes) SHORT
        - SELL with no position: Opens SHORT
        - SELL with LONG position: Closes LONG
        - SELL with SHORT position: Adds to SHORT
        """
        # Apply slippage
        execution_price = self._apply_slippage(price, order.side)

        # Calculate cost
        cost = order.qty * execution_price + self.commission

        if order.side == 'buy':
            # Check if we have a SHORT position to cover
            if order.symbol in self.positions and self.positions[order.symbol]['side'] == 'short':
                # COVERING SHORT: Buy to close short position
                pos = self.positions[order.symbol]
                cover_qty = min(order.qty, pos['qty'])
                cover_cost = cover_qty * execution_price + self.commission

                if cover_cost > self.cash:
                    order.status = 'rejected'
                    self.logger.warning(f"FakeBroker: Order {order.id} rejected - insufficient cash to cover short")
                    return

                self.cash -= cover_cost
                pos['qty'] -= cover_qty

                if pos['qty'] <= 0 or math.isclose(pos['qty'], 0, abs_tol=1e-9):
                    del self.positions[order.symbol]
                    self.current_prices.pop(order.symbol, None)
                    # Clear associated stop order when position fully closed
                    self.stop_orders.pop(order.symbol, None)
                    self.logger.info(f"FakeBroker: Covered SHORT {order.symbol} x{cover_qty} @ ${execution_price:.2f}")
                else:
                    self.logger.info(f"FakeBroker: Partially covered SHORT {order.symbol} x{cover_qty} @ ${execution_price:.2f}")
            else:
                # OPENING/ADDING TO LONG position
                if cost > self.cash:
                    order.status = 'rejected'
                    self.logger.warning(f"FakeBroker: Order {order.id} rejected - insufficient cash")
                    return

                # Update cash
                self.cash -= cost

                # Update position
                if order.symbol in self.positions:
                    # Add to existing LONG position
                    pos = self.positions[order.symbol]
                    total_qty = pos['qty'] + order.qty
                    total_cost = (pos['qty'] * pos['avg_entry_price']) + (order.qty * execution_price)
                    pos['avg_entry_price'] = total_cost / total_qty
                    pos['qty'] = total_qty
                else:
                    # Create new LONG position
                    self.positions[order.symbol] = {
                        'qty': order.qty,
                        'avg_entry_price': execution_price,
                        'side': 'long'
                    }
                    self.current_prices[order.symbol] = execution_price

        else:  # sell
            if order.symbol not in self.positions:
                # OPENING SHORT: No existing position, this is a short sale
                # Simplified margin model: 50% margin requirement
                margin_required = order.qty * execution_price * 0.5
                if margin_required > self.cash:
                    order.status = 'rejected'
                    self.logger.warning(f"FakeBroker: Order {order.id} rejected - insufficient margin for short")
                    return

                # Receive sale proceeds (simplified)
                self.cash += (order.qty * execution_price) - self.commission

                # Create SHORT position
                self.positions[order.symbol] = {
                    'qty': order.qty,
                    'avg_entry_price': execution_price,
                    'side': 'short'
                }
                self.current_prices[order.symbol] = execution_price
                self.logger.info(f"FakeBroker: Opened SHORT {order.symbol} x{order.qty} @ ${execution_price:.2f}")

            elif self.positions[order.symbol]['side'] == 'short':
                # ADDING TO SHORT: Scale into existing short position
                pos = self.positions[order.symbol]
                total_qty = pos['qty'] + order.qty
                total_proceeds = (pos['qty'] * pos['avg_entry_price']) + (order.qty * execution_price)
                pos['avg_entry_price'] = total_proceeds / total_qty
                pos['qty'] = total_qty
                self.cash += (order.qty * execution_price) - self.commission
                self.logger.info(f"FakeBroker: Added to SHORT {order.symbol} x{order.qty} @ ${execution_price:.2f}")

            else:
                # CLOSING LONG: Existing LONG position
                pos = self.positions[order.symbol]
                pos_qty = pos['qty']

                if pos_qty < order.qty and not math.isclose(pos_qty, order.qty, rel_tol=1e-9):
                    order.status = 'rejected'
                    self.logger.warning(f"FakeBroker: Order {order.id} rejected - insufficient shares (have {pos_qty}, need {order.qty})")
                    return

                # Update cash
                self.cash += (order.qty * execution_price) - self.commission

                # Update position
                pos['qty'] -= order.qty
                if pos['qty'] <= 0 or math.isclose(pos['qty'], 0, abs_tol=1e-9):
                    del self.positions[order.symbol]
                    self.current_prices.pop(order.symbol, None)
                    # Clear associated stop order when position fully closed
                    self.stop_orders.pop(order.symbol, None)

        # Mark order as filled
        order.status = 'filled'
        order.filled_qty = order.qty
        order.filled_avg_price = execution_price
        order.filled_at = datetime.now(pytz.UTC)

        # Log execution
        self.execution_log.append({
            'timestamp': datetime.now(pytz.UTC),
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side,
            'qty': order.qty,
            'price': execution_price,
            'commission': self.commission
        })

        self.logger.info(f"FakeBroker: Executed {order.side.upper()} {order.qty} {order.symbol} @ ${execution_price:.2f}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel simulated order (including stop orders)"""
        # Check if this is a stop order and remove from stop_orders tracking
        for symbol, stop_data in list(self.stop_orders.items()):
            if stop_data['order_id'] == order_id:
                del self.stop_orders[symbol]
                self.logger.info(f"FakeBroker: Cancelled stop order {order_id} for {symbol}")
                # Also update the order status in orders dict
                if order_id in self.orders:
                    self.orders[order_id].status = 'cancelled'
                return True

        # Regular order cancellation
        if order_id in self.orders:
            if self.orders[order_id].status == 'new':
                self.orders[order_id].status = 'cancelled'
                self.logger.info(f"FakeBroker: Cancelled order {order_id}")
                return True
        return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders"""
        count = 0
        for order in self.orders.values():
            if order.status == 'new':
                order.status = 'cancelled'
                count += 1
        self.logger.info(f"FakeBroker: Cancelled {count} orders")
        return count

    def close_position(self, symbol: str) -> bool:
        """Close simulated position (LONG or SHORT) and cancel associated stop order"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            side = pos.get('side', 'long')
            close_price = self.current_prices.get(symbol, pos['avg_entry_price'])

            # Cancel any associated stop order first
            if symbol in self.stop_orders:
                stop_order_id = self.stop_orders[symbol]['order_id']
                self.cancel_order(stop_order_id)

            # Use correct side to close position
            if side == 'long':
                close_side = 'sell'
            else:
                close_side = 'buy'

            self.submit_order(
                symbol=symbol,
                qty=pos['qty'],
                side=close_side,
                type='market',
                price=close_price
            )
            return True
        return False

    def close_all_positions(self) -> int:
        """Close all simulated positions"""
        count = 0
        for symbol in list(self.positions.keys()):
            if self.close_position(symbol):
                count += 1
        self.logger.info(f"FakeBroker: Closed {count} positions")
        return count

    def get_broker_name(self) -> str:
        return "FakeBroker"

    def get_portfolio_history(self, period: str = "30D") -> PortfolioHistory:
        """Get mock portfolio history for simulation.

        Generates synthetic equity curve data based on initial cash
        and a slight upward trend with noise.

        Args:
            period: Time period. Options: "7D", "30D", "90D", "1Y", "ALL"

        Returns:
            PortfolioHistory with mock timestamps and equity values
        """
        import random

        # Map period to number of days
        period_days = {
            "7D": 7,
            "30D": 30,
            "90D": 90,
            "1Y": 365,
            "ALL": 365,
        }
        num_days = period_days.get(period, 30)

        # Generate timestamps
        now = datetime.now(pytz.UTC)
        timestamps = [
            now - timedelta(days=num_days - i)
            for i in range(num_days)
        ]

        # Generate mock equity curve
        # Start at initial cash, add small random daily changes
        equity = []
        current_value = self.initial_cash
        random.seed(42)  # Reproducible for testing

        for i in range(num_days):
            # Small daily change: -0.5% to +0.7% (slight positive bias)
            daily_return = random.uniform(-0.005, 0.007)
            current_value *= (1 + daily_return)
            equity.append(round(current_value, 2))

        return PortfolioHistory(
            timestamps=timestamps,
            equity=equity,
            timeframe="1D",
            base_value=self.initial_cash
        )

    def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        stop_loss_percent: float = 0.05,
        time_in_force: str = 'gtc',
        **kwargs
    ) -> Order:
        """Submit bracket order with simulated stop-loss tracking.

        Args:
            symbol: Stock ticker
            qty: Number of shares
            side: 'buy' or 'sell'
            stop_loss_percent: Stop-loss percentage (default 5%)
            time_in_force: Ignored in simulation
            **kwargs: Must include 'price' for stop calculation

        Returns:
            Order with stop_order_id attribute
        """
        price = kwargs.get('price')
        if price is None:
            self.logger.error("FakeBroker: price required for bracket order")
            order = Order(
                id=self._generate_order_id(),
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                status='rejected',
                submitted_at=datetime.now(pytz.UTC)
            )
            order.stop_order_id = None
            return order

        # Execute the entry order
        entry_order = self.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            price=price
        )

        # If entry failed, don't create stop order
        if entry_order.status != 'filled':
            entry_order.stop_order_id = None
            return entry_order

        # Calculate stop price based on direction
        # Use original price (not slipped price) for consistent stop placement
        if side == 'buy':
            # LONG: stop below entry
            stop_price = price * (1 - stop_loss_percent)
            stop_side = 'sell'
        else:
            # SHORT: stop above entry
            stop_price = price * (1 + stop_loss_percent)
            stop_side = 'buy'

        # Create and track the stop order
        stop_order_id = self._generate_order_id()
        self.stop_orders[symbol] = {
            'order_id': stop_order_id,
            'stop_price': stop_price,
            'qty': entry_order.filled_qty,
            'side': stop_side
        }

        # Also track as a pending stop order in orders dict
        stop_order = Order(
            id=stop_order_id,
            symbol=symbol,
            qty=entry_order.filled_qty,
            side=stop_side,
            type='stop',
            status='new',
            stop_price=stop_price,
            submitted_at=datetime.now(pytz.UTC)
        )
        self.orders[stop_order_id] = stop_order

        # Attach stop order ID to entry order for tracking
        entry_order.stop_order_id = stop_order_id

        self.logger.info(
            f"FakeBroker: Bracket order created - Entry {side.upper()} {qty} {symbol}, "
            f"Stop {stop_side.upper()} @ ${stop_price:.2f}"
        )

        return entry_order

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get all executions for analysis"""
        return self.execution_log

    def reset(self):
        """
        Reset FakeBroker to initial state.

        Clears all positions, orders, stop orders, and resets cash to initial value.
        """
        self.cash = self.initial_cash
        self.positions.clear()
        self.current_prices.clear()
        self.orders.clear()
        self.stop_orders.clear()
        self.order_counter = 0
        self.execution_log.clear()
        self.logger.info(f"FakeBroker: Reset to initial state (cash=${self.initial_cash:,.2f})")


class TradeLockerBroker(BrokerInterface):
    """Broker implementation for TradeLocker API (prop firm trading via DNA Funded, etc.)

    Uses direct REST API calls instead of the tradelocker package (which requires Python 3.11+).
    """

    def __init__(self, username: str, password: str, server: str,
                 environment: str = "https://live.tradelocker.com"):
        """
        Initialize TradeLocker broker.

        Args:
            username: TradeLocker account email
            password: TradeLocker account password
            server: Server name (e.g., 'PTTSER')
            environment: API base URL
        """
        import requests as req
        self._requests = req

        self.logger = logging.getLogger(__name__)
        self.username = username
        self.password = password
        self.server = server
        self.base_url = environment.rstrip('/') + '/backend-api'

        # Auth tokens
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        # Account info (populated on first auth)
        self._account_id: Optional[str] = None
        self._acc_num: Optional[str] = None

        # Instrument cache: symbol -> {instrument_id, route_id}
        self._instrument_cache: Dict[str, Dict[str, Any]] = {}
        self._instrument_id_to_symbol: Dict[int, str] = {}
        self._route_id_to_symbol: Dict[int, str] = {}  # Route ID -> symbol mapping

        # Order counter for tracking
        self._order_counter = 0

        # Rate limiting - TradeLocker has strict limits, be conservative
        self._request_times: List[float] = []
        self._rate_limit_window = 60.0  # seconds
        self._rate_limit_max = 30  # Very conservative: 30 requests/minute

        # Positions cache - avoid redundant API calls within same cycle
        self._positions_cache: Optional[List[Position]] = None
        self._positions_cache_time: float = 0
        self._positions_cache_ttl: float = 5.0  # Cache for 5 seconds

        # Account cache - similar caching for account data
        self._account_cache: Optional[Account] = None
        self._account_cache_time: float = 0
        self._account_cache_ttl: float = 5.0  # Cache for 5 seconds

        # Authenticate on init
        self._authenticate()
        self.logger.info(f"TradeLockerBroker connected to {server} (account: {self._acc_num})")

    def _check_rate_limit(self):
        """Check if we're approaching rate limit and sleep if necessary."""
        now = time.time()

        # Remove requests older than the window
        self._request_times = [t for t in self._request_times if now - t < self._rate_limit_window]

        # Check if we're at the limit
        if len(self._request_times) >= self._rate_limit_max:
            oldest = self._request_times[0]
            sleep_time = self._rate_limit_window - (now - oldest) + 0.5  # Add 500ms buffer
            if sleep_time > 0:
                self.logger.warning(f"TradeLocker rate limit approaching ({len(self._request_times)} reqs), sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                now = time.time()
                self._request_times = [t for t in self._request_times if now - t < self._rate_limit_window]

        # Record this request
        self._request_times.append(now)

    def _authenticate(self):
        """Authenticate with TradeLocker and get JWT tokens."""
        try:
            response = self._requests.post(
                f"{self.base_url}/auth/jwt/token",
                json={
                    "email": self.username,
                    "password": self.password,
                    "server": self.server
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()

            self._access_token = data.get('accessToken')
            self._refresh_token = data.get('refreshToken')

            # Token typically expires in 1 hour, refresh before that
            self._token_expiry = datetime.now(pytz.UTC) + timedelta(minutes=50)

            # Get account info
            self._fetch_account_info()

        except Exception as e:
            raise BrokerAPIError(f"TradeLocker authentication failed: {e}")

    def _fetch_account_info(self):
        """Fetch account ID and account number after auth."""
        try:
            response = self._requests.get(
                f"{self.base_url}/auth/jwt/all-accounts",
                headers=self._get_headers()
            )
            response.raise_for_status()
            data = response.json()

            accounts = data.get('accounts', [])
            if not accounts:
                raise BrokerAPIError("No trading accounts found on TradeLocker")

            # Use the first account (or could be configured)
            account = accounts[0]
            self._account_id = str(account.get('id'))
            self._acc_num = str(account.get('accNum'))

            self.logger.info(f"TradeLocker account: {self._acc_num} (ID: {self._account_id})")

        except Exception as e:
            raise BrokerAPIError(f"Failed to fetch TradeLocker account info: {e}")

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with auth token."""
        self._ensure_token_valid()
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json"
        }
        if self._acc_num:
            headers["accNum"] = self._acc_num
        return headers

    def _ensure_token_valid(self):
        """Refresh token if expired or close to expiry."""
        if self._token_expiry and datetime.now(pytz.UTC) >= self._token_expiry:
            self._refresh_auth_token()

    def _refresh_auth_token(self):
        """Refresh the JWT token."""
        try:
            response = self._requests.post(
                f"{self.base_url}/auth/jwt/refresh",
                json={"refreshToken": self._refresh_token},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()

            self._access_token = data.get('accessToken')
            self._refresh_token = data.get('refreshToken')
            self._token_expiry = datetime.now(pytz.UTC) + timedelta(minutes=50)

            self.logger.debug("TradeLocker token refreshed")

        except Exception as e:
            # If refresh fails, re-authenticate
            self.logger.warning(f"Token refresh failed, re-authenticating: {e}")
            self._authenticate()

    def _get_instrument_id(self, symbol: str) -> Dict[str, Any]:
        """Get instrument ID and route ID for a symbol.

        Returns dict with 'instrument_id' and 'route_id'.
        """
        symbol = symbol.upper().strip()

        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        # Fetch all instruments if cache is empty
        if not self._instrument_cache:
            self._load_instruments()

        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        raise BrokerAPIError(f"Symbol {symbol} not found on TradeLocker")

    def _load_instruments(self):
        """Load all tradable instruments into cache."""
        self._check_rate_limit()
        try:
            response = self._requests.get(
                f"{self.base_url}/trade/accounts/{self._account_id}/instruments",
                headers=self._get_headers()
            )
            response.raise_for_status()
            data = response.json()

            instruments = data.get('instruments', data.get('d', {}).get('instruments', []))
            for inst in instruments:
                name = inst.get('name', '').upper()
                inst_id = inst.get('tradableInstrumentId')
                routes = inst.get('routes', [])
                route_id = routes[0].get('id') if routes else None

                if name and inst_id:
                    self._instrument_cache[name] = {
                        'instrument_id': inst_id,
                        'route_id': route_id
                    }
                    self._instrument_id_to_symbol[inst_id] = name

                    # Also map all route IDs to this symbol
                    for route in routes:
                        rid = route.get('id')
                        if rid:
                            self._route_id_to_symbol[rid] = name

            self.logger.info(f"Loaded {len(self._instrument_cache)} instruments from TradeLocker")

        except Exception as e:
            self.logger.error(f"Failed to load instruments: {e}")
            raise BrokerAPIError(f"Failed to load instruments: {e}")

    def _get_symbol_from_id(self, id_value: int) -> str:
        """Reverse lookup: instrument ID or route ID to symbol."""
        # Check instrument ID first
        if id_value in self._instrument_id_to_symbol:
            return self._instrument_id_to_symbol[id_value]

        # Check route ID
        if id_value in self._route_id_to_symbol:
            return self._route_id_to_symbol[id_value]

        # If not in cache, try to load instruments
        if not self._instrument_id_to_symbol:
            self._load_instruments()

        # Try again after loading
        if id_value in self._instrument_id_to_symbol:
            return self._instrument_id_to_symbol[id_value]
        if id_value in self._route_id_to_symbol:
            return self._route_id_to_symbol[id_value]

        return f"UNKNOWN_{id_value}"

    @retry_on_failure(max_retries=3, delay=2.0, backoff=2.0)
    def get_account(self) -> Account:
        """Get account information from TradeLocker.

        TradeLocker returns account details as an array. The field order is:
        0=balance, 1=projectedBalance, 2=availableFunds, 3=blockedBalance,
        4=cashBalance, 5=unsettledCash, 6=withdrawalAvailable, 7=stocksValue,
        8=optionValue, 9=initialMarginReq, 10=maintMarginReq, etc.

        Uses 5-second cache to avoid hitting rate limits.
        """
        # Check cache first
        now = time.time()
        if self._account_cache is not None and (now - self._account_cache_time) < self._account_cache_ttl:
            return self._account_cache

        self._check_rate_limit()
        try:
            response = self._requests.get(
                f"{self.base_url}/trade/accounts/{self._account_id}/state",
                headers=self._get_headers()
            )
            response.raise_for_status()
            data = response.json()

            # TradeLocker returns data in d.accountDetailsData as an array
            details = data.get('d', {}).get('accountDetailsData', [])

            if not details or len(details) < 5:
                raise BrokerAPIError("Invalid account state response")

            # Parse array values by index
            balance = float(details[0])           # balance
            projected_balance = float(details[1]) # projectedBalance (equity)
            available_funds = float(details[2])   # availableFunds (buying power)
            cash_balance = float(details[4])      # cashBalance

            account = Account(
                equity=projected_balance,
                cash=cash_balance,
                buying_power=available_funds,
                portfolio_value=projected_balance,
                last_equity=balance  # Use balance as previous equity approximation
            )

            # Cache the result
            self._account_cache = account
            self._account_cache_time = time.time()

            return account

        except Exception as e:
            self.logger.error(f"Failed to get account: {e}")
            raise BrokerAPIError(f"Failed to get account: {e}")

    @retry_on_failure(max_retries=3, delay=2.0, backoff=2.0)
    def get_positions(self) -> List[Position]:
        """Get all open positions from TradeLocker.

        TradeLocker returns positions as arrays with format:
        [position_id, ?, instrument_id, side, qty, avg_price, ?, ?, timestamp, unrealized_pl, ?]

        Uses 5-second cache to avoid hitting rate limits (1 req/sec for positions).
        """
        # Check cache first
        now = time.time()
        if self._positions_cache is not None and (now - self._positions_cache_time) < self._positions_cache_ttl:
            return self._positions_cache

        self._check_rate_limit()
        try:
            response = self._requests.get(
                f"{self.base_url}/trade/accounts/{self._account_id}/positions",
                headers=self._get_headers()
            )
            response.raise_for_status()
            data = response.json()

            # Handle various response formats from TradeLocker API
            # API can return: list directly, {'positions': [...]}, {'d': [...]}, or {'d': {'positions': [...]}}
            if isinstance(data, list):
                # API returned positions array directly
                positions_data = data
            elif isinstance(data, dict):
                # Try 'positions' key first
                if 'positions' in data:
                    positions_data = data['positions']
                # Then try 'd' key - can be list or dict
                elif 'd' in data:
                    d_value = data['d']
                    if isinstance(d_value, list):
                        positions_data = d_value
                    elif isinstance(d_value, dict):
                        positions_data = d_value.get('positions', [])
                    else:
                        positions_data = []
                else:
                    positions_data = []
            else:
                self.logger.warning(f"Unexpected positions response type: {type(data)}")
                positions_data = []
            positions = []

            for pos in positions_data:
                # Handle array format: [id, ?, route_id, side, qty, avg_price, ?, ?, ts, unrealized_pl, ?]
                if isinstance(pos, list):
                    if len(pos) < 10:
                        continue
                    position_id = pos[0]
                    instrument_id = int(pos[1]) if pos[1] else None
                    route_id = int(pos[2]) if pos[2] else None
                    symbol = self._get_symbol_from_id(instrument_id) if instrument_id else "UNKNOWN"
                    # Log position details for debugging symbol mapping
                    self.logger.info(f"Position parse: id={position_id}, route_id={route_id} -> symbol={symbol}")
                    qty = abs(float(pos[4])) if pos[4] else 0
                    side_str = str(pos[3]).lower() if pos[3] else ''
                    side = 'long' if side_str == 'buy' else 'short'
                    entry_price = float(pos[5]) if pos[5] else 0
                    unrealized_pl = float(pos[9]) if pos[9] else 0
                    # For CFDs/forex, entry_price is not in dollars, so don't derive current_price
                    # Just use entry_price as placeholder, the unrealized_pl is what matters
                    current_price = entry_price
                else:
                    # Handle dict format (legacy fallback)
                    inst_id = pos.get('tradableInstrumentId')
                    symbol = self._get_symbol_from_id(inst_id)
                    qty = abs(float(pos.get('qty', pos.get('quantity', 0))))
                    side_str = pos.get('side', '').lower()
                    side = 'long' if side_str == 'buy' else 'short'
                    entry_price = float(pos.get('avgPrice', pos.get('openPrice', 0)))
                    current_price = float(pos.get('currentPrice', entry_price))
                    unrealized_pl = float(pos.get('unrealizedPl', pos.get('profit', 0)))

                # For TradeLocker CFDs/forex, entry_price isn't in dollars so percentage is meaningless
                # market_value and unrealized_plpc only make sense for stocks
                if isinstance(pos, list):
                    # CFD/forex from TradeLocker - don't calculate percentage
                    market_value = 0
                    unrealized_plpc = 0
                else:
                    # Stock position - calculate normally
                    market_value = qty * current_price
                    unrealized_plpc = unrealized_pl / (entry_price * qty) if entry_price * qty > 0 else 0

                positions.append(Position(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    avg_entry_price=entry_price,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pl=unrealized_pl,
                    unrealized_plpc=unrealized_plpc
                ))

            # Safeguard: ensure unique symbols (append suffix if duplicates found)
            seen_symbols = {}
            for i, pos in enumerate(positions):
                if pos.symbol in seen_symbols:
                    # Duplicate found - make unique by appending index
                    original = pos.symbol
                    unique_symbol = f"{pos.symbol}_{i}"
                    positions[i] = Position(
                        symbol=unique_symbol,
                        qty=pos.qty,
                        side=pos.side,
                        avg_entry_price=pos.avg_entry_price,
                        current_price=pos.current_price,
                        market_value=pos.market_value,
                        unrealized_pl=pos.unrealized_pl,
                        unrealized_plpc=pos.unrealized_plpc
                    )
                    self.logger.warning(f"Duplicate symbol {original} renamed to {unique_symbol}")
                else:
                    seen_symbols[pos.symbol] = i

            # Cache the result
            self._positions_cache = positions
            self._positions_cache_time = time.time()

            return positions

        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            raise BrokerAPIError(f"Failed to get positions: {e}")

    def list_positions(self) -> List[Position]:
        """Alias for get_positions."""
        return self.get_positions()

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        symbol = symbol.upper().strip()
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        return self.list_orders(status='open')

    @retry_on_failure(max_retries=3, delay=2.0, backoff=2.0)
    def list_orders(self, status: str = 'open') -> List[Order]:
        """List orders with status filter."""
        self._check_rate_limit()
        try:
            response = self._requests.get(
                f"{self.base_url}/trade/accounts/{self._account_id}/orders",
                headers=self._get_headers()
            )
            response.raise_for_status()
            data = response.json()

            # Handle various response formats from TradeLocker API
            if isinstance(data, list):
                # API returned orders array directly
                orders_data = data
            elif isinstance(data, dict):
                orders_data = data.get('orders', data.get('d', {}).get('orders', []))
            else:
                self.logger.warning(f"Unexpected orders response type: {type(data)}")
                orders_data = []
            orders = []

            for ord in orders_data:
                inst_id = ord.get('tradableInstrumentId')
                symbol = self._get_symbol_from_id(inst_id)

                order_status = ord.get('status', '').lower()

                # Filter by status if needed
                if status == 'open' and order_status not in ['new', 'pending', 'working']:
                    continue

                orders.append(Order(
                    id=str(ord.get('id')),
                    symbol=symbol,
                    qty=float(ord.get('qty', ord.get('quantity', 0))),
                    side=ord.get('side', '').lower(),
                    type=ord.get('type', 'market').lower(),
                    status=order_status,
                    limit_price=float(ord.get('limitPrice')) if ord.get('limitPrice') else None,
                    stop_price=float(ord.get('stopPrice')) if ord.get('stopPrice') else None,
                    filled_qty=float(ord.get('filledQty', 0)),
                    filled_avg_price=float(ord.get('avgFilledPrice')) if ord.get('avgFilledPrice') else None
                ))

            return orders

        except Exception as e:
            self.logger.error(f"Failed to list orders: {e}")
            raise BrokerAPIError(f"Failed to list orders: {e}")

    @retry_on_failure(max_retries=3, delay=2.0, backoff=2.0)
    def _invalidate_cache(self):
        """Invalidate position and account caches after order activity."""
        self._positions_cache = None
        self._positions_cache_time = 0
        self._account_cache = None
        self._account_cache_time = 0

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        type: str = 'market',
        time_in_force: str = 'day',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """Submit order to TradeLocker with fill verification for market orders."""
        # Invalidate cache since positions will change
        self._invalidate_cache()

        self._check_rate_limit()
        symbol = symbol.upper().strip()
        side = side.lower()

        # Get instrument info
        inst_info = self._get_instrument_id(symbol)
        instrument_id = inst_info['instrument_id']
        route_id = inst_info['route_id']

        # Map order type - IOC required for market orders (GTC/DAY forbidden on TradeLocker)
        tl_type = type.lower()
        if tl_type == 'market':
            validity = 'IOC'  # Immediate-Or-Cancel (only allowed for market orders)
            price = 0
        else:
            validity = 'GTC'  # Good Till Cancel for limit/stop
            price = limit_price or 0

        # Build order payload
        order_payload = {
            "tradableInstrumentId": instrument_id,
            "qty": int(qty),
            "side": side,
            "type": tl_type,
            "validity": validity,
            "routeId": route_id,
            "price": price
        }

        if stop_price:
            order_payload["stopPrice"] = stop_price

        try:
            self.logger.info(f"Submitting order: {side.upper()} {qty} {symbol} @ {type}")

            response = self._requests.post(
                f"{self.base_url}/trade/accounts/{self._account_id}/orders",
                headers=self._get_headers(),
                json=order_payload
            )
            response.raise_for_status()
            data = response.json()

            order_id = data.get('orderId', data.get('d', {}).get('orderId'))

            self.logger.info(f"Order submitted successfully: {order_id}")

            # For market orders, verify fill by checking position
            # Match by instrument/route ID, not symbol (symbol mapping can be unreliable)
            if tl_type == 'market':
                time.sleep(0.5)  # Brief wait for order processing

                filled_qty = 0
                filled_price = None

                try:
                    response = self._requests.get(
                        f"{self.base_url}/trade/accounts/{self._account_id}/positions",
                        headers=self._get_headers()
                    )
                    response.raise_for_status()
                    pos_data = response.json()

                    if isinstance(pos_data, list):
                        positions_raw = pos_data
                    elif isinstance(pos_data, dict):
                        # Handle various formats: {'positions': [...]}, {'d': [...]}, {'d': {'positions': [...]}}
                        if 'positions' in pos_data:
                            positions_raw = pos_data['positions']
                        elif 'd' in pos_data:
                            d_val = pos_data['d']
                            if isinstance(d_val, list):
                                positions_raw = d_val
                            elif isinstance(d_val, dict):
                                positions_raw = d_val.get('positions', [])
                            else:
                                positions_raw = []
                        else:
                            positions_raw = []
                    else:
                        positions_raw = []

                    for pos in positions_raw:
                        if isinstance(pos, list) and len(pos) >= 6:
                            pos_inst_id = int(pos[1]) if pos[1] else None
                            if pos_inst_id == instrument_id:
                                filled_qty = int(abs(float(pos[4]))) if pos[4] else 0
                                filled_price = float(pos[5]) if pos[5] else None
                                self.logger.info(
                                    f"Market order filled: {filled_qty} {symbol} @ ${filled_price:.2f}"
                                )
                                break
                        elif isinstance(pos, dict):
                            pos_inst_id = pos.get('tradableInstrumentId')
                            if pos_inst_id == instrument_id:
                                filled_qty = int(abs(float(pos.get('qty', pos.get('quantity', 0)))))
                                filled_price = float(pos.get('avgPrice', pos.get('openPrice', 0)))
                                self.logger.info(
                                    f"Market order filled: {filled_qty} {symbol} @ ${filled_price:.2f}"
                                )
                                break
                except Exception as e:
                    self.logger.warning(f"Could not verify position after order: {e}")

                if filled_qty == 0:
                    self.logger.error(
                        f"Market order {order_id} not filled - no position for {symbol}"
                    )
                    raise BrokerAPIError(
                        f"Market order not filled for {symbol}: order cancelled or no liquidity"
                    )

                return Order(
                    id=str(order_id),
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type=type,
                    status='filled',
                    filled_qty=filled_qty,
                    filled_avg_price=filled_price,
                    limit_price=limit_price,
                    stop_price=stop_price,
                    submitted_at=datetime.now(pytz.UTC)
                )

            # For limit/stop orders, return with 'new' status (fills later)
            return Order(
                id=str(order_id),
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                status='new',
                limit_price=limit_price,
                stop_price=stop_price,
                submitted_at=datetime.now(pytz.UTC)
            )

        except BrokerAPIError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to submit order: {e}")
            raise BrokerAPIError(f"Order submission failed for {symbol}: {e}")

    @retry_on_failure(max_retries=2, delay=2.0, backoff=2.0)
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order by ID."""
        self._check_rate_limit()
        try:
            response = self._requests.delete(
                f"{self.base_url}/trade/orders/{order_id}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            self.logger.info(f"Cancelled order {order_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        orders = self.get_open_orders()
        count = 0
        for order in orders:
            if self.cancel_order(order.id):
                count += 1
        self.logger.info(f"Cancelled {count} orders")
        return count

    def close_position(self, symbol: str) -> bool:
        """Close position for symbol."""
        self._check_rate_limit()
        try:
            # Get position to find position ID
            positions = self.get_positions()
            position = None
            for pos in positions:
                if pos.symbol.upper() == symbol.upper():
                    position = pos
                    break

            if not position:
                self.logger.warning(f"No position found for {symbol}")
                return False

            # Get position ID from raw API response
            response = self._requests.get(
                f"{self.base_url}/trade/accounts/{self._account_id}/positions",
                headers=self._get_headers()
            )
            response.raise_for_status()
            data = response.json()

            # Handle various response formats from TradeLocker API
            if isinstance(data, list):
                positions_data = data
            elif isinstance(data, dict):
                # Handle various formats: {'positions': [...]}, {'d': [...]}, {'d': {'positions': [...]}}
                if 'positions' in data:
                    positions_data = data['positions']
                elif 'd' in data:
                    d_val = data['d']
                    if isinstance(d_val, list):
                        positions_data = d_val
                    elif isinstance(d_val, dict):
                        positions_data = d_val.get('positions', [])
                    else:
                        positions_data = []
                else:
                    positions_data = []
            else:
                positions_data = []
            position_id = None

            inst_info = self._get_instrument_id(symbol)
            target_inst_id = inst_info['instrument_id']

            for pos in positions_data:
                # Handle array format: [id, ?, inst_id, ...]
                if isinstance(pos, list):
                    if len(pos) >= 3 and int(pos[2]) == target_inst_id:
                        position_id = str(pos[0])
                        break
                else:
                    # Handle dict format (legacy)
                    if pos.get('tradableInstrumentId') == target_inst_id:
                        position_id = pos.get('id')
                        break

            if not position_id:
                self.logger.error(f"Could not find position ID for {symbol}")
                return False

            # Close the position
            close_response = self._requests.delete(
                f"{self.base_url}/trade/positions/{position_id}",
                headers=self._get_headers()
            )
            close_response.raise_for_status()

            self.logger.info(f"Closed position: {symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to close position {symbol}: {e}")
            return False

    def close_all_positions(self) -> int:
        """Close all positions."""
        positions = self.get_positions()
        count = 0
        for pos in positions:
            if self.close_position(pos.symbol):
                count += 1
        self.logger.info(f"Closed {count} positions")
        return count

    def get_broker_name(self) -> str:
        return "TradeLockerBroker"

    def get_portfolio_history(self, period: str = "30D") -> PortfolioHistory:
        """Get portfolio history.

        Note: TradeLocker may not provide historical equity data.
        Returns empty history if not available.
        """
        # TradeLocker doesn't have a direct portfolio history endpoint
        # Return empty history
        return PortfolioHistory(
            timestamps=[],
            equity=[],
            timeframe="1D",
            base_value=0.0
        )

    def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        stop_loss_percent: float = 0.05,
        take_profit_percent: float = 0.05,
        time_in_force: str = 'gtc',
        **kwargs
    ) -> Order:
        """Submit bracket order with stop-loss and take-profit, then verify fill.

        Uses GTC (Good-Till-Cancelled) for market orders with SL/TP attached.
        Verifies the fill by checking if a position was created.

        Args:
            stop_loss_percent: Stop loss percentage (default 5% = 0.05)
            take_profit_percent: Take profit percentage (default 5% = 0.05)
        """
        price = kwargs.get('price')
        if price is None:
            raise BrokerAPIError("price is required for bracket orders")

        symbol = symbol.upper().strip()
        side = side.lower()

        # Calculate stop and take profit prices
        if side == 'buy':
            stop_price = round(price * (1 - stop_loss_percent), 2)
            tp_price = round(price * (1 + take_profit_percent), 2)
        else:
            stop_price = round(price * (1 + stop_loss_percent), 2)
            tp_price = round(price * (1 - take_profit_percent), 2)

        # Get instrument info
        inst_info = self._get_instrument_id(symbol)
        instrument_id = inst_info['instrument_id']
        route_id = inst_info['route_id']

        # Build order with stopLoss and takeProfit (absolute prices)
        # IOC is required for market orders on TradeLocker (GTC/DAY forbidden)
        order_payload = {
            "tradableInstrumentId": instrument_id,
            "qty": int(qty),
            "side": side,
            "type": "market",
            "validity": "IOC",
            "routeId": route_id,
            "price": 0,
            "stopLoss": stop_price,
            "stopLossType": "absolute",
            "takeProfit": tp_price,
            "takeProfitType": "absolute"
        }

        try:
            self.logger.info(
                f"Submitting bracket order: {side.upper()} {qty} {symbol} "
                f"(SL={stop_price}, TP={tp_price})"
            )

            response = self._requests.post(
                f"{self.base_url}/trade/accounts/{self._account_id}/orders",
                headers=self._get_headers(),
                json=order_payload
            )
            self.logger.info(f"Order API response: status={response.status_code}, body={response.text[:500]}")
            response.raise_for_status()
            data = response.json()

            # Check for TradeLocker error response (returns 200 with s:"error")
            if data.get('s') == 'error':
                error_msg = data.get('errmsg', 'Unknown error')
                raise BrokerAPIError(f"Order rejected by TradeLocker: {error_msg}")

            order_id = data.get('orderId', data.get('d', {}).get('orderId'))

            self.logger.info(f"Bracket order submitted: {order_id}")

            # Verify fill by checking for position creation
            # Market orders should fill quickly, wait then verify
            time.sleep(2.0)  # Wait for order processing and rate limit

            filled_qty = 0
            filled_price = None

            # Check if position was created - match by instrument/route ID, not symbol
            # Retry up to 3 times with delay to handle rate limiting
            try:
                pos_data = None
                for attempt in range(3):
                    try:
                        response = self._requests.get(
                            f"{self.base_url}/trade/accounts/{self._account_id}/positions",
                            headers=self._get_headers()
                        )
                        response.raise_for_status()
                        pos_data = response.json()
                        self.logger.info(f"Position check attempt {attempt+1}: {len(pos_data.get('d', {}).get('positions', []))} positions")
                        if pos_data.get('d', {}).get('positions'):
                            break  # Got valid positions data
                    except Exception as e:
                        self.logger.warning(f"Position check attempt {attempt+1} failed: {e}")
                    if attempt < 2:
                        time.sleep(1.5)  # Rate limit delay before retry

                if pos_data is None:
                    pos_data = {}
                    self.logger.warning("All position check attempts failed, using empty data")

                if isinstance(pos_data, list):
                    positions_raw = pos_data
                elif isinstance(pos_data, dict):
                    # Handle various formats: {'positions': [...]}, {'d': [...]}, {'d': {'positions': [...]}}
                    if 'positions' in pos_data:
                        positions_raw = pos_data['positions']
                    elif 'd' in pos_data:
                        d_val = pos_data['d']
                        if isinstance(d_val, list):
                            positions_raw = d_val
                        elif isinstance(d_val, dict):
                            positions_raw = d_val.get('positions', [])
                        else:
                            positions_raw = []
                    else:
                        positions_raw = []
                else:
                    positions_raw = []

                for pos in positions_raw:
                    if isinstance(pos, list) and len(pos) >= 6:
                        # Match by instrument_id (pos[1]) against our order's instrument_id
                        pos_inst_id = int(pos[1]) if pos[1] else None
                        if pos_inst_id == instrument_id:
                            filled_qty = int(abs(float(pos[4]))) if pos[4] else 0
                            filled_price = float(pos[5]) if pos[5] else None
                            self.logger.info(
                                f"Order filled: {filled_qty} {symbol} @ ${filled_price:.2f} (instrument_id={instrument_id})"
                            )
                            break
                    elif isinstance(pos, dict):
                        pos_inst_id = pos.get('tradableInstrumentId')
                        if pos_inst_id == instrument_id:
                            filled_qty = int(abs(float(pos.get('qty', pos.get('quantity', 0)))))
                            filled_price = float(pos.get('avgPrice', pos.get('openPrice', 0)))
                            self.logger.info(
                                f"Order filled: {filled_qty} {symbol} @ ${filled_price:.2f}"
                            )
                            break
            except Exception as e:
                self.logger.warning(f"Could not verify position after order: {e}")

            # If no position found, order was not filled
            if filled_qty == 0:
                self.logger.error(
                    f"Order {order_id} not filled - no position created for {symbol}. "
                    f"IOC order may have been cancelled (no liquidity or symbol unavailable)."
                )
                raise BrokerAPIError(
                    f"Order not filled for {symbol}: IOC market order cancelled (no liquidity)"
                )

            # Log partial fills
            if filled_qty != qty:
                self.logger.warning(
                    f"Partial fill for {symbol}: requested {qty}, filled {filled_qty} "
                    f"(IOC filled available liquidity)"
                )

            order = Order(
                id=str(order_id),
                symbol=symbol,
                qty=qty,  # Requested qty
                side=side,
                type='market',
                status='filled',
                stop_price=stop_price,
                filled_qty=filled_qty,  # Actual filled qty (may be partial)
                filled_avg_price=filled_price,
                submitted_at=datetime.now(pytz.UTC)
            )

            # Attach stop order ID (may be same as main order in TradeLocker)
            order.stop_order_id = str(order_id)

            return order

        except BrokerAPIError:
            # Re-raise our own errors
            raise
        except Exception as e:
            self.logger.error(f"Failed to submit bracket order: {e}")
            raise BrokerAPIError(f"Bracket order failed for {symbol}: {e}")

    def set_position_stop_loss(self, position_id: str, stop_loss_price: float, take_profit_price: float = None) -> bool:
        """Set stop loss (and optionally take profit) on an existing position.

        Args:
            position_id: The TradeLocker position ID
            stop_loss_price: Stop loss price (absolute)
            take_profit_price: Optional take profit price (absolute)

        Returns:
            True if successful, False otherwise
        """
        try:
            payload = {"stopLoss": stop_loss_price}
            if take_profit_price is not None:
                payload["takeProfit"] = take_profit_price

            self.logger.info(f"Setting SL on position {position_id}: SL=${stop_loss_price:.2f}")

            response = self._requests.patch(
                f"{self.base_url}/trade/positions/{position_id}",
                headers=self._get_headers(),
                json=payload
            )

            self.logger.info(f"Modify position response: status={response.status_code}, body={response.text[:200]}")
            response.raise_for_status()

            return True
        except Exception as e:
            self.logger.error(f"Failed to set stop loss on position {position_id}: {e}")
            return False

    def get_positions_with_ids(self) -> list:
        """Get positions with their TradeLocker position IDs for modification."""
        try:
            response = self._requests.get(
                f"{self.base_url}/trade/accounts/{self._account_id}/positions",
                headers=self._get_headers()
            )
            response.raise_for_status()
            data = response.json()

            positions_raw = data.get('d', {}).get('positions', [])
            result = []

            for pos in positions_raw:
                if isinstance(pos, list) and len(pos) >= 6:
                    pos_id = str(pos[0])
                    inst_id = int(pos[1]) if pos[1] else None
                    qty = float(pos[4]) if pos[4] else 0
                    entry_price = float(pos[5]) if pos[5] else 0

                    # Get symbol from instrument_id
                    symbol = None
                    for sym, info in self._instrument_cache.items():
                        if info.get('tradable_instrument_id') == inst_id:
                            symbol = sym
                            break

                    if symbol and qty != 0:
                        result.append({
                            'position_id': pos_id,
                            'symbol': symbol,
                            'qty': qty,
                            'entry_price': entry_price,
                            'instrument_id': inst_id
                        })

            return result
        except Exception as e:
            self.logger.error(f"Failed to get positions with IDs: {e}")
            return []


class BrokerFactory:
    """Factory for creating broker instances based on trading mode"""

    @staticmethod
    def create_broker() -> BrokerInterface:
        """
        Create appropriate broker based on current mode

        Returns:
            BrokerInterface: AlpacaBroker for PAPER/LIVE, TradeLockerBroker for TRADELOCKER,
                           FakeBroker for BACKTEST/DRY_RUN
        """
        config = get_global_config()
        mode = config.get_mode()
        logger = logging.getLogger(__name__)

        if config.requires_tradelocker_broker():
            # TRADELOCKER mode - use TradeLockerBroker for prop firm trading
            username = os.getenv('TRADELOCKER_USERNAME')
            password = os.getenv('TRADELOCKER_PASSWORD')
            server = os.getenv('TRADELOCKER_SERVER')
            environment = os.getenv('TRADELOCKER_ENVIRONMENT', 'https://live.tradelocker.com')

            # Strip whitespace
            if username:
                username = username.strip()
            if password:
                password = password.strip()
            if server:
                server = server.strip()
            if environment:
                environment = environment.strip()

            if not all([username, password, server]):
                raise ValueError(
                    f"Mode {mode} requires TradeLocker credentials. "
                    "Please set TRADELOCKER_USERNAME, TRADELOCKER_PASSWORD, "
                    "TRADELOCKER_SERVER environment variables."
                )

            logger.info(f"Creating TradeLockerBroker (mode={mode}, server={server})")

            return TradeLockerBroker(username, password, server, environment)

        elif config.requires_real_broker():
            # PAPER or LIVE mode - use AlpacaBroker
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')

            # Strip whitespace/newlines from API keys
            if api_key:
                api_key = api_key.strip()
            if secret_key:
                secret_key = secret_key.strip()

            if not api_key or not secret_key:
                raise ValueError(
                    f"Mode {mode} requires Alpaca API credentials. "
                    "Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
                )

            endpoint = config.get_alpaca_endpoint()
            logger.info(f"Creating AlpacaBroker (mode={mode}, endpoint={endpoint})")

            return AlpacaBroker(api_key, secret_key, endpoint)

        else:
            # BACKTEST or DRY_RUN mode - use FakeBroker
            fake_config = config.get_fake_broker_config()
            initial_cash = fake_config.get('initial_cash', 100000)
            commission = fake_config.get('commission_per_trade', 0)
            slippage = fake_config.get('slippage_percent', 0.0005)

            logger.info(f"Creating FakeBroker (mode={mode}, cash=${initial_cash:,.2f})")

            return FakeBroker(
                initial_cash=initial_cash,
                commission=commission,
                slippage=slippage
            )


# Convenience function alias
def create_broker() -> BrokerInterface:
    """Convenience function to create a broker (alias for BrokerFactory.create_broker)"""
    return BrokerFactory.create_broker()
