"""
Broker Abstraction Layer for Trading Bot
Provides unified interface for real (Alpaca) and simulated (Fake) brokers

Combined from:
- broker_interface.py (BrokerInterface ABC, AlpacaBroker, FakeBroker)
- broker_factory.py (BrokerFactory)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
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


class AlpacaBroker(BrokerInterface):
    """Real broker implementation using Alpaca API"""

    def __init__(self, api_key: str, secret_key: str, base_url: str = None):
        """
        Initialize Alpaca broker

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            base_url: Base URL (paper or live endpoint)
        """
        import alpaca_trade_api as tradeapi

        self.logger = logging.getLogger(__name__)

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
        """Submit order to Alpaca."""
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
        except Exception as e:
            self.logger.error(
                f"Failed to submit order: {side.upper()} {qty} {symbol} @ {type} - {e}",
                exc_info=True
            )
            raise BrokerAPIError(
                f"Order submission failed for {symbol}: {e}",
                original_exception=e
            )

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

        # Execution log
        self.execution_log: List[Dict[str, Any]] = []

        self.logger.info(f"FakeBroker initialized with ${initial_cash:,.2f}")

    def update_price(self, symbol: str, price: float):
        """
        Update current price for a symbol (used for P&L calculations).

        Args:
            symbol: Stock ticker
            price: Current market price
        """
        if price > 0:
            if symbol in self.positions:
                self.current_prices[symbol] = price
            else:
                self.logger.debug(
                    f"Price update for {symbol} (${price:.2f}) ignored - no position"
                )

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
        """Cancel simulated order"""
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
        """Close simulated position (LONG or SHORT)"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            side = pos.get('side', 'long')
            close_price = self.current_prices.get(symbol, pos['avg_entry_price'])

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

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get all executions for analysis"""
        return self.execution_log

    def reset(self):
        """
        Reset FakeBroker to initial state.

        Clears all positions, orders, and resets cash to initial value.
        """
        self.cash = self.initial_cash
        self.positions.clear()
        self.current_prices.clear()
        self.orders.clear()
        self.order_counter = 0
        self.execution_log.clear()
        self.logger.info(f"FakeBroker: Reset to initial state (cash=${self.initial_cash:,.2f})")


class BrokerFactory:
    """Factory for creating broker instances based on trading mode"""

    @staticmethod
    def create_broker() -> BrokerInterface:
        """
        Create appropriate broker based on current mode

        Returns:
            BrokerInterface: AlpacaBroker for PAPER/LIVE, FakeBroker for BACKTEST/DRY_RUN
        """
        config = get_global_config()
        mode = config.get_mode()
        logger = logging.getLogger(__name__)

        if config.requires_real_broker():
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
