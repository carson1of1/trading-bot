# TradeLocker Broker Integration Design

**Date:** 2026-01-12
**Status:** Approved
**Purpose:** Enable trading bot to work with DNA Funded prop firm challenge via TradeLocker API

## Overview

Add `TradeLockerBroker` class implementing `BrokerInterface` to support prop firm trading. This allows switching between Alpaca (paper/live) and TradeLocker (prop challenge) via config.

## Architecture

```
┌─────────────────────┐
│      bot.py         │  No changes needed
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   create_broker()   │  Updated to handle TRADELOCKER mode
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌──────────────┐
│ Alpaca  │ │ TradeLocker  │  New implementation
│ Broker  │ │ Broker       │
└─────────┘ └──────────────┘
```

## Components to Modify

| File | Changes |
|------|---------|
| `core/broker.py` | Add `TradeLockerBroker` class (~250 lines) |
| `core/broker.py` | Update `BrokerFactory.create_broker()` |
| `core/config.py` | Add `TRADELOCKER` to valid modes |
| `.env` | Add TradeLocker credentials |
| `config.yaml` | Set `mode: TRADELOCKER` to activate |

**No changes to:** bot.py, strategies, risk management, scanner, exit manager

## TradeLockerBroker Class Design

### Initialization & Authentication

```python
class TradeLockerBroker(BrokerInterface):
    def __init__(self, username: str, password: str, server: str,
                 environment: str = "https://live.tradelocker.com"):
        self.logger = logging.getLogger(__name__)

        try:
            from tradelocker import TLAPI
            self.api = TLAPI(
                environment=environment,
                username=username,
                password=password,
                server=server
            )
            self._verify_connection()
            self.logger.info(f"TradeLockerBroker connected to {server}")
        except Exception as e:
            raise BrokerAPIError(f"TradeLocker connection failed: {e}")

        # Cache for symbol -> instrument_id mapping
        self._instrument_cache: Dict[str, int] = {}
        self._instrument_id_to_symbol: Dict[int, str] = {}
```

### Key Implementation Details

**1. Instrument ID Mapping**

TradeLocker uses numeric IDs, not ticker symbols:
- `AAPL` → `12345` (lookup required)
- Cache lookups to avoid repeated API calls
- Reverse mapping for position retrieval

**2. Order Submission**

```python
def submit_order(self, symbol, qty, side, type='market', ...):
    instrument_id = self._get_instrument_id(symbol)
    order_id = self.api.create_order(
        instrument_id=instrument_id,
        quantity=int(qty),
        side=side,
        type_=type
    )
    return Order(id=str(order_id), symbol=symbol, ...)
```

**3. Account Info**

Map TradeLocker account fields to our `Account` dataclass:
- `equity` → `equity`
- `balance` → `cash`
- `available_margin` → `buying_power`

### Methods to Implement

| Method | Notes |
|--------|-------|
| `get_account()` | Map TL account to Account dataclass |
| `get_positions()` | Reverse instrument ID lookup |
| `list_positions()` | Alias for get_positions |
| `get_position(symbol)` | Filter by symbol |
| `get_open_orders()` | Map TL orders to Order dataclass |
| `list_orders(status)` | Filter by status |
| `submit_order(...)` | Convert symbol to instrument ID |
| `cancel_order(id)` | Direct passthrough |
| `cancel_all_orders()` | Iterate and cancel |
| `close_position(symbol)` | Lookup position ID, close |
| `close_all_positions()` | Iterate and close |
| `get_broker_name()` | Return "TradeLockerBroker" |
| `get_portfolio_history()` | May return empty (prop firms often lack this) |
| `submit_bracket_order()` | May need separate stop order logic |

## Configuration

### .env additions

```
TRADELOCKER_USERNAME=carsonrodell@gmail.com
TRADELOCKER_PASSWORD=<password>
TRADELOCKER_SERVER=PTTSER
TRADELOCKER_ENVIRONMENT=https://live.tradelocker.com
```

### config.yaml

```yaml
mode: TRADELOCKER  # Switch to prop firm trading
```

## BrokerFactory Update

```python
@staticmethod
def create_broker() -> BrokerInterface:
    config = get_global_config()
    mode = config.get_mode()

    if mode == "TRADELOCKER":
        username = os.getenv('TRADELOCKER_USERNAME')
        password = os.getenv('TRADELOCKER_PASSWORD')
        server = os.getenv('TRADELOCKER_SERVER')
        environment = os.getenv('TRADELOCKER_ENVIRONMENT',
                                'https://live.tradelocker.com')

        if not all([username, password, server]):
            raise ValueError(
                "TRADELOCKER mode requires credentials in .env"
            )

        return TradeLockerBroker(username, password, server, environment)

    elif config.requires_real_broker():
        # Existing Alpaca logic
        ...
```

## Testing Strategy

1. **Unit tests:** Mock `tradelocker.TLAPI` to test TradeLockerBroker methods
2. **Integration test:** Connect to DNA Funded demo, verify account/positions
3. **Paper test:** Run bot in TRADELOCKER mode, verify orders execute

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| TradeLocker API differs from docs | Test each method against real API |
| Symbol not found on TradeLocker | Clear error message, skip symbol |
| Rate limiting | Add retry logic similar to AlpacaBroker |
| Bracket orders not supported | Fall back to separate entry + stop orders |

## Implementation Order

1. Add TradeLocker credentials to `.env`
2. Add `TRADELOCKER` mode to config
3. Implement `TradeLockerBroker` class
4. Update `BrokerFactory`
5. Test connection with `get_account()`
6. Test order flow with small position
7. Full integration test
