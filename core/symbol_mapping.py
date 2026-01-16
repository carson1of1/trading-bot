"""
Symbol mapping between different data sources and brokers.

yfinance uses: BTC-USD, ETH-USD, etc.
Alpaca uses: BTC/USD, ETH/USD, etc.
TradeLocker uses: BTCUSD, ETHUSD, etc.
"""

# Known crypto symbols for explicit mapping (optional overrides)
YFINANCE_TO_TRADELOCKER = {
    'BTC-USD': 'BTCUSD',
    'ETH-USD': 'ETHUSD',
    'SOL-USD': 'SOLUSD',
    'ADA-USD': 'ADAUSD',
    'NEAR-USD': 'NEARUSD',
    'ATOM-USD': 'ATOMUSD',
    'AVAX-USD': 'AVAXUSD',
    'LINK-USD': 'LINKUSD',
    'LTC-USD': 'LTCUSD',
    'DOT-USD': 'DOTUSD',
    'XRP-USD': 'XRPUSD',
    'MATIC-USD': 'MATICUSD',
    'UNI-USD': 'UNIUSD',
    'APT-USD': 'APTUSD',
    'DOGE-USD': 'DOGEUSD',
}

# Reverse mapping: TradeLocker -> yfinance
TRADELOCKER_TO_YFINANCE = {v: k for k, v in YFINANCE_TO_TRADELOCKER.items()}

# Alpaca -> yfinance mapping (slash to dash)
ALPACA_TO_YFINANCE = {
    'BTC/USD': 'BTC-USD',
    'ETH/USD': 'ETH-USD',
    'SOL/USD': 'SOL-USD',
    'ADA/USD': 'ADA-USD',
    'NEAR/USD': 'NEAR-USD',
    'ATOM/USD': 'ATOM-USD',
    'AVAX/USD': 'AVAX-USD',
    'LINK/USD': 'LINK-USD',
    'LTC/USD': 'LTC-USD',
    'DOT/USD': 'DOT-USD',
    'XRP/USD': 'XRP-USD',
    'MATIC/USD': 'MATIC-USD',
    'UNI/USD': 'UNI-USD',
    'APT/USD': 'APT-USD',
    'DOGE/USD': 'DOGE-USD',
}


def to_tradelocker(symbol: str) -> str:
    """Convert yfinance symbol to TradeLocker format.

    Handles both explicit mappings and auto-conversion of XXX-USD to XXXUSD.
    """
    # Check explicit mapping first
    if symbol in YFINANCE_TO_TRADELOCKER:
        return YFINANCE_TO_TRADELOCKER[symbol]

    # Auto-convert XXX-USD to XXXUSD for crypto
    if symbol.endswith('-USD'):
        return symbol.replace('-USD', 'USD')
    if symbol.endswith('/USD'):
        return symbol.replace('/USD', 'USD')

    # Return as-is for stocks and other symbols
    return symbol


def to_yfinance(symbol: str) -> str:
    """Convert broker symbol (Alpaca/TradeLocker) to yfinance format.

    Handles:
    - Alpaca format: BTC/USD -> BTC-USD
    - TradeLocker format: BTCUSD -> BTC-USD
    - Already yfinance format: BTC-USD -> BTC-USD (passthrough)
    """
    # Already in yfinance format
    if symbol.endswith('-USD'):
        return symbol

    # Check explicit Alpaca mapping (slash format)
    if symbol in ALPACA_TO_YFINANCE:
        return ALPACA_TO_YFINANCE[symbol]

    # Auto-convert XXX/USD to XXX-USD (Alpaca format)
    if symbol.endswith('/USD'):
        return symbol.replace('/USD', '-USD')

    # Check explicit TradeLocker mapping
    if symbol in TRADELOCKER_TO_YFINANCE:
        return TRADELOCKER_TO_YFINANCE[symbol]

    # Auto-convert XXXUSD to XXX-USD for crypto (TradeLocker format)
    if symbol.endswith('USD') and len(symbol) > 3:
        base = symbol[:-3]  # Remove 'USD'
        return f"{base}-USD"

    # Return as-is for stocks and other symbols
    return symbol


def is_crypto(symbol: str) -> bool:
    """Check if symbol is a crypto pair."""
    # Check explicit mappings
    if symbol in YFINANCE_TO_TRADELOCKER or symbol in TRADELOCKER_TO_YFINANCE:
        return True
    if symbol in ALPACA_TO_YFINANCE:
        return True

    # Check pattern: XXX-USD (yfinance), XXX/USD (Alpaca), or XXXUSD (TradeLocker)
    if symbol.endswith('-USD') or symbol.endswith('/USD'):
        return True
    if symbol.endswith('USD') and len(symbol) > 3:
        return True

    return False
