"""
Symbol mapping between different data sources and brokers.

yfinance uses: BTC-USD, ETH-USD, etc.
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

    # Return as-is for stocks and other symbols
    return symbol


def to_yfinance(symbol: str) -> str:
    """Convert TradeLocker symbol to yfinance format.

    Handles both explicit mappings and auto-conversion of XXXUSD to XXX-USD.
    """
    # Check explicit mapping first
    if symbol in TRADELOCKER_TO_YFINANCE:
        return TRADELOCKER_TO_YFINANCE[symbol]

    # Auto-convert XXXUSD to XXX-USD for crypto
    # Only for symbols ending in USD that look like crypto (e.g., BTCUSD, ETHUSD)
    if symbol.endswith('USD') and len(symbol) > 3:
        base = symbol[:-3]  # Remove 'USD'
        # Avoid converting stock symbols like 'NVDA' that happen to... wait, NVDA doesn't end in USD
        # This should be safe for crypto symbols
        return f"{base}-USD"

    # Return as-is for stocks and other symbols
    return symbol


def is_crypto(symbol: str) -> bool:
    """Check if symbol is a crypto pair."""
    # Check explicit mappings
    if symbol in YFINANCE_TO_TRADELOCKER or symbol in TRADELOCKER_TO_YFINANCE:
        return True

    # Check pattern: XXX-USD (yfinance) or XXXUSD (TradeLocker)
    if symbol.endswith('-USD') or (symbol.endswith('USD') and len(symbol) > 3):
        return True

    return False
