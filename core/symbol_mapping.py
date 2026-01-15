"""
Symbol mapping between different data sources and brokers.

yfinance uses: BTC-USD, ETH-USD, etc.
TradeLocker uses: BTCUSD, ETHUSD, etc.
"""

# yfinance -> TradeLocker mapping
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
    """Convert yfinance symbol to TradeLocker format."""
    return YFINANCE_TO_TRADELOCKER.get(symbol, symbol)


def to_yfinance(symbol: str) -> str:
    """Convert TradeLocker symbol to yfinance format."""
    return TRADELOCKER_TO_YFINANCE.get(symbol, symbol)


def is_crypto(symbol: str) -> bool:
    """Check if symbol is a crypto pair."""
    return symbol in YFINANCE_TO_TRADELOCKER or symbol in TRADELOCKER_TO_YFINANCE
