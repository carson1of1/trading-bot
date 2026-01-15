#!/bin/bash
# Switch between broker modes
# Usage: ./switch_mode.sh [alpaca|tradelocker]

CONFIG="config.yaml"

if [ "$1" == "alpaca" ]; then
    echo "Switching to ALPACA (paper) mode..."
    sed -i 's/^mode: .*/mode: PAPER/' $CONFIG
    sed -i 's/^crypto_mode: .*/crypto_mode: false/' $CONFIG
    sed -i 's/watchlist_file: .*/watchlist_file: universe_alpaca.yaml/' $CONFIG
    sed -i 's/eod_close: .*/eod_close: true/' $CONFIG
    echo "Done! Config set for Alpaca paper trading."
    echo "  - Mode: PAPER"
    echo "  - Universe: universe_alpaca.yaml (stocks + crypto)"
    echo "  - EOD Close: enabled"

elif [ "$1" == "alpaca-live" ]; then
    echo "Switching to ALPACA LIVE mode..."
    sed -i 's/^mode: .*/mode: LIVE/' $CONFIG
    sed -i 's/^crypto_mode: .*/crypto_mode: false/' $CONFIG
    sed -i 's/watchlist_file: .*/watchlist_file: universe_alpaca.yaml/' $CONFIG
    sed -i 's/eod_close: .*/eod_close: true/' $CONFIG
    echo "Done! Config set for Alpaca LIVE trading."
    echo "  - Mode: LIVE"
    echo "  - Universe: universe_alpaca.yaml (stocks + crypto)"
    echo "  - EOD Close: enabled"

elif [ "$1" == "tradelocker" ]; then
    echo "Switching to TRADELOCKER mode..."
    sed -i 's/^mode: .*/mode: TRADELOCKER/' $CONFIG
    sed -i 's/^crypto_mode: .*/crypto_mode: true/' $CONFIG
    sed -i 's/watchlist_file: .*/watchlist_file: universe_tradelocker_crypto.yaml/' $CONFIG
    sed -i 's/eod_close: .*/eod_close: false/' $CONFIG
    echo "Done! Config set for TradeLocker."
    echo "  - Mode: TRADELOCKER"
    echo "  - Universe: universe_tradelocker_crypto.yaml (crypto only)"
    echo "  - EOD Close: disabled (24/7 crypto)"

else
    echo "Usage: ./switch_mode.sh [alpaca|alpaca-live|tradelocker]"
    echo ""
    echo "Current mode:"
    grep "^mode:" $CONFIG
    grep "watchlist_file:" $CONFIG
    exit 1
fi
