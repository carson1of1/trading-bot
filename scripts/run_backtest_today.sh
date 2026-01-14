#!/bin/bash
# Run backtest for today with TradeLocker symbols
# Uses the symbols from universe.yaml (121 TradeLocker-compatible symbols)

set -e

# Get today's date
TODAY=$(date +%Y-%m-%d)

# Default to last 5 trading days for meaningful results
# (single day often has too few signals)
START_DATE=$(date -d "5 days ago" +%Y-%m-%d)

echo "========================================"
echo "  TradeLocker Backtest"
echo "========================================"
echo "  Period: $START_DATE to $TODAY"
echo "  Using: universe.yaml (121 TradeLocker symbols)"
echo "========================================"
echo ""

cd "$(dirname "$0")/.."

# Run the backtest
python3 backtest.py --start "$START_DATE" --end "$TODAY" "$@"
