#!/bin/bash
# Start frontend in production mode
# Usage: ./scripts/start-frontend.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/../frontend"
LOG_FILE="/tmp/frontend.log"
PID_FILE="/tmp/frontend.pid"

# Validate frontend directory structure before starting
validate_frontend() {
    local errors=0

    if [ ! -f "$FRONTEND_DIR/package.json" ]; then
        echo "ERROR: package.json not found in $FRONTEND_DIR"
        errors=$((errors + 1))
    fi

    if [ ! -d "$FRONTEND_DIR/src" ]; then
        echo "ERROR: src directory not found in $FRONTEND_DIR"
        errors=$((errors + 1))
    fi

    if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
        echo "ERROR: node_modules not found. Run 'bun install' in $FRONTEND_DIR"
        errors=$((errors + 1))
    fi

    if [ ! -f "$FRONTEND_DIR/next.config.ts" ] && [ ! -f "$FRONTEND_DIR/next.config.js" ]; then
        echo "ERROR: next.config.ts/js not found in $FRONTEND_DIR"
        errors=$((errors + 1))
    fi

    # Check for corrupted .next cache (has cache but no source)
    if [ -d "$FRONTEND_DIR/.next" ] && [ ! -f "$FRONTEND_DIR/package.json" ]; then
        echo "ERROR: Corrupted state - .next cache exists but no source files"
        echo "This frontend directory appears to be incomplete or stale"
        errors=$((errors + 1))
    fi

    if [ $errors -gt 0 ]; then
        echo ""
        echo "Frontend validation failed with $errors error(s)"
        echo "Expected frontend directory: $FRONTEND_DIR"
        echo "Please ensure the frontend source files are present"
        exit 1
    fi

    echo "Frontend validation passed"
}

validate_frontend

cd "$FRONTEND_DIR"

# Kill existing frontend if running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping existing frontend (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$PID_FILE"
fi

# Also kill any stray next processes
pkill -f "next start" 2>/dev/null || true
pkill -f "next dev" 2>/dev/null || true
sleep 1

# Build if .next doesn't exist or is older than src
if [ ! -d ".next" ] || [ "$(find src -newer .next -print -quit 2>/dev/null)" ]; then
    echo "Building frontend..."
    bun run build
fi

# Start production server
echo "Starting frontend production server..."
nohup bun run start > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

sleep 2

# Verify it started
if curl -s --max-time 5 http://localhost:3000 > /dev/null; then
    echo "Frontend running on http://localhost:3000 (PID: $(cat $PID_FILE))"
else
    echo "ERROR: Frontend failed to start. Check $LOG_FILE"
    exit 1
fi
