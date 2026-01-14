#!/bin/bash
# Frontend health check test script
# Run this to verify the frontend is properly set up and can start
# Usage: ./scripts/test-frontend-health.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/../frontend"
TEST_PORT=3999
PASSED=0
FAILED=0

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    PASSED=$((PASSED + 1))
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    FAILED=$((FAILED + 1))
}

echo "========================================"
echo "Frontend Health Check Tests"
echo "========================================"
echo "Frontend dir: $FRONTEND_DIR"
echo ""

# Test 1: Directory structure
echo "--- Structure Tests ---"

if [ -f "$FRONTEND_DIR/package.json" ]; then
    pass "package.json exists"
else
    fail "package.json missing"
fi

if [ -d "$FRONTEND_DIR/src" ]; then
    pass "src directory exists"
else
    fail "src directory missing"
fi

if [ -d "$FRONTEND_DIR/node_modules" ]; then
    pass "node_modules exists"
else
    fail "node_modules missing (run 'bun install')"
fi

if [ -f "$FRONTEND_DIR/next.config.ts" ] || [ -f "$FRONTEND_DIR/next.config.js" ]; then
    pass "next.config exists"
else
    fail "next.config.ts/js missing"
fi

# Test 2: Check for corrupted state
echo ""
echo "--- Corruption Tests ---"

if [ -d "$FRONTEND_DIR/.next" ] && [ ! -f "$FRONTEND_DIR/package.json" ]; then
    fail "Corrupted state: .next cache exists without source files"
else
    pass "No corrupted .next cache detected"
fi

# Test 3: Required source files
echo ""
echo "--- Source File Tests ---"

if [ -d "$FRONTEND_DIR/src/app" ]; then
    pass "src/app directory exists (Next.js App Router)"
else
    fail "src/app directory missing"
fi

# Test 4: Package.json validation
echo ""
echo "--- Package.json Tests ---"

if [ -f "$FRONTEND_DIR/package.json" ]; then
    if grep -q '"next"' "$FRONTEND_DIR/package.json"; then
        pass "Next.js dependency found"
    else
        fail "Next.js dependency missing from package.json"
    fi

    if grep -q '"dev"' "$FRONTEND_DIR/package.json"; then
        pass "dev script exists"
    else
        fail "dev script missing from package.json"
    fi

    if grep -q '"build"' "$FRONTEND_DIR/package.json"; then
        pass "build script exists"
    else
        fail "build script missing from package.json"
    fi
fi

# Test 5: Build test (optional, only if no failures so far)
echo ""
echo "--- Build Test ---"

if [ $FAILED -eq 0 ]; then
    echo "Running Next.js build check..."
    cd "$FRONTEND_DIR"

    # Try to run a type check / lint
    if bun run lint 2>/dev/null; then
        pass "Lint check passed"
    else
        fail "Lint check failed"
    fi
else
    echo "Skipping build test due to previous failures"
fi

# Test 6: Server start test (only if no failures)
echo ""
echo "--- Server Start Test ---"

if [ $FAILED -eq 0 ]; then
    echo "Testing if dev server can start on port $TEST_PORT..."
    cd "$FRONTEND_DIR"

    # Start server on test port in background
    PORT=$TEST_PORT bun run dev &
    DEV_PID=$!

    # Wait for server to start
    sleep 5

    # Check if it's running
    if curl -s --max-time 5 "http://localhost:$TEST_PORT" > /dev/null 2>&1; then
        pass "Dev server started successfully"
    else
        fail "Dev server failed to start"
    fi

    # Cleanup
    kill $DEV_PID 2>/dev/null || true
    sleep 1
else
    echo "Skipping server start test due to previous failures"
fi

# Summary
echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Please fix the issues above.${NC}"
    exit 1
fi
