# Safe Auto-Start Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement automated startup at market open with preflight safety checks that block trading if any check fails.

**Architecture:** `PreflightChecklist` class in `core/preflight.py` runs 7 safety checks before enabling trading. Bot starts via systemd timer at 9:25 AM ET weekdays. If any check fails, exit with code 1.

**Tech Stack:** Python 3, pytest, systemd, existing MarketHours/broker abstractions

---

### Task 1: Create CheckResult and PreflightChecklist skeleton

**Files:**
- Create: `core/preflight.py`
- Create: `tests/test_preflight.py`
- Modify: `core/__init__.py` (add export)

**Step 1: Write the failing test for CheckResult**

```python
# tests/test_preflight.py
"""Tests for preflight checklist."""
import pytest


class TestCheckResult:
    """Test CheckResult namedtuple."""

    def test_check_result_fields(self):
        """CheckResult has name, passed, and message fields."""
        from core.preflight import CheckResult

        result = CheckResult(name="test_check", passed=True, message="All good")

        assert result.name == "test_check"
        assert result.passed is True
        assert result.message == "All good"

    def test_check_result_failed(self):
        """CheckResult can represent failed check."""
        from core.preflight import CheckResult

        result = CheckResult(name="api_keys", passed=False, message="Missing API key")

        assert result.passed is False
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_preflight.py::TestCheckResult -v`
Expected: FAIL with "cannot import name 'CheckResult'"

**Step 3: Write minimal implementation**

```python
# core/preflight.py
"""
Preflight checklist for safe auto-start.

Runs safety checks before enabling trading. All checks must pass
for trading to be enabled.
"""
import logging
from typing import NamedTuple, List, Tuple

logger = logging.getLogger(__name__)


class CheckResult(NamedTuple):
    """Result of a single preflight check."""
    name: str
    passed: bool
    message: str


class PreflightChecklist:
    """
    Run preflight checks before enabling trading.

    All checks must pass for trading to proceed.
    """

    def __init__(self, config: dict, broker):
        """
        Initialize preflight checklist.

        Args:
            config: Bot configuration dict
            broker: Broker instance for API checks
        """
        self.config = config
        self.broker = broker

    def run_all_checks(self) -> Tuple[bool, List[CheckResult]]:
        """
        Run all preflight checks.

        Returns:
            Tuple of (all_passed, list of CheckResults)
        """
        results = []
        # TODO: Add checks
        all_passed = all(r.passed for r in results)
        return all_passed, results
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_preflight.py::TestCheckResult -v`
Expected: PASS

**Step 5: Add export to core/__init__.py**

Add to `core/__init__.py`:
```python
from core.preflight import PreflightChecklist, CheckResult
```

**Step 6: Commit**

```bash
git add core/preflight.py tests/test_preflight.py core/__init__.py
git commit -m "feat(preflight): add CheckResult and PreflightChecklist skeleton"
```

---

### Task 2: Implement check_api_keys

**Files:**
- Modify: `core/preflight.py`
- Modify: `tests/test_preflight.py`

**Step 1: Write the failing tests**

```python
# tests/test_preflight.py - add to file
import os
from unittest.mock import patch, MagicMock


class TestCheckApiKeys:
    """Test API key validation."""

    def test_check_api_keys_present(self):
        """Passes when both API keys are set."""
        from core.preflight import PreflightChecklist

        with patch.dict(os.environ, {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret'
        }):
            checklist = PreflightChecklist({}, MagicMock())
            result = checklist.check_api_keys()

            assert result.passed is True
            assert result.name == "api_keys"
            assert "loaded" in result.message.lower()

    def test_check_api_keys_missing_key(self):
        """Fails when API key is missing."""
        from core.preflight import PreflightChecklist

        with patch.dict(os.environ, {'ALPACA_SECRET_KEY': 'test_secret'}, clear=True):
            # Ensure ALPACA_API_KEY is not set
            os.environ.pop('ALPACA_API_KEY', None)
            checklist = PreflightChecklist({}, MagicMock())
            result = checklist.check_api_keys()

            assert result.passed is False
            assert "ALPACA_API_KEY" in result.message

    def test_check_api_keys_missing_secret(self):
        """Fails when secret key is missing."""
        from core.preflight import PreflightChecklist

        with patch.dict(os.environ, {'ALPACA_API_KEY': 'test_key'}, clear=True):
            os.environ.pop('ALPACA_SECRET_KEY', None)
            checklist = PreflightChecklist({}, MagicMock())
            result = checklist.check_api_keys()

            assert result.passed is False
            assert "ALPACA_SECRET_KEY" in result.message

    def test_check_api_keys_empty_value(self):
        """Fails when API key is empty string."""
        from core.preflight import PreflightChecklist

        with patch.dict(os.environ, {
            'ALPACA_API_KEY': '',
            'ALPACA_SECRET_KEY': 'test_secret'
        }):
            checklist = PreflightChecklist({}, MagicMock())
            result = checklist.check_api_keys()

            assert result.passed is False
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_preflight.py::TestCheckApiKeys -v`
Expected: FAIL with "has no attribute 'check_api_keys'"

**Step 3: Write implementation**

Add to `core/preflight.py` in PreflightChecklist class:

```python
import os

    def check_api_keys(self) -> CheckResult:
        """Check that Alpaca API keys are loaded."""
        api_key = os.environ.get('ALPACA_API_KEY', '')
        secret_key = os.environ.get('ALPACA_SECRET_KEY', '')

        missing = []
        if not api_key:
            missing.append('ALPACA_API_KEY')
        if not secret_key:
            missing.append('ALPACA_SECRET_KEY')

        if missing:
            return CheckResult(
                name="api_keys",
                passed=False,
                message=f"Missing environment variables: {', '.join(missing)}"
            )

        return CheckResult(
            name="api_keys",
            passed=True,
            message="API keys loaded"
        )
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_preflight.py::TestCheckApiKeys -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add core/preflight.py tests/test_preflight.py
git commit -m "feat(preflight): add check_api_keys"
```

---

### Task 3: Implement check_no_duplicate_process

**Files:**
- Modify: `core/preflight.py`
- Modify: `tests/test_preflight.py`

**Step 1: Write the failing tests**

```python
# tests/test_preflight.py - add to file
from pathlib import Path


class TestCheckNoDuplicateProcess:
    """Test duplicate process detection."""

    def test_no_pid_file(self, tmp_path):
        """Passes when no PID file exists."""
        from core.preflight import PreflightChecklist

        pid_file = tmp_path / "bot.pid"
        checklist = PreflightChecklist({}, MagicMock())

        result = checklist.check_no_duplicate_process(pid_file)

        assert result.passed is True
        assert "no duplicate" in result.message.lower()

    def test_stale_pid_file(self, tmp_path):
        """Passes when PID file exists but process is dead."""
        from core.preflight import PreflightChecklist

        pid_file = tmp_path / "bot.pid"
        pid_file.write_text("999999")  # Non-existent PID

        checklist = PreflightChecklist({}, MagicMock())
        result = checklist.check_no_duplicate_process(pid_file)

        assert result.passed is True
        # Should also clean up stale PID file
        assert not pid_file.exists()

    def test_running_process(self, tmp_path):
        """Fails when another bot process is running."""
        from core.preflight import PreflightChecklist

        pid_file = tmp_path / "bot.pid"
        # Use current process PID (known to be running)
        pid_file.write_text(str(os.getpid()))

        checklist = PreflightChecklist({}, MagicMock())
        result = checklist.check_no_duplicate_process(pid_file)

        assert result.passed is False
        assert "already running" in result.message.lower()

    def test_invalid_pid_file(self, tmp_path):
        """Passes when PID file contains invalid data."""
        from core.preflight import PreflightChecklist

        pid_file = tmp_path / "bot.pid"
        pid_file.write_text("not_a_number")

        checklist = PreflightChecklist({}, MagicMock())
        result = checklist.check_no_duplicate_process(pid_file)

        assert result.passed is True
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_preflight.py::TestCheckNoDuplicateProcess -v`
Expected: FAIL with "has no attribute 'check_no_duplicate_process'"

**Step 3: Write implementation**

Add to `core/preflight.py`:

```python
from pathlib import Path

    def check_no_duplicate_process(self, pid_file: Path = None) -> CheckResult:
        """Check that no other bot process is running."""
        if pid_file is None:
            pid_file = Path(__file__).parent.parent / "logs" / "bot.pid"

        if not pid_file.exists():
            return CheckResult(
                name="no_duplicate_process",
                passed=True,
                message="No duplicate process (no PID file)"
            )

        try:
            pid = int(pid_file.read_text().strip())
            # Check if process exists (signal 0 doesn't kill, just checks)
            os.kill(pid, 0)
            # Process exists
            return CheckResult(
                name="no_duplicate_process",
                passed=False,
                message=f"Bot already running (PID {pid})"
            )
        except (ValueError, ProcessLookupError, OSError):
            # Invalid PID or process doesn't exist - clean up stale file
            try:
                pid_file.unlink()
            except OSError:
                pass
            return CheckResult(
                name="no_duplicate_process",
                passed=True,
                message="No duplicate process (stale PID file removed)"
            )
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_preflight.py::TestCheckNoDuplicateProcess -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add core/preflight.py tests/test_preflight.py
git commit -m "feat(preflight): add check_no_duplicate_process"
```

---

### Task 4: Implement check_universe_loaded

**Files:**
- Modify: `core/preflight.py`
- Modify: `tests/test_preflight.py`

**Step 1: Write the failing tests**

```python
# tests/test_preflight.py - add to file
import yaml


class TestCheckUniverseLoaded:
    """Test universe file loading."""

    def test_universe_loaded(self, tmp_path):
        """Passes when universe file has symbols."""
        from core.preflight import PreflightChecklist

        universe_file = tmp_path / "universe.yaml"
        universe_file.write_text(yaml.dump({
            'scanner_universe': {
                'tech': ['AAPL', 'MSFT', 'GOOGL']
            }
        }))

        config = {'trading': {'watchlist_file': str(universe_file)}}
        checklist = PreflightChecklist(config, MagicMock())
        checklist.bot_dir = tmp_path

        result = checklist.check_universe_loaded()

        assert result.passed is True
        assert "3 symbols" in result.message

    def test_universe_empty(self, tmp_path):
        """Fails when universe file has no symbols."""
        from core.preflight import PreflightChecklist

        universe_file = tmp_path / "universe.yaml"
        universe_file.write_text(yaml.dump({'scanner_universe': {}}))

        config = {'trading': {'watchlist_file': str(universe_file)}}
        checklist = PreflightChecklist(config, MagicMock())
        checklist.bot_dir = tmp_path

        result = checklist.check_universe_loaded()

        assert result.passed is False
        assert "empty" in result.message.lower()

    def test_universe_missing(self, tmp_path):
        """Fails when universe file doesn't exist."""
        from core.preflight import PreflightChecklist

        config = {'trading': {'watchlist_file': 'nonexistent.yaml'}}
        checklist = PreflightChecklist(config, MagicMock())
        checklist.bot_dir = tmp_path

        result = checklist.check_universe_loaded()

        assert result.passed is False
        assert "not found" in result.message.lower() or "error" in result.message.lower()
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_preflight.py::TestCheckUniverseLoaded -v`
Expected: FAIL with "has no attribute 'check_universe_loaded'"

**Step 3: Write implementation**

Add to `core/preflight.py`:

```python
import yaml

    def check_universe_loaded(self) -> CheckResult:
        """Check that symbol universe file is loaded and non-empty."""
        try:
            watchlist_file = self.config.get('trading', {}).get('watchlist_file', 'universe.yaml')

            # Handle absolute vs relative paths
            if not Path(watchlist_file).is_absolute():
                bot_dir = getattr(self, 'bot_dir', Path(__file__).parent.parent)
                watchlist_file = bot_dir / watchlist_file

            with open(watchlist_file, 'r') as f:
                universe = yaml.safe_load(f)

            # Count symbols from scanner_universe
            scanner_universe = universe.get('scanner_universe', {})
            symbols = []
            for category, syms in scanner_universe.items():
                if isinstance(syms, list):
                    symbols.extend(syms)

            # Deduplicate
            symbols = list(set(symbols))

            if not symbols:
                return CheckResult(
                    name="universe_loaded",
                    passed=False,
                    message="Universe file is empty (no symbols)"
                )

            return CheckResult(
                name="universe_loaded",
                passed=True,
                message=f"Universe loaded: {len(symbols)} symbols"
            )

        except FileNotFoundError:
            return CheckResult(
                name="universe_loaded",
                passed=False,
                message=f"Universe file not found: {watchlist_file}"
            )
        except Exception as e:
            return CheckResult(
                name="universe_loaded",
                passed=False,
                message=f"Error loading universe: {e}"
            )
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_preflight.py::TestCheckUniverseLoaded -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add core/preflight.py tests/test_preflight.py
git commit -m "feat(preflight): add check_universe_loaded"
```

---

### Task 5: Implement check_account_balance

**Files:**
- Modify: `core/preflight.py`
- Modify: `tests/test_preflight.py`

**Step 1: Write the failing tests**

```python
# tests/test_preflight.py - add to file

class TestCheckAccountBalance:
    """Test account balance validation."""

    def test_account_balance_success(self):
        """Passes when balance is fetched and positive."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_account = MagicMock()
        mock_account.cash = "10000.00"
        mock_account.portfolio_value = "25000.00"
        mock_broker.get_account.return_value = mock_account

        checklist = PreflightChecklist({}, mock_broker)
        result = checklist.check_account_balance()

        assert result.passed is True
        assert "$25,000" in result.message or "25000" in result.message

    def test_account_balance_zero(self):
        """Fails when balance is zero."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_account = MagicMock()
        mock_account.cash = "0"
        mock_account.portfolio_value = "0"
        mock_broker.get_account.return_value = mock_account

        checklist = PreflightChecklist({}, mock_broker)
        result = checklist.check_account_balance()

        assert result.passed is False
        assert "zero" in result.message.lower() or "0" in result.message

    def test_account_balance_api_error(self):
        """Fails when API call fails."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_broker.get_account.side_effect = Exception("API connection failed")

        checklist = PreflightChecklist({}, mock_broker)
        result = checklist.check_account_balance()

        assert result.passed is False
        assert "error" in result.message.lower() or "failed" in result.message.lower()
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_preflight.py::TestCheckAccountBalance -v`
Expected: FAIL with "has no attribute 'check_account_balance'"

**Step 3: Write implementation**

Add to `core/preflight.py`:

```python
    def check_account_balance(self) -> CheckResult:
        """Check that account balance can be fetched and is positive."""
        try:
            account = self.broker.get_account()
            portfolio_value = float(account.portfolio_value)

            if portfolio_value <= 0:
                return CheckResult(
                    name="account_balance",
                    passed=False,
                    message=f"Account balance is zero or negative: ${portfolio_value:,.2f}"
                )

            return CheckResult(
                name="account_balance",
                passed=True,
                message=f"Account balance: ${portfolio_value:,.2f}"
            )

        except Exception as e:
            return CheckResult(
                name="account_balance",
                passed=False,
                message=f"Failed to fetch account: {e}"
            )
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_preflight.py::TestCheckAccountBalance -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add core/preflight.py tests/test_preflight.py
git commit -m "feat(preflight): add check_account_balance"
```

---

### Task 6: Implement check_market_status

**Files:**
- Modify: `core/preflight.py`
- Modify: `tests/test_preflight.py`

**Step 1: Write the failing tests**

```python
# tests/test_preflight.py - add to file

class TestCheckMarketStatus:
    """Test market status validation."""

    def test_market_open(self):
        """Passes when market is open."""
        from core.preflight import PreflightChecklist

        with patch('core.preflight.MarketHours') as MockMarketHours:
            mock_mh = MagicMock()
            mock_mh.is_market_open.return_value = True
            MockMarketHours.return_value = mock_mh

            checklist = PreflightChecklist({}, MagicMock())
            result = checklist.check_market_status()

            assert result.passed is True
            assert "open" in result.message.lower()

    def test_market_opens_soon(self):
        """Passes when market opens within 10 minutes."""
        from core.preflight import PreflightChecklist

        with patch('core.preflight.MarketHours') as MockMarketHours:
            mock_mh = MagicMock()
            mock_mh.is_market_open.return_value = False
            mock_mh.time_until_market_open.return_value = 5  # 5 minutes
            MockMarketHours.return_value = mock_mh

            checklist = PreflightChecklist({}, MagicMock())
            result = checklist.check_market_status()

            assert result.passed is True
            assert "5" in result.message

    def test_market_closed(self):
        """Fails when market is closed and not opening soon."""
        from core.preflight import PreflightChecklist

        with patch('core.preflight.MarketHours') as MockMarketHours:
            mock_mh = MagicMock()
            mock_mh.is_market_open.return_value = False
            mock_mh.time_until_market_open.return_value = 120  # 2 hours
            MockMarketHours.return_value = mock_mh

            checklist = PreflightChecklist({}, MagicMock())
            result = checklist.check_market_status()

            assert result.passed is False
            assert "closed" in result.message.lower()

    def test_market_opens_in_10_minutes_edge(self):
        """Passes at exactly 10 minutes until open."""
        from core.preflight import PreflightChecklist

        with patch('core.preflight.MarketHours') as MockMarketHours:
            mock_mh = MagicMock()
            mock_mh.is_market_open.return_value = False
            mock_mh.time_until_market_open.return_value = 10
            MockMarketHours.return_value = mock_mh

            checklist = PreflightChecklist({}, MagicMock())
            result = checklist.check_market_status()

            assert result.passed is True
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_preflight.py::TestCheckMarketStatus -v`
Expected: FAIL with "has no attribute 'check_market_status'"

**Step 3: Write implementation**

Add to `core/preflight.py` (also add import at top):

```python
from core.market_hours import MarketHours

    def check_market_status(self) -> CheckResult:
        """Check that market is open or opens within 10 minutes."""
        market_hours = MarketHours()

        if market_hours.is_market_open():
            return CheckResult(
                name="market_status",
                passed=True,
                message="Market is open"
            )

        minutes_until_open = market_hours.time_until_market_open()

        if 0 < minutes_until_open <= 10:
            return CheckResult(
                name="market_status",
                passed=True,
                message=f"Market opens in {minutes_until_open} minutes"
            )

        return CheckResult(
            name="market_status",
            passed=False,
            message=f"Market closed ({minutes_until_open} min until open)"
        )
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_preflight.py::TestCheckMarketStatus -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add core/preflight.py tests/test_preflight.py
git commit -m "feat(preflight): add check_market_status"
```

---

### Task 7: Implement check_daily_loss_reset

**Files:**
- Modify: `core/preflight.py`
- Modify: `tests/test_preflight.py`

**Step 1: Write the failing tests**

```python
# tests/test_preflight.py - add to file

class TestCheckDailyLossReset:
    """Test daily loss reset validation."""

    def test_daily_loss_reset_success(self):
        """Passes when last_equity is accessible."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_account = MagicMock()
        mock_account.last_equity = "25000.00"
        mock_broker.get_account.return_value = mock_account

        checklist = PreflightChecklist({}, mock_broker)
        result = checklist.check_daily_loss_reset()

        assert result.passed is True
        assert "25,000" in result.message or "25000" in result.message

    def test_daily_loss_reset_no_last_equity(self):
        """Fails when last_equity is not available."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_account = MagicMock(spec=[])  # No last_equity attribute
        mock_broker.get_account.return_value = mock_account

        checklist = PreflightChecklist({}, mock_broker)
        result = checklist.check_daily_loss_reset()

        assert result.passed is False

    def test_daily_loss_reset_api_error(self):
        """Fails when API call fails."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_broker.get_account.side_effect = Exception("Connection refused")

        checklist = PreflightChecklist({}, mock_broker)
        result = checklist.check_daily_loss_reset()

        assert result.passed is False
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_preflight.py::TestCheckDailyLossReset -v`
Expected: FAIL with "has no attribute 'check_daily_loss_reset'"

**Step 3: Write implementation**

Add to `core/preflight.py`:

```python
    def check_daily_loss_reset(self) -> CheckResult:
        """Check that daily loss tracking can be initialized."""
        try:
            account = self.broker.get_account()
            last_equity = float(account.last_equity)

            return CheckResult(
                name="daily_loss_reset",
                passed=True,
                message=f"Daily loss reset (starting equity: ${last_equity:,.2f})"
            )

        except AttributeError:
            return CheckResult(
                name="daily_loss_reset",
                passed=False,
                message="Cannot access last_equity for daily loss tracking"
            )
        except Exception as e:
            return CheckResult(
                name="daily_loss_reset",
                passed=False,
                message=f"Failed to fetch account for daily loss: {e}"
            )
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_preflight.py::TestCheckDailyLossReset -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add core/preflight.py tests/test_preflight.py
git commit -m "feat(preflight): add check_daily_loss_reset"
```

---

### Task 8: Implement check_positions_accounted

**Files:**
- Modify: `core/preflight.py`
- Modify: `tests/test_preflight.py`

**Step 1: Write the failing tests**

```python
# tests/test_preflight.py - add to file

class TestCheckPositionsAccounted:
    """Test position accounting validation."""

    def test_no_positions(self):
        """Passes when no positions are open."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = []

        checklist = PreflightChecklist({}, mock_broker)
        checklist.watchlist = ['AAPL', 'MSFT']

        result = checklist.check_positions_accounted()

        assert result.passed is True
        assert "0" in result.message

    def test_positions_in_watchlist(self):
        """Passes when all positions are in watchlist."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_pos = MagicMock()
        mock_pos.symbol = 'AAPL'
        mock_broker.get_positions.return_value = [mock_pos]

        checklist = PreflightChecklist({}, mock_broker)
        checklist.watchlist = ['AAPL', 'MSFT', 'GOOGL']

        result = checklist.check_positions_accounted()

        assert result.passed is True
        assert "1" in result.message

    def test_orphaned_positions(self):
        """Fails when positions exist that aren't in watchlist."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_pos = MagicMock()
        mock_pos.symbol = 'TSLA'  # Not in watchlist
        mock_broker.get_positions.return_value = [mock_pos]

        checklist = PreflightChecklist({}, mock_broker)
        checklist.watchlist = ['AAPL', 'MSFT']

        result = checklist.check_positions_accounted()

        assert result.passed is False
        assert "TSLA" in result.message

    def test_api_error(self):
        """Fails when API call fails."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_broker.get_positions.side_effect = Exception("API error")

        checklist = PreflightChecklist({}, mock_broker)
        checklist.watchlist = ['AAPL']

        result = checklist.check_positions_accounted()

        assert result.passed is False
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_preflight.py::TestCheckPositionsAccounted -v`
Expected: FAIL with "has no attribute 'check_positions_accounted'"

**Step 3: Write implementation**

Add to `core/preflight.py`:

```python
    def check_positions_accounted(self) -> CheckResult:
        """Check that all open positions are in the watchlist."""
        try:
            positions = self.broker.get_positions()

            if not positions:
                return CheckResult(
                    name="positions_accounted",
                    passed=True,
                    message="No open positions (0)"
                )

            # Get watchlist from instance or empty list
            watchlist = getattr(self, 'watchlist', [])
            watchlist_set = set(watchlist)

            position_symbols = [p.symbol for p in positions]
            orphaned = [s for s in position_symbols if s not in watchlist_set]

            if orphaned:
                return CheckResult(
                    name="positions_accounted",
                    passed=False,
                    message=f"Orphaned positions not in watchlist: {', '.join(orphaned)}"
                )

            return CheckResult(
                name="positions_accounted",
                passed=True,
                message=f"All positions accounted ({len(positions)} open)"
            )

        except Exception as e:
            return CheckResult(
                name="positions_accounted",
                passed=False,
                message=f"Failed to fetch positions: {e}"
            )
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_preflight.py::TestCheckPositionsAccounted -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add core/preflight.py tests/test_preflight.py
git commit -m "feat(preflight): add check_positions_accounted"
```

---

### Task 9: Implement run_all_checks

**Files:**
- Modify: `core/preflight.py`
- Modify: `tests/test_preflight.py`

**Step 1: Write the failing tests**

```python
# tests/test_preflight.py - add to file

class TestRunAllChecks:
    """Test run_all_checks aggregation."""

    def test_all_checks_pass(self, tmp_path):
        """Returns True when all checks pass."""
        from core.preflight import PreflightChecklist

        # Setup mocks for all checks to pass
        mock_broker = MagicMock()
        mock_account = MagicMock()
        mock_account.portfolio_value = "25000.00"
        mock_account.last_equity = "25000.00"
        mock_broker.get_account.return_value = mock_account
        mock_broker.get_positions.return_value = []

        universe_file = tmp_path / "universe.yaml"
        universe_file.write_text(yaml.dump({
            'scanner_universe': {'tech': ['AAPL', 'MSFT']}
        }))

        config = {'trading': {'watchlist_file': str(universe_file)}}

        with patch.dict(os.environ, {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret'
        }), patch('core.preflight.MarketHours') as MockMH:
            mock_mh = MagicMock()
            mock_mh.is_market_open.return_value = True
            MockMH.return_value = mock_mh

            checklist = PreflightChecklist(config, mock_broker)
            checklist.bot_dir = tmp_path
            checklist.watchlist = ['AAPL', 'MSFT']

            # Use a non-existent PID file
            pid_file = tmp_path / "bot.pid"

            all_passed, results = checklist.run_all_checks(pid_file=pid_file)

            assert all_passed is True
            assert len(results) == 7
            assert all(r.passed for r in results)

    def test_one_check_fails(self, tmp_path):
        """Returns False when any check fails."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_account = MagicMock()
        mock_account.portfolio_value = "25000.00"
        mock_account.last_equity = "25000.00"
        mock_broker.get_account.return_value = mock_account
        mock_broker.get_positions.return_value = []

        universe_file = tmp_path / "universe.yaml"
        universe_file.write_text(yaml.dump({
            'scanner_universe': {'tech': ['AAPL']}
        }))

        config = {'trading': {'watchlist_file': str(universe_file)}}

        # Missing API key - this will cause one check to fail
        with patch.dict(os.environ, {'ALPACA_SECRET_KEY': 'test_secret'}, clear=True), \
             patch('core.preflight.MarketHours') as MockMH:
            os.environ.pop('ALPACA_API_KEY', None)
            mock_mh = MagicMock()
            mock_mh.is_market_open.return_value = True
            MockMH.return_value = mock_mh

            checklist = PreflightChecklist(config, mock_broker)
            checklist.bot_dir = tmp_path
            checklist.watchlist = ['AAPL']

            pid_file = tmp_path / "bot.pid"
            all_passed, results = checklist.run_all_checks(pid_file=pid_file)

            assert all_passed is False
            failed = [r for r in results if not r.passed]
            assert len(failed) >= 1

    def test_runs_all_checks_even_if_one_fails(self, tmp_path):
        """Runs all checks even when early ones fail (no short-circuit)."""
        from core.preflight import PreflightChecklist

        mock_broker = MagicMock()
        mock_broker.get_account.side_effect = Exception("API down")
        mock_broker.get_positions.side_effect = Exception("API down")

        universe_file = tmp_path / "universe.yaml"
        universe_file.write_text(yaml.dump({
            'scanner_universe': {'tech': ['AAPL']}
        }))

        config = {'trading': {'watchlist_file': str(universe_file)}}

        with patch.dict(os.environ, {}, clear=True), \
             patch('core.preflight.MarketHours') as MockMH:
            mock_mh = MagicMock()
            mock_mh.is_market_open.return_value = False
            mock_mh.time_until_market_open.return_value = 120
            MockMH.return_value = mock_mh

            checklist = PreflightChecklist(config, mock_broker)
            checklist.bot_dir = tmp_path
            checklist.watchlist = ['AAPL']

            pid_file = tmp_path / "bot.pid"
            all_passed, results = checklist.run_all_checks(pid_file=pid_file)

            # Should have run all 7 checks even though many failed
            assert len(results) == 7
            assert all_passed is False
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_preflight.py::TestRunAllChecks -v`
Expected: FAIL (run_all_checks doesn't run actual checks yet)

**Step 3: Write implementation**

Update `run_all_checks` in `core/preflight.py`:

```python
    def run_all_checks(self, pid_file: Path = None) -> Tuple[bool, List[CheckResult]]:
        """
        Run all preflight checks.

        Args:
            pid_file: Optional path to PID file (for testing)

        Returns:
            Tuple of (all_passed, list of CheckResults)
        """
        results = []

        # Run each check and log result
        checks = [
            ("API keys", self.check_api_keys),
            ("Account balance", self.check_account_balance),
            ("Market status", self.check_market_status),
            ("Daily loss reset", self.check_daily_loss_reset),
            ("Positions accounted", self.check_positions_accounted),
            ("Universe loaded", self.check_universe_loaded),
            ("No duplicate process", lambda: self.check_no_duplicate_process(pid_file)),
        ]

        for check_name, check_fn in checks:
            result = check_fn()
            results.append(result)

            # Log each check result
            status = "✓" if result.passed else "✗"
            logger.info(f"[PREFLIGHT] {status} {result.message}")

        all_passed = all(r.passed for r in results)

        if all_passed:
            logger.info(f"[PREFLIGHT] PASSED - {len(results)} of {len(results)} checks passed")
        else:
            failed_count = len([r for r in results if not r.passed])
            logger.error(f"[PREFLIGHT] FAILED - {failed_count} of {len(results)} checks failed")

        return all_passed, results
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_preflight.py::TestRunAllChecks -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add core/preflight.py tests/test_preflight.py
git commit -m "feat(preflight): implement run_all_checks aggregation"
```

---

### Task 10: Integrate with bot.py

**Files:**
- Modify: `bot.py`

**Step 1: Write integration test**

```python
# tests/test_preflight.py - add to file

class TestBotIntegration:
    """Test integration with bot.py."""

    def test_run_preflight_method_exists(self):
        """TradingBot has run_preflight method."""
        # Import to verify method exists (don't actually run)
        from bot import TradingBot
        assert hasattr(TradingBot, 'run_preflight')
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_preflight.py::TestBotIntegration -v`
Expected: FAIL with "has no attribute 'run_preflight'"

**Step 3: Add run_preflight to TradingBot**

Add to `bot.py` in TradingBot class (after `__init__`, around line 220):

```python
    def run_preflight(self) -> bool:
        """
        Run preflight checks before enabling trading.

        Returns:
            True if all checks pass, False otherwise.
        """
        from core.preflight import PreflightChecklist

        logger.info("Running preflight checks...")

        checklist = PreflightChecklist(self.config, self.broker)
        checklist.bot_dir = self.bot_dir
        checklist.watchlist = self.watchlist

        all_passed, results = checklist.run_all_checks()

        if not all_passed:
            failed = [r for r in results if not r.passed]
            logger.error(f"PREFLIGHT FAILED: {len(failed)} check(s) failed")
            for r in failed:
                logger.error(f"  - {r.name}: {r.message}")
            return False

        logger.info("PREFLIGHT PASSED: All checks passed, trading enabled")
        return True
```

**Step 4: Modify main() to call preflight**

In `bot.py`, modify `main()` function (around line 2088):

```python
def main():
    """Main entry point for trading bot."""
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated list of symbols from scanner (overrides config)')
    parser.add_argument('--candle-delay', type=int, default=31,
                        help='Minutes after hour to run cycle (default: 31 for :30 bar alignment)')
    parser.add_argument('--skip-preflight', action='store_true',
                        help='Skip preflight checks (for manual/debug runs)')
    args = parser.parse_args()

    # Parse symbols if provided
    scanner_symbols = None
    if args.symbols:
        scanner_symbols = [s.strip() for s in args.symbols.split(',')]
        print(f"[SCANNER] Using {len(scanner_symbols)} symbols from scanner: {scanner_symbols}")

    bot = TradingBot(config_path=args.config, scanner_symbols=scanner_symbols)

    # Run preflight checks (unless skipped)
    if not args.skip_preflight:
        if not bot.run_preflight():
            logger.error("Exiting due to preflight failure")
            sys.exit(1)
    else:
        logger.warning("Preflight checks SKIPPED (--skip-preflight flag)")

    eastern = pytz.timezone('America/New_York')
    # ... rest of existing code ...
```

Also add `import sys` at the top of bot.py if not already present.

**Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_preflight.py::TestBotIntegration -v`
Expected: PASS

**Step 6: Commit**

```bash
git add bot.py tests/test_preflight.py
git commit -m "feat(bot): integrate preflight checks into startup"
```

---

### Task 11: Create systemd unit files

**Files:**
- Create: `systemd/trading-bot.service`
- Create: `systemd/trading-bot.timer`

**Step 1: Create systemd directory and service file**

```bash
mkdir -p systemd
```

Create `systemd/trading-bot.service`:

```ini
[Unit]
Description=Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=carsonodell
WorkingDirectory=/home/carsonodell/trading-bot
ExecStart=/usr/bin/python3 bot.py
Restart=no
Environment=PYTHONUNBUFFERED=1
# Load environment variables from .env file
EnvironmentFile=/home/carsonodell/trading-bot/.env

[Install]
WantedBy=multi-user.target
```

**Step 2: Create timer file**

Create `systemd/trading-bot.timer`:

```ini
[Unit]
Description=Start Trading Bot before market open

[Timer]
OnCalendar=Mon..Fri 09:25:00 America/New_York
Persistent=false

[Install]
WantedBy=timers.target
```

**Step 3: Create installation script**

Create `systemd/install.sh`:

```bash
#!/bin/bash
# Install trading bot systemd units

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing systemd units..."
sudo cp "$SCRIPT_DIR/trading-bot.service" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/trading-bot.timer" /etc/systemd/system/

echo "Reloading systemd..."
sudo systemctl daemon-reload

echo "Enabling timer..."
sudo systemctl enable trading-bot.timer

echo ""
echo "Installation complete!"
echo ""
echo "Commands:"
echo "  sudo systemctl start trading-bot.timer   # Start the timer"
echo "  sudo systemctl status trading-bot.timer  # Check timer status"
echo "  sudo systemctl list-timers               # List all timers"
echo "  sudo journalctl -u trading-bot           # View bot logs"
```

**Step 4: Make install script executable and commit**

```bash
chmod +x systemd/install.sh
git add systemd/
git commit -m "feat(systemd): add service and timer for auto-start"
```

---

### Task 12: Run full test suite and final commit

**Step 1: Run all preflight tests**

```bash
python3 -m pytest tests/test_preflight.py -v
```

Expected: All tests pass

**Step 2: Run full test suite**

```bash
python3 -m pytest
```

Expected: No regressions

**Step 3: Final commit if any cleanup needed**

```bash
git status
# If any uncommitted changes, commit them
```

---

## Summary

**Files created:**
- `core/preflight.py` - PreflightChecklist class with 7 checks
- `tests/test_preflight.py` - Comprehensive unit tests
- `systemd/trading-bot.service` - Systemd service unit
- `systemd/trading-bot.timer` - Systemd timer for 9:25 AM ET
- `systemd/install.sh` - Installation script

**Files modified:**
- `core/__init__.py` - Export PreflightChecklist
- `bot.py` - Add run_preflight() method and call from main()

**Commits:** 12 atomic commits following TDD pattern
