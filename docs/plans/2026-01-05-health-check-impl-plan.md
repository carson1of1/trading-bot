# Health Check System Implementation Plan (ODE-95)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a health check system that verifies ExitManager and live bot components are functioning correctly, running at the start of each trading cycle.

**Architecture:** Single `run_health_check()` method in `TradingBot` class with helper methods for each component check. Returns structured dict, logs summary at INFO level. Does not block trading.

**Tech Stack:** Python 3.10+, pytest, unittest.mock

---

## Task 1: Add Health Check Test File

**Files:**
- Create: `tests/test_health_check.py`

**Step 1: Create test file with basic structure**

```python
"""
Tests for TradingBot health check system (ODE-95).

Tests verify:
1. Health check returns correct structure
2. Broker connection check works
3. ExitManager registration check works
4. Position sync check works
5. Overall status calculation is correct
6. Log output format is correct
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
import pytz

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import TradingBot


class TestHealthCheckStructure:
    """Test health check returns correct structure."""

    @pytest.fixture
    def mock_bot(self):
        """Create a bot with mocked components."""
        with patch('bot.VolatilityScanner') as mock_scanner, \
             patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = ['AAPL']
            mock_scanner.return_value = mock_scanner_instance

            mock_broker_instance = MagicMock()
            mock_account = MagicMock()
            mock_account.cash = 10000.0
            mock_account.portfolio_value = 50000.0
            mock_account.last_equity = 50000.0
            mock_broker_instance.get_account.return_value = mock_account
            mock_broker_instance.get_positions.return_value = []
            mock_broker.return_value = mock_broker_instance

            bot = TradingBot()
            bot.cash = 10000.0
            bot.portfolio_value = 50000.0
            yield bot

    def test_health_check_returns_dict(self, mock_bot):
        """Health check returns a dictionary."""
        result = mock_bot.run_health_check()
        assert isinstance(result, dict)

    def test_health_check_has_required_keys(self, mock_bot):
        """Health check result has required keys."""
        result = mock_bot.run_health_check()
        assert 'timestamp' in result
        assert 'overall_status' in result
        assert 'checks' in result
        assert 'summary' in result

    def test_health_check_summary_structure(self, mock_bot):
        """Health check summary has correct structure."""
        result = mock_bot.run_health_check()
        summary = result['summary']
        assert 'total_checks' in summary
        assert 'passed' in summary
        assert 'failed' in summary
        assert 'info' in summary
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_health_check.py -v`
Expected: FAIL with `AttributeError: 'TradingBot' object has no attribute 'run_health_check'`

**Step 3: Commit test file**

```bash
git add tests/test_health_check.py
git commit -m "test(ODE-95): Add health check structure tests"
```

---

## Task 2: Add run_health_check() Skeleton

**Files:**
- Modify: `bot.py` (add method around line 1470, before `start()`)

**Step 1: Add skeleton method to TradingBot**

Add this method to `TradingBot` class in `bot.py`:

```python
    def run_health_check(self) -> dict:
        """
        Run comprehensive health check on bot systems.

        Verifies:
        - ExitManager: positions registered, state persistence, stops valid
        - Live Bot: broker connected, account synced, positions synced

        Returns:
            dict with timestamp, overall_status, checks, and summary
        """
        from datetime import datetime
        import pytz

        results = {
            'timestamp': datetime.now(pytz.UTC).isoformat(),
            'overall_status': 'HEALTHY',
            'checks': {},
            'summary': {
                'total_checks': 0,
                'passed': 0,
                'failed': 0,
                'info': 0
            }
        }

        return results
```

**Step 2: Run test to verify it passes**

Run: `python3 -m pytest tests/test_health_check.py::TestHealthCheckStructure -v`
Expected: 3 tests PASS

**Step 3: Commit**

```bash
git add bot.py
git commit -m "feat(ODE-95): Add run_health_check() skeleton method"
```

---

## Task 3: Implement Broker Connection Check

**Files:**
- Modify: `tests/test_health_check.py` (add tests)
- Modify: `bot.py` (add `_check_broker_health()` method)

**Step 1: Add broker health test**

Add to `tests/test_health_check.py`:

```python
class TestBrokerHealthCheck:
    """Test broker connection health check."""

    @pytest.fixture
    def mock_bot(self):
        """Create a bot with mocked components."""
        with patch('bot.VolatilityScanner') as mock_scanner, \
             patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = ['AAPL']
            mock_scanner.return_value = mock_scanner_instance

            mock_broker_instance = MagicMock()
            mock_account = MagicMock()
            mock_account.cash = 10000.0
            mock_account.portfolio_value = 50000.0
            mock_account.last_equity = 50000.0
            mock_broker_instance.get_account.return_value = mock_account
            mock_broker_instance.get_positions.return_value = []
            mock_broker.return_value = mock_broker_instance

            bot = TradingBot()
            bot.cash = 10000.0
            bot.portfolio_value = 50000.0
            yield bot

    def test_broker_connected_passes(self, mock_bot):
        """Broker check passes when connection works."""
        result = mock_bot.run_health_check()
        assert result['checks']['broker_connected']['status'] == 'PASS'

    def test_broker_connected_fails_on_exception(self, mock_bot):
        """Broker check fails when get_account raises exception."""
        mock_bot.broker.get_account.side_effect = Exception("Connection failed")
        result = mock_bot.run_health_check()
        assert result['checks']['broker_connected']['status'] == 'FAIL'
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_health_check.py::TestBrokerHealthCheck -v`
Expected: FAIL with `KeyError: 'broker_connected'`

**Step 3: Implement _check_broker_health()**

Add to `TradingBot` class in `bot.py`:

```python
    def _check_broker_health(self) -> dict:
        """Check broker connection is active."""
        try:
            account = self.broker.get_account()
            broker_name = getattr(self.broker, 'get_broker_name', lambda: 'Unknown')()
            return {
                'status': 'PASS',
                'message': f'Connected to {broker_name}'
            }
        except Exception as e:
            return {
                'status': 'FAIL',
                'message': f'Broker connection failed: {e}'
            }
```

**Step 4: Update run_health_check() to call it**

Update `run_health_check()` in `bot.py`:

```python
    def run_health_check(self) -> dict:
        """
        Run comprehensive health check on bot systems.

        Verifies:
        - ExitManager: positions registered, state persistence, stops valid
        - Live Bot: broker connected, account synced, positions synced

        Returns:
            dict with timestamp, overall_status, checks, and summary
        """
        results = {
            'timestamp': datetime.now(pytz.UTC).isoformat(),
            'overall_status': 'HEALTHY',
            'checks': {},
            'summary': {
                'total_checks': 0,
                'passed': 0,
                'failed': 0,
                'info': 0
            }
        }

        # Check broker connection
        results['checks']['broker_connected'] = self._check_broker_health()

        # Calculate summary
        for check_name, check_result in results['checks'].items():
            results['summary']['total_checks'] += 1
            status = check_result.get('status', 'FAIL')
            if status == 'PASS':
                results['summary']['passed'] += 1
            elif status == 'INFO':
                results['summary']['info'] += 1
            else:
                results['summary']['failed'] += 1

        # Determine overall status
        if results['summary']['failed'] >= 3:
            results['overall_status'] = 'UNHEALTHY'
        elif results['summary']['failed'] >= 1:
            results['overall_status'] = 'DEGRADED'

        return results
```

**Step 5: Run tests**

Run: `python3 -m pytest tests/test_health_check.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add bot.py tests/test_health_check.py
git commit -m "feat(ODE-95): Implement broker connection health check"
```

---

## Task 4: Implement Account Sync Check

**Files:**
- Modify: `tests/test_health_check.py` (add tests)
- Modify: `bot.py` (add check to run_health_check)

**Step 1: Add account sync tests**

Add to `tests/test_health_check.py`:

```python
class TestAccountSyncCheck:
    """Test account sync health check."""

    @pytest.fixture
    def mock_bot(self):
        """Create a bot with mocked components."""
        with patch('bot.VolatilityScanner') as mock_scanner, \
             patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = ['AAPL']
            mock_scanner.return_value = mock_scanner_instance

            mock_broker_instance = MagicMock()
            mock_account = MagicMock()
            mock_account.cash = 10000.0
            mock_account.portfolio_value = 50000.0
            mock_account.last_equity = 50000.0
            mock_broker_instance.get_account.return_value = mock_account
            mock_broker_instance.get_positions.return_value = []
            mock_broker.return_value = mock_broker_instance

            bot = TradingBot()
            yield bot

    def test_account_synced_passes_with_values(self, mock_bot):
        """Account sync check passes when cash and portfolio populated."""
        mock_bot.cash = 10000.0
        mock_bot.portfolio_value = 50000.0
        result = mock_bot.run_health_check()
        assert result['checks']['account_synced']['status'] == 'PASS'

    def test_account_synced_fails_with_zero_values(self, mock_bot):
        """Account sync check fails when values are zero."""
        mock_bot.cash = 0.0
        mock_bot.portfolio_value = 0.0
        result = mock_bot.run_health_check()
        assert result['checks']['account_synced']['status'] == 'FAIL'
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_health_check.py::TestAccountSyncCheck -v`
Expected: FAIL with `KeyError: 'account_synced'`

**Step 3: Add account sync check to run_health_check()**

Add to `run_health_check()` in `bot.py` (after broker check):

```python
        # Check account sync
        if self.cash > 0 and self.portfolio_value > 0:
            results['checks']['account_synced'] = {
                'status': 'PASS',
                'message': f'Cash: ${self.cash:,.2f}, Portfolio: ${self.portfolio_value:,.2f}'
            }
        else:
            results['checks']['account_synced'] = {
                'status': 'FAIL',
                'message': f'Account not synced: cash=${self.cash}, portfolio=${self.portfolio_value}'
            }
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_health_check.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add bot.py tests/test_health_check.py
git commit -m "feat(ODE-95): Implement account sync health check"
```

---

## Task 5: Implement Position Sync Check

**Files:**
- Modify: `tests/test_health_check.py` (add tests)
- Modify: `bot.py` (add `_check_position_sync()`)

**Step 1: Add position sync tests**

Add to `tests/test_health_check.py`:

```python
class TestPositionSyncCheck:
    """Test position sync health check."""

    @pytest.fixture
    def mock_bot(self):
        """Create a bot with mocked components."""
        with patch('bot.VolatilityScanner') as mock_scanner, \
             patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = ['AAPL']
            mock_scanner.return_value = mock_scanner_instance

            mock_broker_instance = MagicMock()
            mock_account = MagicMock()
            mock_account.cash = 10000.0
            mock_account.portfolio_value = 50000.0
            mock_account.last_equity = 50000.0
            mock_broker_instance.get_account.return_value = mock_account
            mock_broker_instance.get_positions.return_value = []
            mock_broker.return_value = mock_broker_instance

            bot = TradingBot()
            bot.cash = 10000.0
            bot.portfolio_value = 50000.0
            yield bot

    def test_positions_synced_passes_when_matching(self, mock_bot):
        """Position sync passes when counts match."""
        # Both have 0 positions
        result = mock_bot.run_health_check()
        assert result['checks']['positions_synced']['status'] == 'PASS'

    def test_positions_synced_fails_when_mismatched(self, mock_bot):
        """Position sync fails when counts don't match."""
        mock_bot.open_positions = {'AAPL': {'qty': 100}}
        mock_bot.broker.get_positions.return_value = []  # Broker has 0
        result = mock_bot.run_health_check()
        assert result['checks']['positions_synced']['status'] == 'FAIL'
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_health_check.py::TestPositionSyncCheck -v`
Expected: FAIL with `KeyError: 'positions_synced'`

**Step 3: Add _check_position_sync() method**

Add to `TradingBot` class in `bot.py`:

```python
    def _check_position_sync(self) -> dict:
        """Check positions match between bot and broker."""
        try:
            broker_positions = self.broker.get_positions()
            broker_count = len(broker_positions) if broker_positions else 0
            bot_count = len(self.open_positions)

            if broker_count == bot_count:
                return {
                    'status': 'PASS',
                    'message': f'{bot_count} positions synced'
                }
            else:
                return {
                    'status': 'FAIL',
                    'message': f'Position mismatch: bot has {bot_count}, broker has {broker_count}'
                }
        except Exception as e:
            return {
                'status': 'FAIL',
                'message': f'Position sync check failed: {e}'
            }
```

**Step 4: Add call to run_health_check()**

Add to `run_health_check()` in `bot.py` (after account sync check):

```python
        # Check position sync
        results['checks']['positions_synced'] = self._check_position_sync()
```

**Step 5: Run tests**

Run: `python3 -m pytest tests/test_health_check.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add bot.py tests/test_health_check.py
git commit -m "feat(ODE-95): Implement position sync health check"
```

---

## Task 6: Implement ExitManager Registration Check

**Files:**
- Modify: `tests/test_health_check.py` (add tests)
- Modify: `bot.py` (add `_check_exit_manager_health()`)

**Step 1: Add ExitManager tests**

Add to `tests/test_health_check.py`:

```python
class TestExitManagerCheck:
    """Test ExitManager health check."""

    @pytest.fixture
    def mock_bot(self):
        """Create a bot with mocked components."""
        with patch('bot.VolatilityScanner') as mock_scanner, \
             patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = ['AAPL']
            mock_scanner.return_value = mock_scanner_instance

            mock_broker_instance = MagicMock()
            mock_account = MagicMock()
            mock_account.cash = 10000.0
            mock_account.portfolio_value = 50000.0
            mock_account.last_equity = 50000.0
            mock_broker_instance.get_account.return_value = mock_account
            mock_broker_instance.get_positions.return_value = []
            mock_broker.return_value = mock_broker_instance

            bot = TradingBot()
            bot.cash = 10000.0
            bot.portfolio_value = 50000.0
            yield bot

    def test_exit_manager_passes_when_all_registered(self, mock_bot):
        """ExitManager check passes when all positions registered."""
        mock_bot.open_positions = {'AAPL': {'qty': 100, 'entry_price': 150.0}}
        mock_bot.exit_manager.positions = {'AAPL': MagicMock()}
        result = mock_bot.run_health_check()
        assert result['checks']['positions_registered']['status'] == 'PASS'

    def test_exit_manager_fails_when_missing(self, mock_bot):
        """ExitManager check fails when position not registered."""
        mock_bot.open_positions = {'AAPL': {'qty': 100, 'entry_price': 150.0}}
        mock_bot.exit_manager.positions = {}  # Empty - not registered
        result = mock_bot.run_health_check()
        assert result['checks']['positions_registered']['status'] == 'FAIL'
        assert 'AAPL' in result['checks']['positions_registered']['message']
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_health_check.py::TestExitManagerCheck -v`
Expected: FAIL with `KeyError: 'positions_registered'`

**Step 3: Add _check_exit_manager_health() method**

Add to `TradingBot` class in `bot.py`:

```python
    def _check_exit_manager_health(self) -> dict:
        """Check ExitManager has all positions registered correctly."""
        if not self.use_tiered_exits or not self.exit_manager:
            return {
                'status': 'INFO',
                'message': 'Tiered exits disabled'
            }

        bot_symbols = set(self.open_positions.keys())
        exit_mgr_symbols = set(self.exit_manager.positions.keys())

        missing = bot_symbols - exit_mgr_symbols
        orphaned = exit_mgr_symbols - bot_symbols

        if not missing and not orphaned:
            return {
                'status': 'PASS',
                'message': f'{len(bot_symbols)}/{len(bot_symbols)} positions registered'
            }
        else:
            issues = []
            if missing:
                issues.append(f'missing: {list(missing)}')
            if orphaned:
                issues.append(f'orphaned: {list(orphaned)}')
            return {
                'status': 'FAIL',
                'message': f'ExitManager mismatch - {", ".join(issues)}'
            }
```

**Step 4: Add call to run_health_check()**

Add to `run_health_check()` in `bot.py`:

```python
        # Check ExitManager registration
        results['checks']['positions_registered'] = self._check_exit_manager_health()
```

**Step 5: Run tests**

Run: `python3 -m pytest tests/test_health_check.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add bot.py tests/test_health_check.py
git commit -m "feat(ODE-95): Implement ExitManager registration health check"
```

---

## Task 7: Implement Data Fetcher Check

**Files:**
- Modify: `tests/test_health_check.py` (add tests)
- Modify: `bot.py` (add check)

**Step 1: Add data fetcher tests**

Add to `tests/test_health_check.py`:

```python
class TestDataFetcherCheck:
    """Test data fetcher health check."""

    @pytest.fixture
    def mock_bot(self):
        """Create a bot with mocked components."""
        with patch('bot.VolatilityScanner') as mock_scanner, \
             patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher') as mock_fetcher:

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = ['AAPL']
            mock_scanner.return_value = mock_scanner_instance

            mock_broker_instance = MagicMock()
            mock_account = MagicMock()
            mock_account.cash = 10000.0
            mock_account.portfolio_value = 50000.0
            mock_account.last_equity = 50000.0
            mock_broker_instance.get_account.return_value = mock_account
            mock_broker_instance.get_positions.return_value = []
            mock_broker.return_value = mock_broker_instance

            bot = TradingBot()
            bot.cash = 10000.0
            bot.portfolio_value = 50000.0
            yield bot

    def test_data_fetcher_passes_with_data(self, mock_bot):
        """Data fetcher check passes when data returned."""
        import pandas as pd
        mock_bot.data_fetcher.get_historical_data_range.return_value = pd.DataFrame({
            'open': [100], 'high': [101], 'low': [99], 'close': [100.5], 'volume': [1000000]
        })
        result = mock_bot.run_health_check()
        assert result['checks']['data_fetcher_valid']['status'] == 'PASS'

    def test_data_fetcher_fails_when_empty(self, mock_bot):
        """Data fetcher check fails when no data returned."""
        mock_bot.data_fetcher.get_historical_data_range.return_value = None
        result = mock_bot.run_health_check()
        assert result['checks']['data_fetcher_valid']['status'] == 'FAIL'
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_health_check.py::TestDataFetcherCheck -v`
Expected: FAIL with `KeyError: 'data_fetcher_valid'`

**Step 3: Add data fetcher check to run_health_check()**

Add to `run_health_check()` in `bot.py`:

```python
        # Check data fetcher
        try:
            test_data = self.fetch_data('SPY', bars=10)
            if test_data is not None and len(test_data) > 0:
                results['checks']['data_fetcher_valid'] = {
                    'status': 'PASS',
                    'message': f'SPY returned {len(test_data)} bars'
                }
            else:
                results['checks']['data_fetcher_valid'] = {
                    'status': 'FAIL',
                    'message': 'Data fetcher returned empty data for SPY'
                }
        except Exception as e:
            results['checks']['data_fetcher_valid'] = {
                'status': 'FAIL',
                'message': f'Data fetcher error: {e}'
            }
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_health_check.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add bot.py tests/test_health_check.py
git commit -m "feat(ODE-95): Implement data fetcher health check"
```

---

## Task 8: Implement Strategy Manager Check

**Files:**
- Modify: `tests/test_health_check.py` (add tests)
- Modify: `bot.py` (add check)

**Step 1: Add strategy manager tests**

Add to `tests/test_health_check.py`:

```python
class TestStrategyManagerCheck:
    """Test strategy manager health check."""

    @pytest.fixture
    def mock_bot(self):
        """Create a bot with mocked components."""
        with patch('bot.VolatilityScanner') as mock_scanner, \
             patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = ['AAPL']
            mock_scanner.return_value = mock_scanner_instance

            mock_broker_instance = MagicMock()
            mock_account = MagicMock()
            mock_account.cash = 10000.0
            mock_account.portfolio_value = 50000.0
            mock_account.last_equity = 50000.0
            mock_broker_instance.get_account.return_value = mock_account
            mock_broker_instance.get_positions.return_value = []
            mock_broker.return_value = mock_broker_instance

            bot = TradingBot()
            bot.cash = 10000.0
            bot.portfolio_value = 50000.0
            yield bot

    def test_strategy_manager_passes_with_strategies(self, mock_bot):
        """Strategy manager check passes when strategies loaded."""
        mock_bot.strategy_manager.strategies = [MagicMock(), MagicMock()]
        result = mock_bot.run_health_check()
        assert result['checks']['strategy_manager_ready']['status'] == 'PASS'

    def test_strategy_manager_fails_with_no_strategies(self, mock_bot):
        """Strategy manager check fails when no strategies."""
        mock_bot.strategy_manager.strategies = []
        result = mock_bot.run_health_check()
        assert result['checks']['strategy_manager_ready']['status'] == 'FAIL'
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_health_check.py::TestStrategyManagerCheck -v`
Expected: FAIL with `KeyError: 'strategy_manager_ready'`

**Step 3: Add strategy manager check to run_health_check()**

Add to `run_health_check()` in `bot.py`:

```python
        # Check strategy manager
        if self.strategy_manager and len(self.strategy_manager.strategies) > 0:
            strategy_names = [s.__class__.__name__ for s in self.strategy_manager.strategies]
            results['checks']['strategy_manager_ready'] = {
                'status': 'PASS',
                'message': f'{len(self.strategy_manager.strategies)} strategies: {strategy_names}'
            }
        else:
            results['checks']['strategy_manager_ready'] = {
                'status': 'FAIL',
                'message': 'No strategies loaded'
            }
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_health_check.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add bot.py tests/test_health_check.py
git commit -m "feat(ODE-95): Implement strategy manager health check"
```

---

## Task 9: Implement Kill Switch and Drawdown Guard Status

**Files:**
- Modify: `tests/test_health_check.py` (add tests)
- Modify: `bot.py` (add INFO-level status checks)

**Step 1: Add status check tests**

Add to `tests/test_health_check.py`:

```python
class TestStatusChecks:
    """Test INFO-level status checks (kill switch, drawdown guard)."""

    @pytest.fixture
    def mock_bot(self):
        """Create a bot with mocked components."""
        with patch('bot.VolatilityScanner') as mock_scanner, \
             patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = ['AAPL']
            mock_scanner.return_value = mock_scanner_instance

            mock_broker_instance = MagicMock()
            mock_account = MagicMock()
            mock_account.cash = 10000.0
            mock_account.portfolio_value = 50000.0
            mock_account.last_equity = 50000.0
            mock_broker_instance.get_account.return_value = mock_account
            mock_broker_instance.get_positions.return_value = []
            mock_broker.return_value = mock_broker_instance

            bot = TradingBot()
            bot.cash = 10000.0
            bot.portfolio_value = 50000.0
            yield bot

    def test_kill_switch_status_is_info(self, mock_bot):
        """Kill switch status is INFO, not PASS/FAIL."""
        result = mock_bot.run_health_check()
        assert result['checks']['kill_switch_status']['status'] == 'INFO'

    def test_kill_switch_triggered_reported(self, mock_bot):
        """Kill switch triggered state is reported."""
        mock_bot.kill_switch_triggered = True
        result = mock_bot.run_health_check()
        assert 'TRIGGERED' in result['checks']['kill_switch_status']['message']

    def test_drawdown_guard_status_is_info(self, mock_bot):
        """Drawdown guard status is INFO."""
        result = mock_bot.run_health_check()
        assert result['checks']['drawdown_guard_status']['status'] == 'INFO'
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_health_check.py::TestStatusChecks -v`
Expected: FAIL with `KeyError: 'kill_switch_status'`

**Step 3: Add status checks to run_health_check()**

Add to `run_health_check()` in `bot.py`:

```python
        # Kill switch status (INFO only)
        results['checks']['kill_switch_status'] = {
            'status': 'INFO',
            'message': 'TRIGGERED' if self.kill_switch_triggered else 'Not triggered'
        }

        # Drawdown guard status (INFO only)
        if self.drawdown_guard.enabled:
            status = self.drawdown_guard.get_status()
            results['checks']['drawdown_guard_status'] = {
                'status': 'INFO',
                'message': f"Tier: {status.get('tier', 'NORMAL')}, Entries: {'allowed' if status.get('entries_allowed', True) else 'BLOCKED'}"
            }
        else:
            results['checks']['drawdown_guard_status'] = {
                'status': 'INFO',
                'message': 'Disabled'
            }
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_health_check.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add bot.py tests/test_health_check.py
git commit -m "feat(ODE-95): Implement kill switch and drawdown guard status checks"
```

---

## Task 10: Add Health Check Logging

**Files:**
- Modify: `tests/test_health_check.py` (add logging test)
- Modify: `bot.py` (add logging to run_health_check)

**Step 1: Add logging test**

Add to `tests/test_health_check.py`:

```python
class TestHealthCheckLogging:
    """Test health check logging output."""

    @pytest.fixture
    def mock_bot(self):
        """Create a bot with mocked components."""
        with patch('bot.VolatilityScanner') as mock_scanner, \
             patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = ['AAPL']
            mock_scanner.return_value = mock_scanner_instance

            mock_broker_instance = MagicMock()
            mock_account = MagicMock()
            mock_account.cash = 10000.0
            mock_account.portfolio_value = 50000.0
            mock_account.last_equity = 50000.0
            mock_broker_instance.get_account.return_value = mock_account
            mock_broker_instance.get_positions.return_value = []
            mock_broker.return_value = mock_broker_instance

            bot = TradingBot()
            bot.cash = 10000.0
            bot.portfolio_value = 50000.0
            yield bot

    def test_health_check_logs_summary(self, mock_bot, caplog):
        """Health check logs summary at INFO level."""
        import logging
        with caplog.at_level(logging.INFO):
            mock_bot.run_health_check()

        assert any('HEALTH_CHECK' in record.message for record in caplog.records)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_health_check.py::TestHealthCheckLogging -v`
Expected: FAIL (no HEALTH_CHECK in logs)

**Step 3: Add logging to run_health_check()**

Add at the end of `run_health_check()` in `bot.py` (before the return statement):

```python
        # Log summary
        summary = results['summary']
        status = results['overall_status']
        positions_count = len(self.open_positions)
        exit_mgr_count = len(self.exit_manager.positions) if self.exit_manager else 0
        kill_switch = 'ON' if self.kill_switch_triggered else 'OFF'

        logger.info(
            f"HEALTH_CHECK | {status} | "
            f"{summary['passed']}/{summary['total_checks']} PASS | "
            f"{summary['failed']} FAIL | {summary['info']} INFO | "
            f"Positions: {positions_count} | ExitMgr: {exit_mgr_count} | "
            f"KillSwitch: {kill_switch}"
        )

        # Log individual failures at WARNING level
        for check_name, check_result in results['checks'].items():
            if check_result.get('status') == 'FAIL':
                logger.warning(f"HEALTH_CHECK | {check_name} FAIL: {check_result.get('message', 'Unknown')}")
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_health_check.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add bot.py tests/test_health_check.py
git commit -m "feat(ODE-95): Add health check logging"
```

---

## Task 11: Integrate Health Check into Trading Cycle

**Files:**
- Modify: `tests/test_health_check.py` (add integration test)
- Modify: `bot.py` (call run_health_check in run_trading_cycle)

**Step 1: Add integration test**

Add to `tests/test_health_check.py`:

```python
class TestHealthCheckIntegration:
    """Test health check integration with trading cycle."""

    @pytest.fixture
    def mock_bot(self):
        """Create a bot with mocked components."""
        with patch('bot.VolatilityScanner') as mock_scanner, \
             patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = ['AAPL']
            mock_scanner.return_value = mock_scanner_instance

            mock_broker_instance = MagicMock()
            mock_account = MagicMock()
            mock_account.cash = 10000.0
            mock_account.portfolio_value = 50000.0
            mock_account.last_equity = 50000.0
            mock_broker_instance.get_account.return_value = mock_account
            mock_broker_instance.get_positions.return_value = []
            mock_broker.return_value = mock_broker_instance

            bot = TradingBot()
            bot.cash = 10000.0
            bot.portfolio_value = 50000.0
            yield bot

    def test_trading_cycle_runs_health_check(self, mock_bot, caplog):
        """Trading cycle calls run_health_check."""
        import logging
        with caplog.at_level(logging.INFO):
            # Mock fetch_data to avoid actual data fetching
            mock_bot.fetch_data = MagicMock(return_value=None)
            mock_bot.run_trading_cycle()

        # Health check should have run
        assert any('HEALTH_CHECK' in record.message for record in caplog.records)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_health_check.py::TestHealthCheckIntegration -v`
Expected: FAIL (no HEALTH_CHECK in trading cycle logs)

**Step 3: Add health check call to run_trading_cycle()**

In `bot.py`, find `run_trading_cycle()` and add after the reconcile call (around line 1213):

```python
            # 0. Reconcile broker state BEFORE syncing (detect divergence)
            self._reconcile_broker_state()

            # 0.5 Run health check (ODE-95)
            self.run_health_check()

            # 1. Sync state
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_health_check.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add bot.py tests/test_health_check.py
git commit -m "feat(ODE-95): Integrate health check into trading cycle"
```

---

## Task 12: Add Hard Stop Validation Check

**Files:**
- Modify: `tests/test_health_check.py` (add tests)
- Modify: `bot.py` (add check)

**Step 1: Add hard stop validation tests**

Add to `tests/test_health_check.py`:

```python
class TestHardStopValidation:
    """Test hard stop validation health check."""

    @pytest.fixture
    def mock_bot(self):
        """Create a bot with mocked components."""
        with patch('bot.VolatilityScanner') as mock_scanner, \
             patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):

            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = ['AAPL']
            mock_scanner.return_value = mock_scanner_instance

            mock_broker_instance = MagicMock()
            mock_account = MagicMock()
            mock_account.cash = 10000.0
            mock_account.portfolio_value = 50000.0
            mock_account.last_equity = 50000.0
            mock_broker_instance.get_account.return_value = mock_account
            mock_broker_instance.get_positions.return_value = []
            mock_broker.return_value = mock_broker_instance

            bot = TradingBot()
            bot.cash = 10000.0
            bot.portfolio_value = 50000.0
            yield bot

    def test_hard_stops_valid_with_correct_calculation(self, mock_bot):
        """Hard stops check passes when calculations correct."""
        from core.risk import PositionExitState
        from datetime import datetime
        import pytz

        # Register a position
        mock_bot.open_positions = {'AAPL': {'qty': 100, 'entry_price': 100.0, 'direction': 'LONG'}}
        state = PositionExitState(
            symbol='AAPL',
            entry_price=100.0,
            entry_time=datetime.now(pytz.UTC),
            quantity=100,
            hard_stop_pct=0.02  # 2%
        )
        mock_bot.exit_manager.positions = {'AAPL': state}

        result = mock_bot.run_health_check()
        assert result['checks']['hard_stops_valid']['status'] == 'PASS'
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_health_check.py::TestHardStopValidation -v`
Expected: FAIL with `KeyError: 'hard_stops_valid'`

**Step 3: Add hard stop validation to run_health_check()**

Add to `run_health_check()` in `bot.py`:

```python
        # Check hard stops are valid
        if self.use_tiered_exits and self.exit_manager and self.open_positions:
            invalid_stops = []
            for symbol, pos in self.open_positions.items():
                if symbol in self.exit_manager.positions:
                    state = self.exit_manager.positions[symbol]
                    entry = pos.get('entry_price', 0)
                    direction = pos.get('direction', 'LONG')

                    if entry > 0:
                        # Calculate expected hard stop
                        if direction == 'LONG':
                            expected_stop = entry * (1 - state.hard_stop_pct)
                        else:
                            expected_stop = entry * (1 + state.hard_stop_pct)

                        # Hard stop should be reasonable (within 10% of entry)
                        stop_distance = abs(expected_stop - entry) / entry
                        if stop_distance > 0.10:  # More than 10% stop is suspicious
                            invalid_stops.append(f"{symbol}: {stop_distance*100:.1f}% stop")

            if not invalid_stops:
                results['checks']['hard_stops_valid'] = {
                    'status': 'PASS',
                    'message': f'All {len(self.open_positions)} stops valid'
                }
            else:
                results['checks']['hard_stops_valid'] = {
                    'status': 'FAIL',
                    'message': f'Invalid stops: {invalid_stops}'
                }
        else:
            results['checks']['hard_stops_valid'] = {
                'status': 'INFO',
                'message': 'No positions to validate'
            }
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_health_check.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add bot.py tests/test_health_check.py
git commit -m "feat(ODE-95): Add hard stop validation health check"
```

---

## Task 13: Run Full Test Suite

**Step 1: Run all health check tests**

Run: `python3 -m pytest tests/test_health_check.py -v`
Expected: All tests PASS

**Step 2: Run all tests to ensure no regressions**

Run: `python3 -m pytest tests/ -v --tb=short`
Expected: All tests PASS (or pre-existing failures only)

**Step 3: Final commit if any cleanup needed**

---

## Task 14: Update Design Document with Final Implementation

**Files:**
- Modify: `docs/plans/2026-01-05-health-check-system-design.md`

**Step 1: Update design doc with actual implementation**

Add section to the design document with final check list and any deviations from original plan.

**Step 2: Commit**

```bash
git add docs/plans/2026-01-05-health-check-system-design.md
git commit -m "docs(ODE-95): Update design document with final implementation"
```

---

## Summary

**Total Tasks:** 14
**Files Modified:**
- `bot.py` (add `run_health_check()`, `_check_broker_health()`, `_check_position_sync()`, `_check_exit_manager_health()`, integration into `run_trading_cycle()`)
- `tests/test_health_check.py` (new file with all health check tests)
- `docs/plans/2026-01-05-health-check-system-design.md` (update)

**Health Checks Implemented:**
1. `broker_connected` - Broker connection active
2. `account_synced` - Cash/portfolio populated
3. `positions_synced` - Positions match broker
4. `positions_registered` - All positions in ExitManager
5. `data_fetcher_valid` - Data fetcher returns data
6. `strategy_manager_ready` - Strategies initialized
7. `kill_switch_status` - Reports state (INFO)
8. `drawdown_guard_status` - Reports tier (INFO)
9. `hard_stops_valid` - Hard stops calculated correctly
