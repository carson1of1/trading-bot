"""Tests for API bot management and PID tracking.

Tests for the Jan 2026 fixes:
- PID file tracking to prevent duplicate bots
- Process detection on status check
- Actual process termination on stop
- Start prevention when bot already running

These tests verify the fixes for the bug where:
1. Multiple bot processes could be started
2. API restart caused status to reset to "stopped"
3. Stop endpoint didn't actually kill the process
"""
import pytest
import os
import sys
import signal
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip entire module if fastapi is not installed (CI environment)
pytest.importorskip("fastapi", reason="fastapi not installed")


class TestPIDFileOperations:
    """Test PID file read/write/clear operations."""

    def test_write_and_read_pid(self, tmp_path):
        """Can write and read PID from file."""
        from api.main import _write_bot_pid, _read_bot_pid, BOT_PID_FILE

        # Temporarily override BOT_PID_FILE
        test_pid_file = tmp_path / "test_bot.pid"

        with patch('api.main.BOT_PID_FILE', test_pid_file):
            # Import fresh to get patched version
            from api import main
            main.BOT_PID_FILE = test_pid_file

            # Write PID
            main._write_bot_pid(12345)
            assert test_pid_file.exists()

            # Read PID
            pid = main._read_bot_pid()
            assert pid == 12345

    def test_read_nonexistent_pid_file(self, tmp_path):
        """Reading nonexistent PID file returns None."""
        test_pid_file = tmp_path / "nonexistent.pid"

        with patch('api.main.BOT_PID_FILE', test_pid_file):
            from api import main
            main.BOT_PID_FILE = test_pid_file

            pid = main._read_bot_pid()
            assert pid is None

    def test_clear_pid_file(self, tmp_path):
        """Can clear PID file."""
        test_pid_file = tmp_path / "test_bot.pid"
        test_pid_file.write_text("12345")

        with patch('api.main.BOT_PID_FILE', test_pid_file):
            from api import main
            main.BOT_PID_FILE = test_pid_file

            main._clear_bot_pid()
            assert not test_pid_file.exists()

    def test_clear_nonexistent_pid_file_no_error(self, tmp_path):
        """Clearing nonexistent PID file doesn't raise error."""
        test_pid_file = tmp_path / "nonexistent.pid"

        with patch('api.main.BOT_PID_FILE', test_pid_file):
            from api import main
            main.BOT_PID_FILE = test_pid_file

            # Should not raise
            main._clear_bot_pid()

    def test_read_invalid_pid_file(self, tmp_path):
        """Reading invalid PID file returns None."""
        test_pid_file = tmp_path / "invalid.pid"
        test_pid_file.write_text("not_a_number")

        with patch('api.main.BOT_PID_FILE', test_pid_file):
            from api import main
            main.BOT_PID_FILE = test_pid_file

            pid = main._read_bot_pid()
            assert pid is None


class TestProcessDetection:
    """Test process detection logic."""

    def test_is_bot_running_with_valid_pid(self, tmp_path):
        """Detects running process correctly."""
        test_pid_file = tmp_path / "test_bot.pid"

        # Use current process PID (known to be running)
        current_pid = os.getpid()
        test_pid_file.write_text(str(current_pid))

        with patch('api.main.BOT_PID_FILE', test_pid_file):
            from api import main
            main.BOT_PID_FILE = test_pid_file

            assert main._is_bot_running() is True

    def test_is_bot_running_with_invalid_pid(self, tmp_path):
        """Detects non-running process correctly."""
        test_pid_file = tmp_path / "test_bot.pid"

        # Use a PID that definitely doesn't exist
        # PID 999999 is almost certainly not running
        test_pid_file.write_text("999999")

        with patch('api.main.BOT_PID_FILE', test_pid_file):
            from api import main
            main.BOT_PID_FILE = test_pid_file

            assert main._is_bot_running() is False
            # Should also clean up stale PID file
            assert not test_pid_file.exists()

    def test_is_bot_running_no_pid_file(self, tmp_path):
        """Returns False when no PID file exists."""
        test_pid_file = tmp_path / "nonexistent.pid"

        with patch('api.main.BOT_PID_FILE', test_pid_file):
            from api import main
            main.BOT_PID_FILE = test_pid_file

            assert main._is_bot_running() is False


class TestKillBotProcess:
    """Test process termination logic."""

    def test_kill_nonexistent_process(self, tmp_path):
        """Killing nonexistent process clears PID file."""
        test_pid_file = tmp_path / "test_bot.pid"
        test_pid_file.write_text("999999")  # Non-existent PID

        with patch('api.main.BOT_PID_FILE', test_pid_file):
            from api import main
            main.BOT_PID_FILE = test_pid_file

            result = main._kill_bot_process()
            assert result is False
            assert not test_pid_file.exists()

    def test_kill_no_pid_file(self, tmp_path):
        """Killing with no PID file returns False."""
        test_pid_file = tmp_path / "nonexistent.pid"

        with patch('api.main.BOT_PID_FILE', test_pid_file):
            from api import main
            main.BOT_PID_FILE = test_pid_file

            result = main._kill_bot_process()
            assert result is False

    def test_kill_actual_subprocess(self, tmp_path):
        """Can kill an actual subprocess."""
        test_pid_file = tmp_path / "test_bot.pid"

        # Start a simple subprocess that sleeps
        proc = subprocess.Popen(
            ["python3", "-c", "import time; time.sleep(60)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        test_pid_file.write_text(str(proc.pid))

        with patch('api.main.BOT_PID_FILE', test_pid_file):
            from api import main
            main.BOT_PID_FILE = test_pid_file

            result = main._kill_bot_process()
            assert result is True
            assert not test_pid_file.exists()

            # Verify process is actually dead
            time.sleep(0.2)
            assert proc.poll() is not None  # Process has exited


class TestBotStatusEndpoint:
    """Test /api/bot/status endpoint with process detection."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create test client with mocked PID file."""
        test_pid_file = tmp_path / "test_bot.pid"

        with patch('api.main.BOT_PID_FILE', test_pid_file):
            from api import main
            main.BOT_PID_FILE = test_pid_file

            # Reset bot state
            main._bot_state["status"] = "stopped"
            main._bot_state["last_action"] = None

            from fastapi.testclient import TestClient
            yield TestClient(main.app), test_pid_file, main

    def test_status_detects_running_process(self, client):
        """Status endpoint detects running process even if state says stopped."""
        test_client, test_pid_file, main_module = client

        # Write current PID (so process exists)
        test_pid_file.write_text(str(os.getpid()))
        main_module._bot_state["status"] = "stopped"

        response = test_client.get("/api/bot/status")
        assert response.status_code == 200
        data = response.json()

        # Should detect running and update status
        assert data["status"] == "running"
        assert "recovered after API restart" in data["last_action"]

    def test_status_detects_stopped_process(self, client):
        """Status endpoint detects dead process even if state says running."""
        test_client, test_pid_file, main_module = client

        # Write non-existent PID
        test_pid_file.write_text("999999")
        main_module._bot_state["status"] = "running"

        response = test_client.get("/api/bot/status")
        assert response.status_code == 200
        data = response.json()

        # Should detect stopped and update status
        assert data["status"] == "stopped"
        assert "process terminated" in data["last_action"]


class TestBotStartEndpoint:
    """Test /api/bot/start endpoint prevents duplicates."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create test client with mocked dependencies."""
        test_pid_file = tmp_path / "test_bot.pid"

        with patch('api.main.BOT_PID_FILE', test_pid_file):
            from api import main
            main.BOT_PID_FILE = test_pid_file
            main._bot_state["status"] = "stopped"

            from fastapi.testclient import TestClient
            yield TestClient(main.app), test_pid_file, main

    @patch('api.main.is_market_open')
    def test_start_blocked_when_already_running(self, mock_market, client):
        """Start returns 409 when bot is already running."""
        test_client, test_pid_file, main_module = client

        # Write current PID (so process appears running)
        test_pid_file.write_text(str(os.getpid()))

        mock_market.return_value = True

        response = test_client.post("/api/bot/start")
        assert response.status_code == 409
        data = response.json()
        assert data["detail"]["reason"] == "already_running"


class TestBotStopEndpoint:
    """Test /api/bot/stop endpoint actually kills process."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create test client with mocked PID file."""
        test_pid_file = tmp_path / "test_bot.pid"

        with patch('api.main.BOT_PID_FILE', test_pid_file):
            from api import main
            main.BOT_PID_FILE = test_pid_file
            main._bot_state["status"] = "running"

            from fastapi.testclient import TestClient
            yield TestClient(main.app), test_pid_file, main

    def test_stop_kills_subprocess(self, client):
        """Stop endpoint actually kills the subprocess."""
        test_client, test_pid_file, main_module = client

        # Start a subprocess to kill
        proc = subprocess.Popen(
            ["python3", "-c", "import time; time.sleep(60)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        test_pid_file.write_text(str(proc.pid))

        response = test_client.post("/api/bot/stop")
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["killed_pid"] == proc.pid

        # Verify process is dead
        time.sleep(0.2)
        assert proc.poll() is not None

    def test_stop_when_not_running(self, client):
        """Stop returns success even if bot wasn't running."""
        test_client, test_pid_file, main_module = client

        # No PID file exists
        if test_pid_file.exists():
            test_pid_file.unlink()

        response = test_client.post("/api/bot/stop")
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "was not running" in data.get("message", "")
