"""Tests for preflight checklist."""
import os
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock


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
