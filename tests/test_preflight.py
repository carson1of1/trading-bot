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
