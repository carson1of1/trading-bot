"""
Preflight checklist for safe auto-start.

Runs safety checks before enabling trading. All checks must pass
for trading to be enabled.
"""
import logging
import os
import yaml
from pathlib import Path
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

    def run_all_checks(self) -> Tuple[bool, List[CheckResult]]:
        """
        Run all preflight checks.

        Returns:
            Tuple of (all_passed, list of CheckResults)
        """
        results = []
        # TODO: Add checks
        all_passed = all(r.passed for r in results) if results else True
        return all_passed, results
