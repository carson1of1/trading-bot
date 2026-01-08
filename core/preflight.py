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
