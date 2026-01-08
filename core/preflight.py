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

    def check_market_status(self) -> CheckResult:
        """Check that market is open or opens within 10 minutes."""
        from core.market_hours import MarketHours

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
                # WARNING only - bot.py syncs ALL positions regardless of watchlist
                # Don't fail preflight for positions from previous day's scanner
                return CheckResult(
                    name="positions_accounted",
                    passed=True,
                    message=f"WARNING: Positions not in watchlist (will still be managed): {', '.join(orphaned)}"
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

    def check_stop_config_alignment(self) -> CheckResult:
        """Verify broker stop_loss_pct matches software tier_0_hard_stop."""
        risk_config = self.config.get('risk_management', {})
        exit_config = self.config.get('exit_manager', {})

        broker_stop = risk_config.get('stop_loss_pct', 5.0)  # Percentage (e.g., 5.0)
        software_stop = abs(exit_config.get('tier_0_hard_stop', -0.05)) * 100  # Convert to percentage

        # Allow small floating point tolerance
        if abs(broker_stop - software_stop) > 0.01:
            return CheckResult(
                name="stop_config_alignment",
                passed=False,
                message=f"MISMATCH: Broker stop ({broker_stop}%) != Software stop ({software_stop}%). Fix config.yaml."
            )

        return CheckResult(
            name="stop_config_alignment",
            passed=True,
            message=f"Stop configs aligned at {broker_stop}%"
        )

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
            ("Stop config alignment", self.check_stop_config_alignment),
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
