"""
Scanner Lookahead Audit

Detects and reports any instances where the scanner uses data from
the trade date or later (look-ahead bias).

Rule: scanner_lookup_date < trade_date ALWAYS
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import json


logger = logging.getLogger(__name__)


@dataclass
class LookaheadViolation:
    """Record of a lookahead violation."""
    scanner_date: str
    trade_date: str
    symbol: str
    description: str = ""


class ScannerLookaheadAudit:
    """
    Audit scanner for lookahead bias.

    Records all scanner lookups and validates that lookup dates
    are strictly before trade dates.
    """

    def __init__(self):
        self.lookups: List[Dict] = []
        self.violations: List[LookaheadViolation] = []

    def record_lookup(
        self,
        scanner_date: str,
        trade_date: str,
        symbol: str,
        description: str = ""
    ) -> bool:
        """
        Record a scanner lookup and check for violations.

        Args:
            scanner_date: Date used for scanner calculation (YYYY-MM-DD)
            trade_date: Date of the trade (YYYY-MM-DD)
            symbol: Symbol being looked up
            description: Optional description

        Returns:
            True if valid (no violation), False if violation detected
        """
        # Parse dates
        scanner_dt = datetime.strptime(scanner_date, '%Y-%m-%d')
        trade_dt = datetime.strptime(trade_date, '%Y-%m-%d')

        # Record the lookup
        self.lookups.append({
            'scanner_date': scanner_date,
            'trade_date': trade_date,
            'symbol': symbol,
            'description': description,
        })

        # Check for violation: scanner date must be BEFORE trade date
        if scanner_dt >= trade_dt:
            violation = LookaheadViolation(
                scanner_date=scanner_date,
                trade_date=trade_date,
                symbol=symbol,
                description=f"Scanner used {scanner_date} data for {trade_date} trade"
            )
            self.violations.append(violation)
            logger.warning(f"LOOKAHEAD VIOLATION: {violation.description}")
            return False

        return True

    @property
    def violation_count(self) -> int:
        """Return number of violations detected."""
        return len(self.violations)

    def is_clean(self) -> bool:
        """Return True if no violations detected."""
        return self.violation_count == 0

    def assert_clean(self) -> None:
        """Raise AssertionError if any violations detected."""
        if not self.is_clean():
            msg = f"Scanner lookahead audit FAILED: {self.violation_count} violations detected"
            for v in self.violations[:5]:  # Show first 5
                msg += f"\n  - {v.symbol}: scanner={v.scanner_date}, trade={v.trade_date}"
            if len(self.violations) > 5:
                msg += f"\n  ... and {len(self.violations) - 5} more"
            raise AssertionError(msg)

    def get_report(self) -> Dict:
        """Generate audit report."""
        return {
            'total_lookups': len(self.lookups),
            'violations': self.violation_count,
            'status': 'CLEAN' if self.is_clean() else 'FAILED',
            'violation_details': [
                {
                    'symbol': v.symbol,
                    'scanner_date': v.scanner_date,
                    'trade_date': v.trade_date,
                }
                for v in self.violations
            ]
        }

    def save_report(self, path: str = 'logs/scanner_lookahead_audit.json') -> None:
        """Save audit report to file."""
        output_path = Path(__file__).parent.parent / path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.get_report(), f, indent=2)

        logger.info(f"Scanner audit report saved to {output_path}")


# Global audit instance for use during backtests
_global_audit: Optional[ScannerLookaheadAudit] = None


def get_audit() -> ScannerLookaheadAudit:
    """Get or create global audit instance."""
    global _global_audit
    if _global_audit is None:
        _global_audit = ScannerLookaheadAudit()
    return _global_audit


def reset_audit() -> ScannerLookaheadAudit:
    """Reset and return new audit instance."""
    global _global_audit
    _global_audit = ScannerLookaheadAudit()
    return _global_audit


def record_scanner_lookup(
    scanner_date: str,
    trade_date: str,
    symbol: str
) -> bool:
    """Convenience function to record lookup in global audit."""
    return get_audit().record_lookup(scanner_date, trade_date, symbol)
