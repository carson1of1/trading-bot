"""Tests for scanner lookahead audit."""

import pytest
from validation.scanner_audit import (
    ScannerLookaheadAudit,
    LookaheadViolation,
    get_audit,
    reset_audit,
    record_scanner_lookup,
)


class TestScannerLookaheadAudit:
    """Test scanner lookahead detection."""

    def test_no_violations_returns_zero(self):
        """Test clean audit returns zero violations."""
        audit = ScannerLookaheadAudit()

        # Record valid lookups (scanner date < trade date)
        audit.record_lookup('2024-01-04', '2024-01-05', 'AAPL')  # OK
        audit.record_lookup('2024-01-05', '2024-01-08', 'NVDA')  # OK

        assert audit.violation_count == 0
        assert audit.is_clean()

    def test_violation_detected(self):
        """Test lookahead violation is detected."""
        audit = ScannerLookaheadAudit()

        # Record invalid lookup (scanner date >= trade date)
        audit.record_lookup('2024-01-05', '2024-01-05', 'AAPL')  # VIOLATION

        assert audit.violation_count == 1
        assert not audit.is_clean()

    def test_assert_clean_raises_on_violation(self):
        """Test assert_clean raises exception on violations."""
        audit = ScannerLookaheadAudit()
        audit.record_lookup('2024-01-05', '2024-01-05', 'AAPL')

        with pytest.raises(AssertionError, match="lookahead"):
            audit.assert_clean()

    def test_assert_clean_passes_when_clean(self):
        """Test assert_clean passes when no violations."""
        audit = ScannerLookaheadAudit()
        audit.record_lookup('2024-01-04', '2024-01-05', 'AAPL')

        # Should not raise
        audit.assert_clean()

    def test_future_scanner_date_violation(self):
        """Test scanner date in future is detected."""
        audit = ScannerLookaheadAudit()

        # Scanner date is after trade date
        result = audit.record_lookup('2024-01-10', '2024-01-05', 'AAPL')

        assert result is False
        assert audit.violation_count == 1

    def test_record_lookup_returns_true_for_valid(self):
        """Test record_lookup returns True for valid lookups."""
        audit = ScannerLookaheadAudit()

        result = audit.record_lookup('2024-01-04', '2024-01-05', 'AAPL')

        assert result is True

    def test_record_lookup_returns_false_for_violation(self):
        """Test record_lookup returns False for violations."""
        audit = ScannerLookaheadAudit()

        result = audit.record_lookup('2024-01-05', '2024-01-05', 'AAPL')

        assert result is False

    def test_multiple_violations_tracked(self):
        """Test multiple violations are all tracked."""
        audit = ScannerLookaheadAudit()

        audit.record_lookup('2024-01-05', '2024-01-05', 'AAPL')
        audit.record_lookup('2024-01-06', '2024-01-05', 'NVDA')
        audit.record_lookup('2024-01-07', '2024-01-05', 'MSFT')

        assert audit.violation_count == 3

    def test_get_report(self):
        """Test report generation."""
        audit = ScannerLookaheadAudit()
        audit.record_lookup('2024-01-04', '2024-01-05', 'AAPL')
        audit.record_lookup('2024-01-05', '2024-01-05', 'NVDA')  # violation

        report = audit.get_report()

        assert report['total_lookups'] == 2
        assert report['violations'] == 1
        assert report['status'] == 'FAILED'
        assert len(report['violation_details']) == 1

    def test_clean_report_status(self):
        """Test clean audit has correct status."""
        audit = ScannerLookaheadAudit()
        audit.record_lookup('2024-01-04', '2024-01-05', 'AAPL')

        report = audit.get_report()

        assert report['status'] == 'CLEAN'
        assert len(report['violation_details']) == 0


class TestLookaheadViolation:
    """Test LookaheadViolation dataclass."""

    def test_violation_creation(self):
        """Test violation can be created."""
        violation = LookaheadViolation(
            scanner_date='2024-01-05',
            trade_date='2024-01-05',
            symbol='AAPL',
            description='Test violation'
        )

        assert violation.scanner_date == '2024-01-05'
        assert violation.symbol == 'AAPL'


class TestGlobalAudit:
    """Test global audit functions."""

    def test_get_audit_creates_instance(self):
        """Test get_audit creates instance if none exists."""
        reset_audit()  # Start fresh
        audit = get_audit()

        assert audit is not None
        assert isinstance(audit, ScannerLookaheadAudit)

    def test_get_audit_returns_same_instance(self):
        """Test get_audit returns same instance."""
        reset_audit()
        audit1 = get_audit()
        audit2 = get_audit()

        assert audit1 is audit2

    def test_reset_audit_creates_new_instance(self):
        """Test reset_audit creates new instance."""
        reset_audit()
        audit1 = get_audit()
        audit1.record_lookup('2024-01-05', '2024-01-05', 'AAPL')

        audit2 = reset_audit()

        assert audit2.violation_count == 0
        assert audit2 is not audit1

    def test_record_scanner_lookup_uses_global(self):
        """Test convenience function uses global audit."""
        reset_audit()

        result = record_scanner_lookup('2024-01-04', '2024-01-05', 'AAPL')

        assert result is True
        assert get_audit().violation_count == 0

    def test_record_scanner_lookup_detects_violation(self):
        """Test convenience function detects violations."""
        reset_audit()

        result = record_scanner_lookup('2024-01-05', '2024-01-05', 'AAPL')

        assert result is False
        assert get_audit().violation_count == 1
