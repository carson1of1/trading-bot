"""Test that error logging includes traceback information."""
import logging
import pytest


def test_error_logging_includes_traceback(caplog):
    """Verify that logger.error with exc_info=True includes traceback."""
    logger = logging.getLogger("test_traceback")

    with caplog.at_level(logging.ERROR):
        try:
            raise ValueError("Test exception message")
        except ValueError as e:
            logger.error(f"Caught error: {e}", exc_info=True)

    log_output = caplog.text

    # Should contain the error message
    assert "Test exception message" in log_output
    # Should contain traceback indicator
    assert "Traceback" in log_output
    # Should contain the file reference
    assert "test_traceback_logging.py" in log_output
    # Should contain the exception type
    assert "ValueError" in log_output
