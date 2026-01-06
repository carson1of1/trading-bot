"""
Pytest configuration and fixtures for test suite.
"""

import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def clear_exit_manager_state():
    """
    Clear ExitManager state file before each test to ensure test isolation.

    The ExitManager persists state to logs/exit_manager_state.json for
    production use (preserving state across bot restarts). However, this
    causes test pollution where state from one test leaks into another.

    This fixture runs automatically before every test to ensure clean state.
    """
    state_file = Path(__file__).parent.parent / 'logs' / 'exit_manager_state.json'
    if state_file.exists():
        state_file.unlink()

    yield  # Run the test

    # Cleanup after test as well
    if state_file.exists():
        state_file.unlink()
