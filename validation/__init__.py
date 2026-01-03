"""Validation module for backtesting integrity."""

from .frozen_spec import FrozenSpec, generate_frozen_spec
from .walk_forward import WalkForwardTest, WalkForwardConfig, run_walk_forward_test
from .scanner_audit import (
    ScannerLookaheadAudit,
    get_audit,
    reset_audit,
    record_scanner_lookup,
)
from .reality_checks import (
    CostSensitivityAnalyzer,
    DelaySensitivityAnalyzer,
    RegimeSplitAnalyzer,
    ShortDisableAnalyzer,
    RealityCheckSuite,
    run_reality_checks,
)
from .run_rolling_walkforward import run_rolling_walkforward, ROLLING_SPLITS

__all__ = [
    # Frozen spec
    'FrozenSpec',
    'generate_frozen_spec',
    # Walk forward
    'WalkForwardTest',
    'WalkForwardConfig',
    'run_walk_forward_test',
    # Scanner audit
    'ScannerLookaheadAudit',
    'get_audit',
    'reset_audit',
    'record_scanner_lookup',
    # Reality checks
    'CostSensitivityAnalyzer',
    'DelaySensitivityAnalyzer',
    'RegimeSplitAnalyzer',
    'ShortDisableAnalyzer',
    'RealityCheckSuite',
    'run_reality_checks',
    # Rolling walk-forward
    'run_rolling_walkforward',
    'ROLLING_SPLITS',
]
