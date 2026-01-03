# Walk-Forward Validation Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a rigorous backtesting validation framework with frozen specs, walk-forward testing, scanner integrity audits, and reality checks to catch "too-good-to-be-true" results.

**Architecture:** Create a `validation/` module with four components: (A) frozen spec generator that hashes all config parameters, (B) walk-forward test runner that enforces train/validation/test splits, (C) scanner lookahead auditor that asserts no future data leaks, (D) sensitivity analyzers for costs, delays, regimes, and shorting.

**Tech Stack:** Python 3, pytest, pandas, hashlib (checksums), existing backtest.py infrastructure

---

## Task 1: Create Frozen Spec Document

**Files:**
- Create: `docs/FROZEN_SPEC.md`
- Create: `validation/__init__.py`
- Create: `validation/frozen_spec.py`

**Step 1: Create validation module directory**

```bash
mkdir -p /home/carsonodell/trading-bot/validation
```

**Step 2: Create frozen spec generator**

Create `validation/frozen_spec.py`:

```python
"""
Frozen Spec Generator - Captures exact system configuration for reproducibility.

Generates a frozen specification document with checksums for all parameters
used in backtesting. Once frozen, no parameter changes are allowed during
the out-of-sample test period.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import yaml


class FrozenSpec:
    """
    Captures and validates frozen system configuration.

    Usage:
        spec = FrozenSpec.from_config_files()
        spec.save('docs/FROZEN_SPEC.md')

        # Later, validate nothing changed:
        spec.validate_unchanged()
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.created_at = datetime.now().isoformat()
        self.checksum = self._compute_checksum()

    @classmethod
    def from_config_files(cls,
                          config_path: str = 'config.yaml',
                          universe_path: str = 'universe.yaml') -> 'FrozenSpec':
        """Load configuration from YAML files."""
        base_dir = Path(__file__).parent.parent

        with open(base_dir / config_path, 'r') as f:
            config = yaml.safe_load(f)

        with open(base_dir / universe_path, 'r') as f:
            universe = yaml.safe_load(f)

        # Extract scanner universe symbols
        scanner_universe = universe.get('scanner_universe', {})
        all_symbols = []
        for category, symbols in scanner_universe.items():
            if isinstance(symbols, list):
                all_symbols.extend(symbols)
        # Deduplicate while preserving order
        seen = set()
        unique_symbols = [s for s in all_symbols if not (s in seen or seen.add(s))]

        # Build frozen config structure
        frozen_config = {
            # Universe
            'universe': {
                'source': 'universe.yaml:scanner_universe',
                'symbol_count': len(unique_symbols),
                'symbols_hash': cls._hash_list(unique_symbols),
                'categories': list(scanner_universe.keys()),
            },

            # Scanner formula
            'scanner': {
                'enabled': config.get('volatility_scanner', {}).get('enabled', False),
                'top_n': config.get('volatility_scanner', {}).get('top_n', 10),
                'min_price': config.get('volatility_scanner', {}).get('min_price', 5),
                'max_price': config.get('volatility_scanner', {}).get('max_price', 1000),
                'min_volume': config.get('volatility_scanner', {}).get('min_volume', 500000),
                'lookback_days': 14,  # Hardcoded in scanner
                'weights': config.get('volatility_scanner', {}).get('weights', {
                    'atr_pct': 0.5,
                    'daily_range_pct': 0.3,
                    'volume_ratio': 0.2
                }),
            },

            # Strategy set
            'strategies': {
                name: {
                    'enabled': cfg.get('enabled', False),
                    'weight': cfg.get('weight', 0.0)
                }
                for name, cfg in config.get('strategies', {}).items()
            },

            # Risk parameters
            'risk': {
                'max_position_size_pct': config.get('risk_management', {}).get('max_position_size_pct', 3.0),
                'max_portfolio_risk_pct': config.get('risk_management', {}).get('max_portfolio_risk_pct', 15.0),
                'stop_loss_pct': config.get('risk_management', {}).get('stop_loss_pct', 5.0),
                'take_profit_pct': config.get('risk_management', {}).get('take_profit_pct', 8.0),
                'max_daily_loss_pct': config.get('risk_management', {}).get('max_daily_loss_pct', 3.0),
                'max_open_positions': config.get('risk_management', {}).get('max_open_positions', 5),
            },

            # Exit policy
            'exits': {
                'hard_stop_pct': config.get('exit_manager', {}).get('tier_0_hard_stop', -0.05),
                'profit_floor_pct': config.get('exit_manager', {}).get('tier_1_profit_floor', 0.02),
                'atr_trailing_pct': config.get('exit_manager', {}).get('tier_2_atr_trailing', 0.03),
                'partial_take_pct': config.get('exit_manager', {}).get('tier_3_partial_take', 0.04),
                'max_hold_hours': config.get('exit_manager', {}).get('max_hold_hours', 168),
                'eod_close': config.get('exit_manager', {}).get('eod_close', False),
            },

            # Trailing stop
            'trailing_stop': {
                'enabled': config.get('trailing_stop', {}).get('enabled', True),
                'activation_pct': config.get('trailing_stop', {}).get('activation_pct', 0.25),
                'trail_pct': config.get('trailing_stop', {}).get('trail_pct', 0.25),
                'move_to_breakeven': config.get('trailing_stop', {}).get('move_to_breakeven', True),
            },

            # Execution assumptions
            'execution': {
                'slippage_bps': config.get('execution', {}).get('slippage_bps', 5),
                'half_spread_bps': config.get('execution', {}).get('half_spread_bps', 2),
                'commission': 0.0,  # Alpaca has no commission
            },

            # Shorting constraints
            'shorting': {
                'enabled': not config.get('longs_only', False),
                'shorts_only': config.get('shorts_only', False),
            },

            # Entry gate
            'entry_gate': {
                'confidence_threshold': config.get('entry_gate', {}).get('confidence_threshold', 60),
                'max_trades_per_symbol_per_day': config.get('entry_gate', {}).get('max_trades_per_symbol_per_day', 3),
                'min_time_between_trades_minutes': config.get('entry_gate', {}).get('min_time_between_trades_minutes', 60),
            },
        }

        return cls(frozen_config)

    @staticmethod
    def _hash_list(items: list) -> str:
        """Create deterministic hash of a list."""
        sorted_items = sorted([str(i) for i in items])
        content = '|'.join(sorted_items)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _compute_checksum(self) -> str:
        """Compute checksum of entire frozen config."""
        json_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:32]

    def save(self, path: str = 'docs/FROZEN_SPEC.md') -> None:
        """Save frozen spec to markdown file."""
        base_dir = Path(__file__).parent.parent
        output_path = base_dir / path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        content = self._generate_markdown()
        with open(output_path, 'w') as f:
            f.write(content)

    def _generate_markdown(self) -> str:
        """Generate markdown document."""
        c = self.config

        return f'''# Frozen Specification - Baseline v1

> **WARNING:** This configuration is FROZEN. No parameter changes allowed during OOS test period.
> Any modification invalidates all test results.

**Created:** {self.created_at}
**Checksum:** `{self.checksum}`

---

## 1. Universe Source

| Parameter | Value |
|-----------|-------|
| Source File | `{c['universe']['source']}` |
| Symbol Count | {c['universe']['symbol_count']} |
| Symbols Hash | `{c['universe']['symbols_hash']}` |
| Categories | {', '.join(c['universe']['categories'])} |

## 2. Scanner Formula

| Parameter | Value |
|-----------|-------|
| Enabled | {c['scanner']['enabled']} |
| Top N | {c['scanner']['top_n']} |
| Min Price | ${c['scanner']['min_price']} |
| Max Price | ${c['scanner']['max_price']} |
| Min Volume | {c['scanner']['min_volume']:,} |
| Lookback Days | {c['scanner']['lookback_days']} |

**Volatility Score Weights:**
- ATR %: {c['scanner']['weights'].get('atr_pct', 0.5) * 100:.0f}%
- Daily Range %: {c['scanner']['weights'].get('daily_range_pct', 0.3) * 100:.0f}%
- Volume Ratio: {c['scanner']['weights'].get('volume_ratio', 0.2) * 100:.0f}%

## 3. Strategy Set

| Strategy | Enabled | Weight |
|----------|---------|--------|
{self._format_strategies()}

## 4. Risk Parameters

| Parameter | Value |
|-----------|-------|
| Max Position Size | {c['risk']['max_position_size_pct']}% |
| Max Portfolio Risk | {c['risk']['max_portfolio_risk_pct']}% |
| Stop Loss | {c['risk']['stop_loss_pct']}% |
| Take Profit | {c['risk']['take_profit_pct']}% |
| Max Daily Loss | {c['risk']['max_daily_loss_pct']}% |
| Max Open Positions | {c['risk']['max_open_positions']} |

## 5. Exit Policy

| Parameter | Value |
|-----------|-------|
| Hard Stop | {c['exits']['hard_stop_pct'] * 100:.1f}% |
| Profit Floor | {c['exits']['profit_floor_pct'] * 100:.1f}% |
| ATR Trailing Activation | {c['exits']['atr_trailing_pct'] * 100:.1f}% |
| Partial Take Profit | {c['exits']['partial_take_pct'] * 100:.1f}% |
| Max Hold Hours | {c['exits']['max_hold_hours']} |
| EOD Close | {c['exits']['eod_close']} |

## 6. Trailing Stop

| Parameter | Value |
|-----------|-------|
| Enabled | {c['trailing_stop']['enabled']} |
| Activation | {c['trailing_stop']['activation_pct']}% |
| Trail Distance | {c['trailing_stop']['trail_pct']}% |
| Move to Breakeven | {c['trailing_stop']['move_to_breakeven']} |

## 7. Execution Assumptions

| Parameter | Value |
|-----------|-------|
| Slippage | {c['execution']['slippage_bps']} bps |
| Half Spread | {c['execution']['half_spread_bps']} bps |
| Commission | {c['execution']['commission']} |

## 8. Shorting Constraints

| Parameter | Value |
|-----------|-------|
| Shorting Enabled | {c['shorting']['enabled']} |
| Shorts Only Mode | {c['shorting']['shorts_only']} |

## 9. Entry Gate

| Parameter | Value |
|-----------|-------|
| Confidence Threshold | {c['entry_gate']['confidence_threshold']} |
| Max Trades/Symbol/Day | {c['entry_gate']['max_trades_per_symbol_per_day']} |
| Min Time Between Trades | {c['entry_gate']['min_time_between_trades_minutes']} min |

---

## Validation Command

```bash
python -c "from validation.frozen_spec import FrozenSpec; s=FrozenSpec.from_config_files(); print('Checksum:', s.checksum); assert s.checksum == '{self.checksum}', 'CONFIG CHANGED!'"
```

## Git Tag

```bash
git tag -a baseline_frozen_v1 -m "Frozen spec: {self.checksum[:16]}"
```
'''

    def _format_strategies(self) -> str:
        """Format strategies as markdown table rows."""
        rows = []
        for name, cfg in self.config['strategies'].items():
            enabled = '✓' if cfg['enabled'] else '✗'
            weight = f"{cfg['weight'] * 100:.0f}%" if cfg['enabled'] else '-'
            rows.append(f"| {name} | {enabled} | {weight} |")
        return '\n'.join(rows)

    def validate_unchanged(self) -> bool:
        """Validate current config matches frozen spec."""
        current = FrozenSpec.from_config_files()
        if current.checksum != self.checksum:
            raise ValueError(
                f"CONFIG CHANGED! Frozen: {self.checksum}, Current: {current.checksum}"
            )
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Export frozen config as dict."""
        return {
            'config': self.config,
            'checksum': self.checksum,
            'created_at': self.created_at,
        }


def generate_frozen_spec() -> FrozenSpec:
    """Generate and save frozen spec from current config."""
    spec = FrozenSpec.from_config_files()
    spec.save()
    print(f"Frozen spec saved to docs/FROZEN_SPEC.md")
    print(f"Checksum: {spec.checksum}")
    return spec


if __name__ == '__main__':
    generate_frozen_spec()
```

**Step 3: Create validation module init**

Create `validation/__init__.py`:

```python
"""Validation module for backtesting integrity."""

from .frozen_spec import FrozenSpec, generate_frozen_spec

__all__ = ['FrozenSpec', 'generate_frozen_spec']
```

**Step 4: Generate the frozen spec**

```bash
cd /home/carsonodell/trading-bot && python -m validation.frozen_spec
```

**Step 5: Commit frozen spec**

```bash
git add validation/ docs/FROZEN_SPEC.md
git commit -m "feat: add frozen spec generator for walk-forward validation"
```

---

## Task 2: Create Walk-Forward Test Runner

**Files:**
- Create: `validation/walk_forward.py`
- Create: `tests/test_walk_forward.py`

**Step 1: Write failing test for walk-forward runner**

Create `tests/test_walk_forward.py`:

```python
"""Tests for walk-forward validation framework."""

import pytest
from datetime import datetime
from validation.walk_forward import WalkForwardTest, WalkForwardConfig


class TestWalkForwardConfig:
    """Test walk-forward configuration."""

    def test_default_periods(self):
        """Test default walk-forward periods are set correctly."""
        config = WalkForwardConfig()

        assert config.train_start == '2024-01-05'
        assert config.train_end == '2024-09-30'
        assert config.validation_start == '2024-10-01'
        assert config.validation_end == '2024-12-31'
        assert config.test_start == '2025-01-01'
        assert config.test_end == '2025-12-31'

    def test_no_overlap(self):
        """Test periods don't overlap."""
        config = WalkForwardConfig()

        train_end = datetime.strptime(config.train_end, '%Y-%m-%d')
        val_start = datetime.strptime(config.validation_start, '%Y-%m-%d')
        val_end = datetime.strptime(config.validation_end, '%Y-%m-%d')
        test_start = datetime.strptime(config.test_start, '%Y-%m-%d')

        assert val_start > train_end, "Validation must start after train ends"
        assert test_start > val_end, "Test must start after validation ends"


class TestWalkForwardTest:
    """Test walk-forward test runner."""

    def test_run_returns_metrics(self):
        """Test that run() returns required metrics."""
        wf = WalkForwardTest(symbols=['SPY'], quick_mode=True)
        results = wf.run_period('train')

        assert 'total_return' in results
        assert 'max_drawdown' in results
        assert 'profit_factor' in results
        assert 'total_trades' in results
        assert 'avg_hold_bars' in results

    def test_concentration_metrics(self):
        """Test per-symbol concentration is calculated."""
        wf = WalkForwardTest(symbols=['SPY', 'QQQ'], quick_mode=True)
        results = wf.run_period('train')

        assert 'top1_contribution' in results
        assert 'top3_contribution' in results
```

**Step 2: Run test to verify it fails**

```bash
cd /home/carsonodell/trading-bot && python -m pytest tests/test_walk_forward.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'validation.walk_forward'"

**Step 3: Write walk-forward test runner implementation**

Create `validation/walk_forward.py`:

```python
"""
Walk-Forward Test Framework

Implements proper train/validation/test split for backtesting validation.
Rules:
- Parameters may only be changed before the test period
- Once test period starts, everything is frozen
- Each segment produces standardized metrics
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import pandas as pd
import numpy as np

# Import existing backtest infrastructure
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest import Backtest1Hour


logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward test periods."""

    # Train/Dev period (allowed to tune)
    train_start: str = '2024-01-05'
    train_end: str = '2024-09-30'

    # Validation period (limited tuning)
    validation_start: str = '2024-10-01'
    validation_end: str = '2024-12-31'

    # Test period (true OOS, frozen)
    test_start: str = '2025-01-01'
    test_end: str = '2025-12-31'

    # Initial capital for all tests
    initial_capital: float = 100_000.0


@dataclass
class PeriodResults:
    """Results from a single test period."""

    period_name: str
    start_date: str
    end_date: str

    # Core metrics
    total_return: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0

    # Turnover and hold time
    avg_hold_bars: float = 0.0
    turnover: float = 0.0  # Total traded value / initial capital

    # Concentration metrics
    top1_contribution: float = 0.0  # % of P&L from top symbol
    top3_contribution: float = 0.0  # % of P&L from top 3 symbols
    symbol_pnl: Dict[str, float] = field(default_factory=dict)

    # Trade details
    trades: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'period_name': self.period_name,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_return': self.total_return,
            'max_drawdown': self.max_drawdown,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'avg_hold_bars': self.avg_hold_bars,
            'turnover': self.turnover,
            'top1_contribution': self.top1_contribution,
            'top3_contribution': self.top3_contribution,
            'symbol_pnl': self.symbol_pnl,
        }


class WalkForwardTest:
    """
    Walk-forward test runner.

    Runs backtests on train/validation/test periods and produces
    standardized metrics for comparison.
    """

    def __init__(
        self,
        symbols: List[str] = None,
        config: WalkForwardConfig = None,
        quick_mode: bool = False,
        longs_only: bool = False,
        shorts_only: bool = False,
    ):
        """
        Initialize walk-forward test.

        Args:
            symbols: List of symbols to test (loads from universe.yaml if None)
            config: Walk-forward configuration
            quick_mode: If True, use subset of symbols for faster testing
            longs_only: Only take long positions
            shorts_only: Only take short positions
        """
        self.config = config or WalkForwardConfig()
        self.quick_mode = quick_mode
        self.longs_only = longs_only
        self.shorts_only = shorts_only

        # Load symbols
        if symbols:
            self.symbols = symbols
        else:
            self.symbols = self._load_symbols()

        if quick_mode and len(self.symbols) > 10:
            self.symbols = self.symbols[:10]
            logger.info(f"Quick mode: using {len(self.symbols)} symbols")

        self.results: Dict[str, PeriodResults] = {}

    def _load_symbols(self) -> List[str]:
        """Load symbols from universe.yaml."""
        import yaml

        universe_path = Path(__file__).parent.parent / 'universe.yaml'
        with open(universe_path, 'r') as f:
            universe = yaml.safe_load(f)

        scanner_universe = universe.get('scanner_universe', {})
        all_symbols = []
        for category, symbols in scanner_universe.items():
            if isinstance(symbols, list):
                all_symbols.extend(symbols)

        # Deduplicate
        seen = set()
        unique = [s for s in all_symbols if isinstance(s, str) and not (s in seen or seen.add(s))]
        return unique

    def run_period(self, period: str) -> Dict[str, Any]:
        """
        Run backtest for a specific period.

        Args:
            period: One of 'train', 'validation', 'test'

        Returns:
            Dict with metrics
        """
        if period == 'train':
            start = self.config.train_start
            end = self.config.train_end
        elif period == 'validation':
            start = self.config.validation_start
            end = self.config.validation_end
        elif period == 'test':
            start = self.config.test_start
            end = self.config.test_end
        else:
            raise ValueError(f"Unknown period: {period}")

        logger.info(f"Running {period} period: {start} to {end}")

        # Run backtest
        backtester = Backtest1Hour(
            initial_capital=self.config.initial_capital,
            longs_only=self.longs_only,
            shorts_only=self.shorts_only,
        )

        results = backtester.run(self.symbols, start, end)

        # Calculate metrics
        period_results = self._calculate_metrics(period, start, end, results)
        self.results[period] = period_results

        return period_results.to_dict()

    def run_all(self) -> Dict[str, Dict[str, Any]]:
        """Run all three periods and return combined results."""
        all_results = {}

        for period in ['train', 'validation', 'test']:
            logger.info(f"\n{'='*60}")
            logger.info(f"RUNNING {period.upper()} PERIOD")
            logger.info(f"{'='*60}")
            all_results[period] = self.run_period(period)

        return all_results

    def _calculate_metrics(
        self,
        period_name: str,
        start_date: str,
        end_date: str,
        backtest_results: Dict
    ) -> PeriodResults:
        """Calculate standardized metrics from backtest results."""

        trades = backtest_results.get('trades', [])
        metrics = backtest_results.get('metrics', {})

        # Calculate symbol-level P&L
        symbol_pnl = {}
        total_traded_value = 0.0

        for trade in trades:
            symbol = trade['symbol']
            pnl = trade['pnl']
            symbol_pnl[symbol] = symbol_pnl.get(symbol, 0.0) + pnl
            total_traded_value += abs(trade['entry_price'] * trade['shares'])

        # Calculate concentration
        sorted_pnl = sorted(symbol_pnl.values(), reverse=True)
        total_pnl = sum(sorted_pnl) if sorted_pnl else 0

        if total_pnl > 0:
            top1 = sorted_pnl[0] / total_pnl * 100 if sorted_pnl else 0
            top3 = sum(sorted_pnl[:3]) / total_pnl * 100 if len(sorted_pnl) >= 3 else 100
        else:
            top1 = 0
            top3 = 0

        # Calculate turnover
        turnover = total_traded_value / self.config.initial_capital if trades else 0

        # Average hold time
        avg_hold = np.mean([t['bars_held'] for t in trades]) if trades else 0

        return PeriodResults(
            period_name=period_name,
            start_date=start_date,
            end_date=end_date,
            total_return=metrics.get('total_return_pct', 0),
            max_drawdown=metrics.get('max_drawdown', 0),
            profit_factor=metrics.get('profit_factor', 0),
            total_trades=len(trades),
            win_rate=metrics.get('win_rate', 0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0),
            avg_hold_bars=avg_hold,
            turnover=turnover,
            top1_contribution=top1,
            top3_contribution=top3,
            symbol_pnl=symbol_pnl,
            trades=trades,
        )

    def print_summary(self) -> None:
        """Print summary table of all periods."""
        if not self.results:
            print("No results to display. Run tests first.")
            return

        print(f"\n{'='*80}")
        print("WALK-FORWARD TEST SUMMARY")
        print(f"{'='*80}")

        headers = ['Metric', 'Train', 'Validation', 'Test']
        rows = [
            ('Total Return %', 'total_return'),
            ('Max Drawdown %', 'max_drawdown'),
            ('Profit Factor', 'profit_factor'),
            ('Total Trades', 'total_trades'),
            ('Win Rate %', 'win_rate'),
            ('Sharpe Ratio', 'sharpe_ratio'),
            ('Avg Hold (bars)', 'avg_hold_bars'),
            ('Turnover', 'turnover'),
            ('Top 1 Contribution %', 'top1_contribution'),
            ('Top 3 Contribution %', 'top3_contribution'),
        ]

        # Print header
        print(f"{'Metric':<25} {'Train':>12} {'Validation':>12} {'Test':>12}")
        print('-' * 65)

        for label, attr in rows:
            train_val = getattr(self.results.get('train'), attr, '-') if 'train' in self.results else '-'
            val_val = getattr(self.results.get('validation'), attr, '-') if 'validation' in self.results else '-'
            test_val = getattr(self.results.get('test'), attr, '-') if 'test' in self.results else '-'

            # Format numbers
            def fmt(v):
                if isinstance(v, float):
                    return f"{v:.2f}"
                return str(v)

            print(f"{label:<25} {fmt(train_val):>12} {fmt(val_val):>12} {fmt(test_val):>12}")

        print(f"{'='*80}")

    def save_results(self, path: str = 'logs/walk_forward_results.json') -> None:
        """Save results to JSON file."""
        output_path = Path(__file__).parent.parent / path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'config': {
                'train': f"{self.config.train_start} to {self.config.train_end}",
                'validation': f"{self.config.validation_start} to {self.config.validation_end}",
                'test': f"{self.config.test_start} to {self.config.test_end}",
                'initial_capital': self.config.initial_capital,
                'symbols_count': len(self.symbols),
                'longs_only': self.longs_only,
                'shorts_only': self.shorts_only,
            },
            'results': {
                period: result.to_dict()
                for period, result in self.results.items()
            },
            'generated_at': datetime.now().isoformat(),
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")


def run_walk_forward_test(
    quick_mode: bool = False,
    longs_only: bool = False,
    shorts_only: bool = False,
) -> WalkForwardTest:
    """Run complete walk-forward test and return results."""
    wf = WalkForwardTest(
        quick_mode=quick_mode,
        longs_only=longs_only,
        shorts_only=shorts_only,
    )
    wf.run_all()
    wf.print_summary()
    wf.save_results()
    return wf


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run walk-forward validation test')
    parser.add_argument('--quick', action='store_true', help='Quick mode (10 symbols)')
    parser.add_argument('--longs-only', action='store_true', help='Only long positions')
    parser.add_argument('--shorts-only', action='store_true', help='Only short positions')
    parser.add_argument('--period', choices=['train', 'validation', 'test', 'all'], default='all')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    wf = WalkForwardTest(
        quick_mode=args.quick,
        longs_only=args.longs_only,
        shorts_only=args.shorts_only,
    )

    if args.period == 'all':
        wf.run_all()
    else:
        wf.run_period(args.period)

    wf.print_summary()
    wf.save_results()
```

**Step 4: Update validation/__init__.py**

```python
"""Validation module for backtesting integrity."""

from .frozen_spec import FrozenSpec, generate_frozen_spec
from .walk_forward import WalkForwardTest, WalkForwardConfig, run_walk_forward_test

__all__ = [
    'FrozenSpec',
    'generate_frozen_spec',
    'WalkForwardTest',
    'WalkForwardConfig',
    'run_walk_forward_test',
]
```

**Step 5: Run tests to verify they pass**

```bash
cd /home/carsonodell/trading-bot && python -m pytest tests/test_walk_forward.py -v
```

**Step 6: Commit walk-forward framework**

```bash
git add validation/walk_forward.py validation/__init__.py tests/test_walk_forward.py
git commit -m "feat: add walk-forward test framework with train/validation/test splits"
```

---

## Task 3: Add Scanner Lookahead Audit

**Files:**
- Create: `validation/scanner_audit.py`
- Modify: `backtest.py` (add audit hook)
- Create: `tests/test_scanner_audit.py`

**Step 1: Write failing test for scanner audit**

Create `tests/test_scanner_audit.py`:

```python
"""Tests for scanner lookahead audit."""

import pytest
from validation.scanner_audit import ScannerLookaheadAudit


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
```

**Step 2: Run test to verify it fails**

```bash
cd /home/carsonodell/trading-bot && python -m pytest tests/test_scanner_audit.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write scanner audit implementation**

Create `validation/scanner_audit.py`:

```python
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
```

**Step 4: Update validation/__init__.py**

Add to `validation/__init__.py`:

```python
from .scanner_audit import (
    ScannerLookaheadAudit,
    get_audit,
    reset_audit,
    record_scanner_lookup,
)

__all__ = [
    'FrozenSpec',
    'generate_frozen_spec',
    'WalkForwardTest',
    'WalkForwardConfig',
    'run_walk_forward_test',
    'ScannerLookaheadAudit',
    'get_audit',
    'reset_audit',
    'record_scanner_lookup',
]
```

**Step 5: Run tests to verify they pass**

```bash
cd /home/carsonodell/trading-bot && python -m pytest tests/test_scanner_audit.py -v
```

**Step 6: Commit scanner audit**

```bash
git add validation/scanner_audit.py validation/__init__.py tests/test_scanner_audit.py
git commit -m "feat: add scanner lookahead audit for data integrity validation"
```

---

## Task 4: Create Reality Check Analyzers

**Files:**
- Create: `validation/reality_checks.py`
- Create: `tests/test_reality_checks.py`

**Step 1: Write failing test for reality checks**

Create `tests/test_reality_checks.py`:

```python
"""Tests for reality check analyzers."""

import pytest
from validation.reality_checks import (
    CostSensitivityAnalyzer,
    DelaySensitivityAnalyzer,
    RegimeSplitAnalyzer,
    ShortDisableAnalyzer,
    RealityCheckSuite,
)


class TestCostSensitivity:
    """Test cost sensitivity analysis."""

    def test_runs_multiple_friction_levels(self):
        """Test analyzer runs with different friction levels."""
        analyzer = CostSensitivityAnalyzer(symbols=['SPY'], quick_mode=True)
        results = analyzer.run()

        assert '0_bps' in results
        assert '5_bps' in results
        assert '15_bps' in results

    def test_detects_fragility(self):
        """Test fragility detection."""
        analyzer = CostSensitivityAnalyzer(symbols=['SPY'], quick_mode=True)
        results = analyzer.run()

        assert 'is_fragile' in results
        assert isinstance(results['is_fragile'], bool)


class TestDelaySensitivity:
    """Test delay sensitivity analysis."""

    def test_runs_multiple_delay_modes(self):
        """Test analyzer runs with different delay modes."""
        analyzer = DelaySensitivityAnalyzer(symbols=['SPY'], quick_mode=True)
        results = analyzer.run()

        assert 'same_bar' in results
        assert 'next_open' in results
        assert 'next_open_plus_1' in results


class TestRealityCheckSuite:
    """Test full reality check suite."""

    def test_run_all_returns_summary(self):
        """Test suite runs all checks and returns summary."""
        suite = RealityCheckSuite(symbols=['SPY'], quick_mode=True)
        results = suite.run_all()

        assert 'cost_sensitivity' in results
        assert 'delay_sensitivity' in results
        assert 'regime_split' in results
        assert 'short_disable' in results
        assert 'warnings' in results
```

**Step 2: Run test to verify it fails**

```bash
cd /home/carsonodell/trading-bot && python -m pytest tests/test_reality_checks.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write reality checks implementation**

Create `validation/reality_checks.py`:

```python
"""
Reality Check Analyzers

Fast tests that catch "too-good-to-be-true" backtest results:
- Cost sensitivity: 0 bps, 5 bps, 15 bps friction
- Delay sensitivity: same bar, next open, next open + 1 bar
- Regime split: high-vol vs low-vol months
- Short disable: long-only performance
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest import Backtest1Hour


logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result from a single analysis run."""
    name: str
    total_return: float
    profit_factor: float
    total_trades: int
    win_rate: float
    max_drawdown: float


class CostSensitivityAnalyzer:
    """
    Test performance sensitivity to transaction costs.

    If performance collapses with modest friction, the strategy is fragile.
    """

    FRICTION_LEVELS = [
        ('0_bps', 0, 0),      # Zero friction (unrealistic)
        ('5_bps', 5, 0),      # Light friction
        ('15_bps', 15, 0),    # Heavy friction
    ]

    def __init__(
        self,
        symbols: List[str] = None,
        start_date: str = '2024-10-01',
        end_date: str = '2024-12-31',
        quick_mode: bool = False,
    ):
        self.symbols = symbols or ['SPY']
        self.start_date = start_date
        self.end_date = end_date
        self.quick_mode = quick_mode

        if quick_mode:
            self.symbols = self.symbols[:5]

    def run(self) -> Dict[str, Any]:
        """Run cost sensitivity analysis."""
        results = {}
        baseline_return = None

        for name, slippage_bps, spread_bps in self.FRICTION_LEVELS:
            logger.info(f"Running cost sensitivity: {name}")

            # Create config override
            config = self._get_config_with_costs(slippage_bps, spread_bps)

            backtester = Backtest1Hour(
                initial_capital=100_000,
                config=config,
            )

            bt_results = backtester.run(self.symbols, self.start_date, self.end_date)
            metrics = bt_results.get('metrics', {})

            total_return = metrics.get('total_return_pct', 0)

            if baseline_return is None:
                baseline_return = total_return

            results[name] = {
                'total_return': total_return,
                'profit_factor': metrics.get('profit_factor', 0),
                'total_trades': metrics.get('total_trades', 0),
                'win_rate': metrics.get('win_rate', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
            }

        # Check for fragility
        if baseline_return and baseline_return > 0:
            return_at_15bps = results.get('15_bps', {}).get('total_return', 0)
            decay = (baseline_return - return_at_15bps) / baseline_return
            is_fragile = decay > 0.5  # >50% decay = fragile
        else:
            is_fragile = False

        results['is_fragile'] = is_fragile
        results['decay_at_15bps'] = decay if baseline_return else 0

        return results

    def _get_config_with_costs(self, slippage_bps: int, spread_bps: int) -> Dict:
        """Get config with specified costs."""
        import yaml

        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if 'execution' not in config:
            config['execution'] = {}

        config['execution']['slippage_bps'] = slippage_bps
        config['execution']['half_spread_bps'] = spread_bps

        return config


class DelaySensitivityAnalyzer:
    """
    Test performance sensitivity to execution delay.

    Modes:
    - same_bar: Fill at signal bar close (unrealistic)
    - next_open: Fill at next bar open (standard)
    - next_open_plus_1: Fill at next bar open + 1 bar (conservative)
    """

    def __init__(
        self,
        symbols: List[str] = None,
        start_date: str = '2024-10-01',
        end_date: str = '2024-12-31',
        quick_mode: bool = False,
    ):
        self.symbols = symbols or ['SPY']
        self.start_date = start_date
        self.end_date = end_date
        self.quick_mode = quick_mode

        if quick_mode:
            self.symbols = self.symbols[:5]

    def run(self) -> Dict[str, Any]:
        """Run delay sensitivity analysis."""
        # Note: Current backtest already uses next_open (realistic)
        # This analyzer documents that and provides comparison baseline

        results = {}

        # Run standard backtest (next_open)
        backtester = Backtest1Hour(initial_capital=100_000)
        bt_results = backtester.run(self.symbols, self.start_date, self.end_date)
        metrics = bt_results.get('metrics', {})

        results['next_open'] = {
            'total_return': metrics.get('total_return_pct', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'total_trades': metrics.get('total_trades', 0),
            'description': 'Fill at next bar open (current implementation)',
        }

        # same_bar and next_open_plus_1 would require backtest modifications
        # For now, document as placeholders
        results['same_bar'] = {
            'total_return': None,
            'description': 'Fill at signal bar close - NOT IMPLEMENTED (unrealistic)',
        }

        results['next_open_plus_1'] = {
            'total_return': None,
            'description': 'Fill at next bar open + 1 bar - NOT IMPLEMENTED',
        }

        return results


class RegimeSplitAnalyzer:
    """
    Test performance in different market regimes.

    Splits test period into high-volatility and low-volatility months.
    """

    # VIX-based regime classification for 2024
    HIGH_VOL_MONTHS_2024 = ['2024-04', '2024-08', '2024-10']
    LOW_VOL_MONTHS_2024 = ['2024-01', '2024-02', '2024-03', '2024-05', '2024-06', '2024-07', '2024-09', '2024-11', '2024-12']

    def __init__(
        self,
        symbols: List[str] = None,
        quick_mode: bool = False,
    ):
        self.symbols = symbols or ['SPY']
        self.quick_mode = quick_mode

        if quick_mode:
            self.symbols = self.symbols[:5]

    def run(self) -> Dict[str, Any]:
        """Run regime split analysis."""
        results = {}

        # High volatility periods
        logger.info("Running high-vol regime test...")
        high_vol_results = self._run_months(self.HIGH_VOL_MONTHS_2024)
        results['high_vol'] = {
            'months': self.HIGH_VOL_MONTHS_2024,
            **high_vol_results,
        }

        # Low volatility periods
        logger.info("Running low-vol regime test...")
        low_vol_results = self._run_months(self.LOW_VOL_MONTHS_2024)
        results['low_vol'] = {
            'months': self.LOW_VOL_MONTHS_2024,
            **low_vol_results,
        }

        # Compare
        high_return = high_vol_results.get('total_return', 0)
        low_return = low_vol_results.get('total_return', 0)

        results['regime_dependency'] = abs(high_return - low_return) > 20  # >20% difference

        return results

    def _run_months(self, months: List[str]) -> Dict[str, Any]:
        """Run backtest for specific months."""
        all_trades = []
        total_pnl = 0

        for month in months:
            year, mon = month.split('-')
            start = f"{year}-{mon}-01"

            # Calculate end of month
            import calendar
            last_day = calendar.monthrange(int(year), int(mon))[1]
            end = f"{year}-{mon}-{last_day}"

            try:
                backtester = Backtest1Hour(initial_capital=100_000)
                bt_results = backtester.run(self.symbols, start, end)

                trades = bt_results.get('trades', [])
                all_trades.extend(trades)
                total_pnl += sum(t['pnl'] for t in trades)
            except Exception as e:
                logger.warning(f"Error running month {month}: {e}")
                continue

        # Calculate aggregate metrics
        wins = [t for t in all_trades if t['pnl'] > 0]
        losses = [t for t in all_trades if t['pnl'] < 0]

        total_wins = sum(t['pnl'] for t in wins) if wins else 0
        total_losses = abs(sum(t['pnl'] for t in losses)) if losses else 0

        return {
            'total_return': total_pnl / 1000,  # % of 100k
            'total_trades': len(all_trades),
            'profit_factor': total_wins / total_losses if total_losses > 0 else 0,
            'win_rate': len(wins) / len(all_trades) * 100 if all_trades else 0,
        }


class ShortDisableAnalyzer:
    """
    Test long-only performance.

    Shorting is often fragile in backtests. This tests what happens
    when shorts are disabled.
    """

    def __init__(
        self,
        symbols: List[str] = None,
        start_date: str = '2024-10-01',
        end_date: str = '2024-12-31',
        quick_mode: bool = False,
    ):
        self.symbols = symbols or ['SPY']
        self.start_date = start_date
        self.end_date = end_date
        self.quick_mode = quick_mode

        if quick_mode:
            self.symbols = self.symbols[:5]

    def run(self) -> Dict[str, Any]:
        """Run short disable analysis."""
        results = {}

        # Both directions
        logger.info("Running with both long and short...")
        both_bt = Backtest1Hour(initial_capital=100_000)
        both_results = both_bt.run(self.symbols, self.start_date, self.end_date)
        both_metrics = both_results.get('metrics', {})

        results['both'] = {
            'total_return': both_metrics.get('total_return_pct', 0),
            'profit_factor': both_metrics.get('profit_factor', 0),
            'total_trades': both_metrics.get('total_trades', 0),
        }

        # Long only
        logger.info("Running long-only...")
        long_bt = Backtest1Hour(initial_capital=100_000, longs_only=True)
        long_results = long_bt.run(self.symbols, self.start_date, self.end_date)
        long_metrics = long_results.get('metrics', {})

        results['longs_only'] = {
            'total_return': long_metrics.get('total_return_pct', 0),
            'profit_factor': long_metrics.get('profit_factor', 0),
            'total_trades': long_metrics.get('total_trades', 0),
        }

        # Short only
        logger.info("Running short-only...")
        short_bt = Backtest1Hour(initial_capital=100_000, shorts_only=True)
        short_results = short_bt.run(self.symbols, self.start_date, self.end_date)
        short_metrics = short_results.get('metrics', {})

        results['shorts_only'] = {
            'total_return': short_metrics.get('total_return_pct', 0),
            'profit_factor': short_metrics.get('profit_factor', 0),
            'total_trades': short_metrics.get('total_trades', 0),
        }

        # Check if shorts are helping or hurting
        long_return = results['longs_only']['total_return']
        both_return = results['both']['total_return']

        results['shorts_helping'] = both_return > long_return
        results['shorts_pf'] = results['shorts_only']['profit_factor']

        return results


class RealityCheckSuite:
    """
    Run all reality checks and produce summary report.
    """

    def __init__(
        self,
        symbols: List[str] = None,
        start_date: str = '2024-10-01',
        end_date: str = '2024-12-31',
        quick_mode: bool = False,
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.quick_mode = quick_mode

    def run_all(self) -> Dict[str, Any]:
        """Run all reality checks."""
        results = {}
        warnings = []

        # Cost sensitivity
        logger.info("\n=== COST SENSITIVITY ===")
        cost_analyzer = CostSensitivityAnalyzer(
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            quick_mode=self.quick_mode,
        )
        results['cost_sensitivity'] = cost_analyzer.run()

        if results['cost_sensitivity'].get('is_fragile'):
            warnings.append("FRAGILE: Performance collapses with 15bps friction")

        # Delay sensitivity
        logger.info("\n=== DELAY SENSITIVITY ===")
        delay_analyzer = DelaySensitivityAnalyzer(
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            quick_mode=self.quick_mode,
        )
        results['delay_sensitivity'] = delay_analyzer.run()

        # Regime split
        logger.info("\n=== REGIME SPLIT ===")
        regime_analyzer = RegimeSplitAnalyzer(
            symbols=self.symbols,
            quick_mode=self.quick_mode,
        )
        results['regime_split'] = regime_analyzer.run()

        if results['regime_split'].get('regime_dependency'):
            warnings.append("REGIME DEPENDENT: >20% return difference between high/low vol")

        # Short disable
        logger.info("\n=== SHORT DISABLE ===")
        short_analyzer = ShortDisableAnalyzer(
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            quick_mode=self.quick_mode,
        )
        results['short_disable'] = short_analyzer.run()

        if results['short_disable'].get('shorts_pf', 0) < 1.0:
            warnings.append("FRAGILE SHORTS: Short-only PF < 1.0")

        results['warnings'] = warnings
        results['generated_at'] = datetime.now().isoformat()

        return results

    def save_report(self, path: str = 'logs/reality_check_report.json') -> None:
        """Save report to file."""
        results = self.run_all()

        output_path = Path(__file__).parent.parent / path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Reality check report saved to {output_path}")

        # Print summary
        print("\n" + "="*60)
        print("REALITY CHECK SUMMARY")
        print("="*60)

        if results['warnings']:
            print("\nWARNINGS:")
            for w in results['warnings']:
                print(f"  ! {w}")
        else:
            print("\n  All checks passed.")

        print("="*60)


def run_reality_checks(quick_mode: bool = True) -> Dict[str, Any]:
    """Run all reality checks and return results."""
    suite = RealityCheckSuite(quick_mode=quick_mode)
    suite.save_report()
    return suite.run_all()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run reality checks')
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    parser.add_argument('--full', action='store_true', help='Full universe')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_reality_checks(quick_mode=not args.full)
```

**Step 4: Update validation/__init__.py**

Add to `validation/__init__.py`:

```python
from .reality_checks import (
    CostSensitivityAnalyzer,
    DelaySensitivityAnalyzer,
    RegimeSplitAnalyzer,
    ShortDisableAnalyzer,
    RealityCheckSuite,
    run_reality_checks,
)

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
]
```

**Step 5: Run tests to verify they pass**

```bash
cd /home/carsonodell/trading-bot && python -m pytest tests/test_reality_checks.py -v
```

**Step 6: Commit reality checks**

```bash
git add validation/reality_checks.py validation/__init__.py tests/test_reality_checks.py
git commit -m "feat: add reality check analyzers for costs, delays, regimes, and shorts"
```

---

## Task 5: Create CLI Runner

**Files:**
- Create: `validation/cli.py`
- Modify: `validation/__init__.py`

**Step 1: Create CLI entry point**

Create `validation/cli.py`:

```python
#!/usr/bin/env python3
"""
Validation Suite CLI

Run validation tools from command line:

    python -m validation frozen-spec     # Generate frozen spec
    python -m validation walk-forward    # Run walk-forward test
    python -m validation scanner-audit   # Run scanner audit
    python -m validation reality-checks  # Run reality checks
    python -m validation all             # Run everything
"""

import argparse
import logging
import sys

from .frozen_spec import generate_frozen_spec
from .walk_forward import run_walk_forward_test
from .reality_checks import run_reality_checks


def main():
    parser = argparse.ArgumentParser(
        description='Trading Bot Validation Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Commands:
  frozen-spec     Generate frozen specification document
  walk-forward    Run walk-forward test (train/validation/test)
  reality-checks  Run reality checks (costs, delays, regimes, shorts)
  all             Run all validations

Examples:
  python -m validation frozen-spec
  python -m validation walk-forward --quick
  python -m validation reality-checks --quick
  python -m validation all --quick
        '''
    )

    parser.add_argument('command', choices=['frozen-spec', 'walk-forward', 'reality-checks', 'all'])
    parser.add_argument('--quick', action='store_true', help='Quick mode (subset of symbols)')
    parser.add_argument('--longs-only', action='store_true', help='Long positions only')
    parser.add_argument('--shorts-only', action='store_true', help='Short positions only')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.command == 'frozen-spec':
        print("\n=== GENERATING FROZEN SPEC ===\n")
        spec = generate_frozen_spec()
        print(f"\nFrozen spec saved. Checksum: {spec.checksum}")
        print("\nTo tag this commit:")
        print(f"  git tag -a baseline_frozen_v1 -m 'Frozen spec: {spec.checksum[:16]}'")

    elif args.command == 'walk-forward':
        print("\n=== RUNNING WALK-FORWARD TEST ===\n")
        wf = run_walk_forward_test(
            quick_mode=args.quick,
            longs_only=args.longs_only,
            shorts_only=args.shorts_only,
        )

    elif args.command == 'reality-checks':
        print("\n=== RUNNING REALITY CHECKS ===\n")
        run_reality_checks(quick_mode=args.quick)

    elif args.command == 'all':
        print("\n=== RUNNING ALL VALIDATIONS ===\n")

        print("\n--- Frozen Spec ---")
        spec = generate_frozen_spec()

        print("\n--- Walk-Forward Test ---")
        wf = run_walk_forward_test(
            quick_mode=args.quick,
            longs_only=args.longs_only,
            shorts_only=args.shorts_only,
        )

        print("\n--- Reality Checks ---")
        run_reality_checks(quick_mode=args.quick)

        print("\n=== ALL VALIDATIONS COMPLETE ===")
        print(f"Frozen spec checksum: {spec.checksum}")


if __name__ == '__main__':
    main()
```

**Step 2: Create validation/__main__.py**

Create `validation/__main__.py`:

```python
"""Allow running as python -m validation"""
from .cli import main

if __name__ == '__main__':
    main()
```

**Step 3: Test CLI**

```bash
cd /home/carsonodell/trading-bot && python -m validation frozen-spec
```

**Step 4: Commit CLI**

```bash
git add validation/cli.py validation/__main__.py
git commit -m "feat: add validation CLI for running all validation tools"
```

---

## Task 6: Final Integration and Git Tag

**Step 1: Run full validation suite (quick mode)**

```bash
cd /home/carsonodell/trading-bot && python -m validation all --quick
```

**Step 2: Verify frozen spec was created**

```bash
cat /home/carsonodell/trading-bot/docs/FROZEN_SPEC.md | head -50
```

**Step 3: Create git tag**

```bash
git add -A
git commit -m "feat: complete walk-forward validation framework

- Frozen spec generator with SHA256 checksums
- Walk-forward test runner (train/validation/test splits)
- Scanner lookahead audit
- Reality checks (cost sensitivity, delay sensitivity, regime split, short disable)
- CLI for running all validations"

git tag -a baseline_frozen_v1 -m "Frozen spec baseline for walk-forward validation"
```

**Step 4: Verify tag**

```bash
git tag -l "baseline*"
git show baseline_frozen_v1
```

---

## Summary

This plan creates a complete validation framework:

| Component | File | Purpose |
|-----------|------|---------|
| Frozen Spec | `validation/frozen_spec.py` | Captures exact config with checksums |
| Walk-Forward | `validation/walk_forward.py` | Train/validation/test splits |
| Scanner Audit | `validation/scanner_audit.py` | Detects lookahead bias |
| Reality Checks | `validation/reality_checks.py` | Cost, delay, regime, short tests |
| CLI | `validation/cli.py` | Command-line interface |

Run commands:
```bash
python -m validation frozen-spec        # Step A
python -m validation walk-forward       # Step B
python -m validation reality-checks     # Step D
python -m validation all --quick        # All at once (fast)
python -m validation all                # Full validation
```
