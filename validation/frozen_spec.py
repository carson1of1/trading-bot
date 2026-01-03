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
            enabled = 'Y' if cfg['enabled'] else 'N'
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
