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
        decay = 0
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
