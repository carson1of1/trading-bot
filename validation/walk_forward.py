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
