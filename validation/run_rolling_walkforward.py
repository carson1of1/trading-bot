#!/usr/bin/env python3
"""
Rolling Walk-Forward Validation

Runs backtest across multiple date splits with FROZEN rules (no tuning).
This validates that performance is consistent across different time periods.

Usage:
    python -m validation.run_rolling_walkforward
    python -m validation.run_rolling_walkforward --longs-only
    python -m validation.run_rolling_walkforward --quick
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from .walk_forward import WalkForwardTest, WalkForwardConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define rolling date splits
# Each split has non-overlapping test periods
ROLLING_SPLITS = [
    {
        'name': 'Split 1 (Q4 2024)',
        'config': WalkForwardConfig(
            train_start='2024-01-05',
            train_end='2024-06-30',
            validation_start='2024-07-01',
            validation_end='2024-09-30',
            test_start='2024-10-01',
            test_end='2024-12-31',
        ),
    },
    {
        'name': 'Split 2 (Q1 2025)',
        'config': WalkForwardConfig(
            train_start='2024-04-01',
            train_end='2024-09-30',
            validation_start='2024-10-01',
            validation_end='2024-12-31',
            test_start='2025-01-01',
            test_end='2025-03-31',
        ),
    },
    {
        'name': 'Split 3 (Q2 2025)',
        'config': WalkForwardConfig(
            train_start='2024-07-01',
            train_end='2024-12-31',
            validation_start='2025-01-01',
            validation_end='2025-03-31',
            test_start='2025-04-01',
            test_end='2025-06-30',
        ),
    },
]


def run_rolling_walkforward(
    longs_only: bool = False,
    shorts_only: bool = False,
    quick_mode: bool = False,
    splits: List[Dict] = None,
) -> Dict[str, Any]:
    """
    Run walk-forward validation across multiple date splits.

    Args:
        longs_only: Only take long positions
        shorts_only: Only take short positions
        quick_mode: Use subset of symbols for faster testing
        splits: Custom splits (uses ROLLING_SPLITS if None)

    Returns:
        Dict with results from all splits
    """
    splits = splits or ROLLING_SPLITS
    all_results = {}

    print("\n" + "=" * 70)
    print("ROLLING WALK-FORWARD VALIDATION")
    print("=" * 70)
    print(f"Splits: {len(splits)}")
    print(f"Mode: {'Longs Only' if longs_only else 'Shorts Only' if shorts_only else 'Both'}")
    print(f"Quick Mode: {quick_mode}")
    print("=" * 70)

    for i, split in enumerate(splits, 1):
        name = split['name']
        config = split['config']

        print(f"\n{'='*70}")
        print(f"SPLIT {i}: {name}")
        print(f"Test Period: {config.test_start} to {config.test_end}")
        print(f"{'='*70}\n")

        # Create walk-forward test with this config
        wf = WalkForwardTest(
            config=config,
            quick_mode=quick_mode,
            longs_only=longs_only,
            shorts_only=shorts_only,
        )

        # Only run the TEST period (OOS, no tuning allowed)
        results = wf.run_period('test')

        all_results[name] = {
            'test_start': config.test_start,
            'test_end': config.test_end,
            'total_return': results.get('total_return', 0),
            'max_drawdown': results.get('max_drawdown', 0),
            'profit_factor': results.get('profit_factor', 0),
            'total_trades': results.get('total_trades', 0),
            'win_rate': results.get('win_rate', 0),
            'sharpe_ratio': results.get('sharpe_ratio', 0),
        }

        # Print individual split results
        print(f"\n  Total Return: {results.get('total_return', 0):.2f}%")
        print(f"  Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
        print(f"  Profit Factor: {results.get('profit_factor', 0):.2f}")
        print(f"  Total Trades: {results.get('total_trades', 0)}")
        print(f"  Win Rate: {results.get('win_rate', 0):.1f}%")

    # Calculate aggregate statistics
    returns = [r['total_return'] for r in all_results.values()]
    drawdowns = [r['max_drawdown'] for r in all_results.values()]
    profit_factors = [r['profit_factor'] for r in all_results.values()]

    import numpy as np

    summary = {
        'splits_run': len(splits),
        'avg_return': np.mean(returns),
        'std_return': np.std(returns),
        'min_return': min(returns),
        'max_return': max(returns),
        'avg_drawdown': np.mean(drawdowns),
        'max_drawdown': max(drawdowns),
        'avg_profit_factor': np.mean(profit_factors),
        'all_profitable': all(r > 0 for r in returns),
        'consistency_score': 1 - (np.std(returns) / abs(np.mean(returns))) if np.mean(returns) != 0 else 0,
    }

    all_results['summary'] = summary
    all_results['generated_at'] = datetime.now().isoformat()

    # Print summary
    print("\n" + "=" * 70)
    print("ROLLING WALK-FORWARD SUMMARY")
    print("=" * 70)
    print(f"\n{'Split':<25} {'Return %':>12} {'Max DD %':>12} {'PF':>10} {'Trades':>10}")
    print("-" * 70)

    for name, result in all_results.items():
        if name in ['summary', 'generated_at']:
            continue
        print(f"{name:<25} {result['total_return']:>12.2f} {result['max_drawdown']:>12.2f} {result['profit_factor']:>10.2f} {result['total_trades']:>10}")

    print("-" * 70)
    print(f"{'Average':<25} {summary['avg_return']:>12.2f} {summary['avg_drawdown']:>12.2f} {summary['avg_profit_factor']:>10.2f}")
    print(f"{'Std Dev':<25} {summary['std_return']:>12.2f}")
    print("=" * 70)

    # Verdict
    print("\nVERDICT:")
    if summary['all_profitable']:
        print("  [PASS] All splits profitable")
    else:
        print("  [FAIL] Some splits unprofitable")

    if summary['consistency_score'] > 0.5:
        print(f"  [PASS] Consistent returns (score: {summary['consistency_score']:.2f})")
    else:
        print(f"  [WARN] High variance in returns (score: {summary['consistency_score']:.2f})")

    if summary['max_drawdown'] < 15:
        print(f"  [PASS] Max drawdown under 15% ({summary['max_drawdown']:.1f}%)")
    else:
        print(f"  [WARN] High max drawdown ({summary['max_drawdown']:.1f}%)")

    print("=" * 70)

    return all_results


def save_results(results: Dict[str, Any], path: str = 'logs/rolling_walkforward_results.json'):
    """Save results to JSON file."""
    output_path = Path(__file__).parent.parent / path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run rolling walk-forward validation across multiple date splits'
    )
    parser.add_argument('--longs-only', action='store_true', help='Only long positions')
    parser.add_argument('--shorts-only', action='store_true', help='Only short positions')
    parser.add_argument('--quick', action='store_true', help='Quick mode (10 symbols)')
    parser.add_argument('--save', action='store_true', help='Save results to JSON', default=True)

    args = parser.parse_args()

    results = run_rolling_walkforward(
        longs_only=args.longs_only,
        shorts_only=args.shorts_only,
        quick_mode=args.quick,
    )

    if args.save:
        save_results(results)


if __name__ == '__main__':
    main()
