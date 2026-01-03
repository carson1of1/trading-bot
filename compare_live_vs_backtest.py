#!/usr/bin/env python3
"""
Live vs Backtest Comparison Script

Compares live trading results with backtest results to validate
that the backtest accurately represents live performance.

Usage:
    python compare_live_vs_backtest.py
    python compare_live_vs_backtest.py --start 2026-01-03 --end 2026-01-17
    python compare_live_vs_backtest.py --live-db logs/trades.db --output comparison_report.json

After running live for 2 weeks, this script will:
1. Load live trades from the database
2. Run a backtest for the same period
3. Compare scanner output, trades, exits, and P&L
4. Generate a detailed comparison report
"""

import argparse
import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveBacktestComparator:
    """Compare live trading results with backtest results."""

    def __init__(self, live_db: str = 'logs/trades.db'):
        self.live_db = live_db
        self.report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {},
            'scanner_comparison': {},
            'trade_comparison': {},
            'exit_analysis': {},
            'pnl_analysis': {},
            'recommendations': []
        }

    def load_live_trades(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load live trades from database."""
        try:
            with sqlite3.connect(self.live_db) as conn:
                query = '''
                    SELECT * FROM trades
                    WHERE timestamp >= ? AND timestamp <= ?
                    AND status = 'closed'
                    ORDER BY timestamp
                '''
                df = pd.read_sql_query(
                    query, conn,
                    params=[f"{start_date} 00:00:00", f"{end_date} 23:59:59"]
                )
            logger.info(f"Loaded {len(df)} live trades from {start_date} to {end_date}")
            return df
        except Exception as e:
            logger.error(f"Error loading live trades: {e}")
            return pd.DataFrame()

    def load_live_scans(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load live scanner results from database."""
        try:
            with sqlite3.connect(self.live_db) as conn:
                query = '''
                    SELECT * FROM scans
                    WHERE scan_date >= ? AND scan_date <= ?
                    AND mode IN ('LIVE', 'PAPER')
                    ORDER BY scan_date
                '''
                df = pd.read_sql_query(query, conn, params=[start_date, end_date])
            logger.info(f"Loaded {len(df)} live scan results")
            return df
        except Exception as e:
            logger.error(f"Error loading live scans: {e}")
            return pd.DataFrame()

    def load_backtest_scans(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load backtest scanner results from database."""
        try:
            with sqlite3.connect(self.live_db) as conn:
                query = '''
                    SELECT * FROM scans
                    WHERE scan_date >= ? AND scan_date <= ?
                    AND mode = 'BACKTEST'
                    ORDER BY scan_date
                '''
                df = pd.read_sql_query(query, conn, params=[start_date, end_date])
            logger.info(f"Loaded {len(df)} backtest scan results")
            return df
        except Exception as e:
            logger.error(f"Error loading backtest scans: {e}")
            return pd.DataFrame()

    def run_backtest(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Run backtest for the comparison period and return trades."""
        from backtest import Backtest1Hour

        logger.info(f"Running backtest from {start_date} to {end_date}...")

        try:
            backtest = Backtest1Hour()
            results = backtest.run(start_date=start_date, end_date=end_date)

            # Convert trades to DataFrame
            if results and 'trades' in results:
                trades = results['trades']
                df = pd.DataFrame(trades)
                logger.info(f"Backtest completed with {len(df)} trades")
                return df
            else:
                logger.warning("Backtest returned no trades")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return pd.DataFrame()

    def compare_scanners(self, live_scans: pd.DataFrame, backtest_scans: pd.DataFrame) -> Dict:
        """Compare scanner output between live and backtest."""
        comparison = {
            'dates_compared': 0,
            'avg_overlap_pct': 0,
            'daily_overlap': [],
            'live_only_symbols': defaultdict(int),
            'backtest_only_symbols': defaultdict(int)
        }

        if live_scans.empty or backtest_scans.empty:
            comparison['error'] = 'Missing scan data for comparison'
            return comparison

        # Get unique dates in both
        live_dates = set(live_scans['scan_date'].unique())
        backtest_dates = set(backtest_scans['scan_date'].unique())
        common_dates = live_dates & backtest_dates

        comparison['dates_compared'] = len(common_dates)
        comparison['live_only_dates'] = len(live_dates - backtest_dates)
        comparison['backtest_only_dates'] = len(backtest_dates - live_dates)

        overlaps = []

        for date in sorted(common_dates):
            live_row = live_scans[live_scans['scan_date'] == date].iloc[0]
            backtest_row = backtest_scans[backtest_scans['scan_date'] == date].iloc[0]

            live_symbols = set(json.loads(live_row['selected_symbols']))
            backtest_symbols = set(json.loads(backtest_row['selected_symbols']))

            overlap = live_symbols & backtest_symbols
            union = live_symbols | backtest_symbols

            overlap_pct = (len(overlap) / len(union) * 100) if union else 0
            overlaps.append(overlap_pct)

            comparison['daily_overlap'].append({
                'date': date,
                'live_count': len(live_symbols),
                'backtest_count': len(backtest_symbols),
                'overlap_count': len(overlap),
                'overlap_pct': round(overlap_pct, 1),
                'live_only': list(live_symbols - backtest_symbols),
                'backtest_only': list(backtest_symbols - live_symbols)
            })

            # Track symbols that appear in one but not the other
            for sym in (live_symbols - backtest_symbols):
                comparison['live_only_symbols'][sym] += 1
            for sym in (backtest_symbols - live_symbols):
                comparison['backtest_only_symbols'][sym] += 1

        comparison['avg_overlap_pct'] = round(np.mean(overlaps), 1) if overlaps else 0
        comparison['min_overlap_pct'] = round(min(overlaps), 1) if overlaps else 0
        comparison['max_overlap_pct'] = round(max(overlaps), 1) if overlaps else 0

        # Convert defaultdicts to regular dicts for JSON serialization
        comparison['live_only_symbols'] = dict(comparison['live_only_symbols'])
        comparison['backtest_only_symbols'] = dict(comparison['backtest_only_symbols'])

        return comparison

    def compare_trades(self, live_trades: pd.DataFrame, backtest_trades: pd.DataFrame) -> Dict:
        """Compare trade characteristics between live and backtest."""
        comparison = {
            'live_trade_count': len(live_trades),
            'backtest_trade_count': len(backtest_trades),
            'symbol_analysis': {},
            'strategy_analysis': {},
            'direction_analysis': {}
        }

        if live_trades.empty and backtest_trades.empty:
            comparison['error'] = 'No trades to compare'
            return comparison

        # Symbol frequency comparison
        live_symbols = live_trades['symbol'].value_counts().to_dict() if not live_trades.empty else {}
        backtest_symbols = backtest_trades['symbol'].value_counts().to_dict() if not backtest_trades.empty else {}

        all_symbols = set(live_symbols.keys()) | set(backtest_symbols.keys())
        comparison['symbol_analysis'] = {
            'unique_live': len(live_symbols),
            'unique_backtest': len(backtest_symbols),
            'common_symbols': len(set(live_symbols.keys()) & set(backtest_symbols.keys())),
            'top_live': dict(list(live_symbols.items())[:10]),
            'top_backtest': dict(list(backtest_symbols.items())[:10])
        }

        # Strategy comparison
        if 'strategy' in live_trades.columns and not live_trades.empty:
            live_strategies = live_trades['strategy'].value_counts().to_dict()
        else:
            live_strategies = {}

        if 'strategy' in backtest_trades.columns and not backtest_trades.empty:
            backtest_strategies = backtest_trades['strategy'].value_counts().to_dict()
        else:
            backtest_strategies = {}

        comparison['strategy_analysis'] = {
            'live': live_strategies,
            'backtest': backtest_strategies
        }

        # Direction comparison (LONG vs SHORT)
        if 'action' in live_trades.columns and not live_trades.empty:
            live_longs = len(live_trades[live_trades['action'] == 'BUY'])
            live_shorts = len(live_trades[live_trades['action'] == 'SHORT'])
        else:
            live_longs = live_shorts = 0

        if 'action' in backtest_trades.columns and not backtest_trades.empty:
            backtest_longs = len(backtest_trades[backtest_trades['action'] == 'BUY'])
            backtest_shorts = len(backtest_trades[backtest_trades['action'] == 'SHORT'])
        else:
            # Backtest may use 'side' instead of 'action'
            if 'side' in backtest_trades.columns:
                backtest_longs = len(backtest_trades[backtest_trades['side'] == 'LONG'])
                backtest_shorts = len(backtest_trades[backtest_trades['side'] == 'SHORT'])
            else:
                backtest_longs = backtest_shorts = 0

        comparison['direction_analysis'] = {
            'live_longs': live_longs,
            'live_shorts': live_shorts,
            'backtest_longs': backtest_longs,
            'backtest_shorts': backtest_shorts,
            'live_long_pct': round(live_longs / max(len(live_trades), 1) * 100, 1),
            'backtest_long_pct': round(backtest_longs / max(len(backtest_trades), 1) * 100, 1)
        }

        return comparison

    def compare_exits(self, live_trades: pd.DataFrame, backtest_trades: pd.DataFrame) -> Dict:
        """Compare exit reasons between live and backtest."""
        comparison = {
            'live_exit_reasons': {},
            'backtest_exit_reasons': {},
            'exit_reason_alignment': {}
        }

        # Live exit reasons
        if 'exit_reason' in live_trades.columns and not live_trades.empty:
            live_exits = live_trades['exit_reason'].value_counts().to_dict()
            live_total = sum(live_exits.values())
            comparison['live_exit_reasons'] = {
                k: {'count': v, 'pct': round(v / live_total * 100, 1)}
                for k, v in live_exits.items()
            }

        # Backtest exit reasons
        if 'exit_reason' in backtest_trades.columns and not backtest_trades.empty:
            backtest_exits = backtest_trades['exit_reason'].value_counts().to_dict()
            backtest_total = sum(backtest_exits.values())
            comparison['backtest_exit_reasons'] = {
                k: {'count': v, 'pct': round(v / backtest_total * 100, 1)}
                for k, v in backtest_exits.items()
            }

        # Compare exit reason distributions
        all_reasons = set(comparison['live_exit_reasons'].keys()) | set(comparison['backtest_exit_reasons'].keys())
        for reason in all_reasons:
            live_pct = comparison['live_exit_reasons'].get(reason, {}).get('pct', 0)
            backtest_pct = comparison['backtest_exit_reasons'].get(reason, {}).get('pct', 0)
            comparison['exit_reason_alignment'][reason] = {
                'live_pct': live_pct,
                'backtest_pct': backtest_pct,
                'diff_pct': round(abs(live_pct - backtest_pct), 1)
            }

        return comparison

    def compare_pnl(self, live_trades: pd.DataFrame, backtest_trades: pd.DataFrame) -> Dict:
        """Compare P&L metrics between live and backtest."""
        comparison = {
            'live': {},
            'backtest': {},
            'alignment': {}
        }

        # Live P&L metrics
        if 'pnl' in live_trades.columns and not live_trades.empty:
            live_pnl = live_trades['pnl'].dropna()
            live_wins = live_pnl[live_pnl > 0]
            live_losses = live_pnl[live_pnl < 0]

            comparison['live'] = {
                'total_pnl': round(live_pnl.sum(), 2),
                'avg_pnl': round(live_pnl.mean(), 2),
                'median_pnl': round(live_pnl.median(), 2),
                'std_pnl': round(live_pnl.std(), 2),
                'win_count': len(live_wins),
                'loss_count': len(live_losses),
                'win_rate': round(len(live_wins) / max(len(live_pnl), 1) * 100, 1),
                'avg_win': round(live_wins.mean(), 2) if len(live_wins) > 0 else 0,
                'avg_loss': round(live_losses.mean(), 2) if len(live_losses) > 0 else 0,
                'best_trade': round(live_pnl.max(), 2),
                'worst_trade': round(live_pnl.min(), 2),
                'profit_factor': round(
                    abs(live_wins.sum() / live_losses.sum()), 2
                ) if len(live_losses) > 0 and live_losses.sum() != 0 else 999.99
            }

        # Backtest P&L metrics
        if 'pnl' in backtest_trades.columns and not backtest_trades.empty:
            backtest_pnl = backtest_trades['pnl'].dropna()
            backtest_wins = backtest_pnl[backtest_pnl > 0]
            backtest_losses = backtest_pnl[backtest_pnl < 0]

            comparison['backtest'] = {
                'total_pnl': round(backtest_pnl.sum(), 2),
                'avg_pnl': round(backtest_pnl.mean(), 2),
                'median_pnl': round(backtest_pnl.median(), 2),
                'std_pnl': round(backtest_pnl.std(), 2),
                'win_count': len(backtest_wins),
                'loss_count': len(backtest_losses),
                'win_rate': round(len(backtest_wins) / max(len(backtest_pnl), 1) * 100, 1),
                'avg_win': round(backtest_wins.mean(), 2) if len(backtest_wins) > 0 else 0,
                'avg_loss': round(backtest_losses.mean(), 2) if len(backtest_losses) > 0 else 0,
                'best_trade': round(backtest_pnl.max(), 2),
                'worst_trade': round(backtest_pnl.min(), 2),
                'profit_factor': round(
                    abs(backtest_wins.sum() / backtest_losses.sum()), 2
                ) if len(backtest_losses) > 0 and backtest_losses.sum() != 0 else 999.99
            }

        # Alignment metrics
        if comparison['live'] and comparison['backtest']:
            comparison['alignment'] = {
                'win_rate_diff': round(
                    abs(comparison['live']['win_rate'] - comparison['backtest']['win_rate']), 1
                ),
                'avg_pnl_diff': round(
                    abs(comparison['live']['avg_pnl'] - comparison['backtest']['avg_pnl']), 2
                ),
                'total_pnl_ratio': round(
                    comparison['live']['total_pnl'] / comparison['backtest']['total_pnl'], 2
                ) if comparison['backtest']['total_pnl'] != 0 else 0,
                'trade_count_ratio': round(
                    len(live_trades) / max(len(backtest_trades), 1), 2
                )
            }

        return comparison

    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on comparison."""
        recommendations = []

        # Scanner recommendations
        scanner = self.report.get('scanner_comparison', {})
        if scanner.get('avg_overlap_pct', 100) < 70:
            recommendations.append(
                f"SCANNER DIVERGENCE: Only {scanner['avg_overlap_pct']}% average overlap. "
                "Check data timing differences between live and backtest."
            )

        # Exit reason recommendations
        exits = self.report.get('exit_analysis', {})
        alignment = exits.get('exit_reason_alignment', {})
        for reason, data in alignment.items():
            if data.get('diff_pct', 0) > 15:
                recommendations.append(
                    f"EXIT DIVERGENCE: '{reason}' differs by {data['diff_pct']}% "
                    f"(live: {data['live_pct']}%, backtest: {data['backtest_pct']}%). "
                    "Investigate exit logic timing."
                )

        # P&L recommendations
        pnl = self.report.get('pnl_analysis', {})
        alignment = pnl.get('alignment', {})

        if alignment.get('win_rate_diff', 0) > 10:
            recommendations.append(
                f"WIN RATE DIVERGENCE: {alignment['win_rate_diff']}% difference. "
                "Check entry signal timing and data quality."
            )

        if alignment.get('trade_count_ratio', 1) < 0.5 or alignment.get('trade_count_ratio', 1) > 2:
            recommendations.append(
                f"TRADE COUNT DIVERGENCE: Ratio is {alignment['trade_count_ratio']}x. "
                "Check if live bot was running continuously."
            )

        # Trade comparison recommendations
        trades = self.report.get('trade_comparison', {})
        if trades.get('live_trade_count', 0) == 0:
            recommendations.append(
                "NO LIVE TRADES: No closed trades found in the period. "
                "Verify the bot was running and the date range is correct."
            )

        if not recommendations:
            recommendations.append(
                "GOOD ALIGNMENT: No major divergences detected. "
                "Backtest appears to represent live trading accurately."
            )

        return recommendations

    def run_comparison(self, start_date: str, end_date: str,
                       run_backtest: bool = True) -> Dict:
        """Run full comparison between live and backtest."""
        logger.info(f"Starting comparison for {start_date} to {end_date}")

        # Load live data
        live_trades = self.load_live_trades(start_date, end_date)
        live_scans = self.load_live_scans(start_date, end_date)

        # Run or load backtest
        if run_backtest:
            backtest_trades = self.run_backtest(start_date, end_date)
            backtest_scans = self.load_backtest_scans(start_date, end_date)
        else:
            backtest_trades = pd.DataFrame()
            backtest_scans = self.load_backtest_scans(start_date, end_date)

        # Update summary
        self.report['summary'] = {
            'comparison_period': f"{start_date} to {end_date}",
            'live_trades': len(live_trades),
            'backtest_trades': len(backtest_trades),
            'live_scan_days': len(live_scans),
            'backtest_scan_days': len(backtest_scans)
        }

        # Run comparisons
        self.report['scanner_comparison'] = self.compare_scanners(live_scans, backtest_scans)
        self.report['trade_comparison'] = self.compare_trades(live_trades, backtest_trades)
        self.report['exit_analysis'] = self.compare_exits(live_trades, backtest_trades)
        self.report['pnl_analysis'] = self.compare_pnl(live_trades, backtest_trades)

        # Generate recommendations
        self.report['recommendations'] = self.generate_recommendations()

        return self.report

    def print_report(self):
        """Print a formatted summary of the comparison."""
        print("\n" + "=" * 70)
        print("LIVE vs BACKTEST COMPARISON REPORT")
        print("=" * 70)

        # Summary
        summary = self.report.get('summary', {})
        print(f"\nPeriod: {summary.get('comparison_period', 'N/A')}")
        print(f"Live Trades: {summary.get('live_trades', 0)}")
        print(f"Backtest Trades: {summary.get('backtest_trades', 0)}")

        # Scanner comparison
        print("\n" + "-" * 40)
        print("SCANNER ALIGNMENT")
        print("-" * 40)
        scanner = self.report.get('scanner_comparison', {})
        print(f"Days Compared: {scanner.get('dates_compared', 0)}")
        print(f"Average Overlap: {scanner.get('avg_overlap_pct', 0)}%")
        print(f"Min/Max Overlap: {scanner.get('min_overlap_pct', 0)}% / {scanner.get('max_overlap_pct', 0)}%")

        # Exit analysis
        print("\n" + "-" * 40)
        print("EXIT REASON COMPARISON")
        print("-" * 40)
        exits = self.report.get('exit_analysis', {})
        print("\nLive Exit Reasons:")
        for reason, data in exits.get('live_exit_reasons', {}).items():
            print(f"  {reason}: {data['count']} ({data['pct']}%)")
        print("\nBacktest Exit Reasons:")
        for reason, data in exits.get('backtest_exit_reasons', {}).items():
            print(f"  {reason}: {data['count']} ({data['pct']}%)")

        # P&L analysis
        print("\n" + "-" * 40)
        print("P&L COMPARISON")
        print("-" * 40)
        pnl = self.report.get('pnl_analysis', {})

        live_pnl = pnl.get('live', {})
        backtest_pnl = pnl.get('backtest', {})

        print(f"\n{'Metric':<20} {'Live':>12} {'Backtest':>12} {'Diff':>12}")
        print("-" * 56)

        metrics = ['total_pnl', 'avg_pnl', 'win_rate', 'profit_factor', 'best_trade', 'worst_trade']
        for metric in metrics:
            live_val = live_pnl.get(metric, 0)
            backtest_val = backtest_pnl.get(metric, 0)
            diff = abs(live_val - backtest_val) if isinstance(live_val, (int, float)) else 'N/A'
            print(f"{metric:<20} {live_val:>12} {backtest_val:>12} {diff:>12}")

        # Recommendations
        print("\n" + "-" * 40)
        print("RECOMMENDATIONS")
        print("-" * 40)
        for rec in self.report.get('recommendations', []):
            print(f"\n* {rec}")

        print("\n" + "=" * 70)

    def save_report(self, output_file: str):
        """Save full report to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)
        logger.info(f"Report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare live trading with backtest results')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--live-db', type=str, default='logs/trades.db',
                        help='Path to live trades database')
    parser.add_argument('--output', type=str, default='comparison_report.json',
                        help='Output JSON file for full report')
    parser.add_argument('--no-backtest', action='store_true',
                        help='Skip running backtest (use existing backtest scan data)')

    args = parser.parse_args()

    # Default to last 14 days if no dates provided
    if not args.end:
        args.end = datetime.now().strftime('%Y-%m-%d')
    if not args.start:
        start_dt = datetime.strptime(args.end, '%Y-%m-%d') - timedelta(days=14)
        args.start = start_dt.strftime('%Y-%m-%d')

    # Run comparison
    comparator = LiveBacktestComparator(live_db=args.live_db)
    report = comparator.run_comparison(
        start_date=args.start,
        end_date=args.end,
        run_backtest=not args.no_backtest
    )

    # Print and save results
    comparator.print_report()
    comparator.save_report(args.output)

    print(f"\nFull report saved to: {args.output}")


if __name__ == '__main__':
    main()
