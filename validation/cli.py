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
