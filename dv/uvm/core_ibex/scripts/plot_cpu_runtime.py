#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Standalone script to generate CPU runtime plots from regression test results.

This script can be run independently to generate or regenerate matplotlib plots
showing CPU runtime analysis for regression tests.

Usage:
    ./plot_cpu_runtime.py --dir-metadata <path_to_metadata_dir>

Example:
    ./plot_cpu_runtime.py --dir-metadata out/metadata
"""

import argparse
import sys
import os
import pathlib3x as pathlib
from typing import List

# Add the scripts directory to Python path to enable imports
script_dir = pathlib.Path(__file__).parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from metadata import RegressionMetadata, LockedMetadata
from test_run_result import TestRunResult
from report_lib.matplotlib_plots import generate_all_runtime_plots

import logging
logging.basicConfig(level=logging.INFO,
                   format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main() -> int:
    """Generate CPU runtime plots from test results."""
    parser = argparse.ArgumentParser(
        description='Generate CPU runtime plots from regression test results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plots from the default metadata location
  %(prog)s --dir-metadata out/metadata

  # Generate plots with custom output directory
  %(prog)s --dir-metadata out/metadata --output-dir custom_plots/

  # Verbose output
  %(prog)s --dir-metadata out/metadata --verbose
        """
    )

    parser.add_argument('--dir-metadata',
                       type=pathlib.Path,
                       required=True,
                       help='Path to the metadata directory containing test results')

    parser.add_argument('--output-dir',
                       type=pathlib.Path,
                       default=None,
                       help='Output directory for plots (default: same as dir_run in metadata)')

    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate metadata directory
    if not args.dir_metadata.exists():
        logger.error(f"Metadata directory does not exist: {args.dir_metadata}")
        return 1

    logger.info(f"Loading test results from {args.dir_metadata}")

    # Load metadata
    with LockedMetadata(args.dir_metadata, __file__) as md:
        # Collect all test results
        all_tests = []
        tests_with_runtime = 0
        tests_without_runtime = 0

        for f in md.tests_pickle_files:
            try:
                trr = TestRunResult.construct_from_pickle(f)
                all_tests.append(trr)

                if trr.runtime_s is not None:
                    tests_with_runtime += 1
                else:
                    tests_without_runtime += 1

            except RuntimeError as e:
                logger.warning(f"Failed to load test result from {f}: {e}")
                continue

        # Report statistics
        total_tests = len(all_tests)
        logger.info(f"Loaded {total_tests} test results")
        logger.info(f"  Tests with runtime data: {tests_with_runtime}")
        logger.info(f"  Tests without runtime data: {tests_without_runtime}")

        if tests_with_runtime == 0:
            logger.error("No tests have runtime data. Cannot generate plots.")
            logger.info("Runtime data is captured during RTL simulation.")
            logger.info("Make sure tests have been run with the updated run_rtl.py")
            return 1

        if tests_without_runtime > 0:
            logger.warning(f"{tests_without_runtime} tests are missing runtime data")

        # Determine output directory
        output_dir = args.output_dir if args.output_dir else md.dir_run

        logger.info(f"Generating plots in {output_dir}")

        # Generate plots
        generate_all_runtime_plots(all_tests, output_dir)

        logger.info("Plot generation complete!")
        logger.info(f"Generated plots:")
        for plot_name in ['runtime_histogram.png', 'runtime_by_test.png',
                         'runtime_scatter.png', 'cumulative_runtime.png']:
            plot_path = output_dir / plot_name
            if plot_path.exists():
                logger.info(f"  - {plot_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
