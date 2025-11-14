#!/usr/bin/env python3
"""Build a coverage attribution database.

This script processes all individual test coverage databases and builds
a comprehensive database that tracks:
- Which tests cover which blocks/instances
- Which blocks/instances are covered by which tests
- Unique coverage contributions per test

The output can be saved as JSON or SQLite for efficient querying.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

# Discover the Ibex DV python modules
CORE_IBEX_SCRIPTS = Path(__file__).resolve().parent
IBEX_ROOT = CORE_IBEX_SCRIPTS.parent.parent.parent.parent
IBEX_UTIL = IBEX_ROOT / "util"
REPORT_LIB = CORE_IBEX_SCRIPTS / "report_lib"

for module_path in (IBEX_UTIL, CORE_IBEX_SCRIPTS, REPORT_LIB, IBEX_ROOT):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

# pylint: disable=wrong-import-position
try:
    import pathlib3x as pathlib3x
    from analyze_single_test_coverage import (
        analyze_single_test_coverage,  # type: ignore
    )
    from merge_cov import find_cov_dbs  # type: ignore
    from metadata import RegressionMetadata  # type: ignore
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    sys.exit(1)


class CoverageDatabase:
    """Coverage attribution database."""

    def __init__(self):
        self.test_to_blocks: Dict[str, Set[str]] = defaultdict(set)
        self.block_to_tests: Dict[str, Set[str]] = defaultdict(set)
        self.test_metrics: Dict[str, Dict[str, float]] = {}
        # New: Store instance-level coverage
        # Format: {test_name: {instance_name: {metric_name: {covered, total, pct}}}}
        self.test_instance_coverage: Dict[
            str, Dict[str, Dict[str, Dict[str, float]]]
        ] = defaultdict(dict)

    def add_test_coverage(
        self,
        test_name: str,
        blocks: List[str],
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Add coverage data for a test.

        Args:
            test_name: Name of the test (e.g., "test_name.seed")
            blocks: List of covered blocks/instances
            metrics: Optional coverage metrics dictionary
        """
        self.test_to_blocks[test_name].update(blocks)
        for block in blocks:
            self.block_to_tests[block].add(test_name)

        if metrics:
            self.test_metrics[test_name] = metrics

    def add_test_instance_coverage(
        self, test_name: str, instance_coverage: Dict[str, Dict[str, Dict[str, float]]]
    ):
        """Add instance-level coverage data for a test.

        Args:
            test_name: Name of the test
            instance_coverage: Dict mapping instance names to coverage metrics
                Example: {
                    "ibex_core.if_stage_i": {
                        "block": {"covered": 230, "total": 250, "pct": 0.92},
                        ...
                    }
                }
        """
        self.test_instance_coverage[test_name] = instance_coverage

        # Also add instances to the block list (instances are blocks)
        for instance_name in instance_coverage.keys():
            self.test_to_blocks[test_name].add(instance_name)
            self.block_to_tests[instance_name].add(test_name)

    def get_tests_covering_block(self, block: str) -> Set[str]:
        """Get all tests that cover a specific block."""
        return self.block_to_tests.get(block, set())

    def get_blocks_covered_by_test(self, test: str) -> Set[str]:
        """Get all blocks covered by a specific test."""
        return self.test_to_blocks.get(test, set())

    def get_unique_coverage_contributors(self) -> List[tuple[str, int]]:
        """Get tests ranked by their unique coverage contribution.

        Returns:
            List of (test_name, unique_blocks_count) tuples, sorted by count
        """
        unique_contributions = []
        for test, blocks in self.test_to_blocks.items():
            # Count blocks that only this test covers
            unique_count = sum(
                1 for block in blocks if len(self.block_to_tests[block]) == 1
            )
            unique_contributions.append((test, unique_count))

        return sorted(unique_contributions, key=lambda x: x[1], reverse=True)

    def save_to_json(self, output_path: Path):
        """Save database to JSON format."""
        data = {
            "test_to_blocks": {k: list(v) for k, v in self.test_to_blocks.items()},
            "block_to_tests": {k: list(v) for k, v in self.block_to_tests.items()},
            "test_metrics": self.test_metrics,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Coverage database saved to: {output_path}")

    def save_to_sqlite(self, output_path: Path):
        """Save database to SQLite format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(output_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tests (
                test_name TEXT PRIMARY KEY,
                total_blocks INTEGER
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blocks (
                block_name TEXT PRIMARY KEY,
                covered_by_count INTEGER
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_block_coverage (
                test_name TEXT,
                block_name TEXT,
                PRIMARY KEY (test_name, block_name),
                FOREIGN KEY (test_name) REFERENCES tests(test_name),
                FOREIGN KEY (block_name) REFERENCES blocks(block_name)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_metrics (
                test_name TEXT,
                metric_name TEXT,
                value REAL,
                PRIMARY KEY (test_name, metric_name),
                FOREIGN KEY (test_name) REFERENCES tests(test_name)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_instance_coverage (
                test_name TEXT,
                instance_name TEXT,
                metric_name TEXT,
                covered REAL,
                total REAL,
                percentage REAL,
                PRIMARY KEY (test_name, instance_name, metric_name),
                FOREIGN KEY (test_name) REFERENCES tests(test_name)
            )
        """)

        # Insert data
        for test, blocks in self.test_to_blocks.items():
            cursor.execute(
                "INSERT OR REPLACE INTO tests (test_name, total_blocks) VALUES (?, ?)",
                (test, len(blocks)),
            )

            for block in blocks:
                cursor.execute(
                    "INSERT OR REPLACE INTO test_block_coverage (test_name, block_name) VALUES (?, ?)",
                    (test, block),
                )

        for block, tests in self.block_to_tests.items():
            cursor.execute(
                "INSERT OR REPLACE INTO blocks (block_name, covered_by_count) VALUES (?, ?)",
                (block, len(tests)),
            )

        for test, metrics in self.test_metrics.items():
            for metric_name, value in metrics.items():
                cursor.execute(
                    "INSERT OR REPLACE INTO test_metrics (test_name, metric_name, value) VALUES (?, ?, ?)",
                    (test, metric_name, value),
                )

        # Insert instance coverage data
        for test, instances in self.test_instance_coverage.items():
            for instance_name, metrics in instances.items():
                for metric_name, metric_data in metrics.items():
                    covered = metric_data.get("covered", 0.0)
                    total = metric_data.get("total", 0.0)
                    pct = metric_data.get("pct", metric_data.get("average", 0.0))

                    cursor.execute(
                        """INSERT OR REPLACE INTO test_instance_coverage
                           (test_name, instance_name, metric_name, covered, total, percentage)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (test, instance_name, metric_name, covered, total, pct),
                    )

        # Create indexes for faster queries
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_test_block ON test_block_coverage(test_name)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_block_test ON test_block_coverage(block_name)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_instance_test ON test_instance_coverage(test_name)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_instance_name ON test_instance_coverage(instance_name)"
        )

        conn.commit()
        conn.close()

        print(f"Coverage database saved to: {output_path}")

    def print_summary(self):
        """Print summary statistics."""
        print("\nCoverage Database Summary:")
        print("=" * 60)
        print(f"Total tests: {len(self.test_to_blocks)}")
        print(f"Total blocks tracked: {len(self.block_to_tests)}")

        if self.test_to_blocks:
            blocks_per_test = [len(blocks) for blocks in self.test_to_blocks.values()]
            print(
                f"Blocks per test (avg): {sum(blocks_per_test) / len(blocks_per_test):.1f}"
            )
            print(f"Blocks per test (max): {max(blocks_per_test)}")
            print(f"Blocks per test (min): {min(blocks_per_test)}")

        if self.block_to_tests:
            tests_per_block = [len(tests) for tests in self.block_to_tests.values()]
            print(
                f"Tests per block (avg): {sum(tests_per_block) / len(tests_per_block):.1f}"
            )
            print(f"Tests per block (max): {max(tests_per_block)}")
            print(f"Tests per block (min): {min(tests_per_block)}")

            # Blocks covered by only one test
            unique_blocks = sum(
                1 for tests in self.block_to_tests.values() if len(tests) == 1
            )
            print(f"Blocks covered by only one test: {unique_blocks}")

        # Top unique contributors
        top_contributors = self.get_unique_coverage_contributors()[:10]
        if top_contributors:
            print("\nTop 10 tests by unique coverage contribution:")
            for i, (test, count) in enumerate(top_contributors, 1):
                print(f"  {i:2d}. {test:40s}: {count:4d} unique blocks")


def filter_instances_by_granularity(
    instance_coverage: Dict[str, Dict[str, Dict[str, float]]], granularity: str
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Filter instance coverage based on granularity level.

    Args:
        instance_coverage: Full instance coverage dictionary
        granularity: One of 'pipeline', 'functional', or 'full'

    Returns:
        Filtered instance coverage dictionary
    """
    if granularity == "full":
        # Return all instances
        return instance_coverage

    # Define pipeline stage instances (Level 1)
    pipeline_stages = {
        "if_stage_i",
        "id_stage_i",
        "ex_block_i",
        "load_store_unit_i",
        "wb_stage_i",
        "cs_registers_i",
    }

    # Define functional unit instances (Level 2) - includes pipeline + more
    functional_units = pipeline_stages | {
        "alu_i",
        "decoder_i",
        "mult_div_i",
        "controller_i",
        "prefetch_buffer_i",
        "compressed_decoder_i",
    }

    # Filter based on granularity
    filtered = {}
    target_set = pipeline_stages if granularity == "pipeline" else functional_units

    for instance_name, metrics in instance_coverage.items():
        # Check if the instance name ends with any of our target instances
        # (handles full hierarchical paths like "ibex_core.if_stage_i")
        if any(instance_name.endswith(target) for target in target_set):
            filtered[instance_name] = metrics

    return filtered


def process_test_coverage(test_dir: Path, granularity: str) -> Optional[Dict]:
    """Process a single test's coverage data.

    Args:
        test_dir: Directory containing test coverage data
        granularity: Coverage granularity level

    Returns:
        Dictionary with test name, instances, and metrics, or None if failed
    """
    test_name = test_dir.name
    json_file = test_dir / "coverage_metrics.json"

    # Check if coverage metrics JSON exists
    if not json_file.exists():
        # Try to find .ucd file and run analysis
        ucd_file = None
        for pattern in ["*.ucd", "coverage/*.ucd", "*/coverage.ucd"]:
            matches = list(test_dir.glob(pattern))
            if matches:
                ucd_file = matches[0]
                break

        if ucd_file and ucd_file.exists():
            print(f"    Generating coverage report for {test_name}...")
            # Call analyze_single_test_coverage directly
            try:
                result = analyze_single_test_coverage(
                    ucd_path=ucd_file,
                    output_dir=test_dir,
                    imc_bin="/opt/coe/cadence/VMANAGER/bin/imc",
                    dut_top="ibex_core",
                    waiver_script=None,
                )

                # Save JSON output
                with json_file.open("w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)

            except Exception as e:
                print(f"    Warning: Failed to analyze {test_name}: {e}")
                return None
        else:
            # No UCD file found, skip this test
            return None

    # Load coverage metrics JSON
    try:
        with json_file.open() as f:
            data = json.load(f)
    except Exception as e:
        print(f"    Warning: Failed to load JSON for {test_name}: {e}")
        return None

    # Extract and filter instance coverage
    instance_coverage = data.get("instance_coverage", {})
    filtered_instances = filter_instances_by_granularity(instance_coverage, granularity)

    return {
        "test_name": test_name,
        "instance_coverage": filtered_instances,
        "metrics": data.get("metrics", {}),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build coverage attribution database from test coverage data."
    )

    # Define default metadata path
    default_metadata = CORE_IBEX_SCRIPTS.parent / "out" / "metadata"

    parser.add_argument(
        "--metadata",
        type=Path,
        default=default_metadata,
        help=f"Path to regression metadata directory (default: {default_metadata})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output SQLite database file path",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["pipeline", "functional", "full"],
        default="functional",
        help="Coverage granularity level: pipeline (major stages), functional (includes ALU/decoder), full (all instances)",
    )

    args = parser.parse_args(argv)

    # Load metadata
    try:
        metadata_dir = pathlib3x.Path(str(args.metadata.resolve()))
        md = RegressionMetadata.construct_from_metadata_dir(metadata_dir)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return 1

    print(f"Building coverage database from regression: {md.dir_run}")
    print(f"Granularity level: {args.granularity}")

    # Find all coverage databases using find_cov_dbs
    run_dir = pathlib3x.Path(str(md.dir_run))
    cov_dbs = find_cov_dbs(run_dir, md.simulator)

    if not cov_dbs:
        print(f"Error: No coverage databases found in {run_dir}")
        return 1

    print(f"Found {len(cov_dbs)} coverage databases")

    # Process all tests
    db = CoverageDatabase()
    processed_count = 0
    failed_count = 0

    for i, ucd_path in enumerate(sorted(cov_dbs), 1):
        # Get test directory from .ucd path (parent directory)
        test_dir = Path(str(ucd_path.parent))
        print(f"[{i}/{len(cov_dbs)}] Processing {test_dir.name}...")

        result = process_test_coverage(test_dir, args.granularity)

        if result:
            # Add to database
            db.add_test_instance_coverage(
                result["test_name"], result["instance_coverage"]
            )

            # Also add metrics
            if result["metrics"]:
                # Get just the top-level metrics (percentages), convert to float
                metrics = {
                    k: float(v)
                    for k, v in result["metrics"].items()
                    if isinstance(v, (int, float))
                }
                db.test_metrics[result["test_name"]] = metrics

            processed_count += 1
        else:
            failed_count += 1

    # Save database
    print(
        f"\nProcessing complete: {processed_count} tests processed, {failed_count} skipped"
    )

    if processed_count == 0:
        print("Error: No tests were successfully processed")
        return 1

    db.save_to_sqlite(args.output)
    db.print_summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())
