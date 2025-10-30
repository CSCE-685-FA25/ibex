#!/usr/bin/env python3
"""Analyze and visualize extracted features for ML model development.

This script provides utilities for exploring the enhanced feature dataset,
computing statistics, and preparing data for ML training.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter


def load_features(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load features from JSONL file."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def analyze_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute dataset statistics."""
    stats = {
        "total_records": len(records),
        "label_distribution": Counter(),
        "test_type_distribution": Counter(),
        "pass_fail_distribution": Counter(),
        "feature_availability": {},
        "coverage_stats": {},
        "execution_stats": {},
    }

    # Collect statistics
    for record in records:
        # Label distribution
        label = record.get("label", 0)
        stats["label_distribution"][label] += 1

        # Test type distribution
        test_type = record.get("test_metadata", {}).get("test_type")
        if test_type:
            stats["test_type_distribution"][test_type] += 1

        # Pass/fail distribution
        passed = record.get("test_metadata", {}).get("passed")
        if passed is not None:
            stats["pass_fail_distribution"]["passed" if passed else "failed"] += 1

    # Feature availability
    if records:
        sample = records[0]
        stats["feature_availability"] = {
            "test_metadata": bool(sample.get("test_metadata")),
            "execution_features": bool(sample.get("execution_features")),
            "derived_features": bool(sample.get("derived_features")),
        }

        # List available features
        stats["available_features"] = {
            "test_metadata_keys": list(sample.get("test_metadata", {}).keys()),
            "execution_features_keys": list(sample.get("execution_features", {}).keys()),
            "derived_features_keys": list(sample.get("derived_features", {}).keys()),
        }

    # Coverage statistics
    coverage_deltas = []
    for record in records:
        deltas = record.get("covered_deltas", {})
        if "block" in deltas:
            coverage_deltas.append(deltas["block"])

    if coverage_deltas:
        stats["coverage_stats"] = {
            "mean_block_delta": sum(coverage_deltas) / len(coverage_deltas),
            "max_block_delta": max(coverage_deltas),
            "min_block_delta": min(coverage_deltas),
            "tests_with_contribution": sum(1 for d in coverage_deltas if d > 0),
        }

    # Execution statistics
    instr_counts = []
    for record in records:
        exec_feat = record.get("execution_features", {})
        count = exec_feat.get("trace_instruction_count")
        if count:
            instr_counts.append(count)

    if instr_counts:
        stats["execution_stats"] = {
            "mean_instruction_count": sum(instr_counts) / len(instr_counts),
            "max_instruction_count": max(instr_counts),
            "min_instruction_count": min(instr_counts),
        }

    return stats


def print_statistics(stats: Dict[str, Any]) -> None:
    """Print dataset statistics in readable format."""
    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"\nTotal Records: {stats['total_records']}")

    print("\nLabel Distribution (Coverage Contribution):")
    for label, count in sorted(stats["label_distribution"].items()):
        pct = count / stats["total_records"] * 100
        label_name = "Contributes" if label == 1 else "No contribution"
        print(f"  {label_name}: {count} ({pct:.1f}%)")

    if stats["test_type_distribution"]:
        print("\nTest Type Distribution:")
        for test_type, count in sorted(stats["test_type_distribution"].items()):
            pct = count / stats["total_records"] * 100
            print(f"  {test_type}: {count} ({pct:.1f}%)")

    if stats["pass_fail_distribution"]:
        print("\nPass/Fail Distribution:")
        for result, count in sorted(stats["pass_fail_distribution"].items()):
            pct = count / stats["total_records"] * 100
            print(f"  {result}: {count} ({pct:.1f}%)")

    print("\nFeature Availability:")
    for feature_group, available in stats.get("feature_availability", {}).items():
        status = "Available" if available else "Missing"
        print(f"  {feature_group}: {status}")

    if stats.get("available_features"):
        print("\nFeature Groups:")
        for group, keys in stats["available_features"].items():
            print(f"  {group}: {len(keys)} features")
            if keys:
                print(f"    {', '.join(keys[:5])}{'...' if len(keys) > 5 else ''}")

    if stats.get("coverage_stats"):
        print("\nCoverage Statistics (Block Coverage):")
        cov = stats["coverage_stats"]
        print(f"  Mean delta: {cov.get('mean_block_delta', 0):.2f} blocks")
        print(f"  Max delta: {cov.get('max_block_delta', 0):.0f} blocks")
        print(f"  Min delta: {cov.get('min_block_delta', 0):.0f} blocks")
        pct = cov.get('tests_with_contribution', 0) / stats['total_records'] * 100
        print(f"  Tests with contribution: {cov.get('tests_with_contribution', 0)} ({pct:.1f}%)")

    if stats.get("execution_stats"):
        print("\nExecution Statistics:")
        exec_stats = stats["execution_stats"]
        print(f"  Mean instruction count: {exec_stats.get('mean_instruction_count', 0):.0f}")
        print(f"  Max instruction count: {exec_stats.get('max_instruction_count', 0):.0f}")
        print(f"  Min instruction count: {exec_stats.get('min_instruction_count', 0):.0f}")

    print("\n" + "=" * 80)


def export_feature_matrix(
    records: List[Dict[str, Any]],
    output_path: Path,
    format: str = "csv"
) -> None:
    """Export flattened feature matrix for ML training."""
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required for feature matrix export.")
        print("Install with: pip install pandas")
        return

    # Flatten records
    flattened = []
    for record in records:
        flat = {
            "testdotseed": record.get("testdotseed"),
            "label": record.get("label"),
            "covergroup_delta": record.get("covergroup_delta"),
        }

        # Add coverage deltas
        for metric, delta in record.get("covered_deltas", {}).items():
            flat[f"delta_{metric}"] = delta

        # Add test metadata
        for key, value in record.get("test_metadata", {}).items():
            flat[f"meta_{key}"] = value

        # Add execution features
        for key, value in record.get("execution_features", {}).items():
            flat[f"exec_{key}"] = value

        # Add derived features
        for key, value in record.get("derived_features", {}).items():
            flat[f"derived_{key}"] = value

        flattened.append(flat)

    df = pd.DataFrame(flattened)

    # Convert boolean columns
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass

    # Save based on format
    if format == "csv":
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} records to {output_path}")
    elif format == "parquet":
        df.to_parquet(output_path, index=False)
        print(f"Exported {len(df)} records to {output_path}")
    elif format == "json":
        df.to_json(output_path, orient="records", lines=True)
        print(f"Exported {len(df)} records to {output_path}")

    print(f"Feature matrix shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


def show_sample_records(records: List[Dict[str, Any]], n: int = 3) -> None:
    """Display sample records."""
    print("\n" + "=" * 80)
    print(f"SAMPLE RECORDS (first {n})")
    print("=" * 80)

    for i, record in enumerate(records[:n], 1):
        print(f"\nRecord {i}: {record.get('testdotseed', 'unknown')}")
        print(f"  Label: {record.get('label')} (triggers: {record.get('triggers', [])})")

        # Coverage
        deltas = record.get("covered_deltas", {})
        print(f"  Coverage delta (block): {deltas.get('block', 0):.0f}")

        # Test metadata
        meta = record.get("test_metadata", {})
        print(f"  Test type: {meta.get('test_type', 'N/A')}")
        print(f"  Passed: {meta.get('passed', 'N/A')}")
        print(f"  Seed: {meta.get('seed', 'N/A')}")

        # Execution features
        exec_feat = record.get("execution_features", {})
        print(f"  Instructions: {exec_feat.get('trace_instruction_count', 'N/A')}")
        print(f"  Cycles: {exec_feat.get('cycle_count', 'N/A')}")

        # Derived features
        derived = record.get("derived_features", {})
        print(f"  Branch ratio: {derived.get('branch_ratio', 0):.3f}")
        print(f"  Control flow complexity: {derived.get('control_flow_complexity', 0):.3f}")
        print(f"  CPI: {derived.get('cpi', 'N/A')}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze enhanced feature dataset for ML development."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSONL file with enhanced features.",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Export flattened feature matrix to file.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet", "json"],
        default="csv",
        help="Export format (requires pandas).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of sample records to display.",
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Skip displaying sample records.",
    )

    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist.")
        return 1

    # Load features
    print(f"Loading features from {args.input}...")
    records = load_features(args.input)

    if not records:
        print("Error: No records found in input file.")
        return 1

    print(f"Loaded {len(records)} records.\n")

    # Analyze dataset
    stats = analyze_dataset(records)
    print_statistics(stats)

    # Show samples
    if not args.no_samples:
        show_sample_records(records, args.samples)

    # Export feature matrix
    if args.export:
        print(f"\nExporting feature matrix...")
        export_feature_matrix(records, args.export, args.format)

    return 0


if __name__ == "__main__":
    sys.exit(main())
