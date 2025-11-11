#!/usr/bin/env python3
"""Extract additional features from test results and augment coverage labels.

This script reads the coverage_labels.jsonl produced by coverage_labeler.py
and augments each record with additional features extracted from:
- Test metadata (TestRunResult pickle files)
- Simulation logs (runtime, instruction counts)
- Trace CSV files (instruction mix, control flow)

The output is an enhanced JSONL file suitable for ML model training.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# Discover the Ibex DV python modules
# This script is now located in dv/uvm/core_ibex/scripts/
CORE_IBEX_SCRIPTS = Path(__file__).resolve().parent
IBEX_ROOT = CORE_IBEX_SCRIPTS.parent.parent.parent.parent
IBEX_UTIL = IBEX_ROOT / "util"

# Ensure the common ibex utility dir is on sys.path
for module_path in (IBEX_UTIL, CORE_IBEX_SCRIPTS, IBEX_ROOT):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

# Import after path setup
try:
    import pathlib3x as pathlib3x
    from metadata import RegressionMetadata  # type: ignore
    from test_run_result import Failure_Modes, TestRunResult, TestType  # type: ignore
except ImportError as e:
    print(f"Warning: Could not import ibex modules: {e}")
    print("Feature extraction will be limited to log parsing only.")
    RegressionMetadata = None
    TestRunResult = None
    TestType = None
    Failure_Modes = None


# Regex patterns for log parsing
INSTR_COUNT_RE = re.compile(r"(\d+)\s+instructions")
EXCEPTION_RE = re.compile(r"exception|interrupt", re.IGNORECASE)
CYCLE_COUNT_RE = re.compile(r"(\d+)\s+cycles")
TIME_RE = re.compile(r"Time:\s+([\d.]+)\s*(\w+)")


def extract_test_metadata_features(test_result: Any) -> Dict[str, Any]:
    """Extract features from TestRunResult metadata."""
    if test_result is None or TestRunResult is None:
        return {}

    features = {}

    # Test execution status
    features["passed"] = test_result.passed if test_result.passed is not None else None
    features["failure_mode"] = (
        test_result.failure_mode.name if test_result.failure_mode else None
    )
    features["timeout_s"] = test_result.timeout_s

    # Test configuration
    features["test_type"] = test_result.testtype.name if test_result.testtype else None
    features["test_name"] = test_result.testname
    features["seed"] = test_result.seed
    features["simulator"] = test_result.rtl_simulator
    features["iss"] = test_result.iss_cosim

    # Test options (for RISCV-DV tests)
    if test_result.gen_opts:
        features["gen_opts_count"] = len(test_result.gen_opts)
    else:
        features["gen_opts_count"] = 0

    if test_result.sim_opts:
        features["sim_opts_count"] = len(test_result.sim_opts)
    else:
        features["sim_opts_count"] = 0

    return features


def parse_simulation_log(log_path: Path) -> Dict[str, Any]:
    """Extract features from simulation log file."""
    features = {
        "log_exists": False,
        "log_size_bytes": 0,
        "instruction_count_from_log": None,
        "cycle_count": None,
        "exception_count": 0,
        "log_line_count": 0,
    }

    if not log_path or not log_path.exists():
        return features

    features["log_exists"] = True
    features["log_size_bytes"] = log_path.stat().st_size

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            features["log_line_count"] = len(lines)

            for line in lines:
                # Look for instruction count
                m = INSTR_COUNT_RE.search(line)
                if m:
                    features["instruction_count_from_log"] = int(m.group(1))

                # Look for cycle count
                m = CYCLE_COUNT_RE.search(line)
                if m:
                    features["cycle_count"] = int(m.group(1))

                # Count exceptions/interrupts
                if EXCEPTION_RE.search(line):
                    features["exception_count"] += 1

    except Exception as e:
        print(f"Warning: Error parsing log {log_path}: {e}")

    return features


def parse_trace_csv(csv_path: Path) -> Dict[str, Any]:
    """Extract features from instruction trace CSV."""
    features = {
        "trace_exists": False,
        "trace_instruction_count": 0,
        "trace_branch_count": 0,
        "trace_load_count": 0,
        "trace_store_count": 0,
        "trace_arithmetic_count": 0,
        "trace_jump_count": 0,
        "trace_system_count": 0,
        "trace_unique_pcs": 0,
    }

    if not csv_path or not csv_path.exists():
        return features

    features["trace_exists"] = True

    # Instruction type patterns
    branch_instrs = {"beq", "bne", "blt", "bge", "bltu", "bgeu"}
    load_instrs = {"lb", "lh", "lw", "lbu", "lhu"}
    store_instrs = {"sb", "sh", "sw"}
    jump_instrs = {"jal", "jalr"}
    system_instrs = {"ecall", "ebreak", "mret", "wfi", "csrrw", "csrrs", "csrrc"}
    arithmetic_instrs = {
        "add",
        "sub",
        "and",
        "or",
        "xor",
        "sll",
        "srl",
        "sra",
        "addi",
        "andi",
        "ori",
        "xori",
        "slli",
        "srli",
        "srai",
        "mul",
        "mulh",
        "div",
        "rem",
    }

    unique_pcs = set()

    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue

                # CSV format from riscv_trace_csv: pc, binary, instr, operands...
                pc = parts[0].strip()
                instr = parts[2].strip().split()[0] if len(parts) > 2 else ""

                if pc:
                    unique_pcs.add(pc)

                features["trace_instruction_count"] += 1

                # Categorize instruction
                if instr in branch_instrs:
                    features["trace_branch_count"] += 1
                elif instr in load_instrs:
                    features["trace_load_count"] += 1
                elif instr in store_instrs:
                    features["trace_store_count"] += 1
                elif instr in jump_instrs:
                    features["trace_jump_count"] += 1
                elif instr in system_instrs:
                    features["trace_system_count"] += 1
                elif any(instr.startswith(a) for a in arithmetic_instrs):
                    features["trace_arithmetic_count"] += 1

        features["trace_unique_pcs"] = len(unique_pcs)

    except Exception as e:
        print(f"Warning: Error parsing trace CSV {csv_path}: {e}")

    return features


def compute_derived_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Compute derived/ratio features from extracted features."""
    derived = {}

    # Instruction mix ratios
    total_instrs = features.get("trace_instruction_count", 0)
    if total_instrs > 0:
        derived["branch_ratio"] = features.get("trace_branch_count", 0) / total_instrs
        derived["load_ratio"] = features.get("trace_load_count", 0) / total_instrs
        derived["store_ratio"] = features.get("trace_store_count", 0) / total_instrs
        derived["memory_ratio"] = (
            features.get("trace_load_count", 0) + features.get("trace_store_count", 0)
        ) / total_instrs
        derived["jump_ratio"] = features.get("trace_jump_count", 0) / total_instrs
        derived["system_ratio"] = features.get("trace_system_count", 0) / total_instrs
        derived["arithmetic_ratio"] = (
            features.get("trace_arithmetic_count", 0) / total_instrs
        )
    else:
        derived["branch_ratio"] = 0.0
        derived["load_ratio"] = 0.0
        derived["store_ratio"] = 0.0
        derived["memory_ratio"] = 0.0
        derived["jump_ratio"] = 0.0
        derived["system_ratio"] = 0.0
        derived["arithmetic_ratio"] = 0.0

    # Control flow complexity (unique PCs / total instructions)
    if total_instrs > 0:
        derived["control_flow_complexity"] = (
            features.get("trace_unique_pcs", 0) / total_instrs
        )
    else:
        derived["control_flow_complexity"] = 0.0

    # Cycles per instruction (if available)
    cycles = features.get("cycle_count")
    if cycles and total_instrs > 0:
        derived["cpi"] = cycles / total_instrs
    else:
        derived["cpi"] = None

    # Exception rate
    if total_instrs > 0:
        derived["exception_rate"] = features.get("exception_count", 0) / total_instrs
    else:
        derived["exception_rate"] = 0.0

    return derived


def find_test_result_pickle(
    metadata_dir: Path, testdotseed: str, coverage_path: str
) -> Optional[Path]:
    """Find the TestRunResult pickle file for a given test."""
    # Try direct path from testdotseed
    pickle_path = metadata_dir / f"{testdotseed}.pickle"
    if pickle_path.exists():
        return pickle_path

    # Try to extract from coverage_path
    # Coverage path format: .../run/coverage/testname.seed/...
    cov_path = Path(coverage_path)
    if "coverage" in cov_path.parts:
        # Extract testdotseed from path
        for i, part in enumerate(cov_path.parts):
            if part == "coverage" and i + 1 < len(cov_path.parts):
                possible_testdotseed = cov_path.parts[i + 1]
                pickle_path = metadata_dir / f"{possible_testdotseed}.pickle"
                if pickle_path.exists():
                    return pickle_path

    return None


def load_test_result(pickle_path: Path) -> Optional[Any]:
    """Load TestRunResult from pickle file."""
    if not pickle_path or not pickle_path.exists():
        return None

    try:
        if TestRunResult and pathlib3x:
            # Use the TestRunResult class method
            return TestRunResult.construct_from_pickle(pathlib3x.Path(str(pickle_path)))
        else:
            # Fallback to direct pickle loading
            with open(pickle_path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load test result from {pickle_path}: {e}")
        return None


def augment_coverage_record(
    record: Dict[str, Any],
    metadata_dir: Optional[Path],
) -> Dict[str, Any]:
    """Augment a single coverage label record with additional features."""
    testdotseed = record.get("testdotseed", "")
    coverage_path = record.get("coverage_path", "")

    # Initialize feature containers
    test_metadata_features = {}
    log_features = {}
    trace_features = {}
    derived_features = {}

    # Try to load test result pickle
    test_result = None
    if metadata_dir:
        pickle_path = find_test_result_pickle(metadata_dir, testdotseed, coverage_path)
        if pickle_path:
            test_result = load_test_result(pickle_path)

    # Extract features from test metadata
    if test_result:
        test_metadata_features = extract_test_metadata_features(test_result)

        # Get log and trace paths from test result
        if hasattr(test_result, "rtl_log") and test_result.rtl_log:
            log_features = parse_simulation_log(Path(test_result.rtl_log))

        if hasattr(test_result, "rtl_trace") and test_result.rtl_trace:
            trace_features = parse_trace_csv(Path(test_result.rtl_trace))

    # Combine all features
    all_features = {**test_metadata_features, **log_features, **trace_features}

    # Compute derived features
    derived_features = compute_derived_features(all_features)

    # Create augmented record
    augmented = record.copy()
    augmented["test_metadata"] = test_metadata_features
    augmented["execution_features"] = {**log_features, **trace_features}
    augmented["derived_features"] = derived_features

    return augmented


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Augment coverage labels with additional ML features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input coverage_labels.jsonl file from coverage_labeler.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file with augmented features.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to regression metadata directory (contains test pickle files).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of records to process (for testing).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information.",
    )

    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist.")
        return 1

    # Read input JSONL
    print(f"Reading coverage labels from {args.input}...")
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if args.limit:
        records = records[: args.limit]

    print(f"Processing {len(records)} records...")

    # Augment each record
    augmented_records = []
    for i, record in enumerate(records, 1):
        if args.verbose or i % 100 == 0:
            print(
                f"  [{i}/{len(records)}] Processing {record.get('testdotseed', 'unknown')}..."
            )

        try:
            augmented = augment_coverage_record(record, args.metadata)
            augmented_records.append(augmented)
        except Exception as e:
            print(f"  Warning: Error processing record {i}: {e}")
            # Include original record on error
            augmented_records.append(record)

    # Write output JSONL
    print(f"Writing augmented features to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in augmented_records:
            f.write(json.dumps(record))
            f.write("\n")

    print(f"Successfully wrote {len(augmented_records)} augmented records.")

    # Print feature summary
    if augmented_records:
        sample = augmented_records[0]
        print("\nFeature groups added:")
        print(f"  - test_metadata: {len(sample.get('test_metadata', {}))} features")
        print(
            f"  - execution_features: {len(sample.get('execution_features', {}))} features"
        )
        print(
            f"  - derived_features: {len(sample.get('derived_features', {}))} features"
        )

        if args.verbose:
            print("\nSample augmented record (first test):")
            print(json.dumps(sample, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
