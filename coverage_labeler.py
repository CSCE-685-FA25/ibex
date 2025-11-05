#!/usr/bin/env python3
"""Label Ibex regression tests by their coverage contribution.

Run this script from the lowRISC workspace root (one level above ``ibex``).
It replays the coverage merge using Cadence IMC for the selected tests and
records whether each test improves the cumulative coverage metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, Optional, Tuple
from collections import Counter

# Discover the Ibex DV python modules when invoked from lowRISC/
REPO_ROOT = Path(__file__).resolve().parent
IBEX_ROOT = REPO_ROOT / "ibex"
IBEX_UTIL = IBEX_ROOT / "util"
CORE_IBEX_SCRIPTS = IBEX_ROOT / "dv" / "uvm" / "core_ibex" / "scripts"
REPORT_LIB = CORE_IBEX_SCRIPTS / "report_lib"
RISCV_DV_SCRIPTS = IBEX_ROOT / "vendor" / "google_riscv-dv" / "scripts"

# Ensure the common ibex utility dir is on sys.path so imports such as
# `import ibex_config` succeed. Also add core scripts and report_lib.
for module_path in (IBEX_UTIL, CORE_IBEX_SCRIPTS, REPORT_LIB, RISCV_DV_SCRIPTS, IBEX_ROOT):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

# pylint: disable=wrong-import-position
from metadata import RegressionMetadata  # type: ignore  # noqa: E402
from report_lib.util import (  # type: ignore  # noqa: E402
    IBEX_COVERAGE_METRICS,
    calc_cg_average,
    parse_xcelium_cov_report,
)

DEFAULT_METADATA_DIR = (
    IBEX_ROOT / "dv" / "uvm" / "core_ibex" / "out" / "metadata"
)
DEFAULT_OUTPUT = REPO_ROOT / "coverage_labels.jsonl"
DEFAULT_LOG_DIR = REPO_ROOT / "cov_labeler_logs"

# We use the repository cov_report.tcl which calls both the legacy textual
# summary reports and the newer `report_metrics` HTML report. This keeps
# behavior identical to the DV flow while avoiding maintaining a separate
# Tcl snippet here.

def run_imc(cmd: List[str], env: Dict[str, str], log_path: Path) -> None:
    """Run an IMC command and capture stdout/err into log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"Command {' '.join(cmd)} failed with code {process.returncode}.\n"
            f"See {log_path} for details."
        )


def parse_cov_metrics(report_dir: Path) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Extract numeric coverage metrics from generated reports."""
    cov_dict = parse_xcelium_cov_report(
        (report_dir / "cov_report.txt").read_text(encoding="utf-8")
    )
    cg_dict = parse_xcelium_cov_report(
        (report_dir / "cov_report_cg.txt").read_text(encoding="utf-8")
    )

    if "ibex_top" not in cov_dict:
        raise RuntimeError("Expected ibex_top module in coverage report.")

    module_metrics = cov_dict["ibex_top"]
    metrics: Dict[str, Dict[str, float]] = {}

    for metric_name in IBEX_COVERAGE_METRICS:
        key = f"{metric_name}-covered"
        if key not in module_metrics:
            continue
        covered = float(module_metrics[key]["covered"])
        total = float(module_metrics[key]["total"])
        pct = covered / total if total else 0.0
        metrics[metric_name] = {
            "covered": covered,
            "total": total,
            "pct": pct,
        }

    covergroup_avg = calc_cg_average(cg_dict) or 0.0
    return metrics, covergroup_avg


def write_results(
    output_path: Path,
    records: List[Dict[str, object]],
    *,
    append: bool = False,
) -> None:
    """Write JSONL records to the requested path."""
    if not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def select_coverage_paths(runfile: Path) -> List[Path]:
    """Return absolute coverage directories in the order tests were merged."""
    selected: List[Path] = []
    for line in runfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "coverage/fcov" in line:
            continue
        cov_dir = Path(line)
        if cov_dir.exists():
            selected.append(cov_dir)
    return selected


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay coverage merge and label tests by coverage gain."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help="Path to the regression metadata directory (metadata.yaml lives here).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSONL file for labeled results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of coverage databases processed (for quick dry runs).",
    )
    parser.add_argument(
        "--imc-bin",
        default="imc",
        help="Cadence IMC binary to invoke (default: imc).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory to store merge/report logs.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate merge data (useful for debugging).",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory to host temporary merged/report data. "
            "If omitted, the system temporary directory is used."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from an existing output JSONL file. Existing entries are "
            "validated against the runfile and skipped."
        ),
    )
    parser.add_argument(
        "--metrics",
        default=",".join(IBEX_COVERAGE_METRICS),
        help=(
            "Comma-separated list of coverage metrics to consider when computing labels. "
            "Defaults to all functional metrics."
        ),
    )
    parser.add_argument(
        "--min-covered",
        type=int,
        default=1,
        help=(
            "Minimum covered delta (per metric) required to treat that metric as a label trigger. "
            "Set to 0 to match the original behavior."
        ),
    )
    parser.add_argument(
        "--min-cg-delta",
        type=float,
        default=1e-6,
        help=(
            "Minimum covergroup delta required to trigger a label. "
            "Set to 0 to include any positive change."
        ),
    )
    args = parser.parse_args(argv)

    # RegressionMetadata expects a pathlib3x.Path instance (used across the
    # ibex scripts). Convert the user-supplied pathlib.Path into the
    # required type to avoid typeguard errors.
    import pathlib3x as pathlib3x

    metadata_dir = args.metadata.resolve()
    metadata_dir = pathlib3x.Path(str(metadata_dir))
    if not metadata_dir.exists():
        raise SystemExit(f"Metadata directory {metadata_dir} does not exist.")

    md = RegressionMetadata.construct_from_metadata_dir(metadata_dir)
    if md.simulator != "xlm":
        raise SystemExit(
            f"Expected simulator 'xlm' but regression used '{md.simulator}'."
        )

    cov_runfile = Path(md.cov_merge_db_list)
    if not cov_runfile.exists():
        raise SystemExit(
            "Coverage runfile missing. Did you run the regression with COV=1?"
        )

    coverage_paths = select_coverage_paths(cov_runfile)
    if not coverage_paths:
        raise SystemExit("No coverage databases found in runfile.")
    if args.limit:
        coverage_paths = coverage_paths[: args.limit]

    waiver_script = Path(md.ibex_dv_root) / "waivers" / "coverage_waivers_xlm.tcl"
    if not waiver_script.exists():
        raise SystemExit(f"Waiver script not found at {waiver_script}.")

    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    env_base["DUT_TOP"] = md.dut_cov_rtl_path

    selected_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    invalid_metrics = [m for m in selected_metrics if m not in IBEX_COVERAGE_METRICS]
    if invalid_metrics:
        raise SystemExit(
            "Unknown metrics specified: " + ", ".join(invalid_metrics) +
            f". Valid metrics: {', '.join(IBEX_COVERAGE_METRICS)}"
        )

    records: List[Dict[str, object]] = []
    cumulative_counts: Dict[str, float] = {metric: 0.0 for metric in IBEX_COVERAGE_METRICS}
    totals: Dict[str, float] = {metric: 0.0 for metric in IBEX_COVERAGE_METRICS}
    cumulative_covergroup = 0.0
    trigger_counts = Counter()

    processed_count = 0
    if args.resume and args.output.exists():
        with args.output.open("r", encoding="utf-8") as existing:
            for line in existing:
                line = line.strip()
                if not line:
                    continue
                if processed_count >= len(coverage_paths):
                    raise SystemExit(
                        "Existing output has more entries than coverage runfile."
                    )
                record = json.loads(line)
                expected_path = str(coverage_paths[processed_count])
                if record.get("coverage_path") != expected_path:
                    raise SystemExit(
                        "Coverage path mismatch when resuming: "
                        f"expected {expected_path} but found {record.get('coverage_path')}"
                    )
                deltas = record.get("covered_deltas", {})
                metrics_after = record.get("metrics_after", {})
                for metric in IBEX_COVERAGE_METRICS:
                    delta = float(deltas.get(metric, 0.0))
                    cumulative_counts[metric] += delta
                    pct_after = metrics_after.get(metric)
                    if totals[metric] == 0.0 and pct_after:
                        # Avoid division by zero; pct_after is already 0-1.
                        if pct_after > 0:
                            totals[metric] = cumulative_counts[metric] / pct_after
                        else:
                            totals[metric] = 0.0
                cumulative_covergroup = record.get(
                    "covergroup_after", cumulative_covergroup
                )
                triggers_existing = record.get("triggers") or []
                for trig in triggers_existing or ["none"]:
                    trigger_counts[trig] += 1
                processed_count += 1
        print(f"Resuming from {processed_count} existing records in {args.output}.")
    elif not args.resume and args.output.exists():
        raise SystemExit(
            f"Output file {args.output} already exists. Use --resume or remove it first."
        )

    output_path = args.output.resolve()

    remaining = len(coverage_paths) - processed_count
    print(
        f"Processing {remaining} coverage databases "
        f"(total: {len(coverage_paths)}, already done: {processed_count})."
    )

    scratch_dir: Optional[Path] = None
    if args.scratch_dir is not None:
        scratch_dir = args.scratch_dir.expanduser().resolve()
        scratch_dir.mkdir(parents=True, exist_ok=True)

    temp_manager: Optional[TemporaryDirectory[str]] = None
    try:
        temp_manager = TemporaryDirectory(
            prefix="cov_labeler_",
            dir=str(scratch_dir) if scratch_dir else None,
        )
        temp_root = Path(temp_manager.name)
        progress_runfile = temp_root / "progress_runfile.txt"
        merge_dir = temp_root / "merged"
        report_dir = temp_root / "report"
        # Use the repository-provided cov_report.tcl for reporting.
        cov_report_tcl = Path(md.ot_xcelium_cov_scripts) / "cov_report.tcl"
        if not cov_report_tcl.exists():
            raise SystemExit(f"cov_report.tcl not found at {cov_report_tcl}")

        processed_paths: List[Path] = list(coverage_paths[:processed_count])

        if processed_count >= len(coverage_paths):
            print("All coverage databases already processed. Nothing to do.")
            return 0

        for index, coverage_dir in enumerate(
            coverage_paths[processed_count:], start=processed_count + 1
        ):
            processed_paths.append(coverage_dir)
            # Write the progress runfile in plain ASCII (no BOM) because IMC
            # can reject non-ASCII or otherwise malformed text files with
            # "not a legal text file" errors. Use newline='\n' and
            # ignore characters that can't be encoded in ASCII to be robust.
            with progress_runfile.open("w", encoding="ascii", newline="\n", errors="ignore") as pf:
                for p in processed_paths:
                    pf.write(str(p))
                    pf.write("\n")

            # Sanity-check the runfile contents before invoking IMC so we can
            # fail early with a helpful message if something is wrong.
            runfile_lines = [l.strip() for l in progress_runfile.read_text(encoding="ascii", errors="ignore").splitlines() if l.strip()]
            if not runfile_lines:
                raise SystemExit(f"Progress runfile {progress_runfile} is empty or unreadable.")
            for line in runfile_lines:
                if not Path(line).exists():
                    raise SystemExit(f"Path listed in runfile does not exist: {line}")

            if merge_dir.exists():
                shutil.rmtree(merge_dir)
            merge_env = env_base.copy()
            merge_env["cov_merge_db_dir"] = str(merge_dir)
            merge_env["cov_db_runfile"] = str(progress_runfile)
            merge_env["cov_db_dirs"] = ""

            merge_log = log_dir / f"merge_{index:05d}.log"
            try:
                run_imc(
                    [
                        args.imc_bin,
                        "-64bit",
                        "-licqueue",
                        "-exec",
                        str(Path(md.ot_xcelium_cov_scripts) / "cov_merge.tcl"),
                        "-logfile",
                        str(merge_log),
                    ],
                    merge_env,
                    merge_log,
                )
            except RuntimeError as err:
                # Show the tail of the merge log to help debugging why IMC
                # rejected the runfile (or failed for other reasons).
                try:
                    tail_lines = Path(merge_log).read_text(encoding="utf-8", errors="ignore").splitlines()[-200:]
                    tail = "\n".join(tail_lines)
                except Exception:
                    tail = f"Could not read merge log {merge_log}"
                preserved_runfile = log_dir / f"runfile_{index:05d}.txt"
                try:
                    shutil.copy2(progress_runfile, preserved_runfile)
                except Exception:
                    preserved_runfile = None
                extra = (
                    f"\nRunfile snapshot: {preserved_runfile}" if preserved_runfile else ""
                )
                raise RuntimeError(
                    f"IMC merge failed: {err}\n--- merge log tail ---\n{tail}{extra}"
                ) from err

            if report_dir.exists():
                shutil.rmtree(report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)

            report_env = env_base.copy()
            report_env["cov_merge_db_dir"] = str(merge_dir)
            report_env["cov_report_dir"] = str(report_dir)

            report_log = log_dir / f"report_{index:05d}.log"
            run_imc(
                [
                    args.imc_bin,
                    "-64bit",
                    "-licqueue",
                    "-load",
                    str(merge_dir),
                    "-init",
                    str(waiver_script),
                    "-exec",
                    str(cov_report_tcl),
                    "-logfile",
                    str(report_log),
                ],
                report_env,
                report_log,
            )

            metrics, covergroup_avg = parse_cov_metrics(report_dir)

            coverage_deltas: Dict[str, float] = {}
            pct_before: Dict[str, float] = {}
            pct_after: Dict[str, float] = {}
            triggers: List[str] = []

            for metric, values in metrics.items():
                if totals[metric] == 0.0:
                    totals[metric] = values["total"]
                pct_before[metric] = (
                    cumulative_counts[metric] / totals[metric] if totals[metric] else 0.0
                )
                pct_after[metric] = values["pct"]
                coverage_deltas[metric] = values["covered"] - cumulative_counts[metric]
                if (
                    metric in selected_metrics
                    and coverage_deltas[metric] >= args.min_covered
                ):
                    triggers.append(metric)
                cumulative_counts[metric] = values["covered"]

            covergroup_before = cumulative_covergroup
            covergroup_delta = covergroup_avg - cumulative_covergroup
            cumulative_covergroup = covergroup_avg

            if covergroup_delta >= args.min_cg_delta:
                triggers.append("covergroup")

            label = 1 if triggers else 0
            for trig in triggers or ["none"]:
                trigger_counts[trig] += 1

            test_name = coverage_dir.name
            records.append(
                {
                    "index": index,
                    "testdotseed": test_name,
                    "coverage_path": str(coverage_dir),
                    "metrics_before": pct_before,
                    "metrics_after": pct_after,
                    "covered_deltas": coverage_deltas,
                    "covergroup_before": covergroup_before,
                    "covergroup_after": covergroup_avg,
                    "covergroup_delta": covergroup_delta,
                    "label": label,
                    "triggers": triggers,
                }
            )

            append_mode = processed_count > 0 or len(records) > 1
            write_results(output_path, [records[-1]], append=append_mode)

            print(
                f"[{index:05d}/{len(coverage_paths):05d}] {test_name}: "
                f"label={label} block_delta={coverage_deltas.get('block', 0.0):.0f} "
                f"triggers={triggers or ['none']}"
            )

    finally:
        if temp_manager and not args.keep_temp:
            temp_manager.cleanup()

    total_processed = processed_count + len(records)
    print(
        f"Processed {len(records)} new entries (total processed: {total_processed})."
    )
    print(f"Detailed IMC logs: {log_dir}")
    print("Label summary: " + ", ".join(f"{k}={v}" for k, v in sorted(trigger_counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
