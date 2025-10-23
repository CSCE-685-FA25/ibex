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


def write_results(output_path: Path, records: List[Dict[str, object]]) -> None:
    """Write JSONL records to the requested path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
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

    print(f"Processing {len(coverage_paths)} coverage databases...")

    waiver_script = Path(md.ibex_dv_root) / "waivers" / "coverage_waivers_xlm.tcl"
    if not waiver_script.exists():
        raise SystemExit(f"Waiver script not found at {waiver_script}.")

    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    env_base["DUT_TOP"] = md.dut_cov_rtl_path

    records: List[Dict[str, object]] = []
    cumulative_counts: Dict[str, float] = {metric: 0.0 for metric in IBEX_COVERAGE_METRICS}
    totals: Dict[str, float] = {metric: 0.0 for metric in IBEX_COVERAGE_METRICS}
    cumulative_covergroup = 0.0

    temp_manager: Optional[TemporaryDirectory[str]] = None
    try:
        temp_manager = TemporaryDirectory(prefix="cov_labeler_")
        temp_root = Path(temp_manager.name)
        progress_runfile = temp_root / "progress_runfile.txt"
        merge_dir = temp_root / "merged"
        report_dir = temp_root / "report"
        # Use the repository-provided cov_report.tcl for reporting.
        cov_report_tcl = Path(md.ot_xcelium_cov_scripts) / "cov_report.tcl"
        if not cov_report_tcl.exists():
            raise SystemExit(f"cov_report.tcl not found at {cov_report_tcl}")

        processed_paths: List[Path] = []

        for index, coverage_dir in enumerate(coverage_paths, start=1):
            processed_paths.append(coverage_dir)
            progress_runfile.write_text(
                "\n".join(str(path) for path in processed_paths) + "\n",
                encoding="utf-8",
            )

            if merge_dir.exists():
                shutil.rmtree(merge_dir)
            merge_env = env_base.copy()
            merge_env["cov_merge_db_dir"] = str(merge_dir)
            merge_env["cov_db_runfile"] = str(progress_runfile)
            merge_env["cov_db_dirs"] = ""

            merge_log = log_dir / f"merge_{index:05d}.log"
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

            for metric, values in metrics.items():
                if totals[metric] == 0.0:
                    totals[metric] = values["total"]
                pct_before[metric] = (
                    cumulative_counts[metric] / totals[metric] if totals[metric] else 0.0
                )
                pct_after[metric] = values["pct"]
                coverage_deltas[metric] = values["covered"] - cumulative_counts[metric]
                cumulative_counts[metric] = values["covered"]

            covergroup_before = cumulative_covergroup
            covergroup_delta = covergroup_avg - cumulative_covergroup
            cumulative_covergroup = covergroup_avg

            label = 1 if (
                any(delta > 0 for delta in coverage_deltas.values())
                or covergroup_delta > 1e-6
            ) else 0

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
                }
            )

            print(
                f"[{index:05d}/{len(coverage_paths):05d}] {test_name}: "
                f"label={label} block_delta={coverage_deltas.get('block', 0.0):.0f}"
            )

    finally:
        if temp_manager and not args.keep_temp:
            temp_manager.cleanup()

    write_results(args.output.resolve(), records)
    print(f"Wrote {len(records)} labeled records to {args.output}.")
    print(f"Detailed IMC logs: {log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
