"""Label Ibex regression tests by their coverage contribution.

This module provides functions to replay the coverage merge using Cadence IMC
for the selected tests and record whether each test improves the cumulative
coverage metrics.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple

# Discover the Ibex DV python modules
# This script is now located in dv/uvm/core_ibex/scripts/
CORE_IBEX_SCRIPTS = Path(__file__).resolve().parent
IBEX_ROOT = CORE_IBEX_SCRIPTS.parent.parent.parent.parent
REPO_ROOT = IBEX_ROOT.parent
IBEX_UTIL = IBEX_ROOT / "util"
REPORT_LIB = CORE_IBEX_SCRIPTS / "report_lib"
RISCV_DV_SCRIPTS = IBEX_ROOT / "vendor" / "google_riscv-dv" / "scripts"

# Ensure the common ibex utility dir is on sys.path so imports such as
# `import ibex_config` succeed. Also add core scripts and report_lib.
for module_path in (
    IBEX_UTIL,
    CORE_IBEX_SCRIPTS,
    REPORT_LIB,
    RISCV_DV_SCRIPTS,
    IBEX_ROOT,
):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

# pylint: disable=wrong-import-position
from merge_cov import select_coverage_paths  # type: ignore  # noqa: E402
from metadata import RegressionMetadata  # type: ignore  # noqa: E402
from report_lib.util import (  # type: ignore  # noqa: E402
    IBEX_COVERAGE_METRICS,
    calc_cg_average,
    parse_xcelium_cov_report,
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


def label_coverage_contributions(
    metadata_dir: Path,
    imc_bin: str = "/opt/coe/cadence/VMANAGER/bin/imc",
    limit: Optional[int] = None,
    selected_metrics: Optional[List[str]] = None,
    min_covered: int = 1,
    min_cg_delta: float = 1e-6,
    log_dir: Optional[Path] = None,
    scratch_dir: Optional[Path] = None,
    keep_temp: bool = False,
) -> List[Dict[str, object]]:
    """Label tests by their coverage contribution.

    Args:
        metadata_dir: Path to regression metadata directory
        imc_bin: Path to IMC binary
        limit: Optional cap on number of tests to process
        selected_metrics: List of metrics to track (defaults to all)
        min_covered: Minimum covered delta to trigger a label
        min_cg_delta: Minimum covergroup delta to trigger a label
        log_dir: Directory for IMC logs (defaults to out/cov_labeler_logs)
        scratch_dir: Temporary directory (defaults to out/.cov_labeler_tmp)
        keep_temp: Keep temporary files for debugging

    Returns:
        List of labeled test records
    """
    import pathlib3x as pathlib3x

    metadata_dir = pathlib3x.Path(str(metadata_dir.resolve()))
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory {metadata_dir} does not exist.")

    md = RegressionMetadata.construct_from_metadata_dir(metadata_dir)
    if md.simulator != "xlm":
        raise ValueError(
            f"Expected simulator 'xlm' but regression used '{md.simulator}'."
        )

    if md.cov_merge_db_list is None:
        raise ValueError(
            "Coverage merge database list not found in metadata. "
            "Did you run the regression with COV=1 and merge coverage?"
        )

    cov_runfile = Path(md.cov_merge_db_list)
    if not cov_runfile.exists():
        raise FileNotFoundError(
            f"Coverage runfile missing: {cov_runfile}. "
            "Did you run the regression with COV=1 and merge coverage?"
        )

    coverage_paths = select_coverage_paths(pathlib3x.Path(str(cov_runfile)))
    if not coverage_paths:
        raise ValueError("No coverage databases found in runfile.")
    if limit:
        coverage_paths = coverage_paths[:limit]

    if md.ibex_dv_root is None:
        raise ValueError("ibex_dv_root not found in metadata.")

    waiver_script = Path(md.ibex_dv_root) / "waivers" / "coverage_waivers_xlm.tcl"
    if not waiver_script.exists():
        raise FileNotFoundError(f"Waiver script not found at {waiver_script}.")

    if log_dir is None:
        log_dir = CORE_IBEX_SCRIPTS.parent / "out" / "cov_labeler_logs"
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    if scratch_dir is None:
        scratch_dir = CORE_IBEX_SCRIPTS.parent / "out" / ".cov_labeler_tmp"
    scratch_dir = scratch_dir.expanduser().resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    env_base["DUT_TOP"] = md.dut_cov_rtl_path

    if selected_metrics is None:
        selected_metrics = list(IBEX_COVERAGE_METRICS)

    invalid_metrics = [m for m in selected_metrics if m not in IBEX_COVERAGE_METRICS]
    if invalid_metrics:
        raise ValueError(
            f"Unknown metrics: {', '.join(invalid_metrics)}. "
            f"Valid metrics: {', '.join(IBEX_COVERAGE_METRICS)}"
        )

    records: List[Dict[str, object]] = []
    cumulative_counts: Dict[str, float] = {
        metric: 0.0 for metric in IBEX_COVERAGE_METRICS
    }
    totals: Dict[str, float] = {metric: 0.0 for metric in IBEX_COVERAGE_METRICS}
    cumulative_covergroup = 0.0

    temp_manager: Optional[TemporaryDirectory[str]] = None
    try:
        temp_manager = TemporaryDirectory(prefix="cov_labeler_", dir=str(scratch_dir))
        temp_root = Path(temp_manager.name)
        progress_runfile = temp_root / "progress_runfile.txt"
        merge_dir = temp_root / "merged"
        report_dir = temp_root / "report"

        if md.ot_xcelium_cov_scripts is None:
            raise ValueError("ot_xcelium_cov_scripts not found in metadata.")

        cov_report_tcl = Path(md.ot_xcelium_cov_scripts) / "cov_report.tcl"
        if not cov_report_tcl.exists():
            raise FileNotFoundError(f"cov_report.tcl not found at {cov_report_tcl}")

        processed_paths: List[Path] = []

        for index, coverage_dir in enumerate(coverage_paths, start=1):
            processed_paths.append(coverage_dir)

            with progress_runfile.open(
                "w", encoding="ascii", newline="\n", errors="ignore"
            ) as pf:
                for p in processed_paths:
                    pf.write(str(p))
                    pf.write("\n")

            if merge_dir.exists():
                shutil.rmtree(merge_dir)
            merge_env = env_base.copy()
            merge_env["cov_merge_db_dir"] = str(merge_dir)
            merge_env["cov_db_runfile"] = str(progress_runfile)
            merge_env["cov_db_dirs"] = ""

            merge_log = log_dir / f"merge_{index:05d}.log"
            merge_log.parent.mkdir(parents=True, exist_ok=True)
            with merge_log.open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    [
                        imc_bin,
                        "-64bit",
                        "-licqueue",
                        "-exec",
                        str(Path(md.ot_xcelium_cov_scripts) / "cov_merge.tcl"),
                        "-logfile",
                        str(merge_log),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=merge_env,
                    check=False,
                )
            if process.returncode != 0:
                raise RuntimeError(
                    f"IMC merge failed with code {process.returncode}. See {merge_log}"
                )

            if report_dir.exists():
                shutil.rmtree(report_dir)
            report_dir.mkdir(parents=True)
            report_env = env_base.copy()
            report_env["cov_report_dir"] = str(report_dir)

            report_log = log_dir / f"report_{index:05d}.log"
            with report_log.open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    [
                        imc_bin,
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
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=report_env,
                    check=False,
                )
            if process.returncode != 0:
                raise RuntimeError(
                    f"IMC report failed with code {process.returncode}. See {report_log}"
                )

            metrics, covergroup_avg = parse_cov_metrics(report_dir)

            coverage_deltas: Dict[str, float] = {}
            triggers: List[str] = []

            for metric_name in selected_metrics:
                if metric_name not in metrics:
                    continue
                covered = metrics[metric_name]["covered"]
                total = metrics[metric_name]["total"]
                if totals[metric_name] == 0:
                    totals[metric_name] = total

                delta_covered = covered - cumulative_counts[metric_name]
                if delta_covered >= min_covered:
                    triggers.append(metric_name)
                    coverage_deltas[metric_name] = delta_covered

                cumulative_counts[metric_name] = covered

            cg_delta = covergroup_avg - cumulative_covergroup
            if cg_delta >= min_cg_delta:
                triggers.append("covergroup")
                coverage_deltas["covergroup"] = cg_delta
            cumulative_covergroup = covergroup_avg

            testdotseed = coverage_dir.name
            record: Dict[str, object] = {
                "testdotseed": testdotseed,
                "coverage_path": str(coverage_dir),
                "labels": triggers,
                "deltas": coverage_deltas,
                "cumulative_covergroup": cumulative_covergroup,
            }
            records.append(record)

    finally:
        if temp_manager and not keep_temp:
            temp_manager.cleanup()

    return records
