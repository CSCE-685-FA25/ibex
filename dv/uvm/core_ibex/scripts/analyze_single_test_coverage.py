"""Analyze coverage for a single test.

This module provides functions to load an individual test's coverage database
(.ucd file for Xcelium) and generate detailed coverage reports including
per-instance and per-block coverage information.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Discover the Ibex DV python modules
# This script is located in dv/uvm/core_ibex/scripts/
CORE_IBEX_SCRIPTS = Path(__file__).resolve().parent
IBEX_ROOT = CORE_IBEX_SCRIPTS.parent.parent.parent.parent
REPO_ROOT = IBEX_ROOT.parent
IBEX_UTIL = IBEX_ROOT / "util"
REPORT_LIB = CORE_IBEX_SCRIPTS / "report_lib"

# Ensure paths are on sys.path
for module_path in (IBEX_UTIL, CORE_IBEX_SCRIPTS, REPORT_LIB, IBEX_ROOT):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

# pylint: disable=wrong-import-position
from report_lib.util import (  # type: ignore  # noqa: E402
    IBEX_COVERAGE_METRICS,
    calc_cg_average,
    parse_xcelium_cov_report,
    parse_xcelium_instance_report,
)


def analyze_single_test_coverage(
    ucd_path: Path,
    output_dir: Path,
    imc_bin: str = "/opt/coe/cadence/VMANAGER/bin/imc",
    dut_top: str = "ibex_core",
    waiver_script: Optional[Path] = None,
) -> Dict[str, Any]:
    """Analyze coverage for a single test database.

    Args:
        ucd_path: Path to the .ucd coverage database
        output_dir: Directory to store output reports
        imc_bin: Path to IMC binary
        dut_top: Top-level DUT module name
        waiver_script: Optional path to coverage waiver TCL script

    Returns:
        Dictionary with coverage metrics and details
    """
    if not ucd_path.exists():
        raise FileNotFoundError(f"Coverage database not found: {ucd_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the TCL script for single test coverage reporting
    tcl_script = CORE_IBEX_SCRIPTS / "tcl" / "single_test_coverage_report.tcl"

    if not tcl_script.exists():
        # Fallback: use vendored OpenTitan scripts
        ot_scripts = REPO_ROOT / "lowrisc_ip" / "dv" / "tools" / "xcelium"
        if not ot_scripts.exists():
            # Try alternative path
            ot_scripts = (
                IBEX_ROOT / "vendor" / "lowrisc_ip" / "dv" / "tools" / "xcelium"
            )
        tcl_script = ot_scripts / "cov_report.tcl"

    if not tcl_script.exists():
        raise FileNotFoundError(f"Coverage report TCL script not found at {tcl_script}")

    # Set up environment variables for TCL script
    env = os.environ.copy()
    env["DUT_TOP"] = dut_top
    env["cov_report_dir"] = str(output_dir)

    # Build IMC command
    imc_cmd = [
        imc_bin,
        "-64bit",
        "-licqueue",
        "-load",
        str(ucd_path),
    ]

    # Add waiver script if provided
    if waiver_script and waiver_script.exists():
        imc_cmd.extend(["-init", str(waiver_script)])

    # Add report generation script
    imc_cmd.extend(
        [
            "-exec",
            str(tcl_script),
            "-logfile",
            str(output_dir / "imc_report.log"),
        ]
    )

    # Run IMC
    print(f"Analyzing coverage for {ucd_path.name}...")
    print(f"  Output directory: {output_dir}")

    log_file = output_dir / "imc_report.log"
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            imc_cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )

    if process.returncode != 0:
        raise RuntimeError(
            f"IMC command failed with code {process.returncode}.\n"
            f"See {log_file} for details."
        )

    # Parse the generated coverage reports
    metrics = {}

    cov_report_file = output_dir / "cov_report.txt"
    cg_report_file = output_dir / "cov_report_cg.txt"

    if cov_report_file.exists():
        cov_dict = parse_xcelium_cov_report(cov_report_file.read_text(encoding="utf-8"))

        # Extract metrics for the top-level module
        if dut_top in cov_dict:
            module_metrics = cov_dict[dut_top]
            for metric_name in IBEX_COVERAGE_METRICS:
                key = f"{metric_name}-covered"
                if key in module_metrics:
                    covered = float(module_metrics[key]["covered"])
                    total = float(module_metrics[key]["total"])
                    pct = covered / total if total else 0.0
                    metrics[metric_name] = {
                        "covered": covered,
                        "total": total,
                        "pct": pct,
                    }

    # Parse covergroup metrics
    covergroup_avg = 0.0
    if cg_report_file.exists():
        cg_dict = parse_xcelium_cov_report(cg_report_file.read_text(encoding="utf-8"))
        covergroup_avg = calc_cg_average(cg_dict) or 0.0

    metrics["covergroup"] = {"pct": covergroup_avg}

    # Parse instance-level coverage
    instance_coverage = {}
    inst_report_file = output_dir / "cov_report_instances.txt"
    if inst_report_file.exists():
        instance_coverage = parse_xcelium_instance_report(
            inst_report_file.read_text(encoding="utf-8")
        )

    # Extract test name from ucd path
    # Typical path: .../coverage/testname.seed/...
    test_name = (
        ucd_path.parent.name if ucd_path.parent.name != "coverage" else "unknown"
    )

    result = {
        "test_name": test_name,
        "ucd_path": str(ucd_path),
        "metrics": metrics,
        "instance_coverage": instance_coverage,
        "output_dir": str(output_dir),
    }

    return result
