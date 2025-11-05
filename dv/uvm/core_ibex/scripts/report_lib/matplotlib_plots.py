#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Generate matplotlib plots for regression test runtime analysis."""

import pathlib3x as pathlib
from typing import List, TextIO, Dict
import sys

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not available, skipping runtime plots")

import logging
logger = logging.getLogger(__name__)


def plot_runtime_histogram(test_results: List, output_path: pathlib.Path,
                           title: str = "CPU Runtime Distribution"):
    """Create a histogram of test runtimes.

    Args:
        test_results: List of TestRunResult objects
        output_path: Path to save the plot
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    # Extract runtimes, filtering out None values
    runtimes = [trr.runtime_s for trr in test_results
                if trr.runtime_s is not None]

    if not runtimes:
        logger.warning("No runtime data available for histogram plot")
        return

    plt.figure(figsize=(12, 6))

    # Create histogram
    n, bins, patches = plt.hist(runtimes, bins=30, edgecolor='black', alpha=0.7)

    # Color bars: green for passing tests, red for failing
    for i, patch in enumerate(patches):
        # Get tests in this bin range
        bin_min = bins[i]
        bin_max = bins[i + 1]
        bin_tests = [trr for trr in test_results
                     if trr.runtime_s is not None and
                     bin_min <= trr.runtime_s < bin_max]

        # Calculate pass rate for this bin
        if bin_tests:
            pass_rate = sum(1 for t in bin_tests if t.passed) / len(bin_tests)
            # Gradient from red (0% pass) to yellow (50%) to green (100%)
            if pass_rate < 0.5:
                color = (1.0, pass_rate * 2, 0.0)  # Red to yellow
            else:
                color = (2.0 - pass_rate * 2, 1.0, 0.0)  # Yellow to green
            patch.set_facecolor(color)
        else:
            patch.set_facecolor('gray')

    plt.xlabel('Runtime (seconds)', fontsize=12)
    plt.ylabel('Number of Tests', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')

    # Add statistics text
    mean_runtime = np.mean(runtimes)
    median_runtime = np.median(runtimes)
    max_runtime = np.max(runtimes)
    min_runtime = np.min(runtimes)

    stats_text = (f'Total Tests: {len(runtimes)}\n'
                 f'Mean: {mean_runtime:.2f}s\n'
                 f'Median: {median_runtime:.2f}s\n'
                 f'Min: {min_runtime:.2f}s\n'
                 f'Max: {max_runtime:.2f}s')

    plt.text(0.98, 0.97, stats_text,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Runtime histogram saved to {output_path}")


def plot_runtime_by_test(test_results: List, output_path: pathlib.Path,
                         title: str = "CPU Runtime by Test"):
    """Create a bar plot of runtime grouped by test name.

    Args:
        test_results: List of TestRunResult objects
        output_path: Path to save the plot
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    # Group tests by name
    test_dict = {}
    for trr in test_results:
        if trr.runtime_s is not None and trr.testname is not None:
            if trr.testname not in test_dict:
                test_dict[trr.testname] = {
                    'runtimes': [],
                    'passed': [],
                }
            test_dict[trr.testname]['runtimes'].append(trr.runtime_s)
            test_dict[trr.testname]['passed'].append(trr.passed)

    if not test_dict:
        logger.warning("No runtime data available for test comparison plot")
        return

    # Calculate statistics for each test
    test_names = []
    mean_runtimes = []
    pass_rates = []

    for test_name in sorted(test_dict.keys()):
        data = test_dict[test_name]
        test_names.append(test_name)
        mean_runtimes.append(np.mean(data['runtimes']))
        pass_rates.append(sum(data['passed']) / len(data['passed'])
                         if data['passed'] else 0)

    # Limit to top 30 tests by runtime if there are too many
    if len(test_names) > 30:
        # Sort by mean runtime and take top 30
        sorted_indices = np.argsort(mean_runtimes)[-30:]
        test_names = [test_names[i] for i in sorted_indices]
        mean_runtimes = [mean_runtimes[i] for i in sorted_indices]
        pass_rates = [pass_rates[i] for i in sorted_indices]
        title += " (Top 30 by Runtime)"

    # Create plot
    fig, ax = plt.subplots(figsize=(14, max(8, len(test_names) * 0.3)))

    # Create color array based on pass rate
    colors = []
    for pr in pass_rates:
        if pr < 0.5:
            colors.append((1.0, pr * 2, 0.0))  # Red to yellow
        else:
            colors.append((2.0 - pr * 2, 1.0, 0.0))  # Yellow to green

    bars = ax.barh(test_names, mean_runtimes, color=colors,
                   edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Mean Runtime (seconds)', fontsize=12)
    ax.set_ylabel('Test Name', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')

    # Add value labels on bars
    for i, (bar, runtime, pr) in enumerate(zip(bars, mean_runtimes, pass_rates)):
        width = bar.get_width()
        label = f'{runtime:.2f}s ({pr*100:.0f}%)'
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f' {label}',
                ha='left', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Runtime by test plot saved to {output_path}")


def plot_runtime_scatter(test_results: List, output_path: pathlib.Path,
                         title: str = "Runtime vs Test Status"):
    """Create a scatter plot of runtime vs test index, colored by pass/fail.

    Args:
        test_results: List of TestRunResult objects
        output_path: Path to save the plot
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    # Extract data
    passing_runtimes = []
    passing_indices = []
    failing_runtimes = []
    failing_indices = []

    for i, trr in enumerate(test_results):
        if trr.runtime_s is not None:
            if trr.passed:
                passing_runtimes.append(trr.runtime_s)
                passing_indices.append(i)
            else:
                failing_runtimes.append(trr.runtime_s)
                failing_indices.append(i)

    if not (passing_runtimes or failing_runtimes):
        logger.warning("No runtime data available for scatter plot")
        return

    plt.figure(figsize=(14, 6))

    # Plot passing tests
    if passing_runtimes:
        plt.scatter(passing_indices, passing_runtimes,
                   c='green', alpha=0.6, s=50, label='Passed',
                   edgecolors='black', linewidth=0.5)

    # Plot failing tests
    if failing_runtimes:
        plt.scatter(failing_indices, failing_runtimes,
                   c='red', alpha=0.6, s=50, label='Failed',
                   edgecolors='black', linewidth=0.5, marker='x')

    plt.xlabel('Test Index', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Add statistics
    all_runtimes = passing_runtimes + failing_runtimes
    if all_runtimes:
        mean_runtime = np.mean(all_runtimes)
        plt.axhline(y=mean_runtime, color='blue', linestyle='--',
                   linewidth=2, alpha=0.5, label=f'Mean: {mean_runtime:.2f}s')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Runtime scatter plot saved to {output_path}")


def plot_cumulative_runtime(test_results: List, output_path: pathlib.Path,
                           title: str = "Cumulative Runtime"):
    """Create a plot showing cumulative runtime over test execution.

    Args:
        test_results: List of TestRunResult objects
        output_path: Path to save the plot
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    # Extract runtimes in order
    runtimes = []
    test_labels = []

    for trr in test_results:
        if trr.runtime_s is not None:
            runtimes.append(trr.runtime_s)
            if trr.testname and trr.seed is not None:
                test_labels.append(f"{trr.testname}.{trr.seed}")
            else:
                test_labels.append("Unknown")

    if not runtimes:
        logger.warning("No runtime data available for cumulative plot")
        return

    # Calculate cumulative runtime
    cumulative = np.cumsum(runtimes)
    indices = list(range(len(runtimes)))

    plt.figure(figsize=(14, 6))
    plt.plot(indices, cumulative, linewidth=2, color='blue', marker='o',
             markersize=3, alpha=0.7)

    plt.xlabel('Test Number', fontsize=12)
    plt.ylabel('Cumulative Runtime (seconds)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')

    # Add total runtime annotation
    total_runtime = cumulative[-1]
    hours = int(total_runtime // 3600)
    minutes = int((total_runtime % 3600) // 60)
    seconds = int(total_runtime % 60)

    time_str = f'{hours}h {minutes}m {seconds}s' if hours > 0 else \
               f'{minutes}m {seconds}s' if minutes > 0 else \
               f'{seconds}s'

    stats_text = (f'Total Tests: {len(runtimes)}\n'
                 f'Total Runtime: {time_str}\n'
                 f'({total_runtime:.2f} seconds)')

    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Cumulative runtime plot saved to {output_path}")


def generate_all_runtime_plots(test_results: List, output_dir: pathlib.Path):
    """Generate all runtime analysis plots.

    Args:
        test_results: List of TestRunResult objects
        output_dir: Directory to save plots
    """
    if not MATPLOTLIB_AVAILABLE:
        print("WARNING: matplotlib not available, skipping runtime plots")
        return

    logger.info(f"Generating runtime plots in {output_dir}")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate each plot type
    try:
        plot_runtime_histogram(test_results, output_dir / 'runtime_histogram.png')
        plot_runtime_by_test(test_results, output_dir / 'runtime_by_test.png')
        plot_runtime_scatter(test_results, output_dir / 'runtime_scatter.png')
        plot_cumulative_runtime(test_results, output_dir / 'cumulative_runtime.png')

        logger.info("All runtime plots generated successfully")
    except Exception as e:
        logger.error(f"Error generating runtime plots: {e}")
        import traceback
        traceback.print_exc()


def output_results_matplotlib(test_results: List, output_dir: pathlib.Path):
    """Main function to generate matplotlib plots for test results.

    This is the main entry point that should be called from collect_results.py.

    Args:
        test_results: List of TestRunResult objects
        output_dir: Directory to save plots (typically md.dir_run)
    """
    generate_all_runtime_plots(test_results, output_dir)
