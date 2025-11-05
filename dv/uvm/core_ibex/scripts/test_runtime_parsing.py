#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Test script for runtime parsing functionality.

This script tests the parse_runtime_from_log function with sample log data.
"""

import re
import sys


def parse_runtime_from_log_sample(log_content: str, simulator: str) -> float:
    """Parse the runtime from simulator log content.

    This is a simplified version for testing.
    """
    # Pattern for xrun/xlm simulator
    # Example: TOOL:   xrun(64)    22.03-s012: Exiting on Nov 04, 2025 at 21:37:58 CST  (total: 00:00:21)
    xrun_pattern = r'TOOL:\s+xrun.*\(total:\s+(\d+):(\d+):(\d+)\)'

    # Pattern for VCS simulator
    # Example: CPU Time: 0.450 seconds; Data structure size: 0.0Mb
    vcs_pattern = r'CPU Time:\s+([\d.]+)\s+seconds'

    runtime_s = None

    if simulator in ['xlm', 'xcelium']:
        match = re.search(xrun_pattern, log_content)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = int(match.group(3))
            runtime_s = hours * 3600 + minutes * 60 + seconds
            print(f"✓ Parsed xrun runtime: {hours:02d}:{minutes:02d}:{seconds:02d} = {runtime_s}s")
    elif simulator == 'vcs':
        match = re.search(vcs_pattern, log_content)
        if match:
            runtime_s = float(match.group(1))
            print(f"✓ Parsed VCS runtime: {runtime_s}s")

    return runtime_s


def test_xrun_parsing():
    """Test parsing of xrun log format."""
    print("\n=== Testing xrun/Xcelium log parsing ===")

    # Test case 1: 21 seconds
    log1 = "TOOL:   xrun(64)    22.03-s012: Exiting on Nov 04, 2025 at 21:37:58 CST  (total: 00:00:21)"
    result1 = parse_runtime_from_log_sample(log1, 'xlm')
    assert result1 == 21, f"Expected 21, got {result1}"

    # Test case 2: 1 minute 30 seconds
    log2 = "TOOL:   xrun(64)    22.03-s012: Exiting on Nov 04, 2025 at 21:37:58 CST  (total: 00:01:30)"
    result2 = parse_runtime_from_log_sample(log2, 'xlm')
    assert result2 == 90, f"Expected 90, got {result2}"

    # Test case 3: 1 hour 5 minutes 45 seconds
    log3 = "TOOL:   xrun(64)    22.03-s012: Exiting on Nov 04, 2025 at 21:37:58 CST  (total: 01:05:45)"
    result3 = parse_runtime_from_log_sample(log3, 'xlm')
    assert result3 == 3945, f"Expected 3945, got {result3}"

    # Test case 4: Multi-line log with noise
    log4 = """
Some random log output
More log lines here
TOOL:   xrun(64)    22.03-s012: Exiting on Nov 04, 2025 at 21:37:58 CST  (total: 00:02:15)
Some trailing output
    """
    result4 = parse_runtime_from_log_sample(log4, 'xlm')
    assert result4 == 135, f"Expected 135, got {result4}"

    print("✓ All xrun tests passed!")


def test_vcs_parsing():
    """Test parsing of VCS log format."""
    print("\n=== Testing VCS log parsing ===")

    # Test case 1: Simple decimal
    log1 = "CPU Time: 0.450 seconds; Data structure size: 0.0Mb"
    result1 = parse_runtime_from_log_sample(log1, 'vcs')
    assert result1 == 0.450, f"Expected 0.450, got {result1}"

    # Test case 2: Longer runtime
    log2 = "CPU Time: 123.456 seconds; Data structure size: 1.2Mb"
    result2 = parse_runtime_from_log_sample(log2, 'vcs')
    assert result2 == 123.456, f"Expected 123.456, got {result2}"

    print("✓ All VCS tests passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Runtime Parsing Test Suite")
    print("=" * 60)

    try:
        test_xrun_parsing()
        test_vcs_parsing()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
