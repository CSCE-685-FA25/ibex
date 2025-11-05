# CPU Runtime Plotting for Ibex Regression Tests

This document describes the CPU runtime tracking and visualization features for Ibex regression tests.

## Overview

The regression test infrastructure now automatically captures and visualizes CPU runtime data for each test execution. This helps identify:
- Tests that take longer to execute
- Performance bottlenecks in the test suite
- Correlation between runtime and test pass/fail status
- Overall regression execution time

## Features

### Automatic Runtime Tracking

- **Captured automatically**: Runtime is measured for each RTL simulation
- **Stored in test results**: Runtime data is saved in the `TestRunResult` pickle/YAML files
- **Field name**: `runtime_s` (measured in seconds as a float)

### Generated Plots

Four types of plots are automatically generated in the regression output directory:

1. **runtime_histogram.png**: Distribution of test runtimes
   - Bins colored by pass rate (green = high pass rate, red = low pass rate)
   - Includes statistics: mean, median, min, max, total tests

2. **runtime_by_test.png**: Bar chart of mean runtime by test name
   - Shows average runtime for each test across all seeds
   - Bars colored by pass rate
   - Limited to top 30 tests by runtime if more tests exist

3. **runtime_scatter.png**: Scatter plot of runtime vs test index
   - Green dots: passing tests
   - Red X marks: failing tests
   - Shows runtime distribution across the entire test suite

4. **cumulative_runtime.png**: Cumulative runtime over test execution
   - Shows total execution time as tests complete
   - Useful for understanding total regression time

## Usage

### Automatic Generation (Default)

Runtime plots are generated automatically when running the regression:

```bash
# Run regression as usual
make -C dv/uvm/core_ibex/ run SIMULATION=xlm TEST=all

# Plots are automatically created in out/run/ directory
ls out/run/*.png
```

### Manual Plot Generation

You can regenerate plots from existing test results using the standalone script.

**Run from anywhere** (script handles paths automatically):

```bash
# Basic usage
./dv/uvm/core_ibex/scripts/plot_cpu_runtime.py --dir-metadata out/metadata

# Or with absolute path
python3 dv/uvm/core_ibex/scripts/plot_cpu_runtime.py --dir-metadata out/metadata

# Specify custom output directory
./dv/uvm/core_ibex/scripts/plot_cpu_runtime.py \
    --dir-metadata out/metadata \
    --output-dir custom_plots/

# Verbose output
./dv/uvm/core_ibex/scripts/plot_cpu_runtime.py \
    --dir-metadata out/metadata \
    --verbose
```

**Note**: The script automatically adds the necessary directories to the Python path, so it can be run from any location.

### Requirements

The plotting functionality requires matplotlib:

```bash
pip install matplotlib numpy
```

If matplotlib is not available, the system will gracefully skip plot generation and print a warning.

## Implementation Details

### Modified Files

1. **test_run_result.py**: Added `runtime_s: Optional[float]` field to TestRunResult dataclass

2. **run_rtl.py**: Modified to capture CPU runtime
   - Parses runtime from simulator log files (rtl_sim.log)
   - Supports xrun/Xcelium and VCS log formats
   - Extracts time in HH:MM:SS or decimal seconds format
   - Stores result in `runtime_s` field

3. **collect_results.py**: Integration with result collection
   - Imports `output_results_matplotlib` function
   - Calls plotting after generating other reports

4. **report_lib/matplotlib_plots.py**: New module with plotting functions
   - `plot_runtime_histogram()`: Runtime distribution histogram
   - `plot_runtime_by_test()`: Bar chart by test name
   - `plot_runtime_scatter()`: Scatter plot of all tests
   - `plot_cumulative_runtime()`: Cumulative runtime over time
   - `generate_all_runtime_plots()`: Generate all plot types
   - `output_results_matplotlib()`: Main entry point

5. **plot_cpu_runtime.py**: Standalone script for manual plot generation

### Data Flow

```
Test Execution (run_rtl.py)
    ↓
Run RTL simulation
    ↓
Parse rtl_sim.log for runtime
    ↓
Extract "TOOL: xrun... (total: HH:MM:SS)" or "CPU Time: X.XXX seconds"
    ↓
Convert to seconds and store in TestRunResult.runtime_s
    ↓
Export to pickle/YAML
    ↓
Result Collection (collect_results.py)
    ↓
Load all TestRunResult objects
    ↓
Generate plots (matplotlib_plots.py)
    ↓
Save PNG files to output directory
```

### Runtime Measurement

Runtime is extracted from simulator log files after simulation completes:

**Xcelium/xrun Log Format:**
```
TOOL:   xrun(64)    22.03-s012: Exiting on Nov 04, 2025 at 21:37:58 CST  (total: 00:00:21)
```
- Regex pattern: `TOOL:\s+xrun.*\(total:\s+(\d+):(\d+):(\d+)\)`
- Extracts hours, minutes, seconds
- Converts to total seconds: `hours * 3600 + minutes * 60 + seconds`

**VCS Log Format:**
```
CPU Time: 0.450 seconds; Data structure size: 0.0Mb
```
- Regex pattern: `CPU Time:\s+([\d.]+)\s+seconds`
- Extracts decimal seconds directly

**Behavior:**
- Parsed from `dv/uvm/core_ibex/out/run/tests/<testname>/rtl_sim.log`
- Represents actual simulator execution time (wall-clock)
- Returns `None` if log parsing fails (warning logged)
- Supports both simulator formats automatically

## Interpreting the Plots

### Histogram
- **Green bars**: High test pass rate in this runtime range
- **Red/yellow bars**: Lower pass rate, may indicate flaky or problematic tests
- **Long tail**: A few very slow tests may be candidates for optimization

### By Test Chart
- **Longer bars**: Tests that consume more execution time
- **Red bars**: Tests with low pass rates
- **Value labels**: Show mean runtime and pass rate percentage

### Scatter Plot
- **Clustering**: Groups of tests with similar runtimes
- **Outliers**: Individual tests that are unusually fast or slow
- **Red X marks**: Failed tests, check if they correlate with longer runtimes

### Cumulative Plot
- **Steep sections**: Periods with many fast tests
- **Flat sections**: Long-running individual tests
- **Final value**: Total regression execution time

## Examples

### Example Output Location
```
out/run/
├── runtime_histogram.png     # Runtime distribution
├── runtime_by_test.png       # Test comparison
├── runtime_scatter.png       # All test runtimes
├── cumulative_runtime.png    # Total execution time
├── report.html               # HTML report
├── report.json               # JSON report
└── summary.svg               # SVG dashboard
```

### Typical Runtime Statistics
```
Total Tests: 150
Mean: 45.23s
Median: 42.10s
Min: 15.30s
Max: 180.45s
Total Runtime: 1h 53m 24s
```

## Troubleshooting

### No Runtime Data
If plots show "No runtime data available":
- Make sure tests have been run with the updated `run_rtl.py`
- Check that `rtl_sim.log` files contain the runtime line:
  - For xrun: Look for "TOOL: xrun... (total: HH:MM:SS)"
  - For VCS: Look for "CPU Time: X.XXX seconds"
- Verify `runtime_s` field is not None in test YAML files
- Check log warnings for parsing errors

### Matplotlib Import Errors
If you see "matplotlib not available":
```bash
# Install matplotlib
pip install matplotlib numpy

# Or use conda
conda install matplotlib numpy
```

### Missing Plots
If some plots are not generated:
- Check log output for error messages
- Ensure output directory is writable
- Verify sufficient test data exists (at least a few tests with runtime data)

## Future Enhancements

Potential improvements to the runtime plotting system:
- Interactive HTML plots using plotly
- Runtime trend analysis across multiple regression runs
- Performance regression detection
- Test parallelization optimization suggestions
- Integration with CI/CD dashboards
- Comparison plots between different configurations

## See Also

- `collect_results.py`: Main result collection script
- `test_run_result.py`: Test result data structure
- `run_rtl.py`: RTL simulation execution
- `report_lib/`: Other report generation modules
