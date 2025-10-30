# Feature Extractor for ML-Based Regression Analysis

## Overview

`feature_extractor.py` augments the coverage labels produced by `coverage_labeler.py` with additional features extracted from test execution data. This creates a comprehensive feature set suitable for training machine learning models for test prioritization, failure prediction, and regression optimization.

## Architecture

```
coverage_labels.jsonl  ────┐
                           │
test metadata pickles  ────┤
                           ├──> feature_extractor.py ──> enhanced_features.jsonl
simulation logs  ──────────┤
                           │
trace CSV files  ──────────┘
```

## Features Extracted

### 1. Test Metadata Features (from TestRunResult pickle)

| Feature | Type | Description |
|---------|------|-------------|
| `passed` | bool | Whether the test passed |
| `failure_mode` | str | Failure mode: NONE, TIMEOUT, FILE_ERROR, LOG_ERROR |
| `timeout_s` | int | Timeout duration in seconds |
| `test_type` | str | RISCVDV or DIRECTED |
| `test_name` | str | Name of the test |
| `seed` | int | Random seed used |
| `simulator` | str | RTL simulator used (e.g., xlm, vcs) |
| `iss` | str | Instruction Set Simulator used for cosimulation |
| `gen_opts_count` | int | Number of generation options |
| `sim_opts_count` | int | Number of simulation options |

### 2. Execution Features (from logs and traces)

#### From Simulation Logs:
| Feature | Type | Description |
|---------|------|-------------|
| `log_exists` | bool | Whether log file exists |
| `log_size_bytes` | int | Size of log file in bytes |
| `instruction_count_from_log` | int | Instruction count parsed from log |
| `cycle_count` | int | Cycle count from simulation |
| `exception_count` | int | Number of exceptions/interrupts |
| `log_line_count` | int | Number of lines in log |

#### From Trace CSV:
| Feature | Type | Description |
|---------|------|-------------|
| `trace_exists` | bool | Whether trace CSV exists |
| `trace_instruction_count` | int | Total instructions in trace |
| `trace_branch_count` | int | Number of branch instructions |
| `trace_load_count` | int | Number of load instructions |
| `trace_store_count` | int | Number of store instructions |
| `trace_arithmetic_count` | int | Number of arithmetic instructions |
| `trace_jump_count` | int | Number of jump instructions (jal, jalr) |
| `trace_system_count` | int | Number of system instructions (ecall, csr, etc.) |
| `trace_unique_pcs` | int | Number of unique program counters visited |

### 3. Derived Features (computed from above)

| Feature | Type | Description |
|---------|------|-------------|
| `branch_ratio` | float | Ratio of branch instructions to total |
| `load_ratio` | float | Ratio of load instructions to total |
| `store_ratio` | float | Ratio of store instructions to total |
| `memory_ratio` | float | Ratio of memory ops (load+store) to total |
| `jump_ratio` | float | Ratio of jump instructions to total |
| `system_ratio` | float | Ratio of system instructions to total |
| `arithmetic_ratio` | float | Ratio of arithmetic instructions to total |
| `control_flow_complexity` | float | Unique PCs / total instructions (code coverage proxy) |
| `cpi` | float | Cycles per instruction (if cycle count available) |
| `exception_rate` | float | Exceptions per instruction |

### 4. Original Coverage Features (from coverage_labeler.py)

All original features are preserved:
- `index`, `testdotseed`, `coverage_path`
- `metrics_before`, `metrics_after`, `covered_deltas`
- `covergroup_before`, `covergroup_after`, `covergroup_delta`
- `label`, `triggers`

## Usage

### Basic Usage

```bash
# After running coverage_labeler.py
python3 feature_extractor.py \
  --input coverage_labels.jsonl \
  --output enhanced_features.jsonl \
  --metadata ibex/dv/uvm/core_ibex/out/metadata
```

### Options

- `--input PATH` (required): Input JSONL file from coverage_labeler.py
- `--output PATH` (required): Output JSONL file with augmented features
- `--metadata PATH`: Path to regression metadata directory containing test pickle files
- `--limit N`: Process only first N records (useful for testing)
- `--verbose`: Print detailed progress information

### Example Workflow

```bash
# Step 1: Run regression with coverage
cd ibex/dv/uvm/core_ibex
make COV=1 ITERATIONS=100

# Step 2: Generate coverage labels
python3 /path/to/coverage_labeler.py \
  --metadata out/metadata \
  --output coverage_labels.jsonl

# Step 3: Extract additional features
python3 /path/to/feature_extractor.py \
  --input coverage_labels.jsonl \
  --output enhanced_features.jsonl \
  --metadata out/metadata \
  --verbose
```

## Output Format

The output is a JSONL file where each line contains a JSON object with the following structure:

```json
{
  "index": 1,
  "testdotseed": "riscv_arithmetic_basic_test.0",
  "coverage_path": "/path/to/coverage.ucd",

  "metrics_before": {"block": 0.75, "branch": 0.68, ...},
  "metrics_after": {"block": 0.77, "branch": 0.69, ...},
  "covered_deltas": {"block": 145.0, "branch": 23.0, ...},

  "covergroup_before": 45.2,
  "covergroup_after": 47.8,
  "covergroup_delta": 2.6,

  "label": 1,
  "triggers": ["block", "branch"],

  "test_metadata": {
    "passed": true,
    "failure_mode": null,
    "timeout_s": 1800,
    "test_type": "RISCVDV",
    "test_name": "riscv_arithmetic_basic_test",
    "seed": 0,
    "simulator": "xlm",
    "iss": "spike",
    "gen_opts_count": 5,
    "sim_opts_count": 3
  },

  "execution_features": {
    "log_exists": true,
    "log_size_bytes": 524288,
    "instruction_count_from_log": 5000,
    "cycle_count": 12000,
    "exception_count": 2,
    "log_line_count": 5123,
    "trace_exists": true,
    "trace_instruction_count": 5000,
    "trace_branch_count": 750,
    "trace_load_count": 800,
    "trace_store_count": 700,
    "trace_arithmetic_count": 1500,
    "trace_jump_count": 100,
    "trace_system_count": 50,
    "trace_unique_pcs": 1250
  },

  "derived_features": {
    "branch_ratio": 0.15,
    "load_ratio": 0.16,
    "store_ratio": 0.14,
    "memory_ratio": 0.30,
    "jump_ratio": 0.02,
    "system_ratio": 0.01,
    "arithmetic_ratio": 0.30,
    "control_flow_complexity": 0.25,
    "cpi": 2.4,
    "exception_rate": 0.0004
  }
}
```

## ML Use Cases

### 1. Test Prioritization
**Goal**: Predict which tests will contribute most to coverage

**Target**: `label` (binary) or `covered_deltas` (regression)

**Key Features**:
- Instruction mix ratios (branch_ratio, memory_ratio, etc.)
- Control flow complexity
- Test type and configuration
- Historical coverage deltas

### 2. Failure Prediction
**Goal**: Predict test failures before running

**Target**: `passed` (binary)

**Key Features**:
- Test type and seed
- Test configuration (gen_opts_count, sim_opts_count)
- Historical failure rates (requires temporal features)

### 3. Runtime Estimation
**Goal**: Predict test execution time

**Target**: `cycle_count` or runtime from logs

**Key Features**:
- Instruction count
- Instruction mix
- Memory access ratio
- Test configuration

### 4. Coverage Estimation
**Goal**: Estimate coverage contribution without full merge

**Target**: `covered_deltas`

**Key Features**:
- All execution features
- Derived features (control_flow_complexity, ratios)
- Test metadata

## Feature Engineering Tips

### Loading Data for ML

```python
import json
import pandas as pd

# Load enhanced features
records = []
with open("enhanced_features.jsonl", "r") as f:
    for line in f:
        records.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(records)

# Flatten nested features
test_meta = pd.json_normalize(df['test_metadata'])
exec_feat = pd.json_normalize(df['execution_features'])
derived = pd.json_normalize(df['derived_features'])

# Combine into feature matrix
features = pd.concat([
    df[['index', 'testdotseed', 'label']],
    test_meta,
    exec_feat,
    derived
], axis=1)

print(features.head())
```

### Handling Missing Values

Some features may be missing if:
- Trace CSV doesn't exist (early test termination)
- Log parsing failed
- Metadata pickle not found

Strategies:
- Impute with median/mean for numeric features
- Create indicator variables for missing data
- Use models that handle missing values (e.g., XGBoost)

### Feature Selection

High-value features for test prioritization:
1. `control_flow_complexity` - Proxy for code coverage
2. `branch_ratio` - Branch-heavy tests often hit new paths
3. `instruction_count` - Longer tests may cover more
4. `test_type` - Random vs directed tests behave differently
5. Coverage deltas from previous runs (temporal features)

## Extending the Feature Extractor

### Adding Custom Features

To add new features, modify `feature_extractor.py`:

1. **Add log parsing**: Update `parse_simulation_log()` with new regex patterns
2. **Add trace analysis**: Update `parse_trace_csv()` with new metrics
3. **Add derived features**: Update `compute_derived_features()` with calculations
4. **Add metadata features**: Update `extract_test_metadata_features()`

Example - Adding custom pattern detection:

```python
def parse_simulation_log(log_path: Path) -> Dict[str, Any]:
    # ... existing code ...

    # Add custom pattern
    CUSTOM_PATTERN = re.compile(r"MY_PATTERN: (\d+)")
    features["my_custom_metric"] = 0

    for line in lines:
        m = CUSTOM_PATTERN.search(line)
        if m:
            features["my_custom_metric"] = int(m.group(1))

    return features
```

## Performance Characteristics

- **Speed**: ~100-1000 records/second (depends on file I/O)
- **Memory**: Minimal - processes one record at a time
- **Disk**: Reads metadata pickles, logs, traces per test
- **Bottlenecks**: File I/O (especially for large trace CSVs)

## Troubleshooting

### Common Issues

**Issue**: "Could not load test result from pickle"
- **Cause**: Metadata directory path incorrect or pickle files missing
- **Fix**: Verify `--metadata` points to correct directory containing `.pickle` files

**Issue**: No execution features extracted
- **Cause**: Log/trace files don't exist or paths not set in TestRunResult
- **Fix**: Ensure regression ran successfully and files weren't cleaned up

**Issue**: All trace features are zero
- **Cause**: Trace CSV format doesn't match expected format
- **Fix**: Check CSV format, update `parse_trace_csv()` if needed

**Issue**: Import errors for ibex modules
- **Cause**: Python path not set correctly
- **Fix**: Run from repository root or adjust sys.path in script

## Future Enhancements

Potential additions to the feature extractor:

1. **Temporal Features**: Track test history across multiple regressions
2. **Hierarchical Coverage**: Extract per-module, per-file coverage
3. **Code Change Correlation**: Link tests to git changes
4. **Functional Coverage Details**: Extract individual coverpoint data
5. **Pipeline Stall Analysis**: Parse performance counters from logs
6. **Data Dependency Graphs**: Build from trace analysis
7. **Batch Processing**: Parallel feature extraction for large datasets

## References

- `coverage_labeler.py` - Generates base coverage labels
- `metadata.py` - Regression metadata schema
- `test_run_result.py` - Test execution metadata schema
- `ibex_log_to_trace_csv.py` - Trace CSV format specification
