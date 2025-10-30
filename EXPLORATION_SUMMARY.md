# IBEX Codebase Exploration - Executive Summary

## What Was Explored

Complete analysis of the IBEX (lowRISC RISC-V processor) hardware verification infrastructure to understand:
1. Types of regressions being run
2. Output data generated
3. Coverage and metrics collected
4. Data processing and analysis capabilities
5. ML/analytics readiness

## Key Findings

### 1. Regression Types

**Primary: UVM-based RTL Simulation** 
- Hardware simulation of RISC-V processor using Xcelium (also supports VCS, Questa, DSIM)
- Two categories of tests:
  - RISC-V-DV: Random instruction generation (from Google's RISC-V-DV framework)
  - Directed: Targeted tests for specific features
- Example tests: arithmetic, machine mode, jump/branch, CSR operations, exceptions

**Secondary Approaches**:
- Formal verification (property checking)
- Cosimulation (RTL vs Spike ISS comparison)
- RISC-V compliance testing
- Verilator open-source simulation
- Control/Status register verification

### 2. Output Data Generated

**Per-Test Outputs**:
- Assembly file (test.S) and compiled binary (test.bin)
- RTL simulation log (rtl_sim.log) with full instruction trace
- Instruction trace in standard CSV format (instr.csv)
- Per-test coverage database (.ucd file)
- Test result metadata (pass/fail, failure mode, timeout info)
- Execution traces with register updates

**Regression-Level Outputs**:
- Merged coverage database (.ucm file) combining all tests
- Text and HTML coverage reports showing:
  - 7 code coverage metrics: block, branch, statement, expression, toggle, FSM, assertion
  - Per-metric: items covered, total items, percentage
  - Functional coverage (UVM covergroups) per module
- Multiple report formats: text, HTML, JSON (dvsim), JUnit XML, SVG
- Regression metadata: test list, git commit, timestamps, all paths

**ML-Ready Outputs**:
- JSONL file (coverage_labels.jsonl) with per-test coverage contribution labels
- Each record includes: test name, seed, coverage before/after, delta, label (contributes or not)

### 3. Coverage Metrics

**Code Coverage Tracked** (7 metrics):
- block: Basic block execution
- branch: Branch decision coverage
- statement: Statement/line coverage
- expression: Expression coverage
- toggle: Signal state transitions
- fsm: State machine coverage
- assertion: Assertion coverage

**Functional Coverage**:
- UVM covergroup averages per module
- Instruction execution patterns
- Register operations
- Memory access patterns
- Exception handling
- CSR state transitions

**Collection Infrastructure**:
- Xcelium simulator generates coverage databases (.ucd)
- Cadence IMC tool merges and reports
- Coverage labeler script computes incremental contribution per test
- Results exported to JSONL for ML feature extraction

### 4. Data Processing Pipeline

**Key Scripts**:
- `metadata.py`: Central configuration and result aggregation (26KB, comprehensive)
- `collect_results.py`: Aggregates per-test results into regression summaries
- `merge_cov.py`: Orchestrates coverage database merging
- `ibex_log_to_trace_csv.py`: Converts raw logs to standardized trace format
- `report_lib/util.py`: Parses coverage reports (regex-based)
- `coverage_labeler.py`: ML feature extraction with coverage labels (435 lines, JSONL output)

**Report Generation**:
- Multiple output formats: HTML (visualization), JSON (machine-readable), XML (CI/CD), SVG (dashboard)
- Text-based coverage reports (cov_report.txt, cov_report_cg.txt)

**Data Formats**:
- Python Pickle: Full object serialization for metadata and test results
- YAML: Human-readable configuration and results
- JSON/JSONL: Machine-readable (especially coverage_labels.jsonl)
- CSV: Instruction traces
- Coverage Databases: .ucd (per-test), .ucm (merged)

### 5. Infrastructure Characteristics

**Scalability**:
- Fully parallelizable per test+seed (8 jobs typical, SLURM support)
- 72-hour timeout for full regressions
- Support for multiple simulators and configurations

**Maturity**:
- Comprehensive error handling and validation
- Multiple output formats
- Reproducibility (git commit tracking, full metadata)
- Integration with CI/CD (JUnit XML output)

**ML Readiness**:
- coverage_labeler.py already produces ML-ready feature vectors (JSONL)
- Test metadata includes all configuration parameters
- Coverage data is granular (per metric, per module)
- Failure modes are categorized (timeout vs error)
- Instruction traces available in standardized format

---

## ML Opportunity Analysis

### High-Value Opportunities

1. **Test Prioritization** (HIGH IMPACT)
   - Predict which tests contribute to coverage
   - Ground truth: coverage_labeler.py labels
   - Feature space: test name, seed patterns, ISA options, test parameters
   - Goal: Reorder tests to maximize coverage early in regression

2. **Failure Prediction** (HIGH IMPACT)
   - Predict test failures before execution
   - Ground truth: pass/fail status, failure modes
   - Could save simulation time
   - Features: test type, seed, configuration options

3. **Coverage Estimation** (MEDIUM)
   - Predict final coverage from partial regression data
   - Could enable early stopping decisions
   - Features: cumulative coverage trajectory, time-series

4. **Anomaly Detection** (MEDIUM)
   - Identify unusual test execution patterns
   - Extract features from instruction traces
   - Find corner cases for manual review

5. **Root Cause Analysis** (MEDIUM)
   - Link failures to code changes
   - Trace analysis to identify problematic instruction sequences
   - Could automate failure diagnosis

---

## Key Data Access Points

### For Immediate Use
1. **coverage_labels.jsonl**: Pre-computed ML labels (binary: contributes to coverage)
2. **report_lib/util.py**: Coverage parsing functions
3. **metadata.py**: Load full regression state
4. **test_run_result.py**: Access individual test results

### Python API
```python
# Load metadata
from metadata import RegressionMetadata
md = RegressionMetadata.construct_from_metadata_dir(Path("out/metadata"))

# Access test results
for pickle_file in md.tests_pickle_files:
    trr = TestRunResult.construct_from_pickle(pickle_file)
    # passed, failure_mode, testname, seed, etc.

# Get coverage
cov_dict = create_cov_summary_dict(md)  # Returns metrics
```

---

## Files Generated

Documentation created in `/home/a2zaustin/ibex/`:

1. **CODEBASE_ANALYSIS.md** (12KB)
   - Comprehensive analysis of regression infrastructure
   - Detailed breakdown of test types, outputs, coverage metrics
   - Complete data processing pipeline description
   - ML opportunity areas

2. **ML_FEATURE_EXTRACTION_GUIDE.md** (10KB)
   - Practical guide with code examples
   - How to access metadata, test results, coverage metrics
   - Building feature matrices for ML
   - Common analysis patterns
   - Tips for feature engineering

3. **EXPLORATION_SUMMARY.md** (this file)
   - Executive summary
   - Key findings
   - ML opportunities
   - Quick reference

---

## Next Steps

### For ML Project Planning
1. Use coverage_labels.jsonl as initial supervised learning dataset
2. Build baseline models for test prioritization
3. Expand features to include instruction count, test parameters, seed patterns
4. Evaluate on multiple regression runs (temporal validation)
5. Consider ensemble methods combining coverage and failure predictions

### For Infrastructure Enhancement
1. Existing infrastructure already supports ML well
2. Consider adding per-test coverage metrics (currently only merged)
3. Export more detailed execution statistics (runtime, memory)
4. Standardize trace format to enable sequence models

### For Data Collection
1. Run regressions with coverage_labeler.py to generate JSONL labels
2. Archive regression runs with metadata for reproducibility
3. Correlate with git commits for root cause analysis

---

## Conclusion

The IBEX codebase contains a **mature, production-ready verification infrastructure** with:
- Multiple verification approaches (UVM, formal, compliance, cosim)
- Rich, diverse output data (logs, traces, coverage databases, metadata)
- Existing ML feature extraction (coverage_labeler.py)
- Scalable, parallelizable execution
- Well-documented, object-oriented Python APIs

The infrastructure is **well-suited for ML applications** with immediate opportunities for:
- Test prioritization and optimization
- Failure prediction and early stopping
- Coverage estimation and tracking
- Anomaly detection in test execution

**Status**: Ready to begin ML feature engineering and model development.

