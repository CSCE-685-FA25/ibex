# IBEX Codebase Exploration: Regression Testing Infrastructure

## Executive Summary

The IBEX (lowRISC's RISC-V processor core) codebase contains a comprehensive hardware verification infrastructure with multiple verification approaches. The system is designed to run hardware regressions, collect coverage metrics, and generate detailed reports. This infrastructure is amenable to machine learning feature extraction for test prioritization and failure prediction.

---

## 1. TYPES OF REGRESSIONS BEING RUN

### 1.1 Main Verification Approach: UVM-based RTL Simulation
- **Location**: `/home/a2zaustin/ibex/dv/uvm/core_ibex/`
- **Simulator**: Xcelium (xlm) - primary, VCS (vcs) supported as alternative, Questa (questa), Vivado Simulator (dsim)
- **RTL Testbench**: SystemVerilog UVM framework

### 1.2 Test Categories

#### A. RISC-V DV Random Instruction Generation Tests
- **Source**: Integrated from Google's RISC-V-DV tool (`vendor/google_riscv-dv/`)
- **Purpose**: Generate random RISC-V instruction sequences to stress-test the processor
- **Config File**: `/home/a2zaustin/ibex/dv/uvm/core_ibex/riscv_dv_extension/testlist.yaml`
- **Test Examples**:
  - `riscv_arithmetic_basic_test`: Pure arithmetic, 10,000 instructions x 10 iterations
  - `riscv_machine_mode_rand_test`: Random machine mode instructions
  - `riscv_rand_instr_test`: Comprehensive random instruction testing
  - `riscv_rand_jump_test`: Jump and branch stress testing
  - CSR (Control & Status Register) tests
  - Interrupt and exception handling tests

#### B. Directed Tests
- **Location**: `/home/a2zaustin/ibex/dv/uvm/core_ibex/directed_tests/`
- **Config File**: `/home/a2zaustin/ibex/dv/uvm/core_ibex/directed_tests/directed_testlist.yaml`
- **Purpose**: Targeted tests for specific features and corner cases

#### C. Other Verification Approaches
- **Formal Verification**: `/home/a2zaustin/ibex/dv/formal/` - Using formal property checking
- **Cosimulation**: `/home/a2zaustin/ibex/dv/cosim/` - RTL vs ISS (Instruction Set Simulator) comparison
- **RISC-V Compliance**: `/home/a2zaustin/ibex/dv/riscv_compliance/` - Standards compliance testing
- **Verilator Simulation**: `/home/a2zaustin/ibex/dv/verilator/` - Open-source simulation
- **Register Verification**: `/home/a2zaustin/ibex/dv/cs_registers/` - Control/Status register testing

---

## 2. OUTPUT/DATA GENERATED FROM REGRESSIONS

### 2.1 Directory Structure (from metadata.py)
```
$OUT_DIR/
├── metadata/
│   ├── metadata.pickle      # Python serialized metadata object
│   ├── metadata.yaml        # YAML representation of metadata
│   └── [testname.seed].pickle  # Per-test result metadata
├── build/
│   ├── instr_gen/           # Instruction generator artifacts
│   └── tb/                  # Testbench compilation artifacts
├── run/
│   ├── tests/               # Per-test execution directories
│   │   └── [test_name.seed]/
│   │       ├── test.S       # Generated assembly file
│   │       ├── test.bin     # Compiled binary
│   │       ├── rtl_sim.log  # RTL simulation log
│   │       ├── ibex.trace   # Instruction trace
│   │       ├── instr.csv    # CSV format instruction trace
│   │       ├── coverage/    # Per-test coverage database
│   │       │   └── test.ucd # Xcelium coverage database
│   │       ├── trr.yaml     # Test Run Result metadata
│   │       └── trr.pickle   # Serialized test result object
│   ├── regr.log             # Summary regression log
│   ├── report.html          # HTML regression report
│   ├── report.json          # JSON regression report (dvsim format)
│   ├── summary.svg          # SVG dashboard visualization
│   ├── regr_junit.xml       # JUnit XML test results
│   └── regr_junit_merged.xml # Merged JUnit XML
└── coverage/
    ├── fcov/                # Functional coverage from RISC-V-DV
    ├── shared_cov/          # Cross-test coverage data
    ├── merged/              # Merged coverage database (.ucm)
    │   └── cov_db_runfile   # List of merged coverage runs
    └── report/              # Coverage reports
        ├── cov_report.txt   # Text-based coverage summary
        ├── cov_report_cg.txt # Functional coverage report
        ├── grading          # Coverage grading/ranking
        └── [HTML reports]   # Generated HTML coverage views
```

### 2.2 Per-Test Output Files

#### Log Files
- **rtl_sim.log**: Main RTL simulation log with instruction trace
- **rtl_stdout**: Standard output from simulator
- Various generation and compilation logs (riscvdv_run, compile_asm, etc.)

#### Binary and Assembly
- **test.S**: Generated or directed RISC-V assembly code
- **test.bin**: Compiled binary executable
- **vmem files**: Memory initialization files

#### Execution Results
- **ibex.trace**: Raw instruction execution trace (PC, instruction, register updates)
- **instr.csv**: CSV format trace (standardized RISC-V-DV format with ABI names)
- **cosim_trace**: Co-simulation trace if ISS comparison enabled

#### Coverage Artifacts
- **.ucd files** (Xcelium): Per-test coverage databases
- **.ucm files**: Merged coverage databases
- **cov_report.txt**: Code coverage percentages (block, branch, statement, expression, toggle, FSM, assertion)
- **cov_report_cg.txt**: Functional coverage (covergroups) averages

### 2.3 Metadata Objects (Python Pickle/YAML)

Each test and regression generates:
- **RegressionMetadata**: Top-level run configuration and results
  - git_commit, creation_datetime
  - simulator type, ISS, test names, seed values
  - Paths to all generated artifacts
  - Compilation and merge commands executed
  
- **TestRunResult**: Individual test results
  - testname, seed, test type (RISCV_DV or DIRECTED)
  - passed/failed status, failure_mode, failure_message
  - Paths to binaries, logs, traces
  - timeout information

### 2.4 Report Formats

Multiple output formats for different purposes:
- **Text**: `regr.log` - Human-readable summary
- **JSON**: `report.json` - dvsim-compatible format
- **JUnit XML**: `regr_junit.xml` - CI/CD integration
- **HTML**: `report.html` - Web-based visualization
- **SVG**: `summary.svg` - Dashboard visualization
- **HTML Coverage Reports**: Generated by Cadence IMC tool

---

## 3. COVERAGE AND METRICS COLLECTED

### 3.1 Code Coverage Metrics (RTL)

Tracked by Xcelium simulator (`parse_xcelium_cov_report` in `report_lib/util.py`):

```python
IBEX_COVERAGE_METRICS = [
    'block',        # Branch coverage - which basic blocks executed
    'branch',       # Branch decision coverage
    'statement',    # Statement/line coverage
    'expression',   # Expression coverage
    'toggle',       # Signal toggle coverage (0->1 and 1->0)
    'fsm',          # FSM state and transition coverage
    'assertion'     # Assertion coverage
]
```

**Coverage Representation**:
- For each metric: `{covered: int, total: int, pct: float}`
- Example: `block-covered: {covered: 850, total: 1200, pct: 0.708}`

**Module-level Granularity**:
- `ibex_top` (main module)
- Sub-modules (ALU, decoder, CSR, MMU, etc.)

### 3.2 Functional Coverage (UVM Covergroups)

Defined in `/home/a2zaustin/ibex/dv/uvm/core_ibex/fcov/`:
- **core_ibex_fcov_if.sv**: ~36KB of SystemVerilog coverage definitions
- **core_ibex_pmp_fcov_if.sv**: Physical Memory Protection coverage
- **core_ibex_csr_categories.svh**: CSR-specific coverage bins

Tracks:
- Instruction execution patterns
- Register operations
- Memory access patterns
- Exception handling
- CSR state transitions

**Functional Coverage Metrics**:
- Per-module covergroup averages (0-100%)
- `calc_cg_average()`: Aggregated covergroup coverage across modules

### 3.3 Coverage Extraction and Processing

**Coverage Databases**:
- Xcelium produces `.ucd` files (Unified Coverage Database)
- Merged into `.ucm` files using Cadence IMC tool
- Processed by Tcl scripts (`cov_merge.tcl`, `cov_report.tcl`)

**Coverage Label Generation** (`coverage_labeler.py`):
- Replays coverage merge incrementally
- For each test: calculates delta in covered items per metric
- Labels tests based on coverage contribution
- Outputs JSONL format with:
  - test name, seed
  - metrics_before/after (percentages)
  - covered_deltas (items gained per metric)
  - covergroup_before/after
  - label (binary: contributes to coverage or not)
  - triggers (which metrics changed)

---

## 4. EXISTING DATA PROCESSING AND ANALYSIS SCRIPTS

### 4.1 Core Processing Pipeline

#### Metadata Management
- **metadata.py** (26KB): Central configuration and result aggregation
  - `RegressionMetadata.construct_from_metadata_dir()`: Load regression state
  - Tracks all paths, commands, and intermediate results
  - Persists as pickle + YAML for reproducibility

#### Test Result Collection
- **collect_results.py**: Aggregates all per-test results into summaries
  - Reads individual `TestRunResult` pickle files
  - Generates multiple report formats
  - Called after all simulations complete

#### Report Generation (`report_lib/`)
- **util.py** (232 lines):
  - `parse_xcelium_cov_report()`: Regex parsing of text coverage reports
  - `create_test_summary_dict()`: Aggregate pass/fail statistics
  - `create_cov_summary_dict()`: Extract coverage metrics to dict
  - `calc_cg_average()`: Average functional coverage
  - CSS gradient coloring for visualization

- **text.py**: Plain-text report generation
- **html.py**: HTML report with embedded visualizations
- **junit_xml.py**: JUnit XML for CI/CD
- **dvsim_json.py**: Machine-readable JSON format
- **svg.py**: Dashboard visualization

#### Coverage Label Generation
- **coverage_labeler.py** (435 lines): Feature extraction for ML
  - Replays entire coverage merge sequence
  - Computes incremental coverage gains per test
  - Produces JSONL output:
    ```json
    {
      "index": 1,
      "testdotseed": "test_name.seed_value",
      "coverage_path": "/path/to/coverage/db",
      "metrics_before": {"block": 0.45, "branch": 0.38, ...},
      "metrics_after": {"block": 0.47, "branch": 0.40, ...},
      "covered_deltas": {"block": 12, "branch": 5, ...},
      "covergroup_before": 0.62,
      "covergroup_after": 0.64,
      "covergroup_delta": 0.02,
      "label": 1,
      "triggers": ["block", "branch"]
    }
    ```

### 4.2 Trace Processing

#### Log-to-CSV Conversion
- **ibex_log_to_trace_csv.py** (279 lines): Standardizes execution traces
  - Parses raw Ibex simulation logs (instruction trace output)
  - Extracts: PC, binary, instruction, register updates
  - Converts to standard RISC-V-DV CSV format
  - Converts register names to ABI format (x6 -> t1)
  - Outputs per-test `instr.csv`

#### Test Validation
- **check_ibex_uvm_log()**: Validates UVM simulation logs
  - Checks for `RISC-V UVM TEST PASSED/FAILED` markers
  - Detects timeout failures
  - Returns failure mode enum

### 4.3 Merge and Analysis Commands

#### Coverage Merge
- **merge_cov.py** (161 lines): Orchestrates coverage database merging
  - Finds all `.ucd` files in test directory
  - Invokes Cadence IMC with `cov_merge.tcl` and `cov_report.tcl`
  - Generates text and HTML coverage reports
  - Supports both Xcelium (xlm) and VCS (vcs) simulators

#### Functional Coverage (RISC-V-DV)
- **get_fcov.py**: Calls riscv-dv's coverage generation
  - Processes instruction traces
  - Generates functional coverage metrics
  - Output: coverage reports in fcov directory

### 4.4 Configuration and Setup

- **scripts_lib.py**: Common utilities
  - `run_one()`: Execute external commands with logging
  - `format_dict_to_printable_dict()`: Serialize for output
  - File I/O, subprocess handling

- **setup_imports.py**: Path setup for Python modules
  - Discovers IBEX, RISC-V-DV, lowRISC IP paths
  - Sets up PYTHONPATH

---

## 5. OVERALL TEST/REGRESSION INFRASTRUCTURE STRUCTURE

### 5.1 Execution Flow

```
Makefile (top level)
    |
    +-> metadata.py (create_metadata)
    |   - Sets up directory structure
    |   - Creates RegressionMetadata object
    |   - Saves to pickle/YAML
    |
    +-> wrapper.mk
        |
        +-> core_config (ibex config extraction)
        |
        +-> instr_gen_build (compile RISC-V-DV generator)
        |
        +-> instr_gen_run (generate random test vectors)
        |
        +-> compile_riscvdv_tests (gcc compile .S -> .bin)
        |
        +-> compile_directed_tests (directed test compilation)
        |
        +-> rtl_tb_compile (compile UVM testbench)
        |
        +-> rtl_sim_run (parallel: execute per-test RTL sims)
        |   - For each test+seed:
        |     - Run RTL simulation
        |     - Collect coverage (.ucd)
        |     - Generate trace log
        |     - Validate pass/fail
        |     - Save TestRunResult pickle
        |
        +-> check_logs (validate all test logs)
        |
        +-> riscv_dv_fcov (functional coverage generation)
        |
        +-> merge_cov (merge coverage databases)
        |   - imc merge (combine .ucd files)
        |   - imc report (generate text/HTML reports)
        |
        +-> collect_results (generate final reports)
            - Read all TestRunResult pickles
            - Generate multiple report formats
            - Compute coverage summary
            - Output: regr.log, report.html, report.json, etc.
```

### 5.2 Configuration System

- **ibex_configs.yaml**: Processor configuration options
  - ISA extensions (M, C, B variants)
  - Pipeline depth
  - Memory protection features
  - PMP (Physical Memory Protection) rules

- **testlist.yaml**: Test definitions
  - Test name, description
  - gen_opts: RISC-V-DV instruction generator options
  - rtl_test: UVM test class
  - iterations: Number of seeds per test

- **metadata.yaml**: Per-regression snapshot
  - Git commit hash
  - Timestamp
  - All configuration parameters
  - Paths to all outputs

### 5.3 Test Execution Parallelism

- Instruction generation: parallelizable (different seeds)
- Compilation: parallelizable per test
- RTL simulation: **fully parallelizable** per test+seed
  - Makefile uses: `make -j 8` (8 parallel jobs)
  - SLURM job script: 72-hour timeout, 8 cores

### 5.4 Supported Simulators

- **Xcelium (xlm)**: Primary - full coverage support
- **VCS (vcs)**: Alternative - coverage supported
- **Questa (questa)**: Supported
- **DSIM (dsim)**: Supported
- **Verilator**: Open-source (limited features)

---

## 6. DATA READY FOR ML/ANALYTICS

### 6.1 Available Features per Test

From metadata and execution:
1. **Test characteristics**:
   - Test name (categorical)
   - Test type (RISCV_DV vs DIRECTED)
   - Seed value (numeric)
   - Instruction count (from test config)
   - Sub-programs (from gen_opts)
   - CSR options (from gen_opts)

2. **Execution metrics**:
   - Wall-clock runtime
   - Simulation cycles
   - Memory usage (from logs)
   - Timeout (yes/no)

3. **Code coverage by test**:
   - block, branch, statement, expression, toggle, fsm, assertion
   - Coverage percentage
   - Items covered delta (absolute)

4. **Functional coverage**:
   - Covergroup average before/after
   - Delta in covergroup coverage

5. **Test outcomes**:
   - Pass/fail
   - Failure mode (timeout, error, etc.)
   - Coverage contribution (label)

### 6.2 Output Formats

- **JSONL** (`coverage_labels.jsonl`): Per-test coverage labels for supervised learning
- **CSV**: Instruction traces (`instr.csv`)
- **Pickle**: Full Python objects for programmatic access
- **HTML/JSON**: Reports for human analysis

### 6.3 Data Extraction Entry Points

```python
# Load regression metadata
from metadata import RegressionMetadata
md = RegressionMetadata.construct_from_metadata_dir(Path("out/metadata"))

# Access test results
for pickle_file in md.tests_pickle_files:
    trr = TestRunResult.construct_from_pickle(pickle_file)
    # trr.passed, trr.failure_mode, trr.testname, trr.seed, etc.

# Access coverage reports
from report_lib.util import parse_xcelium_cov_report, create_cov_summary_dict
cov_dict = create_cov_summary_dict(md)  # Returns coverage metrics

# Access per-test traces
# Each test has: out/run/tests/[testname.seed]/instr.csv
```

---

## 7. ML OPPORTUNITY AREAS

### 7.1 Test Prioritization
- **Predict coverage contribution** of each test
- Use coverage_labeler.py output as ground truth
- Features: test name, seed patterns, test parameters
- Goal: Reorder test execution to maximize coverage early

### 7.2 Failure Prediction
- **Predict test failures** before execution
- Features: test type, seed, ISA options, memory config
- Use pass/fail history to train classifier
- Goal: Skip likely-passing tests, focus on edge cases

### 7.3 Coverage Prediction
- **Predict final coverage** from partial regression data
- Features: per-test coverage metrics to date
- Regression task: estimate total coverage at completion
- Goal: Estimate when regression complete

### 7.4 Anomaly Detection
- **Detect unusual test behaviors**
- Feature extraction from execution traces
- Identify tests that exercise rare code paths
- Flag potential corner cases for manual review

### 7.5 Root Cause Analysis
- **Link failures to code changes**
- Correlate test failures with git commits
- Instruction trace analysis to identify problematic code sequences
- Goal: Automate failure diagnosis

---

## Key Files for ML Data Access

```
/home/a2zaustin/ibex/
├── coverage_labeler.py              # ML-ready coverage labels (JSONL output)
├── dv/uvm/core_ibex/
│   ├── scripts/
│   │   ├── metadata.py              # Load regression state
│   │   ├── test_run_result.py       # Test result objects
│   │   ├── collect_results.py       # Report aggregation
│   │   ├── merge_cov.py             # Coverage database processing
│   │   ├── report_lib/
│   │   │   ├── util.py              # Coverage parsing
│   │   │   └── dvsim_json.py        # JSON format export
│   │   └── riscv_dv_extension/
│   │       └── ibex_log_to_trace_csv.py  # Trace standardization
│   ├── riscv_dv_extension/testlist.yaml  # Test definitions
│   └── Makefile                     # Regression orchestration
├── python-requirements.txt          # Dependencies
└── regression.sh                    # SLURM job script
```

---

## Summary

The IBEX codebase contains a **mature, multi-layered verification infrastructure** ideal for ML applications:

- **Multiple verification approaches**: UVM simulation, formal, compliance, cosimulation
- **Rich output data**: Per-test metadata, coverage databases, execution traces, pass/fail outcomes
- **Standardized formats**: Pickle, YAML, JSON, CSV, coverage databases (UCD/UCM)
- **Existing analysis scripts**: Coverage merge, report generation, trace standardization
- **Scalable infrastructure**: Parallel execution, SLURM support, multiple simulators
- **ML-friendly output**: coverage_labeler.py already produces JSONL feature vectors

The system is well-suited for:
- Test prioritization and ordering
- Failure prediction
- Coverage estimation
- Anomaly detection in test execution
- Correlation analysis between code changes and test results

