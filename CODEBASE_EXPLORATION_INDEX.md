# IBEX Codebase Exploration - Documentation Index

This directory contains comprehensive documentation of the IBEX hardware verification infrastructure, generated through systematic codebase exploration.

## Documents Created

### 1. **EXPLORATION_SUMMARY.md** - START HERE
**Purpose**: Executive summary and quick reference  
**Length**: 8.4 KB, 242 lines  
**Content**:
- What was explored and why
- Key findings on regression types, outputs, coverage metrics
- ML opportunity analysis (5 high-value opportunities identified)
- Quick data access examples
- Next steps and recommendations

**Best For**: Getting oriented, understanding value proposition, planning ML projects

---

### 2. **CODEBASE_ANALYSIS.md** - COMPREHENSIVE REFERENCE
**Purpose**: Detailed technical analysis  
**Length**: 19 KB, 520 lines  
**Content**:
- Detailed breakdown of regression infrastructure (UVM, RISC-V-DV, formal, etc.)
- Complete output directory structure
- Coverage metrics (7 code coverage types + functional coverage)
- Full data processing pipeline walkthrough
- Execution flow and parallelization strategy
- ML-ready data formats and access patterns
- Supported simulators and configurations

**Sections**:
1. Regression Types (UVM, RISC-V-DV, formal, cosim, compliance)
2. Output Data (directory structure, per-test files, metadata objects)
3. Coverage & Metrics (code coverage, functional coverage, extraction)
4. Data Processing Scripts (metadata.py, collect_results.py, coverage_labeler.py, etc.)
5. Infrastructure Structure (execution flow, configuration, parallelism)
6. Data Ready for ML (available features, output formats)
7. ML Opportunity Areas (test prioritization, failure prediction, etc.)

**Best For**: Deep understanding, implementation details, infrastructure design decisions

---

### 3. **ML_FEATURE_EXTRACTION_GUIDE.md** - PRACTICAL GUIDE
**Purpose**: Hands-on guide with code examples  
**Length**: 11 KB, 360 lines  
**Content**:
- How to load regression metadata
- Iterate over test results
- Extract coverage metrics
- Access pre-computed coverage labels (JSONL)
- Parse test configuration
- Read instruction traces
- Build feature matrices
- Common analysis patterns
- ML feature engineering tips

**Code Examples**:
- Load RegressionMetadata object
- Access TestRunResult pickles
- Parse coverage reports
- Build pandas DataFrames
- Extract test parameters
- Plot coverage trajectories

**Best For**: Getting started with data extraction, implementing feature engineering

---

## Quick Start Guide

### Step 1: Understand the System
Read **EXPLORATION_SUMMARY.md** (5-10 minutes)

### Step 2: Learn the Details
Read **CODEBASE_ANALYSIS.md** sections 1-3 (20-30 minutes)

### Step 3: Start Extracting Data
Follow examples in **ML_FEATURE_EXTRACTION_GUIDE.md** (hands-on)

### Step 4: Reference as Needed
Use **CODEBASE_ANALYSIS.md** section 5 for infrastructure details

---

## Key Discoveries

### The Infrastructure
- **Type**: Hardware verification for RISC-V processor (IBEX)
- **Primary Method**: UVM-based RTL simulation (Xcelium)
- **Test Generation**: Google's RISC-V-DV (random instruction generation)
- **Coverage Tools**: Cadence IMC, Xcelium
- **Scale**: Multiple tests, thousands of configurations, fully parallelizable

### The Data
- **Per-Test**: Assembly, binary, simulation log, instruction trace (CSV), coverage database (.ucd)
- **Regression-Level**: Merged coverage (.ucm), multiple report formats, metadata (pickle/YAML)
- **Coverage Metrics**: 7 code coverage types (block, branch, statement, expression, toggle, FSM, assertion) + functional coverage
- **ML-Ready**: coverage_labels.jsonl with test contribution labels

### The Scripts
- **metadata.py** (26KB): Central metadata management
- **collect_results.py**: Result aggregation
- **coverage_labeler.py** (435 lines): ML label generation
- **report_lib/util.py**: Coverage parsing
- **ibex_log_to_trace_csv.py**: Trace standardization
- **merge_cov.py**: Coverage database merging

### The Opportunities
1. **Test Prioritization** - Order tests to maximize coverage early
2. **Failure Prediction** - Predict test failures to save simulation time
3. **Coverage Estimation** - Estimate final coverage from partial runs
4. **Anomaly Detection** - Identify unusual test behaviors
5. **Root Cause Analysis** - Link failures to code changes

---

## File Organization

```
/home/a2zaustin/ibex/
├── EXPLORATION_SUMMARY.md          # START HERE - Executive summary
├── CODEBASE_ANALYSIS.md            # Detailed technical analysis
├── ML_FEATURE_EXTRACTION_GUIDE.md   # Hands-on practical guide
├── CODEBASE_EXPLORATION_INDEX.md    # This file
│
└── dv/uvm/core_ibex/
    ├── Makefile                    # Regression entry point
    ├── scripts/
    │   ├── metadata.py             # Load regression state
    │   ├── test_run_result.py      # Test result class
    │   ├── collect_results.py      # Report aggregation
    │   ├── merge_cov.py            # Coverage merging
    │   ├── report_lib/
    │   │   ├── util.py             # Coverage parsing
    │   │   └── dvsim_json.py       # JSON export
    │   └── riscv_dv_extension/
    │       └── ibex_log_to_trace_csv.py  # Trace conversion
    ├── riscv_dv_extension/testlist.yaml  # Test definitions
    └── out/                        # Regression output directory
        ├── metadata/               # Metadata pickle/YAML
        ├── run/tests/              # Per-test directories
        └── coverage/               # Coverage reports
```

---

## Key Code Sections Referenced

### Loading Data
- `metadata.py`: `RegressionMetadata.construct_from_metadata_dir()`
- `test_run_result.py`: `TestRunResult.construct_from_pickle()`

### Coverage Analysis
- `report_lib/util.py`: `parse_xcelium_cov_report()`, `create_cov_summary_dict()`
- `coverage_labeler.py`: Coverage merge replay and labeling

### Trace Processing
- `ibex_log_to_trace_csv.py`: Raw log to standardized CSV conversion

### Report Generation
- `collect_results.py`: Multi-format report generation
- `report_lib/`: HTML, JSON, XML, SVG outputs

---

## Data Access Examples

### Python - Load Metadata
```python
from metadata import RegressionMetadata
md = RegressionMetadata.construct_from_metadata_dir(Path("out/metadata"))
```

### Python - Access Test Results
```python
from test_run_result import TestRunResult
for pickle_file in md.tests_pickle_files:
    trr = TestRunResult.construct_from_pickle(pickle_file)
```

### Python - Get Coverage
```python
from report_lib.util import create_cov_summary_dict
cov_dict = create_cov_summary_dict(md)  # Returns coverage metrics
```

### Load Coverage Labels (ML)
```json
// coverage_labels.jsonl format
{"index": 1, "testdotseed": "test_name.seed", "label": 1, 
 "covered_deltas": {"block": 12, "branch": 5}, ...}
```

---

## Next Steps

### For ML Project Planning
1. Review EXPLORATION_SUMMARY.md for opportunities
2. Read CODEBASE_ANALYSIS.md sections 6-7 for technical details
3. Run example code from ML_FEATURE_EXTRACTION_GUIDE.md
4. Collect baseline regression runs with coverage_labeler.py
5. Build initial test prioritization models

### For Infrastructure Enhancement
1. Consider per-test coverage extraction
2. Add execution statistics (runtime, memory)
3. Standardize trace formats for sequence models

### For Data Collection
1. Archive regression runs with metadata
2. Correlate with git commits
3. Build multi-run datasets for temporal analysis

---

## Additional Resources

### Original Source Code
- `/home/a2zaustin/ibex/dv/uvm/core_ibex/scripts/` - All analysis code
- `/home/a2zaustin/ibex/dv/uvm/core_ibex/riscv_dv_extension/testlist.yaml` - Test definitions
- `/home/a2zaustin/ibex/python-requirements.txt` - Dependencies

### Related Documentation
- `/home/a2zaustin/ibex/README.md` - IBEX project overview
- `/home/a2zaustin/ibex/dv/uvm/core_ibex/README.md` - DV-specific info
- IBEX readthedocs: https://ibex-core.readthedocs.io/

---

## Document Statistics

| Document | Size | Lines | Purpose |
|----------|------|-------|---------|
| EXPLORATION_SUMMARY.md | 8.4 KB | 242 | Executive summary |
| CODEBASE_ANALYSIS.md | 19 KB | 520 | Detailed reference |
| ML_FEATURE_EXTRACTION_GUIDE.md | 11 KB | 360 | Practical guide |
| **Total** | **38.4 KB** | **1,122** | Complete documentation |

---

## How This Documentation Was Created

1. **File Discovery**: Used glob patterns to identify all Python scripts, Tcl files, makefiles
2. **Code Analysis**: Read and analyzed key files:
   - metadata.py (metadata management)
   - coverage_labeler.py (ML feature extraction)
   - report_lib/util.py (coverage parsing)
   - Test-related classes and utilities
3. **Infrastructure Mapping**: Traced execution flow from Makefile through all stages
4. **Data Format Analysis**: Identified input/output formats at each stage
5. **ML Opportunity Assessment**: Evaluated suitability for machine learning

**Tools Used**:
- Bash for file discovery and system information
- Read tool for source code analysis
- Grep for pattern matching and code search
- Manual analysis for understanding system design

**Coverage**: Comprehensive exploration of regression infrastructure with focus on ML feature extraction capabilities.

---

## Questions Answered

### Original Questions
1. ✓ What type of regressions are being run? → UVM-based RTL simulation with random instruction generation
2. ✓ What output/data is generated? → Per-test logs, traces, coverage databases; regression-level reports
3. ✓ What coverage/metrics are collected? → 7 code coverage types + functional coverage
4. ✓ Any existing data processing or analysis scripts? → coverage_labeler.py, report_lib, metadata pipeline
5. ✓ Overall structure of test/regression infrastructure? → Makefile-driven, parallelizable, multi-format output

### ML-Specific Questions
- Is the data ML-ready? **Yes** - coverage_labels.jsonl provides labels
- What are the opportunities? **5 high-value opportunities identified**
- How to access the data? **Detailed Python API documented**
- What features are available? **Test params, coverage metrics, pass/fail status, traces**

---

## Conclusion

The IBEX codebase contains a **mature, production-ready verification infrastructure** that is **well-suited for ML applications**. The existing coverage_labeler.py script already produces JSONL feature vectors, and the infrastructure provides granular, multi-faceted data for building sophisticated models.

**Immediate Next Steps**: Use ML_FEATURE_EXTRACTION_GUIDE.md to start extracting features and building baseline models for test prioritization and failure prediction.

---

*Documentation created: 2025-10-29*  
*IBEX Codebase: /home/a2zaustin/ibex/*  
*Exploration Tool: Claude Code (File Search & Analysis)*
