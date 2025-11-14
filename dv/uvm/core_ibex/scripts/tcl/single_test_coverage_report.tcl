# TCL script for generating detailed coverage reports for a single test
#
# This script is invoked by IMC (Integrated Metrics Center) to generate
# detailed coverage reports for an individual test's coverage database.
#
# Environment variables expected:
#   DUT_TOP          - Top-level DUT module name (e.g., "ibex_core")
#   cov_report_dir   - Directory to write coverage reports

# Get environment variables
set dut_top $::env(DUT_TOP)
set report_dir $::env(cov_report_dir)

# Create report directory if it doesn't exist
file mkdir $report_dir

# Generate summary text report
set report_file [file join $report_dir "cov_report.txt"]
report -summary \
    -detail \
    -out $report_file

# Generate covergroup report
set cg_report_file [file join $report_dir "cov_report_cg.txt"]
report -summary \
    -detail \
    -covergroup \
    -out $cg_report_file

# Generate detailed instance-level report
set inst_report_file [file join $report_dir "cov_report_instances.txt"]
report -detail \
    -inst $dut_top \
    -out $inst_report_file

# Exit IMC
exit
