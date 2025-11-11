#!/bin/bash
#SBATCH --job-name=regression          # Job name
#SBATCH --mail-type=ALL                # Mail Events (NONE,BEGIN,FAIL,END,ALL)
#SBATCH --mail-user=a2zaustin@tamu.edu # Replace with your email address
#SBATCH --ntasks=8                     # Run on 8 Cores
#SBATCH --time=72:00:00                # Time limit hh:mm:ss
#SBATCH --output=regression%j.log      # Standard output and error log
#SBATCH --qos=olympus-cpu-research     # Do not change
#SBATCH --partition=cpu-research       # Do not change

source ~/lowRISC/ibex/env_setup.bash

#make -C ../ run SIMULATION=xlm TEST=all WAVES=0 COV=1
make -C ~/lowRISC/ibex/dv/uvm/core_ibex run SIMULATION=xlm TEST=riscv_machine_mode_rand_test ITERATIONS=10 COV=1
