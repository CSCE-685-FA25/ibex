#!/bin/bash
#SBATCH --job-name=regression         # Job name
#SBATCH --mail-type=END,FAIL          # Mail Events (NONE,BEGIN,FAIL,END,ALL)
#SBATCH --mail-user=ashwin_k@tamu.edu # Replace with your email address
#SBATCH --ntasks=8                    # Run on 8 Cores
#SBATCH --time=72:00:00               # Time limit hh:mm:ss
#SBATCH --output=regression%j.log     # Standard output and error log
#SBATCH --qos=olympus-cpu-research        # Do not change
#SBATCH --partition=cpu-research      # Do not change

source env_setup.bash

make -C ibex/dv/uvm/core_ibex/ run SIMULATION=xlm TEST=all WAVES=0 COV=1
