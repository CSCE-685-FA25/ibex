#-------------------------------
#!/bin/bash
source /opt/coe/cadence/XCELIUM/setup.XCELIUM.linux.bash
source ~/.bashrc

source ~/lowRISC/ibex/lowrisc_env/bin/activate
export RISCV_TOOLCHAIN=~/lowRISC/lowrisc-toolchain-gcc-rv32imcb-x86_64-20250710-1/
export RISCV_GCC="$RISCV_TOOLCHAIN/bin/riscv32-unknown-elf-gcc"
export RISCV_OBJCOPY="$RISCV_TOOLCHAIN/bin/riscv32-unknown-elf-objcopy"
export OVPSIM_PATH=~/lowRISC/imperas-riscv-tests/riscv-ovpsim/bin
export SPIKE_PATH=~/lowRISC/spike/bin/
export PKG_CONFIG_PATH=~/lowRISC/spike/lib/pkgconfig/

export MDV_XLM_HOME="/opt/coe/cadence/XCELIUM"
export UVMHOME="/opt/coe/cadence/XCELIUM/tools/methodology/UVM/CDNS-1.1d/sv"
source /opt/coe/cadence/XCELIUM/setup.XCELIUM.linux.bash
source /opt/coe/cadence/VMANAGER/setup.VMANAGER.linux.bash
alias vmanager="/opt/coe/cadence/VMANAGER/bin/vmanager -server 192.168.2.4:8081 &"
alias imc="/opt/coe/cadence/VMANAGER/bin/imc"

PATH=${RISCV_TOOLCHAIN}/bin:${PATH}
echo "Done Sourcing!"
#-------------------------------
