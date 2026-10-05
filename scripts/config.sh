# Paths used by the MemPrint scripts. Every value can be overridden from the
# environment, e.g.  PIN_ROOT=/opt/pin-3.30 scripts/run.sh polybench ...

MEMPRINT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Intel Pin kit (needed to build and run the Pin tool)
PIN_ROOT="${PIN_ROOT:-}"
MEMPRINT_TOOL="${MEMPRINT_TOOL:-$MEMPRINT_ROOT/pintool/obj-intel64/memprint_trace.so}"

# Where scripts/setup.sh checks out the workload sources
WORKLOAD_SRC_DIR="${WORKLOAD_SRC_DIR:-$MEMPRINT_ROOT/workloads/src}"

# Outputs of scripts/run.sh: traces/<bench>/*.csv and results/<bench>/*.csv
TRACE_DIR="${TRACE_DIR:-$MEMPRINT_ROOT/traces}"
RESULTS_DIR="${RESULTS_DIR:-$MEMPRINT_ROOT/results}"
