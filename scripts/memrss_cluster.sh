#!/bin/bash
# memrss_cluster.sh: the MemRSS rows of tab:mpr-cluster (Ch3 ladder: request -> MemRSS curve -> MemRSS curve + libmpr).
#   1. runs: every one of the resize study's 29 workloads under the glibc build ("trim" label, libmpr preloaded
#      with no reservations) and the libmpr build ("selftrim", own profile), libmpr's monitor on and trimming off
#      (MONITOR=1 MPR_TRIM=-1), 3 runs each: <label>-<k>.monitor.csv holds the data estimate and the file RSS
#   2. simulation: policies memrss and mpr+memrss of memprint.sched.resize_cluster, with the reference rows, at
#      every load (grid), at load 0.8 over local memory (capacity, for the TB column), and with native costs
#
#   scripts/memrss_cluster.sh [runs|sim|all]   (default all; runs takes about as long as trim_eval.sh's base+self)
set -eu
STEP=${1:-all}
R=$HOME/memprint-data/2026-10-12-resize
D=${D:-$HOME/memprint-data/2026-11-19-memrss-cluster}
PY=${PY:-$HOME/environments/memfp/bin/python}
mkdir -p "$D"

if [[ $STEP == runs || $STEP == all ]]; then
    # directories only (runs/corpus also holds summary.csv); miniVite: the pass build only
    names=$(cd "$R/runs/corpus" && ls -d */ | tr -d / | grep -vx miniVite; echo minivite; ls "$R/runs/ml")
    for w in $names; do
        [[ -f $D/runs/done.txt ]] && grep -qx "$w" "$D/runs/done.txt" && continue
        OUT=$D/runs MONITOR=1 MPR_TRIM=-1 BUILDS="trim selftrim" "$HOME/MemPrint/scripts/trim_eval.sh" "$w"
    done
    # every workload should have 3 monitored runs per build
    for w in $names; do
        n=$(ls "$D/runs/$w"/{trim,selftrim}-[0-9].monitor.csv 2>/dev/null | wc -l)
        [[ $n == 6 ]] || echo "WARNING: $w has $n of 6 monitor files"
    done
fi

if [[ $STEP == sim || $STEP == all ]]; then
    cd "$HOME/MemPrint/analysis"
    export RESIZE_EST_RUNS=$D/runs
    P="exact request mpr-exact mpr memrss mpr+memrss"
    $PY -m memprint.sched.resize_cluster --out "$D/grid_none.csv" --policies $P
    $PY -m memprint.sched.resize_cluster --out "$D/capacity_none.csv" --policies $P --loads 0.8 \
        --locals 64 96 128 160 192 224 256
    $PY -m memprint.sched.resize_cluster --out "$D/grid_native.csv" --policies $P --overheads native --loads 0.6 0.8 1.2
fi
