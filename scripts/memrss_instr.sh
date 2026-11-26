#!/bin/bash
# memrss_instr.sh: MemRSS's curve aligned by instructions (Ch3 future work (c)). The MemRSS cluster study's monitored
# runs again with each run's user-space instructions every 10 ms (PERF_INSTR=1: perf stat -I, left out of the RSS),
# then the cluster grid with truth and estimate from these runs (RESIZE_ALIGN_RUNS: a job on run k, MemRSS's
# estimate from run k+1 of the same build, by wall time or by instructions with a tolerance of +-1%).
#   scripts/memrss_instr.sh [runs|sim|all]
set -eu
STEP=${1:-all}
R=$HOME/memprint-data/2026-10-12-resize
D=${D:-$HOME/memprint-data/2026-11-25-memrss-instr}
PY=${PY:-$HOME/environments/memfp/bin/python}
mkdir -p "$D/runs"
if [[ $STEP == runs || $STEP == all ]]; then
    names=$(cd "$R/runs/corpus" && ls -d */ | tr -d / | grep -vx miniVite; echo minivite; ls "$R/runs/ml")
    for w in $names; do
        [[ -f $D/runs/done.txt ]] && grep -qx "$w" "$D/runs/done.txt" && continue
        OUT=$D/runs PERF_INSTR=1 MONITOR=1 MPR_TRIM=-1 BUILDS="trim selftrim" "$HOME/MemPrint/scripts/trim_eval.sh" "$w"
    done
fi
if [[ $STEP == sim || $STEP == all ]]; then
    cd "$HOME/MemPrint/analysis"
    export RESIZE_ALIGN_RUNS=$D/runs
    P="exact request mpr-exact mpr"
    for a in "" "-i1"; do for sh in "" "-w" "-d2"; do P="$P memrss$a$sh mpr+memrss$a$sh"; done; done
    $PY -m memprint.sched.resize_cluster --out "$D/grid_align.csv" --truths rss reclaim --policies $P memrss-peak mpr+memrss-peak
fi
