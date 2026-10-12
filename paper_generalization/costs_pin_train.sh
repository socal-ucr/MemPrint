#!/bin/bash
# Full traces of the 27 PolyBench kernels at their five other configs (the own model's training traces and the
# one-time traces behind borrowing), one run at a time; with costs_pin.sh's SMALL2 and MEDIUM, all seven configs.
#   PIN_ROOT=<pin kit> paper_generalization/costs_pin_train.sh
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
COSTS=$ROOT/data/costs
SCRATCH=${SCRATCH:-$COSTS/traces}
mkdir -p "$COSTS/results_train" "$SCRATCH"
export RESULTS_DIR=$COSTS/results_train
for bench in $(ls "$ROOT/data/polybench-bytes/data/static" | sed 's/-[A-Z0-9]*\.npz//' | sort -u); do
    echo "$(date +%F_%T) $(cut -d' ' -f1-3 /proc/loadavg) polybench $bench [MINI MINI2 MINI3 SMALL SMALL3] splitter" >> "$COSTS/load.txt"
    TRACE_DIR=$SCRATCH "$ROOT/scripts/run.sh" polybench --bench "$bench" --configs "MINI MINI2 MINI3 SMALL SMALL3" \
        --mode splitter --runs 1 --footprint bytes
    rm -rf "${SCRATCH:?}/$bench"
done
