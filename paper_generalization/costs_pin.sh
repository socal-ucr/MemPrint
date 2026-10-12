#!/bin/bash
# Native and Pin run times for every workload the paper uses, one run at a time (for the cost tables).
#
#   PIN_ROOT=<pin kit> paper_generalization/costs_pin.sh [group ...]
#
# Groups: npb polybench gap blind csr (default: all, in that order). Times go to
# data/costs/results/<bench>/overhead_<mode>.csv (run.sh's format) and data/costs/csr.csv. Traces are
# deleted after each benchmark, except NPB's splitter traces (the second blind test's ground truth),
# which are kept in data/blind2/traces. The node load is logged before each step (data/costs/load.txt).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
COSTS=$ROOT/data/costs
SCRATCH=${SCRATCH:-$COSTS/traces}
mkdir -p "$COSTS/results" "$SCRATCH"
export RESULTS_DIR=$COSTS/results
groups=("$@"); [[ ${#groups[@]} -gt 0 ]] || groups=(npb polybench gap blind csr)

step() {  # step <workload> <bench> <configs> <mode> [run.sh options]: one run.sh call, traces dropped
    local wl=$1 bench=$2 configs=$3 mode=$4; shift 4
    echo "$(date +%F_%T) $(cut -d' ' -f1-3 /proc/loadavg) $wl $bench [$configs] $mode $*" >> "$COSTS/load.txt"
    TRACE_DIR=$SCRATCH "$ROOT/scripts/run.sh" "$wl" --bench "$bench" --configs "$configs" --mode "$mode" "$@"
    rm -rf "${SCRATCH:?}/$bench"
}

timings() {  # timings <workload> <bench> <configs>: native x3, instrumentation only, sampled run (defaults)
    step "$1" "$2" "$3" native --runs 3
    step "$1" "$2" "$3" instr --runs 1 --pin-runs 1
    step "$1" "$2" "$3" sampler --runs 1 --pin-runs 1
}

for g in "${groups[@]}"; do
    case $g in
    npb)  # ground truth for the second blind test: kept
        for b in "npb_is S W" "npb_cg S W" "npb_mg S W A"; do
            read -r bench cls <<< "$b"
            echo "$(date +%F_%T) $(cut -d' ' -f1-3 /proc/loadavg) npb $bench [$cls] splitter (truth)" >> "$COSTS/load.txt"
            TRACE_DIR=$ROOT/data/blind2/traces "$ROOT/scripts/run.sh" npb --bench "$bench" --configs "$cls" \
                --mode splitter --footprint bytes --runs 1
            timings npb "$bench" "$cls"
        done ;;
    polybench)
        # the 27 kernels of the paper (those with interpreter spectra)
        for bench in $(ls "$ROOT/data/polybench-bytes/data/static" | sed 's/-[A-Z0-9]*\.npz//' | sort -u); do
            timings polybench "$bench" "SMALL2 MEDIUM"
            step polybench "$bench" "SMALL2 MEDIUM" splitter --runs 1 --footprint bytes
        done ;;
    gap)
        for k in pr bfs cc bc tc sssp; do
            for bench in gap_$k gap_${k}_kron; do
                step gapbs "$bench" "14 16 18" native --runs 3
                step gapbs "$bench" "16 18" instr --runs 1 --pin-runs 1
                step gapbs "$bench" "16 18" sampler --runs 1 --pin-runs 1
                step gapbs "$bench" "16" splitter --runs 1 --footprint bytes
            done
        done ;;
    blind)
        for b in "hpccg hpccg 20 28" "lulesh lulesh 15 20"; do
            read -r wl bench c1 c2 <<< "$b"
            timings "$wl" "$bench" "$c1 $c2"
            step "$wl" "$bench" "$c1 $c2" splitter --runs 1 --footprint bytes
        done ;;
    csr)  # the two C programs (tests/static, -O0, SCALE at compile time)
        tool=${MEMPRINT_TOOL:-$ROOT/pintool/obj-intel64/memprint_trace.so}
        [[ -f $COSTS/csr.csv ]] || echo "program,scale,mode,seconds" > "$COSTS/csr.csv"
        for b in "csr_pr 12 16" "csr_bfs 10 14"; do
            read -r prog s1 s2 <<< "$b"
            for s in $s1 $s2; do
                exe=$SCRATCH/$prog-$s
                gcc -O0 -DSCALE=$s "$ROOT/tests/static/$prog.c" -o "$exe" -lm
                echo "$(date +%F_%T) $(cut -d' ' -f1-3 /proc/loadavg) csr $prog [$s]" >> "$COSTS/load.txt"
                for m in native native native instr sampler splitter; do
                    case $m in
                        native) cmd=("$exe") ;;
                        instr) cmd=("$PIN_ROOT/pin" -t "$tool" -mode sampler -i 4294967295 -s 1 -r 1 -outdir "$SCRATCH" -name x -- "$exe") ;;
                        sampler) cmd=("$PIN_ROOT/pin" -t "$tool" -mode sampler -i 1000 -s 100 -r 10 -outdir "$SCRATCH" -name x -- "$exe") ;;
                        splitter) cmd=("$PIN_ROOT/pin" -t "$tool" -mode splitter -footprint bytes -outdir "$SCRATCH" -name x -- "$exe") ;;
                    esac
                    t0=$(date +%s.%N); "${cmd[@]}" > /dev/null 2>&1; t1=$(date +%s.%N)
                    echo "$prog,$s,$m,$(echo "$t1 - $t0" | bc)" >> "$COSTS/csr.csv"
                    rm -f "$SCRATCH"/x*
                done
                rm -f "$exe"
            done
        done ;;
    *) echo "unknown group $g" >&2; exit 1 ;;
    esac
done
