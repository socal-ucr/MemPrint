#!/bin/bash
# The remaining sampler runs with the high-bit keep test (hibits2.py plans them): M3 at 20 bins, M11, p50, cp50 and
# c20. Same knobs as the original runs: bootstrap s=100 r=10 i=k/10; disjoint with the set's bin count, i=k/bins;
# bins in bytes; miniVite with -exclude_alloc 1. Run under memsched.py (48 GB, 24 at once).
#   dissertation/hibits2.sh      (a run whose directory exists is skipped)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PIN_ROOT=/home/nmust004/memory_estimator/pin/pin-3.30-98830-g1d7b601b3-gcc-linux OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive
export ROOT N=$HOME/memprint-data/2026-11-23-hibits TOOL=$HOME/memprint-data/2026-11-23-hibits/memprint_trace_hibits.so
export B50=$HOME/memprint-data/2026-11-11-bins50
source <(sed -n '/^corpus() {/,/^ *for s in 16 20 24 28 32 36 40.*; }$/p' $ROOT/dissertation/hibits.sh)

one2() {   # one2 <dir> <sampler> <bins> <seed> <workload> <config> <k> <kind>
    local d=$1 s=$2 b=$3 seed=$4 w=$5 c=$6 k=$7 kind=$8 cwd=- cmd ex=() knobs
    [[ -d $d ]] && return
    if [[ $kind == corpus ]]; then read -r _ _ cwd cmd <<< "$(corpus | grep "^$w $c ")"; else cmd=$B50/bin/$w-$c; fi
    [[ $w == miniVite ]] && { ex=(-exclude_alloc 1); cmd="$cmd 1 100 -n $c"; }
    if [[ $s == boot ]]; then knobs=(-i $((k / 10)) -s 100 -r 10); else knobs=(-i $((k / b)) -s $b -disjoint 1); fi
    mkdir -p $d; local run=$d; [[ $cwd == - ]] || run=$cwd
    (cd $run && /usr/bin/time -f "%e %M" -o $d/time $PIN_ROOT/pin -t $TOOL -mode sampler "${knobs[@]}" -footprint bytes \
        -seed $seed "${ex[@]}" -outdir $d -name $w-$c -- $cmd > /dev/null 2> $d/err)
    local rc=$? n=$(ls $d | grep -c "_bin_")
    echo "$d $rc $n $(cat $d/time)" >> $N/done2.txt; }
export -f one2 corpus

~/environments/memfp/bin/python $ROOT/dissertation/hibits2.py
python3 $ROOT/dissertation/memsched.py 48 24 < $N/jobs2.txt
echo "all-done" >> $N/done2.txt
