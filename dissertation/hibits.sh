#!/bin/bash
# Every sampler run of the 37-workload evaluation set again with the keep test on the LCG's high bits
# (memprint_trace_hibits.so, pintool/memprint_trace.cpp after 3bd86c2), into a mirror of each set's m3 tree:
#   $N/<set>/m3/<sampler>/<seed>/<workload>/<config>/<k>/      sets b50 (B50's 28), c50 (the 7 corpus programs), m5
# Same knobs as the original runs: bootstrap s=100 r=10 i=k/10; disjoint 50 bins i=k/50; bins in bytes; miniVite with
# -exclude_alloc 1. Each job's memory is the original run's peak (its time file) x 1.3, run under memsched.py (48 GB).
#   dissertation/hibits.sh [workload ...]     (default every workload; a run whose directory exists is skipped)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PIN_ROOT=/home/nmust004/memory_estimator/pin/pin-3.30-98830-g1d7b601b3-gcc-linux OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive
export N=$HOME/memprint-data/2026-11-23-hibits TOOL=$HOME/memprint-data/2026-11-23-hibits/memprint_trace_hibits.so
export B50=$HOME/memprint-data/2026-11-11-bins50 MORE=$HOME/memprint-data/2026-11-12-b50more

corpus() {   # as corpusmore.sh: "<workload> <config> <cwd> <command...>"
    local g=/home/nmust004/memory_estimator/workloads/gapbs
    for k in bfs cc pr sssp; do n=8; [[ $k == pr || $k == sssp ]] && n=4
        for s in 12 13 14 15 16 17; do echo "gap-$k $s $g ./$k -g $s -n $n"; done; done
    for s in 16 20 24 28 32 40 48; do echo "hpccg $s - $ROOT/workloads/src/hpccg/test_HPCCG $s $s $s"; done
    for s in 10 14 18 22 26 30; do echo "lulesh $s - $ROOT/workloads/src/lulesh/lulesh2.0 -q -s $s -i 20"; done
    for s in 16 20 24 28 32 36 40; do echo "minife $s - /home/nmust004/memory_estimator/workloads/MiniFE/source/miniFE.x -nx $s -ny $s -nz $s"; done; }

one() {   # one <set> <sampler> <seed> <workload> <config> <k>
    local set=$1 s=$2 seed=$3 w=$4 c=$5 k=$6 cwd=- cmd ex=() knobs
    local d=$N/$set/m3/$s/$seed/$w/$c/$k
    [[ -d $d ]] && return
    case $set in
        c50) read -r _ _ cwd cmd <<< "$(corpus | grep "^$w $c ")" ;;
        m5)  cmd=$MORE/bin/$w-$c ;;
        *)   cmd=$B50/bin/$w-$c ;;
    esac
    [[ $w == miniVite ]] && { ex=(-exclude_alloc 1); cmd="$cmd 1 100 -n $c"; }
    if [[ $s == boot ]]; then knobs=(-i $((k / 10)) -s 100 -r 10); else knobs=(-i $((k / 50)) -s 50 -disjoint 1); fi
    mkdir -p $d; local run=$d; [[ $cwd == - ]] || run=$cwd
    (cd $run && /usr/bin/time -f "%e %M" -o $d/time $PIN_ROOT/pin -t $TOOL -mode sampler "${knobs[@]}" -footprint bytes \
        -seed $seed "${ex[@]}" -outdir $d -name $w-$c -- $cmd > /dev/null 2> $d/err)
    local rc=$? n=$(ls $d | grep -c "_bin_")
    echo "$set $s $seed $w $c $k $rc $n $(cat $d/time)" >> $N/done.txt; }
export -f one corpus; export ROOT

jobs=$N/jobs.txt; : > $jobs
for pair in "b50 $B50" "c50 $MORE/c50" "m5 $MORE/m5"; do
    read -r set root <<< "$pair"
    for d in $(ls -d $root/m3/*/*/*/*/*); do
        IFS=/ read -r s seed w c k <<< "${d#$root/m3/}"
        if [[ $# -gt 0 ]]; then [[ " $* " == *" $w "* ]] || continue; fi
        kb=$(cat $d/time* 2>/dev/null | awk '{m = $2 > m ? $2 : m} END {print m + 0}')
        gb=$(awk -v kb=$kb 'BEGIN {g = kb / 1e6 * 1.3; printf "%.1f", g < 1 ? 1 : g}')
        echo "$gb one $set $s $seed $w $c $k" >> $jobs
    done
done
echo "$(wc -l < $jobs) runs"
python3 $ROOT/dissertation/memsched.py 48 24 < $jobs
echo "all-done $*" >> $N/done.txt
