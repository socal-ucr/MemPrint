#!/bin/bash
# R9: the pages-touched truth (4 KiB pages x 4096) and the bytes-touched truth of every R1 input (all workloads and
# configs; R1's R and transfer rule use them all), one interval and one bin, 8 at a time, into
# $OUT/r9/all/{pages,bytes}/<workload>/<config>; score with score.py r1p
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PIN_ROOT=/home/nmust004/memory_estimator/pin/pin-3.30-98830-g1d7b601b3-gcc-linux OMP_NUM_THREADS=1 ROOT
O=$HOME/memprint-data/2026-11-05-diss-ch2; export O
one() {   # one <workload> <config>
    local w=$1 c=$2 wl=polybench ex=(); [[ $w == miniVite ]] && wl=minivite ex=(-exclude_alloc 1)
    source $ROOT/scripts/lib.sh; load_workload $wl
    local B; B=$(cd $SRC && wl_binary $w $c); read -r -a A <<< "$(wl_args $w $c)"
    for fp in pages bytes; do d=$O/r9/all/$fp/$w/$c
        ls $d/Buffered_*.csv > /dev/null 2>&1 && continue; mkdir -p $d
        $PIN_ROOT/pin -t $O/memprint_trace.so -mode splitter -intervals 1 -bins 1 -footprint $fp "${ex[@]}" -outdir $d \
            -name $w-$c -- $B "${A[@]}" > /dev/null 2>&1
    done
    echo "$w $c" >> $O/r9/all/done.txt; }
export -f one
source $ROOT/scripts/lib.sh   # build serially first (builds share the source tree)
for w in $(cat $ROOT/analysis/workloads.txt); do
    wl=polybench cfgs="MINI MINI2 MINI3 SMALL SMALL2 SMALL3 MEDIUM"
    [[ $w == miniVite ]] && wl=minivite cfgs="1024 2048 4096 8192 16384"
    load_workload $wl
    for c in $cfgs; do (cd $SRC && wl_build $w $c) > /dev/null 2>&1; echo "$w $c"; done
done | xargs -P 8 -L 1 bash -c 'one $0 $1'
echo all-done >> $O/r9/all/done.txt
