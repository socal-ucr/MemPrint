#!/bin/bash
# Sampler runs at the intervals that training-only selection (memprint build --select-on train) newly picks and that
# no earlier sample step ran (PROTOCOL.md, "Training-only selection"). Input: CSV "set,sampler,workload,config,split,k"
# (the missing list); every run at seeds 1 to 3, with its set's knobs, into its set's m3 tree, and a line in its set's
# done file in that file's format, so score.py's parts pick it up:
#   m3   (20 bins, bytes)  $OUT/m3/<s>/<seed>/<w>/<c>/<k>         "<dir> <rc>"                    disjoint s=20 (25)
#   m11  (20 bins, pages)  $OUT/m11/m3/...                         "<s> <w> <c> <seed> <k> <rc>"   disjoint s=20 (25)
#   b50  (50 bins)         $B50/m3/...                             "<s> <w> <c> <seed> <k> <rc> <n> <time>"  s=50
#   p50, c50 (50 bins)     $MORE/<set>/m3/...   in $MORE/done.txt  "<set> <s> <w> <c> <seed> <k> <rc> <n> <time>"
#   cp50 (50), c20 (20), m5 (50)  $CM/<set>/m3 ($MORE/m5/m3) in $CM/done.txt, the same format
# Bootstrap s=100 r=10 i=k/10 everywhere. Sampler bins are bytes under either truth.
#   dissertation/samplefill.sh <missing.csv>     (runs under memsched.py: 48 GB, 12 jobs)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PIN_ROOT=/home/nmust004/memory_estimator/pin/pin-3.30-98830-g1d7b601b3-gcc-linux OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive
export OUT=$HOME/memprint-data/2026-11-05-diss-ch2 B50=$HOME/memprint-data/2026-11-11-bins50
export MORE=$HOME/memprint-data/2026-11-12-b50more CM=$HOME/memprint-data/2026-11-13-corpusmore
export TOOL=$OUT/memprint_trace.so

corpus() {   # as corpusmore.sh: "<workload> <config> <cwd> <command...>"
    local g=/home/nmust004/memory_estimator/workloads/gapbs
    for k in bfs cc pr sssp; do n=8; [[ $k == pr || $k == sssp ]] && n=4
        for s in 12 13 14 15 16 17; do echo "gap-$k $s $g ./$k -g $s -n $n"; done; done
    for s in 16 20 24 28 32 40 48; do echo "hpccg $s - $ROOT/workloads/src/hpccg/test_HPCCG $s $s $s"; done
    for s in 10 14 18 22 26 30; do echo "lulesh $s - $ROOT/workloads/src/lulesh/lulesh2.0 -q -s $s -i 20"; done
    for s in 16 20 24 28 32 36 40; do echo "minife $s - /home/nmust004/memory_estimator/workloads/MiniFE/source/miniFE.x -nx $s -ny $s -nz $s"; done; }

fill() {   # fill <set> <sampler> <workload> <config> <seed> <k>
    local set=$1 s=$2 w=$3 c=$4 seed=$5 k=$6 bins=50 d done cwd=- cmd ex=()
    case $set in
        m3)   d=$OUT/m3; bins=20 ;;
        m11)  d=$OUT/m11/m3; bins=20 ;;
        b50)  d=$B50/m3 ;;
        p50|c50) d=$MORE/$set/m3 ;;
        cp50) d=$CM/cp50/m3 ;;
        c20)  d=$CM/c20/m3; bins=20 ;;
        m5)   d=$MORE/m5/m3 ;;
    esac
    d=$d/$s/$seed/$w/$c/$k; [[ -d $d ]] && return
    if [[ $set == c50 || $set == cp50 || $set == c20 ]]; then
        read -r _ _ cwd cmd <<< "$(corpus | grep "^$w $c ")"
    elif [[ $set == m5 ]]; then cmd=$MORE/bin/$w-$c
    else cmd=$B50/bin/$w-$c; fi
    [[ $w == miniVite ]] && { ex=(-exclude_alloc 1); cmd="$cmd 1 100 -n $c"; }
    local knobs
    if [[ $s == boot ]]; then knobs=(-i $((k / 10)) -s 100 -r 10)
    else local b=$bins; ((bins == 20 && k % 20)) && b=25; knobs=(-i $((k / b)) -s $b -disjoint 1); fi
    mkdir -p $d; local run=$d; [[ $cwd == - ]] || run=$cwd
    (cd $run && /usr/bin/time -f "%e %M" -o $d/time.fill $PIN_ROOT/pin -t $TOOL -mode sampler "${knobs[@]}" -footprint bytes \
        -seed $seed "${ex[@]}" -outdir $d -name $w-$c -- $cmd > /dev/null 2> $d/err.fill)
    local rc=$? n=$(ls $d | grep -c "_bin_") t=$(cat $d/time.fill)
    case $set in
        m3)   echo "$d $rc" >> $OUT/m3/done.txt ;;
        m11)  echo "$s $w $c $seed $k $rc" >> $OUT/m11/done.txt ;;
        b50)  echo "$s $w $c $seed $k $rc $n $t" >> $B50/done.txt ;;
        p50|c50) echo "$set $s $w $c $seed $k $rc $n $t" >> $MORE/done.txt ;;
        *)    echo "$set $s $w $c $seed $k $rc $n $t" >> $CM/done.txt ;;
    esac
    echo "$set $s $w $c $seed $k $rc $n $t" >> $OUT/samplefill_done.txt; }
export -f fill corpus; export ROOT

jobs=$OUT/samplefill_jobs.txt; : > $jobs
tail -n +2 "$1" | while IFS=, read -r set s w c split k; do
    gb=1; [[ $set == c50 || $set == cp50 || $set == c20 ]] && gb=12
    for seed in 1 2 3; do echo "$gb fill $set $s $w $c $seed $k" >> $jobs; done
done
python3 $ROOT/dissertation/memsched.py 48 12 < $jobs
echo all-done >> $OUT/samplefill_done.txt
