#!/bin/bash
# The B50 results that covered PolyBench and miniVite only, on the corpus programs too (PROTOCOL.md, "Corpus
# follow-ups"), into $O = ~/memprint-data/2026-11-13-corpusmore. Traces run under dissertation/memsched.py (48 GB,
# each trace's memory estimated from its 50-bin bytes run in 2026-11-12-b50more).
#   trace     cp50  the corpus programs with -footprint pages, 50 bins        -> $O/cp50/traces/<w>/
#             c20   the corpus programs, bytes, 20 bins                      -> $O/c20/traces/<w>/
#   sample    at the held-out inputs of $O/{cp50,c20}/m3_list.txt and of b50more/m5/m3_list.txt (score.py
#             corpusmore_build): bootstrap s=100 r=10, and disjoint with the set's bin count (50; 20 for c20)
#   splittime alone, one at a time: the 50-bin splitter (bytes) on every corpus input -> $O/splittime.csv
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PIN_ROOT=/home/nmust004/memory_estimator/pin/pin-3.30-98830-g1d7b601b3-gcc-linux OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive
O=$HOME/memprint-data/2026-11-13-corpusmore; export O; TOOL=$HOME/memprint-data/2026-11-05-diss-ch2/memprint_trace.so; export TOOL
M=$HOME/memprint-data/2026-11-12-b50more; export M
mkdir -p $O

corpus() {   # as batch50.sh: "<workload> <config> <cwd> <command...>"
    local g=/home/nmust004/memory_estimator/workloads/gapbs
    for k in bfs cc pr sssp; do n=8; [[ $k == pr || $k == sssp ]] && n=4
        for s in 12 13 14 15 16 17; do echo "gap-$k $s $g ./$k -g $s -n $n"; done; done
    for s in 16 20 24 28 32 40 48; do echo "hpccg $s - $ROOT/workloads/src/hpccg/test_HPCCG $s $s $s"; done
    for s in 10 14 18 22 26 30; do echo "lulesh $s - $ROOT/workloads/src/lulesh/lulesh2.0 -q -s $s -i 20"; done
    for s in 16 20 24 28 32 36 40; do echo "minife $s - /home/nmust004/memory_estimator/workloads/MiniFE/source/miniFE.x -nx $s -ny $s -nz $s"; done; }

one() {   # one <set> <job> <workload> <config> <cwd> [seed k] <command...>
    local set=$1 job=$2 w=$3 c=$4 cwd=$5; shift 5; local seed= kk= fp=bytes bins=50 root=$O/$set
    [[ $set == cp50 ]] && fp=pages
    [[ $set == c20 ]] && bins=20
    [[ $set == m5 ]] && root=$M/m5
    case $job in
        split) d=$root/traces/$w; k=(-mode splitter -bins $bins -footprint $fp)
               ls $d/Buffered_$w-${c}_1_*.csv > /dev/null 2>&1 && return ;;
        boot|disj) seed=$1 kk=$2; shift 2
               d=$root/m3/$job/$seed/$w/$c/$kk; [[ -d $d ]] && return
               if [[ $job == boot ]]; then k=(-mode sampler -i $((kk / 10)) -s 100 -r 10)
               else s=$bins; ((bins == 20 && kk % 20)) && s=25; k=(-mode sampler -i $((kk / s)) -s $s -disjoint 1); fi
               k+=(-footprint bytes -seed $seed) ;;
    esac
    mkdir -p $d; local run=$root/cwd/$w-$c-$job-$seed-$kk; mkdir -p $run; [[ $cwd == - || $cwd == / ]] || run=$cwd
    (cd $run && /usr/bin/time -f "%e %M" -o $d/time.$job.$c.$seed.$kk $PIN_ROOT/pin -t $TOOL "${k[@]}" -outdir $d -name $w-$c \
        -- "$@" > /dev/null 2> $d/err.$job.$c)
    local rc=$? n=$(ls $d | grep -c "^\(Buffered\|Sampled\)_$w-${c}_.*_bin_")
    echo "$set $job $w $c ${seed:--} ${kk:--} $rc $n $(cat $d/time.$job.$c.$seed.$kk)" >> $O/done.txt; }
export -f one

gb() {   # gb <workload> <config> <factor>: the 50-bin bytes trace's peak memory in GB x factor, at least 1
    awk -v w=$1 -v c=$2 -v f=$3 '$1=="c50" && $2=="split" && $3==w && $4==c && $7==0 {g=$10/1e6*f} END{printf "%.1f", (g>1?g:1)}' $M/done.txt; }

case $1 in
trace)
    jobs=$O/jobs_trace.txt; : > $jobs
    corpus | while read -r w c rest; do
        echo "$(gb $w $c 1.05) one cp50 split $w $c $rest" >> $jobs
        echo "$(gb $w $c 0.5) one c20 split $w $c $rest" >> $jobs
    done
    python3 $ROOT/dissertation/memsched.py 48 12 < $jobs ;;
sample)
    jobs=$O/jobs_sample.txt; : > $jobs
    for set in cp50 c20 m5; do
        list=$O/$set/m3_list.txt; [[ $set == m5 ]] && list=$M/m5/m3_list.txt
        while read -r w split c k; do
            if [[ $set == m5 ]]; then l="$w $c / $M/bin/$w-$c"; else l=$(corpus | grep "^$w $c "); fi
            read -r w_ c_ cwd cmd <<< "$l"
            for seed in 1 2 3; do for s in boot disj; do echo "2 one $set $s $w $c $cwd $seed $k $cmd" >> $jobs; done; done
        done < $list
    done
    python3 $ROOT/dissertation/memsched.py 48 12 < $jobs ;;
retry)   # sampler runs the OOM killer stopped (sampler jobs were budgeted at 2 GB; GAP pr at 17 needs 10 to 12): one at a time
    awk '($2=="boot"||$2=="disj") && ($7!=0 || $8==0) {print $1, $2, $3, $4, $5, $6}' $O/done.txt | sort -u | \
    while read -r set job w c seed kk; do
        rm -rf $O/$set/m3/$job/$seed/$w/$c/$kk
        read -r w_ c_ cwd cmd <<< "$(corpus | grep "^$w $c ")"
        one $set $job $w $c $cwd $seed $kk $cmd
    done ;;
splittime)
    f=$O/splittime.csv; [[ -f $f ]] || echo "workload,config,seconds,max_kb" > $f; mkdir -p $O/stime
    corpus | while read -r w c cwd cmd; do
        grep -q "^$w,$c," $f && continue
        run=$O/stime; [[ $cwd == - ]] || run=$cwd
        (cd $run && /usr/bin/time -f "%e %M" -o $O/stime/t $PIN_ROOT/pin -t $TOOL -mode splitter -bins 50 -footprint bytes \
            -outdir $O/stime -name x -- $cmd > /dev/null 2>&1)
        echo "$w,$c,$(tr ' ' , < $O/stime/t)" >> $f; rm -f $O/stime/*.csv
    done ;;
esac
echo "all-done $1" >> $O/done.txt
