#!/bin/bash
# M11 and R11, 12 Pin runs at a time.
#   dissertation/m11.sh trace    M11: the splitter (default intervals and bins) with -footprint pages on every
#                                PolyBench and miniVite input -> $OUT/m11/traces/<workload>/; R11: the pages in use
#                                over the run (-track_frees 1 -snapshot 100000) of miniVite's five inputs
#                                -> $OUT/r11/<config>/
#   dissertation/m11.sh sample   M11: M3's samplers (boot and disj, seeds 1 to 3), bins in bytes as in M3, on the
#                                runs of $OUT/m11/m3_list.txt (score.py m11plan) -> $OUT/m11/m3/<sampler>/<seed>/...
# Every binary is built first, one at a time (the PolyBench Makefile relinks on every build, so a build during a run
# can replace the binary under it), and copied to $OUT/m11/bin.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PIN_ROOT=/home/nmust004/memory_estimator/pin/pin-3.30-98830-g1d7b601b3-gcc-linux OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive
O=$HOME/memprint-data/2026-11-05-diss-ch2; export O; TOOL=$O/memprint_trace.so; export TOOL
mkdir -p $O/m11/bin

build() {   # build <workload> <config>: prints "<workload> <config> <binary> <args...>"
    local w=$1 c=$2 wl=polybench; [[ $w == miniVite ]] && wl=minivite
    source $ROOT/scripts/lib.sh; load_workload $wl
    (cd $SRC && wl_build $w $c) > /dev/null 2>&1 || { echo "build failed: $w $c" >&2; return 1; }
    local b=$O/m11/bin/$w-$c; cp "$(cd $SRC && wl_binary $w $c)" $b
    echo "$w $c $b $(wl_args $w $c)"; }

configs() { [[ $1 == miniVite ]] && echo "1024 2048 4096 8192 16384" || echo "MINI MINI2 MINI3 SMALL SMALL2 SMALL3 MEDIUM"; }

one() {   # one <job> <workload> <config> <binary> <args...>
    local job=$1 w=$2 c=$3 b=$4; shift 4; local ex=(); [[ $w == miniVite ]] && ex=(-exclude_alloc 1)
    case $job in
        split) d=$O/m11/traces/$w; k=(-mode splitter -footprint pages "${ex[@]}")
               ls $d/Buffered_$w-${c}_1_*.csv > /dev/null 2>&1 && return ;;
        inuse) d=$O/r11/$c; k=(-mode splitter -intervals 1 -bins 1 -footprint pages -track_frees 1 -snapshot 100000)
               ls $d/Buffered_*_timeline.csv > /dev/null 2>&1 && return ;;
        boot|disj) local seed=$1 kk=$2; shift 2   # one boot|disj w c b seed k args...
               d=$O/m11/m3/$job/$seed/$w/$c/$kk; [[ -d $d ]] && return
               if [[ $job == boot ]]; then k=(-mode sampler -i $((kk / 10)) -s 100 -r 10)
               else s=20; ((kk % 20)) && s=25; k=(-mode sampler -i $((kk / s)) -s $s -disjoint 1); fi
               k+=(-footprint bytes -seed $seed "${ex[@]}") ;;   # the bins are in bytes under both truths (the tool counts pages only in the exact footprint)
    esac
    mkdir -p $d
    $PIN_ROOT/pin -t $TOOL "${k[@]}" -outdir $d -name $w-$c -- $b "$@" > /dev/null 2> $d/err.$job.$c
    echo "$job $w $c ${seed:-} ${kk:-} $?" >> $O/m11/done.txt; }
export -f one

case $1 in
trace)
    jobs=$O/m11/jobs_trace.txt; : > $jobs
    for w in miniVite $(grep -v miniVite $ROOT/analysis/workloads.txt); do
        for c in $(configs $w); do
            l=$(build $w $c) || continue
            [[ $w == miniVite ]] && echo "inuse $l" >> $jobs
            echo "split $l" >> $jobs
        done
    done
    sed -i "s/[[:space:]]*$//" $jobs; xargs -P 12 -L 1 bash -c 'one "$0" "$@"' < $jobs ;;
sample)
    jobs=$O/m11/jobs_sample.txt; : > $jobs
    while read -r w split c k; do
        l=$(build $w $c) || continue; read -r w_ c_ b args <<< "$l"
        for seed in 1 2 3; do for s in boot disj; do echo "$s $w $c $b $seed $k $args" >> $jobs; done; done
    done < $O/m11/m3_list.txt
    sed -i "s/[[:space:]]*$//" $jobs; xargs -P 12 -L 1 bash -c 'one "$0" "$@"' < $jobs ;;
esac
echo "all-done $1" >> $O/m11/done.txt
