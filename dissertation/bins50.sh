#!/bin/bash
# B50: the published protocol with 50 splitter bins per interval (the chapter's text) instead of 20.
#   dissertation/bins50.sh trace    the splitter (13 default intervals, -bins 50, bytes touched; miniVite with
#                                   -exclude_alloc 1) on every input of the 28 workloads -> $O/traces/<workload>/.
#                                   miniVite one input at a time and alone (its times feed the cumulative cost),
#                                   then PolyBench 12 at a time.
#   dissertation/bins50.sh sample   M3's samplers on the runs of $O/m3_list.txt (score.py b50build): bootstrap
#                                   s=100, r=10, i=k/10 as M3; disjoint with 50 bins, i=k/50 -> $O/m3/<s>/<seed>/...
#   dissertation/bins50.sh time     the splitter at 20 and 50 bins on miniVite 8,192, 5 interleaved rounds, alone
#                                   -> $O/timing.csv (time and peak memory)
# Binaries are built first, one at a time, and copied to $O/bin (the PolyBench Makefile relinks on every build).
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PIN_ROOT=/home/nmust004/memory_estimator/pin/pin-3.30-98830-g1d7b601b3-gcc-linux OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive
O=$HOME/memprint-data/2026-11-11-bins50; export O; TOOL=$HOME/memprint-data/2026-11-05-diss-ch2/memprint_trace.so; export TOOL
mkdir -p $O/bin $O/traces

build() {   # build <workload> <config>: prints "<workload> <config> <binary> <args...>"
    local w=$1 c=$2 wl=polybench; [[ $w == miniVite ]] && wl=minivite
    source $ROOT/scripts/lib.sh; load_workload $wl
    (cd $SRC && wl_build $w $c) > /dev/null 2>&1 || { echo "build failed: $w $c" >&2; return 1; }
    local b=$O/bin/$w-$c; cp "$(cd $SRC && wl_binary $w $c)" $b
    echo "$w $c $b $(wl_args $w $c)" | sed 's/[[:space:]]*$//'; }

configs() { [[ $1 == miniVite ]] && echo "1024 2048 4096 8192 16384" || echo "MINI MINI2 MINI3 SMALL SMALL2 SMALL3 MEDIUM"; }

one() {   # one <job> <workload> <config> <binary> [seed k] <args...>
    local job=$1 w=$2 c=$3 b=$4; shift 4; local ex=() seed= kk=; [[ $w == miniVite ]] && ex=(-exclude_alloc 1)
    case $job in
        split) d=$O/traces/$w; k=(-mode splitter -bins 50 -footprint bytes "${ex[@]}")
               ls $d/Buffered_$w-${c}_1_*.csv > /dev/null 2>&1 && return ;;
        boot|disj) seed=$1 kk=$2; shift 2
               d=$O/m3/$job/$seed/$w/$c/$kk; [[ -d $d ]] && return
               if [[ $job == boot ]]; then k=(-mode sampler -i $((kk / 10)) -s 100 -r 10)
               else k=(-mode sampler -i $((kk / 50)) -s 50 -disjoint 1); fi
               k+=(-footprint bytes -seed $seed "${ex[@]}") ;;
    esac
    mkdir -p $d
    /usr/bin/time -f "%e %M" -o $d/time.$job.$c.${seed}.${kk} $PIN_ROOT/pin -t $TOOL "${k[@]}" -outdir $d -name $w-$c -- $b "$@" \
        > /dev/null 2> $d/err.$job.$c
    local rc=$?
    # Pin exits 0 on a tool usage error: count the files written instead
    local n=$(ls $d | grep -c "^Buffered_$w-${c}_.*_SubSample_\|^Sampled\|_bin_" )
    echo "$job $w $c ${seed:--} ${kk:--} $rc $n $(cat $d/time.$job.$c.${seed}.${kk})" >> $O/done.txt; }
export -f one

case $1 in
trace)
    jobs=$O/jobs_trace.txt; : > $jobs
    for w in miniVite $(grep -v miniVite $ROOT/analysis/workloads.txt); do
        for c in $(configs $w); do l=$(build $w $c) && echo "split $l" >> $jobs; done
    done
    grep "^split miniVite " $jobs | while read -r line; do bash -c 'one "$0" "$@"' $line; done   # alone, one at a time
    grep -v "^split miniVite " $jobs | xargs -P 12 -L 1 bash -c 'one "$0" "$@"' ;;
sample)
    jobs=$O/jobs_sample.txt; : > $jobs
    while read -r w split c k; do
        l=$(build $w $c) || continue; read -r w_ c_ b args <<< "$l"
        for seed in 1 2 3; do for s in boot disj; do echo "$s $w $c $b $seed $k $args" | sed 's/[[:space:]]*$//' >> $jobs; done; done
    done < $O/m3_list.txt
    xargs -P 12 -L 1 bash -c 'one "$0" "$@"' < $jobs ;;
time)
    l=$(build miniVite 8192); read -r w c b args <<< "$l"; f=$O/timing.csv; mkdir -p $O/time
    [[ -f $f ]] || echo "round,bins,seconds,max_kb" > $f
    for round in 1 2 3 4 5; do for bins in 20 50; do
        /usr/bin/time -f "%e %M" -o $O/time/t $PIN_ROOT/pin -t $TOOL -mode splitter -bins $bins -footprint bytes \
            -exclude_alloc 1 -outdir $O/time -name x -- $b $args > /dev/null 2>&1
        echo "$round,$bins,$(tr ' ' , < $O/time/t)" >> $f; rm -f $O/time/*.csv
    done; done ;;
samptime)   # the samplers' cost: miniVite 8,192 at matched bin intervals as M7b (default knobs), then the three
            # PolyBench kernels of M9 at 1 in 1,000 next to the truth run as M9 (-footprint bytes); 5 interleaved rounds
    mkdir -p $O/time; f=$O/samptime.csv; [[ -f $f ]] || echo "round,workload,run,k,seconds" > $f
    t() { local wl=$1 n=$2 kk=$3; shift 3; /usr/bin/time -f "%e" -o $O/time/t "$@" > /dev/null 2>&1
          echo "$round,$wl,$n,$kk,$(cat $O/time/t)" >> $f; rm -f $O/time/*.csv; }
    l=$(build miniVite 8192); read -r w c b args <<< "$l"
    for c_ in gemm 2mm jacobi-2d; do build $c_ MEDIUM > /dev/null; done
    p=($PIN_ROOT/pin -t $TOOL)
    for round in 1 2 3 4 5; do
        t miniVite native 0 $b $args
        for kk in 500 1000 2500; do
            t miniVite boot $kk "${p[@]}" -mode sampler -i $((kk / 10)) -s 100 -r 10 -outdir $O/time -- $b $args
            t miniVite disj20 $kk "${p[@]}" -mode sampler -i $((kk / 20)) -s 20 -disjoint 1 -outdir $O/time -- $b $args
            t miniVite disj50 $kk "${p[@]}" -mode sampler -i $((kk / 50)) -s 50 -disjoint 1 -outdir $O/time -- $b $args
        done
        for wl in gemm 2mm jacobi-2d; do bb=$O/bin/$wl-MEDIUM; fb=(-footprint bytes)
            t $wl truth 1 "${p[@]}" -mode splitter -intervals 1 -bins 1 "${fb[@]}" -outdir $O/time -- $bb
            t $wl boot 1000 "${p[@]}" -mode sampler -i 100 -s 100 -r 10 "${fb[@]}" -outdir $O/time -- $bb
            t $wl disj20 1000 "${p[@]}" -mode sampler -i 50 -s 20 -disjoint 1 "${fb[@]}" -outdir $O/time -- $bb
            t $wl disj50 1000 "${p[@]}" -mode sampler -i 20 -s 50 -disjoint 1 "${fb[@]}" -outdir $O/time -- $bb
        done
    done ;;
esac
echo "all-done $1" >> $O/done.txt
