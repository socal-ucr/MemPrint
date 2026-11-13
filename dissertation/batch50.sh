#!/bin/bash
# The 50-bin follow-ups (PROTOCOL.md, "B50 follow-ups"), into $O = ~/memprint-data/2026-11-12-b50more:
#   trace     12 Pin runs at a time:
#             m5    durbin and jacobi-1d, 7 inputs each, bytes, 50 bins          -> $O/m5/traces/<w>/
#             p50   M11 at 50 bins: every PolyBench and miniVite input, -footprint pages, 50 bins -> $O/p50/traces/<w>/
#             c50   corpus programs with an input size, at 6 or 7 inputs each, bytes, 50 bins -> $O/c50/traces/<w>/
#   sample    M3's samplers (bootstrap s=100 r=10; disjoint 50 bins) at the held-out inputs of the run lists
#             $O/{p50,c50}/m3_list.txt (score.py b50more_build) -> $O/<set>/m3/<s>/<seed>/<w>/<c>/<k>/
#   cost      alone: miniVite 8,192 and gemm, 2mm, jacobi-2d MEDIUM, 5 interleaved rounds of native, instrumentation
#             only, the truth run, bootstrap, disjoint 20 and disjoint 50 at 1 in 1,000 -> $O/cost.csv
#   splittime alone, one at a time: the 50-bin splitter (bytes) on every PolyBench and miniVite input -> $O/splittime.csv
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PIN_ROOT=/home/nmust004/memory_estimator/pin/pin-3.30-98830-g1d7b601b3-gcc-linux OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive
O=$HOME/memprint-data/2026-11-12-b50more; export O; TOOL=$HOME/memprint-data/2026-11-05-diss-ch2/memprint_trace.so; export TOOL
B50=$HOME/memprint-data/2026-11-11-bins50
mkdir -p $O/bin

build() {   # build <workload> <config>: prints "<workload> <config> / <binary> <args...>" (PolyBench, miniVite)
    local w=$1 c=$2 wl=polybench; [[ $w == miniVite ]] && wl=minivite
    source $ROOT/scripts/lib.sh; load_workload $wl
    (cd $SRC && wl_build $w $c) > /dev/null 2>&1 || { echo "build failed: $w $c" >&2; return 1; }
    local b=$O/bin/$w-$c; cp "$(cd $SRC && wl_binary $w $c)" $b
    echo "$w $c / $b $(wl_args $w $c)" | sed 's/[[:space:]]*$//'; }
pbconfigs() { [[ $1 == miniVite ]] && echo "1024 2048 4096 8192 16384" || echo "MINI MINI2 MINI3 SMALL SMALL2 SMALL3 MEDIUM"; }

corpus() {   # the corpus programs with an input size: "<workload> <config> <cwd> <command...>"
    local g=/home/nmust004/memory_estimator/workloads/gapbs
    for k in bfs cc pr sssp; do n=8; [[ $k == pr || $k == sssp ]] && n=4
        for s in 12 13 14 15 16 17; do echo "gap-$k $s $g ./$k -g $s -n $n"; done; done
    for s in 16 20 24 28 32 40 48; do echo "hpccg $s - $ROOT/workloads/src/hpccg/test_HPCCG $s $s $s"; done
    for s in 10 14 18 22 26 30; do echo "lulesh $s - $ROOT/workloads/src/lulesh/lulesh2.0 -q -s $s -i 20"; done
    for s in 16 20 24 28 32 36 40; do echo "minife $s - /home/nmust004/memory_estimator/workloads/MiniFE/source/miniFE.x -nx $s -ny $s -nz $s"; done; }

one() {   # one <set> <job> <workload> <config> <cwd> [seed k] <command...>
    local set=$1 job=$2 w=$3 c=$4 cwd=$5; shift 5; local ex=() seed= kk= fp=bytes
    [[ $w == miniVite ]] && ex=(-exclude_alloc 1)
    [[ $set == p50 ]] && fp=pages
    case $job in
        split) d=$O/$set/traces/$w; k=(-mode splitter -bins 50 -footprint $fp "${ex[@]}")
               ls $d/Buffered_$w-${c}_1_*.csv > /dev/null 2>&1 && return ;;
        boot|disj) seed=$1 kk=$2; shift 2
               d=$O/$set/m3/$job/$seed/$w/$c/$kk; [[ -d $d ]] && return
               if [[ $job == boot ]]; then k=(-mode sampler -i $((kk / 10)) -s 100 -r 10)
               else k=(-mode sampler -i $((kk / 50)) -s 50 -disjoint 1); fi
               k+=(-footprint bytes -seed $seed "${ex[@]}") ;;   # sampler bins are bytes under either truth
    esac
    mkdir -p $d; local run=$O/$set/cwd/$w-$c-$job-$seed-$kk; mkdir -p $run; [[ $cwd == - || $cwd == / ]] || run=$cwd
    (cd $run && /usr/bin/time -f "%e %M" -o $d/time.$job.$c.$seed.$kk $PIN_ROOT/pin -t $TOOL "${k[@]}" -outdir $d -name $w-$c \
        -- "$@" > /dev/null 2> $d/err.$job.$c)
    local rc=$? n=$(ls $d | grep -c "^\(Buffered\|Sampled\)_$w-${c}_.*_bin_")
    echo "$set $job $w $c ${seed:--} ${kk:--} $rc $n $(cat $d/time.$job.$c.$seed.$kk)" >> $O/done.txt; }
export -f one

case $1 in
trace)
    jobs=$O/jobs_trace.txt; : > $jobs
    for w in durbin jacobi-1d; do for c in $(pbconfigs $w); do l=$(build $w $c) && echo "m5 split $l" >> $jobs; done; done
    for w in miniVite $(grep -v miniVite $ROOT/analysis/workloads.txt); do
        for c in $(pbconfigs $w); do l=$(build $w $c) && echo "p50 split $l" >> $jobs; done; done
    corpus | sed 's/^/c50 split /' >> $jobs
    # longest first: corpus and miniVite, then the rest
    { grep -E "^c50|^p50 split miniVite" $jobs; grep -vE "^c50|^p50 split miniVite" $jobs; } | xargs -P 12 -L 1 bash -c 'one "$0" "$@"' ;;
sample)
    jobs=$O/jobs_sample.txt; : > $jobs
    for set in p50 c50; do
        while read -r w split c k; do
            if [[ $set == p50 ]]; then l=$(build $w $c) || continue; else l=$(corpus | grep "^$w $c "); fi
            read -r w_ c_ cwd cmd <<< "$l"
            for seed in 1 2 3; do for s in boot disj; do echo "$set $s $w $c $cwd $seed $k $cmd" >> $jobs; done; done
        done < $O/$set/m3_list.txt
    done
    xargs -P 12 -L 1 bash -c 'one "$0" "$@"' < $jobs ;;
cost)
    f=$O/cost.csv; [[ -f $f ]] || echo "round,workload,run,seconds" > $f; mkdir -p $O/time
    t() { local wl=$1 n=$2; shift 2; /usr/bin/time -f "%e" -o $O/time/t "$@" > /dev/null 2>&1
          echo "$round,$wl,$n,$(cat $O/time/t)" >> $f; rm -f $O/time/*.csv; }
    read -r w c _ mv args <<< "$(build miniVite 8192)"
    for k in gemm 2mm jacobi-2d; do build $k MEDIUM > /dev/null; done
    p=($PIN_ROOT/pin -t $TOOL)
    for round in 1 2 3 4 5; do
        for wl in miniVite gemm 2mm jacobi-2d; do
            if [[ $wl == miniVite ]]; then b=$mv; a=($args); ex=(-footprint bytes -exclude_alloc 1); else b=$O/bin/$wl-MEDIUM; a=(); ex=(-footprint bytes); fi
            t $wl native $b "${a[@]}"
            t $wl instr "${p[@]}" -mode sampler -i 4294967295 -s 1 -r 1 -outdir $O/time -- $b "${a[@]}"
            t $wl truth "${p[@]}" -mode splitter -intervals 1 -bins 1 "${ex[@]}" -outdir $O/time -- $b "${a[@]}"
            t $wl boot-1000 "${p[@]}" -mode sampler -i 100 -s 100 -r 10 "${ex[@]}" -outdir $O/time -- $b "${a[@]}"
            t $wl disj20-1000 "${p[@]}" -mode sampler -i 50 -s 20 -disjoint 1 "${ex[@]}" -outdir $O/time -- $b "${a[@]}"
            t $wl disj50-1000 "${p[@]}" -mode sampler -i 20 -s 50 -disjoint 1 "${ex[@]}" -outdir $O/time -- $b "${a[@]}"
        done
    done ;;
splittime)
    f=$O/splittime.csv; [[ -f $f ]] || echo "workload,config,seconds,max_kb" > $f; mkdir -p $O/stime
    for w in miniVite $(grep -v miniVite $ROOT/analysis/workloads.txt); do for c in $(pbconfigs $w); do
        grep -q "^$w,$c," $f && continue
        read -r w_ c_ _ b args <<< "$(build $w $c)"; ex=(); [[ $w == miniVite ]] && ex=(-exclude_alloc 1)
        /usr/bin/time -f "%e %M" -o $O/stime/t $PIN_ROOT/pin -t $TOOL -mode splitter -bins 50 -footprint bytes "${ex[@]}" \
            -outdir $O/stime -name x -- $b $args > /dev/null 2>&1
        echo "$w,$c,$(tr ' ' , < $O/stime/t)" >> $f; rm -f $O/stime/*.csv
    done; done ;;
esac
echo "all-done $1" >> $O/done.txt
