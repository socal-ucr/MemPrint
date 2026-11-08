#!/bin/bash
# Inputs for a CXL latency model of the emulated pool (cxl_model.py). The runs of run_lplace.sh (GAP, CPython, Lua) and
# run_combo.sh (darknet, llama.cpp), again, under perf stat (user-mode counts of the loads served from local and remote
# DRAM, and of all reads that missed the L3), at two pool latencies:
#   hog 0   node 1 idle (about +55 ns over local)
#   hog H   H threads on node 1's CPUs streaming over node 1's memory (hog.c; H=8: about +108 ns, local +1.5 ns)
# Each repetition runs every program and policy at both hog levels (the hog started once per block). Same plans,
# libmpr and settings as the runs being modelled (MPR_NOTHP=1; MPR_MIN=262144 for the run_lplace programs).
# lat.txt records the pointer-chase latency (chase.c) of each node at each hog level before and after the runs.
#   REPS (default 5), H (default 8), ONLY (regex of programs) -> $D/runs/<prog>/<rep>/<hog>/<pol>.{perf,time}, $D/done.txt
HERE=$(cd "$(dirname "$0")" && pwd)
L=$HOME/memprint-data/2026-11-08-lplace
D=${D:-$HOME/memprint-data/2026-11-08-cxl}; REPS=${REPS:-5}; H=${H:-8}; ONLY=${ONLY:-}
LIB=${LIB:-$L/libmpr-post3.so}
NUMACTL=$(echo $HOME/spack/opt/spack/linux-broadwell/numactl-2.0.19-*/bin/numactl)
EV=mem_load_uops_l3_miss_retired.local_dram:u,mem_load_uops_l3_miss_retired.remote_dram:u
EV=$EV,offcore_response.all_reads.llc_miss.local_dram:u,offcore_response.all_reads.llc_miss.remote_dram:u
LIST=$HOME/memprint-data/2026-11-07-refs/prof_large/list.txt
PPOL="local remote lsame_cold25 lxfer_cold25 lxfer_rand25 lxfer_hot25 lxfer_cold50 lxfer_hot50"
CPOL="none remote cold25 cold50 res res_remote res_cold25 res_cold50 res_hot25 res_hot50"
mkdir -p $D; cp $HERE/cxl/{hog,chase}.c $D/
gcc -O2 -pthread -o $D/hog $HERE/cxl/hog.c && gcc -O2 -o $D/chase $HERE/cxl/chase.c || exit 1
export OMP_WAIT_POLICY=passive

lat() { # $1 = hog level: pointer-chase latency, local and pool
    echo "$1 $($NUMACTL --cpunodebind=0 --membind=0 $D/chase | cut -d' ' -f1) $($NUMACTL --cpunodebind=0 --membind=1 $D/chase | cut -d' ' -f1)"
}
one() { # out-dir label node cwd env... -- cmd...
    local o=$1 pol=$2 node=$3 cwd=$4; shift 4
    (cd "$cwd" && /usr/bin/time -o $o/$pol.time -f "%e %M" perf stat -x, -o $o/$pol.perf -e $EV \
        $NUMACTL --cpunodebind=0 --membind=$node env "$@" < /dev/null > /dev/null 2> $o/$pol.err)
    echo "$o $pol $? $(tail -1 $o/$pol.time)" >> $D/done.txt
}
block() { # $1 rep, $2 hog level
    local rep=$1 hog=$2
    while read -r prog input threads cwd cmd; do
        [[ -n $ONLY && ! $prog =~ $ONLY ]] && continue
        export OMP_NUM_THREADS=$threads; o=$D/runs/$prog/$rep/$hog; mkdir -p $o
        for pol in $PPOL; do
            node=0; pre=(LD_PRELOAD=$LIB MPR_MIN=262144 MPR_NOTHP=1)
            case $pol in
            remote) node=1 ;;
            l*_*) t=${pol%%_*}; pre+=(MPR_PLACE=$L/lplans_${t#l}/$prog/${pol#*_}.txt) ;;
            esac
            one $o $pol $node "$cwd" "${pre[@]}" $cmd
        done
    done < $LIST
    for p in "darknet 1 $L/combo $HOME/memprint-data/2026-10-12-resize/runs/corpus/darknet/res-self.txt" \
             "llamacpp 4 $L/combo2 $HOME/memprint-data/2026-10-12-resize/runs/inputs/res-llamacpp-135m.txt"; do
        read -r prog threads c res <<< "$p"
        [[ -n $ONLY && ! $prog =~ $ONLY ]] && continue
        export OMP_NUM_THREADS=$threads; o=$D/runs/$prog/$rep/$hog; mkdir -p $o
        for pol in $CPOL; do
            node=0; r=/dev/null; plan=
            case $pol in
            remote) node=1 ;;
            res) r=$res ;;
            res_remote) r=$res; node=1 ;;
            res_*) r=$res; plan=$c/lplans/$prog/${pol#res_}.txt ;;
            none) ;;
            *) plan=$c/lplans/$prog/$pol.txt ;;
            esac
            one $o $pol $node / WRAP_PRELOAD=$LIB MPR_NOTHP=1 MPR_RESERVATIONS=$r ${plan:+MPR_PLACE=$plan} $c/$prog.sh
        done
    done
}
lat 0 >> $D/lat.txt
for rep in $(seq 1 $REPS); do
    for hog in 0 $H; do
        hp=; [[ $hog -gt 0 ]] && { $NUMACTL --cpunodebind=1 --membind=1 $D/hog $hog & hp=$!; sleep 2; lat $hog >> $D/lat.txt; }
        block $rep $hog
        [[ -n $hp ]] && { kill $hp; wait $hp 2>/dev/null; }
    done
done
lat 0 >> $D/lat.txt
echo all-done
