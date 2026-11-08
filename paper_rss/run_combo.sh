#!/bin/bash
# Reservations and placement together, through libmpr, on the emulated pool (CPUs and default memory on NUMA node 0, the
# "pool" node 1): darknet alexnet and llama.cpp smollm2-135m (the resize study's programs whose reservations save
# memory; wrappers in $D: darknet.sh, llamacpp.sh, which preload $WRAP_PRELOAD into the program only).
#   1. profile: mprshim chunks run (as run_refs.sh "chunks": 2 MiB chunks, 0.1 s windows, THP off), one run
#   2. plans:   place_lplans.py -> $D/lplans/<prog>/{cold,hot,rand}{25,50}.txt
#   3. runs:    scripts/resize_eval.sh (config linked: the wrapper preloads libmpr; peak RSS, peak commit, output hash)
#               under numactl, REPS repetitions interleaved, every libmpr run with MPR_NOTHP=1 and MPR_NUMA:
#      glibc      no libmpr, --membind=0               none       libmpr, no reservations, no plan
#      res        the resize study's reservations       remote     libmpr, no reservations, --membind=1
#      cold<p>    plan only                             res_cold<p>, res_hot<p>   reservations and the plan
#      res_remote reservations, --membind=1
# The plan is profiled with glibc's committed memory in view: reservations decide what is backed, the plan where the
# backed memory lives. Same input for profile and runs.
# STEPS (default "prof plans runs"), REPS (default 5). -> $D/runs/<prog>/{summary.txt,<label>.*}, $D/done.txt
HERE=$(cd "$(dirname "$0")" && pwd)
R=$HOME/memprint-data/2026-10-12-resize; S=$HOME/memprint-data/2026-11-07-refs
D=${D:-$HOME/memprint-data/2026-11-08-lplace/combo}; REPS=${REPS:-5}; STEPS=" ${STEPS:-prof plans runs} "
LIB=${LIB:-$HOME/memprint-data/2026-11-08-lplace/libmpr-post.so}
SHIM=${SHIM:-$S/mprshim-sites.so}   # the profiler (mprshim-sites2.so: with posix_memalign, aligned_alloc, memalign)
ONLY=${ONLY:-}                      # if set, a regex of the programs to run
NUMACTL=$(echo $HOME/spack/opt/spack/linux-broadwell/numactl-2.0.19-*/bin/numactl)
POLICIES=${POLICIES:-"glibc none res remote res_remote cold25 res_cold25 res_hot25 cold50 res_cold50 res_hot50"}
export OMP_WAIT_POLICY=passive
# prog threads binary-dir reservations wrapper
progs=("darknet 1 /home/nmust004/memory_estimator/workloads/darknet $R/runs/corpus/darknet/res-self.txt $D/darknet.sh"
       "llamacpp 4 $R/ml/llama.cpp/build/bin $R/runs/inputs/res-llamacpp-135m.txt $D/llamacpp.sh")
if [[ $STEPS == *" prof "* ]]; then
    mkdir -p $D/prof; : > $D/prof/list.txt
    for p in "${progs[@]}"; do
        read -r prog threads cwd res wrap <<< "$p"
        [[ -n $ONLY && ! $prog =~ $ONLY ]] && continue
        echo "$prog - $threads $cwd $wrap" >> $D/prof/list.txt
        o=$D/prof/runs/$prog/1; mkdir -p $o
        OMP_NUM_THREADS=$threads WRAP_PRELOAD=$SHIM MPR_NOTHP=1 MPR_MODE=refs MPR_SITES=1 MPR_SITE_MMAP=1 \
            MPR_CHUNK=2097152 MPR_WIN=0.1 MPR_OUT=$o/chunks.csv /usr/bin/time -o $o/chunks.time -f "%e %M" $wrap \
            > /dev/null 2> $o/chunks.err
        echo "$prog prof $? $(tail -1 $o/chunks.time)" >> $D/done.txt
    done
fi
[[ $STEPS == *" plans "* ]] && { ~/environments/memfp/bin/python $HERE/place_lplans.py $D/prof $D/lplans || exit 1; }
if [[ $STEPS == *" runs "* ]]; then
for rep in $(seq 1 $REPS); do
    for p in "${progs[@]}"; do
        read -r prog threads cwd res wrap <<< "$p"
        [[ -n $ONLY && ! $prog =~ $ONLY ]] && continue
        export OMP_NUM_THREADS=$threads
        o=$D/runs/$prog; mkdir -p $o
        for pol in $POLICIES; do
            node=0; r=/dev/null; plan=; lib=$LIB
            case $pol in
            glibc) lib= ;;
            none) ;;
            remote) node=1 ;;
            res) r=$res ;;
            res_remote) r=$res; node=1 ;;
            res_*) r=$res; plan=$D/lplans/$prog/${pol#res_}.txt ;;
            *) plan=$D/lplans/$prog/$pol.txt ;;
            esac
            [[ -n $plan && ! -f $plan ]] && { echo "no plan $plan" >&2; exit 1; }
            label=$pol-$rep
            env WRAP_PRELOAD=$lib MPR_NOTHP=1 ${lib:+MPR_NUMA=$o/$label.numa} ${plan:+MPR_PLACE=$plan} \
                $NUMACTL --cpunodebind=0 --membind=$node \
                $HOME/MemPrint/scripts/resize_eval.sh $o linked $r "/ - ALL $wrap" $label
            echo "$prog $rep $pol $(tail -1 $o/summary.txt)" >> $D/done.txt
        done
    done
done
fi
echo all-done
