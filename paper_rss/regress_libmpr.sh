#!/bin/bash
# Regression check for libmpr changes: the resize study's runs (each corpus program's largest input with its own
# reservations, runs/corpus/<prog>/res-self.txt, and llama.cpp 135m/360m) under two builds of libmpr, PRE and POST,
# REPS repetitions interleaved (scripts/resize_eval.sh, config mpr). Compare peak commit, peak RSS, output hash and
# time per program: -> $O/<name>/summary.txt (labels pre-<rep>, post-<rep>), $O/done.txt
R=$HOME/memprint-data/2026-10-12-resize; C=$HOME/memprint-data/2026-10-11-sched/corpus
D=${D:-$HOME/memprint-data/2026-11-08-lplace}; O=${O:-$D/regress}; REPS=${REPS:-2}
PRE=${PRE:-$D/libmpr-pre.so}; POST=${POST:-$D/libmpr-post.so}
export PATH=/home/nmust004/spack/opt/spack/linux-centos7-haswell/gcc-4.8.5/openmpi-5.0.5-hwdt6fs2qm5ybyrq4u73lbgdv6utoz6x/bin:$PATH OMP_WAIT_POLICY=passive
one() { # name res "cwd stdin pattern cmd" threads
    local o=$O/$1; rm -rf $o; mkdir -p $o
    export OMP_NUM_THREADS=$4
    for rep in $(seq 1 $REPS); do
        LIBMPR=$PRE $HOME/MemPrint/scripts/resize_eval.sh $o mpr $2 "$3" pre-$rep
        LIBMPR=$POST $HOME/MemPrint/scripts/resize_eval.sh $o mpr $2 "$3" post-$rep
    done
    echo $1 >> $O/done.txt
}
mkdir -p $O
for prog in $(awk '{print $1}' $C/list.txt | uniq); do
    mapfile -t lines < <(awk -v p=$prog '$1 == p && !($1 == "npb-mg" && $2 == "A")' $C/list.txt)
    read -r _ input threads cwd cmd <<< "${lines[-1]}"
    stdin=-; [[ $prog == darknet ]] && stdin=/home/nmust004/memprint-data/2026-10-0x-memory_estimator/win/cand/images.txt
    case $prog in sqlite) pattern='^[0-9|]+$' ;; hpccg) pattern='Residual' ;; kripke) pattern='particle.count' ;;
                  quicksilver) pattern='^NO_LINE_MATCHES$' ;; *) pattern=ALL ;; esac
    one $prog $R/runs/corpus/$prog/res-self.txt "$cwd $stdin $pattern $cmd" $threads
done
one llamacpp-135m-n32 $R/runs/inputs/res-llamacpp-135m.txt "/ - . $R/ml/llamacpp_gen.sh 135m 32 0 1" 4
one llamacpp-360m-n64 $R/runs/inputs/res-llamacpp-360m.txt "/ - . $R/ml/llamacpp_gen.sh 360m 64 0 1" 4
echo all-done
