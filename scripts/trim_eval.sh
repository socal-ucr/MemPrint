#!/bin/bash
# trim_eval.sh <workload>: one of the resize study's 29 workloads (a corpus program at its largest input,
# "miniVite (pass)" as minivite, or an LLM configuration such as llama2c-15M) under glibc and libmpr's own and
# held-out reservations, each without and with MPR_TRIM, 3 runs each, into $OUT/<workload>/summary.txt.
# The reservations are the resize study's (runs/corpus, runs/minivite-v4, runs/ml); libmpr is $LIBMPR (default
# libmpr-v8.so), and the pass-built miniVite is a copy of build/minivite-v4 whose rpath names build/mv8.
# BUILDS="trim selftrim" runs only those labels (default: all six).
set -u
R=$HOME/memprint-data/2026-10-12-resize; C=$HOME/memprint-data/2026-10-11-sched/corpus
OUT=${OUT:-$HOME/memprint-data/2026-10-27-rsstrim/runs}
export LIBMPR=${LIBMPR:-$R/libmpr-v8.so}
export PATH=/home/nmust004/spack/opt/spack/linux-centos7-haswell/gcc-4.8.5/openmpi-5.0.5-hwdt6fs2qm5ybyrq4u73lbgdv6utoz6x/bin:$PATH OMP_WAIT_POLICY=passive
EVAL=$HOME/MemPrint/scripts/resize_eval.sh
name=$1
O=$OUT/$name; rm -rf "$O"; mkdir -p "$O"
plain=base glibc=trim mpr=mpr mprt=mprtrim
case $name in
  minivite)
    S=$R/runs/minivite-v4
    export OMP_NUM_THREADS=4
    spec="/ - Modularity, $R/build/minivite-v8/miniVite -n 32768"
    plain=linked glibc=linkedtrim mpr=linked mprt=linkedtrim ;;
  llama2c-*)
    S=$R/runs/ml/$name
    spec="/ - . $R/ml/llama2c.sh ${name#llama2c-}" ;;
  llamacpp-*)
    S=$R/runs/ml/$name; tag=${name#llamacpp-}
    spec="/ - . $R/ml/llamacpp.sh ${tag%-n*} ${tag##*-n}" ;;
  *)
    S=$R/runs/corpus/$name
    mapfile -t lines < <(awk -v p="$name" '$1 == p && !($1 == "npb-mg" && $2 == "A")' $C/list.txt)
    read -r _ input threads cwd cmd <<< "${lines[-1]}"
    export OMP_NUM_THREADS=$threads
    stdin=-; [[ $name == darknet ]] && stdin=/home/nmust004/memprint-data/2026-10-0x-memory_estimator/win/cand/images.txt
    case $name in   # as corpus_eval_v3.sh
      sqlite) pattern='^[0-9|]+$' ;;
      hpccg) pattern='Residual' ;;
      kripke) pattern='particle.count' ;;
      quicksilver) pattern='^NO_LINE_MATCHES$' ;;
      *) pattern=ALL ;;
    esac
    spec="$cwd $stdin $pattern $cmd" ;;
esac
cp "$S/res-self.txt" "$S/res-heldout.txt" "$O/"
want() { [[ " ${BUILDS:-base trim self selftrim heldout heldouttrim} " == *" $1 "* ]]; }
for rep in 1 2 3; do
    want base && $EVAL "$O" $plain /dev/null "$spec" base-$rep
    want trim && $EVAL "$O" $glibc /dev/null "$spec" trim-$rep
    want self && $EVAL "$O" $mpr "$O/res-self.txt" "$spec" self-$rep
    want selftrim && $EVAL "$O" $mprt "$O/res-self.txt" "$spec" selftrim-$rep
    want heldout && $EVAL "$O" $mpr "$O/res-heldout.txt" "$spec" heldout-$rep
    want heldouttrim && $EVAL "$O" $mprt "$O/res-heldout.txt" "$spec" heldouttrim-$rep
done
echo "$name" >> "$OUT/done.txt"
