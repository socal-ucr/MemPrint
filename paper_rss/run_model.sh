#!/bin/bash
# The model on every program of the corpus at its largest input (the inputs of the pages-touched truth), 3
# repetitions, one run at a time: a plain RSS run (base2.csv, the cost baseline) and a run under the shim in its
# pss mode (pss.csv: RSS, PSS, anonymous RSS and glibc's free bytes every 5 ms, from one in-process thread).
# RUNS="files" runs the shim's files mode instead (files.csv: the same from smaps, with mapped data files);
# RUNS="basenothp nothp" a plain RSS run and the files mode with transparent huge pages disabled (MPR_NOTHP).
#   paper_rss/run_model.sh   ->  $OUT/runs_all/<prog>/<rep>/{base2,pss,files,basenothp,nothp}.csv, $OUT/done_all.txt
HERE=$(cd "$(dirname "$0")" && pwd); S=$HERE/../scripts
D=$HOME/memprint-data/2026-10-26-rsscurve; OUT=${OUT:-$D/opts}; PAGES=$HOME/memprint-data/2026-10-12-resize/corpus-pages
gcc -O2 -shared -fPIC -o $OUT/mprshim.so $HERE/tools/mprshim.c -lpthread -ldl || exit 1
for rep in 1 2 3; do
for p in $(ls -d $PAGES/*/ | xargs -n1 basename); do
    input=$(ls -d $PAGES/$p/*/ | head -1 | xargs basename)
    read -r prog input threads cwd cmd <<< "$(grep "^$p $input " $D/list.txt)"
    export OMP_NUM_THREADS=$threads OMP_WAIT_POLICY=passive
    read -r -a args <<< "$cmd"
    for k in "${!args[@]}"; do [[ ${args[k]} == EMPTY ]] && args[k]=""; args[k]=${args[k]//_SP_/ }; done
    stdin=/dev/null; [[ $prog == darknet ]] && stdin=$HOME/memprint-data/2026-10-0x-memory_estimator/win/cand/images.txt
    o=$OUT/runs_all/$prog/$rep; mkdir -p $o; cd "$cwd" || exit 1
    t() { m=$1; shift; /usr/bin/time -o $o/$m.time -f "%e %M" "$@" < $stdin > /dev/null 2> $o/$m.err
          echo "$prog $rep $m $? $(tail -1 $o/$m.time)" >> $OUT/done_all.txt; }
    for run in ${RUNS:-base pss}; do case $run in
      base)  t base python3 $S/rss_sampler.py -i 0.005 -o $o/base2.csv -- "${args[@]}" ;;
      pss)   t pss  env LD_PRELOAD=$OUT/mprshim.so MPR_MODE=pss MPR_OUT=$o/pss.csv "${args[@]}" ;;
      files) t files env LD_PRELOAD=$OUT/mprshim.so MPR_MODE=files MPR_OUT=$o/files.csv "${args[@]}" ;;
      basenothp) t basenothp python3 $S/rss_sampler.py -i 0.005 -o $o/basenothp.csv -- \
                     env LD_PRELOAD=$OUT/mprshim.so MPR_NOTHP=1 "${args[@]}" ;;
      nothp) t nothp env LD_PRELOAD=$OUT/mprshim.so MPR_MODE=files MPR_NOTHP=1 MPR_OUT=$o/nothp.csv "${args[@]}" ;;
    esac; done
done; done
echo all-done
