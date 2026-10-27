#!/bin/bash
# The model on every smaller input of each program of run_model.sh (the training inputs of the transfer study), one
# run each, one at a time: $OUT/runs_small/<prog>/<input>/pss.csv, $OUT/done_small.txt
HERE=$(cd "$(dirname "$0")" && pwd)
D=$HOME/memprint-data/2026-10-26-rsscurve; OUT=${OUT:-$D/opts}; PAGES=$HOME/memprint-data/2026-10-12-resize/corpus-pages
gcc -O2 -shared -fPIC -o $OUT/mprshim.so $HERE/tools/mprshim.c -lpthread -ldl || exit 1
for p in $(ls -d $PAGES/*/ | xargs -n1 basename); do
    big=$(ls -d $PAGES/$p/*/ | head -1 | xargs basename)
    grep "^$p " $D/list.txt | awk -v b="$big" '$2 != b' | while read -r prog input threads cwd cmd; do
        export OMP_NUM_THREADS=$threads OMP_WAIT_POLICY=passive
        read -r -a args <<< "$cmd"
        for k in "${!args[@]}"; do [[ ${args[k]} == EMPTY ]] && args[k]=""; args[k]=${args[k]//_SP_/ }; done
        o=$OUT/runs_small/$prog/$input; mkdir -p $o; cd "$cwd" || exit 1
        /usr/bin/time -o $o/pss.time -f "%e %M" env LD_PRELOAD=$OUT/mprshim.so MPR_MODE=pss MPR_OUT=$o/pss.csv \
            "${args[@]}" < /dev/null > /dev/null 2> $o/pss.err
        echo "$prog $input pss $? $(tail -1 $o/pss.time)" >> $OUT/done_small.txt
    done
done
echo all-done
