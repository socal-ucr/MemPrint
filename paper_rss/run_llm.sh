#!/bin/bash
# The model with mapped data files (MPR_MODE=files) on the LLM engines, and their pages-touched truth.
#   paper_rss/run_llm.sh truth   pages-touched truth of llama2.c and llama.cpp (Pin, 4 at a time; llama.cpp 360M is
#                                left out: Pin's splitter ran out of memory on it): $OUT/<prog>/<input>/truthpages
#   paper_rss/run_llm.sh model   3 repetitions, one run at a time, of a plain RSS run and the model's run (files mode)
#                                of every LLM input: $OUT/runs/<prog>/<input>/<rep>/{base2,files}.csv, $OUT/done_runs.txt
# NOTHP=1: both runs with transparent huge pages disabled (MPR_NOTHP), into runs_nothp/ and done_runs_nothp.txt
HERE=$(cd "$(dirname "$0")" && pwd); S=$HERE/../scripts
D=$HOME/memprint-data/2026-10-26-rsscurve; OUT=${OUT:-$D/llm}
mkdir -p $OUT
sfx=${NOTHP:+_nothp}
case $1 in
truth)
    grep -E "^(llama2c|llamacpp) " $D/list.txt | grep -v "^llamacpp 360m" | while read -r prog input rest; do
        mkdir -p $OUT/$prog/$input; ln -sfn $D/corpus/$prog/$input/native $OUT/$prog/$input/native
        echo "$prog $input $rest"
    done | tr '\n' '\0' | OUT=$OUT S=$S xargs -0 -P 4 -I{} bash -c 'CORPUS=$OUT \
        TOOL=$HOME/memprint-data/2026-10-12-resize/corpus-pages/tool.so $S/footprint_corpus.sh truthpages "$1"' _ {} ;;
model)
    gcc -O2 -shared -fPIC -o $OUT/mprshim.so $HERE/tools/mprshim.c -lpthread -ldl || exit 1
    for rep in 1 2 3; do
    grep -E "^(llama2c|llamacpp) " $D/list.txt | while read -r prog input threads cwd cmd; do
        export OMP_NUM_THREADS=$threads OMP_WAIT_POLICY=passive
        read -r -a args <<< "$cmd"
        for k in "${!args[@]}"; do args[k]=${args[k]//_SP_/ }; done
        o=$OUT/runs$sfx/$prog/$input/$rep; mkdir -p $o; cd "$cwd" || exit 1
        t() { m=$1; shift; /usr/bin/time -o $o/$m.time -f "%e %M" "$@" < /dev/null > /dev/null 2> $o/$m.err
              echo "$prog $input $rep $m $? $(tail -1 $o/$m.time)" >> $OUT/done_runs$sfx.txt; }
        if [[ -n $NOTHP ]]; then   # the shim without a mode only disables THP
            t base python3 $S/rss_sampler.py -i 0.005 -o $o/base2.csv -- env LD_PRELOAD=$OUT/mprshim.so MPR_NOTHP=1 "${args[@]}"
        else
            t base python3 $S/rss_sampler.py -i 0.005 -o $o/base2.csv -- "${args[@]}"
        fi
        t files env LD_PRELOAD=$OUT/mprshim.so MPR_MODE=files MPR_OUT=$o/files.csv ${NOTHP:+MPR_NOTHP=1} "${args[@]}"
    done; done ;;
esac
echo all-done
