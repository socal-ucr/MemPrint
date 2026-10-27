#!/bin/bash
# The model under jemalloc and TCMalloc (LD_PRELOAD), on the 7 allocator-heavy programs at their largest inputs.
#   paper_rss/run_alloc.sh truth   the pages-touched truth under each allocator (Pin, 7 at a time):
#                                  $OUT/<alloc>/<prog>/<input>/truthpages
#   paper_rss/run_alloc.sh model   3 repetitions, one run at a time, of a plain RSS run and the model's run under each
#                                  allocator: $OUT/<alloc>/runs/<prog>/<rep>/{base2,pss}.csv, $OUT/done_alloc.txt
# NOTHP=1: both runs with transparent huge pages disabled (MPR_NOTHP), into runs_nothp/ and done_alloc_nothp.txt
HERE=$(cd "$(dirname "$0")" && pwd); S=$HERE/../scripts
D=$HOME/memprint-data/2026-10-26-rsscurve; OUT=${OUT:-$D/alloc}
declare -A LIB=([jemalloc]=$HOME/memprint-data/allocators/jemalloc/lib/libjemalloc.so.2
                [tcmalloc]=$HOME/spack/opt/spack/linux-broadwell/gperftools-2.18-u4cqvcpknhrzftdrxadqtow2rchwko4l/lib/libtcmalloc_minimal.so)
PROGS="^(miniVite 32768|lua 1|python 1|perl 1|sqlite 1|cc1 large|quicksilver 100000) "
mkdir -p $OUT
sfx=${NOTHP:+_nothp}
case $1 in
truth)
    for al in jemalloc tcmalloc; do
        grep -E "$PROGS" $D/list.txt | while read -r prog input rest; do
            mkdir -p $OUT/$al/$prog/$input; ln -sfn $D/corpus/$prog/$input/native $OUT/$al/$prog/$input/native
            echo "$al|${LIB[$al]}|$prog $input $rest"
        done
    done | tr '\n' '\0' | OUT=$OUT S=$S xargs -0 -P 7 -I{} bash -c 'IFS="|" read -r al lib line <<< "$1"
        CORPUS=$OUT/$al TOOL=$HOME/memprint-data/2026-10-12-resize/corpus-pages/tool.so LD_PRELOAD=$lib \
        $S/footprint_corpus.sh truthpages "$line"' _ {} ;;
model)
    gcc -O2 -shared -fPIC -o $OUT/mprshim.so $HERE/tools/mprshim.c -lpthread -ldl || exit 1
    for rep in 1 2 3; do for al in jemalloc tcmalloc; do
    grep -E "$PROGS" $D/list.txt | while read -r prog input threads cwd cmd; do
        export OMP_NUM_THREADS=$threads OMP_WAIT_POLICY=passive
        read -r -a args <<< "$cmd"
        o=$OUT/$al/runs$sfx/$prog/$rep; mkdir -p $o; cd "$cwd" || exit 1
        t() { m=$1; shift; /usr/bin/time -o $o/$m.time -f "%e %M" "$@" < /dev/null > /dev/null 2> $o/$m.err
              echo "$al $prog $rep $m $? $(tail -1 $o/$m.time)" >> $OUT/done_alloc$sfx.txt; }
        if [[ -n $NOTHP ]]; then   # the shim without a mode only disables THP
            t base python3 $S/rss_sampler.py -i 0.005 -o $o/base2.csv -- \
                env LD_PRELOAD="${LIB[$al]} $OUT/mprshim.so" MPR_NOTHP=1 "${args[@]}"
        else
            t base python3 $S/rss_sampler.py -i 0.005 -o $o/base2.csv -- env LD_PRELOAD=${LIB[$al]} "${args[@]}"
        fi
        t pss  env LD_PRELOAD="${LIB[$al]} $OUT/mprshim.so" MPR_MODE=pss MPR_OUT=$o/pss.csv ${NOTHP:+MPR_NOTHP=1} \
            "${args[@]}"
    done; done; done ;;
esac
echo all-done
