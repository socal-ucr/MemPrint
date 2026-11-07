#!/bin/bash
# Hot and cold memory from the accessed bits (mprshim MPR_MODE=refs), no Pin. The 7 allocator-heavy programs and the 4
# GAP kernels at their largest inputs, THP off (MPR_NOTHP) in every run, one run at a time:
#   base      the shim without a mode (THP off only), 3 repetitions: the runtime reference
#   refs      windows of 0.1 s, 3 repetitions: zones heap/anon/stack/data/code
#   sites     windows of 0.1 s with call sites (-DMPR_SITES build, blocks >= 256 KiB), 3 repetitions
#   sites_w   windows of 0.02 s and 0.5 s with sites, 1 repetition
#   chunks    windows of 0.1 s, sites from big mallocs and the program's own mmaps (>= 64 KiB), blocks split into 2 MiB
#             chunks, 3 repetitions
#   chunks_w  as chunks with windows of 0.02 s and 0.5 s, 1 repetition
# PROGS (regex over list.txt lines), CWIN (chunks window, default 0.1), REPS (default 3).
# RUNS (default "base refs sites sites_w chunks chunks_w") picks the runs.
# -> $OUT/runs/<prog>/<rep>/<run>.csv{,.zones,.sites,.time,.err}, $OUT/done.txt "prog rep run status seconds maxrss_kb"
HERE=$(cd "$(dirname "$0")" && pwd)
D=$HOME/memprint-data/2026-10-26-rsscurve; OUT=${OUT:-$HOME/memprint-data/2026-11-07-refs}
PROGS=${PROGS:-"^(miniVite 32768|lua 1|python 1|perl 1|sqlite 1|cc1 large|quicksilver 100000|gap-(bfs|pr|cc|sssp) 18) "}
CWIN=${CWIN:-0.1}; REPS=${REPS:-3}   # window of the chunks run; repetitions
mkdir -p $OUT; cp $HERE/tools/mprshim.c $OUT/
gcc -O2 -shared -fPIC -o $OUT/mprshim.so $HERE/tools/mprshim.c -lpthread -ldl || exit 1
gcc -O2 -DMPR_SITES -shared -fPIC -o $OUT/mprshim-sites.so $HERE/tools/mprshim.c -lpthread -ldl || exit 1
grep -E "$PROGS" $D/list.txt > $OUT/list.txt
RUNS=" ${RUNS:-base refs sites sites_w chunks chunks_w} "; on() { [[ $RUNS == *" $1 "* ]]; }
for rep in $(seq 1 $REPS); do
while read -r prog input threads cwd cmd; do
    export OMP_NUM_THREADS=$threads OMP_WAIT_POLICY=passive
    read -r -a args <<< "$cmd"
    o=$OUT/runs/$prog/$rep; mkdir -p $o; cd "$cwd" || exit 1
    t() { m=$1; shift; /usr/bin/time -o $o/$m.time -f "%e %M" "$@" < /dev/null > /dev/null 2> $o/$m.err
          echo "$prog $rep $m $? $(tail -1 $o/$m.time)" >> $OUT/done.txt; }
    on base && t base  env LD_PRELOAD=$OUT/mprshim.so MPR_NOTHP=1 "${args[@]}"
    on refs && t refs  env LD_PRELOAD=$OUT/mprshim.so MPR_NOTHP=1 MPR_MODE=refs MPR_WIN=0.1 MPR_OUT=$o/refs.csv "${args[@]}"
    on sites && t sites env LD_PRELOAD=$OUT/mprshim-sites.so MPR_NOTHP=1 MPR_MODE=refs MPR_SITES=1 MPR_WIN=0.1 \
        MPR_OUT=$o/sites.csv "${args[@]}"
    if on sites_w && [[ $rep == 1 ]]; then for w in 0.02 0.5; do
        t sites_w$w env LD_PRELOAD=$OUT/mprshim-sites.so MPR_NOTHP=1 MPR_MODE=refs MPR_SITES=1 MPR_WIN=$w \
            MPR_OUT=$o/sites_w$w.csv "${args[@]}"
    done; fi
    on chunks && t chunks env LD_PRELOAD=$OUT/mprshim-sites.so MPR_NOTHP=1 MPR_MODE=refs MPR_SITES=1 MPR_SITE_MMAP=1 \
        MPR_CHUNK=2097152 MPR_WIN=$CWIN MPR_OUT=$o/chunks.csv "${args[@]}"
    if on chunks_w && [[ $rep == 1 ]]; then for w in 0.02 0.5; do
        t chunks_w$w env LD_PRELOAD=$OUT/mprshim-sites.so MPR_NOTHP=1 MPR_MODE=refs MPR_SITES=1 MPR_SITE_MMAP=1 \
            MPR_CHUNK=2097152 MPR_WIN=$w MPR_OUT=$o/chunks_w$w.csv "${args[@]}"
    done; fi
done < $OUT/list.txt
done
echo all-done
