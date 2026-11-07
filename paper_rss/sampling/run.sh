#!/bin/bash
# The runs of PROTOCOL.md, one at a time, into $OUT:
#   large/<prog>/<rep>/<sampler>.{csv,csv.hwm,time}  every program at its largest input, plain and under each sampler
#   small/<prog>/<input>/<rep>/<sampler>.*            the transfer study's smaller inputs under each sampler
#   blind/<sampler>/<prog>/<config>/<rep>.csv          blind/run.sh's configurations under fast and faults
#   py/<input>/<rep>/pss.*                             python under PYTHONMALLOC=malloc
#   cc1/<rep>/mmap.*, cc1/memreport.txt                cc1 with its own mappings counted; -fmem-report
#   thp.txt                                            the huge-page test (thp_test.c)
#   pages/python/1/{native,truthpages}                 the pages-in-use truth of python under PYTHONMALLOC=malloc
# Samplers: pss5 (pss, 5 ms), pss1 (pss, 1 ms), fast, faults, event (R8 of the dissertation plan: smaps_rollup after
# page faults and before frees, mprshim-event.so); plain is the program alone.
#   SAMPLERS="..." picks the samplers of large and small, BLIND="..." those of blind, LARGE=<dir> the directory of large
#   (R8: SAMPLERS="pss1 event" LARGE=large_r8 run.sh large; SAMPLERS=event BLIND=event run.sh small blind)
#   paper_rss/sampling/run.sh [part ...]   (parts: thp large small blind py cc1 truth; default all)
HERE=$(cd "$(dirname "$0")" && pwd)
D=$HOME/memprint-data/2026-10-26-rsscurve; PAGES=$HOME/memprint-data/2026-10-12-resize/corpus-pages
OUT=${OUT:-$HOME/memprint-data/2026-11-05-rsssample}; W=$HOME/MemPrint/workloads/src
mkdir -p $OUT/scratch
gcc -O2 -shared -fPIC -o $OUT/mprshim.so $HERE/../tools/mprshim.c -lpthread -ldl || exit 1
gcc -O2 -shared -fPIC -DMPR_EVENT_FREE -o $OUT/mprshim-event.so $HERE/../tools/mprshim.c -lpthread -ldl || exit 1
gcc -O2 -o $OUT/thp_test $HERE/thp_test.c || exit 1
SAMPLERS=${SAMPLERS:-"pss5 pss1 fast faults"}; BLIND=${BLIND:-"fast faults"}; LARGE=${LARGE:-large}

env_of() { case $1 in   # the shim's environment for a sampler
    pss5) echo "LD_PRELOAD=$OUT/mprshim.so MPR_MODE=pss" ;;
    pss1) echo "LD_PRELOAD=$OUT/mprshim.so MPR_MODE=pss MPR_INT=0.001" ;;
    fast) echo "LD_PRELOAD=$OUT/mprshim.so MPR_MODE=fast" ;;
    faults) echo "LD_PRELOAD=$OUT/mprshim.so MPR_MODE=faults" ;;
    event) echo "LD_PRELOAD=$OUT/mprshim-event.so MPR_MODE=event" ;;
    plain) echo "" ;;
esac; }

# run <dir> <sampler> <cwd> <threads> <stdin> <cmd...>
run() { local o=$1 s=$2 cwd=$3 threads=$4 stdin=$5; shift 5; mkdir -p $o
    (cd "$cwd" && OMP_NUM_THREADS=$threads OMP_WAIT_POLICY=passive /usr/bin/time -o $o/$s.time -f "%e %M" \
        env $(env_of $s) MPR_OUT=$o/$s.csv "$@" < $stdin > /dev/null 2> $o/$s.err)
    echo "$o $s $? $(tail -1 $o/$s.time)" >> $OUT/done.txt; }

each() {   # each <list filter: large|small> -> lines "prog input threads cwd cmd"
    for p in $(ls -d $PAGES/*/ | xargs -n1 basename); do
        big=$(ls -d $PAGES/$p/*/ | head -1 | xargs basename)
        if [[ $1 == large ]]; then grep "^$p $big " $D/list.txt; else grep "^$p " $D/list.txt | awk -v b="$big" '$2 != b'; fi
    done; }

args_of() { read -r -a A <<< "$1"; for k in "${!A[@]}"; do [[ ${A[k]} == EMPTY ]] && A[k]=""; A[k]=${A[k]//_SP_/ }; done; }

part_thp() { $OUT/thp_test > $OUT/thp.txt 2>&1; }

part_large() {   # plain and every sampler interleaved, 3 rounds
    for rep in 1 2 3; do each large | while read -r prog input threads cwd cmd; do
        args_of "$cmd"; stdin=/dev/null
        [[ $prog == darknet ]] && stdin=$HOME/memprint-data/2026-10-0x-memory_estimator/win/cand/images.txt
        for s in plain $SAMPLERS; do run $OUT/$LARGE/$prog/$rep $s "$cwd" $threads $stdin "${A[@]}"; done
    done; done; }

part_small() {
    for rep in 1 2 3; do each small | while read -r prog input threads cwd cmd; do
        args_of "$cmd"
        for s in $SAMPLERS; do run $OUT/small/$prog/$input/$rep $s "$cwd" $threads /dev/null "${A[@]}"; done
    done; done; }

part_blind() {
    b() { local prog=$1 cfg=$2; shift 2
        for rep in 1 2 3; do for s in $BLIND; do
            run $OUT/blind/$s/$prog/$cfg/$rep $s $OUT/scratch 1 /dev/null "$@"; done; done; }
    for n in 10 12 14 16 20 24 28; do b hpccg $n $W/hpccg/test_HPCCG $n $n $n; done
    for s in 5 8 10 12 15 18 20; do b lulesh $s $W/lulesh/lulesh2.0 -q -s $s -i 20; done
    for c in S W; do b IS $c $W/npb/NPB-SER/bin/is.$c; b CG $c $W/npb/NPB-SER/bin/cg.$c; done
    for c in S W A; do b MG $c $W/npb/NPB-SER/bin/mg.$c; done; }

part_py() {
    grep "^python " $D/list.txt | while read -r prog input threads cwd cmd; do args_of "$cmd"
        for rep in 1 2 3; do PYTHONMALLOC=malloc run $OUT/py/$input/$rep pss5 "$cwd" 1 /dev/null "${A[@]}"; done
    done; }

part_cc1() {
    read -r prog input threads cwd cmd <<< "$(each large | grep '^cc1 ')"; args_of "$cmd"
    for rep in 1 2 3; do o=$OUT/cc1/$rep; mkdir -p $o
        (cd "$cwd" && /usr/bin/time -o $o/mmap.time -f "%e %M" env LD_PRELOAD=$OUT/mprshim.so MPR_MODE=pss MPR_MMAP=1 \
            MPR_OUT=$o/mmap.csv "${A[@]}" < /dev/null > /dev/null 2> $o/mmap.err); done
    (cd "$cwd" && "${A[@]}" -fmem-report < /dev/null > /dev/null 2> $OUT/cc1/memreport.txt); }

part_truth() {   # the corpus's Pin pipeline, in a corpus of its own
    mkdir -p $OUT/pages; cp $PAGES/tool.so $OUT/pages/tool.so
    line=$(each large | grep '^python ')
    for m in native truthpages; do
        PYTHONMALLOC=malloc CORPUS=$OUT/pages TOOL=$OUT/pages/tool.so $HERE/../../scripts/footprint_corpus.sh $m "$line"
    done; }

for part in ${@:-thp large small blind py cc1 truth}; do echo "== $part $(date)"; part_$part; done
echo all-done
