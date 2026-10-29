#!/bin/bash
# The RSS model on paper_generalization's blind configurations (PROTOCOL.md): 3 runs of each under the shim's pss mode
# at 1 ms, into $OUT/<program>/<config>/<rep>.csv.
OUT=${OUT:-$HOME/memprint-data/2026-10-28-rssblind}; W=$HOME/MemPrint/workloads/src
mkdir -p $OUT/scratch; gcc -O2 -shared -fPIC -o $OUT/mprshim.so $HOME/MemPrint/paper_rss/tools/mprshim.c -lpthread -ldl || exit 1
run() { local prog=$1 cfg=$2; shift 2; local o=$OUT/$prog/$cfg; mkdir -p $o
    for rep in 1 2 3; do (cd $OUT/scratch && env LD_PRELOAD=$OUT/mprshim.so MPR_MODE=pss MPR_INT=0.001 MPR_OUT=$o/$rep.csv \
        "$@" > $o/$rep.out 2>&1; echo "$prog $cfg $rep $?" >> $OUT/done.txt); done; }
for n in 10 12 14 16 20 24 28; do run hpccg $n $W/hpccg/test_HPCCG $n $n $n; done
for s in 5 8 10 12 15 18 20; do run lulesh $s $W/lulesh/lulesh2.0 -q -s $s -i 20; done
for c in S W; do run IS $c $W/npb/NPB-SER/bin/is.$c; run CG $c $W/npb/NPB-SER/bin/cg.$c; done
for c in S W A; do run MG $c $W/npb/NPB-SER/bin/mg.$c; done
echo all-done
