#!/bin/bash
# R9: the pages-touched truth (4 KiB pages x 4096) and the bytes-touched truth of every R1 EXTRA and INTER input,
# one interval and one bin, into $OUT/r9/all/{pages,bytes}/<workload>/<config>; score with r9score.py
ROOT=/home/nmust004/MemPrint; source $ROOT/scripts/lib.sh
export PIN_ROOT=/home/nmust004/memory_estimator/pin/pin-3.30-98830-g1d7b601b3-gcc-linux OMP_NUM_THREADS=1
O=$HOME/memprint-data/2026-11-05-diss-ch2
tail -n +2 $O/r1_splits.csv | cut -d, -f1,2 | while IFS=, read -r w c; do
    wl=polybench; ex=(); [[ $w == miniVite ]] && wl=minivite ex=(-exclude_alloc 1)
    load_workload $wl; (cd $SRC && wl_build $w $c) > /dev/null 2>&1
    read -r -a A <<< "$(wl_args $w $c)"; B=$(cd $SRC && wl_binary $w $c)
    for fp in pages bytes; do d=$O/r9/all/$fp/$w/$c; mkdir -p $d
        $PIN_ROOT/pin -t $O/memprint_trace.so -mode splitter -intervals 1 -bins 1 -footprint $fp "${ex[@]}" -outdir $d -name $w-$c -- $B "${A[@]}" > /dev/null 2>&1
    done
    echo "$w $c" >> $O/r9/all/done.txt
done
echo all-done >> $O/r9/all/done.txt
