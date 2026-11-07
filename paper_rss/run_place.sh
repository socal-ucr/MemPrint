#!/bin/bash
# Placement on an emulated pool: CPUs and default memory on NUMA node 0, the "pool" is node 1 (distance 21 vs 10).
# Every run uses the same shim (sites build, MPR_MODE=place, big mallocs and own mmaps as blocks, 2 MiB chunks, THP off,
# numa_maps every 0.25 s) and differs only in where memory goes:
#   local     numactl --membind=0, no plan            remote  numactl --membind=1, no plan
#   <policy><pct>  --membind=0, chunks in $PLANS/<prog>/<policy><pct>.csv bound to node 1 (place_plans.py)
#   r<tag>_<policy><pct>  --membind=0, relative plan rplans_<tag>/<prog>/<policy><pct>.csv (place_rplans.py)
#   dyn<pct>[h]    --membind=0, migration every 0.1 s: the coldest (h: hottest) chunks up to pct % of anonymous RSS
#                  on node 1 (MPR_MIGRATE); dyn0 only monitors
#   dyna<pct>      as dyn<pct> with windows of 0.25 s and a slow score (MPR_ALPHA=0.1)
#   dynt<pct>[w]   as dyn<pct>, but only chunks scored <= 0.1 move (MPR_MIGRATE_MAX); w: windows of 0.25 s
# Repetitions interleave the policies. -> $OUT/$PLACE/<prog>/<rep>/<policy>.csv{,.time,.err}, $OUT/done_$PLACE.txt
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HOME/memprint-data/2026-11-07-refs}; REPS=${REPS:-5}
NUMACTL=$(echo $HOME/spack/opt/spack/linux-broadwell/numactl-2.0.19-*/bin/numactl)
POLICIES=${POLICIES:-"local remote cold25 hot25 rand25 cold50 hot50 rand50"}
PLANS=${PLANS:-plans_time}; PLACE=${PLACE:-place}   # plan directory and output directory under $OUT
SKIP=${SKIP:-sqlite}   # no site memory to place
ONLY=${ONLY:-}         # if set, a regex of the programs to run
for rep in $(seq 1 $REPS); do
while read -r prog input threads cwd cmd; do
    [[ " $SKIP " == *" $prog "* ]] && continue
    [[ -n $ONLY && ! $prog =~ $ONLY ]] && continue
    export OMP_NUM_THREADS=$threads OMP_WAIT_POLICY=passive
    read -r -a args <<< "$cmd"
    o=$OUT/$PLACE/$prog/$rep; mkdir -p $o; cd "$cwd" || exit 1
    for pol in $POLICIES; do
        node=0; extra=(MPR_WIN=0.25)
        case $pol in
        local) ;;
        remote) node=1 ;;
        r*_*) t=${pol%%_*}; extra+=(MPR_RPLAN=$OUT/rplans_${t#r}/$prog/${pol#*_}.csv) ;;   # rxfer_cold25
        dyna*) f=${pol#dyna}                                                               # dyna25, dyna50
               extra=(MPR_WIN=0.25 MPR_ALPHA=0.1 MPR_MIGRATE=$(echo "$f / 100" | bc -l)) ;;
        dynt*) f=${pol#dynt}; w=0.1; [[ $f == *w ]] && { w=0.25; f=${f%w}; }              # dynt50, dynt50w, dynt0w
               extra=(MPR_WIN=$w MPR_MIGRATE=$(echo "$f / 100" | bc -l) MPR_MIGRATE_MAX=0.1) ;;
        dyn*) f=${pol#dyn}; h=; [[ $f == *h ]] && { h=1; f=${f%h}; }                       # dyn50, dyn50h, dyn0
              extra=(MPR_WIN=0.1 MPR_MIGRATE=$(echo "$f / 100" | bc -l) ${h:+MPR_MIGRATE_HOT=1}) ;;
        *) extra+=(MPR_PLAN=$OUT/$PLANS/$prog/$pol.csv)                                    # cold25: chunk plan
           [[ -f $OUT/$PLANS/$prog/$pol.csv ]] || { echo "no plan $OUT/$PLANS/$prog/$pol.csv" >&2; exit 1; } ;;
        esac
        /usr/bin/time -o $o/$pol.time -f "%e %M" $NUMACTL --cpunodebind=0 --membind=$node \
            env LD_PRELOAD=$OUT/mprshim-sites.so MPR_NOTHP=1 MPR_MODE=place MPR_SITES=1 MPR_SITE_MMAP=1 \
            MPR_CHUNK=2097152 "${extra[@]}" MPR_OUT=$o/$pol.csv "${args[@]}" \
            < /dev/null > /dev/null 2> $o/$pol.err
        echo "$prog $rep $pol $? $(tail -1 $o/$pol.time)" >> $OUT/done_$PLACE.txt
    done
done < $OUT/list.txt
done
echo all-done
