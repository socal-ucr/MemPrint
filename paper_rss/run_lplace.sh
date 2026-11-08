#!/bin/bash
# Placement through libmpr (MPR_PLACE) on the emulated pool of run_place.sh: CPUs and default memory on NUMA node 0, the
# "pool" node 1. Every run but "plain" preloads the same libmpr (MPR_MIN=262144, as mprshim's big blocks; THP off;
# numa_maps every 0.25 s) and differs only in where memory goes:
#   plain   no libmpr, --membind=0               local   --membind=0, no plan        remote  --membind=1, no plan
#   l<tag>_<policy><pct>  --membind=0, place lines $D/lplans_<tag>/<prog>/<policy><pct>.txt (place_lplans.py)
# Repetitions interleave the policies. -> $D/$PLACE/<prog>/<rep>/<policy>.csv{,.time,.err}, $D/done_$PLACE.txt
# (place_summary.py $D $PLACE summarises).
D=${D:-$HOME/memprint-data/2026-11-08-lplace}; REPS=${REPS:-5}; PLACE=${PLACE:-place}
LIST=${LIST:-$HOME/memprint-data/2026-11-07-refs/prof_large/list.txt}
LIB=${LIB:-$D/libmpr-place.so}
NUMACTL=$(echo $HOME/spack/opt/spack/linux-broadwell/numactl-2.0.19-*/bin/numactl)
POLICIES=${POLICIES:-"plain local remote lsame_cold25 lxfer_cold25 lxfer_rand25 lxfer_hot25 lsame_cold50 lxfer_cold50 lxfer_hot50"}
ONLY=${ONLY:-}   # if set, a regex of the programs to run
for rep in $(seq 1 $REPS); do
while read -r prog input threads cwd cmd; do
    [[ -n $ONLY && ! $prog =~ $ONLY ]] && continue
    export OMP_NUM_THREADS=$threads OMP_WAIT_POLICY=passive
    read -r -a args <<< "$cmd"
    o=$D/$PLACE/$prog/$rep; mkdir -p $o; cd "$cwd" || exit 1
    for pol in $POLICIES; do
        node=0; pre=(LD_PRELOAD=$LIB MPR_MIN=262144 MPR_NOTHP=1 MPR_NUMA=$o/$pol.csv)
        case $pol in
        plain) pre=() ;;
        local) ;;
        remote) node=1 ;;
        l*_*) t=${pol%%_*}; p=$D/lplans_${t#l}/$prog/${pol#*_}.txt
              [[ -f $p ]] || { echo "no plan $p" >&2; exit 1; }
              pre+=(MPR_PLACE=$p) ;;
        *) echo "unknown policy $pol" >&2; exit 1 ;;
        esac
        /usr/bin/time -o $o/$pol.time -f "%e %M" $NUMACTL --cpunodebind=0 --membind=$node \
            env "${pre[@]}" "${args[@]}" < /dev/null > /dev/null 2> $o/$pol.err
        echo "$prog $rep $pol $? $(tail -1 $o/$pol.time)" >> $D/done_$PLACE.txt
    done
done < $LIST
done
echo all-done
