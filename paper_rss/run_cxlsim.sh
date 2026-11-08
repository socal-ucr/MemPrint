#!/bin/bash
# Per-workload pool costs for the cluster simulation with a CXL pool (cxl_sim.py): the resize study's 29 workloads
# (the 23 corpus programs at their largest inputs, miniVite, llama.cpp and llama2.c), each through a wrapper in
# $D/wrap/<name>.sh that preloads $WRAP_PRELOAD into the program only.
#   wrap   write the wrappers and $D/list.txt ("name - threads / wrapper", the format of place_lplans.py)
#   prof   one mprshim chunks profile per workload (2 MiB chunks, windows of 0.1 s, 0.02 s for runs under 1 s, THP off)
#   plans  place_lplans.py -> $D/lplans/<name>/{cold,hot,rand}{25,50}.txt (plans from the workload's own profile)
#   bw     the pool's loaded latency: hog.c with 0 to 16 threads on node 1 (delivered GB/s) and the pointer chase
#          from node 0 to each node meanwhile -> $D/bw.txt ("threads GB/s local_ns pool_ns")
#   runs   every workload under libmpr (no reservations, blocks of 256 KiB or more served, THP off) with each
#          policy, under perf stat (the DRAM read counts of run_cxl.sh), node 1 idle (hog 0) and busy (hog H):
#            local (all on node 0), remote (--membind=1), and the plans cold/hot 25, 50 and rand 25
#          -> $D/runs/<name>/<rep>/<hog>/<pol>.{perf,time,hash}, $D/done.txt
#   STEPS (default "wrap prof plans bw runs"), REPS (default 3), H (default 8), ONLY (regex of names)
HERE=$(cd "$(dirname "$0")" && pwd)
R=$HOME/memprint-data/2026-10-12-resize; C=$HOME/memprint-data/2026-10-11-sched/corpus
D=${D:-$HOME/memprint-data/2026-11-08-cxlsim}; REPS=${REPS:-3}; H=${H:-8}; ONLY=${ONLY:-}
STEPS=" ${STEPS:-wrap prof plans bw runs} "
LIB=${LIB:-$HOME/memprint-data/2026-11-08-lplace/libmpr-post3.so}
SHIM=${SHIM:-$HOME/memprint-data/2026-11-08-lplace/mprshim-sites2.so}
NUMACTL=$(echo $HOME/spack/opt/spack/linux-broadwell/numactl-2.0.19-*/bin/numactl)
MPI=/home/nmust004/spack/opt/spack/linux-centos7-haswell/gcc-4.8.5/openmpi-5.0.5-hwdt6fs2qm5ybyrq4u73lbgdv6utoz6x/bin
EV=mem_load_uops_l3_miss_retired.local_dram:u,mem_load_uops_l3_miss_retired.remote_dram:u
EV=$EV,offcore_response.all_reads.llc_miss.local_dram:u,offcore_response.all_reads.llc_miss.remote_dram:u
POLICIES=${POLICIES:-"local remote rand25 cold25 cold50 hot25 hot50"}
SHORT="atax npb-mg llamacpp-135m-n32"   # runs under 1 s: profile windows of 0.02 s
export OMP_WAIT_POLICY=passive
mkdir -p $D
want() { [[ -z $ONLY || $1 =~ $ONLY ]]; }

if [[ $STEPS == *" wrap "* ]]; then
    mkdir -p $D/wrap; : > $D/list.txt
    wrap() { # name threads cwd stdin cmd...
        local n=$1 t=$2 cwd=$3 in=$4; shift 4
        local args=""; for a in "$@"; do [[ $a == EMPTY ]] && args+=' ""' || args+=" $a"; done
        printf '#!/bin/bash\nexport PATH=%s:$PATH\ncd %s && exec env ${WRAP_PRELOAD:+LD_PRELOAD=$WRAP_PRELOAD}%s < %s\n' \
            "$MPI" "$cwd" "$args" "$in" > $D/wrap/$n.sh
        chmod +x $D/wrap/$n.sh
        echo "$n - $t / $D/wrap/$n.sh" >> $D/list.txt
    }
    for prog in $(awk '{print $1}' $C/list.txt | sort -u); do
        mapfile -t lines < <(awk -v p=$prog '$1 == p && !($1 == "npb-mg" && $2 == "A")' $C/list.txt)
        read -r _ input threads cwd cmd <<< "${lines[-1]}"
        stdin=/dev/null; [[ $prog == darknet ]] && stdin=$HOME/memprint-data/2026-10-0x-memory_estimator/win/cand/images.txt
        wrap $prog $threads $cwd $stdin $cmd
    done
    for m in "135m 32" "135m 128" "360m 64"; do
        set -- $m
        p=$(sed -n 1p $R/ml/prompts.txt)
        wrap llamacpp-$1-n$2 4 $R/ml/llama.cpp/build/bin /dev/null ./llama-completion -m $R/ml/models/smollm2-$1-q8.gguf \
            -no-cnv -n $2 -c 0 --temp 0 -s 1 -t 4 -p "\"$p\"" --simple-io
    done
    for m in 15M 42M 110M; do
        wrap llama2c-$m 1 $R/ml/llama2.c /dev/null ./run ../models/stories$m.bin -t 0 -n 128 -i '"Once upon a time"'
    done
    echo "wrap $(wc -l < $D/list.txt) workloads" >> $D/done.txt
fi

if [[ $STEPS == *" prof "* ]]; then
    while read -r n _ t _ w; do
        want $n || continue
        o=$D/prof/runs/$n/1; mkdir -p $o
        win=0.1; [[ " $SHORT " == *" $n "* ]] && win=0.02
        OMP_NUM_THREADS=$t WRAP_PRELOAD=$SHIM MPR_NOTHP=1 MPR_MODE=refs MPR_SITES=1 MPR_SITE_MMAP=1 MPR_CHUNK=2097152 \
            MPR_WIN=$win MPR_OUT=$o/chunks.csv /usr/bin/time -o $o/chunks.time -f "%e %M" $w > /dev/null 2> $o/chunks.err
        echo "prof $n $? $(tail -1 $o/chunks.time)" >> $D/done.txt
    done < $D/list.txt
    cp $D/list.txt $D/prof/list.txt
fi
[[ $STEPS == *" plans "* ]] && { ~/environments/memfp/bin/python $HERE/place_lplans.py $D/prof $D/lplans > $D/lplans.log 2>&1 || exit 1; }

gcc -O2 -pthread -o $D/hog $HERE/cxl/hog.c && gcc -O2 -o $D/chase $HERE/cxl/chase.c || exit 1
lat() { echo "$($NUMACTL --cpunodebind=0 --membind=0 $D/chase | cut -d' ' -f1) $($NUMACTL --cpunodebind=0 --membind=1 $D/chase | cut -d' ' -f1)"; }
if [[ $STEPS == *" bw "* ]]; then
    for rep in 1 2; do
        for t in 0 1 2 3 4 6 8 10 12 16; do
            if [[ $t == 0 ]]; then echo "0 0 $(lat)" >> $D/bw.txt; continue; fi
            $NUMACTL --cpunodebind=1 --membind=1 $D/hog $t 8 15 > $D/hog.out & hp=$!
            sleep 3; l=$(lat); wait $hp
            echo "$t $(cat $D/hog.out) $l" >> $D/bw.txt
        done
    done
fi

if [[ $STEPS == *" runs "* ]]; then
    for rep in $(seq 1 $REPS); do
        for hog in 0 $H; do
            hp=; [[ $hog -gt 0 ]] && { $NUMACTL --cpunodebind=1 --membind=1 $D/hog $hog & hp=$!; sleep 2; }
            echo "lat $rep $hog $(lat)" >> $D/done.txt
            while read -r n _ t _ w; do
                want $n || continue
                export OMP_NUM_THREADS=$t; o=$D/runs/$n/$rep/$hog; mkdir -p $o
                for pol in $POLICIES; do
                    node=0; plan=
                    case $pol in remote) node=1 ;; local) ;; *) plan=$D/lplans/$n/$pol.txt ;; esac
                    [[ -n $plan && ! -f $plan ]] && { echo "$n $pol: no plan" >> $D/done.txt; continue; }
                    /usr/bin/time -o $o/$pol.time -f "%e %M" perf stat -x, -o $o/$pol.perf -e $EV \
                        $NUMACTL --cpunodebind=0 --membind=$node env WRAP_PRELOAD=$LIB MPR_MIN=262144 MPR_NOTHP=1 \
                        ${plan:+MPR_PLACE=$plan} $w 2> $o/$pol.err | grep -viE "time|sec|elapsed|ms|t/s|rate|speed" \
                        | md5sum | cut -c1-12 > $o/$pol.hash
                    echo "$n $rep $hog $pol ${PIPESTATUS[0]} $(tail -1 $o/$pol.time) $(cat $o/$pol.hash)" >> $D/done.txt
                done
            done < $D/list.txt
            [[ -n $hp ]] && { kill $hp; wait $hp 2>/dev/null; }
        done
    done
fi
echo all-done
