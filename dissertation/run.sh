#!/bin/bash
# The runs of dissertation/PROTOCOL.md, one at a time, into $OUT.
#   dissertation/run.sh [part ...]     parts: m4 m5 m3 m3mv m6 r1 m7 (default: m4 m5 m3 r1 m6; m3mv after M4's build;
#                                      m7 alone, on an idle machine)
#   m4    $OUT/m4/{traces,results}/miniVite        miniVite splitter, bytes, allocator excluded
#   m5    $OUT/m5/{traces,results}/{durbin,jacobi-1d}
#   m3    $OUT/m3/<sampler>/<seed>/<workload>/<config>/<k>/   samplers boot and disj on $OUT/m3_list.txt
#   m3mv  the same for miniVite from $OUT/m3mv_list.txt
#   m6    $OUT/m6/<workload>-<config>.{massif,dhat}
#   r1    $OUT/r1/<workload>/<config>/<rep>/pss1.{csv,csv.hwm,time}
#   m7    $OUT/m7/timing.csv
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$HERE/..
source "$ROOT/scripts/lib.sh"
export PIN_ROOT=${PIN_ROOT:-/home/nmust004/memory_estimator/pin/pin-3.30-98830-g1d7b601b3-gcc-linux}
OUT=${OUT:-$HOME/memprint-data/2026-11-05-diss-ch2}; mkdir -p "$OUT"
TOOL=$OUT/memprint_trace.so; cp "$ROOT/pintool/obj-intel64/memprint_trace.so" "$TOOL"   # frozen copy
OLD=$PIN_ROOT/source/tools/ManualExamples/obj-intel64/buffer_memtrace_poissonBS.so          # the published sampler
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive

build_of() {   # build_of <workload> <bench> <config>: sets BIN, ARGS and NAME
    load_workload "$1"
    (cd "$SRC" && wl_build "$2" "$3") > /dev/null 2>&1 || return 1
    read -r -a ARGS <<< "$(wl_args "$2" "$3")"
    BIN=$(cd "$SRC" && wl_binary "$2" "$3"); NAME=$(wl_name "$2" "$3"); [[ -x $BIN ]]; }

part_m4() {
    TRACE_DIR=$OUT/m4/traces RESULTS_DIR=$OUT/m4/results MEMPRINT_TOOL=$TOOL \
        "$ROOT/scripts/run.sh" minivite --mode splitter --footprint bytes --exclude-alloc --runs 1; }

part_m5() {
    TRACE_DIR=$OUT/m5/traces RESULTS_DIR=$OUT/m5/results MEMPRINT_TOOL=$TOOL \
        "$ROOT/scripts/run.sh" polybench --bench "durbin jacobi-1d" --mode splitter --footprint bytes --runs 1; }

sample() {   # sample <list> <workload> <extra knobs...>
    local list=$1 wl=$2; shift 2
    while read -r w split config k; do
        build_of "$wl" "$w" "$config" || { log "$w $config: build failed"; continue; }
        for seed in 1 2 3; do
            for s in boot disj; do
                o=$OUT/m3/$s/$seed/$w/$config/$k; [[ -d $o ]] && continue; mkdir -p "$o"
                if [[ $s == boot ]]; then knobs=(-i $((k / 10)) -s 100 -r 10)
                else b=20; ((k % 20)) && b=25; knobs=(-i $((k / b)) -s $b -disjoint 1); fi
                "$PIN_ROOT/pin" -t "$TOOL" -mode sampler "${knobs[@]}" -footprint bytes -seed "$seed" "$@" \
                    -outdir "$o" -name "$NAME" -- "$BIN" "${ARGS[@]}" > /dev/null 2> "$o/err"
                echo "$o $?" >> "$OUT/m3/done.txt"
            done
        done
    done < "$list"; }

part_m3() { sample "$OUT/m3_list.txt" polybench; }
part_m3mv() { sample "$OUT/m3mv_list.txt" minivite -exclude_alloc 1; }

part_m6() {
    mkdir -p "$OUT/m6"
    for c in 1024 2048 4096 8192 16384; do build_of minivite miniVite $c && vg miniVite-$c "$BIN" "${ARGS[@]}"; done
    for c in MINI MINI2 MINI3 SMALL SMALL2 SMALL3 MEDIUM; do build_of polybench gemm $c && vg gemm-$c "$BIN"; done; }
vg() { local n=$1; shift
    valgrind --tool=massif --massif-out-file="$OUT/m6/$n.massif" "$@" > /dev/null 2> "$OUT/m6/$n.massif.err"
    valgrind --tool=dhat --dhat-out-file="$OUT/m6/$n.dhat" "$@" > /dev/null 2> "$OUT/m6/$n.dhat.err"; }

part_r1() {
    gcc -O2 -shared -fPIC -o "$OUT/mprshim.so" "$ROOT/paper_rss/tools/mprshim.c" -lpthread -ldl || return 1
    for rep in 1 2 3; do
        for w in $(grep -v miniVite "$ROOT/analysis/workloads.txt") miniVite; do
            wl=polybench cfgs="MINI MINI2 MINI3 SMALL SMALL2 SMALL3 MEDIUM"
            [[ $w == miniVite ]] && wl=minivite cfgs="1024 2048 4096 8192 16384"
            for c in $cfgs; do
                build_of $wl "$w" $c || continue
                o=$OUT/r1/$w/$c/$rep; mkdir -p "$o"
                (cd "$o" && /usr/bin/time -o pss1.time -f "%e %M" env LD_PRELOAD="$OUT/mprshim.so" MPR_MODE=pss \
                    MPR_INT=0.001 MPR_OUT="$o/pss1.csv" "$BIN" "${ARGS[@]}" > /dev/null 2> pss1.err)
                echo "$o $?" >> "$OUT/r1/done.txt"
            done
        done
    done; }

part_m7() {   # interleaved, 5 rounds
    mkdir -p "$OUT/m7/x"; build_of minivite miniVite 8192 || return 1; bin=$BIN; f=$OUT/m7/timing.csv
    [[ -f $f ]] || echo "round,config,seconds" > "$f"
    for round in 1 2 3 4 5; do
        t() { local c=$1; shift; echo "$round,$c,$(time_cmd "$@")" >> "$f"; }
        t native "$bin" "${ARGS[@]}"
        t instr "$PIN_ROOT/pin" -t "$TOOL" -mode sampler -i 4294967295 -s 1 -r 1 -outdir "$OUT/m7/x" -- "$bin" "${ARGS[@]}"
        for i in 500 1000 2500; do
            t published-$i "$PIN_ROOT/pin" -t "$OLD" -i $i -s 50 -r 20 -o "$OUT/m7/x/old" -- "$bin" "${ARGS[@]}"
            t boot-$i "$PIN_ROOT/pin" -t "$TOOL" -mode sampler -i $i -s 50 -r 20 -outdir "$OUT/m7/x" -- "$bin" "${ARGS[@]}"
            t disj-$i "$PIN_ROOT/pin" -t "$TOOL" -mode sampler -i $i -s 50 -disjoint 1 -outdir "$OUT/m7/x" -- "$bin" "${ARGS[@]}"
        done
        t splitter "$PIN_ROOT/pin" -t "$TOOL" -mode splitter -outdir "$OUT/m7/x" -- "$bin" "${ARGS[@]}"
    done; }

part_m7b() {   # M7 at matched bin intervals k (M2: the published label i*s/r is not the bin interval), 5 rounds
    mkdir -p "$OUT/m7/x"; build_of minivite miniVite 8192 || return 1; bin=$BIN; f=$OUT/m7/timing_k.csv
    [[ -f $f ]] || echo "round,config,seconds" > "$f"
    for round in 1 2 3 4 5; do
        t() { local c=$1; shift; echo "$round,$c,$(time_cmd "$@")" >> "$f"; }
        t native "$bin" "${ARGS[@]}"
        for k in 500 1000 2500; do   # bin interval about i*r for the published sampler and boot, i*s for disj
            t published-$k "$PIN_ROOT/pin" -t "$OLD" -i $((k / 10)) -s 100 -r 10 -o "$OUT/m7/x/old" -- "$bin" "${ARGS[@]}"
            t published50-$k "$PIN_ROOT/pin" -t "$OLD" -i $((k / 5)) -s 50 -r 5 -o "$OUT/m7/x/old" -- "$bin" "${ARGS[@]}"
            t boot-$k "$PIN_ROOT/pin" -t "$TOOL" -mode sampler -i $((k / 10)) -s 100 -r 10 -outdir "$OUT/m7/x" -- "$bin" "${ARGS[@]}"
            t disj-$k "$PIN_ROOT/pin" -t "$TOOL" -mode sampler -i $((k / 20)) -s 20 -disjoint 1 -outdir "$OUT/m7/x" -- "$bin" "${ARGS[@]}"
        done
    done; }

part_m9() {   # the truth run (one interval, one bin: the full bytes-touched trace) against the samplers, 5 rounds
    mkdir -p "$OUT/m9/x"; f=$OUT/m9/timing.csv
    [[ -f $f ]] || echo "round,workload,config,seconds" > "$f"
    for round in 1 2 3 4 5; do
        for wc in "minivite miniVite 8192" "polybench gemm MEDIUM" "polybench 2mm MEDIUM" "polybench atax MEDIUM" \
                  "polybench jacobi-2d MEDIUM"; do
            read -r wl w c <<< "$wc"; build_of $wl $w $c || continue
            ex=(-footprint bytes); [[ $w == miniVite ]] && ex+=(-exclude_alloc 1)
            t() { local n=$1; shift; echo "$round,$w,$n,$(time_cmd "$@")" >> "$f"; }
            p=("$PIN_ROOT/pin" -t "$TOOL")
            t native "$BIN" "${ARGS[@]}"
            t instr "${p[@]}" -mode sampler -i 4294967295 -s 1 -r 1 -outdir "$OUT/m9/x" -- "$BIN" "${ARGS[@]}"
            t truth "${p[@]}" -mode splitter -intervals 1 -bins 1 "${ex[@]}" -outdir "$OUT/m9/x" -- "$BIN" "${ARGS[@]}"
            t boot-1000 "${p[@]}" -mode sampler -i 100 -s 100 -r 10 "${ex[@]}" -outdir "$OUT/m9/x" -- "$BIN" "${ARGS[@]}"
            t disj-1000 "${p[@]}" -mode sampler -i 50 -s 20 -disjoint 1 "${ex[@]}" -outdir "$OUT/m9/x" -- "$BIN" "${ARGS[@]}"
            t splitter "${p[@]}" -mode splitter "${ex[@]}" -outdir "$OUT/m9/x" -- "$BIN" "${ARGS[@]}"
            rm -f "$OUT"/m9/x/*.csv
        done
    done; }

part_r8() {   # R1 again under the event sampler (mprshim-event.so: smaps_rollup after page faults, before frees, at exit)
    gcc -O2 -shared -fPIC -DMPR_EVENT_FREE -o "$OUT/mprshim-event.so" "$ROOT/paper_rss/tools/mprshim.c" -lpthread -ldl || return 1
    for rep in 1 2 3; do
        for w in $(grep -v miniVite "$ROOT/analysis/workloads.txt") miniVite; do
            wl=polybench cfgs="MINI MINI2 MINI3 SMALL SMALL2 SMALL3 MEDIUM"
            [[ $w == miniVite ]] && wl=minivite cfgs="1024 2048 4096 8192 16384"
            for c in $cfgs; do
                build_of $wl "$w" $c || continue
                o=$OUT/r1/$w/$c/$rep; mkdir -p "$o"
                (cd "$o" && /usr/bin/time -o event.time -f "%e %M" env LD_PRELOAD="$OUT/mprshim-event.so" MPR_MODE=event \
                    MPR_OUT="$o/event.csv" "$BIN" "${ARGS[@]}" > /dev/null 2> event.err)
                echo "$o event $?" >> "$OUT/r1/done.txt"
            done
        done
    done; }

for part in ${@:-m4 m5 m3 r1 m6}; do echo "== $part $(date)"; part_$part; done
echo all-done
