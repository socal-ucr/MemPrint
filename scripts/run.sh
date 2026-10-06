#!/bin/bash
# Run a workload natively and under the MemPrint Pin tool, recording traces
# and runtime overheads.
#
#   scripts/run.sh <workload> --mode MODE [options]
#
# Modes:
#   splitter  full trace with memprint_trace -mode splitter (training traces)
#   sampler   memprint_trace -mode sampler -i/-s/-r (prediction traces)
#   spatial   memprint_trace -mode spatial -i/-s: every access to 1-in-i addresses
#   instr     instrumentation only (sampler that never samples), for overhead
#   native    no Pin, timing only
#
# Options:
#   --bench "b1 b2 ..."     benchmarks to run (default: all of the workload's)
#   --configs "c1 c2 ..."   input configurations (default: workload's DEFAULT_CONFIGS)
#   --runs N                native (and massif) repetitions per config (default 5)
#   --pin-runs N            Pin repetitions per config (default 1 for splitter, else --runs)
#   --massif                also run valgrind massif and record the peak heap
#   -i N -s N -r N          sampler knobs (default 1000, 100, 10); spatial uses -i and -s
#   --intervals LIST        splitter sampling intervals (comma-separated)
#   --bins N                splitter bins per interval
#   --snapshot N            also write a footprint timeline every N memory references
#   --track-frees           remove freed (free/realloc/munmap) memory from the footprints
#   --stop N                stop tracing after N memory references (partial trace)
#
# Outputs:
#   $TRACE_DIR/<bench>/              trace CSVs (and *_timeline.csv) written by the Pin tool
#   $RESULTS_DIR/<bench>/overhead_<mode>.csv
#   $RESULTS_DIR/<bench>/massif.csv  (with --massif)

source "$(dirname "$0")/lib.sh"

usage() { sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

[[ $# -gt 0 && $1 != -* ]] || usage
load_workload "$1"; shift

MODE= BENCHES= CONFIGS= RUNS=5 PIN_RUNS= MASSIF=0
INTERVAL=1000 SPLITS=100 REPS=10 SPLIT_KNOBS=() EXTRA_KNOBS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE=$2; shift ;;
        --bench) BENCHES=$2; shift ;;
        --configs) CONFIGS=$2; shift ;;
        --runs) RUNS=$2; shift ;;
        --pin-runs) PIN_RUNS=$2; shift ;;
        --massif) MASSIF=1 ;;
        -i) INTERVAL=$2; shift ;;
        -s) SPLITS=$2; shift ;;
        -r) REPS=$2; shift ;;
        --intervals) SPLIT_KNOBS+=(-intervals "$2"); shift ;;
        --bins) SPLIT_KNOBS+=(-bins "$2"); shift ;;
        --snapshot) EXTRA_KNOBS+=(-snapshot "$2"); shift ;;
        --track-frees) EXTRA_KNOBS+=(-track_frees 1) ;;
        --stop) EXTRA_KNOBS+=(-stop "$2"); shift ;;
        -h|--help) usage ;;
        *) die "unknown option $1" ;;
    esac
    shift
done

case $MODE in
    splitter) PIN_KNOBS=(-mode splitter "${SPLIT_KNOBS[@]}"); INTERVAL=1 SPLITS= REPS=; : "${PIN_RUNS:=1}" ;;
    sampler)  PIN_KNOBS=(-mode sampler -i "$INTERVAL" -s "$SPLITS" -r "$REPS") ;;
    spatial)  REPS= PIN_KNOBS=(-mode spatial -i "$INTERVAL" -s "$SPLITS") ;;
    # Largest -i: the sampling test effectively never fires, so only the
    # instrumentation cost is measured.
    instr)    INTERVAL=4294967295 SPLITS=1 REPS=1
              PIN_KNOBS=(-mode sampler -i "$INTERVAL" -s "$SPLITS" -r "$REPS") ;;
    native)   PIN_KNOBS=() INTERVAL= SPLITS= REPS=; PIN_RUNS=0 ;;
    *) die "--mode must be splitter, sampler, spatial, instr or native" ;;
esac
: "${PIN_RUNS:=$RUNS}"
[[ $MODE == native ]] || require_pin
[[ $MASSIF == 0 ]] || command -v ms_print > /dev/null || die "--massif needs valgrind (ms_print)"

[[ -n $BENCHES ]] || BENCHES=$(cd "$SRC" && wl_benchmarks)
[[ -n $CONFIGS ]] || CONFIGS="${DEFAULT_CONFIGS[*]}"

for bench in $BENCHES; do
    results="$RESULTS_DIR/$bench"
    traces="$TRACE_DIR/$bench"
    mkdir -p "$results" "$traces"

    overhead_csv="$results/overhead_$MODE.csv"
    [[ -f $overhead_csv ]] ||
        echo "Workload,Config,Mode,SamplingInterval,Splits,Replication,NativeRuns,PinRuns,Native(s),Pin(s),PinOverhead(s),Massif(s),MassifOverhead(s)" > "$overhead_csv"
    massif_csv="$results/massif.csv"
    [[ $MASSIF == 0 || -f $massif_csv ]] || echo "Workload,Size,PeakMemory" > "$massif_csv"

    for config in $CONFIGS; do
        log "$bench $config: building"
        if ! (cd "$SRC" && wl_build "$bench" "$config") > /dev/null 2>&1; then
            log "$bench $config: build failed, skipping"
            continue
        fi
        binary=$(cd "$SRC" && wl_binary "$bench" "$config")
        [[ -x $binary ]] || { log "$bench $config: $binary not found, skipping"; continue; }
        read -r -a args <<< "$(wl_args "$bench" "$config")"
        name=$(wl_name "$bench" "$config")

        native=() pin=() massif=()
        for ((run = 1; run <= RUNS; run++)); do
            log "$bench $config: native run $run/$RUNS"
            native+=("$(time_cmd "$binary" "${args[@]}")")
            if [[ $MASSIF == 1 ]]; then
                log "$bench $config: massif run $run/$RUNS"
                massif+=("$(time_cmd valgrind --tool=massif --massif-out-file="$results/massif.out.$config" "$binary" "${args[@]}")")
            fi
        done
        for ((run = 1; run <= PIN_RUNS; run++)); do
            log "$bench $config: $MODE run $run/$PIN_RUNS"
            pin+=("$(time_cmd "$PIN_ROOT/pin" -t "$MEMPRINT_TOOL" "${PIN_KNOBS[@]}" "${EXTRA_KNOBS[@]}" -outdir "$traces" -name "$name" -- "$binary" "${args[@]}")")
        done

        native_s=$(mean "${native[@]}")
        pin_s= pin_overhead= massif_s= massif_overhead=
        if [[ ${#pin[@]} -gt 0 ]]; then
            pin_s=$(mean "${pin[@]}"); pin_overhead=$(minus "$pin_s" "$native_s")
        fi
        if [[ ${#massif[@]} -gt 0 ]]; then
            massif_s=$(mean "${massif[@]}"); massif_overhead=$(minus "$massif_s" "$native_s")
            echo "$bench,$config,$(massif_peak "$results/massif.out.$config")" >> "$massif_csv"
        fi
        echo "$bench,$config,$MODE,$INTERVAL,$SPLITS,$REPS,$RUNS,$PIN_RUNS,$native_s,$pin_s,$pin_overhead,$massif_s,$massif_overhead" >> "$overhead_csv"
        log "$bench $config: native ${native_s}s${pin_s:+, pin ${pin_s}s}${massif_s:+, massif ${massif_s}s}"
    done
done
