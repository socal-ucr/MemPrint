#!/bin/bash
# Deallocation and timeline tests for memprint_trace.
#   PIN_ROOT=/path/to/pin tests/pintool/run_tests.sh
# Builds the test programs, runs them under the splitter with -track_frees and
# -snapshot, and checks the live footprint over time against known values.

source "$(dirname "$0")/../../scripts/lib.sh"
require_pin
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

cc -O1 -o "$WORK/free" "$HERE/free.c"
cc -O1 -o "$WORK/realloc" "$HERE/realloc.c"
c++ -O1 -o "$WORK/calloc_new" "$HERE/calloc_new.cpp"
cc -O1 -o "$WORK/mmap" "$HERE/mmap.c"
cc -O1 -pthread -o "$WORK/threads" "$HERE/threads.c"
cc -O1 -o "$WORK/small_blocks" "$HERE/small_blocks.c"

failed=0
# run <test name> <program> <pin knobs> -- <check.py arguments>
run() {
    local name=$1 program=$2; shift 2
    local knobs=()
    while [[ $1 != -- ]]; do knobs+=("$1"); shift; done; shift
    echo "$name"
    if ! "$PIN_ROOT/pin" -t "$MEMPRINT_TOOL" -mode splitter -snapshot 5000 "${knobs[@]}" -outdir "$WORK/out/$name" \
            -name "$name-test" -- "$WORK/$program" > "$WORK/$name.out" 2>&1; then
        echo "    FAIL program exited with an error"; failed=1; return
    fi
    python3 "$HERE/check.py" "$WORK/out/$name"/*_timeline.csv "$@" || failed=1
}

run free free -track_frees 1 -- --peak 2 --freed 3
run free-untracked free -track_frees 0 -- --peak 3 --final-max 3.5
run realloc realloc -track_frees 1 -- --peak 4 --freed 4.0625
run calloc-new calloc_new -track_frees 1 -- --peak 1.5 --freed 1.5 --slack 0.75  # libstdc++ keeps its own pools
run mmap mmap -track_frees 1 -- --peak 4 --freed 4
run threads threads -track_frees 1 -- --peak 2 --freed 2
run small-blocks small_blocks -track_frees 1 -- --peak 0.125 --freed 2

# -stop: outputs are written after 200000 references and the program finishes natively.
echo "stop"
"$PIN_ROOT/pin" -t "$MEMPRINT_TOOL" -mode splitter -snapshot 20000 -stop 200000 -outdir "$WORK/out/stop" -name stop-test \
    -- "$WORK/free" > /dev/null 2>&1
status=$?
last=$(tail -1 "$WORK"/out/stop/*_timeline.csv 2>/dev/null | cut -d, -f1)
if [[ $status == 0 && $(ls "$WORK/out/stop" | wc -l) -gt 2 && $last -ge 200000 && $last -lt 220000 ]]; then
    echo "    ok   exit 0, outputs written, last snapshot at $last"
else
    echo "    FAIL status=$status last snapshot=$last"; failed=1
fi

[[ $failed == 0 ]] && echo "all tests passed" || { echo "some tests FAILED"; exit 1; }
