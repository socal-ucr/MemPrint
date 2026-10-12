#!/bin/bash
# libmpr runtime and MemPrintResize pass tests:  resize/tests/run_tests.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
failed=0

echo "runtime"
gcc -O2 -g -Wall -fPIC -shared -o "$WORK/libmpr.so" "$HERE/../runtime/mpr.c" -ldl -lpthread || exit 1
gcc -O1 -g -o "$WORK/test_runtime" "$HERE/test_runtime.c" -L"$WORK" -lmpr -lpthread || exit 1
printf 'id:0x1 dead 0 0\nid:0x2 ratio 0.25 0\nid:0x3 const 8192 0\n' > "$WORK/reservations.txt"
LD_LIBRARY_PATH="$WORK" MPR_RESERVATIONS="$WORK/reservations.txt" MPR_ALL=1 MPR_REPORT="$WORK/report.csv" \
    "$WORK/test_runtime" || failed=1
if grep -q '^id:0x1,dead,1,1048576,0,1048576,1,' "$WORK/report.csv"; then
    echo "    ok   telemetry: the dead site's block was all resident, an under-prediction"
else
    echo "    FAIL telemetry:"; cat "$WORK/report.csv"; failed=1
fi

[[ $failed == 0 ]] && echo "all resize tests passed" || { echo "some resize tests FAILED"; exit 1; }
