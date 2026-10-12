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

echo "pass"
LLVM_BIN=${LLVM_BIN:-$HOME/llvm-project/build/bin}
cmake -S "$HERE/../pass" -B "$WORK/pass" -DLLVM_DIR="$(dirname "$LLVM_BIN")/lib/cmake/llvm" -DCMAKE_BUILD_TYPE=Release \
    > "$WORK/cmake.log" 2>&1 && cmake --build "$WORK/pass" >> "$WORK/cmake.log" 2>&1 || { cat "$WORK/cmake.log"; exit 1; }
PLUGIN="$WORK/pass/libMemPrintResize.so"
MPR_PLUGIN="$PLUGIN" LLVM_BIN="$LLVM_BIN" "$LLVM_BIN/llvm-lit" -q "$HERE/pass" && echo "    ok   lit: allocation calls and vector growth rewritten" || failed=1

# a program built with the pass: correct, and with a capacity reservation its vector grows once
GCC8="--gcc-install-dir=/usr/lib/gcc/x86_64-redhat-linux/8"
MPR_PASS_SITEMAP="$WORK/sitemap.txt" "$LLVM_BIN/clang++" $GCC8 -O2 -g -fpass-plugin="$PLUGIN" \
    -o "$WORK/vec" "$HERE/../../tests/pintool/sites_vec.cpp" -L"$WORK" -lmpr -Wl,-rpath,"$WORK" 2> /dev/null || failed=1
cap=$(awk '$2 == "capacity" {print $1; exit}' "$WORK/sitemap.txt")
printf "id:%s const 131072 0\n" "$cap" > "$WORK/vec_res.txt"
MPR_ALL=1 MPR_REPORT="$WORK/vec0.csv" "$WORK/vec" && MPR_ALL=1 MPR_RESERVATIONS="$WORK/vec_res.txt" \
    MPR_REPORT="$WORK/vec1.csv" "$WORK/vec" || { echo "    FAIL the program built with the pass"; failed=1; }
n0=$(awk -F, 'NR == 2 {print $3}' "$WORK/vec0.csv"); n1=$(awk -F, 'NR == 2 {print $3}' "$WORK/vec1.csv")
if [[ $n0 -gt 1 && $n1 == 1 ]]; then
    echo "    ok   vector growth: $n0 large blocks, 1 with a capacity reservation; result unchanged"
else
    echo "    FAIL vector growth: $n0 then $n1 large blocks"; failed=1
fi

# the Pin tool names the sites of a program built with the pass by the pass's ids
if [[ -n "${PIN_ROOT:-}" && -f "$HERE/../../pintool/obj-intel64/memprint_trace.so" ]]; then
    "$PIN_ROOT/pin" -t "$HERE/../../pintool/obj-intel64/memprint_trace.so" -mode splitter -track_frees 1 -sites 1 \
        -outdir "$WORK/pin" -name vec -- "$WORK/vec" > /dev/null 2>&1
    alloc=$(awk '$2 == "_Znwm" {print $1; exit}' "$WORK/sitemap.txt")
    if grep -q "^id:$alloc," "$WORK"/pin/*_sites.csv; then
        echo "    ok   Pin -sites names the vector's site id:$alloc"
    else
        echo "    FAIL Pin -sites: no site id:$alloc"; cat "$WORK"/pin/*_sites.csv; failed=1
    fi
fi

[[ $failed == 0 ]] && echo "all resize tests passed" || { echo "some resize tests FAILED"; exit 1; }
