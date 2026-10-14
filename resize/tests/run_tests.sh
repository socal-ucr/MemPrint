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
printf 'id:0x1 dead 0 0\nid:0x2 ratio 0.25 0\nid:0x3 const 8192 0\nid:0x4 runs 2 0 0.25 0.5 0.75\nid:0x5 periodic 4 1024 0 tokens 1\nid:0x6 ranges 2 prompt 1 0 1024 0 0.5 0 8192\n' > "$WORK/reservations.txt"
LD_LIBRARY_PATH="$WORK" MPR_RESERVATIONS="$WORK/reservations.txt" MPR_PARAMS=steps=3,tokens=10,prompt=20 MPR_ALL=1 MPR_ZERO_MIN=16384 MPR_REPORT="$WORK/report.csv" \
    "$WORK/test_runtime" || failed=1
if grep -q '^id:0x1,dead,1,1048576,0,1048576,1,' "$WORK/report.csv"; then
    echo "    ok   telemetry: the dead site's block was all resident, an under-prediction"
else
    echo "    FAIL telemetry:"; cat "$WORK/report.csv"; failed=1
fi

# MPR_PROFILE: libmpr profiles the program at native speed (tests/pintool/sites.c's known sizes)
cc -O1 -g -o "$WORK/sites" "$HERE/../../tests/pintool/sites.c"
LD_PRELOAD="$WORK/libmpr.so" MPR_PROFILE="$WORK/prof" "$WORK/sites" || failed=1
got=$(tail -n +2 "$WORK"/prof/*_blocks.csv | awk -F, '{print $3, $6, $7}' | sort | tr '\n' ';')
want="1048576 0 0;1048576 1048576 1048576;1048576 262144 262144;524288 65536 524288;65536 4096 4096;65536 65536 65536;"
if [[ $got == "$want" ]] && ls "$WORK"/prof/*_images.csv > /dev/null; then
    echo "    ok   MPR_PROFILE: dead, prefix, tail, calloc and realloc blocks (size, touched, high water) as Pin measures them"
else
    echo "    FAIL MPR_PROFILE: $got"; failed=1
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
# (the growth point is inlined into its caller: its id is the call site's; 1 MB of longs)
cap=$(awk '$2 == "capacity-site" {print $1; exit}' "$WORK/sitemap.txt")
printf "id:%s cap 1048576 0\n" "$cap" > "$WORK/vec_res.txt"
MPR_ALL=1 MPR_REPORT="$WORK/vec0.csv" "$WORK/vec" && MPR_ALL=1 MPR_RESERVATIONS="$WORK/vec_res.txt" \
    MPR_REPORT="$WORK/vec1.csv" "$WORK/vec" || { echo "    FAIL the program built with the pass"; failed=1; }
n0=$(awk -F, 'NR == 2 {print $3}' "$WORK/vec0.csv"); n1=$(awk -F, 'NR == 2 {print $3}' "$WORK/vec1.csv")
if [[ $n0 -gt 1 && $n1 == 1 ]]; then
    echo "    ok   vector growth: $n0 large blocks, 1 with a capacity reservation; result unchanged"
else
    echo "    FAIL vector growth: $n0 then $n1 large blocks"; failed=1
fi

# containers: element sizes from the debug information, one capacity per call site (the vector's
# growth point is inlined into each caller), growth contexts around out-of-line string and hash
# table growth
MPR_PASS_SITEMAP="$WORK/cap_sitemap.txt" "$LLVM_BIN/clang++" $GCC8 -O2 -g -fpass-plugin="$PLUGIN" \
    -o "$WORK/cap" "$HERE/../../tests/pintool/sites_cap.cpp" -L"$WORK" -lmpr -Wl,-rpath,"$WORK" 2> /dev/null || failed=1
e8=$(grep -c "capacity elem=8" "$WORK/cap_sitemap.txt"); e1=$(grep -c "capacity elem=1" "$WORK/cap_sitemap.txt")
cs=$(grep -c " capacity-site " "$WORK/cap_sitemap.txt"); gc=$(grep -c " growth " "$WORK/cap_sitemap.txt")
if [[ $e8 -ge 2 && $e1 -ge 1 && $cs -ge 2 && $gc -ge 2 ]]; then
    echo "    ok   growth points: element sizes 8 and 1 from debug info, $cs inlined call sites, $gc growth contexts"
else
    echo "    FAIL growth points: elem=8 $e8, elem=1 $e1, call sites $cs, contexts $gc"; failed=1
fi

# site ids do not move when lines are added above a function or elsewhere in the file, or when the
# file is built in another directory
mkdir -p "$WORK/moved"
sed '1i // a comment line\n// and another' "$HERE/../../tests/pintool/sites_cap.cpp" |
    sed 's|^__attribute__((noinline)) long text()|// moved down\n&|' > "$WORK/moved/sites_cap.cpp"
MPR_PASS_SITEMAP="$WORK/moved_sitemap.txt" "$LLVM_BIN/clang++" $GCC8 -O2 -g -fpass-plugin="$PLUGIN" \
    -o "$WORK/cap_moved" "$WORK/moved/sites_cap.cpp" -L"$WORK" -lmpr -Wl,-rpath,"$WORK" 2> /dev/null || failed=1
if diff <(awk '{print $1}' "$WORK/cap_sitemap.txt" | sort) <(awk '{print $1}' "$WORK/moved_sitemap.txt" | sort) > /dev/null; then
    echo "    ok   site ids unchanged after inserting lines ($(wc -l < "$WORK/cap_sitemap.txt") sites)"
else
    echo "    FAIL site ids changed after inserting lines"; failed=1
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
    # an allocation wrapper with two callers (one writes its block, one does not) and a buffer
    # cleared with memset: profiled, fitted per calling context, served by libmpr preloaded
    PY=${MPR_PY:-$HOME/environments/memfp/bin/python}
    cc -O1 -g -I"$HERE/../../tests/pintool" -o "$WORK/ctx" "$HERE/../../tests/pintool/sites_ctx.c"
    "$PIN_ROOT/pin" -t "$HERE/../../pintool/obj-intel64/memprint_trace.so" -mode splitter -track_frees 1 -sites 1 \
        -sites_block_min 4096 -outdir "$WORK/pinctx" -name ctx -- "$WORK/ctx" > /dev/null 2>&1
    (cd "$HERE/../../analysis" && "$PY" -m memprint.sites "$WORK/pinctx" -o "$WORK/ctx_res.txt" --min 4096 > /dev/null)
    LD_PRELOAD="$WORK/libmpr.so" MPR_RESERVATIONS="$WORK/ctx_res.txt" MPR_MIN=4096 MPR_REPORT="$WORK/ctx.csv" "$WORK/ctx"
    # expected: the hot caller's block committed (1 MB), the cold caller's not (dead), the cleared
    # block 64 KB with its 1 MB of zeroing skipped
    if awk -F, 'NR > 1 && $1 ~ /@/ && $2 == "dead" {d++} NR > 1 && $1 ~ /@/ && $5 == 1048576 {h++}
                NR > 1 && $5 == 65536 && $9 == 1048576 {z++} END {exit !(d == 1 && h == 1 && z == 1)}' "$WORK/ctx.csv"; then
        echo "    ok   wrapper split by calling context (1 MB and dead), memset zeroing skipped (64 KB committed)"
    else
        echo "    FAIL calling contexts / zeroing:"; cat "$WORK/ctx_res.txt" "$WORK/ctx.csv"; failed=1
    fi
    # address keys are tied to the build they were profiled on: another build of the program (other
    # flags, other build-id) has its address sites ignored, with a warning
    mkdir -p "$WORK/rebuilt"
    cc -O2 -g -I"$HERE/../../tests/pintool" -o "$WORK/rebuilt/ctx" "$HERE/../../tests/pintool/sites_ctx.c"
    warn=$(LD_PRELOAD="$WORK/libmpr.so" MPR_RESERVATIONS="$WORK/ctx_res.txt" MPR_MIN=4096 MPR_REPORT="$WORK/ctx2.csv" \
        "$WORK/rebuilt/ctx" 2>&1 >/dev/null)
    if grep -q "# image ctx build-id" "$WORK/ctx_res.txt" && [[ $warn == *"not the build"* ]] &&
       [[ $(awk 'END {print NR}' "$WORK/ctx2.csv") -le 1 ]]; then
        echo "    ok   reservations of another build: address sites ignored ($warn)" | cut -c1-150
    else
        echo "    FAIL build check: '$warn'"; cat "$WORK/ctx2.csv"; failed=1
    fi
    # the same, with the wrapper in a shared library the program's build does not compile
    cc -O1 -g -fPIC -shared -o "$WORK/libwrap.so" "$HERE/../../tests/pintool/libwrap.c"
    cc -O1 -g -I"$HERE/../../tests/pintool" -o "$WORK/lib" "$HERE/../../tests/pintool/sites_lib.c" -L"$WORK" -lwrap \
        -Wl,-rpath,"$WORK"
    "$PIN_ROOT/pin" -t "$HERE/../../pintool/obj-intel64/memprint_trace.so" -mode splitter -track_frees 1 -sites 1 \
        -sites_block_min 4096 -outdir "$WORK/pinlib" -name lib -- "$WORK/lib" > /dev/null 2>&1
    (cd "$HERE/../../analysis" && "$PY" -m memprint.sites "$WORK/pinlib" -o "$WORK/lib_res.txt" --min 4096 > /dev/null)
    LD_PRELOAD="$WORK/libmpr.so" MPR_RESERVATIONS="$WORK/lib_res.txt" MPR_MIN=4096 MPR_REPORT="$WORK/lib.csv" "$WORK/lib"
    if awk -F, 'NR > 1 && $1 ~ /^libwrap.so\+.*@lib\+/ && $2 == "dead" {d++} NR > 1 && $1 ~ /^libwrap.so\+.*@lib\+/ && $5 == 1048576 {h++}
                END {exit !(d == 1 && h == 1)}' "$WORK/lib.csv"; then
        echo "    ok   wrapper in a shared library split by the program's calling contexts"
    else
        echo "    FAIL library wrapper:"; cat "$WORK/lib_res.txt" "$WORK/lib.csv"; failed=1
    fi
    # a block used in ranges (8 layers, the first 64 KB of each written): a shaped reservation
    cc -O1 -g -I"$HERE/../../tests/pintool" -o "$WORK/strided" "$HERE/../../tests/pintool/sites_strided.c"
    "$PIN_ROOT/pin" -t "$HERE/../../pintool/obj-intel64/memprint_trace.so" -mode splitter -track_frees 1 -sites 1 \
        -outdir "$WORK/pinstrided" -name strided -- "$WORK/strided" > /dev/null 2>&1
    (cd "$HERE/../../analysis" && "$PY" -m memprint.sites "$WORK/pinstrided" -o "$WORK/strided_res.txt" > /dev/null)
    LD_PRELOAD="$WORK/libmpr.so" MPR_RESERVATIONS="$WORK/strided_res.txt" MPR_REPORT="$WORK/strided.csv" "$WORK/strided"
    status=$?
    if [[ $status == 0 ]] && grep -q " runs 8 " "$WORK/strided_res.txt" &&
       awk -F, 'NR > 1 && $2 == "runs" && $5 >= 524288 && $5 <= 655360 {ok = 1; c = $5} END {exit !ok}' "$WORK/strided.csv"; then
        # (17 pages per layer: the profiled block started 16 bytes into a page)
        echo "    ok   shaped reservation: 8 ranges, $(awk -F, 'NR > 1 {print $5 / 1024}' "$WORK/strided.csv") KB of 8 MB committed (a prefix: 7.1 MB)"
    else
        echo "    FAIL shaped reservation (status $status):"; cat "$WORK/strided_res.txt" "$WORK/strided.csv"; failed=1
    fi
    # capacities profiled per call site: the big vector and the string are allocated once; the 2000
    # small vectors at another call site keep their own growth
    "$PIN_ROOT/pin" -t "$HERE/../../pintool/obj-intel64/memprint_trace.so" -mode splitter -track_frees 1 -sites 1 \
        -outdir "$WORK/pincap" -name cap -- "$WORK/cap" > /dev/null 2>&1
    (cd "$HERE/../../analysis" && "$PY" -m memprint.sites "$WORK/pincap" -o "$WORK/cap_res.txt" > /dev/null)
    out0=$(MPR_ALL=1 MPR_REPORT="$WORK/cap0.csv" "$WORK/cap"); out1=$(MPR_ALL=1 MPR_RESERVATIONS="$WORK/cap_res.txt" MPR_REPORT="$WORK/cap1.csv" "$WORK/cap")
    n0=$(awk -F, 'NR > 1 {n += $3} END {print n}' "$WORK/cap0.csv"); n1=$(awk -F, 'NR > 1 {n += $3} END {print n}' "$WORK/cap1.csv")
    caps=$(grep -c " cap " "$WORK/cap_res.txt")
    # (the hash table's bucket array is left out by default: --hash-capacities)
    if [[ "$out0" == "$out1" && $caps -eq 2 && $n1 -lt $n0 ]]; then
        echo "    ok   container capacities: $caps call sites, $n0 -> $n1 large blocks, same output"
    else
        echo "    FAIL container capacities: $caps call sites, $n0 -> $n1 large blocks; output '$out0' / '$out1'"
        cat "$WORK/cap_res.txt"; failed=1
    fi
fi

# sites.py --param: blocks of 4 equal periods, each touched in a prefix of 1 KB per token plus a page, at three
# token counts -> "periodic 4 1024 4096 tokens 64"
echo "fitting"
PYF=${PY:-$HOME/environments/memfp/bin/python}
for t in 8 16 64; do
    mkdir -p "$WORK/per$t"; echo "tokens=$t" > "$WORK/per$t/params.txt"
    pages=$(( (t * 1024 + 4096 + 4095) / 4096 ))
    runs=$(for i in 0 1 2 3; do printf '%s%d:%d' "$([[ $i -gt 0 ]] && echo ';')" $((i * 64)) $pages; done)
    printf 'site,kind,size,born,died,touched,high_water,pages_touched,pages,alloc_touched,grown_from,moved,context,runs\nid:0x9,1,1048576,0,1,%d,%d,%d,256,0,0,0,,%s\n' \
        $((pages * 4 * 4096)) $((192 * 4096 + pages * 4096)) $((pages * 4)) "$runs" > "$WORK/per$t/T_blocks.csv"
done
(cd "$HERE/../../analysis" && "$PYF" -m memprint.sites "$WORK/per8" "$WORK/per16" "$WORK/per64" -o "$WORK/per.txt" --param tokens > /dev/null)
if grep -q '^id:0x9 periodic 4 1024 4096 tokens 64$' "$WORK/per.txt"; then
    echo "    ok   periodic fit: 4 periods, 1024 bytes per token plus a page"
else
    echo "    FAIL periodic fit:"; cat "$WORK/per.txt"; failed=1
fi

[[ $failed == 0 ]] && echo "all resize tests passed" || { echo "some resize tests FAILED"; exit 1; }
