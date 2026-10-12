#!/bin/bash
# Build a program with clang and the MemPrintResize pass, linked with libmpr:
#
#   scripts/resize_build.sh profile <out dir> -- <build command using $CC $CXX $MPR_CFLAGS $MPR_LDFLAGS>
#   scripts/resize_build.sh apply <out dir> <reservations> -- <the same build command>
#
# profile: allocation calls go through libmpr with site ids; run the program under the Pin tool with
#   -sites (sites are named id:0x...), then analysis/memprint/sites.py makes the reservations.
# apply: the same, with the reservations built into the binary (libmpr reads them at start-up;
#   MPR_RESERVATIONS overrides them).
# The pass, libmpr and the site map (id -> source location) are built into <out dir>.
#
# Example (darknet, whose Makefile takes CC, OPTS and LDFLAGS):
#   scripts/resize_build.sh profile build -- make CC="$CC" OPTS="-O3 -ffast-math $MPR_CFLAGS" LDFLAGS="-lm -pthread $MPR_LDFLAGS"
set -eu
mode=$1 out=$(mkdir -p "$2" && cd "$2" && pwd)
shift 2
res=""
if [[ $mode == apply ]]; then res=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); shift; fi
[[ ${1:-} == -- ]] && shift
HERE=$(cd "$(dirname "$0")/.." && pwd)
LLVM_BIN=${LLVM_BIN:-$HOME/llvm-project/build/bin}

cmake -S "$HERE/resize/pass" -B "$out/pass" -DLLVM_DIR="$(dirname "$LLVM_BIN")/lib/cmake/llvm" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$out/pass" > /dev/null
gcc -O2 -g -fPIC -shared -o "$out/libmpr.so" "$HERE/resize/runtime/mpr.c" -ldl -lpthread

# This machine's clang puts spack's GCC 11 headers first; take the system GCC 8 (its libstdc++ is the one
# the programs run with). stb_image's SSE code does not build against GCC 11's intrinsics headers.
GCC=${MPR_GCC_INSTALL:-/usr/lib/gcc/x86_64-redhat-linux/8}
export CC="$LLVM_BIN/clang --gcc-install-dir=$GCC" CXX="$LLVM_BIN/clang++ --gcc-install-dir=$GCC"
export MPR_CFLAGS="-g -fpass-plugin=$out/pass/libMemPrintResize.so -DSTBI_NO_SIMD"
# --build-id: libmpr checks that address-keyed reservations were profiled on this build
export MPR_LDFLAGS="-L$out -lmpr -Wl,-rpath,$out -Wl,--build-id"
export MPR_PASS_MODE=$mode MPR_PASS_RESERVATIONS="$res" MPR_PASS_SITEMAP="$out/sitemap-$mode.txt"
rm -f "$MPR_PASS_SITEMAP"
"$@"
