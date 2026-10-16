#!/bin/bash
# Build a CUDA program for the hybrid interpreter (analysis/memprint/static/hybrid.py): a shared library whose
# main keeps its name (renaming it loses the implicit return 0 of main), with host line information (call
# sites of copies and fills, for addr2line), linked against the stand-in runtime libkdeps_cudart.so instead of CUDA's.
#   gpu/hybrid/build.sh OUT.so SOURCE.cu [SOURCE ...] [nvcc flags]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=$1
shift
[[ -f $here/libkdeps_cudart.so ]] || g++ -O2 -fPIC -shared -I/usr/local/cuda/include \
    -o "$here/libkdeps_cudart.so" "$here/kdeps_cudart.cpp"
nvcc -O2 -g -cudart none -shared -Xcompiler -fPIC "$@" "$here/../probe.cpp" \
    -L"$here" -lkdeps_cudart -Xlinker -rpath="$here" -o "$out"
