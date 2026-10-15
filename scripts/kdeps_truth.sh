#!/bin/bash
# Score MemPrint's static dependences between a CUDA program's kernels against a traced run. Needs a GPU,
# nvcc, the NVBit tool gpu/nvbit_rw/rw_trace.so (gpu/README.md) and a Python with libclang.
#
#   scripts/kdeps_truth.sh OUT_DIR NAME SOURCE.cu [ARG ...]
#
# Builds SOURCE with nvcc, runs it under rw_trace.so (OUT_DIR/NAME/truth), runs the static interpreter on the
# same source and command line (OUT_DIR/NAME/static), and appends the scores to OUT_DIR/scores.csv.
# Environment: PYTHON (python3), RW_TRACE (the tool), ARCH (nvcc -arch, native), UNIT (bytes, 4),
# NVCC_FLAGS (extra flags, e.g. -I for the program's headers), SCHEDULES (the interpreter's thread schedules,
# lockstep,blocks,reverse,random: dependences not in all of them are may).
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
out=$(mkdir -p "$1" && cd "$1" && pwd)
name=$2
src=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
shift 3
PYTHON=${PYTHON:-python3}
TOOL=${RW_TRACE:-$here/gpu/nvbit_rw/rw_trace.so}
ARCH=${ARCH:-native}
UNIT=${UNIT:-4}
dir=$out/$name
mkdir -p "$dir/truth" "$dir/static"
nvcc -O2 -arch="$ARCH" ${NVCC_FLAGS:-} -o "$dir/$name" "$src" "$here/gpu/probe.cpp"
"$dir/$name" "$@" > "$dir/plain.log" 2>&1                     # the program's own output, untraced
LD_PRELOAD=$TOOL KDEPS_OUT="$dir/truth" KDEPS_UNIT=$UNIT "$dir/$name" "$@" > "$dir/traced.log" 2>&1
cd "$here/analysis"
$PYTHON -m memprint.static.kdeps "$src" "$dir/static" --name "$name" --argv "$name" "$@" --no-host \
    --unit "$UNIT" --access --schedules "${SCHEDULES:-lockstep,blocks,reverse,random}" > "$dir/static.log" 2>&1
$PYTHON -m memprint.static.kdeps_score "$dir/static" "$name" "$dir/truth" --unit "$UNIT" --out "$out/scores.csv" |
    tee "$dir/score.txt"
