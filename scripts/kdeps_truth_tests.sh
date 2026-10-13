#!/bin/bash
# The test programs of tests/static/cuda against traced runs (scripts/kdeps_truth.sh), on a machine with a GPU.
#   scripts/kdeps_truth_tests.sh OUT_DIR
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
out=$(mkdir -p "$1" && cd "$1" && pwd)
t=$here/tests/static/cuda
${PYTHON:-python3} - "$out" <<'PY'
import struct, sys, random
out = sys.argv[1]
random.seed(1)
idx = random.sample(range(4096), 300)
open(f"{out}/idx.txt", "w").write(f"{len(idx)}\n" + " ".join(map(str, idx)) + "\n")
open(f"{out}/w.bin", "wb").write(struct.pack("4096f", *[float(i % 5) for i in range(4096)]))
PY
run() { "$here/scripts/kdeps_truth.sh" "$out" "$@" || echo "$1: failed (see $out/$1)"; }
run chain "$t/chain.cu" 1000
run scatter "$t/scatter.cu"
run bfs "$t/bfs.cu" 2000
run histogram "$t/histogram.cu"
run reduce "$t/reduce.cu"
run stencil "$t/stencil.cu"
run fileidx "$t/fileidx.cu" "$out/idx.txt" "$out/w.bin"
run maybe "$t/maybe.cu"
column -s, -t "$out/scores.csv"
