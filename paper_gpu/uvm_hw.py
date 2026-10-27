"""Unified Memory with prefetch plans on the GPU itself: the check of uvm_oversub.py's kernel-level model.

    PAPER_GPU_SIZE=large python paper_gpu/uvm_hw.py --only NAME,... [--fractions 1,0.5] [--reps 3]

Each program is rebuilt with -cudart shared (in SCRATCH/uvm_hw/NAME) and run under gpu/uvm_shim (managed
allocations, copies by the host, device memory limited to a share of the 64 KB blocks its device operations touch,
a synchronisation after each launch) with four plans, prefetching before each kernel:

    demand      nothing (the driver's demand paging)
    buffer      every allocation the kernel touches
    hybrid      the 64 KB blocks of the hybrid interpreter's footprint over every schedule (NAME_access_all.csv)
    truth       the 64 KB blocks of the traced footprint

each only as far as the plan believes them not on the device (a prefetch of resident pages is not free). The
driver decides what to evict. Writes data/uvm_hw{,_large}.csv: per program, share, plan and repetition, the
time in prefetches and kernels, in copies, and the bytes prefetched.
"""

import argparse
import csv
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmarks as B  # noqa: E402
from copy_savings import H2D, D2H, UNIT, accesses  # noqa: E402
from memprint.static.kdeps import align_ends  # noqa: E402
from memprint.static.kdeps_score import _same as same_op  # noqa: E402

DATA = HERE / "data"
SCRATCH = Path(os.environ.get("PAPER_GPU_SCRATCH", "/tmp/nmustakin-memprint")) / "uvm_hw"
SHIM = SCRATCH / "uvm_shim.so"
BLOCK = 64 * 1024
PLANS = ("demand", "buffer", "hybrid", "truth")


def build_shim():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    src = B.ROOT / "gpu" / "uvm_shim" / "uvm_shim.cpp"
    if not SHIM.exists() or SHIM.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["g++", "-O2", "-shared", "-fPIC", "-I/usr/local/cuda/include", "-o", str(SHIM), str(src),
                        "-L/usr/local/cuda/lib64", "-lcudart", "-ldl"], check=True)


def build(b, w):
    out = SCRATCH / b["name"] / "native_shared"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        sd = B.source_dir(b, w)
        subprocess.run(["nvcc", "-O2", "-arch=native", "-w", "-cudart", "shared", *B.flags(b, sd), "-o", str(out),
                        *(str(sd / s) for s in b["sources"])], check=True, capture_output=True, text=True)
    return out


def blocks_of(rows):
    """{alloc: sorted 64 KB blocks} of one event's access rows."""
    out = {}
    for a, lo, hi, _ in rows:
        if hi > lo:
            out.setdefault(a, set()).update(range(lo * UNIT // BLOCK, ((hi - 1) * UNIT) // BLOCK + 1))
    return {a: sorted(v) for a, v in out.items()}


def ranges(blocks):
    """Contiguous runs of blocks as (offset, bytes)."""
    out, start, prev = [], None, None
    for x in blocks + [None]:
        if start is not None and (x is None or x != prev + 1):
            out.append((start * BLOCK, (prev - start + 1) * BLOCK))
            start = None
        if x is not None and start is None:
            start = x
        prev = x
    return out


def plans(name, w, capacity_blocks):
    """{plan: [(launch, alloc, offset, bytes)]}: before each kernel, the 64 KB blocks a plan wants that are not on
    the device as far as the plan can tell (blocks leave it when the host touches them in a copy, or by LRU at
    capacity_blocks, as an estimate of the driver's eviction; a block a kernel touches is on the device after it)."""
    tev = list(csv.DictReader(open(w / "truth" / "truth_events.csv")))
    sizes = {int(r["index"]): int(r["bytes"]) for r in csv.DictReader(open(w / "truth" / "truth_allocs.csv"))}
    tacc = accesses(w / "truth" / "truth_access.csv")
    iev = [e for e in csv.DictReader(open(w / "hybrid" / f"{name}_events.csv")) if e["kind"] != "host"]
    iacc = accesses(w / "hybrid" / f"{name}_access_all.csv")
    to_i = {t: int(iev[k]["index"]) for t, k in align_ends(tev, iev, lambda y, x: same_op(x, y)).items()}
    out = {}
    for plan in PLANS:
        rows, resident, launch = [], OrderedDict(), 0

        def enter(blocks):
            for x in blocks:
                resident[x] = True
                resident.move_to_end(x)
            while len(resident) > capacity_blocks:
                resident.popitem(last=False)

        for i, e in enumerate(tev):
            touched = {(a, x) for a, xs in blocks_of(tacc.get(i, [])).items() for x in xs}
            if e["name"] in (H2D, D2H):
                for x in touched:
                    resident.pop(x, None)
                continue
            if e["kind"] == "kernel":
                if plan == "buffer":
                    want = {(a, x) for a, _ in touched for x in range(-(-sizes[a] // BLOCK))}
                elif plan == "truth":
                    want = touched
                elif plan == "hybrid" and i in to_i:
                    want = {(a, x) for a, xs in blocks_of(iacc.get(to_i[i], [])).items() for x in xs}
                else:
                    want = set()
                new = sorted(x for x in want if x not in resident)
                for a in sorted({a for a, _ in new}):
                    for off, n in ranges([x for b, x in new if b == a]):
                        rows.append((launch, a, off, n))
                enter(new)
                launch += 1
            enter(touched)
        out[plan] = rows
    return out


def footprint_of(w):
    """The 64 KB blocks the device operations touch, and the allocations of the trace."""
    tev = list(csv.DictReader(open(w / "truth" / "truth_events.csv")))
    tacc = accesses(w / "truth" / "truth_access.csv")
    touched = set()
    for i, e in enumerate(tev):
        if e["name"] not in (H2D, D2H):
            touched |= {(a, x) for a, xs in blocks_of(tacc.get(i, [])).items() for x in xs}
    return len(touched), sum(1 for _ in csv.DictReader(open(w / "truth" / "truth_allocs.csv")))


def run(name, fractions, reps):
    b = B.BENCH[name]
    w = B.workdir(b)
    exe = build(b, w)
    n_blocks, n_allocs = footprint_of(w)
    footprint = n_blocks * BLOCK
    d = SCRATCH / name
    out = []
    for frac in fractions:
        keep = int(frac * footprint)
        for p, rows in plans(name, w, max(1, int(frac * n_blocks))).items():
            with open(d / f"plan_{p}.txt", "w") as f:
                f.writelines(f"{k} {a} {off} {n}\n" for k, a, off, n in rows)
        for rep in range(reps):
            for p in PLANS:
                res = d / f"out_{p}.txt"
                res.unlink(missing_ok=True)
                env = dict(os.environ, LD_PRELOAD=str(SHIM), UVM_DEVICE_BYTES=str(keep), UVM_OUT=str(res),
                           UVM_PLAN=str(d / f"plan_{p}.txt"))
                done = subprocess.run([str(exe), *B.argv(b, w)], cwd=w, env=env, capture_output=True, text=True,
                                      timeout=3600)
                r = dict(line.split() for line in res.read_text().splitlines()) if res.exists() else {}
                out.append({"name": name, "suite": b["suite"], "footprint_bytes": footprint, "fraction": frac,
                             "device_bytes": keep, "plan": p, "rep": rep, "rc": done.returncode,
                             "allocs_traced": n_allocs, **r})
                print(name, frac, p, rep, r.get("gpu_ns"), r.get("copy_ns"), "rc", done.returncode, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True)
    ap.add_argument("--fractions", default="1.0,0.5")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    build_shim()
    rows = []
    for n in args.only.split(","):
        rows += run(n, [float(x) for x in args.fractions.split(",")], args.reps)
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    path = DATA / f"uvm_hw{B.SUFFIX}.csv"
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, cols)
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
