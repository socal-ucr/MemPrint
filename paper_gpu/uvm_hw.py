"""Unified Memory with prefetch plans on the GPU itself: the check of uvm_oversub.py's kernel-level model.

    PAPER_GPU_SIZE=large python paper_gpu/uvm_hw.py --only NAME,... [--fractions 1,0.5] [--reps 3]

Each program is rebuilt with -cudart shared (in SCRATCH/uvm_hw/NAME) and run under gpu/uvm_shim (managed
allocations, copies by the host, device memory limited to a share of the 64 KB blocks its device operations touch,
a synchronisation after each launch) with four plans, prefetching before each kernel:

    demand      nothing (the driver's demand paging)
    buffer      every allocation the kernel touches
    hybrid      the 64 KB blocks of the hybrid interpreter's footprint over every schedule (NAME_access_all.csv)
    truth       the 64 KB blocks of the traced footprint
    hybrid_min, truth_min
                the same, but only runs of at least uvm_costs.min_run contiguous blocks (a shorter run costs more
                as a prefetch call than as faults)
    range, history, range_min, history_min
                without the interpreter (uvm_cheap.py): each allocation's blocks from the first the kernel touches
                to the last, or what the same kernel's earlier launches predict

each only as far as the plan believes them not on the device (a prefetch of resident pages is not free). The
driver decides what to evict. Writes data/uvm_hw{,_large}.csv: per program, share, plan and repetition, the
time in prefetches and kernels, in copies, and the bytes prefetched. With --plans-only, writes
data/uvm_hw_plans{,_large}.csv instead: per program, share and plan, the prefetch calls, bytes and a digest.
"""

import argparse
import csv
import hashlib
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
from uvm_cheap import History, span  # noqa: E402
from uvm_costs import min_run  # noqa: E402

DATA = HERE / "data"
SCRATCH = Path(os.environ.get("PAPER_GPU_SCRATCH", "/tmp/nmustakin-memprint")) / "uvm_hw"
SHIM = SCRATCH / "uvm_shim.so"
BLOCK = 64 * 1024
PLANS = ("demand", "buffer", "hybrid", "truth", "hybrid_min", "truth_min", "range", "history", "range_min",
         "history_min")


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


def plans(name, w, capacity_blocks, plans_wanted=None):
    """{plan: [(launch, alloc, offset, bytes)]}: before each kernel, the 64 KB blocks a plan wants that are not on
    the device as far as the plan can tell (blocks leave it when the host touches them in a copy, or by LRU at
    capacity_blocks, as an estimate of the driver's eviction; a block a kernel touches is on the device after it)."""
    tev = list(csv.DictReader(open(w / "truth" / "truth_events.csv")))
    sizes = {int(r["index"]): int(r["bytes"]) for r in csv.DictReader(open(w / "truth" / "truth_allocs.csv"))}
    tacc = accesses(w / "truth" / "truth_access.csv")
    iev = [e for e in csv.DictReader(open(w / "hybrid" / f"{name}_events.csv")) if e["kind"] != "host"]
    iacc = accesses(w / "hybrid" / f"{name}_access_all.csv")
    to_i = {t: int(iev[k]["index"]) for t, k in align_ends(tev, iev, lambda y, x: same_op(x, y)).items()}
    out, shortest = {}, min_run(BLOCK // 1024)
    for plan in plans_wanted or PLANS:
        rows, resident, launch = [], OrderedDict(), 0
        hist = History({a: -(-n // BLOCK) for a, n in sizes.items()})

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
                src = plan.split("_")[0]
                if src == "buffer":
                    want = {(a, x) for a, _ in touched for x in range(-(-sizes[a] // BLOCK))}
                elif src == "truth":
                    want = touched
                elif src == "range":
                    want = span(touched)
                elif src == "history":
                    want = hist.predict(e["name"])
                    hist.record(e["name"], touched)
                elif src == "hybrid" and i in to_i:
                    want = {(a, x) for a, xs in blocks_of(iacc.get(to_i[i], [])).items() for x in xs}
                else:
                    want = set()
                new = sorted(x for x in want if x not in resident)
                sent = []
                for a in sorted({a for a, _ in new}):
                    for off, n in ranges([x for b, x in new if b == a]):
                        if plan.endswith("_min") and n < shortest * BLOCK:
                            continue
                        rows.append((launch, a, off, n))
                        sent += [(a, x) for x in range(off // BLOCK, (off + n) // BLOCK)]
                enter(sent)
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


def run(name, fractions, reps, wanted=PLANS):
    b = B.BENCH[name]
    w = B.workdir(b)
    exe = build(b, w)
    n_blocks, n_allocs = footprint_of(w)
    footprint = n_blocks * BLOCK
    d = SCRATCH / name
    out = []
    for frac in fractions:
        keep = int(frac * footprint)
        for p, rows in plans(name, w, max(1, int(frac * n_blocks)), wanted).items():
            with open(d / f"plan_{p}.txt", "w") as f:
                f.writelines(f"{k} {a} {off} {n}\n" for k, a, off, n in rows)
        for rep in range(reps):
            for p in wanted:
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


def plan_summary(name, fractions, wanted=PLANS):
    """Per share and plan: the prefetch calls, the bytes and a digest of the plan (equal plans, equal digests)."""
    w = B.workdir(B.BENCH[name])
    n_blocks, _ = footprint_of(w)
    out = []
    for frac in fractions:
        for p, rows in plans(name, w, max(1, int(frac * n_blocks)), wanted).items():
            out.append({"name": name, "fraction": frac, "plan": p, "calls": len(rows),
                        "bytes": sum(n for *_, n in rows),
                        "digest": hashlib.sha1(repr(rows).encode()).hexdigest()[:12]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True)
    ap.add_argument("--fractions", default="1.0,0.75,0.5")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--plans-only", action="store_true",
                    help="only describe the plans, in data/uvm_hw_plans{,_large}.csv")
    ap.add_argument("--plans", default=",".join(PLANS))
    ap.add_argument("--tag", default="", help="appended to the file name")
    args = ap.parse_args()
    fractions = [float(x) for x in args.fractions.split(",")]
    rows = []
    if args.plans_only:
        for n in args.only.split(","):
            rows += plan_summary(n, fractions, args.plans.split(","))
    else:
        build_shim()
        for n in args.only.split(","):
            rows += run(n, fractions, args.reps, args.plans.split(","))
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    path = DATA / f"uvm_hw{'_plans' if args.plans_only else ''}{B.SUFFIX}{args.tag}.csv"
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, cols)
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
