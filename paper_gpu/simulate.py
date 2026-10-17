"""Simulate a CUDA program's operations on a GPU under different ways of deciding which ones may overlap.

    python paper_gpu/simulate.py [--profile] [--only NAME,...] [--delay US] [--host-sync]

Each benchmark of benchmarks.py that has a traced run (data/bench/NAME/truth) is run once under Nsight
Systems (--profile; data/bench/NAME/sim_ops.csv): per operation, its duration on the GPU and, for a kernel,
its grid, block, registers and shared memory. The run whose operations match the traced run's is kept, so
the operations, their durations and the dependences that actually hold all come from runs of the same
sequence.

The simulator issues the operations in program order to a GPU that runs a kernel when its predecessors
have finished and enough streaming multiprocessors (SMs) are free, and a copy when its predecessors have
finished and its copy engine (one per direction) is free. Which operations are predecessors depends on the
policy:

    sequential   every operation waits for the one before it
    buffer       ... for every earlier one that accesses an allocation it accesses (a simple oracle that
                 knows each operation's buffers)
    buffer-rw    ... for every earlier one that accesses an allocation it accesses, one of them writing it
    hybrid       ... for every earlier one the hybrid interpreter reports a dependence on (certain or
                 possible); data/bench/NAME/hybrid
    full         ... the same for the whole-program interpreter; data/bench/NAME/full
    truth        ... for every earlier one it actually depends on (the traced run's dependences): the best
                 any dependence checker can do

An operation that waits for a predecessor starts the moment the predecessor ends: the dependence is known
in advance, so the scheduling delay of issuing it is hidden. A dependence the policy does not report but the
run has (a false negative) is detected when the operation starts before its true predecessor has ended:
the operation runs to its end for nothing, is rolled back, and is issued again once its true predecessors
have ended, after the scheduling delay (--delay; by default the delay measured between a device-to-host copy
and the next operation in the profiled runs). Restoring the data a rolled-back operation overwrote is not
charged. Durations do not change when kernels share the GPU. With --host-sync, an operation may not start
before every device-to-host copy issued before it has ended (the host reads what it copied before going on).

Writes data/sim.csv (per benchmark and policy: makespan, speedup over sequential, rollbacks) and
data/sim_delay.csv (the hybrid and full policies over a range of scheduling delays).
"""

import argparse
import csv
import heapq
import json
import math
import os
import sqlite3
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "analysis"))
import benchmarks as B  # noqa: E402
from memprint.static import kdeps_score as ks  # noqa: E402
from memprint.static.kdeps import instantiate_loop  # noqa: E402

DATA = HERE / "data"
GPU = {"sms": 56, "threads": 2048, "blocks": 32, "regs": 65536, "smem": 164 * 1024}  # NVIDIA A30, per SM
POLICIES = ["sequential", "buffer", "buffer-rw", "hybrid", "full", "truth"]
COPY_KIND = {1: "H2D", 2: "D2H", 8: "D2D", 10: "D2D"}


# ---------------------------------------------------------------- profiling

def _rows(db, table, cols):
    try:
        return [dict(zip(cols, r)) for r in db.execute(f"select {', '.join(cols)} from {table}")]
    except sqlite3.OperationalError:
        return []


def nsys_ops(sqlite_path):
    """The device operations of a profiled run in the order they started, and the gaps between the end of a
    device-to-host copy and the start of the next operation (ns)."""
    db = sqlite3.connect(sqlite_path)
    names = dict(db.execute("select id, value from StringIds"))
    ops = []
    for r in _rows(db, "CUPTI_ACTIVITY_KIND_KERNEL",
                   ["start", "end", "demangledName", "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ",
                    "registersPerThread", "staticSharedMemory", "dynamicSharedMemory"]):
        ops.append({"kind": "kernel", "name": names.get(r["demangledName"], ""), "start": r["start"],
                    "dur_ns": r["end"] - r["start"], "blocks": r["gridX"] * r["gridY"] * r["gridZ"],
                    "threads": r["blockX"] * r["blockY"] * r["blockZ"], "regs": r["registersPerThread"],
                    "smem": r["staticSharedMemory"] + r["dynamicSharedMemory"], "bytes": 0})
    for r in _rows(db, "CUPTI_ACTIVITY_KIND_MEMCPY", ["start", "end", "bytes", "copyKind"]):
        ops.append({"kind": "copy", "name": f"cudaMemcpy {COPY_KIND.get(r['copyKind'], '?')}", "start": r["start"],
                    "dur_ns": r["end"] - r["start"], "blocks": 0, "threads": 0, "regs": 0, "smem": 0,
                    "bytes": r["bytes"]})
    for r in _rows(db, "CUPTI_ACTIVITY_KIND_MEMSET", ["start", "end", "bytes"]):
        ops.append({"kind": "set", "name": "cudaMemset", "start": r["start"], "dur_ns": r["end"] - r["start"],
                    "blocks": 0, "threads": 0, "regs": 0, "smem": 0, "bytes": r["bytes"]})
    ops.sort(key=lambda o: o["start"])
    gaps = [ops[i + 1]["start"] - (o["start"] + o["dur_ns"]) for i, o in enumerate(ops[:-1])
            if o["name"] == "cudaMemcpy D2H"]
    return ops, gaps


def _same(o, t):
    if o["kind"] != t["kind"]:
        return False
    if o["kind"] == "copy":
        return o["name"] == t["name"].split(" (")[0]
    return o["kind"] != "kernel" or o["name"].split("(")[0] == t["name"].split("(")[0]


def _fits(ops, truth_events):
    """The profiled operations in the traced run's order, or None. A launch the trace has but the profile has
    not (it failed: an invalid configuration, an empty grid) takes no time."""
    out, k = [], 0
    for t in truth_events:
        if k < len(ops) and _same(ops[k], t):
            out.append(ops[k])
            k += 1
        elif t["kind"] == "kernel":
            out.append({"kind": "kernel", "name": t["name"], "start": 0, "dur_ns": 0, "blocks": 0, "threads": 0,
                        "regs": 0, "smem": 0, "bytes": 0})
        else:
            return None
    return out if k == len(ops) else None


def profile(name, tries=8):
    """Profile one benchmark until a run issues the traced run's operations; write sim_ops.csv. Returns
    (status, gaps)."""
    b = B.BENCH[name]
    w = B.workdir(b)
    truth = list(csv.DictReader(open(w / "truth" / "truth_events.csv")))
    out = w / "nsys"
    for k in range(tries):
        done = subprocess.run(["nsys", "profile", "--trace=cuda", "--force-overwrite=true", "--export=sqlite",
                               "-o", str(out), str(w / "native"), *B.argv(b, w)],
                              cwd=w, capture_output=True, text=True)
        if done.returncode or not (w / "nsys.sqlite").exists():
            return f"nsys failed: {(done.stderr.strip().splitlines() or ['?'])[-1][:100]}", []
        ops, gaps = nsys_ops(w / "nsys.sqlite")
        ops = _fits(ops, truth)
        if ops is not None:
            with open(w / "sim_ops.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, ["index", "kind", "name", "dur_ns", "blocks", "threads", "regs", "smem",
                                        "bytes"])
                wr.writeheader()
                for i, o in enumerate(ops):
                    wr.writerow({"index": i, **{c: o[c] for c in wr.fieldnames if c != "index"}})
            return f"ok (try {k + 1})", gaps
    return f"no run matched the trace ({len(ops)} operations, traced {len(truth)})", gaps


# ---------------------------------------------------------------- dependences per policy

def _touched(truth_dir):
    """Per traced event: {allocation: bits (1 read, 2 write)}."""
    out = {}
    with open(truth_dir / "truth_access.csv") as f:
        for r in csv.DictReader(f):
            d = out.setdefault(int(r["event"]), {})
            d[int(r["alloc"])] = d.get(int(r["alloc"]), 0) | int(r["bits"])
    return out


def _predicted(interp_dir, name, truth_events):
    """An interpreter's dependences (certain or possible) as pairs of the traced run's events, with its loop
    template instantiated for the run's rounds; None if it has no output."""
    if not (interp_dir / f"{name}_events.csv").exists():
        return None
    ie, iedges = ks.interp_edges(interp_dir, name)
    inv = {v: k for k, v in ks.truth_map(ie, truth_events).items()}    # interpreter index -> run index
    pairs = {(inv[s], inv[d]) for (s, d, _, _) in iedges if s in inv and d in inv}
    path = interp_dir / f"{name}_loop.json"
    if path.exists():
        loop = json.load(open(path))
        a, p, post = loop["start"], loop["period"], loop["post"]
        n = len(truth_events)
        if n - post >= a and not (n - post - a) % p and all(
                ks._same(loop["phases"][(i - a) % p], truth_events[i]) for i in range(a, n - post)):
            pairs |= {(s, d) for (s, d, _, _) in instantiate_loop(loop, n)}
    return pairs


def dependences(name):
    """{policy: {event: set of predecessor events}} over the traced run's events."""
    w = B.workdir(B.BENCH[name])
    truth_events = list(csv.DictReader(open(w / "truth" / "truth_events.csv")))
    n = len(truth_events)
    _, tedges = ks.truth_edges(w / "truth", 4)
    touched = _touched(w / "truth")
    pairs = {"sequential": {(i - 1, i) for i in range(1, n)},
             "truth": {(s, d) for (s, d, _, _) in tedges}}
    for policy, write in (("buffer", False), ("buffer-rw", True)):
        got = set()
        for j in range(n):
            for i in range(j):
                common = touched.get(i, {}).keys() & touched.get(j, {}).keys()
                if any(not write or (touched[i][a] | touched[j][a]) & 2 for a in common):
                    got.add((i, j))
        pairs[policy] = got
    for which in ("hybrid", "full"):
        got = _predicted(w / which, name, truth_events)
        if got is not None:
            pairs[which] = got
    out = {}
    for policy, ps in pairs.items():
        preds = {j: set() for j in range(n)}
        for s, d in ps:
            if s < d:
                preds[d].add(s)
        out[policy] = preds
    return out


# ---------------------------------------------------------------- the GPU

def sms_needed(op):
    """SMs a kernel occupies (all its blocks resident at once, or the whole GPU); a memset takes one."""
    if op["kind"] == "set":
        return 1
    if op["kind"] != "kernel":
        return 0
    t = max(1, op["threads"])
    per_sm = min(GPU["blocks"], GPU["threads"] // t if t <= GPU["threads"] else 1,
                 GPU["regs"] // max(1, op["regs"] * t) if op["regs"] else GPU["blocks"],
                 GPU["smem"] // op["smem"] if op["smem"] else GPU["blocks"])
    return min(GPU["sms"], max(1, math.ceil(op["blocks"] / max(1, per_sm))))


def engine(op):
    return op["name"].split()[-1] if op["kind"] == "copy" else "sm"


def simulate(ops, preds, truth, delay_ns, host_sync=False):
    """Run the operations under one policy: (makespan ns, rollbacks, ns of work rolled back)."""
    n = len(ops)
    need = [sms_needed(o) for o in ops]
    if host_sync:
        last_d2h, extra = None, []
        for j, o in enumerate(ops):
            extra.append({last_d2h} if last_d2h is not None else set())
            if o["name"] == "cudaMemcpy D2H":
                last_d2h = j
        preds = {j: preds[j] | extra[j] for j in range(n)}
        truth = {j: truth[j] | extra[j] for j in range(n)}
    end = [None] * n                     # when an operation's final run ended
    waiting = [set(preds[j]) for j in range(n)]
    retry_at = [None] * n                # a rolled-back operation: may not start before
    state = ["wait"] * n                 # wait, run, wasted, done
    free_sm, busy = GPU["sms"], set()    # copy engines in use
    running = []                         # heap of (end, op, final)
    now, rollbacks, wasted = 0, 0, 0
    succ = {j: set() for j in range(n)}
    for j in range(n):
        for i in preds[j] | truth[j]:
            succ[i].add(j)
    done = 0
    while done < n:
        for j in range(n):                                             # start what may start, in program order
            if state[j] != "wait" or waiting[j] or (retry_at[j] is not None and retry_at[j] > now):
                continue
            e = engine(ops[j])
            if (e == "sm" and need[j] > free_sm) or (e != "sm" and e in busy):
                continue
            final = retry_at[j] is not None or all(end[i] is not None and end[i] <= now for i in truth[j])
            if e == "sm":
                free_sm -= need[j]
            else:
                busy.add(e)
            state[j] = "run" if final else "wasted"
            heapq.heappush(running, (now + int(ops[j]["dur_ns"]), j, final))
        nxt = [running[0][0]] if running else []
        nxt += [retry_at[j] for j in range(n) if state[j] == "wait" and not waiting[j] and retry_at[j] is not None
                and retry_at[j] > now]
        if not nxt:
            raise RuntimeError("deadlock")
        now = max(now, min(nxt))
        while running and running[0][0] <= now:
            t, j, final = heapq.heappop(running)
            e = engine(ops[j])
            if e == "sm":
                free_sm += need[j]
            else:
                busy.discard(e)
            if final:
                state[j], end[j] = "done", t
                done += 1
                for k in succ[j]:
                    waiting[k].discard(j)
            else:                                                      # it ran before a true predecessor ended
                rollbacks += 1
                wasted += int(ops[j]["dur_ns"])
                state[j] = "wait"
                waiting[j] = {i for i in truth[j] if end[i] is None}
                retry_at[j] = -1                                       # set once its true predecessors end
        for j in range(n):                                             # a rolled-back operation is issued again
            if state[j] == "wait" and retry_at[j] == -1 and not waiting[j]:   # after the scheduling delay
                retry_at[j] = max([now] + [end[i] for i in truth[j]]) + delay_ns
    return max(end), rollbacks, wasted


# ---------------------------------------------------------------- main

def load_ops(name):
    """A benchmark's profiled operations: data/bench/NAME/sim_ops.csv, else its rows of data/sim_ops.csv."""
    path = B.workdir(B.BENCH[name]) / "sim_ops.csv"
    if path.exists():
        rows = list(csv.DictReader(open(path)))
    elif (DATA / "sim_ops.csv").exists():
        rows = [r for r in csv.DictReader(open(DATA / "sim_ops.csv")) if r["name_of"] == name]
    else:
        return None
    if not rows:
        return None
    return [{**r, "dur_ns": int(r["dur_ns"]), "blocks": int(r["blocks"]), "threads": int(r["threads"]),
             "regs": int(r["regs"]), "smem": int(r["smem"])} for r in rows]


def save_ops(names):
    """All benchmarks' profiled operations in one file (data/sim_ops.csv), to simulate without a GPU."""
    rows = []
    for n in names:
        path = B.workdir(B.BENCH[n]) / "sim_ops.csv"
        if path.exists():
            rows += [{"name_of": n, **r} for r in csv.DictReader(open(path))]
    if rows:
        with open(DATA / "sim_ops.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true", help="profile the benchmarks with nsys first")
    ap.add_argument("--only", default=None)
    ap.add_argument("--delay", type=float, default=None, help="scheduling delay in microseconds")
    ap.add_argument("--host-sync", action="store_true")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else [n for n in B.BENCH
                                                     if (B.workdir(B.BENCH[n]) / "truth" / "truth_events.csv").exists()
                                                     and (B.workdir(B.BENCH[n]) / "hybrid" / f"{n}_events.csv").exists()]
    gaps_path = DATA / "sim_gaps.json"
    gaps = json.load(open(gaps_path)) if gaps_path.exists() else {}
    if args.profile:
        for n in names:
            status, g = profile(n)
            gaps[n] = g
            print(n, status, flush=True)
        json.dump(gaps, open(gaps_path, "w"))
        save_ops(names)
    allgaps = [x for g in gaps.values() for x in g]
    delay_ns = args.delay * 1000 if args.delay is not None else (statistics.median(allgaps) if allgaps else 10000)
    print(f"scheduling delay: {delay_ns / 1000:.1f} us", flush=True)
    rows, sweep = [], []
    for n in names:
        ops = load_ops(n)
        if ops is None:
            print(n, "no profile", flush=True)
            continue
        deps = dependences(n)
        base = None
        for policy in POLICIES:
            if policy not in deps:
                continue
            t, rb, wasted = simulate(ops, deps[policy], deps["truth"], delay_ns, args.host_sync)
            base = base or t
            missed = sum(1 for j in deps["truth"] for i in deps["truth"][j] if i not in deps[policy][j])
            rows.append({"name": n, "suite": B.BENCH[n]["suite"], "policy": policy, "ops": len(ops),
                         "makespan_us": round(t / 1000, 2), "speedup": round(base / t, 4), "rollbacks": rb,
                         "rolled_back_us": round(wasted / 1000, 2), "missed_pairs": missed,
                         "pairs": sum(len(v) for v in deps[policy].values())})
        for d in (0, 5, 10, 20, 50, 100, 200):
            for policy in ("hybrid", "full"):
                if policy in deps:
                    t, rb, _ = simulate(ops, deps[policy], deps["truth"], d * 1000, args.host_sync)
                    sweep.append({"name": n, "policy": policy, "delay_us": d, "makespan_us": round(t / 1000, 2),
                                  "speedup": round(base / t, 4), "rollbacks": rb})
        print(n, " ".join(f"{r['policy']}={r['speedup']:.2f}x/{r['rollbacks']}rb" for r in rows if r["name"] == n),
              flush=True)
    suffix = "_hostsync" if args.host_sync else ""
    for path, data in ((DATA / f"sim{suffix}.csv", rows), (DATA / f"sim_delay{suffix}.csv", sweep)):
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, list(data[0]))
            wr.writeheader()
            wr.writerows(data)
        print("wrote", path)


if __name__ == "__main__":
    main()
