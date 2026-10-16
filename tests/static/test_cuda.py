"""Check the CUDA interpreter (interp_cuda) and the dependences between kernels (kdeps) on the programs in
tests/static/cuda/: values the programs compute (probe), launches, and the dependences each program's
comment describes, with their units, certainty and input dependence. No GPU is needed:
python tests/static/test_cuda.py"""
import csv
import struct
import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from memprint.static import interp_cuda as ic  # noqa: E402
from memprint.static import kdeps, kdeps_score  # noqa: E402
from memprint.static.kdeps import reduce_edges  # noqa: E402

HERE = Path(__file__).with_name("cuda")
ok = True


def check(what, got, expected):
    global ok
    good = got == expected
    ok &= good
    print(f"  {what}: got {got} expected {expected} {'ok' if good else 'FAIL'}")


def run(name, argv=None):
    print(name)
    it, errors, _ = ic.analyze(HERE / name, argv=argv)
    check("parse errors", errors, [])
    probes = [float(v) for p in it.probes if isinstance(p, ic.Real) for v in np.ravel(p.v)] + \
        [int(v) for p in it.probes if not isinstance(p, ic.Real) for v in np.ravel(p)]   # a probe in a loop: one per pass
    return it, probes, it.tracker.edge_rows()


def edge(rows, src, dst, kind, obj):
    """The (units must, units may, input dependent, needed) of a dependence between events src -> dst."""
    for r in rows:
        if r["src"] == src and r["dst"] == dst and r["kind"] == kind and obj in (r["object"], r["object"].split("@")[0]):
            return r["units_must"], r["units_may"], r["input_dependent"], r["needed"]
    return None


def index(it, name, k=0):
    """The event index of the k-th event with this name."""
    return [e["index"] for e in it.tracker.events if e["name"] == name][k]


# a chain of kernels: init a, init b -> add -> scale -> copy back (returns and a grid-stride loop)
it, probes, rows = run("chain.cu", ["chain", "1000"])
check("probes", probes, [18.0])
add, scale = index(it, "add"), index(it, "scale")
check("init a -> add", edge(rows, index(it, "init", 0), add, "RAW", "a"), (1000, 0, 0, 1))
check("init b -> add", edge(rows, index(it, "init", 1), add, "RAW", "b"), (1000, 0, 0, 1))
check("add -> scale", edge(rows, add, scale, "RAW", "c"), (1000, 0, 0, 1))
check("add -> scale WAW", edge(rows, add, scale, "WAW", "c"), (1000, 0, 0, 1))

# out[idx[i]] with idx from the host: the scatter meets gather_even on exactly n units, gather_odd not at all
it, probes, rows = run("scatter.cu")
check("probes", probes, [1, 2])
sc, ev_, od = index(it, "scatter"), index(it, "gather_even"), index(it, "gather_odd")
check("scatter -> gather_even", edge(rows, sc, ev_, "RAW", "d_out"), (1000, 0, 1, 1))
check("scatter -> gather_odd", edge(rows, sc, od, "RAW", "d_out"), None)
check("memset -> gather_odd", edge(rows, index(it, "cudaMemset"), od, "RAW", "d_out"), (1000, 0, 0, 1))

# BFS: as many launches as levels, under data control; each level reads what the one before claimed
it, probes, rows = run("bfs.cu", ["bfs", "2000"])
check("probes (levels, reached, depth)", probes, [15, 2000, 14])
expands = [e for e in it.tracker.events if e["name"] == "expand"]
check("launches", len(expands), 15)
check("under data control", all(e["data_control"] for e in expands), True)
e3, e4 = expands[3]["index"], expands[4]["index"]
r = edge(rows, e3, e4, "RAW", "d_dist")
check("level 3 -> level 4 through dist (input-dependent)", r is not None and r[2] == 1, True)

# atomics: bins chosen by the data, max and min over all threads
it, probes, rows = run("histogram.cu")
check("probes (bin 0, bin 7, total, max, min)", probes, [84, 0, 3000, 998, 0])
h, s = index(it, "hist"), index(it, "scan_bins")
check("hist -> scan_bins (the 32 bins hit)", edge(rows, h, s, "RAW", "bins"), (32, 0, 1, 1))
check("memset -> scan_bins (the bins not hit)", edge(rows, index(it, "cudaMemset"), s, "RAW", "bins"), (32, 0, 0, 0))

# shared memory and __syncthreads: block sums, then their sum
it, probes, rows = run("reduce.cu")
check("probes (sum, block 0)", probes, [238834, 11203])
check("block_sum -> final_sum", edge(rows, index(it, "block_sum"), index(it, "final_sum"), "RAW", "d_partial"),
      (20, 0, 0, 1))

# ping-pong stencil: needed dependences form the chain t -> t+1
it, probes, rows = run("stencil.cu")
steps = [e["index"] for e in it.tracker.events if e["name"] == "jacobi"]
needed = sorted({(r["src"], r["dst"]) for r in rows if r["needed"] and r["src"] in steps and r["dst"] in steps})
check("needed between steps", needed, [(steps[i], steps[i + 1]) for i in range(len(steps) - 1)])
check("step 0 -> step 1", edge(rows, steps[0], steps[1], "RAW", "b"), (1022, 0, 0, 1))
check("step 0 -> step 2 (implied)", edge(rows, steps[0], steps[2], "WAW", "b"), (1022, 0, 0, 0))

# input files: indices by fscanf, weights by fread
with tempfile.TemporaryDirectory() as tmp:
    rng = np.random.default_rng(1)
    idx = rng.choice(4096, 300, replace=False)
    Path(tmp, "idx.txt").write_text(f"{len(idx)}\n" + " ".join(map(str, idx)) + "\n")
    w = [float(i % 5) for i in range(4096)]
    Path(tmp, "w.bin").write_bytes(struct.pack("4096f", *w))
    low = int(np.sum(idx < 2048))
    it, probes, rows = run("fileidx.cu", ["fileidx", str(Path(tmp, "idx.txt")), str(Path(tmp, "w.bin"))])
    check("probes (k, sum)", probes, [300.0, sum(w[:2048]) + low])
    check("update -> sum_half (listed indices below n/2)",
          edge(rows, index(it, "update"), index(it, "sum_half"), "RAW", "d_w"), (low, 0, 1, 1))

# Rodinia-style BFS: a graph file in Rodinia's format, an array of structs, bool masks, two kernels per level
with tempfile.TemporaryDirectory() as tmp:
    rng = np.random.default_rng(7)
    n = 4096
    adj = [list(rng.integers(0, n, rng.integers(1, 6))) for _ in range(n)]
    starts = np.concatenate([[0], np.cumsum([len(a) for a in adj])])
    lines = [str(n)] + [f"{starts[v]} {len(adj[v])}" for v in range(n)] + ["0", str(starts[-1])]
    lines += [f"{u} 1" for a in adj for u in a]
    Path(tmp, "graph.txt").write_text("\n".join(lines) + "\n")
    dist = [-1] * n
    dist[0], queue = 0, deque([0])
    while queue:
        v = queue.popleft()
        for u in adj[v]:
            if dist[u] < 0:
                dist[u] = dist[v] + 1
                queue.append(u)
    it, probes, rows = run("rodinia_bfs.cu", ["rodinia_bfs", str(Path(tmp, "graph.txt"))])
    reached = [d for d in dist if d >= 0]
    check("probes (iterations, reached, cost sum)", probes, [max(reached) + 1, len(reached), sum(reached)])
    launches = [e for e in it.tracker.events if e["kind"] == "kernel"]
    check("launches under data control", sum(e["data_control"] for e in launches), 2 * (max(reached) + 1))

# a branch on values the interpreter does not know: may dependences; the unconditional write stays must
it, probes, rows = run("maybe.cu")
t, c = index(it, "threshold"), index(it, "count")
check("threshold -> count through flag (may)", edge(rows, t, c, "RAW", "flag"), (0, 512, 1, 1))
check("threshold -> count through seen (must)", edge(rows, t, c, "RAW", "seen"), (512, 0, 0, 1))

# buffers chosen by the data: the producer writes through a device table of pointers, the bucket chosen by the
# key; the threads are split by the bucket they load, so each consumer depends on its own bucket, for certain
it, probes, rows = run("ptrsel.cu", ["ptrsel"])
check("probes (sums per bucket)", probes, [166833, 166333, 0, 166334])
dist = index(it, "distribute")
for k, units in enumerate((334, 333, None, 333)):
    obj = "d_bucket@30" + (f"#{k}" if k else "")
    got = edge(rows, dist, index(it, "consume", k), "RAW", obj)
    check(f"distribute -> consume {k}", got[:2] if got else None, (units, 0) if units else None)

# a struct field pointing into one of two lists, a pointer kept in a local, a divergent load in a loop: all
# split; a load after an atomic in the same statement is not, and its references are may, to device objects
it, probes, rows = run("ptrsel2.cu", ["ptrsel2"])
check("probes", probes, [17316, 5798])
split_out, late = index(it, "split_out"), index(it, "late_load")
check("degree_sum -> split_out", edge(rows, index(it, "degree_sum"), split_out, "RAW", "d_out")[:2], (2048, 0))
check("split_out -> sum a", edge(rows, split_out, index(it, "sum", 0), "RAW", "d_a")[:2], (1927, 0))
check("split_out -> sum b", edge(rows, split_out, index(it, "sum", 1), "RAW", "d_b")[:2], (121, 0))
check("split_out -> late_load: may", edge(rows, split_out, late, "WAW", "d_a")[:2], (0, 1927))
check("late_load reaches no host object",
      [r["object"] for r in rows if r["dst"] == late and r["object"].startswith("malloc")], [])

# thread schedules: label propagation with atomicMin in place takes a different number of rounds under each,
# so the rounds past the common ones, and the dependences not in every schedule, are may; a BFS is the same
# under all of them and keeps every dependence must
check("align_ends", kdeps.align_ends(list("abxxyz"), list("abxyz")), {0: 0, 1: 1, 2: 2, 4: 3, 5: 4})
for name, argv, varies in (("cc.cu", ["cc", "1024"], True), ("bfs.cu", ["bfs", "500"], False)):
    runs = [ic.analyze(HERE / name, argv=argv, host_events=False, schedule=s)[0].tracker for s in ic.SCHEDULES]
    counts = [len(t.events) for t in runs]
    matched = kdeps.merge_schedules(runs)
    rows = runs[0].edge_rows()
    print(name, "under", ic.SCHEDULES)
    check("event counts differ", len(set(counts)) > 1, varies)
    check("some dependences schedule-dependent", any(r["schedule_dependent"] for r in rows), varies)
    late = {e["index"] for e in runs[0].events if e["schedule_dependent"]}
    check("no must dependence into an event not in every schedule",
          [r for r in rows if r["certainty"] == "must" and (r["src"] in late or r["dst"] in late)], [])

# the scoring pipeline (kdeps_score against gpu/nvbit_rw's truth format), on the interpreter's own access
# sets written as a truth: every dependence must come back, with its units
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp)
    kdeps.main([str(HERE / "bfs.cu"), str(out), "--no-host", "--access", "--argv", "bfs", "500"])
    with open(out / "bfs_objects.csv") as f:
        allocs = sorted((int(r["alloc"]), int(r["bytes"])) for r in csv.DictReader(f) if r["alloc"])
    with open(out / "truth_allocs.csv", "w") as f:
        f.write("index,base,bytes\n" + "".join(f"{i},0x{(i + 1) << 32:x},{b}\n" for i, b in allocs))
    with open(out / "bfs_events.csv") as f, open(out / "truth_events.csv", "w") as g:
        g.write("index,kind,name,grid,block\n")
        for r in csv.DictReader(f):
            g.write(f"{r['index']},{r['kind']},\"_Z{r['name']}\",{r['grid']},{r['block']}\n")
    (out / "truth_access.csv").write_text((out / "bfs_access.csv").read_text())
    ie, iedges = kdeps_score.interp_edges(out, "bfs")
    te, tedges = kdeps_score.truth_edges(out, 4)
    check("aligned", kdeps_score.aligned(ie, te), (True, ""))
    sc = kdeps_score.score(iedges, tedges)
    check("must precision, recall, units error", (sc["must_precision"], sc["recall"], sc["units_error"]),
          (1.0, 1.0, 0.0))

# the transitive reduction
check("reduce_edges", sorted(reduce_edges({(0, 1), (1, 2), (0, 2), (2, 3), (0, 3), (1, 3)})), [(0, 1), (1, 2), (2, 3)])

sys.exit(0 if ok else 1)
