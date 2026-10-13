"""Check the CUDA interpreter (interp_cuda) and the dependences between kernels (kdeps) on the programs in
tests/static/cuda/: values the programs compute (probe), launches, and the dependences each program's
comment describes, with their units, certainty and input dependence. No GPU is needed:
python tests/static/test_cuda.py"""
import csv
import struct
import sys
import tempfile
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
    probes = [float(p.v) if isinstance(p, ic.Real) else int(p) for p in it.probes]
    return it, probes, it.tracker.edge_rows()


def edge(rows, src, dst, kind, obj):
    """The (units must, units may, input dependent, needed) of a dependence between events src -> dst."""
    for r in rows:
        if r["src"] == src and r["dst"] == dst and r["kind"] == kind and r["object"].split("@")[0] == obj:
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

# a branch on values the interpreter does not know: may dependences; the unconditional write stays must
it, probes, rows = run("maybe.cu")
t, c = index(it, "threshold"), index(it, "count")
check("threshold -> count through flag (may)", edge(rows, t, c, "RAW", "flag"), (0, 512, 1, 1))
check("threshold -> count through seen (must)", edge(rows, t, c, "RAW", "seen"), (512, 0, 0, 1))

# the scoring pipeline (kdeps_score against gpu/nvbit_rw's truth format), on the interpreter's own access
# sets written as a truth: every dependence must come back, with its units
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp)
    kdeps.main([str(HERE / "bfs.cu"), str(out), "--argv", "bfs", "500", "--no-host", "--access"])
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
