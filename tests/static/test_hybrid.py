"""Check the hybrid interpreter (memprint.static.hybrid: host code native, kernels interpreted) against the
whole-program interpreter (memprint.static.kdeps) on the programs in tests/static/cuda/: the same operations and
the same dependences, between the same operations, through the same objects, with the same units and certainty.
Needs nvcc (to build each program as a library), not a GPU:  python tests/static/test_hybrid.py"""
import csv
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).with_name("cuda")
PY = sys.executable
ok = True


def check(what, got, expected):
    global ok
    good = got == expected
    ok &= good
    print(f"  {what}: got {got} expected {expected} {'ok' if good else 'FAIL'}")


def inputs(tmp):
    """The input files the programs read (as scripts/kdeps_truth_tests.sh makes them)."""
    random.seed(1)
    idx = random.sample(range(4096), 300)
    (tmp / "idx.txt").write_text(f"{len(idx)}\n" + " ".join(map(str, idx)) + "\n")
    (tmp / "w.bin").write_bytes(struct.pack("4096f", *[float(i % 5) for i in range(4096)]))
    random.seed(7)
    n = 4096
    adj = [[random.randrange(n) for _ in range(random.randint(1, 5))] for _ in range(n)]
    starts = [0]
    for a in adj:
        starts.append(starts[-1] + len(a))
    lines = [str(n)] + [f"{starts[v]} {len(adj[v])}" for v in range(n)] + ["0", str(starts[-1])] + \
        [f"{u} 1" for a in adj for u in a]
    (tmp / "graph.txt").write_text("\n".join(lines) + "\n")


def edges(out, name):
    """Dependences keyed by (src, dst, kind, object), objects named by allocation order (alloc<k>), by name for
    __device__ variables, and as 'host' for host memory (whose objects the two interpreters name differently)."""
    objs = list(csv.DictReader(open(out / f"{name}_objects.csv")))
    label = {r["name"]: f"alloc{r['alloc']}" if r["alloc"] else r["name"] if r["kind"] == "device-global" else "host"
             for r in objs}
    return {(r["src"], r["dst"], r["kind"], label.get(r["object"], r["object"])):
            (r["units_must"], r["units_may"], r["schedule_dependent"]) for r in csv.DictReader(open(out / f"{name}_edges.csv"))}


def run(tmp, name, argv, schedules="lockstep"):
    print(name, "under", schedules)
    full, hyb = tmp / f"{name}_full_{schedules}", tmp / f"{name}_hybrid_{schedules}"
    subprocess.run([PY, "-m", "memprint.static.kdeps", str(HERE / f"{name}.cu"), str(full), "--name", name,
                    "--no-host", "--schedules", schedules, "--argv", name, *argv], cwd=ROOT / "analysis",
                   check=True, capture_output=True)
    done = subprocess.run([PY, "-m", "memprint.static.hybrid", str(HERE / f"{name}.cu"), str(hyb), "--name", name,
                           "--schedules", schedules, "--argv", name, *argv], cwd=ROOT / "analysis",
                          capture_output=True, text=True)
    check("hybrid ran", done.returncode, 0)
    if done.returncode:
        print(done.stdout[-2000:], done.stderr[-2000:])
        return
    fe = [(r["kind"], r["name"]) for r in csv.DictReader(open(full / f"{name}_events.csv"))]
    he = [(r["kind"], r["name"]) for r in csv.DictReader(open(hyb / f"{name}_events.csv"))]
    check("operations", [k for k, _ in he], [k for k, _ in fe])
    f, h = edges(full, name), edges(hyb, name)
    check("dependences (same keys)", sorted(set(h) ^ set(f)), [])
    check("units and certainty", sorted(k for k in set(f) & set(h) if f[k] != h[k]), [])


def main():
    if shutil.which("nvcc") is None:
        print("nvcc not found: the hybrid interpreter builds each program with it (skipped)")
        return 0
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        inputs(tmp)
        for name, argv in (("chain", ["1000"]), ("reduce", []), ("stencil", []), ("scatter", []),
                           ("fileidx", [str(tmp / "idx.txt"), str(tmp / "w.bin")]), ("histogram", []),
                           ("bfs", ["2000"]), ("rodinia_bfs", [str(tmp / "graph.txt")]), ("wl_bfs", []),
                           ("ptrsel", []), ("ptrsel2", []), ("symbols", [])):
            run(tmp, name, argv)
        # every value is concrete natively: drand48's values are known, so maybe's dependences are must there
        print("maybe (the hybrid knows drand48's values)")
        run_maybe = subprocess.run([PY, "-m", "memprint.static.hybrid", str(HERE / "maybe.cu"), str(tmp / "maybe_h"),
                                    "--name", "maybe", "--argv", "maybe"], cwd=ROOT / "analysis", capture_output=True)
        check("hybrid ran", run_maybe.returncode, 0)
        rows = list(csv.DictReader(open(tmp / "maybe_h" / "maybe_edges.csv")))
        check("no may dependences", [r["object"] for r in rows if r["certainty"] == "may"], [])
        # thread order: several schedules, each run in a process of its own, merged as the whole-program interpreter does
        run(tmp, "cc", ["1024"], "lockstep,blocks,reverse,random")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
