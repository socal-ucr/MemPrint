"""Accuracy data for paper_gpu/main.tex: the interpreter's dependences and address ranges against traced runs.

    python paper_gpu/accuracy.py [--skip-gpu]

Runs scripts/kdeps_truth_tests.sh into paper_gpu/data/runs/truth (every test program built, run under the
NVBit tool and analysed under the four thread schedules), traces cc REPS more times, runs it plainly RUNS
times for its round counts, and repeats the analysis with each of the interpreter's mechanisms turned off.
Writes, in paper_gpu/data/:

    scores.csv        dependence scores per program (kdeps_score)
    ranges.csv        per program, the units each event read and wrote: interpreter against trace
    datadep.csv       per program, kernel-to-kernel dependences through data-dependent references
    cc_runs.csv       cc: rounds and scores of each traced run, rounds of plain runs and of each schedule
    ablation.csv      scores with thread splitting off, and with the lockstep schedule only
    granularity.csv   dependences the traces imply when kernels are tracked by exact units, by one
                      interval per allocation, or by whole allocations
    patterns.csv      rodinia_bfs on three graphs: dependences shared exactly and by pattern

--skip-gpu reuses the traces already in data/runs.
"""

import csv
import io
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from memprint.static import interp_cuda as ic  # noqa: E402
from memprint.static import kdeps, kdeps_score as ks  # noqa: E402

CUDA = ROOT / "tests" / "static" / "cuda"
DATA = ROOT / "paper_gpu" / "data"
RUNS = DATA / "runs"
TRUTH = RUNS / "truth"
TOOL = ROOT / "gpu" / "nvbit_rw" / "rw_trace.so"
PY = sys.executable
REPS, PLAIN = 10, 50
COPIES = ("cudaMemcpy H2D", "cudaMemcpy D2H", "cudaMemcpy D2D", "cudaMemset")


def write(name, rows):
    with open(DATA / name, "w", newline="") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", name, len(rows), "rows")


def programs():
    return sorted(p.name for p in TRUTH.iterdir() if (p / "truth" / "truth_events.csv").exists())


def argv_of(name):
    """The command line kdeps_truth.sh ran the program with (its static.log does not keep it)."""
    return {"chain": ["1000"], "bfs": ["2000"], "fileidx": [str(TRUTH / "idx.txt"), str(TRUTH / "w.bin")],
            "rodinia_bfs": [str(TRUTH / "graph.txt")]}.get(name, [])


def score_dirs(static_dir, name, truth_dir):
    return ks.score_run(static_dir, name, truth_dir, 4)[0]


def analyse(name, out, schedules="lockstep,blocks,reverse,random"):
    """kdeps on a test program into out (quietly)."""
    with redirect_stdout(io.StringIO()):
        kdeps.main([str(CUDA / f"{name}.cu"), str(out), "--name", name, "--no-host", "--access",
                    "--schedules", schedules, "--argv", name, *argv_of(name)])


# ---------------------------------------------------------------- GPU runs

def gpu_runs():
    subprocess.run([str(ROOT / "scripts" / "kdeps_truth_tests.sh"), str(TRUTH)], check=True,
                   env=dict(os.environ, PYTHON=PY))
    exe = TRUTH / "cc" / "cc"
    for i in range(REPS):
        d = RUNS / "cc_reps" / str(i)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "log", "w") as f:
            subprocess.run([str(exe)], check=True, stdout=f, stderr=subprocess.STDOUT,
                           env=dict(os.environ, LD_PRELOAD=str(TOOL), KDEPS_OUT=str(d), NOBANNER="1"))
    plain = []
    for _ in range(PLAIN):
        out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
        plain.append(int(re.findall(r"probe (\d+)", out)[0]))
    (RUNS / "cc_plain_rounds.txt").write_text(" ".join(map(str, plain)) + "\n")


# ---------------------------------------------------------------- tables of results

def scores():
    rows = []
    for name in programs():
        r = {"program": name}
        r.update(score_dirs(TRUTH / name / "static", name, TRUTH / name / "truth"))
        rows.append(r)
    write("scores.csv", rows)


def units(path, event_key="event"):
    s = defaultdict(set)
    for r in csv.DictReader(open(path)):
        e, a, lo, hi, b = (int(r[k]) for k in (event_key, "alloc", "unit_lo", "unit_hi", "bits"))
        for bit, t in ((1, "R"), (2, "W")):
            if b & bit:
                s[(e, a, t)].update(range(lo, hi))
    return s


def ranges():
    rows = []
    for name in programs():
        st = units(TRUTH / name / "static" / f"{name}_access.csv")
        tr = units(TRUTH / name / "truth" / "truth_access.csv")
        keys = set(st) | set(tr)
        inter = sum(len(st.get(k, set()) & tr.get(k, set())) for k in keys)
        su, tu = sum(map(len, st.values())), sum(map(len, tr.values()))
        rows.append({"program": name, "sets": len(keys),
                     "exact_sets": sum(st.get(k, set()) == tr.get(k, set()) for k in keys),
                     "static_units": su, "traced_units": tu,
                     "unit_precision": round(inter / su, 4) if su else 1.0,
                     "unit_recall": round(inter / tu, 4) if tu else 1.0})
    write("ranges.csv", rows)


def datadep():
    rows = []
    for name in programs():
        d = TRUTH / name
        edges = list(csv.DictReader(open(d / "static" / f"{name}_edges.csv")))
        alloc = {r["name"]: int(r["alloc"]) for r in csv.DictReader(open(d / "static" / f"{name}_objects.csv"))
                 if r["alloc"]}
        ie, _ = ks.interp_edges(d / "static", name)
        te, tedges = ks.truth_edges(d / "truth", 4)
        tedges = ks.match_truth(ie, te, tedges)
        kk = [r for r in edges if r["src_name"] not in COPIES and r["dst_name"] not in COPIES and
              r["object"] in alloc]
        dd = [r for r in kk if int(r["units_dd"]) > 0]
        traced = lambda r: (int(r["src"]), int(r["dst"]), r["kind"], alloc[r["object"]]) in tedges  # noqa: E731
        rows.append({"program": name, "kernel_deps": len(kk), "data_deps": len(dd),
                     "data_must": sum(r["certainty"] == "must" for r in dd),
                     "data_must_traced": sum(r["certainty"] == "must" and traced(r) for r in dd),
                     "data_traced": sum(traced(r) for r in dd)})
    write("datadep.csv", rows)


def cc_runs():
    rows = []
    for d in sorted((RUNS / "cc_reps").iterdir(), key=lambda p: int(p.name)):
        rounds = int(re.findall(r"probe (\d+)", (d / "log").read_text())[0])
        r = {"kind": "traced", "run": d.name, "rounds": rounds}
        r.update(score_dirs(TRUTH / "cc" / "static", "cc", d))
        lock = RUNS / "cc_lockstep"
        if not lock.exists():
            analyse("cc", lock, "lockstep")
        r.update({f"lockstep_{k}": v for k, v in score_dirs(lock, "cc", d).items()})
        rows.append(r)
    for i, n in enumerate((RUNS / "cc_plain_rounds.txt").read_text().split()):
        rows.append({"kind": "plain", "run": i, "rounds": int(n)})
    for schedule in ic.SCHEDULES:
        for seed in (range(5) if schedule == "random" else [0]):
            it, _, _ = ic.analyze(CUDA / "cc.cu", argv=["cc"], host_events=False, schedule=schedule, seed=seed)
            rows.append({"kind": f"interp_{schedule}", "run": seed, "rounds": int(it.probes[0])})
    write("cc_runs.csv", rows)


def ablation():
    rows = []
    for label, setting, names in (("no thread splitting", {"MAX_TARGETS": 1}, ("ptrsel", "ptrsel2")),
                                  ("lockstep only", {}, ("cc",))):
        for name in names:
            out = RUNS / "ablation" / label.replace(" ", "_") / name
            saved = {k: getattr(ic, k) for k in setting}
            for k, v in setting.items():
                setattr(ic, k, v)
            try:
                analyse(name, out, "lockstep" if label == "lockstep only" else "lockstep,blocks,reverse,random")
            finally:
                for k, v in saved.items():
                    setattr(ic, k, v)
            for variant, d in (("full", TRUTH / name / "static"), (label, out)):
                r = {"ablation": label, "program": name, "variant": variant}
                r.update(score_dirs(d, name, TRUTH / name / "truth"))
                rows.append(r)
    write("ablation.csv", rows)


def granularity():
    rows = []
    for name in programs():
        t = TRUTH / name / "truth"
        acc = defaultdict(lambda: defaultdict(lambda: [set(), set()]))
        for r in csv.DictReader(open(t / "truth_access.csv")):
            e, a, lo, hi, b = (int(r[k]) for k in ("event", "alloc", "unit_lo", "unit_hi", "bits"))
            for bit in (1, 2):
                if b & bit:
                    acc[e][a][bit - 1].update(range(lo, hi))
        sizes = {int(r["index"]): -(-int(r["bytes"]) // 4) for r in csv.DictReader(open(t / "truth_allocs.csv"))}
        n = sum(1 for _ in csv.DictReader(open(t / "truth_events.csv")))
        row = {"program": name, "events": n}
        for how in ("exact", "interval", "allocation"):
            sets = {}
            for e, m in acc.items():
                sets[e] = {}
                for a, (rd, wr) in m.items():
                    if how == "interval":
                        f = lambda s: set(range(min(s), max(s) + 1)) if s else set()  # noqa: E731
                    elif how == "allocation":
                        f = lambda s, a=a: set(range(sizes[a])) if s else set()  # noqa: E731
                    else:
                        f = lambda s: s  # noqa: E731
                    sets[e][a] = (f(rd), f(wr))
            pairs = set()
            for j in range(n):
                for i in range(j):
                    for a, (rj, wj) in sets.get(j, {}).items():
                        ri, wi = sets.get(i, {}).get(a, (set(), set()))
                        if (wi & (rj | wj)) or (ri & wj):
                            pairs.add((i, j))
                            break
            row[f"{how}_pairs"] = len(pairs)
            row[f"{how}_needed"] = len(kdeps.reduce_edges(pairs))
        rows.append(row)
    write("granularity.csv", rows)


def patterns():
    """rodinia_bfs on three graphs: how many dependences carry over exactly, and how many by pattern
    (source and destination name, kind, object, distance in events)."""
    rows, per = [], {}
    for g, seed, n, lo, hi in (("graph A", 7, 4096, 1, 5), ("graph B", 11, 4096, 1, 5), ("graph C", 3, 8192, 2, 8)):
        random.seed(seed)
        adj = [[random.randrange(n) for _ in range(random.randint(lo, hi))] for _ in range(n)]
        starts = [0]
        for a in adj:
            starts.append(starts[-1] + len(a))
        path = RUNS / "patterns" / f"{g.replace(' ', '_')}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join([str(n)] + [f"{starts[v]} {len(adj[v])}" for v in range(n)] +
                                  ["0", str(starts[-1])] + [f"{u} 1" for a in adj for u in a]) + "\n")
        it, _, _ = ic.analyze(CUDA / "rodinia_bfs.cu", argv=["rodinia_bfs", str(path)], host_events=False)
        er = it.tracker.edge_rows()
        per[g] = er
        rows.append({"input": g, "vertices": n, "edges": starts[-1], "events": len(it.tracker.events),
                     "dependences": len(er), "needed": sum(r["needed"] for r in er)})
    pat = lambda r: (r["src_name"], r["dst_name"], r["kind"], r["object"], r["dst"] - r["src"])  # noqa: E731
    exact = lambda r: (r["src"], r["dst"], r["kind"], r["object"], r["units_must"] + r["units_may"])  # noqa: E731
    names = list(per)
    for r in rows:
        g = r["input"]
        others = [o for o in names if o != g]
        mine = {exact(x) for x in per[g]}
        r["exact_shared_with_all"] = len(mine.intersection(*[{exact(x) for x in per[o]} for o in others]))
        mp = {pat(x) for x in per[g] if x["needed"]}
        r["needed_patterns"] = len(mp)
        r["needed_patterns_in_all"] = len(mp.intersection(*[{pat(x) for x in per[o] if x["needed"]} for o in others]))
    # without the input (the graph file missing): which of the needed patterns shared by all three graphs, between
    # operations at most six apart (two rounds), the interpreter still reports, as certain or possible
    it, _, _ = ic.analyze(CUDA / "rodinia_bfs.cu", argv=["rodinia_bfs", str(RUNS / "patterns" / "missing.txt")],
                          host_events=False)
    er = it.tracker.edge_rows()
    shared = set.intersection(*[{pat(x) for x in per[g] if x["needed"]} for g in names])
    near = {p for p in shared if p[4] <= 6}
    rows.append({"input": "no input", "vertices": 0, "edges": 0, "events": len(it.tracker.events),
                 "dependences": len(er), "needed": sum(r["needed"] for r in er), "exact_shared_with_all": "",
                 "needed_patterns": len(near), "needed_patterns_in_all": len(near & {pat(x) for x in er}),
                 "possible": sum(r["certainty"] == "may" for r in er)})
    for r in rows[:-1]:
        r["possible"] = sum(x["certainty"] == "may" for x in per[r["input"]])
    write("patterns.csv", rows)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    if "--skip-gpu" not in sys.argv:
        gpu_runs()
    scores()
    ranges()
    datadep()
    cc_runs()
    ablation()
    granularity()
    patterns()


if __name__ == "__main__":
    main()
