"""Score the static dependences between a CUDA program's kernels against a traced run (gpu/nvbit_rw).

    python -m memprint.static.kdeps_score INTERP_DIR NAME TRUTH_DIR [--unit 4] [--out scores.csv]

INTERP_DIR/NAME_*.csv come from `python -m memprint.static.kdeps SOURCE INTERP_DIR --no-host` (device
operations only); TRUTH_DIR/truth_*.csv from running the program under rw_trace.so (gpu/README.md). The
truth's per-event bytes are folded into dependences by the same Tracker, so the two sides differ only in
which bytes each event touched. Events are matched by their order among device operations (the run must
issue the same launches, copies and memsets in the same order: the check is reported), objects by the
order of their device allocations (cudaMalloc / cudaMallocManaged); dependences through other objects
(__device__ variables, pinned host memory) are left out.

Scores, over dependences (src, dst, kind, allocation):
    must_precision  the interpreter's must dependences that the run has
    recall          the run's dependences the interpreter has, must or may
    must_recall     ... has as must
    units_error     over dependences both have: |units (must + may) - units in the run| / units in the run
and the same over the needed dependences (the transitive reductions).
"""

import argparse
import csv
import os
import sys

import numpy as np

from .interp_cuda import Event
from .kdeps import Tracker, reduce_edges


class _Alloc:
    """An allocation of the traced run, standing in for an interpreter object."""

    def __init__(self, index, nbytes):
        self.name, self.kind, self.nbytes, self.alloc_index = f"alloc{index}", "device", nbytes, index


def truth_edges(truth_dir, unit):
    """The traced run's events and dependences: ([event rows], {(src, dst, kind, alloc): units})."""
    with open(os.path.join(truth_dir, "truth_allocs.csv")) as f:
        allocs = {int(r["index"]): _Alloc(int(r["index"]), int(r["bytes"])) for r in csv.DictReader(f)}
    with open(os.path.join(truth_dir, "truth_events.csv")) as f:
        events = list(csv.DictReader(f))
    access = {}
    with open(os.path.join(truth_dir, "truth_access.csv")) as f:
        for r in csv.DictReader(f):
            access.setdefault(int(r["event"]), []).append((int(r["alloc"]), int(r["unit_lo"]), int(r["unit_hi"]),
                                                           int(r["bits"])))
    tracker = Tracker(unit)
    for row in events:
        i = int(row["index"])
        ev = Event(i, row["kind"], row["name"], "")
        for a, lo, hi, bits in access.get(i, ()):
            obj = allocs[a]
            ev.mark_range(obj, lo * unit, (hi - lo) * unit, bits)
        tracker.add(ev)
    edges = {}
    for (s, d, kind, obj), (must, may, _) in tracker.edges.items():
        edges[(s, d, kind, int(obj[len("alloc"):]))] = must + may
    return events, edges


def interp_edges(interp_dir, name):
    """The interpreter's events and dependences through device allocations:
    ([event rows], {(src, dst, kind, alloc): (units must, units may, input dependent)})."""
    with open(os.path.join(interp_dir, f"{name}_events.csv")) as f:
        events = list(csv.DictReader(f))
    with open(os.path.join(interp_dir, f"{name}_objects.csv")) as f:
        alloc = {r["name"]: int(r["alloc"]) for r in csv.DictReader(f) if r["alloc"] != ""}
    edges = {}
    with open(os.path.join(interp_dir, f"{name}_edges.csv")) as f:
        for r in csv.DictReader(f):
            if r["object"] in alloc:
                edges[(int(r["src"]), int(r["dst"]), r["kind"], alloc[r["object"]])] = (
                    int(r["units_must"]), int(r["units_may"]), int(r["input_dependent"]))
    return events, edges


def aligned(interp_events, truth_events):
    """Do the two runs issue the same device operations in the same order (kinds, and kernel names)?"""
    a = [e for e in interp_events if e["kind"] != "host"]
    if len(a) != len(truth_events):
        return False, f"{len(a)} device events statically, {len(truth_events)} traced"
    for x, y in zip(a, truth_events):
        if x["kind"] != y["kind"] or (x["kind"] == "kernel" and x["name"] not in y["name"]):
            return False, f"event {x['index']}: {x['kind']} {x['name']} statically, {y['kind']} {y['name']} traced"
    return True, ""


def score(interp, truth):
    must = {k for k, (m, _, _) in interp.items() if m}
    anyk = set(interp)
    t = set(truth)
    err = [abs(sum(interp[k][:2]) - truth[k]) for k in anyk & t]
    return {"truth": len(t), "must": len(must), "may": len(anyk - must),
            "must_precision": len(must & t) / len(must) if must else 1.0,
            "recall": len(t & anyk) / len(t) if t else 1.0,
            "must_recall": len(t & must) / len(t) if t else 1.0,
            "units_error": float(np.sum(err)) / max(1, sum(truth[k] for k in anyk & t))}


def needed(edges):
    keep = reduce_edges({(k[0], k[1]) for k in edges})
    return {k: v for k, v in edges.items() if (k[0], k[1]) in keep}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("interp_dir")
    ap.add_argument("name")
    ap.add_argument("truth_dir")
    ap.add_argument("--unit", type=int, default=4)
    ap.add_argument("--out", help="append a row of scores to this CSV")
    args = ap.parse_args(argv)
    ie, iedges = interp_edges(args.interp_dir, args.name)
    te, tedges = truth_edges(args.truth_dir, args.unit)
    ok, why = aligned(ie, te)
    if not ok:
        print(f"{args.name}: the device operations differ ({why}); scores compare events by position",
              file=sys.stderr)
    row = {"name": args.name, "aligned": int(ok)}
    for prefix, a, b in (("", iedges, tedges), ("needed_", needed(iedges), needed(tedges))):
        for k, v in score(a, b).items():
            row[prefix + k] = round(v, 4) if isinstance(v, float) else v
    print(" ".join(f"{k}={v}" for k, v in row.items()))
    missed = sorted(set(tedges) - set(iedges))
    wrong = sorted(k for k, v in iedges.items() if v[0] and k not in tedges)
    for label, keys in (("missed", missed), ("must but not traced", wrong)):
        for k in keys[:20]:
            print(f"  {label}: {k[0]} -> {k[1]} {k[2]} alloc{k[3]}")
    if args.out:
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            wr = csv.DictWriter(f, list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
