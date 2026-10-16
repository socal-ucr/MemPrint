"""Score the static dependences between a CUDA program's kernels against a traced run (gpu/nvbit_rw).

    python -m memprint.static.kdeps_score INTERP_DIR NAME TRUTH_DIR [--unit 4] [--out scores.csv]

INTERP_DIR/NAME_*.csv come from `python -m memprint.static.kdeps SOURCE INTERP_DIR --no-host` (device
operations only); TRUTH_DIR/truth_*.csv from running the program under rw_trace.so (gpu/README.md). The
truth's per-event bytes are folded into dependences by the same Tracker, so the two sides differ only in
which bytes each event touched. Events are matched by their order among device operations (the run should
issue the same launches, copies and memsets in the same order: the check is reported; if it does not, as
when a loop the host repeats until the device says so runs a different number of rounds, the common start
and the common end are matched, kdeps.align_ends, and the run's dependences through events between count
as missed), objects by the
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
import json
import os
import sys

import numpy as np

from .interp_cuda import Event
from .kdeps import Tracker, align_ends, instantiate_loop, reduce_edges


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


def _same(x, y):
    return x["kind"] == y["kind"] and (x["kind"] != "kernel" or x["name"] in y["name"])


def aligned(interp_events, truth_events):
    """Do the two runs issue the same device operations in the same order (kinds, and kernel names)?"""
    a = [e for e in interp_events if e["kind"] != "host"]
    if len(a) != len(truth_events):
        return False, f"{len(a)} device events statically, {len(truth_events)} traced"
    for x, y in zip(a, truth_events):
        if not _same(x, y):
            return False, f"event {x['index']}: {x['kind']} {x['name']} statically, {y['kind']} {y['name']} traced"
    return True, ""


def truth_map(interp_events, truth_events):
    """The run's device events renumbered as the interpreter's (align_ends): {run index: interpreter index}."""
    a = [e for e in interp_events if e["kind"] != "host"]
    return {t: int(a[i]["index"]) for t, i in align_ends(truth_events, a, lambda y, x: _same(x, y)).items()}


def match_truth(interp_events, truth_events, truth):
    """The run's dependences with its events renumbered as the interpreter's (align_ends over device
    events); a dependence through an event without a match keeps a key no interpreter dependence has."""
    m = truth_map(interp_events, truth_events)
    return {(m.get(s, ("traced", s)), m.get(d, ("traced", d)), kind, obj): v
            for (s, d, kind, obj), v in truth.items()}


def loop_edges(interp_dir, name, interp_events, truth_events):
    """The interpreter's loop template (NAME_loop.json: the rounds that differ between schedules) instantiated
    for the traced run's rounds, as possible dependences keyed like match_truth's: {key: (0, units, 0)}; {} if
    there is no template or the run's events do not fit it."""
    path = os.path.join(interp_dir, f"{name}_loop.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        loop = json.load(f)
    a, p, post = loop["start"], loop["period"], loop["post"]
    n = len(truth_events)
    if n - post < a or (n - post - a) % p or not all(
            _same(loop["phases"][(i - a) % p], truth_events[i]) for i in range(a, n - post)):
        return {}
    with open(os.path.join(interp_dir, f"{name}_objects.csv")) as f:
        alloc = {r["name"]: int(r["alloc"]) for r in csv.DictReader(f) if r["alloc"] != ""}
    m = truth_map(interp_events, truth_events)
    return {(m.get(s, ("traced", s)), m.get(d, ("traced", d)), kind, alloc[obj]): (0, units, 0)
            for (s, d, kind, obj), units in instantiate_loop(loop, n).items() if obj in alloc}


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


def score_run(interp_dir, name, truth_dir, unit=4):
    """Score the interpreter's output in interp_dir against a traced run: (scores, interpreter dependences,
    run's dependences, why the event sequences differ or ""). Both keyed as match_truth's."""
    ie, iedges = interp_edges(interp_dir, name)
    te, tedges = truth_edges(truth_dir, unit)
    ok, why = aligned(ie, te)
    tneeded = match_truth(ie, te, needed(tedges))                     # reduced in the run's own numbering
    tedges = match_truth(ie, te, tedges)
    ineeded = needed(iedges)
    for k, v in loop_edges(interp_dir, name, ie, te).items():         # the rounds the run had, as possible
        if k not in iedges:
            iedges[k] = ineeded[k] = v
    row = {"aligned": int(ok)}
    for prefix, a, b in (("", iedges, tedges), ("needed_", ineeded, tneeded)):
        for k, v in score(a, b).items():
            row[prefix + k] = round(v, 4) if isinstance(v, float) else v
    return row, iedges, tedges, why


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("interp_dir")
    ap.add_argument("name")
    ap.add_argument("truth_dir")
    ap.add_argument("--unit", type=int, default=4)
    ap.add_argument("--out", help="append a row of scores to this CSV")
    args = ap.parse_args(argv)
    row, iedges, tedges, why = score_run(args.interp_dir, args.name, args.truth_dir, args.unit)
    if why:
        print(f"{args.name}: the device operations differ ({why}); the common start and end are matched",
              file=sys.stderr)
    row = {"name": args.name, **row}
    print(" ".join(f"{k}={v}" for k, v in row.items()))
    missed = sorted(set(tedges) - set(iedges), key=str)
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
