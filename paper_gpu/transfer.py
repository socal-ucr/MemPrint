"""Dependences before the input is known: patterns learned on one input, applied to another.

    python paper_gpu/transfer.py

For each benchmark run on both input sets (data/bench and data/bench_large), the hybrid interpreter's dependences
on one input are turned into patterns over the structure of the program's operations, and the patterns are applied
to the operations the program issues on the other input, without running the interpreter on it. The operations are
split into segments: a single operation, or a loop (a run of at least two rounds of the same operations). An
operation's place is its segment, its phase in a loop, and its round, counted from the loop's start, or from its
end when the dependence leaves the loop. A dependence becomes a pattern between two places; between rounds of one
loop its distance in rounds; where the learned input showed it for every round (or every distance) the pattern
holds for every round of the other input. An operation the learned structure does not account for (the other input
issues another sequence), or that ran no thread on the learned input (a launch with an invalid configuration), waits
for every operation before it, as in issue order, and every later one for it.

Scored against the traced run on the other input (missed dependences, dependences that did not occur) and simulated
as simulate.py does, a missed dependence the run violates causing a rollback. Writes data/transfer.csv.
"""

import csv
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "analysis"))
from memprint.static import kdeps_score as ks  # noqa: E402

DATA = HERE / "data"
DIRS = {"small": DATA / "bench", "large": DATA / "bench_large"}
MAX_PERIOD = 16


def sig(e):
    return (e["kind"], e["name"].split("(")[0])


def segments(sigs):
    """[(start, period, rounds)]: greedy from the left, the period whose repetition covers most operations (at least
    two rounds), else a single operation (period 1, one round)."""
    out, i, n = [], 0, len(sigs)
    while i < n:
        best = (1, 1)                                                  # (period, rounds)
        for p in range(1, min(MAX_PERIOD, (n - i) // 2) + 1):
            k = 0
            while i + k + p < n and sigs[i + k] == sigs[i + k + p]:
                k += 1
            rounds = (k + p) // p
            if rounds >= 2 and rounds * p > best[0] * best[1]:
                best = (p, rounds)
        out.append((i, best[0], best[1]))
        i += best[0] * best[1]
    return out


def places(sigs):
    """Per operation: (segment, phase, round, rounds of its segment), and the segments' phase signatures."""
    segs = segments(sigs)
    where, shape = [], []
    for k, (start, p, rounds) in enumerate(segs):
        shape.append(tuple(sigs[start: start + p]) + (("loop",) if rounds > 1 else ()))
        for i in range(p * rounds):
            where.append((k, i % p, i // p, rounds))
    return where, shape


def _general(vals, n, lo_ok=0):
    """A set of rounds (or distances) seen on an input whose loop had n rounds: ("every", k) if it holds from k up to
    the end (n - 1) without a gap, since a longer loop would go on the same way, with the values below k it holds
    for too: ("every", k, set); else ("exact", set)."""
    vals = set(vals)
    if n > 2:
        for k in range(lo_ok, n):
            if set(range(k, n)) <= vals and (k == lo_ok or k - 1 not in vals):
                return ("every", k, {v for v in vals if v < k})
    return ("exact", vals)


def _holds(g, x):
    return (x >= g[1] or x in g[2]) if g[0] == "every" else x in g[1]


def learn(events, pairs):
    """Patterns from dependences (pairs of operation indices) on one input."""
    sigs = [sig(e) for e in events]
    where, shape = places(sigs)
    seen = {}
    for s, d in pairs:
        ks_, ps, rs, ns = where[s]
        kd, pd, rd, nd = where[d]
        if ks_ == kd:                                                  # within a loop: the distance in rounds
            key = ("in", ks_, ps, pd)
            seen.setdefault(key, set()).add(rd - rs)
        else:                                                          # between segments: the rounds, the
            key = ("across", ks_, ps, kd, pd)                          # source's counted from its loop's end
            seen.setdefault(key, set()).add((ns - 1 - rs, rd))
    idle = {}                                                          # launches that ran no thread here (an
    for i, e in enumerate(events):                                     # invalid configuration): nothing learned,
        if e["kind"] == "kernel" and str(e.get("threads", "1")) == "0":  # per round counted from the loop's end
            k, ph, r, rounds = where[i]
            idle.setdefault((k, ph), set()).add(rounds - 1 - r)
    for key, ends in idle.items():                                     # idle in every round: in every round
        rounds = where[[i for i, w in enumerate(where) if w[0] == key[0]][0]][3]
        if len(ends) == rounds:
            idle[key] = "every"
    patterns = {}
    for key, vals in seen.items():
        if key[0] == "in":
            n = where[[i for i, w in enumerate(where) if w[0] == key[1]][0]][3]
            patterns[key] = _general(vals, n, 1 if key[2] >= key[3] else 0)
        else:
            ns = where[[i for i, w in enumerate(where) if w[0] == key[1]][0]][3]
            nd = where[[i for i, w in enumerate(where) if w[0] == key[3]][0]][3]
            patterns[key] = (_general({a for a, _ in vals}, ns), _general({b for _, b in vals}, nd))
    return {"shape": shape, "patterns": patterns, "idle": idle}


def apply(model, events):
    """Predicted predecessors {op: set} for the operations of another input."""
    sigs = [sig(e) for e in events]
    where, shape = places(sigs)
    n = len(events)
    known = 0                                                          # the segments the learned structure covers
    while known < min(len(shape), len(model["shape"])) and shape[known] == model["shape"][known]:
        known += 1
    preds = {j: set() for j in range(n)}
    by_place = {}
    for i, (k, ph, r, rounds) in enumerate(where):
        by_place.setdefault((k, ph), []).append((r, i))
    for key, pat in model["patterns"].items():
        if key[0] == "in":
            _, k, ps, pd = key
            if k >= known:
                continue
            src = dict(by_place.get((k, ps), []))
            for rd, d in by_place.get((k, pd), []):
                for rs, s in src.items():
                    dist = rd - rs
                    if s < d and _holds(pat, dist):
                        preds[d].add(s)
        else:
            _, ks_, ps, kd, pd = key
            if ks_ >= known or kd >= known:
                continue
            srcs, dsts = by_place.get((ks_, ps), []), by_place.get((kd, pd), [])
            ns = len(srcs)
            for rd, d in dsts:
                if not _holds(pat[1], rd):
                    continue
                for rs, s in srcs:
                    if s < d and _holds(pat[0], ns - 1 - rs):
                        preds[d].add(s)
    for i, (k, ph, r, rounds) in enumerate(where):                     # a launch learned from one that did
        ends = model.get("idle", {}).get((k, ph))                      # nothing: issue order around it
        if ends == "every" or (ends and rounds - 1 - r in ends):
            preds[i] = set(range(i))
            for j in range(i + 1, n):
                preds[j].add(i)
    first_unknown = next((i for i, w in enumerate(where) if w[0] >= known), n)
    for j in range(first_unknown, n):                                  # not accounted for: issue order
        preds[j] = set(range(j))
    return preds, known, len(shape), first_unknown


def main():
    import simulate as S
    import benchmarks as B
    rows = []
    gaps = []
    for f in ("sim_gaps.json", "sim_gaps_large.json"):
        if (DATA / f).exists():
            gaps += [x for v in json.load(open(DATA / f)).values() for x in v]
    delay = statistics.median(gaps) if gaps else 10000
    names = sorted(n.name for n in DIRS["small"].iterdir() if (DIRS["large"] / n.name / "hybrid").exists()
                   and (n / "hybrid" / f"{n.name}_edges.csv").exists())
    for name in names:
        for train, target in (("small", "large"), ("large", "small")):
            tw, gw = DIRS[train] / name, DIRS[target] / name
            iev, iedges = ks.interp_edges(tw / "hybrid", name)
            iev = [e for e in iev if e["kind"] != "host"]
            model = learn(iev, {(s, d) for s, d, _, _ in iedges})
            tev = list(csv.DictReader(open(gw / "truth" / "truth_events.csv")))
            preds, known, segs, first_unknown = apply(model, tev)
            _, tedges = ks.truth_edges(gw / "truth", 4)
            truth = {(s, d) for s, d, _, _ in tedges}
            pred = {(s, d) for d, ss in preds.items() for s in ss}
            ops = S.load_ops_dir(gw, name) if hasattr(S, "load_ops_dir") else None
            row = {"name": name, "train": train, "target": target, "ops": len(tev), "segments": segs,
                   "covered_segments": known, "uncovered_ops": len(tev) - first_unknown,
                   "truth_pairs": len(truth), "missed": len(truth - pred), "extra": len(pred - truth)}
            if ops is not None:
                n = len(tev)
                tp = {j: set() for j in range(n)}
                for s, d in truth:
                    tp[d].add(s)
                seq = {j: ({j - 1} if j else set()) for j in range(n)}
                base, _, _ = S.simulate(ops, seq, tp, delay)
                for policy, pp in (("patterns", preds), ("truth", tp)):
                    t, rb, _ = S.simulate(ops, pp, tp, delay)
                    row[f"{policy}_speedup"], row[f"{policy}_rollbacks"] = round(base / t, 4), rb
            rows.append(row)
            print(row, flush=True)
    with open(DATA / "transfer.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", DATA / "transfer.csv")


if __name__ == "__main__":
    main()
