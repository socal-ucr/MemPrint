"""Is an earlier run's RSS curve already enough to schedule on?

An earlier run's RSS (rssrep1: a repeat native run, not the RSS truth's own run) as a peak or as a curve, against
MemPrint's curve on its clock of references (W5), on the native wall clock (W5nat) and the hybrid (W5nat's shape at
the earlier RSS peak), in two settings:
- repeat: every estimate comes from a run of the job's own input;
- transfer: from the program's smaller inputs only, for its largest input (held out). Every method is
  transferred by one rule (transfer), fixed before looking at the results.

The error of each estimate is split into peak, timing and shape (decompose), and each method's acquisition cost
is the wall time of the runs it needs (costs).

    python -m memprint.sched.earlier build CORPUS OUT   ->  OUT/curves.csv, OUT/costs.csv (the study's corpus)
    python -m memprint.sched.earlier errors OUT          ->  OUT/errors.csv, OUT/acquisition.csv
"""

import os
import sys

import numpy as np
import pandas as pd

from . import oracles, real
from .ksplus import INPUT_SIZE
from .model import K
from .run import EXCLUDE, rank

SOURCES = ("rssrep1", "W5", "W5nat")   # the curves an estimate is built from
BAND = 0.25   # largest timing shift the alignment may undo, as a share of the run


def _rebased(g):
    """One (program, input)'s curves by method, rebased as make_jobs does."""
    return real.rebase({m: h.sort_values("k").bytes.to_numpy() for m, h in g.groupby("method")})


def transfer(curves, sources=SOURCES):
    """For each program with a smaller input, every source's curve at its largest input predicted from its smaller
    ones (method + "X"): the peak from a linear regression on the input size (as KS+ predicts its segments; with one
    smaller input, in proportion to the size), never below the largest peak seen; the shape is the largest smaller
    input's curve, scaled to that peak. A peak of 0 is no estimate: its oracle reserves the request."""
    rows = []
    c = curves[[(p, i) not in EXCLUDE for p, i in zip(curves.prog, curves.input)]]
    for prog, g in c.groupby("prog"):
        inputs = sorted(g.input.unique(), key=rank)
        if len(inputs) < 2 or prog not in INPUT_SIZE:
            continue
        target, train = inputs[-1], inputs[:-1]
        by = {i: _rebased(g[g.input == i]) for i in train}
        for m in sources:
            have = [i for i in train if m in by[i]]
            if not have:
                continue
            x = np.array([INPUT_SIZE[prog][i] for i in have], float)
            y = np.array([by[i][m].max() for i in have])
            xt = INPUT_SIZE[prog][target]
            if len(have) >= 2 and np.ptp(x) > 0:
                a, b = np.polyfit(x, y, 1)
                peak = a * xt + b
            else:
                peak = y[-1] * xt / x[-1]
            peak = max(peak, y.max())
            if peak <= 0:   # nothing above the runtime's own RSS at any smaller input: no estimate (the request)
                continue
            shape = by[have[-1]][m]
            rows.append(pd.DataFrame(dict(prog=prog, input=target, method=m + "X", k=np.arange(K),
                                          bytes=shape / max(shape.max(), 1e-12) * peak)))
    return pd.concat(rows, ignore_index=True)


def build(corpus, out):
    """The study's corpus: every program with a smaller input (darknet has none), with the transferred curves."""
    from .corpus import costs, curves
    os.makedirs(out, exist_ok=True)
    c = curves(corpus)
    multi = c.groupby("prog").input.nunique()
    c = c[c.prog.isin(multi[multi >= 2].index)]
    c = pd.concat([c, transfer(c)], ignore_index=True)
    c.to_csv(f"{out}/curves.csv", index=False)
    done = pd.concat([pd.read_csv(f, sep=r"\s+", header=None, dtype={1: str})
                      for f in (f"{corpus}/done.base.txt", f"{corpus}/done.txt") if os.path.exists(f)])
    done.to_csv(f"{out}/done.txt", sep=" ", header=False, index=False)
    costs(out).to_csv(f"{out}/costs.csv", index=False)
    return c


# --- the error of an estimate: peak, timing and shape

def required(r, t):
    """The headroom (a share of the reservation's peak, as run.Setup.required) with which r never falls below t."""
    return max(float(np.max((t - r) / max(r.max(), 1e-12))), 0.0)


def warp(e, t, band=BAND):
    """e aligned in time to t: the monotone alignment (dynamic time warping within +-band of the run) that
    minimises sum |e - t|; each sample of t takes the largest sample of e aligned to it."""
    n, w = len(t), max(1, int(band * len(t)))
    cost = np.abs(t[:, None] - e[None, :])
    D = np.full((n + 1, n + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        lo, hi = max(1, i - w), min(n, i + w)
        for j in range(lo, hi + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    out = np.zeros(n)
    i, j = n, n
    while i > 0 and j > 0:
        out[i - 1] = max(out[i - 1], e[j - 1])
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return out


def _estimate(kind, method, level, info):
    """The curve an oracle reserves from (a hybrid's at its level) and how it turns one into a reservation."""
    e = np.asarray(info[method], float)
    if level:
        e = e * np.max(info[level]) / max(e.max(), 1e-12)
    if kind == "peak":
        return e, lambda c: oracles.peak(c)
    return e, lambda c: oracles.curve(c, windows=20, widen=1, headroom=True)


def decompose(e, reserve, t):
    """The estimate e of the truth t, its reservation reserve(e), split three ways:
    H: the headroom it needs; Hp: with its peak set to the truth's; Hpt: with its peak set and aligned in time;
    then peak = H - Hp, timing = Hp - Hpt, shape = Hpt. The same split of the L1 error of the curve itself (E*) and
    of the reservation it takes to be safe (A*: the safe reservation's area over the truth's, minus 1)."""
    ep = e * t.max() / max(e.max(), 1e-12)
    ept = warp(ep, t)
    out = dict(peak_err=e.max() / t.max() - 1)
    for tag, c in (("", e), ("p", ep), ("pt", ept)):
        r = reserve(c)
        h = required(r, t)
        out["H" + tag] = h
        out["E" + tag] = float(np.abs(c - t).sum() / t.sum())
        out["A" + tag] = float((r + h * r.max()).sum() / t.sum() - 1)
    for x in "HEA":
        out[f"{x}_peak"], out[f"{x}_timing"], out[f"{x}_shape"] = (out[x] - out[x + "p"], out[x + "p"] - out[x + "pt"],
                                                                   out[x + "pt"])
    return out


STUDY = {"repeat": ["rss-peak-rep", "rss-curve-rep", "memprint-W5", "memprint-W5nat", "hybrid-rep"],
         "transfer": ["rss-peak-x", "rss-curve-x", "memprint-W5-x", "memprint-W5nat-x", "hybrid-x"]}
SPEC = {**oracles.EARLIER, "memprint-W5": ("curve", "W5", None), "memprint-W5nat": ("curve", "W5nat", None)}


def errors(out):
    """Per (truth, study, program, oracle): the decomposition at each program's largest input (scale 1)."""
    from .run import largest
    c = largest(pd.read_csv(f"{out}/curves.csv", dtype={"input": str}))
    rows = []
    for prog, g in c.groupby("prog"):
        info = _rebased(g)
        for scenario in ("rss", "bytes"):
            t = info[real.TRUTHS[scenario][0]]
            for study, names in STUDY.items():
                for name in names:
                    kind, method, level = SPEC[name]
                    if method not in info or (level and level not in info):
                        continue
                    e, reserve = _estimate(kind, method, level, info)
                    rows.append(dict(scenario=scenario, study=study, prog=prog, oracle=name, kind=kind,
                                     **decompose(e, reserve, t)))
    d = pd.DataFrame(rows)
    d.to_csv(f"{out}/errors.csv", index=False)
    return d


# --- what each estimate costs to get: the wall time of its runs, over the native run of the job's input

RUNS = {"rss": ["rssrep1"], "W5": ["W5"], "W5nat": ["W5", "nativerefs"], "hybrid": ["W5", "nativerefs", "rssrep1"]}


def acquisition(out):
    k = pd.read_csv(f"{out}/costs.csv", dtype={"input": str})
    k = k[k.exit == 0]
    wall = {(p, i, m): w for p, i, m, w in k[["prog", "input", "method", "wall_s"]].itertuples(index=False)}
    rows = []
    for prog, g in k.groupby("prog"):
        inputs = sorted((i for i in g.input.unique() if (prog, i) not in EXCLUDE), key=rank)
        target, train = inputs[-1], inputs[:-1]
        native = wall.get((prog, target, "native"))
        for what, runs in RUNS.items():
            for study, ins in (("repeat", [target]), ("transfer", train)):
                s = [wall.get((prog, i, m)) for i in ins for m in runs]
                if None in s or not native:
                    continue
                rows.append(dict(prog=prog, study=study, source=what, seconds=sum(s), native_s=native,
                                 x_native=sum(s) / native))
    d = pd.DataFrame(rows)
    d.to_csv(f"{out}/acquisition.csv", index=False)
    return d


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "build":
        build(*args)
    elif cmd == "errors":
        errors(*args)
        acquisition(*args)
