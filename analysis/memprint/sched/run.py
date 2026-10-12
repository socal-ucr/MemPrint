"""Scheduling experiments on real traces with measured footprints.

    python -m memprint sched run --corpus DIR --traces DIR --out results.csv [--loads ...] [--seeds ...]

For every (scenario, load, seed, margin, oracle) one job stream is replayed under one oracle. A stream is n
consecutive single-node jobs of the MetaCentrum2 log (arrivals rescaled to the offered load), each with the
memory curve of a corpus program at its largest input (real.make_jobs).
"""

from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import ml, oracles, real
from .model import Cluster, Job, at_slots
from .sim import Sim
from copy import copy

ORDER = {"SMALL": 0, "MEDIUM": 1, "LARGE": 2, "S": 0, "W": 1, "A": 2, "small": 0, "medium": 1, "large": 2}
EXCLUDE = {("npb-mg", "A")}   # the splitter ran out of memory: no truth
EXACT = {"exact", "exact-coarse", "exact-peak"}
CALIBRATED = {"memprint-W1", "memprint-W5", "memprint-P", "memprint-W5-peak", "memprint-static", "massif",
              "massif-pages", "massif-i", "massif-pages-i", "heaptrack", "dhat", "dhat-curve", "dhat-hold", "gperf", "rss-peak"}
ORACLES = ["exact", "exact-coarse", "exact-peak", "memprint-static", "memprint-W1", "memprint-W5", "memprint-P", "memprint-W5-peak", "massif",
           "massif-pages", "heaptrack", "dhat", "dhat-curve", "dhat-hold", "gperf", "rss-peak", "request", "ml-peak",
           "reactive"]


def rank(s):
    try:
        return float(s)
    except ValueError:
        return ORDER.get(s, 0)


def largest(curves, require=None):
    """Per program, the curves of its largest input (with a `require` method's curve, if given)."""
    c = curves[[(p, i) not in EXCLUDE for p, i in zip(curves.prog, curves.input)]].copy()
    if require:
        have = set(map(tuple, c[c.method == require][["prog", "input"]].drop_duplicates().to_numpy()))
        c = c[[(p, i) in have for p, i in zip(c.prog, c.input)]]
    c["r"] = c.input.map(rank)
    return c[c.r == c.groupby("prog").r.transform("max")].drop(columns="r")


class Setup:
    """Everything a run needs, loaded once per worker."""

    def __init__(self, corpus, traces, n, cluster, kind="synthetic", programs=None):
        self.cluster = cluster
        # programs: None (every program, largest input) or "static" (the programs the static route predicts, at
        # their largest input with a prediction)
        self.panel = programs or "all"
        self.curves = largest(pd.read_csv(f"{corpus}/curves.csv", dtype={"input": str}),
                              require="static" if programs == "static" else None)
        cm5 = real.cm5_ratios(f"{traces}/LANL-CM5-1994-4.1-cln.swf.gz")
        self.cm5_train, self.cm5_test = ml.split_users(cm5)
        self.model = ml.PeakModel(self.cm5_train)
        self.trace_path = f"{traces}/METACENTRUM-2013-3.swf.gz"
        self.n = n
        self.kind = kind
        self._trace = {}
        self._required = {}

    def stream(self, scenario, load, seed):
        # the stream and its arrival rate are set against the default cluster, so that a sweep over the memory
        # size replays the same jobs at the same rate
        ref = Cluster()
        if seed not in self._trace:
            self._trace[seed] = (real.synthetic_trace(self.n, seed, cluster=ref) if self.kind == "synthetic"
                                 else real.trace_jobs(self.trace_path, ref, self.n, seed))
        jobs = real.make_jobs(self._trace[seed], self.curves, self.cm5_test, scenario, seed=seed)
        gbmin = sum(at_slots(j.resident, j.base).sum() for j in jobs)
        t = real.scale_arrivals([j.arrival for j in jobs], load, ref, gbmin)
        for j, a in zip(jobs, t):
            j.arrival = int(a)
        return jobs

    def required(self, scenario, name):
        """Per program: the smallest margin with which the oracle never falls below the truth (the program's
        curves at scale 1; the ratio does not depend on the scale)."""
        key = (scenario, name)
        if key not in self._required:
            truth_method = "rss" if scenario == "rss" else "truth"
            out = {}
            for prog, g in self.curves.groupby("prog"):
                wide = {m: h.sort_values("k").bytes.to_numpy() for m, h in g.groupby("method")}
                if "rss" in wide:
                    wide["rss"] = np.maximum(wide["rss"] - wide.get("rssbase", wide["rss"]).min(), 0)  # as make_jobs
                if truth_method not in wide:
                    continue
                j = Job(id=0, arrival=0, cores=1, base=60, resident=wide[truth_method], info=dict(wide))
                if not oracles.available([j], name):
                    continue
                r = oracles.REAL[name](j, 0.0)
                # margin is headroom as a share of the reservation's peak (equal to a factor for a flat one)
                out[prog] = max(float(np.max((j.resident - r) / max(r.max(), 1e-12))), 0.0)
            self._required[key] = out
        return self._required[key]

    def calibrated(self, scenario, name, prog, q=0.9):
        """The q-quantile of the required margins of the other programs (leave one program out)."""
        req = self.required(scenario, name)
        return float(np.quantile([m for p, m in req.items() if p != prog], q))

    def oracle(self, name):
        return ml.oracle(self.model) if name == "ml-peak" else oracles.REAL[name]


_setup = None


def _init(corpus, traces, n, cluster, kind="synthetic", programs=None):
    global _setup
    _setup = Setup(corpus, traces, n, cluster, kind, programs)


def steady(d, sim, trim=0.1):
    """Metrics of the middle of the stream (warm-up and drain left out): per-job metrics of the jobs that
    arrived in it, and memory reserved / resident over that time as a share of all memory."""
    lo, hi = d.arrival.quantile(trim), d.arrival.quantile(1 - trim)
    m = d[(d.arrival >= lo) & (d.arrival <= hi)]
    c, w = sim.c, slice(int(lo), int(hi) + 1)
    cap = c.nodes * c.local + c.pool
    reserved = (sim.local[:, w].sum(0) + sim.pool[w]).mean() / cap
    return dict(steady_jobs=len(m), steady_reserved=reserved, steady_resident=sim.resident[w].mean() / cap,
                steady_cores=sim.cores[:, w].sum(0).mean() / (c.nodes * c.cores), steady_response=m.response.mean(), steady_p95=m.response.quantile(0.95),
                steady_wait=(m.start - m.arrival).mean(), steady_bsld=m.bsld.mean(),
                steady_stretch=(m.runtime / m.base).mean())


def _one(task):
    try:
        return _run_one(task)
    except Exception as e:  # one bad task must not lose the grid
        import traceback
        return dict(zip(("scenario", "load", "seed", "margin", "oracle"), task), error=repr(e),
                    trace=traceback.format_exc()[-500:])


def _run_one(task):
    scenario, load, seed, margin, name = task
    jobs = _setup.stream(scenario, load, seed)
    if name != "ml-peak" and not oracles.available(jobs, name):
        return None
    f = _setup.oracle(name)
    if margin in ("cal", "cal100"):
        q = 0.9 if margin == "cal" else 1.0
        if name in CALIBRATED:
            base = f
            f = lambda j, m: base(j, _setup.calibrated(scenario, name, j.info["prog"], q))
            f.elastic = getattr(base, "elastic", False)
        else:
            f, margin = f, 0.1   # oracles without a profile to calibrate (request, ML, reactive) keep 10%
    jobs = oracles.assign([copy(j) for j in jobs], f, 0.0 if margin in ("cal", "cal100") else margin)
    sim = Sim(_setup.cluster, jobs)
    d, s = sim.run()
    return dict(stream=_setup.kind, programs=_setup.panel, local_gb=_setup.cluster.local, pool_gb=_setup.cluster.pool,
                scenario=scenario, load=load, seed=seed, margin=margin, oracle=name, **s,
                **steady(d, sim))


def run(corpus, traces, out, scenarios=("rss", "bytes"), loads=(0.5, 0.7, 0.9, 1.2), seeds=(0, 1, 2),
        margins=(0.0, 0.1, 0.3), names=ORACLES, n=3000, cluster=None, procs=24, kind="synthetic", programs=None):
    cluster = cluster or Cluster()
    tasks = list(product(scenarios, loads, seeds, margins, names))
    # exact and exact-peak do not use a margin
    tasks = [t for t in tasks if not (t[4] in EXACT and t[3] != margins[0])]
    rows = []
    with Pool(procs, _init, (corpus, traces, n, cluster, kind, programs)) as p, open(out + ".jsonl", "a") as log:
        for r in p.imap_unordered(_one, tasks, chunksize=1):
            if r:
                rows.append(r)
                log.write(pd.Series(r).to_json() + "\n")   # progress, and the rows so far if the run dies
                log.flush()
    df = pd.DataFrame(rows).sort_values(["scenario", "load", "oracle", "margin", "seed"])
    df.to_csv(out, index=False)
    return df
