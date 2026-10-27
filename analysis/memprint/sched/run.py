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

from . import ksplus, ml, oracles, real
from .model import Cluster, Job, at_slots
from .sim import Sim
from copy import copy

ORDER = {"SMALL": 0, "MEDIUM": 1, "LARGE": 2, "S": 0, "W": 1, "A": 2, "small": 0, "medium": 1, "large": 2,
         "15M": 0, "42M": 1, "110M": 2,                                  # llama2.c models
         "135m-n32": 0, "135m-n128": 1, "360m-n64": 2}                   # llama.cpp model and tokens
LLM = ("llama2c", "llamacpp")
EXCLUDE = {("npb-mg", "A"), ("llamacpp", "360m-n64")}   # the splitter ran out of memory: no truth
EXACT = {"exact", "exact-coarse", "exact-peak"}
NO_MARGIN = EXACT | {"pond", "zacarias", "zacarias-past", "ksplus"}   # run once: they do not use a margin (KS+ has its own)
LEARNED = {"ml-peak", "pond", "ksplus"}
CALIBRATED = {"memprint-W1", "memprint-W5", "memprint-P", "memprint-W5-peak", "memprint-static", "massif",
              "massif-pages", "massif-i", "massif-pages-i", "heaptrack", "dhat", "dhat-curve", "dhat-hold", "gperf", "rss-peak",
              *(f"memprint-{n}" for n in oracles.CLOCK_PAGES), *oracles.HYBRID, *oracles.EARLIER}
ORACLES = ["exact", "exact-coarse", "exact-peak", "memprint-static", "memprint-W1", "memprint-W5", "memprint-P", "memprint-W5-peak", "massif",
           "massif-pages", "heaptrack", "dhat", "dhat-curve", "dhat-hold", "gperf", "rss-peak", "request", "ml-peak",
           "reactive", "pond", "zacarias", "zacarias-past", "ksplus"]


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
        # programs: None (every program, largest input), "static" (the programs the static route predicts, at
        # their largest input with a prediction) or "llm" (the LLM engines at every input, each (engine, input) a
        # program of its own, so that a margin is calibrated on the other models and lengths)
        self.panel = programs or "all"
        all_curves = pd.read_csv(f"{corpus}/curves.csv", dtype={"input": str})
        if programs == "llm":
            keep = np.array([(p, i) not in EXCLUDE for p, i in zip(all_curves.prog, all_curves.input)], bool)
            c = all_curves[all_curves.prog.isin(LLM).to_numpy() & keep].copy()
            targets = [(p, i, f"{p}-{i}") for p, i in c[["prog", "input"]].drop_duplicates().to_numpy()]
            c["prog"] = c.prog + "-" + c.input
            self.curves = c
        else:
            self.curves = largest(all_curves, require="static" if programs == "static" else None)
            targets = [(p, i, p) for p, i in self.curves[["prog", "input"]].drop_duplicates().to_numpy()]
        # KS+: each job's program and input predicted from the program's other inputs
        history = all_curves[[(p, i) not in EXCLUDE for p, i in zip(all_curves.prog, all_curves.input)]]
        self.ksplus = ksplus.models(history, pd.read_csv(f"{corpus}/costs.csv", dtype={"input": str}), targets,
                                    real.rebase)
        cm5 = real.cm5_ratios(f"{traces}/LANL-CM5-1994-4.1-cln.swf.gz")
        self.cm5_train, self.cm5_test = ml.split_users(cm5)
        self.model = ml.PeakModel(self.cm5_train)
        self.pond_model = ml.PeakModel(self.cm5_train, quantile=0.98)   # Pond's TP = 98%
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
            truth_method = real.TRUTHS[scenario][0]
            out = {}
            for prog, g in self.curves.groupby("prog"):
                wide = {m: h.sort_values("k").bytes.to_numpy() for m, h in g.groupby("method")}
                wide = real.rebase(wide)  # as make_jobs
                if truth_method not in wide:
                    continue
                info = dict(wide)
                if real.rss_view(info, scenario) is not None:
                    info["rssview"] = real.rss_view(info, scenario)
                j = Job(id=0, arrival=0, cores=1, base=60, resident=wide[truth_method], info=info)
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
        if name == "ml-peak":
            return ml.oracle(self.model)
        if name == "pond":
            return ml.pond(self.pond_model)
        if name == "ksplus":
            return ksplus.oracle(self.ksplus)
        return oracles.REAL[name]


_setup = None


def _init(corpus, traces, n, cluster, kind="synthetic", programs=None, interval=None):
    global _setup
    if interval:   # the dynamic policies' update interval (minutes), for the sensitivity runs
        oracles.INTERVAL = interval
    if kind.startswith("lisa"):   # the SURF Lisa trace (lisa_run.py)
        from .lisa_run import LisaSetup
        _setup = LisaSetup(corpus, traces, n, cluster, kind, programs)
    else:
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
    if name not in LEARNED and not oracles.available(jobs, name):
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
        margins=(0.0, 0.1, 0.3), names=ORACLES, n=3000, cluster=None, procs=24, kind="synthetic", programs=None,
        interval=None):
    cluster = cluster or Cluster()
    tasks = list(product(scenarios, loads, seeds, margins, names))
    # the exact oracles, Pond and dynamic provisioning do not use a margin
    tasks = [t for t in tasks if not (t[4] in NO_MARGIN and t[3] != margins[0])]
    rows = []
    with Pool(procs, _init, (corpus, traces, n, cluster, kind, programs, interval)) as p, open(out + ".jsonl", "a") as log:
        for r in p.imap_unordered(_one, tasks, chunksize=1):
            if r:
                rows.append(r)
                log.write(pd.Series(r).to_json() + "\n")   # progress, and the rows so far if the run dies
                log.flush()
    df = pd.DataFrame(rows).sort_values(["scenario", "load", "oracle", "margin", "seed"])
    df.to_csv(out, index=False)
    return df
