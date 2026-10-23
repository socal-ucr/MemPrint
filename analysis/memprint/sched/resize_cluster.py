"""Cluster throughput with MemPrintResize against Pond, dynamic provisioning and KS+ (DMSched-Sim).

    ~/environments/memfp/bin/python -m memprint.sched.resize_cluster --out results.csv [--loads ...] [--seeds ...]

A job is one of the resize study's 29 workloads (the 23 corpus programs at their largest inputs, miniVite built
with the pass, six LLM configurations), with the RSS and commit timelines recorded for it under glibc and under
libmpr. The job requests what the program allocates (its peak of commit and RSS), and it is built either without
the pass (glibc) or with it (libmpr). One scale per job maps the program's allocation under glibc to the stream's
request, so a job built with the pass is smaller in the cluster by what libmpr saves, and both builds face the same
arrivals (the offered load is the glibc stream's). The runtime's fixed footprint (the glibc run's smallest RSS) is
taken off both builds before scaling, as in the scheduling study.

The truth (what the job must be given, or it is OOM-killed) is its RSS ("rss") or the larger of its commit and RSS
("strict": strict overcommit). Policies:
  request              glibc build, the request held for the whole run
  pond                 glibc build, the request held, the predicted untouched part (the smallest untouched share of
                       the job's other runs, 1 - peak RSS / request) in the pool, QoS mitigation after 5 minutes
  zacarias             glibc build, dynamic provisioning every 5 minutes (sim interval)
  ksplus               glibc build, KS+ from the RSS of the program's other corpus inputs (sched.ksplus)
  mpr                  libmpr build, its own request held for the whole run (MemPrintResize alone)
  mpr+zacarias         libmpr build, dynamic provisioning
  mpr+pond             libmpr build, Pond
  exact, mpr-exact     the truth's own curve, each build (the upper bounds)
"""

import argparse
import glob
import os
from copy import copy
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import ksplus, real
from .model import K, Cluster, Job, at_slots
from .run import steady
from .sim import Sim

RESIZE = os.path.expanduser("~/memprint-data/2026-10-12-resize")
SCHED = os.path.expanduser("~/memprint-data/2026-10-13-sched-llm")
LLM = {"llamacpp-135m-n32": ("llamacpp", "135m-n32"), "llamacpp-135m-n128": ("llamacpp", "135m-n128"),
       "llamacpp-360m-n64": ("llamacpp", "360m-n64"), "llama2c-15M": ("llama2c", "15M"),
       "llama2c-42M": ("llama2c", "42M"), "llama2c-110M": ("llama2c", "110M")}
POLICIES = ["exact", "request", "pond", "zacarias", "ksplus", "mpr-exact", "mpr", "mpr+zacarias", "mpr+pond"]


def _curve(t, col, base):
    """A run's timeline on K samples of progress (each the largest value in its share), less the baseline."""
    v = np.maximum(t[col].to_numpy(float) - base, 0)
    return at_slots(v, K)


def workloads():
    """name -> dict(prog, input, glibc=[(rss, commit) per run], mpr=[...]) in bytes over progress (K samples)."""
    big = {}
    for line in open(os.path.expanduser("~/memprint-data/2026-10-11-sched/corpus/list.txt")):
        f = line.split()
        if f and not (f[0] == "npb-mg" and f[1] == "A"):
            big[f[0]] = f[1]
    cases = [(d.rstrip("/").split("/")[-1], d, "base") for d in sorted(glob.glob(f"{RESIZE}/runs/corpus/*/"))]
    cases = [(n, d, b, n, big[n]) for n, d, b in cases if n != "miniVite"]
    cases.append(("miniVite (pass)", f"{RESIZE}/runs/minivite-v4/", "none", "miniVite", "32768"))
    cases += [(n, f"{RESIZE}/runs/ml/{n}/", "base", p, i) for n, (p, i) in LLM.items()]
    out = {}
    for name, d, base_label, prog, inp in cases:
        g = [pd.read_csv(f) for f in sorted(glob.glob(f"{d}/{base_label}-[0-9].csv")) if os.path.getsize(f)]
        m = [pd.read_csv(f) for f in sorted(glob.glob(f"{d}/self-[0-9].csv")) if os.path.getsize(f)]
        if not g or not m:
            continue
        b = g[0].rss_bytes.min()
        out[name] = dict(prog=prog, input=inp,
                         glibc=[(_curve(t, "rss_bytes", b), _curve(t, "commit_bytes", b)) for t in g],
                         mpr=[(_curve(t, "rss_bytes", b), _curve(t, "commit_bytes", b)) for t in m])
    return out


def untouched(runs):
    """Pond's history: the smallest untouched share (1 - peak RSS / request) of the runs."""
    return max(min(1 - rss.max() / max(np.maximum(rss, com).max(), 1e-12) for rss, com in runs), 0.0)


class Setup:
    def __init__(self, n=3000):
        self.wl = workloads()
        self.names = sorted(self.wl)
        curves = pd.read_csv(f"{SCHED}/corpus/curves.csv", dtype={"input": str})
        costs = pd.read_csv(f"{SCHED}/corpus/costs.csv", dtype={"input": str})
        self.ks = ksplus.models(curves, costs, [(w["prog"], w["input"], name) for name, w in self.wl.items()],
                                real.rebase)
        self.n = n
        self._trace = {}

    def stream(self, truth, load, seed):
        """The jobs of one stream: workload, scale and both builds' truths; arrivals at the glibc offered load."""
        ref = Cluster()
        if seed not in self._trace:
            self._trace[seed] = real.synthetic_trace(self.n, seed, cluster=ref)
        tr = self._trace[seed]
        rng = np.random.default_rng(seed + 7)
        jobs = []
        for i, r in tr.iterrows():
            name = self.names[rng.integers(len(self.names))]
            w = self.wl[name]
            k = rng.integers(len(w["glibc"]))
            rss, com = w["glibc"][k]
            mrss, mcom = w["mpr"][min(k, len(w["mpr"]) - 1)]
            req = np.maximum(rss, com).max()
            s = r.req_gb / max(req, 1e-12)   # GB per byte of the program's own scale
            pick = (lambda a, c: np.maximum(a, c)) if truth == "strict" else (lambda a, c: a)
            others = [x for j, x in enumerate(w["glibc"]) if j != k] or w["glibc"]
            mothers = [x for j, x in enumerate(w["mpr"]) if j != min(k, len(w["mpr"]) - 1)] or w["mpr"]
            info = dict(prog=name, scale=s, request=req * s, mpr_request=np.maximum(mrss, mcom).max() * s,
                        glibc_truth=pick(rss, com) * s, mpr_truth=pick(mrss, mcom) * s,
                        um=untouched(others), mpr_um=untouched(mothers))
            jobs.append(Job(id=i, arrival=r.arrival, cores=int(r.procs), base=int(r.base),
                            resident=info["glibc_truth"], info=info, shape=name))
        gbmin = sum(at_slots(j.resident, j.base).sum() for j in jobs)
        t = real.scale_arrivals([j.arrival for j in jobs], load, ref, gbmin)
        for j, a in zip(jobs, t):
            j.arrival = int(a)
        return jobs

    def assign(self, jobs, policy):
        """Copies of the jobs with the policy's build, reservation and simulator attributes."""
        out = []
        ks = ksplus.oracle(self.ks)
        for j0 in jobs:
            j = copy(j0)
            i = j.info
            mpr = policy.startswith("mpr")
            j.resident = i["mpr_truth"] if mpr else i["glibc_truth"]
            R = i["mpr_request"] if mpr else i["request"]
            um = i["mpr_um"] if mpr else i["um"]
            if policy in ("exact", "mpr-exact"):
                j.reservation = j.resident.copy()
            elif policy == "ksplus":
                j.reservation = np.asarray(ks(j, 0.0), float)
                j.retry = ks.attrs(j, 0.0)["retry"]
            else:
                j.reservation = np.full(K, R)
                if policy.endswith("pond"):
                    j.local_cap, j.qos_delay = R - np.floor(R * um), 5
                if policy.endswith("zacarias"):
                    j.interval = 5
            out.append(j)
        return out


_setup = None


def _init(n):
    global _setup
    _setup = Setup(n)


def _one(task):
    truth, load, seed, policy, local = task
    try:
        cluster = Cluster(local=local)
        jobs = _setup.assign(_setup.stream(truth, load, seed), policy)
        sim = Sim(cluster, jobs)
        d, s = sim.run()
        return dict(truth=truth, load=load, seed=seed, policy=policy, local_gb=local, **s, **steady(d, sim))
    except Exception as e:  # one bad task must not lose the grid
        import traceback
        return dict(truth=truth, load=load, seed=seed, policy=policy, local_gb=local, error=repr(e),
                    trace=traceback.format_exc()[-500:])


def run(out, truths=("rss", "strict"), loads=(0.4, 0.6, 0.8, 1.0, 1.2), seeds=(0, 1, 2), policies=POLICIES,
        locals_=(128,), n=3000, procs=28):
    tasks = list(product(truths, loads, seeds, policies, locals_))
    rows = []
    with Pool(procs, _init, (n,)) as p, open(out + ".jsonl", "a") as log:
        for r in p.imap_unordered(_one, tasks, chunksize=1):
            rows.append(r)
            log.write(pd.Series(r).to_json() + "\n")
            log.flush()
    df = pd.DataFrame(rows).sort_values(["truth", "load", "policy", "seed", "local_gb"])
    df.to_csv(out, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--loads", type=float, nargs="+", default=[0.4, 0.6, 0.8, 1.0, 1.2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    ap.add_argument("--locals", type=float, nargs="+", default=[128])
    ap.add_argument("--procs", type=int, default=28)
    a = ap.parse_args()
    run(a.out, loads=a.loads, seeds=a.seeds, policies=a.policies, locals_=a.locals, procs=a.procs)


if __name__ == "__main__":
    main()
