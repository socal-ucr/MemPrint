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
  zacarias-past        glibc build, dynamic provisioning as a running system can do it: each update to the previous
                       interval's largest use, grown on demand within the interval (failing and restarting when the
                       growth does not fit), memory committed one minute at a time
  mpr+zacarias-past    the same on the libmpr build
  mprh, mprh+zacarias  the libmpr build with a held-out profile (from the program's other inputs or models)
  trim+zacarias[-past] glibc build with libmpr's MPR_TRIM (glibc's free pages returned to the kernel when they
                       exceed 10% of the anonymous memory), dynamic provisioning; trim alone holds the request
  mprt, mprt+zacarias[-past]  the libmpr build (own profile) with MPR_TRIM
The trim builds need the runs of scripts/trim_eval.sh: RESIZE_RUNS=<its runs directory> reads every build from there
(base, trim, self, selftrim, heldout, heldouttrim), else the resize study's runs (no trim builds).

--inflate o sets every glibc-build request to (1 + o) times the program's allocation, as a user who over-requests
would (Zacarias et al. study +60%); the libmpr builds take their requests from the profile, not from the user, so
they are unaffected.

With --overheads native or pin, each policy also pays its own costs: a job built with the pass runs for its
workload's measured libmpr runtime ratio (libmpr's run against glibc's), and the first job of each workload in the
stream is the profiling run, under libmpr's native-speed profiler (native) or the Pin tool (pin), at its measured
slowdown and holding the glibc build's request; a growing resize of dynamic provisioning costs 10 s of progress
(Zacarias et al.'s simulator); Pond's mitigation costs 50 ms per GB it copies to local memory (the Pond paper). KS+'s
retries and dynamic provisioning's restarts are counted as lost work either way.
"""

import argparse
import glob
import os
from copy import copy
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import ksplus, oracles, real
from .model import K, Cluster, Job, at_slots
from .run import steady
from .sim import Sim

RESIZE = os.path.expanduser("~/memprint-data/2026-10-12-resize")
RUNS = os.environ.get("RESIZE_RUNS")   # scripts/trim_eval.sh's runs: every build, the trim builds among them
BUILDS = ("mprh", "mprt", "mpr", "trim")
SCHED = os.path.expanduser("~/memprint-data/2026-10-13-sched-llm")
LLM = {"llamacpp-135m-n32": ("llamacpp", "135m-n32"), "llamacpp-135m-n128": ("llamacpp", "135m-n128"),
       "llamacpp-360m-n64": ("llamacpp", "360m-n64"), "llama2c-15M": ("llama2c", "15M"),
       "llama2c-42M": ("llama2c", "42M"), "llama2c-110M": ("llama2c", "110M")}
COSTS = dict(resize_s=10.0, migrate_s_per_gb=0.05)
CLUSTER_KW = {}   # cluster settings other than local memory (beta, pool) for the sensitivity runs; set by run()


def overhead_tables():
    """Per workload: libmpr's runtime ratio (own profile, mean of 3 runs against glibc's) and the two profilers'
    slowdowns (libmpr's native-speed profile: profiling run / glibc run; the Pin -sites profile)."""
    def secs(path):
        t = pd.read_csv(path, sep=r"\s+", header=None, names=["label", "rss", "commit", "secs", "hash", "status"],
                        engine="python")
        t["cfg"] = t.label.str.replace(r"-\d+$", "", regex=True)
        return t.groupby("cfg").secs.mean()
    ratio = {}
    for path in glob.glob(f"{RESIZE}/runs/corpus/*/summary.txt") + glob.glob(f"{RESIZE}/runs/ml/*/summary.txt"):
        g = secs(path)
        ratio[path.split("/")[-2]] = g["self"] / g["base"]
    g = secs(f"{RESIZE}/runs/minivite-v4/summary.txt")
    ratio["miniVite (pass)"] = g["self"] / g["none"]
    nat = pd.read_csv(f"{RESIZE}/runs/nprof_summary.csv").set_index("name")
    native = (nat.prof_s / nat.base_secs).dropna().to_dict()
    pin = pd.read_csv(f"{RESIZE}/results/pin_cost.csv").set_index("workload").slowdown.to_dict()
    for d in (native, pin):
        if "miniVite" in d:
            d["miniVite (pass)"] = d["miniVite"]
    # llama.cpp 135M at 128 tokens has no native-profile run: the same model's 32-token profile
    native.setdefault("llamacpp-135m-n128", native.get("llamacpp-135m-n32", 1.0))
    return ratio, native, pin


POLICIES = ["exact", "request", "pond", "zacarias", "zacarias-past", "ksplus", "mpr-exact", "mpr", "mpr+zacarias",
            "mpr+zacarias-past", "mpr+pond", "mprh", "mprh+zacarias"]
TRIM_POLICIES = ["trim", "trim+zacarias", "trim+zacarias-past", "mprt", "mprt+zacarias", "mprt+zacarias-past"]


def build_of(policy):
    """The build a policy runs: glibc (None) or one of BUILDS."""
    return next((b for b in BUILDS if policy in (b, b + "-exact") or policy.startswith(b + "+")), None)


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
    if RUNS:
        cases = [(n, f"{RUNS}/{'minivite' if n == 'miniVite (pass)' else n}/", "base", p, i) for n, _, _, p, i in cases]
    out = {}
    for name, d, base_label, prog, inp in cases:
        load = lambda label: [pd.read_csv(f) for f in sorted(glob.glob(f"{d}/{label}-[0-9].csv")) if os.path.getsize(f)]
        g, m, h = load(base_label), load("self"), load("heldout")
        if not g or not m:
            continue
        b = g[0].rss_bytes.min()
        cv = lambda runs: [(_curve(t, "rss_bytes", b), _curve(t, "commit_bytes", b)) for t in runs]
        out[name] = dict(prog=prog, input=inp, glibc=cv(g), mpr=cv(m), mprh=cv(h) or cv(m))
        if RUNS:   # a build whose runs never trimmed is the build without trimming (its runs differ by the monitor
            # thread's slowdown alone, which would pass for a saving on the progress axis: darknet)
            trims = lambda label: np.median([pd.read_csv(f).trims.iloc[-1] for f in
                                             glob.glob(f"{d}/{label}-[0-9].trims") + glob.glob(f"{d}/{label}-[0-9].monitor.csv")]
                                            or [0])
            out[name].update(trim=cv(load("trim")) if trims("trim") > 0 else cv(g),
                             mprt=cv(load("selftrim")) if trims("selftrim") > 0 else cv(m))
    return out


def untouched(runs):
    """Pond's history: the smallest untouched share (1 - peak RSS / request) of the runs."""
    return max(min(1 - rss.max() / max(np.maximum(rss, com).max(), 1e-12) for rss, com in runs), 0.0)


class Setup:
    def __init__(self, n=3000, overheads="none"):
        self.wl = workloads()
        self.overheads = overheads
        self.ratio, self.native, self.pin = overhead_tables()
        self.names = sorted(self.wl)
        curves = pd.read_csv(f"{SCHED}/corpus/curves.csv", dtype={"input": str})
        costs = pd.read_csv(f"{SCHED}/corpus/costs.csv", dtype={"input": str})
        self.ks = ksplus.models(curves, costs, [(w["prog"], w["input"], name) for name, w in self.wl.items()],
                                real.rebase)
        self.n = n
        self._trace = {}

    def stream(self, truth, load, seed, inflate=0.0):
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
            hrss, hcom = w["mprh"][min(k, len(w["mprh"]) - 1)]
            req = np.maximum(rss, com).max()
            s = r.req_gb / max(req, 1e-12)   # GB per byte of the program's own scale
            extra = {}
            for bld in ("trim", "mprt"):   # (RESIZE_RUNS only)
                if bld in w:
                    trss, tcom = w[bld][min(k, len(w[bld]) - 1)]
                    tothers = [x for j, x in enumerate(w[bld]) if j != min(k, len(w[bld]) - 1)] or w[bld]
                    extra.update({f"{bld}_request": np.maximum(trss, tcom).max() * s, f"{bld}_um": untouched(tothers)})
                    extra[f"{bld}_truth"] = (np.maximum(trss, tcom) if truth == "strict" else trss) * s
            pick = (lambda a, c: np.maximum(a, c)) if truth == "strict" else (lambda a, c: a)
            others = [x for j, x in enumerate(w["glibc"]) if j != k] or w["glibc"]
            mothers = [x for j, x in enumerate(w["mpr"]) if j != min(k, len(w["mpr"]) - 1)] or w["mpr"]
            hothers = [x for j, x in enumerate(w["mprh"]) if j != min(k, len(w["mprh"]) - 1)] or w["mprh"]
            cap = Cluster().local + Cluster().pool
            um = untouched(others)
            info = dict(prog=name, scale=s, request=min(req * s * (1 + inflate), cap),
                        mpr_request=np.maximum(mrss, mcom).max() * s, mprh_request=np.maximum(hrss, hcom).max() * s,
                        glibc_truth=pick(rss, com) * s, mpr_truth=pick(mrss, mcom) * s, mprh_truth=pick(hrss, hcom) * s,
                        # Pond's history sees the user's request: the untouched share includes the over-request
                        um=1 - (1 - um) / (1 + inflate), mpr_um=untouched(mothers), mprh_um=untouched(hothers),
                        **extra)
            if "trim_request" in info:   # the glibc build with trimming: the user's request all the same (or its own
                # peak, when its run happens to peak above the glibc run's: run-to-run noise, not the user's choice)
                info["trim_request"], info["trim_um"] = max(info["request"], info["trim_request"]), info["um"]
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
        cost = self.overheads != "none"
        prof = self.native if self.overheads == "native" else self.pin
        profiled = set()
        for j0 in sorted(jobs, key=lambda j: (j.arrival, j.id)):
            j = copy(j0)
            i = j.info
            build = build_of(policy)
            mpr = build in ("mpr", "mprh", "mprt")
            if cost and mpr and policy != "mpr-exact" and j.shape not in profiled:
                # the workload's first job is its profiling run: the glibc build under the profiler
                profiled.add(j.shape)
                j.resident = i["glibc_truth"]
                j.reservation = np.full(K, i["request"])
                j.base = max(1, int(round(j.base * prof.get(j.shape, 1.0))))
                j.profiling = True
                out.append(j)
                continue
            if cost and mpr:
                j.base = max(1, int(round(j.base * self.ratio.get(j.shape, 1.0))))
            j.resident = i[f"{build}_truth"] if build else i["glibc_truth"]
            R = i[f"{build}_request"] if build else i["request"]
            um = i[f"{build}_um"] if build else i["um"]
            if policy in ("exact", "mpr-exact"):
                j.reservation = j.resident.copy()
            elif policy == "ksplus":
                j.reservation = np.asarray(ks(j, 0.0), float)
                j.retry = ks.attrs(j, 0.0)["retry"]
            else:
                j.reservation = np.full(K, R)
                if policy.endswith("pond"):
                    j.local_cap, j.qos_delay = R - np.floor(R * um), 5
                    j.migrate_cost = COSTS["migrate_s_per_gb"] if cost else 0.0
                if "zacarias" in policy:
                    j.interval = oracles.INTERVAL
                    j.lookback = policy.endswith("-past")
                    j.fallback_after = 3 if j.lookback else None   # repeated failures: run it statically
                    j.resize_cost = COSTS["resize_s"] if cost else 0.0
            out.append(j)
        return out


_setup = None


def _init(n, overheads, interval=None):
    global _setup
    if interval:   # the dynamic policies' update interval (minutes), for the sensitivity runs
        oracles.INTERVAL = interval
    _setup = Setup(n, overheads)


def _one(task):
    truth, load, seed, policy, local, inflate = task
    try:
        import time
        cluster = Cluster(local=local, **CLUSTER_KW)
        jobs = _setup.assign(_setup.stream(truth, load, seed, inflate), policy)
        prof = sum(j.cores * j.base for j in jobs if getattr(j, "profiling", False)) / 60
        t0 = time.process_time()
        sim = Sim(cluster, jobs)
        d, s = sim.run()
        return dict(truth=truth, load=load, seed=seed, policy=policy, local_gb=local, inflate=inflate,
                    overheads=_setup.overheads,
                    profiling_core_h=prof, sim_cpu_s=time.process_time() - t0, **s, **steady(d, sim))
    except Exception as e:  # one bad task must not lose the grid
        import traceback
        return dict(truth=truth, load=load, seed=seed, policy=policy, local_gb=local, inflate=inflate, error=repr(e),
                    trace=traceback.format_exc()[-500:])


def run(out, truths=("rss", "strict"), loads=(0.4, 0.6, 0.8, 1.0, 1.2), seeds=(0, 1, 2), policies=POLICIES,
        locals_=(128,), n=3000, procs=28, overheads="none", inflates=(0.0,), cluster_kw=None, interval=None):
    tasks = list(product(truths, loads, seeds, policies, locals_, inflates))
    rows = []
    global CLUSTER_KW
    CLUSTER_KW = cluster_kw or {}
    with Pool(procs, _init, (n, overheads, interval)) as p, open(out + ".jsonl", "a") as log:
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
    ap.add_argument("--overheads", choices=["none", "native", "pin"], default="none")
    ap.add_argument("--inflate", type=float, nargs="+", default=[0.0])
    a = ap.parse_args()
    run(a.out, loads=a.loads, seeds=a.seeds, policies=a.policies, locals_=a.locals, procs=a.procs, overheads=a.overheads,
        inflates=a.inflate)


if __name__ == "__main__":
    main()
