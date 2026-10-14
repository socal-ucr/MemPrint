"""Job streams from a real trace with measured footprint curves.

- Arrivals, runtimes, core counts and the user's memory request come from an SWF log (MetaCentrum2 by default:
  requested memory per job, no used memory). Only jobs that fit on one node are kept.
- What a job really uses: its peak is request x u, with u the used/requested ratio of a random LANL CM5 job
  (the archive log that records both), keeping u <= 1 (a job above its request is killed by the batch
  system). That CM5 job also supplies the features of the ML oracle.
- The shape of its footprint, and what every estimator says about it, come from one corpus program (its
  largest input): all of that program's curves are scaled by the one factor that puts the truth's peak at the
  job's peak, so each estimator keeps its own over- or under-estimate.

Truths (scenario): "rss", the resident set (what the kernel charges and OOM-kills on); "bytes", bytes
touched (MemPrint's definition: what a page-perfect system would need); and "rssmpr", the resident set of the
program run with libmpr serving its allocations by MemPrint's reservations (resize/).
"""

import numpy as np
import pandas as pd

from .model import Cluster, Job

SWF = ['id', 'submit', 'wait', 'run', 'procs', 'cpu', 'used_mem', 'req_procs', 'req_time', 'req_mem', 'status',
       'uid', 'gid', 'exe', 'queue', 'part', 'prec', 'think']
GB = 1024 ** 3


def read_swf(path):
    return pd.read_csv(path, sep=r"\s+", comment=";", header=None, names=SWF, low_memory=False)


def cm5_ratios(path):
    """LANL CM5 jobs with both memory fields: used/requested (u <= 1) and the features the ML oracle sees."""
    d = read_swf(path)
    d = d[(d.used_mem > 0) & (d.req_mem > 0) & (d.run > 0)].copy()
    d["u"] = d.used_mem / d.req_mem
    return d[d.u <= 1][["u", "req_mem", "procs", "req_time", "uid", "exe"]].reset_index(drop=True)


def trace_jobs(path, cluster: Cluster, n=3000, seed=0, max_hours=24):
    """n consecutive single-node jobs from a random point of the log (GB, minutes)."""
    d = read_swf(path)
    d = d[(d.run >= 60) & (d.procs >= 1) & (d.procs <= cluster.cores) & (d.req_mem > 0)].copy()
    d["req_gb"] = d.req_mem * d.procs * 1024 / GB              # SWF memory is KB per processor
    d = d[d.req_gb <= cluster.local + cluster.pool].sort_values("submit").reset_index(drop=True)
    start = np.random.default_rng(seed).integers(0, len(d) - n)
    d = d.iloc[start:start + n].copy()
    d["arrival"] = (d.submit - d.submit.min()) / 60
    d["base"] = np.clip(np.ceil(d.run / 60), 1, max_hours * 60).astype(int)
    return d[["arrival", "base", "procs", "req_gb", "req_time"]].reset_index(drop=True)


def synthetic_trace(n=3000, seed=0, req_median=100.0, req_sigma=0.8, cluster=None):
    """The prototype's job stream (synthetic.py) as a trace: arrivals in order, 1-16 cores, lognormal base
    runtimes of 5-720 minutes and lognormal memory requests, sized so that memory is the contended resource.
    Footprints then come from the corpus as for a real trace (make_jobs)."""
    cluster = cluster or Cluster()
    rng = np.random.default_rng(seed)
    req = np.minimum(rng.lognormal(np.log(req_median), req_sigma, n), cluster.local + cluster.pool)
    return pd.DataFrame(dict(arrival=np.sort(rng.uniform(0, 1, n)), procs=rng.integers(1, 17, n),
                             base=np.clip(rng.lognormal(np.log(30), 1.0, n), 5, 720).astype(int),
                             req_gb=req, req_time=0))


def scale_arrivals(t, load, cluster, gbmin):
    """Compress or stretch the trace's arrival times to an offered load (resident GB-min / total memory)."""
    capacity = cluster.nodes * cluster.local + cluster.pool
    span = gbmin / (load * capacity)
    return np.floor(np.asarray(t) / max(np.max(t), 1) * span).astype(int)


# scenario -> (the truth's method in the corpus, the method holding its runtime baseline, if any)
TRUTHS = {"rss": ("rss", "rssbase"), "bytes": ("truth", None), "rssmpr": ("rssmpr", "rssmprbase"),
          # the RSS without the resident code (executable mappings: program and libraries): what grows with the
          # job's data. Not rebased: that run's sampler also reads smaps, so its first sample can come after the
          # program has allocated its data (2mm), and the code is the part that does not grow
          "rssdata": ("rssdata", None)}


def rebase(info):
    """The runtime's fixed footprint (loader, libraries, MPI, interpreter: the run's smallest RSS) does not
    grow with the job, so only the RSS above it is scaled."""
    for method, base in TRUTHS.values():
        if base and method in info:
            info[method] = np.maximum(info[method] - info.get(base, info[method]).min(), 0)
    return info


def rss_view(info, scenario):
    """An earlier run's RSS as the oracles that use one see it: measured as the truth is (without the code
    under the rssdata truth), the RSS otherwise."""
    return info.get("rssdata") if scenario == "rssdata" and "rssdata" in info else info.get("rss")


def make_jobs(trace, curves, cm5, scenario="rss", programs=None, seed=0):
    """Jobs whose resident curve is the corpus truth of a random program, scaled to the job's peak.

    curves: corpus curves (prog, input, method, k, bytes), largest input per program. info gets every method's
    curve on the same scale, the user request, and the CM5 record (u and features)."""
    rng = np.random.default_rng(seed + 1)
    truth_method = TRUTHS[scenario][0]
    wide = {(p, m): g.sort_values("k").bytes.to_numpy() for (p, m), g in curves.groupby(["prog", "method"])}
    progs = sorted(p for p in {p for p, _ in wide} if (p, truth_method) in wide)
    if programs:
        progs = [p for p in progs if p in programs]
    jobs = []
    for i, r in trace.iterrows():
        prog = progs[rng.integers(len(progs))]
        c = cm5.iloc[rng.integers(len(cm5))]
        peak = r.req_gb * c.u
        info = {m: wide[(p_, m)] for (p_, m) in wide if p_ == prog}
        info = rebase(info)
        t = info[truth_method]
        s = peak / t.max()
        info = {m: c * s for m, c in info.items()}
        info.update(request=r.req_gb, cm5=c, prog=prog, scale=s)
        if rss_view(info, scenario) is not None:
            info["rssview"] = rss_view(info, scenario)
        # arrival in trace time; run.Setup rescales it to the offered load
        jobs.append(Job(id=i, arrival=r.arrival, cores=int(r.procs), base=int(r.base), resident=info[truth_method],
                        info=info, shape=prog))
    return jobs
