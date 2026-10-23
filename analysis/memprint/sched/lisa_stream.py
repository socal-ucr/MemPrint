"""Job streams from the SURF Lisa trace (lisa.py's jobs.csv and curves_*.npy).

A stream is n consecutive jobs (by submit time) from a random point of the trace, with their real inter-arrival
pattern (rescaled to an offered load, as the other streams), runtimes, node counts and cores per node. Two kinds:

- measured: each job's memory is its own measured curve per node (active memory above the node's idle level);
- hybrid: each job's footprint shape comes from a corpus program at its largest input (as real.make_jobs does for
  the synthetic stream), scaled so its peak is the job's measured peak, so that every corpus estimator and the
  resize study's builds can be replayed on the trace's mix of sizes, runtimes and node counts.

Lisa records no memory request; a job requests what Slurm's default would give it, the node's memory per core times
the cores it uses on each node (a whole 16-core, 103 GB node for most jobs).
"""

import os

import numpy as np
import pandas as pd

from .model import Cluster, Job

LISA = os.path.expanduser("~/memprint-data/traces-lisa/jobs")
NODE_GB, NODE_CORES = 103.0, 16   # Lisa's normal nodes (282 of 343): 103 GB, 16 cores


def load(d=LISA, curve="active"):
    jobs = pd.read_csv(f"{d}/jobs.csv")
    curves = np.load(f"{d}/curves_{curve}.npy")
    return jobs, curves


def trace(n=3000, seed=0, cluster=None, d=LISA, curve="active"):
    """n consecutive jobs that fit the cluster: arrival (minutes from the first), base runtime (minutes), nodes,
    cores per node, the per-node request (GB) and the measured per-node curve (GB)."""
    cluster = cluster or Cluster()
    jobs, curves = load(d, curve)
    per_node = np.ceil(jobs.cores / jobs.nodes).astype(int)
    peak = curves.max(axis=1) / 1e9
    ok = ((jobs.nodes <= cluster.nodes) & (per_node <= cluster.cores) & (peak <= cluster.local + cluster.pool)
          & (peak > 0)).to_numpy()
    jobs, curves, per_node = jobs[ok].reset_index(drop=True), curves[ok], per_node[ok].reset_index(drop=True)
    order = np.argsort(jobs.submit.to_numpy(), kind="stable")
    jobs, curves, per_node = jobs.iloc[order].reset_index(drop=True), curves[order], per_node.iloc[order].reset_index(drop=True)
    start = np.random.default_rng(seed).integers(0, max(len(jobs) - n, 1))
    sel = slice(start, start + n)
    t = jobs.iloc[sel].copy()
    t["arrival"] = (t.submit - t.submit.min()) / 60
    t["base"] = np.clip(np.ceil(t.minutes), 1, None).astype(int)
    t["procs"] = per_node.iloc[sel].to_numpy()
    t["req_gb"] = np.minimum(t.procs * NODE_GB / NODE_CORES, cluster.local + cluster.pool)
    t["curve"] = list(curves[sel] / 1e9)
    return t.reset_index(drop=True)


def measured_jobs(tr):
    """Jobs whose truth is their own measured curve; info holds the request and the curve's peak."""
    jobs = []
    for i, r in tr.iterrows():
        c = np.asarray(r.curve, float)
        info = dict(request=float(max(r.req_gb, c.max())), prog="lisa", scale=1.0, lisa_id=int(r.id))
        jobs.append(Job(id=i, arrival=r.arrival, cores=int(r.procs), base=int(r.base), resident=c, info=info,
                        shape="lisa", nodes=int(r.nodes)))
    return jobs


def hybrid_trace(tr):
    """The trace in real.make_jobs's form, the measured peak standing in for the request x u that sizes a job."""
    t = tr.copy()
    t["peak_gb"] = [float(np.max(c)) for c in t.curve]
    return t
