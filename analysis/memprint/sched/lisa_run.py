"""Scheduling on the SURF Lisa trace (lisa_stream.py) on a Lisa-shaped cluster: 32 nodes of 16 cores, local memory
swept down from Lisa's 103 GB per node, a pool of a quarter of the local memory.

    python -m memprint.sched.lisa_run --corpus DIR --traces DIR --out DIR [--kind measured|hybrid] [--locals ...]

kind "lisa-measured": every job's memory is its own measured curve; the oracles are those that need only the job's
own memory (exact, exact-coarse, exact-peak, request, reactive, zacarias, zacarias-past). kind "lisa-hybrid": the
trace's arrivals, runtimes, node counts and peaks with corpus footprint shapes (real.make_jobs), every oracle. The
arrivals are rescaled to the offered load against each cluster's own memory (against the 103 GB cluster, a small
cluster runs several times over its capacity, its queue grows without bound and so does the simulation).
"""

import argparse

from . import lisa_stream, real, run
from .model import Cluster, at_slots

LISA_NODE = dict(nodes=32, cores=16)
MEASURED = ["exact", "exact-coarse", "exact-peak", "request", "reactive", "zacarias", "zacarias-past"]
HYBRID = ["exact", "exact-coarse", "exact-peak", "memprint-W5", "memprint-P", "massif-i", "massif", "dhat", "dhat-curve",
          "heaptrack", "gperf", "rss-peak", "request", "ml-peak", "reactive", "pond", "zacarias", "zacarias-past",
          "ksplus"]


def lisa_cluster(local):
    return Cluster(local=local, pool=LISA_NODE["nodes"] * local / 4, **LISA_NODE)


class LisaSetup(run.Setup):
    def stream(self, scenario, load, seed):
        ref = self.cluster
        if seed not in self._trace:
            self._trace[seed] = lisa_stream.trace(self.n, seed, cluster=self.cluster)
        tr = self._trace[seed]
        if self.kind == "lisa-measured":
            jobs = lisa_stream.measured_jobs(tr)
        else:
            jobs = real.make_jobs(lisa_stream.hybrid_trace(tr), self.curves, self.cm5_test, scenario, seed=seed)
        gbmin = sum(at_slots(j.resident, j.base).sum() * j.nodes for j in jobs)
        t = real.scale_arrivals([j.arrival for j in jobs], load, ref, gbmin)
        for j, a in zip(jobs, t):
            j.arrival = int(a)
        return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", choices=["measured", "hybrid"], default="measured")
    ap.add_argument("--locals", type=float, nargs="+", default=[103, 64, 32, 16, 8])
    ap.add_argument("--loads", type=float, nargs="+", default=[0.4, 0.6, 0.8, 1.0, 1.2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--procs", type=int, default=20)
    a = ap.parse_args()
    names = MEASURED if a.kind == "measured" else HYBRID
    scenarios = ("rss",) if a.kind == "measured" else ("rss", "bytes")
    for local in a.locals:
        run.run(a.corpus, a.traces, f"{a.out}/lisa_{a.kind}_{local:g}.csv", scenarios=scenarios, loads=a.loads,
                seeds=a.seeds, margins=("cal",), names=names, n=3000, cluster=lisa_cluster(local), procs=a.procs,
                kind=f"lisa-{a.kind}")
        print("done", a.kind, local, flush=True)


if __name__ == "__main__":
    main()
