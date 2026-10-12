"""Synthetic job streams (the prototype's workload).

Each job: Poisson arrival, 1-16 cores, a lognormal base runtime of 5-720 minutes, a footprint shape (model.phases)
scaled to a lognormal heap peak, a fraction of the requested heap actually touched, and a non-heap part. The
profilers see the requested heap on a different (profiling) input: profiled = heap_req / X, X ~ lognormal(0, 0.15).

Offered load is the resident GB-minutes arriving per minute over the cluster's total memory: memory is the
contended resource, so that is what the load axis counts.
"""

import numpy as np

from .model import K, Cluster, Job, at_slots, phases

SHAPES = ("steady", "ramp", "phased", "spike", "shrink")


def make_jobs(n, seed=0, heap_median=24.0, heap_sigma=0.8, touched=(0.4, 1.0), nonheap=(0.2, 1.0),
              profile_sigma=0.15):
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(n):
        shape = SHAPES[rng.integers(len(SHAPES))]
        heap = phases(shape, rng) * min(rng.lognormal(np.log(heap_median), heap_sigma), 600.0)
        tf = rng.uniform(*touched)
        nh = rng.uniform(*nonheap)
        base = int(np.clip(rng.lognormal(np.log(30), 1.0), 5, 720))
        jobs.append(Job(id=i, arrival=0, cores=int(rng.integers(1, 17)), base=base, shape=shape,
                        resident=tf * heap + nh,
                        info=dict(heap_req=heap, touched=tf, nonheap=nh,
                                  profiled=heap / rng.lognormal(0, profile_sigma))))
    return jobs


def offered(jobs, cluster: Cluster):
    """Resident GB-minutes of the stream per minute of arrivals, if every job ran at its base runtime."""
    return sum(at_slots(j.resident, j.base).sum() for j in jobs)


def stream(n, load, seed=0, cluster=None, **kw):
    """n jobs at `load` (fraction of total memory; None or inf = all arrive at once, i.e. saturation)."""
    cluster = cluster or Cluster()
    jobs = make_jobs(n, seed, **kw)
    if load is None or np.isinf(load):
        return jobs
    gbmin = offered(jobs, cluster)
    capacity = cluster.nodes * cluster.local + cluster.pool
    rate = load * capacity * n / gbmin                       # jobs per minute
    t = np.floor(np.cumsum(np.random.default_rng(seed + 10_000).exponential(1 / rate, n))).astype(int)
    for j, a in zip(jobs, t):
        j.arrival = int(a)
    return jobs
