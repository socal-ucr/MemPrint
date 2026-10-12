"""Jobs and the cluster.

A job's memory is a curve over progress p in [0, 1], sampled at K equal steps. Curves are over progress, not
time, so a job slowed by remote memory stretches its curve in wall-clock time. All memory is in GB.
"""

from dataclasses import dataclass, field

import numpy as np

K = 200  # samples of a memory curve over progress


@dataclass
class Cluster:
    nodes: int = 32
    cores: int = 32           # per node
    local: float = 128.0      # GB of local DRAM per node
    pool: float = 1024.0      # GB of shared pool (CXL or RDMA)
    beta: float = 0.5         # runtime stretch when every resident byte is remote (1 + beta)
    window: int = 32          # backfill window (queued jobs considered behind the head)
    oom_growth: float = 1.5   # reservation multiplier after an OOM kill


@dataclass
class Job:
    id: int
    arrival: int              # minute
    cores: int
    base: int                 # runtime in minutes with all memory local
    resident: np.ndarray      # actual resident footprint over progress, GB (K samples)
    info: dict = field(default_factory=dict)  # what the oracles may see (profiled curves, peaks, ...)
    shape: str = ""
    reservation: np.ndarray = None  # set by an oracle, GB (K samples)
    elastic: bool = False     # growth past the reservation is granted from free memory (reactive provisioning)


def at_slots(curve, slots):
    """The curve over `slots` equal wall-clock slots: each slot takes the largest sample in its share of
    progress, so a reservation equal to the curve is never exceeded inside a slot."""
    curve = np.asarray(curve, float)
    starts = np.floor(np.arange(slots) * len(curve) / slots).astype(int)
    if slots >= len(curve):
        return curve[starts]
    return np.maximum.reduceat(curve, starts)


def phases(kind, rng, k=K):
    """A footprint shape over progress, scaled to peak 1: steady, ramp (peaks at the end), phased
    (alternating), spike (a short transient) or shrink (load, then reduce)."""
    p = (np.arange(k) + 0.5) / k
    if kind == "steady":
        c = np.full(k, 0.9) + 0.1 * (p > 0.05)
    elif kind == "ramp":
        c = 0.1 + 0.9 * p ** rng.uniform(0.7, 2.0)
    elif kind == "phased":
        n = rng.integers(2, 5)
        low = rng.uniform(0.15, 0.5)
        c = np.where(np.floor(p * 2 * n) % 2 == 1, 1.0, low)
    elif kind == "spike":
        base = rng.uniform(0.15, 0.4)
        at, width = rng.uniform(0.2, 0.8), rng.uniform(0.03, 0.08)
        c = np.where(np.abs(p - at) < width / 2, 1.0, base)
    elif kind == "shrink":
        cut = rng.uniform(0.15, 0.4)
        c = np.where(p < cut, 1.0, rng.uniform(0.2, 0.5))
    else:
        raise ValueError(kind)
    return c / c.max()
