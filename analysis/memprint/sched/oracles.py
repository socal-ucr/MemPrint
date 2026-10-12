"""Footprint oracles: what the scheduler reserves for a job, as a curve over progress (GB).

Two kinds cover every estimator:
- a curve oracle follows an estimated footprint over progress, optionally coarsened to a few windows, each
  widened by its neighbours to cover timing error (a profiler's time axis is instructions or references, not
  wall-clock time);
- a peak oracle holds one number for the job's whole run.
Both multiply by (1 + margin) and add a fixed overhead for what the estimator cannot see (stack, code, data).
"""

import numpy as np

from .model import K


def curve(est, margin=0.0, add=0.0, windows=None, widen=0, headroom=False):
    """Reservation that follows `est` (K samples over progress). With `windows`, each of that many equal
    windows holds its largest value, widened to the largest of its `widen` neighbours on each side. The margin
    multiplies the curve, or with `headroom` adds margin x the curve's peak everywhere (a curve on another
    clock can be low where the truth is not, which no factor can cover)."""
    est = np.asarray(est, float)
    if windows:
        edges = np.floor(np.arange(windows) * K / windows).astype(int)
        top = np.maximum.reduceat(est, edges)
        if widen:
            padded = np.pad(top, widen, mode="edge")
            top = np.max([padded[i:i + windows] for i in range(2 * widen + 1)], axis=0)
        est = np.repeat(top, np.diff(np.append(edges, K)))
    if headroom:
        return est + margin * est.max() + add
    return est * (1 + margin) + add


def peak(est, margin=0.0, add=0.0):
    """A constant reservation: the largest value of `est` (a curve or a number)."""
    return np.full(K, float(np.max(est)) * (1 + margin) + add)


# The prototype's oracles, on jobs from synthetic.py. info: profiled = heap curve on the profiling input.
NONHEAP_GB = 1.0
SYNTHETIC = {
    "exact": lambda j, m: curve(j.resident),
    "exact-peak": lambda j, m: peak(j.resident),
    "massif": lambda j, m: curve(j.info["profiled"], m, NONHEAP_GB, windows=20, widen=1),
    "dhat-peak": lambda j, m: peak(j.info["profiled"], m, NONHEAP_GB),
}


# Oracles on jobs from real.py: info holds every corpus method's curve on the job's scale. Curve oracles are
# coarsened to 20 windows widened by one on each side, as each method's clock (references, Valgrind ms,
# seconds) maps to wall-clock progress only approximately.
def _curve(method):
    return lambda j, m: curve(j.info[method], m, windows=20, widen=1, headroom=True)


def _peak(method):
    return lambda j, m: peak(j.info[method], m)


def reactive(j, m, interval=5):
    """Dynamic provisioning (monitor and resize every `interval` minutes, as in Zacarias et al., SC-W 2023): the
    request for the first interval, then each interval the largest footprint seen in the previous one x
    (1 + margin). Growth faster than that is an OOM kill. The scheduler is shown the future resizes, which
    favours this baseline."""
    a = np.asarray(j.resident, float)
    lag = max(1, int(np.ceil(K * interval / max(j.base, 1))))
    r = np.empty(K)
    r[:lag] = j.info["request"]
    for s in range(lag, K, lag):
        r[s:s + lag] = a[s - lag:s].max() * (1 + m)
    return r


reactive.elastic = True


REAL = {
    "reactive": reactive,
    "exact": lambda j, m: curve(j.resident),
    "exact-peak": lambda j, m: peak(j.resident),
    "exact-coarse": lambda j, m: curve(j.resident, windows=20, widen=1),   # exact, at the measured curves' resolution
    "memprint-W1": _curve("W1"), "memprint-W5": _curve("W5"), "memprint-P": _curve("P"),
    "memprint-W5-peak": _peak("W5"),
    "memprint-static": _curve("static"),
    "massif": _curve("massif"), "massif-pages": _curve("massifpages"),
    "massif-i": _curve("massifi"), "massif-pages-i": _curve("massifpagesi"), "heaptrack": _curve("heaptrack"),
    "dhat": _peak("dhat"), "dhat-curve": _curve("dhatcurve"), "dhat-hold": _curve("dhathold"), "gperf": _peak("gperf"), "rss-peak": _peak("rss"),
    "request": lambda j, m: np.full(K, j.info["request"]),
}


def available(jobs, name):
    """Whether every job has what the oracle needs (a method may be missing for some programs)."""
    need = {"memprint-static": "static", "memprint-W1": "W1", "memprint-W5": "W5", "memprint-P": "P", "memprint-W5-peak": "W5",
            "massif": "massif", "massif-pages": "massifpages", "massif-i": "massifi",
            "massif-pages-i": "massifpagesi", "heaptrack": "heaptrack", "dhat": "dhat",
            "gperf": "gperf", "rss-peak": "rss",
            "dhat-curve": "dhatcurve", "dhat-hold": "dhathold"}.get(name)
    return need is None or all(need in j.info for j in jobs)


def assign(jobs, oracle, margin=0.3, table=None):
    """Set every job's reservation with `oracle`: a name in `table` (SYNTHETIC by default, REAL for jobs from
    real.py) or a callable (job, margin) -> curve."""
    f = oracle if callable(oracle) else (table or SYNTHETIC)[oracle]
    for j in jobs:
        j.reservation = np.asarray(f(j, margin), float)
        j.elastic = getattr(f, "elastic", False)
    return jobs
