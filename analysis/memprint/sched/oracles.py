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


INTERVAL = 5   # minutes between the dynamic policies' updates (reactive, zacarias, zacarias-past); run._init sets it


def reactive(j, m, interval=None):
    """Dynamic provisioning (monitor and resize every `interval` minutes, as in Zacarias et al., SC-W 2023): the
    request for the first interval, then each interval the largest footprint seen in the previous one x
    (1 + margin). Growth faster than that is an OOM kill. The scheduler is shown the future resizes, which
    favours this baseline."""
    a = np.asarray(j.resident, float)
    interval = interval or INTERVAL
    lag = max(1, int(np.ceil(K * interval / max(j.base, 1))))
    r = np.empty(K)
    r[:lag] = j.info["request"]
    for s in range(lag, K, lag):
        r[s:s + lag] = a[s - lag:s].max() * (1 + m)
    return r


reactive.elastic = True


def zacarias(j, m, interval=None):
    """Dynamic provisioning as Zacarias et al. (SC-W 2023) simulate it: the request is the initial allocation;
    every `interval` minutes the allocation is resized to the largest footprint of the coming interval (their
    simulator reads it from the usage trace), with no headroom, local memory first. Unlike `reactive`, a resize
    is committed only when it happens, so other jobs may take the memory, and a resize that does not fit fails
    the job and restarts it (F/R). The margin is not used."""
    return np.full(K, float(j.info["request"]))


zacarias.attrs = lambda j, m: dict(interval=INTERVAL)


def zacarias_past(j, m):
    """Dynamic provisioning as a running system can do it: each update sets the allocation to the previous interval's
    largest use, and the job grows on demand within the interval (failing and restarting when the growth does not
    fit); memory is committed one minute at a time. After three restarts the job runs with its request held, as
    Zacarias et al. suggest against repeated failures (without it, jobs whose growth others keep taking restart
    indefinitely). The margin is not used."""
    return zacarias(j, m)


zacarias_past.attrs = lambda j, m: dict(interval=INTERVAL, lookback=True, fallback_after=3)


CLOCK_PAGES = {"T": "truth", "Tsec": "truthsec", "Tpages": "truthpages", "Tpagessec": "truthpagessec",
               "W5sec": "W5sec", "W5pages": "W5pages", "W5pagessec": "W5pagessec",
               # on the native wall clock, through the native run's memory references over time (nativerefs)
               "Tnat": "truthnat", "W5nat": "W5nat", "W1nat": "W1nat", "Pnat": "Pnat"}


def _scaled_to_rss(method, level=None):
    """A hybrid: the shape of a MemPrint curve, at the level of an earlier run's RSS (its curve scaled so its peak
    is the RSS peak; `level` names that run's curve, the RSS view by default). MemPrint's curve knows when the
    program uses memory; the RSS knows how much the kernel backs, code and allocator caches included."""
    def f(j, m):
        c = np.asarray(j.info[method], float)
        lv = np.max(j.info[level] if level else j.info.get("rssview", j.info["rss"]))
        return curve(c * (lv / max(c.max(), 1e-12)), m, windows=20, widen=1, headroom=True)
    return f


HYBRID = {"memprint-W5nat-rss": "W5nat", "memprint-Pnat-rss": "Pnat"}

# Is an earlier run's RSS enough (earlier.py)? rssrep1 is a repeat native run of the same input, not the RSS truth's
# own run (which rss-peak and the hybrids above read); -x: every curve predicted for the job's input from the
# program's smaller inputs (earlier.transfer: methods with suffix X). name -> (kind, curve, level for a hybrid)
EARLIER = {"rss-peak-rep": ("peak", "rssrep1", None), "rss-curve-rep": ("curve", "rssrep1", None),
           "hybrid-rep": ("curve", "W5nat", "rssrep1"),
           "rss-peak-x": ("peak", "rssrep1X", None), "rss-curve-x": ("curve", "rssrep1X", None),
           "memprint-W5-x": ("curve", "W5X", None), "memprint-W5nat-x": ("curve", "W5natX", None),
           "hybrid-x": ("curve", "W5natX", "rssrep1X")}


def _earlier(kind, method, level):
    """An EARLIER oracle; without its curves (a transfer with nothing to extrapolate) it reserves the request."""
    f = _scaled_to_rss(method, level) if level else _peak(method) if kind == "peak" else _curve(method)
    return lambda j, m: (f(j, m) if all(x in j.info for x in (method, level) if x)
                         else np.full(K, float(j.info["request"])))

REAL = {
    "reactive": reactive,
    "zacarias": zacarias,
    "zacarias-past": zacarias_past,
    "exact": lambda j, m: curve(j.resident),
    "exact-peak": lambda j, m: peak(j.resident),
    "exact-coarse": lambda j, m: curve(j.resident, windows=20, widen=1),   # exact, at the measured curves' resolution
    "memprint-W1": _curve("W1"), "memprint-W5": _curve("W5"), "memprint-P": _curve("P"),
    "memprint-W5-peak": _peak("W5"),
    "memprint-static": _curve("static"),
    # the clock and footprint options (Pin tool -clock seconds, -footprint pages): the full trace (T) and W5,
    # on the references or wall-time clock, counting bytes or pages
    **{f"memprint-{n}": _curve(m) for n, m in CLOCK_PAGES.items()},
    "massif": _curve("massif"), "massif-pages": _curve("massifpages"),
    "massif-i": _curve("massifi"), "massif-pages-i": _curve("massifpagesi"), "heaptrack": _curve("heaptrack"),
    "dhat": _peak("dhat"), "dhat-curve": _curve("dhatcurve"), "dhat-hold": _curve("dhathold"), "gperf": _peak("gperf"),
    "rss-peak": lambda j, m: peak(j.info.get("rssview", j.info["rss"]), m),
    "request": lambda j, m: np.full(K, j.info["request"]),
    **{n: _scaled_to_rss(m) for n, m in HYBRID.items()},
    **{n: _earlier(*v) for n, v in EARLIER.items()},
}


def available(jobs, name):
    """Whether every job has what the oracle needs (a method may be missing for some programs)."""
    need = {"memprint-static": "static", "memprint-W1": "W1", "memprint-W5": "W5", "memprint-P": "P", "memprint-W5-peak": "W5",
            "massif": "massif", "massif-pages": "massifpages", "massif-i": "massifi",
            "massif-pages-i": "massifpagesi", "heaptrack": "heaptrack", "dhat": "dhat",
            "gperf": "gperf", "rss-peak": "rss",
            "dhat-curve": "dhatcurve", "dhat-hold": "dhathold",
            **{f"memprint-{n}": m for n, m in CLOCK_PAGES.items()}, **HYBRID}.get(name)
    if name in EARLIER:
        return all(all(m in j.info for m in EARLIER[name][1:] if m) or "request" in j.info for j in jobs)
    return need is None or all(need in j.info and (name not in HYBRID or "rss" in j.info) for j in jobs)


def assign(jobs, oracle, margin=0.3, table=None):
    """Set every job's reservation with `oracle`: a name in `table` (SYNTHETIC by default, REAL for jobs from
    real.py) or a callable (job, margin) -> curve."""
    f = oracle if callable(oracle) else (table or SYNTHETIC)[oracle]
    for j in jobs:
        j.reservation = np.asarray(f(j, margin), float)
        j.elastic = getattr(f, "elastic", False)
        for k, v in (f.attrs(j, margin) if hasattr(f, "attrs") else {}).items():
            setattr(j, k, v)   # what the simulator needs beyond the curve (Pond's local cap, a resize interval)
    return jobs
