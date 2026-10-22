"""The KS+ oracle (Bader et al., "KS+: Predicting Workflow Task Memory Usage Over Time", eScience 2024), after the
authors' code (github.com/dos-group/KSPlus, tsb_resource_allocation/predictor.py and model.py).

KS+ learns a task's memory over time from earlier executions of the same task with other inputs:
- each execution's monitored memory series is cut into k segments by a greedy merge into a non-decreasing step
  function (get_segments), giving each segment's end time and peak;
- per segment, two linear regressions on the input size predict the end time and the peak of a new execution;
- the prediction is made safe by ending segments 15% earlier and raising peaks by 10%, and is then forced to be
  non-decreasing;
- after an out-of-memory failure at time t, the ends of the segment that failed and of all later ones are
  scaled so that the failed segment ends at t (the next one starts there); a failure in the last segment raises
  its peak by 20%. The task is re-executed until it succeeds.

Here an execution's monitored series is the RSS of the program's native run at another input of the corpus
(above its runtime baseline, as the RSS truth and the RSS-peak oracle see it), on the native run's wall clock,
and the input size is the size of the data the input describes (INPUT_SIZE). A job uses the program's largest
input, so KS+ extrapolates from the two smaller ones; on the LLM stream, every input is predicted from the other
two of the same engine. A program with no other input (darknet) has no history, and KS+ reserves the request.
"""

import bisect

import numpy as np
from sklearn.linear_model import LinearRegression

from .model import K

SEGMENTS = 4   # the authors' default_num_segments
END_FACTOR, PEAK_FACTOR, RETRY_PEAK = 0.85, 1.1, 1.2
FLOOR_GB = 0.1   # the authors' Model floors every peak at 100 MB


def _pb(*dims):   # PolyBench: total elements of the arrays (sizes from the kernel's header)
    return sum(a * b for a, b in dims)


# the size of each input's data, known before the run (per program, so only ratios matter): elements of the
# arrays a dataset size declares, graph vertices, grid points, particles, the interpreters' scale factor, the
# source file compiled, the model file read
INPUT_SIZE = {
    "2mm": {"SMALL": _pb((40, 70), (70, 50), (50, 80), (40, 80)), "MEDIUM": _pb((180, 210), (210, 190), (190, 220), (180, 220)),
            "LARGE": _pb((800, 1100), (1100, 900), (900, 1200), (800, 1200))},
    "gemm": {"SMALL": _pb((60, 70), (60, 80), (80, 70)), "MEDIUM": _pb((200, 220), (200, 240), (240, 220)),
             "LARGE": _pb((1000, 1100), (1000, 1200), (1200, 1100))},
    "jacobi-2d": {"SMALL": 2 * 90 ** 2, "MEDIUM": 2 * 250 ** 2, "LARGE": 2 * 1300 ** 2},
    "atax": {"SMALL": 116 * 124, "MEDIUM": 390 * 410, "LARGE": 1900 * 2100},
    "miniVite": {"8192": 8192, "16384": 16384, "32768": 32768},
    **{f"gap-{k}": {"16": 2 ** 16, "17": 2 ** 17, "18": 2 ** 18} for k in ("bfs", "pr", "cc", "sssp")},
    **{p: {"0.25": 0.25, "0.5": 0.5, "1": 1.0} for p in ("lua", "python", "perl", "sqlite")},
    "cc1": {"small": 3915, "medium": 45081, "large": 2862216},          # 2mm.c, darknet's parser.c, sqlite3.c
    "hpccg": {n: int(n) ** 3 for n in ("32", "48", "64")},
    "lulesh": {n: int(n) ** 3 for n in ("30", "45", "60")},
    "minife": {n: int(n) ** 3 for n in ("40", "60", "80")},
    "kripke": {n: int(n) ** 3 for n in ("8", "12", "16")},
    "quicksilver": {n: int(n) for n in ("20000", "50000", "100000")},
    "npb-is": {"S": 2 ** 16, "W": 2 ** 20, "A": 2 ** 23},                 # keys
    "npb-cg": {"S": 1400 * 8 ** 2, "W": 7000 * 9 ** 2, "A": 14000 * 12 ** 2},   # na x (nonzer + 1)^2
    "npb-mg": {"S": 32 ** 3, "W": 128 ** 3, "A": 256 ** 3},
    "llama2c": {"15M": 60816028, "42M": 167020572, "110M": 438381596},   # model files (bytes)
    "llamacpp": {"135m-n32": 144811360, "135m-n128": 144811360, "360m-n64": 386405280},
    "darknet": {"alexnet": 1},
}


def get_segments(data, segcount=SEGMENTS):
    """The authors' greedy segmentation: segment sizes (samples) and peaks of a non-decreasing step function."""
    res = [[data[0], 1, 0.0]]   # peak, size, error
    for d in data[1:]:
        if res[-1][0] >= d:
            res[-1][1] += 1
        else:
            res.append([d, 1, 0.0])

    def merge_error(a, b):
        return (a[0] - b[0]) * b[1] if a[0] >= b[0] else (b[0] - a[0]) * a[1]
    while len(res) > segcount:
        i = int(np.argmin([merge_error(a, b) for a, b in zip(res, res[1:])]))
        a, b = res[i], res[i + 1]
        a[2] += b[2] + merge_error(a, b)
        a[0], a[1] = max(a[0], b[0]), a[1] + b[1]
        res.pop(i + 1)
    sizes, peaks = [r[1] for r in res], [r[0] for r in res]
    while len(sizes) < segcount:   # as the authors' code: pad with flat segments
        sizes.append(10)
        peaks.append(peaks[-1])
    return sizes, peaks


def sanitize(ends, peaks):
    """As the authors' Model: ends non-increasing from the back, peaks non-decreasing (the 100 MB floor is applied
    on the job's scale, in oracle)."""
    ends, peaks = [max(e, 0.0) for e in ends], [max(p, 0.0) for p in peaks]
    last = ends[-1]
    for i in range(len(ends) - 2, -1, -1):
        last = ends[i] = min(ends[i], last)
    for i in range(1, len(peaks)):
        peaks[i] = max(peaks[i], peaks[i - 1])
    return ends, peaks


def fit(history, target_size, segcount=SEGMENTS):
    """history: [(input size, memory series, seconds per sample)]. Returns (segment ends in seconds, peaks)."""
    X = np.array([[s] for s, _, _ in history], float)
    segs = [get_segments(list(m), segcount) for _, m, _ in history]
    ends = [np.cumsum(sz) * dt for (sz, _), (_, _, dt) in zip(segs, history)]
    e = [LinearRegression().fit(X, [x[i] for x in ends]).predict([[target_size]])[0] * END_FACTOR for i in range(segcount)]
    p = [LinearRegression().fit(X, [pk[i] for _, pk in segs]).predict([[target_size]])[0] * PEAK_FACTOR
         for i in range(segcount)]
    return sanitize(e, p)


def curve(ends, peaks):
    """The step function on K samples of progress (ends given as progress)."""
    x = (np.arange(K) + 0.5) / K
    idx = np.minimum(np.searchsorted(ends, x, side="right"), len(ends) - 1)
    return np.asarray(peaks, float)[idx]


def fail_at(ends, peaks, t):
    """The authors' get_model_failat: the segment running at progress t and every later one end proportionally
    earlier, so the failed segment ends at t; in the last segment, its peak rises by 20%."""
    i = bisect.bisect_right(ends, t)
    ends, peaks = list(ends), list(peaks)
    if i >= len(ends) - 1:
        peaks[-1] *= RETRY_PEAK
    else:
        f = t / ends[i]
        ends = [e * f if j >= i else e for j, e in enumerate(ends)]
    return ends, peaks


def models(all_curves, costs, targets, rebase):
    """KS+ predictions for `targets` [(prog, input, key)]: key -> (ends as progress of the target's native run,
    peaks in bytes on the target's own scale) or None without history. rebase(info) is real.rebase."""
    wall = costs[costs.method == "native"].set_index(["prog", "input"]).wall_s
    out = {}
    for prog, inp, key in targets:
        sizes = INPUT_SIZE.get(prog, {})
        hist = []
        for other, g in all_curves[(all_curves.prog == prog) & (all_curves.input != inp)].groupby("input"):
            if other not in sizes or (prog, other) not in wall.index:
                continue
            info = {m: h.sort_values("k").bytes.to_numpy() for m, h in g.groupby("method") if m in ("rss", "rssbase")}
            if "rss" not in info:
                continue
            rss = rebase(info)["rss"]
            hist.append((sizes[other], rss, float(wall[(prog, other)]) / len(rss)))
        if not hist or inp not in sizes or (prog, inp) not in wall.index:
            out[key] = None
            continue
        ends, peaks = fit(hist, sizes[inp])
        out[key] = (list(np.asarray(ends) / float(wall[(prog, inp)])), peaks)
    return out


def oracle(table):
    """Oracle on jobs from real.py: the KS+ step function scaled as the job's other curves are, re-fitted after each
    OOM kill by the retry strategy (the simulator calls `retry` instead of growing the reservation)."""
    def f(j, m):
        mdl = table.get(j.info["prog"])
        if mdl is None:
            return np.full(K, float(j.info["request"]))
        ends, peaks = mdl
        j.ksplus = (ends, list(np.maximum.accumulate([max(p * j.info["scale"], FLOOR_GB) for p in peaks])))
        return curve(*j.ksplus)

    def retry(j, t):
        if getattr(j, "ksplus", None) is None:
            return None   # the request: grow as every other oracle does
        j.ksplus = fail_at(*j.ksplus, t)
        return curve(*j.ksplus)

    f.attrs = lambda j, m: dict(retry=retry)
    return f
