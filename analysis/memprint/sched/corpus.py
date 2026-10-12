"""The footprint corpus (scripts/footprint_corpus.sh): every method's footprint of every (program, input), as a
curve over progress p in [0, 1].

Each method has its own clock, so each curve is put on progress by linear rescaling of its own run: memory
references (MemPrint), milliseconds under Valgrind (Massif), seconds (RSS). A method that gives only a peak
(DHAT at t-gmax, gperftools' in-use dumps) is a flat curve at that peak. Curves are resampled to K points by
taking the largest value in each step, so a reservation that follows the curve covers every sample in it.

    python -m memprint sched corpus DIR   ->  DIR/curves.csv (prog, input, method, k, bytes), DIR/costs.csv
"""

import glob
import json
import os
import re
import subprocess

import numpy as np
import pandas as pd

from .model import K

HALF_PAGE = 2048  # expected unused part of a page bordering an untouched one (windowed partial-page correction)
HEAPTRACK_PRINT = os.environ.get(
    "HEAPTRACK_PRINT",
    "/home/nmust004/spack/opt/spack/linux-broadwell/heaptrack-1.3.0-fob5atwjkybije52useb6lqyrp43x4zp/bin/heaptrack_print")


def resample(t, v, k=K):
    """Largest value of the step function (t, v) in each of k equal steps of progress."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    if len(v) == 0:
        return None
    if len(v) == 1 or t.max() <= t.min():
        return np.full(k, v.max())
    p = (t - t.min()) / (t.max() - t.min())
    step = np.minimum((p * k).astype(int), k - 1)
    out = np.full(k, np.nan)
    np.fmax.at(out, step, v)
    # a step with no sample holds the value in force: the last sample before it
    last = pd.Series(out).ffill().bfill().to_numpy()
    return last


def _one(pattern):
    f = sorted(glob.glob(pattern))
    return f[0] if f else None


def truth(d):
    f = _one(f"{d}/Buffered_*_timeline.csv")
    if not f:
        return None
    t = pd.read_csv(f)
    t = t[t.Bin == -1]
    return t.Time, t.MemUsageObs


def windowed(d):
    f = _one(f"{d}/Spatial_*_windowed.csv")
    if not f:
        return None
    w = pd.read_csv(f)
    return w.Time, np.maximum(w.Estimate - HALF_PAGE * w.PartialEdges, 0)


def rss(d):
    f = f"{d}/rss.csv"
    if not os.path.exists(f):
        return None
    r = pd.read_csv(f)
    r = r[r.rss_bytes > 0]
    return r.seconds, r.rss_bytes


def massif_text(text):
    snaps = re.findall(r"time=([\d.]+)\nmem_heap_B=(\d+)\nmem_heap_extra_B=(\d+)\nmem_stacks_B=(\d+)", text)
    if not snaps:
        return None
    a = np.array(snaps, float)
    return a[:, 0], a[:, 1] + a[:, 2] + a[:, 3]


def massif(d):
    f = f"{d}/massif.out"
    return massif_text(open(f).read()) if os.path.exists(f) else None


def dhat(d):
    f = f"{d}/dhat.json"
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    return [0.0], [sum(pp["gb"] for pp in j["pps"])]


def dhat_points(d):
    """DHAT records no time series, only the heap at its global peak (gb, at time tg) and at exit (eb, at te)."""
    f = f"{d}/dhat.json"
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    gb, eb = sum(pp["gb"] for pp in j["pps"]), sum(pp["eb"] for pp in j["pps"])
    return j["tg"] / max(j["te"], 1), gb, eb


def dhat_curve(d, hold=False):
    """A curve from DHAT's three points: from 0 (or, with `hold`, the peak) at the start, to the peak at tg,
    then linearly to the exit value. Only "never above the peak" is guaranteed; the rest is interpolation."""
    got = dhat_points(d)
    if got is None:
        return None
    g, gb, eb = got
    p = np.linspace(0, 1, 4 * K + 1)
    up = np.full_like(p, gb) if hold else gb * p / max(g, 1e-9)
    down = gb + (eb - gb) * (p - g) / max(1 - g, 1e-9)
    return p, np.where(p <= g, up, down)


def heaptrack(d):
    f = _one(f"{d}/heaptrack*.zst") or _one(f"{d}/heaptrack*.gz")
    if not f:
        return None
    out = f"{d}/heaptrack.massif"
    if not os.path.exists(out):
        subprocess.run([HEAPTRACK_PRINT, f, "--print-massif", out, "-p", "0", "-a", "0", "-T", "0", "-l", "0"],
                       capture_output=True, check=False)
    return massif_text(open(out).read()) if os.path.exists(out) else None


def gperf(d):
    peaks = []
    for f in glob.glob(f"{d}/heap.*.heap"):
        m = re.match(r"heap profile:\s*\d+:\s*(\d+)", open(f).readline())
        if m:
            peaks.append(int(m.group(1)))
    return ([0.0], [max(peaks)]) if peaks else None


def static(d):
    f = f"{d}/static.csv"
    if not os.path.exists(f):
        return None
    s = pd.read_csv(f)
    return s.p, s.bytes


PARSERS = {"static": static, "native": rss, "truth": truth, "W1": windowed, "W5": windowed, "P": windowed, "massif": massif,
           "massifpages": massif, "massifi": massif, "massifpagesi": massif, "dhat": dhat, "heaptrack": heaptrack, "gperf": gperf}


def curves(root):
    rows = []
    for d in sorted(glob.glob(f"{root}/*/*/*/")):
        prog, inp, method = d.rstrip("/").split("/")[-3:]
        if method not in PARSERS:
            continue
        try:
            got = PARSERS[method](d.rstrip("/"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"{prog}/{inp}/{method}: {e}")
            continue
        if got is None:
            continue
        c = resample(*got)
        if c is None:
            continue
        rows.append(pd.DataFrame(dict(prog=prog, input=inp, method="rss" if method == "native" else method,
                                      k=np.arange(K), bytes=c)))
        if method == "dhat":
            for name, hold in (("dhatcurve", False), ("dhathold", True)):
                rows.append(pd.DataFrame(dict(prog=prog, input=inp, method=name, k=np.arange(K),
                                              bytes=resample(*dhat_curve(d.rstrip("/"), hold)))))
        if method == "native":
            # the runtime's own RSS (loader, libraries, interpreter): the smallest raw sample, before the program
            # allocates its data
            rows.append(pd.DataFrame(dict(prog=prog, input=inp, method="rssbase", k=np.arange(K),
                                          bytes=float(np.min(got[1])))))
    return pd.concat(rows, ignore_index=True)


def costs(root):
    """done.txt: prog input method exit wall_s maxrss_kb (the last run of each wins)."""
    d = pd.read_csv(f"{root}/done.txt", sep=r"\s+", header=None,
                    names=["prog", "input", "method", "exit", "wall_s", "maxrss_kb"], dtype={"input": str})
    d = d.drop_duplicates(["prog", "input", "method"], keep="last")
    nat = d[d.method == "native"].set_index(["prog", "input"]).wall_s.rename("native_s")
    d = d.join(nat, on=["prog", "input"])
    d["slowdown"] = d.wall_s / d.native_s.clip(lower=0.01)
    return d


def build(root):
    c, k = curves(root), costs(root)
    c.to_csv(f"{root}/curves.csv", index=False)
    k.to_csv(f"{root}/costs.csv", index=False)
    return c, k
