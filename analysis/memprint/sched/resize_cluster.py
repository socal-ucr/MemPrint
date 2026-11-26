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
  est+zacarias[-past]  the bound on returning free memory: the glibc build's RSS less the allocator's free bytes
                       (the data estimate of paper_rss), as if the allocator gave every free byte back; the free
                       bytes from the monitor of trim_eval.sh's trim build with MPR_TRIM=-1 (RESIZE_EST_RUNS)
  mpre, mpre+zacarias[-past]  the same bound on the libmpr build (own profile)
  memrss               glibc build, MemRSS's estimated curve as the reservation: the data estimate plus the file
                       RSS from the monitor of the trim build with MPR_TRIM=-1 (RESIZE_EST_RUNS, the run of the
                       same index), less the glibc run's smallest RSS, with headroom = the q90 over the other
                       workloads of the margin each needs to cover its truth (leave one workload out, as the
                       scheduling study); an OOM kill grows the reservation as for every other oracle
  mpr+memrss           the same on the libmpr build (own profile), from the selftrim build's monitor
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
TRIM_RUNS = os.environ.get("RESIZE_TRIM_RUNS")   # the trim builds alone from another share's runs (default RUNS)
EST_RUNS = os.environ.get("RESIZE_EST_RUNS")   # trim_eval.sh's runs with MONITOR=1: the data estimate's curves
ALIGN_RUNS = os.environ.get("RESIZE_ALIGN_RUNS")   # the same with PERF_INSTR=1: truth and estimate from these runs,
# the estimate from another run of the same build, aligned by wall time or by instructions (align_workloads)
BUILDS = ("mprh", "mprt", "mpre", "mpr", "trim", "est")
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
EST_POLICIES = ["est+zacarias", "est+zacarias-past", "mpre", "mpre+zacarias", "mpre+zacarias-past"]
MEMRSS_POLICIES = ["memrss", "mpr+memrss", "memrss-w", "mpr+memrss-w", "memrss-peak", "mpr+memrss-peak"]   # (RESIZE_EST_RUNS)
# -peak: MemRSS's peak held for the whole run (a flat reservation), its margin calibrated on that peak: the curve's
# level without its timing, which another run of the program does not share
# -w: MemRSS's curve at the scheduling study's resolution for measured curves (20 windows, each widened to its
# neighbours, oracles.curve), so that a run's timing differs from the estimate's without a margin of its peak


def _widened(e):
    return oracles.curve(e, 0.0, windows=20, widen=1)


def _shape(e, mode):
    """MemRSS's estimate as a policy reserves it: the curve (None), widened ("w"), its peak held ("peak"), or with a
    tolerance of +-N% of progress ("dN": each point the largest value within N% of it on either side)."""
    if mode == "w":
        return _widened(e)
    if mode == "peak":
        return np.full(K, float(np.max(e)))
    if mode and mode.startswith("d"):
        r = max(1, int(round(float(mode[1:]) * K / 100)))
        padded = np.pad(np.asarray(e, float), r, mode="edge")
        return np.max([padded[i:i + K] for i in range(2 * r + 1)], axis=0)
    return e


def _mode(policy):
    """A MemRSS policy's shape: [mpr+]memrss[-i][-w|-peak|-dN] (-i: the estimate aligned by instructions)."""
    last = policy.rsplit("-", 1)[-1]
    return last if last in ("w", "peak") or (last.startswith("d") and last[1:].replace(".", "").isdigit()) else None


def _aligned(policy):
    """The estimate's alignment: "" (wall time) or "i<tol>" (instructions, tolerance in %), as -i<tol> in the name."""
    if not is_memrss(policy):
        return ""
    return next((x for x in policy.split("-")[1:] if x.startswith("i") and x[1:].replace(".", "").isdigit()), "")


def is_memrss(policy):
    return policy.split("+")[-1].split("-")[0] == "memrss"


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
            td = f"{TRIM_RUNS}/{'minivite' if name == 'miniVite (pass)' else name}/" if TRIM_RUNS else d
            tload = lambda label: [pd.read_csv(f) for f in sorted(glob.glob(f"{td}/{label}-[0-9].csv")) if os.path.getsize(f)]
            trims = lambda label: np.median([pd.read_csv(f).trims.iloc[-1] for f in
                                             glob.glob(f"{td}/{label}-[0-9].trims") + glob.glob(f"{td}/{label}-[0-9].monitor.csv")]
                                            or [0])
            out[name].update(trim=cv(tload("trim")) if trims("trim") > 0 else cv(g),
                             mprt=cv(tload("selftrim")) if trims("selftrim") > 0 else cv(m))
        if EST_RUNS:
            e = f"{EST_RUNS}/{'minivite' if name == 'miniVite (pass)' else name}/"

            def gaps(label):   # the allocator's free bytes in each monitored run (RSS - data - file) on the progress axis
                return [at_slots(np.maximum(t.rss_bytes - t.data_bytes - t.file_bytes, 0).to_numpy(float), K)
                        for t in map(pd.read_csv, sorted(glob.glob(f"{e}/{label}-[0-9].monitor.csv")))]

            def less(runs, gs, trimmed):   # the build's runs less those bytes (another run's: the monitor's thread
                # slows the program), as if the allocator gave its free memory back to the last byte, and no more than
                # the trimmed run (mallinfo counts glibc's main arena alone, malloc_trim every arena: llama.cpp)
                if not gs:
                    return trimmed
                return [(np.minimum(np.maximum(rss - gs[k % len(gs)], 0), trimmed[k % len(trimmed)][0]), com)
                        for k, (rss, com) in enumerate(runs)]
            if "trim" in out[name]:   # (RESIZE_RUNS: the trim builds' runs)
                out[name].update(est=less(out[name]["glibc"], gaps("trim"), out[name]["trim"]),
                                 mpre=less(out[name]["mpr"], gaps("selftrim"), out[name]["mprt"]))

            def memrss(label):   # MemRSS's data footprint: anon - resident free (data_bytes) + file RSS, less base
                return [_curve(t.assign(memrss=t.data_bytes + t.file_bytes), "memrss", b)
                        for t in map(pd.read_csv, sorted(glob.glob(f"{e}/{label}-[0-9].monitor.csv")))]
            out[name].update(memrss=memrss("trim"), mpr_memrss=memrss("selftrim"))
            # the "reclaim" truth: a node that takes the allocator's free bytes back, so a job holds its RSS less
            # them (the monitored run's of the same index, as est without the trimmed run's bound): MemRSS's target
            g, ms = gaps("trim"), gaps("selftrim")
            if g and ms:
                out[name].update(reclaim=[(np.maximum(rss - g[k % len(g)], 0), com) for k, (rss, com) in
                                          enumerate(out[name]["glibc"])],
                                 mpr_reclaim=[(np.maximum(rss - ms[k % len(ms)], 0), com) for k, (rss, com) in
                                              enumerate(out[name]["mpr"])])
    return out


def _instr(path):
    """A run's cumulative user-space instructions at the end of each perf interval: (seconds, count)."""
    t, c = [], []
    for line in open(path):
        f = line.strip().split(",")
        if len(f) > 3 and not line.startswith("#"):
            t.append(float(f[0]))
            c.append(float(f[1]) if f[1].replace(".", "").isdigit() else 0.0)
    return np.array(t), np.cumsum(c)


def _on_instructions(truth_t, est_t, est_v, it_t, it_c, ie_t, ie_c, tol=0.0):
    """The estimate (est_v at est_t in its own run) at each truth sample, indexed by instructions: at truth time t
    the job has retired I(t), so the reservation is the estimate's largest value between the previous sample's
    instruction share and this one's (shares of the estimate run's total, as a running system would know it),
    widened by tol (a share of the instructions) on both sides: the kernel frees a program's memory after its
    last instructions, so the truth can lag the estimate where memory falls."""
    total = max(ie_c[-1], 1.0)
    fe = np.interp(est_t, ie_t, ie_c, left=0.0) / total
    ft = np.minimum(np.interp(truth_t, it_t, it_c, left=0.0) / total, 1.0)
    hi = np.clip(np.searchsorted(fe, ft + tol, side="right") - 1, 0, len(est_v) - 1)
    lo = np.clip(np.searchsorted(fe, np.concatenate([[0.0], ft[:-1]]) - tol, side="right") - 1, 0, len(est_v) - 1)
    return np.array([est_v[min(a, b):max(a, b) + 1].max() for a, b in zip(lo, hi)])


ITOL = (0, 0.1, 0.5, 1)   # instruction tolerances of the -i<tol> policies (% of the run's instructions)


def align_workloads():
    """workloads() with every curve from the runs of RESIZE_ALIGN_RUNS (scripts/trim_eval.sh with MONITOR=1
    MPR_TRIM=-1 PERF_INSTR=1, labels trim and selftrim): a job on run k is checked against that run's RSS and
    commit, and MemRSS's estimate comes from run k+1 of the same build, on the progress axis by wall time (memrss)
    or by instructions (memrss_i); reclaim is run k's RSS less its own allocator's free bytes."""
    meta = workloads()
    out = {}
    for name, m in meta.items():
        d = f"{ALIGN_RUNS}/{'minivite' if name == 'miniVite (pass)' else name}"
        builds = {}
        for label, key in (("trim", ""), ("selftrim", "mpr_")):
            ks = sorted(int(f.rsplit("-", 1)[1].split(".")[0]) for f in glob.glob(f"{d}/{label}-[0-9].instr"))
            if not ks:
                break
            ts = [pd.read_csv(f"{d}/{label}-{k}.csv") for k in ks]
            ms = [pd.read_csv(f"{d}/{label}-{k}.monitor.csv").dropna() for k in ks]   # (a last row cut off at exit)
            ins = [_instr(f"{d}/{label}-{k}.instr") for k in ks]
            builds[key] = (ts, ms, ins)
        if len(builds) < 2:
            continue
        b = builds[""][0][0].rss_bytes.min()
        w = dict(prog=m["prog"], input=m["input"])
        for key, (ts, ms, ins) in builds.items():
            n = len(ts)
            runs = [(_curve(t, "rss_bytes", b), _curve(t, "commit_bytes", b)) for t in ts]
            est = [(mm.data_bytes + mm.file_bytes).to_numpy(float) for mm in ms]
            gap = [at_slots(np.maximum(mm.rss_bytes - mm.data_bytes - mm.file_bytes, 0).to_numpy(float), K) for mm in ms]
            j = [(k + 1) % n for k in range(n)]   # the estimate's run
            w["mpr" if key else "glibc"] = runs
            w[key + "reclaim"] = [(np.maximum(r - g, 0), c) for (r, c), g in zip(runs, gap)]
            w[key + "memrss"] = [at_slots(np.maximum(est[j[k]] - b, 0), K) for k in range(n)]
            for tol in ITOL:   # memrss_i<tol in % of instructions>
                w[f"{key}memrss_i{tol:g}"] = [at_slots(np.maximum(_on_instructions(
                    ts[k].seconds.to_numpy(float), ms[j[k]].seconds.to_numpy(float), est[j[k]], *ins[k], *ins[j[k]],
                    tol / 100) - b, 0), K) for k in range(n)]
        w["mprh"] = w["mpr"]
        out[name] = w
    return out


def untouched(runs):
    """Pond's history: the smallest untouched share (1 - peak RSS / request) of the runs."""
    return max(min(1 - rss.max() / max(np.maximum(rss, com).max(), 1e-12) for rss, com in runs), 0.0)


class Setup:
    def __init__(self, n=3000, overheads="none"):
        self.wl = align_workloads() if ALIGN_RUNS else workloads()
        self.overheads = overheads
        self.ratio, self.native, self.pin = overhead_tables()
        self.names = sorted(self.wl)
        curves = pd.read_csv(f"{SCHED}/corpus/curves.csv", dtype={"input": str})
        costs = pd.read_csv(f"{SCHED}/corpus/costs.csv", dtype={"input": str})
        self.ks = ksplus.models(curves, costs, [(w["prog"], w["input"], name) for name, w in self.wl.items()],
                                real.rebase)
        self.n = n
        self._trace = {}
        self._need = {}

    def need(self, truth, build, widened=None, aligned=""):
        """Per workload: the headroom (share of the estimate's peak) MemRSS's curve needs to cover the truth in
        every run of the build (glibc: None, or "mpr"), the runs paired by index (widened: of the -w curve)."""
        key = (truth, build, widened, aligned)
        if key not in self._need:
            runs, est = ("mpr", "mpr_memrss") if build == "mpr" else ("glibc", "memrss")
            est += f"_{aligned}" if aligned else ""
            if truth == "reclaim":
                runs = "mpr_reclaim" if build == "mpr" else "reclaim"
            out = {}
            for name, w in self.wl.items():
                if not w.get(est):
                    continue
                m = 0.0
                for k, (rss, com) in enumerate(w[runs]):
                    t = np.maximum(rss, com) if truth == "strict" else rss
                    e = w[est][k % len(w[est])]
                    e = _shape(e, widened if widened != True else "w")
                    m = max(m, float(np.max((t - e) / max(e.max(), 1e-12))))
                out[name] = m
            self._need[key] = out
        return self._need[key]

    def margin(self, truth, build, name, q=0.9, widened=None, aligned=""):
        """The q-quantile of the other workloads' needed headroom (leave one workload out)."""
        need = self.need(truth, build, widened, aligned)
        return float(np.quantile([m for p, m in need.items() if p != name], q))

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
            req = np.maximum(rss, com).max()   # the user's request: what the program allocates, under any truth
            mreq = np.maximum(mrss, mcom).max()   # libmpr's request from its profile, under any truth
            if truth == "reclaim":   # (RESIZE_EST_RUNS only)
                rss = w["reclaim"][k][0]
                mrss = w["mpr_reclaim"][min(k, len(w["mpr_reclaim"]) - 1)][0]
            hrss, hcom = w["mprh"][min(k, len(w["mprh"]) - 1)]
            s = r.req_gb / max(req, 1e-12)   # GB per byte of the program's own scale
            extra = {}
            for bld in ("trim", "mprt", "est", "mpre"):   # (RESIZE_RUNS, RESIZE_EST_RUNS only)
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
                        mpr_request=mreq * s, mprh_request=np.maximum(hrss, hcom).max() * s,
                        glibc_truth=pick(rss, com) * s, mpr_truth=pick(mrss, mcom) * s, mprh_truth=pick(hrss, hcom) * s,
                        # Pond's history sees the user's request: the untouched share includes the over-request
                        um=1 - (1 - um) / (1 + inflate), mpr_um=untouched(mothers), mprh_um=untouched(hothers),
                        truth=truth, **extra)
            for est in [x for x in w if x.endswith("memrss") or "memrss_i" in x]:   # (RESIZE_EST_RUNS, ALIGN_RUNS)
                if w.get(est):
                    info[f"{est}_est"] = w[est][k % len(w[est])] * s
            for bld in ("trim", "est"):   # the glibc build with trimming: the user's request all the same (or its own
                # peak, when its run happens to peak above the glibc run's: run-to-run noise, not the user's choice)
                if f"{bld}_request" in info:
                    info[f"{bld}_request"], info[f"{bld}_um"] = max(info["request"], info[f"{bld}_request"]), info["um"]
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
            mpr = build in ("mpr", "mprh", "mprt", "mpre")
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
            elif is_memrss(policy):
                widened, al = _mode(policy), _aligned(policy)
                e = _shape(i[("mpr_memrss" if mpr else "memrss") + (f"_{al}" if al else "") + "_est"], widened)
                j.reservation = e + self.margin(i["truth"], "mpr" if mpr else None, j.shape, widened=widened,
                                                aligned=al) * e.max()
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
    ap.add_argument("--truths", nargs="+", default=["rss", "strict"])   # or reclaim (RESIZE_EST_RUNS)
    ap.add_argument("--loads", type=float, nargs="+", default=[0.4, 0.6, 0.8, 1.0, 1.2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    ap.add_argument("--locals", type=float, nargs="+", default=[128])
    ap.add_argument("--procs", type=int, default=28)
    ap.add_argument("--overheads", choices=["none", "native", "pin"], default="none")
    ap.add_argument("--inflate", type=float, nargs="+", default=[0.0])
    a = ap.parse_args()
    run(a.out, truths=a.truths, loads=a.loads, seeds=a.seeds, policies=a.policies, locals_=a.locals, procs=a.procs, overheads=a.overheads,
        inflates=a.inflate)


if __name__ == "__main__":
    main()
