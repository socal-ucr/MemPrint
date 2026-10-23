"""MemPrintResize against three published memory-provisioning systems, on the recorded runs of the resize study.

    ~/environments/memfp/bin/python paper_resize/baselines.py [--data DIR] [--sched DIR] [--out paper_resize]

Pond (Li et al., ASPLOS 2023), dynamic provisioning (Zacarias et al., SC-W 2023) and KS+ (Bader et al., eScience
2024) decide how much memory a system holds for a job; the program itself is unchanged, so it commits and touches
what it does under glibc. MemPrintResize changes what the program commits and touches. For every workload, from
the RSS and commit timelines of its runs (glibc, libmpr with its own profile, libmpr with a held-out profile), we
compute what each approach holds for the job over its run:

  demand u(t)   what the job must be given at time t: its RSS (default overcommit, a cgroup limit), or
                max(commit, RSS) (strict overcommit: the commitment must be backed, and what is resident held)
  request R     what the job allocates: its peak max(commit, RSS) under glibc, under both accountings (an exact
                request: no user overestimate)
  request       hold R for the whole run
  Pond          hold R; only R - UM of it local, UM = the untouched memory predicted from the job's history: the
                smallest untouched share (1 - peak RSS / R) of its other runs of the same input (Pond's per-customer
                history at its best), times R; never fails
  dynamic       R for the first interval, then each interval the largest u of the coming interval (as their
                simulator reads it from the usage trace), interval = 1/6 of the run (5 of a median 30 minutes);
                never fails when the job runs alone
  KS+           the step function KS+ predicts from the RSS of the program's other corpus inputs
                (sched.ksplus), and its retries: a run whose u exceeds the reservation fails there and runs again
                with the retry strategy's reservation
  MemPrintResize  the request under libmpr: hold the libmpr run's peak max(commit, RSS) (own or held-out profile)
  + combined    dynamic provisioning and Pond applied to the libmpr run

Metrics, per workload and in total: peak memory held, the time-averaged memory held over the successful run, local
DRAM held (Pond), KS+'s wastage (GB-s held beyond u, plus everything held by failed attempts) and its failures.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from memprint.sched import ksplus  # noqa: E402
from memprint.sched.model import K  # noqa: E402

MB = 1e6
INTERVAL = 1 / 6

LLM = {"llamacpp-135m-n32": ("llamacpp", "135m-n32"), "llamacpp-135m-n128": ("llamacpp", "135m-n128"),
       "llamacpp-360m-n64": ("llamacpp", "360m-n64"), "llama2c-15M": ("llama2c", "15M"),
       "llama2c-42M": ("llama2c", "42M"), "llama2c-110M": ("llama2c", "110M")}


def workloads(data):
    """(name, corpus program, corpus input, run directory, glibc label): the 23 corpus programs at their largest
    inputs, miniVite built with the pass, and the six LLM configurations."""
    R = f"{data}/runs"
    big = {}
    for line in open(os.path.expanduser("~/memprint-data/2026-10-11-sched/corpus/list.txt")):
        f = line.split()
        if f and not (f[0] == "npb-mg" and f[1] == "A"):
            big[f[0]] = f[1]   # the last line of each program is its largest input
    out = []
    for d in sorted(glob.glob(f"{R}/corpus/*/")):
        prog = d.rstrip("/").split("/")[-1]
        if prog == "miniVite":
            continue   # miniVite is evaluated as built with the pass (below), as in the headline results
        out.append((prog, prog, big[prog], d, "base"))
    out.append(("miniVite (pass)", "miniVite", "32768", f"{R}/minivite-v4/", "none"))
    for name, (prog, inp) in LLM.items():
        out.append((name, prog, inp, f"{R}/ml/{name}/", "base"))
    return out


def runs(d, label):
    return [pd.read_csv(f) for f in sorted(glob.glob(f"{d}/{label}-[0-9].csv")) if os.path.getsize(f)]


def request(t):
    return float(np.maximum(t.rss_bytes, t.commit_bytes).max())


def untouched(others):
    """Pond's prediction from history: the smallest untouched share of the request over the other runs."""
    return max(min(1 - x.rss_bytes.max() / request(x) for x in others), 0.0)


def demand(t, strict):
    rss, com = t.rss_bytes.to_numpy(float), t.commit_bytes.to_numpy(float)
    return np.maximum(rss, com) if strict else rss


def held_stats(h, u, secs):
    """Peak and time-averaged memory held (MB), and GB-s held beyond the demand."""
    dt = np.diff(secs, append=secs[-1])
    T = max(secs[-1], 1e-9)
    return h.max() / MB, (h * dt).sum() / T / MB, float((np.maximum(h - u, 0) * dt).sum()) / 1e9


def dynamic(u, secs, R):
    """Dynamic provisioning: R for the first interval, then each interval the largest demand of that interval."""
    T = max(secs[-1], 1e-9)
    k = np.floor(secs / (INTERVAL * T)).astype(int)
    h = np.full(len(u), R, float)
    for b in range(1, k.max() + 1):
        m = k == b
        if m.any():
            h[m] = u[m].max()
    return h


def ksplus_run(model, u, secs, floor=ksplus.FLOOR_GB * 1e9, limit=200):
    """KS+ on one run: the reservation over the run's progress; on a failure (demand above it), the retry
    strategy's next reservation and the run again. Returns the final held curve, failures, and the GB-s held by
    failed attempts."""
    ends, peaks = list(model[0]), [max(p, floor) for p in model[1]]
    peaks = list(np.maximum.accumulate(peaks))
    T = max(secs[-1], 1e-9)
    prog = secs / T
    dt = np.diff(secs, append=secs[-1])
    fails, wasted = 0, 0.0
    while True:
        idx = np.minimum(np.searchsorted(ends, prog, side="right"), len(ends) - 1)
        h = np.asarray(peaks, float)[idx]
        over = np.flatnonzero(u > h + 1)
        if not len(over) or fails >= limit:
            return h, fails, wasted
        i = over[0]
        wasted += float((h[:i] * dt[:i]).sum()) / 1e9   # everything held by the failed attempt
        ends, peaks = ksplus.fail_at(ends, peaks, prog[i])
        fails += 1


def compare(data, sched):
    curves = pd.read_csv(f"{sched}/corpus/curves.csv", dtype={"input": str})
    costs = pd.read_csv(f"{sched}/corpus/costs.csv", dtype={"input": str})
    wl = workloads(data)
    # KS+ from every RSS history the corpus has (it needs no Pin truth), raw RSS as the runs record it
    models = ksplus.models(curves, costs, [(p, i, n) for n, p, i, _, _ in wl], lambda info: info)
    rows = []
    for name, prog, inp, d, base_label in wl:
        base, self_, held = runs(d, base_label), runs(d, "self"), runs(d, "heldout")
        if not base or not self_:
            continue
        for strict in (False, True):
            acc = "strict" if strict else "rss"
            # every quantity is the mean over the run repetitions (run i of glibc with run i of libmpr)
            agg = {}
            for i, b in enumerate(base):
                secs = b.seconds.to_numpy(float)
                u = demand(b, strict)
                R = request(b)
                others = [x for j, x in enumerate(base) if j != i] or [b]
                um = untouched(others) * R
                res = {"request": (np.full(len(u), R), u, secs), "pond": (np.full(len(u), R), u, secs),
                       "dynamic": (dynamic(u, secs, R), u, secs)}
                vals = {k: held_stats(*v) for k, v in res.items()}
                vals["pond_local"] = ((R - um) / MB,) * 2 + (0.0,)
                m = models.get(name)
                for lab, floor in (("ksplus", ksplus.FLOOR_GB * 1e9), ("ksplus1mb", 1e6)):
                    if m is None:   # no other input (darknet): the request, as in the scheduling study
                        vals[lab], vals[lab + "_fails"] = vals["request"], (0, 0, 0)
                        continue
                    h, f, w = ksplus_run(m, u, secs, floor)
                    pk, mean, waste = held_stats(h, u, secs)
                    vals[lab] = (pk, mean, waste + w)
                    vals[lab + "_fails"] = (f, f, f)
                for lab, runs_ in (("mpr", self_), ("mpr_heldout", held)):
                    if i < len(runs_):
                        r = runs_[i]
                        s2, u2, R2 = r.seconds.to_numpy(float), demand(r, strict), request(r)
                        vals[lab] = held_stats(np.full(len(u2), R2), u2, s2)
                        if lab == "mpr":
                            vals["mpr+dynamic"] = held_stats(dynamic(u2, s2, R2), u2, s2)
                            others2 = [x for j, x in enumerate(self_) if j != i] or [r]
                            vals["mpr+pond_local"] = ((R2 * (1 - untouched(others2))) / MB,) * 2 + (0.0,)
                for k, v in vals.items():
                    agg.setdefault(k, []).append(v)
            row = dict(workload=name, accounting=acc)
            for k, v in agg.items():
                a = np.mean(np.array(v, float), axis=0)
                row[f"{k}_peak"], row[f"{k}_mean"], row[f"{k}_waste"] = a
            rows.append(row)
    return pd.DataFrame(rows)


def n0(v):
    return "--" if pd.isna(v) else f"{v:,.0f}".replace(",", "{,}")


APPROACHES = [("request", "Exact request (glibc)"), ("pond", "Pond: all held"), ("pond_local", "Pond: local DRAM"),
              ("dynamic", "Dynamic provisioning"), ("ksplus", "KS+ (100\\,MB floor)"),
              ("ksplus1mb", "KS+ (1\\,MB floor)"), ("mpr", "MemPrintResize"),
              ("mpr_heldout", "MemPrintResize, held-out profile"), ("mpr+pond_local", "MemPrintResize + Pond: local"),
              ("mpr+dynamic", "MemPrintResize + dynamic")]


def table_totals(df, out):
    """Sums over all workloads: peak and time-averaged memory held (MB), under both accountings."""
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Demand = RSS} & \multicolumn{2}{c}{Demand = max(commit, RSS)} \\",
             r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
             r"Approach & Peak & Mean & Peak & Mean \\", r"\midrule"]
    tot = {a: df[df.accounting == a].select_dtypes("number").sum() for a in ("rss", "strict")}
    for k, lab in APPROACHES:
        cells = [lab]
        for a in ("rss", "strict"):
            cells += [n0(tot[a].get(f"{k}_peak")), n0(tot[a].get(f"{k}_mean"))]
        lines.append(" & ".join(cells) + r" \\")
    t = df[df.accounting == "rss"]
    st = df[df.accounting == "strict"]
    lines += [r"\midrule"] + [" & ".join([f"{lab} failures", n0(t[f"{k}_fails_peak"].sum()), "",
                                         n0(st[f"{k}_fails_peak"].sum()), ""]) + r" \\"
                              for k, lab in (("ksplus", "KS+ (100\\,MB)"), ("ksplus1mb", "KS+ (1\\,MB)"))] + [
              r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/baselines_totals.tex", "w").write("\n".join(lines) + "\n")


def table_workloads(df, out, acc="strict"):
    """Per workload, the time-averaged memory held (MB) under one accounting."""
    keys = ["request", "pond_local", "dynamic", "ksplus1mb", "mpr", "mpr+dynamic"]
    head = ["Glibc request", "Pond local", "Dynamic", "KS+ (fails)", "MemPrintResize", "MPR + dynamic"]
    lines = [r"\begin{tabular}{l" + "r" * len(keys) + "}", r"\toprule",
             "Workload & " + " & ".join(head) + r" \\", r"\midrule"]
    for _, r in df[df.accounting == acc].iterrows():
        cells = [r.workload.replace("_", r"\_")]
        for k in keys:
            v = n0(r.get(f"{k}_mean"))
            if k == "ksplus1mb" and not pd.isna(r.get("ksplus1mb_fails_peak")):
                v += f" ({r.ksplus1mb_fails_peak:.0f})"
            cells.append(v)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/baselines_{acc}.tex", "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/memprint-data/2026-10-12-resize"))
    ap.add_argument("--sched", default=os.path.expanduser("~/memprint-data/2026-10-13-sched-llm"))
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    df = compare(a.data, a.sched)
    df.to_csv(f"{a.data}/results/baselines.csv", index=False)
    os.makedirs(f"{a.out}/tables", exist_ok=True)
    table_totals(df, a.out)
    for acc in ("rss", "strict"):
        table_workloads(df, a.out, acc)
    print(f"{len(df)} rows -> {a.data}/results/baselines.csv, tables/baselines_*.tex")


if __name__ == "__main__":
    main()
