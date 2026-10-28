"""libmpr with MPR_TRIM on the resize study's 29 workloads (scripts/trim_eval.sh): what trimming saves, what it costs,
and what dynamic provisioning then holds.

    ~/environments/memfp/bin/python paper_rss/trim.py [RUNS]   ->  paper_rss/data/trim.csv, paper_rss/tables/trim*.tex

Builds (labels of the runs): base (glibc), trim (glibc with MPR_TRIM), self / selftrim (libmpr with its own
profile, without and with MPR_TRIM), heldout / heldouttrim (a held-out profile). Per workload and build, the mean
over the 3 runs of: peak RSS and peak commit (MB), time-averaged RSS (MB), run time, the dynamic-provisioning
reservation's time average (paper_resize/baselines.dynamic: the request for the first sixth of the run, then each
sixth's largest RSS; MB), the trims and whether the output is the glibc run's.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "paper_resize"))
import baselines  # noqa: E402

RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/memprint-data/2026-10-27-rsstrim/runs")
MB = 1e6
BUILDS = ["base", "trim", "self", "selftrim", "heldout", "heldouttrim"]
# the 7 programs whose RSS the allocator inflates most (paper_rss, its allocator-heavy group)
HEAVY = ["minivite", "python", "lua", "sqlite", "perl", "quicksilver", "cc1"]


def trims_of(stem):
    """The trims of one run: its MPR_TRIM_REPORT (.trims), else its monitor's last row; 0 without either."""
    for f, col in ((stem + ".trims", "trims"), (stem + ".monitor.csv", "trims")):
        if os.path.exists(f) and os.path.getsize(f) > 20:
            return int(pd.read_csv(f)[col].iloc[-1])
    return 0


def summary(d):
    t = pd.read_csv(f"{d}/summary.txt", sep=r"\s+", header=None, names=["label", "rss", "commit", "secs", "hash", "status"])
    t["build"], t["rep"] = t.label.str.rsplit("-", n=1).str[0], t.label.str.rsplit("-", n=1).str[1]
    return t


def collect():
    rows = []
    for d in sorted(glob.glob(f"{RUNS}/*/")):
        name = d.rstrip("/").split("/")[-1]
        s = summary(d)
        ref = s[s.build == "base"].hash.mode()
        for b in BUILDS:
            for _, r in s[s.build == b].iterrows():
                f = f"{d}/{r.label}.csv"
                if not os.path.exists(f) or not os.path.getsize(f):
                    continue
                t = pd.read_csv(f)
                secs, u = t.seconds.to_numpy(float), t.rss_bytes.to_numpy(float)
                dt = np.diff(secs, append=secs[-1])
                T = max(secs[-1], 1e-9)
                R = baselines.request(t)
                dyn = baselines.dynamic(u, secs, R)
                trims = trims_of(f"{d}/{r.label}")
                rows.append(dict(workload=name, build=b, rep=r.rep, peak_rss=r.rss / MB, peak_commit=r.commit / MB,
                                 mean_rss=(u * dt).sum() / T / MB, rss_gbs=(u * dt).sum() / 1e9, secs=r.secs, dynamic=(dyn * dt).sum() / T / MB,
                                 trims=trims, same_output=bool(len(ref) and r.hash == ref.iloc[0]),
                                 status=r.status))
    return pd.DataFrame(rows)


def n0(v):
    return "--" if pd.isna(v) else f"{v:,.0f}".replace(",", "{,}")


def tables(df, out):
    m = df.groupby(["workload", "build"]).agg(peak_rss=("peak_rss", "mean"), mean_rss=("mean_rss", "mean"),
                                              peak_commit=("peak_commit", "mean"), secs=("secs", "mean"),
                                              rss_gbs=("rss_gbs", "mean"),
                                              dynamic=("dynamic", "mean"), trims=("trims", "median"),
                                              same=("same_output", "all")).reset_index()
    w = m.pivot(index="workload", columns="build")
    # per workload: glibc and libmpr (own profile), each without and with trimming
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & \multicolumn{4}{c}{Time-averaged RSS (MB)} & \multicolumn{2}{c}{Run time ($\times$ glibc)} "
             r"& \multicolumn{2}{c}{Trims} \\",
             r"\cmidrule(lr){2-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
             r"Workload & glibc & +trim & libmpr & +trim & +trim & libmpr+trim & glibc & libmpr \\", r"\midrule"]
    # the workloads where trimming happened (the others made no trim in any build, or at most one)
    trimmed = [x for x in w.index if max(w.loc[x, ("trims", "trim")], w.loc[x, ("trims", "selftrim")]) > 1]
    order = [x for x in HEAVY if x in trimmed] + [x for x in trimmed if x not in HEAVY]
    for i, k in enumerate(order):
        if i == len([x for x in HEAVY if x in trimmed]):
            lines.append(r"\midrule")
        r = w.loc[k]
        cells = [k.replace("minivite", "miniVite (pass)").replace("_", r"\_")]
        cells += [n0(r[("mean_rss", b)]) for b in ("base", "trim", "self", "selftrim")]
        cells += [f"{r[('secs', b)] / r[('secs', 'base')]:.2f}" for b in ("trim", "selftrim")]
        cells += [n0(r[("trims", b)]) for b in ("trim", "selftrim")]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/trim.tex", "w").write("\n".join(lines) + "\n")
    # totals over the workloads (MB), allocator-heavy and all
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & \multicolumn{4}{c}{Allocator-heavy (7)} & \multicolumn{4}{c}{All (" + str(len(w)) + r")} \\",
             r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
             r"Build & Peak & Mean & RSS$\times$time & Dynamic & Peak & Mean & RSS$\times$time & Dynamic \\",
             r" & (MB) & (MB) & (GB\,s) & (MB) & (MB) & (MB) & (GB\,s) & (MB) \\", r"\midrule"]
    names = {"base": "glibc", "trim": "glibc + trim", "self": "libmpr", "selftrim": "libmpr + trim",
             "heldout": "libmpr, held-out profile", "heldouttrim": "libmpr, held-out + trim"}
    heavy = w.loc[[x for x in HEAVY if x in w.index]]
    for b in BUILDS:
        cells = [names[b]]
        for g in (heavy, w):
            cells += [n0(g[(c, b)].sum()) if c != "rss_gbs" else f"{g[(c, b)].sum():,.1f}".replace(",", "{,}")
                      for c in ("peak_rss", "mean_rss", "rss_gbs", "dynamic")]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/trim_totals.tex", "w").write("\n".join(lines) + "\n")
    return m


def grid_table(out, path=os.path.expanduser("~/memprint-data/2026-10-27-rsstrim/grid.csv")):
    """The cluster simulation (sched.resize_cluster with RESIZE_RUNS): throughput (jobs/h) and mean response (min) of
    each policy without and with trimming, RSS truth, mean of 3 seeds."""
    if not os.path.exists(path):
        return
    g = pd.read_csv(path)
    t = g[g.truth == "rss"].groupby(["policy", "load"])[["throughput", "steady_response"]].mean()
    pairs = [("request", "trim", "Request held"), ("zacarias", "trim+zacarias", "Dynamic provisioning"),
             ("zacarias-past", "trim+zacarias-past", "Dynamic, look-back"), ("mpr", "mprt", "libmpr"),
             ("mpr+zacarias", "mprt+zacarias", "libmpr + dynamic"),
             ("mpr+zacarias-past", "mprt+zacarias-past", "libmpr + dynamic, look-back")]
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & \multicolumn{4}{c}{Load 0.8} & \multicolumn{4}{c}{Load 1.2} \\",
             r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
             r" & \multicolumn{2}{c}{Jobs/h} & \multicolumn{2}{c}{Response (min)} & \multicolumn{2}{c}{Jobs/h} "
             r"& \multicolumn{2}{c}{Response (min)} \\",
             r"Policy & no trim & trim & no trim & trim & no trim & trim & no trim & trim \\", r"\midrule"]
    for a, b, lab in pairs:
        cells = [lab]
        for load in (0.8, 1.2):
            cells += [f"{t.loc[(p, load), 'throughput']:.1f}" for p in (a, b)]
            cells += [n0(t.loc[(p, load), "steady_response"]) for p in (a, b)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/trim_grid.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    df = collect()
    os.makedirs(f"{HERE}/data", exist_ok=True)
    os.makedirs(f"{HERE}/tables", exist_ok=True)
    df.to_csv(f"{HERE}/data/trim.csv", index=False)
    m = tables(df, HERE)
    grid_table(HERE)
    pd.set_option("display.width", 200)
    print(m.pivot(index="workload", columns="build", values="mean_rss").round(1))
    bad = df[(df.status != 0) | ~df.same_output]
    print("runs failed or with other output:", len(bad))
    if len(bad):
        print(bad[["workload", "build", "rep", "status", "same_output"]].to_string())
