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


DATA = os.path.expanduser("~/memprint-data/2026-10-27-rsstrim")


def grid_table(out, paths=(f"{DATA}/grid.csv", f"{DATA}/grid_est.csv")):
    """The cluster simulation (sched.resize_cluster with RESIZE_RUNS, and RESIZE_EST_RUNS for the bound): throughput
    (jobs/h) and mean response (min) of each policy without trimming, with trimming and at the bound, RSS truth, mean
    of 3 seeds; the bound's change against no trimming on each seed and load."""
    if not all(os.path.exists(p) for p in paths):
        return
    g = pd.concat([pd.read_csv(p) for p in paths])
    g = g[g.truth == "rss"]
    t = g.groupby(["policy", "load"])[["throughput", "steady_response"]].mean()
    paired = g.pivot_table(index=["load", "seed"], columns="policy", values="throughput")
    rows = [("request", "trim", None, "Request held"), ("zacarias", "trim+zacarias", "est+zacarias", "Dynamic provisioning"),
            ("zacarias-past", "trim+zacarias-past", "est+zacarias-past", "Dynamic, look-back"),
            ("mpr", "mprt", "mpre", "libmpr"), ("mpr+zacarias", "mprt+zacarias", "mpre+zacarias", "libmpr + dynamic"),
            ("mpr+zacarias-past", "mprt+zacarias-past", "mpre+zacarias-past", "libmpr + dynamic, look-back")]
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{Load 0.8} & \multicolumn{2}{c}{Load 1.2} & Seeds with \\",
             r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
             r"Policy & Memory & Jobs/h & Resp.\ (min) & Jobs/h & Resp.\ (min) & more jobs/h \\", r"\midrule"]
    for i, (a, b, c, lab) in enumerate(rows):
        if i:
            lines.append(r"\addlinespace")
        for p, mem in ((a, "glibc's"), (b, "trimmed"), (c, "bound")):
            if p is None:
                continue
            cells = [lab if p == a else "", mem]
            for load in (0.8, 1.2):
                cells += [f"{t.loc[(p, load), 'throughput']:.1f}", n0(t.loc[(p, load), "steady_response"])]
            d = paired[p] - paired[a]
            cells.append("" if p == a else f"{(d > 0).sum()} of {len(d)}")
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/trim_grid.tex", "w").write("\n".join(lines) + "\n")


def share_table(out):
    """MPR_TRIM's share s on glibc (the trim build of each share's runs; the untrimmed runs from the s = 0.1 runs,
    as the sweep's libmpr builds also trimmed): the trims per run summed over the allocator-heavy programs, the
    time-averaged RSS (MB) summed over the allocator-heavy programs and over all, and the run time over glibc's in the
    same set of runs (median and largest)."""
    global RUNS
    sets = [("none", "runs"), ("0.02", "runs_s0.02"), ("0.1", "runs"), ("0.3", "runs_s0.3")]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & Trims & \multicolumn{2}{c}{Time-averaged RSS (MB)} & \multicolumn{2}{c}{Run time ($\times$ glibc)} \\",
             r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
             r"Share $s$ & per run & Alloc.-heavy & All & Median & Largest \\", r"\midrule"]
    for share, d in sets:
        RUNS = f"{DATA}/{d}"
        m = collect().groupby(["workload", "build"])[["mean_rss", "secs", "trims"]].mean().unstack()
        b = "base" if share == "none" else "trim"
        heavy = [x for x in HEAVY if x in m.index]
        r = m[("secs", b)] / m[("secs", "base")]
        cells = [share, n0(m.loc[heavy, ("trims", b)].sum()), n0(m.loc[heavy, ("mean_rss", b)].sum()),
                 n0(m[("mean_rss", b)].sum()), f"{r.median():.2f}", f"{r.max():.2f}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/trim_share.tex", "w").write("\n".join(lines) + "\n")


def bound_numbers():
    """On the simulator's curves (less the runtime's pages): time-averaged and peak RSS (MB) summed over the
    allocator-heavy programs and over all, under glibc, trimmed and at the bound, for glibc and libmpr."""
    os.environ.setdefault("RESIZE_RUNS", f"{DATA}/runs")
    os.environ.setdefault("RESIZE_EST_RUNS", f"{DATA}/runs_free")
    sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
    from memprint.sched import resize_cluster
    w = resize_cluster.workloads()
    heavy = ["miniVite (pass)" if x == "minivite" else x for x in HEAVY]
    rows = []
    for k in ("glibc", "trim", "est", "mpr", "mprt", "mpre"):
        for grp, names in (("heavy", heavy), ("all", list(w))):
            rows.append(dict(build=k, group=grp,
                             mean=sum(np.mean([c[0].mean() for c in w[n][k]]) for n in names) / MB,
                             peak=sum(np.mean([c[0].max() for c in w[n][k]]) for n in names) / MB))
    return pd.DataFrame(rows).pivot(index="build", columns="group").round(0)


if __name__ == "__main__":
    df = collect()
    os.makedirs(f"{HERE}/data", exist_ok=True)
    os.makedirs(f"{HERE}/tables", exist_ok=True)
    df.to_csv(f"{HERE}/data/trim.csv", index=False)
    m = tables(df, HERE)
    grid_table(HERE)
    share_table(HERE)
    print(bound_numbers())
    pd.set_option("display.width", 200)
    print(m.pivot(index="workload", columns="build", values="mean_rss").round(1))
    bad = df[(df.status != 0) | ~df.same_output]
    print("runs failed or with other output:", len(bad))
    if len(bad):
        print(bad[["workload", "build", "rep", "status", "same_output"]].to_string())
