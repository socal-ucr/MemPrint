"""Cluster throughput with MemPrintResize against Pond, dynamic provisioning and KS+: figure and tables from
analysis/memprint/sched/resize_cluster.py's runs.

    ~/environments/memfp/bin/python paper_resize/cluster.py [--data DIR] [--out paper_resize]

DIR holds grid_<o>.csv (every load) and capacity_<o>.csv (the memory sweep at load 0.8) for o = none (no policy
costs), native (with costs, MemPrintResize profiled at native speed) and pin (profiled with the Pin tool),
~/memprint-data/2026-10-22-resize-cluster.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.2, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
})
# label, colour, marker, line: glibc builds dashed, libmpr builds solid
STYLE = {
    "exact": ("Exact curve (glibc build)", INK, None, (0, (4, 2))),
    "request": ("Request (glibc build)", MUTED, "h", (0, (4, 2))),
    "pond": ("Pond", PAL[6], "H", (0, (4, 2))),
    "zacarias": ("Dynamic provisioning", PAL[7], "8", (0, (4, 2))),
    "zacarias-past": ("Dynamic provisioning, look-back", PAL[3], "8", (0, (4, 2))),
    "ksplus": ("KS+", PAL[4], "d", (0, (4, 2))),
    "mpr-exact": ("Exact curve (pass build)", INK, None, "-"),
    "mpr": ("MemPrintResize", PAL[0], "o", "-"),
    "mpr+zacarias": ("MemPrintResize + dynamic", PAL[7], "8", "-"),
    "mpr+zacarias-past": ("MemPrintResize + dynamic, look-back", PAL[3], "8", "-"),
    "mpr+pond": ("MemPrintResize + Pond", PAL[6], "H", "-"),
    "mprh": ("MemPrintResize, held-out profile", PAL[0], "o", (0, (1, 1.5))),
    "mprh+zacarias": ("MemPrintResize, held-out, + dynamic", PAL[7], "8", (0, (1, 1.5))),
}
TRUTH = {"rss": "demand = RSS", "strict": "demand = max(commit, RSS)"}


def load(data, name, inflate=0.0):
    """One results file; inflate selects the over-request (None: every one)."""
    d = pd.read_csv(f"{data}/{name}")
    if "error" in d:
        d = d[d.error.isna()]
    if "inflate" not in d:
        d["inflate"] = 0.0
    if inflate is not None:
        d = d[d.inflate == inflate]
    d["fails"] = d.kills + d.restarts.fillna(0)
    return d


def fig_load(g, out):
    """Throughput and mean response time against offered load, both demands."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    for col, tr in enumerate(("rss", "strict")):
        d = g[(g.truth == tr) & (g.local_gb == 128)]
        for row, (metric, lab) in enumerate((("throughput", "Throughput (jobs/h)"),
                                             ("steady_response", "Mean response time (min)"))):
            ax = axes[row][col]
            for p, (name, c, mk, ls) in STYLE.items():
                e = d[d.policy == p].groupby("load")[metric].agg(["mean", "std"])
                if e.empty:
                    continue
                ax.errorbar(e.index, e["mean"], yerr=e["std"], color=c, marker=mk, markersize=3.5, linestyle=ls,
                            capsize=1.5, label=name, linewidth=1.6 if p.endswith("exact") else 1.1)
            if metric == "steady_response":
                ax.set_yscale("log")
            ax.set_ylabel(lab)
            if row == 0:
                ax.set_title(TRUTH[tr])
            else:
                ax.set_xlabel("Offered load (glibc stream)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    fig.savefig(f"{out}/figures/cluster.pdf")
    plt.close(fig)


def capacity_needed(cap):
    """Smallest local GB per node at which each policy's mean bounded slowdown is within 1.5x the glibc exact
    curve's at 128 GB (as the scheduling study's memory sweep), per demand."""
    m = cap.groupby(["truth", "policy", "local_gb"]).steady_bsld.mean().reset_index()
    rows = []
    for tr, d in m.groupby("truth"):
        tgt = 1.5 * d[(d.policy == "exact") & (d.local_gb == 128)].steady_bsld.iloc[0]
        for p, e in d.groupby("policy"):
            ok = e[e.steady_bsld <= tgt].local_gb
            rows.append(dict(truth=tr, policy=p, local_gb=ok.min() if len(ok) else np.nan))
    n = pd.DataFrame(rows)
    n["tb"] = (n.local_gb * 32 + 1024) / 1024
    return n


def fmt(v, d=0):
    return "--" if pd.isna(v) else f"{v:,.{d}f}".replace(",", "{,}")


def table(g, cap, out, loads=(0.8, 1.2)):
    """Per policy: throughput and mean response at two loads, OOM kills and restarts at 0.8, and the memory needed
    for the same service (TB), both demands."""
    need = capacity_needed(cap) if cap is not None else None
    t = g[g.local_gb == 128].groupby(["truth", "policy", "load"])[["throughput", "steady_response", "fails"]].mean()
    lines = [r"\begin{tabular}{l" + "r" * 12 + "}", r"\toprule",
             r" & \multicolumn{6}{c}{Demand = RSS} & \multicolumn{6}{c}{Demand = max(commit, RSS)} \\",
             r"\cmidrule(lr){2-7} \cmidrule(lr){8-13}",
             " & " + " & ".join([r"\multicolumn{2}{c}{Jobs/h}", r"\multicolumn{2}{c}{Resp.\ (min)}", "Fails", "TB"] * 2)
             + r" \\",
             "Load & " + " & ".join([f"{loads[0]:g}", f"{loads[1]:g}"] * 2 + [f"{loads[0]:g}", f"{loads[0]:g}"]) + " & "
             + " & ".join([f"{loads[0]:g}", f"{loads[1]:g}"] * 2 + [f"{loads[0]:g}", f"{loads[0]:g}"]) + r" \\",
             r"\midrule"]
    for p, (name, *_ ) in STYLE.items():
        cells = [name]
        for tr in ("rss", "strict"):
            cells += [fmt(t["throughput"].get((tr, p, l)), 1) for l in loads]
            cells += [fmt(t["steady_response"].get((tr, p, l))) for l in loads]
            cells.append(fmt(t["fails"].get((tr, p, loads[0]))))
            tb = None if need is None else need[(need.truth == tr) & (need.policy == p)].tb
            cells.append("--" if tb is None or not len(tb) or pd.isna(tb.iloc[0]) else f"{tb.iloc[0]:.1f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/cluster.tex", "w").write("\n".join(lines) + "\n")
    return t, need


def table_costs(grids, out, load=0.8, truth="rss"):
    """What each policy costs at one load: throughput and response without and with its costs (native-speed and
    Pin profiling for MemPrintResize), work lost to OOM kills and restarts, time its own actions add (resizes,
    migrations, libmpr's runtime ratio is in the runtimes), profiling time, and its control actions per stream."""
    def at(o):
        g = grids[o]
        return g[(g.truth == truth) & (g.load == load) & (g.local_gb == 128)].groupby("policy").mean(numeric_only=True)
    a, n, p = at("none"), at("native"), at("pin")
    lines = [r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{Jobs/h} & \multicolumn{3}{c}{Resp.\ (min)} & Lost & Actions & \multicolumn{2}{c}{Profiling} \\",
             r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){10-11}",
             r"Policy & no cost & native & Pin & no cost & native & Pin & core-h & core-h & native & Pin \\",
             r"\midrule"]
    for pol, (name, *_) in STYLE.items():
        if pol not in a.index:
            continue
        v = lambda d, c: d.loc[pol, c] if pol in d.index else float("nan")   # round 1's Pin runs lack the new policies
        cells = [name, fmt(v(a, "throughput"), 1), fmt(v(n, "throughput"), 1), fmt(v(p, "throughput"), 1),
                 fmt(v(a, "steady_response")), fmt(v(n, "steady_response")), fmt(v(p, "steady_response")),
                 fmt(v(n, "wasted_core_h")), fmt(v(n, "overhead_core_h")), fmt(v(n, "profiling_core_h")),
                 fmt(v(p, "profiling_core_h"))]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/cluster_costs.tex", "w").write("\n".join(lines) + "\n")
    cols = ["throughput", "steady_response", "wasted_core_h", "overhead_core_h", "profiling_core_h", "kills", "restarts",
            "resizes", "migrations", "sim_cpu_s"]
    return pd.concat({"none": a[cols], "native": n[cols], "pin": p.reindex(columns=cols)}, axis=1)


def table_inflate(g, out, load=0.8):
    """Throughput and mean response time at one load as the glibc build's request grows past the allocation."""
    t = g[(g.load == load) & (g.local_gb == 128)].groupby(["truth", "policy", "inflate"])[["throughput", "steady_response"]].mean()
    infl = sorted(g.inflate.unique())
    lines = [r"\begin{tabular}{l" + "r" * (2 * len(infl)) + "}", r"\toprule",
             r" & \multicolumn{%d}{c}{Jobs/h} & \multicolumn{%d}{c}{Resp.\ (min)} \\" % (len(infl), len(infl)),
             r"\cmidrule(lr){2-%d} \cmidrule(lr){%d-%d}" % (1 + len(infl), 2 + len(infl), 1 + 2 * len(infl)),
             "Request over allocation & " + " & ".join([f"+{100 * o:.0f}\\%" for o in infl] * 2) + r" \\", r"\midrule"]
    for p, (name, *_) in STYLE.items():
        if (("rss", p, infl[0])) not in t.index:
            continue
        cells = [name] + [fmt(t["throughput"].get(("rss", p, o)), 1) for o in infl] + \
                [fmt(t["steady_response"].get(("rss", p, o))) for o in infl]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/cluster_inflate.tex", "w").write("\n".join(lines) + "\n")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/memprint-data/2026-10-22-resize-cluster"))
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    # round 2 (grid2_*, capacity2_*): every policy, over-requests; round 1's Pin-profiling runs (grid_pin)
    g = load(a.data, "grid2_none.csv")
    cap = load(a.data, "capacity2_none.csv")
    fig_load(g, a.out)
    t, need = table(g, cap, a.out)
    print(table_inflate(load(a.data, "grid2_none.csv", None), a.out).unstack("inflate").round(1).to_string())
    grids = {"none": g, "native": load(a.data, "grid2_native.csv"), "pin": load(a.data, "grid_pin.csv")}
    if len(grids) == 3:
        c = table_costs(grids, a.out)
        print(c.round(1).to_string())
        caps = {o: capacity_needed(load(a.data, f"capacity_{o}.csv")) for o in ("native", "pin")
                if os.path.exists(f"{a.data}/capacity_{o}.csv")}
        caps["over-request 60%"] = capacity_needed(load(a.data, "capacity2_none.csv", 0.6).assign(
            policy=lambda d: d.policy))
        for o, nd in caps.items():
            print("capacity with costs,", o); print(nd.pivot(index="policy", columns="truth", values="tb").to_string())
    pd.set_option("display.width", 200)
    print(t.unstack("load").round(1).to_string())
    if need is not None:
        print(need.pivot(index="policy", columns="truth", values="tb").to_string())


if __name__ == "__main__":
    main()
