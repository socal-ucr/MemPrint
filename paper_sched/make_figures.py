"""Figures for paper_sched/main.tex: what footprint knowledge is worth to a scheduler on disaggregated memory.

    python paper_sched/make_figures.py --data DIR [--out paper_sched/figures]

DIR is the study's data directory (not in the repository), ~/memprint-data/2026-10-13-sched-llm:

    corpus/curves.csv, corpus/costs.csv     every method's footprint curve and cost (analysis/memprint/sched/corpus.py)
    corpus/static_summary.csv               static predictions and their time
    results/grid_all*.csv                   scheduling runs, every program (analysis/memprint/sched/run.py)
    results/grid_static*.csv                the same on the programs the static route predicts
    results/capacity_<GB>.csv               the memory sweep at load 0.8
"""

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from memprint.sched.run import largest  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"   # as paper/make_figures.py (the timeline report)

# the reference palette's eight categorical slots, in order; each estimator family keeps one slot everywhere
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STYLE = {  # oracle: (label, color, marker, dash)
    "exact": ("Exact curve", INK, None, "-"),
    "exact-coarse": ("Exact, 20 windows", INK, None, (0, (4, 2))),
    "exact-peak": ("Exact peak", MUTED, None, (0, (1, 1.5))),
    "memprint-W5": ("MemPrint windowed 5%", PAL[0], "o", "-"),
    "memprint-W1": ("MemPrint windowed 1%", PAL[0], "o", (0, (4, 2))),
    "memprint-P": ("MemPrint pages only", PAL[0], "D", (0, (1, 1.5))),
    "memprint-W5-peak": ("MemPrint windowed 5%, peak", PAL[0], "o", (0, (6, 2, 1, 2))),
    "memprint-static": ("MemPrint static (source)", PAL[0], "*", "-"),
    "massif-i": ("Massif curve", PAL[1], "s", "-"),
    "massif-pages-i": ("Massif pages-as-heap", PAL[1], "s", (0, (4, 2))),
    "massif": ("Massif curve, ms clock", PAL[1], "s", (0, (1, 1.5))),
    "massif-pages": ("Massif pages-as-heap, ms clock", PAL[1], "s", (0, (6, 2, 1, 2))),
    "dhat": ("DHAT peak", PAL[2], "^", "-"),
    "dhat-curve": ("DHAT curve (3 points)", PAL[2], "^", (0, (4, 2))),
    "dhat-hold": ("DHAT hold-then-decay", PAL[2], "^", (0, (1, 1.5))),
    "heaptrack": ("heaptrack curve", PAL[3], "v", "-"),
    "gperf": ("gperftools peak", PAL[4], "P", "-"),
    "rss-peak": ("RSS peak (earlier run)", PAL[5], "X", "-"),
    "request": ("User request", PAL[6], "h", "-"),
    "ml-peak": ("ML peak (quantile GBM)", PAL[7], "p", "-"),
    "reactive": ("Reactive resizing (5 min)", PAL[7], "p", (0, (4, 2))),
    "pond": ("Pond (untouched in pool)", PAL[6], "H", (0, (1, 1.5))),
    "zacarias": ("Dynamic provisioning (5 min)", PAL[7], "8", (0, (1, 1.5))),
    # section "The time axis and the level": MemPrint curves on the native clock, and scaled to an earlier
    # run's RSS peak
    "memprint-W5nat": ("MemPrint 5%, native clock", PAL[4], "o", "-"),
    "memprint-Pnat": ("MemPrint pages, native clock", PAL[4], "D", (0, (1, 1.5))),
    "memprint-W5nat-rss": ("MemPrint 5%, native clock, RSS level", PAL[7], "o", "-"),
    "memprint-Pnat-rss": ("MemPrint pages, native clock, RSS level", PAL[7], "D", (0, (4, 2))),
}
METHOD_LABEL = {"W5": "MemPrint W5", "W1": "MemPrint W1", "P": "MemPrint pages", "static": "MemPrint static",
                "massif": "Massif (ms clock)", "massifpages": "Massif pages (ms clock)", "massifi": "Massif",
                "massifpagesi": "Massif pages", "dhat": "DHAT", "heaptrack": "heaptrack",
                "gperf": "gperftools", "rss": "RSS", "truth": "Full Pin trace",
                "static_all": "MemPrint static (all inputs)"}
METHOD_ORACLE = {"W5": "memprint-W5", "W1": "memprint-W1", "P": "memprint-P", "static": "memprint-static",
                 "massif": "massif", "massifpages": "massif-pages", "massifi": "massif-i",
                 "massifpagesi": "massif-pages-i", "dhat": "dhat", "heaptrack": "heaptrack",
                 "gperf": "gperf", "rss": "rss-peak"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.2, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
})
SCEN = {"bytes": "truth = bytes touched", "rss": "truth = resident set (RSS)",
        "rssdata": "truth = RSS without code"}


def style(o):
    return STYLE.get(o, (o, MUTED, ".", "-"))


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"))
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- corpus figures

def load_corpus(data):
    cv = pd.read_csv(f"{data}/corpus/curves.csv", dtype={"input": str})
    costs = pd.read_csv(f"{data}/corpus/costs.csv", dtype={"input": str})
    return cv, costs


def fig_llm_curves(cv, out, cases=(("llama2c", "110M", "llama2.c, stories110M, 128 tokens"),
                                    ("llamacpp", "135m-n128", "llama.cpp, SmolLM2-135M, 128 tokens"))):
    """The LLM engines: the true footprint against RSS, MemPrint and the heap profilers over the run."""
    fig, axes = plt.subplots(1, len(cases), figsize=(6.8, 2.3))
    for ax, (prog, inp, title) in zip(axes, cases):
        g = cv[(cv.prog == prog) & (cv.input == inp)]
        c = {m: h.sort_values("k").bytes.to_numpy() / 1e6 for m, h in g.groupby("method")}
        p = (np.arange(len(c["truth"])) + 0.5) / len(c["truth"])
        ax.fill_between(p, 0, c["truth"], color=INK, alpha=0.12, linewidth=0, label="True footprint (bytes touched)")
        ax.plot(p, c["truth"], color=INK, linewidth=1.0)
        ax.plot(p, c["rss"], color=PAL[5], linewidth=1.0, linestyle=(0, (1, 1.5)), label="Resident set (RSS)")
        for m, o in [("W5", "memprint-W5"), ("P", "memprint-P"), ("massifi", "massif-i"), ("heaptrack", "heaptrack")]:
            lab, col, _, dash = style(o)
            ax.plot(p, c[m], color=col, linestyle=dash, label=lab)
        ax.axhline(c["dhat"].max(), color=PAL[2], linewidth=0.9, linestyle=(0, (6, 2, 1, 2)), label="DHAT peak")
        ax.set_title(title)
        ax.set_xlabel("Progress through the run")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("Memory (MB)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.3), ncol=4)
    save(fig, out, "llm_curves")


def fig_motivation(cv, out, prog="python", inp="1"):
    """One program: the true footprint over time against what each estimator would make a scheduler reserve."""
    g = cv[(cv.prog == prog) & (cv.input == inp)]
    c = {m: h.sort_values("k").bytes.to_numpy() / 1e6 for m, h in g.groupby("method")}
    p = (np.arange(len(c["truth"])) + 0.5) / len(c["truth"])
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.fill_between(p, 0, c["truth"], color=INK, alpha=0.12, linewidth=0, label="True footprint (bytes touched)")
    ax.plot(p, c["truth"], color=INK, linewidth=1.0)
    ax.plot(p, c["rss"], color=PAL[5], linewidth=1.0, linestyle=(0, (1, 1.5)), label="Resident set (RSS)")
    lines = [("W5", "memprint-W5"), ("massifi", "massif-i"), ("dhatcurve", "dhat-curve")]
    for m, o in lines:
        lab, col, _, dash = style(o)
        ax.plot(p, c[m], color=col, linestyle=dash, label=lab)
    ax.axhline(c["dhat"].max(), color=PAL[2], linewidth=0.9, linestyle=(0, (6, 2, 1, 2)), label="DHAT peak")
    ax.set_xlabel("Progress through the run")
    ax.set_ylabel("Memory (MB)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    save(fig, out, "motivation_curve")


def fig_peak_accuracy(cv, out):
    """Estimated peak / true peak per program at its largest input, one row per method."""
    big = largest(cv)
    w = big.groupby(["prog", "method"]).bytes.max().unstack()
    methods = ["W5", "P", "static", "massifi", "massifpagesi", "dhat", "heaptrack", "gperf", "rss"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, ref in zip(axes, ["truth", "rss"]):
        r = w[methods].div(w[ref], axis=0)
        for i, m in enumerate(methods):
            v = r[m].dropna()
            lab, col, mk, _ = style(METHOD_ORACLE[m])
            jit = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.35
            ax.scatter(np.log2(v.clip(1 / 64, 64)), i + jit, s=10, color=col, marker=mk, alpha=0.8, linewidths=0)
            ax.plot([np.log2(v.median())] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=1.2)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([METHOD_LABEL[m] for m in methods])
        ticks = [-3, -2, -1, 0, 1, 2, 3, 4, 5]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{2.0 ** t:g}" for t in ticks])
        ax.set_xlabel("Estimated peak / true peak (log scale)")
        ax.set_title(SCEN["bytes" if ref == "truth" else "rss"])
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    save(fig, out, "peak_accuracy")


def fig_cost(costs, cv, out, static_summary=None):
    """Time under each method over the program alone, every program at its largest input."""
    big = largest(cv)[["prog", "input"]].drop_duplicates()
    k = costs.merge(big, on=["prog", "input"])
    methods = ["heaptrack", "gperf", "massifi", "massifpagesi", "dhat", "P", "W1", "W5", "truth"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, m in enumerate(methods):
        v = k[k.method == m].slowdown.dropna()
        col = INK if m == "truth" else style(METHOD_ORACLE.get(m, m))[1]
        mk = "x" if m == "truth" else style(METHOD_ORACLE.get(m, m))[2]
        jit = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.35
        ax.scatter(v, i + jit, s=10, color=col, marker=mk, alpha=0.8, linewidths=0 if mk != "x" else 0.8)
        ax.plot([v.median()] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=1.2)
    if static_summary is not None:
        s = static_summary.merge(costs[costs.method == "native"][["prog", "input", "wall_s"]], on=["prog", "input"])
        v = s.seconds_to_predict / s.wall_s.clip(lower=0.01)
        i = len(methods)
        ax.scatter(v, i + (np.random.default_rng(9).random(len(v)) - 0.5) * 0.35, s=14, color=PAL[0], marker="*",
                   linewidths=0)
        ax.plot([v.median()] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=1.2)
        methods = methods + ["static_all"]
    ax.set_xscale("log")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABEL[m] for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel("Cost: time / native run time (log scale)")
    ax.grid(axis="y", visible=False)
    save(fig, out, "cost")


# ---------------------------------------------------------------- scheduling figures

def load_grid(data, pattern):
    files = sorted(glob.glob(f"{data}/results/{pattern}"))
    if not files:
        return None
    g = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "error" in g:
        bad = g.error.notna().sum()
        if bad:
            print(f"{pattern}: {bad} failed runs left out")
        g = g[g.error.isna()]
    g["margin"] = g.margin.astype(str)
    if "restarts" in g:   # dynamic provisioning's failed resizes restart the job: counted with the OOM kills
        g["kills"] = g.kills + g.restarts.fillna(0)
    return g


UNCALIBRATED = {"request", "ml-peak", "reactive", "pond", "zacarias"}   # no profile to calibrate: run with a 10%
# margin (Pond and dynamic provisioning use none, and their single run carries the same label)


def select(g, margin):
    """The rows of one margin policy. Exact oracles take no margin and the uncalibrated ones always have 10%.
    margin = "best": for every (scenario, load, oracle), the policy with the lowest mean response."""
    if margin == "best":
        m = g.groupby(["scenario", "load", "oracle", "margin"]).steady_response.mean().reset_index()
        best = m.loc[m.groupby(["scenario", "load", "oracle"]).steady_response.idxmin()]
        return g.merge(best[["scenario", "load", "oracle", "margin"]], on=["scenario", "load", "oracle", "margin"])
    keep = (g.margin == margin) | g.oracle.str.startswith("exact") | (g.oracle.isin(UNCALIBRATED) & (g.margin == "0.1"))
    return g[keep]


def mean_ci(v):
    v = np.asarray(v, float)
    m = v.mean()
    h = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return m, h


def fig_response_vs_load(g, out, name, oracles, margin="cal", metric="steady_response",
                         ylabel="Mean response time (min)", scenarios=("bytes", "rss")):
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.5 * len(scenarios), 2.6), sharey=True)
    for ax, sc in zip(axes, scenarios):
        for o in oracles:
            d = select(g, margin)
            d = d[(d.scenario == sc) & (d.oracle == o)]
            if d.empty:
                continue
            s = d.groupby("load")[metric].apply(lambda v: mean_ci(v)).apply(pd.Series)
            lab, col, mk, dash = style(o)
            ax.errorbar(s.index, s[0], yerr=s[1], color=col, marker=mk, markersize=3.5, linestyle=dash, label=lab,
                        capsize=1.5, linewidth=1.1 if o != "exact" else 1.6)
        ax.set_yscale("log")
        ax.set_xlabel("Offered load (resident GB-min / memory capacity)")
        ax.set_title(SCEN[sc])
    lo = min(ax.get_ylim()[0] for ax in axes)
    if metric == "steady_response":
        axes[0].set_ylim(max(lo, 30), None)
    axes[0].set_ylabel(ylabel)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4)
    save(fig, out, name)


def fig_margins(g, out, oracles, load=0.8):
    """Response time against OOM kills for every margin policy: no fixed margin escapes the trade-off."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, sc in zip(axes, ["bytes", "rss"]):
        d = g[(g.scenario == sc) & (g.load == load)]
        for o in oracles:
            e = d[d.oracle == o].groupby("margin")[["steady_response", "kills"]].mean()
            if e.empty:
                continue
            lab, col, mk, dash = style(o)
            e = e.sort_values("kills")
            ax.plot(e.kills + 1, e.steady_response, color=col, marker=mk, markersize=4, linestyle="none",
                    label=lab)
        ex = d[d.oracle == "exact"].steady_response.mean()
        ax.scatter([1], [ex], color=INK, marker="o", s=18, zorder=5, label="Exact curve")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("OOM kills + 1 (log scale)")
        ax.set_title(f"{SCEN[sc]}, load {load}")
    axes[0].set_ylabel("Mean response time (min)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4)
    save(fig, out, "margin_tradeoff")


def capacity_needed(cap, target=None):
    """Smallest local GB per node at which each oracle's mean bounded slowdown is within `target` (default:
    1.5x the exact curve's at 128 GB)."""
    m = cap.groupby(["scenario", "oracle", "local_gb"]).steady_bsld.mean().reset_index()
    rows = []
    for sc, d in m.groupby("scenario"):
        tgt = target or 1.5 * d[(d.oracle == "exact") & (d.local_gb == 128)].steady_bsld.iloc[0]
        for o, e in d.groupby("oracle"):
            ok = e[e.steady_bsld <= tgt].local_gb
            rows.append(dict(scenario=sc, oracle=o, target=tgt, local_gb=ok.min() if len(ok) else np.nan))
    return pd.DataFrame(rows)


def fig_capacity(cap, out, nodes=32, pool=1024):
    need = capacity_needed(cap)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True, gridspec_kw={"wspace": 1.0})
    for ax, sc in zip(axes, ["bytes", "rss"]):
        d = need[need.scenario == sc].copy()
        d["total_tb"] = (d.local_gb * nodes + pool) / 1024
        d = d.sort_values("total_tb", na_position="last")
        for i, r in enumerate(d.itertuples()):
            lab, col, mk, _ = style(r.oracle)
            if np.isnan(r.total_tb):
                ax.text(0.1, i, "not met at 320 GB/node", va="center", fontsize=6, color=MUTED)
                continue
            ax.barh(i, r.total_tb, color=col, height=0.6)
            ax.text(r.total_tb + 0.1, i, f"{r.total_tb:.1f} TB", va="center", fontsize=6, color=INK)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([style(o)[0] for o in d.oracle])
        ax.invert_yaxis()
        ax.set_xlabel("Total memory for equal service (TB)")
        ax.set_xlim(0, 13.5)
        ax.set_title(SCEN[sc])
        ax.grid(axis="y", visible=False)
    save(fig, out, "capacity")
    return need


def fig_roi(g, costs, cv, out, static_summary, load=0.8, margin="cal"):
    """What each estimator's cost buys: profiling cost against the response time it gives, against the user's
    request (no profiling, the baseline)."""
    big = largest(cv)[["prog", "input"]].drop_duplicates()
    k = costs.merge(big, on=["prog", "input"])
    cost = k.groupby("method").slowdown.median().to_dict()
    if static_summary is not None:
        s = static_summary.merge(costs[costs.method == "native"][["prog", "input", "wall_s"]], on=["prog", "input"])
        cost["static"] = float((s.seconds_to_predict / s.wall_s.clip(lower=0.01)).median())
    pts = {"memprint-W1": "W1", "memprint-W5": "W5", "memprint-P": "P", "massif-i": "massifi",
           "massif-pages-i": "massifpagesi", "massif": "massif", "dhat": "dhat", "dhat-curve": "dhat", "heaptrack": "heaptrack",
           "gperf": "gperf", "rss-peak": "native", "exact": "truth", "exact-coarse": "truth",
           "exact-peak": "truth"}
    cost["native"] = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, sc in zip(axes, ["bytes", "rss"]):
        d = select(g, margin)
        d = d[(d.scenario == sc) & (d.load == load)]
        base = d[d.oracle == "request"].groupby("seed").steady_response.mean()
        for o, m in pts.items():
            if sc == "rss" and o.startswith("exact"):
                m = "native"   # under the RSS truth the exact curve is the sampled RSS: the cost of a native run
            e = d[d.oracle == o].groupby("seed").steady_response.mean()
            if e.empty or m not in cost:
                continue
            gain = (base / e).dropna()
            mu, h = mean_ci(gain)
            lab, col, mk, _ = style(o)
            if o.startswith("exact"):
                lab = {"exact": "Exact curve (Pin trace / RSS)", "exact-coarse": "Exact curve, 20 windows",
                       "exact-peak": "Exact peak"}[o]
                mk = {"exact": "o", "exact-coarse": "s", "exact-peak": "^"}[o]
            ax.errorbar(cost[m], mu, yerr=h, color=col, marker=mk, markersize=5, linestyle="none", capsize=1.5,
                        label=lab, markerfacecolor=col if not o.startswith("exact") else "white")
        ax.axhline(1, color=MUTED, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Profiling cost per job (time / native run, log)")
        ax.set_title(f"{SCEN[sc]}, load {load}")
    axes[0].set_ylabel("Response-time gain over user request (x)")
    axes[0].set_ylim(0.3, 60)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4)
    save(fig, out, "roi")
    return cost


def fig_dhat(g, out, load=0.8):
    """DHAT's peak, its three-point curves, and MemPrint, at every margin policy."""
    oracles = ["dhat", "dhat-curve", "dhat-hold", "memprint-W5", "exact-peak", "exact"]
    margins = ["0.1", "0.3", "cal", "cal100"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, sc in zip(axes, ["bytes", "rss"]):
        d = g[(g.scenario == sc) & (g.load == load)]
        w = 0.8 / len(oracles)
        for i, o in enumerate(oracles):
            vals = []
            for mg in margins:
                e = d[(d.oracle == o) & ((d.margin == mg) | o.startswith("exact"))].steady_response
                vals.append(e.mean() if len(e) else np.nan)
            lab, col, _, _ = style(o)
            hatch = "////" if o == "dhat-curve" else ("...." if o == "dhat-hold" else None)
            ax.bar(np.arange(len(margins)) + (i - len(oracles) / 2 + 0.5) * w, vals, width=w * 0.9, color=col,
                   label=lab, hatch=hatch, edgecolor="white", linewidth=0.3)
        ax.set_xticks(range(len(margins)))
        ax.set_xticklabels(["10%", "30%", "cal. 90%", "cal. 100%"])
        ax.set_yscale("log")
        ax.set_title(f"{SCEN[sc]}, load {load}")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean response time (min)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    save(fig, out, "dhat_curves")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/memprint-data/2026-10-13-sched-llm"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cv, costs = load_corpus(a.data)
    ss = f"{a.data}/corpus/static_summary.csv"
    static_summary = pd.read_csv(ss, dtype={"input": str}) if os.path.exists(ss) else None
    fig_motivation(cv, a.out)
    fig_peak_accuracy(cv, a.out)
    fig_cost(costs, cv, a.out, static_summary)
    g = load_grid(a.data, "grid_all*.csv")
    if g is not None:
        main_oracles = ["exact", "exact-coarse", "exact-peak", "memprint-W5", "massif-i", "massif", "dhat", "dhat-curve",
                        "heaptrack", "rss-peak", "request", "ml-peak", "reactive", "pond", "zacarias"]
        fig_response_vs_load(g, a.out, "response_vs_load", main_oracles)
        fig_response_vs_load(g, a.out, "response_vs_load_best", main_oracles, margin="best")
        fig_response_vs_load(g, a.out, "kills_vs_load", main_oracles, metric="kills", ylabel="OOM kills")
        fig_margins(g, a.out, ["memprint-W5", "massif-i", "massif", "dhat", "dhat-curve", "heaptrack", "rss-peak"])
        fig_dhat(g, a.out)
        cost = fig_roi(g, costs, cv, a.out, static_summary, margin="best")
    gs = load_grid(a.data, "grid_static*.csv")
    if gs is not None:
        fig_response_vs_load(gs, a.out, "static_response_vs_load",
                             ["exact", "exact-peak", "memprint-static", "memprint-W5", "massif-i", "massif", "dhat",
                              "dhat-curve", "heaptrack", "request", "reactive", "pond", "zacarias"])
    fig_llm_curves(cv, a.out)
    gl = load_grid(a.data, "grid_llm*.csv")
    if gl is not None:
        fig_response_vs_load(gl, a.out, "llm_response_vs_load",
                             ["exact", "exact-coarse", "exact-peak", "memprint-W5", "memprint-P", "massif-i", "dhat",
                              "heaptrack", "gperf", "rss-peak", "request", "reactive", "pond", "zacarias"])
    v3 = os.path.expanduser("~/memprint-data/2026-10-13-sched-fm")
    gv = load_grid(v3, "v3m_all.csv")
    if gv is not None:
        fig_response_vs_load(gv, a.out, "timeaxis_response_vs_load",
                             ["exact", "exact-peak", "rss-peak", "memprint-W5", "memprint-W5nat",
                              "memprint-W5nat-rss", "memprint-Pnat-rss", "dhat", "request"], scenarios=("rss", "rssdata"))
    cap = load_grid(a.data, "capacity_*.csv")
    if cap is not None:
        need = fig_capacity(cap, a.out)
        need.to_csv(os.path.join(a.out, "capacity_needed.csv"), index=False)


if __name__ == "__main__":
    main()
