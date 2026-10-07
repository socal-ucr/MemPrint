"""Figures for paper/memprint_timeline.tex.

    python paper/make_figures.py --scratch DIR [--out paper/figures]

DIR holds the experiment outputs this paper reports (not in the repository):

    tld5/data/timeline_curves.csv      reference-sampling estimates over time (timeline build)
    tld12/traces/2mm/                   2mm MEDIUM splitter and spatial timelines
    tld12/data/spatial_eval.csv         spatial sampling on PolyBench (held-out runs)
    tld11/spatial_eval.csv              spatial sampling on miniVite, 1/4/16 threads
    win/eval6/<workload>/               windowed runs and their splitter truth (--eval)
    win/eval6/results.csv               timeline windowed results
    win/timing6.txt                     sequential timings (--timing) ("<workload> <mode> <seconds>"; a
                                        repeated line is a rerun and replaces the earlier one)
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reference categorical palette (light mode), slots in fixed order; truth in ink.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DASHES = ["-", (0, (5, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2)), (0, (3, 1, 1, 1, 1, 1))]
MB = 1e6
EVAL, TIMING = "win/eval6", "win/timing6.txt"  # overridden by --eval / --timing

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.2, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
})


def series(ax, x, y, slot, label, **kw):
    ax.plot(x, y, color=SLOTS[slot], linestyle=DASHES[slot], label=label, zorder=3, **kw)


def truth(ax, x, y, label="Truth (full trace)"):
    # drawn under the estimates, wider, so an estimate that matches it stays visible on top
    ax.plot(x, y, color=INK, linewidth=2.4, alpha=0.35, label=label, zorder=2, solid_capstyle="butt")


def splitter_truth(path):
    t = pd.read_csv(path)
    t = t[t["Bin"] == -1]
    return t["Time"] / t["Time"].max(), t["MemUsageObs"]


def fig_reference_vs_spatial(scratch, out):
    """2mm MEDIUM: the estimators of this paper on one held-out run."""
    traces = os.path.join(scratch, "tld12/traces/2mm")
    tx, ty = splitter_truth(glob.glob(f"{traces}/Buffered_2mm-MEDIUM_1_*_timeline.csv")[0])
    curves = pd.read_csv(os.path.join(scratch, "tld5/data/timeline_curves.csv"))
    curves = curves[(curves["workload"] == "2mm") & (curves["config"] == "MEDIUM") & (curves["source"] == "sampler")]
    end = pd.read_csv(glob.glob(f"{traces}/Buffered_2mm-MEDIUM_1_*_timeline.csv")[0])["Time"].max()
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    truth(ax, tx, ty / MB)
    picks = [("NZ", "base", r"$\alpha$ model, 1/25 references"), ("iChao1", "raw", "iChao1, 1/25 references"),
             ("Hybrid", "i=25 snapshot smooth", "Hybrid, 1/25 references")]
    for slot, (subset, variant, label) in enumerate(picks):
        c = curves[(curves["subset"] == subset) & (curves["variant"] == variant)].sort_values("Time")
        series(ax, c["Time"] / end, c["Estimate"] / MB, slot, label)
    s = pd.read_csv(glob.glob(f"{traces}/Spatial_2mm-MEDIUM_100_*_timeline.csv")[0])
    s = s[s["Bin"] == -1]
    series(ax, s["Time"] / s["Time"].max(), s["MemUsageObs"] * 100 / MB, 3, "Spatial, 1/100 addresses")
    ax.set_xlabel("Fraction of the run (memory references)")
    ax.set_ylabel("Live footprint (MB)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    fig.savefig(os.path.join(out, "reference_vs_spatial.pdf"))
    plt.close(fig)


def fig_spatial_rate(scratch, out):
    """Spatial sampling error against the sampling rate."""
    pb = pd.read_csv(os.path.join(scratch, "tld12/data/spatial_eval.csv"))
    mv = pd.read_csv(os.path.join(scratch, "tld11/spatial_eval.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.1), sharey=True)
    for slot, (split, label) in enumerate([("EXTRA", "PolyBench, largest size"), ("INTER", "PolyBench, middle size")]):
        g = pb[pb["split"] == split].groupby("sampler_si")
        for ax, metric in zip(axes, ["sampler_mape", "sampler_peak_error"]):
            y = g[metric].apply(lambda v: np.mean(np.abs(v)))
            series(ax, y.index, y.values, slot, label, marker="o", markersize=3.5)
    for k, threads in enumerate([1, 4, 16]):
        g = mv[mv["threads"] == threads].groupby("sampler_si")
        for ax, metric in zip(axes, ["sampler_mape", "sampler_peak_error"]):
            y = g[metric].apply(lambda v: np.mean(np.abs(v)))
            series(ax, y.index, y.values, 2 + k, f"miniVite, {threads} thread{'s' if threads > 1 else ''}",
                   marker="o", markersize=3.5)
    for ax, title in zip(axes, ["Mean error over the run (%)", "Error of the peak (%)"]):
        ax.set_xscale("log")
        ax.set_xticks([25, 100, 250, 1000], ["1/25", "1/100", "1/250", "1/1000"])
        ax.minorticks_off()
        ax.set_xlabel("Addresses selected")
        ax.set_title(title, loc="left")
        ax.set_ylim(bottom=0)
    axes[1].legend(loc="upper left")
    fig.savefig(os.path.join(out, "spatial_rate.pdf"))
    plt.close(fig)


def windowed_run(directory, rate, watched):
    for f in glob.glob(f"{directory}/Spatial_*_{rate}_*_windowed.csv"):
        w = pd.read_csv(f)
        if abs(w["Window"].iloc[0] / w["Period"].iloc[0] - watched) < 1e-6:
            return f, w
    raise FileNotFoundError(f"{directory}: no 1/{rate} run watching {watched}")


def fig_windowed_curves(scratch, out):
    """Truth and windowed estimates over time, 5% watched, 1/100 chunks."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    for ax, (name, title) in zip(axes, [("2mm-LARGE", "2mm LARGE"), ("miniVite1t-65536", "miniVite 65536, 1 thread")]):
        d = os.path.join(scratch, EVAL, name)
        tx, ty = splitter_truth(glob.glob(f"{d}/Buffered_*_timeline.csv")[0])
        f, w = windowed_run(d, 100, 0.05)
        x = w["Time"] / w["Time"].max()
        own = pd.read_csv(f.replace("_windowed.csv", "_timeline.csv"))
        own = own[own["Bin"] == -1].set_index("Time")["MemUsageObs"] * 100
        truth(ax, tx, ty / MB)
        series(ax, x, w["Estimate"] / MB, 0, "Windowed estimate")
        series(ax, x, (w["FreshResident"] + w["ReusedResident"] + w["OtherEst"]) / MB, 1, "Resident pages, no density")
        series(ax, x, own.reindex(w["Time"]).to_numpy() / MB, 2, "Windows' sample alone")
        series(ax, x, w["AllocatedBytes"] / MB, 3, "Live allocated bytes")
        # windows (each 0.25% of the run) as ticks along the top edge
        starts, _ = window_spans(w)
        ax.plot(starts, [1.0] * len(starts), linestyle="none", marker="|", markersize=5, color=MUTED,
                transform=ax.get_xaxis_transform(), clip_on=False)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Fraction of the run (memory references)")
    axes[0].set_ylabel("Live footprint (MB)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=5)
    fig.savefig(os.path.join(out, "windowed_curves.pdf"))
    plt.close(fig)


def window_spans(w):
    """Start and end (fraction of the run) of each window, from the Watching column."""
    x = (w["Time"] / w["Time"].max()).to_numpy()
    on = w["Watching"].to_numpy().astype(bool)
    edges = np.flatnonzero(np.diff(on.astype(int)))
    starts = [x[i + 1] for i in edges if not on[i]]
    ends = [x[i + 1] for i in edges if on[i]]
    if on[0]:
        starts = [0.0] + starts
    return starts[: len(ends)], ends


def fig_windowed_fraction(scratch, out):
    """miniVite: error against the watched fraction."""
    r = pd.read_csv(os.path.join(scratch, EVAL + "/results.csv"))
    r = r[(r["sample_rate"] == 100) & r["workload"].str.startswith("miniVite")]
    runs = [("miniVite1t", "65536", "65536, 1 thread"), ("miniVite", "65536", "65536, 4 threads"),
            ("miniVite1t", "32768", "32768, 1 thread"), ("miniVite", "32768", "32768, 4 threads")]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.1))
    xs = [0, 1, 2, 3]
    for slot, (wl, config, label) in enumerate(runs):
        for estimate, style in [("windowed", {}), ("no density", {"alpha": 0.45})]:
            g = r[(r["workload"] == wl) & (r["config"].astype(str) == config) & (r["estimate"] == estimate)]
            g = g.sort_values("watched")
            for ax, metric in zip(axes, ["mape", "peak_error"]):
                series(ax, xs, g[metric].to_numpy(), slot, label if estimate == "windowed" else None,
                       marker="o" if estimate == "windowed" else "x", markersize=3.5, **style)
    for ax, title in zip(axes, ["Mean error over the run (%)", "Error of the peak (%)"]):
        ax.set_xticks(xs, ["none", "1%", "5%", "20%"])
        ax.set_xlabel("Fraction of the run watched")
        ax.set_title(title, loc="left")
    axes[0].set_ylim(bottom=0)
    axes[1].axhline(0, color=MUTED, linewidth=0.6)
    axes[0].plot([], [], color=MUTED, marker="x", linestyle="none", alpha=0.6, label="faded: without density")
    axes[0].legend(loc="upper right")
    fig.savefig(os.path.join(out, "windowed_fraction.pdf"))
    plt.close(fig)


def fig_cost(scratch, out):
    """Runtime relative to native against the watched fraction, with full spatial sampling."""
    rows = [l.split() for l in open(os.path.join(scratch, TIMING)) if l.strip() and "ALL DONE" not in l]
    t = pd.DataFrame(rows, columns=["workload", "mode", "seconds"]).astype({"seconds": float})
    t = t.groupby(["workload", "mode"])["seconds"].last().unstack()  # reruns replace earlier runs
    names = [("2mm-LARGE", "2mm LARGE"), ("gemm-LARGE", "gemm LARGE"), ("jacobi-2d-LARGE", "jacobi-2d LARGE"),
             ("miniVite1t-65536", "miniVite 65536, 1 thread"), ("miniVite-65536", "miniVite 65536, 4 threads")]
    modes = ["windowed-0", "windowed-0.01", "windowed-0.05", "windowed-0.2", "spatial-full"]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for slot, (wl, label) in enumerate(names):
        y = (t.loc[wl, modes] / t.loc[wl, "native"]).to_numpy()
        series(ax, range(len(modes)), y, slot, label, marker="o", markersize=3.5)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_xticks(range(len(modes)), ["none", "1%", "5%", "20%", "full\nspatial"])
    ax.set_xlabel("Fraction of the run watched")
    ax.set_ylabel("Runtime / native")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(os.path.join(out, "cost.pdf"))
    plt.close(fig)


def fig_churn(scratch, out):
    """Heap churn: blocks on inherited resident pages, a quarter of each touched."""
    d = os.path.join(scratch, EVAL, "churn-0.25")
    tx, ty = splitter_truth(glob.glob(f"{d}/Buffered_*_timeline.csv")[0])
    _, w = windowed_run(d, 100, 0.05)
    x = w["Time"] / w["Time"].max()
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    truth(ax, tx, ty / MB)
    series(ax, x, w["Estimate"] / MB, 0, "Windowed (written pages)")
    series(ax, x, (w["FreshResident"] + w["ReusedResident"] + w["OtherEst"]) / MB, 1, "Resident pages")
    series(ax, x, w["AllocatedBytes"] / MB, 3, "Live allocated bytes")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Fraction of the run (memory references)")
    ax.set_ylabel("Live footprint (MB)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    fig.savefig(os.path.join(out, "churn.pdf"))
    plt.close(fig)


def main():
    global EVAL, TIMING
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "figures"))
    parser.add_argument("--eval", default=EVAL, help="windowed runs, relative to --scratch")
    parser.add_argument("--timing", default=TIMING, help="timings, relative to --scratch")
    args = parser.parse_args()
    EVAL, TIMING = args.eval, args.timing
    os.makedirs(args.out, exist_ok=True)
    for make in (fig_reference_vs_spatial, fig_spatial_rate, fig_windowed_curves, fig_windowed_fraction, fig_cost,
                 fig_churn):
        make(args.scratch, args.out)
        print(make.__name__, "done")


if __name__ == "__main__":
    main()
