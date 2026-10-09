"""Figures comparing MemPrint's footprints with Valgrind (Massif, DHAT) and peak RSS.

    python paper/valgrind_figures.py --scratch DIR [--out paper/figures]

DIR holds the experiment outputs (not in the repository):

    cmp/static_pred.csv        static (source-only) footprint per PolyBench kernel and config, with the
                               splitter's whole-run footprint (start-address definition, data/*_allData.csv)
    vgpb/<kernel>-<config>/    massif.heap, dhat.json, rss.txt for every PolyBench kernel and config
    vg/<workload>/             the same for the timeline workloads (`--pages-as-heap` too)
    win/{eval6,real,interp}/   timeline truth (Buffered_*_timeline.csv) and windowed runs (Spatial_*_windowed.csv)

Writes static_vs_valgrind.pdf, timeline_peaks_vs_valgrind.pdf, timeline_curves_vs_valgrind.pdf and
the tables cmp/valgrind_static.csv and cmp/valgrind_timeline.csv (in --scratch).
"""

import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_figures import INK, MUTED, GRID, SLOTS, MB, truth as truth_line, series  # same style as the report

# identity is never color alone: every measurement also has its own marker
# Massif and DHAT often report the same heap peak: on categorical axes each measure gets its own offset
DODGE = {"estimate": -0.27, "massif": -0.09, "dhat": 0.09, "rss": 0.27}

MEASURES = [  # (column, label, color, marker)
    ("estimate", "MemPrint estimate", SLOTS[0], "o"),
    ("massif", "Massif heap peak", SLOTS[1], "s"),
    ("dhat", "DHAT heap at t-gmax", SLOTS[2], "^"),
    ("rss", "Peak RSS (native)", MUTED, "x"),
]


def massif(path, pages=False):
    """(peak bytes, time series) from a massif.out file; time in the file's unit (instructions)."""
    snaps, cur = [], None
    for line in open(path):
        m = re.match(r"(\w+)=(.*)", line.strip())
        if not m:
            continue
        k, v = m.groups()
        if k == "snapshot":
            cur = {}
            snaps.append(cur)
        elif cur is not None:
            cur[k] = v
    t = np.array([int(s["time"]) for s in snaps], float)
    heap = np.array([int(s["mem_heap_B"]) for s in snaps], float)
    return heap.max(), pd.DataFrame({"time": t, "bytes": heap})


def dhat(path):
    """(bytes at t-gmax, fraction of the run at which t-gmax occurs, total bytes allocated)."""
    j = json.load(open(path))
    return sum(p["gb"] for p in j["pps"]), j["tg"] / j["te"], sum(p["tb"] for p in j["pps"])


def rss(path):
    return int(open(path).read().split()[-1]) * 1024


def valgrind_row(d):
    peak, _ = massif(f"{d}/massif.heap")
    gb, tg, tb = dhat(f"{d}/dhat.json")
    return dict(massif=peak, dhat=gb, dhat_time=tg, dhat_total=tb, rss=rss(f"{d}/rss.txt"))


# ----------------------------------------------------------------------- static


def static_table(scratch):
    s = pd.read_csv(os.path.join(scratch, "cmp/static_pred.csv"))
    rows = [{**r, **valgrind_row(os.path.join(scratch, "vgpb", f"{r['workload']}-{r['config']}"))}
            for r in s.to_dict("records") if os.path.exists(os.path.join(scratch, "vgpb", f"{r['workload']}-{r['config']}", "dhat.json"))]
    return pd.DataFrame(rows).rename(columns={"static": "estimate"})


def fig_static(t, out):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [1.9, 1]})
    m = t[t["split"] == "EXTRA"].sort_values("truth").reset_index(drop=True)
    x = np.arange(len(m))
    ax.axhline(1, color=INK, linewidth=0.8)
    for col, label, color, marker in MEASURES:
        lab = "Static (source only)" if col == "estimate" else label
        ax.plot(x + DODGE[col], m[col] / m["truth"], linestyle="none", marker=marker, markersize=4.5 if marker != "x" else 4,
                color=color, markeredgecolor="white" if marker != "x" else color, markeredgewidth=0.6, label=lab, zorder=3)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 0.7, 1, 1.5, 2, 3], ["0.5", "0.7", "1", "1.5", "2", "3"])
    ax.minorticks_off()
    ax.set_xticks(x, m["workload"], rotation=90, fontsize=6)
    ax.set_xlim(-0.7, len(m) - 0.3)
    ax.set_ylabel("Measure / splitter footprint")
    ax.set_title("Largest size (MEDIUM), kernels by footprint", loc="left")
    ax.grid(axis="x", visible=False)
    lo, hi = min(t[c].min() for c in ["truth", "estimate", "massif", "dhat", "rss"]), max(t[c].max() for c in ["truth", "rss"])
    bx.plot([lo / MB, hi / MB], [lo / MB, hi / MB], color=INK, linewidth=0.8)
    for col, label, color, marker in MEASURES:
        hollow = col == "massif"  # drawn around DHAT's triangles, which usually coincide
        bx.plot(t["truth"] / MB, t[col] / MB, linestyle="none", marker=marker, markersize=5 if hollow else 3.2,
                color=color, markerfacecolor="none" if hollow else color,
                markeredgecolor=color if hollow or marker == "x" else "white", markeredgewidth=0.8 if hollow else 0.4,
                zorder=2 if hollow else 3)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlabel("Splitter footprint (MB)")
    bx.set_ylabel("Measure (MB)")
    bx.set_title("All 7 sizes", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "static_vs_valgrind.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------- timeline

# (valgrind dir, timeline dir, label)
TIMELINE = [("2mm-LARGE", "win/eval6/2mm-LARGE", "2mm LARGE"), ("gemm-LARGE", "win/eval6/gemm-LARGE", "gemm LARGE"),
            ("jacobi-2d-LARGE", "win/eval6/jacobi-2d-LARGE", "jacobi-2d LARGE"),
            ("miniVite1t-65536", "win/eval6/miniVite1t-65536", "miniVite 65536"),
            ("miniVite1t-32768", "win/eval6/miniVite1t-32768", "miniVite 32768"),
            ("gap-bfs", "win/real/gap-bfs", "GAP bfs"), ("gap-pr", "win/real/gap-pr", "GAP pr"),
            ("gap-cc", "win/real/gap-cc", "GAP cc"), ("gap-sssp", "win/real/gap-sssp", "GAP sssp"),
            ("darknet-alexnet", "win/real/darknet-alexnet", "darknet"), ("python", "win/interp/python", "Python"),
            ("lua", "win/interp/lua", "Lua"), ("perl", "win/interp/perl", "Perl"), ("sqlite", "win/interp/sqlite", "SQLite"),
            ("cc1O0", "win/interp/cc1O0", "GCC cc1"), ("churn-0.25", "win/eval6/churn-0.25", "heap churn")]


def splitter_truth(d):
    t = pd.read_csv(glob.glob(f"{d}/Buffered_*_timeline.csv")[0])
    t = t[t["Bin"] == -1]
    return t["Time"].to_numpy(float) / t["Time"].max(), t["MemUsageObs"].to_numpy(float)


def windowed_5(d):
    """The periodic-windows run watching 5% of the run, 1 in 100 words."""
    for f in sorted(glob.glob(f"{d}/Spatial_*_100_*_windowed.csv")):
        w = pd.read_csv(f)
        if w["TriggeredWindows"].iloc[-1] == 0 and abs(w["Window"].iloc[0] / w["Period"].iloc[0] - 0.05) < 1e-3:
            return w["Time"].to_numpy(float) / w["Time"].max(), w["Estimate"].to_numpy(float)
    raise FileNotFoundError(d)


def timeline_table(scratch):
    rows = []
    for vg, tl, label in TIMELINE:
        tx, ty = splitter_truth(os.path.join(scratch, tl))
        _, est = windowed_5(os.path.join(scratch, tl))
        rows.append(dict(workload=label, truth=ty.max(), estimate=est.max(),
                         massif_pages=massif(os.path.join(scratch, "vg", vg, "massif.pages"))[0],
                         **valgrind_row(os.path.join(scratch, "vg", vg))))
    return pd.DataFrame(rows)


def fig_timeline_peaks(t, out):
    m = t.sort_values("truth").reset_index(drop=True)
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.axvline(1, color=INK, linewidth=0.8)
    for col, label, color, marker in MEASURES:
        lab = "MemPrint windowed, 5% watched" if col == "estimate" else label
        ax.plot(m[col] / m["truth"], y - DODGE[col], linestyle="none", marker=marker, markersize=5 if marker != "x" else 4.5,
                color=color, markeredgecolor="white" if marker != "x" else color, markeredgewidth=0.6, label=lab, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5], ["0.1", "0.2", "0.5", "1", "2", "5"])
    ax.minorticks_off()
    ax.set_yticks(y, [f"{w} ({p / MB:.0f} MB)" if p >= 10 * MB else f"{w} ({p / MB:.1f} MB)" for w, p in zip(m["workload"], m["truth"])],
                  fontsize=6.5)
    ax.set_xlabel("Peak / true live peak (bytes touched, full trace)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2)
    fig.savefig(os.path.join(out, "timeline_peaks_vs_valgrind.pdf"))
    plt.close(fig)


CURVES = ["2mm-LARGE", "miniVite1t-65536", "darknet-alexnet", "lua", "sqlite", "churn-0.25"]


def fig_timeline_curves(scratch, out):
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.0))
    for ax, vg in zip(axes.flat, CURVES):
        tl, label = next((t, l) for v, t, l in TIMELINE if v == vg)
        tx, ty = splitter_truth(os.path.join(scratch, tl))
        wx, wy = windowed_5(os.path.join(scratch, tl))
        _, ms = massif(os.path.join(scratch, "vg", vg, "massif.heap"))
        gb, tg, _ = dhat(os.path.join(scratch, "vg", vg, "dhat.json"))
        truth_line(ax, tx, ty / MB, label="Truth: bytes touched (full trace)")
        series(ax, wx, wy / MB, 0, "MemPrint windowed, 5% watched")
        ax.plot(ms["time"] / ms["time"].max(), ms["bytes"] / MB, color=SLOTS[1], linestyle=(0, (5, 2)), drawstyle="steps-post",
                label="Massif heap", zorder=3)
        ax.plot([tg], [gb / MB], linestyle="none", marker="^", markersize=7, color=SLOTS[2], markeredgecolor="white",
                markeredgewidth=0.8, label="DHAT heap at t-gmax", zorder=4)
        ax.set_title(label, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("Fraction of the run")
    for ax in axes[:, 0]:
        ax.set_ylabel("MB")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "timeline_curves_vs_valgrind.pdf"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scratch", required=True)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "figures"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    st = static_table(a.scratch)
    st.to_csv(os.path.join(a.scratch, "cmp", "valgrind_static.csv"), index=False)
    fig_static(st, a.out)
    tt = timeline_table(a.scratch)
    tt.to_csv(os.path.join(a.scratch, "cmp", "valgrind_timeline.csv"), index=False)
    fig_timeline_peaks(tt, a.out)
    fig_timeline_curves(a.scratch, a.out)
    print("done")


if __name__ == "__main__":
    main()
