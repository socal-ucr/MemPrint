"""Figures for paper_gpu/main.tex, from paper_gpu/data (accuracy.py, overhead.py).

    python paper_gpu/make_figures.py [--out paper_gpu/figures]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bench_tables import small_only_running  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"   # as paper/make_figures.py
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.2, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
})

NAMES = {"chain": "chain", "bfs": "bfs", "wl_bfs": "wl\\_bfs", "rodinia_bfs": "rodinia\\_bfs", "cc": "cc",
         "ptrsel": "ptrsel"}
SIZE_LABEL = {"chain": "elements", "bfs": "vertices", "wl_bfs": "vertices", "rodinia_bfs": "vertices",
              "cc": "vertices", "ptrsel": "elements"}


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"))
    plt.close(fig)
    print("wrote", name)


def fig_cc_rounds(out):
    """How many rounds cc takes: plain and traced runs on the GPU, and the interpreter under each schedule."""
    d = pd.read_csv(os.path.join(DATA, "cc_runs.csv"))
    rows = [("GPU, plain runs", d[d.kind == "plain"].rounds, INK),
            ("GPU, under NVBit", d[d.kind == "traced"].rounds, MUTED),
            ("Interpreter, lockstep", d[d.kind == "interp_lockstep"].rounds, PAL[0]),
            ("Interpreter, blocks in order", d[d.kind == "interp_blocks"].rounds, PAL[0]),
            ("Interpreter, blocks reversed", d[d.kind == "interp_reverse"].rounds, PAL[0]),
            ("Interpreter, random block order", d[d.kind == "interp_random"].rounds, PAL[0])]
    fig, ax = plt.subplots(figsize=(4.6, 1.9))
    for y, (label, vals, col) in enumerate(rows):
        counts = vals.value_counts().sort_index()
        size = 10 + 60 * counts.to_numpy() / max(1, counts.max())
        ax.scatter(counts.index, np.full(len(counts), y), s=size, color=col, edgecolor="white", linewidth=0.8,
                   zorder=3)
        for x, c in counts.items():
            if len(vals) > 1:
                ax.annotate(str(c), (x, y), xytext=(0, 6), textcoords="offset points", ha="center",
                            fontsize=6, color=MUTED)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Rounds of label propagation until no label changes")
    ax.set_ylim(len(rows) - 0.5, -0.8)
    ax.grid(axis="y", visible=False)
    save(fig, out, "cc_rounds")


def fig_cc_scores(out):
    """cc against each traced run: lockstep alone against the four schedules merged."""
    d = pd.read_csv(os.path.join(DATA, "cc_runs.csv"))
    d = d[d.kind == "traced"]
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 1.9), sharey=True)
    for ax, (metric, title) in zip(axes, (("must_precision", "Certain dependences\nconfirmed by the run"),
                                          ("must_recall", "Run's dependences\nreported as certain"))):
        for x, (prefix, label, col) in enumerate((("lockstep_", "Lockstep only", MUTED),
                                                  ("", "Four schedules", PAL[0]))):
            v = d[prefix + metric].to_numpy()
            jitter = (np.arange(len(v)) - (len(v) - 1) / 2) * 0.03
            ax.scatter(np.full(len(v), x) + jitter, v, s=14, color=col, edgecolor="white", linewidth=0.6,
                       zorder=3)
        ax.set_xticks([0, 1], ["Lockstep\nonly", "Four\nschedules"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Share, per traced run")
    save(fig, out, "cc_scores")


def fig_overhead(out):
    """Wall-clock time against input size: the GPU run, the run under NVBit, and the interpreter."""
    d = pd.read_csv(os.path.join(DATA, "overhead.csv"))
    d["interp_all_s"] = d[[c for c in d.columns if c.startswith("interp_") and c.endswith("_s")
                           and c != "interp_kernels_s"]].sum(axis=1)
    progs = [p for p in NAMES if p in set(d.program)]
    cols = 3
    rows = -(-len(progs) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.6, 1.85 * rows), squeeze=False)
    series = (("native_s", "GPU run", INK, "o", "-"), ("nvbit_s", "GPU run under NVBit", MUTED, "s", (0, (4, 2))),
              ("interp_lockstep_s", "Interpreter, lockstep", PAL[0], "D", "-"),
              ("interp_kernels_s", "Interpreter, lockstep, kernels only", PAL[0], "D", (0, (1, 1.5))),
              ("interp_all_s", "Interpreter, four orders", PAL[1], "^", "-"),
              ("hybrid_lockstep_s", "Hybrid, lockstep", PAL[2], "v", "-"),
              ("hybrid_all_s", "Hybrid, four orders", PAL[2], "v", (0, (4, 2))))
    series = tuple(x for x in series if x[0] in d.columns)
    for ax, p in zip(axes.flat, progs):
        g = d[d.program == p].sort_values("n")
        for col, label, color, marker, dash in series:
            ax.plot(g.n, g[col], color=color, marker=marker, markersize=3.5, linestyle=dash, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(g.n.to_numpy(), [f"{v:,}" for v in g.n])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(NAMES[p].replace("\\", ""))
        ax.set_xlabel(SIZE_LABEL[p])
    for ax in axes.flat[len(progs):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Seconds")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    save(fig, out, "overhead")


def fig_sim(out):
    """Speedup over issuing in order, per benchmark, on the simulated GPU (simulate_blocks.py), small and large
    inputs."""
    policies = (("buffer", "Same buffer", MUTED), ("buffer-rw", "Same buffer, one writes", "#9a9893"),
                ("hybrid", "Hybrid interpreter", PAL[0]), ("hybrid-block", "Hybrid interpreter, blocks", PAL[1]),
                ("truth-block", "Traced dependences, blocks", INK))
    runs = [(k, os.path.join(DATA, f)) for k, f in (("Small inputs", "sim_blocks.csv"),
                                                    ("Large inputs", "sim_blocks_large.csv")) if
            os.path.exists(os.path.join(DATA, f))]
    fig, axes = plt.subplots(len(runs), 1, figsize=(6.8, 2.8 * len(runs)), squeeze=False)
    for ax, (title, path) in zip(axes[:, 0], runs):
        d = pd.read_csv(path)
        if title == "Small inputs":
            d = small_only_running(d)
        p = d.pivot(index="name", columns="policy", values="speedup")
        p = p[(p[[q for q, _, _ in policies if q in p]].max(axis=1) > 1.005)]   # programs that gain at all
        order = p["truth-block"].sort_values(ascending=False).index
        x = np.arange(len(order))
        w = 0.8 / len(policies)
        for i, (q, label, color) in enumerate(policies):
            if q in p:
                ax.bar(x + (i - (len(policies) - 1) / 2) * w, p.loc[order, q] - 1, w, bottom=1, color=color,
                       label=label, linewidth=0)
        ax.set_xticks(x, [n.split("-", 1)[1] for n in order], rotation=60, ha="right")
        ax.set_ylabel("Speedup")
        ax.set_title(f"{title}: the {len(order)} of {len(d.name.unique())} programs that gain")
        ax.axhline(1, color=MUTED, linewidth=0.6)
        ax.grid(axis="x", visible=False)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(h, lab, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0))
    save(fig, out, "sim_speedup")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_cc_rounds(args.out)
    fig_cc_scores(args.out)
    if os.path.exists(os.path.join(DATA, "overhead.csv")):
        fig_overhead(args.out)
    fig_sim(args.out)


if __name__ == "__main__":
    main()
