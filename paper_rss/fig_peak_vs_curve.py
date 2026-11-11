"""The peak held for the whole run against the footprint curve, for the corpus program with the smallest and the
one with the largest mean / peak ratio of a truth curve, each at its largest input, over progress.

    PYTHONPATH=analysis ~/environments/memfp/bin/python paper_rss/fig_peak_vs_curve.py [method] [corpus dir]
        ->  paper_rss/figures/peak_vs_curve.pdf
method: pagesnat (default; sched_pages.py's corpus) or truth (the bytes-in-use curves of 2026-10-13-sched-llm).

Written on the data host as a stand-in for the fig_peak_vs_curve of the dissertation's own copy of
paper_sched/make_figures.py, which is not in this repository.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "paper_sched"))
import make_figures as MF  # noqa: E402  (rcParams, palette)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from memprint.sched import real  # noqa: E402
from memprint.sched.run import largest  # noqa: E402

plt = MF.plt
NAMES = {"python": "Python", "gemm": "gemm", "jacobi-2d": "jacobi-2d", "sqlite": "SQLite", "gap-bfs": "GAP bfs"}


def main(method="pagesnat", corpus=None):
    corpus = corpus or os.path.expanduser("~/memprint-data/2026-11-10-pagesnat" if method == "pagesnat"
                                          else "~/memprint-data/2026-10-13-sched-llm")
    c = largest(pd.read_csv(f"{corpus}/corpus/curves.csv", dtype={"input": str}))
    curves = {}
    for prog, g in c.groupby("prog"):
        wide = real.rebase({m: h.sort_values("k").bytes.to_numpy() for m, h in g.groupby("method")})
        if method in wide and wide[method].max() > 0:
            curves[prog] = wide[method] / wide[method].max()
    ratio = pd.Series({p: v.mean() for p, v in curves.items()}).sort_values()
    picks = [ratio.index[0], ratio.index[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2), sharey=True)
    for ax, p in zip(axes, picks):
        v = curves[p]
        x = np.linspace(0, 100, len(v))
        ax.fill_between(x, v, 1, color=MF.PAL[1], alpha=0.18, lw=0, label="Held, unused")
        ax.plot(x, np.ones_like(v), color=MF.MUTED, ls=(0, (4, 2)), lw=1.2, label="Peak held for the run", zorder=3)
        ax.plot(x, v, color=MF.PAL[0], lw=2, label="Footprint")
        ax.set_title(f"{NAMES.get(p, p)}: mean / peak = {ratio[p]:.2f}", fontsize=9)
        ax.set_xlabel("Progress (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("Footprint / peak")
    axes[1].legend(loc="lower right", fontsize=7)
    out = os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/peak_vs_curve.pdf")
    fig.savefig(f"{out}/peak_vs_curve.png", dpi=110)
    print("smallest", picks[0], round(ratio.iloc[0], 2), "largest", picks[1], round(ratio.iloc[-1], 2))


if __name__ == "__main__":
    main(*sys.argv[1:])
