"""Overhead of every method: slowdown and peak memory relative to the native run.

Reads the summary written by the overhead experiment (summary.csv: workload, method,
median seconds, slowdown x, peak RSS and its ratio to native) and draws
figures/overhead.pdf.

    python paper/overhead_figure.py --data ~/memprint-data/2026-10-10-overhead/res/summary.csv
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METHODS = [
    ("pin", "Pin, no tool"),
    ("instr", "Per-reference\ninstrumentation"),
    ("memprint", "MemPrint\nsplitter"),
    ("truth", "Complete\nrecording"),
    ("samp100", "Reference\nsampling 1/100"),
    ("samp25", "Reference\nsampling 1/25"),
    ("word", "Word\nsampling"),
    ("P", "Windowed,\npages only"),
    ("W1", "Windowed,\n1%"),
    ("W5", "Windowed,\n5%"),
    ("W20", "Windowed,\n20%"),
    ("W5F", "Windowed, 5%,\nflush"),
    ("massif", "Massif"),
    ("dhat", "DHAT"),
]
WORKLOADS = [
    ("2mm", "2mm", "o"),
    ("gemm", "gemm", "s"),
    ("jacobi2d", "jacobi-2d", "^"),
    ("mv1t", "miniVite, 1 thread", "D"),
    ("mv4t", "miniVite, 4 threads", "d"),
    ("lua", "Lua", "v"),
    ("sqlite", "SQLite", "P"),
    ("cc1", "GCC cc1", "X"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="summary.csv of the overhead experiment")
    ap.add_argument("--out", default=str(Path(__file__).parent / "figures" / "overhead.pdf"))
    args = ap.parse_args()
    s = pd.read_csv(args.data).set_index(["wl", "m"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    for ax, col, label in [(axes[0], "x", "Time / native time"), (axes[1], "mem", "Peak memory / native")]:
        for k, (wl, name, marker) in enumerate(WORKLOADS):
            xs, ys = [], []
            for i, (m, _) in enumerate(METHODS):
                if (wl, m) in s.index:
                    xs.append(i + (k - 3.5) * 0.07)
                    ys.append(s.loc[(wl, m), col])
            ax.scatter(xs, ys, marker=marker, s=22, label=name, alpha=0.85)
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.grid(axis="y", which="major", alpha=0.3)
        for b in (6.5, 11.5):
            ax.axvline(b, color="0.7", lw=0.8)
    axes[0].legend(ncol=8, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    axes[1].set_xticks(range(len(METHODS)))
    axes[1].set_xticklabels([n for _, n in METHODS], fontsize=7, rotation=40, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
