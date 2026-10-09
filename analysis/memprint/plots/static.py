"""Figures of the leave-one-workload-out evaluation (memprint static lowo)."""

import matplotlib.pyplot as plt
import numpy as np

from .common import save

# Categorical slots 1-3 of the reference palette, in fixed order, plus recessive ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#2b2b2b", "#8a8a85", "#e4e4e0"

FAMILIES = {
    "from source alone": (BLUE, "o", ["static-fp", "static-alpha", "pooled-static"]),
    "borrowed models of known workloads": (ORANGE, "s", ["nn-static", "mix-static", "nn-measured",
                                                             "mix-measured", "nn-ast", "mean"]),
    "references (need the held-out traces)": (AQUA, "D", ["own", "oracle"]),
}
LABELS = {
    "static-fp": "static footprint", "static-alpha": "static alpha", "pooled-static": "pooled + static alpha",
    "nn-static": "nearest by z-hat", "mix-static": "mixture by z-hat", "nn-measured": "nearest by measured z",
    "mix-measured": "mixture by measured z", "nn-ast": "nearest by AST counts", "mean": "uniform mixture",
    "own": "own model", "oracle": "best borrowed (oracle)",
}


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_errors(errors, out_dir):
    """MAPE of each method on each held-out workload (dots) and the median (bar tick), per split."""
    methods = [m for _, (_, _, ms) in FAMILIES.items() for m in ms if m in set(errors["method"])]
    splits = [s for s in ("EXTRA", "INTER") if s in set(errors["split"])]
    fig, axes = plt.subplots(1, len(splits), figsize=(4.2 * len(splits), 0.34 * len(methods) + 1.2), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, split in zip(axes, splits):
        sub = errors[errors["split"] == split]
        for i, method in enumerate(methods):
            color, marker = next((c, mk) for c, mk, ms in FAMILIES.values() if method in ms)
            v = np.maximum(sub.loc[sub["method"] == method, "mape"].to_numpy(), 0.01)
            y = len(methods) - 1 - i
            ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=10, color=color, marker=marker, alpha=0.55,
                       linewidths=0)
            med = np.median(v)
            ax.plot([med, med], [y - 0.32, y + 0.32], color=INK, linewidth=2, solid_capstyle="round")
            ax.text(1.02, y, f"{med:.1f}", transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=INK)
        ax.set_xscale("log")
        ax.set_title(f"{split} ({'largest' if split == 'EXTRA' else 'middle'} input held out)", fontsize=9,
                     color=INK)
        ax.set_xlabel("MAPE of alpha on the unseen workload (%)", fontsize=8, color=INK)
        _style(ax)
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels([LABELS.get(m, m) for m in reversed(methods)], fontsize=8)
    handles = [plt.Line2D([], [], color=c, marker=mk, linestyle="", markersize=5, label=name)
               for name, (c, mk, _) in FAMILIES.items()]
    handles.append(plt.Line2D([], [], color=INK, linewidth=2, label="median (value at right)"))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 0.97, 1))
    save(fig, out_dir / "lowo_errors.pdf")


def plot_validity(validity, out_dir):
    """Spearman rho between descriptor distance and transfer error, per held-out workload."""
    columns = [("rho_static", "z-hat (from source)"), ("rho_measured", "measured z"), ("rho_ast", "AST node counts")]
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    rng = np.random.default_rng(1)
    for i, (col, label) in enumerate(columns):
        v = validity[col].dropna().to_numpy()
        y = len(columns) - 1 - i
        ax.scatter(v, y + rng.uniform(-0.15, 0.15, len(v)), s=10, color=BLUE, alpha=0.55, linewidths=0)
        med = np.median(v)
        ax.plot([med, med], [y - 0.3, y + 0.3], color=INK, linewidth=2, solid_capstyle="round")
        ax.text(1.02, y, f"{med:.2f}", transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=INK)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels([label for _, label in reversed(columns)], fontsize=8)
    ax.set_xlabel("Spearman rho: distance vs error of the borrowed model\n(one dot per held-out workload and split)",
                  fontsize=8, color=INK)
    ax.set_xlim(-1, 1)
    _style(ax)
    fig.tight_layout(rect=(0, 0, 0.95, 1))
    save(fig, out_dir / "lowo_similarity_validity.pdf")


def plot_reuse_curves(descriptors, out_dir, workloads=("gemm", "atax", "floyd-warshall", "jacobi-2d"),
                      split="EXTRA"):
    """Reuse curve log(alpha_k)/log(k) predicted from source (line) and measured (markers)."""
    present = [w for w in workloads if f"{w}|{split}" in descriptors.index]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(2.3 * len(present), 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, w in zip(axes, present):
        row = descriptors.loc[f"{w}|{split}"]
        ks = sorted(int(c.split("_")[-1]) for c in row.index if c.startswith("hat_reuse_"))
        hat = [row[f"hat_reuse_{k}"] for k in ks]
        meas = [row[f"meas_reuse_{k}"] for k in ks]
        ax.plot(ks, hat, color=BLUE, linewidth=2, label="predicted from source")
        ax.scatter(ks, meas, s=16, color=ORANGE, marker="s", zorder=3, label="measured (splitter)",
                   edgecolors="white", linewidths=0.8)
        ax.set_xscale("log")
        ax.set_title(w, fontsize=9, color=INK)
        ax.set_xlabel("sampling interval k", fontsize=8, color=INK)
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    axes[0].set_ylabel("log(alpha) / log(k)", fontsize=8, color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, out_dir / "lowo_reuse_curves.pdf")


def plot_lowo(errors, validity, descriptors, out_dir):
    plot_errors(errors, out_dir)
    plot_validity(validity, out_dir)
    plot_reuse_curves(descriptors, out_dir)
