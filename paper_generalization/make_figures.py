"""Figures of the generalisation paper, from the tables written by compute.py and the evaluation
outputs in data/.

python paper_generalization/make_figures.py   (writes paper_generalization/figures/*.pdf)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
D = HERE / "data"
FIG = HERE / "figures"
PB = ROOT / "data" / "polybench-bytes" / "data"
GAP = ROOT / "data" / "gap-bytes" / "data"
CSR = ROOT / "data" / "csr-bytes" / "data"
sys.path.insert(0, str(ROOT / "analysis"))

# reference palette (dataviz skill): categorical slots in fixed order, text inks, context gray
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29d"
SEQ = "Blues"
INTERVALS = [100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 100000]
W1, W2 = 3.3, 6.8  # one column, full width (inches)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print(name)


def label_end(ax, x, y, text, color=INK, dx=3, dy=0, **kw):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", fontsize=6.5,
                color=color if color else INK, **kw)


# ---------------------------------------------------------------- theory

def fig_inclusion():
    t = pd.read_csv(D / "inclusion.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3))
    ax = axes[0]
    for i, k in enumerate(sorted(t.k.unique())):
        s = t[t.k == k]
        ax.plot(s.c, s.p, color=C[i])
        ax.plot([k], [1 - np.exp(-1)], "o", color=C[i], ms=3)
    ax.set_xscale("log")
    ax.set_xlabel("references to an address, $c$")
    ax.set_ylabel("$p(c,k)$: chance a bin holds it")
    ax.set_title("(a) Inclusion probability of one address", loc="left")
    ax.text(1.5e3, 0.5, "dots: $c = k$, $p = 1 - 1/e$", fontsize=6.5, color=INK2)
    ax.legend([Line2D([], [], color=C[i]) for i in range(4)], [f"$k$ = {k:,}" for k in sorted(t.k.unique())],
              loc="upper left")

    ax = axes[1]
    from memprint.static.spectrum import Spectrum, moments
    ks = np.array(INTERVALS, float)
    for i, c in enumerate([1, 10, 100, 1000, 10000]):
        spec = Spectrum(np.array([float(c)]), np.array([4.0]), np.array([1e5]), 1e5 * c)
        a = [moments(spec, k).alpha for k in ks]
        ax.plot(ks, a, color=C[i], marker="o", ms=2.5)
        label_end(ax, ks[-1], a[-1], f"c={c:,}", INK2)
    ax.plot(ks, ks, color=GRAY, lw=0.8, ls="--")
    label_end(ax, 4e3, 4e3, "α = k", INK2, dx=-4, dy=8, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(80, 5e5)
    ax.set_xlabel("sampling interval $k$")
    ax.set_ylabel("α = truth / E[bin footprint]")
    ax.set_title("(b) α of spectra where every address has $c$ references", loc="left")
    fig.tight_layout()
    save(fig, "inclusion")


# ---------------------------------------------------------------- PolyBench

HIGHLIGHT = ["gemm", "jacobi-2d", "floyd-warshall"]


def fig_pb_spectra():
    t = pd.read_csv(D / "pb_cumulative_spectra.csv")
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    for w, s in t.groupby("workload"):
        if w not in HIGHLIGHT:
            ax.plot(s.c, s.frac_bytes, color=GRAY, lw=0.6, alpha=0.6)
    for i, w in enumerate(HIGHLIGHT):
        s = t[t.workload == w]
        ax.plot(s.c, s.frac_bytes, color=C[i], lw=1.6, label=w)
    ax.set_xscale("log")
    ax.set_xlabel("references per address, $c$")
    ax.set_ylabel("share of bytes with ≤ $c$ references")
    ax.set_title("Static spectra of the 27 PolyBench kernels (MEDIUM)", loc="left")
    ax.legend(loc="lower right", handles=[Line2D([], [], color=C[i], lw=1.6, label=w) for i, w in
                                          enumerate(HIGHLIGHT)] + [Line2D([], [], color=GRAY, lw=0.6,
                                                                          label="other kernels")])
    save(fig, "pb_spectra")


def fig_pb_alpha_curves():
    m = pd.read_csv(D / "pb_moments.csv")
    b = pd.read_csv(D / "pb_bins.csv")
    kernels = ["2mm", "gemm", "atax", "jacobi-2d", "lu", "cholesky", "nussinov", "floyd-warshall", "trisolv"]
    fig, axes = plt.subplots(3, 3, figsize=(W2, 5.6), sharex=True)
    for ax, w in zip(axes.ravel(), kernels):
        mm = m[(m.workload == w) & (m.config == "MEDIUM")].sort_values("k")
        bb = b[(b.workload == w) & (b.config == "MEDIUM")]
        jitter = np.exp(np.random.default_rng(0).uniform(-0.06, 0.06, len(bb)))
        ax.scatter(bb.k * jitter, bb.alpha, s=2, color=GRAY, alpha=0.5, lw=0, label="bins (Pin)")
        ax.plot(mm.k, mm.alpha_meas, color=C[0], marker="o", ms=2.5, label="measured: truth / mean bin")
        ax.plot(mm.k, mm.alpha_pred, color=C[1], ls="--", label="predicted from source")
        err = np.mean(np.abs(mm.alpha_pred - mm.alpha_meas) / mm.alpha_meas) * 100
        ax.set_title(f"{w}   ({err:.1f}% off)", loc="left")
        ax.set_xscale("log")
        ax.set_yscale("log")
    for ax in axes[-1]:
        ax.set_xlabel("sampling interval $k$")
    for ax in axes[:, 0]:
        ax.set_ylabel("α")
    axes[0, 0].legend(loc="upper left", fontsize=6)
    fig.tight_layout()
    save(fig, "pb_alpha_curves")


def fig_pb_moments_scatter():
    m = pd.read_csv(D / "pb_moments.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.3))
    for ax, (pred, meas, name) in zip(axes, [("m_pred", "m_meas", "mean bin footprint (B)"),
                                             ("sd_pred", "sd_meas", "SD of bin footprint (B)"),
                                             ("u_pred", "u_meas", "mean unique addresses")]):
        ax.scatter(m[meas], m[pred], s=3, color=C[0], alpha=0.35, lw=0)
        lo, hi = np.nanmin(m[[pred, meas]].to_numpy()), np.nanmax(m[[pred, meas]].to_numpy())
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.7, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        r = np.abs(m[pred] - m[meas]) / m[meas] * 100
        ax.set_title(f"{name}\nmedian error {np.nanmedian(r):.1f}%", loc="left")
        ax.set_xlabel("measured (Pin)")
    axes[0].set_ylabel("predicted from source")
    fig.tight_layout()
    save(fig, "pb_moments_scatter")


def fig_pb_alpha_heatmap():
    m = pd.read_csv(D / "pb_moments.csv")
    m["err"] = np.abs(m.alpha_pred - m.alpha_meas) / m.alpha_meas * 100
    t = m.groupby(["workload", "config"]).err.mean().unstack()
    order = ["MINI", "MINI2", "MINI3", "SMALL", "SMALL2", "SMALL3", "MEDIUM"]
    t = t[[c for c in order if c in t.columns]]
    t = t.loc[t.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(W1, 4.6))
    im = ax.imshow(np.log10(np.clip(t.to_numpy(), 0.1, 100)), aspect="auto", cmap=SEQ, vmin=-1, vmax=2)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="center", fontsize=5,
                        color="white" if v > 10 else INK)
    ax.set_xticks(range(t.shape[1]), t.columns, rotation=45, ha="right")
    ax.set_yticks(range(t.shape[0]), t.index, fontsize=6)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, ticks=[-1, 0, 1, 2])
    cb.ax.set_yticklabels(["0.1%", "1%", "10%", "100%"])
    cb.set_label("α MAPE over the 13 intervals")
    ax.set_title("Static α error per kernel and input size", loc="left")
    save(fig, "pb_alpha_heatmap")


METHOD_LABEL = {
    "static-fp": "static footprint (no sampling)", "static-alpha": "static α (spectrum moments)",
    "pooled-static": "pooled regression + static α", "nn-static": "nearest model by ẑ (source)",
    "mix-static": "RBF mixture by ẑ (source)", "nn-measured": "nearest model by measured z",
    "mix-measured": "RBF mixture by measured z", "nn-ast": "nearest model by AST counts",
    "mean": "uniform mixture of models", "own": "own model (needs its traces)",
    "oracle": "best borrowed model (oracle)",
}
METHOD_ORDER = ["static-fp", "static-alpha", "pooled-static", "nn-static", "mix-static", "nn-measured",
                "mix-measured", "nn-ast", "mean", "own", "oracle"]


def fig_lowo_methods():
    e = pd.read_csv(PB / "lowo_errors.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.4), sharey=True)
    fig.subplots_adjust(left=0.25, right=0.95, wspace=0.18, bottom=0.25, top=0.9)
    rng = np.random.default_rng(1)
    for ax, split in zip(axes, ["EXTRA", "INTER"]):
        for i, mth in enumerate(METHOD_ORDER):
            v = e[(e.split == split) & (e.method == mth)].mape.to_numpy()
            y = len(METHOD_ORDER) - 1 - i
            color = C[0] if mth.startswith("static") or mth == "pooled-static" else \
                C[1] if mth.endswith("static") else (C[2] if mth in ("own", "oracle") else GRAY)
            ax.scatter(np.clip(v, 0.01, None), y + rng.uniform(-0.22, 0.22, len(v)), s=5, color=color,
                       alpha=0.7, lw=0)
            med = np.median(v)
            ax.plot([med, med], [y - 0.35, y + 0.35], color=INK, lw=1.4)
            ax.text(1.01, y, f"{med:.1f}", va="center", ha="left", fontsize=6.5, color=INK,
                    transform=ax.get_yaxis_transform())
        ax.set_xscale("log")
        ax.set_xlim(0.01, 1000)
        ax.set_xlabel("α MAPE on the held-out kernel (%)")
        ax.set_title(f"{split}: {'largest' if split == 'EXTRA' else 'middle'} input held out "
                     "(bars: median)", loc="left")
    axes[0].set_yticks(range(len(METHOD_ORDER)), [METHOD_LABEL[m] for m in METHOD_ORDER][::-1])
    axes[0].legend(handles=[Line2D([], [], ls="", marker="o", color=C[0], label="predicted from source"),
                            Line2D([], [], ls="", marker="o", color=C[1], label="borrowed, chosen by source"),
                            Line2D([], [], ls="", marker="o", color=C[2], label="references"),
                            Line2D([], [], ls="", marker="o", color=GRAY, label="other baselines")],
                   loc="upper left", bbox_to_anchor=(-0.5, -0.17), ncol=4)
    save(fig, "lowo_methods")


def fig_pb_static_vs_own():
    e = pd.read_csv(PB / "lowo_errors.csv")
    t = e.pivot_table(index=["workload", "split"], columns="method", values="mape").reset_index()
    fig, ax = plt.subplots(figsize=(W1, 2.9))
    for i, split in enumerate(["EXTRA", "INTER"]):
        s = t[t.split == split]
        ax.scatter(s["own"], s["static-alpha"], s=10, color=C[i], label=split, alpha=0.85, lw=0)
        for _, r in s.iterrows():
            if r["workload"] in ("nussinov", "floyd-warshall") and split == "INTER" or \
                    r["workload"] in ("adi", "seidel-2d") and split == "EXTRA":
                label_end(ax, r["own"], r["static-alpha"], r["workload"], INK2, dx=4)
    ax.plot([0.1, 100], [0.1, 100], color=INK2, lw=0.7, ls="--")
    label_end(ax, 40, 40, "equal", INK2, dx=-2, dy=6, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("own model's α MAPE (%), trained on its traces")
    ax.set_ylabel("static α MAPE (%), from source only")
    ax.legend(loc="upper left")
    ax.set_title("Below the line: the source beats the own model", loc="left")
    save(fig, "pb_static_vs_own")


def fig_pb_footprint():
    e = pd.read_csv(PB / "lowo_errors.csv")
    t = e[e.method == "static-fp"].pivot(index="workload", columns="split", values="mape")
    t = t.loc[t.max(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(W1, 3.9))
    y = np.arange(len(t))
    for i, split in enumerate(["EXTRA", "INTER"]):
        ax.scatter(t[split], y, s=12, color=C[i], label=split, zorder=3)
    ax.hlines(y, t.min(axis=1), t.max(axis=1), color=GRAY, lw=0.8)
    ax.set_yticks(y, t.index, fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("|footprint error| (%), source + runtime baseline")
    ax.legend(loc="lower right")
    ax.set_title("Static footprint of the held-out kernel", loc="left")
    save(fig, "pb_footprint")


def fig_reuse_curves():
    d = pd.read_csv(PB / "lowo_descriptors.csv", index_col=0)
    kernels = ["2mm", "atax", "jacobi-2d", "lu", "nussinov", "trisolv"]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 4.4), sharex=True)
    for col, split in enumerate(["EXTRA", "INTER"]):
        for row, part in enumerate(["reuse", "spread"]):
            ax = axes[row, col]
            for i, w in enumerate(kernels):
                r = d.loc[f"{w}|{split}"]
                ax.plot(INTERVALS, [r[f"meas_{part}_{k}"] for k in INTERVALS], color=C[i], lw=1.2)
                ax.plot(INTERVALS, [r[f"hat_{part}_{k}"] for k in INTERVALS], color=C[i], lw=1.0, ls=":",
                        marker="o", ms=2)
            ax.set_xscale("log")
            ax.set_title(f"{split}: {part}(k)", loc="left")
            if row == 1:
                ax.set_xlabel("sampling interval $k$")
    axes[0, 0].set_ylabel("reuse(k) = log α / log k")
    axes[1, 0].set_ylabel("spread(k) = log(SD / m)")
    axes[0, 0].legend(handles=[Line2D([], [], color=INK2, label="measured z (Pin)"),
                               Line2D([], [], color=INK2, ls=":", marker="o", ms=2, label="ẑ from source")],
                      loc="center right")
    axes[0, 1].legend(handles=[Line2D([], [], color=C[i], label=w) for i, w in enumerate(kernels)],
                      loc="center right", ncol=2)
    fig.tight_layout()
    save(fig, "reuse_curves")


def fig_similarity():
    v = pd.read_csv(PB / "lowo_validity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    rng = np.random.default_rng(2)
    names = [("rho_static", "ẑ from source"), ("rho_measured", "measured z"), ("rho_ast", "AST node counts")]
    for i, (col, name) in enumerate(names):
        for j, split in enumerate(["EXTRA", "INTER"]):
            x = v[v.split == split][col].dropna().to_numpy()
            pos = i * 2.6 + j
            ax.scatter(pos + rng.uniform(-0.25, 0.25, len(x)), x, s=5, color=C[j], alpha=0.7, lw=0)
            ax.plot([pos - 0.35, pos + 0.35], [np.median(x)] * 2, color=INK, lw=1.4)
    ax.set_xticks([0.5, 3.1, 5.7], [n for _, n in names])
    ax.set_ylabel("Spearman ρ(distance, transfer error)")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color=C[j], label=s) for j, s in
                       enumerate(["EXTRA", "INTER"])], loc="lower left")
    ax.set_title("(a) Does descriptor distance rank transfer error?", loc="left")
    ax = axes[1]
    for j, split in enumerate(["EXTRA", "INTER"]):
        x = np.sort(v[v.split == split].zhat_reuse_rmse.to_numpy())
        ax.step(x, np.arange(1, len(x) + 1) / len(x), color=C[j], where="post", label=split)
    ax.set_xscale("log")
    ax.set_xlabel("RMSE of reuse(k), ẑ against z")
    ax.set_ylabel("share of held-out kernels")
    ax.legend(loc="lower right")
    ax.set_title("(b) Accuracy of ẑ", loc="left")
    fig.tight_layout()
    save(fig, "similarity")


def fig_baseline():
    pb, gp = pd.read_csv(D / "pb_baseline.csv"), pd.read_csv(D / "gap_baseline.csv")
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    x = np.log2(pb["count"])
    ax.bar(x - 0.2, pb.bytes / 1024, width=0.4, color=C[0], label="PolyBench (C, -O0)")
    ax.bar(x + 0.2, gp.bytes / 1024, width=0.4, color=C[1], label="GAP (C++, -O3)")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("references per address, $\\log_2 c$")
    ax.set_ylabel("bytes (KB, symlog)")
    ax.legend(loc="upper right")
    ax.set_title("Runtime baseline: memory outside the program text", loc="left")
    ax.text(0.98, 0.55, f"total {pb.bytes.sum() / 1024:.0f} KB / {gp.bytes.sum() / 1024:.0f} KB",
            transform=ax.transAxes, ha="right", fontsize=6.5, color=INK2)
    save(fig, "baseline")


def fig_pb_interp_cost():
    m = pd.read_csv(D / "pb_static_meta.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3))
    ax = axes[0]
    ax.scatter(m.total, m.seconds, s=6, color=C[0], alpha=0.7, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("references charged")
    ax.set_ylabel("interpreter time (s)")
    ax.set_title("(a) Cost of the C interpreter, 27 kernels × 7 sizes", loc="left")
    ax = axes[1]
    t = m[m.config == "MEDIUM"].assign(share=lambda d: d.total / d.pin_references).sort_values("share")
    ax.scatter(t.share, range(len(t)), s=10, color=C[0], zorder=3)
    ax.hlines(range(len(t)), 0, t.share, color=GRID, lw=0.8)
    ax.set_yticks(range(len(t)), t.workload, fontsize=5.5)
    ax.set_xlim(0, 1.05)
    ax.axvline(1, color=INK2, lw=0.6, ls="--")
    ax.set_xlabel("references charged / references Pin saw (MEDIUM)")
    ax.set_title(f"(b) Coverage is 1.0 for all; reference share", loc="left")
    fig.tight_layout()
    save(fig, "pb_interp_cost")


def fig_idioms():
    t = pd.read_csv(PB / "static_idioms.csv", index_col=0)
    t["irregular_loops"] = t.loops_data + t.loops_while
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    pb = t[~t.out_of_distribution]
    out = t[t.out_of_distribution]
    ax.scatter(pb.affine, pb.irregular_loops, s=10, color=C[0], label="PolyBench kernels (gate: in)")
    ax.scatter(out.affine, out.irregular_loops, s=14, color=C[1], marker="s", label="irregular (gate: out)")
    for w, r in out.iterrows():
        label_end(ax, r.affine, r.irregular_loops, w, INK2, dx=4, dy=-6 if w == "gap-bfs" else 2)
    ax.axvline(0.95, color=INK2, lw=0.6, ls="--")
    ax.axhline(0.05, color=INK2, lw=0.6, ls="--")
    ax.set_xlabel("affine share of references (weighted by $10^{depth}$)")
    ax.set_ylabel("share of data-bounded or while loops")
    ax.legend(loc="lower left")
    ax.set_title("Access-idiom gate", loc="left")
    save(fig, "idioms")


def fig_minivite():
    t = pd.read_csv(PB / "lowo_transfer.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(W1, 1.6))
    items = [("own model", t["own"]), ("best PolyBench (oracle)", t["oracle"]),
             (f"nearest by z ({t['nearest']})", t["nn-measured"]), ("median PolyBench", t["median_known_model"])]
    ax.barh(range(len(items)), [v for _, v in items], color=[C[2], C[1], C[1], GRAY], height=0.6)
    for i, (_, v) in enumerate(items):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(items)), [n for n, _ in items])
    ax.invert_yaxis()
    ax.set_xlabel("α MAPE on miniVite, 16384 vertices (%)")
    ax.set_title(f"Borrowing affine models for irregular code (z distance {t['min_distance']:.1f} vs "
                 f"≤ {t['known_nn_distance_max']:.1f})", loc="left", fontsize=7.5)
    save(fig, "minivite")


# ---------------------------------------------------------------- GAP

GK = ["pr", "bfs", "cc", "bc", "tc", "sssp"]


def fig_gap_degrees():
    t = pd.read_csv(D / "gap_degrees.csv")
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    for i, g in enumerate(["uniform", "kron"]):
        s = t[(t.graph == g) & (t.degree > 0)]
        ax.plot(s.degree, s.vertices, "o", ms=2.5, color=C[i], label={"uniform": "uniform (-u 14)",
                                                                      "kron": "Kronecker (-g 14)"}[g])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degree")
    ax.set_ylabel("vertices")
    ax.legend(loc="upper right")
    ax.set_title("Degree distributions of GAP's two generators", loc="left")
    save(fig, "gap_degrees")


def fig_gap_skeleton():
    f = pd.read_csv(GAP / "gap_pilot_footprints.csv")
    f["base"] = f.kernel.str.replace("gap_", "").str.replace("_kron", "").str.replace("_t4", "")
    f["graph"] = np.where(f.kernel.str.contains("kron"), "kron", "uniform")
    f["threads"] = np.where(f.kernel.str.contains("_t4"), 4, 1)
    kernels = ["pr", "bfs", "cc", "bc", "tc", "sssp", "prc"]
    fig, axes = plt.subplots(2, 4, figsize=(W2, 3.6), sharex=True)
    for ax, k in zip(axes.ravel(), kernels):
        for i, g in enumerate(["uniform", "kron"]):
            for th, ls in ((1, "-"), (4, ":")):
                s = f[(f.base == k) & (f.graph == g) & (f.threads == th)].sort_values("scale")
                if len(s):
                    ax.plot(s.scale, s.alpha_mape, color=C[i], ls=ls, marker="o", ms=2)
        ax.set_title({"prc": "pr (converging)"}.get(k, k), loc="left")
        ax.set_ylim(0, None)
        ax.set_xticks([10, 12, 14, 16, 18])
    ax = axes.ravel()[-1]
    ax.axis("off")
    ax.legend(handles=[Line2D([], [], color=C[0], label="uniform graph"),
                       Line2D([], [], color=C[1], label="Kronecker graph"),
                       Line2D([], [], color=INK2, ls=":", label="4 OpenMP threads")], loc="center")
    for a in axes[1]:
        a.set_xlabel("scale (log2 vertices)")
    for a in axes[:, 0]:
        a.set_ylabel("α MAPE (%)")
    fig.suptitle("Hand-written skeleton: α error at every scale", x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "gap_skeleton_scales")


def fig_gap_fp():
    f = pd.read_csv(GAP / "gap_pilot_footprints.csv")
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    for kernel, s in f.groupby("kernel"):
        s = s.sort_values("scale")
        ax.plot(s.scale, s.error, color=GRAY, lw=0.7, marker="o", ms=1.5)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("scale")
    ax.set_ylabel("footprint error (%)")
    ax.set_title(f"Skeleton footprint, all {f.kernel.nunique()} GAP workloads", loc="left")
    save(fig, "gap_footprint")


def fig_gap_methods():
    e = pd.read_csv(GAP / "gap_pilot_errors.csv")
    t = e.pivot_table(index=["kernel", "split"], columns="method", values="mape").reset_index()
    t = t.sort_values(["kernel", "split"])
    methods = [("skeleton-alpha", "skeleton α", C[0]), ("own", "own model", C[2]),
               ("pb-oracle", "best PolyBench (oracle)", C[1]), ("pb-nearest", "PolyBench nearest by ẑ", GRAY)]
    fig, ax = plt.subplots(figsize=(W2, 3.6))
    y = np.arange(len(t))
    for col, name, color in methods:
        if col in t:
            ax.scatter(t[col], y, s=12, color=color, label=name, zorder=3)
    ax.hlines(y, t[[m for m, _, _ in methods if m in t]].min(axis=1),
              t[[m for m, _, _ in methods if m in t]].max(axis=1), color=GRID, lw=0.8)
    ax.set_yticks(y, [f"{k.replace('gap_', '')} {s}" for k, s in zip(t.kernel, t.split)], fontsize=5.5)
    ax.set_xscale("log")
    ax.set_xlabel("α MAPE at the held-out scale (%)  —  EXTRA: 18, INTER: 14")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=4)
    ax.set_title("GAP: predicting from the skeleton against borrowing a model", loc="left")
    save(fig, "gap_methods")


def fig_gap_auto():
    s = pd.read_csv(D / "gap_scores.csv")
    fig, axes = plt.subplots(2, 6, figsize=(W2, 3.4), sharey="row")
    for col, k in enumerate(GK):
        for row, g in enumerate(["uniform", "kron"]):
            ax = axes[row, col]
            for i, src in enumerate(["hand", "auto"]):
                t = s[(s.kernel == k) & (s.graph == g) & (s.source == src)].sort_values("scale")
                ax.plot(t.scale, t.alpha_mape, color=C[i], marker="o", ms=2, lw=1.1)
            ax.set_title(f"{k}, {'uniform' if g == 'uniform' else 'Kron.'}", loc="left", fontsize=7)
            ax.set_xticks([10, 12, 14, 16, 18])
            if row == 1:
                ax.set_xlabel("scale")
    axes[0, 0].set_ylabel("α MAPE (%)")
    axes[1, 0].set_ylabel("α MAPE (%)")
    fig.legend(handles=[Line2D([], [], color=C[0], marker="o", ms=2, label="hand skeleton"),
                        Line2D([], [], color=C[1], marker="o", ms=2, label="C++ interpreter on GAP's source")],
               loc="upper left", ncol=2, bbox_to_anchor=(0.06, 1.06))
    fig.tight_layout()
    save(fig, "gap_auto_vs_hand")


def fig_gap_auto_fp():
    s = pd.read_csv(D / "gap_scores.csv")
    s = s[s.source == "auto"]
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    for i, k in enumerate(GK):
        for g, mk in (("uniform", "o"), ("kron", "s")):
            t = s[(s.kernel == k) & (s.graph == g)].sort_values("scale")
            ax.plot(t.scale, t.fp_error, color=C[i], marker=mk, ms=2.5, lw=0.8,
                    label=k if g == "uniform" else None)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("scale")
    ax.set_ylabel("footprint error (%)")
    ax.legend(ncol=3, loc="lower right")
    ax.set_title("C++ interpreter footprint (○ uniform, □ Kronecker)", loc="left")
    save(fig, "gap_auto_footprint")


def fig_gap_alpha_curves():
    c = pd.read_csv(D / "gap_curves.csv")
    cases = [("pr", "uniform", 12), ("bfs", "kron", 12), ("cc", "uniform", 12), ("bc", "uniform", 12),
             ("tc", "uniform", 12), ("sssp", "uniform", 12), ("pr", "uniform", 16), ("bfs", "uniform", 16)]
    fig, axes = plt.subplots(2, 4, figsize=(W2, 3.6), sharex=True)
    for ax, (k, g, sc) in zip(axes.ravel(), cases):
        t = c[(c.kernel == k) & (c.graph == g) & (c.scale == sc)]
        meas = t[t.source == "hand"].sort_values("k")
        ax.plot(meas.k, meas.alpha_meas, color=INK, marker="o", ms=2.5, lw=1.2)
        for i, src in enumerate(["hand", "auto"]):
            s = t[t.source == src].sort_values("k")
            if len(s):
                ax.plot(s.k, s.alpha_pred, color=C[i], ls="--", lw=1.1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{k}, {'Kron.' if g == 'kron' else 'uniform'}, scale {sc}", loc="left", fontsize=7)
    for a in axes[1]:
        a.set_xlabel("$k$")
    for a in axes[:, 0]:
        a.set_ylabel("α")
    axes[0, 0].legend(handles=[Line2D([], [], color=INK, marker="o", ms=2.5, label="Pin"),
                               Line2D([], [], color=C[0], ls="--", label="hand skeleton"),
                               Line2D([], [], color=C[1], ls="--", label="interpreter")], loc="upper left",
                      fontsize=6)
    fig.tight_layout()
    save(fig, "gap_alpha_curves")


def fig_gap_spectra():
    t = pd.read_csv(D / "gap_cumulative_spectra.csv")
    fig, axes = plt.subplots(1, 6, figsize=(W2, 1.9), sharey=True)
    for ax, k in zip(axes, GK):
        for i, src in enumerate(["hand", "auto"]):
            s = t[(t.kernel == k) & (t.graph == "uniform") & (t.source == src)]
            ax.plot(s.c, s.frac_bytes, color=C[i], lw=1.2, ls="-" if src == "hand" else "--")
        ax.set_xscale("log")
        ax.set_title(k, loc="left")
        ax.set_xlabel("$c$")
    axes[0].set_ylabel("share of bytes ≤ $c$")
    axes[0].legend(handles=[Line2D([], [], color=C[0], label="hand"),
                            Line2D([], [], color=C[1], ls="--", label="interpreter")], loc="lower right",
                   fontsize=6)
    fig.tight_layout()
    save(fig, "gap_spectra")


def fig_gap_arrays():
    t = pd.read_csv(D / "gap_arrays.csv")
    t = t[(t.hand_refs > 0) | (t.auto_refs > 0)]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.2), sharey=True)
    for ax, (k, s) in zip(axes, t.groupby("kernel", sort=False)):
        ratio = ((s.auto_refs + 1) / (s.hand_refs + 1)).to_numpy()
        colors = [C[0] if abs(np.log(r)) < np.log(1.05) else C[1] for r in ratio]
        ax.vlines(range(len(s)), 1, ratio, color=colors, lw=1.2)
        ax.scatter(range(len(s)), ratio, color=colors, s=10, zorder=3)
        ax.axhline(1, color=INK2, lw=0.6)
        ax.set_yscale("log")
        ax.set_xticks(range(len(s)), [f"{int(b / 1024)}K" if b >= 1024 else str(int(b)) for b in s.hand_bytes],
                      rotation=90, fontsize=5)
        ax.set_title(f"{k} (scale 10)", loc="left")
    axes[0].set_ylabel("references: interpreter / hand")
    fig.supxlabel("heap blocks in allocation order (labelled by size); orange: more than 5% apart", fontsize=7)
    fig.tight_layout()
    save(fig, "gap_arrays")


def fig_interp_cost():
    t = pd.read_csv(D / "interp_cost.csv")
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    for i, k in enumerate(GK):
        for g, mk, ls in (("uniform", "o", "-"), ("kron", "s", "--")):
            s_ = t[(t.kernel == k) & (t.graph == g)].sort_values("scale")
            ax.plot(s_.scale, s_.seconds, color=C[i], marker=mk, ms=2.2, lw=0.9, ls=ls,
                    label=k if g == "uniform" else None)
    ax.set_yscale("log")
    ax.set_xlabel("scale")
    ax.set_ylabel("interpreter time (s), 6 runs in parallel")
    ax.legend(ncol=3, loc="upper left", fontsize=6)
    ax.set_title("C++ interpreter cost (solid uniform, dashed Kronecker)", loc="left")
    save(fig, "interp_cost")


def fig_blind():
    t = pd.read_csv(D / "blind_curves.csv")
    sc = pd.read_csv(HERE / "blind" / "scores.csv")
    ph = pd.read_csv(HERE / "blind" / "posthoc_scores.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.4), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, prog, configs in ((axes[0], "hpccg", (10, 16, 28)), (axes[1], "lulesh", (5, 12, 20))):
        for i, c in enumerate(configs):
            s_ = t[(t.program == prog) & (t.config == c)].sort_values("k")
            ax.plot(s_.k, s_.alpha_meas, color=C[i], marker="o", ms=2.5, lw=1.1, label=f"{c}")
            ax.plot(s_.k, s_.alpha_pred, color=C[i], ls="--", lw=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$k$")
        ax.set_title(f"{'HPCCG (grid n)' if prog == 'hpccg' else 'LULESH (mesh s)'}", loc="left")
        ax.legend(title="solid Pin, dashed predicted", fontsize=6, title_fontsize=6, loc="upper left")
    axes[0].set_ylabel("α")
    ax = axes[2]
    for i, (prog, d, name) in enumerate((("hpccg", sc, "HPCCG, blind"), ("lulesh", sc, "LULESH, blind"),
                                         ("lulesh", ph, "LULESH, post hoc"))):
        s_ = d[d.program == prog].sort_values("config")
        x = np.arange(len(s_))
        ax.plot(x, s_.alpha_mape, color=C[i], marker="o", ms=3, lw=1.1, ls="-" if "blind" in name else ":",
                label=name)
    ax.set_xticks(range(7), ["1", "2", "3", "4", "5", "6", "7"])
    ax.set_xlabel("config (smallest to largest)")
    ax.set_ylabel("α MAPE (%)")
    ax.legend(fontsize=6)
    ax.set_title("Error per config", loc="left")
    fig.tight_layout()
    save(fig, "blind")


def fig_csr():
    blind = pd.read_csv(CSR / "c_footprints_blind.csv").assign(run="blind")
    post = pd.read_csv(CSR / "c_footprints.csv").assign(run="with libc rand cost (post hoc)")
    t = pd.concat([blind, post])
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.1), sharey=True)
    for ax, k in zip(axes, ["csr_pr", "csr_bfs"]):
        for i, run in enumerate(["blind", "with libc rand cost (post hoc)"]):
            s = t[(t.kernel == k) & (t.run == run)].sort_values("scale")
            ax.plot(s.scale, s.alpha_mape, color=C[i], marker="o", ms=2.5, label=run)
        ax.set_title(k, loc="left")
        ax.set_xlabel("scale")
    axes[0].set_ylabel("α MAPE (%)")
    axes[0].legend()
    fig.tight_layout()
    save(fig, "csr_programs")


def fig_pb_sd_relative():
    """Where the spread prediction misses: relative SD error against k."""
    m = pd.read_csv(D / "pb_moments.csv")
    m["rel"] = (m.sd_pred - m.sd_meas) / m.sd_meas * 100
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    q = m.groupby("k").rel.quantile([0.1, 0.5, 0.9]).unstack()
    ax.fill_between(q.index, q[0.1], q[0.9], color=C[0], alpha=0.2, lw=0, label="10th–90th percentile")
    ax.plot(q.index, q[0.5], color=C[0], marker="o", ms=2.5, label="median")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("sampling interval $k$")
    ax.set_ylabel("SD error (%)")
    ax.legend(loc="lower left")
    ax.set_title("Bin-to-bin spread: binomial model against Pin", loc="left")
    save(fig, "pb_sd_error")


if __name__ == "__main__":
    for f in [fig_inclusion, fig_pb_spectra, fig_pb_alpha_curves, fig_pb_moments_scatter, fig_pb_alpha_heatmap,
              fig_lowo_methods, fig_pb_static_vs_own, fig_pb_footprint, fig_reuse_curves, fig_similarity,
              fig_baseline, fig_pb_interp_cost, fig_idioms, fig_minivite, fig_gap_degrees, fig_gap_skeleton,
              fig_gap_fp, fig_gap_methods, fig_gap_auto, fig_gap_auto_fp, fig_gap_alpha_curves, fig_gap_spectra,
              fig_gap_arrays, fig_interp_cost, fig_csr, fig_pb_sd_relative, fig_blind]:
        f()
