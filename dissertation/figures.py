"""Figures for dissertation/results.tex from the scored data in $OUT (dissertation/score.py).

    PYTHONPATH=analysis ~/environments/memfp/bin/python -W ignore dissertation/figures.py  -> dissertation/figures/*.pdf
Redrawn from the published chapter: sd (gemm_sd_vs_config_highlight_MT), alpha (miniVite_config_vs_alpha and
rate_vs_alpha), valgrind (miniVite_baseline_vs_massif, with gemm), overhead (miniVite_PIN_overhead, at the bins'
actual interval), cumulative (cumulative_overhead_vs_accuracy). New: m3 (sampler against splitter bins).
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from memprint.dataset import prepare  # noqa: E402
from memprint.model import Model, mape, mape_by_interval  # noqa: E402

OUT = os.path.expanduser("~/memprint-data/2026-11-05-diss-ch2")
FINAL = f"{OUT}/m1/bytes-m4/data"
FIG = os.path.join(HERE, "figures")
OLDLOG = os.path.expanduser("~/memory_estimator/workloads/miniVite/overhead_sampler_miniVite_SI.csv")
# validated categorical palette (dataviz skill, light mode), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED, INK, INK2 = "#b9b8b0", "#1f1f1e", "#5c5b55"
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3dd",
    "grid.linewidth": 0.6, "lines.linewidth": 1.5, "lines.markersize": 4.5, "legend.frameon": False,
    "pdf.fonttype": 42, "font.family": "serif"})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{FIG}/{name}.pdf")


def table(w):
    return pd.read_csv(f"{FINAL}/{w}_allData.csv")


def sd():
    """SD of MemUsageObs across the 20 splitter bins against input size, gemm; the monotone (MT) intervals solid."""
    p = prepare(table("gemm"))
    d = p.data.groupby(["SamplingInterval", "Config"]).SD_MemUsage.first().unstack()[p.configs]
    d = d[d.index <= 10000]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(p.configs))
    mt = [k for k in d.index if k in p.monotone_scores]
    for k in d.index:
        if k not in mt:
            ax.plot(x, d.loc[k], color=MUTED, lw=1, ls="--", marker="o", ms=2.5, zorder=1)
    for i, k in enumerate(mt):
        ax.plot(x, d.loc[k], color=C[i % len(C)], marker=MARK[i % len(MARK)], label=f"1 in {k:,}", zorder=3)
    ax.plot([], [], color=MUTED, lw=1, ls="--", label="other intervals")
    ax.set_yscale("log")
    ax.set_xticks(x, p.configs)
    ax.set_xlabel("Input (PolyBench dataset)")
    ax.set_ylabel("SD of footprint across bins (bytes)")
    ax.legend(title="Monotone intervals (MT)", title_fontsize=8, loc="upper left", ncol=2)
    save(fig, "sd_gemm")


def alpha():
    """miniVite: mean alpha (bytes truth over bin footprint) against input size and against sampling interval."""
    p = prepare(table("miniVite"))
    d = p.data.groupby(["SamplingInterval", "Config"]).Alpha.mean().unstack()[p.configs]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.8))
    ks = [k for k in (100, 500, 1000, 5000, 10000, 50000) if k in d.index]
    for i, k in enumerate(ks):
        a.plot(range(len(p.configs)), d.loc[k], color=C[i], marker=MARK[i], label=f"1 in {k:,}")
    a.set_xticks(range(len(p.configs)), [f"{c:,}" for c in p.configs])
    a.set_xlabel("Vertices")
    a.set_ylabel(r"Mean $\alpha$ over bins")
    a.set_yscale("log")
    a.set_ylim(1, None)
    a.legend(ncol=3, loc="lower left", fontsize=7, handlelength=1.5, columnspacing=0.8)
    a.set_title("(a) Against input size", loc="left")
    for i, c in enumerate(p.configs):
        b.plot(d.index, d[c], color=C[i], marker=MARK[i], label=f"{c:,}")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("Sampling interval (1 in k)")
    b.set_title("(b) Against sampling interval", loc="left")
    b.legend(title="Vertices", title_fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "alpha_minivite")


def valgrind():
    """Massif and DHAT peaks over the bytes truth, gemm and miniVite, with MemPrint's L2O estimate from the
    bootstrap sampler at the EXTRA and INTER inputs."""
    m = pd.read_csv(f"{OUT}/m6.csv")
    est = memprint_estimates()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=False)
    for ax, w, configs in ((axes[0], "gemm", ["MINI", "MINI2", "MINI3", "SMALL", "SMALL2", "SMALL3", "MEDIUM"]),
                           (axes[1], "miniVite", ["1024", "2048", "4096", "8192", "16384"])):
        g = m[m.workload == w].set_index(m[m.workload == w].config.astype(str)).loc[configs]
        x = np.arange(len(configs))
        ax.axhline(1, color=INK2, lw=0.8)
        ax.plot(x, g.massif_over_truth, color=C[0], marker="o", label="Massif")
        ax.plot(x, g.dhat_over_truth, color=C[1], marker="s", ls="--", label="DHAT")
        e = est[est.workload == w]
        xs = [configs.index(str(c)) for c in e.config]
        ax.plot(xs, e.ratio, color=C[2], marker="^", ls="none", ms=6, label="MemPrint (L2O, sampler)")
        ax.set_xticks(x, [f"{int(c):,}" if c.isdigit() else c for c in configs], rotation=30 if w == "gemm" else 0)
        ax.set_title(f"({'a' if w == 'gemm' else 'b'}) {w}", loc="left")
        ax.set_xlabel("Input" if w == "gemm" else "Vertices")
    axes[0].set_ylabel("Estimate / bytes touched")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "valgrind")


def memprint_estimates():
    """MemPrint's L2O estimate of the footprint at the EXTRA and INTER inputs from the bootstrap runs (seed 1 to 3,
    the mean predicted footprint over bins, then over seeds), at the fit's training-best interval."""
    import glob
    fits = pd.read_csv(f"{FINAL}/fits.csv")
    rows = []
    for w in ("gemm", "miniVite"):
        p = prepare(table(w))
        for split, c in (("EXTRA", p.last_config), ("INTER", p.middle_config)):
            f = fits[(fits.workload == w) & (fits.split == split) & (fits.subset == "L2O")].iloc[0]
            model = Model.from_row(f)
            preds = []
            for seed in (1, 2, 3):
                d = f"{OUT}/m3/boot/{seed}/{w}/{c}/{int(f.train_si)}"
                b = pd.Series([pd.read_csv(x).MemUsageObs.iloc[0] for x in glob.glob(f"{d}/*_bin_*.csv")], dtype=float)
                t = pd.DataFrame({"MemUsageObs": b, "SamplingInterval": f.train_si, "SD_MemUsage": b.std()})
                preds.append((model.predict(t).Predicted_Alpha * b).mean())
            rows.append(dict(workload=w, split=split, config=c, estimate=np.mean(preds), truth=p.truth[c],
                             ratio=np.mean(preds) / p.truth[c]))
    e = pd.DataFrame(rows)
    e.to_csv(f"{OUT}/m6_memprint.csv", index=False)
    return e


def overhead():
    """(a) The published sampler's log (s=50, r=5) at the bins' actual interval i*5, not the plotted label i*10.
    (b) M7b: each sampler at matched bin intervals, miniVite 8192, median of 5."""
    old = pd.read_csv(OLDLOG, skipinitialspace=True)
    old.columns = ["vertices", "native", "i", "pin", "overhead"]
    old = old[old.i < 1e9]
    k = pd.read_csv(f"{OUT}/m7_k.csv").set_index("config")["median"]
    instr = pd.read_csv(f"{OUT}/m7.csv").set_index("config").loc["instr", "median"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.8))
    for i, (v, g) in enumerate(old.groupby("vertices")):
        a.plot(g.i * 5, g.pin, color=C[i], marker=MARK[i], label=f"{v:,}")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_yticks([20, 30, 50, 100, 200, 300], ["20", "30", "50", "100", "200", "300"])
    a.minorticks_off()
    a.set_ylim(18, 1200)
    a.set_xlabel("Bin interval, actual (1 in k)")
    a.set_ylabel("Wall time (s)")
    a.set_title("(a) Published sampler, published runs", loc="left")
    a.legend(title="Vertices", title_fontsize=8, ncol=2, loc="upper right")
    ks = [500, 1000, 2500]
    names = [("published50", "Published, s=50, r=5"), ("published", "Published, s=100, r=10"),
             ("boot", "Bootstrap, s=100, r=10"), ("disj", "Disjoint, 20 bins")]
    for i, (n, label) in enumerate(names):
        b.plot(ks, [k[f"{n}-{x}"] for x in ks], color=C[i], marker=MARK[i], label=label)
    b.axhline(instr, color=INK2, lw=0.8, ls=":")
    b.text(2500, instr - 0.4, "instrumentation only", ha="right", va="top", fontsize=7, color=INK2)
    b.set_xscale("log")
    b.set_xticks(ks, [f"{x:,}" for x in ks])
    b.minorticks_off()
    b.set_ylim(22, 42)
    b.set_xlabel("Bin interval (1 in k)")
    b.set_title("(b) Matched intervals, 8,192 vertices", loc="left")
    b.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "overhead")


def splitter_seconds():
    """Pin wall time of M4's splitter run per miniVite config (scripts/run.sh's log)."""
    t = {}
    for line in open(f"{OUT}/run.log"):
        m = re.search(r"miniVite (\d+): native [\d.]+s, pin ([\d.]+)s", line)
        if m:
            t[int(m.group(1))] = float(m.group(2))
    return t


def cumulative():
    """miniVite, EXTRA (16384 held out): test MAPE at the training-best interval when the training set is the
    configs up to n, the subsets chosen from the training configs only; and the cumulative splitter time."""
    a = table("miniVite")
    full = prepare(a)
    test_cfg = full.last_config
    rows = []
    for n in full.configs[1:-1]:
        train_cfgs = [c for c in full.configs if c <= n]
        pt = prepare(a[a.Config.isin(train_cfgs)])
        test = full.data[(full.data.Config == test_cfg) & full.data.SamplingInterval.isin(pt.intervals)]
        for s in pt.subsets:
            data = pt.subset(s)
            model = Model.fit(data)
            tm = mape_by_interval(model.predict(data))
            si = int(tm.idxmin())
            rows.append(dict(n=n, subset=s, train_si=si, mape=mape(model.predict(test[test.SamplingInterval == si]))))
    r = pd.DataFrame(rows)
    secs = splitter_seconds()
    r["cum_seconds"] = [sum(v for c, v in secs.items() if c <= n) for n in r.n]
    r.to_csv(f"{OUT}/cumulative.csv", index=False)
    print(r.round(2).to_string(index=False))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.6))
    ns = sorted(r.n.unique())
    for i, s in enumerate(("NZ", "MT", "L2O")):
        g = r[r.subset == s].set_index("n").reindex(ns)
        a1.plot(range(len(ns)), g.mape, color=C[i], marker=MARK[i], label=s)
    a1.set_xticks(range(len(ns)), [f"{x:,}" for x in ns])
    a1.set_xlabel("Training inputs up to n vertices")
    a1.set_ylabel("Test MAPE at 16,384 vertices (%)")
    a1.set_title("(a) Accuracy", loc="left")
    a1.legend()
    cs = [sum(v for c, v in secs.items() if c <= x) for x in ns]
    a2.bar(range(len(ns)), cs, color=C[0], width=0.55)
    for i, v in enumerate(cs):
        a2.text(i, v + 3, f"{v:.0f}", ha="center", fontsize=7, color=INK)
    a2.set_xticks(range(len(ns)), [f"{x:,}" for x in ns])
    a2.set_xlabel("Training inputs up to n vertices")
    a2.set_ylabel("Cumulative splitter time (s)")
    a2.set_title("(b) Training cost", loc="left")
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "cumulative")


def m3():
    """Per workload, L2O test MAPE on the bootstrap sampler's bins (mean of 3 seeds) against on the splitter's bins."""
    m = pd.read_csv(f"{OUT}/m3_mape.csv")
    m = m[(m.basis == "train_si") & (m.subset == "L2O")]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    for ax, s, title in ((axes[0], "boot", "(a) Bootstrap sampler"), (axes[1], "disj", "(b) Disjoint sampler")):
        g = m[m.sampler == s].groupby(["workload", "split"])[["sampler_mape", "splitter_mape"]].mean().reset_index()
        for i, sp in enumerate(("EXTRA", "INTER")):
            h = g[g.split == sp]
            ax.scatter(h.splitter_mape, h.sampler_mape, s=16, color=C[i], marker=MARK[i], label=sp, zorder=3,
                       edgecolor="white", linewidth=0.5)
        lim = [0.3, 100]
        ax.plot(lim, lim, color=INK2, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("MAPE on splitter bins (%)")
        ax.set_title(title, loc="left")
        for r in g[(g.sampler_mape > 50) | (g.splitter_mape > 50)].itertuples():
            ax.annotate(r.workload, (r.splitter_mape, r.sampler_mape), fontsize=7, color=INK2,
                        xytext=(3, -8), textcoords="offset points")
    axes[0].set_ylabel("MAPE on sampler bins (%)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "m3")


if __name__ == "__main__":
    for part in sys.argv[1:] or ["sd", "alpha", "valgrind", "overhead", "cumulative", "m3"]:
        globals()[part]()
