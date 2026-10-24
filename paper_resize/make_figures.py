"""Figures and tables for paper_resize/main.tex: resizing allocations with MemPrint's per-site footprints.

    ~/environments/memfp/bin/python paper_resize/make_figures.py [--data DIR] [--out paper_resize]

DIR is the study's data directory (not in the repository), ~/memprint-data/2026-10-12-resize:

    corpus-sites/<prog>-<input>/, sites/<name>/   Pin -sites profiles (*_blocks.csv)
    runs/<case>/summary.txt                        label, peak RSS, peak commit, seconds, output hash, status
    runs/ml/<input>/summary.txt, *.csv             the LLM runs (scripts/resize_eval.sh, rss_sampler --commit)
    runs/minivite-pass/summary.txt, *.csv          miniVite built with the pass, without and with a capacity
    runs/corpus/summary.csv                        23 programs under glibc and libmpr (summarize_corpus.py)
    sched-pages/margins.csv, results/grid_*.csv    scheduling with the clock and page options, and rssmpr
    results/llamacpp_buffers.csv                   per-buffer RSS of llama.cpp at its peak (pagemap)
    results/minivite_breakdown.csv                 miniVite's RSS and commit by mapping (smaps)
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GLIBC, LIBMPR, HELD = MUTED, PAL[0], PAL[1]   # glibc grey, libmpr blue everywhere, held-out orange

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.2, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
})
MB = 1e6


def n0(v):
    """A whole number for LaTeX, thousands separated."""
    return f"{v:,.0f}".replace(",", "{,}")


def save(fig, out, name):
    fig.savefig(os.path.join(out, "figures", name + ".pdf"))
    plt.close(fig)
    print("wrote", name)


def summary(path):
    """runs/<case>/summary.txt -> one row per configuration: mean RSS, commit (MB), seconds, outputs identical."""
    s = pd.read_csv(path, sep=r"\s+", header=None, names=["label", "rss", "commit", "secs", "hash", "status"][:6],
                    engine="python")
    s["config"] = s.label.str.replace(r"-\d+$", "", regex=True)
    base = set(s[s.config.isin(["base", "none"])].hash)
    g = s.groupby("config")
    return pd.DataFrame(dict(rss=g.rss.mean() / MB, commit=g.commit.mean() / MB, secs=g.secs.mean(),
                             same=g.hash.apply(lambda h: set(h) <= base if base else True), n=g.size()))


def peaks(files):
    """Mean peak RSS and commit (MB) of rss_sampler CSVs."""
    r = [pd.read_csv(f) for f in files]
    return np.mean([x.rss_bytes.max() for x in r]) / MB, np.mean([x.commit_bytes.max() for x in r]) / MB


def blocks(d):
    return pd.read_csv(glob.glob(f"{d}/*_blocks.csv")[0])


# ---------------------------------------------------------------- results tables

def corpus_row(R, prog):
    """A corpus program's glibc, own-profile and held-out runs (runs/corpus/<prog>/summary.txt)."""
    s = summary(f"{R}/corpus/{prog}/summary.txt")
    return s.loc["base"], s.loc["self"], s.loc["heldout"]


def headline(data):
    """Per workload: glibc, libmpr (own profile), libmpr (held-out profile) peak RSS and commit."""
    R = data + "/runs"
    rows = []
    rows.append(("darknet AlexNet",) + corpus_row(R, "darknet"))
    m = summary(f"{R}/minivite-32768/summary.txt")
    p = summary(f"{R}/minivite-v4/summary.txt")
    rows.append(("miniVite 32768 (pass)", m.loc["base"], p.loc["self"], p.loc["heldout"]))
    names = {"llamacpp-135m-n32": "llama.cpp 135M, 32 tok", "llamacpp-135m-n128": "llama.cpp 135M, 128 tok",
             "llamacpp-360m-n64": "llama.cpp 360M, 64 tok", "llama2c-15M": "llama2.c 15M",
             "llama2c-42M": "llama2.c 42M", "llama2c-110M": "llama2.c 110M"}
    for k, v in names.items():
        s = summary(f"{R}/ml/{k}/summary.txt")
        rows.append((v, s.loc["base"], s.loc["self"], s.loc["heldout"]))
    return rows


def table_headline(rows, out):
    lines = [r"\begin{tabular}{lrrrrrrc}", r"\toprule",
             r" & \multicolumn{2}{c}{glibc} & \multicolumn{2}{c}{libmpr, own profile} & \multicolumn{2}{c}{libmpr, held out} & Output \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Workload & RSS & Commit & RSS & Commit & RSS & Commit & identical \\", r"\midrule"]
    for name, b, s, h in rows:
        ok = "yes" if s.same and h.same else "no"
        lines.append(f"{name} & {b.rss:.1f} & {b.commit:.1f} & {s.rss:.1f} & {s.commit:.1f} & {h.rss:.1f} & "
                     f"{h.commit:.1f} & {ok} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/headline.tex", "w").write("\n".join(lines) + "\n")


def fig_headline(rows, out):
    names = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    for ax, key, title in ((axes[0], "rss", "Peak RSS"), (axes[1], "commit", "Peak commit")):
        b = np.array([getattr(r[1], key) for r in rows])
        s = np.array([getattr(r[2], key) for r in rows])
        h = np.array([getattr(r[3], key) for r in rows])
        ax.barh(y + 0.27, np.ones_like(b), height=0.25, color=GLIBC, label="glibc")
        ax.barh(y, s / b, height=0.25, color=LIBMPR, label="libmpr, own profile")
        ax.barh(y - 0.27, h / b, height=0.25, color=HELD, label="libmpr, held-out profile")
        for yi, v in zip(y, s / b):
            ax.text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=6, color=INK)
        ax.axvline(1, color=INK, linewidth=0.6)
        ax.set_xlim(0, 1.25)
        ax.set_xlabel(f"{title}, relative to glibc")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.55, 1.06))
    save(fig, out, "headline")


IMPROVED = [("llama.cpp 135M, 32 tok", "llamacpp-135m-n32"), ("llama.cpp 360M, 64 tok", "llamacpp-360m-n64"),
            ("llama2.c 42M", "llama2c-42M"), ("llama2.c 110M", "llama2c-110M")]


def fig_improvements(data, out):
    """Commit and RSS under glibc, the first libmpr (runs/ml-v2) and the improved one (runs/ml), own profiles."""
    R = data + "/runs"
    rows = []
    for name, d in IMPROVED:
        old, new = summary(f"{R}/ml-v2/{d}/summary.txt"), summary(f"{R}/ml/{d}/summary.txt")
        rows.append((name, new.loc["base"], old.loc["self"], new.loc["self"]))
    m = summary(f"{R}/minivite-v4/summary.txt")
    p2 = summary(f"{R}/minivite-pass/summary.txt") if os.path.exists(f"{R}/minivite-pass/summary.txt") else None
    rows.append(("miniVite 32768 (pass)", m.loc["none"], p2.loc["capacity"] if p2 is not None else m.loc["none"], m.loc["self"]))
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), sharey=True)
    for ax, key, title in ((axes[0], "commit", "Peak commit (MB)"), (axes[1], "rss", "Peak RSS (MB)")):
        for k, (col, lab) in enumerate(((GLIBC, "glibc (pass build: no reservations)"), (PAL[3], "libmpr, first version"),
                                         (LIBMPR, "libmpr, with the refinements"))):
            v = np.array([getattr(r[1 + k], key) for r in rows])
            ax.barh(y + 0.27 - 0.27 * k, v, height=0.25, color=col, label=lab)
            if k == 2:
                for yi, vi in zip(y, v):
                    ax.text(vi + 4, yi - 0.27, f"{vi:.0f}", va="center", fontsize=6)
        ax.set_xlabel(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.55, 1.08))
    save(fig, out, "improvements")
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{Peak commit (MB)} & \multicolumn{3}{c}{Peak RSS (MB)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Workload & glibc & first & refined & glibc & first & refined \\", r"\midrule"]
    for name, b, o, n in rows:
        lines.append(f"{name} & {b.commit:.1f} & {o.commit:.1f} & {n.commit:.1f} & {b.rss:.1f} & {o.rss:.1f} & {n.rss:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/improvements.tex", "w").write("\n".join(lines) + "\n")


def table_pass(data, out):
    """llama.cpp preloaded (GCC build, return-address sites) against built with the pass (pass ids)."""
    R = data + "/runs"
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{Preloaded (GCC build)} & \multicolumn{3}{c}{Built with \pass{} (clang)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Input & none & own & held out & none & own & held out \\", r"\midrule"]
    for name, tag in (("135M, 32 tokens", "135m-n32"), ("135M, 128 tokens", "135m-n128"), ("360M, 64 tokens", "360m-n64")):
        a = summary(f"{R}/ml/llamacpp-{tag}/summary.txt")
        b = summary(f"{R}/ml-pass/lcpass-{tag}/summary.txt")
        for key, unit in (("commit", "commit"), ("rss", "RSS")):
            lines.append(f"{name}, {unit} & {getattr(a.loc['base'], key):.1f} & {getattr(a.loc['self'], key):.1f} & "
                         f"{getattr(a.loc['heldout'], key):.1f} & {getattr(b.loc['none'], key):.1f} & "
                         f"{getattr(b.loc['self'], key):.1f} & {getattr(b.loc['heldout'], key):.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/pass_preload.tex", "w").write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- per-site profiles

def fig_sites(data, out):
    cases = [("darknet AlexNet", f"{data}/sites/darknet-alexnet"),
             ("miniVite 32768", f"{data}/corpus-sites/miniVite-32768"),
             ("llama.cpp 135M", f"{data}/corpus-sites/llamacpp-135m-n32"),
             ("llama2.c 110M", f"{data}/corpus-sites/llama2c-110M")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
    for ax, (name, d) in zip(axes, cases):
        b = blocks(d)
        g = b.groupby("site")[["size", "high_water", "touched"]].sum().sort_values("size", ascending=False).head(5)
        y = np.arange(len(g))[::-1]
        ax.barh(y, g["size"] / MB, height=0.7, color=GRID, edgecolor=MUTED, linewidth=0.5, label="Requested")
        ax.barh(y, g.high_water / MB, height=0.42, color=PAL[1], label="Up to highest byte touched")
        ax.barh(y, g.touched / MB, height=0.18, color=LIBMPR, label="Bytes touched")
        ax.set_yticks(y)
        ax.set_yticklabels([f"site {i + 1}" for i in range(len(g))])
        ax.set_title(name)
        ax.set_xlabel("MB, summed over blocks")
        ax.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=0.6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.1))
    save(fig, out, "sites")


# ---------------------------------------------------------------- llama.cpp

def fig_llamacpp(data, out):
    b = pd.read_csv(f"{data}/results/llamacpp_buffers.csv")
    order = ["glibc", "libmpr no zero skip", "libmpr zero skip + THP", "libmpr (default)"]
    labels = ["glibc", "libmpr,\nno zero skip", "libmpr, zero skip,\nhuge pages kept", "libmpr\n(default)"]
    parts = [("Other (weights, heap, code)", GRID), ("Compute buffer", PAL[1]), ("KV cache", LIBMPR)]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    for part, col in parts:
        v = np.array([b[(b.config == c) & (b.buffer == part)].rss_mb.iloc[0] for c in order])
        ax.bar(x, v, bottom=bottom, width=0.6, color=col, edgecolor="white", linewidth=1, label=part)
        if part == "KV cache":
            kv = v
            for xi, vi, bi in zip(x, v, bottom):
                if vi > 20:
                    ax.text(xi, bi + vi / 2 - 4, f"KV cache\n{vi:.1f}", ha="center", fontsize=5.5, color="white")
        bottom += v
    for xi, t, k in zip(x, bottom, kv):
        ax.text(xi, t + 6, f"{t:.0f} MB" + (f"\n(KV cache {k:.1f})" if k <= 20 else ""), ha="center", fontsize=6.5,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("RSS at peak (MB)")
    ax.set_ylim(0, 420)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6)
    ax.grid(axis="x", visible=False)
    save(fig, out, "llamacpp_buffers")


def fig_timeline(data, out):
    d = f"{data}/runs/ml/llamacpp-135m-n128"
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.0), sharex=True)
    for f, lab, col, ls in ((f"{d}/base-1.csv", "glibc", GLIBC, "-"), (f"{d}/selfnz-1.csv", "libmpr, no zero skip", PAL[1], (0, (4, 2))),
                            (f"{d}/self-1.csv", "libmpr (default)", LIBMPR, "-")):
        r = pd.read_csv(f)
        axes[0].plot(r.seconds, r.rss_bytes / MB, color=col, linestyle=ls, label=lab)
        axes[1].plot(r.seconds, r.commit_bytes / MB, color=col, linestyle=ls, label=lab)
    axes[0].set_ylabel("RSS (MB)")
    axes[1].set_ylabel("Commit (MB)")
    for ax in axes:
        ax.set_xlabel("Seconds")
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.08))
    save(fig, out, "llamacpp_timeline")


# ---------------------------------------------------------------- miniVite

def fig_minivite(data, out):
    b = pd.read_csv(f"{data}/results/minivite_breakdown.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    measures = ["Bytes touched (MemPrint)", "RSS at peak RSS", "Commit at peak commit"]
    colors = {"Data touched": LIBMPR, "brk heap (small blocks)": PAL[2], "edgeList buffer (anon mmap)": PAL[1],
              "Library and program code": PAL[6], "OpenMP thread stacks": PAL[3], "Other": GRID}
    seen = set()
    for i, m in enumerate(measures):
        left = 0
        for _, r in b[b.measure == m].iterrows():
            lab = r.component if r.component not in seen else None
            seen.add(r.component)
            ax.barh(len(measures) - 1 - i, r.mb, left=left, height=0.55, color=colors[r.component],
                    edgecolor="white", linewidth=1, label=lab)
            left += r.mb
        ax.text(left + 0.6, len(measures) - 1 - i, f"{left:.1f} MB", va="center", fontsize=6.5)
    ax.set_yticks(range(len(measures)))
    ax.set_yticklabels(measures[::-1], fontsize=6.5)
    ax.set_xlabel("MB")
    ax.set_xlim(0, 52)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", fontsize=5.5, ncol=3, bbox_to_anchor=(0.45, -0.22))
    ax.set_title("Where miniVite's memory goes (glibc)")
    ax = axes[1]
    for f, lab, col, ls in ((f"{data}/runs/minivite-v4/none-1.csv", "pass build, no reservations", GLIBC, "-"),
                            (f"{data}/runs/minivite-v4/self-1.csv", "pass build, own profile", LIBMPR, "-")):
        r = pd.read_csv(f)
        ax.plot(r.seconds, r.rss_bytes / MB, color=col, linestyle=ls, label=lab + ", RSS")
        ax.plot(r.seconds, r.commit_bytes / MB, color=col, linestyle=(0, (1, 1.5)), label=lab + ", commit")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("MB")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", fontsize=5.3)
    ax.set_title("Predicted vector capacity")
    fig.tight_layout()
    save(fig, out, "minivite")


# ---------------------------------------------------------------- corpus

def fig_corpus(data, out):
    t = pd.read_csv(f"{data}/runs/corpus/summary.csv").set_index("prog").sort_values("base_commit_x")
    y = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4), sharey=True)
    for ax, key, title in ((axes[0], "commit", "Peak commit / bytes touched"), (axes[1], "rss", "Peak RSS / bytes touched")):
        b, s = t[f"base_{key}_x"], t[f"self_{key}_x"]
        for yi, bv, sv in zip(y, b, s):
            ax.plot([bv, sv], [yi, yi], color=GRID, linewidth=1.5, zorder=1)
        ax.scatter(b, y, s=14, color=GLIBC, label="glibc", zorder=2)
        ax.scatter(s, y, s=14, color=LIBMPR, marker="D", label="libmpr, own profile", zorder=3)
        ax.axvline(1, color=INK, linewidth=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 1, 1.5, 2, 3])
        ax.set_xticklabels(["0.5", "1", "1.5", "2", "3"])
        ax.set_xlabel(title)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(t.index, fontsize=6.5)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, out, "corpus")


def table_corpus(data, out):
    t = pd.read_csv(f"{data}/runs/corpus/summary.csv")
    tot = t[["touched", "base_rss", "base_commit", "self_rss", "self_commit", "heldout_rss", "heldout_commit"]].sum() / MB
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"23 programs, largest inputs & glibc & libmpr, own & libmpr, held out \\",
             r"\midrule",
             f"Total peak commit (MB) & {n0(tot.base_commit)} & {n0(tot.self_commit)} & {n0(tot.heldout_commit)} \\\\",
             f"Total peak RSS (MB) & {n0(tot.base_rss)} & {n0(tot.self_rss)} & {n0(tot.heldout_rss)} \\\\",
             f"Median commit / touched & {t.base_commit_x.median():.2f} & {t.self_commit_x.median():.2f} & {t.heldout_commit_x.median():.2f} \\\\",
             f"Median RSS / touched & {t.base_rss_x.median():.2f} & {t.self_rss_x.median():.2f} & {t.heldout_rss_x.median():.2f} \\\\",
             f"Time relative to glibc (range) & 1 & {t.self_time_x.min():.2f} to {t.self_time_x.max():.2f} & {t.heldout_time_x.min():.2f} to {t.heldout_time_x.max():.2f} \\\\",
             f"Programs with identical output & -- & {int(t.self_ok.sum())} of 23 & {int(t.heldout_ok.sum())} of 23 \\\\",
             r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/corpus.tex", "w").write("\n".join(lines) + "\n")
    print("corpus totals", tot.round(1).to_dict())


# ---------------------------------------------------------------- clock and pages

ORACLE_LABEL = {"memprint-T": "Full trace, bytes, references", "memprint-Tsec": "Full trace, bytes, seconds",
                "memprint-Tpages": "Full trace, pages, references", "memprint-Tpagessec": "Full trace, pages, seconds",
                "memprint-W5": "W5, bytes, references", "memprint-W5sec": "W5, bytes, seconds",
                "memprint-W5pages": "W5, pages, references", "memprint-W5pagessec": "W5, pages, seconds"}


def fig_margins(data, out):
    m = pd.read_csv(f"{data}/sched-pages/margins.csv")
    order = list(ORACLE_LABEL)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for i, o in enumerate(order):
        v = m[m.oracle == o].margin.to_numpy()
        yi = len(order) - 1 - i
        jit = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.4
        col = LIBMPR if "sec" in o else GLIBC
        ax.scatter(v, yi + jit, s=9, color=col, alpha=0.8, linewidths=0)
        q = np.quantile(v, 0.9)
        ax.plot([q, q], [yi - 0.32, yi + 0.32], color=INK, linewidth=1.3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([ORACLE_LABEL[o] for o in order][::-1], fontsize=6.5)
    ax.set_xlabel("Margin needed to cover RSS (share of the curve's peak)\nbar: 90th percentile over 23 programs")
    ax.grid(axis="y", visible=False)
    save(fig, out, "margins")


def table_sched(data, out):
    m = pd.read_csv(f"{data}/sched-pages/margins.csv")
    g = pd.read_csv(f"{data}/results/grid_pages.csv").groupby("oracle").steady_response.mean()
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"MemPrint curve & Median margin & 90th percentile & Response (min) \\", r"\midrule"]
    for o, lab in ORACLE_LABEL.items():
        v = m[m.oracle == o].margin
        lines.append(f"{lab} & {v.median():.2f} & {v.quantile(0.9):.2f} & {n0(g[o])} \\\\")
    lines += [r"\midrule", f"Exact RSS peak & -- & -- & {g['exact-peak']:.0f} \\\\",
              f"Exact RSS curve & -- & -- & {g['exact']:.0f} \\\\",
              f"User request & -- & -- & {n0(g['request'])} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    text = "\n".join(lines) + "\n"
    open(f"{out}/tables/sched_pages.tex", "w").write(text)
    r = pd.read_csv(f"{data}/results/grid_rssmpr.csv")
    r = r[r.load == 0.8]
    # calibrated margins; the oracles without a profile to calibrate (exact, request) run at 0.1 only
    cal = r[r.margin == "cal"]
    r = pd.concat([cal, r[(r.margin != "cal") & ~r.oracle.isin(set(cal.oracle))]])
    r = r.groupby(["scenario", "oracle"]).steady_response.mean().unstack(0)
    names = {"exact": "Exact curve", "exact-peak": "Exact peak", "memprint-W5": "MemPrint W5", "dhat": "DHAT peak",
             "request": "User request"}
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Oracle & Truth: bytes touched & Truth: RSS & Truth: RSS under libmpr \\", r"\midrule"]
    for o, lab in names.items():
        lines.append(f"{lab} & {n0(r.loc[o, 'bytes'])} & {n0(r.loc[o, 'rss'])} & {n0(r.loc[o, 'rssmpr'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/sched_rssmpr.tex", "w").write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- unseen inputs and run parameters

INPUT_LABEL = {"lc-135m-n16": "135M, 16 tokens", "lc-135m-p2": "135M, prompt 2", "lc-135m-p3": "135M, prompt 3",
               "lc-135m-sampled": "135M, 128 sampled", "lc-135m-longprompt": "135M, 136-token prompt",
               "lc-135m-n256": "135M, 256 tokens", "lc-135m-n512": "135M, 512 tokens",
               "lc-135m-n1024": "135M, 1024 tokens", "lc-135m-c2048": "135M, context 2048",
               "lc-135m-c4096": "135M, context 4096", "lc-360m-n256": "360M, 256 tokens",
               "lc-360m-c4096": "360M, context 4096, long prompt",
               "l2-15M-n32": "15M, 32 steps", "l2-15M-sampled": "15M, 128 sampled", "l2-15M-full": "15M, 256 steps",
               "l2-42M-n64": "42M, 64 steps", "l2-42M-n512": "42M, 512 steps", "l2-42M-full": "42M, 1024 steps",
               "l2-110M-n256": "110M, 256 steps", "l2-110M-sampled": "110M, 512 sampled",
               "l2-110M-full": "110M, 1024 steps"}


def inputs(data):
    return pd.read_csv(f"{data}/runs/inputs-v2/summary.csv").set_index("input")


def fig_inputs(data, out):
    t = inputs(data)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    ax = axes[0]
    g = t[t.index.str.startswith("lc-135m") & ~t.index.str.contains("c2048|c4096")].sort_values("tokens")
    ax.plot(g.tokens, g.model_commit, "o-", color=HELD, ms=3, label="commit, per-model runs")
    ax.plot(g.tokens, g.param_commit, "o-", color=LIBMPR, ms=3, label="commit, periodic ($T$ from the run)")
    ax.plot(g.tokens, g.model_kv_beyond, "s--", color=HELD, ms=2.5, label="KV cache used beyond commit, runs")
    ax.plot(g.tokens, g.param_kv_beyond, "s--", color=LIBMPR, ms=2.5, label="KV cache used beyond commit, periodic")
    ax.set_xscale("log")
    ax.set_xlabel("Tokens processed $T$ (prompt + generated)")
    ax.set_ylabel("MB")
    ax.set_title("llama.cpp, SmolLM2-135M (glibc commits 405 MB)")
    ax.legend(loc="center left", fontsize=5.8)
    ax = axes[1]
    for i, m in enumerate(["15M", "42M", "110M"]):
        g = t[t.index.str.startswith(f"l2-{m}-")].sort_values("tokens")
        col = PAL[[2, 3, 6][i]]
        ax.plot(g.tokens, g.param_commit, "o-", color=col, ms=3, label=f"{m}: periodic")
        ax.plot(g.tokens, g.model_commit, "o:", color=col, ms=3, alpha=0.7, label=f"{m}: per-model runs")
        ax.plot(g.tokens, g.base_commit, "_", color=col, ms=9, mew=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Steps (an upper bound on the tokens)")
    ax.set_ylabel("Peak commit (MB)")
    ax.set_title("llama2.c (bars: glibc)")
    ax.legend(fontsize=5.8, ncol=1, loc="upper left")
    save(fig, out, "inputs")


def table_inputs(data, out):
    t = inputs(data)
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{glibc} & \multicolumn{3}{c}{per-model (\texttt{runs})} & \multicolumn{3}{c}{periodic, $T$ from the run} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Input & $T$ & RSS & Commit & RSS & Commit & Beyond & RSS & Commit & Beyond \\", r"\midrule"]
    lines[0] = r"\begin{tabular}{lrrrrrrrrr}"
    for name in INPUT_LABEL:
        if name not in t.index:
            continue
        x = t.loc[name]
        prog = "llama.cpp" if name.startswith("lc") else "llama2.c"
        lines.append(f"{prog} {INPUT_LABEL[name]} & {int(x.tokens)} & {x.base_rss:.0f} & {x.base_commit:.1f} & "
                     f"{x.model_rss:.0f} & {x.model_commit:.1f} & {x.model_beyond:.1f} & "
                     f"{x.param_rss:.0f} & {x.param_commit:.1f} & {x.param_beyond:.1f} \\\\")
        if name == "lc-360m-c4096":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/inputs.tex", "w").write("\n".join(lines) + "\n")
    ok = bool(t[["model_ok", "param_ok"]].all().all())
    print(f"inputs: {len(t)} inputs, outputs identical: {ok}, time model {t.model_time_x.min():.2f}-{t.model_time_x.max():.2f}, "
          f"param {t.param_time_x.min():.2f}-{t.param_time_x.max():.2f}")


# ---------------------------------------------------------------- per-site profiles from source


def table_static(data, out):
    """Agreement and cost of the two per-site profilers (static/overhead.csv: one process at a time)."""
    f = pd.read_csv(f"{data}/static/overhead.csv")
    lines = [r"\begin{tabular}{llrrrrrrrrrr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{High water (MB)} & Largest & Native & \multicolumn{2}{c}{Pin \texttt{-sites}} & "
             r"\multicolumn{2}{c}{Interpreter} & Peak RSS (MB) \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
             r"Kernel & Scale & Sites & static & Pin & site diff. & (s) & (s) & slowdown & (s) & slowdown & Pin / interp. \\",
             r"\midrule"]
    for x in f.itertuples():
        lines.append(f"{x.kernel} & {x.scale} & {x.sites_both} / {x.sites_pin} & {x.hw_st:.1f} & {x.hw_pin:.1f} & "
                     f"{100 * x.max_site_hw_err:.1f}\\% & {x.native_s:.2f} & {x.pin_s:.0f} & {x.pin_x:.0f}$\\times$ & "
                     f"{x.st_s:.0f} & {x.st_x:,.0f}$\\times$ & {x.pin_rss_mb:,.0f} / {x.st_rss_mb:,.0f} \\\\".replace(",", "{,}"))
    lines += [r"\midrule",
              f"Total & & & & & & {f.native_s.sum():.1f} & {f.pin_s.sum():.0f} & & {f.st_s.sum():,.0f} & & \\\\".replace(",", "{,}"),
              r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/static.tex", "w").write("\n".join(lines) + "\n")


PIN_GROUPS = [("PolyBench (2mm, gemm, jacobi-2d, atax)", ["2mm", "gemm", "jacobi-2d", "atax"]),
              ("GAP (bfs, pr, cc, sssp; scale 18)", ["gap-bfs", "gap-pr", "gap-cc", "gap-sssp"]),
              ("HPC proxies (HPCCG, LULESH, miniFE, Kripke, Quicksilver)", ["hpccg", "lulesh", "minife", "kripke", "quicksilver"]),
              ("NPB (IS, CG, MG)", ["npb-is", "npb-cg", "npb-mg"]),
              ("Interpreters, database, compiler (Lua, Python, Perl, SQLite, cc1)", ["lua", "python", "perl", "sqlite", "cc1"]),
              ("miniVite 32768", ["miniVite"]), ("darknet AlexNet", ["darknet"]),
              ("llama2.c (15M, 42M, 110M; 128 steps)", ["llama2c-15M", "llama2c-42M", "llama2c-110M"]),
              ("llama.cpp (135M 32/128 tokens, 360M 64)", ["llamacpp-135m-n32", "llamacpp-135m-n128", "llamacpp-360m-n64"])]


def table_pin_cost(data, out):
    t = pd.read_csv(f"{data}/results/pin_cost.csv").set_index("workload")
    rng = lambda v, f: f"{f(v.min())}" if len(v) == 1 or f(v.min()) == f(v.max()) else f"{f(v.min())} to {f(v.max())}"
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Workloads & Native (s) & Pin \texttt{-sites} (s) & Slowdown & Peak RSS under Pin (GB) \\", r"\midrule"]
    for label, names in PIN_GROUPS:
        g = t.loc[[n for n in names if n in t.index]]
        nums = (f"{rng(g.native_s, lambda x: f'{x:.1f}')} & {rng(g.pin_s, lambda x: f'{x:,.0f}')} & "
                f"{rng(g.slowdown, lambda x: f'{x:.0f}')} & {rng(g.pin_rss_mb / 1024, lambda x: f'{x:.1f}')}")
        lines.append(f"{label} & {nums.replace(',', '{,}')} \\\\")
    lines += [r"\midrule", f"All 29 & & & median {t.slowdown.median():.0f} & \\\\", r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/pin_cost.tex", "w").write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- 2026-10-14: stacks, profiling under libmpr, prompt


def _num(x, d=1):
    return "--" if pd.isna(x) else f"{x:,.{d}f}".replace(",", "{,}")


def table_stacks(data, out):
    t = pd.read_csv(f"{data}/runs/stacks_summary.csv").set_index("name")
    keep = t[(t.nostacks_commit - t.stacks_commit).abs() > 0.5]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Program & glibc commit & stacks off & stacks on & RSS off / on & Time on \\", r"\midrule"]
    for n, x in keep.iterrows():
        lines.append(f"{n} & {_num(x.base_commit)} & {_num(x.nostacks_commit)} & {_num(x.stacks_commit)} & "
                     f"{_num(x.nostacks_rss)} / {_num(x.stacks_rss)} & {x.stacks_time_x:.2f} \\\\")
    rest = t.drop(keep.index)
    lines += [r"\midrule", f"{len(rest)} other programs & {_num(rest.base_commit.sum(), 0)} & {_num(rest.nostacks_commit.sum(), 0)} & "
              f"{_num(rest.stacks_commit.sum(), 0)} & {_num(rest.nostacks_rss.sum(), 0)} / {_num(rest.stacks_rss.sum(), 0)} & "
              f"{rest.stacks_time_x.min():.2f}--{rest.stacks_time_x.max():.2f} \\\\", r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/stacks.tex", "w").write("\n".join(lines) + "\n")


def table_nprof(data, out):
    t = pd.read_csv(f"{data}/runs/nprof_summary.csv").set_index("name")
    pin = pd.read_csv(f"{data}/results/pin_cost.csv").set_index("workload")
    pin = pin.rename(index={"llama2c-15M": "llama2c-15M", "llamacpp-135m-n32": "llamacpp-135m-n32"})
    t["pin_s"] = [pin.pin_s.get(n, float("nan")) for n in t.index]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Profiling time (s)} & \multicolumn{3}{c}{Peak commit (MB)} & Peak RSS (MB) \\",
             r"\cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(lr){7-7}",
             r"Program & Pin & libmpr & glibc & Pin's profile & libmpr's & Pin's / libmpr's \\", r"\midrule"]
    for n, x in t.iterrows():
        lines.append(f"{n} & {_num(x.pin_s, 0)} & {_num(x.prof_s, 1)} & {_num(x.base_commit)} & {_num(x.pin_commit)} & "
                     f"{_num(x.native_commit)} & {_num(x.pin_rss)} / {_num(x.native_rss)} \\\\")
    lines += [r"\midrule", f"Total & {_num(t.pin_s.sum(), 0)} & {_num(t.prof_s.sum(), 0)} & {_num(t.base_commit.sum(), 0)} & "
              f"{_num(t.pin_commit.sum(), 0)} & {_num(t.native_commit.sum(), 0)} & {_num(t.pin_rss.sum(), 0)} / "
              f"{_num(t.native_rss.sum(), 0)} \\\\", r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/nprof.tex", "w").write("\n".join(lines) + "\n")


def table_prompt(data, out):
    t = pd.read_csv(f"{data}/runs/prompt_summary.csv").set_index("name")
    order = ["lc-135m-n16", "lc-135m-p2", "lc-135m-p3", "lc-135m-sampled", "lc-135m-longprompt", "lc-135m-n256",
             "lc-135m-n512", "lc-135m-n1024", "lc-135m-c2048", "lc-135m-c4096"]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Commit (MB)} & \multicolumn{2}{c}{Beyond the commitment (MB)} & \\",
             r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
             r"Input & tokens & + prompt & tokens & + prompt & RSS (MB) \\", r"\midrule"]
    for n in order:
        x = t.loc[n]
        lab = INPUT_LABEL.get(n, n).replace("135M, ", "")
        lines.append(f"{lab} & {_num(x.param_commit)} & {_num(x.param2_commit)} & {_num(x.param_beyond, 2)} & "
                     f"{_num(x.param2_beyond, 2)} & {_num(x.param2_rss, 0)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/prompt.tex", "w").write("\n".join(lines) + "\n")


def server_summary(data):
    """runs/server/<model>-c<ctx>-np<clients>/ (ml/server_eval.sh): glibc and libmpr peak RSS and commit (MB, mean of 3
    runs), run time ratio, whether every run served every request with the same token counts, and the profile's time."""
    rows = []
    for d in sorted(glob.glob(f"{data}/runs/server/*-c*-np*/")):
        s = summary(f"{d}summary.txt")
        status = pd.read_csv(f"{d}summary.txt", sep=r"\s+", header=None).iloc[:, 5]
        outs = [l.split() for l in open(f"{d}outputs.txt") if l.strip()]
        model, ctx, np_ = d.rstrip("/").split("/")[-1].split("-")
        rows.append(dict(model=model.upper(), ctx=int(ctx[1:]), clients=int(np_[2:]), base_rss=s.loc["base"].rss,
                         mpr_rss=s.loc["mpr"].rss, base_commit=s.loc["base"].commit, mpr_commit=s.loc["mpr"].commit,
                         time_x=s.loc["mpr"].secs / s.loc["base"].secs, ok=bool((status == 0).all()),
                         same=len({o[1] for o in outs}) == 1, requests=int(outs[0][2]),
                         profile_s=float(open(f"{d}profile.time").read().split()[0])))
    t = pd.DataFrame(rows).sort_values(["model", "ctx", "clients"])
    t.to_csv(f"{data}/results/server_summary.csv", index=False)
    return t


def table_server(data, out):
    t = server_summary(data)
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{Peak RSS (MB)} & \multicolumn{2}{c}{Peak commit (MB)} & & \\",
             r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}",
             r"Model & Context & Clients & glibc & \libmpr{} & glibc & \libmpr{} & Time & Outputs \\", r"\midrule"]
    for r in t.itertuples():
        if not r.ok:   # llama.cpp itself ran out of KV-cache slots (16 clients in an 8192-token context)
            lines.append(f"SmolLM2-{r.model} & {r.ctx} & {r.clients} & \\multicolumn{{6}}{{l}}{{llama.cpp ran out of KV-cache slots "
                         f"under glibc and \\libmpr{{}} alike}} \\\\")
            continue
        lines.append(f"SmolLM2-{r.model} & {r.ctx} & {r.clients} & {_num(r.base_rss, 0)} & {_num(r.mpr_rss, 0)} & "
                     f"{_num(r.base_commit, 0)} & {_num(r.mpr_commit, 0)} & {r.time_x:.2f} & "
                     f"{'same' if r.same else 'differ'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/server.tex", "w").write("\n".join(lines) + "\n")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/memprint-data/2026-10-12-resize"))
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    os.makedirs(f"{a.out}/figures", exist_ok=True)
    os.makedirs(f"{a.out}/tables", exist_ok=True)
    fig_improvements(a.data, a.out)
    table_pass(a.data, a.out)
    rows = headline(a.data)
    table_headline(rows, a.out)
    fig_headline(rows, a.out)
    fig_sites(a.data, a.out)
    fig_llamacpp(a.data, a.out)
    fig_timeline(a.data, a.out)
    fig_minivite(a.data, a.out)
    fig_corpus(a.data, a.out)
    table_corpus(a.data, a.out)
    fig_margins(a.data, a.out)
    table_sched(a.data, a.out)
    if os.path.exists(f"{a.data}/runs/inputs-v2/summary.csv"):
        fig_inputs(a.data, a.out)
        table_inputs(a.data, a.out)
    if os.path.exists(f"{a.data}/results/pin_cost.csv"):
        table_pin_cost(a.data, a.out)
    for f, fn in (("stacks_summary.csv", table_stacks), ("nprof_summary.csv", table_nprof), ("prompt_summary.csv", table_prompt)):
        if os.path.exists(f"{a.data}/runs/{f}"):
            fn(a.data, a.out)
    if os.path.exists(f"{a.data}/static/overhead.csv"):
        table_static(a.data, a.out)
    if glob.glob(f"{a.data}/runs/server/*-c*-np*/summary.txt"):
        print(table_server(a.data, a.out).round(2).to_string(index=False))
    for name, b, s, h in rows:
        print(f"{name:26s} glibc {b.rss:7.1f} {b.commit:7.1f}  own {s.rss:7.1f} {s.commit:7.1f}  held {h.rss:7.1f} {h.commit:7.1f}  "
              f"time {s.secs / b.secs:.2f} same {s.same and h.same}")


if __name__ == "__main__":
    main()
