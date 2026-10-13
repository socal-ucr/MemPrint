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
    for name, b, s, h in rows:
        print(f"{name:26s} glibc {b.rss:7.1f} {b.commit:7.1f}  own {s.rss:7.1f} {s.commit:7.1f}  held {h.rss:7.1f} {h.commit:7.1f}  "
              f"time {s.secs / b.secs:.2f} same {s.same and h.same}")


if __name__ == "__main__":
    main()
