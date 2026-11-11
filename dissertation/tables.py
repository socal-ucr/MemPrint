"""LaTeX tables for dissertation/results.tex from the scored data in $OUT (dissertation/score.py, figures.py).

    PYTHONPATH=analysis ~/environments/memfp/bin/python -W ignore dissertation/tables.py  -> dissertation/tables/*.tex
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from memprint.dataset import prepare  # noqa: E402

OUT = os.path.expanduser("~/memprint-data/2026-11-05-diss-ch2")
TAB = os.path.join(HERE, "tables")
TACO = pd.read_csv(f"{OUT}/m1/taco/data/fits.csv")
BYTES = pd.read_csv(f"{OUT}/m1/final/data/fits.csv")     # 28 workloads with M4's miniVite, plus M5's two kernels
WORKLOADS = open(os.path.join(HERE, "..", "analysis/workloads.txt")).read().split()


def write(name, body):
    os.makedirs(TAB, exist_ok=True)
    open(f"{TAB}/{name}.tex", "w").write(body)
    print("wrote", f"{TAB}/{name}.tex")


def pct(x, d=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def num(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def sampler_mape():
    m = pd.read_csv(f"{OUT}/m3_mape.csv")
    m = m[m.basis == "train_si"]
    return m.groupby(["sampler", "workload", "split", "subset"]).sampler_mape.mean()


def m1():
    s = sampler_mape().loc["boot"]
    rows = []
    for split in ("EXTRA", "INTER"):
        for sub in ("NZ", "MT", "L2O"):
            t = TACO[(TACO.split == split) & (TACO.subset == sub)].test_mape_train_si
            b = BYTES[(BYTES.split == split) & (BYTES.subset == sub) & BYTES.workload.isin(WORKLOADS)]
            sb = s.xs((split, sub), level=("split", "subset"))
            rows.append(f"{split if sub == 'NZ' else ''} & {sub} & {pct(t.mean())} & {pct(t.median())} & {pct(t.max())} & "
                        f"{pct(b.test_mape_train_si.mean())} & {pct(b.test_mape_train_si.median())} & "
                        f"{pct(b.test_mape_train_si.max())} & {pct(sb.mean())} & {pct(sb.median())} \\\\")
        if split == "EXTRA":
            rows.append("\\midrule")
    write("m1", "\n".join(rows) + "\n")


def l2o():
    s = sampler_mape().loc["boot"]
    rows = []
    for w in WORKLOADS:
        cells = [w]
        for split in ("EXTRA", "INTER"):
            t = TACO[(TACO.workload == w) & (TACO.split == split) & (TACO.subset == "L2O")]
            b = BYTES[(BYTES.workload == w) & (BYTES.split == split) & (BYTES.subset == "L2O")]
            cells += [pct(t.test_mape_train_si.iloc[0]) if len(t) else "--",
                      pct(b.test_mape_train_si.iloc[0]) if len(b) else "--",
                      str(int(b.train_si.iloc[0])) if len(b) else "--",
                      pct(s.get((w, split, "L2O"), np.nan))]
        rows.append(" & ".join(cells) + " \\\\")
    write("l2o", "\n".join(rows) + "\n")


def m2():
    t = pd.read_csv(f"{OUT}/m2_runs.csv")
    t = t[t.i < 1e9]
    src = {"traces": "miniVite (published sampler)", "correlation-old": "correlation-old",
           "correlation-recentOld": "correlation-recentOld", "gemm_prev": "gemm\\_prev", "mvt-1000": "mvt-1000",
           "miniVite_fileInput": "miniVite\\_fileInput"}
    rows = []
    for d, g in t.groupby("dir", sort=False):
        r = g.label_over_emp
        rows.append(f"{src[d]} & {len(g)} & {', '.join(str(b) for b in sorted(g.bins.unique()))} & "
                    f"{r.min():.2f} & {r.median():.2f} & {r.max():.2f} \\\\")
    write("m2", "\n".join(rows) + "\n")


def m3():
    r = pd.read_csv(f"{OUT}/m3_runs.csv")
    m = pd.read_csv(f"{OUT}/m3_mape.csv")
    m = m[m.basis == "train_si"]
    g = m.groupby(["sampler", "workload", "split", "subset"])[["sampler_mape", "splitter_mape"]].mean()
    g["diff"] = g.sampler_mape - g.splitter_mape
    rows = []
    for s, name in (("boot", "Bootstrap, $s=100$, $r=10$"), ("disj", "Disjoint, 20 bins")):
        x = r[r.sampler == s]
        d = g.loc[s]
        rows.append(f"{name} & {x.mean_ratio.median():.3f} & {x.sd_ratio.median():.3f} & "
                    f"{x.sd_ratio.quantile(.1):.2f} to {x.sd_ratio.quantile(.9):.2f} & "
                    f"{d['diff'].median():.2f} & {d.xs('L2O', level='subset').loc[:, 'sampler_mape'].groupby('split').mean()['EXTRA']:.2f} & "
                    f"{d.xs('L2O', level='subset').loc[:, 'sampler_mape'].groupby('split').mean()['INTER']:.2f} \\\\")
    write("m3", "\n".join(rows) + "\n")


def m4():
    t = pd.read_csv(f"{OUT}/m4.csv")
    rows = [f"{num(r.Config)} & {num(r.starts_B)} & {num(r.bytes_alloc_B)} & {num(r.bytes_B)} & "
            f"{r.vs_alloc_pct:.2f} & {r.vs_starts_pct:.2f} \\\\" for r in t.itertuples()]
    write("m4_truth", "\n".join(rows) + "\n")
    pub = {"EXTRA": [3.98, 4.98, 12.90], "INTER": [2.94, 1.04, 1.57]}
    s = sampler_mape().loc["boot"]
    f = BYTES[BYTES.workload == "miniVite"]
    rows = []
    for split in ("EXTRA", "INTER"):
        for i, sub in enumerate(("NZ", "MT", "L2O")):
            r = f[(f.split == split) & (f.subset == sub)].iloc[0]
            rows.append(f"{split if i == 0 else ''} & {sub} & {pub[split][i]:.2f} & {int(r.train_si):,} & "
                        f"{r.test_mape_train_si:.2f} & {pct(s.get(('miniVite', split, sub), np.nan))} \\\\".replace(",", "{,}"))
    write("m4_mape", "\n".join(rows) + "\n")


def m6():
    t = pd.read_csv(f"{OUT}/m6.csv")
    e = pd.read_csv(f"{OUT}/m6_memprint.csv")
    e["config"] = e.config.astype(str)
    order = {"gemm": ["MINI", "MINI2", "MINI3", "SMALL", "SMALL2", "SMALL3", "MEDIUM"],
             "miniVite": ["1024", "2048", "4096", "8192", "16384"]}
    rows = []
    for w, cs in order.items():
        g = t[t.workload == w].assign(config=lambda x: x.config.astype(str)).set_index("config").loc[cs]
        for i, (c, r) in enumerate(g.iterrows()):
            mp = e[(e.workload == w) & (e.config == c)]
            mcell = f"{mp.ratio.iloc[0]:.3f} ({mp.split.iloc[0]})" if len(mp) else "--"
            rows.append(f"{w if i == 0 else ''} & {num(int(c)) if c.isdigit() else c} & {num(r.truth_B)} & "
                        f"{num(r.massif_B)} & {num(r.dhat_B)} & {r.massif_over_truth:.3f} & {r.dhat_over_truth:.3f} & "
                        f"{mcell} \\\\")
        if w == "gemm":
            rows.append("\\midrule")
    write("m6", "\n".join(rows) + "\n")


def m7():
    k = pd.read_csv(f"{OUT}/m7_k.csv").set_index("config")
    a = pd.read_csv(f"{OUT}/m7.csv").set_index("config")
    native = k.loc["native", "median"]
    rows = []
    for n, name in (("published50", "Published sampler, $s=50$, $r=5$"), ("published", "Published sampler, $s=100$, $r=10$"),
                    ("boot", "Bootstrap, $s=100$, $r=10$"), ("disj", "Disjoint, 20 bins")):
        cells = [name]
        for x in (500, 1000, 2500):
            v = k.loc[f"{n}-{x}", "median"]
            cells.append(f"{v:.1f} ({v / native:.0f}$\\times$)")
        rows.append(" & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    for n, name in (("instr", "Instrumentation only"), ("splitter", "Splitter, 13 intervals $\\times$ 20 bins")):
        v = a.loc[n, "median"]
        rows.append(f"{name} & \\multicolumn{{3}}{{c}}{{{v:.1f} ({v / a.loc['native', 'median']:.0f}$\\times$)}} \\\\")
    write("m7", "\n".join(rows) + "\n")


def r1():
    s = pd.read_csv(f"{OUT}/r1_splits.csv")
    rows = []
    for split in ("EXTRA", "INTER"):
        g = s[s.split == split]
        cells = [split]
        for c in ("err_a", "err_b", "transfer_err"):
            cells.append(f"{g[c].abs().mean():.2f} / {g[c].abs().median():.2f}")
        for c in ("memprint_splitter", "memprint_sampler"):
            cells.append(f"{g[c].mean():.2f} / {g[c].median():.2f}")
        cells.append(f"{(g.err_b.abs() < 10).sum()} / {(g.transfer_err.abs() < 10).sum()}")
        rows.append(" & ".join(cells) + " \\\\")
    write("r1", "\n".join(rows) + "\n")


def m9():
    m = pd.read_csv(f"{OUT}/m9.csv").set_index("workload")
    names = {"miniVite": "miniVite, 8{,}192", "gemm": "gemm, MEDIUM", "2mm": "2mm, MEDIUM", "jacobi-2d": "jacobi-2d, MEDIUM",
             "atax": "atax, MEDIUM"}
    rows = []
    for w, n in names.items():
        r = m.loc[w]
        rows.append(f"{n} & {r.native:.2f} & {r.instr:.1f} & {r.truth:.1f} & {r.splitter:.1f} & {r['boot-1000']:.1f} & "
                    f"{r['disj-1000']:.1f} & {r['truth_over_boot-1000']:.1f}$\\times$ & {r['truth_over_disj-1000']:.1f}$\\times$ \\\\")
    write("m9", "\n".join(rows) + "\n")


def r1p():
    """R1 against the pages truth (primary) and the bytes truth, under the 1 ms and the event sampler."""
    rows = []
    for split in ("EXTRA", "INTER"):
        for tag, name in (("", "smaps\\_rollup, 1 ms"), ("_r8", "Event (R8)")):
            p = pd.read_csv(f"{OUT}/r1{tag}_pages_splits.csv")
            b = pd.read_csv(f"{OUT}/r1{tag}_splits.csv")
            p, b = p[p.split == split], b[b.split == split]
            f = lambda g, c: f"{g[c].abs().mean():.2f} / {g[c].abs().median():.2f}"
            cells = [split if tag == "" else "", name, f(p, "err_a"), f(p, "err_b"), f(p, "transfer_err"),
                     f"{(p.err_b.abs() < 10).sum()} / {(p.transfer_err.abs() < 10).sum()}", f(b, "err_b"), f(b, "transfer_err")]
            rows.append(" & ".join(cells) + " \\\\")
        if split == "EXTRA":
            rows.append("\\midrule")
    write("r1p", "\n".join(rows) + "\n")


def r8():
    """paper_rss's short-run programs, blind configurations and cost under the 1 ms and the event sampler."""
    d = os.path.join(HERE, "..", "paper_rss", "data")
    tr = pd.read_csv(f"{d}/sample_transfer.csv").groupby(["prog", "sampler"]).signed.median().unstack()
    bl = pd.read_csv(f"{d}/sample_blind.csv")
    bl = bl[bl.truth.notna()].groupby("sampler").err_b.apply(lambda x: x.abs().median())
    sp = pd.read_csv(f"{d}/sample_blind_splits.csv")
    c = pd.read_csv(f"{d}/sample_cost_r8.csv").set_index("prog")
    pk = pd.read_csv(f"{d}/sample_peaks.csv").groupby("sampler").samples.median()
    sgn = lambda v: ("$-$" if v < 0 else "$+$") + f"{abs(v):.1f}"
    cols = ("pss1", "event")
    rows = []
    for prog, name in (("npb-mg", "NPB-MG"), ("2mm", "2mm"), ("gemm", "gemm"), ("jacobi-2d", "jacobi-2d"), ("atax", "atax")):
        rows.append(f"Transfer error, {name} (\\%) & " + " & ".join(sgn(tr.loc[prog, s]) for s in cols) + " \\\\")
    rows.append(f"Transfer, median $|$error$|$ of {len(tr)} programs (\\%) & " + " & ".join(f"{tr[s].abs().median():.1f}" for s in cols) + " \\\\")
    rows.append("Blind configurations, median $|$error$|$ of (b) (\\%) & " + " & ".join(f"{bl[s]:.2f}" for s in cols) + " \\\\")
    for prog in ("CG", "IS", "MG"):
        rows.append(f"Blind NPB {prog}, transfer from S (\\%) & " + " & ".join(
            sgn(sp[(sp.program == prog) & (sp.split == "EXTRA") & (sp.sampler == s)].err.iloc[0]) for s in cols) + " \\\\")
    rows.append("Samples per smaller-input run, median & " + " & ".join(f"{pk[s]:.0f}" for s in cols) + " \\\\")
    rows.append(f"Run time over the program alone, median of {len(c)} ($\\times$) & " + " & ".join(f"{c[s].median():.2f}" for s in cols) + " \\\\")
    rows.append("\\quad largest ($\\times$) & " + " & ".join(f"{c[s].max():.2f} ({c[s].idxmax()})" for s in cols) + " \\\\")
    write("r8", "\n".join(rows) + "\n")


def r10():
    """R10: the in-run base in place of R and in the transfer rule, event sampler, pages truth; the blind NPB
    transfers with and without the base."""
    d = pd.read_csv(f"{OUT}/r10_r1.csv")
    d = d[(d.sampler == "event") & (d.truth == "pages")]
    f = lambda g, c: f"{g[c].abs().mean():.2f} / {g[c].abs().median():.2f}"
    rows = []
    for c, name in (("b_R", "(b) measured $+ R$"), ("b_file0 alone", "(b) measured $+$ file-backed RSS at start"),
                    ("b_file0", "(b) measured $+$ file-backed RSS at start $+$ constant"),
                    ("b_base0", "(b) measured $+$ (anon $-$ free) at start $+$ constant"),
                    ("c_plain", "(c) transfer $+ R$"), ("c_base0", "(c) transfer of measured $-$ base, $+$ base $+ R$")):
        rows.append(f"{name} & " + " & ".join(f(d[d.split == s_], c) for s_ in ("EXTRA", "INTER")) + " \\\\")
    write("r10", "\n".join(rows) + "\n")
    b = pd.read_csv(f"{OUT}/r10_blind.csv")
    b = b[b.program.isin(["CG", "IS", "MG"])].set_index(["program", "sampler"])
    sgn = lambda v: ("$-$" if v < 0 else "$+$") + f"{abs(v):.1f}"
    rows = [f"NPB {p} & " + " & ".join(sgn(b.loc[(p, s_), c]) for s_ in ("pss1", "event") for c in ("c_plain", "c_base0")) + " \\\\"
            for p in ("CG", "IS", "MG")]
    write("r10blind", "\n".join(rows) + "\n")


def m11():
    """M11: the TACO protocol's test MAPE under the bytes and the pages truth, on the splitter's bins and (pages) on
    the samplers' bins at the held-out inputs."""
    s = pd.read_csv(f"{OUT}/m11_summary.csv")
    sam = pd.read_csv(f"{OUT}/m3_mape_pages.csv")
    sam = sam[sam.basis == "train_si"].groupby(["sampler", "split", "subset", "workload"]).sampler_mape.mean()
    rows = []
    for split in ("EXTRA", "INTER"):
        for sub in ("NZ", "MT", "L2O"):
            g = lambda st: s[(s.set == st) & (s.split == split) & (s.subset == sub)].iloc[0]
            b, p = g("bytes"), g("pages")
            sm = [sam.loc[(x, split, sub)] for x in ("boot", "disj")]
            rows.append(f"{split if sub == 'NZ' else ''} & {sub} & {b['mean']:.2f} / {b['median']:.2f} & {b['max']:.2f} & "
                        f"{p['mean']:.2f} / {p['median']:.2f} & {p['max']:.2f} & "
                        + " & ".join(f"{x.mean():.2f} / {x.median():.2f}" for x in sm) + " \\\\")
        if split == "EXTRA":
            rows.append("\\midrule")
    write("m11", "\n".join(rows) + "\n")


def b50():
    """B50: test MAPE with 50 splitter bins against 20, on the splitter's bins and (50) on the samplers' bins."""
    s = pd.read_csv(f"{OUT}/b50_summary.csv")
    sam = pd.read_csv(f"{OUT}/m3_mape_b50.csv")
    sam = sam[sam.basis == "train_si"].groupby(["sampler", "split", "subset", "workload"]).sampler_mape.mean()
    rows = []
    for split in ("EXTRA", "INTER"):
        for sub in ("NZ", "MT", "L2O"):
            g = lambda b: s[(s.bins == b) & (s.split == split) & (s.subset == sub)].iloc[0]
            a, b = g(20), g(50)
            sm = [sam.loc[(x, split, sub)] for x in ("boot", "disj")]
            rows.append(f"{split if sub == 'NZ' else ''} & {sub} & {a['mean']:.2f} / {a['median']:.2f} & {a['max']:.2f} & "
                        f"{b['mean']:.2f} / {b['median']:.2f} & {b['max']:.2f} & "
                        + " & ".join(f"{x.mean():.2f} / {x.median():.2f}" for x in sm) + " \\\\")
        if split == "EXTRA":
            rows.append("\\midrule")
    write("b50", "\n".join(rows) + "\n")


def m8():
    m = pd.read_csv(f"{OUT}/m8.csv")
    rows = []
    for w in WORKLOADS:
        t, b = m[(m.set == "taco") & (m.workload == w)].iloc[0], m[(m.set == "bytes") & (m.workload == w)].iloc[0]
        if t.used == 13 and b.used == 13:
            continue
        f = lambda x: "--" if pd.isna(x) else ", ".join(f"{int(v):,}" for v in str(x).split()).replace(",", "{,}", ).replace("{,} ", ", ")
        rows.append(f"{w} & {t.used} & {f(t.dropped)} & {b.used} & {f(b.dropped)} \\\\")
    write("m8", "\n".join(rows) + "\n")


if __name__ == "__main__":
    for part in sys.argv[1:] or ["m1", "l2o", "m2", "m3", "m4", "m6", "m7", "r1", "m8", "m9", "r1p", "r8"]:
        globals()[part]()
