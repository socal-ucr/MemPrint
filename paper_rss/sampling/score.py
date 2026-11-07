"""Score the runs of PROTOCOL.md (run.sh).

    PYTHONPATH=analysis ~/environments/memfp/bin/python -W ignore paper_rss/sampling/score.py [part ...]
        ->  paper_rss/data/sample_{peaks,transfer,cost,large,blind,thp,py,cc1}.csv
Parts: peaks transfer cost blind thp py cc1 (default all); cost_r8 (large_r8: plain, pss1 and event interleaved).
"""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import compute as M  # noqa: E402
from memprint.sched import corpus as C, earlier  # noqa: E402

OUT = os.path.expanduser("~/memprint-data/2026-11-05-rsssample")
DATA = os.path.join(HERE, "..", "data")
SAMPLERS = ("pss5", "pss1", "fast", "faults", "event")   # event: R8, mprshim-event.so
NAMED = ("npb-mg", "2mm", "gemm", "jacobi-2d")


def load(path):
    """A sampler's CSV, its samples with a nonzero RSS, and the kernel's peak RSS from the .hwm file (NaN if none)."""
    r = pd.read_csv(path)
    r = r[r.rss_bytes > 0]
    h = f"{path}.hwm"
    hwm = float(open(h).read().split()[-1]) if os.path.exists(h) else np.nan
    return r, hwm


def data_curve(r):
    return C.resample(r.seconds, np.clip(r.anon_bytes - r.free_bytes, 0, None))


def peaks():
    """(a) Per smaller-input run and sampler: samples, the sampled RSS peak over VmHWM, the data estimate's peak."""
    rows = []
    for prog in sorted(os.listdir(f"{OUT}/small")):
        for inp in sorted(os.listdir(f"{OUT}/small/{prog}")):
            for rep in (1, 2, 3):
                for s in SAMPLERS:
                    p = f"{OUT}/small/{prog}/{inp}/{rep}/{s}.csv"
                    if not os.path.exists(p):
                        continue
                    r, hwm = load(p)
                    wall = float(open(f"{OUT}/small/{prog}/{inp}/{rep}/{s}.time").read().split()[0])
                    rows.append(dict(prog=prog, input=inp, rep=rep, sampler=s, wall=wall, samples=len(r),
                                     rss_over_hwm=r.rss_bytes.max() / hwm,
                                     data_peak=float((r.anon_bytes - r.free_bytes).clip(lower=0).max())))
    d = pd.DataFrame(rows)
    d.to_csv(f"{DATA}/sample_peaks.csv", index=False)
    g = d.groupby(["prog", "input", "sampler"]).median(numeric_only=True).reset_index()
    print(g.pivot_table(index=["prog", "input"], columns="sampler", values="rss_over_hwm").round(3).to_string())
    print(g.groupby("sampler")[["rss_over_hwm", "samples"]].median().round(3))
    short = g[g.wall < 0.1]
    print("runs under 0.1 s:", short[["prog", "input"]].drop_duplicates().shape[0])
    print(short.groupby("sampler").rss_over_hwm.agg(["median", "min"]).round(3))


def transfer():
    """(b) The transfer to the largest input from each sampler's first runs of the smaller inputs (earlier.transfer),
    plus the runtime's pages, against the pages-in-use truth, as compute.transfer scores it."""
    ins = M.inputs()
    rows = []
    for s, rep in [(s, rep) for s in SAMPLERS for rep in (1, 2, 3)]:   # rep 1 is the protocol's; 2 and 3 a check
        cur = []
        if not os.path.exists(f"{OUT}/small/npb-mg/{os.listdir(f'{OUT}/small/npb-mg')[0]}/{rep}/{s}.csv"):
            continue
        for prog in sorted(os.listdir(f"{OUT}/small")):
            for inp in os.listdir(f"{OUT}/small/{prog}"):
                p = f"{OUT}/small/{prog}/{inp}/{rep}/{s}.csv"
                if os.path.exists(p):
                    v = data_curve(load(p)[0])
                    cur.append(pd.DataFrame(dict(prog=prog, input=inp, method="model1", k=np.arange(len(v)), bytes=v)))
            big = ins[prog]   # the target's own curve is not used; it names the target input
            cur.append(pd.DataFrame(dict(prog=prog, input=big, method="model1", k=np.arange(len(v)), bytes=v * 0)))
        x = earlier.transfer(pd.concat(cur, ignore_index=True), sources=("model1",))
        for prog, g in x.groupby("prog"):
            t, base = M.truth(prog, ins[prog]), M.base_of(prog, ins[prog])
            rows.append(dict(prog=prog, sampler=s, rep=rep, **M.err(g.sort_values("k").bytes.to_numpy() + base, t)))
    d = pd.DataFrame(rows)
    d.to_csv(f"{DATA}/sample_transfer.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.pivot_table(index="prog", columns=["sampler", "rep"], values="signed").round(1).to_string())
    print(d.assign(a=d.signed.abs()).groupby(["sampler", "rep"])[["a", "abs"]].median().round(1).unstack())


def cost(sub="large", tag=""):
    """(d) Median wall time over the plain run's, per program and sampler, and the model's error at the largest
    input under each sampler (est1 = A - F + the runtime's pages, against the truth)."""
    ins = M.inputs()
    wr, er = [], []
    for prog in sorted(os.listdir(f"{OUT}/{sub}")):
        t, base = M.truth(prog, ins[prog]), M.base_of(prog, ins[prog])
        for rep in (1, 2, 3):
            o = f"{OUT}/{sub}/{prog}/{rep}"
            for s in ("plain",) + SAMPLERS:
                if os.path.exists(f"{o}/{s}.time"):
                    wr.append(dict(prog=prog, rep=rep, sampler=s, wall=float(open(f"{o}/{s}.time").read().split()[0])))
                if s != "plain" and os.path.exists(f"{o}/{s}.csv"):
                    r, hwm = load(f"{o}/{s}.csv")
                    er.append(dict(prog=prog, rep=rep, sampler=s, samples=len(r), rss_over_hwm=r.rss_bytes.max() / hwm,
                                   **M.err(data_curve(r) + base, t)))
    w = pd.DataFrame(wr).groupby(["prog", "sampler"]).wall.median().unstack()
    x = w[[s for s in SAMPLERS if s in w]].div(w["plain"], axis=0)
    x.to_csv(f"{DATA}/sample_cost{tag}.csv")
    e = pd.DataFrame(er)
    e.to_csv(f"{DATA}/sample_large{tag}.csv", index=False)
    print(x.round(2).to_string()); print("median", x.median().round(3).to_dict(), "max", x.max().round(2).to_dict())
    g = e.groupby(["prog", "sampler"]).median(numeric_only=True).reset_index()
    print(g.groupby("sampler")[["signed", "abs", "peak", "rss_over_hwm", "samples"]].median().round(2))


def cost_r8():
    cost("large_r8", "_r8")


def blind():
    """(c) The blind configurations under fast and faults, scored by blind/score.py's rule, against the published
    1 ms pss runs (blind/scores.csv, splits.csv)."""
    sys.path.insert(0, os.path.join(HERE, "..", "blind"))
    import score as B
    mm, ss = [], []
    pm, ps = B.score(B.measured())
    for s, (m, sp) in [("pss1", (pm, ps))] + [(s, B.score(B.measured(f"{OUT}/blind/{s}", "{rep}/" + s + ".csv")))
                                               for s in ("fast", "faults", "event") if os.path.isdir(f"{OUT}/blind/{s}")]:
        mm.append(m.assign(sampler=s)); ss.append(sp.assign(sampler=s))
    m, sp = pd.concat(mm), pd.concat(ss)
    m.to_csv(f"{DATA}/sample_blind.csv", index=False)
    sp.to_csv(f"{DATA}/sample_blind_splits.csv", index=False)
    t = m[m.truth.notna()]
    print(t.groupby("sampler")[["err_a", "err_b"]].agg(lambda x: x.abs().median()).round(2))
    print(sp.pivot_table(index=["program", "split"], columns="sampler", values="err").round(1))


def table():
    """tables/sampling.tex from the CSVs of the other parts."""
    tr = pd.read_csv(f"{DATA}/sample_transfer.csv").groupby(["prog", "sampler"]).signed.median().unstack()
    m = pd.read_csv(f"{DATA}/sample_blind.csv"); sp = pd.read_csv(f"{DATA}/sample_blind_splits.csv")
    c = pd.read_csv(f"{DATA}/sample_cost.csv")
    pk = pd.read_csv(f"{DATA}/sample_peaks.csv")
    sgn = lambda v: "--" if pd.isna(v) else ("$-$" if v < 0 else "$+$") + f"{abs(v):.0f}"
    cols = list(SAMPLERS[:4])   # the paper's table; R8's event sampler is reported in the dissertation
    out = [r"\multicolumn{5}{l}{\emph{Transfer to the largest input, signed whole-run error (\%), median of 3 runs}} \\"]
    for prog, name in (("npb-mg", "NPB-MG"), ("2mm", "2mm"), ("gemm", "gemm"), ("jacobi-2d", "jacobi-2d"), ("atax", "atax")):
        out.append(name + " & " + " & ".join(sgn(tr.loc[prog, s]) for s in cols) + r" \\")
    out.append(r"Median $|$error$|$, 23 programs & " + " & ".join(f"{tr[s].abs().median():.1f}" for s in cols) + r" \\")
    out.append(r"\addlinespace")
    out.append(r"\multicolumn{5}{l}{\emph{Blind configurations, error (\%) of the extrapolation to the largest}} \\")
    for prog in ("MG", "IS", "CG"):
        out.append(f"NPB {prog} & -- & " + " & ".join(sgn(sp[(sp.program == prog) & (sp.split == "EXTRA") & (sp.sampler == s)].err.iloc[0])
                                                for s in cols[1:]) + r" \\")
    b = m[m.truth.notna()].groupby("sampler").err_b.apply(lambda x: x.abs().median())
    out.append(r"Median $|$error$|$, all configurations & -- & " + " & ".join(f"{b[s]:.1f}" for s in cols[1:]) + r" \\")
    out.append(r"\addlinespace")
    out.append(r"\multicolumn{5}{l}{\emph{Cost and samples}} \\")
    r = c.set_index("prog")[cols].agg(["median", "max"]).T
    out.append(r"Run time over the program alone ($\times$), median & " + " & ".join(f"{r.loc[s, 'median']:.2f}" for s in cols) + r" \\")
    out.append(r"\quad largest & " + " & ".join(f"{r.loc[s, 'max']:.2f}" for s in cols) + r" \\")
    n = pk.groupby("sampler").samples.median()
    out.append(r"Samples per smaller-input run, median & " + " & ".join(f"{n[s]:,.0f}".replace(",", "{,}") for s in cols) + r" \\")
    open(os.path.join(HERE, "..", "tables", "sampling.tex"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


def thp():
    """Huge pages: per program and allocator, the median over 3 runs of the peak anonymous RSS with THP on over off
    (the shim's files and nothp runs), and of the plain run's RSS peak (base2 / basenothp); and the test program."""
    rows = []
    sets = [("glibc", f"{M.O}/runs_all", "files", "nothp", "base2", "basenothp", list(M.inputs()))]
    for al in ("jemalloc", "tcmalloc"):
        sets.append((al, f"{M.D}/alloc/{al}/runs", "pss", "pss", "base2", "base2", None))
    for name, root, m_on, m_off, b_on, b_off, progs in sets:
        root_off = root if name == "glibc" else root + "_nothp"
        for prog in progs or sorted(os.listdir(root)):
            v = []
            for rep in (1, 2, 3):
                try:
                    on = pd.read_csv(f"{root}/{prog}/{rep}/{m_on}.csv"); off = pd.read_csv(f"{root_off}/{prog}/{rep}/{m_off}.csv")
                    bon = pd.read_csv(f"{root}/{prog}/{rep}/{b_on}.csv"); boff = pd.read_csv(f"{root_off}/{prog}/{rep}/{b_off}.csv")
                except FileNotFoundError:
                    continue
                v.append((on.anon_bytes.max() / off.anon_bytes.max(), bon.rss_bytes.max() / boff.rss_bytes.max(),
                          (on.anon_bytes.max() - off.anon_bytes.max()) / 2 ** 20))
            if v:
                a = np.median(np.array(v), axis=0)
                rows.append(dict(alloc=name, prog=prog, anon_on_over_off=a[0], rss_on_over_off=a[1], anon_added_mib=a[2]))
    for prog in ("llama2c", "llamacpp"):
        for inp in sorted(os.listdir(f"{M.D}/llm/runs/{prog}")):
            v = []
            for rep in (1, 2, 3):
                try:
                    on = pd.read_csv(f"{M.D}/llm/runs/{prog}/{inp}/{rep}/files.csv")
                    off = pd.read_csv(f"{M.D}/llm/runs_nothp/{prog}/{inp}/{rep}/files.csv")
                    bon = pd.read_csv(f"{M.D}/llm/runs/{prog}/{inp}/{rep}/base2.csv")
                    boff = pd.read_csv(f"{M.D}/llm/runs_nothp/{prog}/{inp}/{rep}/base2.csv")
                except FileNotFoundError:
                    continue
                v.append((on.anon_bytes.max() / off.anon_bytes.max(), bon.rss_bytes.max() / boff.rss_bytes.max(),
                          (on.anon_bytes.max() - off.anon_bytes.max()) / 2 ** 20))
            if v:
                a = np.median(np.array(v), axis=0)
                rows.append(dict(alloc="glibc", prog=f"{prog}-{inp}", anon_on_over_off=a[0], rss_on_over_off=a[1],
                                 anon_added_mib=a[2]))
    d = pd.DataFrame(rows)
    d.to_csv(f"{DATA}/sample_thp.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.round(3).to_string(index=False))
    print(d.groupby("alloc")[["anon_on_over_off", "rss_on_over_off", "anon_added_mib"]].agg(["median", "max"]).round(3))
    if os.path.exists(f"{OUT}/thp.txt"):
        print(open(f"{OUT}/thp.txt").read())


def py():
    """CPython under PYTHONMALLOC=malloc: the data estimate against the default truth and the truth traced under
    PYTHONMALLOC=malloc; the run's peak RSS and wall time against the default runs (runs_all)."""
    inp = M.inputs()["python"]
    t0 = M.truth("python", inp)
    base = M.base_of("python", inp)
    pt = f"{OUT}/pages/python/{inp}/truthpages"
    t1 = M.truth("python", inp, path=pt, base=base) if os.path.isdir(pt) else None
    rows = []
    for rep in (1, 2, 3):
        r, _ = load(f"{OUT}/py/{inp}/{rep}/pss5.csv")
        d, _ = load(f"{M.O}/runs_all/python/{rep}/pss.csv")
        for name, run in (("malloc", r), ("pymalloc", d)):
            e = data_curve(run) + base
            row = dict(rep=rep, run=name, free_peak_mb=run.free_bytes.max() / 1e6, rss_peak_mb=run.rss_bytes.max() / 1e6,
                       **{f"vs_default_{k}": v for k, v in M.err(e, t0).items()})
            if t1 is not None:
                row.update({f"vs_malloc_{k}": v for k, v in M.err(e, t1).items()})
            rows.append(row)
    d = pd.DataFrame(rows)
    wall = lambda p: float(open(p).read().split()[0])
    d.to_csv(f"{DATA}/sample_py.csv", index=False)
    print(d.groupby("run").median(numeric_only=True).round(2).T.to_string())
    if t1 is not None:
        print("truth peak MB default %.1f malloc %.1f; whole-run %.3g vs %.3g" % (t0.max() / 1e6, t1.max() / 1e6, t1.sum() / t0.sum(), 1))
    print("wall malloc", np.median([wall(f"{OUT}/py/{inp}/{r}/pss5.time") for r in (1, 2, 3)]))


def cc1():
    """cc1: GCC's own mappings (the GC heap) as a share of anonymous RSS at its peak, and -fmem-report's GC totals."""
    rows = []
    for rep in (1, 2, 3):
        r, _ = load(f"{OUT}/cc1/{rep}/mmap.csv")
        k = int(np.argmax(r.anon_bytes))
        rows.append(dict(rep=rep, anon_peak_mb=r.anon_bytes.iloc[k] / 1e6, mapped_at_peak_mb=r.mapped_bytes.iloc[k] / 1e6,
                         free_at_peak_mb=r.free_bytes.iloc[k] / 1e6, mapped_max_mb=r.mapped_bytes.max() / 1e6))
    d = pd.DataFrame(rows)
    d.to_csv(f"{DATA}/sample_cc1.csv", index=False)
    print(d.round(1).to_string(index=False))
    rep = open(f"{OUT}/cc1/memreport.txt").read()
    for l in rep.splitlines():
        if re.search(r"^Total|GGC|ggc|^String|Number of", l):
            print(l)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    for part in sys.argv[1:] or ["peaks", "transfer", "cost", "blind", "thp", "py", "cc1", "table"]:
        print("==", part)
        globals()[part]()
