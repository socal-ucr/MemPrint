"""Tables of the RSS footprint model paper, from the runs of run_model.sh and run_options.sh.

    PYTHONPATH=analysis python paper_rss/compute.py   ->  paper_rss/tables/*.tex, paper_rss/data/*.csv

Truth: the pages a program touches (Pin, -footprint pages, -track_frees) plus the runtime's own pages (the
smallest RSS of an earlier native run), on the native wall clock (truth()), on 100 steps of progress. Every estimate is resampled to the same steps
by its own run's wall clock. Errors are over the whole run, as a share of the truth's area (sum over the steps):
signed (sum e - sum t) and absolute (sum |e - t|); and the peak's (max e / max t - 1)."""
import os

import numpy as np
import pandas as pd

from memprint.sched import corpus as C
from memprint.sched.run import largest

D = os.path.expanduser("~/memprint-data/2026-10-26-rsscurve")
O = f"{D}/opts"
PAGES = os.path.expanduser("~/memprint-data/2026-10-12-resize/corpus-pages")
HERE = os.path.dirname(os.path.abspath(__file__))
HEAVY = ["miniVite", "python", "lua", "sqlite", "perl", "quicksilver", "cc1"]
MODELS = {"rss": "RSS", "est1": r"Data $\hat f_1$", "est2": r"Charged $\hat f_2$", "memprint": r"\memprint{}"}


def inputs():
    return {p: sorted(os.listdir(f"{PAGES}/{p}"))[0] for p in sorted(os.listdir(PAGES))
            if os.path.isdir(f"{PAGES}/{p}") and os.listdir(f"{PAGES}/{p}")}


def base_of(prog, inp):
    r = pd.read_csv(f"{D}/corpus/{prog}/{inp}/rssrep1/rss.csv")
    return r[r.rss_bytes > 0].rss_bytes.min()


CLOCK = "native"   # the truth's time axis: "native" (wall clock, through the native run's references) or "refs"


def truth(prog, inp, path=None, clock=None, base=None):
    """The pages-touched truth (at `path`, else the corpus's) plus the runtime's pages, on the native wall clock:
    a share f of Pin's memory references is placed at the time the native run had executed f of its references
    (corpus.native_clock, as MemPrint's W5nat)."""
    got = C.truth(path or f"{PAGES}/{prog}/{inp}/truthpages")
    if (clock or CLOCK) == "native":
        got = C.on_native_clock(*got, C.native_clock(f"{D}/corpus/{prog}/{inp}/nativerefs"))
    return C.resample(*got) + (base_of(prog, inp) if base is None else base)


def err(e, t):
    return dict(signed=100 * (e.sum() - t.sum()) / t.sum(), abs=100 * np.abs(e - t).sum() / t.sum(),
                peak=100 * (e.max() / t.max() - 1))


def walls(path):
    w = pd.read_csv(path, sep=" ", header=None, names=["prog", "rep", "m", "exit", "wall", "kb"])
    assert (w.exit == 0).all(), w[w.exit != 0]
    return w.groupby(["prog", "m"]).wall.median()


def native_walls():
    """Wall time of each (program, input, method) run of the corpus, the latest run of each."""
    f = [pd.read_csv(f"{D}/corpus/{n}", sep=r"\s+", header=None, dtype={1: str}, usecols=[0, 1, 2, 3, 4])
         for n in ("done.base.txt", "done.txt")]
    w = pd.concat(f).set_axis(["prog", "input", "method", "exit", "wall"], axis=1)
    return w[w.exit == 0].drop_duplicates(["prog", "input", "method"], keep="last").set_index(
        ["prog", "input", "method"]).wall


def model():
    """Per program and repetition: the errors of plain RSS, estimates 1 and 2 and MemPrint (W5 on the native
    clock), and the components of the model's run as shares of the truth's area."""
    cur = largest(pd.read_csv(f"{D}/study/curves.csv", dtype={"input": str}))
    rows, comp = [], []
    for prog, inp in inputs().items():
        t, base = truth(prog, inp), base_of(prog, inp)
        w5 = cur[(cur.prog == prog) & (cur.input == inp) & (cur.method == "W5nat")].sort_values("k").bytes.to_numpy()
        for rep in (1, 2, 3):
            o = f"{O}/runs_all/{prog}/{rep}"
            b = pd.read_csv(f"{o}/base2.csv"); b = b[b.rss_bytes > 0]
            r = pd.read_csv(f"{o}/pss.csv"); r = r[r.rss_bytes > 0]
            R = lambda v: C.resample(r.seconds, np.clip(v, 0, None))
            est = {"rss": C.resample(b.seconds, b.rss_bytes), "est1": R(r.anon_bytes - r.free_bytes + base),
                   "est2": R(r.pss_bytes - r.free_bytes)}
            if len(w5) and rep == 1:
                est["memprint"] = w5 + base
            for m, e in est.items():
                rows.append(dict(prog=prog, input=inp, rep=rep, model=m, **err(e, t)))
            comp.append(dict(prog=prog, rep=rep, **{k: 100 * R(v).sum() / t.sum() for k, v in (
                ("data", r.anon_bytes - r.free_bytes + base), ("free", r.free_bytes),
                ("file_rss", r.rss_bytes - r.anon_bytes), ("file_pss", r.pss_bytes - r.anon_bytes))}))
    d = pd.DataFrame(rows).groupby(["prog", "input", "model"])[["signed", "abs", "peak"]].median().reset_index()
    comp = pd.DataFrame(comp).groupby("prog").median().drop(columns="rep")
    # cost: the model's run over the plain RSS run (both here); MemPrint's W5 + nativerefs over the earlier RSS run
    w, nw = walls(f"{O}/done_all.txt"), native_walls()
    cost = {}
    for prog, inp in inputs().items():
        cost[(prog, "est1")] = cost[(prog, "est2")] = w[(prog, "pss")] / w[(prog, "base")]
        cost[(prog, "rss")] = 1.0
        if (prog, inp, "W5") in nw and (prog, inp, "nativerefs") in nw:
            cost[(prog, "memprint")] = (nw[(prog, inp, "W5")] + nw[(prog, inp, "nativerefs")]) / nw[(prog, inp, "rssrep1")]
    d["x"] = [cost.get((p, m), np.nan) for p, m in zip(d.prog, d.model)]
    d["group"] = np.where(d.prog.isin(HEAVY), "heavy", "other")
    return d, comp


def options():
    """The alternatives, each alone, on the allocator-heavy programs (run_options.sh): medians of 3 runs."""
    INP = {"miniVite": "32768", "lua": "1", "python": "1", "perl": "1", "sqlite": "1", "cc1": "large",
           "quicksilver": "100000"}
    rows = []
    for prog, inp in INP.items():
        t, base = truth(prog, inp), base_of(prog, inp)
        for rep in (1, 2, 3):
            o = f"{O}/runs/{prog}/{rep}"
            est = {}
            for m in ("base", "opt2a", "opt2b"):
                r = pd.read_csv(f"{o}/{m}.csv"); r = r[r.rss_bytes > 0]; est[m] = C.resample(r.seconds, r.rss_bytes)
            r = pd.read_csv(f"{o}/opt1.csv"); est["opt1"] = C.resample(r.seconds, (r.rss_bytes - r.free_bytes).clip(lower=0))
            if os.path.exists(f"{o}/opt3.csv"):
                r = pd.read_csv(f"{o}/opt3.csv")
                v = r.rss_bytes - r.free_bytes if "free_bytes" in r else base + r.heap_bytes
                est["opt3"] = C.resample(r.seconds, v.clip(lower=0))
            r = pd.read_csv(f"{o}/opt4.csv"); r = r[r.rss_bytes > 0]
            est["opt4"] = C.resample(r.seconds, r.groupby("window").ref_bytes.transform("max"))
            for m, e in est.items():
                rows.append(dict(prog=prog, rep=rep, opt=m, **err(e, t)))
    d = pd.DataFrame(rows).groupby(["prog", "opt"])[["signed", "abs", "peak"]].median()
    w = walls(f"{O}/done.txt")
    d["x"] = w / w.xs("base", level="m").reindex(w.index, level="prog")
    return d.reset_index()


def transfer():
    """Each estimate predicted for the largest input from the smaller ones (grid.py build: earlier.transfer), plus
    the runtime's pages, against the truth. A program without a transferred curve has no estimate."""
    c = pd.read_csv(f"{D}/model/curves.csv", dtype={"input": str})
    X = {"rss": "rssrep1X", "est1": "model1X", "est2": "model2X", "memprint": "W5natX"}
    rows = []
    for prog, inp in inputs().items():
        g = c[(c.prog == prog) & (c.input == inp)]
        if c[(c.prog == prog) & (c.method == "model1")].input.nunique() < 2:
            continue   # no smaller input (darknet)
        t, base = truth(prog, inp), base_of(prog, inp)
        for m, method in X.items():
            e = g[g.method == method].sort_values("k").bytes.to_numpy()
            rows.append(dict(prog=prog, model=m, **(err(e + base, t) if len(e) else
                                                     dict(signed=np.nan, abs=np.nan, peak=np.nan))))
    d = pd.DataFrame(rows)
    d["group"] = np.where(d.prog.isin(HEAVY), "heavy", "other")
    return d


def runs_base(paths):
    """The runtime's pages from plain RSS runs: the median over the runs of each run's smallest RSS."""
    m = []
    for p in paths:
        r = pd.read_csv(p)
        m.append(r[r.rss_bytes > 0].rss_bytes.min())
    return float(np.median(m))


def allocators(sfx=""):
    """The model under jemalloc and TCMalloc (run_alloc.sh) against the pages-touched truth under the same allocator.
    'anon' is the data estimate without the allocator's statistics (free bytes taken as 0)."""
    INP = {"miniVite": "32768", "lua": "1", "python": "1", "perl": "1", "sqlite": "1", "cc1": "large",
           "quicksilver": "100000"}
    rows, w = [], walls_alloc(sfx)
    for al in ("jemalloc", "tcmalloc"):
        for prog, inp in INP.items():
            o = f"{D}/alloc/{al}/runs{sfx}/{prog}"
            base = runs_base([f"{o}/{rep}/base2.csv" for rep in (1, 2, 3)])
            t = truth(prog, inp, path=f"{D}/alloc/{al}/{prog}/{inp}/truthpages", base=base)
            for rep in (1, 2, 3):
                b = pd.read_csv(f"{o}/{rep}/base2.csv"); b = b[b.rss_bytes > 0]
                r = pd.read_csv(f"{o}/{rep}/pss.csv"); r = r[r.rss_bytes > 0]
                R = lambda v: C.resample(r.seconds, np.clip(v, 0, None))
                est = {"rss": C.resample(b.seconds, b.rss_bytes), "anon": R(r.anon_bytes + base),
                       "est1": R(r.anon_bytes - r.free_bytes + base), "est2": R(r.pss_bytes - r.free_bytes)}
                for m, e in est.items():
                    rows.append(dict(alloc=al, prog=prog, rep=rep, model=m, **err(e, t)))
    d = pd.DataFrame(rows).groupby(["alloc", "prog", "model"])[["signed", "abs", "peak"]].median().reset_index()
    d["x"] = [w[(a, p, "pss")] / w[(a, p, "base")] for a, p in zip(d.alloc, d.prog)]
    return d


def walls_alloc(sfx=""):
    w = pd.read_csv(f"{D}/alloc/done_alloc{sfx}.txt", sep=" ", header=None,
                    names=["alloc", "prog", "rep", "m", "exit", "wall", "kb"])
    assert (w.exit == 0).all(), w[w.exit != 0]
    return w.groupby(["alloc", "prog", "m"]).wall.median()


LLM = {"llama2c": ["15M", "42M", "110M"], "llamacpp": ["135m-n32", "135m-n128"]}


def llm(sfx=""):
    """The model on the LLM engines (run_llm.sh): the data estimate with and without mapped data files, the charged
    estimate, RSS and MemPrint (W5 on the native clock), against the pages-touched truth."""
    cur = pd.read_csv(f"{D}/model/curves.csv", dtype={"input": str})
    w = pd.read_csv(f"{D}/llm/done_runs{sfx}.txt", sep=" ", header=None, names=["prog", "input", "rep", "m", "exit", "wall", "kb"])
    assert (w.exit == 0).all(), w[w.exit != 0]
    w = w.groupby(["prog", "input", "m"]).wall.median()
    rows = []
    for prog, inps in LLM.items():
        for inp in inps:
            o = f"{D}/llm/runs{sfx}/{prog}/{inp}"
            base = runs_base([f"{o}/{rep}/base2.csv" for rep in (1, 2, 3)])
            t = truth(prog, inp, path=f"{D}/llm/{prog}/{inp}/truthpages", base=base)
            w5 = cur[(cur.prog == prog) & (cur.input == inp) & (cur.method == "W5nat")].sort_values("k").bytes.to_numpy()
            for rep in (1, 2, 3):
                b = pd.read_csv(f"{o}/{rep}/base2.csv"); b = b[b.rss_bytes > 0]
                r = pd.read_csv(f"{o}/{rep}/files.csv"); r = r[r.rss_bytes > 0]
                R = lambda v: C.resample(r.seconds, np.clip(v, 0, None))
                est = {"rss": C.resample(b.seconds, b.rss_bytes), "est1": R(r.anon_bytes - r.free_bytes + base),
                       "est1f": R(r.anon_bytes - r.free_bytes + r.datafile_bytes + base),
                       "est2": R(r.pss_bytes - r.free_bytes)}
                if len(w5) and rep == 1:
                    est["memprint"] = w5 + base
                for m, e in est.items():
                    rows.append(dict(prog=prog, input=inp, rep=rep, model=m, **err(e, t)))
            nw = native_walls()
            rows.append(dict(prog=prog, input=inp, model="_cost", x=w[(prog, inp, "files")] / w[(prog, inp, "base")],
                             x_memprint=(nw.get((prog, inp, "W5"), np.nan) + nw.get((prog, inp, "nativerefs"), np.nan))
                             / nw[(prog, inp, "rssrep1")]))
    d = pd.DataFrame(rows)
    cost = d[d.model == "_cost"].set_index(["prog", "input"])[["x", "x_memprint"]]
    d = d[d.model != "_cost"].groupby(["prog", "input", "model"])[["signed", "abs", "peak"]].median().reset_index()
    return d.join(cost, on=["prog", "input"])


def thp():
    """The corpus with transparent huge pages on (files.csv, base2.csv) and off (nothp.csv, basenothp.csv): RSS and
    the data estimate with mapped data files, against the truth."""
    rows = []
    for prog, inp in inputs().items():
        t, base = truth(prog, inp), base_of(prog, inp)
        for rep in (1, 2, 3):
            o = f"{O}/runs_all/{prog}/{rep}"
            for thp_on, b_name, m_name in ((True, "base2", "files"), (False, "basenothp", "nothp")):
                b = pd.read_csv(f"{o}/{b_name}.csv"); b = b[b.rss_bytes > 0]
                r = pd.read_csv(f"{o}/{m_name}.csv"); r = r[r.rss_bytes > 0]
                est = {"rss": C.resample(b.seconds, b.rss_bytes),
                       "est1": C.resample(r.seconds, np.clip(r.anon_bytes - r.free_bytes + r.datafile_bytes, 0, None) + base)}
                for m, e in est.items():
                    rows.append(dict(prog=prog, rep=rep, thp=thp_on, model=m, **err(e, t)))
    d = pd.DataFrame(rows).groupby(["prog", "thp", "model"])[["signed", "abs", "peak"]].median().reset_index()
    d["group"] = np.where(d.prog.isin(HEAVY), "heavy", "other")
    return d


def files_overhead():
    """The files mode (smaps instead of smaps_rollup) on the corpus (run_model.sh with RUNS=files): its run time
    over the plain RSS run's."""
    w = walls(f"{O}/done_all.txt")
    return (w.xs("files", level="m") / w.xs("base", level="m")).dropna()


GRID = [("exact", "Exact curve (truth)"), ("exact-peak", "Exact peak"), ("request", "User request"),
        ("ksplus", "KS+"), None,
        ("rss-peak-rep", "RSS peak"), ("rss-curve-rep", "RSS curve"), ("memprint-W5", r"\memprint{} W5"),
        ("memprint-W5nat", r"\memprint{} W5, native clock"), ("model1-rep", r"Data $\hat f_1$"), ("model2-rep", r"Charged $\hat f_2$"),
        None,
        ("rss-peak-x", "RSS peak"), ("rss-curve-x", "RSS curve"), ("memprint-W5-x", r"\memprint{} W5"),
        ("memprint-W5nat-x", r"\memprint{} W5, native clock"), ("model1-x", r"Data $\hat f_1$"), ("model2-x", r"Charged $\hat f_2$")]


def grid(scenario):
    """The scheduling grid (grid.py run) under one truth: per oracle, its calibrated run (q90 margin; 10% for the
    oracles without a profile; no margin for those that take none), means over the 3 seeds."""
    g = pd.read_csv(f"{D}/model/grid.csv")
    g = g[g.scenario == scenario]
    pick = g.assign(rank=g.margin.map({"cal": 0, "0.1": 1, "0.0": 2}).fillna(3))
    pick = pick[pick["rank"] == pick.groupby("oracle")["rank"].transform("min")]
    return pick.groupby(["oracle", "load"])[["response", "throughput", "kills", "wasted_core_h"]].mean()


def tab_grid(scenario):
    m = grid(scenario)
    lines = []
    for row in GRID:
        if row is None:
            lines.append(r"\midrule")
            continue
        o, label = row
        if o not in m.index.get_level_values(0):
            lines.append(f"{label} & \\multicolumn{{6}}{{c}}{{(no curve for every program)}}\\\\")
            continue
        r = m.loc[o]
        lines.append(f"{label} & {r.response[0.6]:.0f} & {r.response[0.8]:.0f} & {r.response[1.0]:.0f} & "
                     f"{r.throughput[0.8]:.1f} & {r.kills[0.8]:.0f} & {r.wasted_core_h[0.8]:.0f}\\\\")
    return "\n".join(lines)


# --- LaTeX

def f(v, sign=True, nd=0):
    if pd.isna(v):
        return "--"
    v = round(v, nd) + 0.0   # no "-0"
    if v == 0:
        return "0"
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "$-$")


def tab_model(d):
    lines = []
    for grp, name in (("heavy", "Allocator-heavy"), ("other", "Others")):
        g = d[d.group == grp]
        lines.append(rf"\multicolumn{{11}}{{l}}{{\emph{{{name}}}}}\\")
        for prog, h in sorted(g.groupby("prog"), key=lambda x: (HEAVY.index(x[0]) if x[0] in HEAVY else 0, x[0])):
            h = h.set_index("model")
            cells = []
            for m in MODELS:
                cells += [f(h.signed.get(m)), f(h["abs"].get(m), sign=False)] if m in h.index else ["--", "--"]
            lines.append(f"{prog} & " + " & ".join(cells) + f" & {f(h.x.get('est1'), False, 2)} & "
                         f"{f(h.x.get('memprint'), False, 1)}\\\\")
        lines.append(r"\addlinespace")
    return "\n".join(lines[:-1])


def tab_summary(d):
    lines = []
    for grp, name in (("heavy", "Allocator-heavy (7)"), ("other", "Others (16)"), (None, "All (23)")):
        g = d if grp is None else d[d.group == grp]
        for m, label in MODELS.items():
            h = g[g.model == m]
            lines.append(f"{name if m == 'rss' else ''} & {label} & {len(h)} & {f(h.signed.median())} & "
                         f"{f(h['abs'].median(), False)} & {f(h['abs'].quantile(0.9), False)} & {f(h.peak.median())} & "
                         f"{f(h.peak.min())} & {f(h.x.median(), False, 1 if m == 'memprint' else 2)}\\\\")
        lines.append(r"\addlinespace")
    return "\n".join(lines[:-1])


def tab_transfer(x):
    lines = []
    for grp, name in (("heavy", "Allocator-heavy"), ("other", "Others")):
        lines.append(rf"\multicolumn{{9}}{{l}}{{\emph{{{name}}}}}\\")
        g = x[x.group == grp]
        for prog, h in sorted(g.groupby("prog"), key=lambda p: (HEAVY.index(p[0]) if p[0] in HEAVY else 0, p[0])):
            h = h.set_index("model")
            lines.append(f"{prog} & " + " & ".join(f"{f(h.signed[m])} & {f(h['abs'][m], sign=False)}" for m in MODELS)
                         + r"\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\midrule"
    for grp, name in (("heavy", "median, allocator-heavy"), ("other", "median, others")):
        g = x[x.group == grp].dropna()
        lines.append(f"{name} & " + " & ".join(
            f"{f(g[g.model == m].signed.median())} & {f(g[g.model == m]['abs'].median(), sign=False)}" for m in MODELS)
            + r"\\")
    return "\n".join(lines)


def tab_alloc(a, an):
    """Signed whole-run error per allocator and program, THP on and off, and the medians."""
    lines = []
    for al, name in (("jemalloc", "jemalloc"), ("tcmalloc", "TCMalloc")):
        lines.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{name}}}}}\\")
        for prog in HEAVY + ["median"]:
            if prog == "median":
                g, h = a[a.alloc == al].groupby("model").signed.median(), an[an.alloc == al].groupby("model").signed.median()
                lines.append(r"\addlinespace")
            else:
                g = a[(a.alloc == al) & (a.prog == prog)].set_index("model").signed
                h = an[(an.alloc == al) & (an.prog == prog)].set_index("model").signed
            lines.append(f"{prog} & {f(g['rss'])} & {f(g['anon'])} & {f(g['est1'])} & {f(g['est2'])} & "
                         f"{f(h['rss'])} & {f(h['est1'])} & {f(h['est2'])}\\\\")
        lines.append(r"\midrule")
    lines[-1] = ("overhead ($\\times$) & " + " & ".join(["1.00"] + [f(a.x.median(), False, 2)] * 3 + ["1.00"] +
                                                    [f(an.x.median(), False, 2)] * 2) + r"\\")
    lines.insert(-1, r"\midrule")
    return "\n".join(lines)


def tab_llm(l, ln):
    lines = []
    for (prog, inp), g in l.groupby(["prog", "input"], sort=False):
        g = g.set_index("model")
        h = ln[(ln.prog == prog) & (ln.input == inp)].set_index("model")
        lines.append(f"{prog} {inp} & " + " & ".join(f(g.signed.get(m)) for m in ("rss", "est1", "est1f", "est2", "memprint"))
                     + f" & {f(h.signed['est1f'])} & {f(g.x.iloc[0], False, 2)} & {f(g.x_memprint.iloc[0], False, 1)}\\\\")
    return "\n".join(lines)


def tab_thp(t):
    lines = []
    for grp, name in (("heavy", "Allocator-heavy (7)"), ("other", "Others (16)")):
        g = t[t.group == grp].groupby(["model", "thp"])
        lines.append(f"{name} & " + " & ".join(f"{f(g.signed.median()[(m, on)])} & {f(g['abs'].median()[(m, on)], sign=False)}"
                                              for m in ("rss", "est1") for on in (True, False)) + r"\\")
    return "\n".join(lines)


def tab_options(o):
    OPTS = [("base", "RSS"), ("opt1", "RSS$-$free"), ("opt2a", "trim 5\\,ms"), ("opt2b", "trim thr.\\ 0"),
            ("opt3", "runtime stats"), ("opt4", "referenced")]
    lines = []
    for prog in HEAVY:
        h = o[o.prog == prog].set_index("opt")
        lines.append(f"{prog} & " + " & ".join(f(h.signed.get(k)) if k in h.index else "--" for k, _ in OPTS) + r"\\")
    lines.append(r"\midrule")
    for what, fn in (("median signed", lambda h: h.signed.median()), ("median absolute", lambda h: h["abs"].median())):
        lines.append(f"{what} & " + " & ".join(f(fn(o[o.opt == k]), sign=what.endswith("signed")) for k, _ in OPTS) + r"\\")
    lines.append("overhead ($\\times$) & " + " & ".join(f(o[o.opt == k].x.median(), False, 2) for k, _ in OPTS) + r"\\")
    return "\n".join(lines)


def tab_components(comp):
    lines = []
    for prog in HEAVY:
        c = comp.loc[prog]
        lines.append(f"{prog} & {c.data:.0f} & {c.free:.0f} & {c.file_rss:.0f} & {c.file_pss:.0f}\\\\")
    rest = comp.drop(index=HEAVY).median()
    lines.append(r"\midrule")
    lines.append(f"others (median of 16) & {rest.data:.0f} & {rest.free:.0f} & {rest.file_rss:.0f} & {rest.file_pss:.0f}\\\\")
    return "\n".join(lines)


if __name__ == "__main__":
    os.makedirs(f"{HERE}/tables", exist_ok=True); os.makedirs(f"{HERE}/data", exist_ok=True)
    d, comp = model()
    o = options()
    x = transfer()
    x.to_csv(f"{HERE}/data/transfer.csv", index=False)
    d.to_csv(f"{HERE}/data/model.csv", index=False); comp.to_csv(f"{HERE}/data/components.csv")
    o.to_csv(f"{HERE}/data/options.csv", index=False)
    for name, s in (("model", tab_model(d)), ("summary", tab_summary(d)), ("options", tab_options(o)),
                    ("components", tab_components(comp)), ("transfer", tab_transfer(x))):
        open(f"{HERE}/tables/{name}.tex", "w").write(s + "\n")
    a, an = allocators(), allocators("_nothp")
    l, ln = llm(), llm("_nothp")
    t = thp()
    for name, df in (("alloc", a), ("alloc_nothp", an), ("llm", l), ("llm_nothp", ln), ("thp", t)):
        df.to_csv(f"{HERE}/data/{name}.csv", index=False)
    for name, tx in (("alloc", tab_alloc(a, an)), ("llm", tab_llm(l, ln)), ("thp", tab_thp(t))):
        open(f"{HERE}/tables/{name}.tex", "w").write(tx + "\n")
    fo = files_overhead()
    print("files mode overhead: median", round(fo.median(), 2), "range", round(fo.min(), 2), round(fo.max(), 2))
    if os.path.exists(f"{D}/model/grid.csv"):
        for scenario in ("pagesnat", "rss"):
            open(f"{HERE}/tables/grid_{scenario}.tex", "w").write(tab_grid(scenario) + "\n")
    pd.set_option("display.width", 250)
    print(d.pivot(index="prog", columns="model", values=["signed", "abs", "peak"]).round(0).to_string())
    print(d.groupby(["group", "model"])[["signed", "abs", "peak", "x"]].median().round(2).to_string())
    print(comp.round(0).to_string()); print(o.round(1).to_string())
    print(x.pivot(index="prog", columns="model", values=["signed", "abs", "peak"]).round(0).to_string())
    print(x.groupby(["group", "model"])[["signed", "abs", "peak"]].median().round(1).to_string())
