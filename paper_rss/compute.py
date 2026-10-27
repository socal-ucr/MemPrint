"""Tables of the RSS footprint model paper, from the runs of run_model.sh and run_options.sh.

    PYTHONPATH=analysis python paper_rss/compute.py   ->  paper_rss/tables/*.tex, paper_rss/data/*.csv

Truth: the pages a program touches (Pin, -footprint pages, -track_frees) plus the runtime's own pages (the
smallest RSS of an earlier native run), on 100 steps of progress. Every estimate is resampled to the same steps
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
MODELS = {"rss": "RSS", "est1": "Data (1)", "est2": "Shared (2)", "memprint": r"\memprint{}"}


def inputs():
    return {p: sorted(os.listdir(f"{PAGES}/{p}"))[0] for p in sorted(os.listdir(PAGES))
            if os.path.isdir(f"{PAGES}/{p}") and os.listdir(f"{PAGES}/{p}")}


def base_of(prog, inp):
    r = pd.read_csv(f"{D}/corpus/{prog}/{inp}/rssrep1/rss.csv")
    return r[r.rss_bytes > 0].rss_bytes.min()


def truth(prog, inp):
    return C.resample(*C.truth(f"{PAGES}/{prog}/{inp}/truthpages")) + base_of(prog, inp)


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
    d.to_csv(f"{HERE}/data/model.csv", index=False); comp.to_csv(f"{HERE}/data/components.csv")
    o.to_csv(f"{HERE}/data/options.csv", index=False)
    for name, s in (("model", tab_model(d)), ("summary", tab_summary(d)), ("options", tab_options(o)),
                    ("components", tab_components(comp))):
        open(f"{HERE}/tables/{name}.tex", "w").write(s + "\n")
    pd.set_option("display.width", 250)
    print(d.pivot(index="prog", columns="model", values=["signed", "abs", "peak"]).round(0).to_string())
    print(d.groupby(["group", "model"])[["signed", "abs", "peak", "x"]].median().round(2).to_string())
    print(comp.round(0).to_string()); print(o.round(1).to_string())
