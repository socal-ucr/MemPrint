"""Tables of the generalisation section: B from the job's own run (inrun_base.py), paper_generalization's blind
configurations (blind/score.py) and borrowing the growth of known programs (borrow.py).

    ~/environments/memfp/bin/python paper_rss/general.py   ->  paper_rss/tables/{inrun,blind,blind_splits,borrow}.tex
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
T = f"{HERE}/tables"
NAMES = {"hpccg": "HPCCG", "lulesh": "LULESH", "IS": "NPB IS", "CG": "NPB CG", "MG": "NPB MG"}


def s(v, nd=1):
    if pd.isna(v):
        return "--"
    t = f"{v:+.{nd}f}"
    return t.replace("-", "$-$")


def inrun():
    d = pd.read_csv(f"{HERE}/data/inrun_base.csv")
    lines = []
    for grp, lab in (("heavy", "Allocator-heavy (7)"), ("other", "Others (16)"), (None, "All (23)")):
        g = d if grp is None else d[d.group == grp]
        cells = [lab]
        for b in ("earlier", "inrun"):
            h = g[g.B == b]
            cells += [s(h.signed.median()), f"{h['abs'].median():.1f}", s(h.peak.median())]
        lines.append(" & ".join(cells) + r" \\")
    mv = d[d.prog == "miniVite"].set_index("B")
    lines.append(r"\midrule")
    lines.append(" & ".join(["miniVite"] + sum([[s(mv.loc[b, "signed"]), f"{mv.loc[b, 'abs']:.1f}",
                                                  s(mv.loc[b, "peak"])] for b in ("earlier", "inrun")], [])) + r" \\")
    open(f"{T}/inrun.tex", "w").write("\n".join(lines) + "\n")


def blind():
    m = pd.read_csv(f"{HERE}/blind/scores.csv", dtype={"config": str})
    m = m[m.truth.notna()].copy()
    m["order"] = m["size"]
    lines = []
    for i, prog in enumerate(NAMES):
        g = m[m.program == prog].sort_values("order")
        if i:
            lines.append(r"\addlinespace")
        for j, (_, r) in enumerate(g.iterrows()):
            lines.append(" & ".join([NAMES[prog] if j == 0 else "", r.config, f"{r.truth / 1e6:.2f}",
                                     f"{r.measured / 1e6:.2f}", s(r.err_a), s(r.err_b), s(r.static_error)]) + r" \\")
    open(f"{T}/blind.tex", "w").write("\n".join(lines) + "\n")
    sp = pd.read_csv(f"{HERE}/blind/splits.csv", dtype={"config": str})
    lines = [" & ".join([NAMES[r.program], r.split, r.config, s(r.err), s(r.err_posthoc), s(r.measured_err),
                         s(r.static_error)]) + r" \\" for _, r in sp.iterrows()]
    open(f"{T}/blind_splits.tex", "w").write("\n".join(lines) + "\n")
    print("median |err| a, b, static:", m.err_a.abs().median().round(2), m.err_b.abs().median().round(2),
          m.static_error.abs().median().round(2))


def borrow():
    out = []
    for f in ("borrow", "borrow_middle"):
        r = pd.read_csv(f"{HERE}/data/{f}.csv")
        r["a"] = r.signed.abs()
        rk = pd.read_csv(f"{HERE}/data/borrow_rank{'_middle' if 'middle' in f else ''}.csv")
        out.append((r, rk))
    names = [("own", "Own transfer (two smaller inputs)"), ("proportional", r"Proportional ($\beta = 1$)"),
             ("uniform", r"Uniform (mean $\beta$)"), ("nearest", r"Nearest program's $\beta$"),
             ("rbf", r"RBF mixture of $\beta$"), ("oracle", r"Best known $\beta$ (oracle)"),
             ("beta_C", r"The program's own $\beta$ (all its inputs)")]
    lines = []
    for m, lab in names:
        cells = [lab]
        for r, _ in out:
            h = r[r.method == m]
            cells += [f"{h.a.median():.1f}", f"{h.a.quantile(0.9):.0f}", f"{(h.a <= 10).sum()}"]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(" & ".join(["Spearman $\\rho$ of distance and error, median"] +
                            sum([[f"{rk.rho.median():.2f}".replace("-", "$-$"), "", ""] for _, rk in out], [])) + r" \\")
    open(f"{T}/borrow.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    inrun()
    blind()
    borrow()
