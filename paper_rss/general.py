"""Tables of the generalisation section: B from the job's own run (inrun_base.py), paper_generalization's blind
configurations (blind/score.py), borrowing the growth of known programs (borrow.py) and code similarity (similar.py).

    ~/environments/memfp/bin/python paper_rss/general.py   ->  paper_rss/tables/{inrun,blind,blind_splits,borrow,similar}.tex
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


def similar():
    """tables/similar.tex: per space, the Mantel rho of code distance and each term's difference (in parentheses when
    p >= 0.05), the median LOPO error of R' from the nearest program (MB), and the median, 90th percentile and count
    within 10% of the absolute signed whole-run error with beta from the nearest program (as tables/borrow.tex)."""
    mt = pd.read_csv(f"{HERE}/data/similar_mantel.csv").set_index(["space", "term"])
    lo = pd.read_csv(f"{HERE}/data/similar_lopo.csv")
    lo["a"] = lo.whole_signed.abs()
    rho = lambda sp, t: "" if (sp, t) not in mt.index else (
        lambda r, p: (f"{r:.2f}" if p < 0.05 else f"({r:.2f})").replace("-", "$-$"))(*mt.loc[(sp, t), ["rho", "p"]])
    def row(lab, sp, m):
        cells = [lab] + [rho(sp, t) for t in ("B", "r", "Rp", "beta")]
        rp = lo[(lo.term == "Rp") & (lo.method == m)].err
        cells.append(f"{rp.median():.2f}" if len(rp) else "")
        b = lo[(lo.term == "beta") & (lo.method == m)].a
        cells += [f"{b.median():.1f}", f"{b.quantile(0.9):.0f}", f"{(b <= 10).sum()}"] if len(b) else ["--"] * 3
        return " & ".join(cells) + r" \\"
    base = lo[(lo.term == "beta") & (lo.method == "proportional")].a
    rp = lo[(lo.term == "Rp") & (lo.method == "median")].err
    lines = [" & ".join([r"Median of the others; $\beta = 1$"] + ["--"] * 4 + [f"{rp.median():.2f}",
             f"{base.median():.1f}", f"{base.quantile(0.9):.0f}", f"{(base <= 10).sum()}"]) + r" \\"]
    lines.append(r"\midrule")
    spaces = [("static", "Static features, all"), ("static_src", "\\quad source only"),
              ("static_bin", "\\quad executable only"), ("descriptor", "Run descriptor (\\tabref{tab:borrow})"),
              ("gte", "gte-modernbert (149M)"), ("bgecode", "bge-code-v1 (1.5B)"), ("qodo", "Qodo-Embed-1 (1.5B)"),
              ("qwen3", "Qwen3-Embedding (8B)")]
    for sp, lab in spaces:
        if (lo.method == f"{sp}:nearest").any():
            lines.append(row(lab, sp, f"{sp}:nearest"))
    lines.append(r"\midrule")
    for sp, lab in spaces[4:]:
        if (lo.method == f"{sp}_mem:nearest").any():
            lines.append(row(lab + ", memory code", f"{sp}_mem", f"{sp}_mem:nearest"))
    b = lo[(lo.term == "beta") & (lo.method == "own_beta")].a
    lines += [r"\midrule", " & ".join([r"The program's own $\beta$"] + [""] * 5 +
                                        [f"{b.median():.1f}", f"{b.quantile(0.9):.0f}", f"{(b <= 10).sum()}"]) + r" \\"]
    open(f"{T}/similar.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    inrun()
    blind()
    borrow()
    similar()
