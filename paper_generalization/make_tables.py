"""LaTeX tables of the generalisation paper, generated from the evaluation outputs so that no number
is copied by hand.

python paper_generalization/make_tables.py   (writes paper_generalization/tables/*.tex)
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
D = HERE / "data"
T = HERE / "tables"
PB = ROOT / "data" / "polybench-bytes" / "data"
GAP = ROOT / "data" / "gap-bytes" / "data"
CSR = ROOT / "data" / "csr-bytes" / "data"

METHODS = [
    ("static-fp", "Static footprint (source + runtime baseline, no sampling)", "source"),
    ("static-alpha", "Static $\\alpha$ (moments of the static spectrum)", "source"),
    ("pooled-static", "Pooled regression with $\\log$ static $\\alpha$ as a feature", "source + known traces"),
    ("nn-static", "Nearest known model by $\\hat z$", "source + known models"),
    ("mix-static", "RBF mixture of known models by $\\hat z$", "source + known models"),
    ("nn-measured", "Nearest known model by measured $z$", "$C$'s traces (upper bound)"),
    ("mix-measured", "RBF mixture by measured $z$", "$C$'s traces (upper bound)"),
    ("nn-ast", "Nearest known model by AST node counts", "source + known models"),
    ("mean", "Uniform mixture of known models", "known models"),
    ("own", "$C$'s own model, trained on its other inputs", "$C$'s traces"),
    ("oracle", "Best known model, chosen after scoring", "$C$'s traces (oracle)"),
]


def write(name, text):
    T.mkdir(parents=True, exist_ok=True)
    (T / f"{name}.tex").write_text(text)
    print(name)


def f(x, d=2):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def lowo_table():
    e = pd.read_csv(PB / "lowo_errors.csv")
    g = e.groupby(["method", "split"]).mape.agg(["median", "mean"]).unstack("split")
    lines = ["\\begin{tabular}{llrrrr}", "\\toprule",
             "Method & Uses & \\multicolumn{2}{c}{EXTRA} & \\multicolumn{2}{c}{INTER} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}", " & & median & mean & median & mean \\\\", "\\midrule"]
    for key, name, uses in METHODS:
        if key not in g.index:
            continue
        r = g.loc[key]
        lines.append(f"{name} & {uses} & {f(r[('median', 'EXTRA')])} & {f(r[('mean', 'EXTRA')])} & "
                     f"{f(r[('median', 'INTER')])} & {f(r[('mean', 'INTER')])} \\\\")
        if key in ("pooled-static", "nn-ast", "mean"):
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("lowo", "\n".join(lines) + "\n")


def lowo_per_kernel():
    e = pd.read_csv(PB / "lowo_errors.csv")
    t = e.pivot_table(index="workload", columns=["split", "method"], values="mape")
    v = pd.read_csv(PB / "lowo_validity.csv").pivot(index="workload", columns="split",
                                                      values=["nn_static", "oracle"])
    lines = ["\\begin{tabular}{lrrrrrrll}", "\\toprule",
             "Kernel & \\multicolumn{3}{c}{EXTRA} & \\multicolumn{3}{c}{INTER} & \\multicolumn{2}{c}{"
             "Nearest by $\\hat z$ (EXTRA)} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}",
             " & fp & static $\\alpha$ & own & fp & static $\\alpha$ & own & chosen & oracle \\\\", "\\midrule"]
    for w in t.index:
        r = t.loc[w]
        lines.append(f"{w} & {f(r[('EXTRA', 'static-fp')])} & {f(r[('EXTRA', 'static-alpha')], 1)} & "
                     f"{f(r[('EXTRA', 'own')], 1)} & {f(r[('INTER', 'static-fp')])} & "
                     f"{f(r[('INTER', 'static-alpha')], 1)} & {f(r[('INTER', 'own')], 1)} & "
                     f"{v.loc[w, ('nn_static', 'EXTRA')]} & {v.loc[w, ('oracle', 'EXTRA')]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("lowo_per_kernel", "\n".join(lines) + "\n")


def similarity_table():
    v = pd.read_csv(PB / "lowo_validity.csv")
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "Distance & \\multicolumn{2}{c}{EXTRA} & \\multicolumn{2}{c}{INTER} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", " & median $\\rho$ & mean $\\rho$ & median $\\rho$ & "
             "mean $\\rho$ \\\\", "\\midrule"]
    for col, name in (("rho_static", "$\\hat z$ predicted from source"), ("rho_measured", "measured $z$"),
                      ("rho_ast", "clang AST node counts")):
        ex, it = v[v.split == "EXTRA"][col], v[v.split == "INTER"][col]
        lines.append(f"{name} & {f(ex.median(), 3)} & {f(ex.mean(), 3)} & {f(it.median(), 3)} & "
                     f"{f(it.mean(), 3)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("similarity", "\n".join(lines) + "\n")


def idioms_table():
    t = pd.read_csv(PB / "static_idioms.csv", index_col=0)
    pb = t[~t.out_of_distribution]
    out = t[t.out_of_distribution]
    lines = ["\\begin{tabular}{lrrrrrl}", "\\toprule",
             "Program & affine & indirect & pointer & data-bounded loops & while loops & gate \\\\", "\\midrule",
             f"PolyBench ({len(pb)} kernels, min / max) & {pb.affine.min():.3f}--{pb.affine.max():.3f} & "
             f"{pb.indirect.max():.3f} & {pb.pointer.max():.3f} & {pb.loops_data.max():.2f} & "
             f"{pb.loops_while.max():.2f} & in \\\\"]
    for w, r in out.iterrows():
        lines.append(f"{w} & {r.affine:.3f} & {r.indirect:.3f} & {r.pointer:.3f} & {r.loops_data:.2f} & "
                     f"{r.loops_while:.2f} & out \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("idioms", "\n".join(lines) + "\n")


def gap_pilot_table():
    e = pd.read_csv(GAP / "gap_pilot_errors.csv")
    t = e.pivot_table(index=["kernel", "split"], columns="method", values="mape")
    blind = {"gap_pr": "no", "gap_bfs": "no", "gap_pr_kron": "yes", "gap_bfs_kron": "yes"}
    lines = ["\\begin{tabular}{llrrrrrrl}", "\\toprule",
             "Workload & split & skel.\\ fp & skel.\\ $\\alpha$ & own & PB nearest & PB mean & PB oracle & "
             "blind \\\\", "\\midrule"]
    prev = None
    for (k, s), r in t.iterrows():
        if prev is not None and k != prev:
            lines.append("\\addlinespace[2pt]")
        prev = k
        b = blind.get(k, "partly" if k.startswith("gap_prc") else "yes")
        lines.append(f"{k.replace('gap_', '').replace('_', '\\_')} & {s} & {f(r.get('skeleton-fp'))} & "
                     f"{f(r.get('skeleton-alpha'), 1)} & {f(r.get('own'), 1)} & {f(r.get('pb-nearest'), 1)} & "
                     f"{f(r.get('pb-mean'), 1)} & {f(r.get('pb-oracle'), 1)} & {b if s == 'EXTRA' else ''} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("gap_pilot", "\n".join(lines) + "\n")


def auto_table():
    s = pd.read_csv(D / "gap_scores.csv")
    lines = ["\\begin{tabular}{llcrrrrr}", "\\toprule",
             "Kernel & graph & scales & \\multicolumn{2}{c}{max $|$footprint error$|$ (\\%)} & "
             "\\multicolumn{2}{c}{mean $\\alpha$ MAPE (\\%)} & interp./hand \\\\",
             "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}", " & & & hand & interp. & hand & interp. & refs \\\\",
             "\\midrule"]
    for k in ["pr", "bfs", "cc", "bc", "tc", "sssp"]:
        for g in ["uniform", "kron"]:
            a = s[(s.kernel == k) & (s.graph == g) & (s.source == "auto")]
            h = s[(s.kernel == k) & (s.graph == g) & (s.source == "hand") & s.scale.isin(a.scale)]
            if a.empty:
                continue
            ref = a.set_index("scale").references / h.set_index("scale").references
            lines.append(f"{k} & {'Kronecker' if g == 'kron' else 'uniform'} & {a.scale.min()}--{a.scale.max()} & "
                         f"{h.fp_error.abs().max():.2f} & {a.fp_error.abs().max():.2f} & {h.alpha_mape.mean():.1f} & "
                         f"{a.alpha_mape.mean():.1f} & {ref.median():.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("auto", "\n".join(lines) + "\n")


def auto_scales_table():
    s = pd.read_csv(D / "gap_scores.csv")
    s = s[s.kernel.isin(["pr", "bfs"])]
    piv = s.pivot_table(index="scale", columns=["kernel", "graph", "source"], values="alpha_mape")
    cols = [(k, g) for k in ("pr", "bfs") for g in ("uniform", "kron")]
    lines = ["\\begin{tabular}{r" + "rr" * len(cols) + "}", "\\toprule",
             "Scale & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{k}, {'Kron.' if g == 'kron' else 'uniform'}}}"
                                     for k, g in cols) + " \\\\",
             "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cols))),
             " & " + " & ".join("hand & interp." for _ in cols) + " \\\\", "\\midrule"]
    for sc in piv.index:
        vals = []
        for k, g in cols:
            for src in ("hand", "auto"):
                v = piv.get((k, g, src))
                vals.append(f(v.loc[sc] if v is not None else np.nan, 1))
        lines.append(f"{sc} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("auto_scales", "\n".join(lines) + "\n")


def csr_table():
    rows = []
    for name, path in (("blind", "c_errors_blind.csv"), ("post hoc", "c_errors.csv")):
        e = pd.read_csv(CSR / path)
        for (k, s), g in e.groupby(["kernel", "split"]):
            r = g.set_index("method").mape
            rows.append((k, s, name, r.get("skeleton-fp"), r.get("skeleton-alpha"), r.get("own")))
    lines = ["\\begin{tabular}{lllrrr}", "\\toprule",
             "Program & split & run & footprint & static $\\alpha$ & own model \\\\", "\\midrule"]
    for k, s, name, fp, a, own in sorted(rows):
        lines.append(f"{k.replace('_', '\\_')} & {s} & {name} & {f(fp)} & {f(a, 1)} & {f(own, 1)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("csr", "\n".join(lines) + "\n")


def cost_table():
    t = pd.read_csv(D / "interp_cost.csv")
    piv = t.pivot_table(index="scale", columns=["kernel", "graph"], values="seconds")
    cols = [(k, g) for k in ("pr", "bfs", "cc", "bc", "tc", "sssp") for g in ("uniform", "kron")]
    lines = ["\\begin{tabular}{r" + "r" * len(cols) + "}", "\\toprule",
             "Scale & " + " & ".join(f"{k} {'K' if g == 'kron' else 'u'}" for k, g in cols) + " \\\\", "\\midrule"]
    for sc in piv.index:
        vals = []
        for kg in cols:
            v = piv.get(kg)
            v = v.loc[sc] if v is not None and sc in v.index else np.nan
            vals.append("--" if not np.isfinite(v) else f"{v:.0f}")
        lines.append(f"{sc} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("cost", "\n".join(lines) + "\n")


def blind_tables():
    sc = pd.read_csv(HERE / "blind" / "scores.csv")
    ph = pd.read_csv(HERE / "blind" / "posthoc_scores.csv").set_index("config")
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             "Program & config & truth (B) & predicted (B) & footprint error & $\\alpha$ MAPE & post hoc $\\alpha$ \\\\",
             "\\midrule"]
    prev = None
    for _, r in sc.iterrows():
        if prev is not None and r.program != prev:
            lines.append("\\addlinespace")
        prev = r.program
        post = f"{ph.loc[r.config, 'alpha_mape']:.1f}" if r.program == "lulesh" and r.config in ph.index else "--"
        lines.append(f"{'HPCCG' if r.program == 'hpccg' else 'LULESH'} & {r.config} & {r.truth:,.0f} & "
                     f"{r.predicted:,.0f} & {r.fp_error:+.2f}\\% & {r.alpha_mape:.2f}\\% & {post} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("blind", "\n".join(lines) + "\n")
    sp = pd.read_csv(HERE / "blind" / "splits.csv")
    lines = ["\\begin{tabular}{llrrrr}", "\\toprule",
             "Program & split & held-out config & static $\\alpha$ (blind) & own model & footprint \\\\", "\\midrule"]
    for _, r in sp.iterrows():
        lines.append(f"{'HPCCG' if r.program == 'hpccg' else 'LULESH'} & {r.split} & {r.config} & "
                     f"{r.static_alpha:.2f} & {r.own_model:.2f} & {r.fp_error:+.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("blind_splits", "\n".join(lines) + "\n")


def baseline_table():
    pb, gp = pd.read_csv(D / "pb_baseline.csv"), pd.read_csv(D / "gap_baseline.csv")
    lines = ["\\begin{tabular}{rrrrr}", "\\toprule",
             "Count class $c_j$ & \\multicolumn{2}{c}{PolyBench} & \\multicolumn{2}{c}{GAP} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", " & bytes & addresses & bytes & addresses \\\\",
             "\\midrule"]
    for i in range(len(pb)):
        if pb.bytes[i] > 1 or gp.bytes[i] > 1:
            lines.append(f"$2^{{{int(np.log2(pb['count'][i]))}}}$ & {pb.bytes[i]:,.0f} & {pb.addresses[i]:,.0f} & "
                         f"{gp.bytes[i]:,.0f} & {gp.addresses[i]:,.0f} \\\\")
    lines.append("\\midrule")
    lines.append(f"total & {pb.bytes.sum():,.0f} & {pb.addresses.sum():,.0f} & {gp.bytes.sum():,.0f} & "
                 f"{gp.addresses.sum():,.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("baseline", "\n".join(lines) + "\n")


def numbers():
    """Single values quoted in the text, as LaTeX macros."""
    e = pd.read_csv(PB / "lowo_errors.csv")
    v = pd.read_csv(PB / "lowo_validity.csv")
    s = pd.read_csv(D / "gap_scores.csv")
    m = pd.read_csv(D / "pb_moments.csv")
    meta = pd.read_csv(D / "pb_static_meta.csv")
    big = s[s.kernel.isin(["pr", "bfs"]) & (s.scale >= 13)]
    big = big[big.set_index(["kernel", "graph", "scale"]).index.isin(
        big[big.source == "auto"].set_index(["kernel", "graph", "scale"]).index)]
    med = e.groupby(["method", "split"]).mape.median()
    out = {
        "fpExtra": med[("static-fp", "EXTRA")], "fpInter": med[("static-fp", "INTER")],
        "alphaExtra": med[("static-alpha", "EXTRA")], "alphaInter": med[("static-alpha", "INTER")],
        "ownExtra": med[("own", "EXTRA")], "ownInter": med[("own", "INTER")],
        "oracleExtra": med[("oracle", "EXTRA")], "oracleInter": med[("oracle", "INTER")],
        "rhoStaticExtra": v[v.split == "EXTRA"].rho_static.median(),
        "rhoStaticInter": v[v.split == "INTER"].rho_static.median(),
        "rhoAstExtra": v[v.split == "EXTRA"].rho_ast.median(),
        "mMedianErr": float(np.median(np.abs(m.m_pred - m.m_meas) / m.m_meas * 100)),
        "sdMedianErr": float(np.median(np.abs(m.sd_pred - m.sd_meas) / m.sd_meas * 100)),
        "interpMaxSeconds": meta.seconds.max(),
        "refShareMedian": float(np.median(meta.total / meta.pin_references)) * 100,
        "autoFpMax": s[s.source == "auto"].fp_error.abs().max(),
        "bigAutoAlphaMin": big[big.source == "auto"].alpha_mape.min(),
        "bigAutoAlphaMax": big[big.source == "auto"].alpha_mape.max(),
        "bigHandAlphaMin": big[big.source == "hand"].alpha_mape.min(),
        "bigHandAlphaMax": big[big.source == "hand"].alpha_mape.max(),
        "bigAutoFpMax": big[big.source == "auto"].fp_error.abs().max(),
        "maxScaleAuto": float(s[s.source == "auto"].scale.max()),
    }
    text = "".join(f"\\newcommand{{\\num{k}}}{{{val:.2f}}}\n" for k, val in out.items())
    write("numbers", text)


if __name__ == "__main__":
    for fn in (lowo_table, lowo_per_kernel, similarity_table, idioms_table, gap_pilot_table, auto_table,
               auto_scales_table, csr_table, cost_table, baseline_table, blind_tables):
        fn()
    try:
        numbers()
    except KeyError as err:
        print("numbers: missing", err)
