"""Benchmark tables and numbers for paper_gpu/main.tex, from paper_gpu/data/benchmarks.csv (benchmarks.py).
Used by make_tables.py."""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "tables")
SUITES = ["Rodinia", "GraphBIG", "PolyBench-GPU"]
INTERPRETERS = (("hybrid", "Hybrid"), ("full", "Whole program"))
ALONE = "graphbig-betweenness"   # the hybrid interpreter's run of it: alone, with a ten-hour limit
# On the small inputs most kernels of these launch no block (grid = input size / 256 = 0), so they do not run:
# left out of every small-input result (the files keep them).
SMALL_EXCLUDED = {"polybench-corr", "polybench-covar"}


def small_only_running(d, col="name"):
    """A small-input frame without SMALL_EXCLUDED."""
    return d[~d[col].isin(SMALL_EXCLUDED)].reset_index(drop=True)


def tt(name):
    return r"\texttt{" + name.replace("_", r"\_") + "}"


def frame():
    """benchmarks.csv with, per interpreter, its dependences against the trace: certain on the GPU (ctp) or not
    (cfp), possible on the GPU (pyes) or not (pno), and the GPU's dependences it did not report (fn)."""
    d = small_only_running(pd.read_csv(os.path.join(DATA, "benchmarks.csv")))
    for w, _ in INTERPRETERS:
        T, C, P = d[f"{w}_truth"], d[f"{w}_must"], d[f"{w}_may"]
        d[f"{w}_ctp"] = (d[f"{w}_must_precision"] * C).round()
        d[f"{w}_cfp"] = C - d[f"{w}_ctp"]
        found = (d[f"{w}_recall"] * T).round()
        found_c = (d[f"{w}_must_recall"] * T).round()
        d[f"{w}_pyes"] = found - found_c
        d[f"{w}_pno"] = P - d[f"{w}_pyes"]
        d[f"{w}_fn"] = T - found
        d[f"{w}_done"] = d[f"{w}_must_precision"].notna()
    return d


def write(name, body):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(body)
    print("wrote", name)


def summary_table():
    d = frame()
    rows = []
    for w, label in INTERPRETERS:
        for i, suite in enumerate(SUITES + [None]):
            g = d[d[f"{w}_done"] & ((d.suite == suite) if suite else True)]
            cells = [str(len(g))] + [f"{int(g[c].sum()):,}" for c in
                                     (f"{w}_truth", f"{w}_ctp", f"{w}_cfp", f"{w}_pyes", f"{w}_pno", f"{w}_fn")]
            first = r"\multirow{4}{*}{" + label + "}" if i == 0 else ""
            name = suite if suite else r"\textbf{Total}"
            rows.append(f"{first} & {name} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    head = (r"\begin{tabular}{llrrrrrrr}" "\n" r"\toprule" "\n"
            r" & & & & \multicolumn{2}{c}{Certain} & \multicolumn{2}{c}{Possible} & \\" "\n"
            r"\cmidrule(lr){5-6} \cmidrule(lr){7-8}" "\n"
            r"Interpreter & Suite & Programs & \shortstack{GPU\\deps} & \shortstack{On the\\GPU} & "
            r"\shortstack{Not on\\the GPU} & \shortstack{On the\\GPU} & \shortstack{Not on\\the GPU} & "
            r"\shortstack{Missed\\(FN)} \\" "\n" r"\midrule" "\n")
    write("bench_summary.tex", head + "\n".join(rows[:-1]) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


def per_benchmark_table():
    d = frame()
    rows, last = [], None
    for _, r in d.iterrows():
        if last is not None and r.suite != last:
            rows.append(r"\midrule")
        last = r.suite

        def cells(w):
            if not r[f"{w}_done"]:
                return [r"\multicolumn{5}{c}{" + ("timeout" if "timeout" in str(r[f"{w}_status"]) else "failed") + "}"]
            return [f"{int(r[f'{w}_ctp']):,}", f"{int(r[f'{w}_pyes']):,}", f"{int(r[f'{w}_pno']):,}",
                    f"{int(r[f'{w}_fn']):,}", f"{r[f'{w}_s']:.0f}"]

        ops = r.hybrid_truth_ops if not pd.isna(r.get("hybrid_truth_ops")) else r.get("full_truth_ops")
        truth = r.hybrid_truth if not pd.isna(r.hybrid_truth) else r.full_truth
        rows.append(" & ".join([r.suite.replace("-GPU", ""), tt(r["name"].split("-", 1)[1]),
                                "--" if pd.isna(ops) else f"{int(ops)}", "--" if pd.isna(truth) else f"{int(truth):,}",
                                *cells("hybrid"), *cells("full")]) + r" \\")
    head = (r"\begin{tabular}{llrr|rrrrr|rrrrr}" "\n" r"\toprule" "\n"
            r" & & & & \multicolumn{5}{c|}{Hybrid} & \multicolumn{5}{c}{Whole program} \\" "\n"
            r"Suite & Program & Ops & \shortstack{GPU\\deps} & Cert. & \shortstack{Poss.\\yes} & \shortstack{Poss.\\no} & "
            r"FN & s & Cert. & \shortstack{Poss.\\yes} & \shortstack{Poss.\\no} & FN & s \\" "\n" r"\midrule" "\n")
    write("benchmarks.tex", head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


def numbers():
    """Macros for the text: \\numBenchHy... and \\numBenchFu... (hybrid and whole-program interpreter)."""
    d = frame()
    n = {"BenchTotal": len(d) + len(SMALL_EXCLUDED), "BenchSmallExcluded": len(SMALL_EXCLUDED),
         "BenchNvbitLo": f"{(d.nvbit_s / d.native_s).min():.1f}",
         "BenchNvbitHi": f"{(d.nvbit_s / d.native_s).max():.0f}",
         "BenchFullHostWarn": int(d.get("full_host_warning", pd.Series(dtype=float)).fillna(0).sum())}
    for w, k in (("hybrid", "Hy"), ("full", "Fu")):
        g = d[d[f"{w}_done"]]
        n[f"Bench{k}Done"] = len(g)
        for c, label in (("truth", "Truth"), ("ctp", "CertTP"), ("cfp", "CertFP"), ("pyes", "PossYes"),
                         ("pno", "PossNo"), ("fn", "FN")):
            n[f"Bench{k}{label}"] = f"{int(g[f'{w}_{c}'].sum()):,}".replace(",", "{,}")
        n[f"Bench{k}Exact"] = int(((g[f"{w}_fn"] == 0) & (g[f"{w}_pyes"] == 0) & (g[f"{w}_pno"] == 0)).sum())
        n[f"Bench{k}NoFN"] = int((g[f"{w}_fn"] == 0).sum())
        t = g[g.name != ALONE][f"{w}_s"]                               # the runs that shared the machine
        n[f"Bench{k}TimeLo"] = f"{t.min():.0f}"
        n[f"Bench{k}TimeHi"] = f"{t.max():.0f}"
        n[f"Bench{k}TimeMedian"] = f"{t.median():.0f}"
    alone = d[(d.name == ALONE) & d.hybrid_done]
    n["BenchHyAloneHours"] = f"{alone.hybrid_s.iloc[0] / 3600:.1f}" if len(alone) else "--"
    return n


def main():
    os.makedirs(OUT, exist_ok=True)
    summary_table()
    per_benchmark_table()


if __name__ == "__main__":
    main()
    print(numbers())
