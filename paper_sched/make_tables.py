"""Tables for paper_sched/main.tex (LaTeX fragments in paper_sched/tables/).

    python paper_sched/make_tables.py --data ~/memprint-data/2026-10-13-sched-llm
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from memprint.sched import run  # noqa: E402
from memprint.sched.run import largest  # noqa: E402

NAMES = {"2mm": "2mm", "atax": "atax", "gemm": "gemm", "jacobi-2d": "jacobi-2d", "cc1": "GCC cc1",
         "darknet": "darknet", "gap-bfs": "GAP bfs", "gap-cc": "GAP cc", "gap-pr": "GAP pr",
         "gap-sssp": "GAP sssp", "hpccg": "HPCCG", "kripke": "Kripke", "lua": "Lua", "lulesh": "LULESH",
         "miniVite": "miniVite", "minife": "MiniFE", "npb-cg": "NPB CG", "npb-is": "NPB IS", "npb-mg": "NPB MG",
         "perl": "Perl", "python": "Python", "quicksilver": "Quicksilver", "sqlite": "SQLite",
         "llama2c": "llama2.c", "llamacpp": "llama.cpp"}


def fmt(x, d=2):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def corpus_table(data, out):
    cv = pd.read_csv(f"{data}/corpus/curves.csv", dtype={"input": str})
    k = pd.read_csv(f"{data}/corpus/costs.csv", dtype={"input": str})
    big = largest(cv)
    w = big.groupby(["prog", "input", "method"]).bytes.max().unstack()
    m = big.groupby(["prog", "input", "method"]).bytes.mean().unstack()
    s = k.pivot_table(index=["prog", "input"], columns="method", values="slowdown")
    nat = k[k.method == "native"].set_index(["prog", "input"]).wall_s
    lines = []
    for (p, i), r in w.iterrows():
        lines.append(" & ".join([NAMES.get(p, p), i.replace("_", r"\_"), fmt(r.truth / 1e6, 1),
                                 fmt(m.loc[(p, i), "truth"] / r.truth), fmt(r.rss / r.truth),
                                 fmt(r.W5 / r.truth), fmt(r.massifi / r.truth), fmt(r.dhat / r.truth),
                                 fmt(r.heaptrack / r.truth), fmt(nat.get((p, i)), 1),
                                 fmt(s.loc[(p, i), "W5"], 1), fmt(s.loc[(p, i), "massifi"], 1),
                                 fmt(s.loc[(p, i), "dhat"], 1), fmt(s.loc[(p, i), "truth"], 0)]) + r" \\")
    head = (r"\begin{tabular}{llrrrrrrrrrrrr}" "\n" r"\toprule" "\n"
            r" & & \multicolumn{7}{c}{Footprint (ratios to the true peak)} & & \multicolumn{4}{c}{Cost (time / native)} \\"
            "\n" r"\cmidrule(lr){3-9} \cmidrule(lr){11-14}" "\n"
            r"Program & Input & Peak (MB) & Mean/peak & RSS & MemPrint W5 & Massif & DHAT & heaptrack & Native (s) & W5 & Massif & DHAT & Full trace \\"
            "\n" r"\midrule")
    open(f"{out}/corpus.tex", "w").write(head + "\n" + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


def margins_table(data, out):
    setup = run.Setup(f"{data}/corpus", f"{data}/traces", 10, run.Cluster(), "synthetic")
    rows = []
    for o in ["memprint-W1", "memprint-W5", "memprint-P", "massif-i", "massif-pages-i", "massif", "massif-pages",
              "heaptrack", "dhat", "dhat-curve",
              "dhat-hold", "gperf", "rss-peak"]:
        cells = [run_label(o)]
        for sc in ("bytes", "rss"):
            v = np.array(list(setup.required(sc, o).values()))
            cells += [fmt(np.median(v) * 100, 0), fmt(np.quantile(v, 0.9) * 100, 0), fmt(v.max() * 100, 0)]
        rows.append(" & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{lrrrrrr}" "\n" r"\toprule" "\n"
            r" & \multicolumn{3}{c}{Truth = bytes touched} & \multicolumn{3}{c}{Truth = RSS} \\" "\n"
            r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}" "\n"
            r"Oracle & Median & 90th pct. & Max & Median & 90th pct. & Max \\" "\n" r"\midrule")
    open(f"{out}/margins.tex", "w").write(head + "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


def run_label(o):
    from make_figures import STYLE
    return STYLE[o][0].replace("%", r"\%")


def grid_table(data, out, load=0.8, pattern="grid_all*.csv", name=None):
    files = sorted(glob.glob(f"{data}/results/{pattern}"))
    if not files:
        return
    g = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "error" in g:
        g = g[g.error.isna()]
    g["margin"] = g.margin.astype(str)
    if "restarts" in g:   # dynamic provisioning's failed resizes, counted with the OOM kills
        g["kills"] = g.kills + g.restarts.fillna(0)
    from make_figures import select
    g = select(g, "cal")
    g = g[g.load == load]
    rows = []
    order = (g[g.scenario == "bytes"].groupby("oracle").steady_response.mean().sort_values().index)
    for o in order:
        cells = [run_label(o)]
        for sc in ("bytes", "rss"):
            d = g[(g.scenario == sc) & (g.oracle == o)]
            cells += [fmt(d.steady_response.mean(), 0), fmt(d.steady_bsld.mean(), 1), fmt(d.kills.mean(), 0),
                      fmt(d.steady_reserved.mean() * 100, 0), fmt(d.steady_resident.mean() * 100, 0)]
        rows.append(" & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{lrrrrrrrrrr}" "\n" r"\toprule" "\n"
            r" & \multicolumn{5}{c}{Truth = bytes touched} & \multicolumn{5}{c}{Truth = RSS} \\" "\n"
            r"\cmidrule(lr){2-6} \cmidrule(lr){7-11}" "\n"
            r"Oracle & Resp. (min) & BSLD & Kills & Res. \% & Used \% & Resp. (min) & BSLD & Kills & Res. \% & Used \% \\"
            "\n" r"\midrule")
    open(f"{out}/{name or 'grid_' + str(load).replace('.', '')}.tex", "w").write(
        head + "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


def file_maps_table(data, fm, out, load=0.8):
    """MemPrint's oracles before and after -file_maps (fm: the data root of the reruns), calibrated 90% margins."""
    from make_figures import select
    mp = ["memprint-W5", "memprint-W1", "memprint-P", "memprint-W5-peak"]

    def get(root, pattern):
        files = sorted(glob.glob(f"{root}/results/{pattern}"))
        g = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        g["margin"] = g.margin.astype(str)
        g = select(g, "cal")
        return g[(g.load == load) & g.oracle.isin(mp)].groupby(["oracle", "scenario"]).agg(
            r=("steady_response", "mean"), k=("kills", "mean"))
    cols = [(get(data, "grid_all.csv"), get(fm, "grid_all_fm.csv")), (get(data, "grid_llm.csv"), get(fm, "grid_llm_fm.csv"))]
    rows = []
    for o in mp:
        cells = [run_label(o)]
        for before, after in cols:
            for sc in ("bytes", "rss"):
                cells += [fmt(before.loc[(o, sc), "r"], 0), fmt(after.loc[(o, sc), "r"], 0)]
        rows.append(" & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{lrrrrrrrr}" "\n" r"\toprule" "\n"
            r" & \multicolumn{4}{c}{Main stream (25 programs)} & \multicolumn{4}{c}{LLM stream} \\" "\n"
            r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}" "\n"
            r" & \multicolumn{2}{c}{Bytes touched} & \multicolumn{2}{c}{RSS} & \multicolumn{2}{c}{Bytes touched} & \multicolumn{2}{c}{RSS} \\" "\n"
            r"Oracle & before & after & before & after & before & after & before & after \\" "\n" r"\midrule")
    open(f"{out}/file_maps.tex", "w").write(head + "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


def time_axis_table(v3, out, load=0.8):
    """The section on the time axis and the level: oracles under the three truths, calibrated 90% margins, main
    stream (rssdata: the RSS without code)."""
    from make_figures import select
    g = pd.read_csv(f"{v3}/results/v3m_all.csv")
    g["margin"] = g.margin.astype(str)
    order = ["exact", "exact-peak", "rss-peak", "memprint-W5", "memprint-W5nat", "memprint-Pnat",
             "memprint-W5nat-rss", "memprint-Pnat-rss", "dhat", "massif-i", "heaptrack", "gperf", "request"]
    s = select(g, "cal")
    t = s[s.load == load].groupby(["oracle", "scenario"]).steady_response.mean()
    rows = [" & ".join([run_label(o)] + [n0f(t.get((o, sc))) for sc in ("bytes", "rss", "rssdata")]) + r" \\"
            for o in order]
    head = (r"\begin{tabular}{lrrr}" "\n" r"\toprule" "\n"
            r"Oracle & Truth = bytes touched & Truth = RSS & Truth = RSS without code \\" "\n" r"\midrule")
    open(f"{out}/time_axis.tex", "w").write(head + "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


def n0f(x):
    return "--" if x is None or pd.isna(x) else f"{x:,.0f}".replace(",", "{,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/memprint-data/2026-10-13-sched-llm"))
    ap.add_argument("--out", default=os.path.join(HERE, "tables"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, HERE)
    corpus_table(a.data, a.out)
    margins_table(a.data, a.out)
    grid_table(a.data, a.out)
    grid_table(a.data, a.out, pattern="grid_llm*.csv", name="grid_llm_08")
    fm = os.path.expanduser("~/memprint-data/2026-10-13-sched-fm")
    if os.path.exists(f"{fm}/results/grid_all_fm.csv"):
        file_maps_table(a.data, fm, a.out)
    if os.path.exists(f"{fm}/results/v3m_all.csv"):
        time_axis_table(fm, a.out)
    print("tables written to", a.out)


if __name__ == "__main__":
    main()
