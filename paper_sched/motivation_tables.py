"""Tables for the motivation version of the paper (motivation.tex), without any MemPrint evaluation: the paper's
tables with MemPrint's rows dropped, and the corpus table without its MemPrint columns.

    ~/environments/memfp/bin/python paper_sched/motivation_tables.py   (after make_tables.py, sensitivity.py, repro.py)
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_tables import NAMES, fmt  # noqa: E402

SRC, OUT = f"{HERE}/tables", f"{HERE}/tables_motivation"
ROWS = ["grid_08", "grid_llm_08", "margins", "capacity_loads", "lisa_measured", "lisa_hybrid", "sens_beta", "sens_pool",
        "sens_interval", "repro_ksplus", "repro_pond"]


def drop_rows(name):
    lines = open(f"{SRC}/{name}.tex").read().splitlines()
    kept = [l for l in lines if not ("&" in l and l.lstrip().startswith("MemPrint"))]
    open(f"{OUT}/{name}.tex", "w").write("\n".join(kept) + "\n")
    return len(lines) - len(kept)


def corpus(data=os.path.expanduser("~/memprint-data/2026-10-13-sched-llm")):
    sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
    from memprint.sched.run import largest
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
                                 fmt(m.loc[(p, i), "truth"] / r.truth), fmt(r.rss / r.truth), fmt(r.massifi / r.truth),
                                 fmt(r.dhat / r.truth), fmt(r.heaptrack / r.truth), fmt(nat.get((p, i)), 1),
                                 fmt(s.loc[(p, i), "massifi"], 1), fmt(s.loc[(p, i), "dhat"], 1),
                                 fmt(s.loc[(p, i), "heaptrack"], 1), fmt(s.loc[(p, i), "truth"], 0)]) + r" \\")
    head = (r"\begin{tabular}{llrrrrrrrrrrr}" "\n" r"\toprule" "\n"
            r" & & \multicolumn{6}{c}{Footprint (ratios to the true peak)} & & \multicolumn{4}{c}{Cost (time / native)} \\"
            "\n" r"\cmidrule(lr){3-8} \cmidrule(lr){10-13}" "\n"
            r"Program & Input & Peak (MB) & Mean/peak & RSS & Massif & DHAT & heaptrack & Native (s) & Massif & DHAT & heaptrack & Full trace \\"
            "\n" r"\midrule")
    open(f"{OUT}/corpus.tex", "w").write(head + "\n" + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for n in ROWS:
        print(n, drop_rows(n), "MemPrint rows dropped")
    corpus()
    left = [n for n in os.listdir(OUT) if "memprint" in open(f"{OUT}/{n}").read().lower()]
    print("tables still mentioning MemPrint:", left)
