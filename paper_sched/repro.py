"""Tables of the reproductions of the published baselines' own results (~/memprint-data/2026-10-24-repro).

    ~/environments/memfp/bin/python paper_sched/repro.py

KS+: the authors' simulation (dos-group/KSPlus) on their traces (dos-group/k-Segments-traces), memory wastage summed
over a workflow's tasks, mean of 10 seeds, as the paper's Figure 6. Pond: Figure 3's DRAM savings from pooling, from
the public Azure packing trace (memprint.repro.pond_pooling), averaged over 10 clusters.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.expanduser("~/memprint-data/2026-10-24-repro")
# the paper's numbers (KS+, eScience 2024, Section III-C): KS+'s reduction against the best baseline (k-Segments
# Selective) and against PPM-Improved, for 25/50/75% training data; sarek against PPM-Improved: 45% on average
KS_PAPER = {("eager", "best"): (36, 39, 40), ("eager", "ppm"): (54, 52, 51), ("sarek", "best"): (31, 28, 29),
            ("sarek", "ppm"): (45, 45, 45)}


def ksplus():
    rows = {}
    for wf in ("eager", "sarek"):
        d = pd.read_csv(f"{R}/ksplus/code/results/results_{wf}_4.csv")
        agg = d.groupby(["Setup", "Predictor", "Task"]).Wastage.mean().groupby(["Setup", "Predictor"]).sum().unstack()
        best = agg[["k-Segments Selective", "k-Segments Partial", "Tovar-Improved", "Tovar", "Default"]].min(axis=1)
        rows[(wf, "best")] = tuple(round(100 * (1 - agg["KS+"] / best)))
        rows[(wf, "ppm")] = tuple(round(100 * (1 - agg["KS+"] / agg["Tovar-Improved"])))
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{Paper} & \multicolumn{3}{c}{Reproduced} \\",
             r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
             r"Workflow & KS+ against & 25\% & 50\% & 75\% & 25\% & 50\% & 75\% \\", r"\midrule"]
    for (wf, what), rep in rows.items():
        paper = KS_PAPER[(wf, what)]
        pp = [f"{p}" for p in paper] if not (wf == "sarek" and what == "ppm") else [r"\multicolumn{3}{c}{45 (mean)}"]
        lab = "k-Segments Selective" if what == "best" else "PPM-Improved"
        lines.append(f"{wf} & {lab} & " + " & ".join(pp) + " & " + " & ".join(f"{r:.0f}" for r in rep) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{HERE}/tables/repro_ksplus.tex", "w").write("\n".join(lines) + "\n")
    return rows


def pond():
    d = pd.read_csv(f"{R}/pond/results/pond_fig3.csv")
    t = d.groupby(["provisioning", "pool_share", "sockets"]).savings.mean().unstack()
    socks = list(t.columns)
    lines = [r"\begin{tabular}{ll" + "r" * len(socks) + "}", r"\toprule",
             r"Provisioning & Pool share & " + " & ".join(f"{s}" for s in socks) + r" \\", r"\midrule"]
    for (prov, p), row in t.iterrows():
        name = {"server": "per server", "fleet": "uniform fleet"}[prov]
        lines.append(f"{name} & {100 * p:.0f}\\% & " + " & ".join(f"{abs(100 * v) if abs(v) < 5e-4 else 100 * v:.1f}" for v in row) + r" \\")
    lines += [r"\midrule", r"Pond (paper) & 50\% & \multicolumn{4}{c}{} & 12 & 13 \\", r"\bottomrule", r"\end{tabular}"]
    open(f"{HERE}/tables/repro_pond.tex", "w").write("\n".join(lines) + "\n")
    return t


if __name__ == "__main__":
    print(ksplus())
    print(pond().round(3).to_string())
