"""Simulation tables and numbers for paper_gpu/main.tex, from paper_gpu/data/sim_blocks*.csv (simulate_blocks.py)
and sim_delay.csv (simulate.py). Used by make_tables.py."""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "tables")
SUITES = ["Rodinia", "GraphBIG", "PolyBench-GPU"]
POLICIES = [("sequential", "In issue order"), ("buffer", "Same buffer"), ("buffer-rw", "Same buffer, one writes"),
            ("full", "Whole-program interpreter"), ("hybrid", "Hybrid interpreter"),
            ("truth", "Traced dependences"), ("hybrid-block", "Hybrid interpreter, blocks"),
            ("truth-block", "Traced dependences, blocks")]
RUNS = (("small", "sim_blocks.csv"), ("large", "sim_blocks_large.csv"))


def geo(x):
    x = np.asarray(x, dtype=float)
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")


def load(name):
    path = os.path.join(DATA, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def summary_table():
    frames = {k: load(f) for k, f in RUNS}
    rows = []
    for policy, label in POLICIES:
        cells = [label]
        for k, _ in RUNS:
            d = frames[k]
            g = d[d.policy == policy] if d is not None else None
            if g is None or g.empty:
                cells += ["--"] * 5
                continue
            cells += [f"{geo(g.speedup):.3f}"] + [f"{geo(g[g.suite == s].speedup):.2f}" for s in SUITES] + \
                [f"{int(g.rollbacks.sum()):,}"]
        rows.append(" & ".join(cells) + r" \\")
        if policy in ("buffer-rw", "truth"):
            rows.append(r"\midrule")
    head = (r"\begin{tabular}{lrrrrrrrrrr}" "\n" r"\toprule" "\n"
            r" & \multicolumn{5}{c}{Small inputs} & \multicolumn{5}{c}{Large inputs} \\" "\n"
            r"\cmidrule(lr){2-6} \cmidrule(lr){7-11}" "\n"
            r"Dependences & All & Rod. & GB & Poly & Rollb. & All & Rod. & GB & Poly & Rollb. \\" "\n"
            r"\midrule" "\n")
    write("sim_summary.tex", head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


def write(name, body):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(body)
    print("wrote", name)


def numbers():
    n = {}
    for k, f in RUNS:
        d = load(f)
        if d is None:
            continue
        K = "Small" if k == "small" else "Large"
        for policy, macro in (("buffer", "Buf"), ("buffer-rw", "BufRw"), ("hybrid", "Hy"), ("truth", "Tr"),
                              ("hybrid-block", "HyBlk"), ("truth-block", "TrBlk"), ("full", "Fu")):
            g = d[d.policy == policy]
            if not g.empty:
                n[f"Sim{K}{macro}"] = f"{geo(g.speedup):.3f}"
                n[f"Sim{K}{macro}Rb"] = f"{int(g.rollbacks.sum()):,}".replace(",", "{,}")
        p = d.pivot(index="name", columns="policy", values="speedup")
        n[f"Sim{K}Programs"] = len(p)
        n[f"Sim{K}Blocks"] = f"{int(d[d.policy == 'sequential'].blocks.sum()):,}".replace(",", "{,}")
        hb = d[d.policy == "hybrid-block"].set_index("name")
        h = d[d.policy == "hybrid"].set_index("name")
        n[f"Sim{K}Refined"] = f"{100 * (1 - hb.whole_preds.sum() / h.whole_preds.sum()):.0f}\\%"
        for prog, macro in (("polybench-lu", "Lu"), ("polybench-fdtd-2d", "Fdtd"), ("polybench-3dconv", "Conv"),
                            ("graphbig-bfs-topo-atomic", "Atomic"), ("graphbig-bfs-topo-thread-centric", "Topo"),
                            ("rodinia-srad", "Srad")):
            if prog in p.index:
                for policy, m2 in (("buffer", "Buf"), ("buffer-rw", "BufRw"), ("hybrid", "Hy"),
                                   ("hybrid-block", "HyBlk"), ("truth-block", "TrBlk")):
                    if policy in p:
                        n[f"Sim{K}{macro}{m2}"] = f"{p.loc[prog, policy]:.2f}"
        if "full" in p:
            rb = d[(d.policy == "full")]
            n[f"Sim{K}FuSlower"] = int((rb.speedup < 0.999).sum())
            n[f"Sim{K}FuRbProgs"] = int((rb.rollbacks > 0).sum())
            worst = rb.sort_values("speedup").iloc[0]
            n[f"Sim{K}FuWorst"] = f"{worst.speedup:.2f}"
            n[f"Sim{K}FuWorstName"] = worst["name"].split("-", 1)[1]
    sw = load("sim_delay.csv")
    if sw is not None:
        for policy, macro in (("full", "Fu"), ("hybrid", "Hy")):
            g = sw[sw.policy == policy]
            for dl, word in ((0, "None"), (200, "Max")):
                n[f"SimDelay{macro}{word}"] = f"{geo(g[g.delay_us == dl].speedup):.2f}"
    import json
    gaps = os.path.join(DATA, "sim_gaps.json")
    if os.path.exists(gaps):
        allg = [x for v in json.load(open(gaps)).values() for x in v]
        n["SimDelayUs"] = f"{np.median(allg) / 1000:.1f}"
    return n


def main():
    summary_table()


if __name__ == "__main__":
    main()
    print(numbers())
