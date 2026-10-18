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


def _gains(name):
    d = load(name)
    if d is None:
        return None
    p = d.pivot(index="name", columns="policy", values="speedup")
    return p


def block_table():
    """Programs where refining to blocks changes the speedup by more than half a percent, in either input size:
    the hybrid interpreter between operations and between blocks, and the traced dependences between blocks."""
    runs = [(k, _gains(f)) for k, f in RUNS]
    names = set()
    for _, p in runs:
        if p is not None:
            gain = (p["hybrid-block"] / p["hybrid"] - 1).abs() > 0.005
            tgain = (p["truth-block"] / p["truth"] - 1).abs() > 0.005
            names |= set(p.index[gain | tgain])
    rows = []
    order = sorted(names, key=lambda n: -max(p.loc[n, "truth-block"] / p.loc[n, "truth"] for _, p in runs
                                             if p is not None and n in p.index))
    for n in order:
        cells = [r"\texttt{" + n.split("-", 1)[1] + "}"]
        for _, p in runs:
            if p is None or n not in p.index:
                cells += ["--"] * 4
                continue
            r = p.loc[n]
            cells += [f"{r.hybrid:.2f}", f"{r['hybrid-block']:.2f}", f"{100 * (r['hybrid-block'] / r.hybrid - 1):+.1f}\\%",
                      f"{r['truth-block']:.2f}"]
        rows.append(" & ".join(cells) + r" \\")
    for label, col in (("Geometric mean, all 37", None),):
        cells = [label]
        for _, p in runs:
            cells += [f"{geo(p.hybrid):.3f}", f"{geo(p['hybrid-block']):.3f}",
                      f"{100 * (geo(p['hybrid-block']) / geo(p.hybrid) - 1):+.1f}\\%", f"{geo(p['truth-block']):.3f}"]
        rows.append(r"\midrule" "\n" + " & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{lrrrrrrrr}" "\n" r"\toprule" "\n"
            r" & \multicolumn{4}{c}{Small inputs} & \multicolumn{4}{c}{Large inputs} \\" "\n"
            r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}" "\n"
            r"Program & \shortstack{Opera-\\tions} & Blocks & Gain & \shortstack{Traced,\\blocks} & "
            r"\shortstack{Opera-\\tions} & Blocks & Gain & \shortstack{Traced,\\blocks} \\" "\n" r"\midrule" "\n")
    write("sim_blocks.tex", head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


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
    for k, f in (("Small", "sim_blocks"), ("Large", "sim_blocks_large")):
        p, q = _gains(f + ".csv"), _gains(f + "_equal.csv")
        if p is None:
            continue
        n[f"Sim{K}BlkGain".replace(K, k)] = f"{100 * (geo(p['hybrid-block']) / geo(p.hybrid) - 1):.1f}\\%"
        n[f"Sim{k}TrBlkGain"] = f"{100 * (geo(p['truth-block']) / geo(p.truth) - 1):.1f}\\%"
        n[f"Sim{k}BlkProgs"] = int(((p["hybrid-block"] / p.hybrid - 1) > 0.005).sum())
        n[f"Sim{k}BlkMax"] = f"{100 * (p['hybrid-block'] / p.hybrid - 1).max():.0f}\\%"
        if q is not None:
            n[f"Sim{k}EqHyBlk"] = f"{geo(q['hybrid-block']):.3f}"
            n[f"Sim{k}EqBlkGain"] = f"{100 * (geo(q['hybrid-block']) / geo(q.hybrid) - 1):.1f}\\%"
            worst = (q["hybrid-block"] / q.hybrid - 1)
            n[f"Sim{k}EqWorst"] = f"{-100 * worst.min():.1f}\\%"
            n[f"Sim{k}EqWorstName"] = worst.idxmin().split("-", 1)[1]
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
    block_table()


if __name__ == "__main__":
    main()
    print(numbers())
