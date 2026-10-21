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
    for label, col in (("Geometric mean, all", None),):
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


COST = (("small", "block_overhead.csv"), ("large", "block_overhead_large.csv"))


def cost_table():
    """What dependences between blocks cost over dependences between operations (block_overhead.py)."""
    frames = [(k, load(f)) for k, f in COST]
    lines = []

    def row(label, fn):
        cells = [label]
        for _, d in frames:
            cells.append(fn(d) if d is not None else "--")
        lines.append(" & ".join(cells) + r" \\")

    short = lambda n: r"\texttt{" + n.split("-", 1)[1] + "}"  # noqa: E731
    row("Interpreter time, operations (s, total)", lambda d: f"{d.ops_s.sum():,.0f}")
    row("Interpreter time, blocks (s, total)", lambda d: f"{d.blocks_s.sum():,.0f}")
    row("\\quad ratio, median / geometric mean", lambda d: f"{(d.blocks_s / d.ops_s).median():.2f} / "
        f"{geo(d.blocks_s / d.ops_s):.2f}")
    row("\\quad ratio, largest", lambda d: f"{(d.blocks_s / d.ops_s).max():.1f} "
        f"({short(d.loc[(d.blocks_s / d.ops_s).idxmax(), 'name'])})")
    row("Peak memory ratio, median", lambda d: f"{(d.blocks_rss_mb / d.ops_rss_mb).median():.2f}")
    row("\\quad largest", lambda d: f"{(d.blocks_rss_mb / d.ops_rss_mb).max():.1f} "
        f"({short(d.loc[(d.blocks_rss_mb / d.ops_rss_mb).idxmax(), 'name'])})")
    row("\\quad most memory, operations / blocks (GB)", lambda d: f"{d.ops_rss_mb.max() / 1024:.1f} / "
        f"{d.blocks_rss_mb.max() / 1024:.1f}")
    row("Output, operations / blocks (MB, total)", lambda d: f"{d.ops_out_mb.sum():.1f} / {d.blocks_out_mb.sum():.0f}")
    lines.append(r"\midrule")
    if all(d is not None and "orders_ops_s" in d for _, d in frames):
        row("Four orders, operations (s, total)", lambda d: f"{d.orders_ops_s.sum():,.0f}")
        row("Four orders, blocks (s, total)", lambda d: f"{d.orders_blocks_s.sum():,.0f}")
        row("\\quad ratio, median / largest", lambda d: f"{(d.orders_blocks_s / d.orders_ops_s).median():.2f} / "
            f"{(d.orders_blocks_s / d.orders_ops_s).max():.1f}")
        row("\\quad orders one after another (s, total)", lambda d: f"{d.orders_blocks_serial_s.sum():,.0f}")
        lines.append(r"\midrule")
    row("Refining to blocks (s, total)", lambda d: f"{d.refine_s.sum():.0f}")
    row("\\quad share of the four-order run, median / largest",
        lambda d: f"{100 * (d.refine_s / d.hybrid_all_s).median():.1f}\\% / "
        f"{100 * (d.refine_s / d.hybrid_all_s).max():.0f}\\%")
    row("Dependences between operations", lambda d: f"{int(d.op_deps.sum()):,}")
    row("Dependences between blocks", lambda d: f"{int(d.block_deps.sum()):,}")
    row("\\quad and between whole operations", lambda d: f"{int(d.whole_deps.sum()):,}")
    if all(d is not None and "block_ints" in d for _, d in frames):
        row("\\quad stored as ranges and runs (integers)", lambda d: f"{int(d.block_ints.sum()):,}")
    head = (r"\begin{tabular}{lrr}" "\n" r"\toprule" "\n" r" & Small inputs & Large inputs \\" "\n" r"\midrule" "\n")
    write("block_cost.tex", head + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


DEDUP = (("small", "block_overhead_nodedup.csv", "block_overhead.csv"),
         ("large", "block_overhead_nodedup_large.csv", "block_overhead_large.csv"))


def dedup_table():
    """Block recording before and after deduplicating references as they come: the lockstep run with --blocks
    against without, and the programs that needed the most memory before."""
    runs = [(k, load(a), load(b)) for k, a, b in DEDUP]
    lines = []

    def row(label, fn):
        cells = [label]
        for _, before, after in runs:
            for d in (before, after):
                cells.append(fn(d) if d is not None else "--")
        lines.append(" & ".join(cells) + r" \\")

    r = lambda d: d.blocks_s / d.ops_s  # noqa: E731
    m = lambda d: d.blocks_rss_mb / d.ops_rss_mb  # noqa: E731
    row("Interpreter time, median ratio", lambda d: f"{r(d).median():.2f}")
    row("\\quad largest", lambda d: f"{r(d).max():.1f}")
    row("Peak memory, median ratio", lambda d: f"{m(d).median():.2f}")
    row("\\quad largest", lambda d: f"{m(d).max():.1f}")
    row("\\quad most of any program (GB)", lambda d: f"{d.blocks_rss_mb.max() / 1024:.1f}")
    lines.append(r"\midrule")
    before = runs[-1][1].set_index("name")
    after = runs[-1][2].set_index("name")
    for n in before.blocks_rss_mb.sort_values(ascending=False).index[:5]:
        cells = [r"\quad\texttt{" + n.split("-", 1)[1] + "} (GB)", "", ""]
        cells += [f"{before.loc[n, 'blocks_rss_mb'] / 1024:.1f}", f"{after.loc[n, 'blocks_rss_mb'] / 1024:.1f}"]
        lines.append(" & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{lrrrr}" "\n" r"\toprule" "\n"
            r" & \multicolumn{2}{c}{Small inputs} & \multicolumn{2}{c}{Large inputs} \\" "\n"
            r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}" "\n"
            r" & Kept & Merged & Kept & Merged \\" "\n" r"\midrule" "\n")
    lines.insert(6, r"\multicolumn{5}{l}{Peak memory with block references, the most memory-hungry programs:} \\")
    write("block_dedup.tex", head + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


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
        n[f"Sim{K}Refined"] = f"{100 * (1 - hb.whole_preds.sum() / h.whole_preds.sum()):.1f}\\%"
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
    for k, f in COST:
        d = load(f)
        if d is None:
            continue
        K = "Small" if k == "small" else "Large"
        r, m = d.blocks_s / d.ops_s, d.blocks_rss_mb / d.ops_rss_mb
        n[f"Cost{K}TimeMed"] = f"{r.median():.2f}"
        n[f"Cost{K}TimeGeo"] = f"{geo(r):.2f}"
        n[f"Cost{K}TimeMax"] = f"{r.max():.1f}"
        n[f"Cost{K}TimeMaxName"] = d.loc[r.idxmax(), "name"].split("-", 1)[1]
        n[f"Cost{K}MemMed"] = f"{m.median():.2f}"
        n[f"Cost{K}MemMax"] = f"{m.max():.0f}"
        n[f"Cost{K}MemMaxName"] = d.loc[m.idxmax(), "name"].split("-", 1)[1]
        n[f"Cost{K}MemMaxGB"] = f"{d.blocks_rss_mb.max() / 1024:.0f}"
        n[f"Cost{K}RefineShare"] = f"{100 * (d.refine_s / d.hybrid_all_s).median():.1f}\\%"
        n[f"Cost{K}RefineShareMax"] = f"{100 * (d.refine_s / d.hybrid_all_s).max():.0f}\\%"
        n[f"Cost{K}OpDeps"] = f"{int(d.op_deps.sum()):,}".replace(",", "{,}")
        n[f"Cost{K}BlockDeps"] = f"{d.block_deps.sum() / 1e6:.1f}"
        n[f"Cost{K}OutMB"] = f"{d.blocks_out_mb.sum():.0f}"
        if "orders_ops_s" in d:
            n[f"Cost{K}OrdersMed"] = f"{(d.orders_blocks_s / d.orders_ops_s).median():.2f}"
            n[f"Cost{K}OrdersMax"] = f"{(d.orders_blocks_s / d.orders_ops_s).max():.1f}"
            n[f"Cost{K}Parallel"] = f"{d.orders_blocks_serial_s.sum() / d.orders_blocks_s.sum():.1f}"
        if "block_ints" in d:
            n[f"Cost{K}Compact"] = f"{2 * d.block_deps.sum() / d.block_ints.sum():.1f}"
    for k, a, b in DEDUP:
        before, after = load(a), load(b)
        if before is None or after is None:
            continue
        K = "Small" if k == "small" else "Large"
        for lab, d in (("Kept", before), ("Merged", after)):
            n[f"Dedup{K}{lab}MaxGB"] = f"{d.blocks_rss_mb.max() / 1024:.1f}"
            n[f"Dedup{K}{lab}MemMax"] = f"{(d.blocks_rss_mb / d.ops_rss_mb).max():.1f}"
            n[f"Dedup{K}{lab}TimeMed"] = f"{(d.blocks_s / d.ops_s).median():.2f}"
            n[f"Dedup{K}{lab}TimeMax"] = f"{(d.blocks_s / d.ops_s).max():.1f}"
        worst = before.set_index("name").blocks_rss_mb.idxmax()
        n[f"Dedup{K}WorstName"] = worst.split("-", 1)[1]
    n.update(transfer_numbers())
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
    cost_table()
    dedup_table()
    transfer_table()


if __name__ == "__main__":
    main()
    print(numbers())


# ---------------------------------------------------------------- dependences before the input is known

def transfer_table():
    """Patterns learned on one input applied to the other (transfer.py), against the policies of simulate.py."""
    d = load("transfer.csv")
    if d is None:
        return
    rows = []
    for train, target, f in (("small", "large", "sim_large.csv"), ("large", "small", "sim.csv")):
        t = d[d.train == train].set_index("name")
        s = load(f)
        p = s.pivot(index="name", columns="policy", values="speedup").loc[t.index] if s is not None else None
        cells = [f"{train.capitalize()} $\\rightarrow$ {target}", f"{int((t.missed == 0).sum())} / {len(t)}",
                 f"{int(t.missed.sum()):,}", f"{int(t.extra.sum()):,}", f"{int(t.patterns_rollbacks.sum())}",
                 f"{geo(t.patterns_speedup):.3f}"]
        cells += [f"{geo(p[c]):.3f}" for c in ("buffer-rw", "hybrid")] if p is not None else ["--", "--"]
        cells.append(f"{geo(t.truth_speedup):.3f}")
        rows.append(" & ".join(cells) + r" \\")
    head = ("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
            " & & & & & \\multicolumn{4}{c}{Speedup} \\\\\n\\cmidrule(lr){6-9}\n"
            "Learned $\\rightarrow$ applied & \\shortstack{No missed\\\\dependence} & Missed & Extra & Rollb. & "
            "Patterns & \\shortstack{Same buffer,\\\\one writes} & \\shortstack{Hybrid on\\\\the input} & Traced \\\\\n"
            "\\midrule\n")
    write("transfer.tex", head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def transfer_numbers():
    d = load("transfer.csv")
    n = {}
    if d is None:
        return n
    for train, K in (("small", "SmallLarge"), ("large", "LargeSmall")):
        t = d[d.train == train]
        n[f"Tr{K}NoMiss"] = int((t.missed == 0).sum())
        n[f"Tr{K}Rb"] = int(t.patterns_rollbacks.sum())
        n[f"Tr{K}Speedup"] = f"{geo(t.patterns_speedup):.3f}"
        n[f"Tr{K}Uncovered"] = int((t.uncovered_ops > 0).sum())
        sim = load("sim_large.csv" if train == "small" else "sim.csv")
        if sim is not None:
            p = sim.pivot(index="name", columns="policy", values="speedup").loc[t.set_index("name").index]
            n[f"Tr{K}BufRw"] = f"{geo(p['buffer-rw']):.3f}"
            n[f"Tr{K}Hy"] = f"{geo(p.hybrid):.3f}"
    n["TrPrograms"] = d.name.nunique()
    return n
