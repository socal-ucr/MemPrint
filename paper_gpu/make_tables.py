"""Tables for paper_gpu/main.tex (LaTeX fragments in paper_gpu/tables/), from paper_gpu/data.

    python paper_gpu/make_tables.py
"""

import os
import re

import pandas as pd

import bench_tables  # noqa: E402  (paper_gpu/bench_tables.py)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CUDA = os.path.join(HERE, "..", "tests", "static", "cuda")
OUT = os.path.join(HERE, "tables")

ORDER = ["chain", "reduce", "stencil", "scatter", "fileidx", "histogram", "maybe", "bfs", "rodinia_bfs", "wl_bfs",
         "ptrsel", "ptrsel2", "symbols", "cc"]
WHAT = {  # program -> what it exercises
    "chain": "elementwise kernels in a chain",
    "reduce": "a reduction through shared memory",
    "stencil": "a stencil repeated over two buffers",
    "scatter": "writes at indices built on the host",
    "fileidx": "indices and weights read from files",
    "histogram": "atomic updates of bins chosen by the data",
    "maybe": "a branch on values the interpreter does not model",
    "bfs": "breadth-first search, one kernel per level",
    "rodinia_bfs": "breadth-first search in the style of Rodinia",
    "wl_bfs": "breadth-first search with a frontier queue",
    "ptrsel": "a buffer chosen by the data through a pointer table",
    "ptrsel2": "pointer fields in structs, pointers in variables",
    "cc": "connected components, labels lowered in place",
    "symbols": "device and constant variables, symbol copies",
}


def tt(name):
    return r"\texttt{" + name.replace("_", r"\_") + "}"


def f2(x):
    return f"{x:.2f}"


def write(name, body):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(body)
    print("wrote", name)


def programs_table():
    sc = pd.read_csv(os.path.join(DATA, "scores.csv")).set_index("program")
    rows = []
    for p in ORDER:
        src = open(os.path.join(CUDA, f"{p}.cu")).read()
        lines = sum(1 for l in src.splitlines() if l.strip() and not l.strip().startswith("//"))
        kernels = len(re.findall(r"__global__", src))
        ev = pd.read_csv(os.path.join(DATA, "runs", "truth", p, "static", f"{p}_events.csv"))
        rows.append(f"{tt(p)} & {WHAT[p]} & {lines} & {kernels} & {len(ev)} & {sc.loc[p, 'truth']} \\\\")
    write("programs.tex", "\\begin{tabular}{llrrrr}\n\\toprule\n"
          "Program & What it exercises & Lines & Kernels & Operations & Dependences \\\\\n\\midrule\n" +
          "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def accuracy_table():
    sc = pd.read_csv(os.path.join(DATA, "scores.csv")).set_index("program")
    hy_path = os.path.join(DATA, "hybrid_scores.csv")
    hy = pd.read_csv(hy_path).set_index("program") if os.path.exists(hy_path) else None
    rg = pd.read_csv(os.path.join(DATA, "ranges.csv")).set_index("program")
    rows = []
    for p in ORDER:
        s, r = sc.loc[p], rg.loc[p]
        cells = [tt(p), str(int(s.must)), str(int(s.may)), f2(s.must_precision), f2(s.recall), f2(s.must_recall)]
        if hy is not None and p in hy.index and not pd.isna(hy.loc[p].get("must")):
            h = hy.loc[p]
            cells += [str(int(h.must)), str(int(h.may)), f2(h.must_precision), f2(h.recall), f2(h.must_recall)]
        else:
            cells += ["--"] * 5
        cells += [f"{int(r.exact_sets)}/{int(r.sets)}", f2(r.unit_precision), f2(r.unit_recall)]
        rows.append(" & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{lrrrrr|rrrrr|rrr}" "\n" r"\toprule" "\n"
            r" & \multicolumn{5}{c|}{Whole program} & \multicolumn{5}{c|}{Hybrid} & "
            r"\multicolumn{3}{c}{Ranges (whole program)} \\" "\n"
            r"Program & Cert. & Poss. & \shortstack{Cert.\\prec.} & Recall & \shortstack{Cert.\\recall} & "
            r"Cert. & Poss. & \shortstack{Cert.\\prec.} & Recall & \shortstack{Cert.\\recall} & "
            r"\shortstack{Exact\\sets} & \shortstack{Unit\\prec.} & \shortstack{Unit\\recall} \\" "\n" r"\midrule" "\n")
    write("accuracy.tex", head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


def datadep_table():
    d = pd.read_csv(os.path.join(DATA, "datadep.csv")).set_index("program")
    rows = []
    for p in ORDER:
        r = d.loc[p]
        if r.data_deps == 0:
            continue
        rows.append(f"{tt(p)} & {r.kernel_deps} & {r.data_deps} & {r.data_must} & {r.data_must_traced} & "
                    f"{r.data_traced} \\\\")
    t = d.sum()
    rows.append(f"\\midrule\nTotal & {t.kernel_deps} & {t.data_deps} & {t.data_must} & {t.data_must_traced} & "
                f"{t.data_traced} \\\\")
    write("datadep.tex", "\\begin{tabular}{lrrrrr}\n\\toprule\n"
          "Program & \\shortstack{Between\\\\kernels} & \\shortstack{Through data-\\\\dependent bytes} & "
          "\\shortstack{Reported\\\\certain} & \\shortstack{Certain and\\\\in the trace} & "
          "\\shortstack{In the trace\\\\(any)} \\\\\n\\midrule\n" + "\n".join(rows) +
          "\n\\bottomrule\n\\end{tabular}\n")


def ablation_table():
    d = pd.read_csv(os.path.join(DATA, "ablation.csv"))
    rows = []
    for (ab, p), g in d.groupby(["ablation", "program"], sort=False):
        for _, r in g.iterrows():
            label = "Full interpreter" if r.variant == "full" else \
                {"no thread splitting": "Without thread splitting", "lockstep only": "Lockstep schedule only"}[r.variant]
            rows.append(" & ".join([tt(p), label, str(r.must), str(r.may), f2(r.must_precision), f2(r.recall),
                                    f2(r.must_recall), f2(r.needed_must_recall), f2(r.units_error)]) + " \\\\")
        rows.append("\\addlinespace")
    write("ablation.tex", "\\begin{tabular}{llrrrrrrr}\n\\toprule\n"
          "Program & Interpreter & Certain & Possible & \\shortstack{Certain\\\\precision} & Recall & "
          "\\shortstack{Certain\\\\recall} & \\shortstack{Needed cert.\\\\recall} & \\shortstack{Units\\\\error} "
          "\\\\\n\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


def overhead_table():
    path = os.path.join(DATA, "overhead.csv")
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    sch = [c for c in d.columns if c.startswith("interp_") and c.endswith("_s") and c != "interp_kernels_s"]
    d["all"] = d[sch].sum(axis=1)
    rows = []
    for p in ["chain", "bfs", "rodinia_bfs", "wl_bfs", "cc", "ptrsel"]:
        for _, r in d[d.program == p].sort_values("n").iterrows():
            rows.append(" & ".join([tt(p), f"{r.n:,}", f"{r.threads:,}", str(r.events), f"{r.native_s:.2f}",
                                    "--" if pd.isna(r.nvbit_s) else f"{r.nvbit_s:.2f}", f"{r.interp_lockstep_s:.2f}",
                                    f"{100 * r.interp_kernels_s / r.interp_lockstep_s:.0f}\\%",
                                    f"{r['all']:.1f}", f"{r.hybrid_lockstep_s:.2f}", f"{r.hybrid_all_s:.1f}"]) + " \\\\")
        rows.append("\\addlinespace")
    write("overhead.tex", "\\begin{tabular}{lrrrrrrrrrr}\n\\toprule\n"
          " & & & & \\multicolumn{2}{c}{GPU (s)} & \\multicolumn{3}{c}{Whole program} & "
          "\\multicolumn{2}{c}{Hybrid} \\\\\n"
          "\\cmidrule(lr){5-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-11}\n"
          "Program & Input & \\shortstack{GPU\\\\threads} & Operations & Run & \\shortstack{Under\\\\NVBit} & "
          "\\shortstack{Lockstep\\\\(s)} & \\shortstack{In\\\\kernels} & \\shortstack{Four\\\\orders (s)} & "
          "\\shortstack{Lockstep\\\\(s)} & \\shortstack{Four\\\\orders (s)} \\\\\n\\midrule\n" + "\n".join(rows[:-1]) +
          "\n\\bottomrule\n\\end{tabular}\n")


def granularity_table():
    d = pd.read_csv(os.path.join(DATA, "granularity.csv")).set_index("program")
    rows = []
    for p in ORDER:
        r = d.loc[p]
        rows.append(f"{tt(p)} & {r.exact_pairs} & {r.exact_needed} & {r.interval_pairs} & {r.interval_needed} & "
                    f"{r.allocation_pairs} & {r.allocation_needed} \\\\")
    write("granularity.tex", "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
          " & \\multicolumn{2}{c}{Exact units} & \\multicolumn{2}{c}{One interval} & "
          "\\multicolumn{2}{c}{Whole allocation} \\\\\n"
          "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n"
          "Program & Pairs & Needed & Pairs & Needed & Pairs & Needed \\\\\n\\midrule\n" +
          "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def patterns_table():
    d = pd.read_csv(os.path.join(DATA, "patterns.csv"))
    rows = [f"{r.input} & {f'{r.vertices:,}' if r.vertices else '--'} & {f'{r.edges:,}' if r.edges else '--'} & "
            f"{r.events} & {r.dependences} & {r.needed} & "
            f"{'--' if pd.isna(r.exact_shared_with_all) else int(r.exact_shared_with_all)} & "
            f"{r.needed_patterns_in_all}/{r.needed_patterns} \\\\" for _, r in d.iterrows()]
    write("patterns.tex", "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
          "Input & Vertices & Edges & Operations & Dependences & Needed & "
          "\\shortstack{Same bytes\\\\in all graphs} & \\shortstack{Needed patterns\\\\in all graphs} \\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def numbers():
    """paper_gpu/numbers.tex: the numbers the text quotes, as macros."""
    sc = pd.read_csv(os.path.join(DATA, "scores.csv")).set_index("program")
    cc = pd.read_csv(os.path.join(DATA, "cc_runs.csv"))
    tr = cc[cc.kind == "traced"]
    gpu = cc[cc.kind.isin(["traced", "plain"])].rounds
    interp = cc[cc.kind.str.startswith("interp_")].rounds
    dd = pd.read_csv(os.path.join(DATA, "datadep.csv")).set_index("program")
    ab = pd.read_csv(os.path.join(DATA, "ablation.csv"))
    pt = pd.read_csv(os.path.join(DATA, "patterns.csv")).set_index("input")
    pct = lambda x: f"{100 * x:.0f}\\%"  # noqa: E731
    n = {
        "Programs": len(sc), "MustTotal": f"{sc.must.sum():,}".replace(",", "{,}"),
        "CcRecall": f2(sc.loc["cc", "recall"]),
        "CcRecallLo": f"{tr.recall.min():.3f}", "CcRecallHi": f"{tr.recall.max():.3f}",
        "CcPossible": f"{int(sc.loc['cc', 'may']):,}".replace(",", "{,}"), "CcMust": int(sc.loc["cc", "must"]),
        "CcTruth": int(sc.loc["cc", "truth"]),
        "CcMustRecallLo": f2(tr.must_recall.min()), "CcMustRecallHi": f2(tr.must_recall.max()),
        "CcLockPrecLo": f2(tr.lockstep_must_precision.min()), "CcLockPrecHi": f2(tr.lockstep_must_precision.max()),
        "CcLockFalseLo": pct(1 - tr.lockstep_must_precision.max()),
        "CcLockFalseHi": pct(1 - tr.lockstep_must_precision.min()),
        "CcPlain": int((cc.kind == "plain").sum()), "CcReps": len(tr), "CcGpuRuns": len(gpu),
        "CcGpuRoundsLo": int(gpu.min()), "CcGpuRoundsHi": int(gpu.max()),
        "CcInterpRoundsLo": int(interp.min()), "CcInterpRoundsHi": int(interp.max()),
        "DataDeps": f"{dd.data_deps.sum():,}".replace(",", "{,}"),
        "DataMust": f"{dd.data_must.sum():,}".replace(",", "{,}"),
        "BfsDeps": int(dd.loc[["bfs", "rodinia_bfs", "wl_bfs"], "data_deps"].sum()),
        "PtrselNoSplitMust": int(ab[(ab.program == "ptrsel") & (ab.variant != "full")].must.iloc[0]),
        "NoInputPatterns": int(pt.loc["no input", "needed_patterns_in_all"]),
        "NoInputPatternsOf": int(pt.loc["no input", "needed_patterns"]),
    }
    path = os.path.join(DATA, "overhead.csv")
    if os.path.exists(path):
        d = pd.read_csv(path)
        sch = [c for c in d.columns if c.startswith("interp_") and c.endswith("_s") and c != "interp_kernels_s"]
        ratio = d.interp_lockstep_s / d.native_s
        kratio = d.interp_kernels_s / d.native_s
        sx = d[sch].sum(axis=1) / d.interp_lockstep_s
        host = 1 - d.interp_kernels_s / d.interp_lockstep_s
        big = d.loc[d.groupby("program").n.idxmax()].set_index("program")
        hyr = d.hybrid_lockstep_s / d.native_s
        hya = d.hybrid_all_s / d.native_s
        n.update({"HyOverheadLo": f"{hyr.min():.1f}", "HyOverheadHi": f"{hyr.max():.0f}",
                  "HyAllLo": f"{hya.min():.1f}", "HyAllHi": f"{hya.max():.0f}",
                  "HyAllS": f"{d.hybrid_all_s.max():.0f}", "NvbitMissing": int(d.nvbit_s.isna().sum()),
                  "HyOverheadMedian": f"{hyr.median():.1f}",
                  "HyLockLoS": f"{d.hybrid_lockstep_s.min():.1f}", "HyLockHiS": f"{d.hybrid_lockstep_s.max():.1f}",
                  "HyOrdersLo": f"{(d.hybrid_all_s / d.hybrid_lockstep_s).min():.1f}",
                  "HyOrdersHi": f"{(d.hybrid_all_s / d.hybrid_lockstep_s).max():.0f}",
                  "BfsHyLockS": f"{big.loc['bfs', 'hybrid_lockstep_s']:.1f}",
                  "WlbfsHyLockS": f"{big.loc['wl_bfs', 'hybrid_lockstep_s']:.1f}",
                  "WlbfsHyAllS": f"{big.loc['wl_bfs', 'hybrid_all_s']:.0f}",
                  "RbfsHyAllS": f"{big.loc['rodinia_bfs', 'hybrid_all_s']:.0f}",
                  "RbfsAllS": f"{big.loc['rodinia_bfs', ['interp_lockstep_s', 'interp_blocks_s', 'interp_reverse_s', 'interp_random_s']].sum():.0f}"})

        def at(p, col, fmt):  # a value of the largest input of p, while overhead.py is still running
            return fmt.format(big.loc[p, col]) if p in big.index else "??"
        n.update({"OverheadLo": f"{ratio.min():.1f}", "OverheadHi": f"{ratio.max():.0f}",
                  "KernelRatioLo": f"{kratio.min():.2f}", "KernelRatioHi": f"{kratio.max():.1f}",
                  "KernelRatioMedian": f"{kratio.median():.1f}",
                  "ScheduleCost": f"{sx.mean():.1f}", "ScheduleLo": f"{sx.min():.1f}", "ScheduleHi": f"{sx.max():.0f}",
                  "NvbitLo": f"{(d.nvbit_s / d.native_s).min():.1f}", "NvbitHi": f"{(d.nvbit_s / d.native_s).max():.1f}",
                  "NativeLo": f"{d.native_s.min():.2f}", "NativeHi": f"{d.native_s.max():.2f}",
                  "ChainThreads": at("chain", "threads", "{:,.0f}").replace(",", "{,}"),
                  "ChainKernelS": at("chain", "interp_kernels_s", "{:.1f}"),
                  "RbfsThreads": at("rodinia_bfs", "threads", "{:,.0f}").replace(",", "{,}"),
                  "RbfsLockS": at("rodinia_bfs", "interp_lockstep_s", "{:.1f}"),
                  "RbfsNvbitS": at("rodinia_bfs", "nvbit_s", "{:.1f}"),
                  "WlbfsLockS": at("wl_bfs", "interp_lockstep_s", "{:.0f}"),
                  "WlbfsKernelS": at("wl_bfs", "interp_kernels_s", "{:.2f}"),
                  "BfsLockS": at("bfs", "interp_lockstep_s", "{:.0f}"),
                  "BfsKernelS": at("bfs", "interp_kernels_s", "{:.1f}"),
                  "HostShareMax": f"{100 * host.max():.1f}\\%"})
    else:
        raise SystemExit("data/overhead.csv is missing (paper_gpu/overhead.py)")
    if os.path.exists(os.path.join(DATA, "benchmarks.csv")):
        n.update(bench_tables.numbers())
    body = "".join(f"\\newcommand{{\\num{k}}}{{{v}}}\n" for k, v in n.items())
    with open(os.path.join(HERE, "numbers.tex"), "w") as f:
        f.write(body)
    print("wrote numbers.tex")


def main():
    os.makedirs(OUT, exist_ok=True)
    bench_tables.main()
    programs_table()
    accuracy_table()
    datadep_table()
    ablation_table()
    overhead_table()
    granularity_table()
    patterns_table()
    numbers()


if __name__ == "__main__":
    main()
