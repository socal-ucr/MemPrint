"""Numbers and tables for paper_footprint/main.tex, from paper_gpu/data/copy_savings{,_large}.csv
(paper_gpu/copy_savings.py) and paper_gpu/data/uvm*.csv (uvm_oversub.py, uvm_blocks.py, uvm_hw.py). Writes
numbers.tex and tables/*.tex here; writes nothing in paper_gpu.

    python paper_footprint/make_numbers.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "paper_gpu", "data")
OUT = os.path.join(HERE, "tables")
DEGENERATE = {"small": {"polybench-corr", "polybench-covar"}}   # most kernels launch no block on the small input
SIZES = (("small", "Small", "copy_savings.csv"), ("large", "Large", "copy_savings_large.csv"))
VARIANTS = (("", "Tr", "Traced"), ("_hy", "Hy", "Interpreter, all orders"), ("_lock", "Lock", "Interpreter, lockstep"))
sys.path.insert(0, os.path.join(HERE, "..", "paper_gpu"))
from copy_savings import fit  # noqa: E402
import uvm_costs  # noqa: E402


def load(size, f):
    d = pd.read_csv(os.path.join(DATA, f))
    d = d[d.status == "ok"]
    return d[~d.name.isin(DEGENERATE.get(size, ()))].reset_index(drop=True)


def share(d, v):
    """Per program, the share of its GPU operation time a variant's footprints save in copies."""
    return (d[f"h2d_saved_s{v}"] + d[f"d2h_saved_s{v}"]) / d.ops_s


def geo_speedup(s):
    return float((1 / (1 - s)).prod() ** (1 / len(s)))


def big(n):
    return f"{n:,}".replace(",", "{,}")


def short(name):
    return name.split("-", 1)[1]


def numbers():
    n = {}
    model = fit()
    for op, key in (("cudaMemcpy H2D", "HtoD"), ("cudaMemcpy D2H", "DtoH")):
        fixed, per_byte = model[op]
        n[f"Copy{key}FixedUs"] = f"{fixed / 1000:.1f}"
        n[f"Copy{key}GBs"] = f"{1 / per_byte:.1f}"
    for size, K, f in SIZES:
        d = load(size, f)
        n[f"{K}Programs"] = len(d)
        n[f"{K}CopyShareMed"] = f"{100 * ((d.h2d_s + d.d2h_s) / d.ops_s).median():.0f}\\%"
        for v, V, _ in VARIANTS:
            s = share(d, v)
            n[f"{K}{V}Geo"] = f"{geo_speedup(s):.3f}"
            n[f"{K}{V}Progs"] = int((s >= 0.01).sum())
            n[f"{K}{V}HtoDUnneeded"] = f"{100 * d[f'h2d_unneeded_bytes{v}'].sum() / d.h2d_bytes.sum():.1f}\\%"
            n[f"{K}{V}DtoHUnneeded"] = f"{100 * d[f'd2h_unneeded_bytes{v}'].sum() / d.d2h_bytes.sum():.1f}\\%"
            if v:
                bad = d[(d[f"h2d_unsafe_bytes{v}"] > 0) | (d[f"d2h_unsafe_bytes{v}"] > 0)]
                n[f"{K}{V}UnsafeProgs"] = len(bad)
                n[f"{K}{V}UnsafeBytes"] = big(int((bad[f"h2d_unsafe_bytes{v}"] + bad[f"d2h_unsafe_bytes{v}"]).sum()))
        s = share(d, "")
        top = s.idxmax()
        n[f"{K}TrMax"] = f"{100 * s[top]:.1f}\\%"
        n[f"{K}TrMaxName"] = short(d.loc[top, "name"])
        n[f"{K}TrMaxHy"] = f"{100 * share(d, '_hy')[top]:.1f}\\%"
        for prog, macro in (("graphbig-kcore", "Kcore"), ("rodinia-bfs", "Bfs")):
            r = d[d.name == prog]
            if len(r):
                i = r.index[0]
                n[f"{K}{macro}Tr"] = f"{100 * share(d, '')[i]:.1f}\\%"
                n[f"{K}{macro}Hy"] = f"{100 * share(d, '_hy')[i]:.1f}\\%"
                n[f"{K}{macro}HtoDUnneeded"] = f"{100 * r.h2d_unneeded_bytes.iloc[0] / r.h2d_bytes.iloc[0]:.0f}\\%"
    return n


def program_table():
    """Programs whose traced footprints save at least 1% of their GPU operation time on either input."""
    frames = {size: load(size, f).set_index("name") for size, _, f in SIZES}
    keep = set()
    for d in frames.values():
        s = share(d.reset_index(), "")
        keep |= set(d.index[s.values >= 0.01])
    order = sorted(keep, key=lambda x: -max(share(d.reset_index(), "")[list(d.index).index(x)]
                                            for d in frames.values() if x in d.index))
    rows = []
    for name in order:
        cells = [r"\texttt{" + short(name) + "}", frames["small"].loc[name, "suite"] if name in frames["small"].index
                 else frames["large"].loc[name, "suite"]]
        for size, _, _ in SIZES:
            d = frames[size]
            if name not in d.index:
                cells += ["--"] * 4
                continue
            r = d.loc[name]
            cells.append(f"{100 * (r.h2d_s + r.d2h_s) / r.ops_s:.0f}")
            for v in ("", "_hy", "_lock"):
                cells.append(f"{100 * (r[f'h2d_saved_s{v}'] + r[f'd2h_saved_s{v}']) / r.ops_s:.1f}")
        rows.append(" & ".join(cells).replace("PolyBench-GPU", "PolyBench") + r" \\")
    head = (r"\begin{tabular}{llrrrrrrrr}" "\n" r"\toprule" "\n"
            r" & & \multicolumn{4}{c}{Small inputs} & \multicolumn{4}{c}{Large inputs} \\" "\n"
            r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}" "\n"
            r"Program & Suite & Copies & Traced & \shortstack{All\\orders} & Lockstep & Copies & Traced & "
            r"\shortstack{All\\orders} & Lockstep \\" "\n" r"\midrule" "\n")
    write("programs.tex", head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


def summary_table(n):
    rows = []
    for v, V, label in VARIANTS:
        cells = [label]
        for _, K, _ in SIZES:
            cells += [n[f"{K}{V}HtoDUnneeded"], n[f"{K}{V}DtoHUnneeded"], n[f"{K}{V}Geo"], str(n[f"{K}{V}Progs"]),
                      str(n.get(f"{K}{V}UnsafeProgs", "--"))]
        rows.append(" & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{lrrrrrrrrrr}" "\n" r"\toprule" "\n"
            r" & \multicolumn{5}{c}{Small inputs (" + str(n["SmallPrograms"]) + r" programs)} & "
            r"\multicolumn{5}{c}{Large inputs (" + str(n["LargePrograms"]) + r" programs)} \\" "\n"
            r"\cmidrule(lr){2-6} \cmidrule(lr){7-11}" "\n"
            r"Footprints & \shortstack{HtoD\\unread} & \shortstack{DtoH\\unwritten} & Speedup & "
            r"\shortstack{$\geq$1\%\\programs} & Unsafe & \shortstack{HtoD\\unread} & \shortstack{DtoH\\unwritten} & "
            r"Speedup & \shortstack{$\geq$1\%\\programs} & Unsafe \\" "\n" r"\midrule" "\n")
    write("summary.tex", head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


# ---------------------------------------------------------------- Unified Memory (uvm_oversub.py, uvm_blocks.py)

UVM = (("KLarge", "uvm_large.csv", "large", "Kernels", "Large", "64~KB"),
       ("WLarge", "uvm_blocks_large.csv", "large", "Waves", "Large", "64~KB"),
       ("WLargeFour", "uvm_blocks_large_4k.csv", "large", "Waves", "Large", "4~KB"),
       ("WSmall", "uvm_blocks.csv", "small", "Waves", "Small", "64~KB"),
       ("WSmallFour", "uvm_blocks_4k.csv", "small", "Waves", "Small", "4~KB"))
UVM_POLICIES = (("buffer", "Buf"), ("hybrid", "Hy"), ("truth", "Tr"), ("hybrid_min", "HyMin"),
                ("truth_min", "TrMin"))
CHEAP_POLICIES = (("range", "Range"), ("range_min", "RangeMin"), ("history", "Hist"), ("history_min", "HistMin"))


def uvm_load(f, size):
    d = pd.read_csv(os.path.join(DATA, f))
    return d[~d.name.isin(DEGENERATE.get(size, ()))].reset_index(drop=True)


def uvm_speedups(d, costs="measured", fault_us=25, oversubscribed=True):
    """Per (program, share of memory), each policy's speedup over demand paging."""
    ok = d[(d.status == "ok") & (d.costs == costs)]
    if costs == "assumed":
        ok = ok[ok.fault_us.astype(float) == fault_us]
    ok = ok[ok.fraction < 1.0] if oversubscribed else ok[ok.fraction == 1.0]
    t = ok.pivot_table(index=["name", "fraction"], columns="policy", values="time_ns")
    return pd.DataFrame({p: t["demand"] / t[p] for p in t.columns if p != "demand"})


def uvm_numbers():
    n = {"UvmFaultOneUs": f"{uvm_costs.fault_ns(1) / 1000:.0f}",
         "UvmFaultSixteenUs": f"{uvm_costs.fault_ns(16) / 1000:.0f}",
         "UvmFaultPerBlockUs": f"{(uvm_costs.fault_ns(256) - uvm_costs.fault_ns(1)) / 255 / 1000:.0f}",
         "UvmFaultPerPageFourUs": f"{(uvm_costs.fault_ns(256, 4) - uvm_costs.fault_ns(1, 4)) / 255 / 1000:.1f}",
         "UvmPrefetchCallUs": f"{uvm_costs._prefetch()[0] / 1000:.0f}",
         "UvmPrefetchGBs": f"{1 / uvm_costs._prefetch()[1]:.1f}",
         "UvmMinRun": str(uvm_costs.min_run(64)), "UvmMinRunFour": str(uvm_costs.min_run(4))}
    for key, f, size, *_ in UVM:
        d = uvm_load(f, size)
        n[f"Uvm{key}Programs"] = d.name.nunique()
        s = uvm_speedups(d)
        n[f"Uvm{key}Over"] = s.index.get_level_values(0).nunique()
        n[f"Uvm{key}Configs"] = len(s)
        for p, P in UVM_POLICIES:
            n[f"Uvm{key}{P}"] = f"{geo_speedup_of(s[p]):.3f}"
        n[f"Uvm{key}HyMinBehind"] = int((s.hybrid_min < 0.99 * s.truth_min).sum())
        n[f"Uvm{key}HyMinLoses"] = int((s.hybrid_min < 0.99).sum())
        fit = uvm_speedups(d, oversubscribed=False)
        n[f"Uvm{key}FitHyMin"] = f"{geo_speedup_of(fit.hybrid_min):.3f}"
        for fu, F in ((10, "Ten"), (25, ""), (50, "Fifty")):
            a = uvm_speedups(d, "assumed", fu)
            n[f"Uvm{key}AssumedHy{F}"] = f"{geo_speedup_of(a.hybrid):.3f}"
            n[f"Uvm{key}AssumedBuf{F}"] = f"{geo_speedup_of(a.buffer):.3f}"
        ok = d[(d.status == "ok") & (d.costs == "measured") & (d.fraction < 1.0)]
        mv = (ok.to_device_bytes + ok.to_host_bytes).groupby([ok.name, ok.fraction, ok.policy]).sum().unstack("policy")
        n[f"Uvm{key}HyMinFewerBytes"] = f"{100 * (1 - mv.hybrid_min.sum() / mv.demand.sum()):.0f}\\%"
        # plans without the interpreter (uvm_cheap.py), from the runs of the same model with only these policies
        c = uvm_speedups(uvm_load(f.replace(".csv", "_cheap.csv"), size))
        assert np.allclose(c.truth_min.sort_index(), s.truth_min.sort_index()), key
        for p, P in CHEAP_POLICIES:
            n[f"Uvm{key}{P}"] = f"{geo_speedup_of(c[p]):.3f}"
        n[f"Uvm{key}RangeMinBehind"] = int((c.range_min < 0.99 * c.truth_min).sum())
        n[f"Uvm{key}RangeMinAhead"] = int((c.range_min > 1.01 * c.truth_min).sum())
        if key == "WLargeFour":
            path = c.loc["rodinia-pathfinder"]
            n["UvmPathTrMinLo"], n["UvmPathTrMinHi"] = f"{path.truth_min.min():.2f}", f"{path.truth_min.max():.2f}"
            n["UvmPathRangeMinLo"], n["UvmPathRangeMinHi"] = (f"{path.range_min.min():.2f}",
                                                              f"{path.range_min.max():.2f}")
            n["UvmRangeAheadOther"] = short(sorted({x for x, _ in c.index[c.range_min > 1.01 * c.truth_min]}
                                                   - {"rodinia-pathfinder"})[0])
    gaps = pd.read_csv(os.path.join(DATA, "uvm_span_gaps_large.csv"))
    gaps = gaps[gaps.gaps > 0].set_index(["name", "page_kb"])
    n["UvmGapProgramsSixtyFour"] = len(gaps.xs(64, level="page_kb"))
    n["UvmGapProgramsFour"] = len(gaps.xs(4, level="page_kb"))
    n["UvmGapSpanPrograms"] = pd.read_csv(os.path.join(DATA, "uvm_span_gaps_large.csv")).name.nunique()
    for (name, kb), r in gaps.iterrows():
        k = {"rodinia-bfs": "Bfs", "graphbig-bfs-topo-thread-centric": "GraphBfs"}[name]
        n[f"UvmGap{k}{'Four' if kb == 4 else 'SixtyFour'}"] = f"{100 * r.gaps / r.touched:.0f}\\%"
    d = uvm_load("uvm_blocks_large_4k.csv", "large")
    g = d[d.name.str.startswith("graphbig") & (d.status == "ok")]
    n["UvmGraphOver"] = int((g.fraction < 1.0).sum())
    n.update(hw_numbers())
    return n


def geo_speedup_of(s):
    return float(np.exp(np.log(np.asarray(s, dtype=float)).mean()))


def uvm_summary_table(n):
    rows = []
    for key, _, _, model, size, page in UVM:
        cells = [model, size, page, f"{n[f'Uvm{key}Over']} of {n[f'Uvm{key}Programs']}", str(n[f"Uvm{key}Configs"])]
        cells += [n[f"Uvm{key}{P}"] for _, P in UVM_POLICIES]
        cells += [n[f"Uvm{key}RangeMin"], n[f"Uvm{key}HistMin"]]
        cells += [n[f"Uvm{key}AssumedBuf"], n[f"Uvm{key}AssumedHy"]]
        rows.append(" & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{lllrrrrrrrrrrr}" "\n" r"\toprule" "\n"
            r" & & & & & \multicolumn{7}{c}{Measured costs} & \multicolumn{2}{c}{Assumed costs} \\" "\n"
            r"\cmidrule(lr){6-12} \cmidrule(lr){13-14}" "\n"
            r"Model & Inputs & Page & Programs & Shares & Buffers & \shortstack{Inter-\\preter} & Traced & "
            r"\shortstack{Inter-\\preter, runs} & \shortstack{Traced,\\runs} & \shortstack{Ranges,\\runs} & "
            r"\shortstack{History,\\runs} & Buffers & \shortstack{Inter-\\preter} \\"
            "\n" r"\midrule" "\n")
    write("uvm_summary.tex", head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


# ---------------------------------------------------------------- the A30 (uvm_hw.py)

HW_PLANS = (("buffer", "Buf"), ("hybrid", "Hy"), ("truth", "Tr"), ("hybrid_min", "HyMin"), ("truth_min", "TrMin"))


def hw_load():
    """Per (program, share, plan): the median time over the repetitions that ran (rc 0), with the number that failed,
    and whether the kernel-level model simulates the share (every operation's pages fit)."""
    d = pd.read_csv(os.path.join(DATA, "uvm_hw_large.csv"))
    d["t"] = d.gpu_ns + d.copy_ns
    fails = d[d.rc != 0].groupby(["name", "fraction"]).size()
    m = d[d.rc == 0].groupby(["name", "fraction", "plan"]).t.median().unstack("plan")
    model = uvm_load("uvm_large.csv", "large")
    fits = set(map(tuple, model[model.status == "ok"][["name", "fraction"]].drop_duplicates().values))
    m["fits"] = [i in fits for i in m.index]
    return d, m, fails


def hw_numbers():
    d, m, fails = hw_load()
    n = {"HwPrograms": d.name.nunique(), "HwReps": d.rep.nunique(),
         "HwFailRuns": int((d.rc != 0).sum()), "HwRuns": len(d),
         "HwFailProgram": short(fails.index[0][0]) if len(fails) else "none"}
    t = d[d.rc == 0]
    cv = t.groupby(["name", "fraction", "plan"]).t.agg(lambda s: s.std() / s.mean())
    n["HwCv"] = f"{100 * cv.median():.0f}\\%"
    spread = t.groupby(["name", "fraction", "plan"]).t.agg(lambda s: s.max() / s.min())
    n["HwSpreadMax"] = f"{spread.max():.2f}"
    n["HwSpreadName"] = short(spread.idxmax()[0])
    plans = pd.read_csv(os.path.join(DATA, "uvm_hw_plans_large.csv"))
    dg = plans.pivot_table(index=["name", "fraction"], columns="plan", values="digest", aggfunc="first")
    n["HwPlanConfigs"] = len(dg)
    n["HwPlansSame"] = int(((dg.hybrid == dg.truth) & (dg.hybrid_min == dg.truth_min)).sum())
    for frac, F in ((1.0, "Full"), (0.75, "ThreeQ"), (0.5, "Half")):
        x = m.xs(frac, level="fraction")
        fit = x[x.fits]
        n[f"Hw{F}Fits"] = len(fit)
        for p, P in HW_PLANS:
            n[f"Hw{F}{P}"] = f"{geo_speedup_of((fit.demand / fit[p]).dropna()):.3f}"
            n[f"Hw{F}{P}All"] = f"{geo_speedup_of((x.demand / x[p]).dropna()):.3f}"
    # the kernel-level model with the measured costs against the A30, where both have the share
    model = uvm_load("uvm_large.csv", "large")
    model = model[(model.status == "ok") & (model.costs == "measured")]
    mt = model.pivot_table(index=["name", "fraction"], columns="policy", values="time_ns")
    both = m[m.fits & (m.index.get_level_values(1) < 1.0)].index.intersection(mt.index)
    n["HwModelConfigs"] = len(both)
    for p, P in HW_PLANS:
        hw = (m.loc[both, "demand"] / m.loc[both, p]).dropna()
        md = (mt.loc[hw.index, "demand"] / mt.loc[hw.index, p])
        n[f"HwModel{P}"] = f"{geo_speedup_of(md):.3f}"
        n[f"HwMeasured{P}"] = f"{geo_speedup_of(hw):.3f}"
    # a third set of runs: plans without the interpreter (uvm_hw.py --tag _cheap), at a lower load of the host
    c = pd.read_csv(os.path.join(DATA, "uvm_hw_large_cheap.csv"))
    c["t"] = c.gpu_ns + c.copy_ns
    n.update({"HwCheapReps": c.rep.nunique(), "HwCheapRuns": len(c), "HwCheapFailRuns": int((c.rc != 0).sum())})
    ct = c[c.rc == 0]
    n["HwCheapCv"] = f"{100 * ct.groupby(['name', 'fraction', 'plan']).t.agg(lambda s: s.std() / s.mean()).median():.0f}\\%"
    cm = ct.groupby(["name", "fraction", "plan"]).t.median().unstack("plan")
    for frac, F in ((1.0, "Full"), (0.75, "ThreeQ"), (0.5, "Half")):
        x = cm.xs(frac, level="fraction").dropna(subset=["demand", "truth_min", "range_min", "history_min"])
        n[f"HwCheap{F}Programs"] = len(x)
        for p, P in (("truth_min", "TrMin"), ("range_min", "RangeMin"), ("history_min", "HistMin")):
            n[f"HwCheap{F}{P}"] = f"{geo_speedup_of(x.demand / x[p]):.3f}"
    half = cm.xs(0.5, level="fraction")
    n["HwCheapJacobiTrMin"] = f"{half.loc['polybench-jacobi2d', 'demand'] / half.loc['polybench-jacobi2d', 'truth_min']:.2f}"
    n["HwCheapJacobiRangeMin"] = f"{half.loc['polybench-jacobi2d', 'demand'] / half.loc['polybench-jacobi2d', 'range_min']:.2f}"
    n["HwCheapHotspotTrMin"] = f"{half.loc['rodinia-hotspot', 'demand'] / half.loc['rodinia-hotspot', 'truth_min']:.2f}"
    n["HwCheapHotspotHistMin"] = f"{half.loc['rodinia-hotspot', 'demand'] / half.loc['rodinia-hotspot', 'history_min']:.2f}"
    cp = pd.read_csv(os.path.join(DATA, "uvm_hw_plans_large_cheap.csv"))
    dg = cp.pivot_table(index=["name", "fraction"], columns="plan", values="digest", aggfunc="first")
    n["HwRangeSame"] = int(((dg.range == dg.truth) & (dg.range_min == dg.truth_min)).sum())
    n["HwRangeDiffer"] = short(sorted(set(dg.index[dg.range_min != dg.truth_min].get_level_values(0)))[0])
    n["HwHistSame"] = int((dg.history_min == dg.truth_min).sum())
    return n


def hw_table():
    """Per program, at the smallest share at which the kernel-level model holds every operation (or 50% if none
    below 100%), the A30's time with demand paging and each plan's speedup, with the model's for the interpreter's
    runs plan."""
    d, m, _ = hw_load()
    model = uvm_load("uvm_large.csv", "large")
    model = model[(model.status == "ok") & (model.costs == "measured")]
    mt = model.pivot_table(index=["name", "fraction"], columns="policy", values="time_ns")
    rows = []
    for name in sorted(m.index.get_level_values(0).unique(), key=short):
        x = m.loc[name]
        over = x[x.fits & (x.index < 1.0)]
        frac = over.index.min() if len(over) else 0.5
        r = x.loc[frac]
        cells = [r"\texttt{" + short(name) + "}", f"{100 * frac:.0f}\\%" + ("" if r.fits else r"$^*$"),
                 f"{r.demand / 1e6:.1f}"]
        cells += ["---" if pd.isna(r[p]) else f"{r.demand / r[p]:.2f}" for p, _ in HW_PLANS]
        cells.append(f"{mt.loc[(name, frac), 'demand'] / mt.loc[(name, frac), 'hybrid_min']:.2f}"
                     if (name, frac) in mt.index else "---")
        rows.append(" & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{llrrrrrrr}" "\n" r"\toprule" "\n"
            r" & & & \multicolumn{5}{c}{Speedup on the A30} & Model \\" "\n"
            r"\cmidrule(lr){4-8} \cmidrule(lr){9-9}" "\n"
            r"Program & Memory & \shortstack{Demand\\(ms)} & Buffers & \shortstack{Inter-\\preter} & Traced & "
            r"\shortstack{Inter-\\preter, runs} & \shortstack{Traced,\\runs} & \shortstack{Inter-\\preter, runs} \\"
            "\n" r"\midrule" "\n")
    write("uvm_hw.tex", head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write(body)
    print("wrote", name)


def main():
    n = numbers()
    n.update(uvm_numbers())
    with open(os.path.join(HERE, "numbers.tex"), "w") as f:
        for k, v in n.items():
            assert re.fullmatch(r"[A-Za-z]+", k), k
            f.write(f"\\newcommand{{\\num{k}}}{{{v}}}\n")
    print("wrote numbers.tex")
    summary_table(n)
    program_table()
    uvm_summary_table(n)
    hw_table()


if __name__ == "__main__":
    main()
