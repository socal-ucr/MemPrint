"""Numbers and tables for paper_footprint/main.tex, from paper_gpu/data/copy_savings{,_large}.csv
(paper_gpu/copy_savings.py) and paper_gpu/data/uvm*.csv (uvm_oversub.py, uvm_blocks.py). Writes numbers.tex and
tables/*.tex here; writes nothing in paper_gpu.

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
UVM_POLICIES = (("buffer", "Buf"), ("hybrid", "Hy"), ("truth", "Tr"))


def uvm_load(f, size):
    d = pd.read_csv(os.path.join(DATA, f))
    return d[~d.name.isin(DEGENERATE.get(size, ()))].reset_index(drop=True)


def uvm_speedups(d, fault_us, oversubscribed=True):
    """Per (program, share of memory), each policy's speedup over demand paging."""
    ok = d[(d.status == "ok") & (d.fault_us == fault_us)]
    ok = ok[ok.fraction < 1.0] if oversubscribed else ok[ok.fraction == 1.0]
    t = ok.pivot_table(index=["name", "fraction"], columns="policy", values="time_ns")
    return pd.DataFrame({p: t["demand"] / t[p] for p, _ in UVM_POLICIES})


def uvm_numbers():
    n = {}
    for key, f, size, *_ in UVM:
        d = uvm_load(f, size)
        n[f"Uvm{key}Programs"] = d.name.nunique()
        for fu in (10, 25, 50):
            s = uvm_speedups(d, fu)
            F = {10: "Ten", 25: "", 50: "Fifty"}[fu]
            if fu == 25:
                n[f"Uvm{key}Over"] = s.index.get_level_values(0).nunique()
                n[f"Uvm{key}Configs"] = len(s)
                n[f"Uvm{key}HyBehind"] = int((s.hybrid < 0.99 * s.truth).sum())
                n[f"Uvm{key}TrLoses"] = int((s.truth < 0.999).sum())
                ok = d[(d.status == "ok") & (d.fault_us == fu) & (d.fraction < 1.0)]
                moved = (ok.to_device_bytes + ok.to_host_bytes).groupby([ok.name, ok.fraction, ok.policy]).sum()
                mv = moved.unstack("policy")
                n[f"Uvm{key}HyFewerBytes"] = f"{100 * (1 - mv.hybrid.sum() / mv.demand.sum()):.0f}\\%"
                n[f"Uvm{key}HyFewerConfigs"] = int((mv.hybrid < 0.999 * mv.demand).sum())
                fit = uvm_speedups(d, fu, oversubscribed=False)
                for p, P in UVM_POLICIES:
                    n[f"Uvm{key}Fit{P}"] = f"{geo_speedup_of(fit[p]):.3f}"
            for p, P in UVM_POLICIES:
                n[f"Uvm{key}{P}{F}"] = f"{geo_speedup_of(s[p]):.3f}"
    d = uvm_load("uvm_blocks_large_4k.csv", "large")
    g = d[d.name.str.startswith("graphbig") & (d.status == "ok")]
    n["UvmGraphOver"] = int((g.fraction < 1.0).sum())
    return n


def geo_speedup_of(s):
    return float(np.exp(np.log(np.asarray(s, dtype=float)).mean()))


def uvm_summary_table(n):
    rows = []
    for key, _, _, model, size, page in UVM:
        cells = [model, size, page, f"{n[f'Uvm{key}Over']} of {n[f'Uvm{key}Programs']}", str(n[f"Uvm{key}Configs"])]
        cells += [n[f"Uvm{key}{P}"] for _, P in UVM_POLICIES]
        cells += [n[f"Uvm{key}Fit{P}"] for _, P in UVM_POLICIES]
        rows.append(" & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{lllrrrrrrrr}" "\n" r"\toprule" "\n"
            r" & & & & & \multicolumn{3}{c}{Oversubscribed} & \multicolumn{3}{c}{All data fits} \\" "\n"
            r"\cmidrule(lr){6-8} \cmidrule(lr){9-11}" "\n"
            r"Model & Inputs & Page & Programs & Shares & Buffers & \shortstack{Inter-\\preter} & Traced & "
            r"Buffers & \shortstack{Inter-\\preter} & Traced \\" "\n" r"\midrule" "\n")
    write("uvm_summary.tex", head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


def uvm_program_table():
    """Large inputs, 4 KB pages, wave by wave: each oversubscribed program at the smallest share that fits."""
    d = uvm_load("uvm_blocks_large_4k.csv", "large")
    s = uvm_speedups(d, 25).reset_index()
    first = s.sort_values("fraction").groupby("name").first().sort_values("hybrid", ascending=False)
    info = d.groupby("name")[["pages", "waves"]].first()
    rows = []
    for name, r in first.iterrows():
        rows.append(" & ".join([r"\texttt{" + short(name) + "}", d[d.name == name].suite.iloc[0].replace("-GPU", ""),
                                f"{int(info.loc[name, 'pages']):,}", str(int(info.loc[name, "waves"])),
                                f"{100 * r.fraction:.1f}\\%", f"{r.buffer:.2f}", f"{r.hybrid:.2f}",
                                f"{r.truth:.2f}"]) + r" \\")
    head = (r"\begin{tabular}{llrrrrrr}" "\n" r"\toprule" "\n"
            r"Program & Suite & Pages & Waves & Memory & Buffers & Interpreter & Traced \\" "\n" r"\midrule" "\n")
    write("uvm_programs.tex", head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


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
    uvm_program_table()


if __name__ == "__main__":
    main()
