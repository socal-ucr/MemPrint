"""Numbers and tables for paper_footprint/main.tex, from paper_gpu/data/copy_savings{,_large}.csv
(paper_gpu/copy_savings.py). Writes numbers.tex and tables/*.tex here; reads nothing else of paper_gpu.

    python paper_footprint/make_numbers.py
"""

import os
import re
import sys

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


def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write(body)
    print("wrote", name)


def main():
    n = numbers()
    with open(os.path.join(HERE, "numbers.tex"), "w") as f:
        for k, v in n.items():
            assert re.fullmatch(r"[A-Za-z]+", k), k
            f.write(f"\\newcommand{{\\num{k}}}{{{v}}}\n")
    print("wrote numbers.tex")
    summary_table(n)
    program_table()


if __name__ == "__main__":
    main()
