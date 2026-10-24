"""Tables of the sensitivity runs (~/memprint-data/2026-10-24-sensitivity/sensitivity.py) for both papers: each of the
simulator's assumptions varied alone from the defaults (beta 0.5, a 1 TB pool, 5-minute updates), offered load 0.8,
mean of three seeds. The defaults' column comes from the main grids.

    ~/environments/memfp/bin/python paper_sched/sensitivity.py
"""
import glob
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_figures import load_grid, select, style  # noqa: E402

S = os.path.expanduser("~/memprint-data/2026-10-24-sensitivity")
L = os.path.expanduser("~/memprint-data/2026-10-13-sched-llm")
R = os.path.expanduser("~/memprint-data/2026-10-22-resize-cluster")
DEFAULT = {"beta": 0.5, "pool": 1024.0, "interval": 5.0}
SCHED = ["exact", "exact-peak", "memprint-W5", "memprint-P", "dhat", "request", "reactive", "pond", "zacarias",
         "zacarias-past", "ksplus"]
DYN = ["reactive", "zacarias", "zacarias-past"]
RESIZE = {"exact": "Exact curve (glibc build)", "request": "Request (glibc build)", "pond": "Pond",
          "zacarias": "Dynamic provisioning", "zacarias-past": "Dynamic provisioning, look-back", "ksplus": "KS+",
          "mpr-exact": "Exact curve (pass build)", "mpr": "MemPrintResize", "mpr+zacarias": "MemPrintResize + dynamic",
          "mpr+zacarias-past": "MemPrintResize + dynamic, look-back"}
HEAD = {"beta": r"$\beta$", "pool": "Pool (GB)", "interval": "Update interval (min)"}


def n0(x):
    return "--" if pd.isna(x) else f"{x:,.0f}".replace(",", "{,}")


def sched_rows():
    rows = []
    for f in sorted(glob.glob(f"{S}/sched_*.csv")):
        k, v = re.match(r".*sched_(\w+?)_([\d.]+)\.csv", f).groups()
        d = pd.read_csv(f)
        d = d[d.error.isna()] if "error" in d else d
        d["margin"] = d.margin.astype(str)
        for (sc, o), g in select(d, "cal").groupby(["scenario", "oracle"]):
            rows.append(dict(param=k, value=float(v), scenario=sc, oracle=o, resp=g.steady_response.mean()))
    base = select(load_grid(L, "grid_all*.csv"), "cal")
    for (sc, o), g in base[base.load == 0.8].groupby(["scenario", "oracle"]):
        for k, v in DEFAULT.items():
            rows.append(dict(param=k, value=v, scenario=sc, oracle=o, resp=g.steady_response.mean()))
    return pd.DataFrame(rows)


def resize_rows():
    rows = []
    for f in sorted(glob.glob(f"{S}/resize_*.csv")):
        k, v = re.match(r".*resize_(\w+?)_([\d.]+)\.csv", f).groups()
        d = pd.read_csv(f)
        d = d[d.error.isna()] if "error" in d else d
        for (tr, p), g in d.groupby(["truth", "policy"]):
            rows.append(dict(param=k, value=float(v), truth=tr, policy=p, tput=g.throughput.mean()))
    b = pd.read_csv(f"{R}/grid2_none.csv")
    b = b[(b.load == 0.8) & (b.inflate == 0)]
    for (tr, p), g in b.groupby(["truth", "policy"]):
        for k, v in DEFAULT.items():
            rows.append(dict(param=k, value=v, truth=tr, policy=p, tput=g.throughput.mean()))
    return pd.DataFrame(rows)


def table(t, param, names, label, col, fmt, out, cols_from=None, group="scenario", groups=("bytes", "rss"),
          group_titles=("Truth = bytes touched", "Truth = RSS")):
    t = t[t.param == param]
    if param == "pool":
        t = t[t.value > 0]   # without a pool the jobs larger than a node are rejected: not comparable
    vals = sorted(t.value.unique())
    n = len(vals)
    lines = [r"\begin{tabular}{l" + "r" * (n * len(groups)) + "}", r"\toprule",
             " & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{{g}}}" for g in group_titles) + r" \\",
             " ".join(f"\\cmidrule(lr){{{2 + i * n}-{1 + (i + 1) * n}}}" for i in range(len(groups))),
             HEAD[param] + " & " + " & ".join((f"{v:g}" for v in vals)) + (" & " + " & ".join(f"{v:g}" for v in vals)
                                                                           if len(groups) > 1 else "") + r" \\",
             r"\midrule"]
    p = t.pivot_table(index=[group, col], columns="value", values="resp" if col == "oracle" else "tput")
    for o in names:
        cells = [label(o)]
        for g in groups:
            cells += [fmt(p.loc[(g, o)].get(v)) if (g, o) in p.index else "--" for v in vals]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(out, "w").write("\n".join(lines) + "\n")


def main():
    s = sched_rows()
    lab = lambda o: style(o)[0].replace("%", r"\%")
    for param in ("beta", "pool"):
        table(s, param, SCHED, lab, "oracle", n0, f"{HERE}/tables/sens_{param}.tex")
    table(s, "interval", DYN, lambda o: lab(o).replace(" (5 min)", ""), "oracle", n0, f"{HERE}/tables/sens_interval.tex")
    r = resize_rows()
    rp = os.path.join(HERE, "..", "paper_resize", "tables")
    f1 = lambda x: "--" if pd.isna(x) else f"{x:.1f}"
    titles = ("Demand = RSS", "Demand = max(commit, RSS)")
    for param in ("beta", "pool"):
        table(r, param, list(RESIZE), RESIZE.get, "policy", f1, f"{rp}/sens_{param}.tex", group="truth",
              groups=("rss", "strict"), group_titles=titles)
    table(r, "interval", ["zacarias", "zacarias-past", "mpr+zacarias", "mpr+zacarias-past"], RESIZE.get, "policy", f1,
          f"{rp}/sens_interval.tex", group="truth", groups=("rss", "strict"), group_titles=titles)
    print("sensitivity tables written")


if __name__ == "__main__":
    main()
