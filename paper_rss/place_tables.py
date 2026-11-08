#!/usr/bin/env python3
"""Tables of the hot/cold and pool placement section (sec:place) of main.tex, from the runs of run_refs.sh
   (refs_summary.py), run_place.sh (place_summary.py), run_lplace.sh and run_combo.sh.
   python3 place_tables.py -> tables/place_{refs,static,migrate,libmpr,combo}.tex, and the numbers the text quotes"""
import os, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.expanduser("~/memprint-data/2026-11-07-refs")
L = os.path.expanduser("~/memprint-data/2026-11-08-lplace")
T = f"{HERE}/tables"
NAMES = {"gap-bfs": "GAP bfs", "gap-cc": "GAP cc", "gap-pr": "GAP pr", "gap-sssp": "GAP sssp", "python": "CPython",
         "lua": "Lua", "perl": "Perl", "cc1": r"\texttt{cc1}", "miniVite": "miniVite", "quicksilver": "Quicksilver",
         "sqlite": "SQLite", "darknet": "darknet", "llamacpp": "llama.cpp"}
ORDER = ["gap-bfs", "gap-cc", "gap-pr", "gap-sssp", "miniVite", "quicksilver", "cc1", "python", "lua", "perl", "sqlite"]


def f(x, d=3):
    return "---" if pd.isna(x) else f"{x:.{d}f}"


def write(name, head, rows, top=""):
    with open(f"{T}/{name}.tex", "w") as o:
        o.write("\\begin{tabular}{l" + "r" * (len(head) - 1) + "}\n\\toprule\n" + top + head[0])
        for h in head[1:]: o.write(" & " + h)
        o.write(" \\\\\n\\midrule\n")
        for r in rows: o.write(" & ".join(r) + " \\\\\n")
        o.write("\\bottomrule\n\\end{tabular}\n")


def pivot(path, col="slowdown"):
    g = pd.read_csv(path)
    return g.pivot(index="prog", columns="pol", values=col)


# 1. the accessed bits: cost, hot share per window length, what call sites and chunks cover
s = pd.read_csv(f"{R}/summary.csv").set_index("prog")
c = pd.read_csv(f"{R}/chunks_summary.csv")
c = c[c.run == "chunks"].groupby("prog")[["cold_page_MB", "cold_chunk_MB", "site_share"]].median()
rows = []
for p in ORDER:
    if p not in s.index: continue
    x = s.loc[p]
    cov = c.cold_chunk_MB[p] / c.cold_page_MB[p] if p in c.index and c.cold_page_MB[p] > 0 else np.nan
    rows.append([NAMES[p], f(x.refs_x, 2), f(x.sites_x, 2), f(x["hot_w0.02"], 2), f(x["hot_w0.1"], 2),
                 f(x["hot_w0.5"], 2), f(c.site_share[p], 2) if p in c.index else "---", f(cov, 2)])
write("place_refs", ["Program", "refs", "sites", "hot 0.02\\,s", "hot 0.1\\,s", "hot 0.5\\,s", "site share", "cold named"],
      rows)
print("refs cost", s.refs_x.min().round(3), s.refs_x.max().round(3), "sites", s.sites_x.min().round(3), s.sites_x.max().round(3))

# 2. static plans: exact (same input) and transferred (from the smaller input)
pt = pivot(f"{R}/place_time/summary.csv"); px = pivot(f"{R}/place_xd/summary.csv")
pts = pivot(f"{R}/place_time/summary.csv", "node1_avg_share")
rows = []
for p in ORDER:
    if p not in pt.index: continue
    r = [NAMES[p], f(pt.remote[p]), f(pt.cold25[p]), f(pt.rand25[p]), f(pt.hot25[p]), f(pt.cold50[p]), f(pt.hot50[p])]
    r += [f(px.rxfer_cold25[p]), f(px.rxfer_hot25[p]), f(px.rxfer_cold50[p])] if p in px.index else ["---"] * 3
    rows.append(r)
write("place_static", ["Program", "remote", "cold 25", "rand 25", "hot 25", "cold 50", "hot 50",
                       "cold 25", "hot 25", "cold 50"], rows,
      " & & \\multicolumn{5}{c}{Plan from the same input} & \\multicolumn{3}{c}{Plan from the smaller input} \\\\\n"
      "\\cmidrule(lr){3-7}\\cmidrule(lr){8-10}\n")
print("static node1 avg share cold25", pts.cold25.round(2).to_dict())

# 3. migration between windows against the static cold plan
pd_, pa = pivot(f"{R}/place_dt/summary.csv"), pivot(f"{R}/place_da/summary.csv")
rows = []
for p in ["gap-bfs", "gap-cc", "gap-pr", "gap-sssp", "python", "lua"]:
    rows.append([NAMES[p], f(px.remote[p]), f(px.cold25[p]), f(px.dyn0[p]), f(px.dyn25[p]), f(pd_.dynt25[p]),
                 f(pd_.dynt0w[p]), f(pa.dyna25[p]), f(pa.dyna50[p])])
write("place_migrate", ["Program", "remote", "static cold 25", "monitor 0.1\\,s", "fill 25", "threshold 25",
                        "monitor 0.25\\,s", "slow 25", "slow 50"], rows)

# 4. placement through libmpr, plans transferred
lp = pivot(f"{L}/place/summary.csv"); lps = pivot(f"{L}/place/summary.csv", "node1_avg_share")
rows = []
for p in ["gap-bfs", "gap-cc", "gap-pr", "gap-sssp", "python", "lua"]:
    rows.append([NAMES[p], f(1 / lp.plain[p]), f(lp.remote[p]), f(lp.lsame_cold25[p]), f(lp.lxfer_cold25[p]),
                 f(lp.lxfer_rand25[p]), f(lp.lxfer_hot25[p]), f(lp.lxfer_cold50[p]), f(lp.lxfer_hot50[p]),
                 f(lps.lxfer_cold25[p], 2)])
write("place_libmpr", ["Program", "libmpr", "remote", "same 25", "cold 25", "rand 25", "hot 25", "cold 50", "hot 50",
                       "pool share"], rows)


# 5. reservations and placement together
def combo(d, prog):
    x = pd.read_csv(f"{d}/runs/{prog}/summary.txt", sep=" ", names=["label", "rss", "commit", "secs", "hash", "status"])
    x["pol"] = x.label.str.rsplit("-", n=1).str[0]
    def numa(l):
        q = f"{d}/runs/{prog}/{l}.numa"
        if not os.path.exists(q): return pd.Series(dict(n0=np.nan, n1=np.nan))
        n = pd.read_csv(q)
        return pd.Series(dict(n0=n.node0_anon_bytes.max() / 2**20, n1=n.node1_anon_bytes.max() / 2**20))
    x = pd.concat([x, x.label.apply(numa)], axis=1)
    g = x.groupby("pol").agg(secs=("secs", "median"), commit=("commit", "median"), n0=("n0", "median"),
                             n1=("n1", "median"), hashes=("hash", "nunique"), bad=("status", lambda v: (v != 0).sum()))
    g["commit"] /= 2**20
    g["slow"] = g.secs / g.secs["none"]
    return g, x.hash.nunique(), len(x)
rows = []
for d, prog in [(f"{L}/combo", "darknet"), (f"{L}/combo2", "llamacpp")]:
    g, nh, n = combo(d, prog)
    print(prog, "runs", n, "distinct outputs", nh, "failed", int(g.bad.sum())); print(g.round(3).to_string())
    for pol, name in [("none", "libmpr alone"), ("res", "reservations"), ("cold25", "cold 25"),
                      ("res_cold25", "reservations + cold 25"), ("res_cold50", "reservations + cold 50"),
                      ("res_hot50", "reservations + hot 50"), ("remote", "remote"), ("res_remote", "reservations + remote")]:
        r = g.loc[pol]
        rows.append([NAMES[prog] if pol == "none" else "", name, f"{r.commit:.0f}", f"{r.n0:.0f}", f"{r.n1:.0f}", f(r.slow)])
    if prog == "darknet": rows.append(["\\addlinespace\n"] + [""] * 5)
rows = [r if not r[0].startswith("\\addlinespace") else None for r in rows]
with open(f"{T}/place_combo.tex", "w") as o:
    o.write("\\begin{tabular}{llrrrr}\n\\toprule\nProgram & Configuration & Commit & Node 0 & Node 1 & Time \\\\\n\\midrule\n")
    for r in rows: o.write("\\addlinespace\n" if r is None else " & ".join(r) + " \\\\\n")
    o.write("\\bottomrule\n\\end{tabular}\n")
for p in ["llamacpp"]:
    pl = pd.read_csv(f"{L}/combo2/lplans/plans.csv"); po = pd.read_csv(f"{L}/combo/lplans/plans.csv")
    print("llama plan share with aligned allocs", pl[pl.prog == p][["plan", "planned_share"]].round(3).values.tolist())
    print("llama plan share without", po[po.prog == p][["plan", "planned_share"]].round(3).values.tolist()[:1])
