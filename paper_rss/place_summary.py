#!/usr/bin/env python3
"""Summarise run_place.sh: per program and policy, the median runtime over repetitions, the slowdown against all-local,
   the share of the all-remote penalty paid, and the anonymous bytes on node 1 (numa_maps, at the sample with the most
   anonymous bytes; median over repetitions).  python3 place_summary.py [OUT [PLACE]] -> OUT/PLACE/summary.csv"""
import sys, os, pandas as pd, numpy as np
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/memprint-data/2026-11-07-refs")
PLACE = sys.argv[2] if len(sys.argv) > 2 else "place"
MB = 2**20
d = pd.read_csv(f"{OUT}/done_{PLACE}.txt", sep=" ", names=["prog", "rep", "pol", "status", "sec", "maxrss_kb"])
bad = d[d.status != 0]
if len(bad): print("failed runs:\n", bad.to_string(index=False))
d = d[d.status == 0]
def node1(r):
    p = f"{OUT}/{PLACE}/{r.prog}/{r.rep}/{r.pol}.csv"
    try:
        x = pd.read_csv(p)
    except Exception:
        return pd.Series(dict(anon_MB=np.nan, node1_MB=np.nan, node1_avg_share=np.nan))
    tot = x.node0_anon_bytes + x.node1_anon_bytes
    i = tot.idxmax()
    return pd.Series(dict(anon_MB=tot[i] / MB, node1_MB=x.node1_anon_bytes[i] / MB,
                          node1_avg_share=x.node1_anon_bytes.sum() / tot.sum()))
d = pd.concat([d, d.apply(node1, axis=1)], axis=1)
g = d.groupby(["prog", "pol"]).agg(sec=("sec", "median"), sec_min=("sec", "min"), sec_max=("sec", "max"), n=("sec", "size"),
                                   anon_MB=("anon_MB", "median"), node1_MB=("node1_MB", "median"),
                                   node1_avg_share=("node1_avg_share", "median")).reset_index()
loc = g[g.pol == "local"].set_index("prog").sec; rem = g[g.pol == "remote"].set_index("prog").sec
g["slowdown"] = g.sec / g.prog.map(loc)
g["penalty_share"] = (g.sec - g.prog.map(loc)) / (g.prog.map(rem) - g.prog.map(loc))
g["node1_share"] = g.node1_MB / g.anon_MB
g.to_csv(f"{OUT}/{PLACE}/summary.csv", index=False)
order = sorted(g.pol.unique(), key=lambda p: (p not in ("local", "remote", "dyn0"), p[-3:] == "50" or p.endswith("50h"), p))
pd.set_option("display.width", 200)
for col, fmt in [("slowdown", 3), ("node1_share", 2), ("node1_avg_share", 2), ("penalty_share", 2)]:
    t = g.pivot(index="prog", columns="pol", values=col)[[o for o in order if o in g.pol.unique()]]
    print(f"\n{col}\n", t.round(fmt).to_string())
