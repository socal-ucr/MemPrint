#!/usr/bin/env python3
"""Placement plans from a chunks profile (run_refs.sh, run "chunks", 0.1 s windows, 2 MiB chunks).
   Each chunk (site signature, block order within the site, chunk) gets hot = sum ref / sum rss over the full windows.
   The candidates are the chunks resident in the window with the most anonymous memory (the peak), each with its rss
   there as its size, so that every policy puts the same bytes on the remote node at the peak. A plan lists candidates
   until their sizes reach f x the peak anonymous RSS: cold (least referenced first), hot (most first), rand (random
   order, seed 0). Chunks not resident at the peak stay local under every policy.
   BUDGET=time (environment): instead, every chunk is a candidate, its size is its memory-time (sum of its rss over the
   windows), and the budget is f x the memory-time of all anonymous memory, so every policy puts the same share of
   memory-time on the remote node.
   python3 place_plans.py OUT PROFILE_REP [PLANDIR] -> OUT/PLANDIR/<prog>/<policy><pct>.csv, OUT/PLANDIR/plans.csv
   (PLANDIR default plans)"""
import sys, os, pandas as pd, numpy as np
OUT, REP = sys.argv[1], int(sys.argv[2])
PD = sys.argv[3] if len(sys.argv) > 3 else "plans"
TIME = os.environ.get("BUDGET") == "time"
FRACS = [0.25, 0.5]
progs = list(dict.fromkeys(l.split()[0] for l in open(f"{OUT}/list.txt")))
rows = []
for prog in progs:
    d = f"{OUT}/runs/{prog}/{REP}"
    m, z = pd.read_csv(f"{d}/chunks.csv"), pd.read_csv(f"{d}/chunks.csv.zones")
    c, s = pd.read_csv(f"{d}/chunks.csv.chunks"), pd.read_csv(f"{d}/chunks.csv.sites", dtype={"sig": str})
    full = m.window[(m.window > 0) & (m.window < m.window.max())]
    z, c = z[z.window.isin(full)], c[c.window.isin(full)]
    anon = z[z.zone.isin(["heap", "anon", "stack"]) | z.zone.str.startswith("site")].groupby("window").rss_bytes.sum()
    peak, pk = anon.max(), anon.idxmax()
    sig = s.drop_duplicates("site").set_index("site").sig
    c = c.groupby(["window", "site", "ord", "chunk"], as_index=False)[["rss_bytes", "ref_bytes"]].sum()
    k = c.groupby(["site", "ord", "chunk"]).agg(rss=("rss_bytes", "sum"), ref=("ref_bytes", "sum")).reset_index()
    atpk = c[c.window == pk].set_index(["site", "ord", "chunk"]).rss_bytes
    k["size"] = [atpk.get(t, 0) for t in zip(k.site, k.ord, k.chunk)]
    if TIME:
        k["size"], peak = k.rss, anon.sum()
    k = k[k["size"] > 0]
    k["hot"] = k.ref / k.rss
    k["sig"] = k.site.map(sig)
    os.makedirs(f"{OUT}/{PD}/{prog}", exist_ok=True)
    rng = np.random.default_rng(0)
    order = {"cold": k.sort_values(["hot", "size"], ascending=[True, False]).index,
             "hot": k.sort_values(["hot", "size"], ascending=[False, False]).index,
             "rand": k.index[rng.permutation(len(k))]}
    for f in FRACS:
        for pol, idx in order.items():
            kk = k.loc[idx]
            cum = kk["size"].cumsum()
            n = int(np.searchsorted(cum.values, f * peak) + 1)   # the first chunks whose sizes reach the budget
            sel = kk.iloc[:n]
            name = f"{pol}{int(f * 100)}"
            sel[["sig", "ord", "chunk"]].to_csv(f"{OUT}/{PD}/{prog}/{name}.csv", index=False)
            rows.append(dict(prog=prog, plan=name, frac=f, budget_MB=peak / 2**20, planned_MB=sel["size"].sum() / 2**20,
                             planned_share=sel["size"].sum() / peak, n_chunks=len(sel),
                             mean_hot=np.average(sel.hot, weights=sel["size"]) if len(sel) else np.nan))
P = pd.DataFrame(rows); P.to_csv(f"{OUT}/{PD}/plans.csv", index=False)
print(P.round(3).to_string(index=False))
