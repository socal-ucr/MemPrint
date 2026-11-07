#!/usr/bin/env python3
"""Summarise run_refs.sh: overhead, hot share per window, cold memory and how much of it call sites can place.
   python3 refs_summary.py [OUT]  -> OUT/summary.csv, OUT/sites_summary.csv, tables on stdout"""
import sys, os, pandas as pd, numpy as np
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/memprint-data/2026-11-07-refs")
MB = 2**20
done = pd.read_csv(f"{OUT}/done.txt", sep=" ", names=["prog", "rep", "run", "status", "sec", "maxrss_kb"])
progs = list(dict.fromkeys(l.split()[0] for l in open(f"{OUT}/list.txt")))
ANON = {"heap", "anon", "stack"}

def load(prog, rep, run):
    p = f"{OUT}/runs/{prog}/{rep}/{run}.csv"
    if not os.path.exists(p): return None, None
    m, z = pd.read_csv(p), pd.read_csv(p + ".zones")
    # full windows only: drop window 0 (starts at the constructor) and the last (partial, at exit)
    full = m.window[(m.window > 0) & (m.window < m.window.max())]
    return m[m.window.isin(full)], z[z.window.isin(full)]

rows, srows = [], []
for prog in progs:
    d = done[done.prog == prog]
    t = d.groupby("run").sec.median()
    r = dict(prog=prog, base_s=t.get("base"), refs_x=t.get("refs") / t.get("base"),
             sites_x=t.get("sites") / t.get("base"))
    for run, tag in [("refs", "w0.1"), ("sites_w0.02", "w0.02"), ("sites", "sites_w0.1"), ("sites_w0.5", "w0.5")]:
        m, z = load(prog, 1, run)
        if m is None or not len(m): continue
        za = z[z.zone.isin(ANON) | z.zone.str.startswith("site")]
        a = za.groupby("window")[["rss_bytes", "ref_bytes"]].sum()
        r[f"hot_{tag}"] = (a.ref_bytes / a.rss_bytes).mean()          # mean over windows of hot share of anon memory
        pk = a.rss_bytes.idxmax()
        r[f"anon_peak_MB_{tag}"] = a.rss_bytes[pk] / MB
        r[f"cold_peak_MB_{tag}"] = (a.rss_bytes[pk] - a.ref_bytes[pk]) / MB
        r[f"cost_ms_{tag}"] = m.cost_s.median() * 1e3
        if run == "sites":
            s = z[z.zone.str.startswith("site")].groupby("zone")[["rss_bytes", "ref_bytes"]].sum()
            s["hot"] = s.ref_bytes / s.rss_bytes
            zp = z[z.window == pk].set_index("zone")
            cold_sites = s.index[s.hot < 0.1]                              # one static decision per site
            r["n_sites"] = len(s)
            r["site_share_peak"] = zp.loc[zp.index.str.startswith("site"), "rss_bytes"].sum() / a.rss_bytes[pk]
            r["site_cold_peak_MB"] = zp.loc[zp.index.isin(cold_sites), "rss_bytes"].sum() / MB
            # cold bytes a per-window oracle at site granularity finds (site whose ref < 10 % of rss in that window)
            zs = z[z.zone.str.startswith("site")]
            r["site_cold_win_MB"] = zs[(zs.window == pk) & (zs.ref_bytes < 0.1 * zs.rss_bytes)].rss_bytes.sum() / MB
            for zn, row in s.sort_values("rss_bytes", ascending=False).head(5).iterrows():
                srows.append(dict(prog=prog, site=zn, mean_rss_MB=row.rss_bytes / MB / a.shape[0], hot=row.hot))
    rows.append(r)
S = pd.DataFrame(rows); S.to_csv(f"{OUT}/summary.csv", index=False)
pd.DataFrame(srows).to_csv(f"{OUT}/sites_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(S.round(3).to_string(index=False))
print(pd.DataFrame(srows).round(3).to_string(index=False))

# Chunk runs: how much of the cold memory sites and 2 MiB chunks can name, at the peak window of each run.
#   cold_page   resident anonymous bytes not referenced in the window (page level, the most the bits can tell)
#   cold_site   resident bytes of sites with < 10 % of their resident bytes referenced in that window
#   cold_chunk  resident bytes of chunks with < 10 % referenced in that window
#   cold_chunk_static  resident bytes, at the peak, of chunks (block, chunk) whose referenced share over the run < 10 %
crow = []
for prog in progs:
    for run in ["chunks_w0.02", "chunks", "chunks_w0.5"]:
        for rep in [1, 2, 3]:
            m, z = load(prog, rep, run)
            if m is None or not len(m): continue
            c = pd.read_csv(f"{OUT}/runs/{prog}/{rep}/{run}.csv.chunks")
            c = c[c.window.isin(m.window)].groupby(["window", "site", "block", "chunk"], as_index=False).sum()
            za = z[z.zone.isin(ANON) | z.zone.str.startswith("site")]
            a = za.groupby("window")[["rss_bytes", "ref_bytes"]].sum()
            pk = a.rss_bytes.idxmax()
            zs = z[(z.window == pk) & z.zone.str.startswith("site")]
            cp = c[c.window == pk]
            tot = c.groupby(["block", "chunk"])[["rss_bytes", "ref_bytes"]].sum()
            cold_keys = set(tot.index[tot.ref_bytes < 0.1 * tot.rss_bytes])
            st = cp.loc[np.array([k in cold_keys for k in zip(cp.block, cp.chunk)], dtype=bool)]
            t = done[(done.prog == prog) & (done.run == run) & (done.rep == rep)].sec
            crow.append(dict(prog=prog, run=run, rep=rep, sec=t.iloc[0] if len(t) else np.nan,
                anon_MB=a.rss_bytes[pk] / MB, site_share=zs.rss_bytes.sum() / a.rss_bytes[pk],
                cold_page_MB=(a.rss_bytes[pk] - a.ref_bytes[pk]) / MB,
                cold_site_MB=zs[zs.ref_bytes < 0.1 * zs.rss_bytes].rss_bytes.sum() / MB,
                cold_chunk_MB=cp[cp.ref_bytes < 0.1 * cp.rss_bytes].rss_bytes.sum() / MB,
                cold_chunk_static_MB=st.rss_bytes.sum() / MB, n_chunks=len(cp), cost_ms=m.cost_s.median() * 1e3))
if crow:
    C = pd.DataFrame(crow); C.to_csv(f"{OUT}/chunks_summary.csv", index=False)
    print(C.groupby(["prog", "run"], sort=False).median(numeric_only=True).drop(columns="rep").round(3).to_string())
