#!/usr/bin/env python3
"""Plans that carry over to other inputs (mprshim MPR_RPLAN, lines "sig,f0,f1"), from a chunks profile with block
   lengths (run_refs.sh, run "chunks"). Each profiled chunk covers [c x C / len, (c + 1) x C / len) of its block; its
   memory-time (sum of rss over the full windows) and referenced bytes are spread over the NB equal bins of the block
   that the range overlaps, in proportion to the overlap, and pooled over the site's blocks. Units (site signature, bin)
   are ranked by referenced share: cold (least first), hot (most first), rand (seed 0); a plan takes units until their
   memory-time reaches f x the memory-time of all anonymous memory in the profile.
   python3 place_rplans.py PROFILE_OUT DEST  -> DEST/<prog>/<policy><pct>.csv, DEST/plans.csv
   (the profile is PROFILE_OUT/runs/<prog>/1/chunks.csv*)"""
import sys, os, pandas as pd, numpy as np
PO, DEST = sys.argv[1], sys.argv[2]
NB, C = 16, 2097152
FRACS = [0.25, 0.5]
progs = list(dict.fromkeys(l.split()[0] for l in open(f"{PO}/list.txt")))
rows = []
for prog in progs:
    d = f"{PO}/runs/{prog}/1"
    m, z = pd.read_csv(f"{d}/chunks.csv"), pd.read_csv(f"{d}/chunks.csv.zones")
    c, s = pd.read_csv(f"{d}/chunks.csv.chunks"), pd.read_csv(f"{d}/chunks.csv.sites", dtype={"sig": str})
    full = m.window[(m.window > 0) & (m.window < m.window.max())]
    z, c = z[z.window.isin(full)], c[c.window.isin(full)]
    total = z[z.zone.isin(["heap", "anon", "stack"]) | z.zone.str.startswith("site")].rss_bytes.sum()
    sig = s.drop_duplicates("site").set_index("site").sig
    k = c.groupby(["site", "chunk", "len"], as_index=False)[["rss_bytes", "ref_bytes"]].sum()
    r0 = (k.chunk * C / k.len).clip(upper=1).values; r1 = ((k.chunk + 1) * C / k.len).clip(upper=1).values
    acc = {}
    for i in range(len(k)):
        if r1[i] <= r0[i]: continue
        for b in range(int(r0[i] * NB), min(NB, int(np.ceil(r1[i] * NB)))):
            ov = (min(r1[i], (b + 1) / NB) - max(r0[i], b / NB)) / (r1[i] - r0[i])
            if ov <= 0: continue
            key = (sig[k.site.iat[i]], b)
            a = acc.setdefault(key, [0.0, 0.0])
            a[0] += ov * k.rss_bytes.iat[i]; a[1] += ov * k.ref_bytes.iat[i]
    u = pd.DataFrame([(sg, b, v[0], v[1]) for (sg, b), v in acc.items()], columns=["sig", "bin", "mt", "ref"])
    u = u[u.mt > 0]; u["hot"] = u.ref / u.mt
    os.makedirs(f"{DEST}/{prog}", exist_ok=True)
    rng = np.random.default_rng(0)
    order = {"cold": u.sort_values(["hot", "mt"], ascending=[True, False]).index,
             "hot": u.sort_values(["hot", "mt"], ascending=[False, False]).index,
             "rand": u.index[rng.permutation(len(u))]}
    for f in FRACS:
        for pol, idx in order.items():
            uu = u.loc[idx]
            n = int(np.searchsorted(uu.mt.cumsum().values, f * total) + 1)
            sel = uu.iloc[:n]
            out = pd.DataFrame(dict(sig=sel.sig, f0=sel.bin / NB, f1=(sel.bin + 1) / NB))
            out.to_csv(f"{DEST}/{prog}/{pol}{int(f * 100)}.csv", index=False)
            rows.append(dict(prog=prog, plan=f"{pol}{int(f * 100)}", planned_share=sel.mt.sum() / total, n_units=len(sel),
                             mean_hot=np.average(sel.hot, weights=sel.mt)))
P = pd.DataFrame(rows); P.to_csv(f"{DEST}/plans.csv", index=False)
print(P.round(3).to_string(index=False))
