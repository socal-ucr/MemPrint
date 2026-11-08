#!/usr/bin/env python3
"""A latency model of a CXL pool from the runs of run_cxl.sh.
   For each program, the time a policy adds over its local base is linear in the loads the pool serves:
       T(policy, d) = T(base) + s * N(policy) * d
   with N the user-mode loads served from node 1 (retired loads that missed the L3: 'ld'; or all reads that missed it,
   prefetches and stores included: 'or'), d the pool's added latency over local (pointer chase, lat.txt), and s
   (seconds per access per ns, the inverse of the memory-level parallelism) fitted on the runs with node 1 idle
   (d about 55 ns). The fit is checked on the runs with the hog (d about 107 ns), which it never saw, and then
   extrapolated to slower pools. The base of a policy is libmpr with no plan ('local'/'none'), or with the reservations
   alone ('res') for the policies that add a plan to them.
   python3 cxl_model.py [data dir] -> <dir>/model.csv, tables/place_cxl.tex, and the numbers the text quotes"""
import os, sys, glob, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/memprint-data/2026-11-08-cxl")
EV = {"mem_load_uops_l3_miss_retired.remote_dram:u": "ld_r", "mem_load_uops_l3_miss_retired.local_dram:u": "ld_l",
      "offcore_response.all_reads.llc_miss.remote_dram:u": "or_r", "offcore_response.all_reads.llc_miss.local_dram:u": "or_l"}
NAMES = {"gap-bfs": "GAP bfs", "gap-cc": "GAP cc", "gap-pr": "GAP pr", "gap-sssp": "GAP sssp", "python": "CPython",
         "lua": "Lua", "darknet": "darknet", "llamacpp": "llama.cpp"}
SCEN = [100, 200, 300]   # added latencies (ns) to predict at: about 2x, 3x and 4x the local latency


def runs():
    rows = []
    for f in glob.glob(f"{D}/runs/*/*/*/*.time"):
        prog, rep, hog, pol = f[len(D) + 6:-5].split("/")
        t = open(f).read().split()
        try: secs = float(t[-2])
        except (ValueError, IndexError): continue
        r = dict(prog=prog, rep=int(rep), hog=int(hog), pol=pol, secs=secs)
        for line in open(f[:-5] + ".perf"):
            x = line.strip().split(",")
            if len(x) > 2 and x[2] in EV: r[EV[x[2]]] = float(x[0]) if x[0][0].isdigit() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


lat = pd.read_csv(f"{D}/lat.txt", sep=" ", names=["hog", "local", "pool"]).groupby("hog").median()
lat["d"] = lat.pool - lat.local
x = runs()
bad = x.secs.isna().sum()
m = x.groupby(["prog", "hog", "pol"]).median(numeric_only=True).reset_index()
out = []
for prog, g in m.groupby("prog"):
    loc = "local" if "local" in set(g.pol) else "none"
    h0 = g[g.hog == 0].set_index("pol"); hh = g[g.hog > 0].set_index("pol")
    d0, dh = lat.d[0], lat.d[lat.index.max()]
    # location-only pairs: libmpr serves the same blocks, only the node differs. In the run_lplace.sh programs every
    # plan (libmpr serves blocks of 256 KiB or more either way); in run_combo.sh's a plan also makes libmpr serve the
    # sites it names (and skip zeroing their untouched pages), so only all-local against all-remote, with and without
    # the reservations
    pairs = [(p, loc) for p in h0.index if p != loc] if loc == "local" else [("remote", "none"), ("res_remote", "res")]
    for kind in ["ld", "or"]:
        n0 = h0[f"{kind}_r"]
        dt = np.array([h0.secs[p] - h0.secs[b] for p, b in pairs])
        nd = np.array([(n0[p] - n0[b]) * d0 for p, b in pairs])
        s_ = (dt * nd).sum() / (nd * nd).sum()           # least squares through the origin
        for p in h0.index:
            if p in (loc, "res"): continue
            b = "res" if p.startswith("res_") and loc == "none" else loc   # the local base of the slowdown
            r = dict(prog=prog, kind=kind, pol=p, s=s_, n=n0[p], d0=d0, dh=dh, meas0=h0.secs[p] / h0.secs[b],
                     meash=hh.secs[p] / hh.secs[b], predh=(h0.secs[p] + s_ * n0[p] * (dh - d0)) / h0.secs[b])
            for d in SCEN: r[f"pred{d}"] = (h0.secs[p] + s_ * n0[p] * (d - d0)) / h0.secs[b]
            out.append(r)
o = pd.DataFrame(out)
o["errh"] = o.predh - o.meash            # error of the predicted slowdown at the hog's latency
o.to_csv(f"{D}/model.csv", index=False)
print("runs", len(x), "unparsed", bad); print(lat.round(1).to_string())
print("median |error| of the predicted slowdown at the hog's latency, by counter:")
print(o.groupby("kind").errh.apply(lambda e: e.abs().median()).round(4).to_string())
print(o.groupby(["kind", "prog"]).errh.apply(lambda e: e.abs().max()).unstack(0).round(3).to_string())
pd.set_option("display.width", 250)
print(o[o.kind == "or"][["prog", "pol", "meas0", "meash", "predh", "pred100", "pred200", "pred300"]].round(3).to_string())

# the table: measured at the two latencies, predicted at +200 ns; darknet and llama.cpp with their reservations
def f(v): return "---" if pd.isna(v) else f"{v:.3f}"
k = o[o.kind == "or"].set_index(["prog", "pol"])
err = k.errh.abs().groupby("prog").max()
d0, dh = lat.d[0], lat.d[lat.index.max()]
rows = []
for p in ["gap-bfs", "gap-cc", "gap-pr", "gap-sssp", "python", "lua", "darknet", "llamacpp"]:
    if p not in err.index: continue
    pre = "res_" if p in ("darknet", "llamacpp") else "lxfer_"
    r = [NAMES[p], f"{err[p]:.3f}"]
    for pol in ["res_remote" if pre == "res_" else "remote", pre + "cold25", pre + "hot25"]:
        x = k.loc[(p, pol)]
        r += [f(x.meas0), f(x.meash), f(x.pred200), f(x.pred300)]
    rows.append(r)
lab = [f"+{d0:.0f}", f"+{dh:.0f}", "+200", "+300"]
with open(f"{HERE}/tables/place_cxl.tex", "w") as t:
    t.write("\\begin{tabular}{lr" + "rrrr" * 3 + "}\n\\toprule\n")
    t.write(" & & \\multicolumn{4}{c}{All in the pool} & \\multicolumn{4}{c}{Cold 25} & \\multicolumn{4}{c}{Hot 25} \\\\\n"
            "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14}\n")
    t.write("Program & Error & " + " & ".join(lab * 3) + " \\\\\n\\midrule\n")
    for r in rows: t.write(" & ".join(r) + " \\\\\n")
    t.write("\\bottomrule\n\\end{tabular}\n")
# numbers for the text: share of the all-remote cost the cold and hot plans pay, at each latency
for p in err.index:
    pre = "res_" if p in ("darknet", "llamacpp") else "lxfer_"; rem = "res_remote" if pre == "res_" else "remote"
    c, h, a = k.loc[(p, pre + "cold25")], k.loc[(p, pre + "hot25")], k.loc[(p, rem)]
    print(p, "cost +300: cold", round(c.pred300 - 1, 3), "hot", round(h.pred300 - 1, 3), "all", round(a.pred300 - 1, 3),
          "| cold share of all +56", round((c.meas0 - 1) / (a.meas0 - 1), 2), "+300", round((c.pred300 - 1) / (a.pred300 - 1), 2))
