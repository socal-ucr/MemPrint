"""The decision rule's spread ratio on bootstrap runs matched in workload, input, interval and seed between M3 (20
splitter bins, m3_runs.csv) and B50 (50 bins, m3_runs_b50.csv); the runs only one set has; the 95% interval of the
matched medians, resampling workloads (2,000 draws, seed 0).

    ~/environments/memfp/bin/python dissertation/matched.py
"""
import numpy as np
import pandas as pd

D = "/home/nmust004/memprint-data/2026-11-05-diss-ch2"
a = pd.read_csv(f"{D}/m3_runs.csv"); b = pd.read_csv(f"{D}/m3_runs_b50.csv")
a = a[(a.sampler == "boot") & (a.rc == 0)]; b = b[(b.sampler == "boot") & (b.rc == 0)]
k = ["workload", "config", "k", "seed"]
for d in (a, b):
    d["config"] = d.config.astype(str)
m = a.merge(b, on=k, suffixes=("_20", "_50"))
print("runs: 20-bin", len(a), "50-bin", len(b), "matched", len(m), "pairs (w,config,k)",
      m[["workload", "config", "k"]].drop_duplicates().shape[0], "workloads", m.workload.nunique())
print("all runs: median sd_ratio 20-bin %.3f, 50-bin %.3f" % (a.sd_ratio.median(), b.sd_ratio.median()))
print("matched : median sd_ratio 20-bin %.3f, 50-bin %.3f" % (m.sd_ratio_20.median(), m.sd_ratio_50.median()))
only20 = a.merge(b[k], on=k, how="left", indicator=True); only20 = only20[only20._merge == "left_only"]
only50 = b.merge(a[k], on=k, how="left", indicator=True); only50 = only50[only50._merge == "left_only"]
print("unmatched: 20-bin %d runs median %.3f; 50-bin %d runs median %.3f" % (len(only20), only20.sd_ratio.median(),
                                                                             len(only50), only50.sd_ratio.median()))
m["splitter_sd_50_over_20"] = m.sd_ratio_20 / m.sd_ratio_50
print("splitter SD 50 / 20 on matched runs: median %.3f, 10-90%%: %.3f to %.3f" % (
    m.splitter_sd_50_over_20.median(), *m.splitter_sd_50_over_20.quantile([.1, .9])))
rng = np.random.default_rng(0); ws = m.workload.unique()
bs = []
for _ in range(2000):
    s = pd.concat([m[m.workload == w] for w in rng.choice(ws, len(ws))])
    bs.append((s.sd_ratio_20.median(), s.sd_ratio_50.median()))
bs = np.array(bs)
print("matched medians, 95%% interval over workloads: 20-bin %.3f to %.3f; 50-bin %.3f to %.3f" % (
    *np.percentile(bs[:, 0], [2.5, 97.5]), *np.percentile(bs[:, 1], [2.5, 97.5])))
print("share of matched runs below 0.9: 20-bin %.2f, 50-bin %.2f" % ((m.sd_ratio_20 < 0.9).mean(), (m.sd_ratio_50 < 0.9).mean()))
