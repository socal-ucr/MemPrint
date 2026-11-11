"""Per-program statistics of the scheduling section, on the pages-in-use curves (sched_pages.py's corpus), next to the
bytes-in-use truth and the RSS. Each program at its largest input (run.largest).

    PYTHONPATH=analysis ~/environments/memfp/bin/python paper_rss/sched_pages_stats.py   ->  $P/stats_*.csv

1. Mean over the run / peak of each truth curve; programs below 0.7; the GAP kernels' range.
2. Pond: its predicted untouched share of the request against the true one (1 - peak / request), over the jobs of the
   load-0.8 streams (seeds 0 to 2), per truth.
3. KS+: the peak of its first prediction over the true peak, per program, per truth (its histories are RSS curves).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from memprint.sched import ksplus, ml, real, run  # noqa: E402
from memprint.sched.model import Cluster, Job  # noqa: E402

P = os.path.expanduser("~/memprint-data/2026-11-10-pagesnat")
BYTES = os.path.expanduser("~/memprint-data/2026-10-13-sched-llm")   # the chapter's bytes-in-use corpus


def mean_peak():
    rows = []
    for name, corpus, methods in (("pages", P, ("pagesnat", "rss")), ("bytes", BYTES, ("truth",))):
        c = run.largest(pd.read_csv(f"{corpus}/corpus/curves.csv", dtype={"input": str}))
        for prog, g in c.groupby("prog"):
            wide = real.rebase({m: h.sort_values("k").bytes.to_numpy() for m, h in g.groupby("method")})
            for m in methods:
                if m in wide and wide[m].max() > 0:
                    rows.append(dict(corpus=name, truth=m, prog=prog, mean_over_peak=wide[m].mean() / wide[m].max()))
    d = pd.DataFrame(rows)
    d.to_csv(f"{P}/stats_mean_peak.csv", index=False)
    w = d.pivot_table(index="prog", columns="truth", values="mean_over_peak")
    print(w.round(2).to_string())
    for t in w.columns:
        v = w[t].dropna()
        gap = v[v.index.str.startswith("gap-")]
        print(f"{t}: below 0.7 for {(v < 0.7).sum()} of {len(v)}: {', '.join(v[v < 0.7].index)}; GAP {gap.min():.2f} to {gap.max():.2f}")


def pond_and_ksplus():
    s = run.Setup(f"{P}/corpus", f"{P}/traces", 3000, Cluster())
    rows, ks = [], []
    for sc in ("pagesnat", "rss"):
        for seed in (0, 1, 2):
            jobs = s.stream(sc, 0.8, seed)
            for j in jobs:
                u = float(s.pond_model.predict(pd.DataFrame([j.info["cm5"]]))[0])
                req = float(j.info["request"])
                rows.append(dict(scenario=sc, seed=seed, prog=j.info["prog"], pred_untouched=1 - u,
                                 pond_pool_share=np.floor(req * (1 - u)) / req, true_untouched=1 - j.resident.max() / req))
        f = s.oracle("ksplus")
        seen = set()
        for j in s.stream(sc, 0.8, 0):
            p = j.info["prog"]
            if p in seen:
                continue
            seen.add(p)
            r = f(j, 0.0)
            ks.append(dict(scenario=sc, prog=p, first_over_peak=float(np.max(r)) / float(j.resident.max())))
    d = pd.DataFrame(rows)
    d.to_csv(f"{P}/stats_pond.csv", index=False)
    g = d.groupby("scenario")
    print("Pond, mean over jobs: predicted untouched", g.pred_untouched.mean().round(3).to_dict(),
          "pool share", g.pond_pool_share.mean().round(3).to_dict(), "true untouched", g.true_untouched.mean().round(3).to_dict(),
          "overpredicted (pred > true)", g.apply(lambda x: (x.pred_untouched > x.true_untouched).mean()).round(3).to_dict())
    k = pd.DataFrame(ks)
    k.to_csv(f"{P}/stats_ksplus.csv", index=False)
    w = k.pivot_table(index="prog", columns="scenario", values="first_over_peak")
    print(w.round(2).to_string())
    for sc in w.columns:
        v = w[sc].dropna()
        inb = v[(v >= 0.86) & (v <= 1.18)]
        print(f"KS+ {sc}: {len(v)} programs; 0.86 to 1.18 for {len(inb)}; within [{inb.min():.2f}, {inb.max():.2f}]; "
              f"outside: {', '.join(f'{p} {x:.2f}' for p, x in v.drop(inb.index).items())}")


if __name__ == "__main__":
    mean_peak()
    pond_and_ksplus()
