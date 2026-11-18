"""MemPrint's signed extrapolation errors on the 35-workload evaluation set (50 bins, bytes touched, selection on the
training inputs), for the scheduling oracle memprint-peak (analysis/memprint/sched, E1 of PROTOCOL.md).

Per workload, the L2O model of the EXTRA split predicts alpha for every bin of a sampler run at its training-best
interval at the largest input; the footprint estimate is the mean over bins of predicted alpha x the bin's
footprint, and the error is estimate / truth - 1. One row per (sampler, workload, seed); the splitter's bins give one
row per workload (sampler "split").

    PYTHONPATH=analysis ~/environments/memfp/bin/python -W ignore dissertation/mp_errors.py   ->  $OUT/mp_errors.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from memprint.dataset import prepare  # noqa: E402
from memprint.model import Model  # noqa: E402

OUT = os.path.expanduser("~/memprint-data/2026-11-05-diss-ch2")
SETS = [os.path.expanduser("~/memprint-data/2026-11-11-bins50"), os.path.expanduser("~/memprint-data/2026-11-12-b50more/c50")]


def estimate(model, bins, k):
    b = pd.Series(bins, dtype=float)
    t = pd.DataFrame({"MemUsageObs": b, "SamplingInterval": k, "SD_MemUsage": b.std()})
    return float((model.predict(t).Predicted_Alpha * b).mean())


def main():
    rows = []
    for root in SETS:
        fits = pd.read_csv(f"{root}/data/fits.csv")
        for f in fits[(fits.split == "EXTRA") & (fits.subset == "L2O")].itertuples():
            a = pd.read_csv(f"{root}/data/{f.workload}_allData.csv")
            p = prepare(a)
            c = str(p.last_config)
            a = a.assign(Config=a.Config.astype(str))
            truth = a[(a.SamplingInterval == 1) & (a.Config == c)].MemUsageObs.iloc[0]
            model = Model.from_row(pd.Series(f._asdict()))
            k = int(f.train_si)
            sp = a[(a.SamplingInterval == k) & (a.Config == c)].MemUsageObs.to_numpy()
            rows.append(dict(sampler="split", workload=f.workload, seed=0, k=k, truth=truth,
                             e=estimate(model, sp, k) / truth - 1))
            for s in ("boot", "disj"):
                for seed in (1, 2, 3):
                    d = f"{root}/m3/{s}/{seed}/{f.workload}/{c}/{k}"
                    files = glob.glob(f"{d}/*_SubSample_*_bin_*.csv")
                    if not files:
                        print("missing", d)
                        continue
                    b = [pd.read_csv(x).MemUsageObs.iloc[0] for x in files]
                    rows.append(dict(sampler=s, workload=f.workload, seed=seed, k=k, truth=truth,
                                     e=estimate(model, b, k) / truth - 1))
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/mp_errors.csv", index=False)
    for s, g in d.groupby("sampler"):
        w = g.groupby("workload").e.mean()
        print(f"{s}: {len(g)} runs, {w.size} workloads; signed e median {g.e.median()*100:+.2f}%, |e| mean "
              f"{g.e.abs().mean()*100:.2f}% median {g.e.abs().median()*100:.2f}%; under-estimates {(g.e < 0).mean()*100:.0f}%, "
              f"10th to 90th pct {g.e.quantile(.1)*100:+.1f}% to {g.e.quantile(.9)*100:+.1f}%")


if __name__ == "__main__":
    main()
