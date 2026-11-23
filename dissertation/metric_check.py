"""Which error the Ch. 2 tables report, against the estimator the chapter defines (5.Results.tex draftnote), on the
37-workload set at 50 bins (L2O, both splits, the training-best interval; splitter bins and both samplers' bins):
  per_bin    mean over bins of |alpha_i x bin_i / truth - 1| (score.py's MAPE, every table's "MAPE")
  mean_est   |mean over bins of alpha_i x bin_i / truth - 1| (the mean of the per-bin estimates; E1's e)
  eq_est     |alpha(mean bin, SD) x mean bin / truth - 1| (eq:mp-estimate, one alpha at the mean bin)
The signed versions are kept in the CSV.

    PYTHONPATH=analysis ~/environments/memfp/bin/python -W ignore dissertation/metric_check.py  ->  $OUT/metric_check.csv
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
SETS = [os.path.expanduser(p) for p in ("~/memprint-data/2026-11-11-bins50", "~/memprint-data/2026-11-12-b50more/c50",
                                        "~/memprint-data/2026-11-12-b50more/m5")]


def errors(model, b, k, truth):
    b = np.asarray(b, float)
    sd = b.std(ddof=1)
    a = model.predict(pd.DataFrame({"MemUsageObs": b, "SamplingInterval": k, "SD_MemUsage": sd})).Predicted_Alpha.to_numpy()
    m = b.mean()
    a0 = model.predict(pd.DataFrame({"MemUsageObs": [m], "SamplingInterval": [k], "SD_MemUsage": [sd]})).Predicted_Alpha.iloc[0]
    return dict(per_bin=np.mean(np.abs(a * b / truth - 1)) * 100, per_bin_signed=np.mean(a * b / truth - 1) * 100,
                mean_est=((a * b).mean() / truth - 1) * 100, eq_est=(a0 * m / truth - 1) * 100)


def main():
    rows = []
    for root in SETS:
        fits = pd.read_csv(f"{root}/data/fits.csv")
        for f in fits[fits.subset == "L2O"].itertuples():
            a = pd.read_csv(f"{root}/data/{f.workload}_allData.csv")
            p = prepare(a)
            c = str(p.last_config if f.split == "EXTRA" else p.middle_config)
            a = a.assign(Config=a.Config.astype(str))
            truth = a[(a.SamplingInterval == 1) & (a.Config == c)].MemUsageObs.iloc[0]
            model, k = Model.from_row(pd.Series(f._asdict())), int(f.train_si)
            sp = a[(a.SamplingInterval == k) & (a.Config == c)].MemUsageObs
            rows.append(dict(sampler="split", workload=f.workload, split=f.split, seed=0, k=k, **errors(model, sp, k, truth)))
            for s in ("boot", "disj"):
                for seed in (1, 2, 3):
                    files = glob.glob(f"{root}/m3/{s}/{seed}/{f.workload}/{c}/{k}/*_SubSample_*_bin_*.csv")
                    if files:
                        b = [pd.read_csv(x).MemUsageObs.iloc[0] for x in files]
                        rows.append(dict(sampler=s, workload=f.workload, split=f.split, seed=seed, k=k, **errors(model, b, k, truth)))
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/metric_check.csv", index=False)
    w = d.assign(mean_abs=d.mean_est.abs(), eq_abs=d.eq_est.abs()).groupby(["sampler", "split", "workload"])[
        ["per_bin", "mean_abs", "eq_abs"]].mean()
    pd.set_option("display.width", 200)
    print(w.groupby(["sampler", "split"]).agg(["mean", "median"]).round(2).to_string())
    print("within 10%:", w.lt(10).groupby(["sampler", "split"]).sum().to_string())
    print(d[d.workload.isin(["bicg", "gap-pr", "lulesh"])].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
