"""The 37-workload set's sampler results with the high-bit keep test (hibits.sh) against the original runs.

    PYTHONPATH=analysis ~/environments/memfp/bin/python -W ignore dissertation/hibits_score.py  ->  $OUT/m3_{runs,mape}_{b50,c50,m5}hb.csv,
                                                                                                  $OUT/mp_errors_hb.csv

Prints M3's decision rule over the 37 workloads, the L2O test MAPE on each sampler's bins (the ch2_accuracy layout),
the workloads within 10%, and C3: the estimate (mean of the per-bin estimates) within a factor of 2 of the truth.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import score as S  # noqa: E402
import mp_errors as MPE  # noqa: E402

N = os.path.expanduser("~/memprint-data/2026-11-23-hibits")
OUT = S.OUT
SETS = {"b50": os.path.expanduser("~/memprint-data/2026-11-11-bins50"), "c50": f"{S.MORE}/c50", "m5": f"{S.MORE}/m5"}


def rescore():
    rc = {}
    for line in open(f"{N}/done.txt"):
        x = line.split()
        if len(x) >= 8 and x[0] in SETS:
            rc[f"{N}/{x[0]}/m3/{x[1]}/{x[2]}/{x[3]}/{x[4]}/{x[5]}"] = x[6] if int(x[7]) > 0 else "1"
    for tag, root in SETS.items():
        runs = sorted(glob.glob(f"{N}/{tag}/m3/*/*/*/*/*"))
        wl = sorted({r.split("/")[-3] for r in runs})
        S.m3(f"{root}/data", [(p, rc.get(p, "1")) for p in runs], f"_{tag}hb", wl)


def m20():
    """M3's 20-bin runs of gemm and miniVite again (run.sh m3 m3mv, OUT=$N/m20) -> m3_{runs,mape}_m20hb.csv, for
    tab:m4mape's sampler column and the heap-profiler comparison (figures.py memprint_estimates)."""
    done = [x.split() for x in open(f"{N}/m20/m3/done.txt")]
    S.m3(S.FINAL, done, "_m20hb", ["gemm", "miniVite"])


def errors():
    """mp_errors.py's rows from the new runs (and the splitter's, unchanged)."""
    MPE_SETS = MPE.SETS
    rows = []
    for root, tag in zip(MPE_SETS, SETS):
        fits = pd.read_csv(f"{root}/data/fits.csv")
        for f in fits[(fits.split == "EXTRA") & (fits.subset == "L2O")].itertuples():
            a = pd.read_csv(f"{root}/data/{f.workload}_allData.csv")
            c = str(MPE.prepare(a).last_config)
            a = a.assign(Config=a.Config.astype(str))
            truth = a[(a.SamplingInterval == 1) & (a.Config == c)].MemUsageObs.iloc[0]
            model, k = MPE.Model.from_row(pd.Series(f._asdict())), int(f.train_si)
            for s in ("boot", "disj"):
                for seed in (1, 2, 3):
                    files = glob.glob(f"{N}/{tag}/m3/{s}/{seed}/{f.workload}/{c}/{k}/*_SubSample_*_bin_*.csv")
                    if files:
                        b = [pd.read_csv(x).MemUsageObs.iloc[0] for x in files]
                        rows.append(dict(sampler=s, workload=f.workload, seed=seed, k=k, truth=truth,
                                         e=MPE.estimate(model, b, k) / truth - 1))
    old = pd.read_csv(f"{OUT}/mp_errors.csv")
    d = pd.concat([pd.DataFrame(rows), old[old.sampler == "split"]], ignore_index=True)
    d.to_csv(f"{OUT}/mp_errors_hb.csv", index=False)
    return d


def report():
    pd.set_option("display.width", 200)
    for name, suffix in (("old keep test", ""), ("high-bit keep test", "hb")):
        mp = pd.concat([pd.read_csv(f"{OUT}/m3_mape_{t}{suffix}.csv") for t in SETS], ignore_index=True)
        runs = pd.concat([pd.read_csv(f"{OUT}/m3_runs_{t}{suffix}.csv") for t in SETS], ignore_index=True)
        mp = mp[mp.basis == "train_si"]
        g = mp.groupby(["sampler", "split", "subset", "workload"])[["sampler_mape", "splitter_mape"]].mean()
        g["diff"] = g.sampler_mape - g.splitter_mape
        sd = runs[runs.sampler == "boot"].sd_ratio.median()
        diff = g.loc["boot", "diff"].median()
        print(f"== {name}: {g.index.get_level_values('workload').nunique()} workloads, {len(runs)} runs; bootstrap SD ratio "
              f"median {sd:.4f}, MAPE difference median {diff:.2f} -> {'disjoint' if diff > 2 or not 0.9 <= sd <= 1.1 else 'bootstrap'}; "
              f"disjoint SD ratio median {runs[runs.sampler == 'disj'].sd_ratio.median():.4f}")
        for s in ("boot", "disj"):
            cells = []
            for split in ("EXTRA", "INTER"):
                for sub in ("NZ", "MT", "L2O"):
                    v = g.loc[(s, split, sub)].sampler_mape
                    cells.append(f"{split[:2]} {sub} {v.mean():.2f}/{v.median():.2f}")
                v = g.loc[(s, split, "L2O")].sampler_mape
                cells.append(f"L2O<10%: {(v < 10).sum()}/{len(v)}")
            print(f"  {s}: " + "  ".join(cells))
    d = pd.read_csv(f"{OUT}/mp_errors_hb.csv")
    o = pd.read_csv(f"{OUT}/mp_errors.csv")
    for name, t in (("old", o), ("high-bit", d)):
        for s in ("boot", "disj"):
            x = t[t.sampler == s]
            r = 1 + x.e
            bad = x[(r <= 0.5) | (r >= 2)]
            print(f"C3 {name} {s}: {x.workload.nunique() - bad.workload.nunique()} of {x.workload.nunique()} workloads have every "
                  f"run within 2x; misses: {sorted(set(bad.workload))}; signed e median {x.e.median() * 100:+.2f}%, "
                  f"|e| median {x.e.abs().median() * 100:.2f}%, under-estimates {(x.e < 0).mean() * 100:.0f}%, worst "
                  f"{x.loc[x.e.idxmin(), 'workload']} {x.e.min() * 100:+.1f}% / {x.loc[x.e.idxmax(), 'workload']} {x.e.max() * 100:+.1f}%")


if __name__ == "__main__":
    for part in sys.argv[1:] or ["rescore", "errors", "report"]:
        r = globals()[part]()
