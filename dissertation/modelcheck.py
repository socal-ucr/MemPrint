"""Offline checks of three modeling statements of Ch. 2 on the 50-bin splitter bins of the 35-workload evaluation set
(B50's 28 workloads and the 7 corpus programs, the data of tables.py unified). No new traces.

    PYTHONPATH=analysis ~/environments/memfp/bin/python dissertation/modelcheck.py   ->  $OUT/modelcheck_*.csv

  terms   the chosen model (eq. mp-alpha-final, 7 log features and their products) against the same model plus the
          squares of the three logs (sq) and plus their squares and cubes (cub), each fitted per workload with the
          same protocol as build.py: per split (EXTRA, INTER) and subset (NZ, MT, L2O), train on the other inputs,
          predict at the interval with the lowest training MAPE (train_si), test MAPE there.
  pooled  one model of the chosen form fitted on the other 34 workloads (same subset, all their inputs), predicted
          at the held-out input of the target at the interval with the lowest MAPE on the target's training inputs
          (pooled); and the same with the target's training inputs added to the pool (pooled_own).
  slope   per workload and input, the least-squares slope of log(mean alpha over bins) on log(k), over all
          intervals the model uses and over k = 1000 to 10000 (alpha.py's figure range).
  rank    per workload and split, the position of the best monotone interval in the list of intervals (0 = densest,
          smallest k), with the selection on the training inputs only and, as build.py does, on all inputs.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from memprint.dataset import prepare  # noqa: E402
from memprint.build import SPLITS, test_rows  # noqa: E402

OUT = os.path.expanduser("~/memprint-data/2026-11-05-diss-ch2")
ROOTS = [os.path.expanduser("~/memprint-data/2026-11-11-bins50/data"),
         os.path.expanduser("~/memprint-data/2026-11-12-b50more/c50/data")]
SUBSETS = ("NZ", "MT", "L2O")


def X(d, form):
    s, k, m = (np.log(d[c].astype(float).values) for c in ("SD_MemUsage", "SamplingInterval", "MemUsageObs"))
    cols = [s, k, m, m * s, k * s, m * k, m * s * k]
    if form in ("sq", "cub"):
        cols += [s ** 2, k ** 2, m ** 2]
    if form == "cub":
        cols += [s ** 3, k ** 3, m ** 3]
    return np.column_stack([np.ones(len(d))] + cols)


def fit(d, form):
    return np.linalg.lstsq(X(d, form), np.log(d["Alpha"].astype(float).values), rcond=None)[0]


def ape(d, b, form):
    return pd.Series(np.abs(np.exp(X(d, form) @ b) / d["Alpha"].values - 1) * 100, index=d.index)


def at_best(train, test, b, form):
    """Test MAPE at the interval with the lowest training MAPE."""
    tr = ape(train, b, form).groupby(train["SamplingInterval"]).mean()
    si = tr.idxmin()
    t = test[test["SamplingInterval"] == si]
    return ape(t, b, form).mean() if len(t) else np.nan, si


def load():
    P = {}
    for r in ROOTS:
        for f in sorted(glob.glob(f"{r}/*_allData.csv")):
            P[os.path.basename(f)[:-len("_allData.csv")]] = prepare(pd.read_csv(f))
    return P


def split_sets(raw, p, tc, sub_of):
    """Training rows from the selection on the training inputs (memprint build --select-on train), and the held-out
    input's rows at the same intervals."""
    q = prepare(raw[raw.Config != tc])
    for sub in q.subsets:
        train = q.subset(sub)
        test, _ = test_rows(raw, p, tc, train.SamplingInterval.unique())
        yield sub, train, test


def terms(P, raw):
    rows = []
    for w, p in P.items():
        for split, attr in SPLITS.items():
            tc = getattr(p, attr)
            for sub, train, test in split_sets(raw[w], p, tc, None):
                for form in ("base", "sq", "cub"):
                    m, si = at_best(train, test, fit(train, form), form)
                    rows.append(dict(workload=w, split=split, subset=sub, form=form, test_mape=m, train_si=si))
    return pd.DataFrame(rows)


def pooled(P, raw):
    """The other workloads enter with all their inputs and their own selection (they are known workloads)."""
    rows = []
    for w, p in P.items():
        for split, attr in SPLITS.items():
            tc = getattr(p, attr)
            for sub, train, test in split_sets(raw[w], p, tc, None):
                others = [q.subset(sub) for v, q in P.items() if v != w and sub in q.subsets]
                for kind, pool in (("pooled", others), ("pooled_own", others + [train])):
                    m, si = at_best(train, test, fit(pd.concat(pool), "base"), "base")
                    rows.append(dict(workload=w, split=split, subset=sub, form=kind, test_mape=m, train_si=si))
    return pd.DataFrame(rows)


def slope(P):
    rows = []
    for w, p in P.items():
        a = p.data.groupby(["Config", "SamplingInterval"]).Alpha.mean().reset_index()
        for c, g in a.groupby("Config"):
            for rng, h in (("all", g), ("1000-10000", g[g.SamplingInterval.between(1000, 10000)])):
                if len(h) >= 2:
                    rows.append(dict(workload=w, config=c, range=rng, n=len(h),
                                     slope=np.polyfit(np.log(h.SamplingInterval), np.log(h.Alpha), 1)[0]))
    return pd.DataFrame(rows)


def rank(P, raw):
    rows = []
    for w, p in P.items():
        iv = p.intervals
        for split, attr in SPLITS.items():
            tc = getattr(p, attr)
            q = prepare(raw[w][raw[w].Config != tc])
            r = dict(workload=w, split=split, n_intervals=len(iv), intervals=" ".join(map(str, iv)))
            r["best_train"] = q.best_interval
            r["rank_train"] = iv.index(q.best_interval) if q.best_interval in iv else np.nan
            r["best_all"] = p.best_interval
            r["rank_all"] = iv.index(p.best_interval) if p.best_interval is not None else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    raw = {}
    for r in ROOTS:
        for f in sorted(glob.glob(f"{r}/*_allData.csv")):
            raw[os.path.basename(f)[:-len("_allData.csv")]] = pd.read_csv(f)
    P = {w: prepare(d) for w, d in raw.items()}
    print(len(P), "workloads", flush=True)

    t = terms(P, raw)
    ref = pd.concat([pd.read_csv(f"{r}/fits.csv") for r in ROOTS])
    chk = t[t.form == "base"].merge(ref, on=["workload", "split", "subset"])
    print("base reproduces fits.csv: max |diff|", round((chk.test_mape - chk.test_mape_train_si).abs().max(), 6), "over", len(chk))
    t = pd.concat([t, pooled(P, raw)], ignore_index=True)
    t.to_csv(f"{OUT}/modelcheck_terms.csv", index=False)
    pd.set_option("display.width", 200)
    for split in SPLITS:
        for sub in SUBSETS:
            g = t[(t.split == split) & (t.subset == sub)].pivot(index="workload", columns="form", values="test_mape")
            line = [f"{split} {sub:3s} n={len(g)}"]
            for f in ("base", "sq", "cub", "pooled", "pooled_own"):
                worse = (g[f] > g.base).sum() if f != "base" else 0
                line.append(f"{f} {g[f].mean():.2f}/{g[f].median():.2f}" + (f" worse {worse}" if f != "base" else ""))
            print("  ".join(line))

    s = slope(P)
    s.to_csv(f"{OUT}/modelcheck_slope.csv", index=False)
    for rng, g in s.groupby("range"):
        print(f"slope {rng}: {len(g)} curves, median {g.slope.median():.3f}, 10th-90th {g.slope.quantile(.1):.3f} to "
              f"{g.slope.quantile(.9):.3f}, range {g.slope.min():.3f} to {g.slope.max():.3f}")

    r = rank(P, raw)
    r.to_csv(f"{OUT}/modelcheck_rank.csv", index=False)
    for col in ("rank_train", "rank_all"):
        print(col, r[col].value_counts(dropna=False).sort_index().to_dict())
    print(r.to_string(index=False))


if __name__ == "__main__" and not sys.argv[1:]:
    main()


def l2o_trainsel(raw):
    """L2O with the best monotone interval chosen on the training inputs only (build.py chooses it on all inputs,
    the held-out one included); the same fit and train_si protocol otherwise."""
    rows = []
    for w, d0 in raw.items():
        p = prepare(d0)
        best_all = p.best_interval
        for split, attr in SPLITS.items():
            tc = getattr(p, attr)
            q = prepare(d0[d0.Config != tc])
            for name, best in (("all", best_all), ("train", q.best_interval)):
                if best is None:
                    continue
                p.best_interval = best
                d = p.subset("L2O")
                train, test = d[d.Config != tc], d[d.Config == tc]
                m, si = at_best(train, test, fit(train, "base"), "base")
                rows.append(dict(workload=w, split=split, selection=name, best=best, test_mape=m, train_si=si))
    return pd.DataFrame(rows)


if __name__ == "__main__" and sys.argv[1:] == ["l2o_trainsel"]:
    raw = {}
    for r in ROOTS:
        for f in sorted(glob.glob(f"{r}/*_allData.csv")):
            raw[os.path.basename(f)[:-len("_allData.csv")]] = pd.read_csv(f)
    t = l2o_trainsel(raw)
    t.to_csv(f"{OUT}/modelcheck_l2o_trainsel.csv", index=False)
    g = t.pivot_table(index=["workload", "split"], columns="selection", values="test_mape")
    for split, h in g.groupby(level="split"):
        print(split, f"n={len(h)} all {h['all'].mean():.2f}/{h['all'].median():.2f}  train {h['train'].mean():.2f}/{h['train'].median():.2f}  "
              f"changed {(h['all'] != h['train']).sum()}  worse {(h['train'] > h['all']).sum()}")
