"""Analysis for dissertation/PROTOCOL.md.

    PYTHONPATH=analysis ~/environments/memfp/bin/python -W ignore dissertation/score.py [part ...]
Parts:
    plan   M3's runs (workload, split, config, interval) from M1's bytes fits -> $OUT/m3_list.txt
    m1     the TACO protocol on the taco, starts and bytes tables -> $OUT/m1/<set>/data/fits.csv, m1_summary.csv
    m8     bins and intervals per table, intervals dropped per workload -> $OUT/m8.csv
    m4     miniVite truths under three definitions and its six test MAPEs -> $OUT/m4.csv
    m3     sampler bins against splitter bins, the decision rule -> $OUT/m3_runs.csv, m3_mape.csv
    m6     Massif and DHAT peaks against the bytes truth and MemPrint -> $OUT/m6.csv
    m7     median wall time per configuration -> $OUT/m7.csv
"""
import glob
import json
import os
import re
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "analysis"))
from memprint.dataset import prepare  # noqa: E402
from memprint.model import Model  # noqa: E402

OUT = os.path.expanduser("~/memprint-data/2026-11-05-diss-ch2")
PY = os.path.expanduser("~/environments/memfp/bin/python")
SETS = {"taco": os.path.expanduser("~/memory_estimator/tools/data"),
        "starts": os.path.join(REPO, "data/polybench-starts/data"),
        "bytes": os.path.join(REPO, "data/polybench-bytes/data")}
FINAL = f"{OUT}/m1/bytes-m4/data"   # bytes tables with M4's miniVite: the chapter's fits
WORKLOADS = open(os.path.join(REPO, "analysis/workloads.txt")).read().split()


def m1(sets=SETS):
    rows = []
    for name, src in sets.items():
        root = f"{OUT}/m1/{name}"
        os.makedirs(f"{root}/data", exist_ok=True)
        for w in WORKLOADS:
            subprocess.run(["cp", f"{src}/{w}_allData.csv", f"{root}/data/"], check=True)
        with open(f"{root}/build.log", "w") as log:
            subprocess.run([PY, "-W", "ignore", "-m", "memprint", "--root", root, "build", "--latex"], check=True,
                           stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": f"{REPO}/analysis"})
        f = pd.read_csv(f"{root}/data/fits.csv")
        g = f.groupby(["split", "subset"]).test_mape_train_si.agg(["mean", "median", "max", "count"]).reset_index()
        rows.append(g.assign(set=name))
    s = pd.concat(rows)
    s.to_csv(f"{OUT}/m1_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.pivot_table(index=["split", "subset"], columns="set", values=["mean", "max"]).round(2).to_string())


def plan():
    f = pd.read_csv(f"{OUT}/m1/bytes/data/fits.csv")
    lines = []
    for w in WORKLOADS:
        if w == "miniVite":   # after M4's retrace
            continue
        p = prepare(pd.read_csv(f"{OUT}/m1/bytes/data/{w}_allData.csv"))
        for split, config in (("EXTRA", p.last_config), ("INTER", p.middle_config)):
            g = f[(f.workload == w) & (f.split == split)]
            ks = set(g.train_si.astype(int)) | set(g.best_interval.dropna().astype(int))
            lines += [f"{w} {split} {config} {k}" for k in sorted(ks)]
    open(f"{OUT}/m3_list.txt", "w").write("\n".join(lines) + "\n")
    print(len(lines), "runs per sampler and seed")


def m8():
    rows = []
    for name, src in SETS.items():
        for w in WORKLOADS:
            d = pd.read_csv(f"{src}/{w}_allData.csv")
            bins = d[d.SamplingInterval > 1].groupby(["Config", "SamplingInterval"]).size()
            kept = prepare(d).intervals
            rows.append(dict(set=name, workload=w, bins_min=bins.min(), bins_max=bins.max(),
                             intervals=d.SamplingInterval[d.SamplingInterval > 1].nunique(), used=len(kept),
                             dropped=" ".join(str(k) for k in sorted(set(d.SamplingInterval) - set(kept) - {1}))))
    m = pd.DataFrame(rows)
    m.to_csv(f"{OUT}/m8.csv", index=False)
    print(m[m.set == "taco"].to_string(index=False))


def m4():
    taco = pd.read_csv(f"{SETS['taco']}/miniVite_allData.csv")
    old = pd.read_csv(f"{SETS['bytes']}/miniVite_allData.csv")
    new = pd.read_csv(f"{FINAL}/miniVite_allData.csv")
    t = lambda d: d[d.SamplingInterval == 1].groupby("Config").MemUsageObs.first()
    m = pd.DataFrame({"starts_B": t(taco), "bytes_alloc_B": t(old), "bytes_B": t(new)})
    m.index = m.index.astype(int)
    m = m.sort_index().assign(vs_alloc_pct=lambda x: (x.bytes_B / x.bytes_alloc_B - 1) * 100,
                              vs_starts_pct=lambda x: (x.bytes_B / x.starts_B - 1) * 100)
    m.to_csv(f"{OUT}/m4.csv")
    print(m.round(2).to_string())
    pub = {"EXTRA": [3.98, 4.98, 12.90], "INTER": [2.94, 1.04, 1.57]}
    f = pd.read_csv(f"{FINAL}/fits.csv")
    f = f[f.workload == "miniVite"].assign(published=lambda x: [pub[r.split][["NZ", "MT", "L2O"].index(r.subset)]
                                                                   for r in x.itertuples()])
    print(f[["split", "subset", "train_si", "test_mape_train_si", "best_interval", "test_mape_best",
             "published"]].round(2).to_string(index=False))


def sampler_bins(d):
    """MemUsageObs of each bin of one sampler run directory."""
    files = glob.glob(f"{d}/*_SubSample_*_bin_*.csv")
    return pd.Series([pd.read_csv(x).MemUsageObs.iloc[0] for x in files], dtype=float)


def m3():
    fits = pd.read_csv(f"{FINAL}/fits.csv")
    tables = {w: pd.read_csv(f"{FINAL}/{w}_allData.csv") for w in WORKLOADS}
    runs, mapes = [], []
    for done in open(f"{OUT}/m3/done.txt"):
        d, rc = done.split()
        s, seed, w, config, k = d.split("/")[-5:]
        k = int(k)
        split = "EXTRA" if config == str(prepare(tables[w]).last_config) else "INTER"
        a = tables[w]
        a = a.assign(Config=a.Config.astype(str))
        truth = a[(a.SamplingInterval == 1) & (a.Config == config)].MemUsageObs.iloc[0]
        ref = a[(a.SamplingInterval == k) & (a.Config == config)]
        b = sampler_bins(d)
        r = dict(sampler=s, seed=int(seed), workload=w, split=split, config=config, k=k, rc=int(rc), bins=len(b),
                 zero_bins=int((b == 0).sum()), mean_ratio=b.mean() / ref.MemUsageObs.mean(),
                 sd_ratio=b.std() / ref.MemUsageObs.std())
        runs.append(r)
        if r["zero_bins"] or len(b) < 2:
            continue
        test = pd.DataFrame({"MemUsageObs": b, "SamplingInterval": k, "SD_MemUsage": b.std(), "Alpha": truth / b})
        for fit in fits[(fits.workload == w) & (fits.split == split)].itertuples():
            pred = Model.from_row(pd.Series(fit._asdict())).predict(test)
            sampler_mape = (pred.ErrorRate.abs()).mean()
            for basis, si, ref_mape in (("train_si", fit.train_si, fit.test_mape_train_si),
                                        ("best", fit.best_interval, fit.test_mape_best)):
                if si == k:
                    mapes.append(dict(sampler=s, seed=int(seed), workload=w, split=split, subset=fit.subset,
                                      basis=basis, k=k, sampler_mape=sampler_mape, splitter_mape=ref_mape))
    runs, mapes = pd.DataFrame(runs), pd.DataFrame(mapes)
    runs.to_csv(f"{OUT}/m3_runs.csv", index=False)
    mapes.to_csv(f"{OUT}/m3_mape.csv", index=False)
    pd.set_option("display.width", 200)
    print("runs", len(runs), "failed", int((runs.rc != 0).sum()), "with a zero bin", int((runs.zero_bins > 0).sum()))
    print(runs.groupby("sampler")[["mean_ratio", "sd_ratio"]].describe(percentiles=[.1, .5, .9]).T.round(3))
    g = mapes.groupby(["sampler", "basis", "split", "subset", "workload"])[["sampler_mape", "splitter_mape"]].mean()
    g["diff"] = g.sampler_mape - g.splitter_mape
    print(g.groupby(["sampler", "basis", "split", "subset"]).agg(
        sampler_mean=("sampler_mape", "mean"), splitter_mean=("splitter_mape", "mean"),
        diff_median=("diff", "median"), n=("diff", "size")).round(2).to_string())
    b = g.loc[("boot", "train_si")]
    sd = runs[runs.sampler == "boot"].sd_ratio.median()
    disj = b["diff"].median() > 2 or not 0.9 <= sd <= 1.1
    print(f"decision: boot diff median {b['diff'].median():.2f} points, SD ratio median {sd:.3f}"
          f" -> report {'disjoint' if disj else 'bootstrap'}")


def m6():
    rows = []
    for x in sorted(glob.glob(f"{OUT}/m6/*.massif")):
        name = os.path.basename(x)[:-7]
        w, c = name.rsplit("-", 1)
        txt = open(x).read()
        heap = [int(h) + int(e) for h, e in re.findall(r"mem_heap_B=(\d+)\nmem_heap_extra_B=(\d+)", txt)]
        useful = [int(h) for h in re.findall(r"mem_heap_B=(\d+)", txt)]
        dhat = json.load(open(f"{OUT}/m6/{name}.dhat"))
        a = pd.read_csv(f"{FINAL}/{w}_allData.csv")
        truth = a[(a.SamplingInterval == 1) & (a.Config.astype(str) == c)].MemUsageObs.iloc[0]
        rows.append(dict(workload=w, config=c, truth_B=truth, massif_B=max(useful), massif_extra_B=max(heap),
                         dhat_B=sum(p["gb"] for p in dhat["pps"])))
    m = pd.DataFrame(rows)
    for c in ("massif_B", "dhat_B"):
        m[c.replace("_B", "_over_truth")] = m[c] / m.truth_B
    m.to_csv(f"{OUT}/m6.csv", index=False)
    print(m.round(3).to_string(index=False))


def m7():
    t = pd.read_csv(f"{OUT}/m7/timing.csv")
    m = t.groupby("config").seconds.agg(["median", "min", "max", "count"])
    m["slowdown"] = m["median"] / m.loc["native", "median"]
    m.to_csv(f"{OUT}/m7.csv")
    print(m.sort_values("median").round(3).to_string())


if __name__ == "__main__":
    for part in sys.argv[1:] or ["m1", "plan", "m8"]:
        print("==", part)
        globals()[part]()
