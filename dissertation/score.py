"""Analysis for dissertation/PROTOCOL.md.

    PYTHONPATH=analysis ~/environments/memfp/bin/python -W ignore dissertation/score.py [part ...]
Parts:
    plan   M3's runs (workload, split, config, interval) from M1's bytes fits -> $OUT/m3_list.txt
    m1     the TACO protocol on the taco, starts and bytes tables -> $OUT/m1/<set>/data/fits.csv, m1_summary.csv
    m8     bins and intervals per table, intervals dropped per workload -> $OUT/m8.csv
"""
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "analysis"))
from memprint.dataset import prepare  # noqa: E402

OUT = os.path.expanduser("~/memprint-data/2026-11-05-diss-ch2")
PY = os.path.expanduser("~/environments/memfp/bin/python")
SETS = {"taco": os.path.expanduser("~/memory_estimator/tools/data"),
        "starts": os.path.join(REPO, "data/polybench-starts/data"),
        "bytes": os.path.join(REPO, "data/polybench-bytes/data")}
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


if __name__ == "__main__":
    for part in sys.argv[1:] or ["m1", "plan", "m8"]:
        print("==", part)
        globals()[part]()
