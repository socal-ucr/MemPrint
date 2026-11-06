"""M2: label against empirical bin interval of every sampler run in the given directories.
    OUTCSV=$OUT/m2_runs.csv python dissertation/m2scan.py ~/memory_estimator/workloads/miniVite/traces ~/memory_estimator/tools/traces/{correlation-old,correlation-recentOld,gemm_prev,mvt-1000,miniVite_fileInput}
"""
import glob, os, re, sys
import pandas as pd
rows = []
for d in sys.argv[1:]:
    runs = {}
    for f in glob.glob(f"{d}/*.csv"):
        b = os.path.basename(f)
        sub = re.search(r"_SubSample_([\d.]+)_bin_(\d+)\.csv$", b)
        stem = b[:sub.start()] if sub else b[:-4]
        label = sub.group(1) if sub else None
        m = re.match(r"(Sampled|Buffered)_(.+?)_(\d+)_(\d+)(?:_(.*))?$", stem)
        if not m: continue
        pre, name, i, pid, args = m.groups()
        key = (d, pre, name, int(i), pid, args or "")
        r = runs.setdefault(key, {"bins": [], "label": None, "main": None})
        try: v = pd.read_csv(f).iloc[0]
        except Exception: continue
        if label is None: r["main"] = v.CountObs
        else: r["label"] = float(label); r["bins"].append(v.CountObs)
    for (d, pre, name, i, pid, args), r in runs.items():
        if not r["bins"] or not r["main"]: continue
        mb = sum(r["bins"]) / len(r["bins"])
        rows.append(dict(dir=os.path.basename(d), prefix=pre, name=name, args=args, i=i, bins=len(r["bins"]),
                         label=r["label"], refs_per_bin_ratio=r["main"] / mb if mb else float("nan"),
                         empirical=i * r["main"] / mb if mb else float("nan")))
t = pd.DataFrame(rows)
t["label_over_emp"] = t.label / t.empirical
pd.set_option("display.width", 220)
g = t.groupby(["dir", "prefix", "i", "bins", "label"]).agg(runs=("pid" if "pid" in t else "name", "size"),
    emp_median=("empirical", "median"), ratio=("label_over_emp", "median")).reset_index()
print(g.round(2).to_string(index=False))
t.to_csv(os.environ.get("OUTCSV", "/dev/null"), index=False)
