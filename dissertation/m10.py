"""M10: the published per-workload models (memory_estimator/tools/data/{extra,inter}polation_models.csv, 4 decimals)
against the refitted selected models of M1 on the published tables ($OUT/m1/taco/data/models.csv) -> $OUT/m10.csv"""
import os
import pandas as pd
OUT = os.path.expanduser("~/memprint-data/2026-11-05-diss-ch2")
PUB = os.path.expanduser("~/memory_estimator/tools/data")
B = ["intercept"] + [f"b{i}" for i in range(1, 8)]
r = pd.read_csv(f"{OUT}/m1/taco/data/models.csv")
r = r.set_index(["split", "workload"])   # the selected model of each workload and split
rows = []
for split, f in (("EXTRA", "extrapolation_models.csv"), ("INTER", "interpolation_models.csv")):
    p = pd.read_csv(f"{PUB}/{f}").set_index("workload")
    for w, x in p.iterrows():
        y = r.loc[(split, w)]
        d = (x[B].astype(float) - y[B].astype(float).round(4)).abs()
        rows.append(dict(split=split, workload=w, pub_rate=int(x.sample_rate), refit_rate=int(y.sample_rate), method=y.method,
                         max_coef_diff=d.max(), coef=d.idxmax(), refit_error=y.error))
m = pd.DataFrame(rows)
m.to_csv(f"{OUT}/m10.csv", index=False)
pd.set_option("display.width", 200)
print(m.round(4).to_string(index=False))
print("same interval:", (m.pub_rate == m.refit_rate).sum(), "of", len(m), "; identical to 4 decimals:", (m.max_coef_diff < 1e-9).sum())
