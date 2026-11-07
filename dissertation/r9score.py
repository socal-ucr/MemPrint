"""R9: R1's estimates (pss1 and R8's event sampler) against the pages-touched truth (r9pages.sh) -> $OUT/r9_pages.csv.
R is recomputed as the median of pages truth - measured over the other workloads' test inputs."""
import glob
import os

import pandas as pd

O = os.path.expanduser("~/memprint-data/2026-11-05-diss-ch2")


def truth(fp, w, c):
    f = [x for x in glob.glob(f"{O}/r9/all/{fp}/{w}/{c}/Buffered_*.csv") if "SubSample" not in x]
    return float(pd.read_csv(f[0]).MemUsageObs.iloc[0]) if f else float("nan")


rows = []
for sampler, tag in (("pss1", ""), ("event", "_r8")):
    for r in pd.read_csv(f"{O}/r1{tag}_splits.csv").itertuples():
        rows.append(dict(sampler=sampler, workload=r.workload, config=r.config, split=r.split, bytes_truth=r.truth,
                         bytes_rerun=truth("bytes", r.workload, r.config), pages_truth=truth("pages", r.workload, r.config),
                         measured=r.measured, err_b_bytes=r.err_b))
d = pd.DataFrame(rows)
for s, g in d.groupby("sampler"):
    d.loc[g.index, "R_pages"] = g.workload.map({w: (g[g.workload != w].pages_truth - g[g.workload != w].measured).median()
                                                for w in g.workload})
d["err_a_pages"] = 100 * (d.measured - d.pages_truth) / d.pages_truth
d["err_b_pages"] = 100 * (d.measured + d.R_pages - d.pages_truth) / d.pages_truth
d["pages_over_bytes"] = d.pages_truth / d.bytes_truth
d.to_csv(f"{O}/r9_pages.csv", index=False)
for (s, sp), g in d.groupby(["sampler", "split"]):
    print(s, sp, "b vs bytes %.2f/%.2f" % (g.err_b_bytes.abs().mean(), g.err_b_bytes.abs().median()),
          "b vs pages %.2f/%.2f" % (g.err_b_pages.abs().mean(), g.err_b_pages.abs().median()),
          "within 10%:", (g.err_b_pages.abs() < 10).sum())
