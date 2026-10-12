"""How much the static footprint depends on the runtime baseline, and how many traced programs the
baseline needs (python paper_generalization/baseline_sensitivity.py; writes data/baseline_sensitivity*.csv).

For every PolyBench kernel held out in turn and every config: the static footprint with no baseline, with a
baseline fitted on 1 or 3 other kernels drawn at random (5 draws each, seed 0), and on all 26 others.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "analysis"))
from memprint import lowo  # noqa: E402
from memprint.static import load  # noqa: E402

PB = ROOT / "data" / "polybench-bytes" / "data"
spectra = {tuple(f.stem.rsplit("-", 1)): load(f)[0] for f in sorted((PB / "static").glob("*.npz"))}
data = {w: pd.read_csv(PB / f"{w}_allData.csv") for w in sorted({w for w, _ in spectra})}
known = lowo.load(data, spectra)
names = sorted(known)
rng = np.random.default_rng(0)
rows = []
for held in names:
    C = known[held]
    others = [w for w in names if w != held]
    variants = [("none", None), ("all 26 others", lowo.fit_baseline([known[w] for w in others]))]
    for n in (1, 3):
        for _ in range(5):
            pick = list(rng.choice(others, n, replace=False))
            variants.append((f"{n} other kernel{'s' if n > 1 else ''}", lowo.fit_baseline([known[w] for w in pick])))
    for cfg, spec in C.spectra.items():
        if cfg not in C.prepared.truth:
            continue
        truth = C.prepared.truth[cfg]
        for name, b in variants:
            bfp = b.footprint if b is not None else 0.0
            rows.append({"held": held, "config": cfg, "variant": name, "truth": truth, "baseline": bfp,
                         "error": abs(spec.footprint + bfp - truth) / truth * 100})
d = pd.DataFrame(rows)
d.to_csv(HERE / "data" / "baseline_sensitivity.csv", index=False)
order = ["none", "1 other kernel", "3 other kernels", "all 26 others"]
s = d.groupby("variant").error.describe(percentiles=[.5, .9])[["50%", "90%", "max"]].reindex(order)
s.to_csv(HERE / "data" / "baseline_sensitivity_summary.csv")
share = d[d.variant == "all 26 others"].assign(share=lambda x: x.baseline / x.truth * 100)
by_cfg = pd.DataFrame({
    "median_truth": d[d.variant == "none"].groupby("config").truth.median(),
    "baseline_share": share.groupby("config").share.median(),
    "error_none": d[d.variant == "none"].groupby("config").error.median(),
    "error_1": d[d.variant == "1 other kernel"].groupby("config").error.median(),
    "error_all": d[d.variant == "all 26 others"].groupby("config").error.median()})
by_cfg = by_cfg.reindex(["MINI", "MINI2", "MINI3", "SMALL", "SMALL2", "SMALL3", "MEDIUM"])
by_cfg.to_csv(HERE / "data" / "baseline_sensitivity_by_config.csv")
print(s.round(2))
print(by_cfg.round(2))
