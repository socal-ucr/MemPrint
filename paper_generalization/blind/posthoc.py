"""Post-hoc LULESH prediction (after seeing the traces): the interpreter with partial values for
branch assignments (Partial, interp.py), spectra in data/blind/data/static_cpp_posthoc. Same baseline
and scoring as the blind run. Writes posthoc_scores.csv. python paper_generalization/blind/posthoc.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from memprint import irregular, similarity as sim  # noqa: E402
from memprint.dataset import prepare  # noqa: E402
from memprint.static import load  # noqa: E402
from memprint.static.runs import predictions  # noqa: E402

GAP = ROOT / "data" / "gap-bytes" / "data"
sources = []
for w in ["gap_pr", "gap_bfs", "gap_cc", "gap_bc", "gap_tc", "gap_sssp", "gap_pr_kron", "gap_bfs_kron", "gap_cc_kron",
          "gap_bc_kron", "gap_tc_kron", "gap_sssp_kron"]:
    a = pd.read_csv(GAP / f"{w}_allData.csv").assign(Config=lambda d: d.Config.astype(str))
    specs = {f.stem.rsplit("-", 1)[1]: load(f)[0] for f in sorted((GAP / "static_auto").glob(f"{w}-*.npz"))}
    sources.append((prepare(a), specs, sim.bin_summary(a)))
baseline = irregular.fit_baseline(sources)
d = ROOT / "data" / "blind" / "data" / "static_cpp_posthoc"
spectra, meta = {}, {}
for f in sorted(d.glob("lulesh-*.npz")):
    c = f.stem.rsplit("-", 1)[1]
    spectra[c], meta[c] = load(f)
pred = predictions("lulesh", spectra, meta, baseline).assign(config=lambda x: x.config.astype(str))
data = pd.read_csv(ROOT / "data" / "blind" / "data" / "lulesh_allData.csv").assign(Config=lambda x: x.Config.astype(str))
P = prepare(data)
bins = P.data[P.data.SamplingInterval.isin(sim.INTERVALS)]
rows = []
for c in P.configs:
    g = pred[pred.config == c]
    b = bins[bins.Config == c]
    a_pred = b.SamplingInterval.map(g.set_index("k")["alpha"]).to_numpy(float)
    a_meas = b.Alpha.to_numpy(float)
    fp = float(g[g.k == 1].footprint.iloc[0])
    rows.append({"program": "lulesh", "config": int(c), "fp_error": (fp - P.truth[c]) / P.truth[c] * 100,
                 "alpha_mape": float(np.mean(np.abs(a_pred - a_meas) / a_meas) * 100),
                 "coverage": float(g.coverage.iloc[0])})
out = pd.DataFrame(rows)
out.to_csv(HERE / "posthoc_scores.csv", index=False)
print(out.round(2).to_string(index=False))
