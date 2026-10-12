"""Score the committed blind predictions against the traces (python paper_generalization/blind/score.py).

Per config: footprint error and alpha MAPE over every bin of the 13 intervals. Per split (EXTRA: largest
config held out, INTER: middle config): the prediction against the program's own MemPrint model trained
on its other configs. Writes paper_generalization/blind/scores.csv and splits.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from memprint import similarity as sim  # noqa: E402
from memprint.dataset import prepare  # noqa: E402
from memprint.model import Model  # noqa: E402

rows, splits = [], []
for prog in ("hpccg", "lulesh"):
    pred = pd.read_csv(HERE / f"{prog}_predicted.csv").assign(config=lambda d: d.config.astype(str))
    data = pd.read_csv(ROOT / "data" / "blind" / "data" / f"{prog}_allData.csv").assign(
        Config=lambda d: d.Config.astype(str))
    P = prepare(data)
    bins = P.data[P.data.SamplingInterval.isin(sim.INTERVALS)]
    for c in P.configs:
        g = pred[pred.config == c]
        d = bins[bins.Config == c]
        alpha = g.set_index("k")["alpha"]
        a_pred = d.SamplingInterval.map(alpha).to_numpy(float)
        a_meas = d.Alpha.to_numpy(float)
        fp = float(g[g.k == 1].footprint.iloc[0])
        rows.append({"program": prog, "config": int(c), "truth": P.truth[c], "predicted": fp,
                     "fp_error": (fp - P.truth[c]) / P.truth[c] * 100,
                     "alpha_mape": float(np.mean(np.abs(a_pred - a_meas) / a_meas) * 100),
                     "coverage": float(g.coverage.iloc[0])})
    for split, test in (("EXTRA", P.last_config), ("INTER", P.middle_config)):
        tint = int(test)
        d = bins[bins.Config == test]
        a_meas = d.Alpha.to_numpy(float)
        g = pred[pred.config == test].set_index("k")["alpha"]
        a_pred = d.SamplingInterval.map(g).to_numpy(float)
        own = Model.fit(bins[bins.Config != test]).predict(d)["Predicted_Alpha"].to_numpy()
        splits.append({"program": prog, "split": split, "config": int(test),
                       "static_alpha": float(np.mean(np.abs(a_pred - a_meas) / a_meas) * 100),
                       "own_model": float(np.mean(np.abs(own - a_meas) / a_meas) * 100),
                       "fp_error": float(pd.DataFrame(rows).query("program == @prog and config == @tint")
                                         .fp_error.iloc[0])})
pd.DataFrame(rows).to_csv(HERE / "scores.csv", index=False)
pd.DataFrame(splits).to_csv(HERE / "splits.csv", index=False)
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(2).to_string(index=False))
print(pd.DataFrame(splits).round(2).to_string(index=False))
