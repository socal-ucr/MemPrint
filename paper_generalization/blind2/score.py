"""Score the committed NPB predictions against the traces (python paper_generalization/blind2/score.py).

As fixed in PROTOCOL.md: per class, the footprint error (predicted against Pin's exact footprint) and the alpha
MAPE over every bin of the 13 splitter intervals. Writes paper_generalization/blind2/scores.csv.
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

rows = []
for prog in ("is", "cg", "mg"):
    pred = pd.read_csv(HERE / f"npb_{prog}_predicted.csv").assign(config=lambda d: d.config.astype(str))
    data = pd.read_csv(ROOT / "data" / "blind2" / "data" / f"npb_{prog}_allData.csv").assign(
        Config=lambda d: d.Config.astype(str))
    P = prepare(data)
    bins = P.data[P.data.SamplingInterval.isin(sim.INTERVALS)]
    for c in sorted(P.configs, key="SWABC".index):
        g = pred[pred.config == c]
        d = bins[bins.Config == c]
        a_pred = d.SamplingInterval.map(g.set_index("k")["alpha"]).to_numpy(float)
        a_meas = d.Alpha.to_numpy(float)
        fp = float(g[g.k == 1].footprint.iloc[0])
        rows.append({"program": prog.upper(), "class": c, "truth": P.truth[c], "predicted": fp,
                     "program_footprint": float(g.program_footprint.iloc[0]),
                     "fp_error": (fp - P.truth[c]) / P.truth[c] * 100,
                     "alpha_mape": float(np.mean(np.abs(a_pred - a_meas) / a_meas) * 100),
                     "coverage": float(g.coverage.iloc[0])})
out = pd.DataFrame(rows)
out.to_csv(HERE / "scores.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))

# the paper's table (MG class A: predicted, but its full trace ran out of memory)
END = r" \\"
lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Program & Class & Pin footprint & Predicted & Program part & Footprint error (\%) & $\alpha$ MAPE (\%)" + END,
         r"\midrule"]
for _, r in out.iterrows():
    lines.append(f"{r.program} & {r['class']} & {r.truth / 1e6:.2f}\\,MB & {r.predicted / 1e6:.2f}\\,MB & "
                 f"{r.program_footprint / 1e6:.2f}\\,MB & {r.fp_error:+.2f} & {r.alpha_mape:.1f}" + END)
mga = pd.read_csv(HERE / "npb_mg_predicted.csv").query("config == 'A' and k == 1").iloc[0]
lines.append(f"MG & A & \\multicolumn{{1}}{{c}}{{--}} & {mga.footprint / 1e6:.2f}\\,MB & {mga.program_footprint / 1e6:.2f}\\,MB & "
             r"\multicolumn{2}{c}{full trace out of memory}" + END)
lines += [r"\bottomrule", r"\end{tabular}"]
(HERE.parent / "tables" / "blind2.tex").write_text("\n".join(lines) + "\n")
