"""The runtime's pages B from the model's own run instead of an earlier plain run.

    PYTHONPATH=analysis ~/environments/memfp/bin/python paper_rss/inrun_base.py   ->  paper_rss/data/inrun_base.csv

compute.model's estimate 1 adds B, the smallest RSS of an earlier plain RSS run. Here B is taken from the model's run
itself: its file-backed RSS at the first sample (RSS - anonymous RSS at the shim's constructor, once the shared
libraries are loaded), so a job needs no earlier run. The truth keeps the earlier run's B, as in compute.truth.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compute as M  # noqa: E402
from memprint.sched import corpus as C  # noqa: E402

rows = []
for prog, inp in M.inputs().items():
    t, base = M.truth(prog, inp), M.base_of(prog, inp)
    for rep in (1, 2, 3):
        r = pd.read_csv(f"{M.O}/runs_all/{prog}/{rep}/pss.csv")
        r = r[r.rss_bytes > 0]
        for name, B in (("earlier", base), ("inrun", r.rss_bytes.iloc[0] - r.anon_bytes.iloc[0])):
            e = C.resample(r.seconds, np.clip(r.anon_bytes - r.free_bytes, 0, None) + B)
            rows.append(dict(prog=prog, rep=rep, B=name, B_mb=B / 1e6, **M.err(e, t)))
d = pd.DataFrame(rows).groupby(["prog", "B"])[["B_mb", "signed", "abs", "peak"]].median().reset_index()
d["group"] = np.where(d.prog.isin(M.HEAVY), "heavy", "other")
d.to_csv(f"{HERE}/data/inrun_base.csv", index=False)
print(d.pivot(index="prog", columns="B", values=["B_mb", "signed"]).round(1).to_string())
print(d.groupby(["group", "B"])[["signed", "abs", "peak"]].median().round(1))
print(d.groupby("B")[["signed", "abs", "peak"]].median().round(1))
