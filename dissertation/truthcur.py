"""The scheduling corpus's truths traced again with the current Pin tool (~/memprint-data/2026-11-17-truthcur, built
by the job list there) against the October tools' (bytes in use: 2026-10-11 tool, 2026-10-13-sched-llm/corpus;
pages in use: 2026-10-12 tool, 2026-10-12-resize/corpus-pages and 2026-10-26-rsscurve/llm).

    PYTHONPATH=analysis ~/environments/memfp/bin/python dissertation/truthcur.py   ->  $N/compare.csv

Per program and truth: the new peak over the old, and the new curve's area over the old (100 steps of the run).
"""
import os

import numpy as np
import pandas as pd

from memprint.sched import corpus as C

N = os.path.expanduser("~/memprint-data/2026-11-17-truthcur")
OLD = {"truth": [os.path.expanduser("~/memprint-data/2026-10-13-sched-llm/corpus")],
       "truthpages": [os.path.expanduser("~/memprint-data/2026-10-12-resize/corpus-pages"),
                      os.path.expanduser("~/memprint-data/2026-10-26-rsscurve/llm")]}


def curve(d):
    got = C.truth(d)
    return None if got is None else C.resample(*got)


def main():
    rows = []
    for prog in sorted(os.listdir(N)):
        if not os.path.isdir(f"{N}/{prog}"):
            continue
        for inp in sorted(os.listdir(f"{N}/{prog}")):
            for m, roots in OLD.items():
                if not os.path.isdir(f"{N}/{prog}/{inp}/{m}"):
                    continue
                new = curve(f"{N}/{prog}/{inp}/{m}")
                old = next((curve(f"{r}/{prog}/{inp}/{m}") for r in roots if os.path.isdir(f"{r}/{prog}/{inp}/{m}")), None)
                if new is None or old is None:
                    rows.append(dict(prog=prog, input=inp, truth=m, missing="new" if new is None else "old"))
                    continue
                rows.append(dict(prog=prog, input=inp, truth=m, old_peak_mb=old.max() / 1e6, new_peak_mb=new.max() / 1e6,
                                 peak_ratio=new.max() / old.max(), area_ratio=new.sum() / old.sum(),
                                 max_step_diff=float(np.abs(new - old).max() / old.max())))
    d = pd.DataFrame(rows)
    d.to_csv(f"{N}/compare.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.round(3).to_string(index=False))
    for m, g in d.dropna(subset=["peak_ratio"]).groupby("truth"):
        off = g[(g.peak_ratio - 1).abs() > 0.01]
        print(f"{m}: {len(g)} curves; peak ratio median {g.peak_ratio.median():.4f}, range {g.peak_ratio.min():.3f} to "
              f"{g.peak_ratio.max():.3f}; more than 1% off: {', '.join(f'{r.prog} {r.input} {r.peak_ratio:.3f}' for r in off.itertuples()) or 'none'}")


if __name__ == "__main__":
    main()
