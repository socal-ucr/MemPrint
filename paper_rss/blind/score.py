"""Score the RSS model on paper_generalization's blind configurations (PROTOCOL.md).

    ~/environments/memfp/bin/python paper_rss/blind/score.py   ->  paper_rss/blind/scores.csv, splits.csv
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(HERE, "..", "..", "paper_generalization")
RUNS = os.path.expanduser("~/memprint-data/2026-10-28-rssblind")
SIZE = {"hpccg": lambda c: int(c) ** 3, "lulesh": lambda c: int(c) ** 3,
        "IS": {"S": 2 ** 16, "W": 2 ** 20, "A": 2 ** 23}.get,
        "CG": {"S": 1400 * 8 ** 2, "W": 7000 * 9 ** 2, "A": 14000 * 12 ** 2}.get,
        "MG": {"S": 32 ** 3, "W": 128 ** 3, "A": 256 ** 3}.get}
INTER = {"hpccg": "16", "lulesh": "12"}


def truths():
    a = pd.read_csv(f"{G}/blind/scores.csv").rename(columns={"fp_error": "static_error"})
    b = pd.read_csv(f"{G}/blind2/scores.csv").rename(columns={"class": "config", "fp_error": "static_error"})
    t = pd.concat([a[["program", "config", "truth", "static_error"]], b[["program", "config", "truth", "static_error"]]])
    t["config"] = t.config.astype(str)
    return t


def measured():
    rows = []
    for prog in SIZE:
        for cfg in sorted(os.listdir(f"{RUNS}/{prog}")):
            peaks = []
            for rep in (1, 2, 3):
                r = pd.read_csv(f"{RUNS}/{prog}/{cfg}/{rep}.csv")
                peaks.append((r.anon_bytes - r.free_bytes).clip(lower=0).max())
            rows.append(dict(program=prog, config=cfg, measured=float(np.median(peaks)), size=SIZE[prog](cfg)))
    return pd.DataFrame(rows)


def transfer(m, prog, target, floor_smaller=False):
    """The peak at `target` from the program's other configurations (earlier.transfer's rule). With floor_smaller
    (post hoc), the floor is the largest peak of the smaller configurations only, as the rule was written for
    extrapolation and floors an interpolation at the larger configurations' peaks."""
    g = m[(m.program == prog) & (m.config != target) & m.truth.notna()].sort_values("size")
    x, y, xt = g["size"].to_numpy(float), g.measured.to_numpy(), float(m[(m.program == prog) & (m.config == target)]["size"].iloc[0])
    if len(g) >= 2 and np.ptp(x) > 0:
        a, b = np.polyfit(x, y, 1)
        peak = a * xt + b
    else:
        peak = y[-1] * xt / x[-1]
    return max(peak, y[x < xt].max() if floor_smaller and (x < xt).any() else y.max())


def main():
    m = measured().merge(truths(), on=["program", "config"], how="left")
    # R: the median of truth - measured over the other programs' configurations
    m["R"] = [float((m[(m.program != p) & m.truth.notna()].truth - m[(m.program != p) & m.truth.notna()].measured).median())
              for p in m.program]
    m["err_a"] = 100 * (m.measured - m.truth) / m.truth
    m["err_b"] = 100 * (m.measured + m.R - m.truth) / m.truth
    splits = []
    for prog in SIZE:
        have = m[(m.program == prog) & m.truth.notna()].sort_values("size")
        for split, target in (("EXTRA", have.config.iloc[-1]), ("INTER", INTER.get(prog))):
            if target is None:
                continue
            row = m[(m.program == prog) & (m.config == target)].iloc[0]
            pred = transfer(m, prog, target) + row.R
            post = transfer(m, prog, target, floor_smaller=True) + row.R
            splits.append(dict(program=prog, split=split, config=target, truth=row.truth, predicted=pred,
                               err=100 * (pred - row.truth) / row.truth,
                               err_posthoc=100 * (post - row.truth) / row.truth, measured_err=row.err_b,
                               static_error=row.static_error))
    m.to_csv(f"{HERE}/scores.csv", index=False)
    s = pd.DataFrame(splits)
    s.to_csv(f"{HERE}/splits.csv", index=False)
    pd.set_option("display.width", 200)
    print(m.round(2).to_string(index=False))
    print(s.round(2).to_string(index=False))
    t = m[m.truth.notna()]
    print("median |err| a, b, static:", round(t.err_a.abs().median(), 2), round(t.err_b.abs().median(), 2),
          round(t.static_error.abs().median(), 2))


if __name__ == "__main__":
    main()
