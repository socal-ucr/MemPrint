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


def measured(runs=RUNS, name="{rep}.csv"):
    """The median over 3 runs of the data estimate's peak; a run is <runs>/<prog>/<config>/<name>."""
    rows = []
    for prog in SIZE:
        for cfg in sorted(os.listdir(f"{runs}/{prog}")):
            peaks, b0, f0 = [], [], []
            for rep in (1, 2, 3):
                r = pd.read_csv(f"{runs}/{prog}/{cfg}/" + name.format(rep=rep))
                peaks.append((r.anon_bytes - r.free_bytes).clip(lower=0).max())
                r = r[r.rss_bytes > 0]   # the first sample: the in-run base of dissertation R10
                b0.append(max(r.anon_bytes.iloc[0] - r.free_bytes.iloc[0], 0)); f0.append(r.rss_bytes.iloc[0] - r.anon_bytes.iloc[0])
            rows.append(dict(program=prog, config=cfg, measured=float(np.median(peaks)), size=SIZE[prog](cfg),
                             base0=float(np.median(b0)), file0=float(np.median(f0))))
    return pd.DataFrame(rows)


def transfer(m, prog, target, floor_smaller=False, base=None):
    """The peak at `target` from the program's other configurations (earlier.transfer's rule). With floor_smaller
    (post hoc), the floor is the largest peak of the smaller configurations only, as the rule was written for
    extrapolation and floors an interpolation at the larger configurations' peaks. With base (a column of m, the
    run's own start-up memory in the units of measured; dissertation R10), the rule scales measured - base and adds
    the target run's base back."""
    g = m[(m.program == prog) & (m.config != target) & m.truth.notna()].sort_values("size")
    t = m[(m.program == prog) & (m.config == target)].iloc[0]
    x, y, xt = g["size"].to_numpy(float), g.measured.to_numpy(), float(t["size"])
    b0, bt = (g[base].to_numpy(), float(t[base])) if base else (0 * y, 0.0)
    if len(g) >= 2 and np.ptp(x) > 0:
        a, b = np.polyfit(x, y - b0, 1)
        peak = a * xt + b + bt
    else:
        peak = (y[-1] - b0[-1]) * xt / x[-1] + bt
    return max(peak, y[x < xt].max() if floor_smaller and (x < xt).any() else y.max())


def score(m):
    """The whole-run errors and the EXTRA and INTER splits of measured() runs."""
    m = m.merge(truths(), on=["program", "config"], how="left")
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
    return m, pd.DataFrame(splits)


def main():
    m, s = score(measured())
    m.to_csv(f"{HERE}/scores.csv", index=False)
    s.to_csv(f"{HERE}/splits.csv", index=False)
    pd.set_option("display.width", 200)
    print(m.round(2).to_string(index=False))
    print(s.round(2).to_string(index=False))
    t = m[m.truth.notna()]
    print("median |err| a, b, static:", round(t.err_a.abs().median(), 2), round(t.err_b.abs().median(), 2),
          round(t.static_error.abs().median(), 2))


if __name__ == "__main__":
    main()
