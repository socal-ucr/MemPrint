"""Borrowing the growth of known programs for a program run once (paper_generalization's family B, for the RSS model).

    PYTHONPATH=analysis ~/environments/memfp/bin/python paper_rss/borrow.py   ->  paper_rss/data/borrow*.csv

Setting: program C has been run once, under the model, at its smallest input, and its largest input (the truth's)
is to be predicted. Its input sizes are known (sched.ksplus.INPUT_SIZE). The prediction is C's smallest-input data
curve (model1, rebased as the transfer study does) scaled to the peak p_S (x_L / x_S)^beta, never below p_S, plus the
runtime's pages, scored against the truth as compute.transfer is.

beta, the exponent of the peak in the input size, is fitted per known program W by least squares of log peak on log
size over all W's inputs. C's own inputs other than the smallest are not used. Methods:
  own           compute.transfer's model1X (C's two smaller inputs; the reference, as the own model in
                paper_generalization)
  proportional  beta = 1
  uniform       the mean beta of the known programs
  nearest       the beta of the known program nearest to C in the descriptor
  rbf           the mean beta weighted by exp(-d^2 / 2h^2), h the median distance of a known program to its nearest
                other known program
  oracle        the known beta with the smallest absolute whole-run error, chosen after the fact
Descriptor (each program at its smallest input, standardised over the known programs): the allocator's share of RSS
(1 - sum data / sum RSS), mean / peak of the data curve, the step of its peak / K, the share of steps at 90% of the
peak or more, the value at 10% of the run / peak, and the final value / peak. Leave one program out over the corpus
programs with a smaller input.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compute as M  # noqa: E402
from memprint.sched.earlier import _rebased  # noqa: E402
from memprint.sched.ksplus import INPUT_SIZE  # noqa: E402
from memprint.sched.model import K  # noqa: E402
from memprint.sched.run import EXCLUDE, rank  # noqa: E402


def programs():
    c = pd.read_csv(f"{M.D}/model/curves.csv", dtype={"input": str})
    c = c[[(p, i) not in EXCLUDE for p, i in zip(c.prog, c.input)]]
    out = {}
    for prog, target in M.inputs().items():
        g = c[c.prog == prog]
        ins = sorted(g[g.method == "model1"].input.unique(), key=rank)
        if len(ins) < 2 or prog not in INPUT_SIZE:
            continue
        by = {i: _rebased(g[g.input == i]) for i in ins}
        own = g[(g.input == target) & (g.method == "model1X")].sort_values("k").bytes.to_numpy()
        out[prog] = dict(target=target, inputs=ins, by=by, own=own,
                         size={i: float(INPUT_SIZE[prog][i]) for i in ins})
    return out


def beta(p):
    x = np.log([p["size"][i] for i in p["inputs"]])
    y = np.log([max(p["by"][i]["model1"].max(), 1.0) for i in p["inputs"]])
    return float(np.polyfit(x, y, 1)[0])


def descriptor(p):
    s = p["by"][p["inputs"][0]]
    d, r = s["model1"], s["rssrep1"]
    pk = max(d.max(), 1e-12)
    return np.array([1 - d.sum() / max(r.sum(), 1e-12), d.mean() / pk, np.argmax(d) / K, (d >= 0.9 * pk).mean(),
                     d[K // 10] / pk, d[-1] / pk])


def predict(p, b):
    s = p["inputs"][0]
    shape = p["by"][s]["model1"]
    pk = max(shape.max(), 1e-12)
    peak = max(pk * (p["size"][p["target"]] / p["size"][s]) ** b, pk)
    return shape / pk * peak


def main():
    P = programs()
    B = {k: beta(p) for k, p in P.items()}
    Z = {k: descriptor(p) for k, p in P.items()}
    rows, ranks = [], []
    for c, p in P.items():
        t, base = M.truth(c, p["target"]), M.base_of(c, p["target"])
        known = [w for w in P if w != c]
        z = np.array([Z[w] for w in known])
        mu, sd = z.mean(0), z.std(0) + 1e-12
        zs, zc = (z - mu) / sd, (Z[c] - mu) / sd
        d = np.linalg.norm(zs - zc, axis=1)
        nn = np.array([np.sort(np.linalg.norm(zs - zs[i], axis=1))[1] for i in range(len(known))])
        h = np.median(nn)
        w = np.exp(-d ** 2 / (2 * h ** 2)); w /= w.sum()
        bk = np.array([B[k] for k in known])
        err_k = np.array([abs(M.err(predict(p, b) + base, t)["abs"]) for b in bk])
        betas = {"proportional": 1.0, "uniform": bk.mean(), "nearest": bk[np.argmin(d)], "rbf": float(w @ bk),
                 "oracle": bk[np.argmin(err_k)]}
        if len(p["own"]):
            rows.append(dict(prog=c, method="own", beta=np.nan, **M.err(p["own"] + base, t)))
        for m, b in betas.items():
            rows.append(dict(prog=c, method=m, beta=b, **M.err(predict(p, b) + base, t)))
        rows.append(dict(prog=c, method="beta_C", beta=B[c], **M.err(predict(p, B[c]) + base, t)))
        ranks.append(dict(prog=c, rho=spearmanr(d, err_k)[0], nearest=known[int(np.argmin(d))]))
    r = pd.DataFrame(rows)
    r["group"] = np.where(r.prog.isin(M.HEAVY), "heavy", "other")
    k = pd.DataFrame(ranks)
    os.makedirs(f"{HERE}/data", exist_ok=True)
    r.to_csv(f"{HERE}/data/borrow.csv", index=False)
    k.to_csv(f"{HERE}/data/borrow_rank.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.Series(B).round(2).to_string())
    print(r.pivot(index="prog", columns="method", values="signed").round(1).to_string())
    print(r.groupby("method")[["signed", "abs", "peak"]].median().round(1))
    print(r.assign(a=r.signed.abs()).groupby("method").a.quantile(0.9).round(1))
    print(k.to_string(index=False)); print("median rho", round(k.rho.median(), 2))


if __name__ == "__main__":
    main()
