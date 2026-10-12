"""Per-site reservations from MemPrint's -sites profiles (the input of libmpr and the MemPrintResize pass).

    python -m memprint.sites DIR [DIR ...] -o reservations.txt [--quantile 0.9] [--min 65536]

Each DIR is one -sites run of the same program (pintool -sites: *_sites.csv, *_blocks.csv), usually
at different inputs. For every site with blocks of at least --min bytes (the ones libmpr serves),
the bytes a block will touch are fitted as a function of its requested size, from the high-water
offsets of its blocks: the highest byte touched, since a block is committed from its start.

A reservation is not binding (libmpr maps the rest of the block MAP_NORESERVE), so a prediction
below a block's use costs only accounting, never correctness. The fit covers --quantile of the
training blocks, by requested bytes (1.0: all of them) and, among the models that do, keeps the one that commits least:
    dead     no block touched anything
    const    a                (the high water does not grow with the request)
    ratio    a * size
    affine   a * size + b     (fitted as the lowest line above the covered blocks)
A site is left out when its best model would commit at least --keep of what was requested (no
saving). Writes "site model a b" lines, which libmpr reads (MPR_RESERVATIONS), and a summary CSV.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def load(dirs, min_size):
    blocks = []
    for i, d in enumerate(dirs):
        f = glob.glob(os.path.join(d, "*_blocks.csv"))
        if not f:
            raise SystemExit(f"{d}: no *_blocks.csv (run the Pin tool with -sites)")
        b = pd.read_csv(f[0])
        b["input"] = os.path.basename(os.path.normpath(d))
        blocks.append(b[b["size"] >= min_size])
    return pd.concat(blocks, ignore_index=True)


def covered(pred, hw, size, q):
    """The prediction is at least the high water for blocks holding q of the requested bytes."""
    return np.sum(size[pred >= hw]) >= q * np.sum(size) - 1e-6


def fit(size, hw, q):
    """(model, a, b, committed bytes over the training blocks) of the cheapest model covering q of them."""
    size, hw = np.asarray(size, float), np.asarray(hw, float)
    cands = []
    if hw.max() == 0:
        return "dead", 0.0, 0.0, 0.0
    # const and ratio: the smallest value covering q of the bytes, from the blocks' own values
    for model, val in (("const", hw), ("ratio", hw / size)):
        for v in np.unique(val):
            pred = np.minimum(v, size) if model == "const" else v * size
            if covered(pred, hw, size, q):
                cands.append((model, v, 0.0, pred))
                break
    # affine: lines through pairs of blocks (and with slope 0 or through the origin are the above)
    pts = sorted(set(zip(size, hw)))
    if len(pts) <= 200:
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                (s1, h1), (s2, h2) = pts[i], pts[j]
                if s2 == s1:
                    continue
                a = (h2 - h1) / (s2 - s1)
                b = h1 - a * s1
                if a < 0:
                    continue
                cands.append(("affine", a, b, np.clip(a * size + b, 0, size)))
    best = None
    for model, a, b, pred in cands:
        if not covered(np.clip(pred, 0, size), hw, size, q):
            continue
        cost = float(np.clip(pred, 0, size).sum())
        if best is None or cost < best[3]:
            best = (model, float(a), float(b), cost)
    if best is None:
        return "ratio", 1.0, 0.0, float(size.sum())
    return best


def reservations(blocks, q, keep):
    rows = []
    for site, g in blocks.groupby("site"):
        model, a, b, cost = fit(g["size"], g["high_water"], q)
        requested = float(g["size"].sum())
        rows.append(dict(site=site, model=model, a=a, b=b, blocks=len(g), inputs=g["input"].nunique(),
                         requested=requested, committed=cost, touched=float(g["touched"].sum()),
                         high_water=float(g["high_water"].sum()), keep=cost < keep * requested))
    return pd.DataFrame(rows).sort_values("requested", ascending=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--min", type=int, default=65536)
    ap.add_argument("--keep", type=float, default=0.95)
    a = ap.parse_args(argv)
    r = reservations(load(a.dirs, a.min), a.quantile, a.keep)
    with open(a.out, "w") as f:
        f.write(f"# MemPrint reservations from {len(a.dirs)} runs, quantile {a.quantile}, blocks >= {a.min} bytes\n")
        for x in r[r.keep].itertuples():
            f.write(f"{x.site} {x.model} {x.a:.6g} {x.b:.6g}\n")
    r.to_csv(os.path.splitext(a.out)[0] + "_summary.csv", index=False)
    kept = r[r.keep]
    print(f"{len(kept)} of {len(r)} sites reserved; on the training blocks they commit "
          f"{kept.committed.sum() / 1e6:.1f} MB of {kept.requested.sum() / 1e6:.1f} MB requested "
          f"(high water {kept.high_water.sum() / 1e6:.1f} MB)")
    return r


if __name__ == "__main__":
    main()
