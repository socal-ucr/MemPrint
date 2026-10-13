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
    runs     k ranges of the block, as fractions of its size: the union of the training blocks'
             touched page runs (column runs), when the blocks are used sparsely across their whole
             length (a KV cache written a few rows per layer) and committing those ranges costs less
             than --runs of the best prefix model
A site is left out when its best model would commit at least --keep of what was requested (no
saving). Writes "site model a b" lines, which libmpr reads (MPR_RESERVATIONS), and a summary CSV.

Calling contexts: a site reached through an allocation wrapper (a function that calls malloc for
several callers, such as ggml_aligned_malloc) holds blocks that are used differently by each
caller. The Pin tool writes the caller frames above each large block's allocation call (column
context, -site_context). When fitting the site's blocks separately per caller (one frame, then two)
commits less by at least --split of the requested bytes, the site gets one line per context,
"site@frame1[@frame2] model a b", besides its merged line, which serves contexts not profiled.
libmpr takes the same frames at run time with backtrace().

Container capacities (programs built with the MemPrintResize pass): the Pin tool links a growing
container's successive buffers into growth chains per call site (*_capacity.csv). For a call site
whose containers grow past --min, the line "cap:0x<call site> cap A B" reserves A bytes, the
--quantile of the final buffers' used bytes (high water), once a container asks for more than B
bytes, the largest any of the call site's small containers grew to (0 when it has none). Hash tables'
bucket arrays are left out unless --hash-capacities, as a precaution: their growth copies only
pointers, and a large bucket array makes every clear() and full traversal of the table slower. (In
our programs no hash table grew past --min, so the choice changed nothing.)
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


PAGE = 4096
MAX_RUNS = 128


def runs_fit(g):
    """(fractional ranges, committed bytes over the training blocks) of the runs model, or None."""
    if "runs" not in g or g["runs"].fillna("").eq("").all():
        return None
    spans = []
    for size, pages, runs in zip(g["size"], g["pages"], g["runs"].fillna("").astype(str)):
        for r in filter(None, runs.split(";")):
            first, n = (int(x) for x in r.split(":"))
            spans.append((first / pages, (first + n) / pages))
    if not spans:
        return None
    spans.sort()
    merged = [list(spans[0])]
    for lo, hi in spans[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    while len(merged) > MAX_RUNS:  # join the two ranges with the smallest gap
        i = min(range(len(merged) - 1), key=lambda k: merged[k + 1][0] - merged[k][1])
        merged[i][1] = merged[i + 1][1]
        del merged[i + 1]
    cost = 0.0
    for size in g["size"]:
        n = -(-size // PAGE)
        cost += min(size, sum((-(-int(hi * n * PAGE) // PAGE) - int(lo * n * PAGE) // PAGE) * PAGE for lo, hi in merged))
    return merged, cost


def row(site, g, q, keep, context="", runs_ratio=0.7):
    model, a, b, cost = fit(g["size"], g["high_water"], q)
    ranges = None
    shaped = runs_fit(g) if model != "dead" else None
    if shaped and shaped[1] < runs_ratio * cost and cost - shaped[1] >= 65536:
        ranges, cost = shaped
        model, a, b = "runs", float(len(ranges)), 0.0
    requested = float(g["size"].sum())
    return dict(site=site, context=context, model=model, a=a, b=b, ranges=ranges, blocks=len(g),
                inputs=g["input"].nunique(), requested=requested, committed=cost, touched=float(g["touched"].sum()),
                high_water=float(g["high_water"].sum()), keep=cost < keep * requested)


def build_id(path):
    """The GNU build-id of an ELF file (hex), or None."""
    import struct
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if data[:4] != b"\x7fELF" or data[4] != 2:
        return None
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    for i in range(shnum):
        sh_type, = struct.unpack_from("<I", data, shoff + i * shentsize + 4)
        if sh_type != 7:  # SHT_NOTE
            continue
        off, size = struct.unpack_from("<QQ", data, shoff + i * shentsize + 0x18)
        pos = off
        while pos + 12 <= off + size:
            namesz, descsz, ntype = struct.unpack_from("<III", data, pos)
            name = data[pos + 12:pos + 12 + namesz]
            desc_at = pos + 12 + ((namesz + 3) & ~3)
            if ntype == 3 and name.rstrip(b"\0") == b"GNU":
                return data[desc_at:desc_at + descsz].hex()
            pos = desc_at + ((descsz + 3) & ~3)
    return None


def images(dirs):
    """name -> build-id of the images the profiles' address keys name (from *_images.csv)."""
    out = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*_images.csv")):
            for r in pd.read_csv(f).itertuples():
                if r.name not in out:
                    out[r.name] = build_id(r.path)
    return out


def capacities(dirs, q, min_size, hashes=False):
    """'key cap A B' rows from the Pin tool's *_capacity.csv files (one per profiling run)."""
    rows = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*_capacity.csv")):
            for r in pd.read_csv(f, dtype=str).fillna("").itertuples():
                kind = int(getattr(r, "kind", 0) or 0)
                if kind == 3 and not hashes:
                    continue
                a = rows.setdefault(r.key, dict(elem=int(r.elem), chains=0, small=0, small_max=0, finals=[]))
                a["chains"] += int(r.chains)
                a["small"] += int(r.small_chains)
                a["small_max"] = max(a["small_max"], int(r.small_max_size))
                a["finals"] += [int(x) for x in str(r.final_high_waters).split(";") if x]
    out = []
    for key, a in rows.items():
        if not a["finals"]:
            continue
        final = float(np.quantile(a["finals"], q, method="higher"))
        trigger = float(a["small_max"])
        if final < min_size or final <= trigger:
            continue
        out.append(dict(site=key, model="cap", a=final, b=trigger, chains=a["chains"], small_chains=a["small"],
                        large_chains=len(a["finals"]), elem=a["elem"]))
    return out


def contexts(g, depth):
    """Each block's first `depth` caller frames, joined by '@' ('' when the profile has none)."""
    return g["context"].fillna("").astype(str).map(lambda c: "@".join(c.split("|")[:depth]) if c else "")


def split(site, g, q, keep, frac, max_depth=2):
    """Per-context rows of a site, at the smallest depth whose saving over the merged fit is at least
    `frac` of the requested bytes (within 1% of the best depth's); [] when no depth saves that much."""
    if "context" not in g or g["context"].fillna("").eq("").all():
        return []
    merged = row(site, g, q, keep)["committed"]
    best = []
    for depth in range(1, max_depth + 1):
        c = contexts(g, depth)
        if c.nunique() < 2 or c.eq("").any():
            continue
        rows = [row(f"{site}@{ctx}", h, q, keep, ctx) for ctx, h in g.groupby(c)]
        cost = sum(r["committed"] for r in rows)
        if merged - cost >= frac * g["size"].sum():
            if not best or cost < best[0] - 0.01 * g["size"].sum():
                best = (cost, rows)
    if not best:
        return []
    for r in best[1]:
        r["keep"] = True   # a context line overrides the merged one, even when it saves nothing
    return best[1]


def reservations(blocks, q, keep, frac=0.05):
    rows = []
    for site, g in blocks.groupby("site"):
        rows.append(row(site, g, q, keep))
        rows.extend(split(site, g, q, keep, frac))
    cols = ["site", "context", "model", "a", "b", "ranges", "blocks", "inputs", "requested", "committed", "touched",
            "high_water", "keep"]
    return pd.DataFrame(rows, columns=cols).sort_values("requested", ascending=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--min", type=int, default=65536)
    ap.add_argument("--keep", type=float, default=0.95)
    ap.add_argument("--hash-capacities", action="store_true", help="also reserve hash tables' bucket counts")
    ap.add_argument("--split", type=float, default=0.05,
                    help="split a site by calling context when that commits less by this share of its requested bytes")
    a = ap.parse_args(argv)
    r = reservations(load(a.dirs, a.min), a.quantile, a.keep, a.split)
    caps = capacities(a.dirs, a.quantile, a.min, a.hash_capacities)
    with open(a.out, "w") as f:
        f.write(f"# MemPrint reservations from {len(a.dirs)} runs, quantile {a.quantile}, blocks >= {a.min} bytes\n")
        for x in r[r.keep.astype(bool)].itertuples():
            if x.model == "runs":
                f.write(f"{x.site} runs {len(x.ranges)} " + " ".join(f"{lo:.6f} {hi:.6f}" for lo, hi in x.ranges) + "\n")
            else:
                f.write(f"{x.site} {x.model} {x.a:.6g} {x.b:.6g}\n")
        for c in caps:
            f.write(f"{c['site']} cap {c['a']:.0f} {c['b']:.0f}\n")
        # the builds the address keys (image+0x...) were profiled on: libmpr ignores an image's
        # address keys when the image it loads has another build-id
        named = {k.split("+")[0] for k in r[r.keep.astype(bool)].site for k in k.split("@") if "+" in k}
        for name, bid in sorted(images(a.dirs).items()):
            if name in named and bid:
                f.write(f"# image {name} build-id {bid}\n")
    if caps:
        print(f"{len(caps)} container capacities: " + ", ".join(
            f"{c['site']} {c['a'] / 1e6:.2f} MB after {c['b'] / 1e3:.0f} KB ({c['large_chains']} of {c['chains']} chains)"
            for c in caps))
    r.to_csv(os.path.splitext(a.out)[0] + "_summary.csv", index=False)
    kept = r[r.keep.astype(bool)]
    # totals over the blocks once: a split site's context rows replace its merged row
    split_sites = {s.split("@")[0] for s in kept.site if "@" in s}
    kept = kept[kept.site.map(lambda s: ("@" in s) or (s not in split_sites)).astype(bool)]
    print(f"{len(kept)} of {len(r)} sites reserved ({len(split_sites)} split by calling context); on the training blocks they commit "
          f"{kept.committed.sum() / 1e6:.1f} MB of {kept.requested.sum() / 1e6:.1f} MB requested "
          f"(high water {kept.high_water.sum() / 1e6:.1f} MB)")
    return r


if __name__ == "__main__":
    main()
