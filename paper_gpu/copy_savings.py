"""What knowing each operation's exact footprint saves in copies between host and device.

    [PAPER_GPU_SIZE=large] python paper_gpu/copy_savings.py [--only NAME,...]

Per benchmark of benchmarks.py with a traced run (data/bench*/NAME/truth) and a hybrid run, from the units each
operation read (bit 1) and wrote (bit 2) in truth_access.csv and the hybrid interpreter's NAME_access.csv:

    host to device   a copied unit is unneeded if no later operation reads it before something overwrites it
                     (the copy could leave it out)
    device to host   a copied unit is unneeded if no device operation wrote it since it was last copied either
                     way (the host already has its value, if the copy goes to the buffer it came from or went
                     to; the data does not say, so this is an upper bound)

The traced run's footprints give what an oracle saves; the interpreter's (its lockstep run) what it saves, and
the units it would leave out that the traced run needs (unsafe). A copy's time is modelled per direction from the
profiled copies of both input sizes (sim_ops.csv) as a fixed cost (the median of copies of 4 KB or less) plus bytes over a bandwidth
(the median over copies of 1 MB or more); leaving out part of a copy costs one fixed cost per contiguous run of
needed units, and saves that share of the copy's measured time, never less than nothing.
Writes data/copy_savings{,_large}.csv.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmarks as B  # noqa: E402
from simulate import ks  # noqa: E402

DATA = HERE / "data"
UNIT = 4
H2D, D2H = "cudaMemcpy H2D", "cudaMemcpy D2H"


def accesses(path):
    """{event: [(alloc, lo, hi, bits)]}."""
    out = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            out[int(r["event"])].append((int(r["alloc"]), int(r["unit_lo"]), int(r["unit_hi"]), int(r["bits"])))
    return out


def needed(events, acc, sizes):
    """Per copy event: [direction, allocation per unit, unit, needed mask]."""
    owner = {a: np.full(n, -1, np.int64) for a, n in sizes.items()}        # the H2D copy whose value a unit holds
    pos = {a: np.zeros(n, np.int64) for a, n in sizes.items()}             # ... and its place among that copy's
    dirty = {a: np.zeros(n, bool) for a, n in sizes.items()}               # written on the device since copied
    out = {}
    for e in events:
        i = int(e["index"])
        rows = [(a, lo, min(hi, sizes[a]), bits) for a, lo, hi, bits in acc.get(i, ())
                if a in sizes and lo < min(hi, sizes[a])]                  # reads past the end left out
        copy = {H2D: "h2d", D2H: "d2h"}.get(e["name"])
        if copy:
            allocs = np.concatenate([np.full(hi - lo, a) for a, lo, hi, _ in rows]) if rows else np.zeros(0, int)
            units = np.concatenate([np.arange(lo, hi) for _, lo, hi, _ in rows]) if rows else np.zeros(0, int)
            mask = np.zeros(len(units), bool)
            if copy == "d2h":
                mask = np.concatenate([dirty[a][lo:hi] for a, lo, hi, _ in rows]) if rows else mask
            out[i] = [copy, allocs, units, mask]
        for a, lo, hi, bits in rows:                                       # reads first: a unit read and
            if bits & 1:                                                   # written reads the old value
                o, p = owner[a][lo:hi], pos[a][lo:hi]
                for c in np.unique(o[o >= 0]):
                    out[c][3][p[o == c]] = True
        if copy == "h2d":
            k = 0
            for a, lo, hi, _ in rows:
                owner[a][lo:hi], pos[a][lo:hi] = i, np.arange(k, k + hi - lo)
                dirty[a][lo:hi] = False
                k += hi - lo
        elif copy == "d2h":
            for a, lo, hi, _ in rows:
                dirty[a][lo:hi] = False
        else:
            for a, lo, hi, bits in rows:
                if bits & 2:
                    owner[a][lo:hi] = -1
                    dirty[a][lo:hi] = True
    return out


def runs(mask, allocs, units):
    """Contiguous runs of needed units (a run breaks at a gap or another allocation)."""
    if not mask.any():
        return 0
    a, u = allocs[mask], units[mask]
    return 1 + int(np.count_nonzero((np.diff(u) != 1) | (np.diff(a) != 0)))


def fit():
    """Per direction: (fixed ns, ns per byte) over the profiled copies of both input sizes (one machine)."""
    pts = defaultdict(list)
    for p in sorted(DATA.glob("bench*/*/sim_ops.csv")):
        for r in csv.DictReader(open(p)):
            if r["kind"] == "copy" and int(r["bytes"]) > 0:
                pts[r["name"]].append((int(r["bytes"]), int(r["dur_ns"])))
    out = {}
    for k, v in pts.items():                                           # a least-squares line is ruled by the
        x, y = np.array(v, float).T                                    # largest copies (its intercept can go
        fixed = float(np.median(y[x <= 4096])) if (x <= 4096).any() else 0.0   # negative): the fixed cost from
        big = x >= (1 << 20) if (x >= (1 << 20)).any() else x >= np.percentile(x, 90)   # small copies, the
        per_byte = float(np.median((y[big] - fixed) / x[big]))         # bandwidth from large ones
        out[k] = (fixed, max(per_byte, 0.0))
    return out


def program(name, model):
    w = B.workdir(B.BENCH[name])
    tdir, hdir = w / "truth", w / "hybrid"
    sizes = {int(r["index"]): -(-int(r["bytes"]) // UNIT) for r in csv.DictReader(open(tdir / "truth_allocs.csv"))}
    tev = list(csv.DictReader(open(tdir / "truth_events.csv")))
    ops = {int(r["index"]): r for r in csv.DictReader(open(w / "sim_ops.csv"))}
    if len(ops) != len(tev):
        return {"name": name, "status": f"profile has {len(ops)} operations, trace {len(tev)}"}
    truth = needed(tev, accesses(tdir / "truth_access.csv"), sizes)
    iev = [e for e in csv.DictReader(open(hdir / f"{name}_events.csv")) if e["kind"] != "host"]
    interp = needed(iev, accesses(hdir / f"{name}_access.csv"), sizes)
    to_interp = ks.truth_map(iev, tev)                                   # trace event -> interpreter event
    row = {"name": name, "suite": B.BENCH[name]["suite"], "status": "ok",
           "ops_s": round(sum(int(o["dur_ns"]) for o in ops.values()) / 1e9, 9)}
    for d, op in (("h2d", H2D), ("d2h", D2H)):
        fixed, per_byte = model.get(op, (0.0, 0.0))
        r = defaultdict(float, {k: 0 for k in ("n", "bytes", "s", "unneeded_bytes", "saved_s", "unsafe_bytes_hy",
                                               "unneeded_bytes_hy", "saved_s_hy", "unmatched")})

        def saved(dur, mask, allocs, units):
            """The measured copy time times the fitted share that leaving out the unneeded units saves."""
            whole = fixed + len(units) * UNIT * per_byte
            part = runs(mask, allocs, units) * fixed + int(mask.sum()) * UNIT * per_byte
            return dur * max(0.0, 1 - part / whole) / 1e9 if whole > 0 else 0.0

        for i, (kind, allocs, units, mask) in truth.items():
            if kind != d:
                continue
            dur = int(ops[i]["dur_ns"])
            r["n"] += 1
            r["bytes"] += len(units) * UNIT
            r["s"] += dur / 1e9
            r["unneeded_bytes"] += int(np.count_nonzero(~mask)) * UNIT
            r["saved_s"] += saved(dur, mask, allocs, units)
            j = to_interp.get(i)
            if j is None or j not in interp:
                r["unmatched"] += 1
                continue
            _, ia, iu, im = interp[j]
            ikey = set(zip(ia[im].tolist(), iu[im].tolist()))                # units the interpreter keeps
            need = set(zip(allocs[mask].tolist(), units[mask].tolist()))
            unsafe = len(need - ikey)
            r["unsafe_bytes_hy"] += unsafe * UNIT
            r["unneeded_bytes_hy"] += int(np.count_nonzero(~im)) * UNIT
            r["saved_s_hy"] += saved(dur, im, ia, iu)
        row.update({f"{d}_{k}": round(v, 9) if k in ("s", "saved_s", "saved_s_hy") else int(v) for k, v in r.items()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    names = args.only.split(",") if args.only else [
        n for n in B.BENCH if (B.workdir(B.BENCH[n]) / "truth" / "truth_access.csv").exists()
        and (B.workdir(B.BENCH[n]) / "hybrid" / f"{n}_access.csv").exists()
        and (B.workdir(B.BENCH[n]) / "sim_ops.csv").exists()]
    model = fit()
    for k, (fixed, per_byte) in model.items():
        print(f"{k}: {fixed / 1000:.2f} us + {1 / per_byte:.1f} GB/s" if per_byte > 0 else f"{k}: {fixed} ns")
    rows = []
    for n in names:
        rows.append(program(n, model))
        print(n, {k: v for k, v in rows[-1].items() if k in ("status", "h2d_bytes", "h2d_unneeded_bytes",
                                                             "h2d_unsafe_bytes_hy", "d2h_unneeded_bytes")},
              flush=True)
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    path = DATA / f"copy_savings{B.SUFFIX}.csv"
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, cols)
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
