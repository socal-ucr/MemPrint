"""Each event's footprint over the hybrid interpreter's four schedules, for runs made before hybrid.py wrote it.

    [PAPER_GPU_SIZE=large] python paper_gpu/access_all.py [--only NAME,...]

hybrid.py now writes NAME_access_all.csv (kdeps.union_footprint) next to NAME_access.csv, which has the lockstep
run's footprint only. For the runs already made, each schedule's kernels are rebuilt from its block references
(schedule_*/NAME_blocks.npz, --blocks): a unit is read or written if some block's reference has R or RM, W or
WM, and written in every legal schedule if one has W and WX; an event with a reference to an unknown object
may read and write every allocation; an event that ran as in lockstep ("same") adds nothing. Copies and
memsets are issued by the host, alike in every schedule, and come from the lockstep run's NAME_access.csv (a
copy or memset writes its units for certain). Copies that only another schedule's extra rounds issue are not
in its block references and are left out.
Writes data/bench*/NAME/hybrid/NAME_access_all.csv.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmarks as B  # noqa: E402
import simulate_blocks as SB  # noqa: E402
from memprint.static.kdeps import R, W, RM, WM, WX, union_footprint  # noqa: E402

SCHEDULES = B.SCHEDULES.split(",")


def schedule_run(hdir, name, schedule, sizes):
    """(event signatures, {event: {alloc: (read, written, killed)}}) of one schedule's kernels."""
    sub = hdir / f"schedule_{schedule}"
    events = list(csv.DictReader(open(sub / f"{name}_events.csv")))
    alloc = {r["name"]: int(r["alloc"]) for r in csv.DictReader(open(sub / f"{name}_objects.csv")) if r["alloc"]}
    z = SB._load_npz(sub / f"{name}_blocks.npz")
    fp = {}
    for e in z["unknown"]:                                             # may reach every allocation
        fp[int(e)] = {a: (np.ones(n, bool), np.ones(n, bool), np.zeros(n, bool)) for a, n in sizes.items()}
    same = {int(e) for e in z.get("same", ())}
    for k in z:
        if not k.endswith("/unit"):
            continue
        e, obj, _ = k.split("/")
        e = int(e)
        if obj not in alloc or e in same or e in {int(x) for x in z["unknown"]}:
            continue
        a, n = alloc[obj], sizes.get(alloc[obj])
        if n is None:
            continue
        u, bits = z[k].astype(np.int64), z[f"{e}/{obj}/bits"]
        keep = u < n                                                   # references past the end left out
        u, bits = u[keep], bits[keep]
        rd, wr, kill = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        rd[u[(bits & (R | RM)) != 0]] = True
        wr[u[(bits & (W | WM)) != 0]] = True
        kill[u[(bits & (W | WX)) == (W | WX)]] = True
        fp.setdefault(e, {})[a] = (rd, wr, kill)
    return [(r["kind"], r["name"], r["site"]) for r in events], fp


def host_ops(hdir, name, sizes):
    """{event: {alloc: (read, written, killed)}} of the lockstep run's copies and memsets."""
    kinds = {int(r["index"]): r["kind"] for r in csv.DictReader(open(hdir / f"{name}_events.csv"))}
    fp = {}
    for r in csv.DictReader(open(hdir / f"{name}_access.csv")):
        e, a = int(r["event"]), int(r["alloc"])
        if kinds.get(e) == "kernel" or a not in sizes:
            continue
        n = sizes[a]
        rd, wr, kill = fp.setdefault(e, {}).setdefault(a, (np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)))
        lo, hi, bits = int(r["unit_lo"]), min(int(r["unit_hi"]), n), int(r["bits"])
        if bits & 1:
            rd[lo:hi] = True
        if bits & 2:
            wr[lo:hi] = True
            kill[lo:hi] = True
    return fp


def program(name):
    hdir = B.workdir(B.BENCH[name]) / "hybrid"
    sizes = {int(r["alloc"]): max(int(r["bytes"]) + 3, 4) // 4
             for r in csv.DictReader(open(hdir / f"{name}_objects.csv")) if r["alloc"]}
    runs = [schedule_run(hdir, name, s, sizes) for s in SCHEDULES]
    runs[0][1].update(host_ops(hdir, name, sizes))                    # lockstep is the first run
    rows = union_footprint(runs, sizes)
    with open(hdir / f"{name}_access_all.csv", "w", newline="") as f:
        f.write("event,alloc,unit_lo,unit_hi,bits\n")
        f.writelines(",".join(map(str, r)) + "\n" for r in rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    assert SCHEDULES[0] == "lockstep"
    names = args.only.split(",") if args.only else [
        n for n in B.BENCH if (B.workdir(B.BENCH[n]) / "hybrid" / "schedule_lockstep" / f"{n}_blocks.npz").exists()]
    for n in names:
        print(n, program(n), "rows", flush=True)


if __name__ == "__main__":
    main()
