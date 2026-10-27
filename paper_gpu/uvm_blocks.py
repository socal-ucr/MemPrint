"""What footprints are worth to Unified Memory under oversubscription, with kernels run wave by wave.

    [PAPER_GPU_SIZE=large] python paper_gpu/uvm_blocks.py [--only NAME,...] [--page KB]

As uvm_oversub.py (managed allocations, copies as host writes and reads, device memory a share of the pages the
device operations touch, the same costs and policies), but a kernel runs as a sequence of waves: its blocks are
issued in block order, as many at once as the A30 holds (simulate_blocks.block_shape: 56 SMs, the blocks per SM
that its threads, registers and shared memory allow), and each wave takes an equal share of the kernel's profiled
time. A wave touches the pages its blocks reference in the traced run's block trace (truth_blocks); a policy
decides before each wave:

    demand      faults on every page of the wave not on the device; LRU eviction
    buffer      at a kernel's first wave, prefetch every page of every allocation the kernel touches (as far as
                it fits beside the wave's own pages); LRU eviction
    hybrid      before each wave, prefetch the pages the hybrid interpreter's blocks of that wave reference in
                any of its four schedules (schedule_*/NAME_blocks.npz; a kernel with a reference to an unknown
                object may touch every allocation); evict the page whose next predicted use is farthest away
    truth       the same with the traced blocks: an oracle

A share is simulated only if every wave's pages fit (thrashing inside a wave depends on the order of its
blocks' accesses, which is not modelled). Writes data/uvm_blocks{,_large}{,_4k}.csv.
"""

import argparse
import csv
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmarks as B  # noqa: E402
import simulate_blocks as SB  # noqa: E402
from copy_savings import H2D, D2H, UNIT, fit  # noqa: E402
from memprint.static.kdeps import R, W, RM, WM, align_ends  # noqa: E402
from memprint.static.kdeps_score import _same as same_op  # noqa: E402

DATA = HERE / "data"
POLICIES = ("demand", "buffer", "hybrid", "truth")


def wave_pages(per_alloc, slots, nb, page_units, sizes, rbits, wbits):
    """[(pages read, pages written)] per wave of a kernel from {alloc: (blocks, units, bits)}."""
    waves = max(1, math.ceil(nb / slots))
    out = [(set(), set()) for _ in range(waves)]
    for a, (blocks, units, bits) in per_alloc.items():
        if a not in sizes:
            continue
        keep = (units >= 0) & (units < sizes[a])
        wv = np.minimum(blocks[keep].astype(np.int64) // slots, waves - 1)
        pg = units[keep].astype(np.int64) // page_units
        bt = bits[keep]
        for mask, k in (((bt & rbits) != 0, 0), ((bt & wbits) != 0, 1)):
            if not mask.any():
                continue
            pairs = np.unique(wv[mask] * (1 << 32) + pg[mask])
            for x in pairs:
                out[int(x >> 32)][k].add((a, int(x & 0xFFFFFFFF)))
    return out


def interp_waves(name, hdir, tev, slots_nb, page_units, sizes):
    """{traced event: [pages per wave]} from the interpreter's four schedules, matched to the block trace."""
    runs = []
    for s in B.SCHEDULES.split(","):
        r = SB.interp_blocks(hdir, name, s)
        if r is None:
            return {}
        runs.append(r)
    base_ev = runs[0][0]
    sig = lambda ev: [(e["kind"], e["name"], e["site"]) for e in ev]  # noqa: E731
    dev = [e for e in base_ev if e["kind"] != "host"]
    to_base = {t: int(dev[k]["index"]) for t, k in align_ends(tev, dev, lambda y, x: same_op(x, y)).items()}
    maps = [None] + [align_ends(sig(ev), sig(base_ev)) for ev, _, _ in runs[1:]]
    everything = {a: (np.zeros(1, np.int32), np.arange(n, dtype=np.int32), np.full(n, R | W, np.uint8))
                  for a, n in sizes.items()}
    out = {}
    for t, b in to_base.items():
        if t not in slots_nb:
            continue
        slots, nb = slots_nb[t]
        per_run = []
        for k, (ev, acc, _) in enumerate(runs):
            e = b if k == 0 else next((o for o, m in maps[k].items() if m == b), None)
            if e is None or e not in acc or acc[e] == "same":
                continue
            per_run.append(everything if acc[e] is None else acc[e])
        waves = None
        for per_alloc in per_run:
            if per_alloc is everything:
                n = max(1, math.ceil(nb / slots))
                pages = {(a, p) for a, u in sizes.items() for p in range(-(-u // page_units))}
                w = [(set(pages), set(pages)) for _ in range(n)]
            else:
                w = wave_pages(per_alloc, slots, nb, page_units, sizes, R | RM, W | WM)
            if waves is None:
                waves = w
            else:
                for (r0, w0), (r1, w1) in zip(waves, w):
                    r0 |= r1
                    w0 |= w1
        if waves is not None:
            out[t] = waves
    return out


def simulate(steps, policy, capacity, fault_ns, batch, page_bytes, model):
    """steps: [(kind, actual (rd, wr), predicted pages, kernel buffers' pages or None, duration ns)], kind 'h2d',
    'd2h' or 'dev'. Returns (time, stall, faults, bytes to the device, bytes to the host)."""
    h2d_ns, d2h_ns = model[H2D][1] * page_bytes, model[D2H][1] * page_bytes
    lru, dirty = OrderedDict(), set()
    time = stall = faults = up = down = 0
    uses = defaultdict(list)
    for i, (_, _, pred, _, _) in enumerate(steps):
        for p in pred or ():
            uses[p].append(i)
    nxt = defaultdict(int)

    def next_use(p, now):
        u, k = uses.get(p, ()), nxt[p]
        while k < len(u) and u[k] <= now:
            k += 1
        nxt[p] = k
        return u[k] if k < len(u) else math.inf

    def evict(now, keep):
        nonlocal time, stall, down
        if policy in ("hybrid", "truth"):
            victim = max((p for p in lru if p not in keep), key=lambda p: (next_use(p, now), p not in dirty),
                         default=None)
        else:
            victim = next((p for p in lru if p not in keep), None)
        if victim is None:
            return False
        del lru[victim]
        if victim in dirty:
            dirty.discard(victim)
            down += page_bytes
            time += d2h_ns
            stall += d2h_ns
        return True

    def bring(p, now, keep):
        nonlocal up
        while len(lru) >= capacity:
            if not evict(now, keep):
                return False
        lru[p] = True
        up += page_bytes
        return True

    for i, (kind, (rd, wr), pred, bufs, dur) in enumerate(steps):
        touched = rd | wr
        if kind != "dev":                                               # host writes or reads managed pages
            back = [p for p in touched if p in lru]
            for p in back:
                del lru[p]
                if p in dirty or kind == "h2d":
                    down += page_bytes
                    time += d2h_ns
                    stall += d2h_ns
                dirty.discard(p)
            if back:
                time += fault_ns
                stall += fault_ns
            continue
        want = set()
        if policy == "buffer" and bufs is not None:
            want = bufs
        elif policy in ("hybrid", "truth"):
            want = pred or set()
        moved = 0
        for p in sorted(p for p in want if p not in lru):               # as far as it fits beside the
            if not bring(p, i, touched | want):                         # wave's own pages
                break
            moved += 1
        if moved:
            t = moved * h2d_ns + model[H2D][0]
            time += t
            stall += t
        missing = [p for p in touched if p not in lru]
        if missing:
            n = len(missing)
            faults += n
            t = math.ceil(n / batch) * fault_ns + n * h2d_ns
            time += t
            stall += t
            for p in missing:
                bring(p, i, touched)
        for p in touched:
            if p in lru:
                lru.move_to_end(p)
        dirty |= {p for p in wr if p in lru}
        time += dur
    return time, stall, faults, up, down


def program(name, fractions, fault_us, batch, page_kb, model):
    w = B.workdir(B.BENCH[name])
    page_bytes = page_kb * 1024
    pu = page_bytes // UNIT
    sizes = {int(r["index"]): -(-int(r["bytes"]) // UNIT)
             for r in csv.DictReader(open(w / "truth_blocks" / "truth_allocs.csv"))}
    tev, tacc = SB.truth_blocks(w)
    ops = SB.load_ops(name)
    if ops is None or len(ops) != len(tev):
        return [{"name": name, "status": "no profile matching the block trace"}]
    slots_nb = {}
    for i, e in enumerate(tev):
        if e["kind"] == "kernel":
            sh = SB.block_shape(ops[i])
            slots_nb[i] = (SB._per_sm(sh) * SB.GPU["sms"] if sh and sh != (0, 0, 0) else 1,
                           max(1, SB._grid_blocks(e["grid"])))
    hy = interp_waves(name, w / "hybrid", tev, slots_nb, pu, sizes)
    steps_actual, preds = [], {"hybrid": [], "truth": []}
    for i, e in enumerate(tev):
        dur = int(ops[i]["dur_ns"])
        if e["kind"] == "kernel":
            slots, nb = slots_nb[i]
            act = wave_pages(tacc.get(i, {}), slots, nb, pu, sizes, 1, 2)
            bufs = {(a, p) for a in tacc.get(i, {}) if a in sizes for p in range(-(-sizes[a] // pu))}
            pw = hy.get(i)
            for k, (rd, wr) in enumerate(act):
                steps_actual.append(("dev", (rd, wr), None, bufs if k == 0 else None, dur // len(act)))
                preds["truth"].append(rd | wr)
                preds["hybrid"].append((pw[k][0] | pw[k][1]) if pw and k < len(pw) else set())
            continue
        rd, wr = set(), set()
        for a, (_, units, bits) in tacc.get(i, {}).items():
            if a not in sizes:
                continue
            u = units[units < sizes[a]]
            b = bits[units < sizes[a]]
            rd |= {(a, int(p)) for p in np.unique(u[(b & 1) != 0] // pu)}
            wr |= {(a, int(p)) for p in np.unique(u[(b & 2) != 0] // pu)}
        kind = {H2D: "h2d", D2H: "d2h"}.get(e["name"], "dev")
        steps_actual.append((kind, (rd, wr), None, None, dur if kind == "dev" else 0))
        preds["truth"].append(rd | wr if kind == "dev" else set())
        preds["hybrid"].append(rd | wr if kind == "dev" else set())     # a memset: the host knows its range
    dev = [s for s in steps_actual if s[0] == "dev"]
    total = set().union(*[s[1][0] | s[1][1] for s in dev]) if dev else set()
    largest = max((len(s[1][0] | s[1][1]) for s in dev), default=0)
    explicit = sum(int(o["dur_ns"]) for o in ops)
    resident = sum(s[4] for s in dev)
    waves = len(dev)
    kernels = sum(1 for e in tev if e["kind"] == "kernel")
    rows = []
    for f in fractions:
        cap = max(1, math.ceil(f * len(total)))
        base = {"name": name, "suite": B.BENCH[name]["suite"], "page_kb": page_kb, "pages": len(total),
                "largest_wave": largest, "waves": waves, "kernels": kernels, "fraction": f, "capacity": cap}
        if cap < largest:
            rows.append({**base, "status": f"a wave needs {largest} pages"})
            continue
        for fu in fault_us:
            for policy in POLICIES:
                steps = [(k, a, preds[policy][j] if policy in preds else None, b, d)
                         for j, (k, a, _, b, d) in enumerate(steps_actual)]
                t, s, n, up, down = simulate(steps, policy, cap, fu * 1000, batch, page_bytes, model)
                rows.append({**base, "status": "ok", "fault_us": fu, "policy": policy, "resident_ns": resident,
                             "explicit_ns": explicit, "time_ns": int(t), "stall_ns": int(s), "faults": n,
                             "to_device_bytes": up, "to_host_bytes": down,
                             "unmatched_kernels": sum(1 for i in slots_nb if i not in hy)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--page", type=int, default=64, help="page size in KB")
    ap.add_argument("--fractions", default="0.25,0.375,0.5,0.625,0.75,0.875,1.0")
    ap.add_argument("--fault-us", default="10,25,50", help="cost of a batch of faults in microseconds")
    ap.add_argument("--batch", type=int, default=256, help="faults handled in one batch")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else [
        n for n in B.BENCH if (B.workdir(B.BENCH[n]) / "truth_blocks" / "truth_block_access.csv").exists()
        and (B.workdir(B.BENCH[n]) / "hybrid" / "schedule_lockstep" / f"{n}_blocks.npz").exists()]
    model = fit()
    fractions = [float(x) for x in args.fractions.split(",")]
    fault_us = [float(x) for x in args.fault_us.split(",")]
    rows = []
    for n in names:
        r = program(n, fractions, fault_us, args.batch, args.page, model)
        rows += r
        ok = [x for x in r if x.get("status") == "ok" and x["fault_us"] == 25]
        lo = min((x["fraction"] for x in ok), default=None)
        print(n, f"waves {r[0].get('waves')} pages {r[0].get('pages')} largest wave {r[0].get('largest_wave')}",
              {x["policy"]: round(x["time_ns"] / x["explicit_ns"], 2) for x in ok if x["fraction"] == lo},
              f"at {lo}" if lo is not None else r[0].get("status"), flush=True)
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    path = DATA / f"uvm_blocks{B.SUFFIX}{'' if args.page == 64 else f'_{args.page}k'}.csv"
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, cols)
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
