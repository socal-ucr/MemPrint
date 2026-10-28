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
    hybrid_min, truth_min
                the same, prefetching only runs of at least uvm_costs.min_run contiguous pages
    range, range_min
                as truth and truth_min, but each allocation's pages from the first the wave touches to the last
                (uvm_cheap.span: what a bounds analysis gives without the interpreter)
    history, history_min
                prefetch what uvm_cheap.History predicts from the same wave of the kernel's earlier launches;
                LRU eviction

Costs are uvm_costs.Measured (costs "measured") or, one row per --fault-us, uvm_costs.Assumed (costs "assumed"), as
in uvm_oversub.py; a wave's faults stall it as the faults of one kernel would.

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
from uvm_cheap import History, span  # noqa: E402
from uvm_costs import Assumed, Measured, runs  # noqa: E402
from memprint.static.kdeps import R, W, RM, WM, align_ends  # noqa: E402
from memprint.static.kdeps_score import _same as same_op  # noqa: E402

DATA = HERE / "data"
POLICIES = ("demand", "buffer", "hybrid", "truth", "hybrid_min", "truth_min", "range", "range_min", "history",
            "history_min")


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


def simulate(steps, policy, capacity, costs):
    """steps: [(kind, actual (rd, wr), predicted pages, kernel buffers' pages or None, duration ns)], kind 'h2d',
    'd2h' or 'dev'; costs: uvm_costs.Measured or Assumed. Returns (time, stall, faults, bytes to the device, bytes
    to the host)."""
    page_bytes, d2h_ns = costs.page_bytes, costs.d2h_ns
    footprints = policy.split("_")[0] in ("hybrid", "truth", "range")    # evict by the predicted next use
    predicts = footprints or policy.startswith("history")
    min_run = costs.min_run if policy.endswith("_min") else 1
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
        if footprints:
            victim = max((p for p in lru if p not in keep), key=lambda p: (next_use(p, now), p not in dirty),
                         default=None)
        else:
            victim = next((p for p in lru if p not in keep), None)
        if victim is None:
            return False
        del lru[victim]
        if victim in dirty or costs.writeback_clean:
            down += page_bytes
            time += d2h_ns
            stall += d2h_ns
        dirty.discard(victim)
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
            moved = [p for p in back if p in dirty or kind == "h2d" or costs.writeback_clean]
            for p in back:
                del lru[p]
                dirty.discard(p)
            down += len(moved) * page_bytes
            if back:
                t = costs.host_fault(len(moved))
                time += t
                stall += t
            continue
        want = set()
        if policy == "buffer" and bufs is not None:
            want = bufs
        elif predicts:
            want = pred or set()
        moved = []
        for p in [p for r in runs(p for p in want if p not in lru) if len(r) >= min_run for p in r]:
            if not bring(p, i, touched | want):                         # as far as it fits beside the
                break                                                   # wave's own pages
            moved.append(p)
        if moved:
            t = costs.prefetch([len(r) for r in runs(moved)])
            time += t
            stall += t
        missing = [p for p in touched if p not in lru]
        if missing:
            n = len(missing)
            faults += n
            t = costs.fault(n)
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


def program(name, fractions, cost_models, page_kb, policies=POLICIES):
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
    steps_actual, preds = [], {"hybrid": [], "truth": [], "range": [], "history": []}
    hist = History({a: -(-n // pu) for a, n in sizes.items()})
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
                preds["range"].append(span(rd | wr))
                preds["history"].append(hist.predict((e["name"], k)))
                hist.record((e["name"], k), rd | wr)
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
        preds["range"].append(rd | wr if kind == "dev" else set())
        preds["history"].append(rd | wr if kind == "dev" else set())
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
        for costs in cost_models:
            for policy in policies:
                src = policy.split("_")[0]
                steps = [(k, a, preds[src][j] if src in preds else None, b, d)
                         for j, (k, a, _, b, d) in enumerate(steps_actual)]
                t, s, n, up, down = simulate(steps, policy, cap, costs)
                rows.append({**base, "status": "ok", "costs": costs.name, "fault_us": costs.fault_us, "policy": policy, "resident_ns": resident,
                             "explicit_ns": explicit, "time_ns": int(t), "stall_ns": int(s), "faults": n,
                             "to_device_bytes": up, "to_host_bytes": down,
                             "unmatched_kernels": sum(1 for i in slots_nb if i not in hy)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--page", type=int, default=64, help="page size in KB")
    ap.add_argument("--fractions", default="0.25,0.375,0.5,0.625,0.75,0.875,1.0")
    ap.add_argument("--fault-us", default="10,25,50", help="assumed cost of a batch of faults in microseconds")
    ap.add_argument("--batch", type=int, default=256, help="faults handled in one batch")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--tag", default="", help="appended to the file name")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else [
        n for n in B.BENCH if (B.workdir(B.BENCH[n]) / "truth_blocks" / "truth_block_access.csv").exists()
        and (B.workdir(B.BENCH[n]) / "hybrid" / "schedule_lockstep" / f"{n}_blocks.npz").exists()]
    model = fit()
    fractions = [float(x) for x in args.fractions.split(",")]
    cost_models = [Measured(args.page, model)] + [Assumed(args.page, model, float(x), args.batch)
                                                  for x in args.fault_us.split(",") if x]
    rows = []
    for n in names:
        r = program(n, fractions, cost_models, args.page, args.policies.split(","))
        rows += r
        ok = [x for x in r if x.get("status") == "ok" and x["costs"] == "measured"]
        lo = min((x["fraction"] for x in ok), default=None)
        print(n, f"waves {r[0].get('waves')} pages {r[0].get('pages')} largest wave {r[0].get('largest_wave')}",
              {x["policy"]: round(x["time_ns"] / x["explicit_ns"], 2) for x in ok if x["fraction"] == lo},
              f"at {lo}" if lo is not None else r[0].get("status"), flush=True)
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    path = DATA / f"uvm_blocks{B.SUFFIX}{'' if args.page == 64 else f'_{args.page}k'}{args.tag}.csv"
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, cols)
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
