"""What footprints are worth to Unified Memory when a program's data does not fit in device memory.

    [PAPER_GPU_SIZE=large] python paper_gpu/uvm_oversub.py [--only NAME,...] [--page KB]

Each benchmark is modelled as if its device allocations were managed memory (cudaMallocManaged): a copy to the
device becomes a host write (the pages it writes are on the host afterwards), a copy to the host a host read (the
pages it reads come back if they are on the device). Device memory holds a share of the pages the program's device
operations touch (--fractions); a page is --page KB (64, Unified Memory's basic block). Operations run one after
another in the traced order. What each one actually touches is the traced run's footprint (truth_access.csv); a
policy's footprints only decide what it prefetches and evicts:

    demand      no footprints: a kernel faults on every page not on the device; LRU eviction
    buffer      before a kernel, prefetch every page of every allocation it touches (the buffers it uses, as a
                compiler that knows only which data structures a kernel uses would); LRU eviction
    hybrid      before a kernel, prefetch the pages the hybrid interpreter's footprint over every schedule
                (NAME_access_all.csv) says it touches; evict the page whose next use in that footprint is
                farthest away (a page no later operation uses first)
    truth       the same with the traced footprints: an oracle
    hybrid_min, truth_min
                the same, but prefetch only runs of at least uvm_costs.min_run contiguous pages (shorter runs cost
                more as a prefetch call than as faults)

Costs (uvm_costs.py): a kernel's own time is its profiled time (sim_ops.csv). With the measured costs (costs
"measured"), the pages a kernel faults on stall it as long as faultbench measured for that many pages, a prefetch
costs prefetchbench's fit per call (one call per contiguous run of pages) and per byte, and a page leaving the
device is always copied back. With the first model's assumptions (costs "assumed", one row per --fault-us), a
fault batch of up to --batch faults costs --fault-us plus the pages' transfers, a prefetch one fixed copy cost
plus the transfers, and only evicting a page the device wrote since it arrived copies it back. Only shares at
which every operation's pages fit are simulated: thrashing within one kernel depends on the order of its
accesses, which the footprints do not record (such rows only say how many pages the largest operation needs). Transfers use the copy model of
copy_savings.py (fitted on the profiled copies). Footprints of the interpreter's events are matched to the trace's
as copy_savings.py does; a traced event without a match gets no prefetch.
Writes data/uvm{,_large}.csv: per program, fraction, costs and policy, the time, the stall, the faults and
the bytes moved each way, with the time of the device operations alone (resident_ns) and of the program as
written, copies included (explicit_ns).
"""

import argparse
import csv
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmarks as B  # noqa: E402
from copy_savings import H2D, D2H, UNIT, accesses, fit  # noqa: E402
from uvm_costs import Assumed, Measured, runs  # noqa: E402
from memprint.static.kdeps import align_ends  # noqa: E402

DATA = HERE / "data"
POLICIES = ("demand", "buffer", "hybrid", "truth", "hybrid_min", "truth_min")


def pages(acc, page_units, write_bit=2):
    """{event: (pages read, pages written)} with a page as (allocation, page index)."""
    out = defaultdict(lambda: (set(), set()))
    for e, rows in acc.items():
        rd, wr = out[e]
        for a, lo, hi, bits in rows:
            span = {(a, p) for p in range(lo // page_units, (hi - 1) // page_units + 1)} if hi > lo else set()
            if bits & 1:
                rd |= span
            if bits & write_bit:
                wr |= span
    return out


class Device:
    """The pages on the device, in LRU order, and which of them the device wrote since they arrived."""

    def __init__(self, capacity):
        self.capacity, self.lru, self.dirty = capacity, OrderedDict(), set()

    def touch(self, p):
        self.lru.move_to_end(p)


def simulate(events, ops, actual, predicted, alloc_pages, policy, capacity, costs):
    """(time ns, stall ns, faults, bytes to the device, bytes to the host) of one run, charged by costs
    (uvm_costs.Measured or Assumed)."""
    page_bytes = costs.page_bytes
    footprints = policy.split("_")[0] in ("hybrid", "truth")
    min_run = costs.min_run if policy.endswith("_min") else 1
    dev = Device(capacity)
    time = stall = faults = to_dev = to_host = 0
    uses = defaultdict(list)                                             # page -> events whose prediction uses it
    if footprints:
        for e in sorted(predicted):
            rd, wr = predicted[e]
            for p in rd | wr:
                uses[p].append(e)
    nxt = {p: 0 for p in uses}                                           # position in uses[p] not yet passed

    def next_use(p, now):
        u, k = uses.get(p, ()), nxt.get(p, 0)
        while k < len(u) and u[k] <= now:
            k += 1
        nxt[p] = k
        return u[k] if k < len(u) else math.inf

    def evict(now, keep, protect, thrash):
        nonlocal to_host, stall, time
        if footprints:
            victim = max((p for p in dev.lru if p not in keep), key=lambda p: (next_use(p, now), p not in dev.dirty),
                         default=None)
        else:
            victim = next((p for p in dev.lru if p not in keep), None)
        if victim is None and thrash:                                    # the operation needs more pages than
            victim = next((p for p in dev.lru if p != protect), None)   # fit: it evicts its own (thrashing),
        if victim is None:                                               # least recently used first
            return False
        del dev.lru[victim]
        if victim in dev.dirty or costs.writeback_clean:
            to_host += page_bytes
            time += costs.d2h_ns
            stall += costs.d2h_ns
        dev.dirty.discard(victim)
        return True

    def bring(p, now, keep, thrash=True):
        nonlocal to_dev
        while len(dev.lru) >= dev.capacity:
            if not evict(now, keep, p, thrash):
                return False
        dev.lru[p] = True
        to_dev += page_bytes
        return True

    for e in events:
        i = int(e["index"])
        rd, wr = actual.get(i, (set(), set()))
        touched = rd | wr
        if e["name"] in (H2D, D2H):                                      # host writes or reads managed pages
            back = [p for p in touched if p in dev.lru]
            moved = [p for p in back if p in dev.dirty or e["name"] == H2D or costs.writeback_clean]
            for p in back:                                               # they move to the host
                del dev.lru[p]
                dev.dirty.discard(p)
            to_host += len(moved) * page_bytes
            if back:                                                     # the host faults them back
                t = costs.host_fault(len(moved))
                time += t
                stall += t
            continue
        keep = set(touched)
        if policy != "demand":                                           # prefetch before the operation
            if policy == "buffer":
                want = set()
                for a in {a for a, _ in touched}:
                    want |= alloc_pages[a]
            else:
                prd, pwr = predicted.get(i, (set(), set()))
                want = prd | pwr
            want = [p for r in runs(p for p in want if p not in dev.lru) if len(r) >= min_run for p in r]
            want = want[:max(0, dev.capacity)]
            moved = []
            for p in want:                                               # not past what fits beside the
                if not bring(p, i, keep | set(want), thrash=False):      # operation's own pages
                    break
                moved.append(p)
            if moved:
                t = costs.prefetch([len(r) for r in runs(moved)])
                time += t
                stall += t
        missing = [p for p in touched if p not in dev.lru]
        if missing:                                                      # faults
            n = len(missing)
            faults += n
            t = costs.fault(n)
            time += t
            stall += t
            for p in missing:
                bring(p, i, keep)
        for p in touched:
            if p in dev.lru:
                dev.touch(p)
        for p in wr:
            if p in dev.lru:
                dev.dirty.add(p)
        time += int(ops[i]["dur_ns"])
    return time, stall, faults, to_dev, to_host


def program(name, fractions, cost_models, page_kb):
    w = B.workdir(B.BENCH[name])
    tdir, hdir = w / "truth", w / "hybrid"
    page_bytes = page_kb * 1024
    pu = page_bytes // UNIT
    sizes = {int(r["index"]): -(-int(r["bytes"]) // UNIT) for r in csv.DictReader(open(tdir / "truth_allocs.csv"))}
    alloc_pages = {a: {(a, p) for p in range(-(-n // pu))} for a, n in sizes.items()}
    tev = list(csv.DictReader(open(tdir / "truth_events.csv")))
    ops = {int(r["index"]): r for r in csv.DictReader(open(w / "sim_ops.csv"))}
    if len(ops) != len(tev):
        return [{"name": name, "status": f"profile has {len(ops)} operations, trace {len(tev)}"}]
    actual = pages(accesses(tdir / "truth_access.csv"), pu)
    iev = [e for e in csv.DictReader(open(hdir / f"{name}_events.csv")) if e["kind"] != "host"]
    same = lambda x, y: x["kind"] == y["kind"] and (x["name"] == y["name"] if x["kind"] != "kernel"  # noqa: E731
                                                    else x["name"].split("(")[0] == y["name"].split("(")[0])
    to_interp = {t: int(iev[k]["index"]) for t, k in align_ends(tev, iev, same).items()}
    ipages = pages(accesses(hdir / f"{name}_access_all.csv"), pu)
    hybrid = {t: ipages[j] for t, j in to_interp.items() if j in ipages}
    device = {i for i, e in enumerate(tev) if e["name"] not in (H2D, D2H)}
    total = set()
    for i in device:
        rd, wr = actual.get(i, (set(), set()))
        total |= rd | wr
    largest = max((len(actual[i][0] | actual[i][1]) for i in device if i in actual), default=0)
    resident = sum(int(ops[i]["dur_ns"]) for i in device)
    explicit = sum(int(o["dur_ns"]) for o in ops.values())             # the program as written, with its copies
    rows = []
    for f in fractions:
        cap = max(1, math.ceil(f * len(total)))
        if cap < largest:                                                # thrashing inside one kernel depends on
            rows.append({"name": name, "suite": B.BENCH[name]["suite"], "page_kb": page_kb,   # the order of its
                         "pages": len(total), "fraction": f, "capacity": cap,                   # accesses, which
                         "status": f"an operation needs {largest} pages"})                     # is not modelled
            continue
        for costs in cost_models:
            for policy in POLICIES:
                pred = {"hybrid": hybrid, "truth": {i: actual[i] for i in device if i in actual}}.get(
                    policy.split("_")[0], {})
                t, s, n, up, down = simulate(tev, ops, actual, pred, alloc_pages, policy, cap, costs)
                rows.append({"name": name, "suite": B.BENCH[name]["suite"], "status": "ok", "page_kb": page_kb,
                             "pages": len(total), "fraction": f, "capacity": cap, "costs": costs.name,
                             "fault_us": costs.fault_us, "policy": policy,
                             "resident_ns": resident, "explicit_ns": explicit, "time_ns": int(t), "stall_ns": int(s), "faults": n,
                             "to_device_bytes": up, "to_host_bytes": down, "unmatched": len(device - set(to_interp))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--page", type=int, default=64, help="page size in KB (data/uvm{,_large}_4k.csv if not 64)")
    ap.add_argument("--fractions", default="0.25,0.375,0.5,0.625,0.75,0.875,1.0")
    ap.add_argument("--fault-us", default="10,25,50", help="assumed cost of a batch of faults in microseconds")
    ap.add_argument("--batch", type=int, default=256, help="faults handled in one batch")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else [
        n for n in B.BENCH if (B.workdir(B.BENCH[n]) / "truth" / "truth_access.csv").exists()
        and (B.workdir(B.BENCH[n]) / "hybrid" / f"{n}_access_all.csv").exists()
        and (B.workdir(B.BENCH[n]) / "sim_ops.csv").exists()]
    model = fit()
    fractions = [float(x) for x in args.fractions.split(",")]
    cost_models = [Measured(args.page, model)] + [Assumed(args.page, model, float(x), args.batch)
                                                  for x in args.fault_us.split(",")]
    rows = []
    for n in names:
        rows += program(n, fractions, cost_models, args.page)
        r = [x for x in rows if x["name"] == n and x.get("fraction") == 0.5 and x.get("costs") == "measured"]
        print(n, {x["policy"]: round(x["time_ns"] / x["resident_ns"], 2) for x in r} or rows[-1].get("status"),
              flush=True)
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    path = DATA / f"uvm{B.SUFFIX}{'' if args.page == 64 else f'_{args.page}k'}.csv"
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, cols)
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
