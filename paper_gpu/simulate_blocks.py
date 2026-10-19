"""Simulate a CUDA program's operations on a GPU block by block, with dependences between thread blocks.

    python paper_gpu/simulate_blocks.py [--profile] [--only NAME,...] [--delay US] [--host-sync]

The block-granularity counterpart of simulate.py (same profiles, same GPU and scheduling-delay model; see its
docstring). A kernel is split into its thread blocks, each taking the kernel's measured duration divided by
the number of waves it needs on the GPU (all blocks of a kernel take equally long), and the simulator places
each block on a streaming multiprocessor (SM) with room for its threads, registers, shared memory and a block
slot. A block may start once the operations, or the blocks of operations, it depends on have ended. A copy or
a memset is one block.

Dependences between blocks: the units each block of each kernel read and wrote are folded in program order
like operations are (kdeps.Tracker), with copies and memsets as single blocks. The traced run's come from
gpu/nvbit_rw with KDEPS_BLOCKS=1 (data/bench/NAME/truth_blocks/). The hybrid interpreter's
(hybrid.py --blocks: data/bench/NAME/hybrid/schedule_*/NAME_blocks.npz) refine a dependence it reports between
two operations to dependences between their blocks only where that is safe for other thread orders: the
dependence is certain, neither operation referenced an object the interpreter could not place, and each
kernel's blocks referenced the same units under all four schedules; a write then hides an earlier one from
later blocks only if every legal order makes it (WX). Every other dependence makes every block wait for the
whole earlier operation.

Policies: those of simulate.py (all between whole operations) and
    hybrid-block   the hybrid interpreter's dependences, refined to blocks where safe
    truth-block    the traced run's dependences between blocks: the best any block-level checker can do
A block that starts before a block it truly depends on has ended (truth-block) is rolled back and issued again
after the scheduling delay.

Writes data/sim_blocks.csv and data/sim_blocks_delay.csv.
"""

import argparse
import csv
import heapq
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import simulate as S  # noqa: E402
from simulate import B, ks, DATA, GPU  # noqa: E402
from memprint.static.kdeps import align_ends, R, W, RM, WM, WX  # noqa: E402

POLICIES = ["sequential", "buffer", "buffer-rw", "hybrid", "full", "truth", "hybrid-block", "truth-block"]
SCHEDULES = ["lockstep", "blocks", "reverse", "random"]


# ---------------------------------------------------------------- who touched what, per block

def _grid_blocks(grid):
    if not grid:
        return 1
    x, y, z = (int(v) for v in grid.split("x"))
    return x * y * z


def _ranges(path, per_block):
    """{event: {alloc: (blocks, units, bits)}} from an access file in truth format (runs of units)."""
    df = pd.read_csv(path)
    out = {}
    if df.empty:
        return out
    df["n"] = df.unit_hi - df.unit_lo
    if not per_block:
        df["block"] = 0
    for (e, a), g in df.groupby(["event", "alloc"], sort=False):
        n = g.n.to_numpy()
        start = np.repeat(g.unit_lo.to_numpy() - np.r_[0, np.cumsum(n)[:-1]], n)
        units = (start + np.arange(int(n.sum()))).astype(np.int32)
        out.setdefault(int(e), {})[int(a)] = (np.repeat(g.block.to_numpy().astype(np.int32), n), units,
                                              np.repeat(g.bits.to_numpy().astype(np.uint8), n))
    return out


def truth_blocks(w):
    """The traced run's events and per-event {alloc: (blocks, units, bits)}: kernels per block (block access
    file), copies and memsets as block 0."""
    tdir = w / "truth_blocks"
    events = list(csv.DictReader(open(tdir / "truth_events.csv")))
    per_block = _ranges(tdir / "truth_block_access.csv", True)
    whole = _ranges(tdir / "truth_access.csv", False)
    acc = {}
    for e in events:
        i = int(e["index"])
        acc[i] = per_block.get(i, {}) if e["kind"] == "kernel" else whole.get(i, {})
    return events, acc


def interp_blocks(hdir, name, schedule):
    """One schedule's run of the hybrid interpreter: (events, {event: {alloc: (blocks, units, bits)} or None})
    for its kernels (None: a reference to an unknown object), from schedule_*/NAME_blocks.npz."""
    sub = hdir / f"schedule_{schedule}"
    path = sub / f"{name}_blocks.npz"
    if not path.exists():
        return None
    events = list(csv.DictReader(open(sub / f"{name}_events.csv")))
    alloc = {r["name"]: int(r["alloc"]) for r in csv.DictReader(open(sub / f"{name}_objects.csv")) if r["alloc"]}
    z = np.load(path)
    acc = {int(e): None for e in z["unknown"]}
    for e in (z["same"] if "same" in z.files else ()):
        acc[int(e)] = "same"                                           # run as in lockstep there
    uniform = {}
    for k in z.files:
        if not k.endswith("/block"):
            continue
        e, obj, _ = k.split("/")
        if obj not in alloc or acc.get(int(e), {}) is None or acc.get(int(e)) == "same":
            continue
        acc.setdefault(int(e), {})[alloc[obj]] = (z[k].astype(np.int64), z[f"{e}/{obj}/unit"].astype(np.int64),
                                                  z[f"{e}/{obj}/bits"])
        if f"{e}/{obj}/uniform" in z.files and int(z[f"{e}/{obj}/uniform"]):
            uniform.setdefault(int(e), set()).add(alloc[obj])
    return events, acc, uniform


# ---------------------------------------------------------------- folding blocks into dependences

def _incidence(b, u, nblocks, nunits):
    return sp.csr_matrix((np.ones(len(b), dtype=np.int32), (b, u)), shape=(nblocks, nunits))


def fold(n_events, acc, pairs, kills, nb, raw_writers=None):
    """Dependences between blocks: {dst event: {dst block: {(src event, src block)}}}, for the pairs of
    operations that depend on each other (pairs: {dst: {src}}). acc: per event {alloc: (blocks, units, bits)},
    bits R/W or RM/WM (possible); kills(event, bits): whether a certain write hides the unit's earlier
    writers and readers from later events. Block sb of s and block db of d depend on each other if one
    writes a unit the other references and no event between them hid the unit. raw_writers: {event: {alloc:
    (blocks, units)}}: for a read of such a unit, the blocks of that event a reader must wait for, in place of
    all its writers (same-value writes, hybrid_block)."""
    raw_writers = raw_writers or {}
    lastkill = {}                                                      # alloc -> per unit, the last event
    edges = {}                                                         # that hid it
    width = {}
    for e in range(n_events):
        for a, (b, u, v) in (acc.get(e) or {}).items():
            if len(u):
                width[a] = max(width.get(a, 0), int(u.max()) + 1)
    for a, n in width.items():
        lastkill[a] = np.full(n, -1, dtype=np.int64)
    for d in range(n_events):
        mine = acc.get(d) or {}
        for s in sorted(pairs.get(d, ())):
            if s >= d:
                continue
            other = acc.get(s) or {}
            for a in mine.keys() & other.keys():
                sb, su, sv = other[a]
                db, du, dv = mine[a]
                n = width[a]
                live = lastkill[a][su] <= s                            # not hidden since s
                s_w = live & ((sv & (W | WM)) != 0)
                s_r = live & ((sv & (R | RM)) != 0)
                d_w = (dv & (W | WM)) != 0
                d_r = (dv & (R | RM)) != 0
                raw_b, raw_u = sb[s_w], su[s_w]                        # whom a read waits for
                if a in raw_writers.get(s, {}):
                    rb, ru = raw_writers[s][a]
                    keep = lastkill[a][ru] <= s
                    raw_b, raw_u = rb[keep], ru[keep]
                got = None
                for (xb, xu), xd in (((raw_b, raw_u), d_r), ((sb[s_w], su[s_w]), d_w), ((sb[s_r], su[s_r]), d_w)):
                    if not len(xb) or not xd.any():
                        continue
                    m = _incidence(xb, xu, nb[s], n) @ _incidence(db[xd], du[xd], nb[d], n).T
                    got = m if got is None else got + m
                if got is not None:
                    got = got.tocoo()
                    per = edges.setdefault(d, {})
                    for x, y in zip(got.row.tolist(), got.col.tolist()):
                        per.setdefault(y, set()).add((s, x))
        for a, (b, u, v) in mine.items():                              # what d hides from later events
            w = (v & W) != 0
            if w.any():
                k = np.array([kills(d, int(x)) for x in np.unique(v[w])])
                hide = np.isin(v, np.unique(v[w])[k]) & w
                lastkill[a][u[hide]] = d
    return edges


# ---------------------------------------------------------------- the policies

def block_policies(name):
    """Per policy: (whole, spec): whole[e] = events all blocks of e wait for; spec[e] = {block: {(s, sb)}}.
    Also the number of blocks per event, the events, and the truth's spec (for rollbacks)."""
    w = B.workdir(B.BENCH[name])
    events, tacc = truth_blocks(w)
    n = len(events)
    nb = [max(1, _grid_blocks(e["grid"])) if e["kind"] == "kernel" else 1 for e in events]
    kernel_level = S.dependences(name, truth_sub="truth_blocks")
    truth_spec = fold(n, tacc, kernel_level["truth"], lambda d, bits: True, nb)
    out = {p: ({e: set(v) for e, v in preds.items()}, {}) for p, preds in kernel_level.items()}
    out["truth-block"] = ({e: set() for e in range(n)}, truth_spec)
    hb = hybrid_block(name, w, events, nb)
    if hb is not None:
        out["hybrid-block"] = hb
    return events, nb, out, truth_spec


def _union_acc(a, b):
    out = dict(a)
    for k, (bb, uu, vv) in b.items():
        if k in out:
            x = out[k]
            out[k] = (np.concatenate([x[0], bb]), np.concatenate([x[1], uu]), np.concatenate([x[2], vv]))
        else:
            out[k] = (bb, uu, vv)
    return out


def hybrid_block(name, w, truth_events, nb_truth, mode="safe"):
    """The hybrid interpreter's dependences, refined to blocks where safe (module docstring).

    A kernel launch is refined if its blocks referenced the same units under all four schedules, or if they
    differ only in writes to objects all of whose writes in the launch stored a value that is the same for every
    thread (UV: a level, a flag): the units such a launch writes end it with that value whichever writer ran last,
    so a later read waits only for one block that read the unit in every schedule and wrote it in lockstep
    (it either wrote the value or found it written), and a later write for every block that wrote it in any
    schedule. mode="union" (an experiment, not safe): refine every dependence between events every schedule
    issued, on the union of what each block referenced under the four schedules, whether certain or not."""
    hdir = w / "hybrid"
    runs = [interp_blocks(hdir, name, s) for s in SCHEDULES]
    if any(r is None for r in runs):
        return None
    ie, iedges = ks.interp_edges(hdir, name)
    m = ks.truth_map(ie, truth_events)
    inv = {v: k for k, v in m.items()}                                  # interpreter index -> run index
    base_events, base_acc, base_uniform = runs[0]
    sig = [(e["kind"], e["name"]) for e in base_events]
    others = {}                                                        # base index -> [its refs in each other run]
    for events, acc, uniform in runs[1:]:
        mp = align_ends([(e["kind"], e["name"]) for e in events], sig)    # their index -> base index
        back = {v: k for k, v in mp.items()}
        for i, e in enumerate(base_events):
            if e["kind"] != "kernel":
                continue
            j = back.get(i)
            ref = acc.get(j, {}) if j is not None else None
            if ref == "same":
                ref = base_acc.get(i, {})
            others.setdefault(i, []).append((ref, uniform.get(j, set()) if j is not None else set()))
    stable, union, raw_writers = {}, {}, {}
    for i, e in enumerate(base_events):
        if e["kind"] != "kernel":
            stable[i] = True
            continue
        mine = base_acc.get(i, {})
        refs = others.get(i, [])
        if mine is None or any(r is None for r, _ in refs):
            stable[i] = False
            continue
        if mode == "union":
            u = mine
            for r, _ in refs:
                u = _union_acc(u, r)
            stable[i], union[i] = True, u
            continue
        if all(_same_acc(mine, r) for r, _ in refs):
            stable[i] = True
            continue
        same_value = set(base_uniform.get(i, set()))                   # objects written only with same values,
        for _, uni in refs:                                            # in every schedule
            same_value &= uni
        ok = all(_same_acc(mine, r, writes_of=same_value) for r, _ in refs)
        stable[i] = ok
        if ok:
            u = mine
            for r, _ in refs:
                u = _union_acc(u, r)
            union[i] = u
            raw_writers[i] = {a: _designated(mine, [r for r, _ in refs], a) for a in same_value if a in mine}
    certain = {}
    for (s, d, _, _), (must, may, _) in iedges.items():
        certain[(s, d)] = certain.get((s, d), True) and must > 0 and may == 0
    # copies and memsets of the interpreter's run, as block 0 (its access file is the lockstep run's)
    acc = {i: v for i, v in base_acc.items() if v is not None and v != "same"}
    acc.update(union)
    path = hdir / f"{name}_access.csv"
    if path.exists():
        for i, d in _ranges(path, False).items():
            if base_events[i]["kind"] != "kernel":
                acc[i] = d
    kernel_pairs = S._predicted(hdir, name, truth_events)
    n = len(truth_events)
    want = {}                                                          # the refinable pairs, interpreter numbering
    for s, d in kernel_pairs:
        si, di = m.get(s), m.get(d)
        if s < d and si is not None and di is not None and (mode == "union" or certain.get((si, di), False)) \
                and stable.get(si) and stable.get(di):
            want.setdefault(di, set()).add(si)
    nb_i = [nb_truth[inv[i]] if i in inv else 1 for i in range(len(base_events))]
    pred_spec = fold(len(base_events), acc, want,
                     lambda d, bits: base_events[d]["kind"] != "kernel" or bool(bits & WX), nb_i, raw_writers)
    whole = {e: set() for e in range(n)}
    spec = {}
    for s, d in kernel_pairs:
        if s >= d:
            continue
        si, di = m.get(s), m.get(d)
        ok = si is not None and di is not None and si in want.get(di, ())
        blocks = {}
        if ok:
            for db, srcs in pred_spec.get(di, {}).items():
                got = {(inv[x], sb) for x, sb in srcs if x == si}
                if got:
                    blocks[db] = got
        if ok and blocks:
            for db, got in blocks.items():
                spec.setdefault(d, {}).setdefault(db, set()).update(got)
        else:
            whole[d].add(s)
    return whole, spec


def _designated(mine, refs, a):
    """For each unit of object a the launch wrote in some schedule, the one block a later read waits for: the
    lowest block that wrote it in lockstep (mine) and read it there (and, the reads being the same, in every
    schedule); units without such a block keep all their writers. (blocks, units)."""
    b, u, v = mine[a][:3]
    rd = (v & (R | RM)) != 0
    wr = (v & (W | WM)) != 0
    both = rd & wr
    designated = {}
    for x, y in zip(u[both].tolist(), b[both].tolist()):
        if x not in designated or y < designated[x]:
            designated[x] = y
    writers = {}                                                       # every writer in any schedule
    for r in [mine] + refs:
        if a not in r:
            continue
        bb, uu, vv = r[a][:3]
        w = (vv & (W | WM)) != 0
        for x, y in zip(uu[w].tolist(), bb[w].tolist()):
            writers.setdefault(x, set()).add(y)
    out_b, out_u = [], []
    for x, ys in writers.items():
        for y in ([designated[x]] if x in designated else sorted(ys)):
            out_b.append(y)
            out_u.append(x)
    return np.array(out_b, dtype=np.int64), np.array(out_u, dtype=np.int64)


def _same_acc(a, b, writes_of=()):
    """Whether two runs' references of a launch are the same; for the objects in writes_of, only the reads."""
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if k in writes_of:
            rx, ry = (x[2] & (R | RM)) != 0, (y[2] & (R | RM)) != 0
            x, y = (x[0][rx], x[1][rx], x[2][rx] & (R | RM)), (y[0][ry], y[1][ry], y[2][ry] & (R | RM))
        if len(x[0]) != len(y[0]):
            return False
        ox, oy = np.lexsort((x[0], x[1])), np.lexsort((y[0], y[1]))
        if not (np.array_equal(x[0][ox], y[0][oy]) and np.array_equal(x[1][ox], y[1][oy])
                and np.array_equal(x[2][ox] & (R | W | RM | WM), y[2][oy] & (R | W | RM | WM))):
            return False
    return True


# ---------------------------------------------------------------- storing them compactly

def compact(spec):
    """Dependences between blocks ({dst event: {dst block: {(src event, src block)}}}) stored compactly: per pair
    of events, each dst block's source blocks as ranges, and runs of consecutive dst blocks whose ranges are the
    same shifted by a constant step per block (block b on blocks b-1 to b+1; every block on one block) as one
    record (first dst block, last, step, ranges), or as plain (dst, src) pairs where those take less. Returns
    (records, integers stored, pairs)."""
    records = ints = pairs = 0

    def close(run):
        nonlocal records, ints
        if run is None:
            return
        cost = 3 + 2 * len(run[2])
        if cost < 2 * run[3]:
            records += 1
            ints += cost
        else:
            records += run[3]
            ints += 2 * run[3]

    for d, per in spec.items():
        by_src = {}
        for db, srcs in per.items():
            for s_, sb in srcs:
                by_src.setdefault(s_, {}).setdefault(db, []).append(sb)
        for s_, rows in by_src.items():
            run = None                                                 # (last db, step, ranges, pairs in run)
            for db in sorted(rows):
                bs = sorted(set(rows[db]))
                pairs += len(bs)
                rng, lo = [], bs[0]
                for x, y in zip(bs, bs[1:] + [None]):
                    if y != x + 1:
                        rng.append((lo, x))
                        lo = y
                rng = tuple(rng)
                if run is not None and db == run[0] + 1 and len(rng) == len(run[2]):
                    step = rng[0][0] - run[2][0][0]
                    if (run[1] is None or step == run[1]) and \
                            all(a0 - b0 == step and a1 - b1 == step for (a0, a1), (b0, b1) in zip(rng, run[2])):
                        run = (db, step, rng, run[3] + len(bs))
                        continue
                close(run)
                run = (db, None, rng, len(bs))
            close(run)
    return records, ints, pairs


# ---------------------------------------------------------------- the GPU, block by block

def block_shape(op):
    """(threads, registers, shared bytes) of one block of an operation on an SM; None for a copy."""
    if op["kind"] == "copy":
        return None
    if op["kind"] == "set" or op["dur_ns"] == 0:
        return (32, 32 * 16, 0) if op["kind"] == "set" else (0, 0, 0)
    t = max(1, op["threads"])
    return (t, max(1, op["regs"]) * t, op["smem"])


def _per_sm(shape):
    t, regs, smem = shape
    return max(1, min(GPU["blocks"], GPU["threads"] // max(1, t), GPU["regs"] // max(1, regs),
                      GPU["smem"] // smem if smem else GPU["blocks"]))


def block_durations(ops, nb, work):
    """Per event, each block's duration (ns). A kernel with instruction counts per block (work: {event:
    {block: warp instructions}}) gets durations in proportion to them, scaled so that the kernel alone, its
    blocks issued in order to the free slots of the GPU, takes its profiled duration; otherwise every block
    takes the duration divided by the number of waves."""
    out = []
    for e, o in enumerate(ops):
        sh = block_shape(o)
        if sh is None or o["dur_ns"] == 0 or sh == (0, 0, 0):
            out.append(np.full(nb[e], int(o["dur_ns"]), dtype=np.int64))
            continue
        slots = _per_sm(sh) * GPU["sms"]
        w = (work or {}).get(e)
        if w:
            weights = np.array([w.get(b, 0) for b in range(nb[e])], dtype=float)
            missing = weights == 0
            if missing.all():
                w = None
            else:
                weights[missing] = np.median(weights[~missing])
        if not w:
            waves = math.ceil(nb[e] / slots)
            out.append(np.full(nb[e], max(1, int(o["dur_ns"]) // max(1, waves)), dtype=np.int64))
            continue
        free = [0.0] * min(slots, nb[e])                               # the kernel alone: list scheduling
        heapq.heapify(free)
        span = 0.0
        for x in weights:
            t = heapq.heappop(free) + x
            span = max(span, t)
            heapq.heappush(free, t)
        out.append(np.maximum(1, np.round(weights * (int(o["dur_ns"]) / span))).astype(np.int64))
    return out


def load_work(name, n_events):
    """{event: {block: warp instructions}} from truth_work, if its run issued the block trace's operations."""
    w = B.workdir(B.BENCH[name])
    path = w / "truth_work" / "truth_block_work.csv"
    if not path.exists():
        return None
    ops = list(csv.DictReader(open(w / "truth_work" / "truth_events.csv")))
    ref = list(csv.DictReader(open(w / "truth_blocks" / "truth_events.csv")))
    if [(r["kind"], r["name"]) for r in ops] != [(r["kind"], r["name"]) for r in ref] or len(ref) != n_events:
        return None
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(int(r["event"]), {})[int(r["block"])] = int(r["warp_instrs"])
    return out


def simulate(ops, nb, whole, spec, truth_spec, delay_ns, host_sync=False, dur=None):
    """Run the blocks under one policy: (makespan ns, rollbacks, ns of work rolled back). dur: per event, each
    block's duration (block_durations); by default equal within a kernel."""
    n = len(ops)
    whole = {e: set(whole.get(e, ())) for e in range(n)}
    if host_sync:
        last = None
        for j, o in enumerate(ops):
            if last is not None:
                whole[j].add(last)
            if o["name"] == "cudaMemcpy D2H":
                last = j
    shape = [block_shape(o) for o in ops]
    if dur is None:
        dur = block_durations(ops, nb, None)
    free = np.array([[GPU["threads"], GPU["blocks"], GPU["regs"], GPU["smem"]]] * GPU["sms"], dtype=np.int64)
    busy = set()
    rem_whole = [len(whole[e]) for e in range(n)]
    whole_succ = {e: [] for e in range(n)}
    for e in range(n):
        for s in whole[e]:
            whole_succ[s].append(e)
    rem_spec = [np.zeros(nb[e], dtype=np.int64) for e in range(n)]
    spec_succ = {}
    for e, per in spec.items():
        for b, srcs in per.items():
            if b >= nb[e]:
                continue
            rem_spec[e][b] = len(srcs)
            for s in srcs:
                spec_succ.setdefault(s, []).append((e, b))
    done = [np.zeros(nb[e], dtype=bool) for e in range(n)]
    end_t = [np.zeros(nb[e], dtype=np.int64) for e in range(n)]
    n_done = [0] * n
    ready = {}                                                          # event -> list of ready blocks
    retry = set()                                                       # (event, block) re-issued after rollback
    retry_wait, retry_succ = {}, {}
    timed = []                                                          # (time, event, block): re-issue at
    running = []                                                        # (end, event, block, sm, final)
    now, rollbacks, wasted, total = 0, 0, 0, sum(nb)
    finished = 0

    def make_ready(e, b):
        ready.setdefault(e, []).append(b)

    for e in range(n):
        if rem_whole[e] == 0:
            for b in np.flatnonzero(rem_spec[e] == 0):
                make_ready(e, int(b))

    def truth_done(e, b):
        return all(done[s][sb] for s, sb in truth_spec.get(e, {}).get(b, ()))

    while finished < total:
        for e in sorted(ready):                                         # dispatch in program order
            blocks = ready[e]
            sh = shape[e]
            if sh is None:                                              # a copy: its engine, one at a time
                eng = ops[e]["name"].split()[-1]
                if eng in busy:
                    continue
                busy.add(eng)
                take, sms = blocks[:1], [-1]
            elif sh == (0, 0, 0):                                       # a launch that did not run
                take, sms = blocks, [-2] * len(blocks)
            else:                                                       # blocks fill free SMs in SM order
                t, regs, smem = sh
                cap = np.minimum(np.minimum(free[:, 0] // max(1, t), free[:, 1]),
                                 np.minimum(free[:, 2] // max(1, regs),
                                            free[:, 3] // smem if smem else free[:, 1]))
                room = int(cap.sum())
                if room == 0:
                    continue
                k = min(room, len(blocks))
                sms = np.repeat(np.arange(len(cap)), cap)[:k]
                used = np.bincount(sms, minlength=len(cap))
                free -= used[:, None] * np.array([t, 1, regs, smem], dtype=np.int64)
                take, sms = blocks[:k], sms.tolist()
            for b, sm in zip(take, sms):
                final = (e, b) in retry or truth_done(e, b)
                heapq.heappush(running, (now + int(dur[e][b]), e, b, sm, final))
            rest = blocks[len(take):]
            if rest:
                ready[e] = rest
            else:
                del ready[e]
        nxt = []
        if running:
            nxt.append(running[0][0])
        if timed:
            nxt.append(timed[0][0])
        if not nxt:
            raise RuntimeError("deadlock")
        now = max(now, min(nxt))
        while timed and timed[0][0] <= now:
            _, e, b = heapq.heappop(timed)
            make_ready(e, b)
        while running and running[0][0] <= now:
            t, e, b, sm, final = heapq.heappop(running)
            sh = shape[e]
            if sm >= 0:
                free[sm] += (sh[0], 1, sh[1], sh[2])
            elif sm == -1:
                busy.discard(ops[e]["name"].split()[-1])
            if not final:                                               # it ran before a true predecessor ended
                rollbacks += 1
                wasted += int(dur[e][b])
                retry.add((e, b))
                waits = {(s, sb) for s, sb in truth_spec.get(e, {}).get(b, ()) if not done[s][sb]}
                if waits:
                    retry_wait[(e, b)] = len(waits)
                    for x in waits:
                        retry_succ.setdefault(x, []).append((e, b))
                else:
                    heapq.heappush(timed, (now + delay_ns, e, b))
                continue
            done[e][b] = True
            end_t[e][b] = t
            finished += 1
            n_done[e] += 1
            for d, db in spec_succ.get((e, b), ()):
                rem_spec[d][db] -= 1
                if rem_spec[d][db] == 0 and rem_whole[d] == 0:
                    make_ready(d, db)
            for d, db in retry_succ.pop((e, b), ()):
                retry_wait[(d, db)] -= 1
                if retry_wait[(d, db)] == 0:
                    heapq.heappush(timed, (now + delay_ns, d, db))
            if n_done[e] == nb[e]:
                for d in whole_succ[e]:
                    rem_whole[d] -= 1
                    if rem_whole[d] == 0:
                        for db in np.flatnonzero(rem_spec[d] == 0):
                            make_ready(d, int(db))
    return int(max(x.max() for x in end_t)), rollbacks, wasted


# ---------------------------------------------------------------- main

def load_ops(name):
    """The profile matching truth_blocks: sim_ops_blocks.csv if the block trace issued other operations than
    the first trace, else sim_ops.csv."""
    w = B.workdir(B.BENCH[name])
    if (w / "sim_ops_blocks.csv").exists():
        rows = list(csv.DictReader(open(w / "sim_ops_blocks.csv")))
        return [{**r, "dur_ns": int(r["dur_ns"]), "blocks": int(r["blocks"]), "threads": int(r["threads"]),
                 "regs": int(r["regs"]), "smem": int(r["smem"])} for r in rows]
    return S.load_ops(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true", help="profile benchmarks whose block trace differs")
    ap.add_argument("--only", default=None)
    ap.add_argument("--delay", type=float, default=None, help="scheduling delay in microseconds")
    ap.add_argument("--host-sync", action="store_true")
    ap.add_argument("--equal", action="store_true", help="every block of a kernel takes equally long (no "
                    "instruction counts)")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else [
        n for n in B.BENCH if (B.workdir(B.BENCH[n]) / "truth_blocks" / "truth_block_access.csv").exists()
        and (B.workdir(B.BENCH[n]) / "hybrid" / f"{n}_events.csv").exists()]
    if args.profile:
        for n in names:
            w = B.workdir(B.BENCH[n])
            a = [(r["kind"], r["name"]) for r in csv.DictReader(open(w / "truth" / "truth_events.csv"))]
            t = [(r["kind"], r["name"]) for r in csv.DictReader(open(w / "truth_blocks" / "truth_events.csv"))]
            if a != t:
                print(n, S.profile(n, truth_sub="truth_blocks", out_name="sim_ops_blocks.csv")[0], flush=True)
    gaps = json.load(open(DATA / f"sim_gaps{B.SUFFIX}.json")) if (DATA / f"sim_gaps{B.SUFFIX}.json").exists() else {}
    allgaps = [x for g in gaps.values() for x in g]
    delay_ns = args.delay * 1000 if args.delay is not None else (statistics.median(allgaps) if allgaps else 10000)
    print(f"scheduling delay: {delay_ns / 1000:.1f} us", flush=True)
    rows, sweep = [], []
    for n in names:
        ops = load_ops(n)
        if ops is None:
            print(n, "no profile", flush=True)
            continue
        events, nb, pol, truth_spec = block_policies(n)
        if len(ops) != len(events):
            print(n, f"profile has {len(ops)} operations, block trace {len(events)}", flush=True)
            continue
        work = None if args.equal else load_work(n, len(events))
        dur = block_durations(ops, nb, work)
        base = None
        for policy in POLICIES:
            if policy not in pol:
                continue
            whole, spec = pol[policy]
            t, rb, wasted = simulate(ops, nb, whole, spec, truth_spec, delay_ns, args.host_sync, dur)
            base = base or t
            rows.append({"name": n, "suite": B.BENCH[n]["suite"], "policy": policy, "ops": len(ops),
                         "durations": "work" if work else "equal",
                         "blocks": sum(nb), "makespan_us": round(t / 1000, 2), "speedup": round(base / t, 4),
                         "rollbacks": rb, "rolled_back_us": round(wasted / 1000, 2),
                         "whole_preds": sum(len(v) for v in whole.values()),
                         "block_preds": sum(len(x) for per in spec.values() for x in per.values())})
        for d in (0, 5, 10, 20, 50, 100, 200):
            for policy in ("hybrid-block", "full"):
                if policy in pol:
                    whole, spec = pol[policy]
                    t, rb, _ = simulate(ops, nb, whole, spec, truth_spec, d * 1000, args.host_sync, dur)
                    sweep.append({"name": n, "policy": policy, "delay_us": d, "makespan_us": round(t / 1000, 2),
                                  "speedup": round(base / t, 4), "rollbacks": rb})
        print(n, " ".join(f"{r['policy']}={r['speedup']:.2f}x/{r['rollbacks']}rb" for r in rows if r["name"] == n),
              flush=True)
    suffix = ("_hostsync" if args.host_sync else "") + ("_equal" if args.equal else "")
    for path, data in ((DATA / f"sim_blocks{B.SUFFIX}{suffix}.csv", rows), (DATA / f"sim_blocks_delay{B.SUFFIX}{suffix}.csv", sweep)):
        if data:
            with open(path, "w", newline="") as f:
                wr = csv.DictWriter(f, list(data[0]))
                wr.writeheader()
                wr.writerows(data)
            print("wrote", path)


if __name__ == "__main__":
    main()
