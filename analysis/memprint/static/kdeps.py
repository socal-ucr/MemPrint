"""Memory dependences between the kernels (and copies) of a CUDA program, from its source.

    python -m memprint.static.kdeps SOURCE[,SOURCE...] OUT_DIR [-D X] [-I DIR] [--unit 4] [--argv ARG ...]

The CUDA interpreter (interp_cuda) runs the program and records, for every kernel launch, device copy and
memset (and the host code between them), which bytes it read and wrote. Events are folded in program order
into the dependences a later event has on an earlier one through an object's bytes, at `unit`-byte
granularity:

    RAW  the later event reads what the earlier one wrote (last writer of each unit)
    WAW  the later event writes what the earlier one wrote
    WAR  the later event writes what the earlier one read (readers since the last write)

Each dependence is must (both references certain, at known addresses) or may (one of them is perhaps:
under control that depends on data the interpreter did not know, or at an address it could not place, which
covers the whole object); and static or input-dependent: input-dependent if the bytes it goes through were
reached by an address or under a condition that reads memory contents (interp_cuda.DataFacts), or if an
event was issued under host control that does. A static dependence holds for any input of the same sizes; an
input-dependent one holds for the input the program was run with.

Writes OUT_DIR/<name>_events.csv (one row per event: kind, name, source location, grid, block, threads) and
OUT_DIR/<name>_edges.csv (one row per dependence: src, dst, kind, object, units must / may /
input-dependent, certainty, input_dependent, needed: not implied by a path through other dependences, the
edges a scheduler must keep).
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

R, W, RM, WM, RD, WD, WX = 1, 2, 4, 8, 16, 32, 64


class _State:
    """An object's history: the last certain writer of each unit, the uncertain writers and the readers since."""

    def __init__(self, n):
        self.last = np.full(n, -1, dtype=np.int64)  # last must writer
        self.last_dd = np.zeros(n, dtype=bool)
        self.may_writers = []  # [event, mask, dd] since the unit's last must write
        self.readers = []  # [event, mask, must mask, dd] since the unit's last must write

    def grow(self, n):
        if n <= len(self.last):
            return
        extra = n - len(self.last)
        self.last = np.concatenate([self.last, np.full(extra, -1, dtype=np.int64)])
        self.last_dd = np.concatenate([self.last_dd, np.zeros(extra, dtype=bool)])
        for lst in (self.may_writers, self.readers):
            for e in lst:
                for i in range(1, len(e)):
                    e[i] = np.concatenate([e[i], np.zeros(extra, dtype=bool)])


class Tracker:
    """Folds events (interp_cuda.Event) into dependences as they close."""

    def __init__(self, unit=4):
        self.unit = unit
        self.states = {}  # Obj -> _State
        self.objects = {}  # Obj -> its row: name, kind, bytes, site
        self.events = []
        self.edges = {}  # (src, dst, kind, object name) -> [must units, may units, dd units]
        self.access = None  # a list to keep each event's units of device allocations in (keep_access)
        self.schedule_edges = set()  # edges keys whose presence or certainty depends on the schedule (merge)
        self.touch = None  # event index -> {object name: (read, write, n, kill) packed unit bits} (keep_touch)
        self.loop = None  # the rounds that differ between schedules, as a template (merge_schedules)
        self.blocks = None  # kernel event index -> {object name: (blocks, units, bits)} or None (keep_blocks)

    def keep_touch(self):
        """Also keep which units each event read and wrote, certainly or possibly, and which it wrote in any
        legal schedule: certainly, at an address and under conditions that read no memory (merge_schedules)."""
        self.touch = {}

    def keep_blocks(self):
        """Also keep, per kernel event and object, which block referenced which unit (Event.block_marks), with
        the event's bits (R, W, RM, WM, RD, WD, WX); None for an event with a reference to an unknown object."""
        self.blocks = {}

    def _block_rows(self, ev):
        out = {}
        for obj, refs in ev.block_marks.items():                       # interp_cuda.BlockRefs
            self._object(obj)
            b, u, bits = refs.rows()
            out[self.objects[obj]["name"]] = (b, u, bits)
        return out

    def keep_access(self):
        """Also keep, per event and device allocation, the runs of units it read (1) and wrote (2), in
        gpu/nvbit_rw's truth_access.csv format."""
        self.access = []

    def _object(self, obj):
        if obj not in self.objects:
            site = getattr(obj, "site", None) or {}
            loc = site.get("loc")
            name = obj.name                                            # one site run several times (a loop of
            seen = sum(1 for o in self.objects if o.name == obj.name)  # cudaMalloc): d_bucket@30, d_bucket@30#1
            if seen:
                name = f"{name}#{seen}"
            self.objects[obj] = {"name": name, "kind": obj.kind, "bytes": obj.nbytes,
                                 "site": f"{loc[0]}:{loc[1]}" if loc else "",
                                 "alloc": getattr(obj, "alloc_index", "")}

    def register(self, obj):
        """A new device allocation. A reference through a pointer the interpreter could not place may reach
        any of them (Event.unknown), also one no event has named yet."""
        self._object(obj)
        if obj not in self.states:
            self.states[obj] = _State(max(obj.nbytes + self.unit - 1, self.unit) // self.unit)

    def _units(self, ev, obj):
        """uint8 bits per unit of an event's references to obj."""
        u = self.unit
        b = ev.bits.get(obj)
        n_units = max((len(b) if b is not None else 0) + u - 1, obj.nbytes + u - 1, u) // u
        out = np.zeros(n_units, dtype=np.uint8)
        if b is not None:
            pad = np.zeros(n_units * u, dtype=np.uint8)
            pad[: len(b)] = b
            out = np.bitwise_or.reduce(pad.reshape(-1, u), axis=1)
        if obj in ev.all:
            out |= np.uint8(ev.all[obj])
        if ev.unknown:
            out |= np.uint8(ev.unknown & (RM | WM | RD | WD))
        return out

    def _edge(self, src, dst, kind, obj, must, may, dd):
        if src == dst:
            return
        key = (int(src), int(dst), kind, self.objects[obj]["name"])
        e = self.edges.setdefault(key, [0, 0, 0])
        e[0] += int(must)
        e[1] += int(may)
        e[2] += int(dd)

    def add(self, ev):
        if self.blocks is not None and ev.kind == "kernel":
            self.blocks[ev.index] = None if ev.block_unknown or ev.block_marks is None else self._block_rows(ev)
            ev.block_marks = None
        self.events.append({"index": ev.index, "kind": ev.kind, "name": ev.name, "site": ev.site,
                            "grid": "x".join(map(str, ev.grid)) if ev.grid else "",
                            "block": "x".join(map(str, ev.block)) if ev.block else "", "threads": ev.threads,
                            "data_control": int(ev.data_control), "note": ev.note,
                            "schedule_dependent": 0})
        objs = list(ev.bits) + [o for o in ev.all if o not in ev.bits]
        if ev.unknown:                                                 # an unknown object: any object so far,
            device = ev.kind == "kernel"                               # in a kernel any the device can reach
            objs += [o for o in self.states if o not in objs and
                     (not device or o.kind in ("device", "managed", "pinned", "device-global"))]
        for obj in objs:
            self._object(obj)
            bits = self._units(ev, obj)
            st = self.states.get(obj)
            if st is None:
                st = self.states[obj] = _State(len(bits))
            st.grow(len(bits))
            if len(bits) < len(st.last):
                bits = np.concatenate([bits, np.zeros(len(st.last) - len(bits), dtype=np.uint8)])
            self._fold(ev, obj, st, bits)
            if self.touch is not None:
                r_any, w_any = (bits & (R | RM)) != 0, (bits & (W | WM)) != 0
                kill = (bits & (W | WX)) == (W | WX)
                if r_any.any() or w_any.any():
                    self.touch.setdefault(ev.index, {})[self.objects[obj]["name"]] = \
                        (np.packbits(r_any), np.packbits(w_any), len(bits), np.packbits(kill))
            if self.access is not None and getattr(obj, "alloc_index", None) is not None:
                rw = ((bits & (R | RM)) != 0).astype(np.uint8) | (((bits & (W | WM)) != 0).astype(np.uint8) << 1)
                cuts = np.flatnonzero(np.diff(rw)) + 1
                for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(rw)]):
                    if rw[lo]:
                        self.access.append((ev.index, obj.alloc_index, int(lo), int(hi), int(rw[lo])))

    def _fold(self, ev, obj, st, bits):
        i = ev.index
        r_must = (bits & R) != 0
        r_any = (bits & (R | RM)) != 0
        w_must = (bits & W) != 0
        w_any = (bits & (W | WM)) != 0
        dd = (bits & (RD | WD)) != 0
        maybe_since = np.zeros(len(st.last), dtype=bool)               # units an event possibly wrote since
        for _, mask, _ in st.may_writers:                              # their last certain write
            maybe_since |= mask
        for kind, mine, mine_must in (("RAW", r_any, r_must), ("WAW", w_any, w_must)):
            if not mine.any():
                continue
            hit = mine & (st.last >= 0) & (st.last != i)
            if hit.any():
                srcs = st.last[hit]
                must = (mine_must & ~maybe_since)[hit]                  # certain only if nothing possibly wrote
                                                                       # the unit in between
                ddh = dd[hit] | st.last_dd[hit]
                for s in np.unique(srcs):
                    sel = srcs == s
                    self._edge(s, i, kind, obj, np.sum(must[sel]), np.sum(~must[sel]), np.sum(ddh[sel]))
            for e, mask, mdd in st.may_writers:
                o = mine & mask
                if o.any():
                    self._edge(e, i, kind, obj, 0, np.sum(o), np.sum(o & (dd | mdd)))
        if w_any.any():
            for e, mask, mmust, mdd in st.readers:
                o = w_any & mask
                if o.any():
                    must = o & mmust & w_must
                    self._edge(e, i, "WAR", obj, np.sum(must), np.sum(o & ~must), np.sum(o & (dd | mdd)))
        # this event's own references become the history
        if w_must.any():
            st.last[w_must] = i
            st.last_dd[w_must] = dd[w_must]
            keep = ~w_must
            st.may_writers = [[e, m & keep, d] for e, m, d in st.may_writers if (m & keep).any()]
            st.readers = [[e, m & keep, mm & keep, d] for e, m, mm, d in st.readers if (m & keep).any()]
        w_may = w_any & ~w_must
        if w_may.any():
            st.may_writers = [x for x in st.may_writers if x[0] != i] + [[i, w_may, dd.copy()]]
        reads = r_any & ~w_must                                        # read then overwritten here: no later WAR
        if reads.any():
            st.readers.append([i, reads, r_must & reads, dd.copy()])

    # ------------------------------------------------------------ results

    def edge_rows(self):
        names = {e["index"]: e["name"] for e in self.events}
        control = {e["index"]: e["data_control"] for e in self.events}
        rows = []
        for (s, d, kind, obj), (must, may, dd) in sorted(self.edges.items()):
            rows.append({"src": s, "dst": d, "src_name": names.get(s, ""), "dst_name": names.get(d, ""),
                         "kind": kind, "object": obj, "units_must": must, "units_may": may, "units_dd": dd,
                         "bytes": (must + may) * self.unit, "certainty": "must" if must else "may",
                         "input_dependent": int(bool(dd) or bool(control.get(s)) or bool(control.get(d))),
                         "schedule_dependent": int((s, d, kind, obj) in self.schedule_edges)})
        needed = reduce_edges({(r["src"], r["dst"]) for r in rows})
        for r in rows:
            r["needed"] = int((r["src"], r["dst"]) in needed)
        return rows


def _signature(e):
    return (e["kind"], e["name"], e["site"])


def align_ends(a, b, same=lambda x, y: x == y):
    """Match two event sequences of one program as a loop with a different trip count leaves them: the common
    start forward, then the common end backward. Returns {index in a: index in b}; the events between (the
    rounds one run has and the other has not) stay unmatched."""
    n = min(len(a), len(b))
    head = next((i for i in range(n) if not same(a[i], b[i])), n)
    out = {i: i for i in range(head)}
    k = 1
    while k <= n - head and same(a[-k], b[-k]):
        out[len(a) - k] = len(b) - k
        k += 1
    return out


def merge_schedules(trackers):
    """Fold the runs of one program under several thread schedules (interp_cuda.SCHEDULES) into the first.
    Events are matched by align_ends (a loop the host repeats until the device says so can run a different
    number of rounds). Events of the first run some other run has no match for, and dependences that are not
    in every run, or not must in every run, become may and schedule-dependent; dependences another run has
    between matched events and the first has not are added as may. Where the runs issue different events
    (the rounds in the middle differ), a dependence from before the point where they part to after the
    differing rounds is may as well, even if every run has it: which event last wrote a unit, or read it, is
    decided by those rounds, and other legal orders than the ones run can decide it differently. Returns the
    number of events matched in every run."""
    base = trackers[0]
    sig0 = [_signature(e) for e in base.events]
    maps = [align_ends([_signature(e) for e in t.events], sig0) for t in trackers[1:]]
    everywhere = set(range(len(base.events)))
    for m in maps:
        everywhere &= set(m.values())
    for e in base.events:
        if e["index"] not in everywhere:
            e["schedule_dependent"] = 1
    spans = []                                     # (events before the parting, first event after the middle)
    for t, m in zip(trackers[1:], maps):
        head = 0
        while m.get(head) == head and head < len(base.events):
            head += 1
        if head == len(base.events) == len(t.events):
            continue
        tail = min((b for o, b in m.items() if o >= head), default=len(base.events))
        spans.append((head, tail))
    crosses = lambda s_, d_: any(s_ < h and d_ >= tl for h, tl in spans)  # noqa: E731
    others = []
    for t, m in zip(trackers[1:], maps):
        others.append({(m[k[0]], m[k[1]], k[2], k[3]): v for k, v in t.edges.items() if k[0] in m and k[1] in m})
    for key, val in base.edges.items():
        runs = [o.get(key) for o in others]
        if key[0] in everywhere and key[1] in everywhere and all(r is not None for r in runs) and \
                not crosses(key[0], key[1]):
            must = min([val[0]] + [r[0] for r in runs])
            total = max([val[0] + val[1]] + [r[0] + r[1] for r in runs])
            if must != val[0] or total != val[0] + val[1]:
                base.schedule_edges.add(key)
            val[0], val[1] = must, total - must
        else:
            val[1] += val[0]
            val[0] = 0
            base.schedule_edges.add(key)
    for o in others:
        for key, val in o.items():
            if key not in base.edges:
                base.edges[key] = [0, val[0] + val[1], val[2]]
                base.schedule_edges.add(key)
    if all(t.touch is not None for t in trackers):
        _union_dependences(base, trackers, maps)
    return len(everywhere)


# ---------------------------------------------------------------- what other schedules could do

def _unpack(entry):
    r, w, n = entry[:3]
    return np.unpackbits(r)[:n].astype(bool), np.unpackbits(w)[:n].astype(bool)


def _kills(entries):
    """Per object name, the units every one of several events (rounds of a loop, in every run) wrote in any
    legal schedule (Tracker.keep_touch): a dependence through such a unit cannot reach past such an event."""
    out = None
    for touch in entries:
        got = {}
        for name, entry in (touch or {}).items():
            if len(entry) > 3:
                got[name] = np.unpackbits(entry[3])[:entry[2]].astype(bool)
        if out is None:
            out = got
            continue
        for name in list(out):
            if name not in got:
                del out[name]
                continue
            n = min(len(out[name]), len(got[name]))
            out[name] = out[name][:n] & got[name][:n]
    return out or {}


def _without(units, kill_sets):
    """A unit set minus the units of every kill set (_kills) that names the same object."""
    name, x = units
    for k in kill_sets:
        if name in k:
            n = min(len(x), len(k[name]))
            x = x.copy()
            x[:n] &= ~k[name][:n]
    return x


def _union_touch(entries):
    """The union of several events' (read, write) unit sets per object name."""
    out = {}
    for touch in entries:
        for name, entry in (touch or {}).items():
            r, w = _unpack(entry)
            if name in out:
                r0, w0 = out[name]
                n = max(len(r0), len(r))
                out[name] = (np.pad(r0, (0, n - len(r0))) | np.pad(r, (0, n - len(r))),
                             np.pad(w0, (0, n - len(w0))) | np.pad(w, (0, n - len(w))))
            else:
                out[name] = (r, w)
    return out


def _conflicts(a, b, between=()):
    """[(kind, object, units)] for an earlier event touching a and a later one touching b (unit sets), with no
    unit some event in between wrote in any legal schedule (between: _kills of those events)."""
    out = []
    for name in a.keys() & b.keys():
        (ra, wa), (rb, wb) = a[name], b[name]
        n = min(len(ra), len(rb))
        for kind, x, y in (("RAW", wa, rb), ("WAW", wa, wb), ("WAR", ra, wb)):
            z = x[:n] & y[:n]
            if between and z.any():
                z = _without((name, z), between)
            k = int(np.count_nonzero(z))
            if k:
                out.append((kind, name, k))
    return out


def _find_loop(sig, a, b):
    """(start, end, period) of the repeating run of signatures that contains [a, b) (b > a), or None."""
    n = len(sig)
    for p in range(1, b - a + 1):
        lo, hi = a, b
        while lo - 1 >= 0 and lo - 1 + p < n and sig[lo - 1] == sig[lo - 1 + p]:
            lo -= 1
        while hi < n and hi - p >= 0 and sig[hi] == sig[hi - p]:
            hi += 1
        if all(sig[i] == sig[i + p] for i in range(lo, hi - p)) and (hi - lo) % p == 0 and hi - lo >= 2 * p:
            return lo, hi, p
    return None


def _union_dependences(base, trackers, maps):
    """Possible dependences that other legal schedules than the ones run could have (merge_schedules).

    Where the runs issue different events (a loop the host repeats until the device says so ran a different
    number of rounds), the rounds are folded into a template: per position in a round (phase), the units any
    run touched in any round; and every conflict, between two phases (in one round, or from a round to any
    later one), between an event before the loop and a phase, a phase and an event after the loop, and an
    event before and one after the loop, is a possible dependence for every number of rounds (base.loop,
    written as NAME_loop.json; kdeps_score instantiates it for the traced run). Where the runs issue the same
    events but an event touched different units in different runs, every conflict between it and another
    event, on the union of what each touched, is a possible dependence."""
    sigs = [[_signature(e) for e in t.events] for t in trackers]
    longest = max(range(len(trackers)), key=lambda k: len(trackers[k].events))
    sl, nl = sigs[longest], len(sigs[longest])
    head = min(next((i for i in range(min(nl, len(sg))) if sg[i] != sl[i]), min(nl, len(sg))) for sg in sigs)
    tail = min(next((i for i in range(min(nl, len(sg)) - head) if sg[-1 - i] != sl[-1 - i]),
                    min(nl, len(sg)) - head) for sg in sigs)
    nb = len(base.events)

    def add(key, units):
        if key[0] == key[1]:
            return
        e = base.edges.get(key)
        if e is None:
            base.edges[key] = [0, units, 0]
            base.schedule_edges.add(key)

    loop = None
    if any(len(sg) != nl for sg in sigs) and nl - tail > head:
        loop = _find_loop(sl, head, nl - tail)
        if loop is not None:
            a, b, p = loop
            post = nl - b
            for sg in sigs:                                            # every run: prefix, rounds, suffix fit
                end = len(sg) - post
                if (end - a) % p or end < a or sg[:a] != sl[:a] or sg[end:] != sl[b:] or \
                        any(sg[i] != sl[a + (i - a) % p] for i in range(a, end)):
                    loop = None
                    break
    if loop is not None:
        a, b, p = loop
        post = nl - b
        pre_t = [_union_touch([t.touch.get(i) for t in trackers]) for i in range(a)]
        post_t = [_union_touch([t.touch.get(len(t.events) - q) for t in trackers]) for q in range(1, post + 1)]
        phase_t = [_union_touch([t.touch.get(i) for t in trackers for i in range(a + ph, len(t.events) - post, p)])
                   for ph in range(p)]
        kill = [_kills([t.touch.get(i) for t in trackers for i in range(a + ph, len(t.events) - post, p)])
                for ph in range(p)]                                    # what each phase overwrites every round
        rows = []

        def tiers(src, dst, near, far, labels):
            """Rows for the conflicts not killed by the events between (near: the nearest pair of rounds,
            far: any farther), labelled labels[0] when farther rounds have them too, else labels[1]."""
            out = []
            nears = {(k, o): u for k, o, u in _conflicts(src, dst, near)}
            fars = {(k, o): u for k, o, u in _conflicts(src, dst, far)}
            for (k, o), u in nears.items():
                out.append((labels[0] if (k, o) in fars else labels[1], k, o, u))
            return out

        for ps in range(p):
            for pd in range(p):
                if ps < pd:                                            # within a round
                    rows += [(("p", ps), ("p", pd), "0", k, o, u)
                             for k, o, u in _conflicts(phase_t[ps], phase_t[pd], kill[ps + 1: pd])]
                rows += [(("p", ps), ("p", pd), dl, k, o, u)          # to the next round ("1") or any later ("+")
                         for dl, k, o, u in tiers(phase_t[ps], phase_t[pd], kill[ps + 1:] + kill[:pd], kill,
                                                  ("+", "1"))]
        for i in range(a):
            for pd in range(p):                                        # to every round ("") or the first ("first")
                rows += [(("e", i), ("p", pd), dl, k, o, u)
                         for dl, k, o, u in tiers(pre_t[i], phase_t[pd], kill[:pd], kill, ("", "first"))]
            for q in range(1, post + 1):                               # past every round ("") or none ("none")
                rows += [(("e", i), ("q", q), dl, k, o, u)
                         for dl, k, o, u in tiers(pre_t[i], post_t[q - 1], (), kill, ("", "none"))]
        for ps in range(p):
            for q in range(1, post + 1):                               # from every round ("") or the last ("last")
                rows += [(("p", ps), ("q", q), dl, k, o, u)
                         for dl, k, o, u in tiers(phase_t[ps], post_t[q - 1], kill[ps + 1:], kill, ("", "last"))]
        base.loop = {"start": a, "period": p, "post": post,
                     "phases": [{"kind": base.events[a + ph]["kind"], "name": base.events[a + ph]["name"]}
                                for ph in range(p)],
                     "rows": [{"src": f"{s_[0]}{s_[1]}", "dst": f"{d_[0]}{d_[1]}", "delta": dl, "kind": k,
                               "object": o, "units": u} for s_, d_, dl, k, o, u in rows]}
        for key, units in instantiate_loop(base.loop, nb).items():    # the template on the first run's rounds
            add(key, units)
        # a dependence of a round's event through an object whose units that phase touched differently in
        # different rounds or runs is not certain: another order can touch other units
        def same(entries):
            entries = [e for e in entries]
            return all((x is None) == (entries[0] is None) and
                       (x is None or all(np.array_equal(x[j], entries[0][j]) for j in range(3)))
                       for x in entries)
        varies = set()
        for ph in range(p):
            insts = [t.touch.get(i) or {} for t in trackers for i in range(a + ph, len(t.events) - post, p)]
            for o in set().union(*[set(x) for x in insts]):
                if not same([x.get(o) for x in insts]):
                    varies.add((ph, o))
        phase_of = lambda i: (i - a) % p if a <= i < nb - post else None  # noqa: E731
        for key, val in base.edges.items():
            s_, d_, _, o = key
            if val[0] and any(phase_of(i) is not None and (phase_of(i), o) in varies for i in (s_, d_)):
                val[1] += val[0]
                val[0] = 0
                base.schedule_edges.add(key)
        for e in base.events[a: nb - post]:
            e["schedule_dependent"] = 1
    # events every run issues, that touched different units in different runs
    matched = [i for i in range(nb) if all(i in m.values() for m in maps)]
    inv = [{v: k for k, v in m.items()} for m in maps]
    in_loop = set(range(loop[0], nb - (nl - loop[1]))) if loop is not None else set()
    union, unstable = {}, set()
    for i in matched:
        if i in in_loop:
            continue
        own = [_union_touch([base.touch.get(i)])] + \
            [_union_touch([t.touch.get(iv[i])]) for t, iv in zip(trackers[1:], inv)]
        union[i] = _union_touch([{n: (np.packbits(r), np.packbits(w), len(r)) for n, (r, w) in o.items()}
                                 for o in own])
        if any(o.keys() != own[0].keys() or any(not (np.array_equal(o[n][0], own[0][n][0]) and
                                                     np.array_equal(o[n][1], own[0][n][1])) for n in o)
               for o in own[1:]):
            unstable.add(i)
    kills = {i: _kills([base.touch.get(i)] + [t.touch.get(iv[i]) for t, iv in zip(trackers[1:], inv)])
             for i in union}                                       # what each wrote in every run, in any order
    for i in sorted(union):
        for j in sorted(union):
            if i < j and (i in unstable or j in unstable):          # not through a unit an event between
                between = [kills[k] for k in range(i + 1, j) if k in kills and kills[k]]   # overwrote
                for k, o, u in _conflicts(union[i], union[j], between):
                    add((i, j, k, o), u)
    for i in unstable:
        base.events[i]["schedule_dependent"] = 1


def instantiate_loop(loop, n_events):
    """A loop template (Tracker.loop) as dependences {(src, dst, kind, object): units} between the events of a
    run with n_events events, or {} if they do not fit it."""
    a, p, post = loop["start"], loop["period"], loop["post"]
    end = n_events - post
    if end < a or (end - a) % p:
        return {}
    rounds = (end - a) // p

    def where(tag, r=None):
        k, x = tag[0], int(tag[1:])
        return x if k == "e" else n_events - x if k == "q" else a + r * p + x

    out = {}
    for row in loop["rows"]:
        s_, d_ = row["src"], row["dst"]
        pairs = []
        dl = row["delta"]                                              # see _union_dependences
        if s_[0] == "p" and d_[0] == "p":
            for r in range(rounds):
                if dl == "0":
                    pairs.append((where(s_, r), where(d_, r)))
                elif dl == "1":
                    pairs += [(where(s_, r), where(d_, r + 1))] if r + 1 < rounds else []
                else:
                    pairs += [(where(s_, r), where(d_, r2)) for r2 in range(r + 1, rounds)]
        elif s_[0] == "p":
            pairs = [(where(s_, r), where(d_)) for r in (range(rounds) if dl != "last" else range(rounds)[-1:])]
        elif d_[0] == "p":
            pairs = [(where(s_), where(d_, r)) for r in (range(rounds) if dl != "first" else range(rounds)[:1])]
        elif dl != "none" or rounds == 0:
            pairs = [(where(s_), where(d_))]
        for x, y in pairs:
            key = (x, y, row["kind"], row["object"])
            out[key] = max(out.get(key, 0), row["units"])
    return out


def reduce_edges(pairs):
    """The transitive reduction of a dependence DAG whose nodes are numbered in program order: the pairs
    (u, v) with no other path from u to v."""
    preds = {}
    for u, v in pairs:
        preds.setdefault(v, set()).add(u)
    anc = {}  # node -> bitset of its ancestors
    keep = set()
    for v in sorted({v for _, v in pairs} | {u for u, _ in pairs}):
        reach = 0
        for u in sorted(preds.get(v, ()), reverse=True):               # nearest first
            if not (reach >> u) & 1:
                keep.add((u, v))
            reach |= anc.get(u, 0) | (1 << u)
        anc[v] = reach
    return keep


def write(out_dir, name, tracker):
    os.makedirs(out_dir, exist_ok=True)
    if tracker.blocks is not None:                                     # NAME_blocks.npz: per kernel event and
        arrays = {"unknown": np.array([e for e, v in tracker.blocks.items() if v is None], dtype=np.int64)}
        for e, objs in tracker.blocks.items():                         # object, which block referenced which
            for obj, (b, u, bits) in (objs or {}).items():             # unit, with the event's bits
                arrays[f"{e}/{obj}/block"], arrays[f"{e}/{obj}/unit"], arrays[f"{e}/{obj}/bits"] = b, u, bits
        np.savez_compressed(os.path.join(out_dir, f"{name}_blocks.npz"), **arrays)
    ev_path = os.path.join(out_dir, f"{name}_events.csv")
    with open(ev_path, "w", newline="") as f:
        cols = ["index", "kind", "name", "site", "grid", "block", "threads", "data_control", "note",
                "schedule_dependent"]
        wr = csv.DictWriter(f, cols)
        wr.writeheader()
        wr.writerows(tracker.events)
    rows = tracker.edge_rows()
    edge_path = os.path.join(out_dir, f"{name}_edges.csv")
    with open(edge_path, "w", newline="") as f:
        cols = ["src", "dst", "src_name", "dst_name", "kind", "object", "units_must", "units_may", "units_dd",
                "bytes", "certainty", "input_dependent", "needed", "schedule_dependent"]
        wr = csv.DictWriter(f, cols)
        wr.writeheader()
        wr.writerows(rows)
    obj_path = os.path.join(out_dir, f"{name}_objects.csv")
    with open(obj_path, "w", newline="") as f:
        wr = csv.DictWriter(f, ["name", "kind", "bytes", "site", "alloc"])
        wr.writeheader()
        wr.writerows(tracker.objects.values())
    if tracker.loop is not None:
        with open(os.path.join(out_dir, f"{name}_loop.json"), "w") as f:
            json.dump(tracker.loop, f)
    if tracker.access is not None:
        with open(os.path.join(out_dir, f"{name}_access.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["event", "alloc", "unit_lo", "unit_hi", "bits"])
            wr.writerows(tracker.access)
    return ev_path, edge_path, rows


def unity(sources, out):
    """Several sources compiled as one translation unit (a file that includes them all)."""
    with open(out, "w") as f:
        for s in sources:
            f.write(f'#include "{os.path.abspath(s)}"\n')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", help="a .cu file, or several separated by commas (one translation unit)")
    ap.add_argument("out_dir")
    ap.add_argument("--name", default=None)
    ap.add_argument("--argv", nargs=argparse.REMAINDER, default=None,
                    help="the command line, argv[0] first (last option: everything after it is the program's, "
                         "--options included)")
    ap.add_argument("-D", dest="defines", action="append", default=[])
    ap.add_argument("-I", dest="includes", action="append", default=[])
    ap.add_argument("--unit", type=int, default=4, help="bytes per dependence unit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-host", action="store_true", help="no host segments: only device operations")
    ap.add_argument("--schedules", default="lockstep",
                    help="thread schedules to run each launch under, comma-separated (lockstep, blocks, reverse, "
                         "random): dependences not in every one become may (schedule_dependent)")
    ap.add_argument("--access", action="store_true",
                    help="also write NAME_access.csv: each event's units of device allocations (truth format)")
    args = ap.parse_args(argv)

    from .interp_cuda import analyze

    sources = args.source.split(",")
    os.makedirs(args.out_dir, exist_ok=True)                       # the unity file goes there
    source = sources[0] if len(sources) == 1 else unity(sources, os.path.join(args.out_dir, "unity.cu"))
    name = args.name or os.path.splitext(os.path.basename(sources[0]))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    start = time.time()
    runs = []
    several = len(args.schedules.split(",")) > 1
    racy = None                                    # after a lockstep run: the kernels other orders reorder
    for schedule in args.schedules.split(","):
        it, errors, _ = analyze(source, args.defines, args.includes, argv=args.argv, seed=args.seed,
                                unit=args.unit, host_events=not args.no_host, keep_access=args.access,
                                schedule=schedule, keep_touch=several, racy=racy)
        if schedule == "lockstep":
            racy = set(it.racy_seen)
        runs.append(it)
    it = runs[0]
    for e in errors[:20]:
        print("parse error:", e, file=sys.stderr)
    unmodelled = {}
    for r in runs:
        for k, v in getattr(r, "unmodelled", {}).items():
            unmodelled[k] = max(unmodelled.get(k, 0), v)
    if errors or unmodelled:
        print(f"{name}: WARNING the host code was not fully understood ({len(errors)} parse errors; library "
              f"calls not modelled: " + (", ".join(f"{k} x{v}" for k, v in sorted(unmodelled.items(),
                                                                                key=lambda kv: -kv[1])[:8]) or "none")
              + "). Values the program computes from them are unknown or wrong, and the operations it issues may "
              "be too; the hybrid interpreter (memprint.static.hybrid) runs the host code natively.", file=sys.stderr)
    oob = {}
    for r in runs:
        for k, v in r.oob.items():
            oob[k] = max(oob.get(k, 0), v)
    if oob:
        print(f"{len(oob)} places reference past the end of a device buffer (taken as possible references to every "
              f"buffer): " + ", ".join(f"{k} x{v}" for k, v in sorted(oob.items(), key=lambda kv: -kv[1])[:5]),
              file=sys.stderr)
    if len(runs) > 1:
        matched = merge_schedules([r.tracker for r in runs])
        counts = ", ".join(f"{s} {len(r.tracker.events)}" for s, r in zip(args.schedules.split(","), runs))
        print(f"{name}: events per schedule: {counts}; {matched} of the first run's matched in all")
    ev_path, edge_path, rows = write(args.out_dir, name, it.tracker)
    kernels = [e for e in it.tracker.events if e["kind"] != "host"]
    print(f"{name}: {len(it.tracker.events)} events ({len(kernels)} device), {len(rows)} dependences "
          f"({sum(r['needed'] for r in rows)} needed, {sum(r['certainty'] == 'may' for r in rows)} may, "
          f"{sum(r['input_dependent'] for r in rows)} input-dependent) in {time.time() - start:.1f} s")
    print(f"  {ev_path}\n  {edge_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
