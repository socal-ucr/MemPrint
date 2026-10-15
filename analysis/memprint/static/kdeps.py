"""Memory dependences between the kernels (and copies) of a CUDA program, from its source.

    python -m memprint.static.kdeps SOURCE[,SOURCE...] OUT_DIR [--argv ARG ...] [-D X] [-I DIR] [--unit 4]

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
import os
import sys
import time

import numpy as np

R, W, RM, WM, RD, WD = 1, 2, 4, 8, 16, 32


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
        for kind, mine, mine_must in (("RAW", r_any, r_must), ("WAW", w_any, w_must)):
            if not mine.any():
                continue
            hit = mine & (st.last >= 0) & (st.last != i)
            if hit.any():
                srcs = st.last[hit]
                must = mine_must[hit]
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
    between matched events and the first has not are added as may. Returns the number of events matched in
    every run."""
    base = trackers[0]
    sig0 = [_signature(e) for e in base.events]
    maps = [align_ends([_signature(e) for e in t.events], sig0) for t in trackers[1:]]
    everywhere = set(range(len(base.events)))
    for m in maps:
        everywhere &= set(m.values())
    for e in base.events:
        if e["index"] not in everywhere:
            e["schedule_dependent"] = 1
    others = []
    for t, m in zip(trackers[1:], maps):
        others.append({(m[k[0]], m[k[1]], k[2], k[3]): v for k, v in t.edges.items() if k[0] in m and k[1] in m})
    for key, val in base.edges.items():
        runs = [o.get(key) for o in others]
        if key[0] in everywhere and key[1] in everywhere and all(r is not None for r in runs):
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
    return len(everywhere)


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
    ap.add_argument("--argv", nargs="*", default=None, help="the command line (argv[0] first)")
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
    source = sources[0] if len(sources) == 1 else unity(sources, os.path.join(args.out_dir, "unity.cu"))
    name = args.name or os.path.splitext(os.path.basename(sources[0]))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    start = time.time()
    runs = []
    for schedule in args.schedules.split(","):
        it, errors, _ = analyze(source, args.defines, args.includes, argv=args.argv, seed=args.seed,
                                unit=args.unit, host_events=not args.no_host, keep_access=args.access,
                                schedule=schedule)
        runs.append(it)
    it = runs[0]
    for e in errors[:20]:
        print("parse error:", e, file=sys.stderr)
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
