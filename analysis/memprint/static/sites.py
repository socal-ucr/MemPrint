"""Per-site profiles from the static interpreter, in the Pin tool's -sites format (no run of the program).

    python -m memprint.static.sites SOURCE OUT_DIR [--name N] [--argv ...] [--sitemap sitemap.txt]
            [--gap KERNEL SCALE] [--min 65536]

The interpreter records, for every heap block, the source location of its allocation call (a malloc-family
call or a new expression) and of the calls that entered the functions above it (Frame.call, innermost
first), and for every std::vector buffer (the containers arena) the call that made it (the constructor,
push_back, resize) and the vector's earlier buffer (a growth chain). From the references it charged, a
block's touched bytes, high water (the highest byte touched, plus one), touched pages and page runs follow,
as the Pin tool measures them on a run. This module writes them as OUT_DIR/Static_<name>_blocks.csv and
_capacity.csv, which memprint.sites fits into reservations like Pin profiles.

Keys. A block is keyed "src:<file>:<line>" with the calls above it as its context ("src:..|src:.."). With
--sitemap (the MemPrintResize pass's site map of the build that will run, id -> the allocation call's
location and the locations it was inlined at), each block gets the pass's id instead: the site whose
location chain is the longest prefix of the block's (locations in the C++ library's headers, which the
interpreter models rather than runs, are skipped), so the profile is keyed as a Pin profile of the pass
build would be ("id:0x...", and "cap:0x<growth call>" for container capacities). Blocks no site matches
keep their src keys (libmpr will not find them; the module reports them).

Limits: sizes are what the interpreter evaluates (an unknown size is 0 and the block is left out); a
vector's buffer is never freed except by growth; realloc is a new block (no copy, no free).
"""

import argparse
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

PAGE = 4096
SYSTEM = re.compile(r"/c\+\+/|/include/c\+\+|/bits/|/ext/")   # libstdc++ / libc++ header paths
VECTOR_OPS = {  # interpreter call -> libstdc++ functions on the pass's chain for that buffer
    "push_back": ("_M_realloc_insert", "emplace_back", "push_back"),
    "emplace_back": ("_M_realloc_insert", "emplace_back"),
    "resize": ("_M_default_append", "_M_fill_insert", "resize"),
    "reserve": ("reserve",),
    "insert": ("_M_range_insert", "_M_fill_insert", "_M_realloc_insert", "insert"),
    "assign": ("_M_fill_assign", "_M_assign_aux", "assign"),
}


def _key(loc):
    return f"src:{loc[0]}:{loc[1]}"


def _runs(cover, size):
    """Touched pages and page runs ("first:count;...") of a block's per-byte reference counts."""
    pages = -(-size // PAGE)
    if not len(cover) or pages == 0:
        return 0, ""
    c = np.zeros(pages * PAGE)
    n = min(len(cover), len(c))
    c[:n] = cover[:n]
    touched = c.reshape(pages, PAGE).max(axis=1) > 0
    idx = np.flatnonzero(touched)
    if not len(idx):
        return 0, ""
    breaks = np.flatnonzero(np.diff(idx) > 1)
    firsts = np.concatenate([[idx[0]], idx[breaks + 1]])
    lasts = np.concatenate([idx[breaks], [idx[-1]]])
    return int(touched.sum()), ";".join(f"{a}:{b - a + 1}" for a, b in zip(firsts, lasts))


def _measure(cover, size):
    cover = np.asarray(cover[:size])
    hit = np.flatnonzero(cover > 0)
    hw = int(hit[-1]) + 1 if len(hit) else 0
    pt, runs = _runs(cover, size)
    return int(len(hit)), hw, pt, runs


def blocks(interp, min_size=0):
    """One row per heap block and vector buffer, with its static site, as the Pin tool's blocks.csv."""
    times = {}
    for (event, obj), t in zip(interp.heap_events, interp.heap_events.times):
        times.setdefault((event, id(obj)), t)
    rows = []
    for obj in interp.objects:
        site = getattr(obj, "site", None)
        if obj.kind != "heap" or site is None or site["loc"] is None or obj.nbytes < max(min_size, 1):
            continue
        touched, hw, pt, runs = _measure(obj.coverage(), obj.nbytes)
        rows.append(dict(site=_key(site["loc"]), kind=obj.alloc_kind, size=obj.nbytes,
                         born=times.get(("new", id(obj)), 0), died=times.get(("free", id(obj)), interp.total),
                         touched=touched, high_water=hw, pages_touched=pt, pages=-(-obj.nbytes // PAGE),
                         alloc_touched=0, grown_from=0, moved=0,
                         context="|".join(_key(c) for c in site["context"]), runs=runs, op="", chain=-1))
    arena = getattr(interp, "_arena_obj", None)
    if arena is not None:
        cover = arena.coverage()
        for b in getattr(interp, "arena_buffers", []):
            site = b["site"]
            if site["loc"] is None or b["nbytes"] < max(min_size, 1):
                continue
            touched, hw, pt, runs = _measure(cover[b["start"]:b["start"] + b["nbytes"]], b["nbytes"])
            rows.append(dict(site=_key(site["loc"]), kind="vector", size=b["nbytes"], born=b["born"],
                             died=b["died"] if b["died"] is not None else interp.total, touched=touched,
                             high_water=hw, pages_touched=pt, pages=-(-b["nbytes"] // PAGE), alloc_touched=0,
                             grown_from=b["grown_from"], moved=min(b["grown_from"], b["nbytes"]),
                             context="|".join(_key(c) for c in site["context"]), runs=runs,
                             op=b.get("op", ""), chain=b["chain"], elem=b["esize"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- the pass's site map

def read_sitemap(path):
    """{id: (kind, [(file base name, line, function), ...])}, the allocation call first."""
    out = {}
    for line in open(path):
        parts = line.rstrip("\n").split(" ", 2)
        if len(parts) < 3 or not parts[0].startswith("0x"):
            continue
        sid, kind, rest = parts
        if kind == "capacity":                                         # "capacity elem=N kind=K <chain>"
            m = re.match(r"(elem=\S+ kind=\S+) (.*)", rest)
            kind, rest = f"capacity {m.group(1)}", m.group(2)
        elif kind == "thunk":
            kind, _, rest = (kind + " " + rest).partition(" ")[0], None, rest.split(" ", 1)[1]
        chain = []
        for entry in rest.split(" < "):
            loc, _, fn = entry.partition(" ")
            f, line_, _col = loc.rsplit(":", 2)
            chain.append((f, int(line_), fn))
        out[int(sid, 16)] = (kind, chain)
    return out


class Translator:
    """Static site (location, context) -> the pass's site id, by the longest matching location chain."""

    def __init__(self, sitemap):
        self.alloc, self.growth, self.capsite = [], [], []
        for sid, (kind, chain) in sitemap.items():
            user = [(os.path.basename(f), l) for f, l, _ in chain if not SYSTEM.search(f)]
            fns = {fn.split("<")[0] for _, _, fn in chain}
            system_first = bool(chain) and SYSTEM.search(chain[0][0]) is not None
            entry = (sid, user, fns, system_first, kind)
            if kind in ("malloc", "calloc", "realloc", "aligned_alloc", "posix_memalign", "memalign", "valloc") \
                    or kind.startswith("_Zn"):
                self.alloc.append(entry)
            elif kind == "growth":
                self.growth.append(entry)
            elif kind == "capacity-site":
                self.capsite.append(entry)

    @staticmethod
    def _best(entries, seq, op=None, vector=None, kind=None):
        best, best_len = None, 0
        for sid, user, fns, system_first, k in entries:
            if kind is not None and not (k == kind or (kind == "new" and k.startswith("_Zn"))):
                continue
            if vector is not None and system_first != vector:          # a vector's buffer comes from the library
                continue
            if op and not fns & set(VECTOR_OPS.get(op, (op,))):
                continue
            n = len(user)
            if 0 < n <= len(seq) and seq[:n] == user and n > best_len:
                best, best_len = sid, n
        return best

    def site(self, row):
        seq = [_parse(row["site"])] + [_parse(c) for c in str(row.get("context") or "").split("|") if c]
        vector = row["kind"] == "vector"
        if vector:
            sid = self._best(self.alloc, seq, row.get("op"), True)
            if sid is None:
                sid = self._best(self.alloc, seq, None, True)
        else:                                                          # the same allocation function, else any
            sid = self._best(self.alloc, seq, kind=row["kind"])
            if sid is None:
                sid = self._best(self.alloc, seq)
        return f"id:0x{sid:x}" if sid is not None else None

    def capacity(self, loc, context, op):
        seq = [_parse(loc)] + [_parse(c) for c in context.split("|") if c]
        sid = self._best(self.growth, seq, op)
        if sid is not None:
            return f"cap:0x{sid:x}"
        sid = self._best(self.capsite, seq, op)
        return f"id:0x{sid:x}" if sid is not None else None


def _parse(key):
    f, line = key[len("src:"):].rsplit(":", 1)
    return (f, int(line))


def capacity_rows(b, min_size, translate=None):
    """The Pin tool's *_capacity.csv rows from the vector buffers' growth chains (final buffers)."""
    v = b[b.kind == "vector"]
    if v.empty:
        return pd.DataFrame(columns=["key", "elem", "kind", "chains", "small_chains", "small_max_size",
                                     "final_sizes", "final_high_waters"])
    final = v.sort_values("born").groupby("chain").tail(1)
    first = v.sort_values("born").groupby("chain").head(1).set_index("chain")
    agg = defaultdict(lambda: dict(elem=0, chains=0, small=0, small_max=0, sizes=[], hws=[]))
    for r in final.itertuples():
        # the chain's key: the call that grew it (its last buffer's), as the pass's growth context
        key = translate(r.site, r.context, r.op) if translate else r.site
        if key is None:
            continue
        if len(v[v.chain == r.chain]) < 2 and first.loc[r.chain].op not in ("push_back", "emplace_back", "resize",
                                                                            "insert", "reserve"):
            continue                                                   # never grew: no capacity to reserve
        a = agg[key]
        a["elem"], a["chains"] = r.elem, a["chains"] + 1
        if r.size < min_size:
            a["small"] += 1
            a["small_max"] = max(a["small_max"], r.size)
        else:
            a["sizes"].append(r.size)
            a["hws"].append(r.high_water)
    return pd.DataFrame([dict(key=k, elem=int(a["elem"]), kind=1, chains=a["chains"], small_chains=a["small"],
                              small_max_size=a["small_max"], final_sizes=";".join(map(str, a["sizes"])),
                              final_high_waters=";".join(map(str, a["hws"]))) for k, a in agg.items()])


COLUMNS = ["site", "kind", "size", "born", "died", "touched", "high_water", "pages_touched", "pages",
           "alloc_touched", "grown_from", "moved", "context", "runs"]


def write(interp, out_dir, name, sitemap=None, min_size=0):
    """Write Static_<name>_blocks.csv and _capacity.csv; returns (blocks, unmatched site keys)."""
    os.makedirs(out_dir, exist_ok=True)
    b = blocks(interp, min_size=0)
    unmatched = set()
    tr = Translator(read_sitemap(sitemap)) if sitemap else None
    if b.empty:
        b = pd.DataFrame(columns=COLUMNS + ["op", "chain", "elem"])
    caps = capacity_rows(b, max(min_size, 65536), tr.capacity if tr else None)
    if tr is not None and not b.empty:
        keys = []
        for r in b.to_dict("records"):
            k = tr.site(r)
            if k is None:
                unmatched.add(r["site"] + (f"@{r['context']}" if r["context"] else ""))
            keys.append(k or r["site"])
        b["source"] = b["site"] + b["context"].map(lambda c: f"|{c}" if c else "")
        b["site"] = keys
        # the pass's id already holds the calling context it inlined; libmpr takes any further frames as
        # return addresses (backtrace), which a static profile does not know, so a context does not split
        b["context"] = ""
    prefix = os.path.join(out_dir, f"Static_{name}")
    b[b["size"] >= min_size].to_csv(prefix + "_blocks.csv", index=False,
                                    columns=COLUMNS + [c for c in ("source", "op") if c in b])
    caps.to_csv(prefix + "_capacity.csv", index=False)
    return b, unmatched


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source")
    ap.add_argument("out_dir")
    ap.add_argument("--name", default=None)
    ap.add_argument("--argv", nargs="*", default=None, help="the command line (argv[0] first)")
    ap.add_argument("--gap", nargs=2, metavar=("KERNEL", "SCALE"), help="a GAP kernel (gap_pr, ...) at a scale")
    ap.add_argument("-D", dest="defines", action="append", default=[])
    ap.add_argument("-I", dest="includes", action="append", default=[])
    ap.add_argument("--sitemap")
    ap.add_argument("--min", type=int, default=0)
    a = ap.parse_args(argv)
    from . import interp_cpp

    name = a.name or os.path.splitext(os.path.basename(a.source))[0]
    stubs = None
    if a.gap:
        from .runs import HEAP_TOP_AT_START, gap_cli, gap_source
        source, uniform = gap_source(a.gap[0], a.source)
        stubs = {"__CL__": gap_cli(int(a.gap[1]), uniform)}
        _, result, errors, secs = interp_cpp.analyze(str(source), heap_top=HEAP_TOP_AT_START, stubs=stubs)
    elif a.source.endswith((".c",)):
        from . import analyze
        _, result, errors, secs = analyze(a.source, a.defines, a.includes, footprint="bytes", argv=a.argv)
    else:
        _, result, errors, secs = interp_cpp.analyze(a.source, a.defines, a.includes, argv=a.argv)
    b, unmatched = write(result.interp, a.out_dir, name, a.sitemap, a.min)
    big = b[b["size"] >= max(a.min, 65536)] if not b.empty else b
    print(f"{name}: {len(b)} blocks ({len(big)} of at least 64 KB, {big['size'].sum() / 1e6:.1f} MB requested, "
          f"high water {big['high_water'].sum() / 1e6:.1f} MB), coverage {result.coverage:.3f}, {secs:.0f} s")
    for u in sorted(unmatched):
        print("  no pass site for", u)


if __name__ == "__main__":
    main()
