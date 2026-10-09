"""Static access-count spectrum of a C program.

A vectorised abstract interpreter over the clang AST runs the program from
main() without its data. Integer variables (sizes, loop indices) are tracked
exactly; floating-point values and memory contents are unknown. A counted loop
runs all its iterations at once as numpy arrays (a batch), so a statement in a
loop nest is executed once, for every iteration point together.

Every memory reference an unoptimised (-O0) build makes is charged to its
address: reads and writes of locals and parameters (stack slots), array
elements, floating-point constants (.rodata) and the call frame (return
address, saved frame pointer). The result is the number of references to each
address, the spectrum that decides what a 1-in-k sample of the references sees.

Integer values stored in memory are remembered (per 4-byte unit), so loops
bounded by loaded values (CSR row offsets) and subscripts that load an index
(A[B[i]]) resolve from the values the program computed. Input that comes from
the C library's random-number generator (rand, random, lrand48) is drawn
from its distribution. `a[i]++` and `a[i] += x` keep their sequential meaning
when a vectorised loop repeats an index (degree counting, `nbr[pos[u]++] = v`);
a loop that otherwise both reads and writes an integer array (a prefix sum,
a BFS claiming vertices) runs one iteration at a time if it is not inside
another such loop and has at most SEQUENTIAL_LIMIT iterations; otherwise it
stays vectorised and the values it computes may differ from a sequential run
(floyd-warshall's path lengths), which only matters where they decide
addresses or control.

References the interpreter cannot place are counted, not guessed: an address
that depends on data it does not know (floating-point values, pointers stored
in memory, files) is unresolved, and
references inside a loop with a data-dependent trip count or under a
data-dependent branch are uncertain. Coverage is the share of references that
are neither.
"""

from dataclasses import dataclass, field

import numpy as np

from . import clangast as ca
from .clangast import K

CHUNK = 1 << 21  # iteration points per batch
MAX_ITERATIONS = 1 << 24  # iterations of a loop run one at a time (a guard against runaway loops)
MAX_DEPTH = 64
FRAME_BYTES = 8192  # stack frame size; frames at the same call depth share addresses
STACK_ARRAY_LIMIT = 1024  # larger local arrays get their own object
GUESS_TRIPS = 1  # trips charged for a loop whose trip count depends on data
SEQUENTIAL_LIMIT = 2_000_000  # iterations a loop carrying values through memory may run one at a time


# An unknown integer in memory: an arbitrary bit pattern (not INT64_MIN, which a bitmap word with
# only bit 63 set equals).
SENTINEL = np.iinfo(np.int64).min + 0x5A3C96E1


_UNSIGNED = None


def wrap_int(value, type_):
    """An integer value converted to the C type: wrapped to its width and signedness (bool: 0/1).
    64-bit values stay int64 (unsigned ones keep their bits)."""
    global _UNSIGNED
    if _UNSIGNED is None:
        tk = ca.ci.TypeKind
        _UNSIGNED = {tk.UINT, tk.ULONG, tk.ULONGLONG, tk.USHORT, tk.UCHAR, tk.CHAR_U, tk.UINT128, tk.CHAR16, tk.CHAR32}
    if value is UNK or isinstance(value, Ptr):
        return value
    t = type_.get_canonical()
    if t.kind == ca.ci.TypeKind.BOOL:
        return _SCALAR((np.asarray(value) != 0).astype(np.int64)) if np.ndim(value) else int(value != 0)
    size = ca.type_size(t)
    if size is None or size >= 8 or not (t.kind in _UNSIGNED or t.kind in (
            ca.ci.TypeKind.INT, ca.ci.TypeKind.SHORT, ca.ci.TypeKind.SCHAR, ca.ci.TypeKind.CHAR_S, ca.ci.TypeKind.LONG)):
        return value
    bits = 8 * size
    if t.kind in _UNSIGNED:
        return _SCALAR(np.asarray(value) & ((1 << bits) - 1)) if np.ndim(value) else int(value) & ((1 << bits) - 1)
    half = 1 << (bits - 1)
    if np.ndim(value):
        return _SCALAR(((np.asarray(value) + half) & ((1 << bits) - 1)) - half)
    return ((int(value) + half) & ((1 << bits) - 1)) - half


def _wrap64(x):
    """A Python integer as the int64 with the same low 64 bits (unsigned 64-bit values)."""
    return ((int(x) + 2 ** 63) % 2 ** 64) - 2 ** 63


class _Unknown:
    def __repr__(self):
        return "UNK"


UNK = _Unknown()


def _is_unk(*values):
    return any(v is UNK for v in values)


@dataclass(eq=False)
class Obj:
    """A memory object: heap block, stack, global, .rodata. Counts are kept per start byte and access size."""

    name: str
    kind: str
    nbytes: int = 0
    by_size: dict = field(default_factory=dict)
    unresolved: float = 0.0
    zeroed: bool = False  # calloc and static storage start as zeros
    values: object = None  # integer contents per 4-byte unit (SENTINEL: unknown)

    def _value_units(self, hi):
        if self.values is None or len(self.values) < hi:
            fill = 0 if self.zeroed else SENTINEL
            new = np.full(max(hi, (self.nbytes + 3) // 4, 1), fill, dtype=np.int64)
            if self.values is not None:
                new[: len(self.values)] = self.values
            self.values = new
        return self.values

    def _byte_values(self, hi):
        if getattr(self, "bytevals", None) is None or len(self.bytevals) < hi:
            fill = 0 if self.zeroed else SENTINEL
            new = np.full(max(hi, self.nbytes, 1), fill, dtype=np.int64)
            if getattr(self, "bytevals", None) is not None:
                new[: len(self.bytevals)] = self.bytevals
            self.bytevals = new
        return self.bytevals

    def load(self, off, size=4):
        """Integer value(s) at byte offset(s). Values of 4 bytes and more are kept per 4-byte unit,
        smaller ones (bool, char, short fields packed together) per byte."""
        if size < 4:
            offs = np.asarray(off, dtype=np.int64)
            if np.any(offs < 0):
                return np.full(np.shape(offs), SENTINEL, dtype=np.int64)
            return self._byte_values(int(np.max(offs)) + 1 if offs.size else 1)[offs]
        units = np.asarray(off, dtype=np.int64) // 4
        if np.any(units < 0):
            return np.full(np.shape(units), SENTINEL, dtype=np.int64)
        vals = self._value_units(int(np.max(units)) + 1 if units.size else 1)
        return vals[units]

    def point_values(self, unit, kind, n):
        """Per-point values a batch of n points stored at one 4-byte unit (a local array written by every
        iteration of a batched loop: one address, one value per iteration), or None."""
        pp = getattr(self, "pp", None)
        if not pp:
            return None
        v = pp.get((kind, int(unit)))
        return v if v is not None and len(v) == n else None

    def _forget_points(self, units):
        pp = getattr(self, "pp", None)
        if not pp:
            return
        us = np.unique(np.asarray(units, dtype=np.int64).ravel())
        for key in [k for k in pp if np.isin(k[1], us)]:
            del pp[key]

    def _keep_points(self, unit, kind, values):
        if getattr(self, "pp", None) is None:
            self.pp = {}
        self.pp[(kind, int(unit))] = np.array(values)

    def _float_units(self, hi):
        fv = getattr(self, "fvals", None)
        if fv is None or len(fv) < hi:
            new = np.full(max(hi, (self.nbytes + 3) // 4, 1), np.nan)
            if fv is not None:
                new[: len(fv)] = fv
            self.fvals = fv = new
        return fv

    def fload(self, off, size=8):
        """Floating-point value(s) at byte offset(s), NaN where unknown. A value never stored as a
        float reads as 0.0 where its integer units hold 0 (zeroed memory, value-initialised elements)."""
        units = np.asarray(off, dtype=np.int64) // 4
        if units.size == 0 or np.any(units < 0):
            return np.full(np.shape(units), np.nan)
        hi = int(np.max(units)) + (2 if size == 8 else 1)
        out = self._float_units(hi)[units].copy()
        missing = np.isnan(out)
        if np.any(missing):
            ints = self._value_units(hi)
            zero = ints[units] == 0
            if size == 8:
                zero &= ints[units + 1] == 0
            out = np.where(missing & zero, 0.0, out)
        return out

    def fstore(self, off, value, size=8):
        """Store floating-point value(s) (NaN: unknown). Repeated offsets keep the last value."""
        units = np.asarray(off, dtype=np.int64) // 4
        if units.size == 0 or np.any(units < 0):
            return
        fv = self._float_units(int(np.max(units)) + (2 if size == 8 else 1))
        v = np.asarray(value, dtype=float)
        self._forget_points(units)
        if np.ndim(units) == 0:
            fv[int(units)] = v.ravel()[-1] if v.ndim else float(v)
            if v.ndim and v.size > 1:
                self._keep_points(units, "f", v.ravel())
        else:
            fv[units] = np.broadcast_to(v, units.shape)
        ints = self._value_units(len(fv))                              # the integer view is not known
        ints[units] = SENTINEL
        if size == 8:
            ints[units + 1] = SENTINEL

    def store(self, off, value, size=4):
        """Store integer value(s) at byte offset(s); UNK stores unknown. Repeated offsets keep the
        last value, as a sequential loop would."""
        fv = getattr(self, "fvals", None)
        if fv is not None and size >= 4:                               # an integer replaces a float there
            u = np.asarray(off, dtype=np.int64) // 4
            if u.size and np.all(u >= 0):
                u = u[u < len(fv)] if np.ndim(u) else (u if u < len(fv) else None)
                if u is not None and np.size(u):
                    fv[u] = np.nan
        if size < 4:
            offs = np.asarray(off, dtype=np.int64)
            if offs.size == 0 or np.any(offs < 0):
                return
            vals = self._byte_values(int(np.max(offs)) + 1)
        else:
            offs = np.asarray(off, dtype=np.int64) // 4
            if offs.size == 0 or np.any(offs < 0):
                return
            vals = self._value_units(int(np.max(offs)) + 1)
        units = offs
        v = SENTINEL if value is UNK else value
        if size >= 4:
            self._forget_points(units)
            if np.ndim(units) == 0 and np.ndim(v) and np.size(v) > 1:
                self._keep_points(units, "i", np.asarray(v, dtype=np.int64).ravel())
        if np.ndim(units) == 0:
            vals[int(units)] = _wrap64(np.asarray(v).ravel()[-1] if np.ndim(v) else v)
        else:
            if not isinstance(v, np.ndarray):
                v = _wrap64(v)
            vals[units] = np.broadcast_to(np.asarray(v, dtype=np.int64), units.shape)

    def _grow(self, size, length):
        arr = self.by_size.get(size)
        if arr is None or len(arr) < length:
            new = np.zeros(max(length, self.nbytes), dtype=np.float64)
            if arr is not None:
                new[: len(arr)] = arr
            self.by_size[size] = arr = new
        return arr

    def add(self, off, size, w):
        """Charge references at byte offsets off (scalar or array aligned with w)."""
        if np.ndim(off) == 0:
            off = int(off)
            if off < 0:
                self.unresolved += float(np.sum(w))
                return
            self._grow(size, off + 1)[off] += float(np.sum(w))
            return
        off = np.broadcast_to(np.asarray(off, dtype=np.int64), np.shape(w))
        bad = off < 0
        if bad.any():
            self.unresolved += float(np.sum(w[bad]))
            off, w = off[~bad], w[~bad]
        if len(off) == 0:
            return
        lo, hi = int(off.min()), int(off.max()) + 1
        arr = self._grow(size, hi)
        if 8 * len(off) < hi - lo:                                     # few offsets over a wide range
            np.add.at(arr, off, w)
        else:
            arr[lo:hi] += np.bincount(off - lo, weights=w, minlength=hi - lo)

    def addresses(self, footprint="starts"):
        """(counts, sizes) of the object's footprint units.

        starts: one unit per start address, sized by its largest access (the paper's footprint).
        bytes: bytes touched; an access counts once on every byte it covers. Bytes are grouped
        into aligned 8-byte units where all eight have the same count, else counted singly."""
        if not self.by_size:
            return np.zeros(0), np.zeros(0)
        if footprint == "bytes":
            return self._bytes()
        length = max(len(a) for a in self.by_size.values())
        total = np.zeros(length)
        size = np.zeros(length)
        for s, arr in sorted(self.by_size.items()):
            total[: len(arr)] += arr
            size[: len(arr)][arr > 0] = s  # largest access size wins (sorted ascending)
        touched = total > 0
        counts, sizes = total[touched], size[touched]
        if self.unresolved and len(counts):
            counts = counts + self.unresolved / len(counts)
        return counts, sizes


    def coverage(self):
        """References covering each byte (bytes touched)."""
        if not self.by_size:
            return np.zeros(0)
        length = max(len(a) + s for s, a in self.by_size.items())
        cover = np.zeros((length + 7) // 8 * 8)
        for s, arr in self.by_size.items():
            c = np.convolve(arr, np.ones(s))
            cover[: len(c)] += c
        touched = cover > 0
        if self.unresolved and touched.any():
            cover[touched] += self.unresolved / touched.sum()
        return cover

    def _bytes(self):
        return group_words(self.coverage())


def group_words(cover):
    """Bytes with references, as aligned 8-byte units where all eight have the same count, else singly."""
    cover = np.concatenate([cover, np.zeros(-len(cover) % 8)])
    words = cover.reshape(-1, 8)
    uniform = (words == words[:, :1]).all(axis=1) & (words[:, 0] > 0)
    rest = words[~uniform].ravel()
    rest = rest[rest > 0]
    counts = np.concatenate([words[uniform, 0], rest])
    sizes = np.concatenate([np.full(int(uniform.sum()), 8.0), np.ones(len(rest))])
    return counts, sizes


@dataclass
class Ptr:
    obj: object  # Obj or UNK
    off: object  # int, int array or UNK (bytes)
    src: object = None  # where the pointer itself was loaded from (its provenance), if from memory


@dataclass(eq=False)
class Real:
    """A floating-point value (scalar or one per batch point), kept only in scalar variables
    (Interpreter.REALS): enough for decisions computed from integers (a ratio against a
    threshold). Floats in memory stay unknown."""
    v: object


def _real(v):
    """v as a float (Real), or UNK."""
    if isinstance(v, Real):
        return v
    v = _as_int(v)
    return UNK if v is UNK else Real(np.asarray(v, dtype=float) if np.ndim(v) else float(v))


@dataclass
class Loc:
    obj: object
    off: object
    size: int
    array: bool = False  # an array lvalue: decays, is never loaded or stored


@dataclass
class Frame:
    func: object  # function definition cursor (None for the globals)
    base: int  # byte offset of the frame in the stack object
    vars: dict = field(default_factory=dict)  # key -> [slot Loc, value]


def _key(decl):
    k = decl.__dict__.get("_memprint_key")
    if k is None:
        k = decl._memprint_key = (decl.spelling, decl.location.offset, str(decl.location.file))
    return k


class Partial:
    """Integer values known for some batch points only (a variable assigned in the branches of an
    if/else that only some points take). Unknown (UNK) to everything else until every point is known."""
    __slots__ = ("v", "known")

    def __init__(self, v, known):
        self.v, self.known = v, known


def _as_int(v):
    if isinstance(v, Partial):
        return v.v if v.known.all() else UNK
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, np.ndarray) and v.dtype.kind in "biu":
        return v.astype(np.int64)
    return UNK


def _tdiv(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.abs(a) // np.where(b == 0, 1, np.abs(b))
        return q * np.sign(a) * np.sign(b)


ARITH = {
    "+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
    "/": _tdiv, "%": lambda a, b: a - b * _tdiv(a, b),
    "<<": lambda a, b: a << b, ">>": lambda a, b: a >> b,
    "&": lambda a, b: a & b, "|": lambda a, b: a | b, "^": lambda a, b: a ^ b,
    "<": lambda a, b: (np.asarray(a) < b).astype(np.int64), ">": lambda a, b: (np.asarray(a) > b).astype(np.int64),
    "<=": lambda a, b: (np.asarray(a) <= b).astype(np.int64), ">=": lambda a, b: (np.asarray(a) >= b).astype(np.int64),
    "==": lambda a, b: (np.asarray(a) == b).astype(np.int64), "!=": lambda a, b: (np.asarray(a) != b).astype(np.int64),
    "&&": lambda a, b: ((np.asarray(a) != 0) & (np.asarray(b) != 0)).astype(np.int64),
    "||": lambda a, b: ((np.asarray(a) != 0) | (np.asarray(b) != 0)).astype(np.int64),
}
_SCALAR = lambda v: v if np.ndim(v) else int(v)  # noqa: E731

ALLOC = {"malloc": (0,), "xmalloc": (0,), "calloc": (0, 1), "polybench_alloc_data": (0, 1),
         "aligned_alloc": (1,), "realloc": (1,), "valloc": (0,), "pvalloc": (0,)}
FREE = {"free", "polybench_free_data"}
ATOI = {"atoi", "atol", "atoll", "strtol", "strtoul", "strtoll", "strtoull", "stoi", "stol", "stoll", "stoul",
        "stoull"}
RANDOM = {"rand": 2 ** 31 - 1, "random": 2 ** 31 - 1, "lrand48": 2 ** 31 - 1}  # name -> largest value
# Inside glibc 2.28, rand() makes about 27 references per call (Pin, a loop of 10^6 calls at -O0):
# about 24 to some 120 bytes of hot state (lock, the random_data fields, its call frame) and 3 to
# the 31-word additive table. Measured after csr_pr / csr_bfs had been scored without it.
RAND_HOT_BYTES, RAND_HOT_REFS = 120, 24.0
RAND_TABLE_WORDS, RAND_TABLE_REFS = 31, 3.0


@dataclass
class Result:
    counts: np.ndarray  # references per start address
    sizes: np.ndarray  # bytes per start address
    objects: list  # per-object summary dicts
    total: float  # all references charged
    unresolved: float  # references whose address is unknown
    uncertain: float  # references under data-dependent control
    data_loops: int
    data_branches: int

    @property
    def footprint(self):
        return float(self.sizes.sum())

    @property
    def coverage(self):
        return 1.0 - (self.unresolved + self.uncertain) / self.total if self.total else 0.0


class Interpreter:
    REALS = False  # track floating-point scalars (Real)

    def __init__(self, tu, seed=0):
        self.tu = tu
        self.w = np.ones(1)
        self.stack = Obj("stack", "stack", MAX_DEPTH * FRAME_BYTES)
        self.globals_obj = Obj("globals", "global", zeroed=True)
        self.rodata = Obj("rodata", "rodata")
        self.objects = [self.stack, self.globals_obj, self.rodata]
        self.rodata_slots = {}
        self.layouts = {}  # function key -> {var key: offset}
        self.frames = [Frame(None, 0)]
        self.charging = True
        self.uncertain_depth = 0
        self.sequential_depth = 0
        self.heap_events = []  # ("new" | "free", Obj) in program order
        self.total = self.unresolved = self.uncertain = 0.0
        self.data_loops = self.data_branches = 0
        self.functions = {}
        self._global_off = 0
        self.rng = np.random.default_rng(seed)
        self.argv = ()  # the command line (strings), argv[0] first
        for c in tu.cursor.get_children():
            if c.kind == K.FUNCTION_DECL and c.is_definition():
                self.functions[c.spelling] = c
            elif c.kind == K.VAR_DECL:
                self._declare_global(c)

    # ------------------------------------------------------------ batches

    @property
    def L(self):
        return len(self.w)

    def _map_frames(self, f):
        def mapv(v):
            if isinstance(v, np.ndarray):
                return f(v)
            if isinstance(v, Ptr) and isinstance(v.off, np.ndarray):
                return Ptr(v.obj, f(v.off), v.src)
            if isinstance(v, Real) and isinstance(v.v, np.ndarray):
                return Real(f(v.v))
            if isinstance(v, Partial):
                return Partial(f(v.v), f(v.known))
            return v

        return [Frame(fr.func, fr.base, {k: [s, mapv(v)] for k, (s, v) in fr.vars.items()}) for fr in self.frames]

    def _merge(self, sub_frames, idx):
        L = self.L
        for fr, sub in zip(self.frames, sub_frames):
            for k, entry in fr.vars.items():
                if k not in sub.vars:
                    continue
                old, new = entry[1], sub.vars[k][1]
                entry[1] = self._merge_value(old, new, idx, L)
            for k, entry in sub.vars.items():                          # declared in the subset
                if k not in fr.vars:
                    fr.vars[k] = [entry[0], self._expand(entry[1], idx, L)]

    @staticmethod
    def _expand(new, idx, L):
        """A value of the points idx as a value of all L points (the others, not running, get the
        first point's value)."""
        def full(a):
            out = np.empty(L, dtype=np.asarray(a).dtype)
            out[:] = np.ravel(a)[0] if np.size(a) else 0
            out[idx] = a
            return out
        if isinstance(new, Ptr):
            return new if not isinstance(new.off, np.ndarray) else Ptr(new.obj, full(new.off), new.src)
        if isinstance(new, Real):
            return new if np.ndim(new.v) == 0 else Real(full(new.v))
        return full(new) if isinstance(new, np.ndarray) else new

    @staticmethod
    def _merge_value(old, new, idx, L):
        if old is new:
            return old
        if len(idx) == L and np.all(idx == np.arange(L)):
            return new
        if isinstance(old, Ptr) and isinstance(new, Ptr):
            if old.obj is not new.obj:
                return Ptr(UNK, UNK)
            off = Interpreter._merge_value(old.off, new.off, idx, L)
            return Ptr(old.obj, off, old.src if old.src == new.src else None)
        if isinstance(old, Real) and isinstance(new, Real):
            full = np.array(np.broadcast_to(old.v, L), dtype=float)
            full[idx] = new.v
            return Real(full)
        if (old is UNK or isinstance(old, Partial)) and _as_int(new) is not UNK:
            v = old.v.copy() if isinstance(old, Partial) else np.zeros(L, dtype=np.int64)
            known = old.known.copy() if isinstance(old, Partial) else np.zeros(L, dtype=bool)
            v[idx] = np.broadcast_to(np.asarray(_as_int(new), dtype=np.int64), np.shape(idx))
            known[idx] = True
            return v if known.all() else Partial(v, known)
        o, n = _as_int(old), _as_int(new)
        if _is_unk(o, n):
            return UNK if not (old is UNK and new is UNK) else UNK
        if np.ndim(o) == 0 and np.ndim(n) == 0 and o == n:
            return o
        full = np.array(np.broadcast_to(o, L), dtype=np.int64)
        full[idx] = n
        return full

    def _link(self, child, parent, index):
        """A batch of weights `child` derived from `parent` (its points parent[index]; None: the same
        points). The C++ interpreter's load cache uses it; the C interpreter has none."""

    def _drop(self, w):
        """A derived batch is finished."""

    def run_subset(self, idx, fn, scale=1.0):
        """Run fn on the batch points idx (weights scaled); variable updates are merged back."""
        if len(idx) == 0:
            return None
        if len(idx) == self.L and scale == 1.0:
            return fn()
        saved_frames, saved_w = self.frames, self.w
        self.frames = self._map_frames(lambda a: a[idx])
        self.w = saved_w[idx] * scale
        try:
            return fn()
        finally:
            sub = self.frames
            self.frames, self.w = saved_frames, saved_w
            self._merge(sub, idx)

    def with_weights(self, w, fn):
        saved = self.w
        self.w = np.broadcast_to(w, saved.shape).astype(np.float64)
        try:
            return fn()
        finally:
            self.w = saved

    def no_charge(self, fn):
        saved = self.charging
        self.charging = False
        try:
            return fn()
        finally:
            self.charging = saved

    def uncertainly(self, fn):
        self.uncertain_depth += 1
        try:
            return fn()
        finally:
            self.uncertain_depth -= 1

    # ------------------------------------------------------------ memory

    def access(self, loc, write=False):
        if not self.charging or loc.array:
            return
        weight = float(self.w.sum())
        self.total += weight
        if self.uncertain_depth:
            self.uncertain += weight
        if loc.obj is UNK or loc.off is UNK or loc.obj is None:
            self.unresolved += weight
            if isinstance(loc.obj, Obj):
                loc.obj.unresolved += weight
            return
        loc.obj.add(loc.off, loc.size or 8, self.w)

    def _declare_global(self, c):
        size = ca.type_size(c.type) or 8
        off = self._global_off
        self._global_off += (size + 7) // 8 * 8
        if ca.is_array(c.type) and size > STACK_ARRAY_LIMIT:
            obj = Obj(c.spelling, "global", size, zeroed=True)
            self.objects.append(obj)
            slot = Loc(obj, 0, size, True)
            value = Ptr(obj, 0)
        else:
            slot = Loc(self.globals_obj, off, size, ca.is_array(c.type))
            value = Ptr(self.globals_obj, off) if ca.is_array(c.type) else UNK
            init = self._initializer(c)
            if init is not None and not ca.is_array(c.type):
                v = ca.evaluate(init)
                value = v if isinstance(v, int) else UNK
        self.frames[0].vars[_key(c)] = [slot, value]

    @staticmethod
    def _initializer(c):
        kids = ca.children(c)
        if kids and any(t.spelling == "=" for t in c.get_tokens()):
            return kids[-1]
        return None

    def _slot_offset(self, frame, decl, size):
        layout = self.layouts.setdefault(_key(frame.func), {"__next": 16})
        k = _key(decl)
        if k not in layout:
            align = min(16, max(1, size))
            layout["__next"] = (layout["__next"] + align - 1) // align * align
            layout[k] = layout["__next"]
            layout["__next"] += size
        return frame.base + min(layout[k], FRAME_BYTES - size)

    def _declare_local(self, decl, value=UNK, pointer=False):
        """A stack slot for a local or parameter (an array parameter is an 8-byte pointer)."""
        frame = self.frames[-1]
        size = 8 if pointer else ca.type_size(decl.type)
        if ca.is_array(decl.type) and not pointer:
            if size is None or size > STACK_ARRAY_LIMIT:
                obj = Obj(f"{frame.func.spelling}.{decl.spelling}", "stack-array", size or 0)
                self.objects.append(obj)
                slot = Loc(obj, 0, size or 0, True)
                value = Ptr(obj, 0)
            else:
                off = self._slot_offset(frame, decl, size)
                slot = Loc(self.stack, off, size, True)
                value = Ptr(self.stack, off)
        else:
            size = size or 8
            slot = Loc(self.stack, self._slot_offset(frame, decl, size), size)
        frame.vars[_key(decl)] = [slot, value]
        return slot

    def _lookup(self, decl):
        k = _key(decl)
        for fr in (self.frames[-1], self.frames[0]):
            if k in fr.vars:
                return fr.vars[k]
        if decl.kind == K.VAR_DECL:  # a local static or a variable declared out of order
            self._declare_global(decl)
            return self.frames[0].vars[k]
        return None

    def _rodata(self, value):
        if value not in self.rodata_slots:
            self.rodata_slots[value] = 8 * len(self.rodata_slots)
        return Loc(self.rodata, self.rodata_slots[value], 8)

    # ------------------------------------------------------------ lvalues

    def lvalue(self, c):
        c = ca.strip(c)
        kind = c.kind
        if kind == K.DECL_REF_EXPR:
            decl = c.referenced
            entry = self._lookup(decl) if decl is not None else None
            if entry is None:
                return Loc(UNK, UNK, 8)
            return entry[0]
        if kind == K.ARRAY_SUBSCRIPT_EXPR:
            base_c, idx_c = ca.children(c)
            base, idx = self.eval(base_c), self.eval(idx_c)
            if isinstance(idx, Ptr):
                base, idx = idx, base
            size = ca.type_size(c.type) or 8
            array = ca.is_array(c.type)
            idx = _as_int(idx)
            if not isinstance(base, Ptr) or base.obj is UNK:
                return Loc(UNK, UNK, size, array)
            if _is_unk(idx, base.off):
                return Loc(base.obj, UNK, size, array)
            loc = Loc(base.obj, base.off + idx * size, size, array)
            loc.src = base.src
            return loc
        if kind == K.UNARY_OPERATOR and ca.unary_op(c) == "*":
            p = self.eval(ca.children(c)[0])
            size = ca.type_size(c.type) or 8
            if not isinstance(p, Ptr):
                return Loc(UNK, UNK, size, ca.is_array(c.type))
            loc = Loc(p.obj, p.off, size, ca.is_array(c.type))
            loc.src = p.src
            return loc
        if kind == K.MEMBER_REF_EXPR:
            kids = ca.children(c)
            fld = c.referenced
            fo = fld.get_field_offsetof() // 8 if fld is not None and fld.get_field_offsetof() >= 0 else 0
            size = ca.type_size(c.type) or 8
            array = ca.is_array(c.type)
            if not kids:
                return Loc(UNK, UNK, size, array)
            base_c = kids[0]
            if ca.is_pointer(base_c.type):
                p = self.eval(base_c)
                if not isinstance(p, Ptr) or _is_unk(p.obj, p.off):
                    return Loc(p.obj if isinstance(p, Ptr) else UNK, UNK, size, array)
                loc = Loc(p.obj, p.off + fo, size, array)
                loc.src = p.src
                return loc
            b = self.lvalue(base_c)
            if _is_unk(b.obj, b.off):
                return Loc(b.obj, UNK, size, array)
            return Loc(b.obj, b.off + fo, size, array)
        # anything else: evaluate for its references, address unknown
        self.eval(c)
        return Loc(UNK, UNK, ca.type_size(c.type) or 8)

    def _store(self, c, value):
        """Assign value to the lvalue c: charge the store, keep integer values of variables."""
        loc = self.lvalue(c)
        self.access(loc, write=True)
        target = ca.strip(c)
        if target.kind == K.DECL_REF_EXPR and target.referenced is not None:
            entry = self._lookup(target.referenced)
            if entry is not None:
                entry[1] = value if (isinstance(value, (Ptr, Real)) or not ca.is_float(target.type)) else UNK
        else:
            self._mem_store(loc, target.type, value)
        return loc

    @staticmethod
    def _tracked(type_):
        """Integer contents are remembered in memory; floats and pointers are not."""
        t = type_.get_canonical()
        return not (ca.is_float(t) or ca.is_pointer(t) or ca.is_array(t)) and ca.type_size(t) in (1, 2, 4, 8)

    def _mem_load(self, loc, type_):
        if not self._tracked(type_) or not isinstance(loc.obj, Obj) or loc.off is UNK:
            return UNK
        size = ca.type_size(type_.get_canonical()) or 4
        vals = loc.obj.load(loc.off, size)
        if np.ndim(loc.off) == 0 and size >= 4 and self.L > 1:          # one address, a value per point
            pp = loc.obj.point_values(int(loc.off) // 4, "i", self.L)
            if pp is not None:
                vals = pp
        if np.any(vals == SENTINEL):
            return UNK
        return _SCALAR(vals) if np.ndim(vals) else int(vals)

    def _mem_store(self, loc, type_, value):
        if not isinstance(loc.obj, Obj) or loc.off is UNK:
            return
        value = _as_int(value) if self._tracked(type_) and not isinstance(value, Ptr) else UNK
        if self.charging:
            loc.obj.store(np.broadcast_to(loc.off, (self.L,)) if np.ndim(loc.off) else loc.off,
                          value if value is UNK or np.ndim(value) or np.ndim(loc.off) == 0
                          else np.broadcast_to(value, (self.L,)), ca.type_size(type_.get_canonical()) or 4)

    def _rmw_add(self, loc, type_, delta):
        """a[i] += delta (or ++, --) for every batch point in order: returns (old, new) per point.
        Repeated offsets see each other's updates as in a sequential loop."""
        delta = _as_int(delta)
        if not self._tracked(type_) or not isinstance(loc.obj, Obj) or loc.off is UNK or delta is UNK:
            if isinstance(loc.obj, Obj) and loc.off is not UNK:
                loc.obj.store(loc.off, UNK, ca.type_size(type_.get_canonical()) or 4)
            return UNK, UNK
        L = self.L
        offs = np.broadcast_to(np.asarray(loc.off, dtype=np.int64), (L,))
        d = np.broadcast_to(np.asarray(delta, dtype=np.int64), (L,))
        size = ca.type_size(type_.get_canonical()) or 4
        base = loc.obj.load(offs, size)
        order = np.argsort(offs, kind="stable")
        so, sd = offs[order], d[order]
        first = np.concatenate([[True], so[1:] != so[:-1]])
        before = np.cumsum(sd) - sd
        group = np.cumsum(first) - 1
        excl = before - before[np.nonzero(first)[0]][group]
        old = np.empty(L, dtype=np.int64)
        old[order] = base[order] + excl
        new = old + d
        last = np.concatenate([so[1:] != so[:-1], [True]])
        unknown = base == SENTINEL
        final = np.where(unknown[order][last], SENTINEL, new[order][last])
        loc.obj.store(so[last], final, size)
        if unknown.any():
            return UNK, UNK
        if L == 1 and np.ndim(loc.off) == 0 and np.ndim(delta) == 0:
            return int(old[0]), int(new[0])
        return old, new

    def _load_lvalue(self, c):
        """Rvalue of an lvalue expression: a load, or a pointer if it is an array."""
        loc = self.lvalue(c)
        if loc.array:
            return Ptr(loc.obj, loc.off) if loc.obj is not UNK else Ptr(UNK, UNK)
        self.access(loc)
        target = ca.strip(c)
        if target.kind == K.DECL_REF_EXPR and target.referenced is not None:
            entry = self._lookup(target.referenced)
            if entry is not None:
                return entry[1]
            return UNK
        return self._mem_load(loc, c.type)

    # ------------------------------------------------------------ rvalues

    def eval(self, c):
        kind = c.kind
        if kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR):
            kids = ca.children(c)
            if len(kids) == 1:
                if ca.is_float(kids[0].type) and not (ca.is_float(c.type) or ca.is_pointer(c.type)):
                    folded = ca.evaluate(c)                            # a constant float -> int (0.57*max)
                    if isinstance(folded, int):
                        return wrap_int(folded, c.type)
                v = self.eval(kids[0])
                if ca.is_float(c.type) and not isinstance(v, Ptr):
                    return _real(v) if self.REALS else UNK
                if isinstance(v, Real):
                    v = self._truncate(v)
                if not isinstance(v, (int, np.integer, np.ndarray)):
                    return v
                return wrap_int(v, c.type)
            for kid in kids:
                self.eval(kid)
            return UNK
        if kind == K.INTEGER_LITERAL or kind == K.CHARACTER_LITERAL:
            v = ca.evaluate(c)
            return v if isinstance(v, int) else UNK
        if kind == K.FLOATING_LITERAL:
            v = ca.evaluate(c)
            if v:
                self.access(self._rodata(v))
            return Real(float(v)) if self.REALS and v is not None else UNK
        if kind == K.STRING_LITERAL:
            return Ptr(UNK, UNK)
        if kind == K.DECL_REF_EXPR:
            decl = c.referenced
            if decl is None:
                return UNK
            if decl.kind == K.ENUM_CONSTANT_DECL:
                return decl.enum_value
            if decl.kind == K.FUNCTION_DECL:
                return UNK
            return self._load_lvalue(c)
        if kind in (K.ARRAY_SUBSCRIPT_EXPR, K.MEMBER_REF_EXPR):
            return self._load_lvalue(c)
        if kind == K.CSTYLE_CAST_EXPR:
            v = self.eval(ca.children(c)[-1])
            if ca.is_float(c.type):
                return _real(v) if self.REALS else UNK
            if isinstance(v, Real):
                v = self._truncate(v)
            return wrap_int(v, c.type) if isinstance(v, (int, np.integer, np.ndarray)) else v
        if kind == K.UNARY_OPERATOR:
            return self._unary(c)
        if kind == K.BINARY_OPERATOR:
            return self._binary(c)
        if kind == K.COMPOUND_ASSIGNMENT_OPERATOR:
            lhs, rhs = ca.children(c)
            loc = self.lvalue(lhs)
            self.access(loc)
            op = ca.compound_op(c)
            if ca.strip(lhs).kind != K.DECL_REF_EXPR:                  # an element in memory
                r = self.eval(rhs)
                self.access(loc, write=True)
                if self.REALS and ca.is_float(lhs.type) and isinstance(loc.obj, Obj) and loc.off is not UNK:
                    return self._float_update(loc, lhs.type, op, r)
                if op in ("+", "-") and not isinstance(r, Ptr):
                    r = _as_int(r)
                    _, new = self._rmw_add(loc, lhs.type, r if op == "+" or r is UNK else _SCALAR(-r))
                    return new
                old = self._mem_load(loc, lhs.type)
                new = self._arith(op, old, r, c)
                repeated = np.ndim(loc.off) and len(np.unique(loc.off)) < np.size(loc.off)
                if repeated and op in ("|", "&", "^") and not _is_unk(old, _as_int(r)):
                    # commutative bit updates: combine every point's operand per location
                    offs = np.broadcast_to(np.asarray(loc.off), (self.L,))
                    rv = np.broadcast_to(np.asarray(_as_int(r), dtype=np.int64), (self.L,))
                    uniq, inv = np.unique(offs, return_inverse=True)
                    base = np.asarray(loc.obj.load(uniq, ca.type_size(lhs.type.get_canonical()) or 4))
                    ufunc = {"|": np.bitwise_or, "&": np.bitwise_and, "^": np.bitwise_xor}[op]
                    acc = base.copy()
                    ufunc.at(acc, inv.ravel(), rv)
                    loc.obj.store(uniq, acc, ca.type_size(lhs.type.get_canonical()) or 4)
                    return _SCALAR(acc[inv.ravel()])
                self._mem_store(loc, lhs.type, UNK if repeated else new)
                return new
            old = self._value_of(lhs)
            r = self.eval(rhs)
            new = self._arith(op, old, r, c)
            if isinstance(new, Real) and not ca.is_float(lhs.type):      # int += double: converted back
                new = self._truncate(new)
                if new is not UNK:
                    new = wrap_int(new, lhs.type)
            self.access(loc, write=True)
            self._set_value(lhs, new)
            return new
        if kind == K.CONDITIONAL_OPERATOR:
            return self._ternary(c)
        if kind == K.CALL_EXPR:
            return self._call(c)
        if kind == K.CXX_UNARY_EXPR:  # sizeof / alignof
            v = ca.evaluate(c)
            return v if isinstance(v, int) else UNK
        for kid in ca.children(c):
            self.eval(kid) if kid.kind.is_expression() else self.exec(kid)
        return UNK

    def _value_of(self, c):
        target = ca.strip(c)
        if target.kind == K.DECL_REF_EXPR and target.referenced is not None:
            entry = self._lookup(target.referenced)
            if entry is not None:
                return entry[1]
        return UNK

    def _set_value(self, c, value):
        target = ca.strip(c)
        if target.kind == K.DECL_REF_EXPR and target.referenced is not None:
            entry = self._lookup(target.referenced)
            if entry is not None:
                entry[1] = value if (isinstance(value, (Ptr, Real)) or not ca.is_float(target.type)) else UNK

    @staticmethod
    def _truncate(v):
        """A Real converted to an integer type (towards zero), or UNK if not finite."""
        x = np.asarray(v.v, dtype=float)
        if not np.all(np.isfinite(x)):
            return UNK
        t = np.trunc(x).astype(np.int64)
        return _SCALAR(t) if np.ndim(t) else int(t)

    def _arith(self, op, a, b, c):
        if isinstance(a, Ptr) or isinstance(b, Ptr):
            return self._ptr_arith(op, a, b, c)
        if isinstance(a, Real) or isinstance(b, Real):
            a, b = _real(a), _real(b)
            if _is_unk(a, b) or op not in ("+", "-", "*", "/", "<", ">", "<=", ">=", "==", "!="):
                return UNK
            x, y = a.v, b.v
            if op == "/":
                if np.any(np.asarray(y) == 0):
                    return UNK
                return Real(np.asarray(x) / y if np.ndim(x) or np.ndim(y) else x / y)
            r = ARITH[op](np.asarray(x) if np.ndim(x) else x, y)
            if op in ("+", "-", "*"):
                return Real(r)
            return _SCALAR(np.asarray(r).astype(np.int64)) if np.ndim(r) else int(r)
        a, b = _as_int(a), _as_int(b)
        if _is_unk(a, b) or op not in ARITH:
            return UNK
        if op in ("/", "%") and np.any(np.asarray(b) == 0):
            return UNK
        return _SCALAR(ARITH[op](a, b))

    def _ptr_arith(self, op, a, b, c):
        if isinstance(a, Ptr) and isinstance(b, Ptr):
            if op == "-" and a.obj is b.obj and not _is_unk(a.off, b.off):
                size = ca.pointee_size(ca.children(c)[0].type) or 1
                return _SCALAR(_tdiv(np.asarray(a.off - b.off), size))
            if op in ("<", ">", "<=", ">=", "==", "!=") and a.obj is b.obj and not _is_unk(a.off, b.off):
                return _SCALAR(ARITH[op](a.off, b.off))
            if op in ("==", "!=") and isinstance(a.obj, Obj) and isinstance(b.obj, Obj) and a.obj is not b.obj:
                return int(op == "!=")                                 # distinct objects never compare equal
            return UNK
        if op in ("+", "-"):
            p, n = (a, b) if isinstance(a, Ptr) else (b, a)
            n = _as_int(n)
            size = ca.pointee_size(c.type) or 1
            if _is_unk(n, p.off):
                return Ptr(p.obj, UNK)
            return Ptr(p.obj, _SCALAR(p.off + n * size if op == "+" else p.off - n * size), p.src)
        return UNK

    def _unary(self, c):
        op = ca.unary_op(c)
        kid = ca.children(c)[0]
        if op in ("post++", "post--", "pre++", "pre--"):
            loc = self.lvalue(kid)
            self.access(loc)
            if ca.strip(kid).kind != K.DECL_REF_EXPR and not ca.is_pointer(kid.type):
                self.access(loc, write=True)
                old, new = self._rmw_add(loc, kid.type, 1 if "++" in op else -1)
                return old if op.startswith("post") else new
            old = self._value_of(kid)
            delta = 1 if "++" in op else -1
            if isinstance(old, Ptr):
                size = ca.pointee_size(kid.type) or 1
                new = Ptr(old.obj, UNK if old.off is UNK else _SCALAR(old.off + delta * size), old.src)
            else:
                o = _as_int(old)
                new = UNK if o is UNK else _SCALAR(o + delta)
            self.access(loc, write=True)
            self._set_value(kid, new)
            return old if op.startswith("post") else new
        if op == "&":
            loc = self.lvalue(kid)
            return Ptr(loc.obj, loc.off)
        if op == "*":
            return self._load_lvalue(c)
        v = self.eval(kid)
        if isinstance(v, Real):
            return Real(-v.v) if op == "-" else v if op == "+" else UNK
        v = _as_int(v)
        if v is UNK:
            return UNK
        if op == "-":
            return _SCALAR(-v)
        if op == "+":
            return v
        if op == "~":
            return _SCALAR(~v)
        if op == "!":
            return _SCALAR((np.asarray(v) == 0).astype(np.int64))
        return UNK

    def _binary(self, c):
        op = ca.binary_op(c)
        lhs, rhs = ca.children(c)
        if op == "=":
            v = self.eval(rhs)
            self._store(lhs, v)
            return v
        if op == ",":
            self.eval(lhs)
            return self.eval(rhs)
        if op in ("&&", "||"):
            return self._logical(op, lhs, rhs)
        a = self.eval(lhs)
        b = self.eval(rhs)
        return self._arith(op, a, b, c)

    def _logical(self, op, lhs, rhs):
        """Short-circuit && and ||: the right side runs only where it is needed."""
        a = _as_int(self.eval(lhs))
        if a is UNK:
            b = self.with_weights(self.w * 0.5, lambda: self.eval(rhs))
            return UNK if _as_int(b) is UNK else UNK
        decided = (np.asarray(a) == 0) if op == "&&" else (np.asarray(a) != 0)
        if np.ndim(a) == 0:
            if decided:
                return int(op == "||")
            b = _as_int(self.eval(rhs))
            return UNK if b is UNK else _SCALAR((np.asarray(b) != 0).astype(np.int64))
        idx = np.nonzero(~np.broadcast_to(decided, (self.L,)))[0]
        b = _as_int(self.run_subset(idx, lambda: self.eval(rhs)))
        if len(idx) and b is UNK:
            return UNK
        out = np.full(self.L, int(op == "||"), dtype=np.int64)
        out[idx] = (np.broadcast_to(np.asarray(b), idx.shape) != 0) if len(idx) else 0
        return out

    def _ternary(self, c):
        cond_c, then_c, else_c = ca.children(c)
        cond = _as_int(self.eval(cond_c))
        if cond is UNK:
            self.data_branches += 1
            a = self.uncertainly(lambda: self.with_weights(self.w * 0.5, lambda: self.eval(then_c)))
            b = self.uncertainly(lambda: self.with_weights(self.w * 0.5, lambda: self.eval(else_c)))
            return a if (np.ndim(a) == 0 and np.ndim(b) == 0 and not _is_unk(a, b) and a == b) else UNK
        if np.ndim(cond) == 0:
            return self.eval(then_c if cond else else_c)
        mask = np.broadcast_to(cond != 0, (self.L,))
        t_idx, f_idx = np.nonzero(mask)[0], np.nonzero(~mask)[0]
        a = self.run_subset(t_idx, lambda: self.eval(then_c))
        b = self.run_subset(f_idx, lambda: self.eval(else_c))
        a, b = _as_int(a) if len(t_idx) else 0, _as_int(b) if len(f_idx) else 0
        if _is_unk(a, b):
            return UNK
        out = np.zeros(self.L, dtype=np.int64)
        out[t_idx] = np.broadcast_to(a, t_idx.shape)
        out[f_idx] = np.broadcast_to(b, f_idx.shape)
        return out

    # ------------------------------------------------------------ calls

    def _call(self, c):
        kids = ca.children(c)
        args_c = kids[1:]
        name = c.spelling
        fdef = self.functions.get(name)
        if fdef is None or len(self.frames) > MAX_DEPTH:
            return self._library(name, args_c)
        args = [self.eval(a) for a in args_c]
        caller_base = self.frames[-1].base
        frame = Frame(fdef, len(self.frames) * FRAME_BYTES)
        ret_slot = Loc(self.stack, frame.base, 8)
        rbp_slot = Loc(self.stack, frame.base + 8, 8)
        self.access(ret_slot, write=True)
        self.access(rbp_slot, write=True)
        self.frames.append(frame)
        try:
            params = [p for p in fdef.get_children() if p.kind == K.PARM_DECL]
            for p, v in zip(params, args):
                slot = self._declare_local(p, v, pointer=ca.is_array(p.type))
                self.access(slot, write=True)
            frame.vars["__ret__"] = [None, UNK]
            body = [k for k in fdef.get_children() if k.kind == K.COMPOUND_STMT]
            if body:
                self.exec(body[0])
            ret = frame.vars["__ret__"][1]
        finally:
            self.frames.pop()
        del caller_base
        self.access(rbp_slot)
        self.access(ret_slot)
        return ret

    def _argv_value(self, expr):
        """atoi(argv[i]) and friends: the integer value of command-line argument i, or UNK."""
        e, queue = None, [(expr, 0)]                                   # through std::string(argv[i]) for stoi
        while queue:
            n, depth = queue.pop(0)
            if n.kind == K.ARRAY_SUBSCRIPT_EXPR:
                e = n
                break
            if depth < 8 and n.kind in (K.UNEXPOSED_EXPR, K.CALL_EXPR, K.PAREN_EXPR, K.CSTYLE_CAST_EXPR):
                queue += [(k, depth + 1) for k in ca.children(n)]
        if e is None:
            return UNK
        base, index = ca.children(e)
        d = ca.strip(base)
        if d.kind != K.DECL_REF_EXPR or d.referenced is None or d.referenced.kind != K.PARM_DECL \
                or d.referenced.spelling != "argv":
            return UNK
        i = _as_int(self.eval(index))
        if i is UNK or np.ndim(i) or not 0 <= i < len(self.argv):
            return UNK
        try:
            return int(str(self.argv[i]).strip(), 0)
        except ValueError:
            return UNK

    def _library(self, name, args_c):
        if name in ATOI and args_c:
            v = self._argv_value(args_c[0])
            if v is not UNK:
                return v
        args = [self.eval(a) for a in args_c]
        if name in ALLOC:
            size = 1
            for i in ALLOC[name]:
                v = _as_int(args[i]) if i < len(args) else UNK
                size = UNK if (v is UNK or size is UNK) else size * v
            nbytes = int(np.max(size)) if size is not UNK else 0
            obj = Obj(f"{name}#{len(self.objects)}", "heap", nbytes, zeroed=name == "calloc")
            self.objects.append(obj)
            self.heap_events.append(("new", obj))
            return Ptr(obj, 0)
        if name in FREE and args and isinstance(args[0], Ptr) and isinstance(args[0].obj, Obj):
            self.heap_events.append(("free", args[0].obj))
            return UNK
        if name in RANDOM:                                             # input drawn from its distribution
            self._libc_random()
            draws = self.rng.integers(0, RANDOM[name] + 1, self.L, dtype=np.int64)
            return int(draws[0]) if self.L == 1 else draws
        if name in ("memset", "memcpy", "memmove") and len(args) == 3:
            n = _as_int(args[2])
            for i, write in ((0, True), (1, False)) if name != "memset" else ((0, True),):
                p = args[i]
                if isinstance(p, Ptr) and isinstance(p.obj, Obj) and np.ndim(p.off) == 0 and p.off is not UNK \
                        and n is not UNK and np.ndim(n) == 0:
                    for off in range(int(p.off), int(p.off) + int(n), 8):
                        self.access(Loc(p.obj, off, 8), write=write)
                else:
                    self.access(Loc(p.obj if isinstance(p, Ptr) else UNK, UNK, 8), write=write)
            return args[0]
        return UNK

    def _libc_random(self):
        """Charge the references a rand() call makes inside the C library (once per batch point)."""
        if not self.charging:
            return
        if not hasattr(self, "libc_random"):
            self.libc_random = Obj("libc.random", "global", RAND_HOT_BYTES + 4 * RAND_TABLE_WORDS)
            self.objects.append(self.libc_random)
        weight = float(self.w.sum())
        hot = np.arange(0, RAND_HOT_BYTES, 4)
        table = RAND_HOT_BYTES + 4 * np.arange(RAND_TABLE_WORDS)
        self.libc_random.add(hot, 4, np.full(len(hot), weight * RAND_HOT_REFS / len(hot)))
        self.libc_random.add(table, 4, np.full(len(table), weight * RAND_TABLE_REFS / len(table)))
        self.total += weight * (RAND_HOT_REFS + RAND_TABLE_REFS)

    # ------------------------------------------------------------ statements

    def exec(self, c):
        kind = c.kind
        if kind == K.COMPOUND_STMT:
            for kid in c.get_children():
                self.exec(kid)
        elif kind == K.DECL_STMT:
            for decl in c.get_children():
                if decl.kind == K.VAR_DECL:
                    self._exec_decl(decl)
        elif kind == K.FOR_STMT:
            self._for(c)
        elif kind == K.WHILE_STMT:
            cond, body = ca.children(c)
            self._loop_general(None, cond, None, body)
        elif kind == K.DO_STMT:
            body, cond = ca.children(c)
            self._loop_general(None, cond, None, body, do=True)
        elif kind == K.IF_STMT:
            self._if(c)
        elif kind == K.RETURN_STMT:
            kids = ca.children(c)
            v = self.eval(kids[0]) if kids else UNK
            frame = self.frames[-1]
            if "__ret__" in frame.vars:
                frame.vars["__ret__"][1] = v
        elif kind in (K.NULL_STMT, K.BREAK_STMT, K.CONTINUE_STMT, K.LABEL_STMT, K.GOTO_STMT):
            pass
        elif kind.is_expression():
            self.eval(c)
        else:
            for kid in c.get_children():
                self.exec(kid)

    def _exec_decl(self, decl):
        if decl.storage_class.name == "STATIC":
            self._lookup(decl)
            return
        init = self._initializer(decl)
        value = self.eval(init) if init is not None and init.kind != K.INIT_LIST_EXPR else UNK
        if ca.is_float(decl.type) and not isinstance(value, (Ptr, Real)):
            value = UNK
        slot = self._declare_local(decl, value if not ca.is_array(decl.type) else UNK)
        if init is not None and not ca.is_array(decl.type):
            self.access(slot, write=True)

    def _if(self, c):
        kids = ca.children(c)
        cond_c, then_c = kids[0], kids[1]
        else_c = kids[2] if len(kids) > 2 else None
        cond = _as_int(self.eval(cond_c))
        if cond is UNK:
            self.data_branches += 1
            self.uncertainly(lambda: self.run_subset(np.arange(self.L), lambda: self.exec(then_c), 0.5))
            if else_c is not None:
                self.uncertainly(lambda: self.run_subset(np.arange(self.L), lambda: self.exec(else_c), 0.5))
            return
        if np.ndim(cond) == 0:
            if cond:
                self.exec(then_c)
            elif else_c is not None:
                self.exec(else_c)
            return
        mask = np.broadcast_to(cond != 0, (self.L,))
        self.run_subset(np.nonzero(mask)[0], lambda: self.exec(then_c))
        if else_c is not None:
            self.run_subset(np.nonzero(~mask)[0], lambda: self.exec(else_c))

    # ------------------------------------------------------------ loops

    def _for(self, c):
        init, cond, inc, body = ca.for_parts(c)
        if init is not None:
            self.exec(init) if init.kind == K.DECL_STMT else self.eval(init)
        pattern = _counted(cond, inc, body)
        if pattern is None:
            return self._loop_general(None, cond, inc, body)
        var, op, bound_c, step = pattern
        entry = self._lookup(var)
        start = _as_int(entry[1]) if entry else UNK
        bound = _as_int(self.no_charge(lambda: self.eval(bound_c)))
        if _is_unk(start, bound):
            return self._loop_general(None, cond, inc, body)
        trips = _trips(start, bound, op, step)
        if trips is None:
            return self._loop_general(None, cond, inc, body)
        trips = np.broadcast_to(np.maximum(trips, 0), (self.L,)).astype(np.int64)
        if self.sequential_depth == 0 and int(trips.sum()) <= SEQUENTIAL_LIMIT and _memory_carried(body):
            self.sequential_depth += 1                                 # iterations depend on each other
            try:
                return self._loop_general(None, cond, inc, body)
            finally:
                self.sequential_depth -= 1
        # the condition is tested trips + 1 times, the increment runs trips times
        self.with_weights(self.w * (trips + 1), lambda: self.eval(cond))
        saved = entry[1]
        self.with_weights(self.w * trips, lambda: self.eval(inc))
        entry[1] = saved

        total = int(trips.sum())
        if total:
            cum = np.cumsum(trips)
            before = cum - trips
            start_arr = np.broadcast_to(start, (self.L,))
            saved_frames, saved_w = self.frames, self.w
            try:
                for a in range(0, total, CHUNK):
                    g = np.arange(a, min(a + CHUNK, total))
                    point = np.searchsorted(cum, g, side="right")
                    within = g - before[point]
                    self.frames = _map_frames_with(saved_frames, lambda arr: arr[point])
                    self.w = saved_w[point]
                    self._link(self.w, saved_w, point)
                    self._lookup(var)[1] = start_arr[point] + step * within
                    self.exec(body)
                    self._drop(self.w)
            finally:
                self.frames, self.w = saved_frames, saved_w
        entry = self._lookup(var)
        entry[1] = _SCALAR(start + step * trips) if np.ndim(start) or np.ndim(trips) else start + step * trips
        self._forget_float_writes(body)

    def _loop_general(self, init, cond, inc, body, do=False):
        """Run a loop one iteration at a time for all batch points still in it."""
        idx = np.arange(self.L)
        for iteration in range(MAX_ITERATIONS):
            if cond is not None and not (do and iteration == 0):
                value = self._eval_condition(idx, cond)
                if value is UNK:
                    self.data_loops += 1
                    self.uncertainly(lambda: self.run_subset(idx, lambda: self._iteration(body, inc),
                                                             GUESS_TRIPS))
                    return
                if np.ndim(value) == 0:
                    if not value:
                        return
                else:
                    idx = idx[np.asarray(value) != 0]
                    if len(idx) == 0:
                        return
            self.run_subset(idx, lambda: self._iteration(body, inc))
        self.data_loops += 1  # gave up: too many iterations

    def _forget_float_writes(self, body, keep=()):
        """After a loop run as a batch: floating-point scalars the body assigns keep no value (their
        updates in the batch are not carried from one iteration to the next)."""
        if not self.REALS:
            return
        for n in _walk_nodes(body):
            if n.kind in (K.BINARY_OPERATOR, K.COMPOUND_ASSIGNMENT_OPERATOR) and \
                    (n.kind == K.COMPOUND_ASSIGNMENT_OPERATOR or ca.binary_op(n) == "="):
                d = _decl_ref(ca.children(n)[0])
                if d is not None and ca.is_float(d.type) and _key(d) not in keep:
                    entry = self._lookup(d)
                    if entry is not None and isinstance(entry[1], Real):
                        entry[1] = UNK

    def _eval_condition(self, idx, cond):
        return self.run_subset(idx, lambda: _as_int(self.eval(cond)))

    def _iteration(self, body, inc):
        self.exec(body)
        if inc is not None:
            self.eval(inc)


def _walk_nodes(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.get_children())


def _map_frames_with(frames, f):
    def mapv(v):
        if isinstance(v, np.ndarray):
            return f(v)
        if isinstance(v, Ptr) and isinstance(v.off, np.ndarray):
            return Ptr(v.obj, f(v.off), v.src)
        if isinstance(v, Real) and isinstance(v.v, np.ndarray):
            return Real(f(v.v))
        if isinstance(v, Partial):
            return Partial(f(v.v), f(v.known))
        return v

    return [Frame(fr.func, fr.base, {k: [s, mapv(v)] for k, (s, v) in fr.vars.items()}) for fr in frames]


# ---------------------------------------------------------------- loop analysis

_FLIP = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "!=": "!="}


def _decl_ref(c):
    c = ca.strip(c) if c is not None else None
    if c is not None and c.kind == K.DECL_REF_EXPR and c.referenced is not None \
            and c.referenced.kind in (K.VAR_DECL, K.PARM_DECL):
        return c.referenced
    return None


def _counted(cond, inc, body):
    """(var, op, bound, step) if the loop is `for (...; v op bound; v += step)` with v and the
    variables of bound not written in the body and no integer variable carried between
    iterations; None otherwise."""
    if cond is None or inc is None:
        return None
    cond = ca.strip(cond)
    if cond.kind != K.BINARY_OPERATOR or ca.binary_op(cond) not in _FLIP:
        return None
    lhs, rhs = ca.children(cond)
    op = ca.binary_op(cond)
    var = _decl_ref(lhs)
    bound = rhs
    if var is None:
        var, bound, op = _decl_ref(rhs), lhs, _FLIP[op]
    if var is None:
        return None
    step = _step(inc, var)
    if step is None or step == 0:
        return None
    first, written = _events(body)
    vk = _key(var)
    if vk in written:
        return None
    if any(_key(d) in written for d in _refs(bound)):
        return None
    for k in written:  # an integer scalar read before it is written carries a value between iterations
        if first.get(k) == "r":
            return None
    return var, op, bound, step


def _step(inc, var):
    inc = ca.strip(inc)
    vk = _key(var)
    if inc.kind == K.UNARY_OPERATOR:
        op = ca.unary_op(inc)
        target = _decl_ref(ca.children(inc)[0])
        if target is not None and _key(target) == vk and op in ("post++", "pre++", "post--", "pre--"):
            return 1 if "++" in op else -1
    if inc.kind == K.COMPOUND_ASSIGNMENT_OPERATOR:
        lhs, rhs = ca.children(inc)
        target = _decl_ref(lhs)
        v = ca.evaluate(rhs)
        if target is not None and _key(target) == vk and isinstance(v, int):
            op = ca.compound_op(inc)
            return v if op == "+" else -v if op == "-" else None
    if inc.kind == K.BINARY_OPERATOR and ca.binary_op(inc) == "=":
        lhs, rhs = ca.children(inc)
        target = _decl_ref(lhs)
        rhs = ca.strip(rhs)
        if target is not None and _key(target) == vk and rhs.kind == K.BINARY_OPERATOR \
                and ca.binary_op(rhs) in ("+", "-"):
            a, b = ca.children(rhs)
            if _decl_ref(a) is not None and _key(_decl_ref(a)) == vk:
                v = ca.evaluate(b)
                if isinstance(v, int):
                    return v if ca.binary_op(rhs) == "+" else -v
    return None


def _trips(start, bound, op, step):
    if op == "!=":
        op = "<" if step > 0 else ">"
    if step > 0 and op == "<":
        return (bound - start + step - 1) // step
    if step > 0 and op == "<=":
        return (bound - start) // step + 1
    if step < 0 and op == ">":
        return (start - bound - step - 1) // (-step)
    if step < 0 and op == ">=":
        return (start - bound) // (-step) + 1
    return None


def _refs(c):
    out = []
    if c is None:
        return out
    stack = [c]
    while stack:
        n = stack.pop()
        d = _decl_ref(n) if n.kind == K.DECL_REF_EXPR else None
        if d is not None:
            out.append(d)
        stack.extend(n.get_children())
    return out


def _subscript_base(c):
    """The declaration of the array (or pointer) variable at the root of a subscript chain."""
    c = ca.strip(c)
    while c.kind == K.ARRAY_SUBSCRIPT_EXPR:
        c = ca.strip(ca.children(c)[0])
    return _decl_ref(c)


def _memory_carried(body):
    """Does the loop body write an integer array (other than by += / ++) that it also reads, or
    read one it updates? Then iterations depend on each other through memory and the loop must
    run one iteration at a time."""
    written, updated, read = set(), set(), set()

    def integer(c):
        return Interpreter._tracked(c.type)

    def walk(n, role=None):
        kind = n.kind
        if kind == K.BINARY_OPERATOR and ca.binary_op(n) == "=":
            lhs, rhs = ca.children(n)
            target = ca.strip(lhs)
            if target.kind == K.ARRAY_SUBSCRIPT_EXPR:
                base = _subscript_base(target)
                if base is not None and integer(target):
                    written.add(_key(base))
                for kid in ca.children(target)[1:]:
                    walk(kid)
                walk(ca.children(target)[0], "base")
            else:
                walk(lhs, "lhs")
            walk(rhs)
            return
        if kind == K.COMPOUND_ASSIGNMENT_OPERATOR or (
                kind == K.UNARY_OPERATOR and ca.unary_op(n) in ("post++", "post--", "pre++", "pre--")):
            kids = ca.children(n)
            target = ca.strip(kids[0])
            if target.kind == K.ARRAY_SUBSCRIPT_EXPR:
                base = _subscript_base(target)
                if base is not None and integer(target):
                    updated.add(_key(base))
                for kid in ca.children(target)[1:]:
                    walk(kid)
                for kid in kids[1:]:
                    walk(kid)
                return
        if kind == K.ARRAY_SUBSCRIPT_EXPR and role != "base":
            base = _subscript_base(n)
            if base is not None and integer(n):
                read.add(_key(base))
            for kid in ca.children(n)[1:]:
                walk(kid)
            walk(ca.children(n)[0], "base")
            return
        for kid in n.get_children():
            walk(kid)

    walk(body)
    return bool((written | updated) & read) or bool(written & updated)


def _events(body):
    """First event ('r' or 'w') of each integer or pointer scalar in evaluation order, and
    the set of scalars written. Writes through subscripts or pointers are not scalar writes."""
    first, written = {}, set()

    def note(decl, event):
        if ca.is_float(decl.type) or ca.is_array(decl.type):
            return
        k = _key(decl)
        first.setdefault(k, event)
        if event == "w":
            written.add(k)

    def walk(n):
        kind = n.kind
        if kind == K.BINARY_OPERATOR and ca.binary_op(n) == "=":
            lhs, rhs = ca.children(n)
            walk(rhs)
            d = _decl_ref(lhs)
            if d is not None:
                note(d, "w")
            else:
                walk(lhs)
            return
        if kind == K.COMPOUND_ASSIGNMENT_OPERATOR or (
                kind == K.UNARY_OPERATOR and ca.unary_op(n) in ("post++", "post--", "pre++", "pre--", "&")):
            kids = ca.children(n)
            d = _decl_ref(kids[0])
            for k in kids[1:]:
                walk(k)
            if d is not None:
                note(d, "r")
                note(d, "w")
            else:
                walk(kids[0])
            return
        if kind == K.VAR_DECL:
            for k in n.get_children():
                walk(k)
            note(n, "w")
            return
        if kind == K.FOR_STMT:
            init, cond, inc, body_ = ca.for_parts(n)
            for part in (init, cond, body_, inc):
                if part is not None:
                    walk(part)
            return
        if kind == K.DECL_REF_EXPR:
            d = _decl_ref(n)
            if d is not None:
                note(d, "r")
            return
        for k in n.get_children():
            walk(k)

    walk(body)
    return first, written


# ---------------------------------------------------------------- entry point


def _place_heap(events, heap_top):
    """Addresses of the heap objects under glibc's allocator (skeleton.Process): {Obj: Block}."""
    from .skeleton import Process

    process, blocks = Process(heap_top=heap_top), {}
    for event, obj in events:
        if event == "new":
            blocks[obj] = process.new(obj.nbytes)
        elif obj in blocks:
            process.delete(blocks[obj])
    return blocks


def run(tu, entry="main", argc=1, footprint="starts", seed=0, heap_top=0, make=None, argv=None):
    """Run the program from `entry` and return its access-count spectrum (footprint: starts or bytes).
    seed: for the values drawn in place of the C library's random numbers. With bytes, heap blocks
    are placed by glibc's allocator (skeleton.Process, starting with heap_top free bytes), so a
    block that reuses a freed block's addresses adds no new bytes. make: a factory for the
    interpreter (the C++ one), called with (tu, seed)."""
    interp = (make or Interpreter)(tu, seed)
    if argv is not None:                                               # the command line, argv[0] first
        interp.argv = tuple(str(a) for a in argv)
        argc = len(interp.argv)
    fdef = interp.functions.get(entry)
    if fdef is None:
        raise ValueError(f"no definition of {entry}")
    frame = Frame(fdef, FRAME_BYTES)
    interp.frames.append(frame)
    for p in [p for p in fdef.get_children() if p.kind == K.PARM_DECL]:
        value = argc if p.spelling == "argc" else (Ptr(UNK, UNK) if ca.is_pointer(p.type) else UNK)
        if p.spelling == "argv" and argv is not None and hasattr(interp, "_argv_array"):
            value = interp._argv_array()
        interp.access(interp._declare_local(p, value, pointer=ca.is_array(p.type)), write=True)
    frame.vars["__ret__"] = [None, UNK]
    for kid in fdef.get_children():
        if kid.kind == K.COMPOUND_STMT:
            interp.exec(kid)
    counts, sizes, objects = [], [], []
    placed = _place_heap(interp.heap_events, heap_top) if footprint == "bytes" else {}
    spaces = {}
    for obj in interp.objects:
        c, s = obj.addresses(footprint)
        if len(c):
            objects.append({"name": obj.name, "kind": obj.kind, "addresses": len(c), "bytes": float(s.sum()),
                            "references": float(c.sum())})
            if obj in placed:                                          # merged by address below
                block = placed[obj]
                cover = obj.coverage()
                space = spaces.setdefault(block.space, np.zeros(0))
                end = block.user + len(cover)
                if len(space) < end:
                    space = np.concatenate([space, np.zeros(end - len(space))])
                space[block.user:end] += cover
                spaces[block.space] = space
                continue
            counts.append(c)
            sizes.append(s)
    for space in spaces.values():
        c, s = group_words(space)
        counts.append(c)
        sizes.append(s)
    return Result(np.concatenate(counts) if counts else np.zeros(0),
                  np.concatenate(sizes) if sizes else np.zeros(0), objects, interp.total, interp.unresolved,
                  interp.uncertain, interp.data_loops, interp.data_branches)
