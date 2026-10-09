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

References the interpreter cannot place are counted, not guessed: an address
that depends on data (indirect subscript, pointer chasing) is unresolved, and
references inside a loop with a data-dependent trip count or under a
data-dependent branch are uncertain. Coverage is the share of references that
are neither.
"""

from dataclasses import dataclass, field

import numpy as np

from . import clangast as ca
from .clangast import K

CHUNK = 1 << 21  # iteration points per batch
MAX_ITERATIONS = 100_000  # iterations of a loop run one at a time
MAX_DEPTH = 64
FRAME_BYTES = 8192  # stack frame size; frames at the same call depth share addresses
STACK_ARRAY_LIMIT = 1024  # larger local arrays get their own object
GUESS_TRIPS = 1  # trips charged for a loop whose trip count depends on data


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
        hi = int(off.max()) + 1
        arr = self._grow(size, hi)
        arr[:hi] += np.bincount(off, weights=w, minlength=hi)

    def addresses(self):
        """(counts, sizes) per start address with at least one reference."""
        if not self.by_size:
            return np.zeros(0), np.zeros(0)
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


@dataclass
class Ptr:
    obj: object  # Obj or UNK
    off: object  # int, int array or UNK (bytes)


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
    return (decl.spelling, decl.location.offset, str(decl.location.file))


def _as_int(v):
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
    def __init__(self, tu):
        self.tu = tu
        self.w = np.ones(1)
        self.stack = Obj("stack", "stack", MAX_DEPTH * FRAME_BYTES)
        self.globals_obj = Obj("globals", "global")
        self.rodata = Obj("rodata", "rodata")
        self.objects = [self.stack, self.globals_obj, self.rodata]
        self.rodata_slots = {}
        self.layouts = {}  # function key -> {var key: offset}
        self.frames = [Frame(None, 0)]
        self.charging = True
        self.uncertain_depth = 0
        self.total = self.unresolved = self.uncertain = 0.0
        self.data_loops = self.data_branches = 0
        self.functions = {}
        self._global_off = 0
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
                return Ptr(v.obj, f(v.off))
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
            return Ptr(old.obj, off)
        o, n = _as_int(old), _as_int(new)
        if _is_unk(o, n):
            return UNK if not (old is UNK and new is UNK) else UNK
        if np.ndim(o) == 0 and np.ndim(n) == 0 and o == n:
            return o
        full = np.array(np.broadcast_to(o, L), dtype=np.int64)
        full[idx] = n
        return full

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
            obj = Obj(c.spelling, "global", size)
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
            return Loc(base.obj, base.off + idx * size, size, array)
        if kind == K.UNARY_OPERATOR and ca.unary_op(c) == "*":
            p = self.eval(ca.children(c)[0])
            size = ca.type_size(c.type) or 8
            if not isinstance(p, Ptr):
                return Loc(UNK, UNK, size, ca.is_array(c.type))
            return Loc(p.obj, p.off, size, ca.is_array(c.type))
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
                return Loc(p.obj, p.off + fo, size, array)
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
                entry[1] = value if (isinstance(value, Ptr) or not ca.is_float(target.type)) else UNK
        return loc

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

    # ------------------------------------------------------------ rvalues

    def eval(self, c):
        kind = c.kind
        if kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR):
            kids = ca.children(c)
            if len(kids) == 1:
                v = self.eval(kids[0])
                return UNK if ca.is_float(c.type) and not isinstance(v, Ptr) else v
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
            return UNK
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
                return UNK
            return v
        if kind == K.UNARY_OPERATOR:
            return self._unary(c)
        if kind == K.BINARY_OPERATOR:
            return self._binary(c)
        if kind == K.COMPOUND_ASSIGNMENT_OPERATOR:
            lhs, rhs = ca.children(c)
            loc = self.lvalue(lhs)
            self.access(loc)
            old = self._value_of(lhs)
            r = self.eval(rhs)
            new = self._arith(ca.compound_op(c), old, r, c)
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
                entry[1] = value if (isinstance(value, Ptr) or not ca.is_float(target.type)) else UNK

    def _arith(self, op, a, b, c):
        if isinstance(a, Ptr) or isinstance(b, Ptr):
            return self._ptr_arith(op, a, b, c)
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
            return UNK
        if op in ("+", "-"):
            p, n = (a, b) if isinstance(a, Ptr) else (b, a)
            n = _as_int(n)
            size = ca.pointee_size(c.type) or 1
            if _is_unk(n, p.off):
                return Ptr(p.obj, UNK)
            return Ptr(p.obj, _SCALAR(p.off + n * size if op == "+" else p.off - n * size))
        return UNK

    def _unary(self, c):
        op = ca.unary_op(c)
        kid = ca.children(c)[0]
        if op in ("post++", "post--", "pre++", "pre--"):
            loc = self.lvalue(kid)
            self.access(loc)
            old = self._value_of(kid)
            delta = 1 if "++" in op else -1
            if isinstance(old, Ptr):
                size = ca.pointee_size(kid.type) or 1
                new = Ptr(old.obj, UNK if old.off is UNK else _SCALAR(old.off + delta * size))
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
        v = _as_int(self.eval(kid))
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

    def _library(self, name, args_c):
        args = [self.eval(a) for a in args_c]
        if name in ALLOC:
            size = 1
            for i in ALLOC[name]:
                v = _as_int(args[i]) if i < len(args) else UNK
                size = UNK if (v is UNK or size is UNK) else size * v
            nbytes = int(np.max(size)) if size is not UNK else 0
            obj = Obj(f"{name}#{len(self.objects)}", "heap", nbytes)
            self.objects.append(obj)
            return Ptr(obj, 0)
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
        if ca.is_float(decl.type) and not isinstance(value, Ptr):
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
                    self._lookup(var)[1] = start_arr[point] + step * within
                    self.exec(body)
            finally:
                self.frames, self.w = saved_frames, saved_w
        entry = self._lookup(var)
        entry[1] = _SCALAR(start + step * trips) if np.ndim(start) or np.ndim(trips) else start + step * trips

    def _loop_general(self, init, cond, inc, body, do=False):
        """Run a loop one iteration at a time for all batch points still in it."""
        idx = np.arange(self.L)
        for iteration in range(MAX_ITERATIONS):
            if cond is not None and not (do and iteration == 0):
                value = self.run_subset(idx, lambda: _as_int(self.eval(cond)))
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

    def _iteration(self, body, inc):
        self.exec(body)
        if inc is not None:
            self.eval(inc)


def _map_frames_with(frames, f):
    def mapv(v):
        if isinstance(v, np.ndarray):
            return f(v)
        if isinstance(v, Ptr) and isinstance(v.off, np.ndarray):
            return Ptr(v.obj, f(v.off))
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


def run(tu, entry="main", argc=1):
    """Run the program from `entry` and return its access-count spectrum."""
    interp = Interpreter(tu)
    fdef = interp.functions.get(entry)
    if fdef is None:
        raise ValueError(f"no definition of {entry}")
    frame = Frame(fdef, FRAME_BYTES)
    interp.frames.append(frame)
    for p in [p for p in fdef.get_children() if p.kind == K.PARM_DECL]:
        value = argc if p.spelling == "argc" else (Ptr(UNK, UNK) if ca.is_pointer(p.type) else UNK)
        interp.access(interp._declare_local(p, value, pointer=ca.is_array(p.type)), write=True)
    frame.vars["__ret__"] = [None, UNK]
    for kid in fdef.get_children():
        if kid.kind == K.COMPOUND_STMT:
            interp.exec(kid)
    counts, sizes, objects = [], [], []
    for obj in interp.objects:
        c, s = obj.addresses()
        if len(c):
            counts.append(c)
            sizes.append(s)
            objects.append({"name": obj.name, "kind": obj.kind, "addresses": len(c), "bytes": float(s.sum()),
                            "references": float(c.sum())})
    return Result(np.concatenate(counts) if counts else np.zeros(0),
                  np.concatenate(sizes) if sizes else np.zeros(0), objects, interp.total, interp.unresolved,
                  interp.uncertain, interp.data_loops, interp.data_branches)
