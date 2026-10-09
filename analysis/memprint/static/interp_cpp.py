"""C++ on top of the C interpreter (interp.py), for programs like GAP built with -O3.

What C++ adds, and how it is run:
- Objects: fields at their offsets, `this`, member calls (including operators
  and lambdas' operator()), constructors with member initialisers, default
  member initialisers and delegation, and destructors at the end of the
  scope (libclang does not show those calls). Objects returned by value are
  constructed in the caller's storage; a returned local is not destroyed.
- References: a reference variable or parameter holds the address it binds
  to; a call returning a reference is an lvalue.
- Pointers stored in memory are remembered (object and offset), so CSR index
  arrays of pointers resolve.
- new / delete, range-for (through begin() and end()), templates (libclang
  gives instantiated bodies with concrete types; class-template patterns are
  run with fields resolved on the concrete type).
- The standard library is not interpreted: std::sort, unique, remove, copy,
  shuffle, min, max, swap, numeric_limits, the Mersenne Twister and
  uniform_int_distribution are modelled (cost and values); streams and
  strings are ignored.
- -O3 counting: scalar locals and small objects (up to REGISTER_BYTES, e.g.
  an Edge, a Neighborhood, a pvector's three pointers) live in registers:
  each batch point has its own copy and their accesses are not charged.
  Accesses through pointers, to arrays and to large objects are charged.
- Inputs that are not memory-relevant (command-line accessors) come from
  `stubs`: method name -> value.
"""

import ctypes
import re
from dataclasses import dataclass

import numpy as np

from . import clangast as ca
from .clangast import K
from .interp import (FREE, SENTINEL, UNK, Frame, Interpreter, Loc, Obj, Ptr, Real, _as_int, _is_unk, _key, _SCALAR, _real)
from .containers import Containers, element_type
from .skeleton import sort_refs

REGISTER_BYTES = 64
# User functions with the meaning of an atomic operation (GAP's platform_atomics.h, also its serial
# fallbacks): run as one read-modify-write with sequential results when a vectorised loop repeats a
# location, instead of interpreting their statements one batch at a time.
ATOMIC_IDIOMS = {"fetch_and_add", "compare_and_swap", "atomic_fetch_add"}
# Library functions that do not write the program's memory (registers survive them).
PURE = {"fabs", "abs", "sqrt", "exp", "log", "pow", "floor", "ceil", "min", "max", "numeric_limits", "size",
        "begin", "end", "move", "forward", "operator==", "operator!=", "operator<", "operator>"}
NULL_UID = -2
# C math functions evaluated on floating-point scalars: name -> (arguments, function)
MATH = {"pow": (2, np.power), "sqrt": (1, np.sqrt), "cbrt": (1, np.cbrt), "exp": (1, np.exp), "exp2": (1, np.exp2),
        "log": (1, np.log), "log2": (1, np.log2), "log10": (1, np.log10), "fabs": (1, np.abs), "abs": (1, np.abs),
        "labs": (1, np.abs), "llabs": (1, np.abs), "floor": (1, np.floor), "ceil": (1, np.ceil),
        "round": (1, np.round), "trunc": (1, np.trunc), "fmin": (2, np.fmin), "fmax": (2, np.fmax),
        "sin": (1, np.sin), "cos": (1, np.cos), "tan": (1, np.tan), "atan": (1, np.arctan),
        "atan2": (2, np.arctan2), "hypot": (2, np.hypot), "fmod": (2, np.fmod)}


@dataclass
class Closure:
    lambda_cursor: object
    frame: object


@dataclass
class Func:
    """A function used as a value (passed to a template's callable parameter)."""
    decl: object


@dataclass
class Str:
    text: str


@dataclass
class InMemory:
    """A scalar local whose address was taken: its value now lives at loc (one slot per batch point)."""

    loc: object


def _record(type_):
    return type_.get_canonical().kind == ca.ci.TypeKind.RECORD


def _reference(type_):
    """A reference type, also behind a typedef (std::vector<T>::reference)."""
    return type_.get_canonical().kind in (ca.ci.TypeKind.LVALUEREFERENCE, ca.ci.TypeKind.RVALUEREFERENCE)


def _referee(type_):
    return type_.get_canonical().get_pointee() if _reference(type_) else type_


def _system(cursor):
    """Declared in a system header (the C or C++ standard library): modelled, not interpreted."""
    if cursor is None or cursor.location.file is None:
        return True
    try:
        return bool(cursor.location.is_in_system_header)
    except Exception:
        return str(cursor.location.file.name).startswith("/usr/")


def _strip(c):
    while c.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR) and len(ca.children(c)) == 1:
        c = ca.children(c)[0]
    return c


_TEMPLATE = None


def _specialized_template(cursor):
    """clang_getSpecializedCursorTemplate: the pattern a class or method was instantiated from."""
    global _TEMPLATE
    if _TEMPLATE is None:
        lib = ca.lib()
        lib.clang_getSpecializedCursorTemplate.argtypes = [ca.ci.Cursor]
        lib.clang_getSpecializedCursorTemplate.restype = ca.ci.Cursor
        lib.clang_getSpecializedCursorTemplate.errcheck = ca.ci.Cursor.from_result
        _TEMPLATE = lib.clang_getSpecializedCursorTemplate
    try:
        return _TEMPLATE(cursor)
    except Exception:
        return None


def _class_members(type_):
    """Member cursors of a class: from its declaration, else from the template it instantiates."""
    decl = type_.get_canonical().get_declaration()
    members = list(decl.get_children())
    if not any(m.kind in (K.CXX_METHOD, K.CONSTRUCTOR, K.DESTRUCTOR, K.FIELD_DECL) for m in members):
        pattern = _specialized_template(decl)
        if pattern is not None and pattern.kind != K.NO_DECL_FOUND:
            members = list(pattern.get_children())
    return members


def _implicit_copy(method, fdef):
    """A copy / move constructor or assignment with no code of its own: compiler-generated (libclang
    shows no parameter for it) or with an empty body. Its effect is a memberwise copy."""
    args = list(method.type.argument_types()) if method.type.kind == ca.ci.TypeKind.FUNCTIONPROTO else []
    parent = method.semantic_parent
    if len(args) != 1 or not _reference(args[0]) or parent is None:
        return False
    if args[0].get_pointee().get_canonical().get_declaration() != parent.type.get_canonical().get_declaration():
        return False
    if not any(p.kind == K.PARM_DECL for p in method.get_children()):
        return True
    body = _body(fdef)
    return body is None or not list(body.get_children())


def _body(fdef):
    if fdef is None:
        return None
    for k in fdef.get_children():
        if k.kind == K.COMPOUND_STMT:
            return k
    return None


def _provenance(obj, off):
    """Where a pointer was loaded from, the same in every batch: a field of a small object (its offset),
    or an element of an array (which element is not recorded)."""
    if np.ndim(off) == 0 and (obj.nbytes or 0) <= 256:
        return (id(obj), int(off))
    return (id(obj), None)


class _Batch:
    """The load cache of one batch: {object id: {(size, provenance): [offsets per point or scalar]}}."""
    __slots__ = ("w", "entries", "link", "absorbed")
    LIMIT = 6

    def __init__(self, w):
        self.w = w
        self.entries = {}
        self.link = None  # (parent weights, index into the parent's points or None)
        self.absorbed = {}  # (object id, key) -> the entry that subsets' latest loads are written into

    def remember(self, obj_key, key, off):
        lst = self.entries.setdefault(obj_key, {}).setdefault(key, [])
        lst.insert(0, np.array(off) if np.ndim(off) else int(off))
        del lst[self.LIMIT:]

    def absorb(self, child, idx, n):
        """A subset's latest loads, as this batch's (n points; the subset's are idx). A point keeps one
        value per pointer, the latest, as a register would: successive subsets (the iterations of a while
        loop's condition) overwrite one entry instead of adding entries."""
        for obj_key, d in child.entries.items():
            for key, lst in d.items():
                if not lst:
                    continue
                lst_here = self.entries.setdefault(obj_key, {}).setdefault(key, [])
                full = self.absorbed.get((obj_key, key))
                if full is None or not any(e is full for e in lst_here):
                    full = np.full(n, -1, dtype=np.int64)
                    lst_here.insert(0, full)
                    del lst_here[self.LIMIT:]
                    self.absorbed[(obj_key, key)] = full
                full[slice(None) if idx is None else idx] = lst[0]


class CppInterpreter(Containers, Interpreter):
    REALS = True
    LOAD_CACHE = "point"  # "point": per-point load cache with parent batches; "batch": one record per batch
    def __init__(self, tu, seed=0, stubs=None, opt=3):
        super().__init__(tu, seed)
        self.stubs = stubs or {}
        self.opt = opt
        self.registry = {}
        self.null = Obj("null", "null")
        self.null.uid = NULL_UID
        self.scratch = Obj("registers", "register")                     # never charged, never in the spectrum
        self.scratch_top = 0
        self.this_types = []
        self.scopes = []
        self.engines = {}
        self.probes = []
        self._caches = {}
        self._batches = {}  # id(w) -> _Batch: per-point load cache of a batch and its parent link
        self.loop_flags = []  # (break key, continue key) of the loops being run one iteration at a time
        self._maps = {}

    # ------------------------------------------------------------ objects and pointers in memory

    def _uid(self, obj):
        if not hasattr(obj, "uid"):
            obj.uid = len(self.registry)
            self.registry[obj.uid] = obj
        return obj.uid

    def access(self, loc, write=False):
        if isinstance(loc.obj, Obj) and loc.obj.kind == "register":
            return
        if getattr(loc, "register", False):
            return
        if self.opt >= 3 and self.charging and isinstance(loc.obj, Obj) and loc.off is not UNK:
            if self.LOAD_CACHE == "point":
                return self._access_point(loc, write)
            # -O3 keeps a loaded or stored value in a register: a load of the same location in the
            # same batch of iterations, with no store to that object in between, costs nothing. Each
            # batch has its own record, kept while nested loops run.
            entry = self._caches.get(id(self.w))
            if entry is None or entry[0] is not self.w:
                if len(self._caches) > 64:
                    self._caches.clear()
                entry = self._caches[id(self.w)] = (self.w, {})
            cache = entry[1]
            off = loc.off
            # keyed by address and by where the base pointer came from: the compiler can only tell
            # that two accesses are the same through the same pointer
            key = (id(loc.obj), int(off) if np.ndim(off) == 0 else
                   (np.size(off), int(np.ravel(off)[0]), int(np.ravel(off)[-1]), int(np.sum(off))), loc.size,
                   getattr(loc, "src", None))
            if write:
                for _, other in self._caches.values():
                    for k in [k for k in other if k[0] == key[0]]:
                        del other[k]
                cache[key] = True
            elif key in cache:
                return
            else:
                cache[key] = True
        super().access(loc, write)

    # ------------------------------------------------------------ the per-point load cache (-O3)
    #
    # Every batch of points (loop iterations run together) records, for each (object, access size,
    # base-pointer provenance), the offsets its points loaded or stored. A point's load costs nothing if
    # that point, or the point it derives from in an enclosing batch, already accessed the offset
    # through the same pointer with no store to the object since: values loaded before a loop stay in
    # registers inside it (loop-invariant loads are hoisted), and a value a while condition loaded is
    # reused after the loop. Each iteration of a loop run one at a time gets its own batch, so an
    # iteration does not reuse what the previous one loaded. A store to an object forgets every entry
    # of that object; a library call that may write memory forgets everything.

    def _record(self, w=None, create=True):
        w = self.w if w is None else w
        rec = self._batches.get(id(w))
        if rec is not None and rec.w is not w:
            rec = None
        if rec is None and create:
            if len(self._batches) > 4096:
                self._batches.clear()
            rec = self._batches[id(w)] = _Batch(w)
        return rec

    def _link(self, child, parent, index):
        self._record(child).link = (parent, None if index is None else np.asarray(index))

    def _drop(self, w):
        rec = self._batches.get(id(w))
        if rec is not None and rec.w is w:
            del self._batches[id(w)]

    def _access_point(self, loc, write):
        obj_key = id(loc.obj)
        key = (loc.size, getattr(loc, "src", None))
        off = loc.off
        if write:
            for rec in self._batches.values():
                rec.entries.pop(obj_key, None)
            self._record().remember(obj_key, key, off)
            return Interpreter.access(self, loc, write)
        hit = self._hits(obj_key, key, off)
        self._record().remember(obj_key, key, off)
        if hit is None or not hit.any():
            return Interpreter.access(self, loc, write)
        if hit.all():
            return
        miss = ~hit
        saved = self.w
        self.w = saved[miss]
        try:
            Interpreter.access(self, Loc(loc.obj, np.asarray(off)[miss] if np.ndim(off) else off, loc.size,
                                         loc.array), write)
        finally:
            self.w = saved

    def _hits(self, obj_key, key, off):
        """Per point of the current batch: was this offset already accessed (through this pointer)?"""
        L = self.L
        offs = np.broadcast_to(np.asarray(off), (L,))
        hit = None
        rec, index = self._record(create=False), None                  # index: current points -> rec's points
        for _ in range(12):
            if rec is None:
                break
            for stored in rec.entries.get(obj_key, {}).get(key, ()):
                s = stored if index is None or np.ndim(stored) == 0 else stored[index]
                h = np.broadcast_to(np.asarray(s) == offs, (L,))
                hit = h.copy() if hit is None else (hit | h)
                if hit.all():
                    return hit
            if rec.link is None:
                break
            parent, pidx = rec.link
            if pidx is not None:
                index = pidx if index is None else pidx[index]
            rec = self._record(parent, create=False)
        return hit

    def run_subset(self, idx, fn, scale=1.0, propagate=False):
        if len(idx) == 0:
            return None
        if len(idx) == self.L and scale == 1.0:
            return fn()
        parent = self.w

        def inner():
            child = self.w
            self._link(child, parent, idx)
            try:
                return fn()
            finally:
                if propagate:                                          # a loop condition's loads
                    rec = self._record(child, create=False)
                    if rec is not None:
                        self._record(parent).absorb(rec, idx, len(parent))
                self._drop(child)
        return super().run_subset(idx, inner, scale)

    def with_weights(self, w, fn):
        parent = self.w

        def inner():
            child = self.w
            self._link(child, parent, None)
            try:
                return fn()
            finally:
                rec = self._record(child, create=False)
                if rec is not None:
                    self._record(parent).absorb(rec, None, len(parent))
                self._drop(child)
        return super().with_weights(w, inner)

    def _fresh(self, fn):
        """Run fn (one loop iteration) in a batch of its own whose parent is the current one."""
        if self.LOAD_CACHE != "point":
            self._forget_loads()
            return fn()
        parent = self.w
        self.w = parent.copy()
        self._link(self.w, parent, None)
        try:
            return fn()
        finally:
            self._drop(self.w)
            self.w = parent

    # ------------------------------------------------------------ loops: pointers and reductions

    @staticmethod
    def _combine_real(op, vals, v0, point, n, n_out):
        """A floating-point reduction over a batch: per enclosing point, its start value combined with
        its iterations' (each iteration started from that value)."""
        vals, v0 = _real(vals), _real(v0) if v0 is not None else UNK
        if _is_unk(vals, v0):
            return UNK
        vv = np.broadcast_to(np.asarray(vals.v, dtype=float), (n,))
        start = np.array(np.broadcast_to(np.asarray(v0.v, dtype=float), (n_out,)))
        if op == "max":
            np.maximum.at(start, point, vv)
            out = start
        elif op == "min":
            np.minimum.at(start, point, vv)
            out = start
        else:
            delta = np.zeros(n_out)
            np.add.at(delta, point, vv - start[point])
            out = start + delta
        return Real(float(out[0]) if n_out == 1 else out)

    def _scans(self, body):
        """Integer or pointer scalars that the body (nested loops and branches included) only changes by
        increments that do not read them (v += x, v -= x, v++, v--), that no condition reads, in a body
        that allocates nothing ({key: decl}): prefix sums, and output pointers that advance as rows are
        filled (*p++ = x). The loop runs as one batch in which iteration j starts from v0 plus the
        increments of the earlier iterations (_run_scans)."""
        from .interp import _decl_ref
        from .idioms import ALLOCATING

        incs, bad = {}, set()
        for n in _walk(body):
            if n.kind == K.CXX_NEW_EXPR or (n.kind == K.CALL_EXPR and n.spelling in ALLOCATING):
                return {}
            target, increment = None, False
            if n.kind == K.BINARY_OPERATOR and ca.binary_op(n) == "=":
                target = _decl_ref(ca.children(n)[0])
            elif n.kind == K.COMPOUND_ASSIGNMENT_OPERATOR:
                lhs, rhs = ca.children(n)
                target = _decl_ref(lhs)
                increment = target is not None and ca.compound_op(n) in ("+", "-") and not any(
                    _decl_ref(r) is not None and _key(_decl_ref(r)) == _key(target) for r in _walk(rhs))
            elif n.kind == K.UNARY_OPERATOR and ca.unary_op(n) in ("post++", "pre++", "post--", "pre--", "&"):
                target = _decl_ref(ca.children(n)[0])
                increment = ca.unary_op(n) != "&"
            if target is None:
                continue
            if increment:
                incs[_key(target)] = target
            else:
                bad.add(_key(target))
        for n in _walk(body):                                          # conditions must not read them
            cond = None
            if n.kind in (K.IF_STMT, K.WHILE_STMT, K.SWITCH_STMT, K.CONDITIONAL_OPERATOR):
                cond = ca.children(n)[0] if ca.children(n) else None
            elif n.kind == K.FOR_STMT:
                cond = ca.for_parts(n)[1]
            elif n.kind == K.DO_STMT:
                cond = ca.children(n)[-1]
            if cond is not None:
                for r in _walk(cond):
                    d = _decl_ref(r)
                    if d is not None:
                        bad.add(_key(d))
        return {k: d for k, d in incs.items() if k not in bad and not ca.is_float(d.type)}

    def _run_scans(self, entries, body, point, n, v0s, n_outer):
        """Scans in a batch of n iterations (point: their enclosing batch point). A first pass, which
        charges and stores nothing, gives each iteration's increment (v starts at 0, or at offset 0 of its
        object for a pointer); each iteration then starts from v0 plus the increments of the earlier
        iterations of its enclosing point. Returns the values after the loop."""
        live = {k: self._lookup_key(k) for k in entries}               # the batch's own copies
        for k, e in live.items():
            v0 = v0s[k]
            e[1] = Ptr(v0.obj, np.zeros(n, dtype=np.int64), v0.src) if isinstance(v0, Ptr) \
                else np.zeros(n, dtype=np.int64)
        # the first pass must leave every other variable as it was (sums it adds to, nested scans, locals)
        snapshot = [(fr, {k: (e, e[0], e[1]) for k, e in fr.vars.items()}) for fr in self.frames]
        saved_top, saved_probes = self.scratch_top, len(self.probes)
        self.no_charge(lambda: self.exec(body))
        incs = {}
        for k, e in live.items():
            v = e[1]
            v = v.off if isinstance(v, Ptr) else _as_int(v)
            if v is UNK:
                incs = None                                            # increments unknown: not a usable scan
                break
            incs[k] = np.broadcast_to(np.asarray(v, dtype=np.int64), (n,))
        for fr, entries in snapshot:
            for k in [k for k in fr.vars if k not in entries]:
                del fr.vars[k]
            for k, (e, slot, value) in entries.items():
                e[0], e[1] = slot, value
                fr.vars[k] = e
        self.scratch_top = saved_top
        del self.probes[saved_probes:]
        if incs is None:                                               # the values after the loop are unknown
            final = {}
            for k, e in live.items():
                v0 = v0s[k]
                e[1] = Ptr(v0.obj, UNK) if isinstance(v0, Ptr) else UNK
                final[k] = e[1]
            return final
        first = np.searchsorted(point, point, side="left")             # the first iteration of each point
        final = {}
        for k, e in live.items():
            v0 = v0s[k]
            base = np.broadcast_to(np.asarray(v0.off if isinstance(v0, Ptr) else v0), (n_outer,))
            excl = np.cumsum(incs[k]) - incs[k]
            excl = excl - excl[first]
            start = base[point] + excl
            e[1] = Ptr(v0.obj, start, v0.src) if isinstance(v0, Ptr) else start
            total = np.zeros(n_outer, dtype=np.int64)
            np.add.at(total, point, incs[k])
            out = base + total
            out = int(out[0]) if n_outer == 1 else out
            final[k] = Ptr(v0.obj, out, v0.src) if isinstance(v0, Ptr) else out
        return final

    def _reductions(self, body):
        """Integer scalars the body only updates as v = std::max(v, x), v = std::min(v, x), v += x,
        v -= x or v++: {key: op}. Such a loop can still run all iterations at once."""
        from .interp import _decl_ref

        found, uses = {}, {}

        nonint = set()                                                 # pointers advance by scans (_scans)

        def note(decl, op, n_refs):
            k = _key(decl)
            if ca.is_pointer(decl.type):
                nonint.add(k)
            if found.get(k, op) != op:
                found[k] = None
            else:
                found[k] = op
            uses[k] = uses.get(k, 0) - n_refs

        def walk(n):
            kind = n.kind
            if kind == K.BINARY_OPERATOR and ca.binary_op(n) == "=":
                lhs, rhs = ca.children(n)
                d = _decl_ref(lhs)
                r = _strip(rhs)
                if d is not None and r.kind == K.CALL_EXPR and r.spelling in ("max", "min"):
                    args = [_decl_ref(a) for a in r.get_arguments()]
                    if any(a is not None and _key(a) == _key(d) for a in args):
                        note(d, r.spelling, 2)
            elif kind == K.COMPOUND_ASSIGNMENT_OPERATOR and ca.compound_op(n) in ("+", "-"):
                d = _decl_ref(ca.children(n)[0])
                if d is not None:
                    note(d, "+", 1)
            elif kind == K.UNARY_OPERATOR and ca.unary_op(n) in ("post++", "pre++", "post--", "pre--"):
                d = _decl_ref(ca.children(n)[0])
                if d is not None:
                    note(d, "+", 1)
            if kind == K.DECL_REF_EXPR:
                d = _decl_ref(n)
                if d is not None:
                    uses[_key(d)] = uses.get(_key(d), 0) + 1
            for kid in n.get_children():
                walk(kid)

        walk(body)
        return {k: op for k, op in found.items() if op is not None and uses.get(k, 0) == 0 and k not in nonint}

    def _for(self, c):
        """Every loop gets its own break / continue flags (continue skips the rest of the body for
        the points that hit it; in a vectorised pass that is per iteration point)."""
        tag = object()
        keys = (("__break__", id(tag)), ("__continue__", id(tag)))
        frame = self.frames[-1]
        for k in keys:
            frame.vars[k] = [None, 0]
        self.loop_flags.append(keys)
        try:
            body = ca.for_parts(c)[3]
            if _breaks(body):
                self.loop_flags.pop()                                  # the break-aware driver has its own
                try:
                    init, cond, inc, body = ca.for_parts(c)
                    if init is not None:
                        self.exec(init) if init.kind == K.DECL_STMT else self.eval(init)
                    return self._loop_with_break(cond, inc, body)
                finally:
                    self.loop_flags.append(keys)
            return self._for_inner(c)
        finally:
            self.loop_flags.pop()
            for k in keys:
                frame.vars.pop(k, None)

    def _for_inner(self, c):
        """Counted loops over pointers (iterators) and loops with reductions run vectorised; the
        rest as in the C interpreter."""
        from .interp import (CHUNK, SEQUENTIAL_LIMIT, _FLIP, _decl_ref, _events, _map_frames_with, _memory_carried,
                             _refs, _step, _trips)

        init, cond, inc, body = ca.for_parts(c)
        if cond is None or inc is None or self.opt < 3:
            return super()._for(c)
        cnd = ca.strip(cond)
        if cnd.kind != K.BINARY_OPERATOR or ca.binary_op(cnd) not in _FLIP:
            return super()._for(c)
        lhs, rhs = ca.children(cnd)
        op = ca.binary_op(cnd)
        var, bound_c = _decl_ref(lhs), rhs
        if var is None:
            var, bound_c, op = _decl_ref(rhs), lhs, _FLIP[op]
        step = _step(inc, var) if var is not None else None
        if var is None or not step:
            return super()._for(c)
        reductions = self._reductions(body)
        first, written = _events(body)
        scans = {k: d for k, d in self._scans(body).items() if k not in reductions}
        carried = [k for k in written if first.get(k) == "r" and k not in reductions and k not in scans]
        pointer = ca.is_pointer(var.type)
        if (not pointer and not reductions and not scans) or carried or _key(var) in written or \
                any(_key(d) in written for d in _refs(bound_c)):
            return super()._for(c)
        if init is not None:
            self.exec(init) if init.kind == K.DECL_STMT else self.eval(init)
        entry = self._lookup(var)
        start = entry[1] if entry else UNK
        bound = self.no_charge(lambda: self.eval(bound_c))
        esize = 1
        if pointer:
            if not (isinstance(start, Ptr) and isinstance(bound, Ptr)) or start.obj is not bound.obj or \
                    _is_unk(start.off, bound.off):
                return self._loop_general(None, cond, inc, body)
            esize = ca.pointee_size(var.type) or 1
            s_off, b_off = np.asarray(start.off) // esize, np.asarray(bound.off) // esize
        else:
            s_off, b_off = _as_int(start), _as_int(bound)
            if _is_unk(s_off, b_off):
                return self._loop_general(None, cond, inc, body)
        trips = _trips(s_off, b_off, op, step)
        if trips is None:
            return self._loop_general(None, cond, inc, body)
        trips = np.broadcast_to(np.maximum(trips, 0), (self.L,)).astype(np.int64)
        if self.sequential_depth == 0 and int(trips.sum()) <= SEQUENTIAL_LIMIT and _memory_carried(body):
            self.sequential_depth += 1
            try:
                return self._loop_general(None, cond, inc, body)
            finally:
                self.sequential_depth -= 1
        total = int(trips.sum())
        scan_entries = {}
        for fr in self.frames[::-1]:
            for k in scans:
                if k in fr.vars and k not in scan_entries:
                    scan_entries[k] = fr.vars[k]
        if scans and (total > CHUNK or len(scan_entries) != len(scans)
                      or any((e[1].off is UNK or e[1].obj is UNK) if isinstance(e[1], Ptr) else _as_int(e[1]) is UNK
                             for e in scan_entries.values())):
            return self._loop_general(None, cond, inc, body)          # a scan we cannot run as a batch
        scan_v0 = {k: e[1] if isinstance(e[1], Ptr) else _as_int(e[1]) for k, e in scan_entries.items()}
        red_entries = {}
        for fr in self.frames[::-1]:
            for k in reductions:
                if k in fr.vars and k not in red_entries:
                    red_entries[k] = fr.vars[k]
        if total:
            cum = np.cumsum(trips)
            before = cum - trips
            base = np.broadcast_to(np.asarray(start.off if pointer else s_off), (self.L,))
            saved_frames, saved_w, saved_top = self.frames, self.w, self.scratch_top
            try:
                for a in range(0, total, CHUNK):
                    g = np.arange(a, min(a + CHUNK, total))
                    point = np.searchsorted(cum, g, side="right")
                    within = g - before[point]
                    self.frames = _map_frames_with(saved_frames, lambda arr: arr[point])
                    self.w = saved_w[point]
                    self._link(self.w, saved_w, point)
                    self._lookup(var)[1] = Ptr(start.obj, base[point] + step * esize * within) if pointer \
                        else base[point] + step * within
                    initial = {k: _as_int(e[1]) for k, e in red_entries.items()}
                    initial_real = {k: e[1] for k, e in red_entries.items() if isinstance(e[1], Real)}
                    scan_final = self._run_scans(scan_entries, body, point, len(g), scan_v0, len(saved_w)) \
                        if scan_entries else None
                    self.exec(body)
                    if scan_final is not None:
                        for k, e in scan_entries.items():
                            e[1] = scan_final[k]
                    for k, e in red_entries.items():                    # combine the reductions
                        here = self._lookup_key(k)
                        if isinstance(here[1] if here else None, Real) or isinstance(initial_real.get(k), Real):
                            e[1] = self._combine_real(reductions[k], here[1] if here else UNK, initial_real.get(k),
                                                      point, len(g), len(saved_w))
                            continue
                        vals, v0 = _as_int(here[1] if here else UNK), initial[k]
                        if _is_unk(vals, v0):
                            e[1] = UNK
                            continue
                        # per enclosing point: its iterations' values combined with its own start value
                        vals = np.broadcast_to(np.asarray(vals, dtype=np.int64), (len(g),))
                        n_out = len(saved_w)
                        start0 = np.array(np.broadcast_to(np.asarray(v0, dtype=np.int64), (n_out,)))
                        if reductions[k] == "max":
                            np.maximum.at(start0, point, vals)
                            out = start0
                        elif reductions[k] == "min":
                            np.minimum.at(start0, point, vals)
                            out = start0
                        else:
                            # each iteration started from its point's value (the batch copy), so its change
                            # is vals minus that start
                            delta = np.zeros(n_out, dtype=np.int64)
                            np.add.at(delta, point, vals - start0[point])
                            out = start0 + delta
                        e[1] = int(out[0]) if n_out == 1 else out
                        initial[k] = e[1]
                    self.scratch_top = saved_top
            finally:
                self.frames, self.w = saved_frames, saved_w
        final = (np.asarray(start.off if pointer else s_off) + step * esize * trips) if pointer \
            else (s_off + step * trips)
        final = _SCALAR(final) if np.ndim(final) else int(final)
        entry = self._lookup(var)
        entry[1] = Ptr(start.obj, final) if pointer else final
        self._forget_float_writes(body, keep=set(reductions) | set(scans))

    def _active_points(self):
        """Batch points still running the current iteration (None: all of them)."""
        if not self.loop_flags:
            return None
        frame = self.frames[-1]
        stop = None
        for key in self.loop_flags[-1]:
            entry = frame.vars.get(key)
            if entry is not None and not (np.ndim(entry[1]) == 0 and not entry[1]):
                flag = np.broadcast_to(np.asarray(entry[1]) != 0, (self.L,))
                stop = flag if stop is None else (stop | flag)
        if stop is None or not stop.any():
            return None
        return np.nonzero(~stop)[0]

    def _loop_with_break(self, cond, inc, body, var_step=None):
        """A loop whose body may break: one iteration at a time for all batch points still in it;
        a point leaves when its condition fails or it breaks."""
        tag = object()
        bkey, ckey = ("__break__", id(tag)), ("__continue__", id(tag))
        self.loop_flags.append((bkey, ckey))
        frame = self.frames[-1]
        frame.vars[bkey] = [None, 0]
        frame.vars[ckey] = [None, 0]
        idx = np.arange(self.L)
        from .interp import MAX_ITERATIONS
        try:
            for _ in range(MAX_ITERATIONS):
                if cond is not None:
                    value = self._eval_condition(idx, cond)
                    if value is UNK:
                        self.data_loops += 1
                        self.uncertainly(lambda: self.run_subset(idx, lambda: self._iteration(body, inc)))
                        return
                    keep = np.broadcast_to(np.asarray(value) != 0, (len(idx),))
                    idx = idx[keep]
                    if len(idx) == 0:
                        return
                self.frames[-1].vars[ckey] = [None, 0]

                def one():
                    self.exec(body)
                    active_inc = self._active_points_break_only(bkey)
                    if inc is not None:
                        if active_inc is None:
                            self.eval(inc)
                        elif len(active_inc):
                            self.run_subset(active_inc, lambda: self.eval(inc))
                self.run_subset(idx, lambda: self._fresh(one))
                broke = self.frames[-1].vars.get(bkey, [None, 0])[1]
                if np.ndim(broke) == 0:
                    if broke:
                        return
                else:
                    idx = idx[np.asarray(broke)[idx] == 0] if len(broke) == self.L else idx
                    if len(idx) == 0:
                        return
        finally:
            self.loop_flags.pop()
            frame.vars.pop(bkey, None)
            frame.vars.pop(ckey, None)

    def _active_points_break_only(self, bkey):
        entry = self.frames[-1].vars.get(bkey)
        if entry is None or (np.ndim(entry[1]) == 0 and not entry[1]):
            return None
        flag = np.broadcast_to(np.asarray(entry[1]) != 0, (self.L,))
        return np.nonzero(~flag)[0] if flag.any() else None

    def _lookup_key(self, k):
        frame = self.frames[-1]
        while frame is not None:
            if k in frame.vars:
                return frame.vars[k]
            frame = getattr(frame, "parent", None)
        return self.frames[0].vars.get(k)

    def _forget_loads(self):
        self._caches.clear()
        for rec in self._batches.values():
            rec.entries.clear()

    def _iteration(self, body, inc):
        if self.loop_flags:
            self.frames[-1].vars[self.loop_flags[-1][1]] = [None, 0]  # continue applies to one iteration
        self._fresh(lambda: Interpreter._iteration(self, body, inc))

    def _eval_condition(self, idx, cond):
        return self.run_subset(idx, lambda: self._condition(cond), propagate=True)

    def _condition(self, cond):
        """A loop condition. In a conjunction (a && b && ...), a part that compares floating-point values
        and is unknown is taken as true when the other parts are known: the loop runs to its integer
        bound, as for a break under an unknown floating-point test (a convergence test with tolerance 0,
        or a simulated time that does not end the run before the cycle limit)."""
        v = _as_int(self.eval(cond))
        if v is not UNK:
            return v
        parts = _conjuncts(cond)
        if len(parts) < 2:
            return UNK
        out = None
        for part in parts:
            pv = _as_int(self.no_charge(lambda: self.eval(part)))
            if pv is UNK:
                if not _float_test(part):
                    return UNK
                continue
            pv = (np.asarray(pv) != 0).astype(np.int64)
            out = pv if out is None else out & pv
        if out is None:
            return UNK
        return _SCALAR(out) if np.ndim(out) else int(out)

    def _ptr_store(self, loc, value):
        obj = loc.obj
        units = np.asarray(loc.off, dtype=np.int64) // 4
        if units.size == 0 or np.any(units < 0):
            return
        hi = int(np.max(units)) + 1
        vals = obj._value_units(hi)
        if getattr(obj, "ptrs", None) is None or len(obj.ptrs) < len(vals):
            ptrs = np.full(len(vals), -1, dtype=np.int64)
            if getattr(obj, "ptrs", None) is not None:
                ptrs[: len(obj.ptrs)] = obj.ptrs
            obj.ptrs = ptrs
        if not isinstance(value, Ptr) or value.obj is UNK or value.off is UNK:
            vals[units] = SENTINEL
            obj.ptrs[units] = -1
            return
        uid = NULL_UID if value.obj is self.null else self._uid(value.obj)
        offs = np.asarray(value.off, dtype=np.int64)
        if np.ndim(units) == 0 and offs.ndim:                          # every point writes one place: the
            offs = offs[-1]                                            # last iteration's value stays
        vals[units] = np.broadcast_to(offs, np.shape(units))
        obj.ptrs[units] = uid

    def _ptr_load(self, loc):
        obj = loc.obj
        units = np.asarray(loc.off, dtype=np.int64) // 4
        ptrs = getattr(obj, "ptrs", None)
        if ptrs is None or units.size == 0 or np.any(units < 0) or int(np.max(units)) >= len(ptrs):
            return Ptr(UNK, UNK)
        uids = ptrs[units]
        first = int(np.ravel(uids)[0])
        if first == -1 or np.any(uids != first):
            return Ptr(UNK, UNK)
        offs = obj.values[units]
        target = self.null if first == NULL_UID else self.registry[first]
        src = _provenance(obj, loc.off)
        return Ptr(target, _SCALAR(offs) if np.ndim(offs) else int(offs), src)

    def _mem_load(self, loc, type_):
        type_ = _referee(type_)
        if not isinstance(loc.obj, Obj) or loc.off is UNK:
            return UNK if not ca.is_pointer(type_) else Ptr(UNK, UNK)
        if ca.is_pointer(type_):
            return self._ptr_load(loc)
        if ca.is_float(type_):                                         # floating-point values in memory
            size = ca.type_size(type_.get_canonical()) or 8
            v = loc.obj.fload(loc.off, size)
            if np.ndim(loc.off) == 0 and self.L > 1:                   # one address, a value per point
                pp = loc.obj.point_values(int(loc.off) // 4, "f", self.L)
                if pp is not None:
                    v = pp
            if np.any(np.isnan(v)):
                return UNK
            if size == 4:
                v = v.astype(np.float32).astype(float)
            return Real(v if np.ndim(v) else float(v))
        return super()._mem_load(loc, type_)

    def _mem_store(self, loc, type_, value):
        type_ = _referee(type_)
        if not isinstance(loc.obj, Obj) or loc.off is UNK:
            return
        if ca.is_pointer(type_) or isinstance(value, Ptr):
            if self.charging:
                self._ptr_store(loc, value)
            return
        if ca.is_float(type_):
            if self.charging:
                size = ca.type_size(type_.get_canonical()) or 8
                r = _real(value) if value is not UNK else UNK
                v = np.nan if r is UNK else (np.asarray(r.v, dtype=np.float32).astype(float) if size == 4 else r.v)
                offs = np.broadcast_to(loc.off, (self.L,)) if np.ndim(loc.off) else loc.off
                loc.obj.fstore(offs, v, size)
            return
        super()._mem_store(loc, type_, value)

    def _float_update(self, loc, type_, op, r):
        """a[i] op= x on floating-point memory: every batch point in order, so repeated offsets
        accumulate (a scatter-add of element forces onto nodes) as in a sequential loop."""
        size = ca.type_size(type_.get_canonical()) or 8
        r = _real(r) if r is not UNK else UNK
        offs = np.broadcast_to(np.asarray(loc.off, dtype=np.int64), (self.L,))
        if r is UNK or not self.charging:
            if self.charging:
                loc.obj.fstore(offs, np.nan, size)
            return UNK
        rv = np.broadcast_to(np.asarray(r.v, dtype=float), (self.L,))
        old = loc.obj.fload(offs, size)
        if op in ("+", "-") and len(np.unique(offs)) < len(offs):
            uniq, inv = np.unique(offs, return_inverse=True)
            acc = loc.obj.fload(uniq, size)
            np.add.at(acc, inv.ravel(), rv if op == "+" else -rv)
            loc.obj.fstore(uniq, acc, size)
            new = acc[inv.ravel()]
        else:
            with np.errstate(all="ignore"):
                new = {"+": old + rv, "-": old - rv, "*": old * rv, "/": old / rv}.get(op, np.full(self.L, np.nan))
            loc.obj.fstore(offs, new, size)
        if np.any(np.isnan(new)):
            return UNK
        return Real(new if self.L > 1 or np.ndim(loc.off) else float(new[0]))

    def _copy_object(self, dest, src, size, charge=True):
        """Memberwise copy of `size` bytes (trivial copy / move): values and pointers, 4-byte units,
        and the per-byte values of small fields."""
        if not (isinstance(dest, Ptr) and isinstance(src, Ptr)) or _is_unk(dest.obj, dest.off, src.obj, src.off):
            return
        if getattr(src.obj, "bytevals", None) is not None:
            for b in range(size):
                vals = src.obj.load(src.off + b, 1)
                dest.obj.store(np.broadcast_to(dest.off + b, np.shape(vals)) if np.ndim(vals) else dest.off + b, vals, 1)
        for u in range(0, size, 4):
            s = Loc(src.obj, src.off + u, 4)
            d = Loc(dest.obj, dest.off + u, 4)
            if charge and u % 8 == 0:
                self.access(Loc(src.obj, src.off + u, 8))
                self.access(Loc(dest.obj, dest.off + u, 8), write=True)
            vals = s.obj.load(s.off)
            d.obj.store(np.broadcast_to(d.off, np.shape(vals)) if np.ndim(vals) else d.off, vals)
            if getattr(s.obj, "fvals", None) is not None:
                fv = s.obj.fload(s.off, 4)
                if not np.all(np.isnan(fv)):
                    dst = np.broadcast_to(d.off, np.shape(fv)) if np.ndim(fv) else d.off
                    keep = d.obj.load(dst)                             # fstore marks the ints unknown; keep them
                    d.obj.fstore(dst, fv, 4)
                    d.obj._value_units(int(np.max(np.asarray(dst) // 4)) + 1)[np.asarray(dst, dtype=np.int64) // 4] = keep
            sp = getattr(s.obj, "ptrs", None)
            if sp is not None:
                su = np.asarray(s.off, dtype=np.int64) // 4
                if np.all(su < len(sp)):
                    dvals = d.obj._value_units(int(np.max(np.asarray(d.off) // 4)) + 1)
                    if getattr(d.obj, "ptrs", None) is None or len(d.obj.ptrs) < len(dvals):
                        ptrs = np.full(len(dvals), -1, dtype=np.int64)
                        if getattr(d.obj, "ptrs", None) is not None:
                            ptrs[: len(d.obj.ptrs)] = d.obj.ptrs
                        d.obj.ptrs = ptrs
                    du, sv = np.asarray(d.off, dtype=np.int64) // 4, sp[su]
                    if np.ndim(du) == 0 and np.ndim(sv):                # every point copies into one place:
                        sv = sv[-1]                                    # the last iteration's value stays
                    d.obj.ptrs[du] = sv

    # ------------------------------------------------------------ storage for objects

    def _new_storage(self, type_):
        """Storage for a local or temporary object: registers (one copy per batch point) for small
        objects at -O3, else memory (a stack slot or an object of its own)."""
        size = ca.type_size(type_.get_canonical()) or 8
        if self.opt >= 3 and size <= REGISTER_BYTES:
            span = (size + 7) // 8 * 8
            base = self.scratch_top
            self.scratch_top += span * self.L
            off = base + span * np.arange(self.L) if self.L > 1 else base
            return Ptr(self.scratch, off)
        obj = Obj(f"object#{len(self.objects)}", "stack-array", size)
        self.objects.append(obj)
        return Ptr(obj, 0)

    # ------------------------------------------------------------ lookup through lambda frames

    def _lookup(self, decl):
        k = _key(decl)
        frame = self.frames[-1]
        while frame is not None:
            if k in frame.vars:
                return frame.vars[k]
            frame = getattr(frame, "parent", None)
        if k in self.frames[0].vars:
            return self.frames[0].vars[k]
        if decl.kind == K.VAR_DECL and decl.semantic_parent is not None and \
                decl.semantic_parent.kind in (K.TRANSLATION_UNIT, K.NAMESPACE, K.CLASS_DECL, K.STRUCT_DECL):
            self._declare_global(decl)
            return self.frames[0].vars[k]
        return None

    def _this(self):
        frame = self.frames[-1]
        while frame is not None:
            if "__this__" in frame.vars:
                return frame.vars["__this__"][1]
            frame = getattr(frame, "parent", None)
        return Ptr(UNK, UNK)

    def _field_offset(self, member):
        fld = member.referenced
        if fld is None:
            return 0
        o = fld.get_field_offsetof()
        if o is not None and o >= 0:
            return o // 8
        kids = ca.children(member)
        for t in ([kids[0].type] if kids else []) + self.this_types[-1:]:
            t = _referee(t)
            if ca.is_pointer(t):
                t = t.get_pointee()
            o = t.get_canonical().get_offset(fld.spelling)
            if o is not None and o >= 0:
                return o // 8
        return 0

    # ------------------------------------------------------------ lvalues

    def lvalue(self, c):
        c = ca.strip(c)
        kind = c.kind
        if kind == K.DECL_REF_EXPR and self.opt >= 3:
            decl, entry = self._scalar_entry(c)
            if entry is not None and isinstance(entry[1], InMemory):
                return entry[1].loc
        if kind == K.DECL_REF_EXPR and c.referenced is not None and _reference(c.referenced.type):
            entry = self._lookup(c.referenced)
            p = entry[1] if entry else Ptr(UNK, UNK)
            size = ca.type_size(c.type) or 8
            if not isinstance(p, Ptr):
                return Loc(UNK, UNK, size)
            return Loc(p.obj, p.off, size, _record(c.type) or ca.is_array(c.type))
        if kind == K.DECL_REF_EXPR and c.referenced is not None and _record(c.type):
            entry = self._lookup(c.referenced)
            p = entry[1] if entry else Ptr(UNK, UNK)
            if isinstance(p, Ptr):
                return Loc(p.obj, p.off, ca.type_size(c.type) or 8, True)
        if kind == K.MEMBER_REF_EXPR:
            kids = ca.children(c)
            size = ca.type_size(c.type) or 8
            array = ca.is_array(c.type) or _record(c.type)
            fo = self._field_offset(c)
            if not kids:
                base = self._this()
            elif kids[0].kind == K.CXX_THIS_EXPR or ca.is_pointer(kids[0].type):
                base = self.eval(kids[0])
            else:
                b = self.lvalue(kids[0])
                base = Ptr(b.obj, b.off)
            if not isinstance(base, Ptr) or _is_unk(base.obj, base.off):
                return Loc(UNK, UNK, size, array)
            if _reference(c.referenced.type if c.referenced is not None else c.type):  # reference member
                p = self._ptr_load(Loc(base.obj, base.off + fo, 8))
                loc = Loc(p.obj, p.off, size, array)
                loc.src = p.src
                return loc
            loc = Loc(base.obj, base.off + fo, size, array)
            loc.src = getattr(base, "src", None)
            return loc
        if kind == K.CALL_EXPR:
            size = ca.type_size(c.type) or 8
            if self._returns_object(c):
                dest = self._new_storage(c.type)                       # the caller owns the result
                v = self._call(c, sret=dest)
                if isinstance(v, (Str, Closure)):
                    return Loc(UNK, UNK, size, True)
                return Loc(dest.obj, dest.off, size, True)
            v = self._call(c)
            if isinstance(v, Ptr):
                return Loc(v.obj, v.off, size, _record(c.type))
            return Loc(UNK, UNK, size)
        if kind == K.UNARY_OPERATOR and ca.unary_op(c) == "*":
            p = self.eval(ca.children(c)[0])
            size = ca.type_size(c.type) or 8
            if not isinstance(p, Ptr):
                return Loc(UNK, UNK, size)
            return Loc(p.obj, p.off, size, ca.is_array(c.type) or _record(c.type))
        if kind in (K.CXX_STATIC_CAST_EXPR, K.CXX_REINTERPRET_CAST_EXPR, K.CXX_CONST_CAST_EXPR,
                    K.CSTYLE_CAST_EXPR, K.CXX_FUNCTIONAL_CAST_EXPR):
            return self.lvalue(ca.children(c)[-1])
        if kind == K.ARRAY_SUBSCRIPT_EXPR:
            loc = super().lvalue(c)
            if _record(c.type):
                loc.array = True                                       # an object: addressed, not loaded
            return loc
        if _record(c.type) and c.kind.is_expression() and kind not in (K.DECL_REF_EXPR, K.MEMBER_REF_EXPR,
                                                                         K.UNARY_OPERATOR):
            v = self.eval(c)
            if isinstance(v, Ptr):
                return Loc(v.obj, v.off, ca.type_size(c.type) or 8, True)
            return Loc(UNK, UNK, ca.type_size(c.type) or 8, True)
        return super().lvalue(c)

    def _materialize(self, decl, entry):
        """Give a register scalar its own memory slot (per batch point) and keep its value there."""
        if isinstance(entry[1], InMemory):
            return entry[1].loc
        # its own region, not the scoped scratch: the variable outlives the block where its address is taken
        size = max(ca.type_size(decl.type.get_canonical()) or 8, 8)
        if getattr(self, "spill", None) is None:
            self.spill = Obj("spilled registers", "register")
            self.spill_top = 0
        base = self.spill_top
        self.spill_top += size * self.L
        off = base + size * np.arange(self.L) if self.L > 1 else base
        loc = Loc(self.spill, off, ca.type_size(decl.type.get_canonical()) or 8)
        value = entry[1]
        if isinstance(value, Ptr) or ca.is_pointer(decl.type):
            self._ptr_store(loc, value if isinstance(value, Ptr) else Ptr(UNK, UNK))
        else:
            self._mem_store(loc, decl.type, value)
        entry[0], entry[1] = loc, InMemory(loc)
        return loc

    def _scalar_entry(self, c):
        t = ca.strip(c)
        if t.kind != K.DECL_REF_EXPR or t.referenced is None or _reference(t.referenced.type) or \
                _record(t.type) or ca.is_array(t.type):
            return None, None
        entry = self._lookup(t.referenced)
        return t.referenced, entry

    def _value_of(self, c):
        decl, entry = self._scalar_entry(c)
        if entry is not None and isinstance(entry[1], InMemory):
            return self._mem_load(entry[1].loc, c.type)
        return super()._value_of(c)

    def _set_value(self, c, value):
        decl, entry = self._scalar_entry(c)
        if entry is not None and isinstance(entry[1], InMemory):
            self._mem_store(entry[1].loc, c.type, value)
            return
        super()._set_value(c, value)

    def _store(self, c, value):
        decl, entry = self._scalar_entry(c)
        if entry is not None and isinstance(entry[1], InMemory):
            self._mem_store(entry[1].loc, c.type, value)
            return entry[1].loc
        target = ca.strip(c)
        if target.kind == K.DECL_REF_EXPR and target.referenced is not None and _reference(target.referenced.type):
            loc = self.lvalue(c)                                       # through the reference
            self.access(loc, write=True)
            self._mem_store(loc, c.type, value)
            return loc
        return super()._store(c, value)

    def _load_lvalue(self, c):
        decl, entry = self._scalar_entry(c)
        if entry is not None and isinstance(entry[1], InMemory):
            return self._mem_load(entry[1].loc, c.type)
        if _record(_referee(c.type)):
            loc = self.lvalue(c)
            return Ptr(loc.obj, loc.off)
        return super()._load_lvalue(c)

    # ------------------------------------------------------------ rvalues

    def _is_ref_var(self, c):
        t = ca.strip(c)
        return t.kind == K.DECL_REF_EXPR and t.referenced is not None and _reference(t.referenced.type)

    def eval(self, c):
        kind = c.kind
        if kind == K.DECL_REF_EXPR and c.referenced is not None and c.referenced.kind == K.FUNCTION_DECL:
            return Func(c.referenced)
        if kind == K.COMPOUND_ASSIGNMENT_OPERATOR and self._is_ref_var(ca.children(c)[0]):
            lhs, rhs = ca.children(c)                                  # through a reference: memory
            loc = self.lvalue(lhs)
            self.access(loc)
            r = self.eval(rhs)
            self.access(loc, write=True)
            op = ca.compound_op(c)
            if op in ("+", "-") and not isinstance(r, Ptr) and not ca.is_pointer(lhs.type):
                r = _as_int(r)
                _, new = self._rmw_add(loc, lhs.type, r if op == "+" or r is UNK else _SCALAR(-r))
                return new
            new = self._arith(op, self._mem_load(loc, lhs.type), r, c)
            self._mem_store(loc, lhs.type, new)
            return new
        if kind == K.UNARY_OPERATOR and ca.unary_op(c) in ("post++", "post--", "pre++", "pre--") \
                and self._is_ref_var(ca.children(c)[0]) and not ca.is_pointer(ca.children(c)[0].type):
            kid = ca.children(c)[0]
            loc = self.lvalue(kid)
            self.access(loc)
            self.access(loc, write=True)
            old, new = self._rmw_add(loc, kid.type, 1 if "++" in ca.unary_op(c) else -1)
            return old if ca.unary_op(c).startswith("post") else new
        if kind == K.UNARY_OPERATOR and ca.unary_op(c) == "&":
            decl, entry = self._scalar_entry(ca.children(c)[0])
            if entry is not None and getattr(entry[0], "register", False):
                loc = self._materialize(decl, entry)                   # its address escapes
                return Ptr(loc.obj, loc.off)
        if kind == K.CXX_BOOL_LITERAL_EXPR:
            return 1 if any(t.spelling == "true" for t in c.get_tokens()) else 0
        if kind in (K.CXX_NULL_PTR_LITERAL_EXPR, K.GNU_NULL_EXPR):
            return Ptr(self.null, 0)
        if kind == K.CXX_THIS_EXPR:
            return self._this()
        if kind == K.STRING_LITERAL:
            return Str(c.spelling.strip('"'))
        if kind in (K.CXX_STATIC_CAST_EXPR, K.CXX_REINTERPRET_CAST_EXPR, K.CXX_CONST_CAST_EXPR,
                    K.CXX_FUNCTIONAL_CAST_EXPR):
            kids = [k for k in ca.children(c) if k.kind.is_expression()]
            if not kids:
                return UNK
            if _record(c.type) and kind == K.CXX_FUNCTIONAL_CAST_EXPR:
                return self.eval(kids[-1])
            v = self.eval(kids[-1])
            if ca.is_float(c.type) and not isinstance(v, Ptr):
                return _real(v)
            if isinstance(v, Real):
                v = self._truncate(v)
            if ca.is_pointer(c.type) and isinstance(v, Ptr):
                return v
            if isinstance(v, (int, np.integer, np.ndarray)):
                from .interp import wrap_int
                return wrap_int(v, c.type)
            return v
        if kind == K.CXX_NEW_EXPR:
            return self._new(c)
        if kind == K.CXX_DELETE_EXPR:
            p = self.eval(ca.children(c)[-1])
            if isinstance(p, Ptr) and isinstance(p.obj, Obj) and p.obj is not self.null:
                self.heap_events.append(("free", p.obj))
            return UNK
        if kind == K.LAMBDA_EXPR:
            return Closure(c, self.frames[-1])
        if kind == K.CALL_EXPR:
            ref = c.referenced
            if ref is not None and ref.kind == K.CONSTRUCTOR:              # a temporary object
                dest = self._new_storage(c.type)
                self._construct_call(c, dest)
                return dest
            if self._returns_object(c):
                dest = self._new_storage(c.type)
                v = self._call(c, sret=dest)
                return v if isinstance(v, (Str, Closure)) else dest
            v = self._call(c)
            if ref is not None and _reference(ref.result_type) and isinstance(v, Ptr) and not _record(c.type):
                loc = Loc(v.obj, v.off, ca.type_size(c.type) or 8)
                self.access(loc)
                return self._mem_load(loc, c.type)
            return v
        if kind == K.DECL_REF_EXPR and c.referenced is not None and _reference(c.referenced.type):
            if _record(c.type):
                return self._load_lvalue(c)
            loc = self.lvalue(c)
            self.access(loc)
            return self._mem_load(loc, c.type)
        if kind == K.MEMBER_REF_EXPR and not ca.children(c):
            if _record(c.type) or ca.is_array(c.type):
                loc = self.lvalue(c)
                return Ptr(loc.obj, loc.off)
            loc = self.lvalue(c)
            self.access(loc)
            return self._mem_load(loc, c.type)
        if kind == K.MEMBER_REF_EXPR:
            loc = self.lvalue(c)
            if loc.array:
                return Ptr(loc.obj, loc.off)
            self.access(loc)
            return self._mem_load(loc, c.type)
        if kind == K.BINARY_OPERATOR and _record(c.type):
            return super().eval(c)
        return super().eval(c)

    def _arith(self, op, a, b, c):
        if isinstance(a, Str) or isinstance(b, Str):
            if op in ("==", "!=") and isinstance(a, Str) and isinstance(b, Str):
                return int((a.text == b.text) == (op == "=="))
            return UNK
        if isinstance(a, Closure) or isinstance(b, Closure):
            return UNK
        return super()._arith(op, a, b, c)

    def _new(self, c):
        """new T[n] / new T(args): a heap block (construction of class elements is not run)."""
        pointee = c.type.get_pointee()
        esize = ca.type_size(pointee.get_canonical()) or 8
        count = 1
        for k in ca.children(c):
            if k.kind.is_expression() and k.kind not in (K.CALL_EXPR, K.INIT_LIST_EXPR) and not _record(k.type):
                n = _as_int(self.eval(k))
                count = n if n is not UNK else UNK
                break
        nbytes = int(np.max(count)) * esize if count is not UNK else 0
        obj = Obj(f"new#{len(self.objects)}", "heap", nbytes)
        self.objects.append(obj)
        self.heap_events.append(("new", obj))
        p = Ptr(obj, 0)
        for k in ca.children(c):
            if k.kind == K.CALL_EXPR and k.referenced is not None and k.referenced.kind == K.CONSTRUCTOR:
                self._construct_call(k, p)
        return p

    # ------------------------------------------------------------ construction

    def _construct_into(self, expr, dest, type_):
        """Initialise the object at dest from expr (a constructor call, a call returning the object
        by value, a lambda, or another object to copy)."""
        e = _strip(expr)
        if e.kind == K.CALL_EXPR and e.referenced is not None and e.referenced.kind == K.CONSTRUCTOR:
            self._construct_call(e, dest)
            return None
        if e.kind == K.CALL_EXPR:
            self._call(e, sret=dest)
            return None
        if e.kind in (K.CXX_FUNCTIONAL_CAST_EXPR, K.CXX_STATIC_CAST_EXPR):
            inner = [k for k in ca.children(e) if k.kind.is_expression()]
            if inner:
                return self._construct_into(inner[-1], dest, type_)
        if e.kind == K.INIT_LIST_EXPR:
            return None
        src = self.eval(e)
        if isinstance(src, Ptr):
            self._copy_object(dest, src, ca.type_size(type_.get_canonical()) or 8)
            moved = _strip(e)
            if moved.kind == K.DECL_REF_EXPR:
                return moved.referenced
        return None

    def _construct_call(self, call, dest):
        ctor = call.referenced
        cdef = ctor.get_definition() if ctor is not None else None
        args_c = list(call.get_arguments())                            # not the template arguments
        if ctor is not None and _system(ctor):
            return self._library_construct(call, dest, args_c)
        if cdef is None or _body(cdef) is None or _implicit_copy(ctor, cdef):
            # implicit (or library) constructor: copy / move from the argument, if it is an object
            if len(args_c) == 1 and _record(_referee(args_c[0].type)):
                src = self.eval(args_c[0])
                if isinstance(src, Ptr):
                    self._copy_object(dest, src, ca.type_size(call.type.get_canonical()) or 8)
                return dest
            if ctor is not None and _system(ctor):
                return self._library_construct(call, dest, args_c)
            if not args_c:                                             # implicit default constructor
                self._default_fields(dest, call.type)
            return dest
        self._invoke(cdef, args_c, this=dest, this_type=call.type, constructor=True)
        return dest

    def _default_construct(self, dest, type_, depth=0):
        """Default-construct an object of class type at dest: a library vector is empty; a class runs its
        default constructor if it has one with code, else default-constructs its own fields."""
        t = type_.get_canonical()
        name = t.spelling
        if name.startswith("std::vector<"):
            from .containers import element_type
            et = element_type(t)
            self._vinit(dest, 0, (ca.type_size(et.get_canonical()) if et is not None else None) or 8, et=et)
            return
        if name.startswith("std::"):
            return
        for m in _class_members(t):
            if m.kind == K.CONSTRUCTOR and not [p for p in m.get_children() if p.kind == K.PARM_DECL]:
                d = m.get_definition()
                if d is not None and _body(d) is not None:
                    self._invoke(d, [], this=dest, this_type=type_, constructor=True)
                    return
        if depth < 8:
            self._default_fields(dest, type_, depth + 1)

    def _default_fields(self, dest, type_, depth=0, skip=()):
        """Default-construct the class-type fields (vectors, nested objects) of the object at dest."""
        if not isinstance(dest, Ptr) or _is_unk(dest.obj, dest.off):
            return
        t = type_.get_canonical()
        for m in _class_members(t):
            if m.kind != K.FIELD_DECL or m.spelling in skip or not _record(m.type):
                continue
            if [k for k in m.get_children() if k.kind.is_expression()]:   # has a default member initialiser
                continue
            offset = t.get_offset(m.spelling)
            if offset is None or offset < 0:
                continue
            off = dest.off + offset // 8
            self._default_construct(Ptr(dest.obj, _SCALAR(off) if np.ndim(off) else int(off)), m.type, depth)

    def _member_inits(self, cdef, this, this_type):
        kids = ca.children(cdef)
        initialised = set()
        i = 0
        while i < len(kids):
            k = kids[i]
            if k.kind == K.MEMBER_REF and i + 1 < len(kids) and kids[i + 1].kind.is_expression():
                fld = k.referenced
                offset = fld.get_field_offsetof() if fld is not None else -1
                if offset is None or offset < 0:
                    offset = this_type.get_canonical().get_offset(k.spelling)
                loc = Loc(this.obj, this.off + max(offset, 0) // 8, ca.type_size(fld.type) if fld else 8)
                self._init_field(loc, fld.type if fld else kids[i + 1].type, kids[i + 1])
                initialised.add(k.spelling)
                i += 2
                continue
            if k.kind in (K.TYPE_REF, K.TEMPLATE_REF) and i + 1 < len(kids) and kids[i + 1].kind == K.CALL_EXPR:
                call = kids[i + 1]
                if call.referenced is not None and call.referenced.kind == K.CONSTRUCTOR:
                    self._construct_call(call, this)                    # base class or delegating constructor
                i += 2
                continue
            i += 1
        # fields of class type not in the list and without an initialiser: default-constructed
        self._default_fields(this, this_type, skip=initialised)
        # default member initialisers of the fields not in the list
        for m in _class_members(this_type):
            if m.kind == K.FIELD_DECL and m.spelling not in initialised:
                init = [k for k in m.get_children() if k.kind.is_expression()]
                if init:
                    offset = this_type.get_canonical().get_offset(m.spelling)
                    loc = Loc(this.obj, this.off + max(offset, 0) // 8, ca.type_size(m.type) or 8)
                    self._init_field(loc, m.type, init[-1])

    def _init_field(self, loc, type_, expr):
        if _reference(type_):
            target = self.lvalue(expr)
            self._ptr_store(loc, Ptr(target.obj, target.off))
            return
        if _record(type_):
            self._construct_into(expr, Ptr(loc.obj, loc.off), type_)
            return
        value = self.eval(expr)
        self.access(loc, write=True)
        self._mem_store(loc, type_, value)

    # ------------------------------------------------------------ declarations and scopes

    def _declare_scalar(self, decl, value):
        if self.opt >= 3 and not ca.is_array(decl.type):
            frame = self.frames[-1]
            loc = Loc(self.scratch, 0, 8)
            loc.register = True
            frame.vars[_key(decl)] = [loc, value]
            return loc
        return self._declare_local(decl, value)

    def _exec_decl(self, decl):
        t = decl.type
        init = self._initializer(decl)
        if init is None:
            exprs = [k for k in decl.get_children() if k.kind.is_expression()]
            init = exprs[-1] if exprs else None
        if decl.storage_class.name == "STATIC":
            self._lookup(decl)
            return
        if _reference(t):
            target = self.lvalue(init) if init is not None else Loc(UNK, UNK, 8)
            loc = Loc(self.scratch, 0, 8)
            loc.register = True
            self.frames[-1].vars[_key(decl)] = [loc, Ptr(target.obj, target.off)]
            return
        if _record(t):
            if init is not None and _strip(init).kind == K.LAMBDA_EXPR:
                loc = Loc(self.scratch, 0, 8)
                loc.register = True
                self.frames[-1].vars[_key(decl)] = [loc, self.eval(_strip(init))]
                return
            dest = self._new_storage(t)
            loc = Loc(dest.obj, dest.off, ca.type_size(t.get_canonical()) or 8, True)
            self.frames[-1].vars[_key(decl)] = [loc, dest]
            moved = None
            if init is not None:
                moved = self._construct_into(init, dest, t)
            if moved is not None:
                self._mark_moved(moved)
            if self.scopes:
                self.scopes[-1].append((decl, dest, t))
            return
        if ca.is_array(t):
            super()._exec_decl(decl)
            if init is not None and _strip(init).kind == K.INIT_LIST_EXPR:
                slot = self._lookup(decl)[0]                           # {a, b, ...}: store the elements
                et = t.get_canonical().element_type
                esize = ca.type_size(et) or 4
                for i, e in enumerate(k for k in ca.children(_strip(init)) if k.kind.is_expression()):
                    loc = Loc(slot.obj, slot.off + i * esize, esize)
                    self.access(loc, write=True)
                    self._mem_store(loc, et, self.eval(e))
            return
        value = self.eval(init) if init is not None and init.kind != K.INIT_LIST_EXPR else UNK
        if ca.is_float(t) and not isinstance(value, (Ptr, Real)):
            value = UNK
        slot = self._declare_scalar(decl, value)
        if init is not None:
            self.access(slot, write=True)

    def _mark_moved(self, decl):
        """A local returned or moved out keeps its contents elsewhere: do not destroy it."""
        for scope in self.scopes:
            for i, (d, dest, t) in enumerate(scope):
                if d is not None and _key(d) == _key(decl):
                    scope[i] = (None, dest, t)

    def _destroy(self, dest, type_):
        decl = type_.get_canonical().get_declaration()
        if _system(decl):
            return
        for m in _class_members(type_):
            if m.kind == K.DESTRUCTOR:
                d = m.get_definition() or m
                if _body(d) is not None:
                    self._invoke(d, [], this=dest, this_type=type_)
                return

    def exec(self, c):
        kind = c.kind
        frame = self.frames[-1]
        if getattr(frame, "returned", False):
            return
        if kind == K.COMPOUND_STMT:
            saved_top = self.scratch_top
            self.scopes.append([])
            try:
                for kid in c.get_children():
                    if getattr(self.frames[-1], "returned", False):
                        break
                    active = self._active_points()
                    if active is None:
                        self.exec(kid)
                    elif len(active):
                        self.run_subset(active, lambda k=kid: self.exec(k))
                    else:
                        break
            finally:
                scope = self.scopes.pop()
                for decl, dest, t in reversed(scope):
                    if decl is not None:
                        self._destroy(dest, t)
                self.scratch_top = max(saved_top, 0) if not scope else self.scratch_top
            return
        if kind == K.DECL_STMT:
            for decl in c.get_children():
                if decl.kind == K.VAR_DECL:
                    self._exec_decl(decl)
            return
        if kind == K.CXX_FOR_RANGE_STMT:
            return self._for_range(c)
        if kind in (K.WHILE_STMT, K.DO_STMT) and _breaks(ca.children(c)[-1 if kind == K.WHILE_STMT else 0]):
            kids = ca.children(c)
            cond, body = (kids[0], kids[-1]) if kind == K.WHILE_STMT else (kids[-1], kids[0])
            if kind == K.DO_STMT:
                self.exec(body)
            return self._loop_with_break(cond, None, body)
        if kind in (K.BREAK_STMT, K.CONTINUE_STMT) and self.loop_flags:
            if self.uncertain_depth:
                # under a condition on unknown (floating-point) data: assume the loop goes on, so it
                # runs to its bound (exact for pr -t 0; an upper bound when it would converge)
                return
            key = self.loop_flags[-1][0 if kind == K.BREAK_STMT else 1]
            self.frames[-1].vars[key] = [None, np.ones(self.L, dtype=np.int64) if self.L > 1 else 1]
            return
        if kind == K.RETURN_STMT:
            kids = ca.children(c)
            fn = frame.func
            if kids and fn is not None and _record(fn.result_type) and getattr(frame, "sret", None) is not None:
                moved = self._construct_into(kids[0], frame.sret, fn.result_type)
                if moved is not None:
                    self._mark_moved(moved)
                frame.vars["__ret__"] = [None, frame.sret]
            elif kids and fn is not None and _reference(fn.result_type):
                loc = self.lvalue(kids[0])
                frame.vars["__ret__"] = [None, Ptr(loc.obj, loc.off)]
            else:
                v = self.eval(kids[0]) if kids else UNK
                frame.vars["__ret__"] = [None, v]
            if self.L == 1:
                frame.returned = True
            return
        if kind in (K.CXX_TRY_STMT, K.CXX_CATCH_STMT):
            for kid in c.get_children():
                if kid.kind == K.COMPOUND_STMT:
                    self.exec(kid)
                    break
            return
        return super().exec(c)

    # ------------------------------------------------------------ calls

    def _call(self, c, sret=None):
        ref = c.referenced
        kids = ca.children(c)
        if ref is None and c.spelling in MATH:                         # an overloaded name libclang left unresolved
            return self._std(c, c.spelling, list(c.get_arguments()), sret)
        if ref is None:
            for k in kids:
                self.eval(k) if k.kind.is_expression() else None
            return UNK
        if ref.kind == K.CONSTRUCTOR:
            dest = sret if sret is not None else self._new_storage(c.type)
            return self._construct_call(c, dest)
        if ref.kind in (K.PARM_DECL, K.VAR_DECL) and kids:              # f(args) through a variable
            f = self.eval(kids[0])
            fdef = f.decl.get_definition() if isinstance(f, Func) else None
            if fdef is None or _body(fdef) is None:
                for a in c.get_arguments():
                    self.eval(a)
                return UNK
            return self._invoke(fdef, list(c.get_arguments()), sret=sret)
        parent = ref.semantic_parent
        stub = self._stub(ref)
        if stub is not None:
            return stub
        obj, args_c = None, list(c.get_arguments())
        if ref.kind in (K.CXX_METHOD, K.CONVERSION_FUNCTION) and not ref.is_static_method():
            if kids and kids[0].kind == K.MEMBER_REF_EXPR and kids[0].referenced is not None and \
                    kids[0].referenced.kind in (K.CXX_METHOD, K.CONVERSION_FUNCTION, K.FUNCTION_TEMPLATE):
                callee = kids[0]                                       # obj.method(args)
                inner = ca.children(callee)
                if not inner:
                    obj = self._this()
                elif inner[0].kind == K.CXX_THIS_EXPR or ca.is_pointer(inner[0].type):
                    obj = self.eval(inner[0])
                else:
                    obj = self.eval(inner[0]) if _strip(inner[0]).kind == K.LAMBDA_EXPR else None
                    if obj is None or not isinstance(obj, Closure):
                        loc = self.lvalue(inner[0])
                        entry_val = self._closure_of(inner[0])
                        obj = entry_val if entry_val is not None else Ptr(loc.obj, loc.off)
                args_c = list(c.get_arguments())
            else:                                                      # operator call: object, then args
                exprs = list(c.get_arguments())
                nparams = len([p for p in ref.get_children() if p.kind == K.PARM_DECL]) or \
                    len(list(ref.type.argument_types()))
                if len(exprs) == nparams:                              # the object is not among them
                    first = [k for k in kids if k.kind.is_expression() and not self._is_callee(k)]
                    exprs = first[:1] + exprs
                obj_c = exprs[0]
                closure = self._closure_of(obj_c)
                if closure is not None:
                    obj = closure
                elif ca.is_pointer(obj_c.type):
                    obj = self.eval(obj_c)
                else:
                    loc = self.lvalue(obj_c)
                    obj = Ptr(loc.obj, loc.off)
                args_c = exprs[1:]
        else:
            args_c = list(c.get_arguments())
        if isinstance(obj, Closure):
            return self._invoke_lambda(obj, args_c)
        if ref.spelling in ATOMIC_IDIOMS and obj is None and args_c:
            return self._atomic(ref.spelling, args_c)
        fdef = ref.get_definition()
        if ref.spelling == "operator=" and obj is not None and _implicit_copy(ref, fdef) \
                and not _system(ref) and args_c:
            src = self.eval(args_c[0])                                 # implicit copy / move assignment
            if isinstance(src, Ptr) and isinstance(obj, Ptr):
                self._copy_object(obj, src, ca.type_size(_referee(c.type).get_canonical()) or 8)
            return obj
        if fdef is None or _body(fdef) is None or _system(ref) or (parent is not None and parent.spelling == "std"):
            return self._library_cpp(c, ref, obj, args_c, sret)
        this_type = None
        if obj is not None:
            this_type = parent.type if parent is not None else None
        return self._invoke(fdef, args_c, this=obj, this_type=this_type, sret=sret)

    @staticmethod
    def _returns_object(c):
        """A call (not a constructor) that returns a class object by value."""
        ref = c.referenced
        return ref is not None and ref.kind != K.CONSTRUCTOR and _record(c.type) and \
            ref.kind in (K.FUNCTION_DECL, K.CXX_METHOD) and not _reference(ref.result_type)

    @staticmethod
    def _is_callee(k):
        """The callee's own name among a call's children (operator calls put it between operands)."""
        e = _strip(k)
        return e.kind in (K.DECL_REF_EXPR, K.OVERLOADED_DECL_REF) and e.referenced is not None and \
            e.referenced.kind in (K.FUNCTION_DECL, K.CXX_METHOD, K.FUNCTION_TEMPLATE, K.CONVERSION_FUNCTION)

    def _closure_of(self, c):
        e = _strip(c)
        while e.kind == K.CALL_EXPR and e.referenced is not None and e.referenced.kind == K.CONSTRUCTOR:
            args = list(e.get_arguments())                             # a copy / move of the closure
            if len(args) != 1:
                return None
            e = _strip(args[0])
        if e.kind == K.DECL_REF_EXPR and e.referenced is not None:
            entry = self._lookup(e.referenced)
            if entry is not None and isinstance(entry[1], Closure):
                return entry[1]
        if e.kind == K.LAMBDA_EXPR:
            return self.eval(e)
        return None

    def _bind_args(self, params, args_c, frame):
        """Evaluate arguments in the caller, then bind them in the callee's frame."""
        bound = []
        for i, p in enumerate(params):
            if i < len(args_c) and not (args_c[i].kind == K.UNEXPOSED_EXPR and not ca.children(args_c[i])):
                a = args_c[i]
                if _reference(p.type):
                    decl, entry = self._scalar_entry(a)
                    if entry is not None and getattr(entry[0], "register", False):
                        loc = self._materialize(decl, entry)           # bound to a reference
                    else:
                        loc = self.lvalue(a)
                    bound.append(("ref", p, Ptr(loc.obj, loc.off)))
                elif _record(p.type):
                    closure = self._closure_of(a)
                    if closure is not None:
                        bound.append(("closure", p, closure))
                    else:
                        dest = self._new_storage(p.type)
                        self._construct_into(a, dest, p.type)
                        bound.append(("object", p, dest))
                else:
                    bound.append(("value", p, self.eval(a)))
            else:
                bound.append(("default", p, None))
        return bound

    def _enter(self, fdef, bound, this, this_type, sret, parent=None):
        frame = Frame(fdef, len(self.frames) * 8192)
        if parent is not None:
            frame.parent = parent
        if this is not None:
            frame.vars["__this__"] = [None, this]
        frame.vars["__ret__"] = [None, UNK]
        frame.sret = sret
        self.frames.append(frame)
        self.this_types.append(this_type)
        for how, p, v in bound:
            reg = Loc(self.scratch, 0, 8)
            reg.register = True
            if how == "default":
                init = [k for k in p.get_children() if k.kind.is_expression()]
                v = self.eval(init[-1]) if init else UNK
                if _reference(p.type) and init:
                    loc = self.lvalue(init[-1])
                    v = Ptr(loc.obj, loc.off)
            if how == "object":
                frame.vars[_key(p)] = [Loc(v.obj, v.off, ca.type_size(p.type) or 8, True), v]
            else:
                frame.vars[_key(p)] = [reg, v]
        return frame

    def _invoke(self, fdef, args_c, this=None, this_type=None, sret=None, constructor=False):
        params = [p for p in fdef.get_children() if p.kind == K.PARM_DECL]
        bound = self._bind_args(params, args_c, None)
        self._enter(fdef, bound, this, this_type, sret)
        try:
            if constructor:
                self._member_inits(fdef, this, this_type)
            body = _body(fdef)
            if body is not None:
                self.exec(body)
            return self.frames[-1].vars["__ret__"][1]
        finally:
            self.frames.pop()
            self.this_types.pop()

    def _invoke_lambda(self, closure, args_c):
        lam = closure.lambda_cursor
        params = [p for p in lam.get_children() if p.kind == K.PARM_DECL]
        bound = self._bind_args(params, args_c, None)
        self._enter(lam, bound, None, None, None, parent=closure.frame)
        try:
            body = _body(lam)
            if body is not None:
                self.exec(body)
            return self.frames[-1].vars["__ret__"][1]
        finally:
            self.frames.pop()
            self.this_types.pop()

    def _stub(self, ref):
        if ref.kind not in (K.CXX_METHOD, K.FUNCTION_DECL):
            return None
        parent = ref.semantic_parent.spelling if ref.semantic_parent is not None else ""
        for key in (f"{parent}::{ref.spelling}", ref.spelling):
            if key in self.stubs:
                v = self.stubs[key]
                if isinstance(v, str):
                    return Str(v)
                if isinstance(v, bool):
                    return int(v)
                return v
        if parent.startswith("CL") and ref.spelling in self.stubs.get("__CL__", {}):
            return self.stubs["__CL__"][ref.spelling]
        return None

    # ------------------------------------------------------------ range-for

    def _range_bounds(self, range_c):
        """begin() and end() of a range expression (an object with those methods, or an array)."""
        rng = self.eval(range_c)
        rtype = _referee(range_c.type)
        if ca.is_array(rtype):
            size = ca.type_size(rtype.get_canonical()) or 0
            return rng, Ptr(rng.obj, rng.off + size) if isinstance(rng, Ptr) else Ptr(UNK, UNK), \
                ca.type_size(rtype.get_canonical().element_type) or 4
        cname = rtype.get_canonical().spelling
        if cname.startswith("std::vector<") or cname.startswith("std::unordered_map<"):
            if not isinstance(rng, Ptr):
                return Ptr(UNK, UNK), Ptr(UNK, UNK), 4
            if cname.startswith("std::vector<"):
                et = element_type(rtype)
                return self._vfield(rng, 0), self._vfield(rng, 1), \
                    (ca.type_size(et.get_canonical()) if et is not None else None) or 8
            return self._map_method("begin", rng, [], rtype), self._map_method("end", rng, [], rtype), 16
        begin = end = None
        for m in _class_members(rtype):
            if m.kind == K.CXX_METHOD and m.spelling == "begin" and begin is None:
                begin = m
            elif m.kind == K.CXX_METHOD and m.spelling == "end" and end is None:
                end = m
        if begin is None or end is None or not isinstance(rng, Ptr):
            return Ptr(UNK, UNK), Ptr(UNK, UNK), 4
        b = self._invoke(begin.get_definition() or begin, [], this=rng, this_type=rtype)
        e = self._invoke(end.get_definition() or end, [], this=rng, this_type=rtype)
        esize = ca.pointee_size(begin.result_type) or 4
        return b, e, esize

    def _for_range(self, c):
        kids = ca.children(c)
        var, body = kids[0], kids[-1]
        range_c = [k for k in kids[1:-1] if k.kind.is_expression()][-1]
        b, e, esize = self._range_bounds(range_c)
        if not (isinstance(b, Ptr) and isinstance(e, Ptr)) or _is_unk(b.obj, b.off, e.off) or b.obj is not e.obj:
            self.data_loops += 1
            self.uncertainly(lambda: self._range_iteration(var, Ptr(UNK, UNK), body))
            return
        trips = np.broadcast_to(np.maximum(_tdiv_floor(np.asarray(e.off) - np.asarray(b.off), esize), 0),
                                (self.L,)).astype(np.int64)
        total = int(trips.sum())
        if not total:
            return
        if _breaks(body):
            return self._range_with_break(var, b, esize, trips, body)
        if self._carried(body, var):
            start = np.broadcast_to(np.asarray(b.off), (self.L,))
            for i in range(int(trips.max())):
                idx = np.nonzero(trips > i)[0]
                saved_top = self.scratch_top
                self.run_subset(idx, lambda: self._range_iteration(
                    var, Ptr(b.obj, _SCALAR(start[idx] + esize * i) if self.L > 1 else int(start[0]) + esize * i),
                    body))
                self.scratch_top = saved_top
            return
        cum = np.cumsum(trips)
        before = cum - trips
        start = np.broadcast_to(np.asarray(b.off), (self.L,))
        saved_frames, saved_w, saved_top = self.frames, self.w, self.scratch_top
        from .interp import CHUNK, _map_frames_with
        try:
            for a in range(0, total, CHUNK):
                g = np.arange(a, min(a + CHUNK, total))
                point = np.searchsorted(cum, g, side="right")
                within = g - before[point]
                self.frames = _map_frames_with(saved_frames, lambda arr: arr[point])
                self.w = saved_w[point]
                self._link(self.w, saved_w, point)
                self._range_iteration(var, Ptr(b.obj, start[point] + esize * within), body)
                self._drop(self.w)
                self.scratch_top = saved_top
        finally:
            self.frames, self.w = saved_frames, saved_w

    @staticmethod
    def _carried(body, var):
        """Does the body read an integer or pointer scalar before writing it (a value carried from
        one iteration to the next), other than the loop variable?"""
        from .interp import _events

        first, written = _events(body)
        own = _key(var)
        return any(first.get(k) == "r" for k in written if k != own)

    def _range_with_break(self, var, b, esize, trips, body):
        """Range-for whose body may break: position by position for all batch points still in it."""
        tag = object()
        bkey, ckey = ("__break__", id(tag)), ("__continue__", id(tag))
        self.loop_flags.append((bkey, ckey))
        frame = self.frames[-1]
        frame.vars[bkey] = [None, 0]
        start = np.broadcast_to(np.asarray(b.off), (self.L,))
        live = np.ones(self.L, bool)
        try:
            for i in range(int(trips.max())):
                idx = np.nonzero(live & (trips > i))[0]
                if len(idx) == 0:
                    break
                frame.vars[ckey] = [None, 0]
                saved_top = self.scratch_top

                def one(i=i, idx=idx):
                    it = Ptr(b.obj, _SCALAR(start[idx] + esize * i) if self.L > 1 or len(idx) > 1
                             else int(start[idx][0]) + esize * i)
                    self._range_iteration(var, it, body)
                self.run_subset(idx, lambda: self._fresh(one))
                self.scratch_top = saved_top
                broke = frame.vars.get(bkey, [None, 0])[1]
                if np.ndim(broke) == 0:
                    if broke:
                        break
                else:
                    live &= np.broadcast_to(np.asarray(broke) == 0, (self.L,))
        finally:
            self.loop_flags.pop()
            frame.vars.pop(bkey, None)
            frame.vars.pop(ckey, None)

    def _range_iteration(self, var, it, body):
        if _reference(var.type):
            value = it
            loc = Loc(self.scratch, 0, 8)
            loc.register = True
            self.frames[-1].vars[_key(var)] = [loc, value]
        elif _record(var.type):
            dest = self._new_storage(var.type)
            if isinstance(it, Ptr) and it.obj is not UNK:
                self._copy_object(dest, it, ca.type_size(var.type.get_canonical()) or 8)
            self.frames[-1].vars[_key(var)] = [Loc(dest.obj, dest.off, ca.type_size(var.type) or 8, True), dest]
        else:
            if isinstance(it, Ptr) and it.obj is not UNK:
                loc = Loc(it.obj, it.off, ca.type_size(var.type) or 4)
                self.access(loc)
                value = self._mem_load(loc, var.type)
            else:
                value = UNK
            self._declare_scalar(var, value)
        self.exec(body)

    # ------------------------------------------------------------ the standard library

    def _library_construct(self, call, dest, args_c):
        name = call.type.get_canonical().spelling
        if name.startswith("std::vector<"):
            return self._vector_construct(dest, args_c, call.type)
        if name.startswith("std::pair<"):
            if len(args_c) == 2:
                self._pair_store(dest, call.type, [self.eval(a) for a in args_c])
            elif len(args_c) == 1 and isinstance(src := self.eval(args_c[0]), Ptr):
                self._copy_object(dest, src, ca.type_size(call.type.get_canonical()) or 16)
            return dest
        if "mersenne_twister_engine" in name:
            words = 312 if "unsigned long, 64" in name or ", 64," in name else 624
            self._engine(dest, words, seed=True)
        elif "uniform_int_distribution" in name and len(args_c) >= 2:
            lo, hi = self.eval(args_c[0]), self.eval(args_c[1])
            self._mem_store(Loc(dest.obj, dest.off, 8), call.type, lo)
            self._mem_store(Loc(dest.obj, dest.off + 8, 8), call.type, hi)
        return dest

    def _pair_store(self, dest, type_, values):
        """Store (first, second) into a std::pair object."""
        t = type_.get_canonical()
        for f, v in zip(t.get_fields(), values):
            fo = t.get_offset(f.spelling)
            if fo is None or fo < 0 or not isinstance(dest, Ptr) or _is_unk(dest.obj, dest.off):
                continue
            loc = Loc(dest.obj, _SCALAR(np.asarray(dest.off) + fo // 8) if np.ndim(dest.off) else dest.off + fo // 8,
                      ca.type_size(f.type.get_canonical()) or 4)
            self.access(loc, write=True)
            self._mem_store(loc, f.type, v)

    def _engine(self, obj, words, seed=False, draws=0):
        """Charge a Mersenne Twister object's state words (8 bytes each): seeding writes every word
        and reads the previous one; each output reads one word, and the twist reads two more and
        writes one per word, so `draws` outputs per batch point cost 4 * draws / words per word."""
        if not self.charging or not isinstance(obj, Ptr) or _is_unk(obj.obj, obj.off):
            return
        if not isinstance(obj.obj, Obj) or obj.obj.kind == "register":
            return
        per_word = (2.0 if seed else 0.0) + float(self.w.sum()) * draws * 4.0 / words
        if per_word <= 0:
            return
        base = int(np.asarray(obj.off).ravel()[0])
        obj.obj.add(base + 8 * np.arange(words), 8, np.full(words, per_word))
        self.total += per_word * words

    def _library_cpp(self, c, ref, obj, args_c, sret):
        if ref.spelling not in PURE:                                   # may write memory
            self._forget_loads()
        name = ref.spelling
        ptype = ref.semantic_parent.type.get_canonical().spelling if ref.semantic_parent is not None else ""
        if ptype.startswith("std::vector<") and obj is not None:
            return self._vector_method(name, obj, args_c, ref.semantic_parent.type, c.type)
        if ptype.startswith("std::unordered_map<") and obj is not None:
            return self._map_method(name, obj, args_c, ref.semantic_parent.type)
        if ptype.startswith("std::pair<") and name == "operator=" and isinstance(obj, Ptr) and args_c:
            src = self.eval(args_c[0])                                 # trivially copyable: copy it
            if isinstance(src, Ptr):
                self._copy_object(obj, src, ca.type_size(ref.semantic_parent.type.get_canonical()) or 16)
            return obj
        if name == "make_pair" and len(args_c) == 2:
            dest = sret if sret is not None else self._new_storage(c.type)
            self._pair_store(dest, c.type, [self.eval(a) for a in args_c])
            return dest
        if name == "max_element" and len(args_c) >= 2:
            return self._max_element(args_c)
        if "mersenne_twister_engine" in ptype:
            words = 312 if ", 64," in ptype else 624
            if name == "seed":
                for a in args_c:
                    self.eval(a)
                self._engine(obj, words, seed=True)
                return UNK
            if name == "operator()":
                self._engine(obj, words, draws=1)
                hi = 2 ** 64 - 1 if words == 312 else 2 ** 32 - 1
                draws = self.rng.integers(0, hi, self.L, dtype=np.uint64, endpoint=True).astype(np.int64) \
                    if words == 312 else self.rng.integers(0, hi + 1, self.L, dtype=np.int64)
                return int(draws[0]) if self.L == 1 else draws
            if name in ("max", "min"):
                return (-1 if words == 312 else 2 ** 32 - 1) if name == "max" else 0   # 2^64 - 1 as int64
        if "uniform_int_distribution" in ptype and name == "operator()":
            if args_c:
                gen = self.lvalue(args_c[0])
                gtype = _referee(args_c[0].type).get_canonical().spelling
                self._engine(Ptr(gen.obj, gen.off), 312 if ", 64," in gtype else 624, draws=1)
            lo = _as_int(self._mem_load(Loc(obj.obj, obj.off, 8), c.type)) if isinstance(obj, Ptr) else UNK
            hi = _as_int(self._mem_load(Loc(obj.obj, obj.off + 8, 8), c.type)) if isinstance(obj, Ptr) else UNK
            if _is_unk(lo, hi):
                return UNK
            v = self.rng.integers(int(np.min(lo)), int(np.max(hi)) + 1, self.L, dtype=np.int64)
            return int(v[0]) if self.L == 1 else v
        if "numeric_limits" in ptype and name in ("max", "min", "lowest"):
            t = c.type.get_canonical()
            size = ca.type_size(t) or 4
            signed = t.kind in (ca.ci.TypeKind.INT, ca.ci.TypeKind.LONG, ca.ci.TypeKind.LONGLONG,
                                ca.ci.TypeKind.SHORT, ca.ci.TypeKind.SCHAR, ca.ci.TypeKind.CHAR_S)
            if ca.is_float(t):
                return UNK
            if name == "max":
                return 2 ** (8 * size - (1 if signed else 0)) - 1
            return -(2 ** (8 * size - 1)) if signed else 0
        return self._std(c, name, args_c, sret)

    def _argv_array(self):
        """argv: an array of pointers to the command-line strings, each an object that knows its text."""
        arr = Obj("argv", "stack", 8 * (len(self.argv) + 1))
        self.objects.append(arr)
        for i, text in enumerate(self.argv):
            sobj = Obj(f"argv[{i}]", "stack", len(text) + 1)
            sobj.text = text
            self.objects.append(sobj)
            self._ptr_store(Loc(arr, 8 * i, 8), Ptr(sobj, 0))
        self._ptr_store(Loc(arr, 8 * len(self.argv), 8), Ptr(self.null, 0))
        return Ptr(arr, 0)

    @staticmethod
    def _text(v):
        """The text of a string value: a literal, or a pointer into a command-line string."""
        if isinstance(v, Str):
            return v.text
        if isinstance(v, Ptr) and isinstance(getattr(v.obj, "text", None), str) and np.ndim(v.off) == 0 \
                and v.off is not UNK:
            return v.obj.text[int(v.off):]
        return None

    def _std(self, c, name, args_c, sret):
        from .interp import ATOI, RANDOM
        if name in RANDOM or name in ("srand", "srandom", "srand48"):  # the C library's generator (interp)
            return Interpreter._library(self, name, args_c)
        if name in ATOI and args_c:
            v = self._argv_value(args_c[0])
            if v is not UNK:
                return v
            text = self._text(self.eval(args_c[0]))
            for a in args_c[1:]:
                self.eval(a)
            if text is not None:
                base = 10
                if name.startswith("strto") and len(args_c) > 2:
                    b = _as_int(self.eval(args_c[2]))
                    base = b if b is not UNK else 10
                m = re.match(r"\s*([+-]?(0[xX][0-9a-fA-F]+|[0-9]+))", text)
                if m:
                    try:
                        return int(m.group(1), 0 if base == 0 else base)
                    except ValueError:
                        return UNK
                return 0
        if name in ("strcmp", "strncmp", "strcasecmp") and len(args_c) >= 2:
            a, b = (self._text(self.eval(x)) for x in args_c[:2])
            if a is not None and b is not None:
                if name == "strncmp":
                    n = _as_int(self.eval(args_c[2])) if len(args_c) > 2 else UNK
                    if n is UNK:
                        return UNK
                    a, b = a[:n], b[:n]
                if name == "strcasecmp":
                    a, b = a.lower(), b.lower()
                return (a > b) - (a < b)
        if name == "strlen" and args_c:
            t = self._text(self.eval(args_c[0]))
            if t is not None:
                return len(t)
        if name == "probe":                                            # test hook: record a value
            v = self.eval(args_c[0]) if args_c else UNK
            self.probes.append(v if not isinstance(v, np.ndarray) else v.copy())
            return UNK
        if name in ("operator==", "operator!=") and len(args_c) == 2:
            a, b = self.eval(args_c[0]), self.eval(args_c[1])
            if isinstance(a, Str) and isinstance(b, Str):
                return int((a.text == b.text) == (name == "operator=="))
            return UNK
        if name in ("min", "max") and len(args_c) >= 2:
            ra, rb = (self._value(x) for x in args_c[:2])
            if isinstance(ra, Real) or isinstance(rb, Real):
                ra, rb = _real(ra), _real(rb)
                if _is_unk(ra, rb):
                    return UNK
                return Real(np.minimum(ra.v, rb.v) if name == "min" else np.maximum(ra.v, rb.v))
            a, b = _as_int(ra), _as_int(rb)
            if _is_unk(a, b):
                return UNK
            return _SCALAR(np.minimum(a, b) if name == "min" else np.maximum(a, b))
        if name in MATH and args_c:                                    # on floating-point scalars
            vals = [self.eval(x) for x in args_c]
            if name in ("abs", "labs", "llabs") and _as_int(vals[0]) is not UNK:
                v = _as_int(vals[0])
                return _SCALAR(np.abs(v)) if np.ndim(v) else abs(v)
            reals = [_real(v) for v in vals[:MATH[name][0]]]
            if any(r is UNK for r in reals):
                return UNK
            with np.errstate(all="ignore"):
                out = MATH[name][1](*[np.asarray(r.v, dtype=float) for r in reals])
            return Real(out if np.ndim(out) else float(out))
        if name in ("sort", "stable_sort") and len(args_c) >= 2:
            b, e = self.eval(args_c[0]), self.eval(args_c[1])
            self._sort(b, e, c, args_c)
            return UNK
        if name in ("unique", "remove") and len(args_c) >= 2:
            b, e = self.eval(args_c[0]), self.eval(args_c[1])
            value = self._value(args_c[2]) if name == "remove" and len(args_c) > 2 else None
            return self._compact(b, e, name, value, args_c)
        if name == "copy" and len(args_c) == 3:
            b, e, out = (self.eval(x) for x in args_c)
            return self._copy_range(b, e, out, args_c)
        if name == "fill" and len(args_c) == 3:
            b, e, value = self.eval(args_c[0]), self.eval(args_c[1]), self._value(args_c[2])
            esize = ca.pointee_size(args_c[0].type) or 4
            seg = self._segments(b, e, esize)
            if seg is not None:
                starts, lengths = seg
                offs = self._charge_range(b.obj, starts, lengths, esize, 1.0)
                if offs is not None:
                    v = _as_int(value)
                    b.obj.store(offs, UNK if v is UNK else np.repeat(np.broadcast_to(np.asarray(v), (self.L,)),
                                                                         lengths), max(esize, 1) if esize < 4 else 4)
            return UNK
        if name in ("fill", "fill_n"):
            for x in args_c:
                self.eval(x)
            return UNK
        if name == "shuffle" and len(args_c) >= 2:
            b, e = self.eval(args_c[0]), self.eval(args_c[1])
            gen = self.lvalue(args_c[2]) if len(args_c) > 2 else None
            return self._shuffle(b, e, gen, args_c)
        if name in ("__sync_fetch_and_add", "__atomic_fetch_add") and len(args_c) >= 2:
            p, v = self.eval(args_c[0]), self.eval(args_c[1])
            if isinstance(p, Ptr):
                loc = Loc(p.obj, p.off, ca.pointee_size(args_c[0].type) or 4)
                self.access(loc)
                self.access(loc, write=True)
                old, _ = self._rmw_add(loc, args_c[0].type.get_pointee(), v)
                return old
            return UNK
        if name in ("__sync_bool_compare_and_swap", "__sync_val_compare_and_swap") and len(args_c) >= 3:
            p, old, new = (self.eval(x) for x in args_c[:3])
            if not isinstance(p, Ptr):
                return UNK
            pt = args_c[0].type.get_pointee()
            loc = Loc(p.obj, p.off, ca.pointee_size(args_c[0].type) or 4)
            self.access(loc)
            self.access(loc, write=True)
            cur = self._mem_load(loc, pt)
            o = _as_int(old) if not isinstance(old, Ptr) else UNK
            if _is_unk(cur, o) or isinstance(cur, Ptr):
                self._mem_store(loc, pt, new if isinstance(new, Ptr) else UNK)
                return UNK
            ok = np.asarray(cur) == np.asarray(o)
            nv = _as_int(new)
            if nv is not UNK:
                merged = np.where(ok, nv, cur)
                self._mem_store(loc, pt, _SCALAR(merged) if np.ndim(merged) else int(merged))
            return _SCALAR(ok.astype(np.int64)) if np.ndim(ok) else int(ok)
        if name in ("swap", "iter_swap") and len(args_c) == 2:
            a, b = (self.lvalue(x) for x in args_c)
            t = _referee(args_c[0].type)
            for loc in (a, b):
                self.access(loc)
                self.access(loc, write=True)
            if not _is_unk(a.obj, a.off, b.obj, b.off):
                size = ca.type_size(t.get_canonical()) or 8
                tmp = self._new_storage(t)
                self._copy_object(tmp, Ptr(a.obj, a.off), size, charge=False)
                self._copy_object(Ptr(a.obj, a.off), Ptr(b.obj, b.off), size, charge=False)
                self._copy_object(Ptr(b.obj, b.off), tmp, size, charge=False)
            return UNK
        if name in ("move", "forward") and len(args_c) == 1:
            return self.eval(args_c[0])
        if name in ("malloc", "calloc", "realloc", "free"):
            return Interpreter._library(self, name, args_c)
        if name in FREE:
            return Interpreter._library(self, name, args_c)
        for x in args_c:
            if x.kind.is_expression() and not _record(_referee(x.type)):
                self.eval(x)
        if sret is not None:
            return sret
        if _record(c.type):
            return self._new_storage(c.type)
        return UNK

    def _value(self, c):
        """An argument's value (through a reference parameter it is still the value)."""
        return self.eval(c)

    def _atomic(self, name, args_c):
        target = self.lvalue(args_c[0])
        t = _referee(args_c[0].type)
        self.access(target)
        self.access(target, write=True)
        if name in ("fetch_and_add", "atomic_fetch_add"):
            inc = self.eval(args_c[1]) if len(args_c) > 1 else 1
            old, _ = self._rmw_add(target, t, inc)
            return old
        old, new = (self.eval(a) for a in args_c[1:3])
        cur = self._mem_load(target, t)
        o, nv = _as_int(old), _as_int(new)
        if _is_unk(cur, o, nv) or isinstance(cur, Ptr):
            self._mem_store(target, t, UNK)
            return UNK
        offs = np.broadcast_to(np.asarray(target.off), (self.L,))
        cur = np.broadcast_to(np.asarray(cur), (self.L,)).copy()
        o = np.broadcast_to(np.asarray(o), (self.L,))
        nv = np.broadcast_to(np.asarray(nv), (self.L,))
        ok = np.zeros(self.L, dtype=np.int64)
        # in order: the first point that finds old_val swaps; later points at that location see new_val
        order = np.argsort(offs, kind="stable")
        value = {}
        for i in order:
            key = int(offs[i])
            c = value.get(key, int(cur[i]))
            if c == int(o[i]):
                ok[i] = 1
                value[key] = int(nv[i])
            else:
                value[key] = c
        keys = np.array(list(value.keys()), dtype=np.int64)
        if self.charging:
            target.obj.store(keys, np.array(list(value.values()), dtype=np.int64),
                             ca.type_size(t.get_canonical()) or 4)
        return _SCALAR(ok) if self.L > 1 else int(ok[0])

    def _max_element(self, args_c):
        """std::max_element(first, last[, comp]): walks the range calling the comparator (a lambda of
        the program, or <) on the values; returns a pointer to the largest."""
        b, e = self.eval(args_c[0]), self.eval(args_c[1])
        esize = ca.pointee_size(args_c[0].type) or 16
        if not (isinstance(b, Ptr) and isinstance(e, Ptr)) or _is_unk(b.obj, b.off, e.off) or self.L > 1:
            return Ptr(UNK, UNK)
        n = (int(e.off) - int(b.off)) // esize
        if n <= 0:
            return e
        comp = self._closure_of(args_c[2]) if len(args_c) > 2 else None
        best = int(b.off)
        for i in range(1, n):
            cur = int(b.off) + esize * i
            if comp is not None:
                less = self._invoke_lambda_ptrs(comp, [Ptr(b.obj, best), Ptr(b.obj, cur)])
            else:
                x, y = b.obj.load(best), b.obj.load(cur)
                self.access(Loc(b.obj, best, esize))
                self.access(Loc(b.obj, cur, esize))
                less = int(x < y)
            if _as_int(less) is not UNK and _as_int(less):
                best = cur
        return Ptr(b.obj, best)

    def _invoke_lambda_ptrs(self, closure, ptrs):
        """Call a lambda whose parameters are references, binding them to the given addresses."""
        lam = closure.lambda_cursor
        params = [p for p in lam.get_children() if p.kind == K.PARM_DECL]
        bound = [("ref", p, v) for p, v in zip(params, ptrs)]
        self._enter(lam, bound, None, None, None, parent=closure.frame)
        try:
            body = _body(lam)
            if body is not None:
                self.exec(body)
            return self.frames[-1].vars["__ret__"][1]
        finally:
            self.frames.pop()
            self.this_types.pop()

    # ------------------------------------------------------------ algorithms on ranges

    def _segments(self, b, e, esize):
        if not (isinstance(b, Ptr) and isinstance(e, Ptr)) or _is_unk(b.obj, b.off, e.off) or b.obj is not e.obj:
            return None
        starts = np.broadcast_to(np.asarray(b.off, dtype=np.int64), (self.L,))
        ends = np.broadcast_to(np.asarray(e.off, dtype=np.int64), (self.L,))
        lengths = np.maximum((ends - starts) // esize, 0)
        return starts, lengths

    def _charge_range(self, obj, starts, lengths, esize, refs):
        lengths = np.asarray(lengths, dtype=np.int64)
        total = int(lengths.sum())
        if total == 0 or not self.charging:
            return
        offs = np.repeat(starts, lengths) + esize * (np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths))
        w = np.repeat(self.w, lengths) * np.repeat(np.broadcast_to(refs, lengths.shape), lengths)
        obj.add(offs, esize, w)
        self.total += float(w.sum())
        return offs

    def _gather(self, obj, starts, lengths, esize):
        total = int(lengths.sum())
        offs = np.repeat(starts, lengths) + esize * (np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths))
        return offs, obj.load(offs)

    def _move_elements(self, src_obj, src, dst_obj, dst, esize):
        """Copy whole elements (every 4-byte unit, pointers and per-byte values) between offsets."""
        src, dst = np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)
        if src.size == 0:
            return
        for u in range(0, max(esize, 4), 4):
            vals = src_obj.load(src + u)
            dst_obj.store(dst + u, vals)
            sp = getattr(src_obj, "ptrs", None)
            if sp is not None and int((src + u).max()) // 4 < len(sp):
                uids = sp[(src + u) // 4]
                dvals = dst_obj._value_units(int((dst + u).max()) // 4 + 1)
                if getattr(dst_obj, "ptrs", None) is None or len(dst_obj.ptrs) < len(dvals):
                    ptrs = np.full(len(dvals), -1, dtype=np.int64)
                    if getattr(dst_obj, "ptrs", None) is not None:
                        ptrs[: len(dst_obj.ptrs)] = dst_obj.ptrs
                    dst_obj.ptrs = ptrs
                dst_obj.ptrs[(dst + u) // 4] = uids
        if getattr(src_obj, "bytevals", None) is not None:
            for u in range(esize):
                dst_obj.store(dst + u, src_obj.load(src + u, 1), 1)

    def _sort(self, b, e, c, args_c):
        """std::sort on memory values: elements are compared field by field (pairs, NodeWeight: first,
        then second), descending with std::greater; the elements move with all their bytes."""
        esize = ca.pointee_size(args_c[0].type) or 4
        seg = self._segments(b, e, esize)
        if seg is None:
            return
        starts, lengths = seg
        self._charge_range(b.obj, starts, lengths, esize, sort_refs(lengths))
        total = int(lengths.sum())
        if total == 0:
            return
        et = args_c[0].type.get_pointee().get_canonical()
        fields = []
        if et.kind == ca.ci.TypeKind.RECORD:
            for f in et.get_fields():
                fo = et.get_offset(f.spelling)
                if fo is not None and fo >= 0:
                    fields.append((fo // 8, ca.type_size(f.type.get_canonical()) or 4))
        if not fields:
            fields = [(0, esize)]
        offs = np.repeat(starts, lengths) + esize * (np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths))
        keys = [b.obj.load(offs + fo, fs if fs < 4 else 4) for fo, fs in fields]
        if any(np.any(k == SENTINEL) for k in keys):
            return
        group = np.repeat(np.arange(len(lengths)), lengths)
        descending = len(args_c) > 2 and "greater" in args_c[2].type.get_canonical().spelling
        sort_keys = [(-k if descending else k) for k in reversed(keys)] + [group]
        order = np.lexsort(sort_keys)
        for u in range(0, esize, 4):                                   # move every 4-byte unit
            vals = b.obj.load(offs + u)
            b.obj.store(offs, vals[order]) if u == 0 else b.obj.store(offs + u, vals[order])
        if getattr(b.obj, "bytevals", None) is not None:
            for u in range(esize):
                vals = b.obj.load(offs + u, 1)
                b.obj.store(offs + u, vals[order], 1)

    def _compact(self, b, e, name, value, args_c):
        """std::unique (adjacent duplicates) or std::remove (a value): compacted in place; returns the
        new end. Charged: two reads per element for unique, one for remove, plus a write per kept
        element that moves."""
        esize = ca.pointee_size(args_c[0].type) or 4
        seg = self._segments(b, e, esize)
        if seg is None:
            return Ptr(UNK, UNK)
        starts, lengths = seg
        self._charge_range(b.obj, starts, lengths, esize, 2.0 if name == "unique" else 1.0)
        offs, vals = self._gather(b.obj, starts, lengths, esize)
        if np.any(vals == SENTINEL):
            return Ptr(b.obj, UNK)
        group = np.repeat(np.arange(len(lengths)), lengths)
        if name == "unique":
            keep = np.ones(len(vals), bool)
            keep[1:] = ~((vals[1:] == vals[:-1]) & (group[1:] == group[:-1]))
        else:
            v = np.repeat(np.broadcast_to(np.asarray(_as_int(value) if value is not UNK else -1), (self.L,)), lengths)
            keep = vals != v
        kept = np.bincount(group[keep], minlength=len(lengths))
        rank = np.cumsum(keep) - 1 - np.repeat(np.cumsum(kept) - kept, lengths)
        dest = np.repeat(starts, lengths) + esize * rank
        moved = keep & (dest != offs)
        if moved.any() and self.charging:
            b.obj.add(dest[moved], esize, np.repeat(self.w, lengths)[moved])
        self._move_elements(b.obj, offs[keep], b.obj, dest[keep], esize)
        new_end = starts + esize * kept
        return Ptr(b.obj, _SCALAR(new_end) if self.L > 1 else int(new_end[0]))

    def _copy_range(self, b, e, out, args_c):
        esize = ca.pointee_size(args_c[0].type) or 4
        seg = self._segments(b, e, esize)
        if seg is None or not isinstance(out, Ptr) or _is_unk(out.obj, out.off):
            return Ptr(UNK, UNK)
        starts, lengths = seg
        self._charge_range(b.obj, starts, lengths, esize, 1.0)
        ostarts = np.broadcast_to(np.asarray(out.off, dtype=np.int64), (self.L,))
        self._charge_range(out.obj, ostarts, lengths, esize, 1.0)
        offs, _ = self._gather(b.obj, starts, lengths, esize)
        doffs = np.repeat(ostarts, lengths) + (offs - np.repeat(starts, lengths))
        self._move_elements(b.obj, offs, out.obj, doffs, esize)
        end = ostarts + esize * lengths
        return Ptr(out.obj, _SCALAR(end) if self.L > 1 else int(end[0]))

    def _shuffle(self, b, e, gen, args_c):
        esize = ca.pointee_size(args_c[0].type) or 4
        seg = self._segments(b, e, esize)
        if seg is None:
            return UNK
        starts, lengths = seg
        n = int(lengths[0])
        if gen is not None:
            gtype = _referee(args_c[2].type).get_canonical().spelling
            words = 312 if ", 64," in gtype else 624
            draws = n // 2 if n * n <= 2 ** 32 else n
            if words and draws:
                saved = self.w
                self.w = saved * draws
                self._engine(Ptr(gen.obj, gen.off), words, draws=1)
                self.w = saved
        i = np.arange(n, dtype=float)
        harmonic = np.cumsum(1.0 / np.arange(1, n + 1)) if n else np.zeros(0)
        refs = np.where(i > 0, 2.0, 0.0) + (2.0 * (harmonic[-1] - harmonic) if n else 0.0)
        self._charge_range(b.obj, starts[:1], lengths[:1], esize, 0.0)
        if n and self.charging:
            b.obj.add(int(starts[0]) + esize * np.arange(n), esize, refs * float(self.w[0]))
            self.total += float(refs.sum())
        offs, vals = self._gather(b.obj, starts[:1], lengths[:1], esize)
        if len(vals) and not np.any(vals == SENTINEL):
            b.obj.store(offs, self.rng.permutation(vals))
        return UNK


def _conjuncts(cond):
    """The parts of a && b && ... (the condition itself if it is not a conjunction)."""
    c = _strip(cond)
    if c.kind == K.BINARY_OPERATOR and ca.binary_op(c) == "&&":
        lhs, rhs = ca.children(c)
        return _conjuncts(lhs) + _conjuncts(rhs)
    return [cond]


def _float_test(cond):
    """A comparison with a floating-point operand."""
    c = _strip(cond)
    if c.kind == K.UNARY_OPERATOR and ca.unary_op(c) == "!":
        return _float_test(ca.children(c)[0])
    if c.kind == K.BINARY_OPERATOR and ca.binary_op(c) in ("<", ">", "<=", ">=", "==", "!="):
        return any(ca.is_float(k.type) for k in ca.children(c))
    if c.kind == K.CALL_EXPR and c.spelling.startswith("operator"):   # a < b through an operator
        return any(ca.is_float(k.type) for k in c.get_arguments())
    return ca.is_float(c.type)


def _walk(node):
    """All nodes of a subtree, the node first."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.get_children())


def _breaks(body):
    """Does the loop body break out of this loop (not out of a nested loop or switch)?"""
    stack = list(body.get_children()) if body.kind == K.COMPOUND_STMT else [body]
    while stack:
        n = stack.pop()
        if n.kind == K.BREAK_STMT:
            return True
        if n.kind in (K.FOR_STMT, K.WHILE_STMT, K.DO_STMT, K.CXX_FOR_RANGE_STMT, K.SWITCH_STMT, K.LAMBDA_EXPR):
            continue
        stack.extend(n.get_children())
    return False


def _tdiv_floor(a, b):
    return np.asarray(a) // b


def run(tu, entry="main", footprint="bytes", seed=0, heap_top=0, stubs=None, opt=3, argc=1, argv=None):
    """Run a C++ program from `entry` and return its access-count spectrum (see interp.run)."""
    from . import interp

    return interp.run(tu, entry, argc, footprint, seed, heap_top,
                      make=lambda tu_, seed_: CppInterpreter(tu_, seed_, stubs, opt), argv=argv)


def analyze(source, defines=(), includes=(), footprint="bytes", seed=0, heap_top=0, stubs=None, opt=3, argv=None):
    """Parse a C++ source and run it (see static.analyze)."""
    import time

    from .spectrum import Spectrum

    start = time.time()
    tu = ca.parse(source, defines, includes, cplusplus=True)
    errors = [str(d) for d in tu.diagnostics if d.severity >= 3]
    result = run(tu, footprint=footprint, seed=seed, heap_top=heap_top, stubs=stubs, opt=opt, argv=argv)
    spec = Spectrum.from_addresses(result.counts, result.sizes, result.total)
    return spec, result, errors, time.time() - start


# Silence the unused-import check for names kept for subclasses and debugging.
_ = (ctypes, _as_int)
