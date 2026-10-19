"""Models of std::vector and std::unordered_map for the C++ interpreter (interp_cpp).

A vector is three pointers in its object (start, finish, end of storage), as
in libstdc++. Buffers come from one arena object, so element pointers from
different vectors can be handled together when a vectorised loop works on
many of them at once (sssp's bins); push_back grows a buffer by doubling
(libstdc++), copying the elements and dropping the old buffer. An
unordered_map keeps its nodes (key, value) contiguously in the arena; its
iterators are pointers to nodes. Bucket arrays and node allocation are not
modelled (small, and hot only in hash-heavy code).
"""

import numpy as np

from . import clangast as ca
from .interp import UNK, Loc, Obj, Ptr, _as_int, _is_unk, _SCALAR

VECTOR_FIELDS = (0, 8, 16)  # start, finish, end_of_storage


def element_type(container_type):
    t = container_type.get_canonical()
    if t.get_num_template_arguments() > 0:
        return t.get_template_argument_type(0)
    return None


def _callee(c):
    """The name of the function a call expression calls (push_back, resize, a constructor)."""
    ref = getattr(c, "referenced", None)
    return (ref.spelling if ref is not None else c.spelling) or ""


class Containers:
    """Mixin for CppInterpreter."""

    def _arena(self):
        if getattr(self, "_arena_obj", None) is None:
            self._arena_obj = Obj("containers", "heap", 0)
            self._arena_obj.zeroed = True
            self._arena_top = 0
            self.objects.append(self._arena_obj)
            self.heap_events.append(("new", self._arena_obj))
        return self._arena_obj

    def _arena_buffer(self, start, nbytes, esize, grown_from=None):
        """A vector buffer, for static.sites: its range in the arena, the call that made it (push_back,
        resize, the constructor), when, and its growth chain (the buffers of one vector, oldest first)."""
        live = self.__dict__.setdefault("arena_live", {})
        bufs = self.__dict__.setdefault("arena_buffers", [])
        stack = self.__dict__.get("call_stack")
        prev = live.pop(grown_from, None) if grown_from is not None else None
        if prev is not None:
            prev["died"] = self.total
        b = {"start": start, "nbytes": int(nbytes), "esize": int(esize), "born": self.total, "died": None,
             "site": self.site_of(stack[-1] if stack else None), "chain": prev["chain"] if prev else len(bufs),
             "grown_from": prev["nbytes"] if prev else 0,
             "op": _callee(stack[-1]) if stack else ""}
        bufs.append(b)
        live[start] = b

    def _arena_alloc(self, nbytes):
        arena = self._arena()
        start = (self._arena_top + 15) // 16 * 16
        self._arena_top = start + max(int(nbytes), 0)
        arena.nbytes = max(arena.nbytes, self._arena_top)
        return start

    # ---------------------------------------------------------------- vector fields

    def _vfield(self, vec, k, charge=True):
        loc = Loc(vec.obj, vec.off + VECTOR_FIELDS[k], 8)
        if charge:
            self.access(loc)
        return self._ptr_load(loc)

    def _vset(self, vec, k, ptr, offs=None, pts=None):
        """Set field k of a vector. pts: the batch points doing it (a vector shared by several points,
        set once for them: charged with their weights, not the whole batch's)."""
        loc = Loc(vec.obj, (vec.off if offs is None else offs) + VECTOR_FIELDS[k], 8)
        if pts is None or self.L == 1:
            self.access(loc, write=True)
        else:
            w = np.zeros(self.L)
            w[pts] = np.broadcast_to(np.asarray(self.w, dtype=float), (self.L,))[pts]
            self.with_weights(w, lambda: self.access(loc, write=True))
        self._ptr_store(loc, ptr)

    def _vinit(self, vec, n, esize, zero=True, et=None):
        """An empty vector (n = 0) or one with n value-initialised elements."""
        n = int(np.max(_as_int(n))) if _as_int(n) is not UNK else 0
        start = self._arena_alloc(n * esize)
        if n:
            self._arena_buffer(start, n * esize, esize)
        arena = self._arena()
        p = Ptr(arena, start)
        self._vset(vec, 0, p)
        self._vset(vec, 1, Ptr(arena, start + n * esize))
        self._vset(vec, 2, Ptr(arena, start + n * esize))
        if n and zero:
            offs = start + esize * np.arange(n)
            arena.add(offs, esize, np.ones(n))
            arena.store(offs, 0, 4)
            self._init_elements(offs, et)

    def _init_elements(self, offs, et):
        """Value-initialised elements that are vectors themselves: empty, with their pointers set
        (to an empty buffer in the arena, so a vectorised loop over many of them finds one object)."""
        if et is None or not et.get_canonical().spelling.startswith("std::vector<"):
            return
        arena = self._arena()
        empty = Ptr(arena, self._arena_alloc(0))
        for k in range(3):
            self._ptr_store(Loc(arena, _SCALAR(np.asarray(offs) + VECTOR_FIELDS[k]), 8), empty)

    def _vsizes(self, vec, esize):
        """(start offs, sizes, capacities) per batch point."""
        s, f, e = (self._vfield(vec, k) for k in range(3))
        if _is_unk(s.obj, s.off, f.off, e.off):
            return None
        L = self.L
        so = np.broadcast_to(np.asarray(s.off), (L,))
        return so, (np.broadcast_to(np.asarray(f.off), (L,)) - so) // esize, \
            (np.broadcast_to(np.asarray(e.off), (L,)) - so) // esize

    def _vgrow(self, vec_off, start, size, newcap, esize):
        """Move one vector's elements to a buffer of newcap elements (libstdc++ reallocation)."""
        arena = self._arena()
        new = self._arena_alloc(newcap * esize)
        self._arena_buffer(new, newcap * esize, esize, grown_from=start)
        if size:
            self._copy_object(Ptr(arena, new), Ptr(arena, start), size * esize, charge=False)
            arena.add(start + esize * np.arange(size), esize, np.ones(size))
            arena.add(new + esize * np.arange(size), esize, np.ones(size))
            self.total += 2.0 * size
        return new

    # ---------------------------------------------------------------- methods

    def _vector_method(self, name, vec, args_c, ctype, ref_type):
        et = element_type(ctype)
        esize = (ca.type_size(et.get_canonical()) if et is not None else None) or 8
        if not isinstance(vec, Ptr) or _is_unk(vec.obj, vec.off):
            return UNK
        if name in ("size", "empty", "capacity"):
            sz = self._vsizes(vec, esize)
            if sz is None:
                return UNK
            _, n, cap = sz
            v = n if name == "size" else (n == 0).astype(np.int64) if name == "empty" else cap
            return _SCALAR(v) if self.L > 1 else int(v[0])
        if name in ("begin", "data"):
            return self._vfield(vec, 0)
        if name == "end":
            return self._vfield(vec, 1)
        if name in ("operator[]", "at"):
            s = self._vfield(vec, 0)
            i = _as_int(self.eval(args_c[0]))
            if _is_unk(s.obj, s.off, i):
                return Ptr(UNK, UNK)
            return Ptr(s.obj, _SCALAR(np.asarray(s.off) + esize * np.asarray(i)) if np.ndim(s.off) or np.ndim(i)
                       else int(s.off) + esize * int(i))
        if name in ("push_back", "emplace_back"):
            value = self.eval(args_c[0]) if args_c else UNK
            return self._push_back(vec, value, esize, et)
        if name == "resize":
            n = _as_int(self.eval(args_c[0]))
            return self._resize(vec, n, esize, et)
        if name == "clear":
            return self._resize(vec, 0, esize, et)
        if name == "reserve":
            for a in args_c:
                self.eval(a)
            return UNK
        for a in args_c:
            self.eval(a)
        return UNK

    def _push_back(self, vec, value, esize, et):
        sz = self._vsizes(vec, esize)
        if sz is None:
            return UNK
        starts, sizes, caps = sz
        vec_offs = np.broadcast_to(np.asarray(vec.off), (self.L,))
        vals = np.broadcast_to(np.asarray(value), (self.L,)) if not isinstance(value, Ptr) and value is not UNK \
            else value
        slots = np.empty(self.L, dtype=np.int64)
        arena = self._arena()
        # group the points by vector, in order: each vector takes its points' values one after another
        order = np.argsort(vec_offs, kind="stable")
        so = vec_offs[order]
        bounds = np.nonzero(np.concatenate([[True], so[1:] != so[:-1], [True]]))[0]
        for g in range(len(bounds) - 1):
            pts = order[bounds[g]:bounds[g + 1]]
            p0 = pts[0]
            start, size, cap = int(starts[p0]), int(sizes[p0]), int(caps[p0])
            need = size + len(pts)
            if need > cap:
                newcap = max(cap, 1)
                while newcap < need:
                    newcap *= 2
                start = self._vgrow(int(vec_offs[p0]), start, size, newcap, esize)
                cap = newcap
                self._vset(Ptr(vec.obj, int(vec_offs[p0])), 0, Ptr(arena, start), pts=pts[:1])
                self._vset(Ptr(vec.obj, int(vec_offs[p0])), 2, Ptr(arena, start + cap * esize), pts=pts[:1])
            slots[pts] = start + esize * (size + np.arange(len(pts)))
            self._vset(Ptr(vec.obj, int(vec_offs[p0])), 1, Ptr(arena, start + need * esize), pts=pts)
        loc = Loc(arena, _SCALAR(slots) if self.L > 1 else int(slots[0]), esize)
        self.access(loc, write=True)
        if et is not None and not ca.is_pointer(et) and et.get_canonical().kind == ca.ci.TypeKind.RECORD:
            if isinstance(value, Ptr) and self.L == 1 and value.obj is not UNK:   # an object: copied (a string
                self._copy_object(Ptr(arena, int(slots[0])), value, esize)        # keeps its text, textlib)
            return UNK
        if isinstance(value, Ptr) or (et is not None and ca.is_pointer(et)):
            self._ptr_store(loc, value if isinstance(value, Ptr) else Ptr(UNK, UNK))
        elif value is not UNK:
            arena.store(loc.off, vals if self.L > 1 else value, min(esize, 4) if esize < 4 else 4)
        return UNK

    def _resize(self, vec, n, esize, et=None):
        sz = self._vsizes(vec, esize)
        if sz is None or n is UNK:
            return UNK
        starts, sizes, caps = sz
        vec_offs = np.broadcast_to(np.asarray(vec.off), (self.L,))
        n = np.broadcast_to(np.asarray(n), (self.L,))
        arena = self._arena()
        for v in np.unique(vec_offs):
            pts = np.nonzero(vec_offs == v)[0]
            target = int(n[pts].max())                                 # sequentially, the largest request wins
            start, size, cap = int(starts[pts[0]]), int(sizes[pts[0]]), int(caps[pts[0]])
            vp = Ptr(vec.obj, int(v))
            if target > cap:
                newcap = max(target, 2 * size)
                start = self._vgrow(int(v), start, size, newcap, esize)
                self._vset(vp, 0, Ptr(arena, start), pts=pts[:1])
                self._vset(vp, 2, Ptr(arena, start + newcap * esize), pts=pts[:1])
            if target > size:                                          # new elements value-initialised
                offs = start + esize * np.arange(size, target)
                arena.add(offs, esize, np.ones(len(offs)))
                self.total += float(len(offs))
                for u in range(0, esize, 4):
                    arena.store(offs + u, 0)
                self._init_elements(offs, et)
            self._vset(vp, 1, Ptr(arena, start + target * esize), pts=pts)
        return UNK

    def _vector_construct(self, dest, args_c, ctype):
        et = element_type(ctype)
        esize = (ca.type_size(et.get_canonical()) if et is not None else None) or 8
        if args_c and ca.type_size(args_c[0].type.get_canonical()) and "vector" in \
                args_c[0].type.get_canonical().spelling:
            src = self.lvalue(args_c[0])                               # copy constructor
            srcp = Ptr(src.obj, src.off)
            sz = self._vsizes(srcp, esize)
            if sz is None:
                return self._vinit(dest, 0, esize)
            starts, sizes, _ = sz
            n = int(sizes[0])
            self._vinit(dest, n, esize, zero=False)
            if n:
                arena = self._arena()
                dst = self._vfield(dest, 0, charge=False)
                self._copy_object(Ptr(arena, int(dst.off)), Ptr(arena, int(starts[0])), n * esize, charge=False)
                arena.add(int(starts[0]) + esize * np.arange(n), esize, np.ones(n))
                arena.add(int(dst.off) + esize * np.arange(n), esize, np.ones(n))
                self.total += 2.0 * n
            return dest
        n = _as_int(self.eval(args_c[0])) if args_c else 0
        self._vinit(dest, n if n is not UNK else 0, esize, et=et)
        if len(args_c) >= 2 and n is not UNK and int(np.max(n)) > 0 and et is not None and \
                not ca.is_float(et) and esize in (4, 8):                # vector(n, value): integer fill
            v = _as_int(self.eval(args_c[1]))
            start = self._vfield(dest, 0, charge=False)
            if isinstance(start, Ptr) and not _is_unk(start.obj, start.off):
                offs = int(np.ravel(start.off)[0]) + esize * np.arange(int(np.max(n)))
                start.obj.store(offs, UNK if v is UNK or np.ndim(v) else int(v), 4)
        return dest

    # ---------------------------------------------------------------- unordered_map

    def _map_state(self, mp, ctype):
        """The map's nodes: one std::pair<const K, V> per key, `stride` bytes apart from `base`; the mapped
        value sits at `voff` in the pair (after the key, aligned), as in libstdc++'s value_type."""
        key_id = (id(mp.obj), int(np.ravel(mp.off)[0]))
        state = self._maps.get(key_id)
        if state is None:
            t = ctype.get_canonical()
            ks = vs = 4
            if t.get_num_template_arguments() >= 2:
                ks = ca.type_size(t.get_template_argument_type(0).get_canonical()) or 4
                vs = ca.type_size(t.get_template_argument_type(1).get_canonical()) or 4
            align = min(max(vs, 1), 8)
            voff = (ks + align - 1) // align * align
            stride = max(16, (voff + vs + 7) // 8 * 8)
            state = self._maps[key_id] = {"keys": {}, "base": self._arena_alloc(stride * 65536), "n": 0,
                                          "voff": voff, "stride": stride, "ksize": ks}
        return state

    def _map_method(self, name, mp, args_c, ctype):
        """unordered_map<K, V>: nodes (K, V) kept in an arena region of the map, one per key."""
        if not isinstance(mp, Ptr) or _is_unk(mp.obj, mp.off):
            return UNK
        state = self._map_state(mp, ctype)
        stride, voff = state["stride"], state["voff"]
        if name == "operator[]":
            k = _as_int(self.eval(args_c[0]))
            if k is UNK:
                return Ptr(UNK, UNK)
            ks = np.broadcast_to(np.asarray(k), (self.L,))
            slots = np.empty(self.L, dtype=np.int64)
            arena = self._arena()
            for i, kv in enumerate(ks):
                kv = int(kv)
                if kv not in state["keys"]:
                    node = state["base"] + stride * state["n"]
                    state["keys"][kv] = node
                    state["n"] += 1
                    arena.store(node, kv, min(state["ksize"], 8))
                    arena.store(node + voff, 0)
                    arena.add(np.array([node, node + voff]), 8, np.ones(2))
                slots[i] = state["keys"][kv] + voff                    # the mapped value
            return Ptr(arena, _SCALAR(slots) if self.L > 1 else int(slots[0]))
        if name == "begin":
            return Ptr(self._arena(), state["base"])
        if name == "end":
            return Ptr(self._arena(), state["base"] + stride * state["n"])
        if name == "size":
            return state["n"]
        if name in ("find", "count") and args_c:                       # the node with the key, or end
            k = _as_int(self.eval(args_c[0]))
            if k is UNK or np.ndim(k):
                return Ptr(UNK, UNK) if name == "find" else UNK
            node = state["keys"].get(int(k))
            self.total += 1.0
            if name == "count":
                return int(node is not None)
            if node is not None:
                self._arena().add(np.array([node]), 8, np.ones(1))
            return Ptr(self._arena(), node if node is not None else state["base"] + stride * state["n"])
        for a in args_c:
            self.eval(a)
        return UNK
