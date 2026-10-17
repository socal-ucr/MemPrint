"""The C++ interpreter for CUDA programs: which bytes each kernel launch reads and writes.

A CUDA source is parsed host side by libclang (clangast.parse(cuda=True), with cuda/cuda_stub.h in place of
CUDA's headers) and run from main() like any C++ program. A kernel launch k<<<grid, block>>>(args) runs
the kernel body once for all its threads together: the threads are the batch points that a vectorised
loop's iterations are in the C++ interpreter, and threadIdx, blockIdx, blockDim and gridDim are integer
values per point. Statements execute in lockstep, which is one legal schedule of a race-free kernel
(__syncthreads is a no-op), and atomics take effect in thread order. Blocks run in groups of at most
THREADS_PER_BATCH threads; __shared__ arrays have one copy per block.

Memory. cudaMalloc and friends make objects of kind device / managed / pinned; __device__ and
__constant__ variables are objects of their own (device-global) whose scalars always live in memory.
cudaMemcpy, cudaMemset, cudaMemcpyToSymbol (and memcpy / memset on the host) carry the contents they copy,
so an index array built on the host and copied to the device keeps its values there, and A[B[i]] in a
kernel resolves to addresses (Obj.copy_from).

Events. Every kernel launch, device copy and memset is an Event, and so is the host code between two of
them (a host segment). An event records, per object and byte, what it did: read or wrote (R, W), whether
only perhaps (RM, WM: under control that depends on data the interpreter does not know, or at an address
it could not resolve, which then covers the whole object), and whether the reference depends on memory
contents (RD, WD: its address or the conditions it runs under read memory, DataFacts). kdeps turns the
events into dependences between them.
"""

import os
import re

import numpy as np

from . import clangast as ca
from .clangast import K
from .interp import (Loc, Obj, Ptr, Real, SENTINEL, UNK, _SCALAR, _as_int, _is_unk, _key, _map_frames_with, _real)
from .interp_cpp import MATH, REGISTER_BYTES, CppInterpreter, InMemory, _body, _breaks, _system

THREADS_PER_BATCH = 1 << 18  # threads (whole blocks) run together
WARP_SIZE = 32
BUILTINS = ("threadIdx", "blockIdx", "blockDim", "gridDim")
LAUNCH_CONFIG = ("cudaConfigureCall", "__cudaPushCallConfiguration")
SCHEDULES = ("lockstep", "blocks", "reverse", "random")  # how a launch's threads are run (CudaInterpreter.schedule)
# math functions of device code beyond interp_cpp.MATH: the single-precision forms and CUDA's fast intrinsics,
# evaluated exactly (values stored to float memory are rounded to float there)
CUDA_MATH = {**{k + "f": v for k, v in MATH.items() if k not in ("abs", "labs", "llabs")},
             "__powf": MATH["pow"], "__expf": MATH["exp"], "__logf": MATH["log"], "__log2f": MATH["log2"],
             "__log10f": MATH["log10"], "__sinf": MATH["sin"], "__cosf": MATH["cos"], "__tanf": MATH["tan"],
             "__fsqrt_rn": MATH["sqrt"], "__fdividef": (2, np.divide),
             "__exp10f": (1, lambda x: np.power(10.0, x)), "exp10": (1, lambda x: np.power(10.0, x)),
             "exp10f": (1, lambda x: np.power(10.0, x)), "rsqrt": (1, lambda x: 1.0 / np.sqrt(x)),
             "rsqrtf": (1, lambda x: 1.0 / np.sqrt(x)), "__saturatef": (1, lambda x: np.clip(x, 0.0, 1.0)),
             "fma": (3, lambda a, b, c: a * b + c), "fmaf": (3, lambda a, b, c: a * b + c)}
MAX_TARGETS = 64  # objects a pointer loaded per thread may name before the threads are no longer split by it
ALLOCS = {"cudaMalloc": "device", "cudaMallocManaged": "managed", "cudaMallocHost": "pinned",
          "cudaHostAlloc": "pinned"}
FREES = {"cudaFree", "cudaFreeHost"}
COPIES = {"cudaMemcpy", "cudaMemcpyAsync"}
SETS = {"cudaMemset", "cudaMemsetAsync"}
SYMBOL_COPIES = {"cudaMemcpyToSymbol", "cudaMemcpyFromSymbol"}
HOST_COPIES = {"memcpy", "memmove"}
ATOMICS = {"atomicAdd", "atomicSub", "atomicExch", "atomicMin", "atomicMax", "atomicCAS", "atomicAnd", "atomicOr",
           "atomicXor", "atomicInc", "atomicDec"}
WARP = {"__shfl_sync", "__shfl_down_sync", "__shfl_up_sync", "__shfl_xor_sync", "__ballot_sync", "__any_sync",
        "__all_sync", "__syncthreads_or", "__syncthreads_and", "__syncthreads_count"}
# allocation results are addresses, not data, whatever their sizes depend on
NOT_DERIVED = set(ALLOCS) | {"malloc", "calloc", "realloc", "aligned_alloc", "fopen", "fopen64"}
FILE_IO = {"fopen", "fopen64", "fclose", "fread", "fscanf", "fgets", "feof", "fseek", "ftell", "rewind", "fgetc",
           "getc"}
COPY_KINDS = {0: "H2H", 1: "H2D", 2: "D2H", 3: "D2D", 4: "default"}
# objects whose bytes events record: device memory, host heap and globals, large host arrays (not the stack
# frames, registers or a kernel thread's own arrays)
TRACKED = {"device", "managed", "pinned", "device-global", "heap", "global", "stack-array"}
DEVICE_KINDS = {"device", "managed", "device-global"}

R, W, RM, WM, RD, WD = 1, 2, 4, 8, 16, 32
WX = 64  # with W: a write every legal schedule makes (its address and conditions read no memory; OrderFacts)
RET_KEY = ("__cuda", "returned")  # per point: has the thread returned from the current function
BLOCK_KEY = ("__cuda", "block")  # per point: its block's index within the threads being run


class Event:
    """A kernel launch, device copy or memset, or a host segment, with the bytes it read and wrote."""

    def __init__(self, index, kind, name, site):
        self.index, self.kind, self.name, self.site = index, kind, name, site
        self.grid = self.block = None
        self.threads = 0
        self.bits = {}  # Obj -> uint8 per byte (R, W, RM, WM, RD, WD, WX)
        self.all = {}  # Obj -> bits for every byte (a reference whose address is unknown)
        self.unknown = 0  # bits of references to an unknown object
        self.data_control = False  # issued under host control that depends on memory contents
        self.note = ""
        self.block_marks = None  # a kernel's references per block (CudaInterpreter.record_blocks): a list of
                                 # (blocks, byte offsets or None for the whole object, size, bits, Obj)
        self.block_unknown = False  # a reference whose object is unknown: which block made it is not kept

    def _arr(self, obj, hi):
        arr = self.bits.get(obj)
        if arr is None or len(arr) < hi:
            new = np.zeros(max(hi, obj.nbytes, 1), dtype=np.uint8)
            if arr is not None:
                new[: len(arr)] = arr
            self.bits[obj] = arr = new
        return arr

    def mark(self, obj, offs, size, bits):
        offs = np.asarray(offs, dtype=np.int64).ravel()
        offs = offs[offs >= 0]
        if offs.size == 0:
            return
        if offs.size > 1:
            offs = np.unique(offs)
        b = (offs[:, None] + np.arange(max(size, 1))).ravel() if size > 1 else offs
        arr = self._arr(obj, int(b[-1]) + 1)
        arr[b] |= _exposed(arr[b], bits)

    def mark_range(self, obj, lo, n, bits):
        if n > 0 and lo >= 0:
            arr = self._arr(obj, lo + n)
            arr[lo: lo + n] |= _exposed(arr[lo: lo + n], bits)

    def mark_all(self, obj, bits):
        self.all[obj] = self.all.get(obj, 0) | bits


def _exposed(cur, bits):
    """The bits to add for a reference to bytes whose bits so far are cur. A read of a byte the event already
    wrote for certain reads the event's own value (tmp[i] = 0; tmp[i] += ...), not an earlier event's: it is
    not recorded as a read. References are marked in the order the interpreter carries them out."""
    if not bits & (R | RM | RD):
        return np.uint8(bits)
    own = (cur & W) != 0
    if not own.any():
        return np.uint8(bits)
    out = np.full(cur.shape, bits, dtype=np.uint8)
    out[own] &= np.uint8(~(R | RM | RD) & 0xFF)
    return out


def _scalarize(v):
    """A host value of one batch point as a scalar (a launch's arguments are the same for every thread)."""
    if isinstance(v, np.ndarray) and v.size == 1:
        return int(v.ravel()[0])
    if isinstance(v, Ptr) and isinstance(v.off, np.ndarray) and v.off.size == 1:
        return Ptr(v.obj, int(v.off.ravel()[0]), v.src)
    if isinstance(v, Real) and isinstance(v.v, np.ndarray) and v.v.size == 1:
        return Real(float(v.v.ravel()[0]))
    return v


def _launch_config(c):
    """The configuration call (grid, block, shared bytes, stream) of a kernel launch, or None."""
    for k in ca.children(c):
        if k.kind == K.CALL_EXPR and k.spelling in LAUNCH_CONFIG:
            return k
    return None


def _attrs(decl):
    return {k.kind for k in decl.get_children() if k.kind.is_attribute()}


def _returns(body):
    """Does a statement contain a return (not one inside a lambda)?"""
    stack = [body]
    while stack:
        n = stack.pop()
        if n.kind == K.RETURN_STMT:
            return True
        if n.kind != K.LAMBDA_EXPR:
            stack.extend(n.get_children())
    return False


def _argv(c):
    """Is c main's argv (or argv + i)?"""
    c = ca.strip(c)
    while c.kind == K.BINARY_OPERATOR and ca.children(c):
        c = ca.strip(ca.children(c)[0])
    return c.kind == K.DECL_REF_EXPR and c.referenced is not None and c.referenced.kind == K.PARM_DECL and \
        c.spelling == "argv"


def _site_key(c):
    k = c.__dict__.get("_mp_site")
    if k is None:
        e = c.extent
        k = c._mp_site = (str(e.start.file), e.start.offset, e.end.offset, c.kind.value)
    return k


def _var_name(c):
    """The first variable or member named in an expression (a cudaMalloc destination: &d_a, &s.buf)."""
    stack = [c]
    while stack:
        n = stack.pop(0)
        if n.kind in (K.DECL_REF_EXPR, K.MEMBER_REF_EXPR) and n.spelling:
            return n.spelling
        stack.extend(n.get_children())
    return "?"


def _loc_text(c):
    if c is None or c.location.file is None:
        return ""
    return f"{str(c.location.file).rsplit('/', 1)[-1]}:{c.location.line}"


class DataFacts:
    """Which references depend on memory contents, from the source alone (flow-insensitive).

    A value is data-derived if it was loaded from memory (an array element, *p, p->f, a __device__ or
    __shared__ scalar, an atomic's result) or computed from such a value, or assigned under control that
    is; variables are data-derived if any assignment to them is (a fixpoint over all functions, through
    parameters and return values). The thread indices, a kernel's scalar arguments computed on the host
    from sizes, and loop counters are not. A reference is indirect if its address is data-derived, and
    guarded if a condition it runs under (if, loop, ?:, &&, ||, or any call site of its function) is.
    A launch is under data control if the host code issuing it is (a loop until a device flag clears)."""

    MAKES_DERIVED = {"cudaMemcpy", "cudaMemcpyAsync", "cudaMemcpyFromSymbol", "memcpy", "memmove", "fread",
                     "read"}  # &x passed as their destination makes x data-derived

    def __init__(self, functions, launches=True, jumps=False):
        self.launches = launches  # a launch under guarded host code guards the kernel
        self.jumps = jumps  # a return, break or continue under a guarded condition guards what follows it
        self._jumped = False
        self.functions = {n: f for n, f in functions.items() if not _system(f)}
        self.derived, self.ret, self.guarded_fn = set(), set(), set()
        self.sites, self.calls = {}, {}
        self.config_derived = set()  # kernels with a launch whose grid or block size is data-derived
        self._fn = None
        for _ in range(16):
            before = (len(self.derived), len(self.ret), len(self.guarded_fn))
            for name, f in self.functions.items():
                body = _body(f)
                if body is not None:
                    self._fn = name
                    self._walk(body, name in self.guarded_fn)
            if (len(self.derived), len(self.ret), len(self.guarded_fn)) == before:
                break

    def site(self, cursor):
        """(indirect, guarded) of a reference's expression."""
        return self.sites.get(_site_key(cursor), (False, False))

    def under_data_control(self, call):
        return self.calls.get(_site_key(call), False)

    # ------------------------------------------------------------ the walk

    def _set(self, decl):
        if decl is not None and decl.kind in (K.VAR_DECL, K.PARM_DECL):
            self.derived.add(_key(decl))

    def _assign(self, lhs, flag):
        t = ca.strip(lhs)
        if flag and t.kind == K.DECL_REF_EXPR:
            self._set(t.referenced)

    def _walk(self, n, g):
        k = n.kind
        kids = ca.children(n)
        if self.jumps:
            if k in (K.RETURN_STMT, K.BREAK_STMT, K.CONTINUE_STMT, K.GOTO_STMT) and g:
                self._jumped = True
            if k == K.COMPOUND_STMT:                                   # after a guarded jump, the rest is guarded
                outer, self._jumped = self._jumped, False
                for kid in kids:
                    self._walk(kid, g or self._jumped)
                self._jumped = outer or self._jumped
                return
        if k == K.IF_STMT:
            gg = g or self.value(kids[0])
            self._walk(kids[0], g)
            for kid in kids[1:]:
                self._walk(kid, gg)
            return
        if k in (K.WHILE_STMT, K.DO_STMT):
            cond = kids[0] if k == K.WHILE_STMT else kids[-1]
            gg = g or self.value(cond)
            for kid in kids:
                self._walk(kid, gg)
            return
        if k == K.FOR_STMT:
            init, cond, inc, body = ca.for_parts(n)
            if init is not None:
                self._walk(init, g)
            gg = g or (cond is not None and self.value(cond))
            for part in (cond, inc, body):
                if part is not None:
                    self._walk(part, gg)
            return
        if k == K.CONDITIONAL_OPERATOR and len(kids) == 3:
            gg = g or self.value(kids[0])
            self._walk(kids[0], g)
            self._walk(kids[1], gg)
            self._walk(kids[2], gg)
            return
        if k == K.BINARY_OPERATOR:
            op = ca.binary_op(n)
            if op in ("&&", "||"):
                self._walk(kids[0], g)
                self._walk(kids[1], g or self.value(kids[0]))
                return
            if op == "=":
                self._assign(kids[0], g or self.value(kids[1]))
        if k == K.COMPOUND_ASSIGNMENT_OPERATOR:
            self._assign(kids[0], g or self.value(kids[1]))
        if k == K.UNARY_OPERATOR and ca.unary_op(n) in ("post++", "post--", "pre++", "pre--") and g:
            self._assign(kids[0], True)
        if k == K.VAR_DECL:
            init = [x for x in kids if x.kind.is_expression()]
            if init and (g or self.value(init[-1])):
                self._set(n)
        if k == K.RETURN_STMT and self._fn is not None and (g or (kids and self.value(kids[0]))):
            self.ret.add(self._fn)
        if k == K.CALL_EXPR:
            self._call(n, g)
        if k in (K.ARRAY_SUBSCRIPT_EXPR, K.MEMBER_REF_EXPR) or (k == K.UNARY_OPERATOR and ca.unary_op(n) == "*"):
            self.sites[_site_key(n)] = (self.address(n), g)
        for kid in kids:
            self._walk(kid, g)

    def _call(self, n, g):
        ref = n.referenced
        name = ref.spelling if ref is not None else n.spelling
        args = list(n.get_arguments())
        if _launch_config(n) is not None or name in COPIES or name in SETS or name in SYMBOL_COPIES:
            self.calls[_site_key(n)] = g
        cfg = _launch_config(n)
        if cfg is not None and any(self.value(a) for a in cfg.get_arguments()):
            self.config_derived.add(name)
        if name in self.MAKES_DERIVED and args:
            t = ca.strip(args[0])
            while t.kind in (K.CSTYLE_CAST_EXPR, K.CXX_STATIC_CAST_EXPR, K.CXX_REINTERPRET_CAST_EXPR) and ca.children(t):
                t = ca.strip(ca.children(t)[-1])
            if t.kind == K.UNARY_OPERATOR and ca.unary_op(t) == "&":
                self._assign(ca.children(t)[0], True)
        if name in ATOMICS and args:
            self.sites[_site_key(n)] = (self.value(args[0]), g)
        f = self.functions.get(name)
        if f is not None and ref is not None and ref.get_definition() is not None:
            params = [p for p in f.get_children() if p.kind == K.PARM_DECL]
            for p, a in zip(params, args):
                if self.value(a):
                    self._set(p)
            if g and (self.launches or _launch_config(n) is None):
                self.guarded_fn.add(name)

    # ------------------------------------------------------------ expressions

    def address(self, n):
        """Is the address of an lvalue expression data-derived?"""
        n = ca.strip(n)
        k = n.kind
        kids = ca.children(n)
        if k == K.ARRAY_SUBSCRIPT_EXPR and len(kids) == 2:
            base, idx = kids
            if ca.is_pointer(idx.type):
                base, idx = idx, base
            return self.value(idx) or (self.address(base) if ca.is_array(base.type) else self.value(base))
        if k == K.UNARY_OPERATOR and ca.unary_op(n) == "*":
            return self.value(kids[0])
        if k == K.MEMBER_REF_EXPR:
            if not kids:
                return False
            return self.value(kids[0]) if ca.is_pointer(kids[0].type) else self.address(kids[0])
        if k == K.DECL_REF_EXPR:
            return False
        return self.value(n)

    def value(self, n):
        """Is the value of an expression data-derived?"""
        n = ca.strip(n)
        k = n.kind
        kids = ca.children(n)
        if k in (K.INTEGER_LITERAL, K.FLOATING_LITERAL, K.CHARACTER_LITERAL, K.STRING_LITERAL,
                 K.CXX_BOOL_LITERAL_EXPR, K.CXX_NULL_PTR_LITERAL_EXPR, K.CXX_UNARY_EXPR, K.CXX_NEW_EXPR):
            return False
        if k == K.DECL_REF_EXPR:
            d = n.referenced
            if d is None or d.kind not in (K.VAR_DECL, K.PARM_DECL):
                return False
            if _key(d) in self.derived:
                return True
            if d.kind == K.VAR_DECL and not ca.is_array(d.type):
                a = _attrs(d)
                return bool(a & {K.CUDADEVICE_ATTR, K.CUDACONSTANT_ATTR, K.CUDASHARED_ATTR}) and \
                    d.spelling not in BUILTINS
            return False
        if k == K.ARRAY_SUBSCRIPT_EXPR:
            if _argv(kids[0]):
                return False                                           # the command line: run parameters
            return self.address(n) if ca.is_array(n.type) else True
        if k == K.UNARY_OPERATOR:
            op = ca.unary_op(n)
            if op == "*":
                return not ca.is_array(n.type) and not _argv(kids[0])
            if op == "&":
                return self.address(kids[0])
            return self.value(kids[0])
        if k == K.MEMBER_REF_EXPR:
            if not kids:
                return True
            base = ca.strip(kids[0])
            if base.kind == K.DECL_REF_EXPR and base.spelling in BUILTINS:
                return False
            return True if ca.is_pointer(kids[0].type) else self.value(kids[0])
        if k == K.CALL_EXPR:
            ref = n.referenced
            name = ref.spelling if ref is not None else n.spelling
            if name in ATOMICS or name in WARP or name == "__ldg":
                return True
            if name in NOT_DERIVED:
                return False
            args = list(n.get_arguments())
            if name in self.functions and name in self.ret:
                return True
            return any(self.value(a) for a in args)
        return any(self.value(x) for x in kids if x.kind.is_expression())


class OrderFacts(DataFacts):
    """Which references a launch makes in every legal schedule, given its arguments: DataFacts within the
    kernel, where whether the host issues the launch at all does not count (the launch's own event says so),
    and a guarded return, break or continue guards what follows it."""

    def __init__(self, functions):
        super().__init__(functions, launches=False, jumps=True)


class _Diverge(Exception):
    """A pointer loaded from memory names a different object for different threads: the statement being run,
    and the rest of its block, run once per group of threads that share a target (CudaInterpreter._stmt)."""

    def __init__(self, groups):
        super().__init__()
        self.groups = groups


class CudaInterpreter(CppInterpreter):
    COPY_VALUES = True

    def __init__(self, tu, seed=0, stubs=None, opt=3, host_events=True):
        self.kernel = None  # the Event of the launch being run
        self.shared = {}  # __shared__ variable key -> (Obj, bytes per block), for the blocks being run
        self.dyn_shared = 0
        super().__init__(tu, seed, stubs, opt)
        self.host_events = host_events
        self.events = []
        self.host_event = None
        self.tracker = None  # kdeps.Tracker: events are folded into dependences as they close
        self._gpu_work_memo = {}
        self.facts = DataFacts(self.functions)
        self.order_facts = OrderFacts(self.functions)
        self.schedule = "lockstep"  # all threads statement by statement; or one block after another, in order,
                                    # in reverse, or in a random order (rng): other legal schedules
        self.oob = {}  # source location (object) -> references past the end of a device buffer
        self.racy_seen = set()  # kernels with a launch that read and wrote one unit (another order may differ)
        self.record_blocks = False  # also keep which block made each reference (Event.block_marks)
        self._b0, self._nb = 0, 1  # the first block of the threads being run, and how many blocks they are
        self.racy = None  # in a schedule other than lockstep: the kernels to reorder (None: all), as racy_seen
        self.effects = 0  # stores so far: a statement can be run again for a subset only if it made none
        self._stmt_marks = []  # per statement being run in a kernel block: (effects, batch weights) at its start

    # ------------------------------------------------------------ events

    def _new_event(self, kind, name, cursor):
        self._close_event()
        ev = Event(len(self.events), kind, name, _loc_text(cursor))
        self.events.append(ev)
        return ev

    def _close_event(self):
        """Fold the last event into the dependences (kdeps.Tracker) and free its byte maps."""
        if self.events and getattr(self.events[-1], "open", True):
            ev = self.events[-1]
            ev.open = False
            if ev.kind == "kernel" and any(((b & (R | RM)) != 0).any() and ((b & (W | WM)) != 0).any() and
                                           (((b & (R | RM)) != 0) & ((b & (W | WM)) != 0)).any()
                                           for b in ev.bits.values()):
                self.racy_seen.add(ev.name)
            if self.tracker is not None:
                self.tracker.add(ev)
            ev.summary = {obj.name: int(np.count_nonzero(b)) for obj, b in ev.bits.items()}
            ev.bits = {}
        self.host_event = None

    def _current_event(self):
        if self.kernel is not None:
            return self.kernel
        if not self.host_events:
            return None
        if self.host_event is None:
            self.host_event = self._new_event("host", f"host#{sum(e.kind == 'host' for e in self.events)}", None)
        return self.host_event

    def finish(self):
        self._close_event()

    # ------------------------------------------------------------ references

    def access(self, loc, write=False):
        if write:
            self.effects += 1
        if self.charging and not loc.array:
            self._note(loc, write)
        return super().access(loc, write)

    def _mem_store(self, loc, type_, value):
        self.effects += 1
        return super()._mem_store(loc, type_, value)

    def _library_cpp(self, c, ref, obj, args_c, sret):
        m = CUDA_MATH.get(ref.spelling)
        if m is None or not args_c:
            return super()._library_cpp(c, ref, obj, args_c, sret)
        reals = [_real(self.eval(x)) for x in args_c][: m[0]]
        if len(reals) < m[0] or any(r is UNK for r in reals):
            return UNK
        with np.errstate(all="ignore"):
            out = m[1](*[np.asarray(r.v, dtype=float) for r in reals])
        return Real(out if np.ndim(out) else float(out))

    def _loop_bound(self, bound_c):
        return self.eval(bound_c)                                      # read once per point (CppInterpreter)

    def _ptr_load(self, loc):
        """A pointer loaded from memory. In a kernel, where the threads load pointers to different objects
        (a table of device pointers indexed by data), the threads are split by target (_Diverge) if the
        statement has had no effect yet; otherwise the pointer is unknown, as on the host."""
        p = super()._ptr_load(loc)
        if p.obj is not UNK or self.kernel is None or not self._stmt_marks:
            return p
        effects, w = self._stmt_marks[-1]
        if effects != self.effects or w is not self.w:
            return p
        groups = self._targets(loc)
        if groups is None:
            return p
        raise _Diverge(groups)

    def _targets(self, loc):
        """The batch points grouped by the object the pointer at loc names, or None."""
        ptrs = getattr(loc.obj, "ptrs", None)
        if ptrs is None or not isinstance(loc.off, np.ndarray) or loc.off.shape != (self.L,):
            return None
        units = loc.off // 4
        if units.min() < 0 or units.max() >= len(ptrs):
            return None
        uids = ptrs[units]
        if np.any(uids == -1):
            return None
        keys = np.unique(uids)
        if not 1 < len(keys) <= MAX_TARGETS:
            return None
        return [np.flatnonzero(uids == k) for k in keys]

    def _note(self, loc, write):
        """Record a reference in the current event."""
        obj = loc.obj
        if isinstance(obj, Obj) and obj.kind not in TRACKED and obj.kind != "device-global":
            return
        if isinstance(obj, Obj) and self.kernel is not None and obj.kind == "stack-array":
            return                                                     # an object a thread keeps for itself
        ev = self._current_event()
        if ev is None:
            return
        bits = W if write else R
        if self.uncertain_depth:
            bits <<= 2
        cur = getattr(loc, "cursor", None)
        if cur is not None:
            indirect, guarded = self.facts.site(cur)
            if indirect or guarded or getattr(self, "_guarded_depth", 0):
                bits |= WD if write else RD
        elif getattr(self, "_guarded_depth", 0):
            bits |= WD if write else RD
        if write and not self.uncertain_depth and cur is not None and self.kernel is not None and \
                self.kernel.name not in self.order_facts.config_derived and not any(self.order_facts.site(cur)):
            bits |= WX
        if not isinstance(obj, Obj):
            ev.unknown |= (bits & (RD | WD)) | (WM if write else RM)
            ev.block_unknown = True
            return
        off = loc.off
        if off is UNK or not isinstance(off, (int, np.integer, np.ndarray)):
            may = (bits & (RD | WD)) | (WM if write else RM)
            ev.mark_all(obj, may)
            self._note_blocks(ev, obj, None, 0, may)
            return
        if obj.kind in ("device", "managed", "device-global") and obj.nbytes > 0:
            offs = np.asarray(off, dtype=np.int64)
            out = (offs < 0) | (offs + (loc.size or 8) > obj.nbytes)
            if out.any():                                              # past the end of its buffer: on the GPU
                site = f"{_loc_text(cur)} ({obj.name})" if cur is not None else obj.name   # it reaches whatever
                self.oob[site] = self.oob.get(site, 0) + int(np.count_nonzero(out))    # lies there: any buffer
                ev.unknown |= (bits & (RD | WD)) | (WM if write else RM)
                ev.block_unknown = True
                ev.note = (ev.note + "; " if ev.note else "") + "out of bounds" if "out of bounds" not in ev.note \
                    else ev.note
                if out.all():
                    return
                self._note_blocks(ev, obj, off, loc.size or 8, bits, keep=~out if offs.ndim else None)
                off = offs[~out] if offs.ndim else off
                ev.mark(obj, off, loc.size or 8, bits)
                return
        self._note_blocks(ev, obj, off, loc.size or 8, bits)
        ev.mark(obj, off, loc.size or 8, bits)

    def _note_blocks(self, ev, obj, off, size, bits, keep=None):
        """Keep which blocks made a reference (record_blocks): the block of each thread taking part, from the
        per-thread block index of the threads being run; every block being run if the shapes do not pair up."""
        if ev.block_marks is None or self.kernel is None or ev is not self.kernel:
            return
        blk = self.frames[0].vars.get(BLOCK_KEY, [None, 0])[1]
        blk = self._b0 + (np.asarray(blk, dtype=np.int64) if np.ndim(blk) else int(blk))
        if off is not None and np.ndim(off) and np.ndim(blk) and np.size(off) == np.size(blk):
            blk = np.asarray(blk).ravel()
            off = np.asarray(off, dtype=np.int64).ravel()
            if keep is not None:
                blk, off = blk[np.asarray(keep).ravel()], off[np.asarray(keep).ravel()]
        elif off is not None and np.ndim(off) == 0:
            blk = np.unique(np.asarray(blk).ravel())
            off = np.full(len(blk), int(off), dtype=np.int64)
        else:                                                          # every block, every offset
            blocks = np.arange(self._b0, self._b0 + self._nb, dtype=np.int64)
            if off is not None:
                offs = np.asarray(off, dtype=np.int64).ravel()
                if keep is not None:
                    offs = offs[np.asarray(keep).ravel()]
                offs = np.unique(offs)
                blk, off = np.repeat(blocks, len(offs)), np.tile(offs, len(blocks))
            else:
                blk = blocks
        ev.block_marks.append((np.asarray(blk, dtype=np.int64), off, size, bits, obj))

    def lvalue(self, c):
        s = ca.strip(c)
        if s.kind == K.DECL_REF_EXPR and s.referenced is not None and _key(s.referenced) in self.shared:
            entry = self._lookup(s.referenced)
            loc = entry[1].loc if isinstance(entry[1], InMemory) else entry[0]
            return loc
        loc = super().lvalue(c)
        if s.kind in (K.ARRAY_SUBSCRIPT_EXPR, K.MEMBER_REF_EXPR) or \
                (s.kind == K.UNARY_OPERATOR and ca.unary_op(s) == "*"):
            if isinstance(loc, Loc) and isinstance(loc.obj, Obj) and loc.obj.kind != "register":
                loc.cursor = s                                         # a fresh Loc: tag it with its expression
        return loc

    def _load_lvalue(self, c):
        decl, entry = self._scalar_entry(c)
        if entry is not None and isinstance(entry[1], InMemory) and entry[1].loc.obj.kind in ("device-global", "shared"):
            loc = entry[1].loc
            self.access(loc)
            return self._mem_load(loc, c.type)
        return super()._load_lvalue(c)

    def _store(self, c, value):
        decl, entry = self._scalar_entry(c)
        if entry is not None and isinstance(entry[1], InMemory) and entry[1].loc.obj.kind in ("device-global", "shared"):
            loc = entry[1].loc
            self.access(loc, write=True)
            self._mem_store(loc, c.type, value)
            return loc
        loc = super()._store(c, value)
        self.effects += 1                                              # after the target is evaluated (_ptr_load)
        return loc

    # ------------------------------------------------------------ variables

    def _declare_global(self, c):
        a = _attrs(c)
        if c.spelling in BUILTINS or c.spelling == "warpSize":
            self.frames[0].vars[_key(c)] = [Loc(UNK, UNK, 4), UNK]
            return
        managed = K.UNEXPOSED_ATTR in a and any(t.spelling == "__managed__" for t in c.get_tokens())
        if not (a & {K.CUDADEVICE_ATTR, K.CUDACONSTANT_ATTR}) and not managed:
            return super()._declare_global(c)
        size = ca.type_size(c.type) or 8
        obj = Obj(c.spelling, "device-global", size, zeroed=True)
        obj.site = {"loc": (str(c.location.file).rsplit("/", 1)[-1], c.location.line), "context": []}
        self.objects.append(obj)
        if ca.is_array(c.type):
            self.frames[0].vars[_key(c)] = [Loc(obj, 0, size, True), Ptr(obj, 0)]
            return
        slot = Loc(obj, 0, size)
        init = self._initializer(c)
        if init is not None:
            v = ca.evaluate(init)
            if isinstance(v, int):
                obj.store(0, v, size)
            elif isinstance(v, float):
                obj.fstore(0, v, size)
        self.frames[0].vars[_key(c)] = [slot, InMemory(slot)]

    def _lookup(self, decl):
        k = _key(decl)
        if k in self.shared:                                           # one copy per block
            obj, size, array = self.shared[k]
            blk = self.frames[0].vars[BLOCK_KEY][1]
            off = _SCALAR(np.asarray(blk) * size) if np.ndim(blk) else int(blk) * size
            loc = Loc(obj, off, size, array)
            return [loc, Ptr(obj, off)] if array else [loc, InMemory(loc)]
        return super()._lookup(decl)

    def _exec_decl(self, decl):
        if self.kernel is not None:
            a = _attrs(decl)
            if K.CUDASHARED_ATTR in a:
                k = _key(decl)
                if k not in self.shared:
                    size = ca.type_size(decl.type) if decl.type.get_canonical().kind != ca.ci.TypeKind.INCOMPLETEARRAY \
                        else None
                    size = size or max(int(self.dyn_shared), 0)        # extern __shared__ T s[]: the launch's bytes
                    nblocks = int(np.max(self.frames[0].vars[BLOCK_KEY][1])) + 1
                    obj = Obj(f"{self.kernel.name}.{decl.spelling}", "shared", size * nblocks)
                    self.shared[k] = (obj, size, ca.is_array(decl.type))
                return
            t = decl.type
            size = ca.type_size(t.get_canonical()) or 0
            if ca.is_array(t) and size > REGISTER_BYTES and decl.storage_class.name != "STATIC":
                # a thread's own array: one copy per thread
                obj = Obj(f"{self.kernel.name}.{decl.spelling}", "stack-array", size * self.L)
                off = size * np.arange(self.L) if self.L > 1 else 0
                self.frames[-1].vars[_key(decl)] = [Loc(obj, off, size, True), Ptr(obj, off)]
                return
        return super()._exec_decl(decl)

    def eval(self, c):
        if c.kind == K.MEMBER_REF_EXPR:
            kids = ca.children(c)
            if kids:
                base = ca.strip(kids[0])
                if base.kind == K.DECL_REF_EXPR and base.spelling in BUILTINS and base.referenced is not None \
                        and _system_stub(base.referenced):
                    entry = self.frames[0].vars.get(("__cuda", base.spelling, c.spelling))
                    return entry[1] if entry is not None and self.kernel is not None else UNK
        elif c.kind == K.DECL_REF_EXPR and c.spelling == "warpSize" and c.referenced is not None \
                and _system_stub(c.referenced):
            return 32
        return super().eval(c)

    # ------------------------------------------------------------ control in kernels

    def _enter(self, fdef, bound, this, this_type, sret, parent=None):
        frame = super()._enter(fdef, bound, this, this_type, sret, parent)
        if self.kernel is not None:
            frame.vars[RET_KEY] = [None, 0]
        return frame

    def _active_points(self):
        act = super()._active_points()
        if self.kernel is None:
            return act
        e = self.frames[-1].vars.get(RET_KEY)
        if e is None or (np.ndim(e[1]) == 0 and not e[1]):
            return act
        stop = np.broadcast_to(np.asarray(e[1]) != 0, (self.L,))
        if act is not None:
            keep = np.zeros(self.L, dtype=bool)
            keep[act] = True
            stop = stop | ~keep
        return np.nonzero(~stop)[0]

    def exec(self, c):
        kind = c.kind
        if self.kernel is not None and kind == K.COMPOUND_STMT and not getattr(self.frames[-1], "returned", False):
            return self._block(c)
        if self.kernel is not None and kind == K.RETURN_STMT:
            frame = self.frames[-1]
            if getattr(frame, "returned", False):
                return
            super().exec(c)
            done = np.ones(self.L, dtype=np.int64) if self.L > 1 else 1
            frame.vars[RET_KEY] = [None, done]
            if self.loop_flags and self.loop_flags[-1][0] in frame.vars:
                frame.vars[self.loop_flags[-1][0]] = [None, done]     # leave the loop being run too
            return
        if kind in (K.WHILE_STMT, K.DO_STMT):
            kids = ca.children(c)
            cond, body = (kids[0], kids[-1]) if kind == K.WHILE_STMT else (kids[-1], kids[0])
            if self._one_at_a_time(body):
                if kind == K.DO_STMT:
                    self.exec(body)
                return self._loop_with_break(cond, None, body)
        return super().exec(c)

    def _block(self, c):
        """A block in a kernel: as CppInterpreter.exec runs it, but a statement whose threads load pointers
        to different objects runs, with the rest of the block, once per group of threads (_stmt)."""
        saved_top = self.scratch_top
        self.scopes.append([])
        try:
            self._stmts(list(c.get_children()))
        finally:
            scope = self.scopes.pop()
            for decl, dest, t in reversed(scope):
                if decl is not None:
                    self._destroy(dest, t)
            self.scratch_top = max(saved_top, 0) if not scope else self.scratch_top

    def _stmts(self, kids):
        for i in range(len(kids)):
            if getattr(self.frames[-1], "returned", False):
                break
            active = self._active_points()
            if active is None:
                done = self._stmt(kids, i)
            elif len(active):
                done = self.run_subset(active, lambda i=i: self._stmt(kids, i))
            else:
                break
            if done:
                break

    def _stmt(self, kids, i):
        """Run kids[i]; True if it diverged and the rest of the block has been run per group of threads."""
        self._stmt_marks.append((self.effects, self.w))
        try:
            self.exec(kids[i])
            return False
        except _Diverge as d:
            groups = d.groups
        finally:
            self._stmt_marks.pop()
        for g in groups:
            self.run_subset(g, lambda: self._stmts(kids[i:]))
        return True

    def _for(self, c):
        init, cond, inc, body = ca.for_parts(c)
        if self._one_at_a_time(body):
            if init is not None:
                self.exec(init) if init.kind == K.DECL_STMT else self.eval(init)
            return self._loop_with_break(cond, inc, body)
        return super()._for(c)

    def _one_at_a_time(self, body):
        """Loops that run one iteration at a time: in a kernel, those a thread may return from; on the
        host, those that launch kernels or copy device memory (in program order)."""
        if self.kernel is not None:
            return _returns(body) and not _breaks(body)
        return self._gpu_work(body)

    def _gpu_work(self, node, depth=0):
        key = _site_key(node)
        hit = self._gpu_work_memo.get(key)
        if hit is not None:
            return hit
        self._gpu_work_memo[key] = False                               # recursion
        found = False
        for n in _walk_nodes(node):
            if n.kind != K.CALL_EXPR:
                continue
            name = n.spelling
            if _launch_config(n) is not None or name in COPIES or name in SETS or name in SYMBOL_COPIES:
                found = True
                break
            f = self.functions.get(name)
            if f is not None and depth < 8 and not _system(f):
                body = _body(f)
                if body is not None and self._gpu_work(body, depth + 1):
                    found = True
                    break
        self._gpu_work_memo[key] = found
        return found

    # ------------------------------------------------------------ calls

    def _call(self, c, sret=None):
        ref = c.referenced
        name = ref.spelling if ref is not None else c.spelling
        if _launch_config(c) is not None:
            return self._launch(c)
        if name in ATOMICS and self.kernel is not None:
            return self._atomic_cuda(name, c)
        if name in WARP:
            for a in c.get_arguments():
                self.eval(a)
            return UNK
        if name in ("__syncthreads", "__syncwarp", "__threadfence", "__threadfence_block"):
            return UNK
        if name == "__ldg":
            args = list(c.get_arguments())
            p = self.eval(args[0]) if args else UNK
            if not isinstance(p, Ptr):
                return UNK
            t = args[0].type.get_canonical().get_pointee()
            loc = Loc(p.obj, p.off, ca.type_size(t) or 4)
            loc.cursor = c
            self.access(loc)
            return self._mem_load(loc, t)
        if name in FILE_IO and (ref is None or _system(ref)):
            return self._file_io(name, c)
        if ref is not None and (name in ALLOCS or name in FREES or name in COPIES or name in SETS or
                                name in SYMBOL_COPIES or name in HOST_COPIES or name == "memset" or
                                (name.startswith("cuda") and _system_stub(ref))):
            return self._runtime(name, c)
        return super()._call(c, sret)

    def _runtime(self, name, c):
        if self.L > 1 and (name in ALLOCS or name in COPIES or name in SETS or name in SYMBOL_COPIES):
            for p in range(self.L):                                    # in a batched host region, one point at
                self.run_subset(np.array([p]), lambda: self._runtime(name, c))  # a time: an object or event each
            return 0
        args_c = list(c.get_arguments())
        if name in ALLOCS:
            p = self.eval(args_c[0])
            n = _as_int(self.eval(args_c[1])) if len(args_c) > 1 else UNK
            for a in args_c[2:]:
                self.eval(a)
            nbytes = int(np.max(n)) if n is not UNK else 0
            obj = Obj(f"{_var_name(args_c[0])}@{c.location.line}", ALLOCS[name], nbytes)
            obj.site = self.site_of(c)
            obj.alloc_kind = name
            if ALLOCS[name] in ("device", "managed"):                  # the n-th device allocation (kdeps_score)
                obj.alloc_index = self.__dict__.setdefault("device_allocs", 0)
                self.device_allocs += 1
                if self.tracker is not None:
                    self.tracker.register(obj)
            self.objects.append(obj)
            if isinstance(p, Ptr) and isinstance(p.obj, Obj) and p.off is not UNK:
                self._ptr_store(Loc(p.obj, p.off, 8), Ptr(obj, 0))
            return 0
        if name in FREES:
            for a in args_c:
                self.eval(a)
            return 0
        if name in COPIES or name in HOST_COPIES:
            args = [self.eval(a) for a in args_c]
            n = _as_int(args[2]) if len(args) > 2 else UNK
            if name in COPIES:
                kind = _as_int(args[3]) if len(args) > 3 else UNK
                ev = self._new_event("copy", f"{name} {COPY_KINDS.get(kind, '?') if kind is not UNK else '?'}", c)
                ev.data_control = self.facts.under_data_control(c) or bool(self.uncertain_depth)
            else:
                ev = self._current_event()
            self._copy(ev, args[0], args[1], n)
            if name in COPIES:
                self._close_event()
                return 0
            return args[0]
        if name in SETS or name == "memset":
            args = [self.eval(a) for a in args_c]
            n = _as_int(args[2]) if len(args) > 2 else UNK
            if name in SETS:
                ev = self._new_event("set", name, c)
                ev.data_control = self.facts.under_data_control(c) or bool(self.uncertain_depth)
            else:
                ev = self._current_event()
            self._fill(ev, args[0], _as_int(args[1]) if len(args) > 1 else UNK, n)
            if name in SETS:
                self._close_event()
                return 0
            return args[0]
        if name in SYMBOL_COPIES:
            to = name == "cudaMemcpyToSymbol"
            sym_c, other_c = (args_c[0], args_c[1]) if to else (args_c[1], args_c[0])
            sym = self.lvalue(sym_c)
            other = self.eval(other_c)
            n = _as_int(self.eval(args_c[2])) if len(args_c) > 2 else UNK
            off = _as_int(self.eval(args_c[3])) if len(args_c) > 3 and ca.children(args_c[3]) else 0
            off = 0 if off is UNK else off
            symp = Ptr(sym.obj, sym.off + off) if isinstance(sym.obj, Obj) and sym.off is not UNK else Ptr(UNK, UNK)
            ev = self._new_event("copy", f"{name} {sym_c.spelling or _var_name(sym_c)}", c)
            ev.data_control = self.facts.under_data_control(c) or bool(self.uncertain_depth)
            self._copy(ev, symp if to else other, other if to else symp, n)
            self._close_event()
            return 0
        if name in ("cudaGetSymbolAddress", "cudaGetSymbolSize") and len(args_c) > 1:
            p = self.eval(args_c[0])
            sym = self.lvalue(args_c[1])
            if isinstance(p, Ptr) and isinstance(p.obj, Obj) and p.off is not UNK:
                if name == "cudaGetSymbolAddress":
                    self._ptr_store(Loc(p.obj, p.off, 8), Ptr(sym.obj, sym.off) if isinstance(sym.obj, Obj)
                                    else Ptr(UNK, UNK))
                else:
                    p.obj.store(p.off, sym.obj.nbytes if isinstance(sym.obj, Obj) else UNK, 8)
            return 0
        if name == "cudaGetDeviceCount" and args_c:
            p = self.eval(args_c[0])
            if isinstance(p, Ptr) and isinstance(p.obj, Obj) and p.off is not UNK:
                p.obj.store(p.off, 1, 4)
            return 0
        for a in args_c:                                               # synchronisation, events, streams, errors
            if a.kind.is_expression():
                self.eval(a)
        return 0

    # ------------------------------------------------------------ input files

    def _file_io(self, name, c):
        """The C library's stdio on real files: a program's input (a graph, a matrix) is read from the file its
        command line names, so the values it loads are known (index arrays resolve)."""
        args_c = list(c.get_arguments())
        if name in ("fopen", "fopen64"):
            path = self._text(self.eval(args_c[0])) if args_c else None
            mode = self._text(self.eval(args_c[1])) if len(args_c) > 1 else "r"
            if path is None or "r" not in (mode or "r") or not os.path.isfile(path):
                return Ptr(self.null, 0)
            with open(path, "rb") as f:
                data = f.read()
            obj = Obj(f"FILE {os.path.basename(path)}", "file", 0)
            obj.data, obj.pos = data, 0
            return Ptr(obj, 0)
        args = [self.eval(a) for a in args_c]
        fp = args[{"fread": 3, "fgets": 2}.get(name, 0)] if args else None
        f = fp.obj if isinstance(fp, Ptr) and isinstance(fp.obj, Obj) and hasattr(fp.obj, "data") else None
        if name == "fclose":
            return 0
        if f is None:
            return UNK
        if name == "feof":
            return int(f.pos >= len(f.data))
        if name == "ftell":
            return f.pos
        if name == "rewind":
            f.pos = 0
            return UNK
        if name == "fseek":
            off, whence = _as_int(args[1]), _as_int(args[2])
            if _is_unk(off, whence) or np.ndim(off):
                return UNK
            f.pos = max(0, min(len(f.data), int(off) + {0: 0, 1: f.pos, 2: len(f.data)}.get(int(whence), 0)))
            return 0
        if name in ("fgetc", "getc"):
            if f.pos >= len(f.data):
                return -1
            f.pos += 1
            return f.data[f.pos - 1]
        if name == "fread":
            return self._fread(f, args_c, args)
        if name == "fgets":
            return self._fgets(f, args)
        return self._fscanf(f, args_c, args)

    def _fread(self, f, args_c, args):
        size, count = _as_int(args[1]), _as_int(args[2])
        buf = args[0]
        if _is_unk(size, count) or np.ndim(size) or np.ndim(count) or not isinstance(buf, Ptr) or \
                not isinstance(buf.obj, Obj) or buf.off is UNK or np.ndim(buf.off):
            return UNK
        n = min(int(size) * int(count), len(f.data) - f.pos)
        raw = f.data[f.pos: f.pos + n]
        f.pos += n
        self._charge(self._current_event(), buf.obj, int(buf.off), n, True)
        t = _pointee(args_c[0])
        self._store_raw(buf.obj, int(buf.off), raw, t)
        return n // int(size) if int(size) else 0

    def _store_raw(self, obj, off, raw, t):
        """Bytes read from a file into memory, kept as values of the buffer's element type."""
        size = ca.type_size(t.get_canonical()) if t is not None else None
        if not self.charging or not raw:
            return
        if t is not None and ca.is_float(t) and size in (4, 8):
            vals = np.frombuffer(raw[: len(raw) // size * size], dtype=np.float32 if size == 4 else np.float64)
            obj.fstore(off + size * np.arange(len(vals)), vals.astype(float), size)
            return
        if size in (1, 2):
            obj.store(off + np.arange(len(raw)), np.frombuffer(raw, dtype=np.uint8).astype(np.int64), 1)
            if size == 1:
                return
        width = 8 if size == 8 else 4
        vals = np.frombuffer(raw[: len(raw) // width * width], dtype=np.int64 if width == 8 else np.int32)
        obj.store(off + width * np.arange(len(vals)), vals.astype(np.int64), 4)

    def _fgets(self, f, args):
        buf, n = args[0], _as_int(args[1])
        if f.pos >= len(f.data):
            return Ptr(self.null, 0)
        if not isinstance(buf, Ptr) or not isinstance(buf.obj, Obj) or buf.off is UNK or n is UNK or np.ndim(n) \
                or np.ndim(buf.off):
            return UNK
        end = f.data.find(b"\n", f.pos)
        end = len(f.data) if end < 0 else end + 1
        line = f.data[f.pos: min(end, f.pos + int(n) - 1)]
        f.pos += len(line)
        self._charge(self._current_event(), buf.obj, int(buf.off), len(line) + 1, True)
        if self.charging:
            buf.obj.store(int(buf.off) + np.arange(len(line) + 1),
                          np.frombuffer(line + b"\0", dtype=np.uint8).astype(np.int64), 1)
        buf.obj.text = line.decode(errors="replace")
        return buf

    def _fscanf(self, f, args_c, args):
        """fscanf with %d %i %u %ld %lld %x %f %lf %e %g %s %c: one point at a time, in order."""
        fmt = self._text(args[1]) if len(args) > 1 else None
        if fmt is None:
            return UNK
        specs = re.findall(r"%[*]?\d*(?:hh|h|ll|l|L|z)?[diuxXfeEgGsc]", fmt)
        slots, k = [], 2
        for spec in specs:                                             # %*d is read and dropped
            slots.append(None if "*" in spec else k)
            k += "*" not in spec
        targets = [(spec, args[k], args_c[k]) for spec, k in zip(specs, slots) if k is not None and k < len(args)]
        L = self.L
        values = [[] for _ in specs]
        counts = []
        for _ in range(L):
            got = 0
            for i, spec in enumerate(specs):
                text = f.data
                while f.pos < len(text) and text[f.pos: f.pos + 1].isspace() and spec[-1] != "c":
                    f.pos += 1
                if f.pos >= len(text):
                    values[i].append(None)
                    continue
                m = _TOKEN.get(spec[-1], _TOKEN["s"]).match(text, f.pos)
                if m is None or not m.group(0):
                    values[i].append(None)
                    continue
                f.pos = m.end()
                tok = m.group(0).decode(errors="replace")
                try:
                    base = 16 if spec[-1] in "xX" else 0 if spec[-1] == "i" else 10
                    v = float(tok) if spec[-1] in "feEgG" else int(tok, base)
                except ValueError:
                    v = tok
                values[i].append(v)
                got += slots[i] is not None
            counts.append(got if got or f.pos < len(f.data) else -1)
        kept = [v for v, k in zip(values, slots) if k is not None and k < len(args)]
        for (spec, p, a), vals in zip(targets, kept):
            if not isinstance(p, Ptr) or not isinstance(p.obj, Obj) or p.off is UNK:
                continue
            t = _pointee(a)
            size = (ca.type_size(t.get_canonical()) if t is not None else None) or 4
            loc = Loc(p.obj, p.off, size)
            self.access(loc, write=True)
            if spec[-1] in "sc":
                continue
            if any(v is None or isinstance(v, str) for v in vals):
                self._mem_store(loc, t, UNK) if t is not None else None
                continue
            v = vals[0] if L == 1 else np.asarray(vals)
            if spec[-1] in "feEgG":
                self._mem_store(loc, t, Real(v if L == 1 else v.astype(float)))
            else:
                self._mem_store(loc, t, int(v) if L == 1 else v.astype(np.int64))
        return counts[0] if L == 1 else np.asarray(counts, dtype=np.int64)

    def _charge(self, ev, obj, off, n, write):
        """A bulk copy's references: 8 bytes at a time, and the bytes in the event."""
        if not self.charging or n <= 0:
            return
        w0 = float(np.ravel(self.w)[0])
        offs = np.arange(off, off + n, 8, dtype=np.int64)
        obj.add(offs, 8, np.full(len(offs), w0))
        self.total += w0 * len(offs)
        if self.uncertain_depth:
            self.uncertain += w0 * len(offs)
        if ev is not None and (obj.kind in TRACKED or obj.kind == "device-global"):
            bits = (W if write else R) << (2 if self.uncertain_depth else 0)
            if write and not self.uncertain_depth and self.kernel is None:
                bits |= WX                                             # the host's copy: in every schedule
            ev.mark_range(obj, off, n, bits)

    def _copy(self, ev, d, s, n):
        def known(p):
            return isinstance(p, Ptr) and isinstance(p.obj, Obj) and p.off is not UNK and np.ndim(p.off) == 0

        if n is UNK or np.ndim(n):
            for p, bits in ((s, RM), (d, WM)):
                if isinstance(p, Ptr) and isinstance(p.obj, Obj):
                    if ev is not None:
                        ev.mark_all(p.obj, bits)
                    if bits == WM:
                        p.obj.forget(0, max(p.obj.nbytes, 0))
                elif ev is not None:
                    ev.unknown |= bits
            return
        n = int(n)
        if known(s):
            self._charge(ev, s.obj, int(s.off), n, False)
        elif ev is not None:
            if isinstance(s, Ptr) and isinstance(s.obj, Obj):
                ev.mark_all(s.obj, RM)
            else:
                ev.unknown |= RM
        if known(d):
            self._charge(ev, d.obj, int(d.off), n, True)
            if self.charging:
                if known(s):
                    d.obj.copy_from(s.obj, int(s.off), int(d.off), n)
                else:
                    d.obj.forget(int(d.off), n)
        elif ev is not None:
            if isinstance(d, Ptr) and isinstance(d.obj, Obj):
                ev.mark_all(d.obj, WM)
            else:
                ev.unknown |= WM

    def _fill(self, ev, d, v, n):
        if not (isinstance(d, Ptr) and isinstance(d.obj, Obj) and d.off is not UNK and np.ndim(d.off) == 0) \
                or n is UNK or np.ndim(n):
            if ev is not None:
                if isinstance(d, Ptr) and isinstance(d.obj, Obj):
                    ev.mark_all(d.obj, WM)
                else:
                    ev.unknown |= WM
            return
        self._charge(ev, d.obj, int(d.off), int(n), True)
        if self.charging:
            if v is UNK or np.ndim(v):
                d.obj.forget(int(d.off), int(n))
            else:
                d.obj.fill(int(d.off), int(n), int(v))

    # ------------------------------------------------------------ launches

    def _dim3(self, expr):
        """(x, y, z) of a launch's grid or block argument, or UNK."""
        t = expr.type
        if not ca.is_float(t) and t.get_canonical().kind != ca.ci.TypeKind.RECORD:
            v = _as_int(self.eval(expr))
            return UNK if v is UNK or np.ndim(v) else (int(v), 1, 1)
        dest = self._new_storage(t)
        self._construct_into(expr, dest, t)
        if not isinstance(dest, Ptr) or not isinstance(dest.obj, Obj):
            return UNK
        vals = [dest.obj.load(np.asarray(dest.off) + 4 * i, 4) for i in range(3)]
        out = []
        for v in vals:
            v = np.ravel(v)
            if v.size == 0 or np.any(v == SENTINEL) or np.any(v != v[0]):
                return UNK
            out.append(int(v[0]))
        return tuple(out)

    def _launch(self, c):
        if self.L > 1:                                                 # launches in a batched host region:
            for p in range(self.L):                                    # one point at a time
                self.run_subset(np.array([p]), lambda: self._launch(c))
            return UNK
        ref = c.referenced
        fdef = ref.get_definition() if ref is not None else None
        cfg = _launch_config(c)
        cargs = [a for a in cfg.get_arguments()]
        grid = self._dim3(cargs[0]) if cargs else UNK
        block = self._dim3(cargs[1]) if len(cargs) > 1 else UNK
        shmem = 0
        if len(cargs) > 2 and ca.children(cargs[2]):
            v = _as_int(self.eval(cargs[2]))
            shmem = int(v) if v is not UNK and np.ndim(v) == 0 else 0
        if len(cargs) > 3 and ca.children(cargs[3]):
            self.eval(cargs[3])
        name = ref.spelling if ref is not None else c.spelling
        ev = self._new_event("kernel", name, c)
        ev.data_control = self.facts.under_data_control(c) or bool(self.uncertain_depth)
        params = [p for p in fdef.get_children() if p.kind == K.PARM_DECL] if fdef is not None else []
        stack = self.__dict__.setdefault("call_stack", [])
        stack.append(c)
        try:
            bound = [(how, p, _scalarize(v)) for how, p, v in self._bind_args(params, list(c.get_arguments()), None)]
            if fdef is None or _body(fdef) is None or grid is UNK or block is UNK:
                # not run: it may read and write whatever its arguments point to
                ev.note = "not run: " + ("no definition" if fdef is None or _body(fdef) is None else "unknown grid")
                for _, _, v in bound:
                    if isinstance(v, Ptr) and isinstance(v.obj, Obj):
                        ev.mark_all(v.obj, RM | WM)
                ev.unknown |= RM | WM
                return UNK
            ev.grid, ev.block = grid, block
            self._run_kernel(ev, fdef, bound, grid, block, shmem)
        finally:
            stack.pop()
            self._close_event()
        return UNK

    def _run_kernel(self, ev, fdef, bound, grid, block, shmem):
        bx, by, bz = block
        gx, gy, gz = grid
        bsz, nblocks = bx * by * bz, gx * gy * gz
        ev.threads = bsz * nblocks
        if bsz <= 0 or nblocks <= 0:
            return
        if self.record_blocks and ev.block_marks is None:
            ev.block_marks = []
        reorder = self.schedule != "lockstep" and (self.racy is None or fdef.spelling in self.racy)
        per = max(1, THREADS_PER_BATCH // bsz) if not reorder else 1   # a kernel no launch of which read and
                                                                       # wrote one unit gives the same in any order
        saved = (self.frames, self.w, self.loop_flags, self.kernel, self.shared, self.dyn_shared)
        w0 = float(np.ravel(self.w)[0])
        globals_frame = self.frames[0]

        def split(v, nx, ny):
            x = v % nx if nx > 1 else 0
            y = (v // nx) % ny if ny > 1 else 0
            z = v // (nx * ny) if np.size(v) and int(np.max(v)) >= nx * ny else 0
            return tuple(_SCALAR(a) if np.ndim(a) else int(a) for a in (x, y, z))

        try:
            starts = list(range(0, nblocks, per))
            if not reorder:
                pass
            elif self.schedule == "reverse":
                starts.reverse()
            elif self.schedule == "random":
                starts = [starts[i] for i in self.rng.permutation(len(starts))]
            # batches: (first block, blocks, first thread, end thread) of a block. In the schedules other than
            # lockstep, a block's warps also run one after another (in the same order as the blocks) when the
            # kernel has no barrier and no shared memory, which is then a legal order too
            warps = list(range(0, bsz, WARP_SIZE))
            if reorder and len(warps) > 1 and not self._block_sync(fdef):
                if self.schedule == "reverse":
                    warps.reverse()
                batches = []
                for b0 in starts:
                    order = [warps[i] for i in self.rng.permutation(len(warps))] if self.schedule == "random" \
                        else warps
                    batches += [(b0, 1, w, min(w + WARP_SIZE, bsz)) for w in order]
            else:
                batches = [(b0, min(per, nblocks - b0), 0, bsz) for b0 in starts]
            for b0, nb, t_lo, t_hi in batches:
                if nb > 1 or (t_lo, t_hi) == (0, bsz):
                    L = nb * bsz
                    tid = np.arange(L, dtype=np.int64)
                    local_block = tid // bsz
                    tin = tid % bsz
                else:
                    tin = np.arange(t_lo, t_hi, dtype=np.int64)
                    L = len(tin)
                    local_block = np.zeros(L, dtype=np.int64)
                g0 = _map_frames_with([globals_frame], lambda a: a[np.zeros(L, dtype=np.int64)])[0]
                before = {k: e[1] for k, e in g0.vars.items()}
                g0.vars[BLOCK_KEY] = [None, local_block if nb > 1 else 0]
                self._b0, self._nb = b0, nb
                for bname, vals in (("threadIdx", split(tin, bx, by)),
                                    ("blockIdx", split(b0 + local_block, gx, gy)),
                                    ("blockDim", (bx, by, bz)), ("gridDim", (gx, gy, gz))):
                    for f, v in zip("xyz", vals):
                        g0.vars[("__cuda", bname, f)] = [None, v]
                self.frames = [g0]
                self.w = np.full(L, w0)
                self.loop_flags = []
                self.kernel, self.shared, self.dyn_shared = ev, {}, shmem
                self._enter(fdef, bound, None, None, None)
                try:
                    self.exec(_body(fdef))
                finally:
                    self.frames.pop()
                    self.this_types.pop()
                for k, v in before.items():                           # host scalars a kernel assigned
                    if g0.vars[k][1] is not v and k in globals_frame.vars:
                        globals_frame.vars[k][1] = UNK
        finally:
            self.frames, self.w, self.loop_flags, self.kernel, self.shared, self.dyn_shared = saved

    def _block_sync(self, fdef):
        """Whether a kernel (or a function it calls) uses __syncthreads or __shared__ memory: its warps must then
        run together (the threads of a block meet at barriers and share memory)."""
        memo = self.__dict__.setdefault("_block_sync_memo", {})
        key = _key(fdef)
        if key in memo:
            return memo[key]
        memo[key] = False                                              # recursion
        found = False
        for n in fdef.walk_preorder():
            if n.kind == K.CALL_EXPR:
                if n.spelling.startswith("__syncthreads"):                 # __syncwarp: within a warp only
                    found = True
                    break
                ref = n.referenced
                d = ref.get_definition() if ref is not None else None
                if d is not None and d != fdef and not _system(d) and self._block_sync(d):
                    found = True
                    break
            elif n.kind == K.VAR_DECL and K.CUDASHARED_ATTR in _attrs(n):
                found = True
                break
        memo[key] = found
        return found

    # ------------------------------------------------------------ atomics

    def _atomic_cuda(self, name, c):
        args_c = list(c.get_arguments())
        p = self.eval(args_c[0])
        vals = [self.eval(a) for a in args_c[1:]]
        t = args_c[0].type.get_canonical().get_pointee()
        size = ca.type_size(t) or 4
        if not isinstance(p, Ptr):
            return UNK
        loc = Loc(p.obj, p.off, size)
        loc.cursor = c
        self.access(loc)
        self.access(loc, write=True)
        if not isinstance(p.obj, Obj) or p.off is UNK or not self.charging or isinstance(p.off, np.ndarray) and \
                p.off.size not in (1, self.L):
            if isinstance(p.obj, Obj) and p.off is not UNK and self.charging:
                p.obj.store(p.off, UNK, size)
            return UNK
        if ca.is_float(t):
            return self._atomic_float(name, loc, size, vals)
        if name in ("atomicAdd", "atomicSub"):
            d = _as_int(vals[0]) if vals else UNK
            if d is not UNK and name == "atomicSub":
                d = _SCALAR(-np.asarray(d)) if np.ndim(d) else -d
            old, _ = self._rmw_add(loc, t, d)
            return old
        return self._atomic_serial(name, loc, size, [_as_int(v) for v in vals])

    def _groups(self, off):
        """Thread order within each address: (order, starts, group, rank, unique offsets)."""
        offs = np.broadcast_to(np.asarray(off, dtype=np.int64), (self.L,))
        order = np.argsort(offs, kind="stable")
        so = offs[order]
        first = np.concatenate([[True], so[1:] != so[:-1]])
        starts = np.nonzero(first)[0]
        group = np.cumsum(first) - 1
        rank = np.arange(self.L) - starts[group]
        return order, starts, group, rank, so[starts]

    def _atomic_serial(self, name, loc, size, vals):
        if any(v is UNK for v in vals) or not vals:
            loc.obj.store(loc.off, UNK, size)
            return UNK
        L = self.L
        order, starts, group, rank, uniq = self._groups(loc.off)
        cur = np.asarray(loc.obj.load(uniq, size), dtype=np.int64).copy()
        if np.any(cur == SENTINEL):
            loc.obj.store(uniq, UNK, size)
            return UNK
        a = np.broadcast_to(np.asarray(vals[0], dtype=np.int64), (L,))[order]
        b = np.broadcast_to(np.asarray(vals[1], dtype=np.int64), (L,))[order] if len(vals) > 1 else None
        old_sorted = np.empty(L, dtype=np.int64)
        if name == "atomicExch":
            old_sorted[1:] = a[:-1]
            old_sorted[starts] = cur
            last = np.concatenate([starts[1:] - 1, [L - 1]])
            cur = a[last]
        elif name in ("atomicMin", "atomicMax"):
            incl = _seg_accumulate(name == "atomicMax", group, a)
            op = np.maximum if name == "atomicMax" else np.minimum
            prev = np.concatenate([[0], incl[:-1]])
            old_sorted = np.where(rank == 0, cur[group], op(cur[group], prev))
            last = np.concatenate([starts[1:] - 1, [L - 1]])
            cur = op(cur, incl[last])
        else:
            ops = {"atomicCAS": lambda o, x, y: np.where(o == x, y, o), "atomicAnd": lambda o, x, y: o & x,
                   "atomicOr": lambda o, x, y: o | x, "atomicXor": lambda o, x, y: o ^ x,
                   "atomicInc": lambda o, x, y: np.where(o >= x, 0, o + 1),
                   "atomicDec": lambda o, x, y: np.where((o == 0) | (o > x), x, o - 1)}
            op = ops[name]
            by_rank = np.argsort(rank, kind="stable")
            pos = 0
            for cnt in np.bincount(rank):
                sel = by_rank[pos: pos + cnt]
                pos += cnt
                g = group[sel]
                o = cur[g]
                old_sorted[sel] = o
                cur[g] = op(o, a[sel], b[sel] if b is not None else None)
        loc.obj.store(uniq, cur, size)
        old = np.empty(L, dtype=np.int64)
        old[order] = old_sorted
        return int(old[0]) if L == 1 else old

    def _atomic_float(self, name, loc, size, vals):
        r = _real(vals[0]) if vals and vals[0] is not UNK else UNK
        order, starts, group, rank, uniq = self._groups(loc.off)
        if r is UNK or name not in ("atomicAdd", "atomicExch"):
            loc.obj.fstore(uniq, np.nan, size)
            return UNK
        L = self.L
        v = np.broadcast_to(np.asarray(r.v, dtype=float), (L,))[order]
        base = loc.obj.fload(uniq, size)
        if name == "atomicExch":
            old_sorted = np.empty(L)
            old_sorted[1:] = v[:-1]
            old_sorted[starts] = base
            last = np.concatenate([starts[1:] - 1, [L - 1]])
            loc.obj.fstore(uniq, v[last], size)
        else:
            cs = np.cumsum(v)
            excl = cs - v
            excl = excl - excl[starts][group]
            old_sorted = base[group] + excl
            ends = np.concatenate([starts[1:], [L]])
            loc.obj.fstore(uniq, base + (cs[ends - 1] - (cs[starts] - v[starts])), size)
        if np.any(np.isnan(old_sorted)):
            return UNK
        old = np.empty(L)
        old[order] = old_sorted
        return Real(float(old[0]) if L == 1 else old)


def _seg_accumulate(maximum, group, x):
    """Inclusive running max (or min) of x within each group (x sorted by group)."""
    u, inv = np.unique(x, return_inverse=True)
    m = len(u)
    r = inv if maximum else (m - 1 - inv)
    key = group.astype(np.int64) * m + r
    acc = np.maximum.accumulate(key) - group.astype(np.int64) * m
    return u[acc] if maximum else u[m - 1 - acc]


_TOKEN = {"d": re.compile(rb"[-+]?\d+"), "i": re.compile(rb"[-+]?(0[xX][0-9a-fA-F]+|\d+)"),
          "u": re.compile(rb"[-+]?\d+"), "x": re.compile(rb"[-+]?(0[xX])?[0-9a-fA-F]+"),
          "X": re.compile(rb"[-+]?(0[xX])?[0-9a-fA-F]+"), "s": re.compile(rb"\S+"), "c": re.compile(rb"."),
          **{k: re.compile(rb"[-+]?(\d+\.?\d*([eE][-+]?\d+)?|\.\d+([eE][-+]?\d+)?|inf|nan)") for k in "feEgG"}}


def _pointee(arg):
    """The element type a pointer argument points to, before its conversion to void *."""
    c = arg
    while True:
        t = c.type.get_canonical()
        if t.kind == ca.ci.TypeKind.POINTER and t.get_pointee().get_canonical().kind != ca.ci.TypeKind.VOID:
            return t.get_pointee()
        kids = [k for k in ca.children(c) if k.kind.is_expression()]
        if c.kind == K.UNARY_OPERATOR and ca.unary_op(c) == "&" and kids:
            return kids[0].type
        if not kids or c.kind not in (K.UNEXPOSED_EXPR, K.PAREN_EXPR, K.CSTYLE_CAST_EXPR, K.CXX_STATIC_CAST_EXPR,
                                      K.CXX_REINTERPRET_CAST_EXPR):
            return None
        c = kids[-1]


def _system_stub(decl):
    """Declared by cuda_stub.h (the runtime), not by the program."""
    f = decl.location.file
    return f is not None and str(f).endswith("cuda_stub.h")


def _walk_nodes(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.get_children())


def run(tu, argv=None, seed=0, stubs=None, opt=3, unit=4, host_events=True, keep_access=False, schedule="lockstep",
        keep_touch=False, racy=None):
    """Run a CUDA program from main; returns the interpreter, with its events (Event) and their dependences
    (interp.tracker, a kdeps.Tracker at `unit` bytes)."""
    from . import interp
    from .kdeps import Tracker

    made = []

    def make(tu_, seed_):
        it = CudaInterpreter(tu_, seed_, stubs, opt, host_events)
        it.schedule = schedule
        it.racy = racy
        it.tracker = Tracker(unit)
        if keep_access:
            it.tracker.keep_access()
        if keep_touch:
            it.tracker.keep_touch()
        made.append(it)
        return it

    result = interp.run(tu, footprint="starts", seed=seed, make=make, argv=argv)
    it = made[0]
    it.finish()
    it.result = result
    return it


def analyze(source, defines=(), includes=(), argv=None, seed=0, unit=4, host_events=True, keep_access=False,
            schedule="lockstep", keep_touch=False, racy=None):
    """Parse a CUDA source and run it: (interpreter, parse errors, seconds)."""
    import time

    start = time.time()
    tu = ca.parse(source, defines, includes, cuda=True)
    errors = [str(d) for d in tu.diagnostics if d.severity >= 3]
    it = run(tu, argv=argv, seed=seed, unit=unit, host_events=host_events, keep_access=keep_access,
             schedule=schedule, keep_touch=keep_touch, racy=racy)
    return it, errors, time.time() - start
