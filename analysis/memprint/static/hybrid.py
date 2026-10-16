"""The hybrid interpreter: a CUDA program's host code runs natively, its kernels in MemPrint's interpreter.

    python -m memprint.static.hybrid SOURCE[,SOURCE...] OUT_DIR --so PROGRAM.so [--argv ARG ...]
        [-D X] [-I DIR] [--unit 4] [--access] [--schedule lockstep|blocks|reverse|random]

The program is compiled by nvcc as a shared library (main keeps its name) which is linked against
gpu/hybrid/libkdeps_cudart.so instead of the CUDA runtime (gpu/hybrid/build.sh). This module loads both, sets
the stand-in runtime's callbacks and calls the program's main, so the host code runs at native speed with every
library it uses. Device memory is ordinary host memory. At each kernel launch the stand-in runtime calls
back here, and the interpreter (interp_cuda.CudaInterpreter) runs that one kernel, all threads together, on the
real contents of device memory: its loads read the program's bytes and its stores write them, so the host
code sees the kernel's results and continues from them. Copies and fills are done natively and recorded.
Every launch, copy and fill becomes an Event, and kdeps.Tracker folds them into dependences; the output is
that of `python -m memprint.static.kdeps --no-host` (kdeps.write), scored the same way (kdeps_score).

Differences from the whole-program interpreter: the program's input must exist; every value is concrete, so
dependences are certain unless a kernel computes a value the interpreter does not model (it is then not
written to device memory, and counted in the summary as an unknown store); device allocations are named
alloc0, alloc1, ... in allocation order; a launch is under data control if any launch of its kernel in the
source is (the host code is not run in the interpreter, so a launch is not matched to its site);
__device__ and __constant__ variables live in the program's own memory (their host shadow).
"""

import argparse
import ctypes
import os
import pickle
import subprocess
import sys
import time
import traceback

import numpy as np

from . import clangast as ca
from . import interp_cuda as ic
from .clangast import K
from .interp import Loc, Obj, Ptr, Real, SENTINEL, UNK, _key
from .interp_cpp import _referee
from .kdeps import R, W, Tracker, merge_schedules, write

HERE = os.path.dirname(os.path.abspath(__file__))
SHIM = os.path.join(HERE, "..", "..", "..", "gpu", "hybrid", "libkdeps_cudart.so")
INT_TYPES = {1: np.uint8, 2: np.int16, 4: np.int32, 8: np.int64}
UINT_TYPES = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.int64}  # 64-bit: int64 with the same bits, as the
                                                                     # interpreter keeps unsigned 64-bit values
FLOAT_TYPES = {4: np.float32, 8: np.float64}
UNSIGNED = {ca.ci.TypeKind.BOOL, ca.ci.TypeKind.UCHAR, ca.ci.TypeKind.CHAR_U, ca.ci.TypeKind.USHORT,
            ca.ci.TypeKind.UINT, ca.ci.TypeKind.ULONG, ca.ci.TypeKind.ULONGLONG}
COPY_KINDS = {1: "H2D", 2: "D2H", 3: "D2D"}


class RawObj(Obj):
    """A device allocation backed by the program's own memory: loads and stores read and write its bytes.
    Integer loads read the bytes as signed unless RawObj.unsigned is set (by HybridInterpreter._typed, for a
    load of an unsigned type): a uint32_t 0xffffffff is 4294967295, as the whole-program interpreter has it."""

    unsigned = False

    def __init__(self, name, kind, addr, nbytes):
        super().__init__(name, kind, nbytes)
        self.addr = addr
        self.buf = np.ctypeslib.as_array((ctypes.c_uint8 * max(nbytes, 1)).from_address(addr))
        self.unknown_stores = 0

    def _offsets(self, off, size):
        offs = np.asarray(off, dtype=np.int64)
        ok = (offs >= 0) & (offs + size <= self.nbytes)
        return offs, ok

    def _get(self, offs, size, dtype):
        if size == 1:
            return self.buf[offs].view(dtype) if dtype != np.uint8 else self.buf[offs]
        if np.all(offs % size == 0):
            return self.buf[: self.nbytes // size * size].view(dtype)[offs // size]
        idx = offs.reshape(-1, 1) + np.arange(size)
        return np.ascontiguousarray(self.buf[idx]).view(dtype).reshape(offs.shape)

    def _put(self, offs, vals, size, dtype):
        vals = np.asarray(vals).astype(dtype)
        if size == 1:
            self.buf[offs] = vals.view(np.uint8)
        elif np.all(offs % size == 0):
            self.buf[: self.nbytes // size * size].view(dtype)[offs // size] = vals
        else:
            for o, v in zip(offs.ravel(), np.broadcast_to(vals, offs.shape).ravel()):
                self.buf[o: o + size] = np.frombuffer(np.asarray(v, dtype=dtype).tobytes(), dtype=np.uint8)

    def load(self, off, size=4):
        dtype = (UINT_TYPES if RawObj.unsigned else INT_TYPES).get(size)
        offs, ok = self._offsets(off, size)
        out = np.full(offs.shape, SENTINEL, dtype=np.int64)
        if dtype is not None and np.any(ok):
            out[ok] = self._get(offs[ok], size, dtype).astype(np.int64)
        return out if out.ndim else out[()]

    def store(self, off, value, size=4):
        dtype = INT_TYPES.get(size)
        offs, ok = self._offsets(off, size)
        if value is UNK or dtype is None:
            self.unknown_stores += int(np.count_nonzero(ok))
            return
        vals = np.asarray(value, dtype=np.int64)
        if offs.ndim == 0 and vals.ndim:                               # one address, a value per thread: the last
            vals = vals.ravel()[-1]
        vals = np.broadcast_to(vals, offs.shape)
        keep = ok & (vals != SENTINEL)
        self.unknown_stores += int(np.count_nonzero(ok & ~keep))
        if np.any(keep):
            if offs.ndim == 0:
                self._put(offs, vals, size, dtype)
            else:
                self._put(offs[keep], vals[keep], size, dtype)

    def fload(self, off, size=8):
        dtype = FLOAT_TYPES.get(size)
        offs, ok = self._offsets(off, size)
        out = np.full(offs.shape, np.nan)
        if dtype is not None and np.any(ok):
            out[ok] = self._get(offs[ok], size, dtype).astype(float)
        return out if out.ndim else out[()]

    def fstore(self, off, value, size=8):
        dtype = FLOAT_TYPES.get(size)
        offs, ok = self._offsets(off, size)
        vals = np.asarray(value, dtype=float)
        if offs.ndim == 0 and vals.ndim:
            vals = vals.ravel()[-1]
        vals = np.broadcast_to(vals, offs.shape)
        keep = ok & ~np.isnan(vals)
        self.unknown_stores += int(np.count_nonzero(ok & ~keep))
        if dtype is not None and np.any(keep):
            if offs.ndim == 0:
                self._put(offs, vals, size, dtype)
            else:
                self._put(offs[keep], vals[keep], size, dtype)

    def point_values(self, unit, kind, n):
        return None

    def _forget_points(self, units):
        pass

    def _keep_points(self, unit, kind, values):
        pass

    def forget(self, off, n):
        pass


class HybridInterpreter(ic.CudaInterpreter):
    """Runs single kernels on device memory owned by a natively running host program."""

    def __init__(self, tu, seed=0, unit=4, keep_access=False, schedule="lockstep"):
        self.seed = seed
        self.raw_vars = {}  # __device__ / __constant__ variable name -> RawObj over the program's own bytes
        super().__init__(tu, seed, None, 3, host_events=False)
        self.schedule = schedule
        self.tracker = Tracker(unit)
        if keep_access:
            self.tracker.keep_access()
        self.live = []  # (base, end, RawObj), sorted by base
        self.kernel_defs = self._kernels(tu)
        self.launch_control = self._launch_control()
        self.shim = None  # the stand-in runtime (ctypes), for the call site of the last copy or fill
        self.host_regions = []  # [base, end, Obj]: host memory that copies read or wrote
        self._site_cache = {}
        self.by_device_name = {}
        self.failures = []
        self.launches = 0
        self.unknown_sites = {}  # source location -> stores of values the interpreter could not compute

    # -------------------------------------------------------------- kernels in the source

    @staticmethod
    def _kernels(tu):
        out = {}
        stack = [tu.cursor]
        while stack:
            c = stack.pop()
            if c.kind in (K.FUNCTION_DECL, K.FUNCTION_TEMPLATE) and c.is_definition():
                if any(k.kind == K.CUDAGLOBAL_ATTR for k in c.get_children()):
                    out.setdefault(c.spelling, []).append(c)
                continue
            if c.kind in (K.TRANSLATION_UNIT, K.NAMESPACE, K.UNEXPOSED_DECL, K.LINKAGE_SPEC):
                stack.extend(c.get_children())
        return out

    def _launch_control(self):
        """Per kernel name: whether a launch of it in the source is under data control (a host loop that runs
        until the device says so, DataFacts). The host code runs natively, so a launch is not matched to its
        site; a kernel launched from several sites is marked if any of them is."""
        out = {}
        for f in self.functions.values():
            for c in f.walk_preorder():
                if c.kind == K.CALL_EXPR and ic._launch_config(c) is not None and c.referenced is not None:
                    name = c.referenced.spelling
                    out[name] = out.get(name, False) or bool(self.facts.under_data_control(c))
        return out

    def _site_control(self, names):
        """Whether the copy or fill just reported was issued under data control: its call site (the stand-in
        runtime keeps the return address; the program is built with line information) mapped to the call in the
        source, and DataFacts asked about that call."""
        if self.shim is None:
            return False
        buf = ctypes.create_string_buffer(1024)
        off = self.shim.kdeps_last_site(buf, len(buf))
        if not off:
            return False
        key = (buf.value, off)
        if key not in self._site_cache:
            out = subprocess.run(["addr2line", "-e", buf.value.decode(), hex(off)], capture_output=True,
                                 text=True).stdout.strip()
            file, _, line = out.rpartition(":")
            line = int(line.split()[0]) if line.split() and line.split()[0].isdigit() else 0
            base = os.path.basename(file)
            found = False
            for f in self.functions.values():
                for c in f.walk_preorder():
                    if c.kind == K.CALL_EXPR and c.spelling in names and c.location.line == line and \
                            os.path.basename(str(c.location.file)) == base:
                        found = found or bool(self.facts.under_data_control(c))
            self._site_cache[key] = found
        return self._site_cache[key]

    def _kernel(self, device_name):
        fdef = self.by_device_name.get(device_name)
        if fdef is None:
            text = subprocess.run(["c++filt", device_name], capture_output=True, text=True).stdout.strip()
            base = text.split("(")[0].split("<")[0].split("::")[-1]
            nparams = 0 if text.endswith("()") else text.count(",") + 1 if "(" in text else -1
            cands = self.kernel_defs.get(base, [])
            match = [c for c in cands if len([p for p in c.get_children() if p.kind == K.PARM_DECL]) == nparams]
            fdef = (match or cands or [None])[0]
            self.by_device_name[device_name] = fdef
        return fdef

    # -------------------------------------------------------------- __device__ and __constant__ variables

    def _declare_global(self, c):
        """As the CUDA interpreter, but a device variable the program registered lives in the program's bytes."""
        super()._declare_global(c)
        raw = self.raw_vars.get(c.spelling)
        if raw is not None and _key(c) in self.frames[0].vars:
            self._bind_var(c, raw)

    def _bind_var(self, c, raw):
        entry = self.frames[0].vars.get(_key(c))
        if entry is None or not isinstance(entry[0], Loc) or getattr(entry[0].obj, "kind", None) != "device-global":
            return
        if ca.is_array(c.type):
            self.frames[0].vars[_key(c)] = [Loc(raw, 0, raw.nbytes, True), Ptr(raw, 0)]
        else:
            slot = Loc(raw, 0, raw.nbytes)
            self.frames[0].vars[_key(c)] = [slot, ic.InMemory(slot)]

    def on_var(self, host, nbytes, device_name, constant):
        """The program registers a __device__ or __constant__ variable (while it loads): its host shadow holds
        its bytes, and copies to and from the symbol go there."""
        text = subprocess.run(["c++filt", device_name], capture_output=True, text=True).stdout.strip()
        name = text.split("::")[-1]
        raw = RawObj(name, "device-global", host, nbytes)
        raw.site = {}
        self.raw_vars[name] = raw
        self.objects.append(raw)
        self.live.append((host, host + nbytes, raw))
        self.live.sort(key=lambda e: e[0])
        self.tracker.register(raw)
        for c in self._globals_named(name):
            self._bind_var(c, raw)

    def _globals_named(self, name):
        stack = [self.tu.cursor] if getattr(self, "tu", None) is not None else []
        while stack:
            c = stack.pop()
            if c.kind == K.VAR_DECL and c.spelling == name:
                yield c
            elif c.kind in (K.TRANSLATION_UNIT, K.NAMESPACE, K.UNEXPOSED_DECL, K.LINKAGE_SPEC):
                stack.extend(c.get_children())

    # -------------------------------------------------------------- device memory

    def _find_alloc(self, addrs):
        """The live allocation each address falls in: (indices into self.live, valid mask)."""
        bases = np.array([b for b, _, _ in self.live], dtype=np.int64)
        ends = np.array([e for _, e, _ in self.live], dtype=np.int64)
        a = np.asarray(addrs, dtype=np.int64)
        if not len(bases):
            return np.zeros(a.shape, dtype=np.int64), np.zeros(a.shape, dtype=bool)
        idx = np.searchsorted(bases, a, side="right") - 1
        ok = (idx >= 0) & (a < ends[np.maximum(idx, 0)])
        return np.maximum(idx, 0), ok

    def _ptr_at(self, addrs):
        """A pointer for raw address(es): Ptr(object, offset), or None if they name different objects."""
        a = np.asarray(addrs, dtype=np.int64)
        if np.all(a == 0):
            return Ptr(self.null, a * 0 if a.ndim else 0)
        idx, ok = self._find_alloc(a)
        if not np.all(ok):
            return Ptr(UNK, UNK)
        first = int(np.ravel(idx)[0])
        if np.any(idx != first):
            return None
        base, _, obj = self.live[first]
        off = a - base
        return Ptr(obj, off if off.ndim else int(off))

    @staticmethod
    def _typed(type_, fn):
        """Run fn with integer loads from the program's memory read with type_'s signedness."""
        t = type_.get_canonical() if type_ is not None else None
        saved = RawObj.unsigned
        RawObj.unsigned = t is not None and t.kind in UNSIGNED
        try:
            return fn()
        finally:
            RawObj.unsigned = saved

    def _mem_load(self, loc, type_):
        return self._typed(_referee(type_), lambda: super(HybridInterpreter, self)._mem_load(loc, type_))

    def _rmw_add(self, loc, type_, delta):
        return self._typed(type_, lambda: super(HybridInterpreter, self)._rmw_add(loc, type_, delta))

    def _atomic_cuda(self, name, c):
        args_c = list(c.get_arguments())
        t = args_c[0].type.get_canonical().get_pointee() if args_c else None
        return self._typed(t, lambda: super(HybridInterpreter, self)._atomic_cuda(name, c))

    def _ptr_load(self, loc):
        if not isinstance(loc.obj, RawObj):
            return super()._ptr_load(loc)
        addrs = loc.obj.load(loc.off, 8)
        if np.any(np.asarray(addrs) == SENTINEL):
            return Ptr(UNK, UNK)
        p = self._ptr_at(addrs)
        if p is not None:
            return p
        idx, _ = self._find_alloc(addrs)                                   # threads point into different objects
        if self.kernel is not None and self._stmt_marks and np.shape(addrs) == (self.L,):
            effects, w = self._stmt_marks[-1]
            keys = np.unique(idx)
            if effects == self.effects and w is self.w and len(keys) <= ic.MAX_TARGETS:
                raise ic._Diverge([np.flatnonzero(idx == k) for k in keys])
        return Ptr(UNK, UNK)

    def _ptr_store(self, loc, value):
        if not isinstance(loc.obj, RawObj):
            return super()._ptr_store(loc, value)
        if isinstance(value, Ptr) and value.obj is self.null:
            loc.obj.store(loc.off, 0, 8)
        elif isinstance(value, Ptr) and isinstance(value.obj, RawObj) and value.off is not UNK:
            loc.obj.store(loc.off, value.obj.addr + np.asarray(value.off, dtype=np.int64), 8)
        else:
            loc.obj.store(loc.off, UNK, 8)

    def _bind(self, p, addr):
        """A kernel parameter bound as the interpreter's _bind_args would: a value, or for a struct passed by
        value an object of its own, filled from the argument's bytes (pointer fields become pointers)."""
        t = p.type.get_canonical()
        if t.kind == ca.ci.TypeKind.RECORD:
            obj = Obj(f"{p.spelling}#arg", "stack-array", ca.type_size(t) or 8)
            self.objects.append(obj)
            self._fill(obj, 0, t, addr)
            return ("object", p, Ptr(obj, 0))
        return ("value", p, self._arg(p.type, addr))

    def _fill(self, obj, off, type_, addr):
        t = type_.get_canonical()
        if t.kind == ca.ci.TypeKind.RECORD:
            for f in t.get_fields():
                fo = t.get_offset(f.spelling)
                if fo >= 0:
                    self._fill(obj, off + fo // 8, f.type, addr + fo // 8)
        elif t.kind == ca.ci.TypeKind.CONSTANTARRAY:
            esize = ca.type_size(t.element_type.get_canonical()) or 1
            for i in range(t.element_count):
                self._fill(obj, off + i * esize, t.element_type, addr + i * esize)
        else:
            v = self._arg(t, addr)
            if isinstance(v, Ptr):
                self._ptr_store(Loc(obj, off, 8), v)
            elif isinstance(v, Real):
                obj.fstore(off, v.v, ca.type_size(t) or 8)
            elif v is not UNK:
                obj.store(off, v, ca.type_size(t) or 4)

    def _store(self, c, value):
        """Note where a kernel stores a value the interpreter could not compute (it is not written)."""
        if self.kernel is not None and (value is UNK or (isinstance(value, Ptr) and value.obj is UNK)):
            loc = f"{str(c.location.file).rsplit('/', 1)[-1]}:{c.location.line}"
            self.unknown_sites[loc] = self.unknown_sites.get(loc, 0) + 1
        return super()._store(c, value)

    def _arg(self, type_, addr):
        """A kernel argument's value, read from the launch's argument array."""
        t = type_.get_canonical()
        size = ca.type_size(t) or 8
        if ca.is_pointer(t):
            p = self._ptr_at(ctypes.c_uint64.from_address(addr).value)
            return p if p is not None else Ptr(UNK, UNK)
        if ca.is_float(t):
            return Real(float((ctypes.c_float if size == 4 else ctypes.c_double).from_address(addr).value))
        if size in INT_TYPES and t.kind not in (ca.ci.TypeKind.RECORD, ca.ci.TypeKind.ELABORATED):
            raw = ctypes.string_at(addr, size)
            return int.from_bytes(raw, "little", signed=t.kind not in UNSIGNED)
        return UNK

    # -------------------------------------------------------------- callbacks from the stand-in runtime

    def on_alloc(self, ptr, nbytes, kind):
        index = self.__dict__.setdefault("device_allocs", 0)
        self.device_allocs += 1
        obj = RawObj(f"alloc{index}", "managed" if kind else "device", ptr, nbytes)
        obj.alloc_index, obj.site, obj.alloc_kind = index, {}, "cudaMalloc"
        self.objects.append(obj)
        self.live.append((ptr, ptr + nbytes, obj))
        self.live.sort(key=lambda e: e[0])
        self.tracker.register(obj)

    def on_free(self, ptr):
        self.live = [e for e in self.live if e[0] != ptr]

    def _range(self, ptr, n):
        idx, ok = self._find_alloc(np.array([ptr]))
        if not ok[0]:
            return None, 0
        base, _, obj = self.live[int(idx[0])]
        return obj, ptr - base

    @staticmethod
    def _stack():
        """The main thread's stack, where the program's main and its locals live (/proc/self/maps)."""
        with open("/proc/self/maps") as f:
            for line in f:
                if line.rstrip().endswith("[stack]"):
                    lo, hi = line.split()[0].split("-")
                    return int(lo, 16), int(hi, 16)
        return 0, 0

    def _host_range(self, ptr, n):
        """The host memory a copy reads or writes, as an object of its own (the program's allocations are not
        seen, so a region is the union of the copies that overlap it; a copy starting before a region's base
        starts a new region). Locals on the stack are left out, as the whole-program interpreter keeps scalar
        locals out of memory: a flag copied back every round would otherwise chain all the rounds."""
        if not hasattr(self, "_stack_range"):
            self._stack_range = self._stack()
        if self._stack_range[0] <= ptr < self._stack_range[1]:
            return None, 0
        for r in self.host_regions:
            if r[0] <= ptr < r[1] or (ptr == r[1] and n == 0):
                r[1] = max(r[1], ptr + n)
                r[2].nbytes = max(r[2].nbytes, r[1] - r[0])
                return r[2], ptr - r[0]
        obj = Obj(f"host{len(self.host_regions)}", "heap", n)       # by order: addresses differ between runs
        obj.site = {}
        self.objects.append(obj)
        self.host_regions.append([ptr, ptr + n, obj])
        return obj, 0

    def on_copy(self, dst, src, n, kind):
        ev = self._new_event("copy", f"cudaMemcpy {COPY_KINDS.get(kind, '?')}", None)
        ev.data_control = self._site_control(ic.COPIES | ic.SYMBOL_COPIES)
        for ptr, bits in ((src, R), (dst, W)):
            obj, off = self._range(ptr, n)
            if obj is None and ptr:
                obj, off = self._host_range(ptr, n)
            if obj is not None:
                ev.mark_range(obj, off, n, bits)
        self._close_event()

    def on_set(self, ptr, value, n):
        ev = self._new_event("set", "cudaMemset", None)
        ev.data_control = self._site_control(ic.SETS)
        obj, off = self._range(ptr, n)
        if obj is not None:
            ev.mark_range(obj, off, n, W)
        self._close_event()

    def on_launch(self, name, grid, block, shmem, args):
        self.launches += 1
        fdef = self._kernel(name)
        if fdef is None:
            self.failures.append(f"no definition for kernel {name}")
            ev = self._new_event("kernel", name, None)
            ev.note, ev.unknown = "not run: no definition", ic.RM | ic.WM
            self._close_event()
            return
        params = [p for p in fdef.get_children() if p.kind == K.PARM_DECL]
        argv = (ctypes.c_void_p * max(len(params), 1)).from_address(args) if params else []
        bound = [self._bind(p, argv[i]) for i, p in enumerate(params)]
        ev = self._new_event("kernel", fdef.spelling, fdef)
        ev.grid, ev.block = grid, block
        ev.data_control = self.launch_control.get(fdef.spelling, False)
        try:
            self._run_kernel(ev, fdef, bound, grid, block, shmem)
        finally:
            self._close_event()


def build(sources, out, nvcc_flags=()):
    """Compile a CUDA program for the hybrid interpreter (as gpu/hybrid/build.sh)."""
    shim_dir = os.path.dirname(os.path.abspath(SHIM))
    probe = os.path.join(shim_dir, "..", "probe.cpp")
    cmd = ["nvcc", "-O2", "-g", "-cudart", "none", "-shared", "-Xcompiler", "-fPIC",
           *nvcc_flags, "-o", out, *sources, probe, f"-L{shim_dir}", "-lkdeps_cudart", f"-Xlinker=-rpath={shim_dir}"]
    subprocess.run(cmd, check=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", help="a .cu file, or several separated by commas")
    ap.add_argument("out_dir")
    ap.add_argument("--so", help="the program built for the hybrid interpreter (gpu/hybrid/build.sh); "
                                 "built from SOURCE with nvcc if omitted")
    ap.add_argument("--name", default=None)
    ap.add_argument("--argv", nargs=argparse.REMAINDER, default=None,
                    help="the command line, argv[0] first (last option: everything after it is passed on)")
    ap.add_argument("-D", dest="defines", action="append", default=[])
    ap.add_argument("-I", dest="includes", action="append", default=[])
    ap.add_argument("--unit", type=int, default=4)
    ap.add_argument("--access", action="store_true")
    ap.add_argument("--schedule", default="lockstep", choices=ic.SCHEDULES)
    ap.add_argument("--schedules", default=None,
                    help="several thread schedules, comma-separated: the program runs once per schedule (each in a "
                         "process of its own) and the runs are merged as kdeps --schedules does")
    ap.add_argument("--seed", type=int, default=0, help="the random schedule's block order")
    ap.add_argument("--dump", help=argparse.SUPPRESS)  # one schedule's dependences, for the merging process
    ap.add_argument("--racy", default=None, help=argparse.SUPPRESS)  # the kernels to reorder (lockstep's racy_seen)
    args = ap.parse_args(argv)
    if args.schedules and len(args.schedules.split(",")) > 1:
        return _several(args, argv)
    if args.schedules:
        args.schedule = args.schedules

    sources = args.source.split(",")
    name = args.name or os.path.splitext(os.path.basename(sources[0]))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    start = time.time()
    so = args.so or build(sources, os.path.join(args.out_dir, f"{name}.so"),
                          [f"-D{d}" for d in args.defines] + [f"-I{i}" for i in args.includes])
    source = sources[0] if len(sources) == 1 else _unity(sources, os.path.join(args.out_dir, "unity.cu"))
    tu = ca.parse(source, args.defines, args.includes, cuda=True,
                  extra=["-include", "stdlib.h", "-include", "math.h", "-include", "string.h"])
    errors = [str(d) for d in tu.diagnostics if d.severity >= 3]
    for e in errors[:20]:
        print("parse error:", e, file=sys.stderr)
    it = HybridInterpreter(tu, seed=args.seed, unit=args.unit, keep_access=args.access, schedule=args.schedule)
    if args.dump:
        it.tracker.keep_touch()                                        # merged with the other schedules' runs
    if args.racy is not None:
        it.racy = set(x for x in args.racy.split(",") if x)
    done = []

    def guarded(fn):
        def call(*a):
            try:
                fn(*a)
            except Exception:                                          # never unwind into the C program
                it.failures.append(traceback.format_exc(limit=4))
                traceback.print_exc()
        return call

    def finish():
        if done:
            return
        done.append(1)
        it._close_event()
        if args.dump:
            t = it.tracker
            with open(args.dump, "wb") as f:
                pickle.dump({"unit": t.unit, "events": t.events, "edges": t.edges, "access": t.access,
                             "touch": t.touch, "objects": list(t.objects.values()),
                             "failures": len(it.failures), "racy": sorted(it.racy_seen), "oob": it.oob}, f)
        ev_path, edge_path, rows = write(args.out_dir, name, it.tracker)
        unknown = sum(o.unknown_stores for o in it.objects if isinstance(o, RawObj))
        print(f"{name}: {len(it.tracker.events)} events, {len(rows)} dependences "
              f"({sum(r['needed'] for r in rows)} needed, {sum(r['certainty'] == 'may' for r in rows)} may), "
              f"{it.launches} launches, {unknown} unknown stores, {len(it.failures)} failures, "
              f"{time.time() - start:.1f} s", flush=True)
        print(f"  {ev_path}\n  {edge_path}", flush=True)
        if it.oob:
            print(f"  {len(it.oob)} places reference past the end of a device buffer (possible references to every "
                  f"buffer): " + ", ".join(f"{k} x{v}" for k, v in sorted(it.oob.items(), key=lambda kv: -kv[1])[:5]),
                  flush=True)
        if it.unknown_sites:
            top = sorted(it.unknown_sites.items(), key=lambda kv: -kv[1])[:5]
            print("  unknown values stored at: " + ", ".join(f"{k} ({v})" for k, v in top), flush=True)

    shim = ctypes.CDLL(os.path.abspath(SHIM), mode=ctypes.RTLD_GLOBAL)
    LAUNCH = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                              ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p)
    ALLOC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    FREE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    COPY = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    SET = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t)
    FIN = ctypes.CFUNCTYPE(None)
    VAR = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_int)
    callbacks = (
        LAUNCH(guarded(lambda n, gx, gy, gz, bx, by, bz, s, a: it.on_launch(n.decode(), (gx, gy, gz), (bx, by, bz), s, a))),
        ALLOC(guarded(it.on_alloc)), FREE(guarded(it.on_free)), COPY(guarded(it.on_copy)), SET(guarded(it.on_set)),
        FIN(guarded(finish)))
    shim.kdeps_set_callbacks(*callbacks)
    shim.kdeps_last_site.restype = ctypes.c_size_t
    shim.kdeps_last_site.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    it.shim = shim
    var_callback = VAR(guarded(lambda h, n, name, const: it.on_var(h, n, name.decode(), const)))
    shim.kdeps_set_var_callback(var_callback)                         # before the program loads and registers
    prog = ctypes.CDLL(os.path.abspath(so), mode=ctypes.RTLD_GLOBAL)
    cmd = [a.encode() for a in (args.argv or [name])]
    c_argv = (ctypes.c_char_p * (len(cmd) + 1))(*cmd, None)
    entry = None                                   # the library's own main (dlsym on its handle finds it first)
    for sym in ("main", "_Z10kdeps_mainiPPc", "kdeps_main"):
        try:
            entry = getattr(prog, sym)
            break
        except AttributeError:
            continue
    if entry is None:
        raise SystemExit(f"{so}: no main (build it with gpu/hybrid/build.sh)")
    entry.restype = ctypes.c_int
    status = entry(len(cmd), c_argv)
    sys.stdout.flush()
    finish()
    sys.stderr.flush()
    ctypes.CDLL(None).fflush(None)                                     # the program's buffered output
    os._exit(0 if status == 0 and not it.failures else 1)              # not the program's static destructors


def _several(args, argv):
    """Run the program once per schedule, each run in a process of its own (the program's native state cannot
    be reset), and merge the runs (kdeps.merge_schedules) into OUT_DIR."""
    sources = args.source.split(",")
    name = args.name or os.path.splitext(os.path.basename(sources[0]))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    start = time.time()
    so = args.so or build(sources, os.path.join(args.out_dir, f"{name}.so"),
                          [f"-D{d}" for d in args.defines] + [f"-I{i}" for i in args.includes])
    rest = list(argv if argv is not None else sys.argv[1:])
    if "--argv" in rest:                                   # the program's command line goes last
        cut = rest.index("--argv")
        rest, prog_argv = rest[:cut], rest[cut:]
    else:
        prog_argv = []
    drop = {"--schedules", "--so", "--dump", "--schedule", "--racy"}
    opts, i = [], 2                                        # SOURCE OUT_DIR first
    while i < len(rest):
        if rest[i] in drop:
            i += 2
            continue
        opts.append(rest[i])
        i += 1
    trackers, failures, counts, racy, oob = [], 0, [], None, {}
    order = args.schedules.split(",")
    order = [s for s in order if s == "lockstep"] + [s for s in order if s != "lockstep"]   # lockstep first
    for schedule in order:
        sub = os.path.join(args.out_dir, f"schedule_{schedule}")
        dump = os.path.join(sub, "run.pkl")
        os.makedirs(sub, exist_ok=True)
        cmd = [sys.executable, "-m", "memprint.static.hybrid", rest[0], sub, *opts, "--so", so,
               "--schedule", schedule, "--dump", dump, *(["--racy", ",".join(racy)] if racy is not None else []),
               *prog_argv]
        pkg = os.path.abspath(os.path.join(HERE, "..", ".."))           # the program runs where we were started
        env = dict(os.environ, PYTHONPATH=pkg + os.pathsep + os.environ.get("PYTHONPATH", ""))
        done = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if not os.path.exists(dump):
            raise SystemExit(f"{name}: the run under schedule {schedule} failed:\n{done.stdout}{done.stderr}")
        with open(dump, "rb") as f:
            d = pickle.load(f)
        os.remove(dump)                                    # large (every event's units); merged below
        for f in os.listdir(sub):                          # the merged run's access file is the one kept
            if f.endswith("_access.csv"):
                os.remove(os.path.join(sub, f))
        t = Tracker(d["unit"])
        t.events, t.edges, t.access, t.touch = d["events"], d["edges"], d["access"], d.get("touch")
        t.objects = {i: row for i, row in enumerate(d["objects"])}
        trackers.append(t)
        failures += d["failures"]
        if schedule == "lockstep":
            racy = d.get("racy", [])
        for k, v in d.get("oob", {}).items():
            oob[k] = max(oob.get(k, 0), v)
        if d["failures"]:                                  # the run's own report of what failed
            print(f"{name}: schedule {schedule}: {d['failures']} failures:\n{done.stderr[-3000:]}", file=sys.stderr)
        counts.append(f"{schedule} {len(t.events)}")
    matched = merge_schedules(trackers)
    ev_path, edge_path, rows = write(args.out_dir, name, trackers[0])
    if oob:
        print(f"{name}: {len(oob)} places reference past the end of a device buffer (possible references to "
              f"every buffer): " + ", ".join(f"{k} x{v}" for k, v in sorted(oob.items(), key=lambda kv: -kv[1])[:5]),
              file=sys.stderr)
    print(f"{name}: events per schedule: {', '.join(counts)}; {matched} of the first run's matched in all")
    print(f"{name}: {len(trackers[0].events)} events, {len(rows)} dependences ({sum(r['needed'] for r in rows)} "
          f"needed, {sum(r['certainty'] == 'may' for r in rows)} may), {failures} failures, "
          f"{time.time() - start:.1f} s", flush=True)
    print(f"  {ev_path}\n  {edge_path}", flush=True)
    return 0 if not failures else 1


def _unity(sources, out):
    with open(out, "w") as f:
        for s in sources:
            f.write(f'#include "{os.path.abspath(s)}"\n')
    return out


if __name__ == "__main__":
    sys.exit(main())
