"""Access-count spectra of irregular programs: structure from the code, data from the input's distribution.

An irregular program's access counts depend on its input (a vertex is read
once per neighbour). A skeleton replays the program's memory accesses with
numpy: which arrays it allocates and frees, and how often each element is
read or written per element of the input. The input itself (a graph's edges)
is sampled from its distribution, e.g. the generator a benchmark uses, not
read from the actual input.

The footprint is the bytes touched (-footprint bytes): counts are kept per
4-byte unit, and an access covering a unit counts once for it whatever its
width, so vectorised copies and scalar loops give the same counts.

Allocation follows glibc 2.28. A request is served from a free heap chunk
(best fit; freed chunks coalesce, and into the top chunk when adjacent), else
from the top chunk if it fits. Only when the top chunk is too small does the
size decide: at or above the mmap threshold the block is a mapping, below it
the heap grows by the request plus a 128 KB pad, rounded to pages. A free that
leaves the top chunk above the trim threshold (128 KB) gives memory back to
the OS down to the pad. Freeing a mapping raises the mmap threshold to its
size (up to 32 MB) and the trim threshold to twice that. So an array just
above 128 KB lands on the heap or in a mapping depending on how much room the
top chunk has, which is why the heap starts from the runtime's measured state
(HEAP_TOP_AT_START). Reuse matters because the footprint counts an
address once however many blocks occupied it. Under Pin, holes left by munmap
are not reused by the program: the tool's own mappings, which grow with the
footprint it records, take them first (GAP bfs at scale 18 shows no reuse of
its 33.5 MB edge list's addresses). reuse_mmap=True models a native run.
"""

from dataclasses import dataclass

import numpy as np

from .spectrum import Spectrum

UNIT = 4  # bytes per counted unit
PAGE = 4096
MMAP_THRESHOLD = 128 * 1024
MMAP_THRESHOLD_MAX = 32 * 1024 * 1024
TOP_PAD = 128 * 1024
MINSIZE = 32
# Free bytes in the heap's top chunk when a GAP program (libstdc++, glibc 2.28) makes its first
# large allocation: libstdc++'s 72704-byte emergency pool and the command line's strings are
# already there. Measured with mallinfo() from an LD_PRELOAD shim (bfs -u 14).
HEAP_TOP_AT_START = 59328


def _align(n, a):
    return (n + a - 1) // a * a


class Space:
    """Address space with first- or best-fit allocation of free intervals and coalescing on free."""

    def __init__(self, best_fit):
        self.free = []  # sorted (start, size)
        self.end = 0
        self.best_fit = best_fit

    def alloc(self, size):
        fits = [(i, s, n) for i, (s, n) in enumerate(self.free) if n >= size]
        if fits:
            i, start, n = min(fits, key=lambda f: f[2]) if self.best_fit else fits[0]
            if n == size:
                self.free.pop(i)
            else:
                self.free[i] = (start + size, n - size)
            return start
        if self.free and self.free[-1][0] + self.free[-1][1] == self.end:  # extend the free tail
            start = self.free.pop()[0]
            self.end = start + size
            return start
        start = self.end
        self.end += size
        return start

    def release(self, start, size):
        self.free.append((start, size))
        self.free.sort()
        merged = []
        for s, n in self.free:
            if merged and merged[-1][0] + merged[-1][1] == s:
                merged[-1] = (merged[-1][0], merged[-1][1] + n)
            else:
                merged.append((s, n))
        self.free = merged


class Heap:
    """glibc's main arena: free chunks below a top chunk that grows with brk."""

    def __init__(self, top_free):
        self.free = []  # sorted (start, size) below the top chunk
        self.top = 0  # start of the top chunk
        self.end = top_free  # end of the heap
        self.trim_threshold = MMAP_THRESHOLD

    def alloc(self, nb, threshold):
        """Chunk start for nb bytes, or None if glibc would map it instead."""
        fits = [(i, s, n) for i, (s, n) in enumerate(self.free) if n >= nb]
        if fits:
            i, start, n = min(fits, key=lambda f: f[2])
            if n - nb < MINSIZE:
                self.free.pop(i)
            else:
                self.free[i] = (start + nb, n - nb)
            return start
        if self.end - self.top < nb + MINSIZE:
            if nb >= threshold:
                return None
            self.end += _align(nb + TOP_PAD + MINSIZE - (self.end - self.top), PAGE)
        start = self.top
        self.top += nb
        return start

    def release(self, start, size):
        self.free.append((start, size))
        self.free.sort()
        merged = []
        for s, n in self.free:
            if merged and merged[-1][0] + merged[-1][1] == s:
                merged[-1] = (merged[-1][0], merged[-1][1] + n)
            else:
                merged.append((s, n))
        if merged and merged[-1][0] + merged[-1][1] == self.top:  # coalesce into the top chunk
            self.top = merged.pop()[0]
            top_size = self.end - self.top
            if top_size >= self.trim_threshold:                      # systrim
                extra = (top_size - TOP_PAD - MINSIZE - 1) // PAGE * PAGE
                if extra > 0:
                    self.end -= extra
        self.free = merged


@dataclass(eq=False)
class Block:
    space: str
    chunk: int  # chunk start in its space
    chunk_size: int
    user: int  # first user byte
    nbytes: int


class Process:
    """Allocator plus per-unit reference counts of one run."""

    def __init__(self, reuse_mmap=False, heap_top=HEAP_TOP_AT_START, threads=1):
        """threads: OpenMP threads. Threads other than the main one allocate from their own
        glibc arena (a separate heap that starts empty); their locals live on their own stacks."""
        self.spaces = {"heap": Heap(heap_top), "mmap": Space(best_fit=False), "stack": Space(best_fit=False)}
        self.counts = {"heap": np.zeros(0), "mmap": np.zeros(0), "stack": np.zeros(0)}
        for t in range(1, threads):
            self.spaces[f"arena{t}"] = Heap(0)
            self.counts[f"arena{t}"] = np.zeros(0)
        self.threads = threads
        self.reuse_mmap = reuse_mmap
        self.threshold = MMAP_THRESHOLD
        self.references = 0.0

    def new(self, nbytes, thread=0):
        nbytes = max(int(nbytes), 1)
        nb = max(MINSIZE, _align(nbytes + 8, 16))                    # request2size
        space = "heap" if thread == 0 else f"arena{thread}"
        chunk = self.spaces[space].alloc(nb, self.threshold)
        if chunk is not None:
            return Block(space, chunk, nb, chunk + 16, nbytes)
        size = _align(nb + 8, PAGE)
        chunk = self.spaces["mmap"].alloc(size)
        return Block("mmap", chunk, size, chunk + 16, nbytes)

    def local(self, nbytes):
        """A long-lived object outside the heap (a local on the stack); never reused here."""
        size = _align(int(nbytes), 16)
        start = self.spaces["stack"].alloc(size)
        return Block("stack", start, size, start, int(nbytes))

    def delete(self, block):
        if block.space == "stack":
            return
        if block.space != "mmap" or self.reuse_mmap:
            self.spaces[block.space].release(block.chunk, block.chunk_size)
        if block.space == "mmap" and self.threshold < block.chunk_size <= MMAP_THRESHOLD_MAX:
            self.threshold = block.chunk_size
            self.spaces["heap"].trim_threshold = 2 * block.chunk_size

    def touch(self, block, index, elem_size, refs=1.0):
        """Charge refs references to each element `index` (int array, or slice) of block."""
        if isinstance(index, slice):
            index = np.arange(*index.indices(block.nbytes // elem_size))
        index = np.asarray(index, dtype=np.int64)
        refs = np.broadcast_to(np.asarray(refs, dtype=float), index.shape)
        if index.size == 0:
            return
        first = (block.user + index * elem_size) // UNIT
        units = elem_size // UNIT
        hi = int(first.max()) + units
        counts = self.counts[block.space]
        if len(counts) < hi:
            grown = np.zeros(max(hi, 2 * len(counts)))
            grown[: len(counts)] = counts
            self.counts[block.space] = counts = grown
        for j in range(units):
            counts[: hi] += np.bincount(first + j, weights=refs, minlength=hi)
        self.references += float(refs.sum())

    def touch_all(self, block, elem_size, refs=1.0):
        self.touch(block, slice(None), elem_size, refs)

    def spectrum(self):
        parts = [c[c > 0] for c in self.counts.values()]
        counts = np.concatenate(parts) if parts else np.zeros(0)
        return Spectrum.from_addresses(counts, np.full(len(counts), float(UNIT)), self.references)


def mt19937_refs(p, state, words, draws, seeds=1):
    """References to a libstdc++ Mersenne Twister state of `words` 8-byte words (std::mt19937
    keeps 624 uint_fast32_t, 8 bytes on x86-64; std::mt19937_64 312 uint64_t) for `draws` outputs
    and `seeds` seedings. Seeding writes every word and reads the previous one; each output reads
    one word; every `words` outputs the twist reads two more words and writes one per word."""
    per_word = 2.0 * seeds + draws / words * 4.0
    p.touch(state, np.arange(words), 8, per_word)


# References per element of std::sort on d elements in random order, measured under Pin
# (bytes touched, 4-byte ints, g++ 8.5 -O3; microbenchmark: lists of length d sorted in place).
SORT_REFS = {1: 0.0, 2: 3.25, 3: 4.54, 4: 5.55, 6: 7.07, 8: 8.22, 12: 10.29, 16: 12.61, 20: 12.35,
             24: 13.23, 32: 13.52, 48: 14.66, 64: 15.41, 128: 17.23, 256: 19.08, 512: 20.99,
             1024: 23.09, 4096: 26.60, 16384: 30.77}
_SORT_D = np.log2(np.array(sorted(SORT_REFS)))
_SORT_R = np.array([SORT_REFS[k] for k in sorted(SORT_REFS)])


def sort_refs(d):
    """Expected references per element of std::sort on d elements in random order: the measured
    curve, interpolated in log2(d) and extrapolated beyond 16384 with its last slope."""
    x = np.log2(np.maximum(np.asarray(d, dtype=float), 1))
    out = np.interp(x, _SORT_D, _SORT_R)
    slope = (_SORT_R[-1] - _SORT_R[-2]) / (_SORT_D[-1] - _SORT_D[-2])
    return np.where(x > _SORT_D[-1], _SORT_R[-1] + slope * (x - _SORT_D[-1]), out)
