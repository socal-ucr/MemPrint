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

Allocation follows glibc: requests below the mmap threshold come from the
heap (best fit, freed chunks coalesce), larger ones are mappings, and freeing
a mapping raises the threshold to its size (up to 32 MB), so later blocks of
that size come from the heap. Reuse matters because the footprint counts an
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


@dataclass(eq=False)
class Block:
    space: str
    chunk: int  # chunk start in its space
    chunk_size: int
    user: int  # first user byte
    nbytes: int


class Process:
    """Allocator plus per-unit reference counts of one run."""

    def __init__(self, reuse_mmap=False):
        self.spaces = {"heap": Space(best_fit=True), "mmap": Space(best_fit=False), "stack": Space(best_fit=False)}
        self.reuse_mmap = reuse_mmap
        self.counts = {"heap": np.zeros(0), "mmap": np.zeros(0), "stack": np.zeros(0)}
        self.threshold = MMAP_THRESHOLD
        self.references = 0.0

    def new(self, nbytes):
        nbytes = max(int(nbytes), 1)
        if nbytes + 16 >= self.threshold:
            size = _align(nbytes + 16, PAGE)
            chunk = self.spaces["mmap"].alloc(size)
            return Block("mmap", chunk, size, chunk + 16, nbytes)
        size = max(32, _align(nbytes + 16, 16))
        chunk = self.spaces["heap"].alloc(size)
        return Block("heap", chunk, size, chunk + 16, nbytes)

    def local(self, nbytes):
        """A long-lived object outside the heap (a local on the stack); never reused here."""
        size = _align(int(nbytes), 16)
        start = self.spaces["stack"].alloc(size)
        return Block("stack", start, size, start, int(nbytes))

    def delete(self, block):
        if block.space == "stack":
            return
        if block.space == "heap" or self.reuse_mmap:
            self.spaces[block.space].release(block.chunk, block.chunk_size)
        if block.space == "mmap" and self.threshold < block.chunk_size <= MMAP_THRESHOLD_MAX:
            self.threshold = block.chunk_size

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


def sort_refs(d):
    """Expected references per element of std::sort on d elements in random order.

    Introsort partitions while more than 16 elements remain (about 1.5 reads and
    0.5 writes per element per level), then one insertion sort finishes: for a
    run of r elements, about (r - 1) / 4 inversions per element, each a read and
    a write, plus a read and a write to place it."""
    d = np.asarray(d, dtype=float)
    levels = np.maximum(np.ceil(np.log2(np.maximum(d, 1) / 16)), 0)
    run = np.minimum(d, 16)
    return 2.0 * levels + 2.0 * np.maximum(run - 1, 0) / 4 + 2.0
