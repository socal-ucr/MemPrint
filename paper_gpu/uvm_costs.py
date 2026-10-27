"""Unified Memory costs measured on the A30, for uvm_oversub.py, uvm_blocks.py and uvm_hw.py.

    fault_ns(n, page_kb)        the stall of an operation that faults on n pages not on the device:
                                gpu/uvm_shim/faultbench (data/faultbench_{64k,4k}.csv), the faulting kernel's time less
                                the same kernel's time with the pages resident, interpolated in n (linearly beyond the
                                largest measurement)
    prefetch_ns(calls, bytes)   cudaMemPrefetchAsync of that many bytes in that many calls, waited for:
                                gpu/uvm_shim/prefetchbench (data/prefetchbench.csv), least squares over calls and bytes
    min_run(page_kb)            the fewest contiguous pages one prefetch call must move to cost less than faulting on
                                them (the policies that are aware of costs prefetch only runs this long)

Measured and Assumed hold what the simulators charge: the measured costs above, or the first models' assumptions
(one fault cost per batch of faults, one fixed copy cost per prefetch, clean pages dropped for free).
"""

import csv
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"


@lru_cache(None)
def _faults(page_kb):
    rows = list(csv.DictReader(open(DATA / f"faultbench_{page_kb}k.csv")))
    n = np.array([float(r["blocks"]) for r in rows])
    us = np.array([max(0.0, float(r["fault_us"]) - float(r["resident_us"])) for r in rows])
    return n, us * 1000


@lru_cache(None)
def _prefetch():
    """(ns per call, ns per byte) fitted on the separate and the contiguous calls together."""
    a, y = [], []
    for r in csv.DictReader(open(DATA / "prefetchbench.csv")):
        k = int(r["calls"])
        a += [[k, k * 65536], [1, k * 65536]]
        y += [float(r["separate_us"]) * 1000, float(r["contiguous_us"]) * 1000]
    (call, per_byte), *_ = np.linalg.lstsq(np.array(a, float), np.array(y), rcond=None)
    return call, per_byte


def fault_ns(n, page_kb=64):
    if n <= 0:
        return 0.0
    x, y = _faults(page_kb)
    if n <= x[-1]:
        return float(np.interp(n, x, y))
    return float(y[-1] + (n - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2]))


def prefetch_ns(calls, nbytes):
    if calls <= 0:
        return 0.0
    call, per_byte = _prefetch()
    return calls * call + nbytes * per_byte


def runs(pages):
    """Contiguous runs of (alloc, page) pairs, as lists."""
    out = []
    for p in sorted(pages):
        if out and out[-1][-1][0] == p[0] and out[-1][-1][1] + 1 == p[1]:
            out[-1].append(p)
        else:
            out.append([p])
    return out


@lru_cache(None)
def min_run(page_kb=64):
    """Smallest r with prefetch_ns(1, r pages) below the marginal cost of faulting on r more pages (at the slope of
    the fault curve over its first 16 MB, where most operations' faults lie)."""
    x, y = _faults(page_kb)
    k = int(np.searchsorted(x, (16 << 20) // (page_kb << 10)))
    slope = (y[k] - y[0]) / (x[k] - x[0])
    call, per_byte = _prefetch()
    per_page = per_byte * page_kb * 1024
    if slope <= per_page:
        return math.inf
    return max(1, math.ceil(call / (slope - per_page)))


class Measured:
    """A page lives in one place: evicting it, or a host access to it, always moves it."""
    name, fault_us, writeback_clean = "measured", "", True

    def __init__(self, page_kb, model):
        self.page_kb, self.page_bytes = page_kb, page_kb * 1024
        self.d2h_ns = model["cudaMemcpy D2H"][1] * self.page_bytes
        self.min_run = min_run(page_kb)

    def fault(self, n):
        return fault_ns(n, self.page_kb)

    def host_fault(self, moved):
        return fault_ns(moved, self.page_kb)

    def prefetch(self, lengths):
        return prefetch_ns(len(lengths), sum(lengths) * self.page_bytes)


class Assumed:
    """The first models: fault_us per batch of batch faults plus the transfers; one call per prefetch."""
    name, writeback_clean = "assumed", False

    def __init__(self, page_kb, model, fault_us, batch=256):
        self.page_bytes, self.fault_us, self.batch = page_kb * 1024, fault_us, batch
        self.h2d_ns = model["cudaMemcpy H2D"][1] * self.page_bytes
        self.d2h_ns = model["cudaMemcpy D2H"][1] * self.page_bytes
        self.call_ns = model["cudaMemcpy H2D"][0]
        self.min_run = 1

    def fault(self, n):
        return math.ceil(n / self.batch) * self.fault_us * 1000 + n * self.h2d_ns if n else 0

    def host_fault(self, moved):
        return self.fault_us * 1000 + moved * self.d2h_ns

    def prefetch(self, lengths):
        return self.call_ns + sum(lengths) * self.h2d_ns if lengths else 0


if __name__ == "__main__":
    c, b = _prefetch()
    print(f"prefetch: {c / 1000:.1f} us per call, {1 / b:.2f} GB/s")
    for kb in (64, 4):
        print(f"{kb} KB pages: fault 1 {fault_ns(1, kb) / 1000:.1f} us, 16 {fault_ns(16, kb) / 1000:.1f} us, "
              f"256 {fault_ns(256, kb) / 1000:.1f} us; min run {min_run(kb)} pages")
