"""Footprints without the interpreter, for uvm_oversub.py, uvm_blocks.py and uvm_hw.py.

    span(pages)                     every page from the first to the last a step touches, per allocation: what a
                                    static bounds analysis of the index expressions, or the lowest and highest
                                    address a run records, gives
    History(limit)                  a runtime predictor: a kernel touches what its previous launch (of the same
                                    key) touched, shifted per allocation by the step its first page moved between
                                    the two launches before; nothing at a kernel's first launch

Pages are (allocation, index) pairs, as in the models.
"""

from collections import defaultdict


def span(pages):
    lo, hi = {}, {}
    for a, p in pages:
        lo[a] = min(lo.get(a, p), p)
        hi[a] = max(hi.get(a, p), p)
    return {(a, p) for a in lo for p in range(lo[a], hi[a] + 1)}


def by_alloc(pages):
    out = defaultdict(set)
    for a, p in pages:
        out[a].add(p)
    return out


class History:
    def __init__(self, limit):
        self.limit, self.seen = limit, {}                  # limit: {alloc: pages}; seen: key -> last two

    def predict(self, key):
        past = self.seen.get(key, [])
        if not past:
            return set()
        last = by_alloc(past[-1])
        before = by_alloc(past[-2]) if len(past) > 1 else {}
        out = set()
        for a, ps in last.items():
            d = min(ps) - min(before[a]) if a in before else 0
            out |= {(a, p + d) for p in ps if 0 <= p + d < self.limit.get(a, p + d + 1)}
        return out

    def record(self, key, pages):
        self.seen[key] = (self.seen.get(key, []) + [set(pages)])[-2:]
