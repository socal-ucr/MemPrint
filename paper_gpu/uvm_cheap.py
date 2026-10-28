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


if __name__ == "__main__":
    # data/uvm_span_gaps{,_large}.csv: per program and page size, the pages the kernels touch (per launch and
    # allocation) and the pages inside their spans that they do not touch
    import csv
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import benchmarks as B
    from copy_savings import H2D, D2H, UNIT, accesses
    rows = []
    for name in sys.argv[1].split(","):
        w = B.workdir(B.BENCH[name])
        tev = list(csv.DictReader(open(w / "truth" / "truth_events.csv")))
        tacc = accesses(w / "truth" / "truth_access.csv")
        for kb in (64, 4):
            pu = kb * 1024 // UNIT
            touched = gaps = 0
            for i, e in enumerate(tev):
                if e["kind"] != "kernel" or e["name"] in (H2D, D2H):
                    continue
                pages = {(a, p) for a, lo, hi, _ in tacc.get(i, []) if hi > lo
                         for p in range(lo // pu, (hi - 1) // pu + 1)}
                touched += len(pages)
                gaps += len(span(pages)) - len(pages)
            rows.append({"name": name, "page_kb": kb, "touched": touched, "gaps": gaps})
    path = Path(__file__).resolve().parent / "data" / f"uvm_span_gaps{B.SUFFIX}.csv"
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", path)
