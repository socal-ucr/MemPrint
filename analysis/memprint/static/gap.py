"""Skeletons of GAP benchmark kernels (src/builder.h, pr.cc, bfs.cc at b5e3e19).

The graph is sampled from the generator's distribution: -u SCALE -k DEGREE
draws DEGREE * 2^SCALE edges with uniform endpoints. Builds are
single-threaded (SERIAL=1). A generated graph is always symmetrised
(command_line.h: `if (scale_ != -1) symmetrize_ = true`): one CSR holds both
directions of every edge and serves as in- and out-graph. Element sizes:
NodeID 4 B, Edge 8 B, offsets and index pointers 8 B, ScoreT (float) 4 B.
"""

import numpy as np

from .skeleton import Process, mt19937_refs, sort_refs


def uniform_edges(scale, degree, seed=0):
    rng = np.random.default_rng(seed)
    n = 1 << scale
    m = n * degree
    return n, rng.integers(0, n, m, dtype=np.int64), rng.integers(0, n, m, dtype=np.int64)


def _prefix_sum(p, degrees, n):
    """ParallelPrefixSum: one block below 2^20 elements; reads degrees twice."""
    blocks = max(1, -(-n // (1 << 20)))
    local = p.new(8 * blocks)
    p.touch_all(degrees, 4)                    # local sums
    p.touch_all(local, 8, 2)                   # written, then read
    bulk = p.new(8 * (blocks + 1))
    p.touch_all(bulk, 8, 2)
    prefix = p.new(8 * (n + 1))
    p.touch_all(degrees, 4)                    # second pass
    p.touch_all(prefix, 8)                     # written
    p.delete(local)
    p.delete(bulk)
    return prefix


def _gen_index(p, offsets, n):
    index = p.new(8 * (n + 1))
    p.touch_all(offsets, 8)
    p.touch_all(index, 8)
    return index


def _make_csr(p, el, src, dst, n):
    """MakeCSR: count degrees of src, prefix sum, fill neighbours with dst. For a symmetrised
    graph src and dst hold both directions of every edge; each edge list entry is still read
    once per pass."""
    degrees = p.new(4 * n)
    p.touch_all(degrees, 4)                                            # fill(0)
    p.touch_all(el, 8)                                                 # Edge e = *it
    deg = np.bincount(src, minlength=n)
    p.touch(degrees, np.arange(n), 4, 2.0 * deg)                       # fetch_and_add
    offsets = _prefix_sum(p, degrees, n)
    m = len(src)
    neighs = p.new(4 * m)
    index = _gen_index(p, offsets, n)
    p.touch_all(el, 8)
    p.touch(offsets, np.arange(n), 8, 2.0 * deg)                       # fetch_and_add per edge
    p.touch_all(neighs, 4)                                             # one write per slot
    p.delete(offsets)
    p.delete(degrees)
    return neighs, index, deg


def _ranges(starts, lengths):
    """Concatenated arange(starts[i], starts[i] + lengths[i])."""
    lengths = np.asarray(lengths, dtype=np.int64)
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offs = np.repeat(np.cumsum(lengths) - lengths, lengths)
    return np.repeat(np.asarray(starts, dtype=np.int64), lengths) + np.arange(total) - offs


def _squish(p, neighs, index, src, dst, n):
    """SquishCSR: sort, unique and drop self loops per vertex, then copy into a new CSR.
    Returns (sq_neighs block, sq_index block, squished degrees, squished CSR offsets, neighbours)."""
    deg = np.bincount(src, minlength=n)
    starts = np.cumsum(deg) - deg
    diffs = p.new(4 * n)
    p.touch(index, np.arange(n + 1), 8, 2.0)                          # begin/end per vertex
    per_edge = np.repeat(sort_refs(deg) + 2.0 + 1.0, deg)            # sort, unique (2 reads), remove (1)
    p.touch(neighs, np.arange(int(deg.sum())), 4, per_edge)
    pairs = np.unique(src * n + dst)
    pairs = pairs[pairs // n != pairs % n]                           # self loops removed
    sq_src, sq_dst = pairs // n, pairs % n
    kept = np.bincount(sq_src, minlength=n)
    p.touch_all(diffs, 4)
    offsets = _prefix_sum(p, diffs, n)
    sq_neighs = p.new(4 * len(pairs))
    sq_index = _gen_index(p, offsets, n)
    p.touch(index, np.arange(n), 8)                                    # begin for the copy
    p.touch(diffs, np.arange(n), 4)
    p.touch(sq_index, np.arange(n), 8)
    p.touch(neighs, _ranges(starts, kept), 4)                          # copy reads
    p.touch_all(sq_neighs, 4)                                          # copy writes
    p.delete(offsets)
    p.delete(diffs)
    return sq_neighs, sq_index, kept, np.concatenate([[0], np.cumsum(kept)]), sq_dst


def build(p, scale, degree, seed=0):
    """Builder::MakeGraph for -u scale -k degree: returns the squished CSR (as out- and in-graph)."""
    n, u, v = uniform_edges(scale, degree, seed)
    m = len(u)
    el = p.new(8 * m)
    p.touch_all(el, 8)                                                 # generated
    rng = p.local(624 * 8)                                             # std::mt19937, reseeded every 2^18 edges
    mt19937_refs(p, rng, 624, 2 * m, seeds=-(-m // (1 << 18)))
    src, dst = np.concatenate([u, v]), np.concatenate([v, u])
    neighs, index, _ = _make_csr(p, el, src, dst, n)
    p.delete(el)                                                       # end of MakeGraph's scope
    sq = _squish(p, neighs, index, src, dst, n)
    p.delete(index)                                                    # the unsquished graph
    p.delete(neighs)
    return n, sq, sq                                                   # out- and in-graph are the same


def pagerank(scale, degree=16, iterations=20, seed=0):
    """pr -u scale -k degree -n 1 -i iterations -t 0 (fixed iterations, pull, Gauss-Seidel).

    At -O3 scores[u] is read and written once per iteration (the error uses the
    register), and outgoing_contrib[v] is read once per in-edge of every u, i.e.
    out-degree(v) times."""
    p = Process()
    n, (_, out_index, out_kept, _, _), (in_neighs, in_index, _, _, _) = build(p, scale, degree, seed)
    scores = p.new(4 * n)
    contrib = p.new(4 * n)
    p.touch_all(scores, 4)                                             # fill(init_score)
    p.touch(out_index, np.arange(n + 1), 8, 2.0)                       # out_degree(n) = index[n+1] - index[n]
    p.touch_all(contrib, 4)
    for _ in range(iterations):
        p.touch(in_index, np.arange(n + 1), 8, 2.0)                    # in_neigh(u): begin, end
        p.touch_all(in_neighs, 4)                                      # each in-edge once
        p.touch(contrib, np.arange(n), 4, out_kept.astype(float))      # gather
        p.touch_all(scores, 4, 2.0)                                    # read, write
        p.touch(out_index, np.arange(n + 1), 8, 2.0)                   # out_degree(u)
        p.touch_all(contrib, 4)                                        # write
    p.delete(contrib)
    return p


def _first_hits(hit, lengths):
    """Per segment (lengths), the index of the first True in hit, or -1."""
    n = len(lengths)
    out = np.full(n, -1, dtype=np.int64)
    if len(hit) == 0:
        return out
    seg = np.repeat(np.arange(n), lengths)
    pos = np.arange(len(hit)) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    idx = np.nonzero(hit)[0]
    seg_hit, first = np.unique(seg[idx], return_index=True)
    out[seg_hit] = pos[idx[first]]
    return out


class _Queue:
    """SlidingQueue plus the QueueBuffer (16384 entries, heap) each step pushes through."""

    LOCAL = 16384

    def __init__(self, p, n):
        self.p = p
        self.block = p.new(4 * n)
        self.items = np.zeros(n, dtype=np.int64)
        self.start = self.end = self.fill = 0

    def push_one(self, v):
        self.p.touch(self.block, [self.fill], 4)
        self.items[self.fill] = v
        self.fill += 1

    def push(self, values):
        """Push values through a QueueBuffer: written locally, copied into the queue on flush."""
        local = self.p.new(4 * self.LOCAL)
        k = len(values)
        if k:
            slots = np.arange(k) % self.LOCAL
            self.p.touch(local, slots, 4, 2.0)                       # write, then read by the copy
            self.p.touch(self.block, np.arange(self.fill, self.fill + k), 4)
            self.items[self.fill:self.fill + k] = values
            self.fill += k
        self.p.delete(local)

    def slide(self):
        self.start, self.end = self.end, self.fill

    def window(self):
        self.p.touch(self.block, np.arange(self.start, self.end), 4)
        return self.items[self.start:self.end]


def bfs(scale, degree=16, seed=0, alpha=15, beta=18):
    """bfs -u scale -k degree -n 1: direction-optimising BFS from a random source with out-edges."""
    p = Process()
    n, (out_neighs, out_index, out_deg, out_off, out_nbr), (in_neighs, in_index, in_deg, in_off, in_nbr) = \
        build(p, scale, degree, seed)
    picker = p.local(312 * 8)                                          # SourcePicker's std::mt19937_64
    rng = np.random.default_rng(seed + 1)
    draws = 0
    while True:                                                        # SourcePicker::PickNext
        source = int(rng.integers(n))
        draws += 1
        p.touch(out_index, [source, source + 1], 8)
        if out_deg[source]:
            break
    mt19937_refs(p, picker, 312, max(draws, 312), seeds=1)             # the first output twists the state
    parent = p.new(4 * n)                                              # InitParent
    p.touch(out_index, np.arange(n + 1), 8, 2.0)
    p.touch_all(parent, 4)
    par = np.where(out_deg > 0, -out_deg, -1).astype(np.int64)
    par[source] = source
    p.touch(parent, [source], 4)
    queue = _Queue(p, n)
    queue.push_one(source)
    queue.slide()
    words = (n + 63) // 64
    bitmaps = [p.new(8 * words), p.new(8 * words)]                    # curr, front
    bits = [np.zeros(n, bool), np.zeros(n, bool)]
    for b in bitmaps:
        p.touch_all(b, 8)                                              # reset
    CURR, FRONT = 0, 1
    edges_to_check = int(out_deg.sum())
    scout = int(out_deg[source])
    while queue.start != queue.end:
        if scout > edges_to_check / alpha:
            frontier = queue.window()                                  # QueueToBitmap
            p.touch(bitmaps[FRONT], frontier // 64, 8, 3.0)            # read, then CAS (read, write)
            bits[FRONT][frontier] = True
            awake = len(frontier)
            queue.slide()
            while True:
                old = awake
                p.touch_all(bitmaps[CURR], 8)                          # next.reset()
                bits[CURR][:] = False
                p.touch_all(parent, 4)                                 # parent[u] < 0
                unvisited = np.nonzero(par < 0)[0]
                p.touch(in_index, np.concatenate([unvisited, unvisited + 1]), 8)
                lengths = in_deg[unvisited]
                edges = _ranges(in_off[unvisited], lengths)
                nbrs = in_nbr[edges]
                first = _first_hits(bits[FRONT][nbrs], lengths)
                scanned = np.where(first >= 0, first + 1, lengths)
                scanned_edges = _ranges(in_off[unvisited], scanned)
                p.touch(in_neighs, scanned_edges, 4)
                p.touch(bitmaps[FRONT], in_nbr[scanned_edges] // 64, 8)  # get_bit
                found = first >= 0
                woken = unvisited[found]
                par[woken] = nbrs[(np.cumsum(lengths) - lengths)[found] + first[found]]
                p.touch(parent, woken, 4)
                p.touch(bitmaps[CURR], woken // 64, 8, 2.0)             # set_bit: read, write
                bits[CURR][woken] = True
                awake = len(woken)
                CURR, FRONT = FRONT, CURR                              # front.swap(curr)
                if not (awake >= old or awake > n / beta):
                    break
            p.touch(bitmaps[FRONT], np.arange(n) // 64, 8)             # BitmapToQueue: get_bit per vertex
            queue.push(np.nonzero(bits[FRONT])[0])
            queue.slide()
            scout = 1
        else:
            edges_to_check -= scout                                    # TDStep
            frontier = queue.window()
            p.touch(out_index, np.concatenate([frontier, frontier + 1]), 8)
            lengths = out_deg[frontier]
            edges = _ranges(out_off[frontier], lengths)
            v = out_nbr[edges]
            p.touch(out_neighs, edges, 4)
            p.touch(parent, v, 4)                                      # curr_val = parent[v]
            cand = np.nonzero(par[v] < 0)[0]
            claimed_v, first = np.unique(v[cand], return_index=True)
            order = np.argsort(first)
            claimed_v = claimed_v[order]
            claimer = np.repeat(frontier, lengths)[cand[first[order]]]
            p.touch(parent, claimed_v, 4, 2.0)                         # compare_and_swap
            scout = int(-par[claimed_v].sum())
            par[claimed_v] = claimer
            queue.push(claimed_v)
            queue.slide()
    p.touch_all(parent, 4)                                             # parent[n] < -1 -> -1
    p.touch(parent, np.nonzero(par < -1)[0], 4)
    return p
