"""Skeletons of more GAP kernels (cc.cc, sssp.cc, tc.cc, bc.cc, and pr.cc converging) at b5e3e19.

Each kernel is replayed on the graph sampled by gap.build: the sequential
parts (union-find, delta-stepping, ordered intersections, Brandes) run in
numba and count the reads and writes of every array element, which are then
charged to the arrays' blocks. Single-threaded builds (SERIAL=1): OpenMP
pragmas are ignored, GAP's compare_and_swap / fetch_and_add are still locked
instructions (a read and a write each). As in gap.py, choices about what -O3
keeps in registers are noted where they matter.
"""

import numpy as np
from numba import njit

from .gap import build
from .skeleton import Process, mt19937_refs, sort_refs

K_DAMP = np.float32(0.85)


def _pick_source(p, g, rng, picker=None, draws=None):
    """SourcePicker::PickNext: uniform vertices until one has out-edges; each draw reads the
    vertex's two index entries. Returns the source and the number of draws."""
    count = 0
    while True:
        source = int(rng.integers(len(g.deg)))
        count += 1
        p.touch(g.index, [source, source + 1], 8)
        if g.deg[source]:
            return source, count


def _picker(p, draws):
    """SourcePicker's std::mt19937_64: seeded once; the first output twists the state."""
    mt19937_refs(p, p.local(312 * 8), 312, max(draws, 312), seeds=1)


def _charge_csr(p, g, idx_cnt, nb_cnt):
    p.touch(g.index, np.arange(len(idx_cnt)), 8, idx_cnt)
    elem = 8 if g.w is not None else 4
    p.touch(g.neighs, np.arange(len(nb_cnt)), elem, nb_cnt)


# ---------------------------------------------------------------- cc (Afforest)


@njit(cache=True)
def _link(u, v, comp, cnt):
    p1 = comp[u]
    p2 = comp[v]
    cnt[u] += 1
    cnt[v] += 1
    while p1 != p2:
        high = max(p1, p2)
        low = p1 + (p2 - high)
        p_high = comp[high]
        cnt[high] += 1
        if p_high == low:
            break
        if p_high == high:                       # compare_and_swap succeeds (one thread)
            cnt[high] += 2
            comp[high] = low
            break
        c = comp[high]
        cnt[high] += 1
        p1 = comp[c]
        cnt[c] += 1
        p2 = comp[low]
        cnt[low] += 1


@njit(cache=True)
def _compress(comp, cnt):
    for n in range(len(comp)):
        c = comp[n]
        cc = comp[c]
        cnt[n] += 1
        cnt[c] += 1
        while c != cc:
            comp[n] = cc
            cnt[n] += 1
            c = cc
            cc = comp[c]
            cnt[c] += 1


@njit(cache=True)
def _afforest_rounds(off, nbr, rounds, comp, cnt, nb_cnt, idx_cnt):
    n = len(off) - 1
    for r in range(rounds):
        for u in range(n):
            idx_cnt[u] += 1
            idx_cnt[u + 1] += 1
            if off[u + 1] - off[u] > r:
                e = off[u] + r
                nb_cnt[e] += 1
                _link(u, nbr[e], comp, cnt)
        _compress(comp, cnt)


@njit(cache=True)
def _afforest_finish(off, nbr, rounds, largest, comp, cnt, nb_cnt, idx_cnt):
    n = len(off) - 1
    for u in range(n):
        cnt[u] += 1
        if comp[u] == largest:
            continue
        idx_cnt[u] += 1
        idx_cnt[u + 1] += 1
        for e in range(min(off[u] + rounds, off[u + 1]), off[u + 1]):
            nb_cnt[e] += 1
            _link(u, nbr[e], comp, cnt)
    _compress(comp, cnt)


def _afforest(off, nbr, samples, rounds):
    n = len(off) - 1
    comp = np.arange(n)
    cnt, nb_cnt, idx_cnt = np.zeros(n), np.zeros(len(nbr)), np.zeros(n + 1)
    _afforest_rounds(off, nbr, rounds, comp, cnt, nb_cnt, idx_cnt)
    np.add.at(cnt, samples, 1.0)                                       # SampleFrequentElement
    values, counts = np.unique(comp[samples], return_counts=True)
    _afforest_finish(off, nbr, rounds, values[np.argmax(counts)], comp, cnt, nb_cnt, idx_cnt)
    return cnt, nb_cnt, idx_cnt


def cc(scale, degree=16, seed=0, uniform=True, rounds=2):
    """cc -u|-g scale -k degree -n 1: Afforest with two neighbour rounds, a 1024-sample guess of the
    largest component, then linking of every vertex outside it, and a final compress."""
    p = Process()
    n, g, _ = build(p, scale, degree, seed, uniform)
    comp = p.new(4 * n)
    p.touch_all(comp, 4)                                               # comp[n] = n
    samples = np.random.default_rng(seed + 2).integers(0, n, 1024)
    cnt, nb_cnt, idx_cnt = _afforest(g.off, g.nbr, samples, rounds)
    p.touch(comp, np.arange(n), 4, cnt)
    _charge_csr(p, g, idx_cnt, nb_cnt)
    mt19937_refs(p, p.local(624 * 8), 624, 1024, seeds=1)             # SampleFrequentElement's generator
    p.delete(comp)
    return p


# ---------------------------------------------------------------- sssp (delta-stepping)


@njit(cache=True)
def _eccentricity(off, nbr, source):
    n = len(off) - 1
    depth = np.full(n, -1, dtype=np.int64)
    depth[source] = 0
    queue = np.empty(n, dtype=np.int64)
    queue[0] = source
    head, tail = 0, 1
    while head < tail:
        u = queue[head]
        head += 1
        for e in range(off[u], off[u + 1]):
            v = nbr[e]
            if depth[v] < 0:
                depth[v] = depth[u] + 1
                queue[tail] = v
                tail += 1
    return depth.max()


@njit(cache=True)
def _delta_step(off, nbr, w, source, delta, bin_threshold, max_bins):
    """Delta-stepping as in DeltaStep with one thread. Bins are linked lists over a push log.
    Returns per-element reference counts of dist, the neighbours, the index and the frontier;
    per bin its header references, slot references and largest size; and the sizes of the
    temporary copies of bins processed locally."""
    n = len(off) - 1
    m = len(nbr)
    inf = np.int64(2 ** 31 - 1) // 2
    dist = np.full(n, inf, dtype=np.int64)
    dist_cnt = np.zeros(n)
    nb_cnt = np.zeros(m)
    idx_cnt = np.zeros(n + 1)
    fr_cnt = np.zeros(m + 1)
    head = np.full(max_bins, -1, dtype=np.int64)
    last = np.full(max_bins, -1, dtype=np.int64)
    size = np.zeros(max_bins, dtype=np.int64)
    cap = np.zeros(max_bins, dtype=np.int64)
    largest = np.zeros(max_bins, dtype=np.int64)
    slot_refs = np.zeros(max_bins)
    header_cnt = np.zeros(max_bins)
    log_v = np.empty(m + 1, dtype=np.int64)
    log_next = np.empty(m + 1, dtype=np.int64)
    npush = 0
    nbins = 0
    copies = np.zeros(m + 1, dtype=np.int64)
    ncopies = 0
    dist[source] = 0
    dist_cnt[source] += 1
    frontier = np.empty(m + 1, dtype=np.int64)
    frontier[0] = source
    fr_cnt[0] += 1
    tail = 1
    kmax = np.int64(2 ** 62)
    curr = 0
    work = np.empty(m + 1, dtype=np.int64)
    while curr != kmax:
        nwork = 0
        for i in range(tail):                                          # the shared frontier
            fr_cnt[i] += 1
            u = frontier[i]
            dist_cnt[u] += 1
            if dist[u] >= delta * curr:
                work[nwork] = u
                nwork += 1
        while True:
            for j in range(nwork):                                     # RelaxEdges(u)
                u = work[j]
                idx_cnt[u] += 1
                idx_cnt[u + 1] += 1
                for e in range(off[u], off[u + 1]):
                    nb_cnt[e] += 1
                    v = nbr[e]
                    dist_cnt[v] += 1                                   # old_dist = dist[v]
                    dist_cnt[u] += 1                                   # dist[u] + w
                    nd = dist[u] + w[e]
                    if nd < dist[v]:
                        dist_cnt[v] += 2                               # compare_and_swap
                        dist[v] = nd
                        b = nd // delta
                        if b >= nbins:
                            nbins = b + 1                              # local_bins.resize
                        header_cnt[b] += 3                             # push_back: end, capacity, end
                        if size[b] == cap[b]:
                            slot_refs[b] += 2 * size[b]                # reallocation moves the elements
                            cap[b] = max(1, 2 * cap[b])
                        log_v[npush] = v
                        log_next[npush] = -1
                        if head[b] < 0:
                            head[b] = npush
                        else:
                            log_next[last[b]] = npush
                        last[b] = npush
                        npush += 1
                        size[b] += 1
                        slot_refs[b] += 1
                        if size[b] > largest[b]:
                            largest[b] = size[b]
            # process the current bin locally while it is small
            if curr < nbins and size[curr] > 0 and size[curr] < bin_threshold:
                k = size[curr]
                header_cnt[curr] += 2
                slot_refs[curr] += k                                   # copied into curr_bin_copy
                copies[ncopies] = k
                ncopies += 1
                nwork = 0
                x = head[curr]
                while x >= 0:
                    work[nwork] = log_v[x]
                    nwork += 1
                    x = log_next[x]
                head[curr] = -1
                last[curr] = -1
                size[curr] = 0
                continue
            break
        nxt = kmax
        for i in range(curr, nbins):
            header_cnt[i] += 1
            if size[i] > 0:
                nxt = i
                break
        tail = 0
        if nxt != kmax:
            x = head[nxt]
            while x >= 0:
                frontier[tail] = log_v[x]
                fr_cnt[tail] += 1
                tail += 1
                x = log_next[x]
            slot_refs[nxt] += size[nxt]
            head[nxt] = -1
            last[nxt] = -1
            size[nxt] = 0
        curr = nxt
    return (dist_cnt, nb_cnt, idx_cnt, fr_cnt, header_cnt[:nbins], slot_refs[:nbins], cap[:nbins],
            copies[:ncopies])


def sssp(scale, degree=16, seed=0, uniform=True, delta=1):
    """sssp -u|-g scale -k degree -n 1 (delta 1): delta-stepping from a random source on the
    weighted graph. Bins are vectors of NodeIDs that keep their buffers when emptied (modelled
    as one block holding every bin's final buffer); every bin processed locally is first copied
    into a temporary vector (allocated and freed each time)."""
    p = Process()
    n, g, _ = build(p, scale, degree, seed, uniform, weighted=True)
    source, draws = _pick_source(p, g, np.random.default_rng(seed + 1))
    _picker(p, draws)
    dist = p.new(4 * n)
    p.touch_all(dist, 4)                                               # fill(kDistInf)
    frontier = p.new(4 * len(g.nbr))                                   # pvector(num_edges_directed), unfilled
    max_bins = 255 * (int(_eccentricity(g.off, g.nbr, source)) + 1) + 2
    d_cnt, nb_cnt, idx_cnt, fr_cnt, header_cnt, slot_refs, caps, copies = _delta_step(
        g.off, g.nbr, g.w.astype(np.int64), source, delta, 1000, max_bins)
    p.touch(dist, np.arange(n), 4, d_cnt)
    _charge_csr(p, g, idx_cnt, nb_cnt)
    p.touch(frontier, np.arange(len(fr_cnt)), 4, fr_cnt)
    headers = p.new(24 * max(len(header_cnt), 1))                      # vector<vector<NodeID>> local_bins
    p.touch(headers, np.arange(len(header_cnt)), 8, header_cnt)
    bins = p.new(4 * max(int(caps.sum()), 1))                          # every bin's final buffer
    starts = np.cumsum(caps) - caps
    slots = np.concatenate([s + np.arange(c) for s, c in zip(starts, caps)]) if caps.sum() else np.zeros(0, int)
    per_slot = np.repeat(np.divide(slot_refs, caps, out=np.zeros(len(caps)), where=caps > 0), caps)
    p.touch(bins, slots, 4, per_slot)
    for k in copies:                                                   # curr_bin_copy
        tmp = p.new(4 * int(k))
        p.touch_all(tmp, 4, 2.0)                                       # written by the copy, read by the loop
        p.delete(tmp)
    p.delete(bins)
    p.delete(headers)
    p.delete(frontier)
    p.delete(dist)
    return p


# ---------------------------------------------------------------- tc (ordered count)


@njit(cache=True)
def _ordered_count(off, nbr):
    n = len(off) - 1
    nb_cnt = np.zeros(len(nbr) + 1)
    idx_cnt = np.zeros(n + 1)
    total = 0
    for u in range(n):
        idx_cnt[u] += 1
        idx_cnt[u + 1] += 1
        for e in range(off[u], off[u + 1]):
            nb_cnt[e] += 1
            v = nbr[e]
            if v > u:
                break
            idx_cnt[v] += 1                                            # out_neigh(v).begin()
            it = off[v]
            for f in range(off[u], off[u + 1]):
                nb_cnt[f] += 1
                x = nbr[f]
                if x > v:
                    break
                while True:
                    nb_cnt[it] += 1
                    if nbr[it] < x:
                        it += 1
                    else:
                        break
                if nbr[it] == x:
                    total += 1
    return nb_cnt[:len(nbr)], idx_cnt, total


def _relabel(p, g, n):
    """Builder::RelabelByDegree: sort (degree, id) pairs descending, new IDs by rank, then a new
    CSR with relabelled, sorted neighbour lists. Returns the new CSR's blocks, offsets and neighbours."""
    pairs = p.new(16 * n)
    p.touch(g.index, np.arange(n + 1), 8, 2.0)                         # out_degree(n)
    p.touch_all(pairs, 16)
    p.touch_all(pairs, 16, sort_refs(n))                               # std::sort of n pairs
    order = np.lexsort((-np.arange(n), -g.deg))                         # greater<pair>: degree, then id
    new_ids = np.empty(n, dtype=np.int64)
    new_ids[order] = np.arange(n)
    degrees, ids = p.new(4 * n), p.new(4 * n)
    p.touch_all(pairs, 16)
    p.touch_all(degrees, 4)
    p.touch_all(ids, 4)
    from .gap import _gen_index, _prefix_sum
    offsets = _prefix_sum(p, degrees, n)
    new_deg = g.deg[order]
    new_off = np.concatenate([[0], np.cumsum(new_deg)])
    neighs = p.new(4 * int(new_off[-1]))
    index = _gen_index(p, offsets, n)
    p.touch(ids, np.arange(n), 4, 1.0 + np.bincount(g.nbr, minlength=n))  # new_ids[u], new_ids[v]
    p.touch(g.index, np.arange(n + 1), 8, 2.0)
    p.touch(g.neighs, np.arange(len(g.nbr)), 4)
    p.touch(offsets, new_ids, 8, 2.0 * g.deg)                          # offsets[new_ids[u]]++
    p.touch(index, np.arange(n + 1), 8, 2.0)                           # sort(index[u'], index[u' + 1])
    p.touch(neighs, np.arange(len(g.nbr)), 4, 1.0 + np.repeat(sort_refs(new_deg), new_deg))
    src = np.repeat(np.arange(n), g.deg)
    nsrc, ndst = new_ids[src], new_ids[g.nbr]
    order2 = np.lexsort((ndst, nsrc))
    p.delete(offsets)
    p.delete(ids)
    p.delete(degrees)
    p.delete(pairs)
    return neighs, index, new_off, ndst[order2]


def tc(scale, degree=16, seed=0, uniform=True):
    """tc -u|-g scale -k degree -n 1: relabel by degree if 1000 sampled degrees say the graph is
    skewed (WorthRelabelling), then OrderedCount."""
    p = Process()
    n, g, _ = build(p, scale, degree, seed, uniform)
    relabel = False
    if (len(g.nbr) // 2) // n >= 10:                                   # average_degree = num_edges / num_nodes
        rng = np.random.default_rng(seed + 3)
        draws, degs = 0, []
        for _ in range(min(1000, n)):
            s, d = _pick_source(p, g, rng)
            draws += d
            degs.append(int(g.deg[s]))
        _picker(p, draws)
        samples = p.new(8 * len(degs))
        p.touch_all(samples, 8, 2.0 + sort_refs(len(degs)))             # written, sorted, summed
        p.delete(samples)
        relabel = np.mean(degs) / 1.3 > np.sort(degs)[len(degs) // 2]
    if relabel:
        neighs, index, off, nbr = _relabel(p, g, n)
    else:
        neighs, index, off, nbr = g.neighs, g.index, g.off, g.nbr
    nb_cnt, idx_cnt, _ = _ordered_count(off, nbr)
    p.touch(index, np.arange(n + 1), 8, idx_cnt)
    p.touch(neighs, np.arange(len(nbr)), 4, nb_cnt)
    if relabel:
        p.delete(index)
        p.delete(neighs)
    return p


# ---------------------------------------------------------------- bc (Brandes, one source)


@njit(cache=True)
def _brandes(off, nbr, source):
    n = len(off) - 1
    m = len(nbr)
    depths = np.full(n, -1, dtype=np.int64)
    dep_cnt = np.zeros(n)
    pc = np.zeros(n)
    pc_cnt = np.zeros(n)
    nb_cnt = np.zeros(m)
    idx_cnt = np.zeros(n + 1)
    succ = np.zeros(m, dtype=np.bool_)
    word_cnt = np.zeros((m + 63) // 64)
    queue = np.empty(n, dtype=np.int64)
    q_cnt = np.zeros(n)
    depths[source] = 0
    dep_cnt[source] += 1
    pc[source] = 1.0
    pc_cnt[source] += 1
    queue[0] = source
    q_cnt[0] += 1
    levels = [0]
    start, end, fill = 0, 1, 1
    pushes = 0
    flushes = 0
    depth = 0
    while start != end:
        depth += 1
        for q in range(start, end):
            q_cnt[q] += 1
            u = queue[q]
            idx_cnt[u] += 1
            idx_cnt[u + 1] += 1
            for e in range(off[u], off[u + 1]):
                nb_cnt[e] += 1
                v = nbr[e]
                dep_cnt[v] += 1
                if depths[v] == -1:
                    dep_cnt[v] += 2                                    # compare_and_swap
                    depths[v] = depth
                    queue[fill] = v
                    q_cnt[fill] += 1                                   # copied in by the flush
                    fill += 1
                    pushes += 1
                dep_cnt[v] += 1
                if depths[v] == depth:
                    word_cnt[e // 64] += 3                             # set_bit_atomic: read, CAS
                    succ[e] = True
                    pc_cnt[u] += 1
                    pc_cnt[v] += 2
                    pc[v] += pc[u]
        flushes += 1
        levels.append(end)
        start, end = end, fill
    levels.append(end)
    deltas = np.zeros(n)
    dl_cnt = np.zeros(n)
    sc_cnt = np.zeros(n)
    for d in range(len(levels) - 3, -1, -1):
        for q in range(levels[d], levels[d + 1]):
            q_cnt[q] += 1
            u = queue[q]
            idx_cnt[u] += 1
            idx_cnt[u + 1] += 1
            delta_u = 0.0
            hit = False
            for e in range(off[u], off[u + 1]):
                nb_cnt[e] += 1
                word_cnt[e // 64] += 1
                if succ[e]:
                    v = nbr[e]
                    hit = True
                    pc_cnt[v] += 1
                    dl_cnt[v] += 1
                    delta_u += (pc[u] / pc[v]) * (1 + deltas[v])
            if hit:
                pc_cnt[u] += 1                                         # path_counts[u], hoisted
            deltas[u] = delta_u
            dl_cnt[u] += 1
            sc_cnt[u] += 2
    return dep_cnt, pc_cnt, nb_cnt, idx_cnt, word_cnt, q_cnt, dl_cnt, sc_cnt, pushes, flushes, len(levels)


def bc(scale, degree=16, seed=0, uniform=True):
    """bc -u|-g scale -k degree -n 1 (-i 1): Brandes from one random source, counting shortest
    paths in a BFS that marks successor edges in a bitmap, then accumulating dependencies back."""
    p = Process()
    n, g, _ = build(p, scale, degree, seed, uniform)
    scores = p.new(4 * n)
    p.touch_all(scores, 4)                                             # fill(0)
    path_counts = p.new(8 * n)
    words = (len(g.nbr) + 63) // 64
    succ = p.new(8 * words)
    queue = p.new(4 * n)
    source, draws = _pick_source(p, g, np.random.default_rng(seed + 1))
    _picker(p, draws)
    p.touch_all(path_counts, 8)                                        # fill(0)
    p.touch_all(succ, 8)                                               # reset
    depths = p.new(4 * n)
    p.touch_all(depths, 4)                                             # fill(-1)
    (dep_cnt, pc_cnt, nb_cnt, idx_cnt, word_cnt, q_cnt, dl_cnt, sc_cnt,
     pushes, flushes, nlevels) = _brandes(g.off, g.nbr, source)
    local = p.new(4 * 16384)                                           # QueueBuffer
    p.touch(local, np.arange(min(pushes, 16384)), 4, 2.0 * max(1, pushes // 16384 + 1))
    p.touch(depths, np.arange(n), 4, dep_cnt)
    p.touch(path_counts, np.arange(n), 8, pc_cnt)
    _charge_csr(p, g, idx_cnt, nb_cnt)
    p.touch(succ, np.arange(words), 8, word_cnt)
    p.touch(queue, np.arange(n), 4, q_cnt)
    levels = p.new(8 * max(nlevels, 1))                                # depth_index
    p.touch_all(levels, 8, 2.0)
    p.delete(local)
    p.delete(depths)
    deltas = p.new(4 * n)
    p.touch_all(deltas, 4)                                             # fill(0)
    p.touch(deltas, np.arange(n), 4, dl_cnt)
    p.touch(scores, np.arange(n), 4, sc_cnt)
    p.delete(deltas)
    p.touch_all(scores, 4, 3.0)                                        # max, then scores[n] /= max
    p.delete(levels)
    p.delete(queue)
    p.delete(succ)
    p.delete(path_counts)
    p.delete(scores)
    return p


# ---------------------------------------------------------------- pr until convergence


@njit(cache=True, error_model="numpy")                                # x / 0 is inf, as in C++
def _pagerank_iterations(off, nbr, deg, max_iters, epsilon):
    """PageRankPullGS in float32 on the sampled graph: the number of iterations it runs."""
    n = len(off) - 1
    init = np.float32(1.0) / np.float32(n)
    base = (np.float32(1.0) - K_DAMP) / np.float32(n)
    scores = np.full(n, init, dtype=np.float32)
    contrib = np.empty(n, dtype=np.float32)
    for u in range(n):
        contrib[u] = init / np.float32(deg[u])
    for it in range(max_iters):
        error = 0.0
        for u in range(n):
            total = np.float32(0.0)
            for e in range(off[u], off[u + 1]):
                total += contrib[nbr[e]]
            old = scores[u]
            scores[u] = base + K_DAMP * total
            error += abs(float(scores[u]) - float(old))
            contrib[u] = scores[u] / np.float32(deg[u])
        if error < epsilon:
            return it + 1
    return max_iters


def pagerank_converge(scale, degree=16, seed=0, uniform=True, max_iters=20, tolerance=1e-4):
    """pr -u|-g scale -k degree -n 1 with its default -i 20 -t 1e-4: the iteration count comes
    from running the float32 Gauss-Seidel update on the sampled graph."""
    from .gap import pagerank

    p0 = Process()
    n, g, _ = build(p0, scale, degree, seed, uniform)
    iterations = _pagerank_iterations(g.off, g.nbr, g.deg, max_iters, tolerance)
    p = pagerank(scale, degree, iterations, seed, uniform)
    p.iterations = iterations
    return p
