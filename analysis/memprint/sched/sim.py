"""The scheduler: FCFS with EASY backfilling over memory curves, local-first placement, remote-memory slowdown
and OOM kills.

Every node keeps a timeline of committed local memory and cores, and the pool one of committed memory, one
slot per minute. A job starts on a node only if its whole reservation curve fits from now to its projected
end: at every slot, the part that does not fit in the node's free local memory must fit in the free pool.

The runtime depends on the placement and the placement on the runtime: a job runs T = base (1 + beta f) where
f is the share of its actual resident GB-minutes that lie above its local reservation, so the two are solved
together by fixed-point iteration. If at some slot the actual footprint exceeds the reservation, the job is
OOM-killed there, its work is lost, and it is requeued at the head with its reservation x 1.5 (or, for KS+,
with the reservation its retry strategy gives).

Two baselines change this. A Pond job (local_cap) keeps part of its reservation in the pool even where local
memory is free, and its QoS monitor (qos_delay) moves that part to local memory after the job spills into it.
A dynamic-provisioning job (interval) commits only its first interval at the start and each resize when it
happens; a resize that does not fit fails the job, which restarts from the beginning.
"""

import heapq
import itertools
import math

import numpy as np
import pandas as pd

from .model import Cluster, at_slots

EPS = 1e-9


class Sim:
    def __init__(self, cluster: Cluster, jobs):
        self.c = cluster
        self.jobs = sorted(jobs, key=lambda j: (j.arrival, j.id))
        self.h = 4096
        self.local = np.zeros((cluster.nodes, self.h))
        self.cores = np.zeros((cluster.nodes, self.h), dtype=np.int32)
        self.pool = np.zeros(self.h)
        self.resident = np.zeros(self.h)   # actually resident GB at each slot, over all jobs
        self.mult = {j.id: 1.0 for j in self.jobs}
        self.rev = {j.id: 0 for j in self.jobs}   # bumped when a retry replaces a job's reservation
        self._slots = {}   # (job id, which, multiplier, slots) -> the curve on that many slots

    def _ensure(self, end):
        if end <= self.h:
            return
        h = max(end, 2 * self.h)
        grow = lambda a: np.concatenate([a, np.zeros(a.shape[:-1] + (h - self.h,), a.dtype)], axis=-1)
        self.local, self.cores, self.pool = grow(self.local), grow(self.cores), grow(self.pool)
        self.resident, self.h = grow(self.resident), h

    def plan(self, job, node, s):
        """The placement of `job` on `node` starting at slot `s`, or None if it does not fit."""
        c = self.c
        if self.cores[node, s] + job.cores > c.cores:
            return None
        res = job.reservation * self.mult[job.id]
        if res.max() > c.local + c.pool:
            return None
        T = job.base
        for it in range(8):
            self._ensure(s + T)
            r, a = self._at(job, "r", T), self._at(job, "a", T)
            if np.any(self.cores[node, s:s + T] + job.cores > c.cores):
                return None
            if job.interval:
                r = self._resizes(job, r, a)
            win = min(T, job.interval) if job.interval else T   # what must fit now
            free = np.maximum(c.local - self.local[node, s:s + T], 0)
            local = np.minimum(r, free)
            if job.local_cap is not None:
                local = np.minimum(local, job.local_cap)
            pool = r - local
            if np.any(pool[:win] > c.pool - self.pool[s:s + win] + EPS):
                return None
            extra = 0.0   # minutes the policy's own actions add to the run (resize and migration costs)
            if job.interval and job.resize_cost:
                L = job.interval
                grows = sum(r[i] > r[i - L] + EPS for i in range(L, T, L))
                extra += grows * job.resize_cost / 60
            if job.qos_delay is not None:
                before = pool
                local, pool = self._mitigate(job, a, r, local, free)
                moved = float(np.max(before - pool, initial=0.0))
                if job.migrate_cost:
                    extra += moved * job.migrate_cost / 60
            if job.elastic:
                local, pool, r, kill = self._grant(node, s, T, a, r, local, pool)
            else:
                over = np.flatnonzero(a > r + EPS)
                kill = int(over[0]) if len(over) else None
            run = slice(0, T if kill is None else kill)
            f = float(np.maximum(a[run] - local[run], 0).sum() / max(a[run].sum(), EPS))
            T2 = max(1, math.ceil(job.base * (1 + c.beta * f) + extra))
            if T2 == T or it == 7:   # unconverged after 8 rounds: keep the last placement, computed for T
                break
            T = T2
        p = dict(node=node, s=s, T=T, local=local, pool=pool, r=r, a=a, kill=kill, remote=f, extra=extra,
                 migrated=job.qos_delay is not None and moved > EPS)
        if job.interval:
            # only the first interval is committed now; the resizes are committed at their ticks (run)
            p["resizes"] = r
            p["local"], p["pool"], p["r"] = (np.where(np.arange(T) < win, v, 0.0) for v in (local, pool, r))
        return p

    @staticmethod
    def _resizes(job, r, a):
        """Dynamic provisioning's reservation on T slots: the request for the first interval, then each
        interval the largest footprint of that interval (the trace's next-interval maximum, as in Zacarias et
        al.'s simulator)."""
        L, out = job.interval, a.copy()
        for i in range(L, len(a), L):
            out[i:i + L] = a[i:i + L].max()
        out[:L] = r[:L]
        return out

    @staticmethod
    def _mitigate(job, a, r, local, free):
        """Pond's QoS monitor: `qos_delay` slots after the job first touches its pool memory, the pool part
        moves to the node's free local memory (a reconfiguration to all-local, where there is room)."""
        spill = np.flatnonzero(a > local + EPS)
        if not len(spill):
            return local, r - local
        k = spill[0] + job.qos_delay
        local = local.copy()
        local[k:] = np.minimum(r[k:], free[k:])
        return local, r - local

    def _grant(self, node, s, T, a, r, local, pool):
        """An elastic job's growth past its reservation, granted from the node's free local memory, then the
        free pool. It is killed at the first slot where neither has room; the grants before that count as
        reserved."""
        c = self.c
        extra = np.maximum(a - r, 0)
        free_local = np.maximum(c.local - self.local[node, s:s + T] - local, 0)
        add_local = np.minimum(extra, free_local)
        add_pool = extra - add_local
        short = np.flatnonzero(add_pool > c.pool - self.pool[s:s + T] - pool + EPS)
        kill = int(short[0]) if len(short) else None
        end = T if kill is None else kill
        local, pool, r = local.copy(), pool.copy(), r.copy()
        local[:end] += add_local[:end]
        pool[:end] += add_pool[:end]
        r[:end] += extra[:end]
        return local, pool, r, kill

    def _at(self, job, which, T):
        m = self.mult[job.id] if which == "r" else 1.0
        key = (job.id, which, m, T, self.rev[job.id])
        v = self._slots.get(key)
        if v is None:
            v = self._slots[key] = at_slots(job.reservation * m if which == "r" else job.resident, T)
        return v

    def choose(self, job, s):
        """Least pool use, then best fit (least free local memory left over), over the nodes that fit. The
        nodes are ranked on the job's base runtime, all at once; the runtime stretch is then solved (plan) on
        the best node, falling back to the next if the stretched job no longer fits."""
        c = self.c
        if job.reservation.max() * self.mult[job.id] > c.local + c.pool:
            return None
        T = job.base
        self._ensure(s + T)
        win = min(T, job.interval) if job.interval else T   # a dynamic job's resizes are checked when they happen
        r = self._at(job, "r", T)[:win]
        ok = (self.cores[:, s:s + T] + job.cores <= c.cores).all(axis=1)
        if not ok.any():
            return None
        free = np.maximum(c.local - self.local[:, s:s + win], 0)
        local = np.minimum(r, free)
        if job.local_cap is not None:
            local = np.minimum(local, job.local_cap)
        pool = r - local
        ok &= (pool <= c.pool - self.pool[s:s + win] + EPS).all(axis=1)
        if not ok.any():
            return None
        nodes = np.flatnonzero(ok)
        order = nodes[np.lexsort(((free - local)[nodes].sum(1), pool[nodes].sum(1)))]
        for n in order:
            p = self.plan(job, int(n), s)
            if p is not None:
                return p
        return None

    def _commit(self, job, p, sign, frm=0):
        n, s, T = p["node"], p["s"], p["T"]
        self.local[n, s + frm:s + T] += sign * p["local"][frm:]
        self.pool[s + frm:s + T] += sign * p["pool"][frm:]
        self.cores[n, s + frm:s + T] += sign * job.cores

    def run(self):
        c, jobs = self.c, self.jobs
        queue, ends, ticks, done, rejected = [], [], [], {}, []
        seq = itertools.count()   # tie-break: a restarted job's dead end event can share its new one's time
        stats = dict(kills=0, restarts=0, wasted_core_min=0.0, reserved=0.0, resident=0.0, remote=0.0, resizes=0,
                     migrations=0, overhead_core_min=0.0)
        nxt = 0

        def account(p, t, n):
            used = slice(0, n)
            stats["reserved"] += p["r"][used].sum()
            stats["resident"] += p["a"][used].sum()
            self.resident[t:t + n] += p["a"][used]
            stats["remote"] += np.maximum(p["a"][used] - p["local"][used], 0).sum()

        def start(job, p, t):
            self._commit(job, p, +1)
            stats["migrations"] += int(p.get("migrated", False))
            stats["overhead_core_min"] += job.cores * p.get("extra", 0.0)
            end = t + (p["T"] if p["kill"] is None else p["kill"])
            heapq.heappush(ends, (end, job.id, next(seq), job, p))
            if job.interval:   # accounted when it ends, once its resizes are known
                if job.interval < p["T"]:
                    heapq.heappush(ticks, (t + job.interval, job.id, next(seq), job, p))
                return
            account(p, t, end - t)

        def resize(job, p, t):
            """A dynamic job's resize at slot t: local first, then the pool. If the pool cannot take the rest,
            the job fails and restarts from the beginning with the same request (Zacarias et al.'s F/R)."""
            n, k0 = p["node"], t - p["s"]
            k1 = min(k0 + job.interval, p["T"])
            need = p["resizes"][k0:k1]
            local = np.minimum(need, np.maximum(c.local - self.local[n, t:t + k1 - k0], 0))
            pool = need - local
            if np.any(pool > c.pool - self.pool[t:t + k1 - k0] + EPS):
                self.cores[n, t:p["s"] + p["T"]] -= job.cores
                p["dead"] = True
                stats["restarts"] += 1
                stats["wasted_core_min"] += job.cores * k0
                account(p, p["s"], k0)
                return job
            stats["resizes"] += int(k0 == 0 or not np.isclose(need[0], p["r"][k0 - 1]))
            p["local"][k0:k1], p["pool"][k0:k1], p["r"][k0:k1] = local, pool, need
            self.local[n, t:t + k1 - k0] += local
            self.pool[t:t + k1 - k0] += pool
            if k1 < p["T"]:
                heapq.heappush(ticks, (t + job.interval, job.id, next(seq), job, p))
            return None

        def earliest(job, t):
            for s in sorted({e for e, *_ in ends if e > t}):
                p = self.choose(job, s)
                if p is not None:
                    return p
            return None

        def schedule(t):
            while queue:
                p = self.choose(queue[0], t)
                if p is not None:
                    start(queue.pop(0), p, t)
                    continue
                shadow = earliest(queue[0], t)
                if shadow is None:
                    rejected.append(queue.pop(0))
                    continue
                self._commit(queue[0], shadow, +1)
                started = []
                for j in queue[1:1 + c.window]:
                    p = self.choose(j, t)
                    if p is not None:
                        start(j, p, t)
                        started.append(j)
                self._commit(queue[0], shadow, -1)
                for j in started:
                    queue.remove(j)
                return

        while nxt < len(jobs) or ends or queue:
            t = min(jobs[nxt].arrival if nxt < len(jobs) else math.inf, ends[0][0] if ends else math.inf,
                    ticks[0][0] if ticks else math.inf)
            if t == math.inf:
                rejected.extend(queue)
                break
            killed = []
            while ends and ends[0][0] == t:
                _, _, _, job, p = heapq.heappop(ends)
                if p.get("dead"):
                    continue
                if job.interval:
                    account(p, p["s"], p["T"])
                if p["kill"] is not None:
                    self._commit(job, p, -1, frm=p["kill"])
                    stats["kills"] += 1
                    stats["wasted_core_min"] += job.cores * p["kill"]
                    new = job.retry(job, p["kill"] / p["T"]) if job.retry else None
                    if new is None:
                        self.mult[job.id] *= c.oom_growth
                    else:
                        job.reservation = np.asarray(new, float)
                        self.rev[job.id] += 1
                    killed.append(job)
                else:
                    done[job.id] = dict(id=job.id, shape=job.shape, arrival=job.arrival, start=p["s"], end=t,
                                        base=job.base, runtime=p["T"], cores=job.cores, remote=p["remote"],
                                        node=p["node"])
            while ticks and ticks[0][0] == t:
                _, _, _, job, p = heapq.heappop(ticks)
                if not p.get("dead") and resize(job, p, t) is not None:
                    killed.append(job)
            queue[:0] = killed
            while nxt < len(jobs) and jobs[nxt].arrival == t:
                queue.append(jobs[nxt])
                nxt += 1
            schedule(t)

        d = pd.DataFrame(done.values())
        if len(d):
            d["response"] = d.end - d.arrival
            d["bsld"] = d.response / np.maximum(d.base, 10)
        return d, summarize(d, stats, c, len(rejected))


def summarize(d, stats, c, rejected):
    if not len(d):
        return dict(jobs=0, rejected=rejected)
    span = d.end.max() - d.arrival.min()
    capacity = (c.nodes * c.local + c.pool) * span
    return dict(
        jobs=len(d), rejected=rejected,
        throughput=len(d) / (span / 60),
        response=d.response.mean(), response_p95=d.response.quantile(0.95), wait=(d.start - d.arrival).mean(),
        bsld=d.bsld.mean(), stretch=(d.runtime / d.base).mean(),
        kills=stats["kills"], restarts=stats["restarts"], resizes=stats["resizes"], migrations=stats["migrations"],
        overhead_core_h=stats["overhead_core_min"] / 60, wasted_core_h=stats["wasted_core_min"] / 60,
        reserved_util=stats["reserved"] / capacity, resident_util=stats["resident"] / capacity,
        pool_share=stats["remote"] / max(stats["resident"], EPS), makespan_h=span / 60,
    )


def simulate(cluster, jobs, oracle, margin=0.3, table=None):
    """Run one job stream under one oracle (jobs are copied, so one stream can be replayed per oracle)."""
    from copy import copy

    from .oracles import assign
    jobs = assign([copy(j) for j in jobs], oracle, margin, table)
    return Sim(cluster, jobs).run()
