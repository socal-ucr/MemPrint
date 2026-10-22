"""Toy cases for the scheduler (python tests/sched/test_sim.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from memprint.sched.model import K, Cluster, Job  # noqa: E402
from memprint.sched.oracles import curve, peak  # noqa: E402
from memprint.sched.sim import Sim  # noqa: E402

half = np.arange(K) < K // 2
failed = 0


def check(name, ok, detail=""):
    global failed
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    failed |= not ok


def job(i, resident, reservation, base=100, cores=1, arrival=0):
    j = Job(id=i, arrival=arrival, cores=cores, base=base, resident=np.asarray(resident, float))
    j.reservation = np.asarray(reservation, float)
    return j


# Two jobs whose peaks do not overlap share one node under their curves, but not under their peaks.
one_node = Cluster(nodes=1, cores=8, local=100, pool=0)
a = np.where(half, 80.0, 10.0)
b = np.where(half, 10.0, 80.0)
d, s = Sim(one_node, [job(0, a, curve(a)), job(1, b, curve(b))]).run()
check("curves overlap in time", d.start.max() == 0, f"starts {d.start.tolist()}")
d, s = Sim(one_node, [job(0, a, peak(a)), job(1, b, peak(b))]).run()
check("peaks run one after the other", d.start.max() == 100, f"starts {d.start.tolist()}")

# An under-reservation is OOM-killed where the footprint passes it, and requeued with x1.5.
d, s = Sim(one_node, [job(0, np.where(half, 30.0, 60.0), np.full(K, 50.0))]).run()
check("OOM kill and requeue", s["kills"] == 1 and d.start.iloc[0] == 50 and d.end.iloc[0] == 150,
      f"kills {s['kills']}, start {d.start.iloc[0]}, end {d.end.iloc[0]}")
check("wasted work counted", s["wasted_core_h"] == 50 / 60)

# Backfill: a small job may start now only if it does not delay the head.
c = Cluster(nodes=1, cores=8, local=100, pool=0)
jobs = [job(0, np.full(K, 50.0), np.full(K, 50.0), base=100),
        job(1, np.full(K, 60.0), np.full(K, 60.0), base=100, arrival=1),   # head: waits for job 0
        job(2, np.full(K, 30.0), np.full(K, 30.0), base=50, arrival=2),    # fits beside job 0 and ends first
        job(3, np.full(K, 45.0), np.full(K, 45.0), base=500, arrival=3)]   # fits at t=52, but not beside the head
d, s = Sim(c, jobs).run()
st = d.set_index("id").start
check("backfill a short job", st[2] == 2, f"start {st[2]}")
check("head not delayed", st[1] == 100, f"start {st[1]}")
check("no backfill that delays the head", st[3] >= 100, f"start {st[3]}")

# Remote memory stretches the runtime: all of it in the pool runs 1 + beta times as long.
c = Cluster(nodes=1, cores=8, local=0.0, pool=100, beta=0.5)
d, s = Sim(c, [job(0, np.full(K, 40.0), np.full(K, 40.0), base=100)]).run()
check("pool slowdown", d.runtime.iloc[0] == 150, f"runtime {d.runtime.iloc[0]}")

# Elastic (reactive) jobs: growth past the reservation is granted while memory is free, killed when not.
c = Cluster(nodes=1, cores=8, local=100, pool=0)
grow = np.where(half, 20.0, 60.0)
e = job(0, grow, np.full(K, 20.0))
e.elastic = True
d, s = Sim(c, [e]).run()
check("elastic growth granted", s["kills"] == 0 and d.end.iloc[0] == 100, f"kills {s['kills']}")
e = job(0, grow, np.full(K, 20.0), arrival=1)
e.elastic = True
d, s = Sim(c, [job(1, np.full(K, 50.0), np.full(K, 50.0), base=300), e]).run()
check("elastic growth killed when memory is taken", s["kills"] >= 1, f"kills {s['kills']}")

# Pond: the predicted untouched memory goes to the pool even with local memory free; a correct prediction
# costs nothing, an overprediction is slowed down until the QoS monitor moves it to local memory.
c = Cluster(nodes=1, cores=8, local=100, pool=100, beta=0.5)
pj = job(0, np.full(K, 40.0), np.full(K, 80.0))
pj.local_cap = 40.0
d, s = Sim(c, [pj]).run()
check("Pond: untouched part in the pool, no slowdown", d.runtime.iloc[0] == 100 and s["pool_share"] == 0,
      f"runtime {d.runtime.iloc[0]}")
sim = Sim(c, [pj])
p = sim.plan(pj, 0, 0)
check("Pond: pool holds request - local cap", np.allclose(p["pool"], 40.0) and np.allclose(p["local"], 40.0))
over = lambda: job(0, np.full(K, 60.0), np.full(K, 80.0))
pj = over()
pj.local_cap = 40.0
d, s = Sim(c, [pj]).run()
slow = d.runtime.iloc[0]
pj = over()
pj.local_cap, pj.qos_delay = 40.0, 10
d, s = Sim(c, [pj]).run()
check("Pond: overprediction slows, QoS mitigation recovers", slow > d.runtime.iloc[0] > 100 and s["kills"] == 0,
      f"runtime {slow} without, {d.runtime.iloc[0]} with mitigation")

# Dynamic provisioning: the request for the first interval, then each interval's largest footprint; a resize
# that does not fit fails the job, which restarts from the beginning.
c = Cluster(nodes=1, cores=8, local=100, pool=0)
dj = job(0, np.where(half, 20.0, 60.0), np.full(K, 90.0))
dj.interval = 10
sim = Sim(c, [dj])
d, s = sim.run()
check("dynamic: resized to the coming interval", s["restarts"] == 0 and d.end.iloc[0] == 100
      and sim.local[0, :10].max() == 90 and sim.local[0, 10:50].max() == 20 and sim.local[0, 50:100].max() == 60,
      f"local {sim.local[0, [0, 20, 70]].tolist()}")
dj = job(0, np.where(half, 20.0, 60.0), np.full(K, 90.0))
dj.interval = 10
other = job(1, np.full(K, 50.0), np.full(K, 50.0), base=100, arrival=20)   # takes the memory freed at t=10
d, s = Sim(c, [dj, other]).run()
st = d.set_index("id")
check("dynamic: growth that does not fit fails and restarts", s["restarts"] == 1 and st.start[1] == 20
      and st.start[0] >= 120, f"restarts {s['restarts']}, starts {st.start.to_dict()}")

# KS+: segmentation into a non-decreasing step function, the retry strategy, and the simulator's retry hook.
from memprint.sched import ksplus  # noqa: E402
sizes, peaks = ksplus.get_segments([1, 1, 3, 2, 3, 5, 5, 4, 8, 8], 3)
check("KS+: segments non-decreasing, k of them", len(sizes) == 3 and sum(sizes) == 10 and peaks == sorted(peaks)
      and peaks[-1] == 8, f"sizes {sizes} peaks {peaks}")
ends, pk = ksplus.fail_at([0.2, 0.5, 1.0], [1, 2, 3], 0.3)
check("KS+: failure before the last segment pulls the later ends in", np.allclose(ends, [0.2, 0.3, 0.6]) and pk == [1, 2, 3],
      f"ends {ends}")
ends, pk = ksplus.fail_at([0.2, 0.5, 1.0], [1, 2, 3], 0.7)
check("KS+: failure in the last segment raises its peak 20%", np.allclose(pk, [1, 2, 3.6]) and ends == [0.2, 0.5, 1.0])
c = Cluster(nodes=1, cores=8, local=100, pool=0)
kj = job(0, np.where(half, 30.0, 60.0), np.full(K, 50.0))
seen = []


def retry(j, t):
    seen.append(round(t, 2))
    return np.full(K, 70.0)


kj.retry = retry
d, s = Sim(c, [kj]).run()
check("KS+: the simulator retries with the strategy's reservation", s["kills"] == 1 and seen == [0.5]
      and d.end.iloc[0] == 150, f"kills {s['kills']}, failed at {seen}")

sys.exit(failed)
