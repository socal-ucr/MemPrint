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

sys.exit(failed)
