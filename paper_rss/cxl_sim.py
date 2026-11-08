#!/usr/bin/env python3
"""Cluster throughput with a CXL pool: the resize study's cluster simulation (memprint.sched.resize_cluster) with each
job's pool cost and pool traffic from the measured runs of run_cxlsim.sh, at an assumed pool latency and bandwidth.

Per workload (run_cxlsim.sh, node 1 idle, median of the repetitions):
  the run time T_k and the reads node 1 served N_k (user-mode reads that missed the L3: loads, reads for ownership,
  prefetches) under each policy k: local, remote (all on node 1), and the plans rand25, cold25/50, hot25/50, which
  put the planned share f_k of the anonymous memory-time in the pool (place_lplans.py's planned_share);
  s, the seconds a pool read adds per ns of added latency (cxl_model.py's fit: least squares over every policy
  against local, the policies differing only in where the memory lies).
At an added pool latency d over local memory, policy k runs T_k(d) = T_k + s (N_k - N_local) (d - d0) (d0: node 1's, about 56 ns),
an extra share e_k(d) = T_k(d) / T_local - 1, and reads the pool at (N_k - N_local) 64 B / T_k(d) / threads per thread.

Placement (which bytes of a job go to the pool when its reservation spills past the node's free local memory):
  beta  the simulator's own cost, beta f with beta = 0.5 for every job (no traffic)
  rand  the random plan: e(f) through (0, 0), (f_rand25, e_rand25), (1, e_remote)
  cold  the cold-first plan: through (f_cold25, e_cold25), (f_cold50, e_cold50), (1, e_remote)
  hot   the hot-first plan, the same with the hot plans
each made non-decreasing in f and at least 0. A job is its workload scaled: its pool cost is the workload's e(f), and its
pool traffic the workload's traffic per thread at f times its cores (on each of its nodes).

The pool's loaded latency (iterate-until-settles): the measured curve of node 1 (run_cxlsim.sh bw: hog threads,
delivered GB/s, pointer chase from node 0) gives M(u) = latency(u) / latency(0) at utilisation u = GB/s / the largest
measured. A pool with idle added latency d and bandwidth B has the latency (L_local + d) M(min(U, 1)) at demand
U = traffic / B, and when U > 1 the pool's part of every run is stretched by U as well. The simulation runs at the
idle latency, the mean pool traffic over the steady window gives U, the latency and stretch follow, and the simulation
runs again (half-way damped) until the latency moves by less than 1 ns and the stretch by less than 0.01.

    PYTHONPATH=analysis python paper_rss/cxl_sim.py model           # the per-workload model and its check
    PYTHONPATH=analysis python paper_rss/cxl_sim.py grid [out.csv]  # the cluster grid (OMP_NUM_THREADS=1)
    PYTHONPATH=analysis python paper_rss/cxl_sim.py minute [out.csv]  # the same, the pool's latency minute by minute

Minute by minute (memprint.sched.sim's contention): each minute, the running jobs' reads give the pool's demand, the
demand its latency and stretch as above, and each job advances by 1 / (1 + e(f, d)) minutes of its base run. A job is
planned at the latency of the minute it starts and ends when its progress is done.
"""
import glob
import os
import sys
from copy import copy
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd

D = os.path.expanduser(os.environ.get("CXLSIM", "~/memprint-data/2026-11-08-cxlsim"))
POLS = ["local", "remote", "rand25", "cold25", "cold50", "hot25", "hot50"]
CURVES = {"rand": ["rand25"], "cold": ["cold25", "cold50"], "hot": ["hot25", "hot50"]}
ALIAS = {"miniVite (pass)": "miniVite"}
MAXIT = 10


def runs():
    rows = []
    for f in glob.glob(f"{D}/runs/*/*/*/*.time"):
        prog, rep, hog, pol = f[len(D) + 6:-5].split("/")
        t = open(f).read().split()
        try:
            secs = float(t[-2])
        except (ValueError, IndexError):
            continue
        r = dict(prog=prog, rep=int(rep), hog=int(hog), pol=pol, secs=secs)
        for line in open(f[:-5] + ".perf"):
            x = line.strip().split(",")
            if len(x) > 2 and x[2].startswith("offcore_response.all_reads.llc_miss.remote"):
                r["n"] = float(x[0]) if x[0][:1].isdigit() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def latencies():
    """hog level -> (local ns, pool ns), medians of the pointer chases during the runs"""
    l = [x.split()[1:] for x in open(f"{D}/done.txt") if x.startswith("lat ")]
    l = pd.DataFrame([(int(h), float(a), float(b)) for _, h, a, b in l], columns=["hog", "local", "pool"])
    return l.groupby("hog").median()


def bandwidth_curve():
    """(u, M): utilisation of the largest measured bandwidth, latency multiplier of the pool"""
    b = pd.read_csv(f"{D}/bw.txt", sep=" ", names=["t", "gbs", "local", "pool"]).groupby("t").median()
    u, m = (b.gbs / b.gbs.max()).to_numpy(), (b.pool / b.pool.iloc[0]).to_numpy()
    o = np.argsort(u)
    return u[o], np.maximum.accumulate(m[o]), b


class Model:
    def __init__(self):
        x = runs()
        self.lat = latencies()
        self.d0 = self.lat.pool[0] - self.lat.local[0]
        self.local_ns = self.lat.local[0]
        self.u, self.m, self.bw = bandwidth_curve()
        self._lin = {}
        threads = {l.split()[0]: int(l.split()[2]) for l in open(f"{D}/list.txt")}
        share = pd.read_csv(f"{D}/lplans/plans.csv").set_index(["prog", "plan"]).planned_share
        med = x.groupby(["prog", "hog", "pol"]).median(numeric_only=True)
        self.w, rows = {}, []
        for prog in sorted(set(x.prog)):
            if (prog, 0, "local") not in med.index:
                continue
            h0 = med.loc[(prog, 0)]
            pols = [p for p in POLS if p in h0.index]
            T, N = h0.secs, h0.n
            pairs = [p for p in pols if p != "local"]
            dt = np.array([T[p] - T["local"] for p in pairs])
            nd = np.array([(N[p] - N["local"]) * self.d0 for p in pairs])
            s = max((dt * nd).sum() / max((nd * nd).sum(), 1e-30), 0.0)
            f = {"local": 0.0, "remote": 1.0}
            for p in pols:
                if p not in f:
                    f[p] = float(share.get((prog, p), np.nan))
            self.w[prog] = dict(T=T[pols], N=N[pols], s=s, f=f, threads=threads.get(prog, 1))
            # the check at the busy pool's latency, which the fit never saw
            if (prog, H) in med.index.droplevel(2).unique():
                hh = med.loc[(prog, H)]
                dh = self.lat.pool[H] - self.lat.local[H]
                for p in pairs:
                    if p in hh.index:
                        # node 1 also serves some reads of the local run (llama2c's weights: file pages in node 1's
                        # page cache), and they slow down as well
                        pred = (T[p] + s * N[p] * (dh - self.d0)) / (T["local"] + s * N["local"] * (dh - self.d0))
                        rows.append(dict(prog=prog, pol=p, meas0=T[p] / T["local"], meash=hh.secs[p] / hh.secs["local"],
                                         predh=pred, f=f[p]))
        self.check = pd.DataFrame(rows)
        if len(self.check):
            self.check["err"] = self.check.predh - self.check.meash

    def extra(self, prog, pol, d):
        w = self.w[prog]
        # only the reads a policy adds over the local run are the pool's: a local job reads no pool
        return (w["T"][pol] + w["s"] * self.pool_reads(prog, pol) * (d - self.d0)) / w["T"]["local"] - 1

    def pool_reads(self, prog, pol):
        w = self.w[prog]
        return max(w["N"][pol] - w["N"]["local"], 0.0)

    def points(self, prog, placement):
        w = self.w[ALIAS.get(prog, prog)]
        pts = [("local", 0.0)] + [(p, w["f"][p]) for p in CURVES[placement]
                                   if p in w["T"].index and np.isfinite(w["f"][p]) and 0 < w["f"][p] < 1] + [("remote", 1.0)]
        return sorted(pts, key=lambda q: q[1])

    def cost(self, prog, placement, d, stretch=1.0):
        """f -> the run's extra share with share f of its memory-time in the pool"""
        name = ALIAS.get(prog, prog)
        pts = self.points(prog, placement)
        fs = np.array([q[1] for q in pts])
        es = np.maximum.accumulate(np.maximum([0.0 if p == "local" else self.extra(name, p, d) for p, _ in pts], 0))
        return lambda f: float(np.interp(f, fs, es)) * stretch

    def traffic(self, prog, placement, d):
        """f -> GB/s of pool reads per thread"""
        name = ALIAS.get(prog, prog)
        w, pts = self.w[name], self.points(prog, placement)
        fs = np.array([q[1] for q in pts])
        gb = [0.0 if p == "local" else self.pool_reads(name, p) * 64 / (w["T"]["local"] * (1 + max(self.extra(name, p, d), 0)))
              / w["threads"] / 1e9 for p, _ in pts]
        return lambda f: float(np.interp(f, fs, np.maximum.accumulate(gb)))

    def linear(self, prog, placement):
        """(f points, e = a + b d at each point, pool reads at each point): the cost and traffic curves of a workload
        as functions of the latency, cached"""
        key = (prog, placement)
        if key not in self._lin:
            name = ALIAS.get(prog, prog)
            w, pts = self.w[name], self.points(prog, placement)
            fs = np.array([q[1] for q in pts])
            a = np.array([0.0 if p == "local" else (w["T"][p] - w["s"] * self.pool_reads(name, p) * self.d0)
                          / w["T"]["local"] - 1 for p, _ in pts])
            b = np.array([0.0 if p == "local" else w["s"] * self.pool_reads(name, p) / w["T"]["local"] for p, _ in pts])
            gb = np.array([0.0 if p == "local" else self.pool_reads(name, p) * 64 / w["T"]["local"] / w["threads"] / 1e9
                           for p, _ in pts])
            self._lin[key] = (fs, a, b, gb)
        return self._lin[key]

    def cost_d(self, prog, placement):
        """(f, d, stretch) -> the extra share of the run, as cost(prog, placement, d, stretch)(f)"""
        fs, a, b, _ = self.linear(prog, placement)
        return lambda f, d, st=1.0: float(np.interp(f, fs, np.maximum.accumulate(np.maximum(a + b * d, 0)))) * st

    def traffic_d(self, prog, placement):
        """(f, d) -> GB/s of pool reads per thread, as traffic(prog, placement, d)(f)"""
        fs, a, b, gb = self.linear(prog, placement)
        return lambda f, d: float(np.interp(f, fs, np.maximum.accumulate(gb / (1 + np.maximum(a + b * d, 0)))))

    def loaded(self, d, U):
        return (self.local_ns + d) * float(np.interp(min(U, 1.0), self.u, self.m)) - self.local_ns


H = 8
_model = _setup = None


def _init(n):
    global _model, _setup
    from memprint.sched import resize_cluster
    _model, _setup = Model(), resize_cluster.Setup(n)


def simulate(task):
    from memprint.sched.model import Cluster
    from memprint.sched.run import steady
    from memprint.sched.sim import Sim
    truth, load, seed, policy, local, placement, d, B = task
    try:
        jobs = _setup.assign(_setup.stream(truth, load, seed), policy)
        de, st, hist = float(d), 1.0, []
        for it in range(MAXIT):
            js = []
            for j0 in jobs:
                j = copy(j0)
                j.pool_cost = None if placement == "beta" else _model.cost(j.shape, placement, de, st)
                js.append(j)
            sim = Sim(Cluster(local=local), js)
            dd, s = sim.run()
            if placement == "beta":
                break
            rate = np.zeros(sim.h)
            for r in dd.itertuples():
                rate[int(r.start):int(r.end)] += _model.traffic(r.shape, placement, de)(r.remote) * r.cores * r.nodes
            lo, hi = dd.arrival.quantile(0.1), dd.arrival.quantile(0.9)
            demand = float(rate[int(lo):int(hi) + 1].mean())
            U = demand / B if B else 0.0
            dn, sn = _model.loaded(d, U), max(1.0, U)
            hist.append((round(de, 1), round(st, 3), round(demand, 1)))
            if not B or (abs(dn - de) < 1 and abs(sn - st) < 0.01):
                break
            de, st = (de + dn) / 2, (st + sn) / 2
        return dict(truth=truth, load=load, seed=seed, policy=policy, local_gb=local, placement=placement, d_idle=d,
                    bw_gbs=B, d_eff=de, stretch_pool=st, demand_gbs=demand if placement != "beta" else 0.0,
                    util=U if placement != "beta" else 0.0, iterations=it + 1, history=str(hist), **s,
                    **steady(dd, sim))
    except Exception as e:
        import traceback
        return dict(truth=truth, load=load, seed=seed, policy=policy, local_gb=local, placement=placement, d_idle=d,
                    bw_gbs=B, error=repr(e), trace=traceback.format_exc()[-600:])


class PoolLatency:
    """the pool's added latency and bandwidth stretch at a demand of GB/s (Sim's contention)"""
    def __init__(self, model, d, B):
        self.model, self.d, self.B, self.idle = model, float(d), B, float(d)

    def at(self, demand):
        U = demand / self.B if self.B else 0.0
        return self.model.loaded(self.d, U), max(1.0, U)


def simulate_minute(task):
    """the cluster with the pool's latency set each minute by the jobs running in it (Sim's contention)"""
    from memprint.sched.model import Cluster
    from memprint.sched.run import steady
    from memprint.sched.sim import Sim
    truth, load, seed, policy, local, placement, d, B = task
    try:
        jobs = []
        for j0 in _setup.assign(_setup.stream(truth, load, seed), policy):
            j = copy(j0)
            j.pool_cost_d, j.traffic_d = _model.cost_d(j.shape, placement), _model.traffic_d(j.shape, placement)
            jobs.append(j)
        sim = Sim(Cluster(local=local), jobs, contention=PoolLatency(_model, d, B))
        dd, s = sim.run()
        lo, hi = int(dd.arrival.quantile(0.1)), int(dd.arrival.quantile(0.9))
        dem, lat = sim.demand[lo:hi + 1], sim.lat[lo:hi + 1]
        util = dem / B if B else np.zeros_like(dem)
        return dict(truth=truth, load=load, seed=seed, policy=policy, local_gb=local, placement=placement, d_idle=d,
                    bw_gbs=B, demand_gbs=dem.mean(), demand_p95=np.quantile(dem, 0.95), demand_max=dem.max(),
                    util=util.mean(), util_max=util.max(), min_over=int((util > 1).sum()), window=hi - lo + 1, d_eff=lat.mean(),
                    d_p95=np.quantile(lat, 0.95), d_max=lat.max(), **s, **steady(dd, sim))
    except Exception as e:
        import traceback
        return dict(truth=truth, load=load, seed=seed, policy=policy, local_gb=local, placement=placement, d_idle=d,
                    bw_gbs=B, error=repr(e), trace=traceback.format_exc()[-600:])


def grid(out, policies=("request", "mpr", "zacarias", "mpr+zacarias", "pond", "mpr+pond"), loads=(0.8, 1.2),
         seeds=(0, 1, 2), locals_=(128,), placements=("rand", "cold", "hot"), ds=(56, 107, 200, 300), bws=(0, 256, 64, 32),
         truth="rss", n=3000, procs=28, minute=False):
    tasks = [(truth, l, s, p, lo, "beta", 0, 0) for l, s, p, lo in product(loads, seeds, policies, locals_)]
    tasks += [(truth, l, s, p, lo, pl, d, b) for l, s, p, lo, pl, d, b in
              product(loads, seeds, policies, locals_, placements, ds, bws)]
    rows, fn = [], simulate
    if minute:   # the pool's latency set each minute: no beta runs, no unbounded bandwidth (it would be the idle one)
        tasks, fn = [t for t in tasks if t[5] != "beta" and t[7]], simulate_minute
    with Pool(procs, _init, (n,)) as p, open(out + ".jsonl", "a") as log:
        for r in p.imap_unordered(fn, tasks, chunksize=1):
            rows.append(r)
            log.write(pd.Series(r).to_json() + "\n")
            log.flush()
    pd.DataFrame(rows).to_csv(out, index=False)


NAMES = {"request": "Request", "mpr": "libmpr", "zacarias": "Dynamic provisioning", "mpr+zacarias": "libmpr + dynamic",
         "pond": "Pond", "mpr+pond": "libmpr + Pond"}


def table(csv, load=0.8, bw=64):
    """tables/cxl_grid.tex: throughput (jobs per hour, mean of the seeds) at the load, the flat beta and the pool of
    bandwidth bw at each latency with the cold plan, and at 300 ns with the random and the hot one"""
    x = pd.read_csv(csv)
    x = x[x.load == load]
    beta = x[x.placement == "beta"].groupby("policy").throughput.mean()
    y = x[(x.placement != "beta") & (x.bw_gbs == bw)].groupby(["policy", "placement", "d_idle"]).throughput.mean()
    cols = [("cold", d) for d in (56, 107, 200, 300)] + [("rand", 300), ("hot", 300)]
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables/cxl_grid.tex"), "w") as t:
        t.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n & & \\multicolumn{4}{c}{Cold first} & Random & Hot first \\\\\n"
                "\\cmidrule(lr){3-6}\\cmidrule(lr){7-7}\\cmidrule(lr){8-8}\n"
                "Policy & $\\beta = 0.5$ & +56 & +107 & +200 & +300 & +300 & +300 \\\\\n\\midrule\n")
        for p in NAMES:
            t.write(" & ".join([NAMES[p], f"{beta[p]:.1f}"] + [f"{y[(p, pl, d)]:.1f}" for pl, d in cols]) + " \\\\\n")
        t.write("\\bottomrule\n\\end{tabular}\n")


def table_minute(csv, it_csv, d=300):
    """tables/cxl_minute.tex: at the idle added latency d, per pool bandwidth, the pool's demand and latency minute by
    minute and the throughput by placement (means over the policies, loads and seeds), with the throughput of the
    iterate-until-settles runs for the cold plan"""
    from memprint.sched import resize_cluster
    m, g = pd.read_csv(csv), pd.read_csv(it_csv)
    st = resize_cluster.Setup(int(m.jobs.max()) + int(m.rejected.max()))
    win = {}
    for load, seed in set(zip(m.load, m.seed)):
        a = np.array([j.arrival for j in st.stream(m.truth.iloc[0], load, seed)])
        win[(load, seed)] = int(np.quantile(a, 0.9)) - int(np.quantile(a, 0.1)) + 1
    m["over"] = m.min_over / [win[(l, s)] for l, s in zip(m.load, m.seed)]
    x, y = m[m.d_idle == d], g[(g.d_idle == d) & (g.placement == "cold")]
    it = y.groupby("bw_gbs").throughput.mean()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables/cxl_minute.tex"), "w") as t:
        t.write("\\begin{tabular}{rrrrrrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{Demand (GB/s)} & Minutes & "
                "\\multicolumn{2}{c}{Latency (ns)} & \\multicolumn{3}{c}{Jobs/h, minute by minute} & Jobs/h, \\\\\n"
                "\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}\\cmidrule(lr){8-10}\n"
                "Bandwidth & Mean & p95 & Max & over & Mean & p95 & Cold & Random & Hot & averaged \\\\\n\\midrule\n")
        for b in sorted(x.bw_gbs.unique(), reverse=True):
            r = x[x.bw_gbs == b]
            tp = r.groupby("placement").throughput.mean()
            t.write(" & ".join([f"{b:.0f}\\,GB/s", f"{r.demand_gbs.mean():.1f}", f"{r.demand_p95.mean():.1f}",
                                f"{r.demand_max.mean():.1f}", f"{100 * r.over.mean():.0f}\\%", f"{r.d_eff.mean():.0f}",
                                f"{r.d_p95.mean():.0f}", f"{tp['cold']:.1f}", f"{tp['rand']:.1f}", f"{tp['hot']:.1f}",
                                f"{it[b]:.1f}" if b in it.index else "---"]) + " \\\\\n")
        t.write("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    if sys.argv[1] == "model":
        m = Model()
        pd.set_option("display.width", 200)
        print("d0", round(m.d0, 1), "local", round(m.local_ns, 1)); print(m.bw.round(2).to_string())
        c = m.check
        print("check at the busy pool: median |err|", c.err.abs().median().round(4), "mean", c.err.mean().round(4))
        print(c.groupby("prog").err.apply(lambda e: e.abs().max()).round(3).to_string())
        for prog in sorted(m.w):
            w = m.w[prog]
            print(prog, "thr", w["threads"], "e_remote@56/300", round(m.extra(prog, "remote", m.d0), 3),
                  round(m.extra(prog, "remote", 300), 3), "cold(.25)@300", round(m.cost(prog, "cold", 300)(0.25), 3),
                  "rand(.25)@300", round(m.cost(prog, "rand", 300)(0.25), 3),
                  "GB/s/thr remote", round(m.traffic(prog, "rand", m.d0)(1.0), 2))
    elif sys.argv[1] == "table":
        table(sys.argv[2] if len(sys.argv) > 2 else f"{D}/grid.csv")
    elif sys.argv[1] == "table_minute":
        table_minute(f"{D}/grid_minute.csv", f"{D}/grid.csv")
    elif sys.argv[1] == "minute":
        grid(sys.argv[2] if len(sys.argv) > 2 else f"{D}/grid_minute.csv", bws=(256, 64, 32, 16), minute=True)
    else:
        out = sys.argv[2] if len(sys.argv) > 2 else f"{D}/grid.csv"
        grid(out)
