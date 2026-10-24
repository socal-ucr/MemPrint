"""Reproduction of Pond's Figure 3 (Li et al., ASPLOS 2023): DRAM saved by pooling a fixed share of every VM's memory
across 2 to 64 sockets, from the public Azure VM packing trace (AzurePublicDataset, AzureTracesForPacking2020: one
zone, 14 days, VM types with cores and memory as fractions of a machine generation).

    ~/environments/memfp/bin/python -m memprint.repro.pond_pooling --db packing_trace_zone_a_v1.sqlite --out DIR

Pond used Azure's internal VM-to-server traces; the public trace has no placements, so VMs are packed here: the
trace's tenants are split at random into `clusters` groups, each a cluster; within a cluster, VMs of machine generation
`machine` are placed first-fit on cores and memory in arrival order (VMs alive at the start first), opening servers as
needed. Each server's memory demand (the sum of its VMs' memory) is sampled hourly over the 14 days. With no pool a
server needs DRAM for its peak demand; with a pool share p, a server needs (1 - p) of its peak and each pool, shared by
`sockets / 2` dual-socket servers (consecutive in placement order), needs the peak of its servers' combined p share.
Savings = 1 - (local + pool DRAM) / (DRAM without a pool), averaged over the clusters. Two ways to provision a server
are reported: "server", each server with DRAM for its own peak demand (an upper bound on what any fleet could do
without pooling), and "fleet", every server of the cluster with the same DRAM, enough for the most demanding one, as a
fleet of identical servers is built and as Pond's stranding analysis assumes; pooling then saves the memory that the
uniform provisioning strands.
"""

import argparse
import os
import sqlite3

import numpy as np
import pandas as pd

DAYS = 14.0


def load(db, machine):
    c = sqlite3.connect(db)
    vm = pd.read_sql("select vmId, tenantId, vmTypeId, starttime, endtime from vm", c)
    ty = pd.read_sql(f"select vmTypeId, core, memory from vmType where machineId = {machine}", c)
    vm = vm.merge(ty, on="vmTypeId")
    vm = vm[(vm.starttime < DAYS) & (vm.endtime.isna() | (vm.endtime > 0))]
    vm["start"] = vm.starttime.clip(lower=0.0)
    vm["end"] = vm.endtime.fillna(np.inf)
    return vm


def pack(vm, step=1 / 24):
    """First-fit on (cores, memory); returns the servers' memory demand sampled every `step` days (servers x times)."""
    times = np.arange(0.0, DAYS, step)
    ev = pd.concat([pd.DataFrame(dict(t=vm.start, kind=1, i=np.arange(len(vm)))),
                    pd.DataFrame(dict(t=vm.end, kind=0, i=np.arange(len(vm))))])
    ev = ev[ev.t < DAYS].sort_values(["t", "kind"]).to_numpy()   # departures before arrivals at equal times
    core, mem = vm.core.to_numpy(), vm.memory.to_numpy()
    cap = 64
    free_c, free_m = np.ones(cap), np.ones(cap)
    used = np.zeros(cap, bool)
    where = np.full(len(vm), -1)
    samples, k, n = [], 0, 0
    for t, kind, i in ev:
        while k < len(times) and times[k] <= t:
            samples.append(1 - free_m[:n].copy())
            k += 1
        i = int(i)
        if kind == 0:
            s = where[i]
            if s >= 0:
                free_c[s] += core[i]
                free_m[s] += mem[i]
            continue
        fit = np.flatnonzero((free_c[:n] >= core[i] - 1e-12) & (free_m[:n] >= mem[i] - 1e-12))
        if len(fit):
            s = fit[0]
        else:
            if n == cap:
                cap *= 2
                free_c = np.concatenate([free_c, np.ones(cap - n)])
                free_m = np.concatenate([free_m, np.ones(cap - n)])
            s, n = n, n + 1
        free_c[s] -= core[i]
        free_m[s] -= mem[i]
        where[i] = s
    while k < len(times):
        samples.append(1 - free_m[:n].copy())
        k += 1
    out = np.zeros((n, len(samples)))
    for j, x in enumerate(samples):
        out[:len(x), j] = x
    return np.clip(out, 0, None)


def savings(demand, shares=(0.1, 0.3, 0.5), sockets=(2, 4, 8, 16, 32, 64)):
    peak = demand.max(axis=1)
    rows = []
    for prov in ("server", "fleet"):
        # local DRAM per server without a pool: its own peak, or the fleet's largest peak on every server
        local = peak if prov == "server" else np.full(len(peak), peak.max())
        base = local.sum()
        for p in shares:
            for s in sockets:
                g = max(s // 2, 1)   # dual-socket servers per pool
                pool = sum((p * demand[a:a + g]).sum(axis=0).max() for a in range(0, len(demand), g))
                rows.append(dict(provisioning=prov, pool_share=p, sockets=s,
                                 savings=1 - ((1 - p) * base + pool) / base))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--machine", type=int, default=12)
    ap.add_argument("--clusters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    vm = load(a.db, a.machine)
    tenants = vm.tenantId.unique()
    group = dict(zip(tenants, np.random.default_rng(a.seed).integers(0, a.clusters, len(tenants))))
    vm["cluster"] = vm.tenantId.map(group)
    rows = []
    os.makedirs(a.out, exist_ok=True)
    for cl, v in vm.groupby("cluster"):
        d = pack(v.sort_values("start").reset_index(drop=True))
        np.save(f"{a.out}/demand_cluster{cl}.npy", d)   # servers x hourly samples, to recompute variants
        for r in savings(d):
            rows.append(dict(cluster=cl, servers=len(d), vms=len(v), **r))
        print(f"cluster {cl}: {len(v)} VMs on {len(d)} servers", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/pond_fig3.csv", index=False)
    print(df.groupby(["provisioning", "pool_share", "sockets"]).savings.mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
