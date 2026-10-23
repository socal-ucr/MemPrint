"""Jobs with measured memory over time from the SURF Lisa trace (Zenodo 13625495, CC-BY 4.0): Slurm jobs joined with
their nodes' Prometheus metrics, 30-second samples, June to November 2022.

    ~/environments/traces/run.sh -m memprint.sched.lisa --data ~/memprint-data/traces-lisa --out DIR

(the traces venv: it needs pyarrow). The joined table has one row per (node, sample) with the Slurm job on that node.
A node's memory is a job's only when the job has the node to itself, so for every (job, node) we keep the series
only if no other job shared the node at any of its samples; a job is kept when all its nodes are kept and at least
`min_coverage` of the samples its runtime implies are present. Memory used is the node's active memory (anonymous
and active file pages) less the node's idle level (its 1st percentile over the trace), and, as an upper variant,
total minus free memory less its idle level. Each node's series is resampled to K samples over the job's run (the
largest value in each share), and a job's curve is the largest over its nodes at each sample (every node is assumed
to hold the same footprint, as the simulator does).

Writes DIR/jobs.csv (one row per kept job: id, submit/start/end, minutes, nodes, cores, state, peaks, coverage) and
DIR/curves_active.npy, DIR/curves_used.npy (K samples per job, bytes, in jobs.csv order), plus DIR/summary.txt.
"""

import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

K = 200
STEP = 30.0   # seconds between samples
COLS = ["node", "timestamp_seconds", "slurm_id", "start_date", "end_date", "submit_date", "numnodes", "numcores",
        "state", "gpu_node", "node_memory_Active_bytes", "node_memory_MemFree_bytes"]


def to_progress(t, v, start, end):
    """A series over [start, end] on K samples of progress, each the largest value in its share; NaN where empty."""
    p = np.clip(((t - start) / max(end - start, 1.0) * K).astype(int), 0, K - 1)
    out = np.full(K, np.nan)
    np.fmax.at(out, p, v)
    return out


def one_part(args):
    path, total = args
    d = pq.read_table(path, columns=COLS).to_pandas()
    d = d[d.gpu_node == 0]
    if not len(d):
        return []
    jobs_at = d.groupby(["node", "timestamp_seconds"]).slurm_id.transform("nunique")
    d["shared"] = jobs_at > 1
    d["total"] = d.node.map(total)
    d["active"] = d.node_memory_Active_bytes
    d["used"] = d.total - d.node_memory_MemFree_bytes
    out = []
    for node, g in d.groupby("node"):
        # the node's idle level: its 1st percentile over the trace (operating system, daemons, page cache floor)
        base_a, base_u = np.nanpercentile(g.active, 1), np.nanpercentile(g.used, 1)
        for jid, h in g.groupby("slurm_id"):
            r = h.iloc[0]
            start, end = r.start_date.timestamp(), r.end_date.timestamp()
            t = h.timestamp_seconds.to_numpy(float)
            out.append(dict(
                slurm_id=int(jid), node=node, exclusive=not h.shared.any(), samples=len(h),
                start=start, end=end, submit=r.submit_date.timestamp(), numnodes=int(r.numnodes),
                numcores=int(r.numcores), state=str(r.state),
                active=to_progress(t, np.maximum(h.active.to_numpy(float) - base_a, 0), start, end),
                used=to_progress(t, np.maximum(h.used.to_numpy(float) - base_u, 0), start, end)))
    return out


def ingest(data, out, procs=8, min_coverage=0.8, min_minutes=5):
    hw = pd.read_parquet(f"{data}/raw/node_hardware_info.parquet").set_index("node").cpu_memory_bytes_total.to_dict()
    parts = sorted(glob.glob(f"{data}/joined/prom_slurm_joined.parquet/part-*.parquet"))
    with Pool(procs) as p:
        rows = [r for part in p.imap_unordered(one_part, [(f, hw) for f in parts]) for r in part]
    per = pd.DataFrame([{k: v for k, v in r.items() if k not in ("active", "used")} for r in rows])
    jobs, act, use = [], [], []
    stats = dict(jobs=per.slurm_id.nunique(), node_series=len(per))
    for jid, g in per.groupby("slurm_id"):
        r = g.iloc[0]
        minutes = (r.end - r.start) / 60
        nodes_seen = len(g)
        expected = max((r.end - r.start) / STEP, 1) * r.numnodes
        coverage = g.samples.sum() / expected
        if not (g.exclusive.all() and nodes_seen == r.numnodes and coverage >= min_coverage and minutes >= min_minutes):
            continue
        idx = g.index.to_numpy()
        a = np.nanmax(np.vstack([rows[i]["active"] for i in idx]), axis=0)
        u = np.nanmax(np.vstack([rows[i]["used"] for i in idx]), axis=0)
        if np.isnan(a).all():
            continue
        # empty shares (a missed sample) take the previous value
        a, u = pd.Series(a).ffill().bfill().to_numpy(), pd.Series(u).ffill().bfill().to_numpy()
        jobs.append(dict(id=int(jid), submit=r.submit, start=r.start, end=r.end, minutes=minutes, nodes=int(r.numnodes),
                         cores=int(r.numcores), state=r.state, peak_active=a.max(), peak_used=u.max(),
                         coverage=coverage))
        act.append(a)
        use.append(u)
    os.makedirs(out, exist_ok=True)
    j = pd.DataFrame(jobs)
    j.to_csv(f"{out}/jobs.csv", index=False)
    np.save(f"{out}/curves_active.npy", np.array(act))
    np.save(f"{out}/curves_used.npy", np.array(use))
    stats.update(kept=len(j), multi_node=int((j.nodes > 1).sum()) if len(j) else 0,
                 median_minutes=float(j.minutes.median()) if len(j) else 0,
                 median_peak_active_gb=float(j.peak_active.median() / 1e9) if len(j) else 0)
    with open(f"{out}/summary.txt", "w") as f:
        for k, v in stats.items():
            f.write(f"{k} {v}\n")
    return j, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/memprint-data/traces-lisa"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    _, stats = ingest(a.data, a.out, a.procs)
    print(stats)


if __name__ == "__main__":
    main()
