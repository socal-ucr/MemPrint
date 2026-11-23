"""Jobs per hour beside the mean response time for every scheduling table of Ch. 2 and Ch. 3 that lacked it.

    PYTHONPATH=analysis:paper_sched ~/environments/memfp/bin/python dissertation/jobs_per_hour.py   ->  dissertation/tables/jobs_per_hour.csv

Jobs/h is the simulator's throughput, the stream's jobs over the time from the first arrival to the last completion
(sim.py); the response time is the steady-state mean (the middle 80% of arrivals), as every table reports it.
Load 0.8, calibrated 90% margins (10% for the oracles without a profile), mean of three seeds.
  ch2_load08   tab:mp-load08 (2026-10-13-sched-llm, grid_all*.csv), truths bytes and rss
  ch2_e1       E1, memprint-peak per error source and margin policy (2026-11-18-motivchecks, e1_*.csv)
  ch3_policies tab:rss-policies (2026-11-10-pagesnat, grid_all*.csv with the *_model.csv rows), truths pagesnat and rss
  ch3_memprint memprint-peak beside MemRSS (2026-11-10-pagesnat, grid_memprint.csv)
  libmpr_cost  the resize cluster with libmpr's own cost (native) and with the Pin profiler's (pin), every policy
               (2026-10-22-resize-cluster, grid_native.csv, grid2_native.csv, grid_pin.csv), and without costs (grid_none)
"""
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "paper_sched"))
from make_figures import select  # noqa: E402

H = os.path.expanduser("~/memprint-data")


def load(pattern):
    g = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pattern))], ignore_index=True)
    if "error" in g:
        g = g[g.error.isna()]
    g["margin"] = g.margin.astype(str) if "margin" in g else ""
    if "restarts" in g:
        g["kills"] = g.kills + g.restarts.fillna(0)
    key = [c for c in ("scenario", "truth", "load", "seed", "margin", "oracle", "policy", "local_gb", "overheads",
                       "opt_mp_sampler") if c in g]
    return g.drop_duplicates(key, keep="last")   # grid_all*.csv repeats some oracles across its files


def summ(g, keys):
    return g.groupby(keys).agg(response_min=("steady_response", "mean"), jobs_per_h=("throughput", "mean"),
                               kills=("kills", "mean"), runs=("seed", "size")).reset_index()


def main():
    out = []
    g = select(load(f"{H}/2026-10-13-sched-llm/results/grid_all*.csv"), "cal")
    out.append(summ(g[g.load == 0.8], ["scenario", "oracle"]).assign(table="ch2_load08"))
    g = load(f"{H}/2026-11-18-motivchecks/results/e1_*.csv")
    out.append(summ(g[g.load == 0.8], ["scenario", "opt_mp_sampler", "margin"]).assign(table="ch2_e1", oracle="memprint-peak"))
    P = f"{H}/2026-11-10-pagesnat/results"
    g = load(f"{P}/grid_all.csv")
    m = load(f"{P}/grid_all_model.csv")
    g = pd.concat([g[~g.oracle.isin(m.oracle.unique())], m], ignore_index=True)   # the model's rows from its rerun
    out.append(summ(select(g, "cal").query("load == 0.8"), ["scenario", "oracle"]).assign(table="ch3_policies"))
    g = select(load(f"{P}/grid_memprint.csv"), "cal")
    out.append(summ(g[g.load == 0.8], ["scenario", "oracle"]).assign(table="ch3_memprint"))
    R = f"{H}/2026-10-22-resize-cluster"
    for f in ("grid_none", "grid_native", "grid2_native", "grid_pin"):
        g = load(f"{R}/{f}.csv")
        out.append(summ(g[g.load == 0.8], ["truth", "overheads", "policy"]).rename(columns={"truth": "scenario", "policy": "oracle"})
                   .assign(table=f"libmpr_cost:{f}"))
    d = pd.concat(out, ignore_index=True)
    cols = ["table", "scenario", "oracle", "opt_mp_sampler", "margin", "overheads", "response_min", "jobs_per_h", "kills", "runs"]
    d = d[[c for c in cols if c in d]]
    d.to_csv(os.path.join(HERE, "tables", "jobs_per_hour.csv"), index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    for t, h in d.groupby("table", sort=False):
        print(f"== {t}")
        print(h.drop(columns="table").dropna(axis=1, how="all").sort_values(["scenario", "response_min"]).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
