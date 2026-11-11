"""The scheduling results of the dissertation's resident-memory chapter under the pages-in-use truth.

    PYTHONPATH=analysis OMP_NUM_THREADS=1 ~/environments/memfp/bin/python paper_rss/sched_pages.py run [part ...]
    PYTHONPATH=analysis ~/environments/memfp/bin/python paper_rss/sched_pages.py tables   ->  paper_rss/tables/sched/

Corpus ($P/corpus): the study corpus of grid.py (the 24 programs with a smaller input: no darknet), its curves at
every input, with the pages-in-use truth on the native wall clock (pagesnat) for all 24, the LLM engines' built from
their pages-touched runs (run_llm.sh truth, -track_frees) and native reference counts (footprint_corpus.sh
nativerefs). Each job takes the curve of a program at its largest input (run.largest). Scenarios: pagesnat and rss.
Calibrated margins only (the q90 of the other programs' required margins; 10% for the oracles without a profile).

Parts (run): grid (loads 0.4 to 1.2), llm (the stream of the two engines' five configurations), sens (beta, pool,
update interval at load 0.8), lisa (the hybrid Lisa stream at 32 GB per node), capacity (local GB per node at every
load).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
sys.path.insert(0, os.path.join(HERE, "..", "paper_sched"))

P = os.path.expanduser("~/memprint-data/2026-11-10-pagesnat")
SC = ("pagesnat", "rss")
LOADS = (0.4, 0.6, 0.8, 1.0, 1.2)
SEEDS = (0, 1, 2)
PROCS = 28
SCHED = ["exact", "exact-coarse", "exact-peak", "memprint-W1", "memprint-W5", "memprint-P", "memprint-W5-peak",
         "massif", "massif-i", "massif-pages", "massif-pages-i", "heaptrack", "dhat", "dhat-curve", "dhat-hold", "gperf",
         "rss-peak", "request", "ml-peak", "reactive", "pond", "zacarias", "zacarias-past", "ksplus"]
MODEL = ["rss-peak-rep", "rss-curve-rep", "memprint-W5nat", "model1-rep", "model2-rep", "rss-peak-x", "rss-curve-x",
         "memprint-W5-x", "memprint-W5nat-x", "model1-x", "model2-x"]   # paper_rss/grid.py's
CAPACITY = ["exact", "exact-coarse", "exact-peak", "memprint-W5", "memprint-P", "massif", "massif-i", "heaptrack", "dhat",
            "dhat-curve", "gperf", "rss-peak", "request", "ml-peak", "reactive", "pond", "zacarias", "zacarias-past",
            "ksplus", "rss-curve-rep", "model1-rep", "rss-curve-x", "model1-x"]
LOCALS = (48, 64, 96, 128, 160, 192, 256, 320, 384, 512)
SENS = ["exact", "exact-coarse", "exact-peak", "memprint-W5", "memprint-P", "massif-i", "dhat", "rss-peak", "request",
        "ml-peak", "reactive", "pond", "zacarias", "zacarias-past", "ksplus", "rss-curve-rep", "model1-x"]
DYN = ["reactive", "zacarias", "zacarias-past"]


def go(out, **kw):
    from memprint.sched import run
    if os.path.exists(f"{P}/results/{out}.csv"):
        print("have", out, flush=True)
        return
    common = dict(scenarios=SC, loads=LOADS, seeds=SEEDS, margins=("cal",), n=3000, procs=PROCS, kind="synthetic")
    common.update(kw)
    df = run.run(f"{P}/corpus", f"{P}/traces", f"{P}/results/{out}.csv", **common)
    print(out, len(df), "errors", int(df["error"].notna().sum()) if "error" in df else 0, flush=True)


def part_grid():
    go("grid_all", names=SCHED + MODEL)


def part_llm():
    go("grid_llm", names=SCHED, programs="llm")


def part_sens():
    from memprint.sched.model import Cluster
    for k, vals in (("beta", (0.1, 0.25, 1.0, 2.0)), ("pool", (0.0, 512.0, 2048.0, 4096.0))):
        for v in vals:
            go(f"sens/sched_{k}_{v:g}", loads=(0.8,), names=SENS, cluster=Cluster(**{k: v}))
    for v in (1, 15, 30):
        go(f"sens/sched_interval_{v}", loads=(0.8,), names=DYN, interval=v)


def part_lisa():
    from memprint.sched.lisa_run import HYBRID, lisa_cluster
    go("lisa/lisa_hybrid_32", names=HYBRID, cluster=lisa_cluster(32), kind="lisa-hybrid")


def part_capacity():
    from memprint.sched.model import Cluster
    for load in LOADS:
        for local in LOCALS:
            go(f"capacity_l{load:g}_{local}", loads=(load,), names=CAPACITY, cluster=Cluster(local=local))


def run_parts(parts):
    for d in ("sens", "lisa"):
        os.makedirs(f"{P}/results/{d}", exist_ok=True)
    for p in parts or ["grid", "llm", "sens", "lisa", "capacity"]:
        print("==", p, flush=True)
        globals()[f"part_{p}"]()
    print("all-done", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run_parts(sys.argv[2:])
    else:
        import sched_pages_tables
        sched_pages_tables.main()
