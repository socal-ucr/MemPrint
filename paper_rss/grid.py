"""The model in the scheduling simulator (DMSched-Sim, analysis/memprint/sched).

    PYTHONPATH=analysis python paper_rss/grid.py build   ->  $D/model/{curves,costs,done}.csv|txt
    PYTHONPATH=analysis python paper_rss/grid.py run     ->  $D/model/grid.csv (run with OMP_NUM_THREADS=1)

The model corpus is the earlier-run study's corpus (study/: RSS truth, bytes truth, rssrep1, MemPrint W5, with
their transfers) plus the pages-touched truth at each program's largest input, on Pin's clock of references
(pages) and placed on the native wall clock by the native run's memory references (pagesnat), and the model's two
estimates at every input, above the runtime's pages as every curve of the simulator is: model1, anonymous RSS -
glibc's free bytes; model2, PSS - free bytes less its smallest value in the run. model1X and model2X are their
transfers from the smaller inputs (earlier.transfer, the rule of the other X curves)."""
import os
import sys

import numpy as np
import pandas as pd

from memprint.sched import corpus as C, earlier, run
from memprint.sched.model import K

D = os.path.expanduser("~/memprint-data/2026-10-26-rsscurve")
PAGES = os.path.expanduser("~/memprint-data/2026-10-12-resize/corpus-pages")
TRACES = os.path.expanduser("~/memprint-data/2026-10-13-sched-fm/traces")
OUT = f"{D}/model"
NAMES = ["exact", "exact-peak", "request", "ksplus",
         "rss-peak-rep", "rss-curve-rep", "memprint-W5", "memprint-W5nat", "model1-rep", "model2-rep",
         "rss-peak-x", "rss-curve-x", "memprint-W5-x", "memprint-W5nat-x", "model1-x", "model2-x"]


def model_curves(path):
    r = pd.read_csv(path)
    r = r[r.rss_bytes > 0]
    m1 = C.resample(r.seconds, np.clip(r.anon_bytes - r.free_bytes, 0, None))
    m2 = np.clip(r.pss_bytes - r.free_bytes, 0, None)
    # the smallest sample, not the smallest step: a step holds its largest sample, so the runtime's alone is lost
    return {"model1": m1, "model2": C.resample(r.seconds, m2) - m2.min()}


def build():
    os.makedirs(OUT, exist_ok=True)
    study = pd.read_csv(f"{D}/study/curves.csv", dtype={"input": str})
    rows = []
    add = lambda prog, inp, method, v: rows.append(pd.DataFrame(dict(prog=prog, input=inp, method=method,
                                                                     k=np.arange(K), bytes=v)))
    for prog in sorted(os.listdir(PAGES)):
        if not os.path.isdir(f"{PAGES}/{prog}") or prog not in set(study.prog):
            continue   # the study's programs: those with a smaller input (not darknet)
        big = os.listdir(f"{PAGES}/{prog}")[0]
        got = C.truth(f"{PAGES}/{prog}/{big}/truthpages")
        add(prog, big, "pages", C.resample(*got))
        add(prog, big, "pagesnat", C.resample(*C.on_native_clock(*got, C.native_clock(
            f"{D}/corpus/{prog}/{big}/nativerefs"))))
        runs = {big: f"{D}/opts/runs_all/{prog}/1/pss.csv"}
        small = f"{D}/opts/runs_small/{prog}"
        if os.path.isdir(small):
            runs.update({i: f"{small}/{i}/pss.csv" for i in os.listdir(small)})
        for inp, path in runs.items():
            for m, v in model_curves(path).items():
                add(prog, inp, m, v)
    new = pd.concat(rows, ignore_index=True)
    new = pd.concat([new, earlier.transfer(new, sources=("model1", "model2"))], ignore_index=True)
    c = pd.concat([study, new], ignore_index=True)
    c.to_csv(f"{OUT}/curves.csv", index=False)
    for f in ("costs.csv", "done.txt"):
        if not os.path.exists(f"{OUT}/{f}"):
            os.symlink(f"{D}/study/{f}", f"{OUT}/{f}")
    return c


def grid():
    run.run(OUT, TRACES, f"{OUT}/grid.csv", scenarios=("pagesnat", "rss"), loads=(0.6, 0.8, 1.0), seeds=(0, 1, 2),
            margins=("cal", "cal100", 0.0), n=3000, procs=28, kind="synthetic", names=NAMES)
    print("all-done", flush=True)


if __name__ == "__main__":
    {"build": build, "run": grid}[sys.argv[1]]()
