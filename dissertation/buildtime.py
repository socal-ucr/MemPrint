"""The time MemPrint-Build takes on the 50-bin traces of the 28 workloads (B50), split into its parts, 3 repetitions,
one process, alone on the machine.

    PYTHONPATH=analysis ~/environments/memfp/bin/python dissertation/buildtime.py   ->  $O/buildtime.csv

Parts (cumulative time of the functions, from cProfile):
  preprocess   reading the splitter's CSVs and adding the spread across bins (traces.load_traces, add_sample_spread)
  selection    preparing each workload's table and choosing the intervals and subsets (dataset.prepare, the
               monotone scores) and each subset's training-best interval (model predictions and MAPE by interval)
  regression   the least-squares fits (model.Model.fit)
  build total  memprint build (cmd_build), and the wall time of preprocess + build
"""
import cProfile
import os
import pstats
import shutil
import sys
import tempfile
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from memprint import __main__ as M  # noqa: E402

B50 = os.path.expanduser("~/memprint-data/2026-11-11-bins50")
O = os.path.expanduser("~/memprint-data/2026-11-12-b50more")
PARTS = {"preprocess": [("traces.py", "load_traces"), ("traces.py", "add_sample_spread")],
         "prepare": [("dataset.py", "prepare")],
         "fit": [("model.py", "fit")], "predict": [("model.py", "predict")],
         "mape_by_interval": [("model.py", "mape_by_interval")]}


def cum(stats, file, func):
    return sum(v[3] for (f, _, fn), v in stats.stats.items() if f.endswith(file) and fn == func)


def one(rep):
    root = tempfile.mkdtemp(prefix="buildtime-")
    os.symlink(f"{B50}/traces", f"{root}/traces")
    rows = {}
    for cmd in ("preprocess", "build"):
        pr = cProfile.Profile()
        t0 = time.perf_counter()
        pr.enable()
        M.main(["--root", root, cmd])
        pr.disable()
        rows[f"{cmd}_wall"] = time.perf_counter() - t0
        st = pstats.Stats(pr)
        for part, funcs in PARTS.items():
            rows[f"{cmd}_{part}"] = sum(cum(st, f, fn) for f, fn in funcs)
    shutil.rmtree(root)
    r = dict(rep=rep, preprocess=rows["preprocess_wall"], build=rows["build_wall"],
             regression=rows["build_fit"],
             selection=rows["build_prepare"] + rows["build_predict"] + rows["build_mape_by_interval"],
             predict=rows["build_predict"], prepare=rows["build_prepare"])
    print(r, flush=True)
    return r


if __name__ == "__main__":
    import contextlib
    import io
    out = []
    for rep in (1, 2, 3):
        with contextlib.redirect_stdout(io.StringIO()):   # the build's own progress lines
            r = one(rep)
        out.append(r)
        print(r, flush=True)
    d = pd.DataFrame(out)
    d.to_csv(f"{O}/buildtime.csv", index=False)
    print(d.median().round(2).to_string())
