"""The interpreter's predictions at the corpus's largest inputs, for the zero-run prediction (zero.py).

    PYTHONPATH=analysis ~/memprint-venv/bin/python paper_rss/zero_static.py [job ...]

Jobs: the four PolyBench kernels at LARGE in timeline mode (the live footprint over the interpreter's reference
clock, as paper_generalization/timeline_static.py), and HPCCG (n = 64) and LULESH (s = 60, 20 cycles) with the C++
interpreter (the spectrum's footprint only; it has no timeline mode). GAP at scale 18 (Kronecker, one trial) and NPB
MG at W are paper_generalization's spectra (data/gap-bytes/data/static_auto, data/blind2/data/static_cpp). NPB IS and
CG at class A are not interpreted: their random-number chains run one call at a time (4 h at W, paper_generalization).
Writes ~/memprint-data/2026-10-28-rsszero/<job>.npz.
"""
import glob
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
OUT = Path(os.path.expanduser("~/memprint-data/2026-10-28-rsszero"))
POLY = ROOT / "workloads" / "src" / "polybench"
W = ROOT / "workloads" / "src"
HEAP_TOP = 59328   # glibc's top chunk at start, as paper_generalization measured for GAP


def polybench(kernel):
    from memprint.static import clangast as ca, interp, polybench_sources
    t0 = time.time()
    tu = ca.parse(polybench_sources(POLY)[kernel], ["LARGE_DATASET"], [POLY / "utilities"])
    r = interp.run(tu, footprint="bytes", timeline=True)
    u, live = r.timeline
    np.savez_compressed(OUT / f"{kernel}.npz", u=u, live=live, total=r.total, seconds=time.time() - t0)
    return f"{kernel}: peak {live.max():.0f} B, {time.time() - t0:.0f} s"


def polybench_peak(kernel):
    """The footprint alone (no timeline), when the timeline mode needs more memory than the machine has."""
    from memprint.static import analyze_polybench
    t0 = time.time()
    spec, result, errors, seconds = analyze_polybench(POLY, kernel, "LARGE", footprint="bytes")
    np.savez_compressed(OUT / f"{kernel}.npz", footprint=spec.footprint, seconds=seconds)
    return f"{kernel}: footprint {spec.footprint:.0f} B (no timeline), {seconds:.0f} s"


def cpp(name, sources, args):
    from memprint.static import runs
    t0 = time.time()
    sp, me = runs.program_spectra(name, sources, ["X"], args, heap_top=HEAP_TOP, cache_dir=OUT / "cpp", n_jobs=1)
    s = sp["X"]
    np.savez_compressed(OUT / f"{name}.npz", footprint=s.footprint, seconds=time.time() - t0, **{
        k: v for k, v in me["X"].items() if k != "seconds"})
    return f"{name}: footprint {s.footprint:.0f} B, {time.time() - t0:.0f} s"


JOBS = {
    **{k: (polybench, (k,)) for k in ("2mm", "atax", "gemm", "jacobi-2d")},
    **{k + "-peak": (polybench_peak, (k,)) for k in ("2mm", "atax", "gemm", "jacobi-2d")},
    "hpccg": (cpp, ("hpccg", sorted(glob.glob(str(W / "hpccg" / "*.cpp"))), ["64", "64", "64"])),
    "lulesh": (cpp, ("lulesh", [str(W / "lulesh" / f) for f in ("lulesh.cc", "lulesh-comm.cc", "lulesh-viz.cc",
                                                                 "lulesh-util.cc", "lulesh-init.cc")],
                     ["-q", "-s", "60", "-i", "20"])),
}


def one(name):
    f, a = JOBS[name]
    try:
        return f(*a)
    except Exception as e:  # one job must not lose the others
        return f"{name}: failed {e!r}"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(JOBS)
    if len(names) == 1:   # the C++ jobs start their own workers, which a pool's process may not
        print(one(names[0]), flush=True)
    else:
        with Pool(len(names)) as p:
            for line in p.imap_unordered(one, names):
                print(line, flush=True)
