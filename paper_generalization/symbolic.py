"""Symbolic counting of affine innermost loops against enumeration (python paper_generalization/symbolic.py).

For every PolyBench kernel and config: the C interpreter with and without the symbolic path (SYMBOLIC), its
time, and whether the two spectra are identical. Then, for larger inputs (LARGE, EXTRALARGE), the time of
both (enumeration with a time limit). Writes data/symbolic_check.csv and data/symbolic_scaling.csv.
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "analysis"))
POLY = ROOT / "workloads" / "src" / "polybench"
CONFIGS = ["MINI", "MINI2", "MINI3", "SMALL", "SMALL2", "SMALL3", "MEDIUM"]


def run(kernel, config, symbolic):
    from memprint.static import analyze_polybench, interp
    interp.Interpreter.SYMBOLIC = symbolic
    t = time.time()
    spec, r, _, _ = analyze_polybench(POLY, kernel, config, footprint="bytes")
    return {"workload": kernel, "config": config, "symbolic": symbolic, "seconds": time.time() - t,
            "footprint": spec.footprint, "references": spec.references,
            "checksum": float(np.sum(spec.n * spec.s * np.log1p(spec.c))), "symbolic_loops": r and 0}


def job(args):
    try:
        return run(*args)
    except Exception as e:                                             # a config the kernel does not define
        return {"workload": args[0], "config": args[1], "symbolic": args[2], "error": str(e)[:80]}


def main():
    kernels = [k for k in (ROOT / "analysis" / "workloads.txt").read_text().split() if k != "miniVite"]
    jobs = [(k, c, s) for k in kernels for c in CONFIGS for s in (False, True)]
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(job, jobs))
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "data" / "symbolic_check.csv", index=False)
    a = d[~d.symbolic].set_index(["workload", "config"])
    b = d[d.symbolic].set_index(["workload", "config"])
    same = (a.footprint == b.footprint) & (np.abs(a.references - b.references) <= 1e-9 * a.references) & \
        (np.abs(a.checksum - b.checksum) <= 1e-9 * np.abs(a.checksum))
    print(f"identical spectra: {int(same.sum())} of {len(same)}")
    print("speed-up at MEDIUM: median", round(float((a.seconds / b.seconds).xs("MEDIUM", level="config").median()), 1),
          "max", round(float((a.seconds / b.seconds).xs("MEDIUM", level="config").max()), 1))




SCALING_KERNELS = ["gemm", "2mm", "syrk", "covariance", "jacobi-2d", "heat-3d", "cholesky", "lu", "atax", "trmm"]


def scaling(limit=900):
    """Time of both modes at MEDIUM, LARGE and EXTRALARGE (each run in its own process, killed at `limit` s)."""
    import subprocess
    rows = []
    for k in SCALING_KERNELS:
        for c in ("MEDIUM", "LARGE", "EXTRALARGE"):
            for s in (True, False):
                code = (f"import sys; sys.path.insert(0, {str(ROOT / 'analysis')!r}); sys.path.insert(0, {str(HERE)!r});"
                        f"import symbolic, json; print(json.dumps(symbolic.run({k!r}, {c!r}, {s})))")
                try:
                    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=limit)
                    import json
                    line = [x for x in out.stdout.splitlines() if x.startswith("{")]
                    rows.append(json.loads(line[-1]) if line else {"workload": k, "config": c, "symbolic": s,
                                                                   "error": out.stderr[-120:]})
                except subprocess.TimeoutExpired:
                    rows.append({"workload": k, "config": c, "symbolic": s, "seconds": float("nan"), "timeout": limit})
                print(rows[-1].get("workload"), c, s, rows[-1].get("seconds"), flush=True)
    pd.DataFrame(rows).to_csv(HERE / "data" / "symbolic_scaling.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "scaling":
        scaling()
    else:
        main()
