"""Measured cost of every step the paper's methods run outside Pin (python paper_generalization/costs.py [part ...]).

Parts (default: all), each appending rows (part, workload, config, step, seconds, reps, load) to
data/costs/static.csv; one measurement at a time, in this process, median of `reps` runs:
    pb_interp     PolyBench, 27 kernels at SMALL2 and MEDIUM: clang parse; the C interpreter with symbolic
                  counting (the method), enumerating every iteration (as in the leave-one-out table's
                  times), and in timeline mode (footprint over time); AST node counts (the AST baseline's
                  descriptor); the access-idiom features (the gate)
    pb_analysis   PolyBench, per held-out kernel and split: the runtime baseline fit (NNLS on the other 26),
                  the moments at the 13 intervals (static alpha), the descriptor z-hat, the known programs'
                  measured z, distances and weights, the own model's fit, the 27 per-program models' fits,
                  the pooled regression, and applying the borrowed models (nearest, mixtures)
    gap_skeleton  the hand skeleton (GAP's six kernels, both graph families, scales 10-18) and its spectrum
    cpp           the C++ interpreter, one run at a time, on GAP's 12 serial workloads at scale 16 (a check
                  of the times recorded in the spectra, which ran 4-16 at a time; HPCCG, LULESH and NPB
                  times are those recorded in their spectra)
    csr           the C interpreter on csr_pr and csr_bfs at their traced scales
Load is the 1-minute load average when the step started (the node is shared).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "analysis"))
from memprint import lowo, similarity as sim  # noqa: E402
from memprint.model import Model, features  # noqa: E402
from memprint.static import clangast as ca, idioms, interp, load, polybench_sources  # noqa: E402
from memprint.static.spectrum import moments  # noqa: E402

OUT = ROOT / "data" / "costs" / "static.csv"
POLY = ROOT / "workloads" / "src" / "polybench"
PB = ROOT / "data" / "polybench-bytes" / "data"
CONFIGS = ["SMALL2", "MEDIUM"]


def record(part, workload, config, step, fn, reps=3, max_single=5.0):
    """Time fn (median of reps; one run if the first takes longer than max_single seconds)."""
    load_avg = os.getloadavg()[0]
    times, value = [], None
    for i in range(reps):
        t0 = time.perf_counter()
        value = fn()
        times.append(time.perf_counter() - t0)
        if times[0] > max_single:
            break
    row = pd.DataFrame([{"part": part, "workload": workload, "config": config, "step": step,
                         "seconds": float(np.median(times)), "reps": len(times), "load": load_avg}])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    return value


def pb_interp():
    src = polybench_sources(POLY)
    for k in sorted(src):
        if not (PB / "static" / f"{k}-MEDIUM.npz").exists():
            continue
        for cfg in CONFIGS:
            args = (src[k], [f"{cfg}_DATASET"], [POLY / "utilities"])
            tu = record("pb_interp", k, cfg, "clang parse", lambda: ca.parse(*args))
            record("pb_interp", k, cfg, "AST node counts",
                   lambda: pd.Series([n.kind.name for n in tu.cursor.walk_preorder()]).value_counts())
            record("pb_interp", k, cfg, "access-idiom features",
                   lambda: idioms.features([src[k]], defines=[f"{cfg}_DATASET"], includes=[POLY / "utilities"]))
            for step, symbolic, timeline in (("interpreter (symbolic)", True, False),
                                             ("interpreter (enumerated)", False, False),
                                             ("interpreter (timeline)", False, True)):
                interp.Interpreter.SYMBOLIC = symbolic
                record("pb_interp", k, cfg, step,
                       lambda: interp.run(ca.parse(*args), footprint="bytes", timeline=timeline))
            interp.Interpreter.SYMBOLIC = True


def pb_analysis():
    spectra = {tuple(f.stem.rsplit("-", 1)): load(f)[0] for f in sorted((PB / "static").glob("*.npz"))}
    data = {w: pd.read_csv(PB / f"{w}_allData.csv") for w in sorted({w for w, _ in spectra})}
    W = lowo.load(data, spectra)
    names = sorted(W)
    full = record("pb_analysis", "all", "", "fit the 27 per-program models", lambda: {w: Model.fit(W[w].rows())
                                                                                     for w in names})
    for held in names:
        C = W[held]
        known = [W[w] for w in names if w != held]
        baseline = record("pb_analysis", held, "", "runtime baseline (NNLS on 26)", lambda: lowo.fit_baseline(known))
        for split, attr in lowo.SPLITS.items():
            test = getattr(C.prepared, attr)
            if test not in C.spectra:
                continue
            rows = C.rows(test)
            alpha = rows["Alpha"].to_numpy(float)
            X = features(rows)
            record("pb_analysis", held, split, "moments at 13 intervals (static alpha)",
                   lambda: [moments(C.spectra[test], k, baseline) for k in [1] + list(sim.INTERVALS)])
            record("pb_analysis", held, split, "static alpha for the sampled bins",
                   lambda: lowo.static_alpha(C, rows, baseline))
            z_hat = record("pb_analysis", held, split, "descriptor z-hat", lambda: sim.predicted(C.spectra[test], baseline))
            z_known = record("pb_analysis", held, split, "known programs' measured z",
                             lambda: pd.DataFrame({w.name: sim.measured(w.all_data, lowo._config_like(w, test))
                                                   for w in known}).T)
            scaler = sim.Scaler(z_known)

            def nearest_and_weights():
                d = sim.distances(z_hat, z_known, scaler)
                nn = [sim.distances(z_known.loc[w], z_known.drop(index=w), scaler).min() for w in z_known.index]
                return d, sim.kernel_weights(d, float(np.median(nn)))
            d_hat, weights = record("pb_analysis", held, split, "distances and RBF weights", nearest_and_weights)
            log_pred = record("pb_analysis", held, split, "apply the 26 borrowed models",
                              lambda: pd.DataFrame({w.name: X.to_numpy() @ full[w.name].coef + full[w.name].intercept
                                                    for w in known}, index=rows.index))
            record("pb_analysis", held, split, "nearest model (pick, apply)", lambda: np.exp(log_pred[d_hat.idxmin()]))
            record("pb_analysis", held, split, "RBF mixture (combine)", lambda: lowo._mix(log_pred, weights, alpha))
            own_train = C.rows()[lambda d: d["Config"] != test]
            record("pb_analysis", held, split, "own model: fit on 6 configs and apply",
                   lambda: Model.fit(own_train).predict(rows))
            record("pb_analysis", held, split, "pooled regression with static alpha (fit and apply)",
                   lambda: lowo._pooled_static(known, baseline, rows, C, alpha))


def gap_skeleton():
    from memprint.static import gap, gap_kernels
    hand = {"pr": gap.pagerank, "bfs": gap.bfs, "cc": gap_kernels.cc, "bc": gap_kernels.bc, "tc": gap_kernels.tc,
            "sssp": gap_kernels.sssp}
    for k, f in hand.items():
        for uniform in (True, False):
            for scale in range(10, 19):
                name = f"gap_{k}" + ("" if uniform else "_kron")
                record("gap_skeleton", name, str(scale), "hand skeleton and spectrum",
                       lambda: f(scale, uniform=uniform).spectrum(), reps=1)


def cpp():
    from memprint.static import runs
    work = Path("/tmp") if not os.environ.get("COSTS_TMP") else Path(os.environ["COSTS_TMP"])
    for k in ["gap_pr", "gap_bfs", "gap_cc", "gap_bc", "gap_tc", "gap_sssp", "gap_pr_kron", "gap_bfs_kron",
              "gap_cc_kron", "gap_bc_kron", "gap_tc_kron", "gap_sssp_kron"]:
        p = work / f"costs_{k}.npz"
        record("cpp", k, "16", "C++ interpreter", lambda: runs._gap_job(k, 16, str(ROOT / "workloads/src/gapbs"), str(p)),
               reps=1)
        p.unlink(missing_ok=True)


def csr():
    for prog, scales in (("csr_pr", range(8, 17)), ("csr_bfs", range(8, 15))):
        for s in scales:
            record("csr", prog, str(s), "C interpreter",
                   lambda: interp.run(ca.parse(ROOT / "tests" / "static" / f"{prog}.c", [f"SCALE={s}"]),
                                      footprint="bytes"), reps=1)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["pb_interp", "pb_analysis", "gap_skeleton", "cpp", "csr"]
    for p in parts:
        globals()[p]()
