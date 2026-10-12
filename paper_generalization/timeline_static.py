"""The footprint over time predicted from source, against Pin's exact live-footprint timelines
(python paper_generalization/timeline_static.py <dir with traces/<kernel>/Buffered_<kernel>-<config>_1_*_timeline.csv>).

For each kernel with a truth timeline (2mm, atax, gemm, jacobi-2d at seven sizes) the interpreter runs in
timeline mode and gives the program's live footprint against its own reference clock. Pin's clock also
counts the runtime's start-up and the -O0 stack traffic the interpreter does not charge, so the prediction is
    truth(t) ~ runtime(t) + program((t - R_s) / s)
with R_s the start-up references and s Pin's references per interpreter reference: the medians of per-kernel
fits (Pin's total references against the interpreter's, over the seven sizes) of the other 26 kernels. The
runtime's footprint is the runtime baseline B (the NNLS baseline of the static footprint, fitted on the other
26 kernels) once the program has started, and during start-up B times the shape of the other timeline
kernels' start-up ramps (their truth before their programs start, over their final runtime-only footprint).
Nothing of the predicted kernel's traces is used.
Writes data/timeline_static.csv (errors) and data/timeline_static_curves.csv (curves).
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "analysis"))
from memprint.static import clangast as ca, interp, polybench_sources  # noqa: E402

POLY = ROOT / "workloads" / "src" / "polybench"
TRACES = Path(sys.argv[1]) if len(sys.argv) > 1 else None
KERNELS = ["2mm", "atax", "gemm", "jacobi-2d"]
CONFIGS = ["MINI", "MINI2", "MINI3", "SMALL", "SMALL2", "SMALL3", "MEDIUM"]
CACHE = HERE / "data" / "timeline_program"


def program_curve(kernel, config):
    """The interpreter's live footprint of the program against its reference clock (cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{kernel}-{config}.npz"
    if path.exists():
        d = np.load(path)
        return d["t"], d["live"], float(d["total"])
    tu = ca.parse(polybench_sources(POLY)[kernel], [f"{config}_DATASET"], [POLY / "utilities"])
    r = interp.run(tu, footprint="bytes", timeline=True)
    u, live = r.timeline
    np.savez_compressed(path, t=u * r.total, live=live, total=r.total)
    return u * r.total, live, float(r.total)


def truth(kernel, config):
    f = glob.glob(str(TRACES / kernel / f"Buffered_{kernel}-{config}_1_*_timeline.csv"))[0]
    t = pd.read_csv(f)
    t = t[t.SamplingInterval == 1].sort_values("Time")
    return t.Time.to_numpy(float), t.MemUsageObs.to_numpy(float)


def clock_fits():
    """Per kernel: Pin's total references = R_s + s * interpreter's, over the seven sizes (least squares)."""
    m = pd.read_csv(HERE / "data" / "pb_static_meta.csv")
    fits = {}
    for w, d in m.groupby("workload"):
        A = np.vstack([np.ones(len(d)), d.total]).T
        (r0, s), *_ = np.linalg.lstsq(A, d.pin_references.to_numpy(float), rcond=None)
        fits[w] = (r0, s)
    return fits


def baselines():
    """The static footprint's runtime baseline for each kernel, fitted on the other 26 (as lowo does)."""
    from memprint import lowo
    from memprint.static import load
    pb = ROOT / "data" / "polybench-bytes" / "data"
    spectra = {tuple(f.stem.rsplit("-", 1)): load(f)[0] for f in sorted((pb / "static").glob("*.npz"))}
    data = {w: pd.read_csv(pb / f"{w}_allData.csv") for w in sorted({w for w, _ in spectra})}
    known = lowo.load(data, spectra)
    return {k: lowo.fit_baseline([known[w] for w in known if w != k]).footprint for k in KERNELS}


def main():
    fits = clock_fits()
    base = baselines()
    progs = {(k, c): program_curve(k, c) for k in KERNELS for c in CONFIGS}
    truths = {(k, c): truth(k, c) for k in KERNELS for c in CONFIGS}
    rows, curves = [], []
    for k in KERNELS:
        others = [w for w in fits if w != k]
        r_s = float(np.median([fits[w][0] for w in others]))
        s = float(np.median([fits[w][1] for w in others]))
        # the runtime's start-up ramp: the shape of the other timeline kernels' footprint before their programs
        # start (truth over their final, runtime-only footprint), on start-up time normalised by their R_s
        grid = np.linspace(0, 1, 200)
        shapes = []
        for w in KERNELS:
            if w == k:
                continue
            for c in CONFIGS:
                tt, ff = truths[(w, c)]
                if tt[0] < fits[w][0]:
                    shapes.append(np.interp(grid * fits[w][0], tt, ff) / ff[-1])
        ramp = np.minimum(np.median(np.array(shapes), axis=0), 1.0) * base[k]
        for c in CONFIGS:
            t_prog, live, total = progs[(k, c)]
            tt, ff = truths[(k, c)]
            runtime = np.where(tt < r_s, np.interp(tt / r_s, grid, ramp), base[k])
            pred = runtime + np.interp((tt - r_s) / s, t_prog, live, left=0.0, right=live[-1])
            err = np.abs(pred - ff) / ff * 100
            # the same on normalised time (each curve over its own length): the shape, without the run-length error
            scale = (r_s + s * total) / tt[-1]
            ts = tt * scale
            runtime_n = np.where(ts < r_s, np.interp(ts / r_s, grid, ramp), base[k])
            pred_n = runtime_n + np.interp((ts - r_s) / s, t_prog, live, left=0.0, right=live[-1])
            err_n = np.abs(pred_n - ff) / ff * 100
            peak_t, peak_p = ff.max(), pred.max()
            rows.append({"workload": k, "config": c, "snapshots": len(tt), "mean_error": float(err.mean()),
                         "median_error": float(np.median(err)), "p90_error": float(np.percentile(err, 90)),
                         "mean_error_normalised": float(err_n.mean()), "median_error_normalised": float(np.median(err_n)),
                         "peak_truth": peak_t, "peak_pred": peak_p, "peak_error": (peak_p - peak_t) / peak_t * 100,
                         "end_truth": tt[-1], "end_pred": r_s + s * total,
                         "length_error": (r_s + s * total - tt[-1]) / tt[-1] * 100})
            if c in ("MINI", "SMALL", "MEDIUM"):
                curves += [{"workload": k, "config": c, "time": a, "truth": b, "pred": p, "pred_normalised": q}
                           for a, b, p, q in zip(tt, ff, pred, pred_n)]
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "data" / "timeline_static.csv", index=False)
    pd.DataFrame(curves).to_csv(HERE / "data" / "timeline_static_curves.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(2).to_string(index=False))
    print("normalised time: mean error median", round(out.mean_error_normalised.median(), 2), "max",
          round(out.mean_error_normalised.max(), 2))
    print("mean error over all: median", round(out.mean_error.median(), 2), "max", round(out.mean_error.max(), 2),
          "| peak error: median", round(out.peak_error.abs().median(), 2), "max", round(out.peak_error.abs().max(), 2))


if __name__ == "__main__":
    main()
