"""Interpreter runs of whole programs: GAP from its C++ source, and any C or C++ program given its
command line per input config.

The spectra are cached as npz (static.save format, with coverage and timing). Predictions need a
runtime baseline fitted on other workloads of the same runtime (their traces and spectra), and are
written before the program is traced, so that a prediction can be committed first and scored later
(a blind test).
"""

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import load, save
from .skeleton import HEAP_TOP_AT_START

# prc (pr until convergence) is left out: its convergence test compares floating-point values in memory,
# which the interpreter does not track, so it would run the iteration cap instead.
GAP_KERNEL = re.compile(r"^gap_(bfs|pr|cc|sssp|tc|bc)(_kron)?$")


def gap_cli(scale, uniform, degree=16):
    """Values of GAP's command-line accessors (CLApp and friends) for -u/-g <scale> -k <degree> -n 1."""
    from .interp_cpp import Str

    return {"scale": scale, "degree": degree, "uniform": int(uniform), "symmetrize": 1, "in_place": 0,
            "filename": Str(""), "num_trials": 1, "max_iters": 20, "tolerance": 0, "logging_en": 0,
            "do_analysis": 0, "do_verify": 0, "start_vertex": -1, "ParseArgs": 1, "num_iters": 1, "delta": 1}


def gap_source(kernel, gapbs):
    m = GAP_KERNEL.match(kernel)
    if m is None:
        raise ValueError(f"{kernel}: not a serial GAP workload (gap_<k>[_kron])")
    return Path(gapbs) / "src" / f"{m.group(1)}.cc", m.group(2) is None


def _gap_job(kernel, scale, gapbs, path):
    from . import interp_cpp

    source, uniform = gap_source(kernel, gapbs)
    stubs = {"__CL__": gap_cli(int(scale), uniform)}
    spec, result, errors, seconds = interp_cpp.analyze(source, heap_top=HEAP_TOP_AT_START, stubs=stubs)
    save(path, spec, result, errors, seconds)
    return f"{kernel}-{scale}: footprint {spec.footprint:.0f} B, coverage {result.coverage:.4f}, {seconds:.0f} s"


def _program_job(source, argv, defines, includes, cplusplus, heap_top, path):
    from . import analyze, interp_cpp

    if cplusplus:
        spec, result, errors, seconds = interp_cpp.analyze(source, defines, includes, heap_top=heap_top,
                                                           argv=argv)
    else:
        spec, result, errors, seconds = analyze(source, defines, includes, footprint="bytes", argv=argv)
    save(path, spec, result, errors, seconds)
    return f"{Path(path).stem}: footprint {spec.footprint:.0f} B, coverage {result.coverage:.4f}, {seconds:.0f} s"


def compute(jobs, n_jobs):
    """jobs: (function, args) with args ending in the output path; skips outputs that exist."""
    todo = [(f, a) for f, a in jobs if not Path(a[-1]).exists()]
    if not todo:
        return
    with ProcessPoolExecutor(max(1, n_jobs)) as pool:
        futures = [pool.submit(f, *a) for f, a in todo]
        for (f, a), fut in zip(todo, futures):
            try:
                print(fut.result(), flush=True)
            except Exception as e:
                print(f"{Path(a[-1]).stem}: failed ({e})", flush=True)


def gap_spectra(kernels, scales, gapbs, cache_dir, n_jobs=4):
    """Interpreter spectra of GAP workloads: {kernel: {scale: Spectrum}}."""
    cache_dir = Path(cache_dir)
    compute([(_gap_job, (k, s, str(gapbs), str(cache_dir / f"{k}-{s}.npz"))) for k in kernels for s in scales],
            n_jobs)
    return {k: {str(s): load(cache_dir / f"{k}-{s}.npz")[0] for s in scales if (cache_dir / f"{k}-{s}.npz").exists()}
            for k in kernels}


def program_spectra(name, source, configs, args, defines=(), includes=(), cplusplus=None, heap_top=0,
                    cache_dir="data/static_cpp", n_jobs=4):
    """Spectra of a program at each config. args: the command line after the program name, with
    {config} replaced by the config; defines may contain {config} too."""
    source = Path(source)
    cplusplus = source.suffix in (".cc", ".cpp", ".cxx", ".C") if cplusplus is None else cplusplus
    cache_dir = Path(cache_dir)
    jobs = []
    for c in configs:
        argv = [source.stem] + [a.replace("{config}", str(c)) for a in args]
        d = [x.replace("{config}", str(c)) for x in defines]
        jobs.append((_program_job, (str(source), argv, d, list(includes), cplusplus, heap_top,
                                    str(cache_dir / f"{name}-{c}.npz"))))
    compute(jobs, n_jobs)
    out, meta = {}, {}
    for c in configs:
        p = cache_dir / f"{name}-{c}.npz"
        if p.exists():
            out[str(c)], meta[str(c)] = load(p)
    return out, meta


def predictions(name, spectra, meta, baseline):
    """Predicted footprint and alpha at every interval, per config (written before tracing)."""
    from ..similarity import INTERVALS
    from .spectrum import moments

    rows = []
    for c, spec in spectra.items():
        m = meta.get(c, {})
        coverage = 1 - (m.get("unresolved", 0) + m.get("uncertain", 0)) / m["total"] if m.get("total") else np.nan
        for k in [1] + INTERVALS:
            bm = moments(spec, k, baseline)
            rows.append({"workload": name, "config": c, "k": k, "footprint": bm.truth, "mean_bin_footprint": bm.m,
                         "sd_bin_footprint": bm.sd, "alpha": bm.alpha, "program_footprint": spec.footprint,
                         "baseline_footprint": baseline.footprint, "coverage": coverage,
                         "seconds": m.get("seconds", np.nan)})
    return pd.DataFrame(rows)


def score(predicted, all_data):
    """Score predictions against a program's traces: footprint error and alpha MAPE per config."""
    from ..dataset import prepare
    from ..similarity import INTERVALS

    P = prepare(all_data.assign(Config=all_data["Config"].astype(str)))
    rows = []
    for c, g in predicted.groupby("config"):
        if c not in P.truth:
            continue
        truth = P.truth[c]
        d = P.data[(P.data.Config == c) & P.data.SamplingInterval.isin(INTERVALS)]
        alpha = g.set_index("k")["alpha"]
        pred = d.SamplingInterval.map(alpha).to_numpy(float)
        meas = d.Alpha.to_numpy(float)
        fp = float(g[g.k == 1]["footprint"].iloc[0])
        rows.append({"workload": g.workload.iloc[0], "config": c, "truth": truth, "predicted": fp,
                     "fp_error": (fp - truth) / truth * 100,
                     "alpha_mape": float(np.mean(np.abs(pred - meas) / meas) * 100) if len(meas) else np.nan})
    return pd.DataFrame(rows)
