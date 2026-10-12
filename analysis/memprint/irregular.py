"""GAP pilot: predict an irregular workload's footprint and alpha from a skeleton of its code and the
distribution of its input, without tracing it.

For each kernel (pr, bfs on uniform graphs; pr_kron, bfs_kron on Kronecker
graphs), the skeleton (static.gap) gives the access-count spectrum at every
scale. The runtime baseline (libstdc++ start-up, stack) is fitted on other
workloads' traces (BASELINE_FROM), so nothing of the predicted workload's
traces is used: the other uniform kernel for a uniform one, both uniform
kernels for a Kronecker one (the blind test). Methods, scored on the kernel's splitter bins:

    skeleton-fp     skeleton footprint + baseline (no sampling at all)
    skeleton-alpha  alpha from the spectrum's bin moments
    own             the kernel's own MemPrint model, trained on its other scales (needs its traces)
    pb-nearest      the PolyBench model nearest to the skeleton's z_hat
    pb-mean         all PolyBench models mixed uniformly
    pb-oracle       the PolyBench model with the lowest error, chosen after the fact
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import similarity as sim
from .dataset import prepare
from .model import Model, features
from .static import gap
from .static import gap_kernels as gk
from .static.spectrum import Baseline, Spectrum, moments

KERNELS = {
    "gap_pr": gap.pagerank,
    "gap_bfs": gap.bfs,
    "gap_pr_kron": lambda scale: gap.pagerank(scale, uniform=False),
    "gap_bfs_kron": lambda scale: gap.bfs(scale, uniform=False),
    "gap_prc": gk.pagerank_converge,
    "gap_prc_kron": lambda scale: gk.pagerank_converge(scale, uniform=False),
    "gap_cc": gk.cc,
    "gap_cc_kron": lambda scale: gk.cc(scale, uniform=False),
    "gap_sssp": gk.sssp,
    "gap_sssp_kron": lambda scale: gk.sssp(scale, uniform=False),
    "gap_tc": gk.tc,
    "gap_tc_kron": lambda scale: gk.tc(scale, uniform=False),
    "gap_bc": gk.bc,
    "gap_bc_kron": lambda scale: gk.bc(scale, uniform=False),
}
BASELINE_FROM = {
    "gap_pr": ["gap_bfs"],
    "gap_bfs": ["gap_pr"],
    "gap_pr_kron": ["gap_pr", "gap_bfs"],
    "gap_bfs_kron": ["gap_pr", "gap_bfs"],
}
# Step 4 (blind): OpenMP builds with 4 threads; the baseline (now with libgomp) comes from the
# other kernel's threaded traces.
for _k, _other in (("pr", "bfs"), ("bfs", "pr")):
    for _uniform, _suffix in ((True, ""), (False, "_kron")):
        _f = gap.pagerank if _k == "pr" else gap.bfs
        KERNELS[f"gap_{_k}{_suffix}_t4"] = (lambda f, u: lambda scale: f(scale, uniform=u, threads=4))(_f, _uniform)
        BASELINE_FROM[f"gap_{_k}{_suffix}_t4"] = [f"gap_{_other}_t4", f"gap_{_other}_kron_t4"]
# The step-2 kernels (blind): the runtime baseline comes from the four pr / bfs workloads.
for _k in ("prc", "cc", "sssp", "tc", "bc"):
    for _suffix in ("", "_kron"):
        BASELINE_FROM[f"gap_{_k}{_suffix}"] = ["gap_pr", "gap_bfs", "gap_pr_kron", "gap_bfs_kron"]
SPLITS = {"EXTRA": "last_config", "INTER": "middle_config"}


def skeleton_spectra(kernel, scales, cache_dir):
    """Spectrum of the kernel's skeleton at every scale (cached as npz)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in scales:
        path = cache_dir / f"{kernel}-{s}.npz"
        if path.exists():
            out[str(s)] = Spectrum.from_dict(np.load(path))
            continue
        spec = KERNELS[kernel](int(s)).spectrum()
        np.savez_compressed(path, **spec.to_dict())
        out[str(s)] = spec
    return out


def _rows(prepared, config=None):
    data = prepared.data[prepared.data["SamplingInterval"].isin(sim.INTERVALS)]
    return data if config is None else data[data["Config"] == config]


def fit_baseline(sources):
    """sources: (prepared, spectra, summary) of the workloads the baseline is fitted on."""
    rows = []
    for prepared, spectra, summary in sources:
        for _, r in summary.iterrows():
            spec = spectra.get(str(r["Config"]))
            if spec is not None and r["SamplingInterval"] in sim.INTERVALS:
                rows.append((spec, r["SamplingInterval"], r["m"], r["u"]))
        for config, truth in prepared.truth.items():
            if str(config) in spectra:
                rows.append((spectra[str(config)], 1, truth, truth / 4))
    return Baseline.fit(rows)


def _mape(pred, truth):
    return float(np.mean(np.abs(pred - truth) / truth) * 100)


def evaluate(all_data, spectra, polybench=None, baselines=None, kernels=None):
    """all_data / spectra: kernel -> allData table / {config: Spectrum}. polybench: name -> allData
    (its models are borrowed). baselines: kernel -> Baseline to use instead of fitting one on
    BASELINE_FROM. Returns (errors, footprints)."""
    prepared = {k: prepare(d.assign(Config=d["Config"].astype(str))) for k, d in all_data.items()}
    summaries = {k: sim.bin_summary(d.assign(Config=d["Config"].astype(str))) for k, d in all_data.items()}
    pb_models, pb_z = {}, {}
    for name, d in (polybench or {}).items():
        p = prepare(d)
        pb_models[name] = Model.fit(_rows(p))
        pb_z[name] = sim.measured(d, p.configs[-1])
    pb_z = pd.DataFrame(pb_z).T

    errors, footprints = [], []
    for kernel in kernels or all_data:
        if baselines and kernel in baselines:
            baseline = baselines[kernel]
        else:
            sources = [k for k in BASELINE_FROM[kernel] if k in all_data]
            baseline = fit_baseline([(prepared[k], spectra[k], summaries[k]) for k in sources])
        P = prepared[kernel]
        for config in P.configs:
            spec = spectra[kernel][config]
            truth = P.truth[config]
            predicted = spec.footprint + baseline.footprint
            rows = _rows(P, config)
            alpha = rows["Alpha"].to_numpy(float)
            static_alpha = rows["SamplingInterval"].map(lambda k: moments(spec, k, baseline).alpha).to_numpy()
            footprints.append({"kernel": kernel, "scale": int(config), "truth": truth, "skeleton": spec.footprint,
                               "baseline": baseline.footprint, "error": (predicted - truth) / truth * 100,
                               "alpha_mape": _mape(static_alpha, alpha)})
        for split, attr in SPLITS.items():
            test = getattr(P, attr)
            rows = _rows(P, test)
            alpha = rows["Alpha"].to_numpy(float)
            spec = spectra[kernel][test]
            res = {"skeleton-fp": abs(spec.footprint + baseline.footprint - P.truth[test]) / P.truth[test] * 100,
                   "skeleton-alpha": _mape(rows["SamplingInterval"].map(
                       lambda k: moments(spec, k, baseline).alpha).to_numpy(), alpha),
                   "own": _mape(Model.fit(_rows(P)[lambda d: d["Config"] != test]).predict(rows)
                                ["Predicted_Alpha"].to_numpy(), alpha)}
            if pb_models:
                X = features(rows).to_numpy()
                e = pd.Series({w: _mape(np.exp(X @ m.coef + m.intercept), alpha) for w, m in pb_models.items()})
                z_hat = sim.predicted(spec, baseline)
                d = sim.distances(z_hat, pb_z)
                res.update({"pb-nearest": float(e[d.idxmin()]), "pb-mean": _mape(
                    np.exp(np.mean([X @ m.coef + m.intercept for m in pb_models.values()], axis=0)), alpha),
                    "pb-oracle": float(e.min())})
            for method, value in res.items():
                errors.append({"kernel": kernel, "split": split, "config": test, "method": method, "mape": value})
    return pd.DataFrame(errors), pd.DataFrame(footprints)
