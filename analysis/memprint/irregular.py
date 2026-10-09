"""GAP pilot: predict an irregular workload's footprint and alpha from a skeleton of its code and the
distribution of its input, without tracing it.

For each kernel (pr, bfs), the skeleton (static.gap) gives the access-count
spectrum at every scale; the runtime baseline (libstdc++ start-up, stack) is
fitted on the *other* kernel's traces, so nothing of the predicted kernel's
traces is used. Methods, scored on the kernel's splitter bins:

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
from .static.spectrum import Baseline, Spectrum, moments

KERNELS = {"gap_pr": gap.pagerank, "gap_bfs": gap.bfs}
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


def fit_baseline(prepared, spectra, summary):
    rows = []
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


def evaluate(all_data, spectra, polybench=None):
    """all_data / spectra: kernel -> allData table / {config: Spectrum}. polybench: name -> allData
    (its models are borrowed). Returns (errors, footprints)."""
    prepared = {k: prepare(d.assign(Config=d["Config"].astype(str))) for k, d in all_data.items()}
    summaries = {k: sim.bin_summary(d.assign(Config=d["Config"].astype(str))) for k, d in all_data.items()}
    pb_models, pb_z = {}, {}
    for name, d in (polybench or {}).items():
        p = prepare(d)
        pb_models[name] = Model.fit(_rows(p))
        pb_z[name] = sim.measured(d, p.configs[-1])
    pb_z = pd.DataFrame(pb_z).T

    errors, footprints = [], []
    for kernel in all_data:
        others = [k for k in all_data if k != kernel]
        baseline = fit_baseline(prepared[others[0]], spectra[others[0]], summaries[others[0]])
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
