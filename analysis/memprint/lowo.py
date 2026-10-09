"""Leave one workload out: predict an unseen workload's alpha from its source code
and the models of the known workloads.

For each held-out workload C (only its source is used, except for scoring) and
split (EXTRA: largest config, INTER: middle config), the methods predict alpha
for C's bins at the test config, at every interval in similarity.INTERVALS:

    own            C's own MemPrint model, trained on C's other configs (needs C's traces; reference)
    static-fp      static footprint + runtime baseline (no sampling at all)
    static-alpha   alpha from the static spectrum's bin moments, times the observed bin footprint
    pooled-static  one regression over the known workloads: the seven model features + log static alpha
    nn-static      the known model nearest to C in z_hat (z predicted from C's source)
    mix-static     known models mixed with RBF weights on the z_hat distance
    nn-measured    as nn-static with C's measured z (upper bound for z_hat)
    mix-measured   as mix-static with C's measured z
    nn-ast         nearest known model by clang AST node counts (baseline similarity)
    mean           all known models mixed uniformly (no similarity)
    oracle         the known model with the lowest error on C (chosen after the fact)

The runtime baseline (memory outside the program text) is fitted on the known
workloads only. Known workloads' models are fitted on all their configs.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import similarity as sim
from .dataset import prepare
from .model import Model, features
from .static.spectrum import Baseline, moments

SPLITS = {"EXTRA": "last_config", "INTER": "middle_config"}


@dataclass
class Workload:
    name: str
    prepared: object  # dataset.Prepared
    summary: pd.DataFrame  # similarity.bin_summary
    spectra: dict  # config -> Spectrum
    all_data: pd.DataFrame

    def rows(self, config=None):
        data = self.prepared.data
        data = data[data["SamplingInterval"].isin(sim.INTERVALS) & data["Config"].isin(self.spectra)]
        return data if config is None else data[data["Config"] == config]


def load(all_data_by_workload, spectra):
    """spectra: (workload, config) -> Spectrum. Workloads without spectra are left out."""
    out = {}
    for w, df in all_data_by_workload.items():
        specs = {c: s for (ww, c), s in spectra.items() if ww == w and s.footprint > 1000}
        if not specs:
            continue
        df = df.assign(Config=df["Config"].astype(str).str.split("-").str[-1])  # fdtd-2d traces say 2d-MINI
        out[w] = Workload(w, prepare(df), sim.bin_summary(df), specs, df)
    return out


def fit_baseline(workloads):
    rows = []
    for wl in workloads:
        for _, r in wl.summary.iterrows():
            spec = wl.spectra.get(r["Config"])
            if spec is not None and r["SamplingInterval"] in sim.INTERVALS:
                rows.append((spec, r["SamplingInterval"], r["m"], r["u"]))
        for config, spec in wl.spectra.items():
            if config in wl.prepared.truth:
                truth_u = wl.all_data[(wl.all_data["SamplingInterval"] == 1) & (wl.all_data["Config"] == config)]
                rows.append((spec, 1, wl.prepared.truth[config], float(truth_u["UniqueAddresses"].min())))
    return Baseline.fit(rows)


def static_alpha(wl, rows, baseline):
    """Static alpha for each row (its config and interval)."""
    cache = {}
    out = np.empty(len(rows))
    for i, (config, k) in enumerate(zip(rows["Config"], rows["SamplingInterval"])):
        if (config, k) not in cache:
            cache[(config, k)] = moments(wl.spectra[config], k, baseline).alpha
        out[i] = cache[(config, k)]
    return out


def _mape(pred, truth):
    return float(np.mean(np.abs(pred - truth) / truth) * 100)


def _config_like(wl, config):
    """The config of wl with the same name, else the one whose true footprint rank matches."""
    if config in wl.spectra and config in wl.prepared.truth:
        return config
    return wl.prepared.configs[-1]


def evaluate(workloads, ast=None, splits=SPLITS):
    """Run the leave-one-out evaluation. Returns (errors, validity, descriptors) tables."""
    names = sorted(workloads)
    full_models = {w: Model.fit(workloads[w].rows()) for w in names}
    errors, validity, descriptors = [], [], []
    for held in names:
        C = workloads[held]
        known = [workloads[w] for w in names if w != held]
        baseline = fit_baseline(known)
        for split, attr in splits.items():
            test = getattr(C.prepared, attr)
            if test not in C.spectra:
                continue
            rows = C.rows(test)
            if rows.empty:
                continue
            alpha = rows["Alpha"].to_numpy(float)
            X = features(rows)
            log_pred = pd.DataFrame({w.name: X.to_numpy() @ full_models[w.name].coef + full_models[w.name].intercept
                                     for w in known}, index=rows.index)
            e_row = log_pred.apply(lambda col: _mape(np.exp(col.to_numpy()), alpha))

            z_known = pd.DataFrame({w.name: sim.measured(w.all_data, _config_like(w, test)) for w in known}).T
            z_hat = sim.predicted(C.spectra[test], baseline)
            z_meas = sim.measured(C.all_data, test)
            scaler = sim.Scaler(z_known)
            d_hat = sim.distances(z_hat, z_known, scaler)
            d_meas = sim.distances(z_meas, z_known, scaler)
            nn_known = [sim.distances(z_known.loc[w], z_known.drop(index=w), scaler).min() for w in z_known.index]
            bandwidth = float(np.median(nn_known))

            res = {}
            own_train = C.rows()[lambda d: d["Config"] != test]
            res["own"] = _mape(Model.fit(own_train).predict(rows)["Predicted_Alpha"].to_numpy(), alpha)
            truth = C.prepared.truth[test]
            res["static-fp"] = abs(C.spectra[test].footprint + baseline.footprint - truth) / truth * 100
            res["static-alpha"] = _mape(static_alpha(C, rows, baseline), alpha)
            res["pooled-static"] = _pooled_static(known, baseline, rows, C, alpha)
            res["nn-static"] = float(e_row[d_hat.idxmin()])
            res["mix-static"] = _mix(log_pred, sim.kernel_weights(d_hat, bandwidth), alpha)
            res["nn-measured"] = float(e_row[d_meas.idxmin()])
            res["mix-measured"] = _mix(log_pred, sim.kernel_weights(d_meas, bandwidth), alpha)
            res["mean"] = _mix(log_pred, pd.Series(1.0 / len(known), index=log_pred.columns), alpha)
            res["oracle"] = float(e_row.min())
            d_ast = None
            if ast is not None and held in ast.index:
                common = [w for w in log_pred.columns if w in ast.index]
                d_ast = sim.distances(ast.loc[held], ast.loc[common])
                res["nn-ast"] = float(e_row[d_ast.idxmin()])
            for method, value in res.items():
                errors.append({"workload": held, "split": split, "config": test, "method": method, "mape": value})

            def rho(d):
                if d is None:
                    return np.nan
                common = [w for w in d.index if w in e_row.index]
                return spearmanr(d[common], e_row[common])[0]

            reuse = [c for c in z_hat.index if c.startswith("reuse")]
            validity.append({"workload": held, "split": split, "rho_static": rho(d_hat), "rho_measured": rho(d_meas),
                             "rho_ast": rho(d_ast),
                             "zhat_reuse_rmse": float(np.sqrt(np.nanmean((z_hat[reuse] - z_meas[reuse]) ** 2))),
                             "nn_static": d_hat.idxmin(), "nn_measured": d_meas.idxmin(), "oracle": e_row.idxmin(),
                             "min_distance": float(d_hat.min()), "bandwidth": bandwidth})
            descriptors.append(pd.concat([z_hat.add_prefix("hat_"), z_meas.add_prefix("meas_")])
                               .rename(f"{held}|{split}"))
    return pd.DataFrame(errors), pd.DataFrame(validity), pd.DataFrame(descriptors)


def transfer_check(name, all_data, workloads):
    """How the known workloads' models and descriptors fare on a workload with no static spectrum
    (miniVite): MAPE of each known model on its largest config, and the distance of its measured z
    from the known workloads relative to their own nearest-neighbour distances."""
    prepared = prepare(all_data)
    test = prepared.configs[-1]
    data = prepared.data
    rows = data[(data["Config"] == test) & data["SamplingInterval"].isin(sim.INTERVALS)]
    alpha = rows["Alpha"].to_numpy(float)
    X = features(rows).to_numpy()
    e_row = {}
    for w, wl in sorted(workloads.items()):
        model = Model.fit(wl.rows())
        e_row[w] = _mape(np.exp(X @ model.coef + model.intercept), alpha)
    e_row = pd.Series(e_row)
    z_known = pd.DataFrame({w: sim.measured(wl.all_data, wl.prepared.configs[-1]) for w, wl in workloads.items()}).T
    scaler = sim.Scaler(z_known)
    d = sim.distances(sim.measured(all_data, test), z_known, scaler)
    nn_known = [sim.distances(z_known.loc[w], z_known.drop(index=w), scaler).min() for w in z_known.index]
    own = Model.fit(data[(data["Config"] != test) & data["SamplingInterval"].isin(sim.INTERVALS)])
    return {"workload": name, "config": test, "own": _mape(own.predict(rows)["Predicted_Alpha"].to_numpy(), alpha),
            "nn-measured": float(e_row[d.idxmin()]), "nearest": d.idxmin(), "oracle": float(e_row.min()),
            "median_known_model": float(e_row.median()),
            "min_distance": float(d.min()), "known_nn_distance_median": float(np.median(nn_known)),
            "known_nn_distance_max": float(np.max(nn_known))}


def _mix(log_pred, weights, alpha):
    w = weights.reindex(log_pred.columns).fillna(0.0)
    return _mape(np.exp(log_pred.to_numpy() @ w.to_numpy()), alpha)


def _pooled_static(known, baseline, rows, C, alpha):
    """log alpha = seven model features + log static alpha, fitted on the known workloads."""
    from sklearn.linear_model import LinearRegression

    Xs, ys = [], []
    for wl in known:
        r = wl.rows()
        X = features(r)
        X["log_static"] = np.log(static_alpha(wl, r, baseline))
        Xs.append(X)
        ys.append(np.log(r["Alpha"].to_numpy(float)))
    reg = LinearRegression().fit(pd.concat(Xs), np.concatenate(ys))
    X = features(rows)
    X["log_static"] = np.log(static_alpha(C, rows, baseline))
    return _mape(np.exp(reg.predict(X)), alpha)


def summarize(errors):
    """Median and mean MAPE per split and method."""
    table = errors.groupby(["split", "method"])["mape"].agg(["median", "mean"]).unstack("split")
    order = ["own", "static-fp", "static-alpha", "pooled-static", "nn-static", "mix-static", "nn-measured",
             "mix-measured", "nn-ast", "mean", "oracle"]
    return table.reindex([m for m in order if m in table.index])
