"""Memory footprint over time.

Timelines (memprint_trace -snapshot) give, every N memory references, the
exact live footprint (splitter main row) and the footprint observed in every
subsample bin. From them:

- reconstruct: estimate the true footprint at every snapshot of a sparse run
  with the MemPrint model, alpha(sigma across bins, mean observed footprint,
  sampling interval [, time]) x mean observed footprint (Eq. 8 per snapshot),
  or with Chao1/iChao1 on the union of the bins: the addresses sampled
  exactly once, twice, ... estimate how many were never sampled; or with the
  known-rate estimator (richness.py), which also uses the sampling rate;
- forecast: from the beginning of a run, predict the rest of its footprint
  curve by fitting time- and magnitude-scaled versions of the training
  configs' curves.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .build import SPLITS
from .dataset import EXCLUDED_INTERVALS, prepare
from .model import Model, mape_by_interval
from .richness import known_rate_bytes

SUBSETS = ("NZ", "MT", "L2O")
# Extra log features of the model. ReuseObs (sampled references per observed
# address) tells a plateau, where bins keep re-sampling the same addresses,
# apart from growth.
VARIANTS = {"base": (), "time": ("Time",), "reuse": ("ReuseObs",)}
PREFIXES = (0.1, 0.25, 0.5, 0.75)


# ------------------------------------------------------------------ data


def truth_curves(timeline):
    """Exact live footprint over time for each config (first splitter run)."""
    main = timeline[(timeline["Kind"] == "splitter") & (timeline["Bin"] == -1)]
    main = main[main["PID"] == main.groupby("Config")["PID"].transform("first")]
    return (
        main[["Config", "Time", "MemUsageObs"]]
        .rename(columns={"MemUsageObs": "Truth"})
        .sort_values(["Config", "Time"])
        .reset_index(drop=True)
    )


def bin_rows(timeline, kind):
    """Bin rows of one kind of run, with the spread (SD) and mean of the
    observed footprint across the bins of the same snapshot."""
    rows = timeline[(timeline["Kind"] == kind) & (timeline["Bin"] >= 0)].copy()
    group = rows.groupby(["Config", "PID", "Time", "SamplingInterval"])["MemUsageObs"]
    rows["SD_MemUsage"] = group.transform("std")
    rows["MeanObs"] = group.transform("mean")
    return rows


def training_rows(timeline):
    """Splitter bin rows at every snapshot with Alpha = true / observed footprint."""
    rows = bin_rows(timeline, "splitter").merge(truth_curves(timeline), on=["Config", "Time"])
    rows = rows[
        ~rows["SamplingInterval"].isin(EXCLUDED_INTERVALS) & (rows["MemUsageObs"] > 0) & (rows["SD_MemUsage"] > 0)
    ].copy()
    rows["Alpha"] = rows["Truth"] / rows["MemUsageObs"]
    rows["ReuseObs"] = rows["CountObs"] / rows["UniqueAddresses"]
    return rows


def peak_snapshot(timeline):
    """The splitter rows at each config's peak footprint, in allData form, so
    the training subsets are chosen exactly as for whole-run models."""
    truth = truth_curves(timeline)
    peaks = truth.loc[truth.groupby("Config")["Truth"].idxmax(), ["Config", "Time"]]
    rows = timeline[timeline["Kind"] == "splitter"].merge(peaks, on=["Config", "Time"])
    rows = rows[rows["PID"] == rows.groupby("Config")["PID"].transform("first")].copy()
    spread = (
        rows[rows["Bin"] >= 0]
        .groupby(["Config", "SamplingInterval"])
        .agg(SD_Unique=("UniqueAddresses", "std"), SD_MemUsage=("MemUsageObs", "std"), SD_CountObs=("CountObs", "std"))
        .reset_index()
    )
    rows = rows.merge(spread, on=["Config", "SamplingInterval"], how="left").assign(FunctionName="Total")
    return rows[["FunctionName", "MemUsageObs", "UniqueAddresses", "CountObs", "SamplingInterval", "PID", "Config",
                 "SD_Unique", "SD_MemUsage", "SD_CountObs"]]


# ------------------------------------------------------------ reconstruct


def estimate_curve(bins, model):
    """Estimated true footprint at each snapshot of one run, from its bins at
    one sampling interval: alpha x mean observed footprint (Eq. 8)."""
    snapshots = (
        bins.groupby("Time")
        .agg(MemUsageObs=("MemUsageObs", "mean"), SD_MemUsage=("MemUsageObs", "std"),
             CountObs=("CountObs", "mean"), UniqueAddresses=("UniqueAddresses", "mean"),
             SamplingInterval=("SamplingInterval", "first"))
        .reset_index()
    )
    snapshots = snapshots[(snapshots["MemUsageObs"] > 0) & (snapshots["SD_MemUsage"] > 0)]
    snapshots["ReuseObs"] = snapshots["CountObs"] / snapshots["UniqueAddresses"]
    predicted = model.predict(snapshots)
    predicted["Estimate"] = predicted["Predicted_Alpha"] * predicted["MemUsageObs"]
    return predicted[["Time", "SamplingInterval", "MemUsageObs", "SD_MemUsage", "Estimate"]]


def curve_error(curve, truth, value="Estimate", run_end=None):
    """MAPE (%) of a curve against the truth (interpolated at the curve's
    times), and the relative error (%) of its peak.

    run_end: length of the run the curve comes from, if it is not the run the
    truth comes from. Times are then compared as fractions of each run: the
    number of references a multithreaded program executes varies between runs
    (threads spin while they wait), so absolute times do not line up.
    """
    times = curve["Time"] if run_end is None else curve["Time"] / run_end * truth["Time"].max()
    true_at = np.interp(times, truth["Time"], truth["Truth"])
    ok = true_at > 0
    mape = float(np.mean(np.abs(curve[value][ok] - true_at[ok]) / true_at[ok]) * 100) if ok.any() else np.nan
    peak = (curve[value].max() - truth["Truth"].max()) / truth["Truth"].max() * 100 if len(curve) else np.nan
    return mape, float(peak)


def sampler_runs(sampler_bins, config):
    """{bin interval: bin rows} for the sampler runs of one config."""
    rows = sampler_bins[sampler_bins["Config"] == config]
    return {si: group for si, group in rows.groupby("SamplingInterval")}


def closest(options, target):
    return min(options, key=lambda si: abs(np.log(si) - np.log(target))) if options else None


@dataclass
class Reconstruction:
    split: str
    subset: str
    variant: str
    model: Model
    train_si: int
    train_mape: float


def fit_reconstruction(rows, train_configs, intervals, variant):
    """Fit alpha on every snapshot (training_rows) of the training configs;
    return the model and the interval with the lowest training MAPE."""
    train = rows[rows["Config"].isin(train_configs) & rows["SamplingInterval"].isin(intervals)]
    model = Model.fit(train, extra=VARIANTS[variant])
    by_si = mape_by_interval(model.predict(train))
    return model, int(by_si.idxmin()), float(by_si.min())


# ------------------------------------------------------------------ Chao1

ESTIMATORS = ("Chao1", "iChao1", "KnownRate")


def union_rows(timeline, kind):
    """Union-of-bins rows with species-richness estimates of the number of
    addresses: the splitter's per-interval unions (Bin -2) or the sampler's
    union (Bin -1). f_k = addresses sampled exactly k times.

    Chao1 (bias-corrected):  S + f1 (f1 - 1) / (2 (f2 + 1))
    iChao1 (Chiu et al. 2014): Chao1 + f3 / (4 f4) * max(f1 - f2 f3 / (2 f4), 0)
    iChao1 also uses f3, f4 and corrects Chao1's underestimate when reuse
    differs across addresses. Both are converted to bytes with the union's
    bytes per address.
    """
    if kind == "splitter":
        rows = timeline[(timeline["Kind"] == "splitter") & (timeline["Bin"] == -2)]
    else:
        rows = timeline[(timeline["Kind"] == "sampler") & (timeline["Bin"] == -1)]
    rows = rows[rows["UniqueAddresses"] > 0].copy()
    seen, f1, f2 = rows["UniqueAddresses"], rows["Singletons"], rows["Doubletons"]
    chao = seen + f1 * (f1 - 1) / (2 * (f2 + 1))
    per_address = rows["MemUsageObs"] / seen
    rows["Chao1"] = chao * per_address
    if "Quadrupletons" in rows:
        f3, f4 = rows["Tripletons"], rows["Quadrupletons"].clip(lower=1)
        rows["iChao1"] = (chao + f3 / (4 * f4) * np.maximum(f1 - f2 * f3 / (2 * f4), 0)) * per_address
    return rows


def add_known_rate(rows):
    """Known-rate estimates are expensive; compute them only for the rows used."""
    if "Quadrupletons" in rows and "KnownRate" not in rows:
        rows = rows.copy()
        rows["KnownRate"] = known_rate_bytes(rows)
    return rows


def chao_curve(rows, estimator):
    curve = rows[["Time", "SamplingInterval"]].copy()
    curve["Estimate"] = rows[estimator]
    return curve


def truth_frame(truths):
    """{config: truth curve} -> one table (Config, Time, Truth)."""
    return pd.concat([t.assign(Config=c) for c, t in truths.items()])[["Config", "Time", "Truth"]]


def evaluate_chao(timeline, truths, train_configs, test_config, workload, split):
    """Chao1/iChao1 reconstruction of the held-out config, from the
    splitter's union rows (interval with the lowest MAPE on the training
    configs) and from the sampler run with the closest sampling interval."""
    unions = union_rows(timeline, "splitter").merge(truth_frame(truths), on=["Config", "Time"])
    unions = unions[unions["Truth"] > 0]
    train = unions[unions["Config"].isin(train_configs)]
    # every 10th training snapshot is enough to pick the interval
    train = add_known_rate(train[train.groupby(["Config", "SamplingInterval"]).cumcount() % 10 == 0])
    samplers = union_rows(timeline, "sampler")
    samplers = samplers[samplers["Config"] == test_config]
    if "Quadrupletons" in unions:
        unions["KnownRate"] = np.nan
    truth = truths[test_config]
    rows, curves = [], []
    for estimator in [e for e in ESTIMATORS if e in train]:
        labels = dict(workload=workload, split=split, config=test_config, subset=estimator, variant="raw")
        per_si = (np.abs(train[estimator] - train["Truth"]) / train["Truth"]).groupby(train["SamplingInterval"]).mean() * 100
        si = int(per_si.idxmin())
        row = {**{k: labels[k] for k in ("workload", "split", "config", "subset", "variant")},
               "train_si": si, "train_mape": float(per_si.min())}
        test = unions[(unions["Config"] == test_config) & (unions["SamplingInterval"] == si)]
        if estimator == "KnownRate":
            test = add_known_rate(test.drop(columns="KnownRate"))
        curve = chao_curve(test, estimator)
        row["splitter_mape"], row["splitter_peak_error"] = curve_error(curve, truth)
        curves.append(curve.assign(source="splitter bins", **labels))
        if len(samplers):
            # union of a sampler run: a 1-in-(i / (1 - e^-lambda)) sample; take the run closest to si
            run_si = closest(sorted(samplers["RunInterval"].unique()), si)
            run = samplers[samplers["RunInterval"] == run_si]
            run = run[run["PID"] == run["PID"].iloc[0]]
            if estimator == "KnownRate":
                run = add_known_rate(run)
            run_end = float(run["Time"].max())
            curve = chao_curve(run, estimator)
            row["sampler_si"] = run_si
            row["sampler_mape"], row["sampler_peak_error"] = curve_error(curve, truth, run_end=run_end)
            row["sampler_length_error"] = (run_end - truth["Time"].max()) / truth["Time"].max() * 100
            row["sampler_coverage"] = 1.0
            curves.append(curve.assign(Time=curve["Time"] / run_end * truth["Time"].max(), source="sampler", **labels))
        rows.append(row)
    return rows, curves


# --------------------------------------------------------------- forecast


@dataclass
class Forecast:
    curve: pd.DataFrame  # Time, Forecast (after the prefix)
    end_time: float
    final: float
    peak: float

    def at(self, times):
        """Forecast at the given times (the final value after the predicted end)."""
        values = np.interp(times, self.curve["Time"], self.curve["Forecast"]) if len(self.curve) else np.full(len(times), self.final)
        return np.where(np.asarray(times) > self.end_time, self.final, values)


def fit_template(prefix, template, length_law, base=0.0, prior_weight=0.1):
    """Scale a template curve as base + a * (F(t / b) - base) to match a
    prefix, in log space. `base` is the runtime's own footprint, which does
    not grow with the input.

    Returns (residual, a, b). The time scale b is regularised by the
    length-vs-peak power law of the training configs (length_law), since a
    flat prefix says little about how long the run will last.
    """
    t, f = prefix["Time"].to_numpy(float), prefix["Truth"].to_numpy(float) - base
    tt, tf = template["Time"].to_numpy(float), template["Truth"].to_numpy(float) - base
    keep = f > 0
    t, f = t[keep], f[keep]
    if len(t) == 0:
        return np.inf, 1.0, 1.0
    b_min = t.max() / tt.max()  # the prefix must fit inside the stretched template
    best = (np.inf, 1.0, 1.0)
    for b in b_min * np.exp(np.linspace(0, np.log(1e4), 400)):
        g = np.interp(t / b, tt, tf)
        ok = g > 0
        if ok.sum() < max(2, len(t) // 2):
            continue
        log_ratio = np.log(f[ok]) - np.log(g[ok])
        log_a = log_ratio.mean()
        residual = np.mean((log_ratio - log_a) ** 2)
        slope, intercept = length_law
        expected_log_length = slope * (log_a + np.log(tf.max())) + intercept
        residual += prior_weight * (np.log(b * tt.max()) - expected_log_length) ** 2
        if residual < best[0]:
            best = (residual, float(np.exp(log_a)), float(b))
    return best


def length_law(templates, base=0.0):
    """Least-squares fit of log(run length) = slope * log(peak - base) + intercept."""
    peaks = np.log([max(tpl["Truth"].max() - base, 1.0) for tpl in templates])
    lengths = np.log([tpl["Time"].max() for tpl in templates])
    if len(templates) < 2 or np.ptp(peaks) == 0:
        return 0.0, float(np.mean(lengths))
    slope, intercept = np.polyfit(peaks, lengths, 1)
    return float(slope), float(intercept)


def runtime_base(templates, prefix):
    """The footprint that does not grow with the input: the smallest early
    footprint seen in any template or the prefix (the C runtime, libraries)."""
    early = [tpl["Truth"].iloc[0] for tpl in templates] + [prefix["Truth"].iloc[0]]
    return 0.9 * float(min(early))


def forecast(prefix, templates, points=200, temperature=0.05):
    """Forecast the rest of a footprint curve from its prefix (Time, Truth)
    using the training configs' curves as templates. Templates are averaged
    with soft-min weights exp(-(residual - best) / temperature)."""
    base = runtime_base(templates, prefix)
    law = length_law(templates, base)
    fits = [(fit_template(prefix, tpl, law, base), tpl) for tpl in templates]
    fits = [(fit, tpl) for fit, tpl in fits if np.isfinite(fit[0])]
    t0 = prefix["Time"].max()
    if not fits:
        last = float(prefix["Truth"].iloc[-1])
        return Forecast(pd.DataFrame({"Time": [], "Forecast": []}), t0, last, float(prefix["Truth"].max()))

    residuals = np.array([fit[0] for fit, _ in fits])
    weights = np.exp(-(residuals - residuals.min()) / temperature)
    weights /= weights.sum()
    ends = np.array([fit[2] * tpl["Time"].max() for fit, tpl in fits])
    end_time = float(weights @ ends)
    times = np.linspace(t0, max(end_time, t0), points)[1:]
    values = np.zeros(len(times))
    total = np.zeros(len(times))
    for w, ((_, a, b), tpl), end in zip(weights, fits, ends):
        running = times <= end
        values[running] += w * (base + a * (np.interp(times[running] / b, tpl["Time"], tpl["Truth"]) - base))
        total[running] += w
    curve = pd.DataFrame({"Time": times, "Forecast": np.where(total > 0, values / np.maximum(total, 1e-12), np.nan)})
    curve = curve.dropna()
    final = float(sum(w * (base + a * (tpl["Truth"].iloc[-1] - base)) for w, ((_, a, _b), tpl) in zip(weights, fits)))
    peak = float(max(prefix["Truth"].max(), curve["Forecast"].max() if len(curve) else 0))
    return Forecast(curve, end_time, final, peak)


def forecast_error(fc, truth, t0):
    rest = truth[truth["Time"] > t0]
    true_final, true_peak, true_end = truth["Truth"].iloc[-1], truth["Truth"].max(), truth["Time"].max()
    ok = rest["Truth"] > 0
    rest_mape = (
        float(np.mean(np.abs(fc.at(rest["Time"][ok]) - rest["Truth"][ok]) / rest["Truth"][ok]) * 100) if ok.any() else np.nan
    )
    return {
        "rest_mape": rest_mape,
        "final_error": (fc.final - true_final) / true_final * 100 if true_final else np.nan,
        "peak_error": (fc.peak - true_peak) / true_peak * 100,
        "end_error": (fc.end_time - true_end) / true_end * 100,
    }


# --------------------------------------------------------------- evaluate


def evaluate(workload, timeline):
    """Hold out the largest (EXTRA) and middle (INTER) config, as for whole-run
    models, and evaluate reconstruction and forecasting on it.

    Returns (reconstruction table, forecast table, curves) where curves holds
    the per-snapshot results for plotting.
    """
    prepared = prepare(peak_snapshot(timeline))
    truths = {c: t for c, t in truth_curves(timeline).groupby("Config")}
    rows = training_rows(timeline)
    all_splitter_bins = bin_rows(timeline, "splitter")
    all_sampler_bins = bin_rows(timeline, "sampler")
    recon_rows, forecast_rows, curves = [], [], []

    for split, attr in SPLITS.items():
        test_config = getattr(prepared, attr)
        train_configs = [c for c in prepared.configs if c != test_config]
        truth = truths[test_config]
        splitter_bins = all_splitter_bins[all_splitter_bins["Config"] == test_config]
        samplers = sampler_runs(all_sampler_bins, test_config)

        best = None
        for subset in prepared.subsets:
            intervals = sorted(prepared.subset(subset)["SamplingInterval"].unique())
            for variant in VARIANTS:
                model, si, train_mape = fit_reconstruction(rows, train_configs, intervals, variant)
                row = {"workload": workload, "split": split, "config": test_config, "subset": subset,
                       "variant": variant, "train_si": si, "train_mape": train_mape}

                curve = estimate_curve(splitter_bins[splitter_bins["SamplingInterval"] == si], model)
                row["splitter_mape"], row["splitter_peak_error"] = curve_error(curve, truth)
                curves.append(curve.assign(workload=workload, split=split, config=test_config, subset=subset,
                                           variant=variant, source="splitter bins"))

                sampler_si = closest(list(samplers), si)
                row["sampler_si"] = sampler_si
                if sampler_si is not None:
                    pid = samplers[sampler_si]["PID"].iloc[0]
                    run = samplers[sampler_si][samplers[sampler_si]["PID"] == pid]
                    curve = estimate_curve(run, model)
                    run_end = float(run["Time"].max())
                    row["sampler_mape"], row["sampler_peak_error"] = curve_error(curve, truth, run_end=run_end)
                    row["sampler_length_error"] = (run_end - truth["Time"].max()) / truth["Time"].max() * 100
                    curve = curve.assign(Time=curve["Time"] / run_end * truth["Time"].max())
                    row["sampler_coverage"] = len(curve) / max(run["Time"].nunique(), 1)
                    curves.append(curve.assign(workload=workload, split=split, config=test_config, subset=subset,
                                               variant=variant, source="sampler"))
                recon_rows.append(row)
                if variant == "base" and (best is None or train_mape < best[0]):
                    best = (train_mape, subset)

        chao_rows, chao_curves = evaluate_chao(timeline, truths, train_configs, test_config, workload, split)
        recon_rows += chao_rows
        curves += chao_curves

        templates = [truths[c] for c in train_configs]
        for fraction in PREFIXES:
            t0 = fraction * truth["Time"].max()
            fc = forecast(truth[truth["Time"] <= t0], templates)
            forecast_rows.append({"workload": workload, "split": split, "config": test_config, "prefix": fraction,
                                  **forecast_error(fc, truth, t0)})
            curves.append(fc.curve.rename(columns={"Forecast": "Estimate"}).assign(
                workload=workload, split=split, config=test_config, subset=best[1], variant=f"prefix {fraction:g}",
                source="forecast"))
    return pd.DataFrame(recon_rows), pd.DataFrame(forecast_rows), pd.concat(curves, ignore_index=True)


def fit_final_model(timeline, subset="L2O", variant="base"):
    """Model trained on all configs of a workload, for estimating new runs.
    Returns (model, recommended sampling interval)."""
    prepared = prepare(peak_snapshot(timeline))
    if subset not in prepared.subsets:
        subset = "NZ"
    intervals = sorted(prepared.subset(subset)["SamplingInterval"].unique())
    model, si, _ = fit_reconstruction(training_rows(timeline), prepared.configs, intervals, variant)
    return model, si, subset
