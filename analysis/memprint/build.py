"""Fit MemPrint models for each workload.

For every training subset (NZ, MT, L2O) and held-out config (EXTRA: largest
input, INTER: middle input) a model is trained on the other configs. The
sampling interval used for prediction is either the one with the lowest
training MAPE or the best monotone interval of the workload.

The selection (the intervals pruned for empty bins, the monotone scores and the
best monotone interval) uses the training configs only (select_on="train").
select_on="all" computes it once on all configs, the held-out one included, as
the published tables did.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset import EXCLUDED_INTERVALS, prepare
from .model import COEF_COLS, Model, mape, mape_by_interval

SPLITS = {"EXTRA": "last_config", "INTER": "middle_config"}


@dataclass
class Fit:
    workload: str
    split: str  # EXTRA or INTER
    subset: str  # NZ, MT or L2O
    model: Model
    train_si: int  # interval with the lowest training MAPE
    train_mape: float  # training MAPE at train_si
    test_mape_train_si: float  # test MAPE at train_si
    best_interval: int  # best monotone interval of the workload (None if there is none)
    test_mape_best: float  # test MAPE at best_interval
    train_mape_best: float = np.nan  # training MAPE at best_interval
    test_dropped: int = 0  # test bins left out because they are empty at the held-out config (alpha infinite)


def test_rows(all_data, prepared, config, intervals):
    """The held-out config's bins at the given intervals, with Alpha from its truth. Bins with no footprint or no
    references (alpha infinite) are dropped; returns (rows, number dropped)."""
    d = all_data[(all_data["Config"] == config) & all_data["SamplingInterval"].isin(intervals)
                 & ~all_data["SamplingInterval"].isin(EXCLUDED_INTERVALS)].dropna()
    empty = (d["MemUsageObs"] == 0) | (d["CountObs"] == 0)
    d = d[~empty].copy()
    d["Alpha"] = prepared.truth[config] / d["MemUsageObs"]
    return d, int(empty.sum())


def fit_workload(workload, all_data, sd_col="SD_MemUsage", select_on="train"):
    """Fit every (split, subset) model for one workload. select_on: "train" (the selection on the training configs
    of each split) or "all" (once on all configs)."""
    prepared = prepare(all_data, sd_col)
    fits = []
    for split, attr in SPLITS.items():
        test_config = getattr(prepared, attr)
        sel = prepared if select_on == "all" else prepare(all_data[all_data["Config"] != test_config], sd_col)
        for subset in sel.subsets:
            data = sel.subset(subset)
            train = data[data["Config"] != test_config]
            if select_on == "all":
                test, dropped = data[data["Config"] == test_config], 0
            else:
                test, dropped = test_rows(all_data, prepared, test_config, data["SamplingInterval"].unique())

            model = Model.fit(train, sd_col)
            train_mape = mape_by_interval(model.predict(train, sd_col))
            train_si = int(train_mape.idxmin())
            test_pred = model.predict(test, sd_col)

            def test_mape_at(si):
                if si is None:
                    return np.nan
                t = test_pred[test_pred["SamplingInterval"] == si]
                return mape(t) if len(t) else np.nan

            fits.append(
                Fit(
                    workload=workload,
                    split=split,
                    subset=subset,
                    model=model,
                    train_si=train_si,
                    train_mape=train_mape.min(),
                    test_mape_train_si=test_mape_at(train_si),
                    best_interval=sel.best_interval,
                    test_mape_best=test_mape_at(sel.best_interval),
                    train_mape_best=train_mape.get(sel.best_interval, np.nan) if sel.best_interval else np.nan,
                    test_dropped=dropped,
                )
            )
    return fits


def select_models(fits, candidates=("NZ", "L2O"), by="train"):
    """Pick one model per (workload, split) among the candidate subsets: by="train", the lowest training MAPE at the
    workload's best monotone interval (the interval the model is deployed at); by="test", the lowest test MAPE there
    (the published selection). NZ at its training-best interval when there is no monotone interval.
    Returns the models table (workload, split, method, sample_rate, error, coefficients)."""
    rows = []
    for (workload, split), group in _group(fits).items():
        options = [f for f in group if f.subset in candidates and f.best_interval is not None]
        if options:
            key = (lambda f: f.train_mape_best) if by == "train" else (lambda f: f.test_mape_best)
            best = min(options, key=key)
            sample_rate, error = best.best_interval, best.test_mape_best
        else:
            best = next(f for f in group if f.subset == "NZ")
            sample_rate, error = best.train_si, best.test_mape_train_si
        rows.append([workload, split, best.subset, int(sample_rate), error] + best.model.coefficients)
    return pd.DataFrame(rows, columns=["workload", "split", "method", "sample_rate", "error"] + COEF_COLS)


def _group(fits):
    groups = {}
    for f in fits:
        groups.setdefault((f.workload, f.split), []).append(f)
    return groups


def _pct(value):
    return "--" if value is None or np.isnan(value) else f"{value:.2f}\\%"


def latex_tables(fits):
    """LaTeX rows for the paper's accuracy tables.

    - train: per split, the lowest training MAPE and its interval for NZ, MT, L2O
    - test: test MAPE at each subset's training-best interval (EXTRA then INTER)
    - test_best: test MAPE when every subset uses the best monotone interval
    """
    by_key = {(f.workload, f.split, f.subset): f for f in fits}
    workloads = list(dict.fromkeys(f.workload for f in fits))
    lines = {"train EXTRA": [], "train INTER": [], "test": [], "test_best": []}

    def get(w, split, subset):
        return by_key.get((w, split, subset))

    for w in workloads:
        for split in SPLITS:
            cells = []
            for subset in ("NZ", "MT", "L2O"):
                f = get(w, split, subset)
                cells += [str(f.train_si), _pct(f.train_mape)] if f else ["--", "--"]
            lines[f"train {split}"].append(" & ".join([w] + cells) + " \\\\")

        test, test_best = [w], [w]
        best_interval = None
        for split in SPLITS:
            for subset in ("NZ", "MT", "L2O"):
                f = get(w, split, subset)
                if f is None:
                    test.append("--")
                    test_best.append("--")
                    continue
                best_interval = f.best_interval
                test.append(_pct(f.test_mape_train_si))
                test_best.append(_pct(f.test_mape_best))
        lines["test"].append(" & ".join(test) + " \\\\")
        lines["test_best"].append(" & ".join([test_best[0], str(best_interval)] + test_best[1:]) + " \\\\")
    return lines


def fits_table(fits):
    """All fits as a flat table (one row per workload, split and subset)."""
    rows = []
    for f in fits:
        rows.append(
            {
                "workload": f.workload,
                "split": f.split,
                "subset": f.subset,
                "train_si": f.train_si,
                "train_mape": f.train_mape,
                "test_mape_train_si": f.test_mape_train_si,
                "best_interval": f.best_interval,
                "test_mape_best": f.test_mape_best,
                "train_mape_best": f.train_mape_best,
                "test_dropped": f.test_dropped,
                **dict(zip(COEF_COLS, f.model.coefficients)),
            }
        )
    return pd.DataFrame(rows)
