"""Fit MemPrint models for each workload.

For every training subset (NZ, MT, L2O) and held-out config (EXTRA: largest
input, INTER: middle input) a model is trained on the other configs. The
sampling interval used for prediction is either the one with the lowest
training MAPE or the best monotone interval of the workload.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset import prepare
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


def fit_workload(workload, all_data, sd_col="SD_MemUsage"):
    """Fit every (split, subset) model for one workload."""
    prepared = prepare(all_data, sd_col)
    fits = []
    for split, attr in SPLITS.items():
        test_config = getattr(prepared, attr)
        for subset in prepared.subsets:
            data = prepared.subset(subset)
            train = data[data["Config"] != test_config]
            test = data[data["Config"] == test_config]

            model = Model.fit(train, sd_col)
            train_mape = mape_by_interval(model.predict(train, sd_col))
            train_si = int(train_mape.idxmin())
            test_pred = model.predict(test, sd_col)

            def test_mape_at(si):
                return mape(test_pred[test_pred["SamplingInterval"] == si]) if si is not None else np.nan

            fits.append(
                Fit(
                    workload=workload,
                    split=split,
                    subset=subset,
                    model=model,
                    train_si=train_si,
                    train_mape=train_mape.min(),
                    test_mape_train_si=test_mape_at(train_si),
                    best_interval=prepared.best_interval,
                    test_mape_best=test_mape_at(prepared.best_interval),
                )
            )
    return fits


def select_models(fits, candidates=("NZ", "L2O")):
    """Pick one model per (workload, split): the candidate subset with the lowest
    test MAPE at the workload's best monotone interval (NZ when there is none).
    Returns the models table (workload, split, method, sample_rate, error, coefficients)."""
    rows = []
    for (workload, split), group in _group(fits).items():
        options = [f for f in group if f.subset in candidates and f.best_interval is not None]
        if options:
            best = min(options, key=lambda f: f.test_mape_best)
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
                **dict(zip(COEF_COLS, f.model.coefficients)),
            }
        )
    return pd.DataFrame(rows)
