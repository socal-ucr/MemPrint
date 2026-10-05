"""Turn a workload's trace table (allData) into training data for the model.

Alpha is the ratio of the true footprint (full trace, SamplingInterval 1) to
the footprint observed in one subsample; the model predicts it from the
subsample's footprint, sampling interval and spread across subsamples.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr

# Intervals never used for training: 1 is the ground truth, 10 is too dense.
EXCLUDED_INTERVALS = (1, 10)


@dataclass
class Prepared:
    data: "pd.DataFrame"  # subsample rows with an Alpha column
    configs: list  # input configs, smallest true footprint first
    truth: dict  # config -> true footprint (bytes)
    monotone_scores: dict = field(default_factory=dict)  # interval -> score
    best_interval: int = None  # monotone interval with the highest score

    @property
    def last_config(self):
        """Held out for extrapolation."""
        return self.configs[-1]

    @property
    def middle_config(self):
        """Held out for interpolation."""
        return self.configs[len(self.configs) // 2]

    @property
    def intervals(self):
        return sorted(self.data["SamplingInterval"].unique())

    def subset(self, label):
        """Training subsets: NZ (all non-zero intervals), MT (intervals whose
        spread grows monotonically with input size) and L2O (the best monotone
        interval with one interval below and three above it)."""
        if label == "NZ":
            return self.data
        if label == "MT":
            return self.data[self.data["SamplingInterval"].isin(self.monotone_scores.keys())]
        if label == "L2O":
            intervals = self.intervals
            idx = intervals.index(self.best_interval)
            picked = intervals[max(0, idx - 1) : idx + 4] if idx else intervals[:5]
            return self.data[self.data["SamplingInterval"].isin(picked)]
        raise ValueError(f"unknown subset {label}")

    @property
    def subsets(self):
        """The subsets available for this workload (MT and L2O need a monotone interval)."""
        return ["NZ", "MT", "L2O"] if self.best_interval is not None else ["NZ"]


def prepare(all_data, sd_col="SD_MemUsage"):
    """Filter the trace table, order configs by true footprint, compute Alpha
    and score the sampling intervals by how monotonically their spread grows."""
    truth_rows = all_data[all_data["SamplingInterval"] == 1]

    data = all_data[~all_data["SamplingInterval"].isin(EXCLUDED_INTERVALS)]
    data = data.dropna()
    bad_intervals = data.loc[(data["MemUsageObs"] == 0) | (data["CountObs"] == 0), "SamplingInterval"].unique()
    data = data[~data["SamplingInterval"].isin(bad_intervals)].copy()

    configs = (
        truth_rows.loc[truth_rows.groupby("Config")["MemUsageObs"].idxmin()]
        .sort_values("MemUsageObs")["Config"]
        .tolist()
    )
    truth = truth_rows.set_index("Config")["MemUsageObs"].to_dict()
    data["Alpha"] = data["Config"].map(truth) / data["MemUsageObs"]

    prepared = Prepared(data=data, configs=configs, truth=truth)
    prepared.monotone_scores = monotone_scores(data, configs, sd_col)
    if prepared.monotone_scores:
        prepared.best_interval = max(prepared.monotone_scores, key=prepared.monotone_scores.get)
    return prepared


def monotone_scores(data, configs, sd_col="SD_MemUsage"):
    """Score each interval whose spread increases (Spearman rho ~ 1) with
    input size by rho * mean increase between consecutive configs."""
    spread = data.groupby(["SamplingInterval", "Config"])[sd_col].first().reset_index()
    scores = {}
    for interval in sorted(spread["SamplingInterval"].unique()):
        rows = spread[spread["SamplingInterval"] == interval]
        y = rows.set_index("Config").reindex(configs)[sd_col].dropna().values
        if len(y) > 1:
            rho, _ = spearmanr(np.arange(len(y)), y)
            if rho + 0.01 >= 1:
                scores[interval] = rho * np.mean(np.diff(y))
    return scores
