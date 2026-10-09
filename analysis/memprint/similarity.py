"""Memory-behaviour descriptor z(W) and distances between workloads.

z is measured from the splitter's bins at one input config, for every sampling
interval k in INTERVALS:

    reuse(k)  = log(alpha_k) / log(k)   1 when no address is seen twice in a bin's
                                         share of the run, towards 0 with heavy reuse
    spread(k) = log(SD_k / m_k)         bin-to-bin spread relative to the bin footprint

with alpha_k = true footprint / mean bin footprint. The same quantities follow
from a static access-count spectrum (static.spectrum.moments), so z can be
predicted for a workload from its source alone (z_hat) and compared with the
measured z of known workloads.
"""

import numpy as np
import pandas as pd

from .static.spectrum import moments

INTERVALS = [100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 100000]
PARTS = ("reuse", "spread")


def columns(parts=PARTS):
    return [f"{p}_{k}" for p in parts for k in INTERVALS]


def bin_summary(all_data):
    """Per (Config, SamplingInterval): mean bin footprint, mean unique addresses, SD of the bin
    footprint and the true footprint."""
    truth = all_data[all_data["SamplingInterval"] == 1].groupby("Config")["MemUsageObs"].min()
    bins = all_data[all_data["SamplingInterval"] > 1]
    summary = bins.groupby(["Config", "SamplingInterval"]).agg(
        m=("MemUsageObs", "mean"), u=("UniqueAddresses", "mean"), sd=("MemUsageObs", "std")).reset_index()
    summary["truth"] = summary["Config"].map(truth)
    return summary


def measured(all_data, config, parts=PARTS):
    """z of a workload at one config, from its traces."""
    s = bin_summary(all_data)
    s = s[s["Config"] == config].set_index("SamplingInterval")
    values = {}
    for k in INTERVALS:
        if k in s.index and s.loc[k, "m"] > 0:
            row = s.loc[k]
            values[f"reuse_{k}"] = np.log(row["truth"] / row["m"]) / np.log(k)
            values[f"spread_{k}"] = np.log(max(row["sd"], 1e-9) / row["m"])
        else:
            values[f"reuse_{k}"] = values[f"spread_{k}"] = np.nan
    return pd.Series(values)[columns(parts)]


def predicted(spec, baseline=None, parts=PARTS):
    """z_hat from a static spectrum (plus the runtime baseline)."""
    values = {}
    for k in INTERVALS:
        bm = moments(spec, k, baseline)
        values[f"reuse_{k}"] = np.log(bm.alpha) / np.log(k)
        values[f"spread_{k}"] = np.log(max(bm.sd, 1e-9) / bm.m)
    return pd.Series(values)[columns(parts)]


class Scaler:
    """Standardise descriptors with the statistics of the known workloads (NaN-aware)."""

    def __init__(self, known):
        self.mean = known.mean()
        self.std = known.std().replace(0, 1).fillna(1)

    def __call__(self, z):
        return ((z - self.mean) / self.std).fillna(0.0)


def distances(z, known, scaler=None):
    """Euclidean distance from descriptor z (Series) to every row of known (DataFrame), after
    standardising with the known workloads' statistics."""
    scaler = scaler or Scaler(known)
    zk = scaler(known)
    zc = scaler(z.to_frame().T).iloc[0]
    return np.sqrt(((zk - zc) ** 2).sum(axis=1))


def kernel_weights(d, bandwidth):
    w = np.exp(-0.5 * (d / bandwidth) ** 2)
    return w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(d), index=d.index)
