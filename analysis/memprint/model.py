"""The MemPrint regression model.

log(Alpha) is linear in log(spread), log(SamplingInterval), log(MemUsageObs)
and their pairwise and three-way products. A model is an intercept plus the
seven coefficients b1..b7 in FEATURES order.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

COEF_COLS = ["intercept", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]


def features(data, sd_col="SD_MemUsage"):
    log_sd = np.log(data[sd_col].astype(float))
    log_si = np.log(data["SamplingInterval"].astype(float))
    log_mem = np.log(data["MemUsageObs"].astype(float))
    return pd.DataFrame(
        {
            f"log_{sd_col}": log_sd,
            "log_SamplingInterval": log_si,
            "log_MemUsage": log_mem,
            "inter1": log_mem * log_sd,
            "inter2": log_si * log_sd,
            "inter3": log_mem * log_si,
            "inter4": log_mem * log_sd * log_si,
        },
        index=data.index,
    )


@dataclass
class Model:
    intercept: float
    coef: np.ndarray

    @classmethod
    def fit(cls, data, sd_col="SD_MemUsage"):
        reg = LinearRegression(fit_intercept=True)
        reg.fit(features(data, sd_col), np.log(data["Alpha"].astype(float)))
        return cls(reg.intercept_, reg.coef_)

    @classmethod
    def from_row(cls, row):
        return cls(float(row["intercept"]), row[COEF_COLS[1:]].to_numpy(dtype=float))

    @property
    def coefficients(self):
        return [self.intercept] + list(self.coef)

    def predict(self, data, sd_col="SD_MemUsage"):
        """Return a copy of data with Predicted_Alpha and ErrorRate (%) columns."""
        data = data.copy()
        log_alpha = features(data, sd_col).to_numpy() @ self.coef + self.intercept
        data["Predicted_Alpha"] = np.exp(log_alpha)
        data["ErrorRate"] = (data["Predicted_Alpha"] - data["Alpha"]) / data["Alpha"] * 100
        return data


def mape(predicted):
    """Mean absolute percentage error of Predicted_Alpha against Alpha."""
    return np.mean(np.abs((predicted["Alpha"] - predicted["Predicted_Alpha"]) / predicted["Alpha"])) * 100


def mape_by_interval(predicted):
    """MAPE for each sampling interval, as a Series indexed by interval."""
    return predicted.groupby("SamplingInterval")[["Alpha", "Predicted_Alpha"]].apply(mape)
