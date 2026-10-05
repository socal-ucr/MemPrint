"""Relate model similarity to cross-workload error.

S is an RBF kernel over the standardised model coefficients and E the
extrapolation error matrix (rows: data, columns: model). The transformation
T = pinv(S) @ E maps similarity to error profiles; reconstructing E as S @ T
shows how much of the error structure the coefficients explain.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

from .model import COEF_COLS


@dataclass
class Transformation:
    similarity: pd.DataFrame  # S
    transformation: pd.DataFrame  # T
    rmse: float
    mape: float


def transform(models, errors):
    """models: workload + coefficient columns; errors: error matrix indexed by data workload."""
    common = [w for w in errors.index if w in models["workload"].values]
    coeffs = models.set_index("workload").loc[common]
    E = errors.loc[common, common].astype(float).values

    Cz = StandardScaler().fit_transform(coeffs[COEF_COLS].astype(float).values)
    S = rbf_kernel(Cz, gamma=1 / Cz.shape[1])
    T = np.linalg.pinv(S) @ E
    E_pred = S @ T

    return Transformation(
        similarity=pd.DataFrame(S, index=common, columns=common),
        transformation=pd.DataFrame(T, index=common, columns=common),
        rmse=float(np.sqrt(np.mean((E - E_pred) ** 2))),
        mape=float(np.mean(np.abs((E - E_pred) / (E + 1e-8))) * 100),
    )
