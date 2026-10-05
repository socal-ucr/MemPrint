"""Cross-workload prediction: apply every workload's model to every workload's
data, giving one error matrix per split (rows: data, columns: model)."""

import pandas as pd

from .dataset import prepare
from .model import COEF_COLS, Model, mape

MATRIX_INDEX = "data\\model"


def error_matrix(models, all_data_by_workload, split, sd_col="SD_MemUsage"):
    """MAPE (%) of each model of `split` (EXTRA or INTER) on each workload's data,
    using the model's own sampling interval."""
    chosen = models[models["split"] == split]
    prepared = {w: prepare(d, sd_col).data for w, d in all_data_by_workload.items()}

    matrix = pd.DataFrame(index=list(prepared), columns=chosen["workload"].tolist(), dtype=float)
    matrix.index.name = MATRIX_INDEX
    for _, row in chosen.iterrows():
        model = Model.from_row(row)
        for workload, data in prepared.items():
            rows = data[data["SamplingInterval"] == row["sample_rate"]]
            matrix.loc[workload, row["workload"]] = mape(model.predict(rows, sd_col))
    return matrix


def model_table(models, split):
    """Coefficients and sampling interval of each workload's model for one split."""
    chosen = models[models["split"] == split]
    return chosen[["workload"] + COEF_COLS + ["sample_rate"]].reset_index(drop=True)


def read_matrix(path):
    return pd.read_csv(path, index_col=0)
