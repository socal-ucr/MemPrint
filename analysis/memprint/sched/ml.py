"""The ML-peak oracle (Pond-style): predict what fraction of its request a job will use from what the scheduler
knows at submission (request, cores, requested time, user, executable), with a quantile model so that it
rarely under-predicts.

Trained on LANL CM5 jobs (the archive log that records both requested and used memory), split by user: the
model never sees the users whose jobs it predicts. real.make_jobs pairs each simulated job with a CM5 job
from the held-out users, whose u sets the job's real peak and whose features the model sees.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = ["log_req", "procs", "log_req_time", "user_median", "exe_median"]


def split_users(cm5, frac=0.5, seed=0):
    users = cm5.uid.unique()
    rng = np.random.default_rng(seed)
    train = set(rng.choice(users, int(len(users) * frac), replace=False))
    return cm5[cm5.uid.isin(train)].reset_index(drop=True), cm5[~cm5.uid.isin(train)].reset_index(drop=True)


def features(d, history):
    """History features come from the training users' jobs only: per-user / per-executable median u (unseen
    users and executables get the overall median)."""
    med = history.u.median()
    by_user, by_exe = history.groupby("uid").u.median(), history.groupby("exe").u.median()
    return pd.DataFrame({
        "log_req": np.log(d.req_mem), "procs": d.procs, "log_req_time": np.log(d.req_time.clip(lower=1)),
        "user_median": d.uid.map(by_user).fillna(med).to_numpy(),
        "exe_median": d.exe.map(by_exe).fillna(med).to_numpy(),
    })


class PeakModel:
    def __init__(self, train, quantile=0.95):
        self.history = train
        self.m = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, max_iter=300,
                                               random_state=0)
        self.m.fit(features(train, train), train.u)

    def predict(self, d):
        return np.clip(self.m.predict(features(d, self.history)), 0.01, 1.0)


def oracle(model):
    """REAL-style oracle: the job's request x predicted u, held for the whole run, x (1 + margin)."""
    from .oracles import peak

    def f(j, m):
        c = j.info["cm5"]
        u = j.info.get("ml_u")
        if u is None:
            u = j.info["ml_u"] = float(model.predict(pd.DataFrame([c]))[0])
        return peak(j.info["request"] * u, m)
    return f
