"""The ML-peak oracle (after Pond's model) and the Pond oracle: predict what fraction of its request a job will use from what the scheduler
knows at submission (request, cores, requested time, user, executable), with a quantile model so that it
rarely under-predicts.

Trained on LANL CM5 jobs (the archive log that records both requested and used memory), split by user: the
model never sees the users whose jobs it predicts. real.make_jobs pairs each simulated job with a CM5 job
from the held-out users, whose u sets the job's real peak and whose features the model sees.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .model import K

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


def pond(model, qos_delay=5):
    """Pond (Li et al., ASPLOS 2023): the job keeps its whole request, so it is never OOM-killed, but only
    request - UM of it is local, where UM is the predicted untouched memory, rounded down to whole GB and placed
    in the pool (Pond's zNUMA node). `model` is a quantile model of u = peak/request; Pond's label is the
    lifetime minimum of untouched memory, 1 - u, and its quantile sets the target rate of overpredictions (2%
    at TP = 98%). The QoS monitor moves the pool part to local memory `qos_delay` minutes after the job first
    touches it. Pond's latency-insensitivity model (all-pool VMs, from 200 PMU counters) has no counterpart in
    the job streams and is left out. The margin is not used."""
    def u(j):
        if "pond_u" not in j.info:
            j.info["pond_u"] = float(model.predict(pd.DataFrame([j.info["cm5"]]))[0])
        return j.info["pond_u"]

    def f(j, m):
        return np.full(K, float(j.info["request"]))

    def attrs(j, m):
        req = float(j.info["request"])
        return dict(local_cap=req - np.floor(req * (1 - u(j))), qos_delay=qos_delay)

    f.attrs = attrs
    return f


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
