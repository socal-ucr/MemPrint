"""Known-rate estimate of how many addresses a sample missed.

A union row is a Bernoulli sample of the references at a known rate p
(sampled references / references executed). An address accessed r times is
sampled Binomial(r, p) times. Across addresses, r = 1 + X with X negative
binomial (mean m, shape a), which covers both mostly-touched-once and heavily
reused memory. The sample count then has pgf (q + p z) B(z) with
B(z) = ((1 + c) - c z)^-a, c = m p / a, so

    P(k) = q b_k + p b_{k-1},  b_k = (1 + c)^-a C(a + k - 1, k) theta^k,  theta = c / (1 + c).

Two constraints are known exactly: U * mean(r) = T (references executed) and
U (1 - P(0)) = S (addresses seen). For each shape a, U follows from them; the
shape is chosen by how well U P(1..4) and U P(>=5) match the observed counts
f1..f4 and S - f1 - ... - f4. Accurate when the sample is dense enough to see
repeats (union of 1 in 5-12 references); at sparser rates f2..f4 are too small
to pin the shape down.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import gammaln

SHAPES = np.exp(np.linspace(np.log(0.02), np.log(200), 40))


def count_probabilities(U, a, T, p, kmax=4):
    """P(an address is sampled k times), k = 0..kmax."""
    q = 1 - p
    c = max(T / U - 1, 1e-12) * p / a
    k = np.arange(kmax + 1)
    b = np.exp(-a * np.log1p(c) + gammaln(a + k) - gammaln(a) - gammaln(k + 1) + k * np.log(c / (1 + c)))
    P = q * b
    P[1:] += p * b[:-1]
    return P


def solve_addresses(S, T, p, a):
    """Addresses U for which U (1 - P(0)) = S, with the reuse shape a fixed (None if no solution)."""
    seen = lambda U: U * (1 - count_probabilities(U, a, T, p)[0]) - S
    lo, hi = S * (1 + 1e-9), T * (1 - 1e-12)
    if seen(lo) > 0 or seen(hi) < 0:
        return None
    return brentq(seen, lo, hi, xtol=1e-6 * S)


def fit_error(U, a, T, p, observed):
    """Pearson chi-square of U P(1..4), U P(>=5) against the observed counts."""
    P = count_probabilities(U, a, T, p)
    expected = U * np.append(P[1:5], max(1 - P[:5].sum(), 0))
    return float(np.sum((observed - expected) ** 2 / (expected + 1)))


def estimate_addresses(S, counts, T, p):
    """Addresses touched (seen and unseen), from S seen, counts = (f1..f4),
    T references executed and sampling rate p."""
    counts = np.asarray(counts, float)
    if S <= 0 or T <= S or not 0 < p < 1:
        return np.nan
    observed = np.append(counts, S - counts.sum())
    best, best_U = np.inf, np.nan
    for a in SHAPES:
        U = solve_addresses(S, T, p, a)
        if U is None:
            continue
        chi2 = fit_error(U, a, T, p, observed)
        if chi2 < best:
            best, best_U = chi2, U
    return best_U


def known_rate_bytes(rows):
    """Known-rate footprint estimate (bytes) for union rows of a timeline
    (timeline.union_rows: needs the union's sampling Rate)."""
    estimates = [
        estimate_addresses(r.UniqueAddresses, (r.Singletons, r.Doubletons, r.Tripletons, r.Quadrupletons), r.Time,
                           r.Rate) * r.MemUsageObs / r.UniqueAddresses
        for r in rows.itertuples()
    ]
    return np.array(estimates, float)


def _row_args(r):
    return r.UniqueAddresses, np.array([r.Singletons, r.Doubletons, r.Tripletons, r.Quadrupletons], float), r.Time, r.Rate


def run_shape(rows, window=(0.5, 0.95)):
    """One reuse shape for a whole run: the shape that best fits the counts
    of its data-rich snapshots (between `window` fractions of the run, before
    the frees at exit), jointly."""
    end = rows["Time"].max()
    fit = rows[(rows["Time"] >= window[0] * end) & (rows["Time"] <= window[1] * end)]
    fit = fit.iloc[:: max(1, len(fit) // 30)]
    best, best_a = np.inf, np.nan
    for a in SHAPES:
        total = 0.0
        for r in fit.itertuples():
            S, counts, T, p = _row_args(r)
            if S <= 0 or T <= S or not 0 < p < 1:
                continue
            U = solve_addresses(S, T, p, a)
            if U is None:
                total = np.inf
                break
            total += fit_error(U, a, T, p, np.append(counts, S - counts.sum()))
        if total < best:
            best, best_a = total, a
    return best_a


def known_rate_bytes_pooled(rows):
    """Known-rate estimates with one reuse shape per run (Config, PID),
    fitted on the run's data-rich snapshots."""
    out = pd.Series(np.nan, index=rows.index)
    for _, run in rows.groupby(["Config", "PID"]):
        a = run_shape(run)
        if not np.isfinite(a):
            continue
        for idx, r in zip(run.index, run.itertuples()):
            S, counts, T, p = _row_args(r)
            U = solve_addresses(S, T, p, a) if S > 0 and T > S and 0 < p < 1 else None
            if U is not None:
                out[idx] = U * r.MemUsageObs / r.UniqueAddresses
    return out.to_numpy(float)
