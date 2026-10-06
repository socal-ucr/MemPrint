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


def estimate_addresses(S, counts, T, p):
    """Addresses touched (seen and unseen), from S seen, counts = (f1..f4),
    T references executed and sampling rate p."""
    counts = np.asarray(counts, float)
    if S <= 0 or T <= S or not 0 < p < 1:
        return np.nan
    observed = np.append(counts, S - counts.sum())
    best, best_U = np.inf, np.nan
    for a in SHAPES:
        seen = lambda U: U * (1 - count_probabilities(U, a, T, p)[0]) - S
        lo, hi = S * (1 + 1e-9), T * (1 - 1e-12)
        if seen(lo) > 0 or seen(hi) < 0:
            continue
        U = brentq(seen, lo, hi, xtol=1e-6 * S)
        P = count_probabilities(U, a, T, p)
        expected = U * np.append(P[1:5], max(1 - P[:5].sum(), 0))
        chi2 = np.sum((observed - expected) ** 2 / (expected + 1))
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
