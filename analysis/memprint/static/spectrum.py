"""From an access-count spectrum to what the splitter's bins see.

The splitter keeps each reference with probability 1/k and puts it in a bin,
independently of the others. An address referenced c times therefore appears
in a given bin with probability p = 1 - (1 - 1/k)^c, and a bin's footprint m
(bytes of the start addresses it saw) has

    E[m] = sum_a s_a p_a,    Var[m] = sum_a s_a^2 p_a (1 - p_a),

with s_a the bytes of address a. The true footprint is sum_a s_a, so the
spectrum gives alpha = truth / m, the bins' spread (SD_MemUsage) and their
unique addresses, at every interval k, without running the program.

The static spectrum misses what the runtime touches outside the program text
(loader, libc start-up, malloc metadata). That part is modelled as a shared
baseline spectrum: non-negative bytes and addresses at access counts 2^0 ..
2^26, fitted (NNLS) to the bin footprints of known workloads.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

BASIS = 2.0 ** np.arange(0, 27)  # access counts of the baseline classes


@dataclass
class Spectrum:
    """Addresses grouped by (access count, size): n addresses of s bytes referenced c times each."""

    c: np.ndarray
    s: np.ndarray
    n: np.ndarray
    references: float = 0.0  # all references charged, including the always-sampled stack

    @classmethod
    def from_addresses(cls, counts, sizes, references=None):
        if len(counts) == 0:
            return cls(np.zeros(0), np.zeros(0), np.zeros(0), 0.0)
        # group counts on a fine log grid (0.1%) to keep the table small
        key = np.round(np.log(np.maximum(counts, 1e-9)) * 1000).astype(np.int64)
        table = np.stack([key, sizes.astype(np.int64)], axis=1)
        uniq, inverse = np.unique(table, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        n = np.bincount(inverse).astype(float)
        c = np.bincount(inverse, weights=counts) / n
        return cls(c, uniq[:, 1].astype(float), n, float(counts.sum() if references is None else references))

    @property
    def footprint(self):
        return float(np.sum(self.n * self.s))

    @property
    def addresses(self):
        return float(np.sum(self.n))

    def to_dict(self):
        return {"c": self.c, "s": self.s, "n": self.n, "references": np.array(self.references)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["c"], d["s"], d["n"], float(d["references"]))


def sample_probability(c, k):
    """P(an address referenced c times is in one 1-in-k bin)."""
    c = np.asarray(c, dtype=float)
    return -np.expm1(c * np.log1p(-1.0 / k)) if k > 1 else np.ones_like(c)


@dataclass
class BinMoments:
    m: float  # E[bin footprint] (bytes)
    sd: float  # SD of the bin footprint
    u: float  # E[unique addresses in the bin]
    truth: float  # true footprint

    @property
    def alpha(self):
        return self.truth / self.m if self.m > 0 else np.nan


def moments(spec, k, baseline=None):
    p = sample_probability(spec.c, k)
    m = float(np.sum(spec.n * spec.s * p))
    var = float(np.sum(spec.n * spec.s ** 2 * p * (1 - p)))
    u = float(np.sum(spec.n * p))
    truth = spec.footprint
    if baseline is not None:
        bm = baseline.moments(k)
        m, var, u, truth = m + bm.m, var + bm.sd ** 2, u + bm.u, truth + bm.truth
    return BinMoments(m, np.sqrt(var), u, truth)


@dataclass
class Baseline:
    """Runtime memory outside the program text: bytes[j] and addresses[j] at access count BASIS[j]."""

    bytes: np.ndarray
    addresses: np.ndarray

    def moments(self, k):
        p = sample_probability(BASIS, k)
        size = np.divide(self.bytes, self.addresses, out=np.full_like(self.bytes, 8.0), where=self.addresses > 0)
        return BinMoments(float(self.bytes @ p), float(np.sqrt(np.sum(self.addresses * size ** 2 * p * (1 - p)))),
                          float(self.addresses @ p), float(self.bytes.sum()))

    @property
    def footprint(self):
        return float(self.bytes.sum())

    @classmethod
    def fit(cls, rows):
        """rows: iterable of (spectrum, k, observed mean bin footprint, observed mean unique addresses).
        k = 1 rows are the true footprint. Residuals are fitted relative to the observation."""
        a_rows, bm, bu, m_scale, u_scale = [], [], [], [], []
        for spec, k, m_obs, u_obs in rows:
            static = moments(spec, k)
            a_rows.append(sample_probability(BASIS, k))
            bm.append(m_obs - static.m)
            bu.append(u_obs - static.u)
            m_scale.append(m_obs)
            u_scale.append(u_obs)
        A = np.array(a_rows)
        wm = 1.0 / np.maximum(np.array(m_scale), 1.0)
        wu = 1.0 / np.maximum(np.array(u_scale), 1.0)
        bytes_, _ = nnls(A * wm[:, None], np.array(bm) * wm)
        addresses, _ = nnls(A * wu[:, None], np.array(bu) * wu)
        return cls(bytes_, addresses)

    def to_dict(self):
        return {"bytes": self.bytes, "addresses": self.addresses}
