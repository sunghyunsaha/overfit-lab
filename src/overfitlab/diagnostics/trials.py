"""Effective number of independent trials.

The 5,000 variants are far from independent (an MA crossover with windows 10/50 and one
with 10/100 on the same asset are nearly the same bet). The Deflated Sharpe Ratio needs
N, the number of INDEPENDENT trials, i.e. the N for which "the best of N independent
skill-less strategies" has the same expected maximum Sharpe ratio as our correlated
family. Four estimators are implemented, all computed from the correlation matrix C of
the variants' returns:

``cluster``       number of clusters found by hierarchical clustering on the correlation
                  distance, with the number of clusters chosen by silhouette quality.
                  This is a simplified version of the ONC algorithm of Lopez de Prado &
                  Lewis (2019): ONC runs k-means repeatedly; here average-linkage
                  clustering replaces k-means (deterministic, no extra dependency) and
                  the same quality score mean(silhouette)/std(silhouette) picks k.
``avg_corr``      N_eff = rho + (1 - rho) * N with rho the average pairwise correlation
                  (Bailey & Lopez de Prado 2014, section on the number of trials).
``eigen``         participation ratio of the eigenvalues, (sum l)^2 / sum l^2 = N^2/||C||_F^2.
``max_matching``  simulation: draw z ~ N(0, C), record max(z), average. N_eff is the
                  number of independent normals with that same expected maximum. This is
                  the definition above applied literally, so it is the estimator used
                  for the headline DSR; the others are reported for comparison.

References
----------
* Lopez de Prado, M. and Lewis, M. J. (2019). "Detection of false investment strategies
  using unsupervised learning methods." Quantitative Finance 19(9).
* Lopez de Prado, M. (2019). "A Data Science Solution to the Multiple-Testing Crisis in
  Financial Research." Journal of Financial Data Science 1(1).
* Bailey, D. H. and Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." JPM 40(5).
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import brentq
from scipy.spatial.distance import squareform
from scipy.stats import norm

from overfitlab.diagnostics.sharpe import EULER_GAMMA


def standardise(returns: np.ndarray) -> np.ndarray:
    """Z-score each column (ddof=1) after dropping zero-variance columns."""
    x = np.asarray(returns, dtype=float)
    std = x.std(axis=0, ddof=1)
    live = std > 0
    return (x[:, live] - x[:, live].mean(axis=0)) / std[live]


def correlation_matrix(z: np.ndarray) -> np.ndarray:
    """Correlation matrix (float32) of standardised columns."""
    z32 = z.astype(np.float32)
    corr = (z32.T @ z32) / np.float32(z.shape[0] - 1)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def neff_avg_corr(corr: np.ndarray) -> tuple[float, float]:
    """``(N_eff, average off-diagonal correlation)`` with N_eff = rho + (1 - rho) N."""
    n = corr.shape[0]
    if n < 2:
        return float(n), 0.0
    rho = float((corr.sum(dtype=np.float64) - n) / (n * (n - 1)))
    return float(np.clip(rho + (1.0 - rho) * n, 1.0, n)), rho


def neff_eigen(corr: np.ndarray) -> float:
    """Participation ratio of the eigenvalues of C: N^2 / sum_ij C_ij^2."""
    n = corr.shape[0]
    return float(n**2 / np.sum(np.square(corr, dtype=np.float64)))


def _raw_expected_max_z(n: float) -> float:
    """The expected-maximum formula without the clamp at zero (monotone in n > 1)."""
    return float(
        (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n)
        + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n * np.e))
    )


def invert_expected_max(e_max: float) -> float:
    """The N for which ``expected_max_z(N)`` equals ``e_max`` (N >= 1)."""
    if e_max <= 0.0:
        return 1.0
    return float(brentq(lambda n: _raw_expected_max_z(n) - e_max, 1.0 + 1e-9, 1e12))


def neff_max_matching(z: np.ndarray, n_sim: int = 2000, seed: int = 0) -> tuple[float, float]:
    """``(N_eff, simulated E[max])`` by matching the expected maximum.

    ``g @ z / sqrt(T-1)`` with g ~ N(0, I_T) is a draw from N(0, C) where C is the sample
    correlation matrix of the columns of ``z``; under the null hypothesis this is how the
    vector of t-statistics of the variants is distributed. Its maximum is averaged over
    ``n_sim`` draws and the result is mapped back to a number of independent trials.
    """
    rng = np.random.default_rng(seed)
    t = z.shape[0]
    z32 = z.astype(np.float32)
    maxima = []
    for start in range(0, n_sim, 500):
        g = rng.standard_normal((min(500, n_sim - start), t)).astype(np.float32)
        maxima.append((g @ z32).max(axis=1) / np.sqrt(t - 1.0))
    e_max = float(np.concatenate(maxima).mean())
    return invert_expected_max(e_max), e_max


def _silhouette_quality(dist: np.ndarray, labels: np.ndarray) -> float:
    """mean(silhouette) / std(silhouette) for a clustering (ONC's quality score)."""
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.flatnonzero(np.r_[True, sorted_labels[1:] != sorted_labels[:-1]])
    sizes = np.diff(np.r_[starts, len(labels)])
    sums = np.add.reduceat(dist[:, order], starts, axis=1, dtype=np.float64)  # (N, K)
    own = np.searchsorted(sorted_labels[starts], labels)
    rows = np.arange(len(labels))
    own_size = sizes[own]
    intra = sums[rows, own] / np.maximum(own_size - 1, 1)
    mean_to = sums / sizes
    mean_to[rows, own] = np.inf
    inter = mean_to.min(axis=1)
    denom = np.maximum(intra, inter)
    sil = np.where(
        (own_size > 1) & (denom > 0), (inter - intra) / np.where(denom > 0, denom, 1), 0.0
    )
    sd = sil.std()
    return float(sil.mean() / sd) if sd > 0 else 0.0


def neff_cluster(corr: np.ndarray, n_candidates: int = 24) -> tuple[int, dict[int, float]]:
    """``(number of clusters, {k: quality})`` from hierarchical clustering.

    Distance ``d_ij = sqrt((1 - C_ij) / 2)`` (0 for identical, 1 for opposite strategies).
    Average-linkage clustering is cut into k clusters for k = 2..30 and then a geometric
    grid up to N/2; the k with the best silhouette quality is returned.
    """
    n = corr.shape[0]
    if n < 4:
        return n, {}
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None)).astype(np.float32)
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False).astype(np.float64), method="average")
    top = max(2, n // 2)
    dense = np.arange(2, min(30, top) + 1)  # every small k, then a geometric grid
    sparse = np.geomspace(min(30, top), top, n_candidates).round().astype(int)
    ks = np.unique(np.r_[dense, sparse])
    quality = {}
    for k in ks:
        labels = fcluster(link, int(k), criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        quality[int(k)] = _silhouette_quality(dist, labels)
    best = max(quality, key=quality.get) if quality else n
    return int(best), quality


def effective_trials(
    returns: np.ndarray,
    methods: tuple[str, ...] = ("cluster", "avg_corr", "eigen", "max_matching"),
    n_sim: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Estimate the effective number of independent trials with each requested method.

    Also returns ``raw`` (the number of variants with non-zero variance), ``avg_corr_value``
    and, when computed, ``sim_expected_max`` (in t-statistic units).
    """
    z = standardise(returns)
    out: dict[str, float] = {"raw": float(z.shape[1])}
    if z.shape[1] < 2:
        return {**out, **{m: float(z.shape[1]) for m in methods}}
    corr = None
    if any(m in methods for m in ("cluster", "avg_corr", "eigen")):
        corr = correlation_matrix(z)
    if "avg_corr" in methods:
        out["avg_corr"], out["avg_corr_value"] = neff_avg_corr(corr)
    if "eigen" in methods:
        out["eigen"] = neff_eigen(corr)
    if "cluster" in methods:
        out["cluster"] = float(neff_cluster(corr)[0])
    if "max_matching" in methods:
        out["max_matching"], out["sim_expected_max"] = neff_max_matching(z, n_sim, seed)
    return out
