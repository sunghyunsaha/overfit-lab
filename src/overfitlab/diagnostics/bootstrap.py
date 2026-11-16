"""Data-snooping tests: White's Reality Check and Hansen's SPA test.

Both ask one family-level question about the whole T x N matrix of strategy returns:
"is the BEST strategy's mean return greater than zero (the benchmark is holding cash),
once we account for the fact that it is the best of N?"

References
----------
* White, H. (2000). "A Reality Check for Data Snooping." Econometrica 68(5), 1097-1126.
* Hansen, P. R. (2005). "A Test for Superior Predictive Ability." Journal of Business &
  Economic Statistics 23(4), 365-380.
* Politis, D. N. and Romano, J. P. (1994). "The Stationary Bootstrap." Journal of the
  American Statistical Association 89(428), 1303-1313.

The stationary bootstrap resamples blocks of consecutive rows whose lengths are
geometric with mean ``mean_block``, wrapping around the end of the sample. Resampling
whole ROWS keeps the cross-sectional dependence between strategies intact; resampling
BLOCKS keeps short-range time dependence (e.g. volatility clustering) intact.

Efficiency: both tests only need the MEAN of each resampled column. If ``counts[b, t]``
is the number of times row t appears in resample b, all bootstrap means are one matrix
product ``counts @ X / T``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def stationary_bootstrap_indices(
    n_obs: int, n_boot: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """(n_boot, n_obs) row indices drawn with the stationary bootstrap.

    At each position a new block starts with probability ``q = 1/mean_block`` at a
    uniformly random row; otherwise the next row (cyclically) is taken.
    """
    q = 1.0 / mean_block
    new_block = rng.random((n_boot, n_obs)) < q
    new_block[:, 0] = True
    starts = rng.integers(0, n_obs, size=(n_boot, n_obs))
    pos = np.arange(n_obs)
    last_start = np.maximum.accumulate(np.where(new_block, pos, 0), axis=1)
    start_row = np.take_along_axis(starts, last_start, axis=1)
    return (start_row + pos - last_start) % n_obs


def bootstrap_counts(indices: np.ndarray, n_obs: int) -> np.ndarray:
    """(n_boot, n_obs) matrix: how often each original row appears in each resample."""
    n_boot = indices.shape[0]
    flat = (indices + n_obs * np.arange(n_boot)[:, None]).ravel()
    return np.bincount(flat, minlength=n_boot * n_obs).reshape(n_boot, n_obs).astype(float)


def bootstrap_means(x: np.ndarray, n_boot: int, mean_block: float, seed: int) -> np.ndarray:
    """(n_boot, N) column means of stationary-bootstrap resamples of ``x``."""
    rng = np.random.default_rng(seed)
    t = x.shape[0]
    counts = bootstrap_counts(stationary_bootstrap_indices(t, n_boot, mean_block, rng), t)
    return (counts @ x) / t


def stationary_long_run_variance(x: np.ndarray, mean_block: float, tol: float = 1e-8) -> np.ndarray:
    """Long-run variance of each column: the variance of sqrt(T) * mean.

        omega^2 = gamma_0 + 2 * sum_{i>=1} kappa(i) * gamma_i,
        kappa(i) = (T - i)/T * (1 - q)^i + i/T * (1 - q)^(T - i),   q = 1/mean_block

    This is the exact variance implied by the stationary bootstrap (Politis & Romano
    1994) and the estimator Hansen (2005) uses to studentise. ``gamma_i`` is the i-th
    sample autocovariance. Simplification: the sum is truncated once ``(1-q)^i < tol``
    and the wrap-around term ``i/T (1-q)^(T-i)`` is dropped (it only matters for lags
    close to T, where it multiplies autocovariances estimated from a handful of points).
    """
    t = x.shape[0]
    q = 1.0 / mean_block
    dev = x - x.mean(axis=0)
    omega2 = np.einsum("ij,ij->j", dev, dev) / t
    gamma0 = omega2.copy()
    if q < 1.0:
        max_lag = min(t - 1, int(np.ceil(np.log(tol) / np.log(1.0 - q))))
        for i in range(1, max_lag + 1):
            gamma = np.einsum("ij,ij->j", dev[i:], dev[:-i]) / t
            omega2 += 2.0 * ((t - i) / t) * (1.0 - q) ** i * gamma
    return np.where(omega2 > 0, omega2, gamma0)


@dataclass
class SnoopingResult:
    """P-values of the family-level data-snooping tests."""

    rc_pvalue: float  # White's Reality Check
    spa_lower: float  # Hansen SPA, lower bound  (most liberal)
    spa_consistent: float  # Hansen SPA, consistent (the one to quote)
    spa_upper: float  # Hansen SPA, upper bound  (most conservative)
    spa_hac_consistent: float  # SPA studentised by the long-run s.d., as in Hansen (2005)
    hac_scale_ratio: float  # long-run s.d. / sample s.d. of the long-run version's winner
    rc_stat: float
    spa_stat: float
    best_mean_index: int  # column with the highest mean return
    best_t_index: int  # column with the highest studentised mean
    n_boot: int
    mean_block: float


def _spa_pvalues(dbar: np.ndarray, boot: np.ndarray, scale: np.ndarray, t: int) -> dict:
    """SPA statistic and its three p-values for a given studentising scale."""
    root_t = np.sqrt(t)
    stat = max(float((root_t * dbar / scale).max()), 0.0)
    threshold = -scale * np.sqrt(2.0 * np.log(np.log(t)) / t)
    centres = {
        "lower": np.maximum(dbar, 0.0),
        "consistent": np.where(dbar >= threshold, dbar, 0.0),
        "upper": dbar,
    }
    out = {"stat": stat, "argmax": int((dbar / scale).argmax())}
    for name, g in centres.items():
        stat_boot = np.maximum((root_t * (boot - g) / scale).max(axis=1), 0.0)
        out[name] = float(np.mean(stat_boot > stat))
    return out


def reality_check_and_spa(
    returns: np.ndarray, n_boot: int = 1000, mean_block: float = 10.0, seed: int = 0
) -> SnoopingResult:
    """White's Reality Check and Hansen's SPA test against a zero benchmark.

    H0: no strategy has a positive expected return, max_k E[d_k] <= 0, with d_k the
    return of strategy k (its excess over holding cash).

    Reality Check (White 2000):
        V   = max_k sqrt(T) * dbar_k
        V*_b = max_k sqrt(T) * (dbar*_{k,b} - dbar_k)
        p   = fraction of V*_b > V

    SPA (Hansen 2005) improves on it in two ways. (1) It studentises by a scale s_k, so
    one very volatile strategy cannot dominate the maximum. (2) It re-centres the
    bootstrap distribution with g(dbar_k) instead of dbar_k, so that strategies that are
    clearly bad do not inflate the null distribution (the Reality Check's known weakness):
        T_SPA = max( max_k sqrt(T) * dbar_k / s_k , 0 )
        T*_b  = max( max_k sqrt(T) * (dbar*_{k,b} - g(dbar_k)) / s_k , 0 )
        lower:      g_l(d) = max(d, 0)
        consistent: g_c(d) = d * 1{ d >= -s_k * sqrt(2 ln ln T / T) }
        upper:      g_u(d) = d
        p = fraction of T*_b > T_SPA

    DEVIATION FROM THE PAPER, found by experiment E1. Hansen studentises with the
    long-run standard deviation omega_k (``stationary_long_run_variance``). Here the
    default scale is the plain sample standard deviation of each strategy, and the
    omega-studentised p-value is reported separately as ``spa_hac_consistent``. Reason:
    omega_k adds up more than a hundred noisy sample autocovariances, so it is a much
    noisier estimate than the sample standard deviation (about three times noisier for
    white noise at the default block length). The statistic takes a MAXIMUM over
    thousands of ratios dbar_k / omega_k, and a maximum seeks out the strategies whose
    omega_k happens to be too small. The bootstrap holds omega_k fixed, so its null
    distribution does not contain that extra source of extreme values, and the test
    rejects too often (the measured rates, and how far below the sample standard
    deviation omega_k was for the winners, are in REPORT section 5.1).
    Under H0 strategy returns are uncorrelated, so both scales estimate the same
    quantity, and the block bootstrap still accounts for time dependence in the null
    distribution whichever scale is used.

    The same bootstrap resamples are used for all p-values. Columns with zero variance
    (strategies that never trade) are ignored.
    """
    x = np.asarray(returns, dtype=float)
    live = x.std(axis=0) > 0
    cols = np.flatnonzero(live)
    if len(cols) == 0:
        return SnoopingResult(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, -1, -1, n_boot, mean_block)
    x = x[:, cols]
    t = x.shape[0]
    root_t = np.sqrt(t)
    dbar = x.mean(axis=0)
    boot = bootstrap_means(x, n_boot, mean_block, seed)

    rc_stat = root_t * dbar.max()
    rc_boot = root_t * (boot - dbar).max(axis=1)
    rc_p = float(np.mean(rc_boot > rc_stat))

    sd = x.std(axis=0, ddof=1)
    spa = _spa_pvalues(dbar, boot, sd, t)
    omega = np.sqrt(stationary_long_run_variance(x, mean_block))
    spa_hac = _spa_pvalues(dbar, boot, omega, t)
    return SnoopingResult(
        rc_pvalue=rc_p,
        spa_lower=spa["lower"],
        spa_consistent=spa["consistent"],
        spa_upper=spa["upper"],
        spa_hac_consistent=spa_hac["consistent"],
        hac_scale_ratio=float(omega[spa_hac["argmax"]] / sd[spa_hac["argmax"]]),
        rc_stat=float(rc_stat),
        spa_stat=spa["stat"],
        best_mean_index=int(cols[dbar.argmax()]),
        best_t_index=int(cols[spa["argmax"]]),
        n_boot=n_boot,
        mean_block=mean_block,
    )
