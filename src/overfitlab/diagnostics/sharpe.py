"""Sharpe-ratio inference: PSR, expected maximum Sharpe, DSR, MinTRL and MinBTL.

All Sharpe ratios inside this module are PER-PERIOD (not annualised) unless a name says
``annual``. To annualise multiply by ``sqrt(periods_per_year)``.

References
----------
* Bailey, D. H. and Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."
  Journal of Risk 15(2). -> Probabilistic Sharpe Ratio (PSR), Minimum Track Record Length.
* Bailey, D. H. and Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality." Journal of Portfolio
  Management 40(5). -> expected maximum Sharpe ratio, Deflated Sharpe Ratio (DSR).
* Bailey, D. H., Borwein, J. M., Lopez de Prado, M. and Zhu, Q. J. (2014).
  "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on
  Out-of-Sample Performance." Notices of the AMS 61(5). -> Minimum Backtest Length.
* Mertens, E. (2002). "Variance of the IID estimator in Lo (2002)." -> the standard
  error of the Sharpe ratio under non-normal returns used by the PSR.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def return_moments(returns: np.ndarray) -> dict[str, np.ndarray]:
    """Column-wise mean, std (ddof=1), skewness, kurtosis and per-period Sharpe ratio.

    Skewness ``g3 = m3 / m2^1.5`` and kurtosis ``g4 = m4 / m2^2`` are the plain moment
    estimators (a normal distribution has g3 = 0, g4 = 3). Columns with zero variance
    (a strategy that never trades) get Sharpe 0, skewness 0 and kurtosis 3.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    mean = x.mean(axis=0)
    dev = x - mean
    m2 = np.mean(dev**2, axis=0)
    live = m2 > 0
    safe_m2 = np.where(live, m2, 1.0)
    skew = np.where(live, np.mean(dev**3, axis=0) / safe_m2**1.5, 0.0)
    kurt = np.where(live, np.mean(dev**4, axis=0) / safe_m2**2, 3.0)
    std = np.sqrt(m2 * n / (n - 1))
    sharpe = np.where(live, mean / np.where(live, std, 1.0), 0.0)
    return {"mean": mean, "std": std, "skew": skew, "kurt": kurt, "sharpe": sharpe, "live": live}


def sharpe_ratio(returns: np.ndarray) -> np.ndarray:
    """Per-period Sharpe ratio (mean / std, zero risk-free rate) of each column."""
    return return_moments(returns)["sharpe"]


def sharpe_std_error(sr, n_obs: int, skew=0.0, kurt=3.0):
    """Standard error of the Sharpe-ratio estimator under non-normal iid returns.

        se(SR) = sqrt( (1 - g3*SR + (g4 - 1)/4 * SR^2) / (T - 1) )

    (Mertens 2002; Bailey & Lopez de Prado 2012, eq. 8). With normal returns
    (g3 = 0, g4 = 3) this is the familiar sqrt((1 + SR^2/2)/(T-1)).
    """
    sr = np.asarray(sr, dtype=float)
    return np.sqrt((1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / (n_obs - 1.0))


def probabilistic_sharpe_ratio(sr, n_obs: int, skew=0.0, kurt=3.0, sr_benchmark=0.0):
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012).

        PSR(SR*) = Phi( (SR - SR*) / se(SR) )

    The probability, in the sampling-distribution sense, that the TRUE Sharpe ratio
    exceeds the benchmark ``SR*`` given an estimate ``SR`` from ``n_obs`` returns with
    the given skewness and kurtosis. ``1 - PSR(0)`` is a one-sided p-value for
    H0: true SR <= 0.
    """
    return norm.cdf((np.asarray(sr) - sr_benchmark) / sharpe_std_error(sr, n_obs, skew, kurt))


def expected_max_z(n_trials: float) -> float:
    """Expected maximum of ``n_trials`` independent standard normal draws (approximation).

        E[max] ~ (1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e))

    with gamma the Euler-Mascheroni constant (Bailey & Lopez de Prado 2014, eq. for the
    "False Strategy" theorem). Returns 0 for N <= 1. The approximation is slightly high
    for small N (about 2.5% at N = 10, under 1.5% for N >= 100).
    """
    if n_trials <= 1.0:
        return 0.0
    value = (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials) + EULER_GAMMA * norm.ppf(
        1.0 - 1.0 / (n_trials * np.e)
    )
    return max(float(value), 0.0)  # the approximation dips below 0 for N barely above 1


def expected_max_sharpe(n_trials: float, sr_std: float, sr_mean: float = 0.0) -> float:
    """Expected maximum Sharpe ratio among ``n_trials`` independent skill-less trials.

    ``sr_std`` is the standard deviation of the Sharpe-ratio estimates across trials
    (same units as the output). This is the benchmark SR0 the Deflated Sharpe Ratio
    deflates against: even with zero true skill, the best of N backtests is expected to
    show a Sharpe ratio this high.
    """
    return sr_mean + sr_std * expected_max_z(n_trials)


def deflated_sharpe_ratio(
    sr: float,
    n_obs: int,
    skew: float,
    kurt: float,
    n_trials: float,
    sr_std: float,
) -> tuple[float, float]:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    ``DSR = PSR(SR0)`` with ``SR0 = expected_max_sharpe(n_trials, sr_std)``: the
    probability that the true Sharpe ratio of the selected strategy is positive after
    accounting for (i) selection among ``n_trials`` independent trials, (ii) the sample
    length and (iii) non-normal returns. Returns ``(DSR, SR0)``.

    All Sharpe quantities are per-period.
    """
    sr0 = expected_max_sharpe(n_trials, sr_std)
    return float(probabilistic_sharpe_ratio(sr, n_obs, skew, kurt, sr0)), sr0


def min_track_record_length(
    sr: float, skew: float = 0.0, kurt: float = 3.0, sr_benchmark: float = 0.0, alpha: float = 0.05
) -> float:
    """Minimum Track Record Length (Bailey & Lopez de Prado 2012), in observations.

        MinTRL = 1 + (1 - g3*SR + (g4-1)/4*SR^2) * (z_{1-alpha} / (SR - SR*))^2

    The number of observations needed for ``PSR(SR*)`` to reach ``1 - alpha``.
    Infinite if ``SR <= SR*``.
    """
    if sr <= sr_benchmark:
        return float("inf")
    z = norm.ppf(1.0 - alpha)
    return float(
        1.0 + (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) * (z / (sr - sr_benchmark)) ** 2
    )


def min_backtest_length(n_trials: float, sr_annual_target: float) -> float:
    """Minimum Backtest Length in YEARS (Bailey, Borwein, Lopez de Prado & Zhu 2014).

        MinBTL ~ ( E[max_N] / SR_target )^2,   E[max_N] = expected_max_z(N)

    If ``N`` independent skill-less strategies are tried on a backtest shorter than this,
    the best in-sample annualised Sharpe ratio is EXPECTED to reach ``sr_annual_target``
    by luck alone. (An annualised Sharpe estimated over y years has standard error
    ~ 1/sqrt(y), so E[max] ~ expected_max_z(N)/sqrt(y); setting that equal to the target
    and solving for y gives the formula.) The paper's bound is MinBTL < 2 ln(N) / SR^2.
    """
    if sr_annual_target <= 0:
        return float("inf")
    return (expected_max_z(n_trials) / sr_annual_target) ** 2


def max_trials_for_length(years: float, sr_annual_target: float) -> float:
    """Largest number of independent trials for which MinBTL stays below ``years``."""
    target = sr_annual_target * np.sqrt(years)
    if expected_max_z(2.0) >= target:
        return 1.0
    if expected_max_z(1e15) <= target:
        return float("inf")
    return float(brentq(lambda n: expected_max_z(n) - target, 2.0, 1e15))
