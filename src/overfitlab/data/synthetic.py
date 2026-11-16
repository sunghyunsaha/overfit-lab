"""Synthetic markets with known ground truth.

Three generators, selected by ``kind`` and ``signal_strength``:

(a) ``kind="gbm"``  constant volatility, Gaussian shocks, no signal: a pure random walk.
(b) ``kind="garch"`` GARCH(1,1) volatility clustering with Student-t (fat-tailed)
    shocks and no signal.
(c) either of the above with ``signal_strength > 0``: a small, slowly varying,
    hidden drift is added, so that past returns really do predict future returns.

Design choice that matters for the experiments: the *simple* return has conditional
mean exactly zero when ``signal_strength = 0`` and ``drift_annual = 0``. Prices are
then a martingale, so EVERY trading rule that only looks at the past has an expected
gross return of exactly zero and a true Sharpe ratio of exactly zero. That is what
makes experiment E1 a clean test: anything flagged there is a false positive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

MAX_ABS_RETURN = 0.5  # symmetric clip on one-bar simple returns (keeps prices positive)


def _shocks(rng: np.random.Generator, shape: tuple[int, ...], kind: str, df: float) -> np.ndarray:
    """Unit-variance shocks: Gaussian for GBM, standardised Student-t for GARCH."""
    if kind == "gbm":
        return rng.standard_normal(shape)
    t = rng.standard_t(df, size=shape)
    return t * np.sqrt((df - 2.0) / df)  # Var[t_df] = df/(df-2), rescale to variance 1


def simulate_returns(
    n_bars: int,
    n_assets: int,
    *,
    kind: str = "garch",
    vol_annual: float = 0.8,
    garch_alpha: float = 0.06,
    garch_beta: float = 0.92,
    student_df: float = 6.0,
    asset_corr: float = 0.0,
    signal_strength: float = 0.0,
    signal_half_life: float = 20.0,
    drift_annual: float = 0.0,
    periods_per_year: float = 365.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Simulate one-bar simple returns for ``n_assets`` assets.

    Model, for asset i and bar t:

        R[i,t]      = mu[i,t] + sigma[i,t] * z[i,t]                (clipped to +-50%)
        z[i,t]      = sqrt(rho) * f[t] + sqrt(1 - rho) * e[i,t]    (common factor f)
        sigma2[i,t] = omega + alpha * eps[i,t-1]^2 + beta * sigma2[i,t-1],
                      eps = sigma * z                              (GARCH only)
        mu[i,t]     = drift + s * sigma[i,t] * x[i,t]
        x[i,t]      = phi * x[i,t-1] + sqrt(1 - phi^2) * eta[i,t]  (hidden AR(1), variance 1)

    ``s = signal_strength / sqrt(periods_per_year)``. An oracle that could see ``x`` and
    hold a position ``x/sigma`` would earn a per-bar Sharpe ratio of about ``s``, i.e. an
    annualised Sharpe of about ``signal_strength``. Real strategies only see prices, so
    they capture a fraction of it.

    Returns a dict with ``returns``, ``sigma`` and ``state`` (the hidden x), each (T, n).
    """
    if kind not in ("gbm", "garch"):
        raise ValueError(f"unknown kind {kind!r}")
    if not 0.0 <= asset_corr < 1.0:
        raise ValueError("asset_corr must be in [0, 1)")
    rng = np.random.default_rng(seed)
    sigma_bar = vol_annual / np.sqrt(periods_per_year)

    common = _shocks(rng, (n_bars, 1), kind, student_df)
    idio = _shocks(rng, (n_bars, n_assets), kind, student_df)
    z = np.sqrt(asset_corr) * common + np.sqrt(1.0 - asset_corr) * idio

    # Hidden predictable state: stationary AR(1) with unit variance (burn-in removes the
    # dependence on the starting value).
    phi = 0.5 ** (1.0 / signal_half_life)
    burn = int(10 * signal_half_life)
    eta = rng.standard_normal((n_bars + burn, n_assets))
    state = lfilter([np.sqrt(1.0 - phi**2)], [1.0, -phi], eta, axis=0)[burn:]

    s = signal_strength / np.sqrt(periods_per_year)
    drift = drift_annual / periods_per_year

    sigma = np.full((n_bars, n_assets), sigma_bar)
    if kind == "garch":
        omega = sigma_bar**2 * (1.0 - garch_alpha - garch_beta)
        if omega <= 0:
            raise ValueError("need garch_alpha + garch_beta < 1")
        var_t = np.full(n_assets, sigma_bar**2)
        for t in range(n_bars):  # the recursion is sequential in time, vectorised over assets
            sigma[t] = np.sqrt(var_t)
            eps = sigma[t] * z[t]
            var_t = omega + garch_alpha * eps**2 + garch_beta * var_t

    mu = drift + s * sigma * state
    returns = np.clip(mu + sigma * z, -MAX_ABS_RETURN, MAX_ABS_RETURN)
    return {"returns": returns, "sigma": sigma, "state": state}


def returns_to_ohlcv(
    returns: np.ndarray,
    sigma: np.ndarray,
    *,
    start: str = "2018-01-01",
    freq: str = "1D",
    seed: int = 0,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    """Turn one asset's return path into an OHLCV frame.

    The market is treated as continuous (24/7), so ``open[t] = close[t-1]``. High and low
    are drawn from the exact distribution of the maximum / minimum of a Brownian bridge
    that starts at 0 and ends at the bar's log return ``l`` with variance ``v``:

        max = (l + sqrt(l^2 - 2 v ln U)) / 2,   min = (l - sqrt(l^2 - 2 v ln U')) / 2,

    with U, U' uniform on (0, 1). This guarantees low <= min(open, close) and
    high >= max(open, close). (The two extremes are drawn independently, which ignores
    their mild dependence; only Donchian channels read highs and lows.)
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    close = initial_price * np.cumprod(1.0 + returns)
    open_ = np.concatenate([[initial_price], close[:-1]])
    log_ret = np.log1p(returns)
    var = sigma**2
    u1 = rng.uniform(1e-12, 1.0, n)
    u2 = rng.uniform(1e-12, 1.0, n)
    bar_max = 0.5 * (log_ret + np.sqrt(log_ret**2 - 2.0 * var * np.log(u1)))
    bar_min = 0.5 * (log_ret - np.sqrt(log_ret**2 - 2.0 * var * np.log(u2)))
    high = open_ * np.exp(bar_max)
    low = open_ * np.exp(bar_min)
    volume = 1e6 * np.exp(0.5 * rng.standard_normal(n)) * (1.0 + np.abs(returns) / sigma)
    index = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="time")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )


def simulate_market(
    n_bars: int,
    n_assets: int,
    *,
    seed: int = 0,
    start: str = "2018-01-01",
    freq: str = "1D",
    **model_kwargs,
) -> dict[str, pd.DataFrame]:
    """Simulate ``n_assets`` OHLCV frames named SYN00, SYN01, ... (see ``simulate_returns``)."""
    sim = simulate_returns(n_bars, n_assets, seed=seed, **model_kwargs)
    out = {}
    for i in range(n_assets):
        out[f"SYN{i:02d}"] = returns_to_ohlcv(
            sim["returns"][:, i],
            sim["sigma"][:, i],
            start=start,
            freq=freq,
            seed=seed * 1000 + i + 1,
        )
    return out


def synthetic_kwargs(cfg_synth: dict, periods_per_year: float) -> dict:
    """Extract the model keyword arguments of ``simulate_returns`` from a config section."""
    keys = (
        "kind",
        "vol_annual",
        "garch_alpha",
        "garch_beta",
        "student_df",
        "asset_corr",
        "signal_strength",
        "signal_half_life",
        "drift_annual",
    )
    out = {k: cfg_synth[k] for k in keys if k in cfg_synth}
    out["periods_per_year"] = periods_per_year
    return out
