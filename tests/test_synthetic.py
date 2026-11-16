"""The synthetic generators have the ground truth they claim to have."""

from __future__ import annotations

import numpy as np

from overfitlab.data.synthetic import simulate_market, simulate_returns


def test_reproducible_and_seed_dependent():
    a = simulate_returns(500, 3, seed=1)["returns"]
    b = simulate_returns(500, 3, seed=1)["returns"]
    c = simulate_returns(500, 3, seed=2)["returns"]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_null_world_is_a_martingale():
    sim = simulate_returns(200_000, 2, kind="garch", seed=3)
    r = sim["returns"]
    se = r.std(axis=0) / np.sqrt(len(r))
    assert np.all(np.abs(r.mean(axis=0)) < 4 * se)  # zero mean
    for lag in (1, 5, 20):  # no linear predictability
        ac = np.corrcoef(r[lag:, 0], r[:-lag, 0])[0, 1]
        assert abs(ac) < 0.02
    # ...but volatility clusters and tails are fat.
    ac_sq = np.corrcoef(r[1:, 0] ** 2, r[:-1, 0] ** 2)[0, 1]
    assert ac_sq > 0.05
    kurt = np.mean((r[:, 0] - r[:, 0].mean()) ** 4) / r[:, 0].var() ** 2
    assert kurt > 4.0


def test_gbm_is_gaussian_with_requested_volatility():
    r = simulate_returns(100_000, 1, kind="gbm", vol_annual=0.73, seed=4)["returns"][:, 0]
    assert abs(r.std() * np.sqrt(365) - 0.73) < 0.01
    kurt = np.mean((r - r.mean()) ** 4) / r.var() ** 2
    assert abs(kurt - 3.0) < 0.1


def test_planted_signal_gives_the_oracle_its_stated_sharpe():
    strength = 2.0
    sim = simulate_returns(
        300_000, 1, kind="garch", signal_strength=strength, signal_half_life=20, seed=5
    )
    r, sigma, x = sim["returns"][:, 0], sim["sigma"][:, 0], sim["state"][:, 0]
    oracle = (x / sigma) * r  # position x/sigma held over the bar whose drift it knows
    sharpe = oracle.mean() / oracle.std() * np.sqrt(365)
    assert abs(sharpe - strength) < 0.15
    # Returns are now positively autocorrelated (past returns predict future returns).
    assert np.corrcoef(r[1:], r[:-1])[0, 1] > 0.002


def test_asset_correlation():
    r = simulate_returns(50_000, 2, kind="gbm", asset_corr=0.6, seed=6)["returns"]
    assert abs(np.corrcoef(r.T)[0, 1] - 0.6) < 0.02


def test_ohlc_is_consistent():
    prices = simulate_market(1000, 2, seed=8)
    for df in prices.values():
        assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-12).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-12).all()
        assert (df[["open", "high", "low", "close"]] > 0).all().all()
        np.testing.assert_allclose(df["open"].to_numpy()[1:], df["close"].to_numpy()[:-1])
