"""Stationary bootstrap, Reality Check and SPA."""

from __future__ import annotations

import numpy as np
import pytest

from overfitlab.diagnostics import bootstrap as bs


def test_stationary_bootstrap_indices():
    rng = np.random.default_rng(0)
    t, b, block = 400, 300, 10.0
    idx = bs.stationary_bootstrap_indices(t, b, block, rng)
    assert idx.shape == (b, t) and idx.min() >= 0 and idx.max() < t
    consecutive = idx[:, 1:] == (idx[:, :-1] + 1) % t
    # A step continues the block with probability 1 - 1/block (plus a 1/T fluke).
    assert consecutive.mean() == pytest.approx(1 - 1 / block, abs=0.01)
    # Every row is (roughly) equally likely to be drawn.
    counts = bs.bootstrap_counts(idx, t)
    np.testing.assert_array_equal(counts.sum(axis=1), t)
    assert counts.mean(axis=0) == pytest.approx(np.ones(t), abs=0.5)
    # Wrap-around: row T-1 can be followed by row 0.
    assert np.any((idx[:, :-1] == t - 1) & (idx[:, 1:] == 0))


def test_bootstrap_means_equal_direct_resampling():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((120, 4))
    idx = bs.stationary_bootstrap_indices(120, 50, 8.0, np.random.default_rng(5))
    direct = np.stack([x[row].mean(axis=0) for row in idx])
    np.testing.assert_allclose(bs.bootstrap_means(x, 50, 8.0, seed=5), direct, atol=1e-12)


def test_long_run_variance():
    rng = np.random.default_rng(2)
    t = 100_000
    iid = rng.standard_normal((t, 1))
    assert bs.stationary_long_run_variance(iid, 10.0)[0] == pytest.approx(1.0, abs=0.05)
    # AR(1) with phi = 0.5 and unit innovation variance: gamma_0 = 4/3 and the kernel
    # weights (1-q)^i give omega^2 = gamma_0 * (1 + 2 * sum_i (0.95 * 0.5)^i) = 3.746
    # (the unweighted long-run variance would be 4.0).
    e = rng.standard_normal(t)
    ar = np.zeros(t)
    for i in range(1, t):
        ar[i] = 0.5 * ar[i - 1] + e[i]
    expected = (4 / 3) * (1 + 2 * 0.475 / (1 - 0.475))
    assert bs.stationary_long_run_variance(ar[:, None], 20.0)[0] == pytest.approx(
        expected, rel=0.04
    )


def test_kernel_variance_equals_bootstrap_variance():
    """The closed-form omega^2 is the variance the stationary bootstrap itself produces."""
    rng = np.random.default_rng(3)
    t = 600
    e = rng.standard_normal((t, 4))
    x = e.copy()
    for i in range(1, t):
        x[i] = 0.4 * x[i - 1] + e[i]
    boot = bs.bootstrap_means(x, 6000, 12.0, seed=9)
    boot_var = t * boot.var(axis=0)
    kernel_var = bs.stationary_long_run_variance(x, 12.0)
    np.testing.assert_allclose(kernel_var, boot_var, rtol=0.08)


def test_size_under_the_null():
    """With no skill anywhere, both tests should reject about 5% of the time."""
    rejections = {"rc": 0, "spa": 0}
    reps = 300
    for seed in range(reps):
        x = np.random.default_rng(1000 + seed).standard_normal((250, 30))
        res = bs.reality_check_and_spa(x, n_boot=200, mean_block=5.0, seed=seed)
        rejections["rc"] += res.rc_pvalue < 0.05
        rejections["spa"] += res.spa_consistent < 0.05
    assert 0.015 < rejections["rc"] / reps < 0.10
    assert 0.015 < rejections["spa"] / reps < 0.10


def test_power_and_pvalue_ordering():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((500, 40))
    x[:, 11] += 0.30  # one strategy with a per-period Sharpe of 0.3
    res = bs.reality_check_and_spa(x, n_boot=500, mean_block=5.0, seed=1)
    assert res.rc_pvalue < 0.01 and res.spa_consistent < 0.01
    assert res.best_mean_index == 11 and res.best_t_index == 11
    assert res.spa_lower <= res.spa_consistent <= res.spa_upper


def test_spa_is_not_fooled_by_terrible_strategies_but_reality_check_is():
    # One modestly good strategy plus many volatile, clearly bad ones. The bad ones
    # inflate the Reality Check's null distribution; SPA studentises and re-centres.
    rng = np.random.default_rng(5)
    t = 500
    good = 0.16 + rng.standard_normal((t, 1))
    bad = -0.5 + 8.0 * rng.standard_normal((t, 60))
    res = bs.reality_check_and_spa(np.hstack([good, bad]), n_boot=1000, mean_block=5.0, seed=2)
    assert res.spa_consistent < 0.05
    assert res.rc_pvalue > 0.2
    assert res.spa_consistent <= res.rc_pvalue


def test_never_trading_strategies_are_ignored():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((200, 10))
    padded = np.hstack([np.zeros((200, 3)), x])
    a = bs.reality_check_and_spa(x, 200, 5.0, seed=3)
    b = bs.reality_check_and_spa(padded, 200, 5.0, seed=3)
    assert a.spa_consistent == b.spa_consistent and b.best_t_index == a.best_t_index + 3
    assert a.spa_hac_consistent == b.spa_hac_consistent


def test_long_run_scale_is_noisier_and_inflates_the_maximum():
    """The E1 finding behind the default scale (see ``reality_check_and_spa``).

    For white noise both scales estimate 1, but the long-run estimate is far noisier,
    and the maximum over many strategies of mean/scale picks out the under-estimates.
    """
    extra = []
    for seed in range(20):
        x = np.random.default_rng(seed).standard_normal((1000, 400))
        sd = x.std(axis=0, ddof=1)
        omega = np.sqrt(bs.stationary_long_run_variance(x, 10.0))
        if seed == 0:
            assert omega.std() > 2.5 * sd.std()
        t_plain = np.sqrt(1000) * x.mean(axis=0) / sd
        t_long_run = np.sqrt(1000) * x.mean(axis=0) / omega
        extra.append(t_long_run.max() - t_plain.max())
    assert np.mean(extra) > 0.05  # the studentised maximum is biased upwards
