"""Effective number of independent trials."""

from __future__ import annotations

import numpy as np
import pytest

from overfitlab.diagnostics import trials as tr
from overfitlab.diagnostics.sharpe import expected_max_z


def test_identical_strategies_count_as_one_trial():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((500, 1))
    x = np.hstack([base * s for s in np.linspace(0.5, 2.0, 40)])  # perfectly correlated
    out = tr.effective_trials(x, n_sim=500)
    assert out["raw"] == 40
    assert out["avg_corr"] == pytest.approx(1.0, abs=1e-3)
    assert out["eigen"] == pytest.approx(1.0, abs=1e-3)
    assert out["max_matching"] < 1.6


def test_independent_strategies_count_fully():
    x = np.random.default_rng(1).standard_normal((3000, 150))
    out = tr.effective_trials(x, n_sim=3000)
    assert out["avg_corr"] == pytest.approx(150, rel=0.02)
    assert out["eigen"] == pytest.approx(150 / (1 + 150 / 3000), rel=0.03)  # sampling noise
    assert out["max_matching"] == pytest.approx(150, rel=0.30)


def test_block_structure_is_recovered():
    # 12 independent "bets", each expressed by 25 near-identical variants.
    rng = np.random.default_rng(2)
    bets = rng.standard_normal((2000, 12))
    x = np.repeat(bets, 25, axis=1) + 0.15 * rng.standard_normal((2000, 300))
    out = tr.effective_trials(x, n_sim=3000)
    assert out["cluster"] == 12
    assert 10 < out["eigen"] < 14
    assert 12 * 0.6 < out["max_matching"] < 12 * 3  # a bit above 12: variants are not identical
    assert out["max_matching"] < out["avg_corr"] < 300


def test_inverse_of_expected_max():
    for n in (2.0, 37.5, 800.0, 12345.0):
        assert tr.invert_expected_max(expected_max_z(n)) == pytest.approx(n, rel=1e-6)
    assert tr.invert_expected_max(-1.0) == 1.0


def test_constant_columns_are_dropped():
    x = np.random.default_rng(3).standard_normal((300, 20))
    x[:, :4] = 0.0
    assert tr.effective_trials(x, methods=("avg_corr",))["raw"] == 16
