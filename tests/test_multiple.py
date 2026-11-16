"""Multiple-testing adjustments against statsmodels, and the haircut Sharpe ratio."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from overfitlab.diagnostics import multiple as mt


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adjusted_pvalues_match_statsmodels(seed):
    rng = np.random.default_rng(seed)
    p = np.r_[rng.uniform(size=180), rng.uniform(size=20) * 1e-4]
    rng.shuffle(p)
    for ours, theirs in (
        ("bonferroni", "bonferroni"),
        ("holm", "holm"),
        ("bh", "fdr_bh"),
        ("by", "fdr_by"),
    ):
        expected = multipletests(p, alpha=0.05, method=theirs)[1]
        np.testing.assert_allclose(mt.ADJUSTMENTS[ours](p), expected, atol=1e-12)


def test_hand_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.60])
    np.testing.assert_allclose(mt.bonferroni(p), [0.005, 0.04, 0.195, 0.205, 1.0])
    # Holm: 5*0.001, 4*0.008, 3*0.039, 2*0.041 (-> max with previous), 1*0.60
    np.testing.assert_allclose(mt.holm(p), [0.005, 0.032, 0.117, 0.117, 0.60])
    # BH: 5/1*0.001, 5/2*0.008, 5/3*0.039, 5/4*0.041, 5/5*0.60 with running min from the right
    np.testing.assert_allclose(mt.benjamini_hochberg(p), [0.005, 0.02, 0.05125, 0.05125, 0.60])
    assert mt.survivors(p, 0.05) == {"naive": 4, "bonferroni": 2, "holm": 2, "bh": 2, "by": 2}


def test_ordering_of_strictness():
    p = np.random.default_rng(3).uniform(size=500) ** 3
    s = mt.survivors(p, 0.05)
    assert s["bonferroni"] <= s["holm"] <= s["bh"] <= s["naive"]
    assert s["by"] <= s["bh"]


def test_haircut_sharpe_bonferroni_by_hand():
    # Annualised Sharpe 1.0 over 9 years: t = 3.0, two-sided p = 0.0026998.
    # Bonferroni with 10 tests: p = 0.026998 -> t = 2.2114 -> Sharpe 0.7371, haircut 26.3%.
    res = mt.haircut_sharpe(1.0, n_obs=9 * 365, periods_per_year=365, n_tests=10)
    p_single = 2 * (1 - norm.cdf(3.0))
    assert res["single"]["p_adjusted"] == pytest.approx(p_single)
    t_adj = norm.ppf(1 - 10 * p_single / 2)
    assert res["bonferroni"]["sharpe_annual"] == pytest.approx(t_adj / 3.0)
    assert res["bonferroni"]["sharpe_annual"] == pytest.approx(0.7371, abs=1e-3)
    assert res["bonferroni"]["haircut"] == pytest.approx(0.263, abs=1e-3)


def test_haircut_uses_all_trials_and_can_wipe_out_the_sharpe():
    rng = np.random.default_rng(4)
    years = 8
    all_sr = rng.standard_normal(2000) / math.sqrt(years)  # 2000 skill-less strategies
    best = float(all_sr.max())
    res = mt.haircut_sharpe(best, years * 365, 365, all_sr_annual=all_sr)
    for method in ("bonferroni", "holm", "bhy"):
        assert 0.0 <= res[method]["sharpe_annual"] < best
        assert res[method]["haircut"] > 0.5  # the best of 2000 noise strategies is mostly luck
    assert res["holm"]["p_adjusted"] <= res["bonferroni"]["p_adjusted"] + 1e-12
