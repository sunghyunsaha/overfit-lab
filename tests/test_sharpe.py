"""PSR, DSR, MinTRL and MinBTL against paper numbers and independent calculations."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from overfitlab.diagnostics import sharpe as sh


def _phi(z: float) -> float:
    """Standard normal CDF written out independently of the code under test."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def test_moments_match_scipy():
    rng = np.random.default_rng(0)
    x = rng.gamma(2.0, 1.0, size=(500, 3)) - 1.5
    m = sh.return_moments(x)
    np.testing.assert_allclose(m["skew"], stats.skew(x, axis=0))
    np.testing.assert_allclose(m["kurt"], stats.kurtosis(x, axis=0, fisher=False))
    np.testing.assert_allclose(m["sharpe"], x.mean(0) / x.std(0, ddof=1))
    flat = sh.return_moments(np.zeros((10, 2)))
    assert np.all(flat["sharpe"] == 0) and np.all(flat["kurt"] == 3) and not flat["live"].any()


def test_psr_hand_calculation_normal_returns():
    # SR = 0.1 per period, T = 101, normal returns:
    # se = sqrt((1 + 0.1^2 / 2) / 100) = 0.1002497,  z = 0.1 / se = 0.997509
    se = math.sqrt((1 + 0.005) / 100)
    assert sh.sharpe_std_error(0.1, 101) == pytest.approx(se, rel=1e-12)
    assert sh.probabilistic_sharpe_ratio(0.1, 101) == pytest.approx(_phi(0.1 / se), abs=1e-12)
    assert sh.probabilistic_sharpe_ratio(0.1, 101) == pytest.approx(0.8407, abs=1e-4)


def test_psr_hand_calculation_non_normal_returns():
    # SR = 0.1, skew = -1, kurtosis = 6, T = 1001, benchmark SR* = 0.02:
    # variance = (1 - (-1)(0.1) + (6-1)/4 * 0.01) / 1000 = 1.1125e-3
    se = math.sqrt(1.1125e-3)
    expected = _phi((0.1 - 0.02) / se)
    got = sh.probabilistic_sharpe_ratio(0.1, 1001, skew=-1.0, kurt=6.0, sr_benchmark=0.02)
    assert got == pytest.approx(expected, abs=1e-12)
    # Negative skew and fat tails make the same Sharpe ratio LESS convincing.
    assert got < sh.probabilistic_sharpe_ratio(0.1, 1001, sr_benchmark=0.02)


def test_dsr_worked_example_from_bailey_lopez_de_prado_2014():
    # The paper's numerical example: annualised SR 2.5 from T = 1250 daily returns
    # (250 per year), skewness -3, kurtosis 10, best of N = 100 independent trials whose
    # annualised Sharpe ratios have variance 1/2.  Paper: SR0 = 0.1132 (daily),
    # DSR = 0.9004, i.e. NOT significant at 95% despite an annualised Sharpe of 2.5.
    sr = 2.5 / math.sqrt(250)
    sr_std = math.sqrt(0.5 / 250)
    dsr, sr0 = sh.deflated_sharpe_ratio(sr, 1250, -3.0, 10.0, 100, sr_std)
    assert sr0 == pytest.approx(0.1132, abs=1e-4)
    assert dsr == pytest.approx(0.9004, abs=1e-3)
    assert dsr < 0.95
    # Without deflation the same track record looks overwhelming.
    assert sh.probabilistic_sharpe_ratio(sr, 1250, -3.0, 10.0) > 0.9999


def test_expected_max_matches_simulation():
    rng = np.random.default_rng(1)
    # The formula is an asymptotic approximation: about 2.5% high at N = 10 and within
    # 1.5% from N = 100 upwards.
    for n, tol in ((10, 0.035), (100, 0.015), (1000, 0.015)):
        sim = rng.standard_normal((20_000, n)).max(axis=1).mean()
        assert sh.expected_max_z(n) == pytest.approx(sim, rel=tol)
        assert sh.expected_max_z(n) > sim * 0.999  # it errs on the conservative side
    assert sh.expected_max_z(1) == 0.0
    assert sh.expected_max_sharpe(100, 0.05) == pytest.approx(0.05 * sh.expected_max_z(100))


def test_min_backtest_length_paper_example():
    # Bailey, Borwein, Lopez de Prado & Zhu (2014): with 5 years of data, no more than
    # about 45 independent configurations should be tried, or an annualised in-sample
    # Sharpe ratio of 1 is expected from pure noise.
    assert sh.min_backtest_length(45, 1.0) == pytest.approx(5.0, abs=0.05)
    assert sh.max_trials_for_length(5.0, 1.0) == pytest.approx(45, abs=1.5)
    for n in (10, 45, 1000, 5000):  # the paper's upper bound 2 ln(N) / SR^2
        assert sh.min_backtest_length(n, 1.0) < 2 * math.log(n)
    assert sh.min_backtest_length(1000, 2.0) == pytest.approx(sh.min_backtest_length(1000, 1.0) / 4)


def test_min_track_record_length_is_where_psr_hits_the_confidence_level():
    sr, skew, kurt, bench = 0.08, -0.7, 5.5, 0.03
    n = sh.min_track_record_length(sr, skew, kurt, bench, alpha=0.05)
    assert sh.probabilistic_sharpe_ratio(sr, n, skew, kurt, bench) == pytest.approx(0.95, abs=1e-9)
    assert sh.min_track_record_length(0.02, sr_benchmark=0.03) == float("inf")


def test_psr_is_calibrated_by_monte_carlo():
    """If the true Sharpe equals the benchmark, PSR > 0.95 should happen ~5% of the time."""
    rng = np.random.default_rng(2)
    t, reps, true_sr = 600, 4000, 0.08
    # Skewed, fat-tailed returns: a centred gamma shifted to have Sharpe exactly true_sr.
    shape = 4.0
    x = (rng.gamma(shape, 1.0, size=(t, reps)) - shape) / math.sqrt(shape) + true_sr
    m = sh.return_moments(x)
    psr = sh.probabilistic_sharpe_ratio(m["sharpe"], t, m["skew"], m["kurt"], true_sr)
    rate = np.mean(psr > 0.95)
    assert 0.035 < rate < 0.07


def test_null_pvalue_is_the_t_test_and_psr_overstates_sparse_strategies():
    """Regression test for the E1 finding (see ``sharpe_pvalue``)."""
    assert sh.sharpe_pvalue(0.0, 500) == pytest.approx(0.5)
    assert sh.sharpe_pvalue(0.1, 401) == pytest.approx(1 - _phi(0.1 * 20.0), rel=1e-9)
    # 4000 skill-less strategies that each hold a position on only 6 of 2000 bars.
    rng = np.random.default_rng(5)
    t, n, active = 2000, 4000, 6
    x = np.zeros((t, n))
    for j in range(n):
        x[rng.choice(t, active, replace=False), j] = rng.standard_normal(active)
    m = sh.return_moments(x)
    p_null = sh.sharpe_pvalue(m["sharpe"], t)
    p_psr = 1 - sh.probabilistic_sharpe_ratio(m["sharpe"], t, m["skew"], m["kurt"])
    bonferroni = 0.05 / n
    assert (p_null < bonferroni).sum() == 0  # the t-statistic of 6 points is at most sqrt(6)
    assert (p_psr < bonferroni).sum() >= 5  # in-sample skewness fakes significance
    # For strategies that trade all the time the two p-values agree closely.
    dense = sh.return_moments(rng.standard_normal((t, 200)))
    p1 = sh.sharpe_pvalue(dense["sharpe"], t)
    p2 = 1 - sh.probabilistic_sharpe_ratio(dense["sharpe"], t, dense["skew"], dense["kurt"])
    np.testing.assert_allclose(p1, p2, atol=5e-3)
