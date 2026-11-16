"""CSCV / PBO against an independent brute-force implementation and known cases."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from overfitlab.diagnostics.pbo import cscv


def _reference_pbo(x: np.ndarray, n_blocks: int) -> tuple[float, list[float]]:
    """Slow, literal implementation of the paper's algorithm."""
    t, n = x.shape
    length = t // n_blocks
    x = x[t - length * n_blocks :]
    blocks = [x[i * length : (i + 1) * length] for i in range(n_blocks)]
    logits = []
    for combo in combinations(range(n_blocks), n_blocks // 2):
        rest = [i for i in range(n_blocks) if i not in combo]
        ins = np.vstack([blocks[i] for i in combo])
        oos = np.vstack([blocks[i] for i in rest])
        sr_is = ins.mean(0) / ins.std(0, ddof=1)
        sr_oos = oos.mean(0) / oos.std(0, ddof=1)
        star = int(np.argmax(sr_is))
        rank = 1 + int(np.sum(sr_oos < sr_oos[star]))  # no ties with continuous data
        w = rank / (n + 1)
        logits.append(float(np.log(w / (1 - w))))
    return float(np.mean(np.array(logits) <= 0)), logits


@pytest.mark.parametrize("n_blocks", [4, 6, 8])
def test_matches_brute_force(n_blocks):
    rng = np.random.default_rng(n_blocks)
    x = 0.01 * rng.standard_normal((243, 17)) + 0.001 * rng.standard_normal(17)
    ref_pbo, ref_logits = _reference_pbo(x, n_blocks)
    res = cscv(x, n_blocks)
    assert res.pbo == pytest.approx(ref_pbo, abs=1e-12)
    np.testing.assert_allclose(res.logits, ref_logits, atol=1e-9)
    assert res.n_splits == len(ref_logits)


def test_number_of_splits():
    x = np.random.default_rng(0).standard_normal((320, 5))
    assert cscv(x, 16).n_splits == 12870  # C(16, 8)
    assert cscv(x, 10).n_splits == 252


def test_a_genuinely_superior_strategy_gives_pbo_zero():
    rng = np.random.default_rng(1)
    x = 0.01 * rng.standard_normal((800, 50))
    x[:, 7] += 0.004  # one strategy with real, persistent edge
    res = cscv(x, 8)
    assert res.pbo == 0.0
    assert np.all(res.best_index == 7)
    assert res.prob_oos_loss == 0.0


def test_pure_noise_gives_pbo_near_one_half():
    pbos = []
    for seed in range(12):
        x = np.random.default_rng(100 + seed).standard_normal((800, 100))
        pbos.append(cscv(x, 8).pbo)
    assert 0.35 < np.mean(pbos) < 0.65


def test_anti_persistent_performance_gives_pbo_one():
    # Strategy k is brilliant in block k and poor everywhere else, so whatever wins
    # in-sample must lose out-of-sample.
    rng = np.random.default_rng(2)
    n_blocks, length = 4, 50
    x = 0.001 * rng.standard_normal((n_blocks * length, n_blocks)) - 0.01
    for k in range(n_blocks):
        x[k * length : (k + 1) * length, k] += 0.05
    res = cscv(x, n_blocks)
    assert res.pbo == 1.0
    assert res.prob_oos_loss == 1.0
    assert np.all(res.is_sharpe > 0) and np.all(res.oos_sharpe < 0)


def test_never_trading_strategies_do_not_break_it():
    rng = np.random.default_rng(3)
    x = 0.01 * rng.standard_normal((400, 20))
    x[:, :5] = 0.0
    res = cscv(x, 8)
    assert np.isfinite(res.logits).all() and 0.0 <= res.pbo <= 1.0


def test_rejects_odd_block_counts():
    with pytest.raises(ValueError):
        cscv(np.zeros((100, 3)), 5)
