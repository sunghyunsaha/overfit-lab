"""Walk-forward selection and IS/OOS decay."""

from __future__ import annotations

import numpy as np

from overfitlab.diagnostics.walkforward import split_decay, walk_forward


def test_noise_shows_full_decay():
    x = 0.01 * np.random.default_rng(0).standard_normal((2000, 400))
    d = split_decay(x)
    assert d["best_is"] > 0.05  # the in-sample winner looks great...
    assert abs(d["slope"]) < 0.15 and abs(d["rank_corr"]) < 0.15  # ...but nothing persists
    assert d["top_is_mean"] > 5 * abs(d["top_oos_mean"])
    wf = walk_forward(x, train_bars=500, test_bars=100)
    assert wf["n_windows"] == 15
    assert wf["mean_is_sharpe"] > 0.08 and abs(wf["oos_sharpe"]) < 0.06


def test_real_edge_persists():
    rng = np.random.default_rng(1)
    x = 0.01 * rng.standard_normal((2000, 100))
    x[:, 42] += 0.003
    d = split_decay(x)
    assert d["best_index"] == 42 and d["best_oos"] > 0.2 and d["best_oos_percentile"] > 0.98
    wf = walk_forward(x, train_bars=500, test_bars=100)
    assert np.all(wf["picks"] == 42)
    np.testing.assert_allclose(wf["oos_returns"][500:], x[500:, 42])
    assert not wf["covered"][:500].any() and wf["covered"][500:].all()


def test_walk_forward_uses_only_training_rows():
    rng = np.random.default_rng(2)
    x = 0.01 * rng.standard_normal((900, 50))
    a = walk_forward(x, 300, 60)
    y = x.copy()
    y[660:] = 0.01 * rng.standard_normal((240, 50))  # rewrite the future
    b = walk_forward(y, 300, 60)
    n_same = (660 - 300) // 60  # windows whose training data ends at or before row 660
    np.testing.assert_array_equal(a["picks"][: n_same + 1], b["picks"][: n_same + 1])
