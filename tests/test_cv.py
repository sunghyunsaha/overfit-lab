"""Purged k-fold and CPCV: no leakage, correct counts, correct path assembly."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from overfitlab.diagnostics import cv


def _assert_no_leakage(n_obs, train, test, horizon, embargo):
    """Brute force: no training window may touch a test window or its embargo."""
    ends = cv.label_ends(n_obs, horizon)
    train_set = set(train.tolist())
    assert train_set.isdisjoint(test.tolist())
    for a, b in cv.contiguous_blocks(np.sort(test)):
        last_test_info = max(ends[j] for j in range(a, b + 1))
        for i in train:
            # purging: window [i, ends[i]] must not overlap [a, last_test_info]
            assert ends[i] < a or i > last_test_info, f"train {i} overlaps test block {a}-{b}"
            # embargo: nothing in (last_test_info, last_test_info + embargo]
            assert not (last_test_info < i <= last_test_info + embargo), f"train {i} in embargo"


@pytest.mark.parametrize(
    ("n_obs", "n_splits", "horizon", "embargo"),
    [(200, 5, 0, 0), (200, 5, 7, 0), (203, 6, 4, 9), (150, 10, 12, 5), (97, 4, 30, 3)],
)
def test_purged_kfold_no_training_observation_overlaps_a_test_label_window(
    n_obs, n_splits, horizon, embargo
):
    splits = cv.purged_kfold(n_obs, n_splits, horizon, embargo)
    assert len(splits) == n_splits
    all_test = np.sort(np.concatenate([test for _, test in splits]))
    np.testing.assert_array_equal(all_test, np.arange(n_obs))  # each obs tested exactly once
    for train, test in splits:
        _assert_no_leakage(n_obs, train, test, horizon, embargo)


@pytest.mark.parametrize(
    ("n_obs", "n_groups", "k", "horizon", "embargo"),
    [(240, 6, 2, 5, 4), (300, 10, 2, 20, 3), (181, 8, 3, 6, 6), (120, 6, 1, 3, 2)],
)
def test_cpcv_no_training_observation_overlaps_a_test_label_window(
    n_obs, n_groups, k, horizon, embargo
):
    splits = cv.cpcv_splits(n_obs, n_groups, k, horizon, embargo)
    assert len(splits) == comb(n_groups, k)
    for sp in splits:
        _assert_no_leakage(n_obs, sp.train, sp.test, horizon, embargo)
        assert len(sp.train) > 0


def test_purging_removes_exactly_the_expected_observations():
    # 100 obs, test block = [40, 59], horizon 5, embargo 3.
    ends = cv.label_ends(100, 5)
    train = cv.purged_train_indices(100, np.arange(40, 60), ends, embargo=3)
    # Purged before: 35..39 (their windows reach bar 40). Test labels reach bar 64, so
    # 60..64 are purged and 65..67 embargoed. First training obs after the block: 68.
    np.testing.assert_array_equal(train, np.r_[0:35, 68:100])
    # Without purging or embargo only the test block itself is removed.
    plain = cv.purged_train_indices(100, np.arange(40, 60), cv.label_ends(100, 0), embargo=0)
    np.testing.assert_array_equal(plain, np.r_[0:40, 60:100])


def test_cpcv_path_structure():
    n_groups, k = 6, 2
    paths = cv.cpcv_path_assignments(n_groups, k)
    assert paths.shape == (cv.n_cpcv_paths(n_groups, k), n_groups) == (5, 6)
    assert cv.n_cpcv_paths(10, 2) == 9 and cv.n_cpcv_paths(6, 3) == 10
    splits = cv.cpcv_splits(120, n_groups, k)
    used = set()
    for p in range(paths.shape[0]):
        for g in range(n_groups):
            s = paths[p, g]
            assert g in splits[s].test_groups  # the split really tests this group
            used.add((s, g))
    assert len(used) == comb(n_groups, k) * k  # every (split, test group) used exactly once


def test_cpcv_select_best_matches_direct_computation():
    rng = np.random.default_rng(0)
    x = 0.01 * rng.standard_normal((300, 12))
    res = cv.cpcv_select_best(x, n_groups=6, n_test_groups=2, label_horizon=5, embargo=4)
    splits = cv.cpcv_splits(300, 6, 2, 5, 4)
    assert res["n_splits"] == 15 and res["n_paths"] == 5
    for s, sp in enumerate(splits):
        tr = x[sp.train]
        sr = tr.mean(0) / tr.std(0, ddof=1)
        best = int(np.argmax(sr))
        assert res["chosen"][s] == best
        assert res["split_is_sharpe"][s] == pytest.approx(sr[best])
        te = x[sp.test, best]
        assert res["split_oos_sharpe"][s] == pytest.approx(te.mean() / te.std(ddof=1))
    # Each path covers the whole sample exactly once.
    assert res["path_returns"].shape == (5, 300)


def test_cpcv_recovers_a_dominant_strategy():
    rng = np.random.default_rng(1)
    x = 0.01 * rng.standard_normal((400, 30))
    x[:, 3] += 0.01
    res = cv.cpcv_select_best(x, 8, 2, 10, 4)
    assert np.all(res["chosen"] == 3)
    np.testing.assert_allclose(res["path_returns"], np.tile(x[:, 3], (res["n_paths"], 1)))
