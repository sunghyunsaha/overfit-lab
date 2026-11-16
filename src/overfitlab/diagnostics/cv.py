"""Purged k-fold and Combinatorial Purged Cross-Validation (CPCV) with embargo.

Reference
---------
Lopez de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley.
Chapter 7 (purged k-fold, embargo) and chapter 12 (CPCV, backtest paths).

Why ordinary k-fold fails on time series: observation i carries information over a
window of time [i, label_end[i]] (here: positions persist and indicators overlap, so
neighbouring observations are not independent). If a training observation's window
overlaps a test observation's window, the "out-of-sample" test is contaminated.

* PURGING removes from the training set every observation whose window overlaps the
  window of any test observation.
* The EMBARGO additionally removes the ``embargo`` observations that follow each test
  block, because serial dependence leaks forward in time past the label window.

In this project the "model" being trained is the selection rule "pick the variant with
the best training-set Sharpe ratio", and it is evaluated on the test set.

CPCV: split the sample into N contiguous groups and use every combination of k groups as
the test set. That gives C(N, k) splits, and because each group is tested
C(N-1, k-1) times, the test results can be reassembled into C(N-1, k-1) complete
backtest PATHS, each covering the whole sample out-of-sample. One historical path
becomes a distribution of out-of-sample Sharpe ratios.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np


def label_ends(n_obs: int, horizon: int) -> np.ndarray:
    """``label_end[i] = min(i + horizon, n_obs - 1)``: last bar observation i depends on."""
    return np.minimum(np.arange(n_obs) + int(horizon), n_obs - 1)


def contiguous_blocks(indices: np.ndarray) -> list[tuple[int, int]]:
    """Split sorted indices into runs of consecutive integers, as (first, last) pairs."""
    idx = np.asarray(indices)
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def purged_train_indices(
    n_obs: int, test_indices: np.ndarray, label_end: np.ndarray, embargo: int = 0
) -> np.ndarray:
    """Training indices left after purging and embargoing around ``test_indices``.

    For each contiguous test block [a, b] let e = max(label_end[a..b]) be the last bar
    any test label depends on. A candidate training observation i is dropped if

    * it is a test observation, or
    * its window [i, label_end[i]] overlaps [a, e]            (purging), or
    * e < i <= e + embargo                                    (embargo).
    """
    keep = np.ones(n_obs, dtype=bool)
    obs = np.arange(n_obs)
    for a, b in contiguous_blocks(np.sort(test_indices)):
        e = int(label_end[a : b + 1].max())
        overlap = (obs <= e) & (label_end >= a)
        embargoed = (obs > e) & (obs <= e + embargo)
        keep &= ~(overlap | embargoed)
    return obs[keep]


def group_bounds(n_obs: int, n_groups: int) -> list[np.ndarray]:
    """Indices of ``n_groups`` contiguous, (almost) equally sized groups."""
    return [g for g in np.array_split(np.arange(n_obs), n_groups)]


def purged_kfold(
    n_obs: int, n_splits: int, label_horizon: int = 0, embargo: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged k-fold splits: each contiguous fold is the test set once."""
    ends = label_ends(n_obs, label_horizon)
    return [
        (purged_train_indices(n_obs, test, ends, embargo), test)
        for test in group_bounds(n_obs, n_splits)
    ]


@dataclass
class CPCVSplit:
    test_groups: tuple[int, ...]
    train: np.ndarray
    test: np.ndarray


def cpcv_splits(
    n_obs: int, n_groups: int, n_test_groups: int, label_horizon: int = 0, embargo: int = 0
) -> list[CPCVSplit]:
    """All C(n_groups, n_test_groups) combinatorial purged splits."""
    groups = group_bounds(n_obs, n_groups)
    ends = label_ends(n_obs, label_horizon)
    out = []
    for combo in combinations(range(n_groups), n_test_groups):
        test = np.concatenate([groups[g] for g in combo])
        out.append(CPCVSplit(combo, purged_train_indices(n_obs, test, ends, embargo), test))
    return out


def n_cpcv_paths(n_groups: int, n_test_groups: int) -> int:
    """Number of backtest paths: phi = k/N * C(N, k) = C(N-1, k-1)."""
    return comb(n_groups - 1, n_test_groups - 1)


def cpcv_path_assignments(n_groups: int, n_test_groups: int) -> np.ndarray:
    """``paths[p, g]`` = index of the split that supplies group g's test data in path p.

    Group g is a test group in C(N-1, k-1) splits; the j-th of them (in split order)
    feeds path j. Every (split, test group) pair is used exactly once.
    """
    combos = list(combinations(range(n_groups), n_test_groups))
    paths = np.full((n_cpcv_paths(n_groups, n_test_groups), n_groups), -1, dtype=int)
    used = np.zeros(n_groups, dtype=int)
    for s, combo in enumerate(combos):
        for g in combo:
            paths[used[g], g] = s
            used[g] += 1
    return paths


def _sharpe_rows(weights: np.ndarray, x: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Sharpe ratio of every column of x over the rows selected by each 0/1 weight row."""
    n = weights.sum(axis=1, keepdims=True)
    mean = (weights @ x) / n
    var = ((weights @ x2) - n * mean**2) / (n - 1)
    ok = var > 1e-20
    return np.where(ok, mean / np.sqrt(np.where(ok, var, 1.0)), 0.0)


def cpcv_select_best(
    returns: np.ndarray,
    n_groups: int = 10,
    n_test_groups: int = 2,
    label_horizon: int = 0,
    embargo: int = 0,
) -> dict:
    """CPCV evaluation of the rule "trade the variant with the best training Sharpe".

    Returns per-split and per-path statistics (per-period Sharpe ratios):

    ``split_is_sharpe``   training Sharpe of the selected variant, one per split
    ``split_oos_sharpe``  test Sharpe of the selected variant, one per split
    ``path_sharpe``       Sharpe of each reassembled out-of-sample path
    ``path_returns``      (n_paths, T) out-of-sample returns of each path
    """
    x = np.asarray(returns, dtype=float)
    t = x.shape[0]
    x2 = np.square(x)
    splits = cpcv_splits(t, n_groups, n_test_groups, label_horizon, embargo)
    groups = group_bounds(t, n_groups)

    w_train = np.zeros((len(splits), t))
    w_test = np.zeros((len(splits), t))
    for s, sp in enumerate(splits):
        w_train[s, sp.train] = 1.0
        w_test[s, sp.test] = 1.0
    sr_train = _sharpe_rows(w_train, x, x2)
    chosen = sr_train.argmax(axis=1)
    rows = np.arange(len(splits))
    sr_test = _sharpe_rows(w_test, x, x2)[rows, chosen]

    paths = cpcv_path_assignments(n_groups, n_test_groups)
    path_returns = np.empty((paths.shape[0], t))
    for p in range(paths.shape[0]):
        for g, idx in enumerate(groups):
            path_returns[p, idx] = x[idx, chosen[paths[p, g]]]
    sd = path_returns.std(axis=1, ddof=1)
    path_sharpe = np.where(sd > 0, path_returns.mean(axis=1) / np.where(sd > 0, sd, 1.0), 0.0)
    return {
        "n_splits": len(splits),
        "n_paths": int(paths.shape[0]),
        "chosen": chosen,
        "split_is_sharpe": sr_train[rows, chosen],
        "split_oos_sharpe": sr_test,
        "path_sharpe": path_sharpe,
        "path_returns": path_returns,
        "train_sizes": w_train.sum(axis=1),
    }
