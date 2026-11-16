"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation.

Reference
---------
Bailey, D. H., Borwein, J. M., Lopez de Prado, M. and Zhu, Q. J. (2017). "The Probability
of Backtest Overfitting." Journal of Computational Finance 20(4), 39-69.

Algorithm (CSCV), for a T x N matrix of strategy returns:

1. Cut the rows into S contiguous blocks of equal length (S even).
2. For every way c of choosing S/2 blocks: the chosen blocks form the in-sample (IS) set,
   the other S/2 blocks the out-of-sample (OOS) set. There are C(S, S/2) such splits and
   both sets always have the same size (hence "symmetric").
3. On each split: find the strategy n* with the best IS Sharpe ratio, compute the OOS
   Sharpe ratio of every strategy, and take the relative OOS rank of n*,
   ``w = rank / (N + 1)`` in (0, 1) (1 = best OOS). The logit is ``lambda = ln(w/(1-w))``.
4. PBO = fraction of splits with ``lambda <= 0``: how often the IS winner lands in the
   bottom half OOS.

If selection carried real information the IS winner would stay near the top OOS
(lambda >> 0, PBO near 0). If selection is pure noise its OOS rank is uniform and
PBO is about 0.5.

Implementation note: the Sharpe ratio of a union of blocks only needs each block's sum
and sum of squares, so every split costs O(S N) instead of O(T N).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class PBOResult:
    pbo: float
    logits: np.ndarray  # one per split
    is_sharpe: np.ndarray  # IS Sharpe of the IS winner (per-period), one per split
    oos_sharpe: np.ndarray  # OOS Sharpe of the IS winner (per-period), one per split
    best_index: np.ndarray  # column index of the IS winner, one per split
    n_blocks: int
    n_splits: int
    block_length: int

    @property
    def prob_oos_loss(self) -> float:
        """Fraction of splits in which the IS winner has a negative OOS Sharpe ratio."""
        return float(np.mean(self.oos_sharpe < 0))

    @property
    def degradation(self) -> tuple[float, float]:
        """(slope, intercept) of the regression OOS Sharpe = a + b * IS Sharpe."""
        if np.ptp(self.is_sharpe) == 0:
            return 0.0, float(self.oos_sharpe.mean())
        slope, intercept = np.polyfit(self.is_sharpe, self.oos_sharpe, 1)
        return float(slope), float(intercept)


def _sharpe_from_sums(s1: np.ndarray, s2: np.ndarray, n: int) -> np.ndarray:
    """Sharpe ratio (ddof=1) from sums and sums of squares over ``n`` observations."""
    mean = s1 / n
    var = (s2 - n * mean**2) / (n - 1)
    ok = var > 1e-20
    return np.where(ok, mean / np.sqrt(np.where(ok, var, 1.0)), 0.0)


def cscv(returns: np.ndarray, n_blocks: int = 16, chunk: int = 256) -> PBOResult:
    """Run CSCV on a T x N return matrix and return the PBO and its logit distribution.

    Rows that do not fit into ``n_blocks`` equal blocks are dropped from the START of the
    sample. Ties in the OOS ranking receive the average rank.
    """
    if n_blocks % 2 or n_blocks < 2:
        raise ValueError("n_blocks must be an even number >= 2")
    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    length = t // n_blocks
    if length < 2:
        raise ValueError("not enough rows for this many blocks")
    x = x[t - length * n_blocks :]
    blocks = x.reshape(n_blocks, length, n)
    s1 = blocks.sum(axis=1)  # (S, N)
    s2 = np.square(blocks).sum(axis=1)
    tot1, tot2 = s1.sum(axis=0), s2.sum(axis=0)

    half = n_blocks // 2
    combos = np.array(list(combinations(range(n_blocks), half)))
    n_half = half * length
    logits, is_best, oos_best, best_idx = [], [], [], []
    for start in range(0, len(combos), chunk):
        sel = np.zeros((min(chunk, len(combos) - start), n_blocks))
        np.put_along_axis(sel, combos[start : start + chunk], 1.0, axis=1)
        is1, is2 = sel @ s1, sel @ s2
        sr_is = _sharpe_from_sums(is1, is2, n_half)
        sr_oos = _sharpe_from_sums(tot1 - is1, tot2 - is2, n_half)
        star = sr_is.argmax(axis=1)
        rows = np.arange(len(star))
        star_oos = sr_oos[rows, star]
        below = (sr_oos < star_oos[:, None]).sum(axis=1)
        equal = (sr_oos == star_oos[:, None]).sum(axis=1)  # includes n* itself
        rank = below + (equal + 1) / 2.0  # average rank, 1 = worst, N = best
        w = rank / (n + 1.0)
        logits.append(np.log(w / (1.0 - w)))
        is_best.append(sr_is[rows, star])
        oos_best.append(star_oos)
        best_idx.append(star)
    logits = np.concatenate(logits)
    return PBOResult(
        pbo=float(np.mean(logits <= 0)),
        logits=logits,
        is_sharpe=np.concatenate(is_best),
        oos_sharpe=np.concatenate(oos_best),
        best_index=np.concatenate(best_idx),
        n_blocks=n_blocks,
        n_splits=len(combos),
        block_length=length,
    )
