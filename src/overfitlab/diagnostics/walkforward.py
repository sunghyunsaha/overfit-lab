"""Naive in-sample selection and its out-of-sample decay.

Two views of the same phenomenon:

``split_decay``     cut the sample at one point; compare every variant's in-sample (IS)
                    Sharpe ratio with its out-of-sample (OOS) Sharpe ratio, and look at
                    what happens to the IS winner.
``walk_forward``    the realistic procedure: repeatedly pick the variant with the best
                    Sharpe ratio over the trailing ``train_bars``, trade it for the next
                    ``test_bars``, roll forward. The stitched OOS returns are what a
                    researcher who "just picks the best backtest" would have earned.

This is the baseline every other diagnostic is trying to predict without needing the
out-of-sample data.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from overfitlab.diagnostics.sharpe import sharpe_ratio


def split_decay(returns: np.ndarray, split: float = 0.5, top: int = 50) -> dict:
    """IS vs OOS Sharpe ratio of every variant for a single chronological split.

    Returns per-period Sharpe arrays plus summary statistics: the regression slope of OOS
    on IS Sharpe (1 = performance persists, 0 = pure noise), the Spearman rank
    correlation, and the fate of the IS winner and of the IS top ``top``.
    """
    x = np.asarray(returns, dtype=float)
    cut = int(x.shape[0] * split)
    sr_is, sr_oos = sharpe_ratio(x[:cut]), sharpe_ratio(x[cut:])
    live = (x[:cut].std(axis=0) > 0) & (x[cut:].std(axis=0) > 0)
    slope, intercept = np.polyfit(sr_is[live], sr_oos[live], 1)
    rank_corr = float(spearmanr(sr_is[live], sr_oos[live]).statistic)
    best = int(np.argmax(sr_is))
    order = np.argsort(sr_is)[::-1][:top]
    return {
        "cut": cut,
        "sr_is": sr_is,
        "sr_oos": sr_oos,
        "slope": float(slope),
        "intercept": float(intercept),
        "rank_corr": rank_corr,
        "best_index": best,
        "best_is": float(sr_is[best]),
        "best_oos": float(sr_oos[best]),
        "best_oos_percentile": float(np.mean(sr_oos[live] < sr_oos[best])),
        "top_is_mean": float(sr_is[order].mean()),
        "top_oos_mean": float(sr_oos[order].mean()),
        "median_oos": float(np.median(sr_oos[live])),
    }


def walk_forward(returns: np.ndarray, train_bars: int, test_bars: int) -> dict:
    """Rolling walk-forward selection of the best-Sharpe variant.

    Window j trains on rows ``[s_j, s_j + train_bars)`` and tests on the next
    ``test_bars`` rows; ``s_{j+1} = s_j + test_bars``. Only information from the training
    window is used to choose what is traded in the test window.
    """
    x = np.asarray(returns, dtype=float)
    t = x.shape[0]
    oos = np.zeros(t)
    covered = np.zeros(t, dtype=bool)
    picks, is_sr, oos_sr = [], [], []
    start = 0
    while start + train_bars + 2 <= t:
        train = slice(start, start + train_bars)
        test = slice(start + train_bars, min(start + train_bars + test_bars, t))
        sr_train = sharpe_ratio(x[train])
        best = int(np.argmax(sr_train))
        seg = x[test, best]
        oos[test] = seg
        covered[test] = True
        picks.append(best)
        is_sr.append(float(sr_train[best]))
        oos_sr.append(float(seg.mean() / seg.std(ddof=1)) if seg.std() > 0 else 0.0)
        start += test_bars
    stitched = oos[covered]
    sd = stitched.std(ddof=1) if stitched.size > 1 else 0.0
    return {
        "n_windows": len(picks),
        "picks": np.array(picks),
        "window_is_sharpe": np.array(is_sr),
        "window_oos_sharpe": np.array(oos_sr),
        "oos_returns": oos,
        "covered": covered,
        "oos_sharpe": float(stitched.mean() / sd) if sd > 0 else 0.0,
        "mean_is_sharpe": float(np.mean(is_sr)) if is_sr else 0.0,
        "frac_windows_oos_negative": float(np.mean(np.array(oos_sr) < 0)) if oos_sr else 0.0,
    }
