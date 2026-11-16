"""Rolling indicators.

Every function returns an array aligned with its input where element ``t`` depends only
on inputs ``0..t`` (never on later bars). Warm-up elements are NaN. The rolling
functions use ``sliding_window_view``: row ``j`` of the view is ``x[j : j + w]``, and the
result is written to position ``j + w - 1`` (the LAST bar of the window), which is what
makes them free of lookahead by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _rolling(x: np.ndarray, window: int, func) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= window:
        out[window - 1 :] = func(sliding_window_view(x, window), axis=1)
    return out


def sma(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average of the last ``window`` values (including the current one)."""
    return _rolling(x, window, np.mean)


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Sample standard deviation (ddof=1) of the last ``window`` values."""
    return _rolling(x, window, lambda v, axis: np.std(v, axis=axis, ddof=1))


def rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    return _rolling(x, window, np.max)


def rolling_min(x: np.ndarray, window: int) -> np.ndarray:
    return _rolling(x, window, np.min)


def ema(x: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average, ``y[t] = a x[t] + (1 - a) y[t-1]`` with ``a = 2/(span+1)``.

    The first ``span - 1`` values are set to NaN so that an unseasoned average is never
    traded on.
    """
    out = pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy(copy=True)
    out[: span - 1] = np.nan
    return out


def rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's Relative Strength Index in [0, 100].

    ``RSI = 100 - 100 / (1 + RS)``, ``RS = avg_gain / avg_loss`` where both averages are
    Wilder-smoothed (an EMA with ``alpha = 1/period``) of the positive and negative parts
    of the one-bar price change.
    """
    delta = np.diff(close, prepend=close[0])
    gain = pd.Series(np.clip(delta, 0.0, None)).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = pd.Series(np.clip(-delta, 0.0, None)).ewm(alpha=1.0 / period, adjust=False).mean()
    gain, loss = gain.to_numpy(), loss.to_numpy()
    total = gain + loss
    out = np.where(total > 0, 100.0 * gain / np.where(total > 0, total, 1.0), 50.0)
    out[:period] = np.nan
    return out


def simple_returns(close: np.ndarray) -> np.ndarray:
    """One-bar simple returns ``close[t]/close[t-1] - 1`` with ``r[0] = 0``."""
    r = np.zeros(len(close))
    r[1:] = close[1:] / close[:-1] - 1.0
    return r


def trailing_vol(returns: np.ndarray, window: int) -> np.ndarray:
    """Trailing standard deviation of returns over ``window`` bars, expanding at the start.

    Element ``t`` uses returns up to and including bar ``t``. Bar 0 carries the dummy
    return 0 and is excluded. Elements with fewer than two usable returns are 0.
    """
    out = rolling_std(returns, window)
    for t in range(min(window, len(returns))):  # expanding warm-up
        out[t] = np.std(returns[1 : t + 1], ddof=1) if t >= 2 else 0.0
    return out
