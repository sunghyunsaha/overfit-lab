"""Strategy families.

Each family is a function ``f(px, **params) -> positions`` where ``px`` is a
``PriceData`` (one asset) and ``positions[t]`` is the target position in {-1, 0, +1}
decided at the close of bar ``t`` from data up to and including bar ``t``. The engine
then applies the one-bar execution lag, so nothing here may look ahead.

Stateful rules (enter on one condition, exit on another) are vectorised with an
"event + forward fill" trick: write +1/-1 where an entry fires, 0 where an exit fires,
NaN elsewhere, then carry the last event forward (``_hold_last_event``).

``mode`` is ``"long_only"`` (short signals become flat) or ``"long_short"``.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

from overfitlab.backtest import indicators as ind


@dataclass
class PriceData:
    """One asset's bars plus a cache so indicators are computed once per window."""

    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    _cache: dict = field(default_factory=dict, repr=False)

    def cached(self, key: tuple, func: Callable[[], np.ndarray]) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = func()
        return self._cache[key]

    @property
    def log_close(self) -> np.ndarray:
        return self.cached(("log_close",), lambda: np.log(self.close))

    @property
    def returns(self) -> np.ndarray:
        return self.cached(("returns",), lambda: ind.simple_returns(self.close))


def _hold_last_event(events: np.ndarray) -> np.ndarray:
    """Forward-fill an event array (NaN = no event); the state before any event is 0."""
    n = len(events)
    seen = ~np.isnan(events)
    last = np.maximum.accumulate(np.where(seen, np.arange(n), -1))
    return np.where(last >= 0, events[np.maximum(last, 0)], 0.0)


def _two_legs(
    long_entry: np.ndarray,
    long_exit: np.ndarray,
    short_entry: np.ndarray,
    short_exit: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Combine independent long and short legs, each an entry/exit state machine."""
    long_events = np.full(len(long_entry), np.nan)
    long_events[long_exit] = 0.0
    long_events[long_entry] = 1.0
    pos = _hold_last_event(long_events)
    if mode == "long_short":
        short_events = np.full(len(short_entry), np.nan)
        short_events[short_exit] = 0.0
        short_events[short_entry] = -1.0
        pos = pos + _hold_last_event(short_events)
    elif mode != "long_only":
        raise ValueError(f"unknown mode {mode!r}")
    return pos


def _directional(score: np.ndarray, mode: str) -> np.ndarray:
    """Long when ``score > 0``; short (or flat in long-only mode) when ``score < 0``."""
    valid = ~np.isnan(score)
    safe = np.where(valid, score, 0.0)
    pos = np.sign(safe)
    if mode == "long_only":
        pos = np.clip(pos, 0.0, None)
    elif mode != "long_short":
        raise ValueError(f"unknown mode {mode!r}")
    return pos


def ma_cross(px: PriceData, fast: int, slow: int, ma_type: str, mode: str) -> np.ndarray:
    """Moving-average crossover: long while the fast average is above the slow one."""
    func = {"sma": ind.sma, "ema": ind.ema}[ma_type]
    f = px.cached((ma_type, fast), lambda: func(px.close, fast))
    s = px.cached((ma_type, slow), lambda: func(px.close, slow))
    return _directional(f - s, mode)


def tsmom(px: PriceData, lookback: int, mode: str) -> np.ndarray:
    """Time-series momentum: the sign of the asset's own return over ``lookback`` bars."""
    mom = np.full(len(px.close), np.nan)
    mom[lookback:] = px.close[lookback:] / px.close[:-lookback] - 1.0
    return _directional(mom, mode)


def zscore_mr(px: PriceData, window: int, entry: float, exit: float, mode: str) -> np.ndarray:
    """Z-score mean reversion on the log price.

    ``z = (log close - rolling mean) / rolling std``. Go long when ``z < -entry`` and close
    the long when ``z >= -exit``; go short when ``z > entry`` and cover when ``z <= exit``.
    """
    mean = px.cached(("sma_log", window), lambda: ind.sma(px.log_close, window))
    std = px.cached(("std_log", window), lambda: ind.rolling_std(px.log_close, window))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(std > 0, (px.log_close - mean) / std, np.nan)
    ok = ~np.isnan(z)
    zz = np.where(ok, z, 0.0)
    return _two_legs(
        long_entry=ok & (zz < -entry),
        long_exit=ok & (zz >= -exit),
        short_entry=ok & (zz > entry),
        short_exit=ok & (zz <= exit),
        mode=mode,
    )


def _prior_channel(px: PriceData, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Highest high / lowest low of the ``window`` bars BEFORE the current bar."""

    def build() -> np.ndarray:
        hi = np.full(len(px.high), np.nan)
        lo = np.full(len(px.low), np.nan)
        hi[1:] = ind.rolling_max(px.high, window)[:-1]
        lo[1:] = ind.rolling_min(px.low, window)[:-1]
        return np.stack([hi, lo])

    both = px.cached(("channel", window), build)
    return both[0], both[1]


def donchian(px: PriceData, entry_window: int, exit_frac: float, mode: str) -> np.ndarray:
    """Donchian channel breakout.

    Go long when the close exceeds the highest high of the previous ``entry_window`` bars;
    close the long when the close falls below the lowest low of the previous
    ``exit_window = round(entry_window * exit_frac)`` bars. Shorts mirror this.
    """
    exit_window = max(2, int(round(entry_window * exit_frac)))
    hi_n, lo_n = _prior_channel(px, entry_window)
    hi_m, lo_m = _prior_channel(px, exit_window)
    ok = ~np.isnan(hi_n)
    c = px.close
    with np.errstate(invalid="ignore"):
        return _two_legs(
            long_entry=ok & (c > hi_n),
            long_exit=ok & (c < lo_m),
            short_entry=ok & (c < lo_n),
            short_exit=ok & (c > hi_m),
            mode=mode,
        )


def rsi_threshold(px: PriceData, period: int, band: float, style: str, mode: str) -> np.ndarray:
    """RSI threshold rules; ``band`` is the oversold level, ``100 - band`` the overbought.

    ``reversal``: buy oversold (RSI < band), sell overbought, exit at the midline 50.
    ``trend``:    buy strength (RSI > 100 - band), sell weakness, exit at the midline 50.
    """
    r = px.cached(("rsi", period), lambda: ind.rsi(px.close, period))
    ok = ~np.isnan(r)
    rr = np.where(ok, r, 50.0)
    low, high = band, 100.0 - band
    if style == "reversal":
        return _two_legs(
            long_entry=ok & (rr < low),
            long_exit=ok & (rr >= 50.0),
            short_entry=ok & (rr > high),
            short_exit=ok & (rr <= 50.0),
            mode=mode,
        )
    if style == "trend":
        return _two_legs(
            long_entry=ok & (rr > high),
            long_exit=ok & (rr < 50.0),
            short_entry=ok & (rr < low),
            short_exit=ok & (rr > 50.0),
            mode=mode,
        )
    raise ValueError(f"unknown style {style!r}")


def vol_target_scale(
    px: PriceData,
    target_annual: float,
    window: int,
    max_leverage: float,
    periods_per_year: float,
) -> np.ndarray:
    """Volatility-target overlay: ``scale[t] = min(target / realised_vol[t], max_leverage)``.

    ``realised_vol[t]`` is the annualised standard deviation of the last ``window`` returns
    up to bar ``t``. Multiplying any base position by this scale keeps its expected risk
    roughly constant through time. Warm-up bars get scale 0 (no position).
    """
    vol = ind.rolling_std(px.returns, window) * np.sqrt(periods_per_year)
    vol[:window] = np.nan  # the first window contains the dummy return r[0] = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.minimum(target_annual / vol, max_leverage)
    return np.where(np.isnan(scale) | ~np.isfinite(scale), 0.0, scale)


FAMILIES: dict[str, Callable[..., np.ndarray]] = {
    "ma_cross": ma_cross,
    "tsmom": tsmom,
    "zscore_mr": zscore_mr,
    "donchian": donchian,
    "rsi": rsi_threshold,
}


def expand_grid(universe_cfg: dict) -> Iterator[tuple[str, dict]]:
    """Yield ``(family, params)`` for every base variant in the configured grids.

    The order is deterministic (family order as in ``FAMILIES``, then the Cartesian
    product in the order the parameters appear in the config). MA crossovers require
    ``fast < slow``.
    """
    for family in FAMILIES:
        grid = universe_cfg.get(family)
        if not grid:
            continue
        names = list(grid)
        for values in itertools.product(*(grid[n] for n in names)):
            params = dict(zip(names, values, strict=True))
            if family == "ma_cross" and params["fast"] >= params["slow"]:
                continue
            yield family, params


def param_label(family: str, params: dict) -> str:
    """Compact, stable text label for a variant, e.g. ``ma_cross{"fast":5,...}``."""
    return family + json.dumps(params, separators=(",", ":"), sort_keys=True)
