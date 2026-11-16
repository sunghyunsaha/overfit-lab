"""Sanity checks of each strategy family on constructed price paths."""

from __future__ import annotations

import numpy as np

from overfitlab.backtest import indicators as ind
from overfitlab.backtest import strategies as st
from overfitlab.config import load_config


def _px(close: np.ndarray) -> st.PriceData:
    return st.PriceData(close=close, high=close * 1.001, low=close * 0.999)


def test_indicators_match_naive_loops():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(80).cumsum() + 50
    w = 7
    for t in range(w - 1, 80):
        win = x[t - w + 1 : t + 1]
        assert np.isclose(ind.sma(x, w)[t], win.mean())
        assert np.isclose(ind.rolling_std(x, w)[t], win.std(ddof=1))
        assert np.isclose(ind.rolling_max(x, w)[t], win.max())
        assert np.isclose(ind.rolling_min(x, w)[t], win.min())
    assert np.all(np.isnan(ind.sma(x, w)[: w - 1]))
    # EMA recursion by hand.
    a, y = 2 / (5 + 1), x[0]
    for t in range(1, 80):
        y = a * x[t] + (1 - a) * y
        if t >= 4:
            assert np.isclose(ind.ema(x, 5)[t], y)


def test_rsi_extremes():
    up = np.linspace(100, 200, 60)
    assert np.all(ind.rsi(up, 14)[14:] > 99.9)
    assert np.all(ind.rsi(up[::-1].copy(), 14)[14:] < 0.1)


def test_trend_rules_go_long_in_an_uptrend_and_short_in_a_downtrend():
    up = 100 * 1.01 ** np.arange(300)
    down = up[::-1].copy()
    for pos_fn in (
        lambda px: st.ma_cross(px, 10, 50, "sma", "long_short"),
        lambda px: st.ma_cross(px, 10, 50, "ema", "long_short"),
        lambda px: st.tsmom(px, 20, "long_short"),
        lambda px: st.donchian(px, 20, 0.5, "long_short"),
        lambda px: st.rsi_threshold(px, 14, 30, "trend", "long_short"),
    ):
        assert np.all(pos_fn(_px(up))[60:] == 1.0)
        assert np.all(pos_fn(_px(down))[60:] == -1.0)
    assert np.all(st.ma_cross(_px(down), 10, 50, "sma", "long_only") == 0.0)


def test_zscore_state_machine():
    # Flat, then a sharp dip (z very negative), then recovery above the mean.
    close = np.r_[np.full(30, 100.0) + 0.01 * np.arange(30) % 2, 90.0, 91.0, 100.5, 101.0]
    pos = st.zscore_mr(_px(close), window=20, entry=2.0, exit=0.0, mode="long_short")
    assert pos[29] == 0.0  # nothing before the dip
    assert pos[30] == 1.0  # dip -> long
    assert pos[31] == 1.0  # still below the mean -> hold
    assert pos[-1] != 1.0  # back above the mean -> long closed


def test_donchian_breakout_and_exit():
    close = np.r_[np.full(25, 100.0), 105.0, 106.0, 107.0, 95.0, 94.0]
    px = st.PriceData(close=close, high=close, low=close)
    pos = st.donchian(px, entry_window=20, exit_frac=0.5, mode="long_only")
    assert pos[24] == 0.0 and pos[25] == 1.0 and pos[27] == 1.0
    assert pos[28] == 0.0  # close below the 10-bar low -> exit


def test_hold_last_event():
    ev = np.array([np.nan, 1.0, np.nan, np.nan, 0.0, np.nan, -1.0, np.nan])
    np.testing.assert_array_equal(st._hold_last_event(ev), [0, 1, 1, 1, 0, 0, -1, -1])


def test_grid_size_is_250_base_variants_per_asset():
    cfg = load_config()
    grid = list(st.expand_grid(cfg["universe"]))
    assert len(grid) == 250
    counts = {}
    for family, _ in grid:
        counts[family] = counts.get(family, 0) + 1
    assert counts == {"ma_cross": 68, "tsmom": 26, "zscore_mr": 80, "donchian": 28, "rsi": 48}
    assert len({st.param_label(f, p) for f, p in grid}) == 250  # all distinct


def test_vol_target_scale_inverse_to_volatility():
    rng = np.random.default_rng(3)
    r = np.r_[0.01 * rng.standard_normal(200), 0.04 * rng.standard_normal(200)]
    px = _px(100 * np.cumprod(1 + r))
    scale = st.vol_target_scale(px, 0.4, 30, 2.0, 365.0)
    assert np.all(scale[:30] == 0.0)
    assert scale[150] > scale[380]  # calm period gets more leverage
    assert scale.max() <= 2.0
