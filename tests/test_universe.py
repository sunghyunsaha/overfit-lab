"""The universe matrix: size, alignment and bookkeeping identities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from overfitlab.backtest.universe import Universe, _daily_sum, build_universe
from overfitlab.config import load_config
from overfitlab.data.loader import load_market
from overfitlab.data.synthetic import simulate_market


def test_default_universe_has_5000_variants():
    cfg = load_config()
    n_assets = len(cfg["data"]["symbols"])
    assert n_assets == 10 and cfg["synthetic"]["n_assets"] == 10
    prices = simulate_market(300, n_assets, seed=1)
    uni = build_universe(prices, cfg)
    assert uni.shape == (300, 5000)
    assert len(uni.meta) == 5000 and uni.meta["variant_id"].tolist() == list(range(5000))
    assert uni.meta.groupby("asset").size().eq(500).all()
    assert uni.meta["vol_target"].sum() == 2500


def test_net_equals_gross_minus_costs_and_roundtrips(small_cfg, small_prices, tmp_path):
    uni = build_universe(small_prices, small_cfg)
    np.testing.assert_allclose(
        uni.net(), uni.gross - 10e-4 * uni.turnover - uni.slippage, atol=1e-15
    )
    assert np.all(uni.turnover >= 0) and np.all(uni.slippage >= 0)
    assert np.all(uni.net(30.0) <= uni.net(0.0) + 1e-15)
    uni.save(tmp_path / "u")
    back = Universe.load(tmp_path / "u")
    np.testing.assert_allclose(back.gross, uni.gross, atol=1e-6)
    assert back.meta.equals(uni.meta) and back.index.equals(uni.index)


def test_column_matches_a_direct_single_strategy_backtest(small_cfg, small_prices):
    """Column j of the matrix is exactly the strategy its metadata row describes."""
    from overfitlab.backtest.engine import CostModel, backtest
    from overfitlab.backtest.strategies import PriceData, ma_cross

    uni = build_universe(small_prices, small_cfg)
    label = 'ma_cross{"fast":10,"ma_type":"ema","mode":"long_short","slow":50}'
    row = uni.meta[
        (uni.meta["asset"] == "SYN01") & (uni.meta["params"] == label) & ~uni.meta["vol_target"]
    ]
    assert len(row) == 1
    df = small_prices["SYN01"]
    px = PriceData(df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy())
    res = backtest(px.close, ma_cross(px, 10, 50, "ema", "long_short"), CostModel())
    np.testing.assert_allclose(uni.net()[:, row.index[0]], res.net(10.0)[:, 0], atol=1e-15)


def test_hourly_bars_are_aggregated_to_daily():
    idx = pd.date_range("2020-01-01", periods=72, freq="1h", tz="UTC")
    vals = np.arange(72.0).reshape(-1, 1)
    days, summed = _daily_sum(idx, vals)
    assert len(days) == 3
    np.testing.assert_allclose(summed[:, 0], [vals[:24].sum(), vals[24:48].sum(), vals[48:].sum()])

    cfg = load_config(
        overrides={
            "data": {"source": "synthetic", "interval": "1h"},
            "synthetic": {"n_assets": 1, "n_bars": 40},
        }
    )
    prices, info = load_market(cfg)
    assert info["n_bars"] == 40 * 24
    uni = build_universe(prices, cfg)
    assert uni.shape == (40, 500)
