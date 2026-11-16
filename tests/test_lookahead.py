"""Lookahead test: changing any future data must not change the past.

For several cut points t, every price after bar t is replaced by a completely different
path. Signals and positions up to and including bar t, and returns up to and including
bar t, must be bit-for-bit identical for EVERY variant in the universe.
"""

from __future__ import annotations

import numpy as np
import pytest

from overfitlab.backtest.engine import CostModel, backtest
from overfitlab.backtest.strategies import PriceData
from overfitlab.backtest.universe import asset_positions
from overfitlab.data.synthetic import simulate_market


def _px(df) -> PriceData:
    return PriceData(df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy())


@pytest.mark.parametrize("cut", [260, 333, 500, 698])
def test_future_data_cannot_change_past_positions_or_returns(small_cfg, small_prices, cut):
    base = small_prices["SYN00"]
    other = simulate_market(700, 1, seed=999, kind="garch")["SYN00"]

    tampered = base.copy()
    scale = base["close"].iloc[cut] / other["close"].iloc[cut]
    for col in ("open", "high", "low", "close"):
        tampered.iloc[cut + 1 :, tampered.columns.get_loc(col)] = (
            other[col].iloc[cut + 1 :].to_numpy() * scale
        )

    pos_a, meta = asset_positions(_px(base), small_cfg, 365.0)
    pos_b, _ = asset_positions(_px(tampered), small_cfg, 365.0)
    assert pos_a.shape[1] == len(meta) == 500
    # Positions decided at bars 0..cut are identical...
    np.testing.assert_array_equal(pos_a[: cut + 1], pos_b[: cut + 1])
    # ...and the test has teeth: the future really is different.
    assert not np.array_equal(pos_a[cut + 1 :], pos_b[cut + 1 :])

    cost = CostModel()
    res_a = backtest(base["close"].to_numpy(), pos_a, cost)
    res_b = backtest(tampered["close"].to_numpy(), pos_b, cost)
    for name in ("gross", "turnover", "slippage"):
        np.testing.assert_array_equal(
            getattr(res_a, name)[: cut + 1], getattr(res_b, name)[: cut + 1]
        )


def test_position_at_t_earns_return_of_t_plus_1(small_cfg, small_prices):
    """A position decided at bar t must not earn bar t's own return."""
    df = small_prices["SYN01"]
    pos, _ = asset_positions(_px(df), small_cfg, 365.0)
    res = backtest(df["close"].to_numpy(), pos, CostModel())
    r = df["close"].pct_change().fillna(0.0).to_numpy()
    np.testing.assert_allclose(res.gross[1:], pos[:-1] * r[1:, None], atol=1e-15)
