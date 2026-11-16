"""The backtester against a hand-calculated example."""

from __future__ import annotations

import numpy as np
import pytest

from overfitlab.backtest.engine import CostModel, backtest, slippage_rate
from overfitlab.backtest.indicators import simple_returns


def test_hand_calculated_example():
    # Prices: 100 -> 110 (+10%) -> 99 (-10%) -> 108.9 (+10%) -> 108.9 (0%)
    close = np.array([100.0, 110.0, 99.0, 108.9, 108.9])
    # Target positions decided at each close.
    target = np.array([1.0, 1.0, -1.0, 0.0, 0.0])
    cost = CostModel(fee_bps=10.0, slippage_vol_mult=0.0)
    res = backtest(close, target, cost, lag=1)

    # Held position is the target of the previous bar: [0, 1, 1, -1, 0].
    # Gross = held * return = [0, +0.10, -0.10, -0.10, 0].
    np.testing.assert_allclose(res.gross[:, 0], [0.0, 0.10, -0.10, -0.10, 0.0], atol=1e-12)
    # Turnover = |change in held| = [0, 1, 0, 2, 1]  (flip from +1 to -1 trades 2 units).
    np.testing.assert_allclose(res.turnover[:, 0], [0.0, 1.0, 0.0, 2.0, 1.0], atol=1e-12)
    # Net at 10 bps per unit traded = gross - 0.001 * turnover.
    np.testing.assert_allclose(res.net(10.0)[:, 0], [0.0, 0.099, -0.10, -0.102, -0.001], atol=1e-12)
    # Net is linear in the fee.
    np.testing.assert_allclose(res.net(0.0), res.gross, atol=1e-15)
    np.testing.assert_allclose(res.net(30.0), res.gross - 0.003 * res.turnover, atol=1e-15)


def test_slippage_uses_only_past_volatility():
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + 0.02 * rng.standard_normal(60))
    cost = CostModel(fee_bps=0.0, slippage_vol_mult=0.5, slippage_window_days=5)
    r = simple_returns(close)
    rate = slippage_rate(r, cost)
    # Hand check at t = 10: 0.5 * std of the 5 returns ending at bar 9.
    assert rate[10] == pytest.approx(0.5 * np.std(r[5:10], ddof=1))
    # The cost charged on bar 10 must not move when bar 10's own price changes.
    close2 = close.copy()
    close2[10:] *= 1.3
    rate2 = slippage_rate(simple_returns(close2), cost)
    np.testing.assert_allclose(rate[:11], rate2[:11])


def test_extra_lag_delays_everything_by_one_bar():
    close = np.array([100.0, 101.0, 103.0, 102.0, 104.0, 107.0])
    target = np.array([1.0, 0.0, 1.0, 1.0, -1.0, 0.0])
    cost = CostModel(fee_bps=0.0, slippage_vol_mult=0.0)
    r = simple_returns(close)
    res2 = backtest(close, target, cost, lag=2)
    np.testing.assert_allclose(res2.gross[2:, 0], target[:-2] * r[2:])
    assert np.all(res2.gross[:2] == 0)


def test_zero_lag_is_rejected():
    with pytest.raises(ValueError):
        backtest(np.array([1.0, 2.0]), np.array([1.0, 1.0]), CostModel(), lag=0)
