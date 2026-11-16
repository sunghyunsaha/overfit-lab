"""The vectorised backtester.

Timing convention (the single most important thing in this file):

* A strategy decides its target position ``w[t]`` at the CLOSE of bar ``t`` using data up
  to and including bar ``t``.
* That position is HELD during bar ``t + lag`` (default ``lag = 1``, i.e. the next bar):

      held[t]  = w[t - lag]
      gross[t] = held[t] * r[t],          r[t] = close[t] / close[t-1] - 1

* Trading from ``held[t-1]`` to ``held[t]`` happens at the start of bar ``t`` and is
  charged to bar ``t``:

      turnover[t] = |held[t] - held[t-1]|
      cost[t]     = turnover[t] * (fee + slip_rate[t])
      net[t]      = gross[t] - cost[t]

  ``slip_rate[t]`` is ``slippage_vol_mult`` times the trailing DAILY volatility measured
  up to bar ``t-1``, so the cost model itself uses no future information either.

Positions are fractions of equity (1 = fully long, -1 = fully short), returns are simple
per-bar returns on equity, and the position is assumed to be rebalanced back to target
each bar at no cost for the drift (a standard small approximation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from overfitlab.backtest.indicators import simple_returns, trailing_vol


@dataclass(frozen=True)
class CostModel:
    """Transaction cost parameters."""

    fee_bps: float = 10.0
    slippage_vol_mult: float = 0.01
    slippage_window_days: int = 20
    bars_per_day: int = 1

    @property
    def fee(self) -> float:
        return self.fee_bps * 1e-4


@dataclass
class BacktestResult:
    """Per-bar outputs, each of shape (T, K) for K strategies on one asset.

    ``net(fee_bps)`` is linear in the fee, so cost sensitivity needs no re-run:
    ``net = gross - fee * turnover - slippage``.
    """

    gross: np.ndarray
    turnover: np.ndarray
    slippage: np.ndarray

    def net(self, fee_bps: float) -> np.ndarray:
        return self.gross - fee_bps * 1e-4 * self.turnover - self.slippage


def slippage_rate(returns: np.ndarray, cost: CostModel) -> np.ndarray:
    """Slippage per unit of turnover charged on bar ``t`` (uses returns up to ``t-1``)."""
    window = cost.slippage_window_days * cost.bars_per_day
    daily_vol = trailing_vol(returns, window) * np.sqrt(cost.bars_per_day)
    rate = np.zeros(len(returns))
    rate[1:] = cost.slippage_vol_mult * daily_vol[:-1]
    return rate


def backtest(
    close: np.ndarray,
    positions: np.ndarray,
    cost: CostModel,
    lag: int = 1,
) -> BacktestResult:
    """Backtest target positions on one asset.

    Parameters
    ----------
    close : (T,) close prices.
    positions : (T,) or (T, K) target positions decided at the close of each bar.
    cost : transaction cost model.
    lag : bars between decision and holding (>= 1; a lag of 0 would be lookahead).
    """
    if lag < 1:
        raise ValueError("lag must be >= 1: a position cannot earn the bar it was decided on")
    pos = np.asarray(positions, dtype=float)
    if pos.ndim == 1:
        pos = pos[:, None]
    returns = simple_returns(np.asarray(close, dtype=float))

    held = np.zeros_like(pos)
    held[lag:] = pos[:-lag]
    gross = held * returns[:, None]

    turnover = np.abs(np.diff(held, axis=0, prepend=0.0))
    slip = turnover * slippage_rate(returns, cost)[:, None]
    return BacktestResult(gross=gross, turnover=turnover, slippage=slip)
