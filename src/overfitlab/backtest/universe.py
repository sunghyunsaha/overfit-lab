"""Build the strategy universe: one T x N matrix of daily returns for all variants.

A variant is (asset, family, parameters, volatility-target overlay on/off). All
diagnostics read the same matrices, so there is exactly one source of truth:

* ``gross``    T x N gross returns
* ``turnover`` T x N traded notional per bar (fraction of equity)
* ``slippage`` T x N slippage cost per bar

and ``net(fee) = gross - fee * turnover - slippage``.

If the bars are hourly, the backtest runs on hourly bars and the three matrices are
then SUMMED within each UTC day, so the diagnostics always see daily observations. (The
sum of simple returns is the daily P&L of a constant-notional book; it keeps net returns
linear in the fee, which the cost-sensitivity experiment relies on.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from overfitlab.backtest.engine import CostModel, backtest
from overfitlab.backtest.strategies import (
    FAMILIES,
    PriceData,
    expand_grid,
    param_label,
    vol_target_scale,
)
from overfitlab.data.loader import BARS_PER_DAY


@dataclass
class Universe:
    """The return matrices plus one metadata row per column."""

    index: pd.DatetimeIndex
    meta: pd.DataFrame
    gross: np.ndarray
    turnover: np.ndarray
    slippage: np.ndarray
    fee_bps: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.gross.shape

    def net(self, fee_bps: float | None = None) -> np.ndarray:
        """Net returns for a given per-side fee (default: the configured fee)."""
        fee = self.fee_bps if fee_bps is None else fee_bps
        return self.gross - fee * 1e-4 * self.turnover - self.slippage

    def save(self, directory: Path) -> None:
        """Store the matrices as float32 ``.npy`` plus Parquet metadata."""
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("gross", "turnover", "slippage"):
            np.save(directory / f"{name}.npy", getattr(self, name).astype(np.float32))
        self.meta.to_parquet(directory / "meta.parquet")
        pd.DataFrame(index=self.index).to_parquet(directory / "index.parquet")
        (directory / "info.json").write_text(json.dumps({"fee_bps": self.fee_bps}))

    @classmethod
    def load(cls, directory: Path) -> Universe:
        info = json.loads((directory / "info.json").read_text())
        return cls(
            index=pd.read_parquet(directory / "index.parquet").index,
            meta=pd.read_parquet(directory / "meta.parquet"),
            gross=np.load(directory / "gross.npy").astype(float),
            turnover=np.load(directory / "turnover.npy").astype(float),
            slippage=np.load(directory / "slippage.npy").astype(float),
            fee_bps=info["fee_bps"],
        )


def asset_positions(px: PriceData, cfg: dict, bars_per_year: float) -> tuple[np.ndarray, list]:
    """Target positions of every variant on one asset: (T, K) matrix and K metadata rows."""
    base, rows = [], []
    for family, params in expand_grid(cfg["universe"]):
        base.append(FAMILIES[family](px, **params))
        rows.append((family, params))
    base_pos = np.column_stack(base)

    vt = cfg["universe"]["vol_target"]
    blocks, meta = [], []
    for overlay in vt["variants"]:
        if overlay:
            scale = vol_target_scale(
                px, vt["target_annual"], vt["window"], vt["max_leverage"], bars_per_year
            )
            blocks.append(base_pos * scale[:, None])
        else:
            blocks.append(base_pos)
        meta += [(family, params, bool(overlay)) for family, params in rows]
    return np.concatenate(blocks, axis=1), meta


def _daily_sum(index: pd.DatetimeIndex, values: np.ndarray) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Sum rows within each UTC calendar day."""
    days = index.floor("D")
    starts = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
    return days[starts], np.add.reduceat(values, starts, axis=0)


def build_universe(prices: dict[str, pd.DataFrame], cfg: dict) -> Universe:
    """Backtest every variant on every asset and assemble the T x N matrices."""
    interval = cfg["data"]["interval"]
    bars_per_day = BARS_PER_DAY[interval]
    bars_per_year = cfg["periods_per_year"] * bars_per_day
    cost = CostModel(
        fee_bps=cfg["costs"]["fee_bps"],
        slippage_vol_mult=cfg["costs"]["slippage_vol_mult"],
        slippage_window_days=cfg["costs"]["slippage_window"],
        bars_per_day=bars_per_day,
    )
    gross, turnover, slippage, meta_rows = [], [], [], []
    index = None
    for asset, df in prices.items():
        px = PriceData(
            close=df["close"].to_numpy(dtype=float),
            high=df["high"].to_numpy(dtype=float),
            low=df["low"].to_numpy(dtype=float),
        )
        positions, meta = asset_positions(px, cfg, bars_per_year)
        res = backtest(px.close, positions, cost, lag=cfg["execution_lag"])
        parts = [res.gross, res.turnover, res.slippage]
        if bars_per_day > 1:
            parts = [_daily_sum(df.index, p)[1] for p in parts]
            asset_index = _daily_sum(df.index, np.zeros((len(df), 1)))[0]
        else:
            asset_index = df.index
        if index is None:
            index = asset_index
        elif not index.equals(asset_index):
            raise ValueError("assets are not aligned on a common index")
        gross.append(parts[0])
        turnover.append(parts[1])
        slippage.append(parts[2])
        for family, params, overlay in meta:
            meta_rows.append(
                {
                    "asset": asset,
                    "family": family,
                    "params": param_label(family, params),
                    "vol_target": overlay,
                }
            )
    meta_df = pd.DataFrame(meta_rows)
    meta_df.insert(0, "variant_id", np.arange(len(meta_df)))
    return Universe(
        index=index,
        meta=meta_df,
        gross=np.concatenate(gross, axis=1),
        turnover=np.concatenate(turnover, axis=1),
        slippage=np.concatenate(slippage, axis=1),
        fee_bps=cost.fee_bps,
    )
