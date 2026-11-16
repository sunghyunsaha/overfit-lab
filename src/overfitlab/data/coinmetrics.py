"""Fallback real-data source: CoinMetrics community data (daily, close-only).

Source: https://github.com/coinmetrics/data (CSV per asset, licensed CC BY-NC 4.0; the
files are downloaded to the local cache and are NOT redistributed in this repository).

Why it exists: the environment this project was built in could reach GitHub but not
data.binance.vision. CoinMetrics publishes one daily reference price per asset
(``PriceUSD``, the price at 00:00 UTC) and a reported spot volume. There is no intraday
open/high/low, so the frame is built with ``open = previous close`` and
``high/low = max/min(open, close)``. Consequence: Donchian channels on this source are
channels of daily closes, not of intraday highs and lows.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

RAW_URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/{asset}.csv"


def parse_coinmetrics_csv(text: str) -> pd.DataFrame:
    """Build a close-only OHLCV frame from a CoinMetrics community CSV."""
    cols = ["time", "PriceUSD", "volume_reported_spot_usd_1d"]
    raw = pd.read_csv(io.StringIO(text), usecols=lambda c: c in cols, low_memory=False)
    raw = raw.dropna(subset=["PriceUSD"])
    index = pd.DatetimeIndex(pd.to_datetime(raw["time"], utc=True), name="time").as_unit("ns")
    close = pd.Series(raw["PriceUSD"].to_numpy(dtype=float), index=index)
    open_ = close.shift(1).fillna(close)
    volume = raw.get("volume_reported_spot_usd_1d")
    return pd.DataFrame(
        {
            "open": open_,
            "high": pd.concat([open_, close], axis=1).max(axis=1),
            "low": pd.concat([open_, close], axis=1).min(axis=1),
            "close": close,
            "volume": float("nan") if volume is None else volume.to_numpy(dtype=float),
        },
        index=index,
    )


def download_all(cfg: dict, cache_dir: Path) -> None:
    """Download every configured asset and cache it as Parquet."""
    out_dir = cache_dir / "coinmetrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    for base in cfg["data"]["symbols"]:
        url = RAW_URL.format(asset=base.lower())
        with urllib.request.urlopen(url, timeout=120) as resp:
            text = resp.read().decode("utf-8")
        df = parse_coinmetrics_csv(text)
        df.to_parquet(out_dir / f"{base}.parquet")
        print(f"{base}: {len(df)} daily prices {df.index[0].date()} .. {df.index[-1].date()}")
