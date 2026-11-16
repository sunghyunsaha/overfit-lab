"""Loading and validating OHLCV data.

``validate_ohlcv`` checks one asset for the usual bar-data problems (duplicates, gaps,
impossible prices, inconsistent OHLC, zero volume, outliers), repairs what can be
repaired without inventing information, and records every finding in a
``ValidationLog`` so nothing is changed silently.

``load_market`` is the single entry point used by the experiments. It resolves the data
source (Binance cache, CoinMetrics cache, or the synthetic stand-in), validates each
asset and aligns all assets on one common time index.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from overfitlab.config import resolve_path
from overfitlab.data.synthetic import simulate_market, synthetic_kwargs

logger = logging.getLogger(__name__)

OHLCV = ["open", "high", "low", "close", "volume"]
BARS_PER_DAY = {"1d": 1, "1h": 24}
PANDAS_FREQ = {"1d": "1D", "1h": "1h"}


@dataclass
class ValidationLog:
    """Collects one record per (symbol, check) finding."""

    records: list[dict] = field(default_factory=list)

    def add(self, symbol: str, check: str, n: int, action: str, detail: str = "") -> None:
        if n <= 0:
            return
        rec = {"symbol": symbol, "check": check, "n": int(n), "action": action, "detail": detail}
        self.records.append(rec)
        logger.info("data check %s", rec)

    def count(self, check: str | None = None) -> int:
        return sum(r["n"] for r in self.records if check is None or r["check"] == check)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")


def validate_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    log: ValidationLog,
    *,
    outlier_sigma: float = 12.0,
    repair_outliers: bool = False,
) -> pd.DataFrame:
    """Validate and clean one asset's OHLCV frame (indexed by UTC bar open time).

    Checks, in order:

    1. duplicates       duplicated timestamps -> keep the last one.
    2. bad prices       NaN or non-positive OHLC -> treated as a missing bar.
    3. gaps             missing bars on the regular grid -> filled flat at the previous
                        close with zero volume (no information is invented: a flat bar
                        produces a zero return and cannot trigger a breakout).
    4. OHLC consistency high < max(open, close) or low > min(open, close) -> widened.
    5. zero volume      counted and logged, not altered.
    6. outliers         |log return| above ``outlier_sigma`` robust standard deviations
                        (rolling median absolute deviation) -> logged; only altered if
                        ``repair_outliers`` is set, in which case the bar is made flat.
    """
    df = df[OHLCV].copy()
    df = df.sort_index()

    dup = df.index.duplicated(keep="last")
    log.add(symbol, "duplicates", int(dup.sum()), "kept last")
    df = df[~dup]

    prices = df[["open", "high", "low", "close"]]
    bad = (prices.isna() | (prices <= 0)).any(axis=1)
    log.add(symbol, "bad_prices", int(bad.sum()), "treated as missing bar")
    df = df[~bad]
    if df.empty:
        raise ValueError(f"{symbol}: no valid bars")

    full = pd.date_range(df.index[0], df.index[-1], freq=PANDAS_FREQ[interval], name="time")
    missing = full.difference(df.index)
    if len(missing):
        detail = f"first {missing[0]}, last {missing[-1]}"
        log.add(symbol, "gaps", len(missing), "flat-filled at previous close, volume 0", detail)
    df = df.reindex(full)
    gap = df["close"].isna()
    df["close"] = df["close"].ffill()
    for col in ("open", "high", "low"):
        df[col] = df[col].where(~gap, df["close"])
    df["volume"] = df["volume"].where(~gap, 0.0)

    hi = df[["open", "high", "close"]].max(axis=1)
    lo = df[["open", "low", "close"]].min(axis=1)
    inconsistent = (df["high"] < hi) | (df["low"] > lo)
    log.add(symbol, "ohlc_inconsistent", int(inconsistent.sum()), "high/low widened")
    df["high"] = hi
    df["low"] = lo

    zero_vol = (df["volume"].fillna(0.0) <= 0) & ~gap
    log.add(symbol, "zero_volume", int(zero_vol.sum()), "kept (logged only)")

    log_ret = np.log(df["close"]).diff()
    med = log_ret.rolling(90, min_periods=30).median()
    mad = (log_ret - med).abs().rolling(90, min_periods=30).median()
    robust_sigma = 1.4826 * mad.shift(1)  # shift: the bar being judged is not in its own scale
    outlier = (log_ret - med.shift(1)).abs() > outlier_sigma * robust_sigma
    outlier &= robust_sigma > 0
    n_out = int(outlier.sum())
    if n_out:
        worst = log_ret[outlier].abs().idxmax()
        detail = f"largest |log return| {abs(log_ret[worst]):.3f} at {worst}"
        action = "made flat" if repair_outliers else "kept (logged only)"
        log.add(symbol, "outliers", n_out, action, detail)
        if repair_outliers:
            prev_close = df["close"].shift(1)
            for col in ("open", "high", "low", "close"):
                df[col] = df[col].where(~outlier, prev_close)
    return df


def _read_cached(cache_dir: Path, source: str, cfg_data: dict) -> dict[str, pd.DataFrame] | None:
    """Read all configured symbols from a cache directory, or None if any is missing."""
    frames = {}
    for sym in cfg_data["symbols"]:
        if source == "binance":
            path = (
                cache_dir / "binance" / cfg_data["interval"] / f"{sym}{cfg_data['quote']}.parquet"
            )
        else:
            path = cache_dir / "coinmetrics" / f"{sym}.parquet"
        if not path.exists():
            return None
        frames[sym] = pd.read_parquet(path)
    return frames


SOURCE_LABELS = {
    "binance": "REAL - Binance spot OHLCV (data.binance.vision)",
    "coinmetrics": "REAL - CoinMetrics community daily reference prices (close-only)",
    "synthetic": "SYNTHETIC - simulated stand-in (no real data was cached)",
}


SOURCE_NOTES = {
    "binance": (
        "This run uses the project's primary source: Binance spot OHLCV bulk data from "
        "`data.binance.vision`, which are one exchange's own bars (open, high, low, close, "
        "volume). The downloader verifies each archive against the SHA-256 checksum file "
        "that Binance publishes beside it."
    ),
    "coinmetrics": (
        "The primary source in the spec is Binance spot OHLCV (`data.binance.vision`). The "
        "machine this run was made on could not reach that host, so this run uses the "
        "fallback: real daily prices from the CoinMetrics community dataset (one reference "
        "price per day, no intraday high or low, so Donchian channels are channels of daily "
        "closes). Binance data replaces it with `make data && make all`."
    ),
    "synthetic": (
        "NO REAL DATA WAS AVAILABLE FOR THIS RUN. E3, E4 and E5 ran on a simulated stand-in "
        "market, so their numbers illustrate the pipeline and say nothing about real "
        "markets. Run `make data && make all` (or `make data-fallback && make all`) to "
        "replace them."
    ),
}


def load_market(cfg: dict, *, seed_offset: int = 0) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load, validate and align the market described by ``cfg``.

    Source resolution for ``data.source: auto``: Binance cache if complete, otherwise the
    CoinMetrics cache if complete, otherwise the synthetic stand-in. The returned ``info``
    dict records which one was used so every report can state it.
    """
    cfg_data = cfg["data"]
    interval = cfg_data["interval"]
    cache_dir = resolve_path(cfg_data, "cache_dir")
    source = cfg_data["source"]
    log = ValidationLog()

    frames = None
    if source in ("auto", "binance"):
        frames = _read_cached(cache_dir, "binance", cfg_data)
        if frames is not None:
            source = "binance"
    if frames is None and source in ("auto", "coinmetrics"):
        if interval == "1d":
            frames = _read_cached(cache_dir, "coinmetrics", cfg_data)
        if frames is not None:
            source = "coinmetrics"
    if frames is None:
        if source not in ("auto", "synthetic"):
            target = "make data-hourly" if interval == "1h" else "make data"
            raise FileNotFoundError(
                f"data.source={source!r} but the {interval} cache is incomplete; "
                f"run `{target}` first"
            )
        source = "synthetic"
        bars_per_year = cfg["periods_per_year"] * BARS_PER_DAY[interval]
        frames = simulate_market(
            cfg["synthetic"]["n_bars"] * BARS_PER_DAY[interval],
            cfg["synthetic"]["n_assets"],
            seed=cfg["seed"] + seed_offset,
            freq=PANDAS_FREQ[interval],
            **synthetic_kwargs(cfg["synthetic"], bars_per_year),
        )

    val = cfg_data.get("validation", {})
    clean = {}
    for sym, df in frames.items():
        if source != "synthetic":
            df = df.loc[cfg_data["start"] : cfg_data.get("end") or None]
        clean[sym] = validate_ohlcv(
            df,
            sym,
            interval,
            log,
            outlier_sigma=val.get("outlier_sigma", 12.0),
            repair_outliers=val.get("repair_outliers", False),
        )

    # Align on the common window so every asset contributes the same T rows.
    start = max(df.index[0] for df in clean.values())
    end = min(df.index[-1] for df in clean.values())
    for sym, df in clean.items():
        trimmed = len(df) - len(df.loc[start:end])
        log.add(sym, "trimmed_to_common_window", trimmed, "dropped", f"window {start}..{end}")
        clean[sym] = df.loc[start:end]
    n_bars = len(next(iter(clean.values())))
    info = {
        "source": source,
        "label": SOURCE_LABELS[source],
        "note": SOURCE_NOTES[source],
        "is_real": source != "synthetic",
        "interval": interval,
        "symbols": list(clean),
        "start": str(start.date()),
        "end": str(end.date()),
        "n_bars": n_bars,
        "validation": log.records,
        "validation_counts": {
            check: log.count(check)
            for check in (
                "duplicates",
                "bad_prices",
                "gaps",
                "ohlc_inconsistent",
                "zero_volume",
                "outliers",
            )
        },
    }
    return clean, info
