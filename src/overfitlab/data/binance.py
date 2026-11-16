"""Downloader for Binance public bulk data (https://data.binance.vision).

Monthly kline archives are fetched per symbol, checksum-verified, parsed and cached as
one Parquet file per month; the months are then stitched into one Parquet per symbol.
Days of the current (incomplete) month come from the daily archives.

File format notes (spot klines):
* CSV columns: open_time, open, high, low, close, volume, close_time, quote_volume,
  trade_count, taker_buy_base, taker_buy_quote, ignore. Older files have no header row.
* Timestamps are epoch milliseconds up to 2024 and epoch MICROseconds from 2025-01-01,
  so the unit is detected from the magnitude.

Only the standard library is used for HTTP, to keep dependencies minimal.
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot"
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def kline_url(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    day: int | None = None,
    base_url: str = BASE_URL,
) -> str:
    """URL of a monthly (``day=None``) or daily kline archive.

    Layout confirmed against the live site (2026-11-16):
    ``.../monthly/klines/NEOUSDT/1d/NEOUSDT-1d-2026-10.zip`` and
    ``.../daily/klines/BTCUSDT/1h/BTCUSDT-1h-2026-11-15.zip``, each with a ``.CHECKSUM``
    file next to it. Monthly archives appear a few days into the following month and
    daily archives the following day.
    """
    if day is None:
        name = f"{symbol}-{interval}-{year}-{month:02d}.zip"
        return f"{base_url}/monthly/klines/{symbol}/{interval}/{name}"
    name = f"{symbol}-{interval}-{year}-{month:02d}-{day:02d}.zip"
    return f"{base_url}/daily/klines/{symbol}/{interval}/{name}"


def month_range(start: date, end: date) -> list[tuple[int, int]]:
    """All (year, month) pairs from ``start``'s month to ``end``'s month inclusive."""
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def verify_checksum(payload: bytes, checksum_text: str) -> bool:
    """Compare a payload's SHA-256 with the first token of a Binance ``.CHECKSUM`` file."""
    expected = checksum_text.split()[0].strip().lower()
    return hashlib.sha256(payload).hexdigest() == expected


def parse_kline_zip(payload: bytes) -> pd.DataFrame:
    """Parse one kline zip archive into an OHLCV frame indexed by UTC open time."""
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            raw = pd.read_csv(fh, header=None, names=KLINE_COLUMNS, dtype=str)
    if not raw.iloc[0, 0].strip().isdigit():  # newer files carry a header row
        raw = raw.iloc[1:]
    open_time = raw["open_time"].astype("int64")
    unit = "us" if open_time.iloc[0] > 10**14 else "ms"
    index = pd.to_datetime(open_time.to_numpy(), unit=unit, utc=True)
    df = raw[["open", "high", "low", "close", "volume"]].astype(float)
    df.index = pd.DatetimeIndex(index, name="time").as_unit("ns")
    return df


def _fetch(url: str, retries: int = 3, timeout: float = 60.0) -> bytes | None:
    """GET ``url``. Returns None on HTTP 404 (file does not exist, e.g. before listing)."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return None
            last = err
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            last = err
        if attempt + 1 < retries:
            time.sleep(1.5 * (attempt + 1))
    raise ConnectionError(f"could not fetch {url}: {last}")


def _fetch_archive(url: str) -> pd.DataFrame | None:
    payload = _fetch(url)
    if payload is None:
        return None
    checksum = _fetch(url + ".CHECKSUM")
    if checksum is not None and not verify_checksum(payload, checksum.decode()):
        raise ValueError(f"checksum mismatch for {url}")
    return parse_kline_zip(payload)


def _fetch_month(
    symbol: str, interval: str, year: int, month: int, today: date, base_url: str
) -> pd.DataFrame | None:
    """One calendar month of klines: the monthly archive, else that month's daily archives."""
    complete = (year, month) < (today.year, today.month)
    prev_month = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    df = None
    if complete:
        df = _fetch_archive(kline_url(symbol, interval, year, month, base_url=base_url))
    if df is None and (year, month) >= prev_month:
        # The current month, or last month before its monthly archive is published.
        days = []
        d = date(year, month, 1)
        while d.month == month and d < today:
            url = kline_url(symbol, interval, d.year, d.month, d.day, base_url=base_url)
            day_df = _fetch_archive(url)
            if day_df is not None:
                days.append(day_df)
            d += timedelta(days=1)
        df = pd.concat(days) if days else None
    return df


def download_symbol(
    symbol: str,
    interval: str,
    start: date,
    end: date,
    cache_dir: Path,
    *,
    base_url: str = BASE_URL,
    today: date | None = None,
    workers: int = 8,
) -> pd.DataFrame:
    """Download (or read from cache) all klines of ``symbol`` between ``start`` and ``end``.

    Completed months are cached as one Parquet file each, so a second run only fetches
    the current month. Months are fetched by a small thread pool (the work is network
    I/O). ``today`` and ``base_url`` exist so the function can be tested against a local
    replica of the site; in normal use leave them at their defaults.
    """
    month_dir = cache_dir / "binance" / "_months" / symbol / interval
    month_dir.mkdir(parents=True, exist_ok=True)
    today = today or datetime.now(UTC).date()
    months = month_range(start, min(end, today))

    def load(ym: tuple[int, int]) -> pd.DataFrame | None:
        year, month = ym
        path = month_dir / f"{year}-{month:02d}.parquet"
        complete = (year, month) < (today.year, today.month)
        if complete and path.exists():
            return pd.read_parquet(path)
        df = _fetch_month(symbol, interval, year, month, today, base_url)
        if df is None:
            logger.info("%s %s %d-%02d: no archive (not listed yet?)", symbol, interval, *ym)
        elif complete:
            df.to_parquet(path)
        return df

    # The first month is fetched on its own, so an unreachable host fails within seconds
    # instead of after every worker has exhausted its retries.
    first = load(months[0])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = [df for df in [first, *pool.map(load, months[1:])] if df is not None]
    if not parts:
        raise FileNotFoundError(f"no Binance data found for {symbol} {interval}")
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    target = cache_dir / "binance" / interval / f"{symbol}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target)
    return out


def _settings(cfg: dict, today: date | None) -> tuple[date, date, date, str]:
    cfg_data = cfg["data"]
    today = today or datetime.now(UTC).date()
    start = date.fromisoformat(cfg_data["start"])
    end = date.fromisoformat(cfg_data["end"]) if cfg_data.get("end") else today
    return start, end, today, cfg_data.get("binance_base_url", BASE_URL)


def download_all(cfg: dict, cache_dir: Path, today: date | None = None) -> None:
    """Download every configured symbol. Raises ConnectionError if the host is unreachable."""
    cfg_data = cfg["data"]
    start, end, today, base_url = _settings(cfg, today)
    for base in cfg_data["symbols"]:
        symbol = f"{base}{cfg_data['quote']}"
        df = download_symbol(
            symbol, cfg_data["interval"], start, end, cache_dir, base_url=base_url, today=today
        )
        print(f"{symbol}: {len(df):,} bars {df.index[0]} .. {df.index[-1]}")


def check_source(cfg: dict, today: date | None = None) -> list[dict]:
    """Quick pre-flight check before a bulk download (``make data-check``).

    For every configured symbol, fetch the archive of the FIRST month of the sample and
    of the LAST completed month, verify the checksum and parse it. Returns (and prints)
    one row per symbol, so a blocked network, a renamed or delisted symbol, or a listing
    date after the sample start is visible in seconds instead of after a long download.
    """
    cfg_data = cfg["data"]
    start, end, today, base_url = _settings(cfg, today)
    last = min(end, today).replace(day=1) - timedelta(days=1)  # last day of the previous month
    rows = []
    for base in cfg_data["symbols"]:
        symbol = f"{base}{cfg_data['quote']}"
        row = {"symbol": symbol, "interval": cfg_data["interval"]}
        for label, (y, m) in (
            ("first", (start.year, start.month)),
            ("last", (last.year, last.month)),
        ):
            df = _fetch_archive(kline_url(symbol, cfg_data["interval"], y, m, base_url=base_url))
            row[f"{label}_month"] = f"{y}-{m:02d}"
            row[f"{label}_bars"] = 0 if df is None else len(df)
            row[f"{label}_from"] = None if df is None else str(df.index[0])
        row["ok"] = row["first_bars"] > 0 and row["last_bars"] > 0
        rows.append(row)
        status = "ok" if row["ok"] else "MISSING"
        first = f"{row['first_month']}: {row['first_bars']:>4} bars (from {row['first_from']})"
        latest = f"{row['last_month']}: {row['last_bars']:>4} bars"
        print(f"{symbol:10s} {row['interval']}  {first}   {latest}   {status}")
    return rows
