"""The Binance downloader, end to end, against a local replica of data.binance.vision.

The real host could not be reached while this project was built, so these tests stand a
small HTTP server up on localhost that serves archives in exactly the layout and file
format of the live site (layout confirmed by browsing the site on 2026-11-16):

    /data/spot/monthly/klines/<SYMBOL>/<interval>/<SYMBOL>-<interval>-<YYYY>-<MM>.zip
    /data/spot/daily/klines/<SYMBOL>/<interval>/<SYMBOL>-<interval>-<YYYY>-<MM>-<DD>.zip
    ...each with a "<name>.CHECKSUM" file holding "<sha256>  <name>"

and then run the REAL download code over HTTP: fetching, 404 handling, checksum
verification, parsing (millisecond and microsecond timestamps, with and without a header
row), the per-month cache, the daily-file fallback for a month whose monthly archive is
not published yet, the current month, and finally the loader and the universe builder.
"""

from __future__ import annotations

import hashlib
import io
import threading
import zipfile
from datetime import date
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from overfitlab.backtest.universe import build_universe
from overfitlab.config import load_config
from overfitlab.data import binance
from overfitlab.data.loader import load_market
from overfitlab.data.synthetic import simulate_market

TODAY = date(2025, 3, 10)  # the replica's "now"
FIRST, LAST = "2024-10-01", "2025-03-09"  # LAST = yesterday: the newest daily file
HEADER = "open_time,open,high,low,close,volume,close_time,quote_volume,count,tb,tq,ignore\n"


def _kline_csv(df: pd.DataFrame, bar_ms: int, micro: bool, header: bool) -> str:
    """Rows in the Binance kline CSV format (12 columns)."""
    scale = 1000 if micro else 1
    lines = [HEADER] if header else []
    for ts, row in df.iterrows():
        t0 = int(ts.timestamp() * 1000)
        o, h, lo, c, v = (repr(float(row[k])) for k in ("open", "high", "low", "close", "volume"))
        quote = repr(float(row["volume"] * row["close"]))
        lines.append(
            f"{t0 * scale},{o},{h},{lo},{c},{v},{(t0 + bar_ms) * scale - 1},{quote},100,1.0,1.0,0\n"
        )
    return "".join(lines)


def _write_archive(path: Path, csv_text: str, corrupt_checksum: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.name.replace(".zip", ".csv"), csv_text)
    payload = buf.getvalue()
    path.write_bytes(payload)
    digest = "0" * 64 if corrupt_checksum else hashlib.sha256(payload).hexdigest()
    Path(str(path) + ".CHECKSUM").write_text(f"{digest}  {path.name}\n")


def build_replica(root: Path, frames: dict[str, pd.DataFrame], interval: str) -> None:
    """Write monthly and daily archives for every symbol, the way the live site has them.

    * months before the current one get a monthly archive, EXCEPT the previous month
      (February 2025), which is "not published yet" and only exists as daily files;
    * the previous and the current month get one daily archive per day up to yesterday;
    * 2024 files use millisecond timestamps and no header; 2025 files use microseconds
      and a header row (the format change Binance made on 2025-01-01).
    """
    bar_ms = 86_400_000 if interval == "1d" else 3_600_000
    base = root / "data" / "spot"
    for symbol, df in frames.items():
        for (year, month), chunk in df.groupby([df.index.year, df.index.month]):
            micro = year >= 2025
            if (year, month) < (2025, 2):
                name = f"{symbol}-{interval}-{year}-{month:02d}.zip"
                path = base / "monthly" / "klines" / symbol / interval / name
                _write_archive(path, _kline_csv(chunk, bar_ms, micro, header=micro))
            if (year, month) >= (2025, 2):
                for day, day_chunk in chunk.groupby(chunk.index.day):
                    name = f"{symbol}-{interval}-{year}-{month:02d}-{day:02d}.zip"
                    path = base / "daily" / "klines" / symbol / interval / name
                    _write_archive(path, _kline_csv(day_chunk, bar_ms, micro, header=micro))


class _CountingHandler(SimpleHTTPRequestHandler):
    hits: list[str] = []

    def do_GET(self):  # noqa: N802 (name fixed by http.server)
        type(self).hits.append(self.path)
        super().do_GET()

    def log_message(self, *args):  # keep the test output quiet
        pass


@pytest.fixture(scope="module")
def replica(tmp_path_factory):
    """A running replica plus the frames it serves: ``(base_url, root, frames_by_interval)``."""
    root = tmp_path_factory.mktemp("replica")
    frames = {}
    for interval, freq, bars in (("1d", "1D", 160), ("1h", "1h", 160 * 24)):
        sim = simulate_market(bars, 2, seed=11, start=FIRST, freq=freq, kind="gbm", vol_annual=0.6)
        aaa = sim["SYN00"].loc[FIRST:LAST]
        bbb = sim["SYN01"].loc["2024-12-01":LAST]  # listed two months later: 404 before that
        frames[interval] = {"AAAUSDT": aaa, "BBBUSDT": bbb}
        build_replica(root, frames[interval], interval)
    _CountingHandler.hits = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_CountingHandler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/data/spot", root, frames
    server.shutdown()


@pytest.fixture(autouse=True)
def _no_proxy_for_localhost(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


def _cfg(base_url: str, cache: Path, interval: str) -> dict:
    return load_config(
        overrides={
            "data": {
                "source": "binance",
                "interval": interval,
                "start": FIRST,
                "symbols": ["AAA", "BBB"],
                "cache_dir": str(cache),
                "binance_base_url": base_url,
            }
        }
    )


@pytest.mark.parametrize("interval", ["1d", "1h"])
def test_download_reproduces_the_served_data_exactly(replica, tmp_path, interval):
    base_url, _, frames = replica
    cfg = _cfg(base_url, tmp_path, interval)
    binance.download_all(cfg, tmp_path, today=TODAY)
    for symbol, expected in frames[interval].items():
        got = pd.read_parquet(tmp_path / "binance" / interval / f"{symbol}.parquet")
        assert len(got) == len(expected)
        assert got.index[0] == expected.index[0] and got.index[-1] == expected.index[-1]
        np.testing.assert_allclose(got[["open", "high", "low", "close", "volume"]], expected)
    # The late-listed symbol starts at its listing date (earlier months were HTTP 404).
    listed = pd.read_parquet(tmp_path / "binance" / interval / "BBBUSDT.parquet")
    assert listed.index[0] == pd.Timestamp("2024-12-01", tz="UTC")


def test_month_cache_daily_fallback_and_current_month(replica, tmp_path):
    base_url, _, _ = replica
    cfg = _cfg(base_url, tmp_path, "1d")
    _CountingHandler.hits = []
    binance.download_all(cfg, tmp_path, today=TODAY)
    first = list(_CountingHandler.hits)
    # February's monthly archive is missing, so its 28 daily files were fetched instead...
    assert any("monthly/klines/AAAUSDT/1d/AAAUSDT-1d-2025-02.zip" in h for h in first)
    feb_daily = [h for h in first if "daily/klines/AAAUSDT/1d/AAAUSDT-1d-2025-02-" in h]
    assert len([h for h in feb_daily if h.endswith(".zip")]) == 28
    # ...and the current month came from daily files up to yesterday (9 March), not today.
    march = [h for h in first if "AAAUSDT-1d-2025-03-" in h and h.endswith(".zip")]
    assert len(march) == 9
    months = sorted(p.name for p in (tmp_path / "binance" / "_months" / "AAAUSDT" / "1d").iterdir())
    assert months == [
        "2024-10.parquet",
        "2024-11.parquet",
        "2024-12.parquet",
        "2025-01.parquet",
        "2025-02.parquet",
    ]

    # A second run reads completed months from the cache and only refetches the current one.
    _CountingHandler.hits = []
    binance.download_all(cfg, tmp_path, today=TODAY)
    # (Months before a symbol's listing are HTTP 404 and are simply probed again.)
    hits = [h for h in _CountingHandler.hits if "BBBUSDT-1d-2024-1" not in h]
    assert hits and all("-2025-03-" in h for h in hits)
    assert not any("AAAUSDT-1d-2024" in h or "-2025-01" in h or "-2025-02" in h for h in hits)


def test_checksum_mismatch_is_fatal(replica, tmp_path):
    base_url, root, _ = replica
    df = simulate_market(31, 1, seed=5, start="2024-10-01", kind="gbm")["SYN00"]
    path = root / "data/spot/monthly/klines/BADUSDT/1d/BADUSDT-1d-2024-10.zip"
    _write_archive(
        path, _kline_csv(df, 86_400_000, micro=False, header=False), corrupt_checksum=True
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        binance.download_symbol(
            "BADUSDT",
            "1d",
            date(2024, 10, 1),
            date(2024, 10, 31),
            tmp_path,
            base_url=base_url,
            today=TODAY,
        )


def test_unknown_symbol_and_unreachable_host(replica, tmp_path):
    base_url, _, _ = replica
    with pytest.raises(FileNotFoundError):
        binance.download_symbol(
            "NOPEUSDT", "1d", date(2024, 10, 1), TODAY, tmp_path, base_url=base_url, today=TODAY
        )
    with pytest.raises(ConnectionError):  # nothing listens on port 9
        binance._fetch("http://127.0.0.1:9/x.zip", retries=1)


def test_preflight_check_reports_listing_gaps(replica, tmp_path, capsys):
    base_url, _, _ = replica
    rows = binance.check_source(_cfg(base_url, tmp_path, "1d"), today=TODAY)
    by_symbol = {r["symbol"]: r for r in rows}
    assert (
        by_symbol["AAAUSDT"]["first_bars"] == 31
        and by_symbol["AAAUSDT"]["first_month"] == "2024-10"
    )
    assert by_symbol["BBBUSDT"]["first_bars"] == 0  # not listed at the start of the sample
    # The last completed month (February) has no monthly archive yet in the replica.
    assert by_symbol["AAAUSDT"]["last_month"] == "2025-02" and not by_symbol["AAAUSDT"]["ok"]
    assert "MISSING" in capsys.readouterr().out


@pytest.mark.parametrize("interval", ["1d", "1h"])
def test_downloaded_data_flows_through_loader_and_universe(replica, tmp_path, interval):
    base_url, _, _ = replica
    cfg = _cfg(base_url, tmp_path, interval)
    binance.download_all(cfg, tmp_path, today=TODAY)
    prices, info = load_market(cfg)
    assert info["source"] == "binance" and info["is_real"] and "Binance" in info["label"]
    # Both assets are trimmed to the common window, which starts at the later listing.
    assert info["start"] == "2024-12-01" and info["end"] == "2025-03-09"
    per_day = 24 if interval == "1h" else 1
    assert info["n_bars"] == 99 * per_day
    assert info["validation_counts"]["gaps"] == 0 and info["validation_counts"]["duplicates"] == 0
    uni = build_universe(prices, cfg)
    assert uni.shape == (99, 1000)  # hourly backtests are summed to 99 daily rows
    assert np.isfinite(uni.gross).all() and (uni.turnover >= 0).all()
