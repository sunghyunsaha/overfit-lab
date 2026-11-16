"""The loader finds, fixes and logs each kind of data problem."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from overfitlab.config import load_config
from overfitlab.data import binance
from overfitlab.data.coinmetrics import parse_coinmetrics_csv
from overfitlab.data.loader import ValidationLog, load_market, validate_ohlcv
from overfitlab.data.synthetic import simulate_market


def _dirty() -> pd.DataFrame:
    df = simulate_market(400, 1, seed=11, kind="gbm", vol_annual=0.5)["SYN00"].copy()
    df = df.drop(df.index[[100, 101, 102]])  # a 3-bar gap
    df = pd.concat([df, df.iloc[[50]]])  # a duplicated timestamp (unsorted, too)
    df.iloc[200, df.columns.get_loc("close")] = np.nan  # a bad price
    df.iloc[210, df.columns.get_loc("high")] = df["low"].iloc[210] * 0.5  # high below low
    df.iloc[220, df.columns.get_loc("volume")] = 0.0  # zero volume
    df.iloc[300, df.columns.get_loc("close")] *= 3.0  # a 200% one-bar spike
    return df


def test_validation_finds_and_logs_every_problem():
    log = ValidationLog()
    clean = validate_ohlcv(_dirty(), "TEST", "1d", log)
    counts = {r["check"]: r["n"] for r in log.records}
    assert counts["duplicates"] == 1
    assert counts["bad_prices"] == 1
    assert counts["gaps"] == 4  # 3 dropped bars + the bad-price bar
    assert counts["ohlc_inconsistent"] >= 1
    assert counts["zero_volume"] == 1
    assert counts["outliers"] >= 1
    # The cleaned frame is a complete regular grid of valid, consistent bars.
    assert clean.index.is_monotonic_increasing and clean.index.is_unique
    assert len(clean) == 400
    assert clean[["open", "high", "low", "close"]].notna().all().all()
    assert (clean["high"] >= clean[["open", "close"]].max(axis=1)).all()
    assert (clean["low"] <= clean[["open", "close"]].min(axis=1)).all()
    # Gap bars are flat at the previous close with zero volume.
    gap_bar = clean.iloc[100]
    assert gap_bar["close"] == clean["close"].iloc[99] == gap_bar["high"] == gap_bar["low"]
    assert gap_bar["volume"] == 0.0


def test_outliers_are_only_altered_on_request():
    kept = validate_ohlcv(_dirty(), "TEST", "1d", ValidationLog())
    fixed = validate_ohlcv(_dirty(), "TEST", "1d", ValidationLog(), repair_outliers=True)
    spike = kept.index[303]  # original row 300 after the gap is refilled
    assert kept["close"].pct_change().abs().max() > 1.0
    assert fixed["close"].pct_change().abs().loc[spike] == 0.0


def test_clean_data_produces_an_empty_log():
    log = ValidationLog()
    df = simulate_market(300, 1, seed=12, kind="gbm", vol_annual=0.5)["SYN00"]
    out = validate_ohlcv(df, "OK", "1d", log)
    assert log.records == []
    pd.testing.assert_frame_equal(out, df, check_freq=False, check_names=False)


def test_load_market_synthetic_is_aligned_and_labelled():
    cfg = load_config(
        overrides={"data": {"source": "synthetic"}, "synthetic": {"n_assets": 3, "n_bars": 200}}
    )
    prices, info = load_market(cfg)
    assert info["source"] == "synthetic" and not info["is_real"]
    assert "SYNTHETIC" in info["label"]
    idx = [df.index for df in prices.values()]
    assert all(i.equals(idx[0]) for i in idx) and info["n_bars"] == 200


def _zip(csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BTCUSDT-1d-2018-01.csv", csv_text)
    return buf.getvalue()


def test_binance_parser_handles_ms_us_and_header():
    row = "{t},100.0,110.0,90.0,105.0,12.5,{c},1300.0,42,6.0,630.0,0\n"
    ms = 1514764800000  # 2018-01-01 00:00 UTC in milliseconds
    plain = "".join(
        row.format(t=ms + i * 86_400_000, c=ms + (i + 1) * 86_400_000 - 1) for i in range(3)
    )
    df = binance.parse_kline_zip(_zip(plain))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index[0] == pd.Timestamp("2018-01-01", tz="UTC")
    assert df.index[2] == pd.Timestamp("2018-01-03", tz="UTC")
    assert df["close"].iloc[0] == 105.0 and df["volume"].iloc[0] == 12.5

    us = 1735689600000000  # 2025-01-01 00:00 UTC in MICROseconds (format change)
    micro = row.format(t=us, c=us + 86_400_000_000 - 1)
    assert binance.parse_kline_zip(_zip(micro)).index[0] == pd.Timestamp("2025-01-01", tz="UTC")

    header = "open_time,open,high,low,close,volume,close_time,quote_volume,count,tb,tq,ignore\n"
    assert len(binance.parse_kline_zip(_zip(header + plain))) == 3


def test_binance_helpers():
    assert binance.kline_url("BTCUSDT", "1d", 2018, 1).endswith(
        "/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2018-01.zip"
    )
    assert binance.kline_url("BTCUSDT", "1h", 2024, 3, 9).endswith(
        "/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2024-03-09.zip"
    )
    months = binance.month_range(
        pd.Timestamp("2018-11-15").date(), pd.Timestamp("2019-02-01").date()
    )
    assert months == [(2018, 11), (2018, 12), (2019, 1), (2019, 2)]
    payload = b"hello"
    digest = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert binance.verify_checksum(payload, f"{digest}  file.zip\n")
    assert not binance.verify_checksum(b"tampered", f"{digest}  file.zip\n")


def test_coinmetrics_parser_builds_close_only_bars():
    text = (
        "time,PriceUSD,volume_reported_spot_usd_1d,Other\n"
        "2020-01-01,,1,x\n2020-01-02,100,5,x\n2020-01-03,110,6,x\n2020-01-04,99,7,x\n"
    )
    df = parse_coinmetrics_csv(text)
    assert len(df) == 3 and df.index[0] == pd.Timestamp("2020-01-02", tz="UTC")
    assert df["open"].tolist() == [100.0, 100.0, 110.0]
    assert df["high"].tolist() == [100.0, 110.0, 110.0]
    assert df["low"].tolist() == [100.0, 100.0, 99.0]
    assert df["volume"].tolist() == [5.0, 6.0, 7.0]
