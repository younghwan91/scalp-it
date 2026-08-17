"""장중 궤적 피처 테스트 — 합성 분봉만 쓴다 (네트워크·실 DB 없음).

가장 중요한 것은 :func:`test_close_auction_bar_does_not_change_features` 다.
15:30 단일가는 종가베팅의 **결과**이므로 피처에 스며들면 안 된다. 나머지 테스트가
전부 통과해도 이 하나가 깨지면 모듈 전체가 무효다.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from scalp_it.intraday.features import (
    INTRADAY_FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    intraday_features,
)
from scalp_it.intraday.loader import available_codes, coverage, load_minute_bars
from scalp_it.intraday.store import init_minute_bars, upsert_minute_bars

DATE = "2026-08-17"

#: 손계산 기준일. ``(hhmm, close, high, low, volume)``.
#: 09:01 에 고가 130 이 한 번 튀고, 15:00 종가 100 이 최대낙폭 저점이며,
#: 15:19 에 120 으로 마감한다.
BASE_DAY = [
    ("09:00", 100.0, 100.0, 100.0, 100),
    ("09:01", 110.0, 130.0, 100.0, 200),
    ("09:02", 109.0, 109.0, 108.0, 100),
    ("13:00", 108.0, 108.0, 107.0, 100),
    ("15:00", 100.0, 100.0, 99.0, 100),
    ("15:19", 120.0, 120.0, 100.0, 400),
]
CLOSE_AUCTION = ("15:30", 150.0, 150.0, 150.0, 9999)


def make_bars(rows, code: str = "000001", date: str = DATE) -> pd.DataFrame:
    """``minute_bars`` 형식 프레임. ``trade_value`` 는 실 DB 처럼 ``close × volume``."""
    return pd.DataFrame(
        [
            {
                "code": code,
                "ts": pd.Timestamp(f"{date} {hhmm}:00"),
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "trade_value": close * volume,
            }
            for hhmm, close, high, low, volume in rows
        ]
    )


@pytest.fixture
def base() -> pd.Series:
    out = intraday_features(make_bars(BASE_DAY))
    assert len(out) == 1
    return out.iloc[0]


def test_late_ret_is_1519_over_1500(base):
    # 120 / 100 - 1
    assert base["late_ret"] == pytest.approx(0.2)


def test_high_time_pct_uses_clock_position(base):
    # 고가 130 은 09:01. (541 - 540) / (919 - 540)
    assert base["high_time_pct"] == pytest.approx(1 / 379)


def test_tv_pm_share_is_afternoon_over_cumulative(base):
    # 오후 10800+10000+48000, 전체 10000+22000+10900+10800+10000+48000
    assert base["tv_pm_share"] == pytest.approx(68800 / 111700)


def test_n_shock_counts_3pct_minute_moves(base):
    # +10.0%, -0.91%, -0.92%, -7.41%, +20.0% → 3 회
    assert base["n_shock"] == 3


def test_max_dd_is_peak_to_trough_on_closes(base):
    # 누적 고점 110 → 저점 100
    assert base["max_dd"] == pytest.approx(1 - 100 / 110)


def test_recovery_is_rebound_from_drawdown_trough(base):
    # (120 - 100) / 100
    assert base["recovery"] == pytest.approx(0.2)


def test_trend_r2_is_signed_r_squared(base):
    # 종가 [100,110,109,108,100,120] 를 0..5 에 회귀. 기울기 양수.
    assert base["trend_r2"] == pytest.approx(1190.25 / 4844.583333, rel=1e-9)


def test_close_vs_vwap_uses_close_times_volume(base):
    # VWAP = 111700 / 1000 = 111.7
    assert base["close_vs_vwap"] == pytest.approx(120 / 111.7 - 1)


def test_p1519_and_tv1519_are_exposed(base):
    assert base["p1519"] == pytest.approx(120.0)
    assert base["tv1519"] == pytest.approx(111700.0)


def test_close_auction_bar_does_not_change_features():
    """**회귀 테스트.** 15:30 봉을 붙여도 피처 8개가 한 톨도 안 변해야 한다."""
    without = intraday_features(make_bars(BASE_DAY)).iloc[0]
    with_auction = intraday_features(make_bars([*BASE_DAY, CLOSE_AUCTION])).iloc[0]

    for col in [*INTRADAY_FEATURE_COLUMNS, "p1519", "tv1519"]:
        assert with_auction[col] == pytest.approx(without[col]), col
    # 단일가 자체는 라벨용으로 실려 나온다.
    assert with_auction["p1530"] == pytest.approx(150.0)
    assert np.isnan(without["p1530"])


def test_late_ret_falls_back_to_first_bar_after_1500():
    rows = [
        ("09:00", 100.0, 100.0, 100.0, 100),
        ("14:59", 100.0, 100.0, 100.0, 100),
        ("15:07", 80.0, 80.0, 80.0, 100),  # 15:00 봉이 없다 → 이 봉이 기준
        ("15:19", 88.0, 88.0, 80.0, 100),
    ]
    row = intraday_features(make_bars(rows)).iloc[0]
    assert row["late_ret"] == pytest.approx(88 / 80 - 1)


def test_recovery_is_zero_when_no_drawdown():
    rows = [
        ("09:00", 100.0, 100.0, 100.0, 100),
        ("12:00", 105.0, 105.0, 100.0, 100),
        ("15:19", 110.0, 110.0, 105.0, 100),
    ]
    row = intraday_features(make_bars(rows)).iloc[0]
    assert row["max_dd"] == pytest.approx(0.0)
    assert row["recovery"] == 0.0


def test_trend_r2_is_zero_with_two_bars():
    rows = [
        ("09:00", 100.0, 100.0, 100.0, 100),
        ("15:19", 120.0, 120.0, 100.0, 100),
    ]
    row = intraday_features(make_bars(rows)).iloc[0]
    # 2점은 항상 R²=1 이라 '일관성'을 재지 못한다.
    assert row["trend_r2"] == 0.0


def test_no_leakage_between_codes():
    other = [
        ("09:00", 50.0, 50.0, 50.0, 100),
        ("13:00", 45.0, 45.0, 45.0, 100),
        ("15:00", 40.0, 40.0, 40.0, 100),
        ("15:19", 44.0, 44.0, 40.0, 100),
    ]
    mixed = pd.concat(
        [
            make_bars([*BASE_DAY, CLOSE_AUCTION], code="000001"),
            make_bars(other, code="000002"),
        ]
    ).sample(frac=1.0, random_state=0)  # 순서를 섞어도 결과가 같아야 한다

    out = intraday_features(mixed).set_index("code")
    alone = intraday_features(make_bars(BASE_DAY, code="000001")).iloc[0]

    for col in INTRADAY_FEATURE_COLUMNS:
        assert out.loc["000001", col] == pytest.approx(alone[col]), col
    assert out.loc["000002", "p1519"] == pytest.approx(44.0)
    assert out.loc["000002", "late_ret"] == pytest.approx(44 / 40 - 1)
    assert out.loc["000002", "max_dd"] == pytest.approx(1 - 40 / 50)


def test_day_with_only_close_auction_bar_is_dropped():
    bars = pd.concat(
        [
            make_bars(BASE_DAY, date="2026-08-17"),
            make_bars([CLOSE_AUCTION], date="2026-08-18"),  # 정규장 봉이 없다
        ]
    )
    out = intraday_features(bars)
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2026-08-17"]


def test_output_contract_and_empty_input():
    out = intraday_features(make_bars(BASE_DAY))
    assert list(out.columns) == OUTPUT_COLUMNS
    empty = intraday_features(make_bars(BASE_DAY).iloc[0:0])
    assert list(empty.columns) == OUTPUT_COLUMNS
    assert empty.empty


def test_missing_column_raises():
    bad = make_bars(BASE_DAY).drop(columns=["volume"])
    with pytest.raises(ValueError, match="volume"):
        intraday_features(bad)


# --- loader — sqlite fixture 로만. 실 DB 는 건드리지 않는다. -------------------


@pytest.fixture
def db():
    conn = init_minute_bars(sqlite3.connect(":memory:"))
    rows = pd.concat(
        [
            make_bars([*BASE_DAY, CLOSE_AUCTION], code="000001", date="2026-08-17"),
            make_bars(BASE_DAY, code="000002", date="2026-08-18"),
        ]
    )
    upsert_minute_bars(conn, rows.to_dict("records"))
    yield conn
    conn.close()


def test_load_minute_bars_includes_whole_end_day(db):
    df = load_minute_bars(db, "2026-08-17", "2026-08-17")
    # 15:30 봉까지 들어와야 한다 — 자르는 건 피처 쪽 몫이다.
    assert len(df) == len(BASE_DAY) + 1
    assert set(df["code"]) == {"000001"}
    assert df["ts"].max() == pd.Timestamp("2026-08-17 15:30:00")


def test_load_minute_bars_skips_absent_codes(db):
    df = load_minute_bars(db, "2026-08-17", "2026-08-18", codes=["000002", "999999"])
    assert set(df["code"]) == {"000002"}
    assert load_minute_bars(db, "2026-08-17", "2026-08-18", codes=[]).empty


def test_available_codes_and_coverage(db):
    assert available_codes(db) == ["000001", "000002"]
    cov = coverage(db)
    assert cov["codes"] == 2
    assert cov["rows"] == len(BASE_DAY) * 2 + 1
    assert cov["start"] == pd.Timestamp("2026-08-17 09:00:00")
    assert cov["end"] == pd.Timestamp("2026-08-18 15:19:00")


def test_coverage_on_empty_table():
    conn = init_minute_bars(sqlite3.connect(":memory:"))
    assert coverage(conn) == {"codes": 0, "rows": 0, "start": None, "end": None}
    assert available_codes(conn) == []
    conn.close()
