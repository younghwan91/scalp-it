"""분봉 저장소 테스트.

운영은 postgres 지만 테스트는 sqlite 로 돈다. 그래서 스키마·SQL 이 양쪽에서
같아야 의미가 있다 — placeholder 만 갈리고 나머지 문자열은 하나다.
"""

import sqlite3

import pandas as pd
import pytest

from swing_it.intraday.store import (
    _CREATE_TABLE,
    _placeholder,
    count_bars,
    init_minute_bars,
    latest_ts,
    top_universe,
    upsert_minute_bars,
)


def _bar(ts, code="005930", close=274500, volume=1000):
    return {
        "code": code, "ts": ts,
        "open": close, "high": close, "low": close, "close": close,
        "volume": volume, "trade_value": float(close) * volume,
    }


@pytest.fixture()
def conn(tmp_path):
    connection = sqlite3.connect(tmp_path / "minute.db")
    init_minute_bars(connection)
    yield connection
    connection.close()


def test_init_creates_table_and_ts_index(conn):
    names = {r[0] for r in conn.execute(
        "select name from sqlite_master where type in ('table','index')").fetchall()}
    assert "minute_bars" in names
    assert "idx_minute_bars_ts" in names


def test_init_is_idempotent(conn):
    init_minute_bars(conn)
    init_minute_bars(conn)
    assert count_bars(conn) == 0


def test_schema_names_only_portable_types():
    """postgres 정식 타입만 쓴다. sqlite 는 타입 친화성으로 받아들인다."""
    lowered = _CREATE_TABLE.lower()
    assert "double precision" in lowered
    assert "timestamp" in lowered
    assert "autoincrement" not in lowered
    assert "serial" not in lowered


def test_placeholder_is_qmark_for_sqlite(conn):
    assert _placeholder(conn) == "?"


def test_upsert_inserts_rows(conn):
    rows = [_bar(pd.Timestamp(2026, 8, 14, 9, m)) for m in range(3)]
    assert upsert_minute_bars(conn, rows) == 3
    assert count_bars(conn, "005930") == 3


def test_upsert_is_idempotent(conn):
    rows = [_bar(pd.Timestamp(2026, 8, 14, 9, m)) for m in range(5)]
    upsert_minute_bars(conn, rows)
    upsert_minute_bars(conn, rows)
    upsert_minute_bars(conn, rows)
    assert count_bars(conn) == 5


def test_upsert_overwrites_on_conflict(conn):
    ts = pd.Timestamp(2026, 8, 14, 9, 0)
    upsert_minute_bars(conn, [_bar(ts, close=100, volume=10)])
    upsert_minute_bars(conn, [_bar(ts, close=200, volume=20)])
    row = conn.execute("select close, volume, trade_value from minute_bars").fetchone()
    assert row == (200.0, 20, 4000.0)
    assert count_bars(conn) == 1


def test_upsert_empty_is_zero_not_error(conn):
    assert upsert_minute_bars(conn, []) == 0


def test_primary_key_is_code_and_ts(conn):
    ts = pd.Timestamp(2026, 8, 14, 9, 0)
    upsert_minute_bars(conn, [_bar(ts, code="005930"), _bar(ts, code="000660")])
    assert count_bars(conn) == 2


def test_latest_ts_returns_none_when_empty(conn):
    assert latest_ts(conn, "005930") is None


def test_latest_ts_returns_max_per_code(conn):
    upsert_minute_bars(conn, [
        _bar(pd.Timestamp(2026, 8, 14, 9, 0)),
        _bar(pd.Timestamp(2026, 8, 14, 15, 30)),
        _bar(pd.Timestamp(2026, 8, 15, 9, 0), code="000660"),
    ])
    assert latest_ts(conn, "005930") == pd.Timestamp(2026, 8, 14, 15, 30)
    assert latest_ts(conn, "000660") == pd.Timestamp(2026, 8, 15, 9, 0)


def test_top_universe_ranks_by_average_trade_value(conn):
    conn.executescript(
        """
        CREATE TABLE daily_bars_adjusted (
            code text, date text, open real, close real, trade_value real
        );
        INSERT INTO daily_bars_adjusted VALUES
            ('005930','2026-08-14',1,1,900),
            ('005930','2026-08-13',1,1,900),
            ('000660','2026-08-14',1,1,500),
            ('000660','2026-08-13',1,1,500),
            ('035720','2026-08-14',1,1,100),
            ('035720','2026-08-13',1,1,100);
        """
    )
    conn.commit()
    assert top_universe(conn, 2, lookback_days=5) == ["005930", "000660"]


def test_top_universe_respects_lookback_window(conn):
    conn.executescript(
        """
        CREATE TABLE daily_bars_adjusted (
            code text, date text, open real, close real, trade_value real
        );
        INSERT INTO daily_bars_adjusted VALUES
            ('OLD','2026-01-02',1,1,99999),
            ('NEW','2026-08-14',1,1,10);
        """
    )
    conn.commit()
    # 최근 1 거래일만 보면 옛날에만 거래된 종목은 빠진다.
    assert top_universe(conn, 5, lookback_days=1) == ["NEW"]


def test_top_universe_empty_table_returns_empty(conn):
    conn.execute("CREATE TABLE daily_bars_adjusted (code text, date text, trade_value real)")
    conn.commit()
    assert top_universe(conn, 10) == []
