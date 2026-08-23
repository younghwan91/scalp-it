# ruff: noqa: DTZ001 — 테스트 datetime 은 KST naive(프로젝트 표준)
"""실시간 저장소 테스트 — 운영은 postgres 지만 테스트는 sqlite fixture 로 돈다.

스키마·SQL 이 양쪽에서 같아야 의미가 있으므로 placeholder 만 갈린다.
"""

import json
import sqlite3
from datetime import datetime

import pytest

from scalp_it.realtime.store import (
    _CREATE_QUOTES,
    _CREATE_TICKS,
    QUOTE_COLUMNS,
    TICK_COLUMNS,
    count_quotes,
    count_ticks,
    event_candidates,
    init_realtime_tables,
    insert_quotes,
    insert_ticks,
)


@pytest.fixture()
def conn(tmp_path):
    c = sqlite3.connect(tmp_path / "rt.db")
    init_realtime_tables(c)
    yield c
    c.close()


def _tick(ts, code="005930", seq=0, **kw):
    row = {
        "code": code, "ts": ts, "seq": seq, "price": 274500, "volume": 30,
        "side": -1, "strength": 120.5, "cum_volume": 1_000_000,
        "best_bid": 274500, "best_ask": 274600, "raw": {"20": "093015"},
    }
    row.update(kw)
    return row


def _quote(ts, code="005930", **kw):
    row = {"code": code, "ts": ts, "total_bid_qty": 500, "total_ask_qty": 400,
           "vi_flag": None, "raw": {"21": "093015"}}
    for i in range(1, 11):
        row[f"bid{i}"] = 274000 + i
        row[f"ask{i}"] = 275000 + i
        row[f"bidqty{i}"] = i * 10
        row[f"askqty{i}"] = i * 5
    row.update(kw)
    return row


def test_init_creates_tables_and_indexes(conn):
    names = {r[0] for r in conn.execute(
        "select name from sqlite_master where type in ('table','index')").fetchall()}
    assert {"ticks", "quotes", "idx_ticks_ts", "idx_quotes_ts"} <= names


def test_init_idempotent(conn):
    init_realtime_tables(conn)
    init_realtime_tables(conn)
    assert count_ticks(conn) == 0


def test_schema_names_only_portable_types():
    for ddl in (_CREATE_TICKS, _CREATE_QUOTES):
        low = ddl.lower()
        assert "serial" not in low
        assert "autoincrement" not in low
        assert "jsonb" not in low  # sqlite 에 없다 — raw 는 text
    assert "double precision" in _CREATE_TICKS.lower()


def test_quote_columns_cover_ten_levels():
    for i in range(1, 11):
        for prefix in ("bid", "ask", "bidqty", "askqty"):
            assert f"{prefix}{i}" in QUOTE_COLUMNS
    assert {"total_bid_qty", "total_ask_qty", "vi_flag", "raw"} <= set(QUOTE_COLUMNS)
    assert TICK_COLUMNS[-1] == "raw"


def test_insert_ticks_roundtrip(conn):
    ts = datetime(2026, 8, 24, 9, 30, 15)
    n = insert_ticks(conn, [_tick(ts, seq=0), _tick(ts, seq=1)])
    assert n == 2
    assert count_ticks(conn) == 2
    row = conn.execute(
        "select price, volume, side, strength, best_bid, raw from ticks "
        "where seq = 0").fetchone()
    assert row[0] == 274500
    assert row[2] == -1
    assert row[3] == pytest.approx(120.5)
    assert json.loads(row[5]) == {"20": "093015"}


def test_insert_ticks_idempotent_on_pk(conn):
    ts = datetime(2026, 8, 24, 9, 30, 15)
    insert_ticks(conn, [_tick(ts, seq=0, price=100)])
    insert_ticks(conn, [_tick(ts, seq=0, price=999)])  # 같은 PK — DO NOTHING
    assert count_ticks(conn) == 1
    assert conn.execute("select price from ticks").fetchone()[0] == 100


def test_insert_quotes_upsert_overwrites(conn):
    ts = datetime(2026, 8, 24, 9, 30, 15)
    insert_quotes(conn, [_quote(ts, total_bid_qty=1)])
    insert_quotes(conn, [_quote(ts, total_bid_qty=999)])  # 같은 (code,ts) — 덮는다
    assert count_quotes(conn) == 1
    assert conn.execute("select total_bid_qty from quotes").fetchone()[0] == 999
    assert conn.execute("select bid1, ask10 from quotes").fetchone() == (274001, 275010)


def test_insert_empty_is_zero(conn):
    assert insert_ticks(conn, []) == 0
    assert insert_quotes(conn, []) == 0


def _daily_fixture(conn):
    conn.execute(
        "CREATE TABLE daily_bars_adjusted (code text, date text, open real, "
        "close real, trade_value real)")
    rows = [
        # (code, date, close, tv) — 20240102 가 최신 거래일
        ("A", "2024-01-01", 100, 9000), ("A", "2024-01-02", 130, 9000),  # +30%
        ("B", "2024-01-01", 100, 8000), ("B", "2024-01-02", 110, 8000),  # +10%
        ("C", "2024-01-01", 100, 1000), ("C", "2024-01-02", 200, 1000),  # 급등이나 tv 미달
        ("D", "2024-01-01", 100, 7000), ("D", "2024-01-02", 105, 7000),
    ]
    for code, d, close, tv in rows:
        conn.execute("INSERT INTO daily_bars_adjusted VALUES (?,?,?,?,?)",
                     (code, d, close, close, tv))
    conn.commit()


def test_event_candidates_combines_movers_and_liquidity(conn):
    _daily_fixture(conn)
    codes = event_candidates(conn, 4, tv_floor=5000.0)
    assert "A" in codes  # 최대 급등 + 유동성
    assert "C" not in codes  # 거래대금 하한 미달로 급등에서 탈락
    assert len(codes) <= 4
