# ruff: noqa: DTZ001 — 테스트 datetime 은 KST naive(프로젝트 표준)
"""PG 쓰기 경로(``execute_values`` + 배치 안 중복 제거) — 실제 PG 에서만 의미가 있다.

``SCALP_IT_TEST_PG_DSN`` 에 버려도 되는 빈 DB 를 주면 돈다(테이블을 만들고 비운다). 예::

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=x -p 127.0.0.1:55440:5432 \\
        --tmpfs /var/lib/postgresql/data postgres:16
    SCALP_IT_TEST_PG_DSN=postgresql://postgres:x@127.0.0.1:55440/postgres uv run pytest tests/test_store_pg.py

기대값은 원래 경로(``executemany``: 한 줄씩, 같은 키는 뒤 행이 덮는다)의 결과다.
"""

import os
from datetime import datetime

import pytest

from scalp_it.intraday.store import ts_key
from scalp_it.realtime.store import (
    init_realtime_tables,
    insert_quotes,
    insert_ticks,
    insert_vi_events,
)
from scalp_it.theme.store import init_theme_tables, upsert_theme_groups, upsert_theme_members

DSN = os.environ.get("SCALP_IT_TEST_PG_DSN")
T = datetime(2026, 10, 29, 9, 0, 1)


def test_ts_key_reads_spool_strings_as_naive_datetime():
    assert ts_key("2026-10-29 09:00:01") == T
    assert ts_key("2026-10-29T09:00:01") == T
    assert ts_key(T) == T
    # 오프셋이 붙으면 합치지 않는다(PG timestamp 는 오프셋을 버려서 순간 비교와 다르다).
    assert ts_key("2026-10-29 09:00:01+09:00") == "2026-10-29 09:00:01+09:00"
    assert ts_key("not a time") == "not a time"


pg = pytest.mark.skipif(not DSN, reason="SCALP_IT_TEST_PG_DSN 없음")


@pytest.fixture()
def conn():
    import psycopg2

    c = psycopg2.connect(DSN)
    cur = c.cursor()
    for t in ("ticks", "quotes", "quote_events", "vi_events", "theme_groups", "theme_members"):
        cur.execute(f"DROP TABLE IF EXISTS {t}")
    c.commit()
    init_realtime_tables(c)
    init_theme_tables(c)
    yield c
    c.close()


def _rows(conn, sql):
    cur = conn.cursor()
    cur.execute(sql)
    out = cur.fetchall()
    cur.close()
    return out


def _quote(ts, bid):
    return {"code": "005930", "ts": ts, "bid1": bid, "ask1": bid + 100}


@pg
def test_quotes_dedup_spool_string_and_datetime_last_wins(conn):
    # 스풀에서 되살린 문자열 ts + 같은 초의 라이브 datetime ts — 전엔 21000 으로 배치가 통째로 막혔다.
    rows = [
        _quote("2026-10-29 09:00:01", 100),
        _quote(T, 200),
        _quote("2026-10-29T09:00:01", 300),
        {**_quote(datetime(2026, 10, 29, 9, 0, 2), 400), "code": "000660"},
    ]
    assert insert_quotes(conn, rows) == 4
    got = _rows(conn, "SELECT code, ts, bid1 FROM quotes ORDER BY code, ts")
    assert got == [("000660", datetime(2026, 10, 29, 9, 0, 2), 400), ("005930", T, 300)]


@pg
def test_quotes_pg_equal_python_different_key_falls_back(conn):
    # 파이썬 키로는 달라도(오프셋 문자열은 정규화하지 않는다) PG 에선 같은 키 → 원래 경로로 다시 넣는다.
    rows = [_quote(T, 100), _quote("2026-10-29 09:00:01+09:00", 200)]
    assert insert_quotes(conn, rows) == 2
    assert _rows(conn, "SELECT ts, bid1 FROM quotes") == [(T, 200)]
    # 커넥션이 다음 배치를 계속 받는다(롤백 뒤 깨끗한 상태).
    assert insert_quotes(conn, [_quote(T, 500)]) == 1
    assert _rows(conn, "SELECT ts, bid1 FROM quotes") == [(T, 500)]


@pg
def test_vi_events_dedup_and_overwrite_existing(conn):
    def vi(ts, rate, vi_type="1"):
        return {"code": "005930", "ts": ts, "vi_type": vi_type, "gap_rate": rate}

    insert_vi_events(conn, [vi(T, 1.0)])
    insert_vi_events(conn, [vi("2026-10-29 09:00:01", 2.0), vi(T, 3.0), vi(T, 4.0, vi_type="2")])
    got = _rows(conn, "SELECT vi_type, gap_rate FROM vi_events ORDER BY vi_type")
    assert got == [("1", 3.0), ("2", 4.0)]


@pg
def test_ticks_duplicate_keys_first_wins(conn):
    def tick(ts, price):
        return {"code": "005930", "ts": ts, "seq": 0, "price": price}

    assert insert_ticks(conn, [tick(T, 100), tick("2026-10-29 09:00:01", 200)]) == 2
    assert _rows(conn, "SELECT price FROM ticks") == [(100,)]


@pg
def test_theme_upserts_dedup_last_wins(conn):
    day = "2026-10-29"
    upsert_theme_groups(conn, day, [
        {"theme_code": "100", "theme_name": "a"},
        {"theme_code": "100", "theme_name": "b"},
        {"theme_code": "200", "theme_name": "c"},
    ])
    upsert_theme_members(conn, day, [
        {"theme_code": "100", "code": "005930", "name": "x"},
        {"theme_code": "100", "code": "005930", "name": "y"},
    ])
    assert _rows(conn, "SELECT theme_code, theme_name FROM theme_groups ORDER BY 1") == [
        ("100", "b"), ("200", "c"),
    ]
    assert _rows(conn, "SELECT name FROM theme_members") == [("y",)]
