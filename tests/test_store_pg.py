"""PG 쓰기 경로(``pg_write``: ``execute_values`` + 배치 안 중복 제거 + 옛 행 분리) — 실제 PG 에서만 의미가 있다.

``SCALP_IT_TEST_PG_DSN`` 에 버려도 되는 빈 DB 를 주면 돈다(테이블을 지우고 다시 만든다). 운영 DB 와 같은
TimescaleDB 이미지에서 돌려야 압축 청크 테스트까지 돈다. 예::

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=x -p 127.0.0.1:55440:5432 \\
        --tmpfs /var/lib/postgresql/data timescale/timescaledb:2.17.2-pg16
    SCALP_IT_TEST_PG_DSN=postgresql://postgres:x@127.0.0.1:55440/postgres uv run pytest tests/test_store_pg.py

기대값은 원래 경로(``executemany``: 한 줄씩, 같은 키는 뒤 행이 덮는다)의 결과다.
"""

import os
from datetime import datetime, timedelta

import pytest

from scalp_it.intraday.store import FAST_PATH_MAX_AGE, ts_key
from scalp_it.realtime.store import (
    init_realtime_tables,
    insert_quote_events,
    insert_quotes,
    insert_ticks,
    insert_vi_events,
)
from scalp_it.theme.store import init_theme_tables, upsert_theme_groups, upsert_theme_members

DSN = os.environ.get("SCALP_IT_TEST_PG_DSN")
# 빠른 경로(FAST_PATH_MAX_AGE 안)를 타도록 날짜를 지금 기준으로 잡는다. KST naive(프로젝트 표준).
T = datetime.now().replace(microsecond=0) - timedelta(days=1)  # noqa: DTZ005
OLD = T - timedelta(days=40)


def test_ts_key_reads_only_str_datetime_shapes():
    assert ts_key(str(T)) == T
    assert ts_key(T.isoformat()) == T
    assert ts_key(str(T + timedelta(microseconds=500))) == T + timedelta(microseconds=500)
    assert ts_key(T) == T
    # 오프셋·소수 7자리 이상·그 밖의 모양은 합치지 않는다(PG 와 해석이 달라 행을 잃을 수 있다).
    for s in (f"{T}+09:00", f"{T}Z", f"{T}.0000009", f"{T}.123456789", f" {T}", "not a time",
              "2026-10-29 24:00:00", "2026-10-29 23:59:60"):
        assert ts_key(s) == s


pg = pytest.mark.skipif(not DSN, reason="SCALP_IT_TEST_PG_DSN 없음")


def _fresh(*, timescale: bool):
    import psycopg2
    import psycopg2.errors

    c = psycopg2.connect(DSN)
    cur = c.cursor()
    if timescale:
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
    for t in ("ticks", "quotes", "quote_events", "vi_events", "theme_groups", "theme_members"):
        for attempt in range(5):  # 압축 정책 백그라운드 작업과 DROP 이 교착할 수 있다
            try:
                cur.execute(f"DROP TABLE IF EXISTS {t}")
                c.commit()
                break
            except psycopg2.errors.DeadlockDetected:
                c.rollback()
                if attempt == 4:
                    raise
    init_realtime_tables(c)
    init_theme_tables(c)
    return c


@pytest.fixture()
def conn():
    c = _fresh(timescale=False)
    yield c
    c.rollback()  # 실패한 테스트가 연 트랜잭션이 다음 테스트의 DROP 을 막지 않게
    c.close()


@pytest.fixture()
def ts_conn():
    c = _fresh(timescale=True)
    yield c
    c.rollback()
    c.close()


def _rows(conn, sql):
    cur = conn.cursor()
    cur.execute(sql)
    out = cur.fetchall()
    cur.close()
    return out


def _quote(ts, bid, code="005930"):
    return {"code": code, "ts": ts, "bid1": bid, "ask1": bid + 100}


def _tick(ts, seq, price=1, code="005930"):
    return {"code": code, "ts": ts, "seq": seq, "price": price}


@pg
def test_quotes_dedup_spool_string_and_datetime_last_wins(conn):
    # 스풀에서 되살린 문자열 ts + 같은 초의 라이브 datetime ts — Antigravity 판은 21000 으로 배치가 막혔다.
    rows = [
        _quote(str(T), 100),
        _quote(T, 200),
        _quote(T.isoformat(), 300),
        _quote(T + timedelta(seconds=1), 400, code="000660"),
    ]
    assert insert_quotes(conn, rows) == 4
    got = _rows(conn, "SELECT code, ts, bid1 FROM quotes ORDER BY code, ts")
    assert got == [("000660", T + timedelta(seconds=1), 400), ("005930", T, 300)]


@pg
def test_quotes_pg_equal_python_different_key_falls_back(conn):
    # 파이썬 키로는 달라도(오프셋 문자열은 정규화하지 않는다) PG 에선 같은 키 → 원래 경로로 다시 넣는다.
    rows = [_quote(T, 100), _quote(f"{T}+09:00", 200)]
    assert insert_quotes(conn, rows) == 2
    assert _rows(conn, "SELECT ts, bid1 FROM quotes") == [(T, 200)]
    # 커넥션이 다음 배치를 계속 받는다(롤백 뒤 깨끗한 상태).
    assert insert_quotes(conn, [_quote(T, 500)]) == 1
    assert _rows(conn, "SELECT ts, bid1 FROM quotes") == [(T, 500)]


@pg
def test_quotes_dedup_keeps_last_position_across_pages(conn):
    # 같은 키는 마지막 자리에 둔다. 첫 자리에 두면 PG 에서만 같은 키(오프셋 문자열)인 행과의 순서가
    # 페이지 경계 너머로 뒤집혀 21000 없이 최종값이 달라진다(리뷰 M1).
    others = [_quote(T + timedelta(seconds=10 + i), i, code="000660") for i in range(1999)]
    rows = [_quote(T, 1), *others, _quote(f"{T}+09:00", -1), _quote(T, -2)]
    insert_quotes(conn, rows)
    assert _rows(conn, "SELECT bid1 FROM quotes WHERE code = '005930'") == [(-2,)]


@pg
def test_quotes_seven_digit_fraction_is_not_merged(conn):
    # 파이썬은 소수 7자리를 자르고 PG 는 반올림한다 — 합치면 PG 에서 다른 행을 잃는다(리뷰 M2).
    insert_quotes(conn, [_quote(f"{T}.0000009", 1), _quote(T, 2)])
    assert len(_rows(conn, "SELECT 1 FROM quotes")) == 2


@pg
def test_vi_events_dedup_and_overwrite_existing(conn):
    def vi(ts, rate, vi_type="1"):
        return {"code": "005930", "ts": ts, "vi_type": vi_type, "gap_rate": rate}

    insert_vi_events(conn, [vi(T, 1.0)])
    insert_vi_events(conn, [vi(str(T), 2.0), vi(T, 3.0), vi(T, 4.0, vi_type="2")])
    got = _rows(conn, "SELECT vi_type, gap_rate FROM vi_events ORDER BY vi_type")
    assert got == [("1", 3.0), ("2", 4.0)]


@pg
def test_ticks_duplicate_keys_first_wins(conn):
    assert insert_ticks(conn, [_tick(T, 0, 100), _tick(str(T), 0, 200)]) == 2
    assert _rows(conn, "SELECT price FROM ticks") == [(100,)]


@pg
def test_old_and_new_rows_in_one_batch(conn):
    # 옛 행(원래 경로)과 최근 행(한 문장)이 섞여도 전부 들어가고 같은 키는 덮인다.
    insert_quotes(conn, [_quote(OLD, 1), _quote(T, 2)])
    insert_quotes(conn, [_quote(OLD, 3), _quote(str(OLD), 4), _quote(T, 5), _quote(str(T), 6)])
    assert _rows(conn, "SELECT ts, bid1 FROM quotes ORDER BY ts") == [(OLD, 4), (T, 6)]


def _timescale(conn) -> bool:
    rows = _rows(conn, "SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
    return bool(rows) and bool(
        _rows(conn, "SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'ticks'")
    )


@pg
@pytest.mark.parametrize("table", ["ticks", "quote_events"])
@pytest.mark.parametrize("age", [timedelta(days=40), timedelta(days=2)], ids=["policy-age", "manual"])
def test_compressed_chunk_conflict_does_not_drop_rows(ts_conn, table, age):
    # TimescaleDB 2.17: 압축 청크와 충돌한 다중 행 ON CONFLICT DO NOTHING 은 그 뒤 행을 오류 없이 버린다.
    # 오래된 스풀 + 오늘 틱이 한 배치로 오면 오늘 틱이 사라졌다(리뷰 C1). 압축 청크에 닿을 행은 행마다
    # 보내야 한다. manual: 정책 나이(14일)보다 새 청크를 손으로 압축해도 경계를 조회해 잡는다.
    c = ts_conn
    old_ts = T - age
    if not _timescale(c):
        pytest.skip("timescaledb 없음 — 압축 청크 재현 불가")
    insert = insert_ticks if table == "ticks" else insert_quote_events
    seq = "seq" if table == "ticks" else "qseq"

    def row(ts, n):
        return {"code": "005930", "ts": ts, seq: n, "price": n, "bid1": n}

    insert(c, [row(old_ts, n) for n in range(50)])
    _rows(c, f"SELECT compress_chunk(ch) FROM show_chunks('{table}') ch")
    c.commit()
    compressed = _rows(c, "SELECT count(*) FROM timescaledb_information.chunks "
                          f"WHERE hypertable_name = '{table}' AND is_compressed")
    assert compressed[0][0] >= 1
    # T 도 압축 청크(손 압축)에 들어갈 수 있어, 확실히 새 청크인 시각을 따로 둔다
    now = T + timedelta(days=30) if age < FAST_PATH_MAX_AGE else T
    batch = [row(old_ts, 0)] + [row(old_ts, n) for n in range(50, 70)] + [row(now, n) for n in range(3000)]
    assert insert(c, batch) == len(batch)
    assert _rows(c, f"SELECT count(*) FROM {table} WHERE ts = '{old_ts}'")[0][0] == 70
    assert _rows(c, f"SELECT count(*) FROM {table} WHERE ts = '{now}'")[0][0] == 3000


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


def test_fast_path_window_is_inside_compress_after():
    from scalp_it.realtime.store import _COMPRESS_AFTER

    assert FAST_PATH_MAX_AGE < timedelta(days=int(_COMPRESS_AFTER.split()[0]))
