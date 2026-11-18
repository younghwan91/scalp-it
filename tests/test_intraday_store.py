"""분봉 저장소 테스트.

운영은 postgres 지만 테스트는 sqlite 로 돈다. 그래서 스키마·SQL 이 양쪽에서
같아야 의미가 있다 — placeholder 만 갈리고 나머지 문자열은 하나다.
"""

import sqlite3

import pandas as pd
import pytest

from scalp_it.intraday.store import (
    _CREATE_TABLE,
    _placeholder,
    collection_universe,
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


def _daily(conn, rows):
    conn.execute(
        "CREATE TABLE daily_bars_adjusted (code text, date text, trade_value real)")
    conn.executemany("INSERT INTO daily_bars_adjusted VALUES (?,?,?)", rows)
    conn.commit()


def test_collection_universe_keeps_recently_collected_codes(conn):
    """ADV 순위 밖으로 밀려도 최근 수집 종목은 계속 받는다 (2026-10-12 감사).

    상위 N 하드캡에선 순위에서 빠진 종목이 그날로 영영 멈췄다.
    """
    days = ["2026-10-06", "2026-10-07", "2026-10-08", "2026-10-12"]
    _daily(conn, [(c, d, tv) for d in days
                  for c, tv in (("TOP1", 900), ("TOP2", 500), ("FELL", 100), ("GONE", 50))])
    upsert_minute_bars(conn, [
        _bar(pd.Timestamp(2026, 10, 7, 15, 30), code="FELL"),   # 창 안에서 수집됨
        _bar(pd.Timestamp(2026, 9, 21, 15, 30), code="GONE"),   # 창 밖에서 멈춤
        _bar(pd.Timestamp(2026, 10, 12, 9, 0), code="TOP1"),
    ])
    # 최근 3 거래일(10-07~10-12) 창: FELL 은 순위 밖이어도 유지, GONE 은 탈락.
    got = collection_universe(conn, 2, lookback_days=4, keep_days=3)
    assert got == ["TOP1", "TOP2", "FELL"]
    # 창을 넓히면 순위 밖 GONE 은 여전히 9-21 이라 안 들어온다(daily 날짜 하한이 10-06).
    assert "GONE" not in collection_universe(conn, 2, lookback_days=4, keep_days=10)


def test_collection_universe_has_no_duplicates(conn):
    _daily(conn, [("TOP1", "2026-10-12", 900), ("TOP2", "2026-10-12", 500)])
    upsert_minute_bars(conn, [_bar(pd.Timestamp(2026, 10, 12, 9, 0), code="TOP1")])
    assert collection_universe(conn, 2, lookback_days=1, keep_days=1) == ["TOP1", "TOP2"]


def test_collection_universe_empty_daily_returns_empty(conn):
    _daily(conn, [])
    upsert_minute_bars(conn, [_bar(pd.Timestamp(2026, 10, 12, 9, 0))])
    assert collection_universe(conn, 5) == []


# ------------------------------------------------- 그날 구독 종목(신규상장) 같은 날 수집
# 신규상장주는 상장 당일 일봉이 없어 ADV 순위에 못 든다. 그날 분봉을 15:45 에 안 받으면 다음 날
# 08:55 감지기가 기준가(전일 정규장 종가)를 몰라 §4-5 고정 편입이 꺼진다(486510 09-29, 10-12 리뷰).

def _ticks(conn, rows):
    conn.execute("CREATE TABLE ticks (code text, ts timestamp, seq integer, price integer)")
    conn.executemany("INSERT INTO ticks VALUES (?,?,?,?)", rows)
    conn.commit()


def test_collection_universe_adds_codes_ticked_that_day(conn):
    from datetime import date

    _daily(conn, [("TOP1", "2026-10-12", 900), ("TOP2", "2026-10-12", 500)])
    _ticks(conn, [
        ("NEWIPO", "2026-10-13 09:02:42", 0, 18_000),   # 상장 당일 — 일봉·ADV 없음
        ("TOP1", "2026-10-13 10:00:00", 0, 1_000),      # 이미 대상 — 중복 없음
        ("OLDDAY", "2026-10-12 10:00:00", 0, 1_000),    # 다른 날 구독 — 안 더한다
    ])
    got = collection_universe(conn, 2, lookback_days=1, keep_days=1, tick_day=date(2026, 10, 13))
    assert got == ["TOP1", "TOP2", "NEWIPO"]
    # tick_day 를 안 주면(옛 동작) 신규상장주는 빠진다 — 이 테스트가 지키는 회귀.
    assert "NEWIPO" not in collection_universe(conn, 2, lookback_days=1, keep_days=1)


def test_collection_universe_survives_missing_ticks_table(conn, capsys):
    from datetime import date

    _daily(conn, [("TOP1", "2026-10-12", 900)])
    got = collection_universe(conn, 1, lookback_days=1, keep_days=1, tick_day=date(2026, 10, 13))
    assert got == ["TOP1"]
    assert "구독 종목 조회 실패" in capsys.readouterr().err


def test_cli_intraday_passes_today_as_tick_day(monkeypatch):
    """15:45 크론 실행줄(--universe-top 1400 --pages 6)은 플래그 없이 그날 구독 종목을 더한다."""
    from datetime import date

    import scalp_it.cli_intraday as ci

    seen = {}

    def fake(conn, limit, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(ci, "collection_universe", fake)
    ci.resolve_codes(ci.build_parser().parse_args(["--universe-top", "1400", "--pages", "6"]), None)
    assert seen["tick_day"] == date.today()  # noqa: DTZ011 — KST naive
    ci.resolve_codes(ci.build_parser().parse_args(["--universe-top", "5", "--no-tick-codes"]), None)
    assert seen["tick_day"] is None


def test_cli_intraday_skips_a_code_on_gateway_502(monkeypatch, capsys):
    """키움 502(httpx.HTTPStatusError)는 그 종목만 실패로 두고 나머지를 계속 받는다.

    2026-11-06 15:45 크론이 1,921/2,557 번째 종목의 502 한 번에 통째로 죽었다.
    """
    import httpx

    import scalp_it.cli_intraday as ci

    class FakeConn:
        def close(self):
            pass

    upserted = []
    monkeypatch.setattr(ci, "connect", lambda dsn: FakeConn())
    monkeypatch.setattr(ci, "init_minute_bars", lambda conn: None)
    monkeypatch.setattr(ci, "latest_ts", lambda conn, code: None)
    monkeypatch.setattr(ci, "count_bars", lambda conn: 0)
    monkeypatch.setattr(ci, "upsert_minute_bars", lambda conn, rows: upserted.append(rows) or len(rows))
    monkeypatch.setattr(ci, "api_credentials", lambda: ("k", "s"))

    req = httpx.Request("POST", "https://api.kiwoom.com/api/dostk/chart")

    class FakeKiwoom:
        def __init__(self, *a, **kw):
            pass

        def minute_chart(self, code, **kw):
            if code == "BAD":
                raise httpx.HTTPStatusError("502 Bad Gateway", request=req,
                                            response=httpx.Response(502, request=req))
            if code == "DROP":
                raise httpx.ConnectError("Temporary failure in name resolution", request=req)
            from datetime import datetime
            return [{"code": code, "ts": datetime(2026, 11, 6, 15, 30)}]  # noqa: DTZ001 — KST naive(프로젝트 표준)

    monkeypatch.setattr(ci, "KiwoomClient", FakeKiwoom)
    rc = ci.main(["--codes", "AAA,BAD,DROP,ZZZ", "--pages", "1"])
    err = capsys.readouterr().err
    assert rc == 0
    assert [rows[0]["code"] for rows in upserted] == ["AAA", "ZZZ"]
    assert "실패 2종목: BAD,DROP" in err


def _fake_intraday(monkeypatch, fail):
    """minute_chart 가 ``fail(code)`` 이 돌려주는 예외를 던지는 가짜 환경."""
    from datetime import datetime

    import scalp_it.cli_intraday as ci

    class FakeConn:
        def close(self):
            pass

    calls = []
    monkeypatch.setattr(ci, "connect", lambda dsn: FakeConn())
    monkeypatch.setattr(ci, "init_minute_bars", lambda conn: None)
    monkeypatch.setattr(ci, "latest_ts", lambda conn, code: None)
    monkeypatch.setattr(ci, "count_bars", lambda conn: 0)
    monkeypatch.setattr(ci, "upsert_minute_bars", lambda conn, rows: len(rows))
    monkeypatch.setattr(ci, "api_credentials", lambda: ("k", "s"))

    class FakeKiwoom:
        def __init__(self, *a, **kw):
            pass

        def minute_chart(self, code, **kw):
            calls.append(code)
            exc = fail(code)
            if exc is not None:
                raise exc
            return [{"code": code, "ts": datetime(2026, 11, 6, 15, 30)}]  # noqa: DTZ001 — KST naive

    monkeypatch.setattr(ci, "KiwoomClient", FakeKiwoom)
    return ci, calls


def test_cli_intraday_stops_after_consecutive_failures(monkeypatch, capsys):
    """키움이 계속 죽어 있으면 2,500종목을 30초 타임아웃으로 다 돌지 않고 멈춘다."""
    import httpx

    req = httpx.Request("POST", "https://api.kiwoom.com/api/dostk/chart")
    ci, calls = _fake_intraday(monkeypatch, lambda code: None if code == "OK1"
                               else httpx.ReadTimeout("timed out", request=req))
    codes = ["OK1"] + [f"C{i}" for i in range(10)]
    rc = ci.main(["--codes", ",".join(codes), "--pages", "1", "--max-consecutive-failures", "3"])
    assert rc == 1
    assert calls == ["OK1", "C0", "C1", "C2"]
    assert "수집 중단 — 연속 3종목 실패" in capsys.readouterr().err


def test_cli_intraday_stops_at_once_on_auth_error(monkeypatch, capsys):
    """401·403 은 종목 탓이 아니다 — 수천 번 토큰을 재발급하지 않고 바로 멈춘다."""
    import httpx

    req = httpx.Request("POST", "https://api.kiwoom.com/api/dostk/chart")
    ci, calls = _fake_intraday(monkeypatch, lambda code: httpx.HTTPStatusError(
        "401", request=req, response=httpx.Response(401, request=req)))
    rc = ci.main(["--codes", "A,B,C", "--pages", "1"])
    assert rc == 1
    assert calls == ["A"]
    assert "인증 오류 HTTP 401" in capsys.readouterr().err
