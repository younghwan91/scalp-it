# ruff: noqa: DTZ001 — 테스트 datetime 은 KST naive(프로젝트 표준)
"""``scalp-tick-health`` 판정 로직 테스트 — sqlite fixture 로.

정상 / 부호결손(전부 +1) / 빈 케이스를 넣어 각 항목의 OK·경고·실패 판정과
종료코드를 고정한다. daily_bars 를 만들어 가격 스케일 검사도 태운다.
"""

import sqlite3
from datetime import datetime, timedelta

import pytest

from scalp_it import cli_tick_health as H
from scalp_it.realtime.store import (
    init_realtime_tables,
    insert_quotes,
    insert_ticks,
    insert_vi_events,
)

# 검사 기준 시각 — 장중(평일 10:00)으로 고정해 '장중 데이터 기대' 경로를 태운다.
NOW = datetime(2026, 8, 24, 10, 0, 0)  # 월요일
TRADE_DATE = "2026-08-24"


@pytest.fixture()
def conn(tmp_path):
    c = sqlite3.connect(tmp_path / "rt.db")
    init_realtime_tables(c)
    _make_daily_bars(c)
    yield c
    c.close()


def _make_daily_bars(c):
    c.execute(
        "create table if not exists daily_bars ("
        "code text, date text, open real, high real, low real, close real, "
        "volume real, trade_value real)"
    )
    # 005930 당일 고저 274000~276000. 스케일 검사 기준.
    c.execute(
        "insert into daily_bars(code,date,open,high,low,close) values (?,?,?,?,?,?)",
        ("005930", TRADE_DATE, 274500, 276000, 274000, 275000),
    )
    c.commit()


def _tick(secs, *, code="005930", seq=0, side=-1, price=274500, strength=120.5, **kw):
    ts = (NOW - timedelta(seconds=secs)).strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "code": code, "ts": ts, "seq": seq, "price": price, "volume": 30,
        "side": side, "strength": strength, "cum_volume": 1_000_000,
        "best_bid": 274500, "best_ask": 274600, "raw": {"15": "-30", "10": "274500"},
    }
    row.update(kw)
    return row


def _quote(secs, *, code="005930", **kw):
    ts = (NOW - timedelta(seconds=secs)).strftime("%Y-%m-%d %H:%M:%S")
    row = {"code": code, "ts": ts, "total_bid_qty": 500, "total_ask_qty": 400,
           "vi_flag": None, "raw": {"21": "100000"}}
    for i in range(1, 11):
        row[f"bid{i}"] = 274000 - i
        row[f"ask{i}"] = 275000 + i
        row[f"bidqty{i}"] = i * 10
        row[f"askqty{i}"] = i * 5
    row.update(kw)
    return row


def _healthy(conn, n=20):
    """정상 수집: 매수·매도 섞이고, 강도 분포·가격 정합·호가 정상."""
    ticks = []
    for i in range(n):
        side = 1 if i % 2 == 0 else -1
        ticks.append(_tick(i, seq=i, side=side, strength=100 + i))
    insert_ticks(conn, ticks)
    insert_quotes(conn, [_quote(i, code="005930") for i in range(n)])


def _by_name(checks):
    return {c.name: c for c in checks}


# --------------------------------------------------------------- 정상 케이스


def test_healthy_all_ok_and_exit_zero(conn):
    _healthy(conn)
    checks = H.run_checks(conn, minutes=15, expected=1, now=NOW)
    named = _by_name(checks)
    assert named["1. 수집 여부"].status is H.Status.OK
    assert named["2. 체결 방향 부호"].status is H.Status.OK
    assert named["3. 체결강도 분포"].status is H.Status.OK
    assert named["4. 가격 정합"].status is H.Status.OK
    assert named["5. 호가"].status is H.Status.OK
    assert H.worst_status(checks) is H.Status.OK


def test_side_ratio_reported(conn):
    _healthy(conn)
    checks = H.run_checks(conn, minutes=15, expected=1, now=NOW)
    detail = _by_name(checks)["2. 체결 방향 부호"].detail
    assert "매수비율" in detail


# --------------------------------------------------------- 부호 결손(핵심 검사)


def test_all_buy_side_is_fail(conn):
    """side 전부 +1 → 매수/매도 부호 미매핑, 실패 + raw 덤프."""
    insert_ticks(conn, [_tick(i, seq=i, side=1) for i in range(20)])
    insert_quotes(conn, [_quote(i) for i in range(20)])
    checks = H.run_checks(conn, minutes=15, expected=1, now=NOW)
    side = _by_name(checks)["2. 체결 방향 부호"]
    assert side.status is H.Status.FAIL
    assert "미매핑" in side.detail
    assert side.raw is not None  # 항목 6: 이상 시 raw 원문 덤프
    # 실패가 있으면 CLI 종료코드 비0
    assert H.worst_status(checks) is H.Status.FAIL


def test_all_null_side_is_fail(conn):
    insert_ticks(conn, [_tick(i, seq=i, side=None) for i in range(10)])
    checks = H.run_checks(conn, minutes=15, expected=1, now=NOW)
    assert _by_name(checks)["2. 체결 방향 부호"].status is H.Status.FAIL


# --------------------------------------------------------------- 빈 케이스


def test_empty_during_market_is_fail(conn):
    """장중인데 0건이면 연결·구독 문제 → 실패."""
    checks = H.run_checks(conn, minutes=15, expected=80, now=NOW)
    coll = _by_name(checks)["1. 수집 여부"]
    assert coll.status is H.Status.FAIL
    assert H.worst_status(checks) is H.Status.FAIL


def test_empty_after_hours_is_ok(conn):
    """장 마감 후 0건은 정상 — '수집 없음(장중 아님)'."""
    after = datetime(2026, 8, 24, 18, 0, 0)  # 월 18:00
    checks = H.run_checks(conn, minutes=15, expected=80, now=after)
    coll = _by_name(checks)["1. 수집 여부"]
    assert coll.status is H.Status.OK
    assert "장중 아님" in coll.detail
    assert H.worst_status(checks) is H.Status.OK


def test_weekend_empty_is_ok(conn):
    weekend = datetime(2026, 8, 23, 10, 0, 0)  # 일요일 장중 시간대지만 휴일
    assert not H.is_market_hours(weekend)
    checks = H.run_checks(conn, minutes=15, expected=80, now=weekend)
    assert _by_name(checks)["1. 수집 여부"].status is H.Status.OK


# --------------------------------------------------------------- 강도 분포


def test_constant_strength_is_fail(conn):
    insert_ticks(conn, [_tick(i, seq=i, side=(1 if i % 2 else -1), strength=100.0)
                        for i in range(10)])
    checks = H.run_checks(conn, minutes=15, expected=1, now=NOW)
    assert _by_name(checks)["3. 체결강도 분포"].status is H.Status.FAIL


def test_all_null_strength_is_fail(conn):
    insert_ticks(conn, [_tick(i, seq=i, side=(1 if i % 2 else -1), strength=None)
                        for i in range(10)])
    assert _by_name(H.run_checks(conn, minutes=15, expected=1, now=NOW))[
        "3. 체결강도 분포"
    ].status is H.Status.FAIL


# --------------------------------------------------------------- 가격 정합


def test_price_out_of_daily_range_same_day_is_fail(conn):
    """당일 daily_bars 고저(274000~276000) 밖 → 스케일 의심, 실패."""
    # 100배 스케일 오류: 27,450,000
    insert_ticks(conn, [_tick(i, seq=i, side=(1 if i % 2 else -1), price=27_450_000)
                        for i in range(10)])
    price = _by_name(H.run_checks(conn, minutes=15, expected=1, now=NOW))["4. 가격 정합"]
    assert price.status is H.Status.FAIL
    assert "스케일" in price.detail


def test_best_bid_gt_ask_is_fail(conn):
    insert_ticks(conn, [_tick(i, seq=i, side=(1 if i % 2 else -1),
                              best_bid=274600, best_ask=274500) for i in range(5)])
    price = _by_name(H.run_checks(conn, minutes=15, expected=1, now=NOW))["4. 가격 정합"]
    assert price.status is H.Status.FAIL


def test_price_fallback_when_no_today_bars(conn):
    """당일 daily_bars 가 없으면 최근일로 폴백해 스케일만 넓게 본다(경고 아님)."""
    conn.execute("delete from daily_bars where date = ?", (TRADE_DATE,))
    conn.execute(
        "insert into daily_bars(code,date,open,high,low,close) values (?,?,?,?,?,?)",
        ("005930", "2026-08-21", 274500, 276000, 274000, 275000),
    )
    conn.commit()
    _healthy(conn)
    price = _by_name(H.run_checks(conn, minutes=15, expected=1, now=NOW))["4. 가격 정합"]
    assert price.status is H.Status.OK
    assert "최근" in price.detail


# --------------------------------------------------------------- 호가


def test_bid_ge_ask_quote_is_fail(conn):
    _healthy(conn, n=2)
    insert_quotes(conn, [_quote(0, code="000660", bid1=100, ask1=99)])
    q = _by_name(H.run_checks(conn, minutes=15, expected=1, now=NOW))["5. 호가"]
    assert q.status is H.Status.FAIL
    assert q.raw is not None


def test_vi_flag_seen_reported(conn):
    _healthy(conn, n=2)
    insert_quotes(conn, [_quote(0, code="000660", vi_flag=1)])
    q = _by_name(H.run_checks(conn, minutes=15, expected=1, now=NOW))["5. 호가"]
    assert "vi_flag 1건" in q.detail


# --------------------------------------------------------------- 항목 6 VI 이벤트


def _vi(secs, *, code="005930", vi_type="발동", **kw):
    ts = (NOW - timedelta(seconds=secs)).strftime("%Y-%m-%d %H:%M:%S")
    row = {"code": code, "ts": ts, "vi_type": vi_type, "vi_kind": "정적",
           "base_price": 272000, "trigger_price": 300000, "gap_rate": 10.29,
           "release_time": "093215", "raw": {"9068": "1"}}
    row.update(kw)
    return row


def test_vi_events_reported(conn):
    _healthy(conn, n=2)
    insert_vi_events(conn, [_vi(10), _vi(5, code="000660")])
    vi = _by_name(H.run_checks(conn, minutes=15, expected=1, now=NOW))["6. VI 이벤트"]
    assert vi.status is H.Status.OK
    assert "2건" in vi.detail


def test_vi_absent_during_market_is_ok(conn):
    # VI 0건은 장중이라도 정상(드문 이벤트). 매번 WARN 을 내면 경보가 무의미하다.
    _healthy(conn, n=2)  # 틱·호가는 정상, VI 만 없음
    checks = H.run_checks(conn, minutes=15, expected=1, now=NOW)
    vi = _by_name(checks)["6. VI 이벤트"]
    assert vi.status is H.Status.OK
    assert H.worst_status(checks) is not H.Status.FAIL


def test_vi_absent_after_hours_is_ok(conn):
    _healthy(conn, n=2)
    after = datetime(2026, 8, 24, 18, 0, 0)
    vi = _by_name(H.run_checks(conn, minutes=15, expected=1, now=after))["6. VI 이벤트"]
    assert vi.status is H.Status.OK


# --------------------------------------------------------------- main / 종료코드


def test_main_exit_zero_after_hours(conn, tmp_path, monkeypatch, capsys):
    """장 마감 후 실행: 빈 상태라도 정상 리포트 + 종료코드 0."""
    # 새 커넥션이 같은 파일을 열도록 conn 파일 경로를 dsn 으로 넘긴다.
    monkeypatch.setattr(
        H, "datetime", _fixed_datetime(datetime(2026, 8, 24, 18, 0, 0))
    )
    rc = H.main(["--dsn", str(_conn_path(conn)), "--minutes", "20",
                 "--log-path", str(tmp_path / "h.log")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "장중 아님" in out
    assert (tmp_path / "h.log").exists()


def test_main_exit_nonzero_on_fail(conn, tmp_path, monkeypatch, capsys):
    insert_ticks(conn, [_tick(i, seq=i, side=1) for i in range(10)])  # 전부 매수
    insert_quotes(conn, [_quote(i) for i in range(10)])
    monkeypatch.setattr(H, "datetime", _fixed_datetime(NOW))
    rc = H.main(["--dsn", str(_conn_path(conn)), "--no-log"])
    assert rc == 1
    assert "실패" in capsys.readouterr().out


# --------------------------------------------------------------- 헬퍼


def _conn_path(conn):
    # sqlite 파일 경로 회수
    return next(r[2] for r in conn.execute("PRAGMA database_list"))


def _fixed_datetime(fixed):
    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):  # datetime.now 시그니처 유지
            return fixed

    return _DT


def test_empty_quote_rows_are_not_counted_as_crossed_book(conn):
    """bid1·ask1 이 **둘 다 0** 인 공백 호가를 역전으로 세면 안 된다.

    2026-08-28 실측: 호가 21,131행 중 bid1>=ask1 이 521행인데 **둘 다 0 초과인 건
    0행**이었다. 키움은 호가가 아직 안 선 구간을 `"44":"-0"` 처럼 0 으로 채워 보낸다.
    그걸 역전으로 세니 헬스체크가 매일 [실패]를 찍었고, 진짜 문제가 생겨도 그 실패에
    묻힌다. 잔량 0 문제는 `total_qty_zero` 가 따로 센다.
    """
    ts = (NOW - timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
    insert_quotes(conn, [{
        "code": "000001", "ts": ts, "bid1": 0, "ask1": 0,
        "total_bid_qty": 0, "total_ask_qty": 0,
    }])
    qs = H.fetch_quote_stats(conn, NOW - timedelta(minutes=20))
    assert qs.bid_ge_ask == 0, "0 채움 호가는 역전이 아니다"
    assert qs.total_qty_zero >= 1, "잔량 0 은 이쪽이 센다"


def test_real_crossed_book_still_counts(conn):
    """진짜 역전(둘 다 0 초과인데 bid1 >= ask1)은 여전히 잡힌다."""
    ts = (NOW - timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
    insert_quotes(conn, [{
        "code": "000001", "ts": ts, "bid1": 1000, "ask1": 990,
        "total_bid_qty": 5, "total_ask_qty": 5,
    }])
    assert H.fetch_quote_stats(conn, NOW - timedelta(minutes=20)).bid_ge_ask == 1
