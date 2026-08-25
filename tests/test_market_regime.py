"""시황 게이트 테스트 — 순수 분류는 DB 없이, DB 함수는 합성 sqlite 로."""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from scalp_it.market_regime import (
    COLD,
    DEAD,
    HOT,
    MILD,
    RegimeThresholds,
    classify_regime,
    evaluate_gate,
    fetch_leader_turnover,
    fetch_market_turnover,
    fetch_recent_turnover,
    previous_session,
    size_multiplier,
)

# ------------------------------------------------------------------ 순수 로직

@pytest.mark.parametrize(
    ("turnover", "expected"),
    [
        (13.0, HOT),      # 만쥬 호황 상단
        (10.0, HOT),      # 경계는 활황 쪽 (10조'이상')
        (9.99, MILD),
        (8.0, MILD),      # 8~10조 = 원자료 공백 구간
        (7.99, COLD),
        (5.0, COLD),      # 5~8조'이하' 침체
        (4.69, DEAD),     # 2026-08-25 코스닥 실제값
        (0.0, DEAD),
    ],
)
def test_classify_regime_boundaries(turnover: float, expected: str) -> None:
    assert classify_regime(turnover) == expected


def test_thresholds_are_tunable() -> None:
    loose = RegimeThresholds(hot=6.0, mild=4.0, cold=2.0)
    assert classify_regime(4.69, loose) == MILD


def test_thresholds_reject_disordered_values() -> None:
    with pytest.raises(ValueError):
        RegimeThresholds(hot=5.0, mild=8.0, cold=1.0)


def test_size_multiplier_is_monotone_and_caps_at_one() -> None:
    assert size_multiplier(HOT) == 1.0
    assert size_multiplier(HOT) > size_multiplier(MILD) > size_multiplier(COLD)
    assert size_multiplier(DEAD) == 0.0
    assert size_multiplier("모르는라벨") == 0.0  # 모르면 가장 보수적으로


# ----------------------------------------------------------------- 합성 DB

def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("create table stocks (code text, name text, market text, sector text, kind text)")
    conn.execute(
        "create table daily_bars (code text, date text, open int, high int, low int, "
        "close int, volume int, trade_value int, source text)"
    )
    stocks = [
        ("111111", "대장", "코스닥", "전기/전자", "A"),
        ("222222", "후속", "코스닥", "전기/전자", "A"),
        ("333333", "코스피주", "거래소", "화학", "A"),
    ]
    conn.executemany("insert into stocks values (?,?,?,?,?)", stocks)
    # trade_value 단위는 백만원. 1e6 백만원 = 1조.
    bars = [
        # 2026-08-24 코스닥 합계 6.0조 (침체)
        ("111111", "2026-08-24", 100, 110, 90, 100, 10, 4_000_000, "t"),
        ("222222", "2026-08-24", 100, 110, 90, 100, 10, 2_000_000, "t"),
        ("333333", "2026-08-24", 100, 110, 90, 100, 10, 9_000_000, "t"),
        # 2026-08-25 코스닥 합계 11.0조 (활황), 대장 1위 8조
        ("111111", "2026-08-25", 100, 110, 90, 100, 10, 8_000_000, "t"),
        ("222222", "2026-08-25", 100, 110, 90, 100, 10, 3_000_000, "t"),
        ("333333", "2026-08-25", 100, 110, 90, 100, 10, 9_000_000, "t"),
    ]
    conn.executemany("insert into daily_bars values (?,?,?,?,?,?,?,?,?)", bars)
    conn.commit()
    return conn


def test_fetch_market_turnover_is_jo_won_and_market_scoped() -> None:
    conn = _make_db()
    assert fetch_market_turnover(conn, "2026-08-25") == pytest.approx(11.0)
    assert fetch_market_turnover(conn, "2026-08-25", market="거래소") == pytest.approx(9.0)
    assert fetch_market_turnover(conn, "2026-08-26") is None


def test_fetch_recent_turnover_is_oldest_first() -> None:
    conn = _make_db()
    rows = fetch_recent_turnover(conn, "2026-08-25", days=5)
    assert [r[0] for r in rows] == ["2026-08-24", "2026-08-25"]
    assert rows[-1][1] == pytest.approx(11.0)


def test_fetch_leader_turnover_picks_top_of_that_market() -> None:
    conn = _make_db()
    leader = fetch_leader_turnover(conn, "2026-08-25")
    assert leader is not None
    assert leader[0] == "111111"
    assert leader[2] == pytest.approx(8.0)


def test_previous_session_is_strictly_before() -> None:
    conn = _make_db()
    assert previous_session(conn, "2026-08-26") == "2026-08-25"
    assert previous_session(conn, "2026-08-25") == "2026-08-24"
    assert previous_session(conn, "2026-08-24") is None


def test_evaluate_gate_hot_day_allows_trading() -> None:
    conn = _make_db()
    reading = evaluate_gate(conn, "2026-08-25", target=date(2026, 8, 26))
    assert reading.regime == HOT
    assert reading.tradable is True
    assert reading.size_multiplier == 1.0
    assert reading.target == date(2026, 8, 26)
    assert reading.leader_code == "111111"
    assert reading.leader_healthy is True  # 주도주 1조↑


def test_evaluate_gate_cold_day_still_tradable_but_shrunk() -> None:
    conn = _make_db()
    reading = evaluate_gate(conn, "2026-08-24")
    assert reading.regime == COLD
    assert reading.tradable is True
    assert 0.0 < reading.size_multiplier < 1.0


def test_evaluate_gate_dead_day_blocks_trading() -> None:
    conn = _make_db()
    # 임계를 올려 같은 데이터를 극침체로 만든다(합성 데이터 재작성 없이 경계만 이동).
    reading = evaluate_gate(
        conn, "2026-08-24", thresholds=RegimeThresholds(hot=20.0, mild=15.0, cold=10.0)
    )
    assert reading.regime == DEAD
    assert reading.tradable is False
    assert reading.size_multiplier == 0.0


def test_evaluate_gate_raises_when_no_data() -> None:
    conn = _make_db()
    with pytest.raises(ValueError):
        evaluate_gate(conn, "2026-08-26")
