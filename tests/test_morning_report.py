"""아침 리포트 — 월요일에 한 장만 열면 되게.

이 리포트의 존재 이유는 지표 하나다: **방아쇠 구간(대장이 상한가 3호가를 남긴
구간, 아직 안 잠김)에 틱이 들어온 종목이 몇 개인가.** 2026-08-24~28 에 2개였고,
10~20 이 돼야 만쥬 규칙이 처음으로 시험대에 오른다(STATUS "다음에 볼 것").
"""
from scalp_it.morning_report import in_trigger_zone


def test_three_ticks_below_limit_is_the_trigger_zone():
    """만쥬 §2 — "상한가 호가를 3개 남겨둔 3~5초 사이에 매수"."""
    # 전일종가 10,000 → 상한가 13,000. 5천~2만원 구간 호가단위는 10원(2023 개편)이라
    # 3호가 아래 = 12,970. 만쥬가 말한 구간은 여기부터 상한가 직전까지다.
    assert in_trigger_zone(price=12_970, prev_close=10_000, ticks=3)
    assert in_trigger_zone(price=12_990, prev_close=10_000, ticks=3)
    assert not in_trigger_zone(price=12_960, prev_close=10_000, ticks=3)


def test_already_locked_is_not_the_trigger_zone():
    """상한가에 닿은 건 방아쇠가 지나간 것이다 — 만쥬는 그걸 안 산다.

    `--limitup-require-surge` 와 미잠김 우선 선정이 같은 얘기다.
    """
    assert not in_trigger_zone(price=13_000, prev_close=10_000, ticks=3)


def test_no_prev_close_is_not_a_zone():
    """전일종가가 없으면 상한가를 계산할 수 없다 — 신규 상장 첫날 등."""
    assert not in_trigger_zone(price=1_000, prev_close=0, ticks=3)


def test_yesterday_on_monday_is_friday_not_sunday():
    """08:30 크론의 ``--date yesterday`` 는 직전 거래일이다.

    달력 어제로 두면 월요일 latest.md 가 일요일 "방아쇠 0개" 가 된다(2026-11-16).
    """
    from datetime import date

    from scalp_it.morning_report import resolve_day

    calls = []

    def fake_previous_session(conn, before):
        calls.append(before)
        return date(2026, 11, 13)  # 금요일

    monday = date(2026, 11, 16)
    assert resolve_day("yesterday", monday, object(),
                       previous_session=fake_previous_session) == date(2026, 11, 13)
    assert calls == [monday]  # '오늘 미만' 의 최근 거래일을 묻는다


def test_today_and_explicit_dates_do_not_touch_the_calendar():
    from datetime import date

    from scalp_it.morning_report import resolve_day

    def boom(conn, before):
        raise AssertionError("달력 조회 불필요")

    d = date(2026, 11, 19)
    assert resolve_day("today", d, None, previous_session=boom) == d
    assert resolve_day("2026-11-13", d, None, previous_session=boom) == date(2026, 11, 13)
