"""커버리지 계기판 — 그날 급등 이벤트 중 우리가 보고 있던 비율.

이 파일이 존재하는 이유는 숫자 하나다. `leader_scan.py` 주석의 **커버리지 2%**
(최근 7거래일 상한가 52건 중 우리 유니버스 안에 있던 것 1건). 2026-08-26~28 에
시장엔 상한가가 4·8·7건 터졌는데 우리 신호는 사흘 내리 0건이었고, 매매일지엔
"희소 이벤트라 정상"이라고 적혔다. 정상이 아니라 **못 본 것**이었다.

구독 못 함 → 신호 0 → 주문 0 → 체결 0. 이 사슬의 첫 칸을 매일 재는 게 이 모듈이다.
재지 않으면 슬롯 선점(2026-08-31)이 효과가 있었는지도 감으로 판단하게 된다.
"""
import pytest

from scalp_it.realtime.coverage import (
    Coverage,
    measure_coverage,
    surge_events,
)


# ── 이벤트 추출: 그날 얼마나 튄 종목이 있었나 ────────────────────────────────
def test_picks_codes_above_the_threshold_biggest_first():
    """가장 크게 튄 것부터 — 못 본 종목을 볼 때 위에서부터 아프다."""
    got = surge_events([("A", 130, 100), ("B", 100, 100), ("C", 150, 100)],
                       min_rate=0.29)
    assert [e.code for e in got] == ["C", "A"]
    assert got[0].rate == pytest.approx(0.5)


def test_threshold_is_inclusive():
    """상한가 +29.5% 는 상한가다 — 경계에서 떨어뜨리면 그날 이벤트를 놓친다."""
    assert [e.code for e in surge_events([("A", 1295, 1000)], min_rate=0.295)] == ["A"]


def test_integer_prices_do_not_become_integer_division():
    """정수 종가를 정수로 나누면 등락률이 0 이 된다.

    market_regime.count_limitups 가 실제로 이 함정에 빠져 1년 2,313건을 5건으로
    셌던 적이 있다. 같은 실수를 여기서 반복하지 않는다.
    """
    got = surge_events([("A", 1300, 1000)], min_rate=0.29)
    assert [e.code for e in got] == ["A"]


def test_skips_rows_without_a_usable_previous_close():
    """신규 상장·데이터 구멍 — 전일종가가 없으면 등락률을 만들 수 없다."""
    assert surge_events([("A", 1300, 0), ("B", 1300, None)], min_rate=0.29) == []


# ── 커버리지: 그중 우리가 보고 있던 게 몇 개인가 ─────────────────────────────
def test_counts_what_we_were_watching():
    events = surge_events([("A", 150, 100), ("B", 150, 100)], min_rate=0.29)
    cov = measure_coverage(events, {"A": "09:10:00"})
    assert (cov.n_events, cov.n_covered) == (2, 1)
    assert cov.rate == pytest.approx(0.5)
    assert cov.missed == ["B"]


def test_ignores_subscriptions_that_were_not_events():
    """삼성전자를 종일 붙여둬도 그날 상한가를 본 게 아니다."""
    events = surge_events([("A", 150, 100)], min_rate=0.29)
    cov = measure_coverage(events, {"005930": "09:00:00", "A": "09:10:00"})
    assert cov.rate == pytest.approx(1.0)


def test_subscribing_after_the_cutoff_does_not_count():
    """장 끝날 때 붙은 건 본 게 아니다.

    만쥬는 오전에 대장을 낙점한다. 14시에 구독이 붙었으면 그날 그 종목으로는
    아무것도 못 했으므로 커버리지로 세면 숫자가 거짓말을 한다.
    """
    events = surge_events([("A", 150, 100)], min_rate=0.29)
    cov = measure_coverage(events, {"A": "14:20:00"}, before="10:00:00")
    assert (cov.n_covered, cov.missed) == (0, ["A"])


def test_no_events_is_not_a_division_by_zero():
    """이벤트가 없던 날은 커버리지가 0% 가 아니라 **판정 불가**다."""
    cov = measure_coverage([], {})
    assert cov.n_events == 0 and cov.rate is None


# ── DB 에서 하루치를 뽑아온다 ────────────────────────────────────────────────
import sqlite3

from scalp_it.realtime.coverage import (
    fetch_first_ticks,
    fetch_surge_rows,
    format_report,
)


def _db(tmp_path):
    conn = sqlite3.connect(tmp_path / "cov.db")
    conn.executescript(
        """
        CREATE TABLE stocks (code TEXT PRIMARY KEY, market TEXT, name TEXT);
        CREATE TABLE daily_bars (code TEXT, date TEXT, close INTEGER);
        CREATE TABLE ticks (code TEXT, ts TIMESTAMP, seq INTEGER);
        INSERT INTO stocks VALUES ('A','코스닥','에이'), ('B','코스닥','비'),
                                  ('C','거래소','씨');
        INSERT INTO daily_bars VALUES
            ('A','2026-08-28',100), ('A','2026-08-31',150),
            ('B','2026-08-28',100), ('B','2026-08-31',101),
            ('C','2026-08-28',100), ('C','2026-08-31',150);
        INSERT INTO ticks VALUES
            ('A','2026-08-31 09:12:03',1), ('A','2026-08-31 09:40:00',2),
            ('Z','2026-08-31 09:00:00',1);
        """
    )
    conn.commit()
    return conn


def test_fetch_surge_rows_uses_the_previous_session_close(tmp_path):
    """전일종가는 **직전 거래일**이다 — 달력 하루 전이 아니다(주말·휴장)."""
    rows = fetch_surge_rows(_db(tmp_path), day="2026-08-31", market="코스닥")
    assert sorted((r[0], r[1], r[2]) for r in rows) == [
        ("A", 150, 100), ("B", 101, 100)]        # C 는 거래소라 빠진다


def test_fetch_first_ticks_returns_when_each_code_first_appeared(tmp_path):
    """종목별 **첫 틱 시각** — 늦게 붙은 구독을 걸러내는 근거다."""
    got = fetch_first_ticks(_db(tmp_path), day="2026-08-31")
    assert got == {"A": "09:12:03", "Z": "09:00:00"}


def test_report_names_the_missed_codes(tmp_path):
    """놓친 종목을 이름으로 적는다 — 숫자만 보면 다음 행동이 안 나온다."""
    conn = _db(tmp_path)
    rows = fetch_surge_rows(conn, day="2026-08-31", market="코스닥")
    events = surge_events(rows, min_rate=0.29)
    cov = measure_coverage(events, fetch_first_ticks(conn, day="2026-08-31"))
    text = format_report(cov, day="2026-08-31", min_rate=0.29)
    assert "100.0%" in text and "1/1" in text


def test_report_says_undecidable_when_there_were_no_events(tmp_path):
    """이벤트 0건인 날을 0% 로 적으면 조용한 날과 못 본 날이 섞인다."""
    text = format_report(Coverage(), day="2026-08-31", min_rate=0.295)
    assert "판정 불가" in text and "0.0%" not in text


# ── CLI ─────────────────────────────────────────────────────────────────────
import json

from scalp_it.cli_coverage import build_parser, run_day
from scalp_it.realtime.coverage import fetch_names


def test_fetch_names_labels_the_missed_codes(tmp_path):
    assert fetch_names(_db(tmp_path), ["A", "C", "없음"]) == {"A": "에이", "C": "씨"}


def test_defaults_to_the_limit_up_threshold():
    """기본 문턱은 상한가다 — 가장 아픈 이벤트부터 센다."""
    args = build_parser().parse_args([])
    assert args.min_rate == [0.295] and args.market == "코스닥"


def test_run_day_reports_every_threshold_it_was_given(tmp_path):
    """문턱마다 한 줄 — 상한가는 놓쳐도 12% 급등은 봤을 수 있고, 그 차이가 정보다."""
    rows = run_day(_db(tmp_path), day="2026-08-31", market="코스닥",
                   min_rates=[0.295, 0.005], before=None)
    assert [r["min_rate"] for r in rows] == [0.295, 0.005]
    by = {r["min_rate"]: r for r in rows}
    assert by[0.295]["n_events"] == 1 and by[0.295]["rate"] == 1.0
    # 문턱을 낮추면 B(+1%) 가 이벤트로 들어오고, 우리는 B 를 못 봤다.
    assert by[0.005]["n_events"] == 2 and by[0.005]["missed"] == ["B"]


def test_run_day_rows_are_json_serialisable(tmp_path):
    """jsonl 로 매일 append 할 거라 직렬화가 깨지면 기록이 끊긴다."""
    rows = run_day(_db(tmp_path), day="2026-08-31", market="코스닥",
                   min_rates=[0.295], before=None)
    assert json.loads(json.dumps(rows[0]))["day"] == "2026-08-31"


def test_missing_daily_bars_is_not_a_quiet_day(tmp_path):
    """일봉이 아직 안 들어온 날을 '이벤트 0건'으로 적으면 계기판이 거짓말한다.

    장중이나 적재 전에 돌리면 daily_bars 에 그날 행이 없어 이벤트가 0건으로 나온다.
    조용한 날(진짜 급등이 없던 날)과 **모르는 날**은 완전히 다른 얘기라, 섞이면
    0% 와 판정 불가와 미적재가 한 칸에 뭉개진다.
    """
    rows = run_day(_db(tmp_path), day="2026-09-01", market="코스닥",
                   min_rates=[0.295], before=None)
    assert rows[0]["n_bars"] == 0
    text = format_report(Coverage(), day="2026-09-01", min_rate=0.295, n_bars=0)
    assert "일봉 미적재" in text


def test_a_genuinely_quiet_day_is_reported_as_no_events(tmp_path):
    """일봉은 있는데 문턱을 넘은 종목이 없으면 그건 진짜 조용한 날이다."""
    rows = run_day(_db(tmp_path), day="2026-08-31", market="코스닥",
                   min_rates=[9.0], before=None)
    assert rows[0]["n_bars"] == 2 and rows[0]["n_events"] == 0
    text = format_report(Coverage(), day="2026-08-31", min_rate=9.0, n_bars=2)
    assert "이벤트 0건" in text and "일봉 미적재" not in text
