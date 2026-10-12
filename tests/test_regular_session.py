"""정규장 기준가 정문 — 애프터마켓 종가로 상한가를 계산하면 실패해야 한다.

2026-09-14 KRX 애프터마켓(16:00~20:00) 뒤로 ``daily_bars`` 종가는 정규장 종가가 아니다. 다음 날
가격제한폭의 기준가는 전일 **정규장** 종가다. 여기 테스트는 일봉 종가를 일부러 애프터마켓 값으로
오염시킨 합성 DB 로, 상한가 계산에 들어가는 전일종가가 전부 정규장 값인지 본다. 마지막 묶음은
일봉 종가를 직접 SELECT 하는 코드가 새로 생기면 잡는 정적 규칙이다(규칙이 죽어도 초록이지
않도록 위반을 주입하는 테스트를 같이 둔다).
"""

from __future__ import annotations

import ast
import re
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pytest

from scalp_it.realtime.pair_detector import limit_up_price
from scalp_it.regular_session import (
    SOURCE_DAILY,
    SOURCE_MINUTE,
    SOURCE_SUPPLY,
    base_price_from_supply,
    base_price_sources,
    base_prices,
    limitup_codes,
    regular_close_sources,
    regular_closes,
    regular_session_bars,
)

D0, D1, D2 = "2026-10-06", "2026-10-07", "2026-10-08"   # 전부 애프터마켓 이후
PRE0, PRE1 = "2026-09-10", "2026-09-11"                   # 애프터마켓 이전


def _schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE stocks (code TEXT, name TEXT, market TEXT, sector TEXT, kind TEXT);
        CREATE TABLE daily_bars (code TEXT, date TEXT, open INT, high INT, low INT, close INT,
                                 volume INT, trade_value INT, source TEXT);
        CREATE TABLE minute_bars (code TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL,
                                  volume INT, trade_value REAL);
        CREATE TABLE supply_demand (code TEXT, date TEXT, close INT, flu_rt REAL);
        """
    )


def _day(conn, code, day, *, open_, high, low, close, after_close, after_high,
         tv_won=11e9, minute=True, close_bar="15:30:00"):
    """하루치: 정규장 분봉 2개(09:00, 종가 봉) + 애프터마켓 분봉 1개, 오염된 일봉 1행.

    일봉 종가·고가·거래대금은 키움처럼 애프터마켓 체결까지 섞인 값이다.
    """
    if minute:
        conn.executemany("insert into minute_bars values (?,?,?,?,?,?,?,?)", [
            (code, f"{day} 09:00:00", open_, high, low, open_, 600, tv_won * 0.6),
            (code, f"{day} {close_bar}", close, close, close, close, 400, tv_won * 0.4),
            (code, f"{day} 17:00:00", after_close, after_high, after_close, after_close, 100, 1e9),
        ])
    conn.execute("insert into daily_bars values (?,?,?,?,?,?,?,?,?)",
                 (code, day, open_, max(high, after_high), low, after_close, 1100,
                  int((tv_won + 1e9) / 1e6), "t"))


def _supply(conn, code, day, *, contaminated_close, base):
    """``supply_demand`` 한 행. close 는 오염값, flu_rt 는 그 close 와 기준가로 계산(키움과 같다)."""
    flu = round((contaminated_close / base - 1) * 100, 2) * 100
    conn.execute("insert into supply_demand values (?,?,?,?)", (code, day, contaminated_close, flu))


@pytest.fixture()
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    _schema(conn)
    conn.executemany("insert into stocks values (?,?,?,?,?)", [
        ("111111", "대장", "코스닥", "전기/전자", "A"),
        ("222222", "후속", "코스닥", "전기/전자", "A"),
        ("333333", "분봉밖", "코스닥", "화학", "A"),
    ])
    # 111111: 정규장 종가 10,000 → 11,000(+10%) → 11,500. 애프터마켓 종가는 매번 더 높다.
    _day(conn, "111111", D0, open_=9_900, high=10_100, low=9_500, close=10_000,
         after_close=10_100, after_high=10_150)
    _day(conn, "111111", D1, open_=10_050, high=11_100, low=10_000, close=11_000,
         after_close=11_200, after_high=11_300)
    _day(conn, "111111", D2, open_=11_000, high=11_600, low=10_900, close=11_500,
         after_close=11_800, after_high=12_400)
    # 222222: 같은 섹터 후속. 마지막 날은 동시호가 체결이 없어 15:19 봉이 종가다.
    _day(conn, "222222", D0, open_=5_000, high=5_100, low=4_800, close=5_000,
         after_close=5_050, after_high=5_100)
    _day(conn, "222222", D1, open_=5_010, high=5_600, low=5_000, close=5_500,
         after_close=5_600, after_high=5_650)
    _day(conn, "222222", D2, open_=5_500, high=5_800, low=5_450, close=5_700,
         after_close=5_790, after_high=5_900, close_bar="15:19:00")
    # 333333: 분봉 수집 밖(애프터마켓 이후엔 일봉만 있다).
    _day(conn, "333333", D1, open_=2_000, high=2_100, low=1_950, close=2_050,
         after_close=2_090, after_high=2_100, minute=False)
    _day(conn, "333333", D2, open_=2_080, high=2_200, low=2_050, close=2_150,
         after_close=2_190, after_high=2_200, minute=False)
    # 수급표: D2 행만(그날 16:00 적재). D2 의 기준가 = D1 정규장 종가.
    _supply(conn, "111111", D2, contaminated_close=11_800, base=11_000)
    _supply(conn, "333333", D2, contaminated_close=2_190, base=2_050)
    conn.commit()
    return conn


# ---------------------------------------------------------------- 정문 함수

def test_base_price_from_supply_inverts_flu_rt_to_the_tick():
    # 12,000 / 1.0909 → 11,000.09… → 10원 틱으로 11,000
    assert base_price_from_supply(12_000, 909.0) == 11_000
    assert base_price_from_supply(13_000, 3000.0) == 10_000
    assert base_price_from_supply(0, 100.0) is None
    assert base_price_from_supply(1_000, None) is None


def test_base_price_from_supply_keeps_won_precision_off_the_stock_tick_grid():
    """ETF(2천원 이상 5원 틱)처럼 주식 호가표에 안 맞는 값은 원 단위로만 맞춘다."""
    assert base_price_from_supply(10_005, 0.0) == 10_005


def test_premarket_base_is_the_previous_regular_close_not_the_aftermarket_close(db):
    """장 전(그날 수급표 없음) 기준가 = 전일 분봉 정규장 종가. 일봉 종가(애프터마켓)는 쓰지 않는다."""
    got = base_price_sources(db, "2026-10-12", ["111111", "222222", "333333"])
    assert got["111111"] == (11_500, SOURCE_MINUTE)
    assert got["222222"] == (5_700, SOURCE_MINUTE)     # 동시호가 체결 없음 → 15:19 봉
    assert "333333" not in got, "분봉도 수급표도 없으면 모른다고 해야 한다(오염 일봉으로 메우지 않음)"
    # 이 값으로 구한 상한가는 실제 상한가이고, 일봉 종가로 구하면 틀린다.
    assert limit_up_price(got["111111"][0]) == 14_950
    assert limit_up_price(11_800) != 14_950


def test_session_base_prefers_supply_demand_of_that_day(db):
    """그날 수급표가 있으면 그 역산값(기준가 재산정까지 반영)이 1순위다."""
    got = base_price_sources(db, D2, ["111111", "222222", "333333"])
    assert got["111111"] == (11_000, SOURCE_SUPPLY)
    assert got["333333"] == (2_050, SOURCE_SUPPLY)     # 분봉 밖 종목도 수급표로는 안다
    assert got["222222"] == (5_500, SOURCE_MINUTE)     # 수급표 없으면 전일 분봉


def test_daily_close_fallback_only_before_the_aftermarket(db):
    db.execute("insert into daily_bars values ('333333', ?, 1, 1, 1, 1_000, 1, 1, 't')", (PRE0,))
    db.execute("insert into daily_bars values ('333333', ?, 1, 1, 1, 1_100, 1, 1, 't')", (PRE1,))
    assert base_price_sources(db, PRE1, ["333333"]) == {"333333": (1_000, SOURCE_DAILY)}
    assert regular_close_sources(db, PRE1, ["333333"]) == {"333333": (1_100, SOURCE_DAILY)}
    # 09-14 이후 날짜는 일봉 종가로 돌아가지 않는다.
    assert "333333" not in regular_closes(db, D2, ["333333"])


def test_regular_close_falls_back_to_next_session_supply(db):
    """분봉 밖 종목의 D1 정규장 종가 = D2 수급표 기준가."""
    assert regular_close_sources(db, D1, ["333333"]) == {"333333": (2_050, SOURCE_SUPPLY)}
    assert regular_closes(db, D2, ["111111"]) == {"111111": 11_500}


def test_regular_session_bars_drop_aftermarket_minutes(db):
    bar = regular_session_bars(db, D2, ["111111"])["111111"]
    assert bar.high == 11_600            # 애프터마켓 고가 12,400 이 빠졌다
    assert bar.trade_value == pytest.approx(11_000.0)   # 백만원, 애프터마켓 10억 제외
    assert bar.volume == 1_000


def test_missing_tables_degrade_to_unknown_not_crash():
    conn = sqlite3.connect(":memory:")
    assert base_prices(conn, D2, ["111111"]) == {}
    assert regular_closes(conn, D2) == {}


def test_limitup_codes_use_regular_close_over_base(db):
    # 111111 D2: 11,500 / 11,000 = +4.5% → 상한가 아님. 정규장 종가를 상한가로 바꾸면 잡혀야 한다.
    assert limitup_codes(db, D2, ["111111"]) == set()
    db.execute("update minute_bars set close = 14300 where code='111111' and ts = ?", (f"{D2} 15:30:00",))
    assert limitup_codes(db, D2, ["111111"]) == {"111111"}


# ---------------------------------------------------------------- 호출부 배선

def test_universe_prev_close_is_the_regular_base(db):
    from scalp_it.universe import UniverseConfig, build_universe

    cfg = UniverseConfig(lookback_days=3, limit=10, min_trade_value_eok=100.0,
                         min_range_pct=1.0, min_surge_days=1, min_price=1_000)
    uni = build_universe(db, D2, cfg)
    assert uni.prev_close_by_code["111111"] == 11_500, "애프터마켓 종가 11,800 이면 상한가가 틀린다"
    assert uni.prev_close_by_code["222222"] == 5_700
    cand = {c.code: c for c in uni.pool}["111111"]
    # 전일 레인지·거래대금도 정규장 값: (11,600−10,900)/11,500, 110억
    assert cand.last_range_pct == pytest.approx(700 / 11_500 * 100)
    assert cand.trade_value_eok == pytest.approx(110.0)


def test_universe_prev_close_takes_a_recomputed_base_when_known(db):
    """과거 기준일로 돌리면 다음 세션 수급표가 있다 — 기준가 재산정(권리락 등)이 있으면 그 값을 쓴다.

    종가(정규화된 일봉)와 기준가가 갈리는 유일한 경우라, 전일종가가 정문 기준가에서 오는지 여기서 본다.
    """
    from scalp_it.universe import UniverseConfig, build_universe

    db.execute("update supply_demand set flu_rt = ? where code='111111' and date = ?",
               (round((11_800 / 10_000 - 1) * 100, 2) * 100, D2))   # D2 기준가 10,000 으로 재산정
    cfg = UniverseConfig(lookback_days=2, limit=10, min_trade_value_eok=100.0,
                         min_range_pct=1.0, min_surge_days=1, min_price=1_000)
    uni = build_universe(db, D1, cfg)
    assert uni.prev_close_by_code["111111"] == 10_000
    assert uni.prev_close_by_code["222222"] == 5_500


def test_breaking_meta_prev_close_is_the_regular_base(db, tmp_path, monkeypatch):
    import argparse

    import scalp_it.cli_pair_detect as cpd

    path = tmp_path / "kq.db"
    disk = sqlite3.connect(path)
    db.backup(disk)
    disk.close()

    class _Now(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 12, 10, 0)  # noqa: DTZ001 — KST naive(프로젝트 표준)

    monkeypatch.setattr(cpd, "datetime", _Now)
    meta = cpd._breaking_meta(argparse.Namespace(dsn=str(path)), ["111111", "333333"])
    assert meta["prev_close"] == {"111111": 11_500.0}
    assert meta["sectors"]["111111"] == "전기/전자"


def test_coverage_surge_rows_are_regular_close_over_base(db):
    from scalp_it.realtime.coverage import fetch_surge_rows

    rows = sorted(fetch_surge_rows(db, day=D2, market="코스닥"))
    # 111111: 11,500 / 11,000. 222222: 5,700 / 5,500(전일 분봉). 333333: D2 정규장 종가 모름 → 없음.
    assert rows == [("111111", 11_500, 11_000), ("222222", 5_700, 5_500)]


def test_count_limitups_and_sector_limitups_use_regular_prices(db):
    from scalp_it.market_regime import count_limitups
    from scalp_it.universe import fetch_sector_limitups

    # 일봉 종가로 세면 D1 의 111111 은 11,200/10,100 = +10.9% 라 어차피 아니다. 상한가 하나를 심는다:
    # 222222 D1 정규장 종가 6,500(= 5,000 × 1.3) — 일봉 종가는 애프터마켓에서 밀린 6,300.
    db.execute("update minute_bars set close = 6500 where code='222222' and ts = ?", (f"{D1} 15:30:00",))
    db.execute("update daily_bars set close = 6300 where code='222222' and date = ?", (D1,))
    assert count_limitups(db, D1) == 1
    assert fetch_sector_limitups(db, D2, days=2) == {"전기/전자": 1}


def test_loader_regular_only_drops_aftermarket(db):
    from scalp_it.intraday.loader import load_minute_bars

    full = load_minute_bars(db, D2, D2, ["111111"])
    reg = load_minute_bars(db, D2, D2, ["111111"], regular_only=True)
    assert len(full) == 3 and len(reg) == 2
    assert reg["ts"].max().strftime("%H:%M") < "15:33"


# ---------------------------------------------------------------- 정적 규칙
# 일봉 종가를 SELECT 하는 SQL 은 정문(regular_session) 밖에 새로 생기면 안 된다. 상한가·등락률의
# 분모로 흘러가면 2026-09-14 이후 애프터마켓 종가가 된다. 허용 목록은 이유가 있는 곳만.
# daily_bars 만이 아니다 — daily_bars_adjusted 와 supply_demand.close 도 같은 오염이다(10-12 정규장
# 종가 일치 36%, 10-12 리뷰). 셋 다 잡는다.

SRC = Path(__file__).resolve().parents[1] / "src" / "scalp_it"
_FROM_DAILY = re.compile(
    r"\b(?:from|join)\s+(?:daily_bars|daily_bars_adjusted|supply_demand)\b", re.IGNORECASE)
_CLOSE = re.compile(r"\bclose\b", re.IGNORECASE)

#: (파일, 함수) — 이유
ALLOWED = {
    ("regular_session.py", "_daily_closes"),   # 정문 자신. 09-14 이전 날짜만 쓴다
    ("regular_session.py", "_supply_bases"),   # 정문 자신. close/(1+flu_rt) 항등식 — 오염과 무관하다
    ("universe.py", "fetch_bars"),             # 읽은 뒤 _regularize 가 09-14 이후 종가를 바꾼다
    ("adapters/db.py", "load_daily_bars_raw"), # 기존 연구 재현용. 독스트링에 base_prices 를 쓰라고 적었다
    ("adapters/db.py", "load_prices"),         # 연구용 연속 시계열(수정주가). 독스트링에 오염 경고
    ("adapters/db.py", "load_ohlcv"),          # 연구용 연속 시계열(종가베팅). 독스트링에 오염 경고
}


def _sql_strings(tree: ast.AST):
    """(함수 이름, SQL 문자열) — 암묵적 이어붙이기·f-string 상수 조각까지 합친다."""
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                yield fn.name, node.value
            elif isinstance(node, ast.JoinedStr):
                yield fn.name, "".join(v.value if isinstance(v, ast.Constant) else "{}"
                                       for v in node.values)


def find_daily_close_sql(source: str, rel: str) -> list[str]:
    out = []
    for fn, sql in _sql_strings(ast.parse(source)):
        if _FROM_DAILY.search(sql) and _CLOSE.search(sql) and (rel, fn) not in ALLOWED:
            out.append(f"{rel}:{fn}: {sql[:80]!r}")
    return sorted(set(out))


def test_no_new_daily_close_sql_outside_the_gate():
    hits: list[str] = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        hits += find_daily_close_sql(path.read_text(encoding="utf-8"), rel)
    assert not hits, ("일봉 종가 직접 SELECT — 2026-09-14 이후 애프터마켓 종가다. "
                      "scalp_it.regular_session.base_prices/regular_closes 를 써라:\n" + "\n".join(hits))


def test_the_static_rule_catches_injected_violations():
    """규칙이 죽으면 위 테스트는 늘 초록이다 — 위반을 넣어 실제로 잡는지 본다."""
    plain = 'def f(cur):\n    cur.execute("select d.close from daily_bars d where d.date = %s")\n'
    split = ('def g(cur, ph):\n    cur.execute(\n        "select code, date, close "\n'
             '        f"from daily_bars where date = {ph}")\n')
    joined = ('def n(cur):\n    cur.execute("select t.code, d.close::numeric from ticks t '
              'join daily_bars d on d.code = t.code")\n')
    lag = ('def h(cur):\n    cur.execute("""with x as (select lag(b.close) over (partition by b.code)'
           ' pc from daily_bars b) select * from x""")\n')
    # daily_bars_adjusted·supply_demand.close 도 09-14 이후 애프터마켓 종가다(옛 _prev_day_movers 모양).
    adjusted = ('def k(cur, ph):\n    cur.execute("select t.code from daily_bars_adjusted t "\n'
                '        "join daily_bars_adjusted y on y.code = t.code "\n'
                '        f"where t.date = {ph} and (t.close / y.close - 1.0) >= {ph}")\n')
    supply = 'def s(cur):\n    cur.execute("select code, close from supply_demand where date = %s")\n'
    for src in (plain, split, joined, lag, adjusted, supply):
        assert find_daily_close_sql(src, "x.py"), src
    ok_no_close = 'def m(cur):\n    cur.execute("select code, low, high from daily_bars")\n'
    ok_adj_tv = 'def t(cur):\n    cur.execute("select code from daily_bars_adjusted where trade_value > 1")\n'
    ok_supply_flow = 'def u(cur):\n    cur.execute("select code, individual, flu_rt from supply_demand")\n'
    for src in (ok_no_close, ok_adj_tv, ok_supply_flow):
        assert not find_daily_close_sql(src, "x.py"), src


def test_static_rule_would_catch_the_old_prev_day_movers():
    """기존 위반(realtime/store.py 옛 _prev_day_movers)을 되돌리면 규칙이 실패해야 한다."""
    old = ('def _prev_day_movers(conn, limit, *, tv_floor, min_gain=0.10):\n'
           '    cur = conn.cursor()\n'
           '    cur.execute(\n'
           '        "select t.code from daily_bars_adjusted t "\n'
           '        "join daily_bars_adjusted y on y.code = t.code "\n'
           '        "where t.date = %s and y.date = %s "\n'
           '        "and t.close is not null and y.close is not null and y.close > 0 "\n'
           '        "order by (t.close / y.close) desc limit %s", (1, 2, 3))\n')
    assert find_daily_close_sql(old, "realtime/store.py")


def test_cutoff_constant_is_the_aftermarket_start():
    from scalp_it.regular_session import AFTERMARKET_START

    assert date(2026, 9, 14) == AFTERMARKET_START


def test_coverage_reports_codes_whose_regular_close_is_unknown(db):
    """분봉 밖 종목은 그날 정규장 종가를 모른다 — 이벤트가 덜 잡힌 걸 숨기지 않는다."""
    from scalp_it.cli_coverage import run_day
    from scalp_it.realtime.coverage import Coverage, format_report

    db.execute("create table ticks (code TEXT, ts TIMESTAMP, seq INTEGER)")
    row = run_day(db, day=D2, market="코스닥", min_rates=[0.295], before=None)[0]
    assert row["n_bars"] == 2 and row["n_unknown"] == 1
    text = format_report(Coverage(), day=D2, min_rate=0.295, n_bars=2, n_unknown=1)
    assert "정규장 종가 미상 1종목" in text



# ---------------------------------------------------------------- 10-12 리뷰 결함 회귀
# M1 신규상장 다음 날 탈락 · M2 애프터마켓 종가로 잡힌 배율 · M3 장중 편입 기준가 · 크론 실패 경로.

@pytest.fixture(autouse=True)
def _fresh_session_warnings():
    """직전 거래일 불일치 경고는 프로세스에서 한 번만 찍는다 — 테스트마다 비운다."""
    import scalp_it.regular_session as rs

    rs._WARNED_SESSIONS.clear()
    yield
    rs._WARNED_SESSIONS.clear()


SESSION = date(2026, 10, 12)   # D2(10-08) 다음 세션 — 장 전 08:55


def _listing(db, code, *, minute: bool):
    """D2 상장 신규주. 일봉 1행(애프터마켓 오염), 분봉은 ``minute`` 일 때만(15:45 수집 여부)."""
    db.execute("insert into stocks values (?,?,?,?,?)", (code, "신규", "코스닥", "전기/전자", "A"))
    _day(db, code, D2, open_=18_000, high=20_000, low=17_000, close=18_220,
         after_close=19_500, after_high=21_000, minute=minute)
    db.commit()


def _cfg(**kw):
    from scalp_it.universe import UniverseConfig

    base = {"lookback_days": 3, "limit": 10, "min_trade_value_eok": 100.0,
            "min_range_pct": 1.0, "min_surge_days": 1, "min_price": 1_000}
    base.update(kw)
    return UniverseConfig(**base)


def test_new_listing_is_in_next_day_universe_with_listing_day_minute_base(db):
    """상장 당일 분봉을 그날 15:45 에 받으면(cli_intraday tick_day) 다음 날 기준가 = 상장일 정규장 종가."""
    from scalp_it.universe import build_universe

    _listing(db, "444444", minute=True)
    uni = build_universe(db, D2, _cfg(pinned=("444444",)), session=SESSION)
    assert "444444" in uni.codes
    assert uni.prev_close_by_code["444444"] == 18_220, "애프터마켓 종가 19,500 이 아니다"


def test_new_listing_without_minutes_stays_pinned_and_warns(db, capsys):
    """상장일 분봉이 아직 없으면 기준가는 모른다 — 그래도 고정 편입은 꺼지지 않고, 이유를 찍는다.

    옛 코드는 _regularize 가 그 봉을 버리고(정규장 종가 미상) build_universe 가 기준가 없는 후보를
    빼서 상장 다음 날 §4-5 편입이 조용히 사라졌다(486510 09-29).
    """
    from scalp_it.universe import build_universe

    _listing(db, "444444", minute=False)
    uni = build_universe(db, D2, _cfg(pinned=("444444",)), session=SESSION)
    assert "444444" in uni.codes
    assert "444444" not in uni.prev_close_by_code, "모르는 기준가를 오염 일봉으로 메우면 안 된다"
    err = capsys.readouterr().err
    assert "기준가 미상 1/" in err and "444444" in err
    assert "분봉 수집 대상 밖" in err
    # 고정 편입이 아닌 종목은 예전처럼 뺀다(틀린 상한가보다 모르는 게 낫다).
    uni2 = build_universe(db, D2, _cfg(), session=SESSION)
    assert "444444" not in uni2.codes


def test_minute_cron_failure_warns_with_the_cause(db, capsys):
    """15:45 분봉 크론이 실패한 날: 일봉은 D2 까지 있는데 D2 분봉이 없다 → 후보 전부 탈락 + 원인 경고."""
    from scalp_it.universe import build_universe

    db.execute("delete from minute_bars where ts >= ?", (D2,))
    db.commit()
    uni = build_universe(db, D2, _cfg(), session=SESSION)
    assert uni.codes == []
    err = capsys.readouterr().err
    assert "분봉 종가 봉이 한 종목도 없다" in err, err   # explain_missing_bases 의 원인 줄
    assert "기준가 미상" in err


def test_previous_session_uses_minutes_when_daily_load_is_late(db, capsys):
    """전일 일봉 적재가 밀리면 일봉 최신은 이틀 전이다 — 분봉 최신과 대조해 실제 직전 거래일을 쓴다."""
    from scalp_it.regular_session import previous_session

    db.execute("delete from daily_bars where date = ?", (D2,))
    db.commit()
    assert previous_session(db, SESSION) == date(2026, 10, 8)
    assert "일봉 적재 전이거나 실패" in capsys.readouterr().err
    # 기준가도 이틀 전(D1 11,000)이 아니라 D2 정규장 종가 11,500.
    assert base_prices(db, SESSION, ["111111"]) == {"111111": 11_500}


def test_previous_session_agreeing_sources_are_silent(db, capsys):
    from scalp_it.regular_session import previous_session

    assert previous_session(db, SESSION) == date(2026, 10, 8)
    assert "불일치" not in capsys.readouterr().err


def test_fetch_failure_is_logged_not_silent(capsys):
    conn = sqlite3.connect(":memory:")
    assert base_prices(conn, D2, ["111111"]) == {}
    err = capsys.readouterr().err
    assert "[regular_session]" in err and "조회 실패" in err


def test_rescale_after_aftermarket_does_not_fire_on_an_aftermarket_move(db):
    """애프터마켓에서 +30% 간 날 — 옛 코드는 일봉 종가/분봉 15:30 로 배율을 걸어 레인지를 부풀렸다.

    087010 10-07: 범위 중앙값 35.9% vs 정상 6.7%. 시가(오염 없음)로 비교하면 배율이 안 걸린다.
    """
    from scalp_it.universe import build_candidates, fetch_bars

    db.execute("update daily_bars set close = 15000, high = 15000 where code='111111' and date = ?", (D2,))
    db.commit()
    bars = fetch_bars(db, D2, _cfg())
    cand = {c.code: c for c in build_candidates(bars, date(2026, 10, 8), _cfg())}["111111"]
    assert cand.last_range_pct == pytest.approx((11_600 - 10_900) / 11_500 * 100)
    assert {b.day: b.open for b in bars if b.code == "111111"}[date(2026, 10, 7)] == 10_050


def test_rescale_after_aftermarket_still_catches_a_real_merge(db):
    """기업행동(10:1 병합) 뒤 분봉이 수정주가로 재수집된 종목은 시가 비교로도 배율이 걸린다."""
    from scalp_it.universe import build_candidates, fetch_bars

    db.execute("insert into stocks values ('555555', '병합', '코스닥', '화학', 'A')")
    for day, (o, h, lo, c_after, c_reg) in {D1: (1_000, 1_100, 990, 1_080, 1_050),
                                            D2: (1_050, 1_200, 1_040, 1_190, 1_100)}.items():
        db.executemany("insert into minute_bars values (?,?,?,?,?,?,?,?)", [
            ("555555", f"{day} 09:00:00", o * 10, h * 10, lo * 10, o * 10, 600, 6e9),
            ("555555", f"{day} 15:30:00", c_reg * 10, c_reg * 10, c_reg * 10, c_reg * 10, 400, 4e9),
        ])
        db.execute("insert into daily_bars values (?,?,?,?,?,?,?,?,?)",
                   ("555555", day, o, h, lo, c_after, 1000, 11_000, "t"))
    db.commit()
    bars = [b for b in fetch_bars(db, D2, _cfg()) if b.code == "555555"]
    by_day = {b.day: b for b in bars}
    assert by_day[date(2026, 10, 8)].open == 10_500         # 일봉 시가 × 10
    assert by_day[date(2026, 10, 8)].close == 11_000        # 정규장 종가(분봉)
    cand = {c.code: c for c in build_candidates(bars, date(2026, 10, 8), _cfg())}["555555"]
    assert cand.last_range_pct == pytest.approx((12_000 - 10_400) / 11_000 * 100)


def test_breaking_meta_fills_unknown_bases_from_the_scan_row(db, tmp_path, monkeypatch):
    """장중 대장 스캔 편입 종목 중 분봉 밖 종목은 DB 기준가가 없다 — 순위 응답 현재가·등락률로 메운다."""
    import argparse

    import scalp_it.cli_pair_detect as cpd

    path = tmp_path / "kq.db"
    disk = sqlite3.connect(path)
    db.backup(disk)
    disk.close()

    class _Now(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 12, 10, 0)  # noqa: DTZ001 — KST naive(프로젝트 표준)

    monkeypatch.setattr(cpd, "datetime", _Now)
    scan = {"111111": 99_999.0, "333333": 2_150.0}
    meta = cpd._breaking_meta(argparse.Namespace(dsn=str(path)), ["111111", "333333"], scan_bases=scan)
    assert meta["prev_close"] == {"111111": 11_500.0, "333333": 2_150.0}, "DB 정문이 1순위, 스캔은 빈 곳만"

    import scalp_it.intraday.store as st

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(st, "connect", boom)
    meta = cpd._breaking_meta(argparse.Namespace(dsn=None), ["111111", "333333"], scan_bases=scan)
    assert meta == {"prev_close": {"111111": 99_999.0, "333333": 2_150.0}}


def test_scan_bases_convert_percent_rate_to_the_session_base():
    """ka10027 ``flu_rt`` 는 % 다(수급표는 ×100). 1,331 / 1.2998 → 1,024 — 상한가 1,331 과 맞는다."""
    import scalp_it.cli_pair_detect as cpd
    from scalp_it.realtime.leader_scan import parse_scan_rows
    from scalp_it.regular_session import base_price_from_pct

    rows = parse_scan_rows([{"stk_cd": "043200_AL", "cur_prc": "+1331", "flu_rt": "+29.98"},
                            {"stk_cd": "000001_AL", "cur_prc": "0", "flu_rt": "0"}])
    assert cpd._scan_bases(rows) == {"043200": 1_024.0}
    assert limit_up_price(1_024) == 1_331
    assert base_price_from_pct(13_000, 30.0) == 10_000
    assert base_price_from_pct(13_000, None) is None


def test_prev_day_movers_rank_by_regular_close_over_base(db):
    """cli_tick --auto 의 전일 급등주 — 일봉(수정) 종가 비율이 아니라 정규장 종가 / 기준가."""
    from scalp_it.realtime.store import _prev_day_movers

    db.execute("create table daily_bars_adjusted as select * from daily_bars")
    db.commit()
    # 정규장: 111111 +4.5%(11,500/11,000), 222222 +3.6%, 333333 D2 정규장 종가 모름.
    # 오염 일봉으로 세면 111111 +5.4%, 333333 +4.8% 가 잡힌다.
    assert _prev_day_movers(db, 10, tv_floor=0.0, min_gain=0.04) == ["111111"]
    assert _prev_day_movers(db, 10, tv_floor=0.0, min_gain=0.05) == []
