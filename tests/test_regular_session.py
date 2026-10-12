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

SRC = Path(__file__).resolve().parents[1] / "src" / "scalp_it"
_FROM_DAILY = re.compile(r"\b(?:from|join)\s+daily_bars\b(?!_)", re.IGNORECASE)
_CLOSE = re.compile(r"\bclose\b", re.IGNORECASE)

#: (파일, 함수) — 이유
ALLOWED = {
    ("regular_session.py", "_daily_closes"),   # 정문 자신. 09-14 이전 날짜만 쓴다
    ("universe.py", "fetch_bars"),             # 읽은 뒤 _regularize 가 09-14 이후 종가를 바꾼다
    ("adapters/db.py", "load_daily_bars_raw"), # 기존 연구 재현용. 독스트링에 base_prices 를 쓰라고 적었다
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
    for src in (plain, split, joined, lag):
        assert find_daily_close_sql(src, "x.py"), src
    ok_adjusted = 'def k(cur):\n    cur.execute("select close from daily_bars_adjusted")\n'
    ok_no_close = 'def m(cur):\n    cur.execute("select code, low, high from daily_bars")\n'
    assert not find_daily_close_sql(ok_adjusted, "x.py")
    assert not find_daily_close_sql(ok_no_close, "x.py")


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

