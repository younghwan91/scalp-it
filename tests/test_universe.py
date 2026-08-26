"""유니버스 선정 테스트 — 순수 로직은 합성 Bar 로, DB 경로는 합성 sqlite 로."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from scalp_it.universe import (
    UNSECTORED,
    Bar,
    Candidate,
    UniverseConfig,
    build_candidates,
    build_universe,
    fetch_bars,
    rank_candidates,
    select_universe,
    trading_days,
)

ASOF = date(2026, 8, 25)


def _bars(
    code: str,
    *,
    sector: str = "전기/전자",
    market: str = "코스닥",
    close: int = 10_000,
    range_pct: float = 8.0,
    trade_value: int = 20_000,  # 백만원 = 200억
    days: int = 5,
    daily_ret: float = 0.0,
    name: str = "",
) -> list[Bar]:
    """등락률 `daily_ret` 로 매일 오르는 일봉 `days` 개. 마지막 날이 ASOF."""
    out: list[Bar] = []
    price = close / ((1 + daily_ret) ** (days - 1))
    for i in range(days):
        c = round(price * (1 + daily_ret) ** i)
        half = int(c * range_pct / 200.0)
        out.append(
            Bar(
                code=code,
                day=ASOF - timedelta(days=days - 1 - i),
                open=c,
                high=c + half,
                low=c - half,
                close=c,
                trade_value=trade_value,
                volume=1_000_000,
                name=name or code,
                sector=sector,
                market=market,
            )
        )
    return out


# ------------------------------------------------------- build_candidates

def test_build_candidates_computes_metrics() -> None:
    cands = build_candidates(_bars("111111", range_pct=10.0, daily_ret=0.15), ASOF)
    assert len(cands) == 1
    c = cands[0]
    assert c.trade_value_eok == pytest.approx(200.0)      # 20,000 백만원 = 200억
    assert c.range_pct == pytest.approx(10.0, abs=0.2)
    assert c.surge_days == 4                              # 15% 상승 4회
    assert c.prev_close == 10_000


def test_build_candidates_counts_limitup_at_29pct() -> None:
    """상한가는 30% 지만 호가단위 반올림으로 29.x% 로 찍히는 날이 있다."""
    cands = build_candidates(_bars("111111", daily_ret=0.295), ASOF)
    assert cands[0].limitup_days == 4


def test_build_candidates_drops_stocks_without_asof_bar() -> None:
    """거래정지로 기준일 봉이 없으면 전일 종가를 못 주므로 버린다."""
    stale = [b for b in _bars("111111") if b.day < ASOF]
    assert build_candidates(stale, ASOF) == []


def test_build_candidates_uses_float_division() -> None:
    """close 가 정수라 캐스팅을 빠뜨리면 등락폭이 0 으로 뭉개진다(정수 나눗셈 함정)."""
    bar = Bar("111111", ASOF, 1000, 1050, 990, 1000, 10_000, 100)
    assert build_candidates([bar], ASOF)[0].range_pct == pytest.approx(6.0)


def test_empty_sector_becomes_unsectored_label() -> None:
    cands = build_candidates(_bars("111111", sector=""), ASOF)
    assert cands[0].sector == UNSECTORED


# ------------------------------------------------------------ rank_candidates

def _cand(code: str, **kw) -> Candidate:
    base = {
        "name": code,
        "sector": "전기/전자",
        "market": "코스닥",
        "prev_close": 10_000,
        "trade_value_eok": 200.0,
        "range_pct": 8.0,
        "last_range_pct": 8.0,
        "surge_days": 2,
        "limitup_days": 0,
        "avg_fill_won": 10_000.0,
    }
    base.update(kw)
    return Candidate(code=code, **base)  # type: ignore[arg-type]


def test_rank_filters_out_thin_and_quiet_and_cheap_and_wrong_market() -> None:
    cands = [
        _cand("111111"),                              # 통과
        _cand("222222", trade_value_eok=50.0),        # 거래대금 미달
        _cand("333333", range_pct=2.0),               # 변동성 미달
        _cand("444444", prev_close=500),              # 저가주(틱 노이즈)
        _cand("555555", market="거래소"),              # 시장 제외
        _cand("666666", surge_days=0),                # 급등 이력 없음
    ]
    ranked = rank_candidates(cands)
    assert [c.code for c in ranked] == ["111111"]


def test_rank_orders_turnover_first_then_volatility() -> None:
    """§2-1: 거래대금이 최우선. 변동성은 같은 거래대금대에서만 순위를 뒤집는다."""
    big_quiet = _cand("111111", trade_value_eok=2000.0, range_pct=5.5)
    small_wild = _cand("222222", trade_value_eok=150.0, range_pct=25.0)
    same_wild = _cand("333333", trade_value_eok=2000.0, range_pct=20.0)
    ranked = rank_candidates([small_wild, big_quiet, same_wild])
    assert [c.code for c in ranked] == ["333333", "111111", "222222"]


def test_pinned_bypasses_filters_and_is_marked() -> None:
    cfg = UniverseConfig(pinned=("999999",))
    ranked = rank_candidates([_cand("999999", trade_value_eok=1.0, range_pct=0.1)], cfg)
    assert [c.code for c in ranked] == ["999999"]
    assert ranked[0].pinned is True


# ------------------------------------------------------------ select_universe

def _pool(n_per_sector: dict[str, int], base_score: float = 500.0) -> list[Candidate]:
    """섹터별 n 종목. 코드 순서가 곧 점수 내림차순이 되게 만든다."""
    out: list[Candidate] = []
    i = 0
    for sector, n in n_per_sector.items():
        for k in range(n):
            i += 1
            out.append(
                _cand(f"{i:06d}", sector=sector, trade_value_eok=base_score * 10 ** (-k * 0.01))
            )
    return rank_candidates(out)


def test_select_respects_limit() -> None:
    ranked = _pool({"A": 20, "B": 20, "C": 20})
    uni = select_universe(ranked, ASOF, UniverseConfig(limit=10, max_per_sector=8))
    assert len(uni.codes) == 10


def test_select_gives_every_included_sector_at_least_two() -> None:
    """짝꿍은 대장+후속이 둘 다 구독돼야 성립한다 — 외톨이 섹터를 만들지 않는다."""
    ranked = _pool({"A": 9, "B": 9, "C": 9, "D": 9, "E": 9})
    uni = select_universe(ranked, ASOF, UniverseConfig(limit=21, max_per_sector=8))
    assert len(uni.codes) == 21
    assert all(n >= 2 for n in uni.sector_counts.values()), uni.sector_counts
    assert uni.pairable_codes == uni.codes


def test_select_caps_a_dominant_sector() -> None:
    """코스닥은 '전기/전자'가 후보의 절반이라 상한이 없으면 한 테마가 유니버스를 먹는다."""
    ranked = _pool({"전기/전자": 40, "화학": 5, "제약": 5})
    uni = select_universe(ranked, ASOF, UniverseConfig(limit=20, max_per_sector=8))
    assert uni.sector_counts["전기/전자"] == 8
    # 상한 때문에 limit(20)을 다 못 채운다 — 8 + 5 + 5. 한 테마로 채우느니 비우는 게 낫다.
    assert len(uni.codes) == 18


def test_select_skips_sectors_that_cannot_field_a_pair() -> None:
    ranked = _pool({"A": 1, "B": 4})
    uni = select_universe(ranked, ASOF, UniverseConfig(limit=10, max_per_sector=8))
    assert "A" not in uni.sector_counts
    assert uni.sector_counts == {"B": 4}


def test_select_excludes_unsectored_by_default() -> None:
    ranked = _pool({UNSECTORED: 4, "B": 4})
    uni = select_universe(ranked, ASOF, UniverseConfig(limit=10))
    assert UNSECTORED not in uni.sector_counts


def test_select_admits_unsectored_when_slots_given() -> None:
    ranked = _pool({UNSECTORED: 4, "B": 4})
    uni = select_universe(ranked, ASOF, UniverseConfig(limit=10, unsectored_slots=2))
    assert uni.sector_counts[UNSECTORED] == 2


def test_select_puts_pinned_in_first_even_when_lowest_scored() -> None:
    cfg = UniverseConfig(limit=4, pinned=("999999",))
    ranked = rank_candidates(
        [*_pool({"B": 6}), _cand("999999", sector=UNSECTORED, trade_value_eok=1.0)], cfg
    )
    uni = select_universe(ranked, ASOF, cfg)
    assert "999999" in uni.codes


def test_universe_exports_sector_and_prev_close_maps() -> None:
    """감지기가 상한가 계산(전일종가)과 섹터 순위에 쓰는 두 맵."""
    ranked = _pool({"A": 4})
    uni = select_universe(ranked, ASOF, UniverseConfig(limit=4))
    assert set(uni.sector_by_code) == set(uni.codes)
    assert set(uni.prev_close_by_code) == set(uni.codes)
    assert all(v > 0 for v in uni.prev_close_by_code.values())


# ----------------------------------------------------------------- 합성 DB

def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("create table stocks (code text, name text, market text, sector text, kind text)")
    conn.execute(
        "create table daily_bars (code text, date text, open int, high int, low int, "
        "close int, volume int, trade_value int, source text)"
    )
    rows_s = [
        ("111111", "대장", "코스닥", "전기/전자", "A"),
        ("222222", "후속", "코스닥", "전기/전자", "A"),
        ("333333", "얇은주", "코스닥", "화학", "A"),
        ("444444", "코스피주", "거래소", "화학", "A"),
    ]
    conn.executemany("insert into stocks values (?,?,?,?,?)", rows_s)
    bars: list[tuple] = []
    for code, tv in (("111111", 30_000), ("222222", 25_000), ("333333", 10), ("444444", 90_000)):
        prev = 8_000
        for i in range(5):
            day = (ASOF - timedelta(days=4 - i)).isoformat()
            close = int(prev * 1.15)
            bars.append((code, day, prev, close + 400, prev - 400, close, 1_000_000, tv, "t"))
            prev = close
    conn.executemany("insert into daily_bars values (?,?,?,?,?,?,?,?,?)", bars)
    conn.commit()
    return conn


def test_trading_days_returns_oldest_first() -> None:
    conn = _make_db()
    days = trading_days(conn, ASOF.isoformat(), 3)
    assert days == sorted(days)
    assert len(days) == 3


def test_fetch_bars_is_market_scoped() -> None:
    conn = _make_db()
    bars = fetch_bars(conn, ASOF.isoformat(), UniverseConfig(lookback_days=5))
    assert {b.code for b in bars} == {"111111", "222222", "333333"}


def test_fetch_bars_lets_pinned_escape_the_market_filter() -> None:
    conn = _make_db()
    cfg = UniverseConfig(lookback_days=5, pinned=("444444",))
    bars = fetch_bars(conn, ASOF.isoformat(), cfg)
    assert "444444" in {b.code for b in bars}


def test_build_universe_end_to_end_on_sqlite() -> None:
    conn = _make_db()
    cfg = UniverseConfig(lookback_days=5, limit=10, min_trade_value_eok=100.0)
    uni = build_universe(conn, ASOF.isoformat(), cfg)
    assert uni.asof == ASOF
    assert uni.codes == ["111111", "222222"]   # 거래대금 순, 얇은주·거래소주 탈락
    assert uni.sector_by_code == {"111111": "전기/전자", "222222": "전기/전자"}
    assert uni.prev_close_by_code["111111"] > 0


# ----- 만쥬 §4-5 신규상장 · §5-5 테마 신선도 --------------------------------

def test_pick_recent_listings_from_first_days():
    """상장일이 최근인 종목만 고른다 — 만쥬 §4-5 "상장 당일은 무조건 관심종목"."""
    from scalp_it.universe import pick_recent_listings

    first = {"111111": date(2026, 8, 25), "222222": date(2026, 8, 14),
             "333333": date(2025, 1, 2)}
    got = pick_recent_listings(first, asof=date(2026, 8, 25), within_days=5)
    assert got == ["111111"]                       # 08-14 는 5일(달력) 밖
    # 경계 포함: 정확히 within_days 일 전이면 들어온다
    assert pick_recent_listings(first, asof=date(2026, 8, 25), within_days=11) == [
        "111111", "222222"]
    assert pick_recent_listings(first, asof=date(2026, 8, 25), within_days=0) == []


def test_stale_sectors_flags_overused_themes():
    """최근 상한가가 잦은 섹터는 '소비된 테마' — 만쥬 §5-5.

    "이미 수차례 소비되어 신선도가 떨어진 테마는 대장이 상을 가도 후발주가 잘
    따라가지 못한다". 섹터별 상한가 횟수를 그 대용치로 쓴다.
    """
    from scalp_it.universe import stale_sectors

    counts = {"전기/전자": 7, "제약": 3, "금융": 0}
    assert stale_sectors(counts, max_limitups=5) == {"전기/전자"}
    assert stale_sectors(counts, max_limitups=0) == set()      # 0 = 필터 끔
    assert stale_sectors(counts, max_limitups=2) == {"전기/전자", "제약"}
