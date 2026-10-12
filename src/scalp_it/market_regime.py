"""시황 게이트 — 코스닥 전체 거래대금으로 그날 매매 여부·비중을 정한다.

근거: `docs/research/manju/44-manju-method-from-videos.md` §1 (묶음 9).

    호황/상승장  코스닥 전체 거래대금 **10조~13조 원 이상** → 비중·매매 횟수를 늘린다
    침체/하락장  **5조~8조 원 이하** → 몸을 사리고 관망, 매매 횟수 극도로 축소
    "거래대금이 마른 장에서 단기 변동성 트레이더는 수익을 낼 수 없으므로
     매매를 쉬는 것이 계좌를 지키는 최선" [03:48]

**8~10조 구간은 원자료에 지침이 없다**(문서 §11 #8). 우리 판단으로 `MILD` 를 만들었고
배수는 활황과 침체의 중간(0.6)으로 뒀다. 근거: 만쥬는 10조↑에서 "늘린다", 8조↓에서
"줄인다"고 말했으므로 그 사이는 **평시(늘리지도 줄이지도 않음)** 로 읽는 것이 서술과
정합적이다. 다만 우리 배수는 "정상 상한 대비 축소율"로 정의하므로 활황 1.0 이 상한이고
평시는 그 아래가 된다 — 절대 비중 상한은 risk_guard 가 따로 정하고, 여기서 나온 배수는
거기에 곱해지는 감산 계수다.

**한계 — 장전 판정은 전일 값으로 한다.** 당일 거래대금은 장이 끝나야 확정되므로 08:55
게이트는 직전 영업일 값을 쓸 수밖에 없다. 즉 "어제 말랐으니 오늘도 마를 것"이라는 지속성
가정에 기댄다. 실제로 코스닥 거래대금은 하루 사이에 두 배가 되기도 하므로, 장중에
당일 값이 쌓이면 재판정하는 것이 옳다(이 모듈은 그 재판정 함수를 따로 주지 않는다 —
:func:`evaluate_gate` 에 오늘 날짜를 넣으면 장중 누적으로 판정된다).

DB 커넥션은 **주입받는다**. 이 모듈은 접속 문자열을 모른다(테스트 가능성).
쿼리는 SELECT 뿐이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from .adapters.db import _placeholder as _ph

__all__ = [
    "COLD",
    "DEAD",
    "HOT",
    "MILD",
    "RegimeReading",
    "RegimeThresholds",
    "classify_regime",
    "evaluate_gate",
    "fetch_leader_turnover",
    "fetch_market_turnover",
    "fetch_recent_turnover",
    "previous_session",
    "size_multiplier",
]

KOSDAQ = "코스닥"

HOT = "활황"      # 10조↑ — 만쥬: 비중·횟수를 늘린다
MILD = "중간"     # 8~10조 — 원자료 공백. 우리가 만든 구간
COLD = "침체"     # 5~8조 — 몸을 사리고 관망
DEAD = "극침체"   # 5조↓ — 매매를 쉰다


@dataclass(frozen=True)
class RegimeThresholds:
    """국면 경계. 단위는 **조원**이고 전부 인자로 덮어쓸 수 있다.

    기본값은 만쥬 원문 그대로다(10조 / 8조 / 5조).
    """

    hot: float = 10.0
    mild: float = 8.0
    cold: float = 5.0

    def __post_init__(self) -> None:
        if not (self.hot > self.mild > self.cold > 0):
            raise ValueError("임계값은 hot > mild > cold > 0 이어야 한다")


#: 국면별 포지션 사이즈 감산 계수. 활황 1.0 이 상한이고 아래로만 깎는다.
#: 만쥬는 활황에 "늘린다"고 했지만 우리는 절대 상한을 risk_guard 가 쥐고 있으므로
#: 그 상한을 넘겨 키우는 방향은 여기서 만들지 않는다(안전 쪽으로 비대칭).
_SIZE_MULTIPLIER = {HOT: 1.0, MILD: 0.6, COLD: 0.3, DEAD: 0.0}


@dataclass(frozen=True)
class RegimeReading:
    """게이트 판정 결과."""

    asof: date                      # 거래대금을 읽은 날(보통 직전 영업일)
    target: date | None             # 이 판정을 적용할 날(장전 판정 시 오늘). 모르면 None
    turnover_jo: float              # 코스닥 전체 거래대금(조원)
    regime: str
    tradable: bool                  # 오늘 매매해도 되는가
    size_multiplier: float
    leader_code: str | None = None  # 그날 거래대금 1위(보조 지표, §1 [02:59])
    leader_name: str | None = None
    leader_turnover_jo: float | None = None
    note: str = ""

    @property
    def leader_healthy(self) -> bool | None:
        """주도주 1종목 거래대금이 활황 수준(1조↑)인가. §1: 침체장은 1~2천억."""
        if self.leader_turnover_jo is None:
            return None
        return self.leader_turnover_jo >= 1.0


# ------------------------------------------------------------------ 순수 로직

def classify_regime(turnover_jo: float, thresholds: RegimeThresholds | None = None) -> str:
    """거래대금(조원)을 국면 라벨로 바꾼다. DB 를 모르는 순수 함수."""
    th = thresholds or RegimeThresholds()
    if turnover_jo >= th.hot:
        return HOT
    if turnover_jo >= th.mild:
        return MILD
    if turnover_jo >= th.cold:
        return COLD
    return DEAD


def size_multiplier(regime: str) -> float:
    """국면별 포지션 감산 계수. 모르는 라벨은 가장 보수적으로 0."""
    return _SIZE_MULTIPLIER.get(regime, 0.0)


def _note_for(regime: str, turnover_jo: float) -> str:
    if regime == HOT:
        return f"활황({turnover_jo:.2f}조) — 만쥬 기준 공격 구간"
    if regime == MILD:
        return (
            f"중간({turnover_jo:.2f}조) — 원자료에 지침이 없는 8~10조 구간. "
            "우리 판단으로 비중 축소 후 매매 허용"
        )
    if regime == COLD:
        return f"침체({turnover_jo:.2f}조) — 몸을 사리고 매매 횟수 극도로 축소"
    return f"극침체({turnover_jo:.2f}조) — 만쥬 기준 '쉬는 것이 최선'. 매매 금지"


# --------------------------------------------------------------------- DB 읽기


def fetch_market_turnover(conn: Any, day: date | str, market: str = KOSDAQ) -> float | None:
    """그날 `market` 전체 거래대금을 **조원**으로 돌려준다. 데이터가 없으면 None.

    `daily_bars.trade_value` 의 단위는 **백만원**이라 1e6 으로 나누면 조원이 된다.

    2026-09-14 이후 이 값에는 애프터마켓(16:00~20:00) 거래대금이 섞여 있다. 정규장만의 시장 합계는
    분봉(~1,400종목만 덮음)으로 다시 만들 수 없어 일봉 합계를 그대로 쓴다 — 게이트(5조·8조)는
    그만큼 후하게 열린다. 정규장 값이 필요한 종목 단위 계산은 :mod:`scalp_it.regular_session` 으로.
    """
    ph = _ph(conn)
    sql = (
        "select sum(d.trade_value) from daily_bars d join stocks s on s.code = d.code "
        f"where s.market = {ph} and d.date = {ph}"
    )
    cur = conn.cursor()
    cur.execute(sql, (market, day))
    row = cur.fetchone()
    cur.close()
    if not row or row[0] is None:
        return None
    return float(row[0]) / 1e6


def fetch_recent_turnover(
    conn: Any, asof: date | str, days: int = 5, market: str = KOSDAQ
) -> list[tuple[date, float]]:
    """`asof` 이하 최근 `days` 영업일의 (날짜, 거래대금 조원). 오래된 순."""
    ph = _ph(conn)
    sql = (
        "select d.date, sum(d.trade_value) from daily_bars d join stocks s on s.code = d.code "
        f"where s.market = {ph} and d.date <= {ph} "
        f"group by d.date order by d.date desc limit {ph}"
    )
    cur = conn.cursor()
    cur.execute(sql, (market, asof, days))
    rows = cur.fetchall()
    cur.close()
    return [(r[0], float(r[1]) / 1e6) for r in reversed(rows) if r[1] is not None]


def fetch_leader_turnover(
    conn: Any, day: date | str, market: str = KOSDAQ
) -> tuple[str, str, float] | None:
    """그날 거래대금 1위 종목 (코드, 이름, 조원). §1 [02:59] 의 보조 지표.

    활황기 주도주는 하루 1~2조↑, 침체장은 1~2천억 수준이라는 대용 지표다.
    """
    ph = _ph(conn)
    sql = (
        "select d.code, s.name, d.trade_value from daily_bars d join stocks s on s.code = d.code "
        f"where s.market = {ph} and d.date = {ph} "
        "order by d.trade_value desc limit 1"
    )
    cur = conn.cursor()
    cur.execute(sql, (market, day))
    row = cur.fetchone()
    cur.close()
    if not row or row[2] is None:
        return None
    return str(row[0]), str(row[1] or ""), float(row[2]) / 1e6


def previous_session(conn: Any, before: date | str, market: str = KOSDAQ) -> date | None:
    """`before` **미만**의 가장 최근 거래일. 장전 게이트가 쓸 기준일이다."""
    ph = _ph(conn)
    sql = (
        "select max(d.date) from daily_bars d join stocks s on s.code = d.code "
        f"where s.market = {ph} and d.date < {ph}"
    )
    cur = conn.cursor()
    cur.execute(sql, (market, before))
    row = cur.fetchone()
    cur.close()
    return row[0] if row and row[0] else None


# ----------------------------------------------------------------------- 게이트

def evaluate_gate(
    conn: Any,
    asof: date | str,
    *,
    target: date | None = None,
    thresholds: RegimeThresholds | None = None,
    market: str = KOSDAQ,
    with_leader: bool = True,
) -> RegimeReading:
    """`asof` 거래대금으로 국면을 판정한다.

    장전 08:55 에는 `asof=직전 영업일`, `target=오늘` 로 부른다
    (:func:`previous_session` 참고). 당일 값을 못 쓰는 한계는 모듈 docstring 참고.
    """
    turnover = fetch_market_turnover(conn, asof, market)
    if turnover is None:
        raise ValueError(f"{market} {asof} 거래대금 데이터가 없다")
    regime = classify_regime(turnover, thresholds)
    leader = fetch_leader_turnover(conn, asof, market) if with_leader else None
    asof_date = asof if isinstance(asof, date) else date.fromisoformat(str(asof))
    return RegimeReading(
        asof=asof_date,
        target=target,
        turnover_jo=turnover,
        regime=regime,
        tradable=regime != DEAD,
        size_multiplier=size_multiplier(regime),
        leader_code=leader[0] if leader else None,
        leader_name=leader[1] if leader else None,
        leader_turnover_jo=leader[2] if leader else None,
        note=_note_for(regime, turnover),
    )


#: 광기 구간 문턱 — 만쥬 §4-3 은 판정 기준을 말하지 않는다(문서 44 §11 #12 "정의 공백").
#: 그래서 우리 1년 실측 분위(2025-08~2026-08, 259거래일)로 정한다:
#:   코스닥 거래대금 중앙 10.1조 · 90% 16.0조 · 최대 25.3조
#:   일별 상한가   중앙 9건  · 90% 15건  · 최대 25건
#: 둘 다 90분위를 넘는 날을 광기로 본다. **한 축만 극단인 건 광기가 아니다** —
#: 거래대금만 크면 대형주 장세일 수 있고, 상한가만 많으면 테마 하나가 뜬 것일 수 있다.
MANIA_TURNOVER_JO = 16.0
MANIA_LIMITUPS = 15


def is_mania(*, turnover_jo: float, limitups: int,
             min_turnover_jo: float = MANIA_TURNOVER_JO,
             min_limitups: int = MANIA_LIMITUPS) -> bool:
    """시장이 '광기 구간'인가(만쥬 §4-3).

    > "시장의 광기가 극에 달한 구간에서는 **최대 1시간 이내 단기 매매로 제한**하고
    > **오버나이트 절대 금지**" [04:29]

    우리 보유 시간이 이미 4분이라 1시간 제한과 오버나이트 금지는 자동으로 지켜진다.
    그래서 이 판정의 쓸모는 **기록**이다 — 그날이 광기였는지 알아야 나중에 성적을
    제대로 읽는다. 광기 구간의 성적을 평상시와 섞으면 판단이 무너진다.
    """
    return turnover_jo >= min_turnover_jo and limitups >= min_limitups


def count_limitups(conn: Any, day: date | str, market: str = KOSDAQ) -> int:
    """그날 상한가 종목 수(정규장 종가가 기준가 대비 +29.5% 이상).

    정문 :func:`scalp_it.regular_session.limitup_codes` 로 센다. 일봉 ``close / lag(close)`` 는
    2026-09-14 이후 애프터마켓 종가라 틀린다(그 전에는 ``::float`` 없는 정수 나눗셈으로 1년 2,313건을
    5건으로 센 적이 있다).
    """
    from .regular_session import limitup_codes

    ph = _ph(conn)
    cur = conn.cursor()
    cur.execute(f"select code from stocks where market = {ph}", [market])
    codes = [str(r[0]) for r in cur.fetchall()]
    cur.close()
    return len(limitup_codes(conn, day, codes))


#: 만쥬 A1(2026-08-30) — Gate 0 아래 구간의 비중. "짝꿍주가 보이더라도 평소 비중의
#: **1/10** 또는 소액 정찰병으로만 대응하며 시드를 온전히 보존해야 하는 구간."
SCOUT_MULTIPLIER = 0.1


def scout_multiplier(*, turnover_jo: float, gate_jo: float) -> float:
    """Gate 0(최소 유동성) 아래면 정찰병 비중, 위면 평소 비중.

    만쥬 A1 은 거래대금 기준을 **두 층**으로 갈랐다.

        Gate 0   코스닥 7~8조 이하 = 돈이 마른 침체기. 2등주 호가가 비어 슬리피지와
                 윗꼬리 리스크가 폭증 → "쉬거나 **극소액**으로 통제"
        Regime   8조 이상에서 지수가 횡보·조정하면 대형주 수급이 개별 테마로 확산되어
                 **짝꿍 효율이 최대**

    둘은 모순이 아니라 계층이다. 우리는 게이트를 8조 → 5조로 낮춰 실측 데이터를
    얻고 있는데(2026-08-27 사용자 결정), 그 구간의 **비중이 1/10 이어야** 만쥬
    기준에 맞는다. ``gate_jo`` 가 0 이면 게이트를 끈 것이므로 평소 비중.
    """
    if gate_jo <= 0:
        return 1.0
    return SCOUT_MULTIPLIER if turnover_jo < gate_jo else 1.0
