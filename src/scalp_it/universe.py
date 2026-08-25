"""그날의 후보 유니버스 — 아침에 실시간 구독할 ≤50 종목을 전일 일봉으로 고른다.

만쥬는 고정 종목을 쓰지 않는다. 그날 주도 테마에서 대장/후속을 새로 고른다
(`docs/research/manju/44-manju-method-from-videos.md` §2·§3·§4). 그런데 우리는
**실시간 구독 상한**이 있다:

    키움 WebSocket 은 App key 당 동시 세션 1개, 등록 한도 ≈ 100건(종목×type).
    0B+0D 면 종목당 2건이라 **커넥션당 ~50종목이 실질 상한**이다. 넘기면 두 번째
    커넥션이 열려 서로 kick 한다 — 2026-08-24 에 80종목 중 30종목이 10:58 에 끊겼다
    (`realtime/collector.py` warning 참고).

그래서 이 모듈은 **매일 아침 50종목 이내**를 뽑는다.

선정 근거 — 만쥬의 주도주 3대 요건(§2-1):

1. **풍부한 거래대금** (최우선) — 등급 A. `daily_bars.trade_value`(백만원)로 그대로 쓴다.
2. **하루 10%~상한가 수준의 변동성** — 등급 A. 거래대금이 커도 변동성이 미미한 대형주는
   배제한다(그래서 삼성전자·SK하이닉스는 거래대금 1·2위여도 걸러진다).
3. **명확한 재료·스토리** — 등급 C, 자동화 불가. 여기서는 쓰지 않는다.

**호가 두께(§4-1, 만쥬의 최우선 필터)는 넣지 못했다.** 1년치 호가 스냅샷이 없고
`quotes` 는 2일치뿐이라 과거 기준으로 판정할 수가 없다. 대신 **전일 거래대금과 평균
체결단가(거래대금/거래량)를 대용**으로 쓴다 — 거래대금이 크면 호가도 대체로 두껍다는
경험칙이며, **이것은 대용이지 §4-1 의 판정식이 아니다**. 진짜 판정
(`상하 10호가 합산 ≥ k × 주문금액`)은 장중에 호가 스트림으로 해야 한다.

**섹터 다양성**: 한 섹터에 몰리면 그날 그 테마가 안 뜰 때 통째로 공친다. 더 결정적인 건
만쥬 짝꿍이 **대장 + 같은 섹터 후속**을 둘 다 구독해야 성립한다는 점이다. 그래서 선정은
"섹터당 최소 2종목"을 먼저 채우고 남은 슬롯을 전체 점수순으로 메운다
(:func:`select_universe`).

가격은 **원주가(`daily_bars`)** 로 읽는다. 상한가·등락률 판정은 분할이 반영된
`daily_bars_adjusted` 로 하면 비율이 왜곡된다(`adapters/db.py` 의 같은 주석 참고).

DB 커넥션은 **주입받는다**. 쿼리는 SELECT 뿐이다.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from itertools import pairwise
from typing import Any

__all__ = [
    "Bar",
    "Candidate",
    "Universe",
    "UniverseConfig",
    "build_candidates",
    "build_universe",
    "fetch_bars",
    "rank_candidates",
    "select_universe",
    "trading_days",
]

KOSDAQ = "코스닥"
KRX = "거래소"
UNSECTORED = "미분류"


@dataclass(frozen=True)
class UniverseConfig:
    """후보 선정 파라미터. 전부 인자로 조절할 수 있다."""

    limit: int = 45
    """최종 종목 수 상한. 키움 한도가 ~50 이라 안전 마진 5 를 뒀다."""

    markets: Sequence[str] = (KOSDAQ,)
    """대상 시장. 만쥬의 무대는 코스닥이다 — 거래소는 변동성 요건에서 대부분 탈락한다."""

    lookback_days: int = 20
    """변동성·급등 이력을 볼 영업일 수. 한 달이면 테마 한 사이클이 들어온다."""

    min_trade_value_eok: float = 100.0
    """전일 거래대금 하한(억원). 100억 ≈ 코스닥 상위 100종목(2026-08-25 기준).
    §2-1 ① '풍부한 거래대금'의 정량화이자 §4-1 호가 두께의 **대용**이다."""

    min_range_pct: float = 5.0
    """lookback 일중 등락폭 `(고−저)/종가` 의 **중앙값** 하한(%). §2-1 ② 의 정량화.
    만쥬는 '하루 10%~상한가'라 했지만 그건 테마가 터진 날의 값이고, 상시 중앙값에
    10% 를 요구하면 후보가 거의 남지 않는다. 중앙값 5% + 급등 이력(아래)으로 나눴다."""

    min_surge_days: int = 1
    """lookback 중 일간 등락 |close/prev_close − 1| ≥ `surge_pct` 인 날의 최소 개수.
    '이 종목은 뜰 때 10% 이상 뛴다'는 이력 확인이다."""

    surge_pct: float = 10.0

    min_price: int = 1000
    """종가 하한(원). 저가주는 1틱이 수십 bp라 우리 브래킷(+0.5%/−0.4%)이
    몇 틱 안 되어 노이즈 청산이 터진다(`pair_detector` 의 min_sl_ticks 와 같은 이유)."""

    min_per_sector: int = 2
    """섹터당 최소 종목 수. 대장 후보 + 후속 후보가 둘 다 있어야 짝꿍이 성립한다."""

    max_per_sector: int = 8
    """섹터당 상한. 코스닥은 '전기/전자'·'기계/장비'가 후보의 절반을 먹어서
    상한이 없으면 유니버스가 반도체 한 테마가 된다."""

    unsectored_slots: int = 0
    """`stocks.sector` 가 빈 종목(스팩·외국주·일부 신규상장)에 줄 슬롯.
    섹터를 모르면 짝꿍 상대를 고를 수 없어 기본 0 이다. 신규상장주를 꼭 넣어야 하면
    이 값을 올리거나 `pinned` 로 직접 박아라."""

    pinned: Sequence[str] = ()
    """무조건 포함할 코드(신규상장 당일 §4-5 등 사람이 아는 재료). 필터를 우회한다."""

    weight_volatility: float = 0.5
    """점수에서 변동성 항의 가중. 거래대금(log10)이 주항이고 변동성이 보조다 —
    §2-1 이 거래대금을 '가장 우선시되는 기준'이라 못 박았기 때문이다."""

    weight_limitup: float = 0.2
    """lookback 상한가 일수당 가산. 테마 대장 이력의 대용."""


@dataclass(frozen=True)
class Bar:
    """일봉 한 줄. DB 없이 테스트하려고 원시 튜플 대신 이걸 쓴다."""

    code: str
    day: date
    open: int
    high: int
    low: int
    close: int
    trade_value: int  # 백만원
    volume: int = 0
    name: str = ""
    sector: str = ""
    market: str = KOSDAQ


@dataclass(frozen=True)
class Candidate:
    code: str
    name: str
    sector: str
    market: str
    prev_close: int
    trade_value_eok: float
    range_pct: float          # lookback 일중 등락폭 중앙값(%)
    last_range_pct: float     # 전일 일중 등락폭(%)
    surge_days: int
    limitup_days: int
    avg_fill_won: float       # 거래대금/거래량 — 호가 두께 **대용**
    score: float = 0.0
    pinned: bool = False


@dataclass(frozen=True)
class Universe:
    """감지기에 주입할 그날의 유니버스."""

    asof: date                                   # 후보를 뽑은 기준일(직전 영업일)
    codes: list[str]
    sector_by_code: dict[str, str]
    prev_close_by_code: dict[str, int]
    candidates: list[Candidate] = field(default_factory=list)  # 선정된 것만, 점수순
    pool: list[Candidate] = field(default_factory=list)        # 필터 통과 전체(점수순)

    @property
    def sector_counts(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for code in self.codes:
            out[self.sector_by_code.get(code, UNSECTORED)] += 1
        return dict(out)

    @property
    def pairable_codes(self) -> list[str]:
        """짝꿍이 성립하는(같은 섹터 동료가 있는) 코드만."""
        counts = self.sector_counts
        return [c for c in self.codes if counts.get(self.sector_by_code.get(c, UNSECTORED), 0) >= 2]


# ------------------------------------------------------------------ 순수 로직

def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def build_candidates(
    bars: Iterable[Bar],
    asof: date,
    config: UniverseConfig | None = None,
) -> list[Candidate]:
    """일봉을 종목별로 접어 후보 지표를 만든다. 필터는 아직 걸지 않는다.

    `asof` 일 봉이 없는 종목(거래정지 등)은 버린다 — 전일 종가를 못 주면
    감지기가 상한가 계산을 못 한다.
    """
    cfg = config or UniverseConfig()
    by_code: dict[str, list[Bar]] = defaultdict(list)
    for bar in bars:
        by_code[bar.code].append(bar)

    out: list[Candidate] = []
    for code, rows in by_code.items():
        rows.sort(key=lambda b: b.day)
        last = rows[-1]
        if last.day != asof or last.close <= 0:
            continue
        # ::float 캐스팅 함정 — close 가 정수라 나눗셈 전에 반드시 float 로 올린다.
        ranges = [
            (b.high - b.low) / float(b.close) * 100.0
            for b in rows
            if b.close > 0
        ]
        surge = 0
        limitup = 0
        for prev, cur in pairwise(rows):
            if prev.close <= 0:
                continue
            ret = cur.close / float(prev.close) - 1.0
            if abs(ret) * 100.0 >= cfg.surge_pct:
                surge += 1
            # 상한가 30% 인데 호가단위 반올림으로 29.x% 로 찍히는 날이 있어 29% 로 본다.
            if ret >= 0.29:
                limitup += 1
        volume = sum(b.volume for b in rows) or 0
        tv_won = sum(b.trade_value for b in rows) * 1_000_000.0
        out.append(
            Candidate(
                code=code,
                name=last.name,
                sector=last.sector or UNSECTORED,
                market=last.market,
                prev_close=last.close,
                trade_value_eok=last.trade_value / 100.0,  # 백만원 → 억원
                range_pct=_median(ranges),
                last_range_pct=(last.high - last.low) / float(last.close) * 100.0,
                surge_days=surge,
                limitup_days=limitup,
                avg_fill_won=(tv_won / volume) if volume else 0.0,
            )
        )
    return out


def _score(cand: Candidate, cfg: UniverseConfig) -> float:
    """거래대금이 주항, 변동성·상한가 이력이 보조항(§2-1 의 우선순위 그대로)."""
    turnover_term = math.log10(max(cand.trade_value_eok, 1.0))
    vol_term = cfg.weight_volatility * (min(cand.range_pct, 30.0) / 10.0)
    limitup_term = cfg.weight_limitup * cand.limitup_days
    return turnover_term + vol_term + limitup_term


def rank_candidates(
    candidates: Iterable[Candidate],
    config: UniverseConfig | None = None,
) -> list[Candidate]:
    """필터를 통과한 후보에 점수를 매겨 내림차순으로 돌려준다.

    `pinned` 는 필터를 우회하되 점수는 그대로 계산한다(선정 단계에서 먼저 들어간다).
    """
    cfg = config or UniverseConfig()
    pinned = set(cfg.pinned)
    scored: list[Candidate] = []
    for cand in candidates:
        is_pinned = cand.code in pinned
        if not is_pinned:
            if cand.market not in cfg.markets:
                continue
            if cand.prev_close < cfg.min_price:
                continue
            if cand.trade_value_eok < cfg.min_trade_value_eok:
                continue
            if cand.range_pct < cfg.min_range_pct:
                continue
            if cand.surge_days < cfg.min_surge_days:
                continue
        scored.append(replace(cand, score=_score(cand, cfg), pinned=is_pinned))
    scored.sort(key=lambda c: (-c.score, c.code))
    return scored


def select_universe(
    ranked: Sequence[Candidate],
    asof: date,
    config: UniverseConfig | None = None,
) -> Universe:
    """섹터 다양성을 지키며 `limit` 개를 고른다.

    순서:
      1. `pinned` 를 먼저 넣는다(사람이 아는 재료가 규칙을 이긴다).
      2. **섹터 라운드** — 섹터를 강도순(상위 `min_per_sector` 종목 점수 합)으로 돌며
         각 섹터에 `min_per_sector` 개씩 준다. 후보가 `min_per_sector` 개에 못 미치는
         섹터는 건너뛴다 — 혼자 들어가 봐야 짝꿍 상대가 없어 슬롯만 먹는다.
      3. **채우기 라운드** — 남은 슬롯을 전체 점수순으로, 섹터당 `max_per_sector` 상한을
         지키며 메운다.
    """
    cfg = config or UniverseConfig()
    by_sector: dict[str, list[Candidate]] = defaultdict(list)
    for cand in ranked:
        by_sector[cand.sector].append(cand)

    chosen: list[Candidate] = []
    taken: set[str] = set()
    per_sector: dict[str, int] = defaultdict(int)

    def take(cand: Candidate) -> None:
        chosen.append(cand)
        taken.add(cand.code)
        per_sector[cand.sector] += 1

    for cand in ranked:
        if cand.pinned and len(chosen) < cfg.limit and cand.code not in taken:
            take(cand)

    # 섹터 강도 = 그 섹터 상위 min_per_sector 종목 점수 합. 대장+후속이 둘 다 센 섹터가
    # 앞에 온다(한 종목만 튀는 섹터는 짝꿍이 안 되므로 뒤로).
    def strength(sector: str) -> float:
        return sum(c.score for c in by_sector[sector][: cfg.min_per_sector])

    sectors = [
        s for s in by_sector
        if s != UNSECTORED and len(by_sector[s]) >= cfg.min_per_sector
    ]
    sectors.sort(key=lambda s: (-strength(s), s))

    for sector in sectors:
        if len(chosen) + cfg.min_per_sector > cfg.limit:
            continue  # 이 섹터엔 최소 인원을 못 넣는다 — 다음(더 작은) 섹터를 본다
        need = cfg.min_per_sector - per_sector[sector]
        for cand in by_sector[sector]:
            if need <= 0:
                break
            if cand.code in taken:
                continue
            take(cand)
            need -= 1

    for cand in ranked:
        if len(chosen) >= cfg.limit:
            break
        if cand.code in taken:
            continue
        if cand.sector == UNSECTORED:
            if per_sector[UNSECTORED] >= cfg.unsectored_slots:
                continue
            take(cand)
            continue
        if per_sector[cand.sector] >= cfg.max_per_sector:
            continue
        if per_sector[cand.sector] == 0:
            # 아직 한 종목도 없는 섹터는 **짝을 통째로** 넣는다. 혼자 넣으면 그 종목은
            # 짝꿍 상대가 없어 구독 슬롯만 먹는다(대장이 떠도 살 후속이 없다).
            partners = [
                c for c in by_sector[cand.sector]
                if c.code not in taken and c.code != cand.code
            ][: cfg.min_per_sector - 1]
            if len(partners) < cfg.min_per_sector - 1:
                continue
            if len(chosen) + 1 + len(partners) > cfg.limit:
                continue
            take(cand)
            for partner in partners:
                take(partner)
            continue
        take(cand)

    chosen.sort(key=lambda c: (-c.score, c.code))
    return Universe(
        asof=asof,
        codes=[c.code for c in chosen],
        sector_by_code={c.code: c.sector for c in chosen},
        prev_close_by_code={c.code: c.prev_close for c in chosen},
        candidates=chosen,
        pool=list(ranked),
    )


# --------------------------------------------------------------------- DB 읽기

def _ph(conn: Any) -> str:
    return "%s" if conn.__class__.__module__.startswith("psycopg2") else "?"


def trading_days(conn: Any, asof: date | str, count: int) -> list[date]:
    """`asof` 이하 최근 `count` 영업일. 오래된 순."""
    ph = _ph(conn)
    sql = f"select distinct date from daily_bars where date <= {ph} order by date desc limit {ph}"
    cur = conn.cursor()
    cur.execute(sql, (asof, count))
    rows = [r[0] for r in cur.fetchall()]
    cur.close()
    return sorted(rows)


def fetch_bars(
    conn: Any,
    asof: date | str,
    config: UniverseConfig | None = None,
) -> list[Bar]:
    """`asof` 기준 lookback 구간의 일봉을 읽는다(원주가, 분할 미반영).

    `pinned` 종목은 시장 필터를 우회해야 하므로 SQL 의 market 조건에 OR 로 붙인다.
    """
    cfg = config or UniverseConfig()
    days = trading_days(conn, asof, cfg.lookback_days)
    if not days:
        return []
    ph = _ph(conn)
    markets = list(cfg.markets)
    market_clause = " or ".join([f"s.market = {ph}"] * len(markets)) or "false"
    params: list[Any] = [*markets]
    pinned = list(cfg.pinned)
    if pinned:
        market_clause = (
            f"({market_clause}) or d.code in ({', '.join([ph] * len(pinned))})"
        )
        params.extend(pinned)
    sql = (
        "select d.code, d.date, d.open, d.high, d.low, d.close, d.trade_value, d.volume, "
        "s.name, s.sector, s.market "
        "from daily_bars d join stocks s on s.code = d.code "
        f"where ({market_clause}) and d.date >= {ph} and d.date <= {ph} "
        "order by d.code, d.date"
    )
    params.extend([days[0], days[-1]])
    cur = conn.cursor()
    cur.execute(sql, tuple(params))
    rows = cur.fetchall()
    cur.close()
    return [
        Bar(
            code=str(r[0]),
            # sqlite 는 date 를 문자열로 돌려준다 — 여기서 한 번에 date 로 맞춘다.
            day=r[1] if isinstance(r[1], date) else date.fromisoformat(str(r[1])),
            open=int(r[2] or 0),
            high=int(r[3] or 0),
            low=int(r[4] or 0),
            close=int(r[5] or 0),
            trade_value=int(r[6] or 0),
            volume=int(r[7] or 0),
            name=str(r[8] or ""),
            sector=str(r[9] or ""),
            market=str(r[10] or ""),
        )
        for r in rows
    ]


def build_universe(
    conn: Any,
    asof: date | str,
    config: UniverseConfig | None = None,
) -> Universe:
    """DB → 후보 → 랭킹 → 선정. 장전에 이 함수 하나만 부르면 된다.

    `asof` 는 **직전 영업일**이다(당일 일봉은 장이 끝나야 생긴다).
    """
    cfg = config or UniverseConfig()
    bars = fetch_bars(conn, asof, cfg)
    asof_date = asof if isinstance(asof, date) else date.fromisoformat(str(asof))
    if bars:
        asof_date = max(b.day for b in bars)
    cands = build_candidates(bars, asof_date, cfg)
    return select_universe(rank_candidates(cands, cfg), asof_date, cfg)
