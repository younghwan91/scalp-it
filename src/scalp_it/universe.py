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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
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

    #: lookback 상한가 일수당 가산. 테마 대장 이력의 대용.
    weight_limitup: float = 0.2
    #: 재료 가중치(만쥬 §2-1 3대 요건의 세 번째). 거래대금·변동성만으로 고르면
    #: "수급은 있는데 이야기가 없는" 종목이 올라온다 — 만쥬는 그걸 주도주로 안 본다.
    #: 재료는 DART 공시에서 온다(뉴스 크롤러 금지 정책, docs/research/manju/17).
    weight_material: float = 0.5


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


def _score(cand: Candidate, cfg: UniverseConfig, *, material: float = 0.0) -> float:
    """거래대금이 주항, 변동성·상한가 이력·재료가 보조항(§2-1 의 우선순위 그대로)."""
    turnover_term = math.log10(max(cand.trade_value_eok, 1.0))
    vol_term = cfg.weight_volatility * (min(cand.range_pct, 30.0) / 10.0)
    limitup_term = cfg.weight_limitup * cand.limitup_days
    material_term = cfg.weight_material * material
    return turnover_term + vol_term + limitup_term + material_term


def rank_candidates(
    candidates: Iterable[Candidate],
    config: UniverseConfig | None = None,
    materials: Mapping[str, Sequence[str]] | None = None,
    material_scores: Mapping[str, float] | None = None,
) -> list[Candidate]:
    """필터를 통과한 후보에 점수를 매겨 내림차순으로 돌려준다.

    `pinned` 는 필터를 우회하되 점수는 그대로 계산한다(선정 단계에서 먼저 들어간다).
    `materials` 는 종목별 최근 재료 유형 목록(만쥬 §2-1). 안 주면 재료항은 0 이다 —
    있으면 쓰고 없으면 기존대로 도는 게 규약이다(재료 DB 가 없어도 기동해야 한다).
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
        # LLM 이 매긴 점수가 있으면 그게 우선한다 — 유형 가중치표보다 판단이 세다
        # (만쥬 §2-1 "미래 가치"·"지속시킬 스토리"는 표로 표현되지 않는다).
        if material_scores and cand.code in material_scores:
            mat = float(material_scores[cand.code])
        else:
            mat = material_score((materials or {}).get(cand.code, ()))
        scored.append(replace(cand, score=_score(cand, cfg, material=mat),
                              pinned=is_pinned))
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
    materials: Mapping[str, Sequence[str]] | None = None,
    material_scores: Mapping[str, float] | None = None,
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
    ranked = rank_candidates(cands, cfg, materials, material_scores)
    return select_universe(ranked, asof_date, cfg)

# ---------------------------------------------- 만쥬 §4-5 신규상장 · §5-5 신선도

def pick_recent_listings(first_day_by_code: Mapping[str, date], *, asof: date,
                         within_days: int) -> list[str]:
    """상장일이 ``asof`` 로부터 ``within_days`` 일 이내인 종목(만쥬 §4-5).

    > "상장 당일은 풍부한 거래대금과 높은 변동성을 보이므로 **무조건** 관심종목에
    > 등록 후 매매 기회를 포착한다"

    신규 상장주는 과거 일봉이 없어 거래대금·변동성 필터를 통과할 수가 없다. 그래서
    일반 랭킹으로는 절대 안 뽑힌다 — ``UniverseConfig.pinned`` 로 박아야 한다.
    ``within_days`` 는 **달력일**이다(영업일 아님) — 상장 직후 며칠만 보므로
    둘의 차이가 실질적이지 않고, 영업일로 세려면 DB 를 알아야 해서 규약이 깨진다.
    ``within_days=0`` 이면 끈다(빈 리스트).
    """
    if within_days <= 0:
        return []
    cutoff = asof - timedelta(days=int(within_days))
    return sorted(code for code, day in first_day_by_code.items()
                  if day is not None and cutoff <= day <= asof)


def stale_sectors(limitup_counts: Mapping[str, int], *, max_limitups: int) -> set[str]:
    """최근 상한가가 ``max_limitups`` 를 넘긴 섹터 = 소비된 테마(만쥬 §5-5).

    > "이미 수차례 소비되어 **신선도가 떨어진 테마**는 대장이 상을 가도 후발주가
    > 잘 따라가지 못하므로 주의해야 한다"

    신선도는 이 문서에서 가장 자동화하기 어려운 항목이라(등급 C) 정확한 재현이
    아니라 **대용치**다 — 그 섹터에서 최근 상한가가 몇 번 나왔는지로 '소비 정도'를
    근사한다. 섹터(27종 산업분류)가 당일 테마와 다르다는 한계도 그대로 있다.
    ``max_limitups=0`` 이면 끈다.
    """
    if max_limitups <= 0:
        return set()
    return {sec for sec, n in limitup_counts.items() if n > max_limitups}


def fetch_first_days(conn: Any, codes: Sequence[str] | None = None) -> dict[str, date]:
    """종목별 일봉 최초 등장일 = 상장일 근사. 상장 피드가 없어 일봉으로 대신한다."""
    ph = _ph(conn)
    sql = "select code, min(date) from daily_bars"
    params: list[Any] = []
    if codes:
        sql += " where code in (" + ",".join([ph] * len(codes)) + ")"
        params = list(codes)
    sql += " group by 1"
    cur = conn.cursor()
    cur.execute(sql, params)
    out: dict[str, date] = {}
    for code, day in cur.fetchall():
        if day is None:
            continue
        out[str(code)] = day if isinstance(day, date) else date.fromisoformat(str(day))
    return out


def fetch_sector_limitups(conn: Any, asof: date | str, *, days: int = 10,
                          market: str = KOSDAQ) -> dict[str, int]:
    """최근 ``days`` 영업일 섹터별 상한가 발생 횟수(만쥬 §5-5 신선도 대용치).

    상한가 근사는 전일종가 대비 +29.5% 이상. ``daily_bars.close`` 가 **정수형**이라
    ``::float`` 캐스팅이 없으면 정수 나눗셈이 되어 거의 안 걸린다 — 실제로 그 함정에
    빠져 1년 2,313건을 5건으로 셌던 적이 있다.
    """
    sessions = trading_days(conn, asof, days + 1)
    if len(sessions) < 2:
        return {}
    ph = _ph(conn)
    cur = conn.cursor()
    cur.execute(
        f"""
        with d as (
          select b.code, b.date, cast(b.close as float) c,
                 lag(cast(b.close as float)) over (partition by b.code order by b.date) pc
          from daily_bars b join stocks s on s.code = b.code
          where s.market = {ph} and b.date >= {ph} and b.date <= {ph}
        )
        select s.sector, count(*)
        from d join stocks s on s.code = d.code
        where d.pc > 0 and d.c / d.pc >= 1.295 and d.date >= {ph}
        group by 1
        """,
        [market, sessions[0], sessions[-1], sessions[1]],
    )
    return {str(sec or UNSECTORED): int(n) for sec, n in cur.fetchall()}

# ------------------------------------------------ 만쥬 §2-1 재료 — DART 공시 기반

#: 재료 유형별 무게. 만쥬 원문(§2-1)은 "단순 수급뿐 아니라 **시장의 관심을 지속시킬
#: 강력한 스토리**"를 요구한다 — 공급계약·기술 인허가·시설투자는 미래 실적으로 이어지는
#: 이야기가 되고, 정기보고·행정은 그런 이야기가 아니다. 악재는 반대 방향이라 감점이다.
#:
#: 뉴스가 아니라 공시를 쓰는 이유: `docs/research/manju/17` 이 "1년치 종목 뉴스를
#: 프로그램으로 합법 확보할 경로는 없다"고 결론짓고 **크롤러를 새로 짜지 않는다**는
#: 정책을 세웠다(ToS·저작권). 같은 문서가 지목한 현실 경로가 DART 공시 원문이다.
#: 공시는 재료의 1차 소스이고 제목이 법정 양식이라 규칙만으로 분류된다(LLM 불필요).
MATERIAL_WEIGHTS: dict[str, float] = {
    "공급계약": 1.0,        # 수주 = 가장 직접적인 미래 실적
    "기술_인허가": 1.0,     # 특허·임상·품목허가 — 스토리가 길게 간다
    "시설투자": 0.7,
    "실적": 0.6,
    "구조조정": 0.4,
    "재량공시": 0.3,
    "주주환원": 0.3,
    "자본조달": 0.1,        # 증자는 재료이기도 악재이기도 하다 — 거의 중립
    "지분변동": 0.1,
    "내부자": 0.0,
    "정기보고": 0.0,
    "행정": 0.0,
    "기타": 0.0,
    "악재": -1.0,           # 감점. 재료가 아니라 반대다
}

#: 한 종목이 공시 여러 건으로 목록을 독식하지 못하게 하는 상한(가중치 합의 배수).
_MATERIAL_CAP = 3.0


def material_score(material_types: Sequence[str]) -> float:
    """재료 유형 목록 → 점수. 강한 재료는 쌓이되 상한이 있다."""
    if not material_types:
        return 0.0
    total = sum(MATERIAL_WEIGHTS.get(str(t), 0.0) for t in material_types)
    best = max((MATERIAL_WEIGHTS.get(str(t), 0.0) for t in material_types), default=0.0)
    if best <= 0:
        return total          # 악재만 있으면 그대로 감점
    return min(total, best * _MATERIAL_CAP)


def group_materials(rows: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    """``(코드, 재료유형)`` 행들을 종목별 목록으로 접는다."""
    out: dict[str, list[str]] = defaultdict(list)
    for code, mtype in rows:
        if code and mtype:
            out[str(code)].append(str(mtype))
    return dict(out)


def fetch_recent_materials(db_path: str, *, since: date | str,
                           until: date | str | None = None) -> dict[str, list[str]]:
    """DART sqlite 에서 최근 공시를 종목별 재료 유형 목록으로.

    `rcept_dt` 는 ``YYYYMMDD`` 문자열이다. 파일이 없거나 읽기 실패면 빈 dict —
    재료를 못 읽는다고 기동을 막지는 않는다(만쥬 3대 요건 중 하나가 빠질 뿐이다).
    """
    import sqlite3

    p = Path(db_path)
    if not p.exists():
        return {}
    lo = (since if isinstance(since, str) else since.isoformat()).replace("-", "")
    hi = until or date.today()  # noqa: DTZ011 — KST naive, 프로젝트 표준
    hi = (hi if isinstance(hi, str) else hi.isoformat()).replace("-", "")
    try:
        with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "select code, material_type from disclosures "
                "where rcept_dt >= ? and rcept_dt <= ? and code is not null",
                (lo, hi),
            ).fetchall()
    except sqlite3.Error:
        return {}
    return group_materials(rows)


def breaking_materials(rows: Iterable[tuple[str, str]], *,
                       min_weight: float = 0.7) -> list[str]:
    """장중 신규 공시 중 **강한 재료** 종목만, 강한 순으로(만쥬 §6).

    > "예상치 못한 돌발 뉴스(전쟁 개전 속보 등)가 나올 경우 즉각적인 판단으로 관련
    > 종목 매수 대응" [02:29]

    다만 공시는 하루 수백 건이 쏟아진다. 전부 좇으면 구독 자리(단일 앱키 50종목)가
    순식간에 동난다. 그래서 미래 실적으로 이어지는 유형만 남긴다 — 기본 0.7 이면
    공급계약·기술_인허가·시설투자다. 악재는 당연히 제외(매수 대상이 아니다).
    """
    best: dict[str, float] = {}
    for code, mtype in rows:
        if not code or not mtype:
            continue
        w = MATERIAL_WEIGHTS.get(str(mtype), 0.0)
        if w >= min_weight:
            best[str(code)] = max(best.get(str(code), 0.0), w)
    return [c for c, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))]


def fetch_new_disclosures(db_path: str, *, day: date | str,
                          after_rcept: str | None = None) -> list[tuple[str, str, str]]:
    """그날 공시 중 ``after_rcept`` 이후 접수분 → ``(접수번호, 코드, 재료유형)``.

    접수번호는 시간순으로 증가하므로 커서로 쓴다. 장중에 주기적으로 불러 **새로 들어온
    것만** 가져오는 용도다(만쥬 §6). 파일이 없거나 읽기 실패면 빈 리스트 — 재료를
    못 읽는다고 감지기를 죽이지는 않는다.
    """
    import sqlite3

    p = Path(db_path)
    if not p.exists():
        return []
    d = (day if isinstance(day, str) else day.isoformat()).replace("-", "")
    sql = ("select rcept_no, code, material_type from disclosures "
           "where rcept_dt = ? and code is not null")
    params: list[Any] = [d]
    if after_rcept:
        sql += " and rcept_no > ?"
        params.append(after_rcept)
    sql += " order by rcept_no"
    try:
        with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as conn:
            return [(str(a), str(b), str(c)) for a, b, c in
                    conn.execute(sql, params).fetchall()]
    except sqlite3.Error:
        return []


def ipo_day_avoid(listing_sectors: Mapping[str, str],
                  candidates: Mapping[str, Mapping[str, Any]],
                  *, min_pre_run_pct: float = 15.0,
                  falling_pct: float = 0.0) -> set[str]:
    """상장 당일 '재료 소멸' 관련주로 의심되는 종목(만쥬 §4-5).

    > "대형 공모주 **상장 전**에는 관련주/테마주를 공략하되, **상장 당일**에는 기존
    > 관련주가 **재료 소멸로 하락**할 가능성이 높으므로 상장 당사자에만 집중"

    **'관련주' 식별은 정확히 못 한다** — 그건 재료 판단(등급 C)이다. 그래서 섹터만으로
    거르지 않는다. 전기/전자에 신규상장이 하나 떴다고 그 섹터 전체가 그 공모주의
    관련주는 아니고, 그렇게 거르면 멀쩡한 종목을 배제해 오히려 해롭다.

    세 조건을 **AND** 로 건다:
      ① 그날 신규상장 종목과 같은 섹터
      ② 상장 전 구간에 ``min_pre_run_pct`` 이상 올랐다 — 기대감으로 오른 흔적
      ③ 지금 ``falling_pct`` 아래로 빠지고 있다 — 재료 소멸이 실제로 일어나는 중

    ②③ 이 함께 있어야 "기대감에 올랐다가 상장으로 재료가 소멸했다"는 이야기가 성립한다.
    그래도 근사다 — 기본은 **꺼 두고**(호출부에서 옵트인) 켤 때 이 한계를 알고 켠다.
    """
    if not listing_sectors:
        return set()
    sectors = {str(v) for v in listing_sectors.values() if v}
    if not sectors:
        return set()
    out: set[str] = set()
    for code, info in candidates.items():
        if str(code) in listing_sectors:
            continue                       # 상장 당사자는 오히려 집중 대상이다
        if str(info.get("sector") or "") not in sectors:
            continue
        try:
            pre = float(info.get("pre_run_pct") or 0.0)
            today = float(info.get("today_pct") or 0.0)
        except (TypeError, ValueError):
            continue
        if pre >= min_pre_run_pct and today < falling_pct:
            out.add(str(code))
    return out
