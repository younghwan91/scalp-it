"""짝꿍(pair) 실시간 감지 로직 — 순수 계산부. I/O 는 CLI 가 담당한다.

규칙(`docs/research/manju/37`, 재검 반영):

* **등록한 쌍만 감시.** 대장 서지마다 전 종목에서 후속주를 기계로 고르는 "자동 매칭"은
  꼭지 눌린 강한 종목을 사게 되어 마이너스(−0.11~−0.21%)로 확인됐다. 엣지는 **대장 →
  알려진 소형 후속**이라는 특정 관계에서만 살아있고, 그 쌍은 사람이 큐레이션한다(만쥬 방식).
  감지기는 :class:`Pair` 목록(`data/pair_watchlist.json` 또는 CLI ``--pairs``)만 본다.

* **진입 신호**: 대장주가 직전 ``surge_window`` 초(기본 30s)에 ``surge_ret``(+1.5%) 이상
  급등(서지)한 순간, 같은 시각 후속주가 **동반 급등**(직전 30s ≥ ``cosurge_min`` +0.5%,
  대장과 같은 방향)이면서 **아직 안 뛴 것**(그 직전 30s 프리무브 ≤ ``premove_max`` +0.2%)
  이면 → 후속주 **bid1 지정가 매수** 알림. 프리무브 필터가 doc37 의 결정적 엣지다(추격이
  아니라 선행을 산다). 여기에 **매수체결비율 필터**(doc39)를 AND 로 더한다: 후속주
  직전 ``buy_ratio_window`` 초의 매수체결비율(side=+1 체결량/전체 체결량)이
  ``min_buy_ratio``(기본 0.30) 미만이면 기각 — 실제 매수로 따라붙는 이벤트만 사서
  가격이 내 지정가로 흘러내린 약한 역선택 fill 을 배제한다. ``min_buy_ratio=0`` 이면 끈다.

* **청산(알림에 표시만, 주문은 사람이)** — 두 모드, 기본은 안정형:
  안정형 익절 +0.5% / 일상 손절 −0.4% / 90초 시간청산(세후 +0.19%·승률 71%·MDD −0.74%),
  공격형 익절 +0.8% / 일상 손절 없음 + 재난선 −0.7% / 90초(세후 +0.27%·MDD −1.27%).
  두 모드 공통: bid1 지정가 진입 · 90초 시간청산 · 트레일 없음. 값은 :class:`DetectorConfig`.

* **호가단위(틱) 필터**: 브래킷이 비율 고정(+0.5%/−0.4%)이라 저가주는 손절폭이 몇 틱
  안 된다(539원 = 1틱 0.186% → 손절 2.2틱). 그러면 스프레드 왕복 한 번, 호가 한두 칸
  흔들림만으로 손절이 터진다 — 신호와 무관한 노이즈 청산이다. 손절폭(없으면 재난선,
  그것도 없으면 익절폭)이 ``min_sl_ticks`` 틱 미만인 후속주는 알림하지 않고
  :class:`FilteredSignal` 로 남긴다. ``min_sl_ticks=0`` 이면 끈다(과거 동작).

* **상한가 방아쇠**(문서 44 §5-1, 기본 꺼짐): 만쥬의 실제 방아쇠는 서지가 아니라
  "대장이 상한가에 안착하거나 상한가 직전 급등"이다. 원자료가 터치인지 잠김인지
  확정되지 않아(§11 #4) ``limitup_touch``/``limitup_near_pct``/``limitup_lock_secs``
  셋을 각각 켤 수 있게 두고 OR 로 판정한다. 상한가는 주입받은 **전일 종가**로만
  계산하고(:func:`limit_up_price`), 전일 종가가 없는 종목은 방아쇠가 없다(안전 스킵).

* **동적 대장/후속**(문서 44 §3·§4-2, 기본 꺼짐): ``universe`` + ``sectors`` 를 주면
  방아쇠가 터진 종목을 그날의 대장으로 삼고, 같은 섹터에서 그 시점 당일등락률
  ``follower_rank_min~max`` 위(기본 2~3)를 후속으로 고른다. 만쥬는 쌍을 매일 새로
  고르기 때문이다. 사람이 큐레이션한 고정 쌍 모드는 그대로 함께 돈다.

* **호가 두께 필터**(문서 44 §4-1 — 만쥬가 **최우선**이라 한 필터): ``order_krw`` 원을
  시장가로 왕복 칠 때의 충격(:func:`roundtrip_bp`)이 ``max_roundtrip_bp`` (기본 20bp)를
  넘거나 10호가로 금액을 못 채우면 배제한다. 실측(2026-08-24, 80종목)에서 1000만원
  왕복충격이 종목별로 39배(5.1~195.9bp) 벌어졌고 두께 5분위 충격후 손익이
  Q1 −22bp vs Q5 −219bp 였다. 10호가 스냅샷을 안 주면 잴 수 없으므로 통과시킨다.
  ``min_sl_ticks`` 와는 성격이 다른 필터라 둘 다 남아 있고, **두께를 먼저** 본다.

* **순간 체결량**(문서 44 §5-1, 기본 꺼짐): 대장에 ``big_trade_krw``(5천만원) 이상
  체결이 ``big_trade_window`` 초 안에 ``min_big_trades`` 건 이상 있어야 통과.

* **대장 이탈 통보**(문서 44 §5-3 — 짝꿍의 진짜 청산 규칙): "대장이 상한가를 못 말거나
  풀리는 순간 손익 무관 전량 매도". 상한가를 계산할 수 있는 건 전일 종가를 주입받은
  이 감지기뿐이라 **판정은 여기서 내리고** :meth:`PairDetector.drain_leader_events` 로
  ``(대장코드, 시각, holding|lost)`` 를 내보낸다(가드·실행기는 결론만 받는다).
  상태가 **바뀔 때만** 한 번씩 넣고, 알림 반환값에는 절대 섞지 않는다.
  이탈 **사유**(:data:`LOST_UNLOCK`/:data:`LOST_DUMP`/:data:`LOST_WALL`)는 상태에 섞지
  않고 :class:`LeaderEvent` 의 ``.reason`` 으로 나른다 — 상태 문자열을 늘리면 가드가
  ``state != LEADER_LOST`` 로 걸러 **청산이 조용히 안 도는** 사고가 난다.

* **상한가 매수벽**(문서 44 §5-2, 기본 꺼짐): "수십만~수백만 주를 **안정적으로 굳히면**".
  가격만으로는 굳힘을 못 잰다 — 상한가 호가의 매수 잔량이 ``lock_min_qty``(주)·
  ``lock_min_krw``(원) 문턱을 넘어야 잠김·터치 방아쇠를 인정한다. 그 벽이 최근
  ``lock_wall_window`` 초 최대치의 ``lock_wall_decay`` 배 아래로 깎이면, 가격이 아직
  상한가여도 이탈(:data:`LOST_WALL`)로 통보한다 — 풀린 뒤에 파는 건 한 박자 늦다.

* **매도 폭탄**(문서 44 §5-4, 기본 꺼짐): "프로그램 매도 폭탄(투매)이 쏟아져 막히면
  지체 없이 손절". 대장의 최근 ``dump_window`` 초 매도 체결대금이 ``dump_krw`` 이상이면
  가격과 **무관하게** 이탈(:data:`LOST_DUMP`) — 만쥬는 "풀리거나, **또는** 막히면"이라고
  or 로 말한다.

* **테마 신선도**(문서 44 §5-5, 기본 꺼짐): 소비된 테마는 대장이 상을 가도 후발주가 안
  따라간다. 대용치로 **"그 섹터가 최근 N일 내 낸 상한가 건수"**를 주입받아
  (``sector_limitup_counts``) ``max_sector_limitups`` 를 넘으면 알림을 배제하고
  ``FilteredSignal(sector_stale)`` 로 남긴다. 섹터·집계를 모르면 막지 않는다(등급 C).

* **쿨다운**: 같은 쌍은 ``cooldown`` 초(기본 90s) 안에 재알림하지 않는다.
* **거래대금 하한**: 후속주 당일 누적 체결대금이 ``min_turnover`` 미만이면 알림하지 않는다.

이 모듈은 DB·시계를 모르고, 오름차순으로 들어오는 틱을 :meth:`PairDetector.on_tick` 으로
먹고 알림 리스트를 돌려줄 뿐이라 mock 틱으로 단위 테스트한다.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Self

from scalp_it.ticksize import tick_size, ticks_in

__all__ = [
    "DEFAULT_WATCHLIST",
    "LEADER_HOLDING",
    "LEADER_LOST",
    "LOST_DUMP",
    "LOST_UNLOCK",
    "LOST_WALL",
    "Alert",
    "DetectorConfig",
    "FilteredSignal",
    "LeaderEvent",
    "Pair",
    "PairDetector",
    "limit_up_price",
    "load_watchlist",
    "parse_pairs_arg",
    "roundtrip_bp",
]


#: 호가 한 단계 = (가격, 잔량).
Level = tuple[float, float]

#: 방아쇠 코드 → 사람 말. **여기가 유일한 어휘 출처다.**
#: 2026-08-26 까지 `journal._TRIGGER_STORY` 가 한국어 키("상한가터치")로 적혀 있었는데
#: `Alert.trigger` 에 실제로 들어가는 값은 영문 코드라 매번 miss 였다 — 상한가로 잡은
#: 알림이 전부 "대장 서지"로 일지에 남았다. 복기의 출발점이 통째로 틀리는 버그다.
TRIGGER_NAMES: dict[str, str] = {
    "surge": "서지",
    "touch": "상한가터치",
    "near": "상한가근접",
    "locked": "상한가잠김",
    "vi": "VI근접",
}

#: 대장 상태 — :meth:`PairDetector.drain_leader_events` 가 내보내는 값.
#: ``scalp_it.realtime.risk_guard`` 의 동명 상수와 **글자까지 같아야** 한다(테스트로 고정).
#: 여기서 import 하지 않는 이유: 감지기는 순수 계산부라 가드·실행기에 의존하지 않는다.
LEADER_HOLDING = "holding"   # 상한가를 말고 있음(유지 중)
LEADER_LOST = "lost"         # 상한가를 못 말거나 풀리고 꺾임 = 매수 이유 상실

#: 이탈 **사유** — 만쥬 §5-4 는 "풀리거나, **또는** 매도 폭탄에 막히면"이라고 or 로 말한다.
#: 사유는 상태(``lost``)와 **별개 축**이라 큐에 넣는 문자열을 늘리지 않는다. 늘리면
#: ``risk_guard``/``order_executor`` 가 ``state != LEADER_LOST`` 로 걸러버려 **청산이
#: 조용히 안 도는** 사고가 난다. 그래서 :class:`LeaderEvent` 의 곁다리 속성으로 나른다.
LOST_UNLOCK = "unlock"       # 가격이 상한가 아래로 — "상한가를 말지 못하고 풀림"
LOST_DUMP = "dump"           # 대량 매도 체결이 쏟아짐 — "프로그램 매도 폭탄에 막힘"
LOST_WALL = "wall"           # 상한가 매수벽이 급격히 깎임 — 풀리기 직전의 예고


class LeaderEvent(tuple):
    """``(대장코드, 시각, 상태)`` **3-튜플 그대로**. 사유는 ``.reason`` 으로 붙는다.

    기존 호출부(``for code, ts, state in det.drain_leader_events()``)와 기존 테스트의
    ``ev == [(code, ts, "lost")]`` 비교를 한 글자도 안 깨면서 "왜 이탈인지"를 실어
    나르려고 튜플을 상속했다. 로그·복기는 ``.reason`` 을 읽으면 되고, 가드·실행기는
    지금처럼 3-튜플만 봐도 정상 동작한다.
    """

    def __new__(cls, code: str, ts: datetime, state: str, reason: str = "") -> Self:
        return super().__new__(cls, (code, ts, state))

    def __init__(self, code: str, ts: datetime, state: str, reason: str = "") -> None:
        self.reason = reason


def limit_up_price(prev_close: float) -> float:
    """전일종가 × 1.3 을 그 가격대 호가단위로 **내림** = 상한가.

    ``scripts/limitup_timing.py::limit_up_price`` 와 값이 한 치도 달라지면 안 된다 —
    그 스크립트로 뽑은 상한가 검증 결과와 실시간 판정이 어긋나면 백테스트가 무의미해진다.
    스크립트는 pandas·DB 어댑터를 끌고 들어와 실시간 감지기에서 import 할 수 없으므로
    식만 여기 옮겼고, **동일성은 테스트가 스크립트를 직접 로드해 고정한다**.
    """
    raw = float(prev_close) * 1.3
    tick = tick_size(raw)
    return math.floor(raw / tick + 1e-9) * tick


def _sweep_vwap(levels: Sequence[Level], krw: float) -> float | None:
    """``krw`` 원을 호가 단계별로 순서대로 소진했을 때의 체결 VWAP.

    잔량을 다 긁어도 금액을 못 채우면 None — 그 종목은 시장가로 치면 호가가 튄다
    (만쥬 §4-1: "상하 10호가를 합쳐도 소화 못 하면 재료가 좋아도 즉시 제외").
    """
    if krw <= 0:
        return None
    remain, shares = krw, 0.0
    for px, qty in levels:
        px, qty = float(px), float(qty)
        if px <= 0 or qty <= 0:
            continue
        take = min(px * qty, remain)
        shares += take / px
        remain -= take
        if remain <= 0:
            break
    if remain > 0 or shares <= 0:
        return None
    return krw / shares


def roundtrip_bp(bids: Sequence[Level], asks: Sequence[Level], krw: float) -> float | None:
    """``krw`` 원 시장가 **왕복** 충격(bp). 10호가로 못 채우면 ``inf``, 호가가 없으면 None.

    매수는 ask1→ask10, 매도는 bid1→bid10 을 순서대로 소진한 VWAP 을 쓰고
    ``(VWAP_buy − VWAP_sell) / mid × 10⁴`` 로 잰다. 실측(2026-08-24, 80종목)에서
    1000만원 기준 중앙 왕복충격이 5.1bp(LG전자)~195.9bp(인디에프)로 39배 벌어졌고,
    두께 5분위 충격후 손익이 Q1 −22bp vs Q5 −219bp 였다 — 두께가 곧 손익이다.
    """
    if not bids or not asks:
        return None
    mid = (float(bids[0][0]) + float(asks[0][0])) / 2.0
    if mid <= 0:
        return None
    buy = _sweep_vwap(asks, krw)
    sell = _sweep_vwap(bids, krw)
    if buy is None or sell is None:
        return math.inf                   # 10호가로도 못 채움 = 호가 튐 → 즉시 배제
    return (buy - sell) / mid * 1e4


# --------------------------------------------------------------------- 데이터

@dataclass(frozen=True)
class Pair:
    """감시할 대장→후속 관계. 코드는 6자리 문자열."""

    leader: str
    follower: str
    leader_name: str = ""
    follower_name: str = ""


@dataclass(frozen=True)
class DetectorConfig:
    """감지·청산 파라미터. 모두 CLI 로 덮어쓸 수 있다."""

    surge_ret: float = 0.015          # 대장 서지 임계 (직전 surge_window 초 수익)
    surge_window: float = 30.0        # 대장 서지 관측 창(초)
    cosurge_min: float = 0.005        # 후속 동반급등 하한 (직전 cosurge_window 초 수익)
    cosurge_window: float = 30.0      # 후속 동반급등 관측 창(초)
    premove_max: float = 0.002        # 후속 '아직 안 뜀' 상한 (그 직전 premove_window 초 수익)
    premove_window: float = 30.0      # 후속 프리무브 관측 창(초)
    # 역선택 방어(doc39): 후속주가 실제 매수체결로 따라붙을 때만 진입한다.
    #   진입 순간 직전 buy_ratio_window 초의 매수체결비율(side=+1 체결량/전체 체결량)이
    #   min_buy_ratio 미만이면 기각 — 가격이 내 지정가로 흘러내린 약한(역선택) fill 배제.
    #   through(보수) net +0.05%→+0.31%·승률 88%. 0 = 필터 끔(기존 동작).
    #   창에 체결 데이터가 없으면(워밍업) 안전 스킵(진입 안 함).
    min_buy_ratio: float = 0.30       # 후속 직전30초 매수체결비율 하한. 0 = 끔
    buy_ratio_window: float = 30.0    # 매수비율 관측 창(초)
    # 청산 — 두 모드를 한 config 로 표현한다. 기본값 = 안정형.
    #   안정형(기본): 익절 +0.5% / 일상 손절 −0.4% / 재난선 없음 / 90초.
    #                세후 +0.19%·승률 71%·MDD −0.74%. 소액·심리 편함용.
    #   공격형(옵션): 익절 +0.8% / 일상 손절 없음 / 재난선 −0.7% / 90초.
    #                세후 +0.27%·MDD −1.27%. (bid1 눌림매수라 역행이 얕아 타이트 손절이
    #                스프레드 노이즈에 튕기므로 일상 손절을 빼고 폭탄방지선만 둔다.)
    take_profit: float | None = 0.005  # 익절. None/0 = 상한 없음(수익을 열어둠)
    stop_loss: float | None = -0.004  # 일상 손절 (안정형 −0.3~0.4%). None = 없음(공격형)
    disaster_line: float | None = None  # 재난 방지선 (공격형 −0.7%). None = 없음(안정형)
    time_exit: float = 90.0           # 시간청산(초) — 두 모드 공통, 실질적 손절 역할
    # 호가단위 방어: 손절폭이 몇 틱 안 되면 스프레드·호가 흔들림만으로 손절이 터진다.
    #   손절폭(없으면 재난선, 그것도 없으면 익절폭) < min_sl_ticks 틱이면 알림 안 함.
    #   3.0 = 진입 1틱 + 왕복 스프레드 1틱 + 여유 1틱. 0 = 필터 끔(과거 동작).
    min_sl_ticks: float = 3.0         # 손절폭 최소 호가단위 수. 0 = 끔
    # 가속도 방어 — 프로그램매매성 급등만 고른다. 둘 다 0 = 끔(기존 동작).
    #   같은 +1.5% 라도 30초에 걸쳐 야금야금 오른 것과 마지막 3초에 튄 것은 다르다.
    #   전자는 감지 시점에 이미 짝꿍이 반응을 끝냈을 공산이 크다(진입이 늦다).
    #   accel_min_ratio: 대장 서지창의 뒤 절반 수익률 / 앞 절반 수익률. 1보다 크면
    #     "뒤로 갈수록 빨라지는" 상승만 통과. 앞 절반이 0 이하인데 뒤가 양이면 통과.
    #   volume_burst_ratio: 대장의 최근 volume_burst_window 초 체결량률 ÷ 그 직전
    #     volume_burst_base 초 체결량률. 프로그램매매는 가격보다 체결량에 먼저 찍힌다.
    accel_min_ratio: float = 0.0      # 가격 가속 하한(배). 0 = 끔
    volume_burst_ratio: float = 0.0   # 거래량 폭발 하한(배). 0 = 끔
    volume_burst_window: float = 10.0  # 폭발 관측 창(초)
    volume_burst_base: float = 30.0   # 폭발 비교 기준 창(초, 그 직전 구간)
    # 만쥬 상한가 방아쇠(문서 44 §5-1) — 서지(+1.5%/30s)를 **대체**한다.
    #   원자료가 터치인지 잠김인지 확정되지 않아(§11 #4) 셋을 각각 켤 수 있게 뒀다.
    #   셋 다 꺼짐(기본) = 기존 서지 방아쇠 그대로. 하나라도 켜면 OR 로 판정하고
    #   서지는 limitup_require_surge=True 일 때만 AND 로 더한다.
    #   전일종가를 모르는 종목은 상한가를 계산할 수 없으므로 **방아쇠 없음**(안전 스킵).
    limitup_touch: bool = False       # 현재가 ≥ 상한가
    limitup_near_pct: float = 0.0     # 상한가까지 이 비율 이내면 발동. 0.04 = 4%p. 0 = 끔
    # 상한가까지 **남은 호가 수**로 재는 방아쇠. 만쥬가 말한 진입 시점이 이것이다 —
    # "대장이 상한가 호가를 3개 남겨둔 3~5초 사이에 매수". 비율(`limitup_near_pct`)
    # 로는 이 시점을 못 집는다: 0.05 는 등락률 +23%대에서 터져 만쥬보다 한참 이르고,
    # 그만큼 상한가에 못 가는 건을 다 사게 된다. 0 = 끔.
    limitup_near_ticks: int = 0
    # VI 방아쇠(만쥬 §5-1 "상한가 안착하거나 상한가 직전 급등·**VI에 들어가는 순간**").
    #   VI 는 +10% 에서 걸려 상한가(+30%)보다 훨씬 이르다 — 상한가만 보면 놓치는 구간이다.
    #   실시간 VI 피드(`1h`)를 구독하면 종목당 등록 단위가 2→3 이 되어 구독 가능 종목이
    #   50→33 으로 준다. 게다가 그 FID 매핑은 아직 장중 대조가 안 됐다(collector 주석).
    #   정적 VI 는 **당일 기준가 대비 ±10%** 라 가격만으로 계산된다 — 구독 비용 0.
    #   기준가는 그날 처음 본 체결가로 근사한다(시가를 따로 안 받으므로).
    vi_band_pct: float = 0.10         # 정적 VI 발동 폭(기준가 대비). KRX 기준 10%
    vi_near_pct: float = 0.0          # VI 상단까지 이 비율 이내면 발동. 0 = 끔
    limitup_lock_secs: float = 0.0    # 상한가를 이 초만큼 연속 유지(잠김 근사). 0 = 끔
    limitup_require_surge: bool = False  # 상한가 방아쇠에 서지 조건까지 AND 로 요구
    # 동적 대장/후속 선정(문서 44 §3·§4-2) — universe·sectors 를 주입했을 때만 돈다.
    #   대장 = 방아쇠가 터진 종목, 후속 = 같은 섹터 당일등락률 2~3위(대장 제외).
    # 진입 지정가를 현재가 위 몇 호가에 낼지. 0 = 매수호가(bid1)에 수동으로 건다.
    # 만쥬는 2~3 이다 — "현재가 기준 위 2~3호가 지정가 원클릭 주문".
    entry_ticks_up: int = 0
    # 등락률 2~3위 후보가 여럿일 때 **무엇으로 하나를 고르는가**. 만쥬가 준 순서:
    #   (1) 호가 두께 — 1회 주문을 1~2호가 안에서 받아줄 잔량이 깔려 있는가
    #   (2) 당일 틱 체결 강도 — 대장이 쏠 때 매수 호가로 가장 빠르게 반응하는가
    #   (3) 가벼운 소형주 우선(단, 호가가 얇지 않은 범위 내 — (1)이 이미 보장한다)
    # "rank" 는 종전 동작(등락률 순서 그대로). 만쥬와 같게 하려면 "manju".
    follower_priority: str = "rank"
    follower_rank_min: int = 2        # 후속 후보 순위 하한(1 = 대장 자리)
    follower_rank_max: int = 3        # 후속 후보 순위 상한
    # 호가 두께 — 만쥬가 **최우선**이라 한 필터(§4-1).
    #   order_krw 원을 시장가로 왕복 칠 때의 충격(bp)이 max_roundtrip_bp 를 넘으면 배제.
    #   10호가 스냅샷이 없으면(기존 호출부) 잴 수 없으므로 통과 — 기존 동작 보존.
    order_krw: float = 10_000_000     # 왕복 충격을 재는 주문금액(원)
    max_roundtrip_bp: float = 20.0    # 왕복 충격 상한(bp). 0 = 끔
    # 순간 체결량(§5-1 수급 확인) — 대장에 대량 자금이 몰려 상한가를 굳히는지.
    big_trade_krw: float = 50_000_000  # '대량 체결' 한 건의 금액 기준(원)
    min_big_trades: int = 0           # 관측창 안 대량체결 최소 건수. 0 = 끔
    big_trade_window: float = 30.0    # 대량체결 관측 창(초)
    # 상한가 매수벽 두께(§5-2 "수십만~수백만 주를 **안정적으로 굳히면**") — 기본 꺼짐.
    #   지금까지 '굳힘'은 가격만 봤다(현재가 ≥ 상한가). 만쥬는 잔량으로 판정한다:
    #   상한가 호가에 대량이 쌓여야 굳힌 것이고, 벽이 얇으면 그 상한가는 곧 풀린다.
    #   **주 수와 원화를 둘 다** 재는 이유: 만쥬의 "수십만~수백만 주"는 그가 주로 보는
    #   저가 소형주 기준이라 3만원짜리에 그대로 대면 문턱이 100배 과해진다. 반대로
    #   원화만 보면 초저가주에서 주 수 감각이 사라진다. 그래서 각각 켤 수 있게 두고,
    #   **켠 것끼리는 AND** 로 본다(둘 다 켜면 둘 다 넘어야 굳힘).
    #   호가(bids)를 안 주면 벽을 잴 수 없다 — 필터가 켜져 있으면 **안전 기각**한다
    #   (두께 필터와 반대 방향인데, 여기서 통과시키면 "굳힘 확인"이 거짓말이 된다).
    #   '상한가 직전'(near) 방아쇠에는 걸지 않는다 — 아직 상한가 호가 자체가 없다.
    lock_min_qty: float = 0.0         # 상한가 매수 잔량 하한(주). 0 = 끔
    lock_min_krw: float = 0.0         # 상한가 매수 잔량 하한(원). 0 = 끔
    # 매수벽 깎임 = 이탈 예고(§5-4 "말지 못하고 풀리거나"의 **직전** 신호).
    #   최근 lock_wall_window 초 최대 잔량 대비 현재 잔량 비율이 이 값 미만이면 이탈로
    #   본다. 가격이 아직 상한가여도 발동한다 — 벽이 절반 넘게 먹혔으면 이미 매물이
    #   이기고 있고, 가격이 풀린 뒤에 파는 건 한 박자 늦다. 0 = 끔.
    lock_wall_decay: float = 0.0      # 최근 최대 대비 잔량 비율 하한(0.5 = 반토막). 0 = 끔
    lock_wall_window: float = 30.0    # 벽 감소 관측 창(초)
    # 매도 폭탄(§5-4 "프로그램 매도 폭탄(투매)이 쏟아져 막히면 지체 없이 손절").
    #   최근 dump_window 초 대장 **매도 체결대금**(side=−1) 합이 dump_krw 이상이면 이탈.
    #   가격이 아직 상한가 위에 있어도 발동한다 — 만쥬는 "풀리거나, 또는 매도 폭탄에
    #   막히면"이라고 or 로 말한다. 0 = 끔.
    dump_krw: float = 0.0             # 관측창 매도 체결대금 임계(원). 0 = 끔
    dump_window: float = 10.0         # 매도 폭탄 관측 창(초)
    # 테마 신선도 대용치(§5-5 "신선도가 떨어진 테마는 대장이 상을 가도 후발주가 못 따라감").
    #   감지기는 DB 를 모르므로 "그 섹터가 최근 N일 내 상한가를 몇 번 냈는가"를
    #   ``sector_limitup_counts`` 로 **주입받는다**. 대장 섹터의 횟수가 이 값을 넘으면
    #   알림을 배제하고 FilteredSignal(sector_stale)로 남긴다. 0 = 끔.
    max_sector_limitups: int = 0      # 최근 N일 섹터 상한가 횟수 상한. 0 = 끔
    # 대장 이탈 통보(문서 44 §5-3): "대장이 상한가를 못 말거나 풀리는 순간 = 손익 무관
    #   전량 매도". 그 판정은 상한가를 아는 감지기만 내릴 수 있어 여기서 내고
    #   drain_leader_events() 로 밖에 알린다(가드·실행기는 판정만 받는다).
    #   확인 구간: 상한가 아래로 내려온 뒤 이 초만큼 지나서도 아래면 이탈로 본다.
    #   기본 0 = 첫 틱 즉시("풀리는 순간"). 상한가 매수벽에 체결이 한 틱 스치는 노이즈가
    #   실측으로 확인되면 1~2초를 주면 된다 — 늦추는 만큼 청산이 늦는다는 게 비용이다.
    leader_lost_confirm_secs: float = 0.0
    cooldown: float = 90.0            # 같은 쌍 재알림 억제(초)
    min_turnover: float = 0.0         # 후속 당일 누적 체결대금 하한(원). 0 = 끔
    history: float = 240.0            # 종목별 가격 히스토리 보관(초)

    @classmethod
    def stable(cls, **kw: Any) -> DetectorConfig:
        """안정형 기본(익절 +0.5% / 일상 손절 −0.4% / 90초, 재난선 없음)."""
        base: dict[str, Any] = {"take_profit": 0.005, "stop_loss": -0.004, "disaster_line": None}
        base.update(kw)
        return cls(**base)

    @classmethod
    def aggressive(cls, **kw: Any) -> DetectorConfig:
        """공격형(익절 +0.8% / 일상 손절 없음 / 재난선 −0.7% / 90초)."""
        base: dict[str, Any] = {"take_profit": 0.008, "stop_loss": None, "disaster_line": -0.007}
        base.update(kw)
        return cls(**base)


@dataclass(frozen=True)
class Alert:
    """매칭 순간의 알림 이벤트. 진입가·익절·손절·청산시각까지 사람이 칠 값으로 채운다."""

    ts: datetime
    leader: str
    follower: str
    leader_surge: float               # 대장 서지 수익률 (관측된 값)
    cosurge: float                    # 후속 동반급등 수익률
    premove: float                    # 후속 프리무브 수익률 (필터 통과값)
    price: float                      # 후속 현재가(마지막 체결)
    bid1: float                       # 후속 최우선매수호가 = 제안 진입가
    entry: float                      # 제안 진입가(bid1 지정가)
    take_profit: float | None         # 익절가. None = 상한 없음(만쥬 손익 구조)
    stop_loss: float | None           # 일상 손절가(안정형). None = 없음
    disaster_line: float | None       # 재난 방지선가(공격형 −0.7%). None = 없음
    exit_ts: datetime                 # 시간청산 시각(90초) — 실질적 손절
    turnover: float                   # 후속 당일 누적 체결대금(원)
    buy_ratio: float = -1.0           # 진입 직전30초 매수체결비율(필터 통과값). -1 = 미측정
    trigger: str = "surge"            # 방아쇠 종류: surge|touch|near|locked
    leader_ret: float = 0.0           # 대장 당일등락률(전일종가 대비). 0 = 미측정
    leader_limit: float = 0.0         # 대장 상한가(원). 0 = 미측정
    rt_bp: float = -1.0               # 후속 왕복 충격(bp, 필터 통과값). -1 = 미측정
    big_trades: int = -1              # 대장 관측창 대량체결 건수. -1 = 미측정
    lock_qty: float = -1.0            # 대장 상한가 매수벽 잔량(주). -1 = 미측정
    lock_krw: float = -1.0            # 대장 상한가 매수벽 금액(원). -1 = 미측정
    sector: str = ""                  # 대장 섹터(주입된 경우)
    sector_limitups: int = -1         # 그 섹터 최근 N일 상한가 횟수. -1 = 미측정
    leader_name: str = ""
    follower_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts"] = self.ts.isoformat(sep=" ", timespec="seconds")
        d["exit_ts"] = self.exit_ts.isoformat(sep=" ", timespec="seconds")
        return d

    def to_line(self) -> str:
        lead = self.leader_name or self.leader
        fol = self.follower_name or self.follower
        if self.stop_loss is not None:
            protect = f"손절 {self.stop_loss:,.0f}"
        elif self.disaster_line is not None:
            protect = f"재난선 {self.disaster_line:,.0f}(일상 손절 없음, −0.7% 관통 시만 즉시 청산)"
        else:
            protect = "손절 없음(90초 시간청산)"
        buy = f" 매수{self.buy_ratio:.0%}" if self.buy_ratio >= 0 else ""
        # 벽 두께는 "굳혔는가"의 근거라 방아쇠 옆에 같이 남긴다(복기 때 문턱 조정용).
        wall = f" 벽{self.lock_qty:,.0f}주" if self.lock_qty >= 0 else ""
        # 방아쇠가 무엇이었는지 한눈에 — 상한가 방아쇠는 당일등락률이 서지보다 정보가 많다.
        if self.trigger == "surge":
            head = f"+{self.leader_surge:.2%}/서지"
        else:
            kind = TRIGGER_NAMES
            head = f"{self.leader_ret:+.1%}/{kind.get(self.trigger, self.trigger)}"
        return (
            f"[{self.ts:%H:%M:%S}] 짝꿍 신호  {lead}({self.leader}) {head}{wall}"
            f"  →  {fol}({self.follower}) 동반+{self.cosurge:.2%} 프리무브{self.premove:+.2%}{buy}"
            f"  | bid1 지정가매수 {self.entry:,.0f}"
            f"  {'익절 없음' if self.take_profit is None else f'익절 {self.take_profit:,.0f}'}"
            f"  {protect}"
            f"  청산 {self.exit_ts:%H:%M:%S}"
        )


@dataclass(frozen=True)
class FilteredSignal:
    """다른 필터는 다 통과했는데 마지막 관문에서 버린 신호의 기록.

    조용히 버리면 며칠 뒤 "몇 건이 필터로 빠졌나"를 셀 수 없다. 알림과 같은 로그에
    한 줄로 남겨(:meth:`AlertBus.notice`) 나중에 세고, 필터가 과했는지 판단한다.

    ``reason`` 으로 어느 필터인지 구분한다:

    * ``min_sl_ticks`` — 보호폭이 몇 틱 안 됨(노이즈 청산 위험).
    * ``roundtrip_bp`` — 10호가 왕복 충격이 상한 초과(호가가 얇음).
    * ``depth_short``  — 10호가를 다 긁어도 주문금액을 못 채움(만쥬 §4-1 즉시 제외).
    * ``sector_stale`` — 그 테마가 최근 N일 상한가를 너무 여러 번 냄(만쥬 §5-5 신선도).
    """

    ts: datetime
    leader: str
    follower: str
    price: float                      # 후속 현재가(마지막 체결)
    entry: float                      # 제안 진입가(bid1) — 폭은 이걸 기준으로 잰다
    tick: int = 0                     # 그 가격대 호가단위(원)
    width: float = 0.0                # 판정에 쓴 보호폭(원)
    width_kind: str = ""              # "손절" | "재난선" | "익절" — 무엇의 폭인지
    sl_ticks: float = 0.0             # 보호폭 ÷ 호가단위
    min_sl_ticks: float = 0.0         # 요구 하한
    rt_bp: float = -1.0               # 왕복 충격(bp). inf = 10호가로 못 채움
    order_krw: float = 0.0            # 충격을 잰 주문금액(원)
    max_roundtrip_bp: float = 0.0     # 왕복 충격 상한(bp)
    sector: str = ""                  # 대장 섹터(신선도 배제일 때)
    sector_limitups: int = -1         # 그 섹터 최근 N일 상한가 횟수
    max_sector_limitups: int = 0      # 요구 상한
    reason: str = "min_sl_ticks"
    leader_name: str = ""
    follower_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts"] = self.ts.isoformat(sep=" ", timespec="seconds")
        # 알림 레코드와 같은 파일에 섞이므로 한눈에 구분되는 키를 둔다(집계용 grep 키).
        d["filtered"] = self.reason
        return d

    def to_line(self) -> str:
        lead = self.leader_name or self.leader
        fol = self.follower_name or self.follower
        head = (f"[{self.ts:%H:%M:%S}] 짝꿍 필터배제({self.reason})  {lead}({self.leader})"
                f"  →  {fol}({self.follower})  진입 {self.entry:,.0f}")
        if self.reason == "depth_short":
            return (f"{head}  10호가를 다 긁어도 {self.order_krw:,.0f}원을 못 채움"
                    f" — 시장가로 치면 호가가 튄다(만쥬 최우선 배제)")
        if self.reason == "sector_stale":
            return (f"{head}  테마 '{self.sector}' 최근 상한가 {self.sector_limitups}회"
                    f" > {self.max_sector_limitups}회 — 소비된 테마라 후발주가 안 따라온다")
        if self.reason == "roundtrip_bp":
            return (f"{head}  {self.order_krw:,.0f}원 왕복충격 {self.rt_bp:.1f}bp"
                    f" > {self.max_roundtrip_bp:.1f}bp — 호가가 얇아 충격이 수익을 먹는다")
        return (
            f"{head} 1틱 {self.tick:,}원"
            f"  {self.width_kind}폭 {self.width:,.1f}원 = {self.sl_ticks:.1f}틱"
            f" < {self.min_sl_ticks:.1f}틱 — 스프레드·호가 흔들림만으로 터질 폭"
        )


DEFAULT_WATCHLIST: list[dict[str, str]] = [
    # doc36/37 에서 발견된 시작 쌍. 대장(강한 급등) → 알려진 소형 후속.
    {"leader": "014990", "follower": "033340",
     "leader_name": "인디에프", "follower_name": "좋은사람들"},
    {"leader": "462860", "follower": "094480",
     "leader_name": "더즌", "follower_name": "갤럭시아머니트리"},
    {"leader": "124500", "follower": "294570",
     "leader_name": "아이티센글로벌", "follower_name": "쿠콘"},
    {"leader": "001210", "follower": "002990",
     "leader_name": "금호전기", "follower_name": "금호건설"},
]


# --------------------------------------------------------------------- 로딩

def _pair_from_dict(d: dict[str, str]) -> Pair:
    return Pair(
        leader=str(d["leader"]).strip(),
        follower=str(d["follower"]).strip(),
        leader_name=str(d.get("leader_name", "")).strip(),
        follower_name=str(d.get("follower_name", "")).strip(),
    )


def load_watchlist(path: str | Path) -> list[Pair]:
    """JSON 워치리스트를 :class:`Pair` 목록으로. 리스트[dict] 또는 {"pairs": [...]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("pairs", [])
    return [_pair_from_dict(d) for d in data]


def parse_pairs_arg(spec: str) -> list[Pair]:
    """``"014990:033340,462860:094480"`` → 쌍 목록. 이름은 비운다."""
    pairs: list[Pair] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"쌍은 '대장코드:후속코드' 형식이어야 한다: {chunk!r}")
        lead, fol = (x.strip() for x in chunk.split(":", 1))
        pairs.append(Pair(leader=lead, follower=fol))
    return pairs


# --------------------------------------------------------------------- 감지기

class PairDetector:
    """등록 쌍만 감시하는 실시간 감지기.

    틱을 :meth:`on_tick` 으로(오름차순) 먹인다. 대장 틱이 서지를 만들면 해당 대장의
    후속주(들)를 검사해 :class:`Alert` 를 돌려준다. 호가(bid1)는 :meth:`on_quote`
    또는 틱의 ``best_bid`` 로 갱신된다.
    """

    def __init__(
        self,
        pairs: list[Pair],
        config: DetectorConfig | None = None,
        *,
        prev_close: Mapping[str, float] | None = None,
        sectors: Mapping[str, str] | None = None,
        universe: Iterable[str] | None = None,
        names: Mapping[str, str] | None = None,
        sector_limitup_counts: Mapping[str, int] | None = None,
        themes: Mapping[str, Any] | None = None,
    ) -> None:
        """``pairs`` 는 사람이 큐레이션한 고정 쌍(기존 모드).

        ``universe`` + ``sectors`` 를 함께 주면 **동적 모드**가 켜진다 — 그날 섹터에서
        방아쇠가 터진 종목을 대장으로 삼고 같은 섹터 등락률 2~3위를 후속으로 고른다
        (만쥬는 쌍을 매일 새로 고른다, 문서 44 §3·§4-2). 두 모드는 공존한다.

        ``prev_close`` 는 상한가·당일등락률 계산의 유일한 입력이다. 이 모듈은 DB 도
        시계도 모르므로 전일종가·섹터·유니버스는 전부 **주입받는다**.

        ``sector_limitup_counts`` 는 ``{섹터명: 최근 N일 상한가 건수}`` — 테마 신선도
        대용치(§5-5)다. 역시 DB 를 아는 쪽에서 계산해 넣고, 장중에도
        :meth:`set_sector_limitup_counts` 로 갈아끼울 수 있다.
        """
        self.config = config or DetectorConfig()
        self.pairs = list(pairs)
        self._by_leader: dict[str, list[Pair]] = defaultdict(list)
        self._codes: set[str] = set()
        for p in self.pairs:
            self._by_leader[p.leader].append(p)
            self._codes.add(p.leader)
            self._codes.add(p.follower)
        self._prev_close: dict[str, float] = {
            str(k): float(v) for k, v in (prev_close or {}).items() if v}
        self._sectors: dict[str, str] = {
            str(k): str(v) for k, v in (sectors or {}).items() if v}
        # 종목 → 소속 테마 코드들. 만쥬가 "27종으론 실전 테마를 못 잡는다"고 한 자리를
        # 메운다. 없으면 섹터로 떨어지므로 주입은 선택이다.
        self._themes: dict[str, frozenset[str]] = {
            str(k): frozenset(str(t) for t in v)
            for k, v in (themes or {}).items() if v}
        self._names: dict[str, str] = {str(k): str(v) for k, v in (names or {}).items()}
        self._sector_limitups: dict[str, int] = {}
        self.set_sector_limitup_counts(sector_limitup_counts)
        self._universe: set[str] = {str(c) for c in (universe or ())}
        # 정적 VI 기준가 — 그날 처음 본 체결가로 근사한다(시가를 따로 안 받는다).
        # 날짜가 바뀌면 비운다. 만쥬 §5-1 VI 방아쇠의 유일한 입력이다.
        self._session_base: dict[str, float] = {}
        self._session_day: Any = None
        # 동적 모드는 유니버스와 섹터가 **둘 다** 있어야 성립한다. 하나만 오면 고정 쌍만 돈다.
        self._dynamic = bool(self._universe and self._sectors)
        if self._dynamic:
            self._codes |= self._universe
            self._by_sector: dict[str, list[str]] = defaultdict(list)
            for code in sorted(self._universe):
                sec = self._sectors.get(code)
                if sec:
                    self._by_sector[sec].append(code)
        else:
            self._by_sector = defaultdict(list)
        self._rebuild_themes()
        # 대장 상태를 추적할 종목 — 고정 쌍의 대장 + (동적 모드면) 유니버스 전부.
        # 후속주만인 종목까지 통보하면 가드가 쓰지도 않을 이벤트를 받는다.
        self._leader_codes: set[str] = set(self._by_leader)
        if self._dynamic:
            self._leader_codes |= self._universe
        self._prices: dict[str, deque[tuple[datetime, float]]] = defaultdict(deque)
        # 종목별 체결 흐름 (ts, 매수체결량, 전체체결량) — 롤링 매수비율용.
        self._flows: dict[str, deque[tuple[datetime, float, float]]] = defaultdict(deque)
        # 종목별 대량체결 시각 — 순간 체결량 필터(만쥬 §5-1 수급 확인).
        self._big: dict[str, deque[datetime]] = defaultdict(deque)
        self._bid: dict[str, float] = {}
        # 종목별 10호가 스냅샷 (매수단계, 매도단계) — 왕복 충격 계산용.
        self._book: dict[str, tuple[list[Level], list[Level]]] = {}
        # 종목별 매도 체결대금 (ts, 금액) — 매도 폭탄(§5-4) 감지용.
        self._sells: dict[str, deque[tuple[datetime, float]]] = defaultdict(deque)
        # 상한가 매수벽 잔량 이력 (ts, 잔량) — 잠김 구간에서만 쌓고 풀리면 버린다.
        self._wall: dict[str, deque[tuple[datetime, float]]] = defaultdict(deque)
        # 가격 외 사유(매도폭탄·벽깎임)로 이탈시킨 종목 — 가격이 아직 상한가라
        # 다음 틱에 곧바로 holding 으로 되돌아가는 깜빡임을 막는다. 실제로 상한가
        # 아래로 내려갔다 다시 붙을 때 풀린다(그때는 진짜 재잠김이다).
        self._relock_block: set[str] = set()
        # 상한가를 연속으로 유지한 시작 시각(잠김 근사). 이탈하면 지운다.
        self._lock_since: dict[str, datetime] = {}
        # 대장 상태 기계 — 종목별 마지막으로 내보낸 상태와, 상한가 아래로 처음 내려온 시각.
        self._leader_state: dict[str, str] = {}
        self._below_since: dict[str, datetime] = {}
        self._leader_events: list[LeaderEvent] = []
        self._turnover: dict[str, float] = defaultdict(float)
        self._last_alert: dict[tuple[str, str], datetime] = {}
        # 필터로 버린 신호 — 누계(집계용)와 아직 안 알린 큐(CLI 가 비워간다).
        self.filtered_count = 0
        self._filtered: list[FilteredSignal] = []

    # ----- 구독/커버리지 확인 -------------------------------------------------

    @property
    def codes(self) -> set[str]:
        """감시에 필요한 전 종목코드(대장+후속)."""
        return set(self._codes)

    def missing_followers(self, subscribed: set[str]) -> list[Pair]:
        """구독 종목에 후속주가 없어 감시 불가한 쌍(경고용)."""
        return [p for p in self.pairs if p.follower not in subscribed]

    def set_sector_limitup_counts(self, counts: Mapping[str, int] | None) -> None:
        """테마 신선도 대용치를 갈아끼운다 — ``{섹터명: 최근 N일 상한가 건수}``.

        장중에 새 상한가가 나면 신선도가 그 자리에서 떨어진다. 감지기는 DB 를 모르니
        세는 쪽(유니버스 갱신)이 다시 넣어준다.
        """
        self._sector_limitups = {
            str(k): int(v) for k, v in (counts or {}).items() if k}

    # ----- 입력 ---------------------------------------------------------------

    def on_quote(
        self,
        code: str,
        ts: datetime,
        bid1: float | None = None,
        *,
        bids: Sequence[Level] | None = None,
        asks: Sequence[Level] | None = None,
    ) -> None:
        """호가 스냅샷으로 bid1 과 10호가 잔량을 갱신(틱의 best_bid 보다 우선).

        ``bids``/``asks`` 는 우선순위 순서의 ``(가격, 잔량)`` 목록(보통 10단계)이다.
        주지 않으면 두께 필터는 잴 수 없어 통과시킨다 — 기존 호출부가 안 깨지도록.
        """
        if code not in self._codes:
            return
        if bids or asks:
            b = [(float(p), float(q)) for p, q in (bids or ()) if p and q]
            a = [(float(p), float(q)) for p, q in (asks or ()) if p and q]
            self._book[code] = (b, a)
            if not bid1 and b:
                bid1 = b[0][0]
            self._update_wall(code, ts, b)
        if bid1:
            self._bid[code] = float(bid1)

    def on_tick(
        self,
        code: str,
        ts: datetime,
        price: float,
        *,
        bid1: float | None = None,
        volume: float | None = None,
        side: int | None = None,
    ) -> list[Alert]:
        """틱 하나를 먹고 발생한 알림 목록을 돌려준다(대개 빈 리스트).

        ``side`` 는 체결 방향(+1 매수/−1 매도). ``volume`` 과 함께 후속주의 롤링
        매수체결비율(역선택 방어 필터)을 유지하는 데 쓴다.
        """
        if code not in self._codes:
            return []
        price = float(price)
        day = ts.date()
        if self._session_day != day:
            # 새 세션 — 어제 기준가로 오늘 VI 를 재면 안 된다.
            self._session_day = day
            self._session_base.clear()
        self._session_base.setdefault(code, price)
        hist = self._prices[code]
        hist.append((ts, price))
        self._trim(hist, ts)
        if bid1:
            self._bid[code] = float(bid1)
        if volume:
            vol = float(volume)
            self._turnover[code] += price * vol
            buy = vol if (side is not None and side > 0) else 0.0
            flow = self._flows[code]
            flow.append((ts, buy, vol))
            self._trim_flow(flow, ts)
            if side is not None and side < 0:
                sells = self._sells[code]
                sells.append((ts, price * vol))
                self._trim(sells, ts)          # (ts, 값) 모양이 같아 가격 히스토리와 공용
            # big_trade_krw 가 0 이면 모든 체결이 '대량'이 된다 — 그건 필터가 아니라
            # 무조건 통과다. 임계가 양수일 때만 센다(꺼진 것과 같은 효과).
            if self.config.big_trade_krw > 0 and price * vol >= self.config.big_trade_krw:
                big = self._big[code]
                big.append(ts)
                self._trim_big(big, ts)
        # 상한가 잠김은 틱마다 갱신해야 한다(방아쇠 판정 시점에 되돌아볼 수 없다).
        self._update_lock(code, ts, price)
        # 매도 폭탄은 **가격과 무관하게** 이탈을 부른다(§5-4 "풀리거나, 또는 막히면").
        self._check_dump(code, ts)

        is_fixed_leader = code in self._by_leader
        if not is_fixed_leader and not (self._dynamic and code in self._universe):
            return []
        fired = self._leader_trigger(code, ts, price)
        if fired is None:
            return []
        kind, surge = fired
        pairs = list(self._by_leader[code]) if is_fixed_leader else []
        if self._dynamic and code in self._universe:
            pairs += self._dynamic_pairs(code, ts)
        alerts: list[Alert] = []
        for pair in pairs:
            alert = self._check_follower(pair, ts, surge, trigger=kind)
            if alert is not None:
                alerts.append(alert)
        return alerts

    # ----- 대장 방아쇠 --------------------------------------------------------

    @property
    def _limitup_on(self) -> bool:
        cfg = self.config
        return bool(cfg.limitup_touch or cfg.limitup_near_pct > 0
                    or cfg.limitup_near_ticks > 0
                    or cfg.limitup_lock_secs > 0 or cfg.vi_near_pct > 0)

    def add_codes(self, codes: Sequence[str], *,
                  sectors: Mapping[str, str] | None = None,
                  prev_close: Mapping[str, float] | None = None,
                  session_base: Mapping[str, float] | None = None,
                  themes: Mapping[str, Any] | None = None) -> list[str]:
        """장중에 감시 종목을 추가한다 → 실제로 새로 들어온 코드 목록(만쥬 §6).

        수집기 구독만 늘리는 걸로는 아무 일도 일어나지 않는다 — :meth:`on_tick` 과
        :meth:`on_quote` 가 첫 줄에서 ``code not in self._codes`` 로 되돌아가기
        때문이다. 2026-08-26 에 실제로 그 상태였다: 로그에는 "장중 구독 추가"가
        찍히는데 알림은 영원히 안 나오는 완전한 무동작.

        섹터·전일종가를 같이 줘야 대장이 될 수 있다 — 섹터가 없으면 후속주를 못 고르고
        전일종가가 없으면 상한가를 못 구한다(둘 다 억지 추정 금지 원칙).
        """
        added: list[str] = []
        for raw in codes:
            code = str(raw)
            if not code or code in self._codes:
                continue
            self._codes.add(code)
            added.append(code)
            sec = (sectors or {}).get(code)
            if sec:
                self._sectors[code] = str(sec)
                self._by_sector[str(sec)].append(code)
            th = (themes or {}).get(code)
            if th:
                self._themes[code] = frozenset(str(t) for t in th)
            pc = (prev_close or {}).get(code)
            if pc:
                self._prev_close[code] = float(pc)
            base = (session_base or {}).get(code)
            if base:
                self._session_base[code] = float(base)
        if added and self._dynamic:
            # 동적 모드는 유니버스를 대장 후보로 쓴다 — 추가분도 대장이 될 수 있어야
            # "돌발 재료 종목이 대장이 되는" §6 의 목적이 성립한다.
            self._universe |= set(added)
            self._leader_codes |= set(added)
        if added:
            # 테마 역인덱스는 구독이 바뀌면 다시 만들어야 한다 — 안 그러면 장중에
            # 붙은 종목이 테마 무리에서 빠져 짝꿍 후보가 되지 못한다.
            self._rebuild_themes()
        return added

    def limit_up(self, code: str) -> float | None:
        """그 종목의 상한가. 전일종가를 주입받지 못했으면 None(억지 추정 금지)."""
        prev = self._prev_close.get(code)
        return limit_up_price(prev) if prev else None

    def day_return(self, code: str, ts: datetime) -> float | None:
        """당일등락률 = 현재가/전일종가 − 1. 전일종가나 가격이 없으면 None."""
        prev = self._prev_close.get(code)
        cur = self._price_at(code, ts)
        if not prev or cur is None:
            return None
        return cur / prev - 1.0

    def _update_lock(self, code: str, ts: datetime, price: float) -> None:
        """상한가 유지/이탈 상태를 갱신하고, **상태가 바뀔 때만** 이벤트를 큐에 넣는다.

        만쥬 §5-3 이 짝꿍의 진짜 청산 규칙이다 — "대장이 상한가를 못 말거나 풀리는
        순간 손익 무관 전량 매도". 상한가를 계산할 수 있는 건 전일종가를 주입받은
        이 감지기뿐이라 판정은 여기서 내리고, 가드·실행기는 결론만 받는다.

        전이할 때 한 번씩만 넣는다 — 매 틱 ``lost`` 를 쏟으면 로그도 주문도 폭주한다.
        """
        limit = self.limit_up(code)
        if limit is None:
            return                             # 전일종가 미주입 — 상한가를 모르니 통보 없음
        if price >= limit:
            self._lock_since.setdefault(code, ts)
            self._below_since.pop(code, None)
            if code in self._relock_block:
                return                         # 폭탄·벽깎임으로 이미 이탈시킨 자리 — 깜빡임 금지
            if code in self._leader_codes and self._leader_state.get(code) != LEADER_HOLDING:
                # 처음 닿은 순간, 그리고 풀렸다 다시 붙는 재잠김에서도 알린다.
                self._leader_state[code] = LEADER_HOLDING
                self._leader_events.append(LeaderEvent(code, ts, LEADER_HOLDING))
            return
        self._lock_since.pop(code, None)
        self._wall.pop(code, None)             # 잠김이 풀렸으면 벽 이력도 이 회차로 끝난다
        self._relock_block.discard(code)       # 진짜로 내려왔으니 다음 재잠김은 인정한다
        if self._leader_state.get(code) != LEADER_HOLDING:
            return                             # 애초에 상한가에 닿은 적 없음 — 이탈도 없다
        since = self._below_since.setdefault(code, ts)
        if (ts - since).total_seconds() < self.config.leader_lost_confirm_secs:
            return                             # 확인 구간(기본 0초 = 즉시) 안 — 아직 보류
        self._below_since.pop(code, None)
        self._emit_lost(code, ts, LOST_UNLOCK)

    def _emit_lost(self, code: str, ts: datetime, reason: str) -> None:
        """대장 이탈을 **한 번만** 큐에 넣는다. 사유는 곁다리 속성으로 싣는다.

        큐에 들어가는 상태 문자열은 언제나 :data:`LEADER_LOST` 다 — 가드·실행기가
        모르는 값을 조용히 무시하도록 짜여 있어서, 사유를 상태에 섞으면 청산이 안 돈다.
        """
        if self._leader_state.get(code) != LEADER_HOLDING:
            return                             # 잠긴 적이 없거나 이미 이탈 — 중복 통보 금지
        self._leader_state[code] = LEADER_LOST
        if reason != LOST_UNLOCK:
            # 가격은 아직 상한가일 수 있다 — 그대로 두면 다음 틱에 재잠김으로 읽힌다.
            self._relock_block.add(code)
        self._leader_events.append(LeaderEvent(code, ts, LEADER_LOST, reason))

    # ----- 상한가 매수벽 (§5-2 굳힘 판정) -------------------------------------

    def _update_wall(self, code: str, ts: datetime, bids: Sequence[Level]) -> None:
        """상한가 호가의 매수 잔량을 기록하고, 급격히 깎였으면 이탈로 통보한다.

        상한가에서는 bid1 가격이 곧 상한가다(그 위 호가는 존재할 수 없다). 그래서
        bid1 이 상한가에 닿아 있을 때의 잔량이 만쥬가 말한 "굳힌 물량"이다.
        """
        limit = self.limit_up(code)
        if limit is None or not bids:
            return
        px, qty = float(bids[0][0]), float(bids[0][1])
        if px < limit - 1e-9:
            self._wall.pop(code, None)         # 상한가 호가가 아니면 잴 벽이 없다
            return
        hist = self._wall[code]
        hist.append((ts, qty))
        self._trim(hist, ts)
        cfg = self.config
        if cfg.lock_wall_decay <= 0 or code not in self._leader_codes:
            return
        start = ts - timedelta(seconds=cfg.lock_wall_window)
        peak = max((q for t, q in hist if start <= t <= ts), default=0.0)
        if peak > 0 and qty / peak < cfg.lock_wall_decay:
            # 벽이 이만큼 먹혔으면 매물이 이기고 있다. 가격이 풀린 뒤에 파는 건 늦다.
            self._emit_lost(code, ts, LOST_WALL)

    def _wall_now(self, code: str) -> float | None:
        """마지막으로 관측한 상한가 매수벽 잔량(주). 관측이 없으면 None."""
        hist = self._wall.get(code)
        return hist[-1][1] if hist else None

    def _wall_ok(self, code: str) -> bool:
        """만쥬의 '굳힘' 판정 — 상한가 매수 잔량이 문턱을 넘었는가.

        둘 다 꺼져 있으면 항상 True(기존 동작). 켜져 있는데 호가를 못 받아 잴 수
        없으면 **False** — "굳힘을 확인했다"고 말할 근거가 없기 때문이다.
        """
        cfg = self.config
        if cfg.lock_min_qty <= 0 and cfg.lock_min_krw <= 0:
            return True
        qty = self._wall_now(code)
        if qty is None:
            return False
        limit = self.limit_up(code) or 0.0
        if cfg.lock_min_qty > 0 and qty < cfg.lock_min_qty:
            return False
        return not (cfg.lock_min_krw > 0 and qty * limit < cfg.lock_min_krw)

    # ----- 매도 폭탄 (§5-4) ---------------------------------------------------

    def _sell_krw(self, code: str, ts: datetime) -> float:
        """최근 ``dump_window`` 초 매도 체결대금 합(원)."""
        start = ts - timedelta(seconds=self.config.dump_window)
        return sum(v for t, v in self._sells.get(code, ()) if start <= t <= ts)

    def _check_dump(self, code: str, ts: datetime) -> None:
        """대장에 매도 폭탄이 쏟아지면 가격이 상한가여도 이탈로 통보한다."""
        cfg = self.config
        if cfg.dump_krw <= 0 or code not in self._leader_codes:
            return
        if self._leader_state.get(code) != LEADER_HOLDING:
            return                             # 잠긴 적 없는 종목은 잃을 것도 없다
        if self._sell_krw(code, ts) >= cfg.dump_krw:
            self._emit_lost(code, ts, LOST_DUMP)

    def _limitup_kind(self, code: str, ts: datetime, price: float) -> str | None:
        """발동한 상한가 방아쇠 이름. 구체적인 것(잠김)부터 본다. 없으면 None."""
        cfg = self.config
        limit = self.limit_up(code)
        if limit is None or limit <= 0:
            return None                        # 전일종가 미주입 — 안전 스킵
        # 상한가 위 방아쇠(잠김·터치)는 만쥬가 말한 '굳힘'이므로 매수벽 두께를 함께 본다.
        # '상한가 직전'(near)은 아직 상한가 호가가 없어 벽을 잴 수 없으니 제외한다.
        if price >= limit and not self._wall_ok(code):
            return None
        if cfg.limitup_lock_secs > 0:
            since = self._lock_since.get(code)
            if since is not None and (ts - since).total_seconds() >= cfg.limitup_lock_secs:
                return "locked"
        if cfg.limitup_touch and price >= limit:
            return "touch"
        # 만쥬의 진입 시점 — 상한가까지 남은 호가가 N개 이하. 호가 단위가 가격대마다
        # 달라서 비율로는 같은 "3호가"가 종목마다 다른 지점이 된다.
        if cfg.limitup_near_ticks > 0:
            if price >= limit - cfg.limitup_near_ticks * tick_size(limit):
                return "near"
            return None
        # "20%대 중후반에서 상한가로 치고 들어가는" 구간 — 상한가까지 N% 이내.
        if cfg.limitup_near_pct > 0 and price >= limit * (1.0 - cfg.limitup_near_pct):
            return "near"
        return None

    def _at_limit(self, code: str, price: float) -> bool:
        """지금 상한가 이상인가. 상한가를 모르면 False(억지 추정 금지)."""
        limit = self.limit_up(code)
        return bool(limit and price >= limit)

    def _vi_kind(self, code: str, price: float) -> str | None:
        """정적 VI 상단 근접 판정(만쥬 §5-1). 기준가를 모르면 None(억지 추정 금지)."""
        cfg = self.config
        if cfg.vi_near_pct <= 0:
            return None
        base = self._session_base.get(code)
        if not base or base <= 0:
            return None
        upper = base * (1.0 + cfg.vi_band_pct)
        return "vi" if price >= upper * (1.0 - cfg.vi_near_pct) else None

    def _leader_trigger(
        self, code: str, ts: datetime, price: float
    ) -> tuple[str, float] | None:
        """대장 방아쇠 판정. ``(방아쇠이름, 서지수익률)`` 또는 None."""
        cfg = self.config
        surge = self._ret(code, ts, cfg.surge_window)
        if self._limitup_on:
            # 상한가 계열 → VI 순으로 본다. 상한가가 더 강한 신호라 먼저다.
            # 상한가 계열 → VI 순으로 본다. 상한가가 더 강한 신호라 먼저다.
            # **다만 상한가에 붙었는데 매수벽이 얇아 거부된 건은 VI 로 되살리지
            # 않는다.** 상한가는 정적 VI 상단(기준가 +10%)을 당연히 넘으므로, 그냥
            # or 로 이으면 §5-2 굳힘 판정이 통째로 무력해지고 실제로는 상한가 터치인
            # 건이 "VI근접"으로 기록돼 복기까지 틀어진다.
            kind = self._limitup_kind(code, ts, price)
            if kind is None and not self._at_limit(code, price):
                kind = self._vi_kind(code, price)
            if kind is None:
                return None
            if cfg.limitup_require_surge and (surge is None or surge < cfg.surge_ret):
                return None
        else:
            if surge is None or surge < cfg.surge_ret:
                return None
            kind = "surge"
        # 가속도·순간체결량 관문은 **대장**에 건다 — 프로그램 매수가 때리는 건 대장이고,
        # 짝꿍이 끌려오는 것도 그 순간이다. 쌍을 돌기 전에 한 번만 판정하면 된다.
        if not self._accel_ok(code, ts):
            return None
        if cfg.min_big_trades > 0 and self._big_trade_count(code, ts) < cfg.min_big_trades:
            return None                        # 대량 자금이 안 붙음 — 상한가를 못 굳힌다
        return kind, (surge if surge is not None else 0.0)

    def _big_trade_count(self, code: str, ts: datetime) -> int:
        """최근 ``big_trade_window`` 초 안의 대량체결(≥ ``big_trade_krw``) 건수."""
        start = ts - timedelta(seconds=self.config.big_trade_window)
        return sum(1 for t in self._big.get(code, ()) if start <= t <= ts)

    def _rebuild_themes(self) -> None:
        """테마 → 소속 종목 역인덱스. 구독이 바뀔 때마다 다시 만든다."""
        self._by_theme: dict[str, list[str]] = defaultdict(list)
        for code in sorted(self._codes):
            for theme in self._themes.get(code, ()):
                self._by_theme[theme].append(code)

    def themes_of(self, code: str) -> frozenset[str]:
        """그 종목의 실전 테마 코드들. 주입 안 됐으면 빈 집합."""
        return self._themes.get(code, frozenset())

    def peers(self, leader: str) -> list[str]:
        """대장과 **같은 무리**. 테마 소속이 있으면 테마, 없으면 27종 섹터.

        만쥬: "27종 표준 산업분류로는 실전 테마를 절대 잡아낼 수 없습니다." 맞는
        말이라 테마가 있으면 그걸 먼저 쓴다. 다만 우리가 가진 테마 스냅샷은 전
        종목의 25% 만 덮어서(643/2700) 주 선택자로는 못 쓴다 — 있으면 쓰고 없으면
        섹터로 떨어진다. **섹터보다 나쁠 일은 없다.**
        """
        mine = self._themes.get(leader)
        if mine:
            peers = {c for t in mine for c in self._by_theme.get(t, ())}
            peers.discard(leader)
            if peers:
                return sorted(peers)
        sec = self._sectors.get(leader)
        return list(self._by_sector.get(sec, ())) if sec else []

    def _dynamic_pairs(self, leader: str, ts: datetime) -> list[Pair]:
        """대장과 같은 무리에서 당일등락률 ``follower_rank_min~max`` 위를 후속으로.

        순위는 대장을 **포함해** 매기고(대장이 1위인 게 보통이다) 대장 자신만 뺀다 —
        그래야 "2~3등주"라는 말이 만쥬의 관심종목 창 정렬과 같은 뜻이 된다.
        무리는 :meth:`peers` 가 정한다 — 테마가 있으면 테마, 없으면 섹터.
        """
        cfg = self.config
        group = self.peers(leader)
        if not group:
            return []
        ranked: list[tuple[float, str]] = []
        # 순위는 대장을 **포함해** 매긴다. 섹터 무리에는 대장이 이미 들어 있고 테마
        # 무리에는 빠져 있으므로, 합집합으로 한 번 맞춘 뒤 센다 — 안 그러면 대장이
        # 두 번 세어져 2·3등주가 한 칸씩 밀린다.
        for code in sorted({leader, *group}):
            ret = self.day_return(code, ts)
            if ret is not None:
                ranked.append((ret, code))
        # 등락률 내림차순, 동률은 코드 오름차순으로 고정(재현 가능한 순위).
        ranked.sort(key=lambda x: (-x[0], x[1]))
        picked: list[str] = []
        for rank, (_ret, code) in enumerate(ranked, start=1):
            if code == leader or rank < cfg.follower_rank_min or rank > cfg.follower_rank_max:
                continue
            picked.append(code)
        if cfg.follower_priority == "manju" and len(picked) > 1:
            picked.sort(key=lambda c: self._manju_priority(c, ts))
        return [Pair(leader=leader, follower=code,
                     leader_name=self._names.get(leader, ""),
                     follower_name=self._names.get(code, "")) for code in picked]

    def _manju_priority(self, code: str, ts: datetime) -> tuple[float, float, float]:
        """만쥬가 준 2등주 서열 — 작을수록 먼저 산다.

        (1) 호가 두께: 1회 주문을 왕복시키는 비용(bp)이 낮을수록 두껍다. 못 재면
            맨 뒤로 보낸다 — 두께를 모르는 종목을 두껍다고 볼 근거가 없다.
        (2) 틱 체결 강도: 매수체결비율이 높을수록 "매수 호가로 빠르게 반응"이다.
        (3) 가벼운 소형주: 당일 거래대금이 작을수록 가볍다. (1)이 이미 얇은 호가를
            걸러내므로 여기서 가벼움을 좇아도 못 팔 종목을 집지 않는다.
        """
        cfg = self.config
        book = self._book.get(code)
        bp = roundtrip_bp(book[0], book[1], cfg.order_krw) if book else None
        ratio = self._buy_ratio(code, ts, cfg.buy_ratio_window)
        return (bp if bp is not None else float("inf"),
                -(ratio if ratio is not None else 0.0),
                self._turnover.get(code, 0.0))

    def drain_filtered(self) -> list[FilteredSignal]:
        """필터로 버린 신호를 꺼내고 큐를 비운다(누계 ``filtered_count`` 는 유지).

        알림처럼 :meth:`on_tick` 반환값에 섞으면 호출부가 "알림"으로 오해해 주문까지
        갈 수 있다. 그래서 별도 큐로 빼고, CLI 가 ``bus.notice`` 로 흘린다.
        """
        out = self._filtered
        self._filtered = []
        return out

    def drain_leader_events(self) -> list[tuple[str, datetime, str]]:
        """대장 상태 **전이**를 꺼내고 큐를 비운다 — ``(대장코드, 시각, 상태)``.

        원소는 :class:`LeaderEvent` — 3-튜플로 그대로 언패킹되고, 이탈 사유가 필요하면
        ``ev.reason``(:data:`LOST_UNLOCK`/:data:`LOST_DUMP`/:data:`LOST_WALL`)을 읽는다.

        상태는 :data:`LEADER_HOLDING` / :data:`LEADER_LOST`. 호출부는 이걸
        ``guard.on_leader_state`` · ``executor.on_leader_state`` 로 그대로 흘리면 된다.

        :meth:`on_tick` 반환값(알림)에 섞지 않는 이유는 :meth:`drain_filtered` 와 같다 —
        호출부가 진입 신호로 오해해 주문까지 갈 수 있다. 여긴 **청산** 통보다.
        """
        out = self._leader_events
        self._leader_events = []
        return out

    # ----- 내부 --------------------------------------------------------------

    def _trim(self, hist: deque[tuple[datetime, float]], now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.config.history)
        while hist and hist[0][0] < cutoff:
            hist.popleft()

    def _trim_flow(self, flow: deque[tuple[datetime, float, float]], now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.config.history)
        while flow and flow[0][0] < cutoff:
            flow.popleft()

    def _trim_big(self, big: deque[datetime], now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.config.history)
        while big and big[0] < cutoff:
            big.popleft()

    def _buy_ratio(self, code: str, end: datetime, window: float) -> float | None:
        """``[end-window, end]`` 구간 매수체결비율(매수량/전체량). 체결 없으면 None."""
        flow = self._flows.get(code)
        if not flow:
            return None
        start = end - timedelta(seconds=window)
        buy = tot = 0.0
        for ts_, b, v in flow:
            if start <= ts_ <= end:
                buy += b
                tot += v
        if tot <= 0:
            return None
        return buy / tot

    def _price_at(self, code: str, t: datetime) -> float | None:
        """``t`` 이하 마지막 체결가. 없으면 None."""
        hist = self._prices.get(code)
        if not hist:
            return None
        for dt_, px in reversed(hist):
            if dt_ <= t:
                return px
        return None

    def _ret(self, code: str, end: datetime, window: float) -> float | None:
        """``[end-window, end]`` 구간 수익률. 기준가·현재가 없으면 None."""
        cur = self._price_at(code, end)
        ref = self._price_at(code, end - timedelta(seconds=window))
        if cur is None or ref is None or ref <= 0:
            return None
        return cur / ref - 1.0

    def _accel_ok(self, code: str, ts: datetime) -> bool:
        """대장의 상승이 '가속 중'인가. 두 필터 다 꺼져 있으면 항상 True."""
        cfg = self.config
        if cfg.accel_min_ratio > 0 and not self._price_accel_ok(code, ts):
            return False
        return not (cfg.volume_burst_ratio > 0 and not self._volume_burst_ok(code, ts))

    def _price_accel_ok(self, code: str, ts: datetime) -> bool:
        """서지창을 반으로 갈라 뒤 절반이 앞 절반보다 가파른지."""
        half = self.config.surge_window / 2.0
        recent = self._ret(code, ts, half)
        prior = self._ret(code, ts - timedelta(seconds=half), half)
        if recent is None or prior is None or recent <= 0:
            return False                       # 히스토리 부족·상승 아님 — 안전 기각
        if prior <= 0:
            return True                        # 앞은 잠잠, 뒤만 튐 = 가장 뚜렷한 가속
        return recent / prior >= self.config.accel_min_ratio

    def _volume_burst_ok(self, code: str, ts: datetime) -> bool:
        """최근 구간 체결량률이 그 직전 기준 구간의 몇 배인지."""
        cfg = self.config
        w, base_w = cfg.volume_burst_window, cfg.volume_burst_base
        recent = self._volume_rate(code, ts - timedelta(seconds=w), ts)
        base = self._volume_rate(code, ts - timedelta(seconds=w + base_w),
                                 ts - timedelta(seconds=w))
        if recent is None or base is None:
            return False                       # 워밍업 — 안전 기각
        if base <= 0:
            return recent > 0                  # 직전에 거래가 없었다면 폭발로 본다
        return recent / base >= cfg.volume_burst_ratio

    def _volume_rate(self, code: str, start: datetime, end: datetime) -> float | None:
        """``[start, end]`` 구간 초당 체결량. 흐름 기록이 없으면 None."""
        flow = self._flows.get(code)
        if not flow:
            return None
        span = (end - start).total_seconds()
        if span <= 0:
            return None
        tot = sum(v for ts_, _b, v in flow if start <= ts_ <= end)
        return tot / span

    def _check_follower(self, pair: Pair, ts: datetime, surge: float,
                        *, trigger: str = "surge") -> Alert | None:
        cfg = self.config
        fol = pair.follower
        cw, pw = cfg.cosurge_window, cfg.premove_window
        cur = self._price_at(fol, ts)
        p_cw = self._price_at(fol, ts - timedelta(seconds=cw))
        p_pw = self._price_at(fol, ts - timedelta(seconds=cw + pw))
        if cur is None or p_cw is None or p_pw is None or p_cw <= 0 or p_pw <= 0:
            return None  # 히스토리 부족(워밍업) — 안전하게 스킵
        cosurge = cur / p_cw - 1.0            # 동반 급등 (직전 cosurge_window 초)
        premove = p_cw / p_pw - 1.0           # 프리무브 (그 직전 premove_window 초)
        if cosurge < cfg.cosurge_min:
            return None                        # 대장과 같이 뛰지 않음
        if premove > cfg.premove_max:
            return None                        # 이미 뛴 것(추격) — 기각
        buy_ratio = self._buy_ratio(fol, ts, cfg.buy_ratio_window)
        if cfg.min_buy_ratio > 0:              # 역선택 방어(doc39)
            if buy_ratio is None:
                return None                    # 체결 데이터 부족(워밍업) — 안전 스킵
            if buy_ratio < cfg.min_buy_ratio:
                return None                    # 매수체결로 안 따라붙음(역선택 fill) — 기각
        if cfg.min_turnover > 0 and self._turnover.get(fol, 0.0) < cfg.min_turnover:
            return None
        key = (pair.leader, fol)
        last = self._last_alert.get(key)
        if last is not None and (ts - last).total_seconds() < cfg.cooldown:
            return None                        # 쿨다운
        self._last_alert[key] = ts
        bid1 = self._bid.get(fol, cur)
        # 진입 지정가. 기본은 매수호가에 걸고 기다리는 수동 주문이지만, 만쥬는
        # **현재가 위 2~3호가를 지정가로 긁어서** 산다 — 2등주가 대장을 따라 뛰는
        # 순간엔 걸어두고 기다리면 호가가 위로 도망가 아예 못 산다. 위 호가 주문은
        # 매도 최우선 잔량부터 즉시 체결되므로 사실상 시장가이되 상한이 있는 주문이다.
        entry = bid1
        if cfg.entry_ticks_up > 0:
            entry = cur + cfg.entry_ticks_up * tick_size(cur)
        # 마지막 두 관문은 쿨다운을 이미 소비한 자리에 둔다. 가격대·호가 두께는 90초
        # 안에 크게 안 변해 어차피 다시 통과 못 하고, 그래야 서지 한 번에 배제 기록이
        # 수십 줄 쌓이지 않는다(알림과 같은 빈도로 한 줄).
        #
        # 순서: **호가 두께 먼저**, 그 다음 호가단위. 만쥬가 최우선이라 한 필터가
        # 두께이고("호가가 튀면 재료가 좋아도 즉시 제외"), 두 필터가 동시에 걸리는
        # 신호는 두께 쪽 이유로 기록된다.
        wall = self._wall_now(pair.leader)
        sector = self._sectors.get(pair.leader, "")
        n_limitups = self._sector_limitups.get(sector, -1) if sector else -1
        if not self._sector_fresh_ok(pair, ts, cur, entry, sector, n_limitups):
            return None
        rt_bp = self._roundtrip_ok(pair, ts, cur, entry)
        if rt_bp is False:
            return None
        if not self._tick_width_ok(pair, ts, cur, entry):
            return None
        return Alert(
            ts=ts,
            leader=pair.leader,
            follower=fol,
            leader_surge=surge,
            cosurge=cosurge,
            premove=premove,
            price=cur,
            bid1=bid1,
            entry=entry,
            take_profit=(None if not cfg.take_profit
                         else round(entry * (1 + cfg.take_profit), 4)),
            stop_loss=(round(entry * (1 + cfg.stop_loss), 4)
                       if cfg.stop_loss is not None else None),
            disaster_line=(round(entry * (1 + cfg.disaster_line), 4)
                           if cfg.disaster_line is not None else None),
            exit_ts=ts + timedelta(seconds=cfg.time_exit),
            turnover=self._turnover.get(fol, 0.0),
            buy_ratio=(buy_ratio if buy_ratio is not None else -1.0),
            trigger=trigger,
            leader_ret=(self.day_return(pair.leader, ts) or 0.0),
            leader_limit=(self.limit_up(pair.leader) or 0.0),
            rt_bp=(rt_bp if isinstance(rt_bp, float) else -1.0),
            big_trades=(self._big_trade_count(pair.leader, ts)
                        if cfg.min_big_trades > 0 else -1),
            lock_qty=(wall if wall is not None else -1.0),
            lock_krw=(wall * (self.limit_up(pair.leader) or 0.0)
                      if wall is not None else -1.0),
            sector=sector,
            sector_limitups=n_limitups,
            leader_name=pair.leader_name,
            follower_name=pair.follower_name,
        )

    def _sector_fresh_ok(self, pair: Pair, ts: datetime, cur: float, entry: float,
                         sector: str, n_limitups: int) -> bool:
        """테마 신선도 관문(§5-5). 소비된 테마면 기록을 남기고 False.

        섹터를 모르거나 집계를 못 받았으면(``-1``) 판단 근거가 없으므로 막지 않는다 —
        신선도는 등급 C 짜리 대용치라 근거 없이 기각하면 놓치는 쪽이 더 비싸다.
        """
        cfg = self.config
        if cfg.max_sector_limitups <= 0 or not sector or n_limitups < 0:
            return True
        if n_limitups <= cfg.max_sector_limitups:
            return True
        self.filtered_count += 1
        self._filtered.append(FilteredSignal(
            ts=ts,
            leader=pair.leader,
            follower=pair.follower,
            price=cur,
            entry=entry,
            sector=sector,
            sector_limitups=n_limitups,
            max_sector_limitups=cfg.max_sector_limitups,
            reason="sector_stale",
            leader_name=pair.leader_name,
            follower_name=pair.follower_name,
        ))
        return False

    def _roundtrip_ok(self, pair: Pair, ts: datetime, cur: float,
                      entry: float) -> float | bool:
        """호가 두께 관문(만쥬 최우선). 통과면 측정한 왕복충격(bp), 배제면 ``False``.

        10호가 스냅샷이 없으면 잴 수 없으므로 ``-1.0`` 을 돌려 통과시킨다 —
        호가를 안 넣어주던 기존 호출부의 동작을 그대로 보존하기 위해서다.
        """
        cfg = self.config
        if cfg.max_roundtrip_bp <= 0:
            return -1.0                        # 필터 끔
        book = self._book.get(pair.follower)
        if not book:
            return -1.0                        # 10호가 미주입 — 잴 수 없으니 막지 않는다
        bids, asks = book
        bp = roundtrip_bp(bids, asks, cfg.order_krw)
        if bp is None:
            return -1.0
        if bp <= cfg.max_roundtrip_bp:
            return bp
        self.filtered_count += 1
        self._filtered.append(FilteredSignal(
            ts=ts,
            leader=pair.leader,
            follower=pair.follower,
            price=cur,
            entry=entry,
            rt_bp=bp,
            order_krw=cfg.order_krw,
            max_roundtrip_bp=cfg.max_roundtrip_bp,
            reason=("depth_short" if math.isinf(bp) else "roundtrip_bp"),
            leader_name=pair.leader_name,
            follower_name=pair.follower_name,
        ))
        return False

    def _tick_width_ok(self, pair: Pair, ts: datetime, cur: float, entry: float) -> bool:
        """보호폭이 ``min_sl_ticks`` 틱 이상인가. 아니면 기록을 남기고 False."""
        cfg = self.config
        if cfg.min_sl_ticks <= 0:
            return True                        # 필터 끔(과거 동작 재현)
        # 일상 손절 → 없으면 재난선 → 그것도 없으면 익절폭. 잴 게 없으면 막지 않는다.
        if cfg.stop_loss is not None:
            width, kind = abs(cfg.stop_loss) * entry, "손절"
        elif cfg.disaster_line is not None:
            width, kind = abs(cfg.disaster_line) * entry, "재난선"
        elif cfg.take_profit:
            width, kind = abs(cfg.take_profit) * entry, "익절"
        else:
            return True
        if width <= 0:
            return True
        n = ticks_in(width, entry)
        if n >= cfg.min_sl_ticks:
            return True
        self.filtered_count += 1
        self._filtered.append(FilteredSignal(
            ts=ts,
            leader=pair.leader,
            follower=pair.follower,
            price=cur,
            entry=entry,
            tick=tick_size(entry),
            width=width,
            width_kind=kind,
            sl_ticks=n,
            min_sl_ticks=cfg.min_sl_ticks,
            leader_name=pair.leader_name,
            follower_name=pair.follower_name,
        ))
        return False
