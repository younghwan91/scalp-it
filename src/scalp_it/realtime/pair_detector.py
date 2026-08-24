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
  아니라 선행을 산다).

* **청산(알림에 표시만, 주문은 사람이)** — 두 모드, 기본은 안정형:
  안정형 익절 +0.5% / 일상 손절 −0.4% / 90초 시간청산(세후 +0.19%·승률 71%·MDD −0.74%),
  공격형 익절 +0.8% / 일상 손절 없음 + 재난선 −0.7% / 90초(세후 +0.27%·MDD −1.27%).
  두 모드 공통: bid1 지정가 진입 · 90초 시간청산 · 트레일 없음. 값은 :class:`DetectorConfig`.

* **쿨다운**: 같은 쌍은 ``cooldown`` 초(기본 90s) 안에 재알림하지 않는다.
* **거래대금 하한**: 후속주 당일 누적 체결대금이 ``min_turnover`` 미만이면 알림하지 않는다.

이 모듈은 DB·시계를 모르고, 오름차순으로 들어오는 틱을 :meth:`PairDetector.on_tick` 으로
먹고 알림 리스트를 돌려줄 뿐이라 mock 틱으로 단위 테스트한다.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_WATCHLIST",
    "Alert",
    "DetectorConfig",
    "Pair",
    "PairDetector",
    "load_watchlist",
    "parse_pairs_arg",
]


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
    # 청산 — 두 모드를 한 config 로 표현한다. 기본값 = 안정형.
    #   안정형(기본): 익절 +0.5% / 일상 손절 −0.4% / 재난선 없음 / 90초.
    #                세후 +0.19%·승률 71%·MDD −0.74%. 소액·심리 편함용.
    #   공격형(옵션): 익절 +0.8% / 일상 손절 없음 / 재난선 −0.7% / 90초.
    #                세후 +0.27%·MDD −1.27%. (bid1 눌림매수라 역행이 얕아 타이트 손절이
    #                스프레드 노이즈에 튕기므로 일상 손절을 빼고 폭탄방지선만 둔다.)
    take_profit: float = 0.005        # 익절 (+0.5% 안정형 / +0.8% 공격형)
    stop_loss: float | None = -0.004  # 일상 손절 (안정형 −0.3~0.4%). None = 없음(공격형)
    disaster_line: float | None = None  # 재난 방지선 (공격형 −0.7%). None = 없음(안정형)
    time_exit: float = 90.0           # 시간청산(초) — 두 모드 공통, 실질적 손절 역할
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
    take_profit: float                # 익절가
    stop_loss: float | None           # 일상 손절가(안정형). None = 없음
    disaster_line: float | None       # 재난 방지선가(공격형 −0.7%). None = 없음
    exit_ts: datetime                 # 시간청산 시각(90초) — 실질적 손절
    turnover: float                   # 후속 당일 누적 체결대금(원)
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
        return (
            f"[{self.ts:%H:%M:%S}] 짝꿍 신호  {lead}({self.leader}) +{self.leader_surge:.2%}/서지"
            f"  →  {fol}({self.follower}) 동반+{self.cosurge:.2%} 프리무브{self.premove:+.2%}"
            f"  | bid1 지정가매수 {self.entry:,.0f}"
            f"  익절 {self.take_profit:,.0f}  {protect}"
            f"  청산 {self.exit_ts:%H:%M:%S}"
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

    def __init__(self, pairs: list[Pair], config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.pairs = list(pairs)
        self._by_leader: dict[str, list[Pair]] = defaultdict(list)
        self._codes: set[str] = set()
        for p in self.pairs:
            self._by_leader[p.leader].append(p)
            self._codes.add(p.leader)
            self._codes.add(p.follower)
        self._prices: dict[str, deque[tuple[datetime, float]]] = defaultdict(deque)
        self._bid: dict[str, float] = {}
        self._turnover: dict[str, float] = defaultdict(float)
        self._last_alert: dict[tuple[str, str], datetime] = {}

    # ----- 구독/커버리지 확인 -------------------------------------------------

    @property
    def codes(self) -> set[str]:
        """감시에 필요한 전 종목코드(대장+후속)."""
        return set(self._codes)

    def missing_followers(self, subscribed: set[str]) -> list[Pair]:
        """구독 종목에 후속주가 없어 감시 불가한 쌍(경고용)."""
        return [p for p in self.pairs if p.follower not in subscribed]

    # ----- 입력 ---------------------------------------------------------------

    def on_quote(self, code: str, ts: datetime, bid1: float | None) -> None:
        """호가 스냅샷으로 bid1 갱신(틱의 best_bid 보다 우선)."""
        if code in self._codes and bid1:
            self._bid[code] = float(bid1)

    def on_tick(
        self,
        code: str,
        ts: datetime,
        price: float,
        *,
        bid1: float | None = None,
        volume: float | None = None,
    ) -> list[Alert]:
        """틱 하나를 먹고 발생한 알림 목록을 돌려준다(대개 빈 리스트)."""
        if code not in self._codes:
            return []
        price = float(price)
        hist = self._prices[code]
        hist.append((ts, price))
        self._trim(hist, ts)
        if bid1:
            self._bid[code] = float(bid1)
        if volume:
            self._turnover[code] += price * float(volume)

        if code not in self._by_leader:
            return []
        surge = self._ret(code, ts, self.config.surge_window)
        if surge is None or surge < self.config.surge_ret:
            return []
        alerts: list[Alert] = []
        for pair in self._by_leader[code]:
            alert = self._check_follower(pair, ts, surge)
            if alert is not None:
                alerts.append(alert)
        return alerts

    # ----- 내부 --------------------------------------------------------------

    def _trim(self, hist: deque[tuple[datetime, float]], now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.config.history)
        while hist and hist[0][0] < cutoff:
            hist.popleft()

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

    def _check_follower(self, pair: Pair, ts: datetime, surge: float) -> Alert | None:
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
        if cfg.min_turnover > 0 and self._turnover.get(fol, 0.0) < cfg.min_turnover:
            return None
        key = (pair.leader, fol)
        last = self._last_alert.get(key)
        if last is not None and (ts - last).total_seconds() < cfg.cooldown:
            return None                        # 쿨다운
        self._last_alert[key] = ts
        bid1 = self._bid.get(fol, cur)
        entry = bid1
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
            take_profit=round(entry * (1 + cfg.take_profit), 4),
            stop_loss=(round(entry * (1 + cfg.stop_loss), 4)
                       if cfg.stop_loss is not None else None),
            disaster_line=(round(entry * (1 + cfg.disaster_line), 4)
                           if cfg.disaster_line is not None else None),
            exit_ts=ts + timedelta(seconds=cfg.time_exit),
            turnover=self._turnover.get(fol, 0.0),
            leader_name=pair.leader_name,
            follower_name=pair.follower_name,
        )
