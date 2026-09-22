"""RL 정책 실시간 추론 그림자 — 주문 없음. 승격 체크리스트 9번(의사결정 패리티)의 경로.

**규칙: 피처는 백테스트와 같은 커널에서 나온 값만 쓴다.** 여기서 따로 계산하지 않는다 — 두 번 구현하면
값이 어긋나는 순간 백테스트가 무의미해진다. 커널이 정책이 요구하는 피처를 못 주면 **추론을 끈다**
(0 으로 채우고 돌리면 조용한 오답이 된다).

흐름: 틱·호가 → 초 단위 집계 → (초 마감) 커널 update → 피처 벡터 → 정책 → 행동 → 가상 체결 기록.
체결 가정은 백테스트와 같다 — 시장가 매수는 다음 초 매도1호가, 매도는 다음 초 매수1호가.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["PolicyWeights", "RLShadow"]

COST = 0.0023
CANCEL_SECS = 10           # 지정가 취소 시한 — 86번 §3 체결 모형의 전제와 같다


@dataclass
class PolicyWeights:
    """학습 산출물 — 피처 이름·MLP 가중치·표준화 통계. 프레임워크 없이 numpy 로 추론한다."""

    features: list[str]
    layers: list[tuple[np.ndarray, np.ndarray]]
    mu: np.ndarray
    sd: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    actions: tuple[str, ...] = ("hold", "market_buy", "limit_buy", "sell")
    """행동 공간 — **정책이 알려준다.** 96번은 3개(hold·limit_buy·sell)다.

    인덱스로 고정하면 행동 수가 바뀌는 순간 한 칸씩 밀려 "매도를 눌렀는데 지정가를 건다"가 된다.
    """

    @classmethod
    def from_npz(cls, path: str | Path, *, features: list[str], mu, sd_, lo, hi,
                 actions: tuple[str, ...] | None = None) -> PolicyWeights:
        """numpy 만으로 읽는다 — 운용 환경에 torch 를 넣지 않기 위해서다."""
        z = np.load(path)
        n = len([k for k in z.files if k.startswith("w")])
        layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(n)]
        return cls(features=list(features), layers=layers, mu=np.asarray(mu), sd=np.asarray(sd_),
                   lo=np.asarray(lo), hi=np.asarray(hi), **({"actions": tuple(actions)} if actions else {}))

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, features: list[str], mu, sd_, lo, hi,
                        actions: tuple[str, ...] | None = None) -> PolicyWeights:
        import torch

        sd = torch.load(path, map_location="cpu", weights_only=False)
        keys = sorted({k.rsplit(".", 1)[0] for k in sd if k.endswith(".weight")},
                      key=lambda k: int(k.rsplit(".", 1)[-1]))
        layers = [(sd[f"{k}.weight"].numpy(), sd[f"{k}.bias"].numpy()) for k in keys]
        return cls(features=list(features), layers=layers, mu=np.asarray(mu), sd=np.asarray(sd_),
                   lo=np.asarray(lo), hi=np.asarray(hi), **({"actions": tuple(actions)} if actions else {}))

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h

    def scale(self, raw: np.ndarray) -> np.ndarray:
        z = (np.clip(raw, self.lo, self.hi) - self.mu) / np.where(self.sd == 0, 1.0, self.sd)
        return np.nan_to_num(z).astype(np.float32)


@dataclass
class _Code:
    sec: int = -1
    buy: float = 0.0
    sell: float = 0.0
    cnt: float = 0.0
    bid: float = float("nan")
    ask: float = float("nan")
    strength: float = float("nan")
    bidqty1: float = float("nan")
    askqty1: float = float("nan")
    bid_depth3: float = float("nan")
    ask_depth3: float = float("nan")
    stream: Any = None
    pos: int = 0
    entry: float = 0.0
    entry_sec: int = 0
    mark: float = 0.0
    # **커널 입력 전용** — 그 초에 실제로 온 값만. 안 오면 NaN 이다.
    # 배치(aggregate_seconds_v2)가 호가 없는 초에 NaN 을 넣기 때문이다(069500/09-22: 22,800초 중 3,881초).
    # 위의 sticky 필드(bid/ask/…)는 체결 판정·평가손익용이라 직전 값을 유지한다. 두 용도를 섞으면 안 된다.
    q: dict = field(default_factory=dict, repr=False)
    # 그 초의 체결 — core v2 증분 커널이 집계값이 아니라 **체결 배열**을 받는다.
    px: list = field(default_factory=list, repr=False)
    vol: list = field(default_factory=list, repr=False)
    sd: list = field(default_factory=list, repr=False)
    pending: str = ""          # "buy" | "sell" | "limit" — 앞의 둘은 다음 초 호가로 체결한다
    lim_P: float = 0.0         # 걸어둔 지정가(0 = 없음)
    lim_sec: int = 0           # 건 시각 — 취소 시한 판정용
    lim_queue: float = 0.0     # 걸 때 앞에 있던 잔량(매수1호가 수량)
    lim_cum: float = 0.0       # 건 뒤 우리 가격에 지나간 매도 체결량
    lim_through: bool = False  # 가격이 우리 지정가 아래로 뚫렸는가
    feats: np.ndarray | None = field(default=None, repr=False)


class RLShadow:
    EXTRA = 4                  # 보유 여부 · 미실현% · 보유초/60 · 주문중

    def __init__(self, policy: PolicyWeights, *, stream_factory: Callable[[], Any],
                 sink: Callable[[dict[str, Any]], None], codes: set[str] | None = None,
                 extra_features: Callable[[str, int], dict[str, float]] | str | None = None) -> None:
        """extra_features: 커널이 못 주는 종목 간 피처(cum_rank·dayret_rank)의 공급자.

        문자열 "turnover_rank" 를 주면 그림자가 당일 누적 체결대금으로 순위를 직접 만든다
        (백테스트 정의와 같다). 부족한 피처를 0 으로 채우는 일은 하지 않는다 — 없으면 추론을 끈다.
        """
        self.policy = policy
        self.stream_factory = stream_factory
        self.sink = sink
        self.codes = codes
        self.extra_features = extra_features
        self.state: dict[str, _Code] = {}
        self.cumval: dict[str, float] = {}
        self.disabled_reason: str | None = None

    def open_count(self) -> int:
        return sum(1 for c in self.state.values() if c.pos == 1)

    # ----------------------------------------------------------------- 입력
    def on_quote(self, code: str, ts: datetime, *, bid1=None, ask1=None, bidqty1=None, askqty1=None,
                 bid_depth3=None, ask_depth3=None) -> None:
        if self.codes is not None and code not in self.codes:
            return
        st = self._roll(code, ts)
        if bid1 and ask1:
            st.bid, st.ask = float(bid1), float(ask1)
        if bid1:
            st.q["bid"] = float(bid1)
        if ask1:
            st.q["ask"] = float(ask1)
        for name, v in (("bidqty1", bidqty1), ("askqty1", askqty1),
                        ("bid_depth3", bid_depth3), ("ask_depth3", ask_depth3)):
            if v is not None:
                setattr(st, name, float(v))
                st.q[name] = float(v)
        feed = getattr(st.stream, "feed_quote", None)
        if feed is not None and bid1 and ask1:
            feed(ts, float(bid1), float(ask1), bidqty1 or 0.0, 0.0, 0.0, askqty1 or 0.0, 0.0, 0.0)

    def on_tick(self, code: str, ts: datetime, price: float, *, volume=None, side=None, strength=None,
                best_bid=None, best_ask=None) -> None:
        if self.codes is not None and code not in self.codes:
            return
        st = self._roll(code, ts)
        if best_bid and best_ask:
            st.bid, st.ask = float(best_bid), float(best_ask)
            st.q.setdefault("bid", float(best_bid))
            st.q.setdefault("ask", float(best_ask))
        v = float(price) * float(volume or 0)
        st.cnt += 1
        st.px.append(float(price))
        st.vol.append(float(volume or 0))
        st.sd.append(int(side or 0))
        self.cumval[code] = self.cumval.get(code, 0.0) + v
        if side is not None and side > 0:
            st.buy += v
        elif side is not None and side < 0:
            st.sell += v
        if strength is not None:
            st.strength = float(strength)
            st.q["strength"] = float(strength)
        if st.lim_P > 0:
            # 대기열 체결(86 §3): 우리 가격에 지나간 매도 체결량을 세고, 뚫림도 본다.
            if float(price) < st.lim_P:
                st.lim_through = True
            elif side is not None and side < 0 and float(price) == st.lim_P:
                st.lim_cum += float(volume or 0)
        feed = getattr(st.stream, "feed_tick", None)      # 창 재계산 커널은 원시 틱을 그대로 받는다
        if feed is not None:
            feed(ts, float(price), float(volume or 0), int(side or 0), strength, best_bid, best_ask)

    # ----------------------------------------------------------------- 초 마감
    def _roll(self, code: str, ts: datetime) -> _Code:
        sec = ts.hour * 3600 + ts.minute * 60 + ts.second
        st = self.state.get(code)
        if st is None:
            try:
                stream = self.stream_factory(code)       # 창 커널은 종목을 알아야 한다
            except TypeError:
                stream = self.stream_factory()
            st = self.state[code] = _Code(stream=stream)
        if st.sec != sec:
            if st.sec >= 0 and sec > st.sec:
                self._finalize(code, st)
            st.sec = sec
            st.buy = st.sell = st.cnt = 0.0
            st.px.clear()
            st.vol.clear()
            st.sd.clear()
            st.q.clear()
        return st

    def extra_for(self, code: str, sec: int) -> dict[str, float]:
        """종목 간 피처. 공급자가 함수면 그대로 쓰고, "turnover_rank" 면 누적 체결대금 순위를 만든다."""
        if callable(self.extra_features):
            return dict(self.extra_features(code, sec))
        if self.extra_features == "turnover_rank":
            mine = self.cumval.get(code, 0.0)
            return {"cum_rank": 1 + sum(1 for v in self.cumval.values() if v > mine)}
        return {}

    def _vector(self, st: _Code, feats: dict[str, float]) -> np.ndarray | None:
        missing = [f for f in self.policy.features if f not in feats]
        if missing:
            if self.disabled_reason is None:
                self.disabled_reason = (f"피처 커널이 정책이 요구하는 값을 못 준다: {missing[:3]}"
                                        f"{'…' if len(missing) > 3 else ''} — 추론을 끈다")
                # 그 자리에서 남긴다 — 종료 때 알면 하루를 버린다(2026-09-17).
                self.sink({"event": "disabled", "sec": st.sec, "reason": self.disabled_reason})
            return None
        raw = np.array([feats[f] for f in self.policy.features], float)
        z = self.policy.scale(raw)
        unreal = (st.mark / st.entry - 1) * 100 if st.pos == 1 and st.entry else 0.0
        hold = min((st.sec - st.entry_sec) / 60.0, 10.0) if st.pos == 1 else 0.0
        return np.concatenate([z, [float(st.pos), unreal, hold, 1.0 if st.pending else 0.0]]).astype(np.float32)

    def _resolve_limit(self, code: str, st: _Code) -> None:
        """걸어둔 지정가를 이 초 마감 시점에 판정한다 — 체결이면 보유 시작, 시한이면 취소."""
        if st.lim_through or st.lim_cum >= st.lim_queue + 1:
            st.pos, st.entry, st.entry_sec = 1, st.lim_P, st.sec
            st.mark = st.bid if st.bid > 0 else st.lim_P
            self.sink({"event": "buy", "code": code, "sec": st.sec, "price": st.lim_P, "kind": "지정가",
                       "wait": st.sec - st.lim_sec,
                       "how": "뚫림" if st.lim_through else "대기열"})
            self._clear_limit(st)
        elif st.sec - st.lim_sec >= CANCEL_SECS:
            self.sink({"event": "limit_cancel", "code": code, "sec": st.sec, "price": st.lim_P,
                       "queue": st.lim_queue, "filled_qty": st.lim_cum})
            self._clear_limit(st)

    @staticmethod
    def _clear_limit(st: _Code) -> None:
        st.lim_P, st.lim_cum, st.lim_queue, st.lim_through = 0.0, 0.0, 0.0, False
        st.pending = ""

    def _finalize(self, code: str, st: _Code) -> None:
        # 1) 직전 초에 정한 행동을 이 초 호가로 체결(백테스트와 같은 가정)
        if st.pending == "buy" and st.ask > 0:
            st.pos, st.entry, st.entry_sec, st.mark = 1, st.ask, st.sec, st.bid
            self.sink({"event": "buy", "code": code, "sec": st.sec, "price": st.ask, "kind": "시장가"})
            st.pending = ""
        elif st.pending == "sell" and st.bid > 0 and st.pos == 1:
            ret = st.bid / st.entry - 1 - COST
            self.sink({"event": "sell", "code": code, "sec": st.sec, "price": st.bid, "ret": ret,
                       "hold": st.sec - st.entry_sec})
            st.pos, st.entry, st.pending = 0, 0.0, ""
        elif st.lim_P > 0:
            self._resolve_limit(code, st)
        else:
            st.pending = ""
        if st.pos == 1 and st.bid > 0:
            st.mark = st.bid
        if self.disabled_reason:
            return
        # 2) 이 초 상태로 커널을 돌리고 다음 행동을 정한다
        if hasattr(st.stream, "sec"):
            st.stream.sec = st.sec
        nan = float("nan")
        kern = st.stream.update(buy=st.buy, sell=st.sell, cnt=st.cnt,
                                bid=st.q.get("bid", nan), ask=st.q.get("ask", nan),
                                strength=st.q.get("strength", nan),
                                bidqty1=st.q.get("bidqty1", nan), askqty1=st.q.get("askqty1", nan),
                                bid_depth3=st.q.get("bid_depth3", nan),
                                ask_depth3=st.q.get("ask_depth3", nan),
                                price=tuple(st.px), volume=tuple(st.vol), side=tuple(st.sd))
        if not kern:
            # 창이 아직 안 데워졌다(장 초반·거래 공백). **결손이 아니라 준비 안 됨**이므로 이 초만 건너뛴다.
            # 2026-09-17 실측: 이걸 결손으로 보고 하루 종일 추론을 꺼서 기록이 0건이었다.
            return
        feats = dict(self.extra_for(code, st.sec))
        feats.update(kern)
        x = self._vector(st, feats)
        if x is None:
            return
        q = self.policy.forward(x)
        acts = self.policy.actions
        busy = st.pos == 1 or st.lim_P > 0
        mask = np.array([a == "hold"
                         or (a in ("market_buy", "limit_buy") and not busy)
                         or (a == "sell" and st.pos == 1) for a in acts[:len(q)]], bool)
        q = np.where(mask, q, -np.inf)
        act = acts[int(np.argmax(q))]
        if act == "market_buy":
            st.pending = "buy"
        elif act == "limit_buy" and st.pos == 0 and st.lim_P == 0:
            # 지정가 매수 — 86 §3 대기열 모형을 실시간에서도 돌린다. 잔량을 모르면 체결을 지어내지 않는다.
            if st.bid > 0 and st.bidqty1 == st.bidqty1:
                st.lim_P, st.lim_sec, st.lim_queue = st.bid, st.sec, st.bidqty1
                st.lim_cum, st.lim_through, st.pending = 0.0, False, "limit"
                self.sink({"event": "limit_place", "code": code, "sec": st.sec, "price": st.bid,
                           "queue": st.bidqty1})
            else:
                self.sink({"event": "limit_intent", "code": code, "sec": st.sec, "price": st.bid,
                           "why": "잔량 미상 — 체결 모형 불가"})
        elif act == "sell" and st.pos == 1:
            st.pending = "sell"

    def flush(self) -> None:
        for code, st in self.state.items():
            if st.pos == 1 and st.bid > 0:
                self.sink({"event": "sell", "code": code, "sec": st.sec, "price": st.bid,
                           "ret": st.bid / st.entry - 1 - COST, "hold": st.sec - st.entry_sec,
                           "reason": "마감"})
                st.pos = 0
