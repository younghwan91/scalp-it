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


@dataclass
class PolicyWeights:
    """학습 산출물 — 피처 이름·MLP 가중치·표준화 통계. 프레임워크 없이 numpy 로 추론한다."""

    features: list[str]
    layers: list[tuple[np.ndarray, np.ndarray]]
    mu: np.ndarray
    sd: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, features: list[str], mu, sd_, lo, hi) -> PolicyWeights:
        import torch

        sd = torch.load(path, map_location="cpu", weights_only=False)
        keys = sorted({k.rsplit(".", 1)[0] for k in sd if k.endswith(".weight")},
                      key=lambda k: int(k.rsplit(".", 1)[-1]))
        layers = [(sd[f"{k}.weight"].numpy(), sd[f"{k}.bias"].numpy()) for k in keys]
        return cls(features=list(features), layers=layers, mu=np.asarray(mu), sd=np.asarray(sd_),
                   lo=np.asarray(lo), hi=np.asarray(hi))

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
    pending: str = ""          # "buy" | "sell" — 다음 초 호가로 체결한다
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
        for name, v in (("bidqty1", bidqty1), ("askqty1", askqty1),
                        ("bid_depth3", bid_depth3), ("ask_depth3", ask_depth3)):
            if v is not None:
                setattr(st, name, float(v))

    def on_tick(self, code: str, ts: datetime, price: float, *, volume=None, side=None, strength=None,
                best_bid=None, best_ask=None) -> None:
        if self.codes is not None and code not in self.codes:
            return
        st = self._roll(code, ts)
        if best_bid and best_ask:
            st.bid, st.ask = float(best_bid), float(best_ask)
        v = float(price) * float(volume or 0)
        st.cnt += 1
        self.cumval[code] = self.cumval.get(code, 0.0) + v
        if side is not None and side > 0:
            st.buy += v
        elif side is not None and side < 0:
            st.sell += v
        if strength is not None:
            st.strength = float(strength)

    # ----------------------------------------------------------------- 초 마감
    def _roll(self, code: str, ts: datetime) -> _Code:
        sec = ts.hour * 3600 + ts.minute * 60 + ts.second
        st = self.state.get(code)
        if st is None:
            st = self.state[code] = _Code(stream=self.stream_factory())
        if st.sec != sec:
            if st.sec >= 0 and sec > st.sec:
                self._finalize(code, st)
            st.sec = sec
            st.buy = st.sell = st.cnt = 0.0
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
            self.disabled_reason = (f"피처 커널이 정책이 요구하는 값을 못 준다: {missing[:3]}"
                                    f"{'…' if len(missing) > 3 else ''} — 추론을 끈다")
            return None
        raw = np.array([feats[f] for f in self.policy.features], float)
        z = self.policy.scale(raw)
        unreal = (st.mark / st.entry - 1) * 100 if st.pos == 1 and st.entry else 0.0
        hold = min((st.sec - st.entry_sec) / 60.0, 10.0) if st.pos == 1 else 0.0
        return np.concatenate([z, [float(st.pos), unreal, hold, 1.0 if st.pending else 0.0]]).astype(np.float32)

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
        else:
            st.pending = ""
        if st.pos == 1 and st.bid > 0:
            st.mark = st.bid
        if self.disabled_reason:
            return
        # 2) 이 초 상태로 커널을 돌리고 다음 행동을 정한다
        feats = dict(self.extra_for(code, st.sec))
        feats.update(st.stream.update(buy=st.buy, sell=st.sell, cnt=st.cnt, bid=st.bid, ask=st.ask,
                                 strength=st.strength, bidqty1=st.bidqty1, askqty1=st.askqty1,
                                 bid_depth3=st.bid_depth3, ask_depth3=st.ask_depth3))
        x = self._vector(st, feats)
        if x is None:
            return
        q = self.policy.forward(x)
        mask = np.ones(len(q), bool)
        if st.pos == 1:
            mask[1] = mask[2] = False
        else:
            mask[3] = False
        q = np.where(mask, q, -np.inf)
        act = int(np.argmax(q))
        if act == 1:
            st.pending = "buy"
        elif act == 2:                      # 지정가 매수는 대기열 모형이 필요하다 — 기록만 하고 체결은 안 만든다
            self.sink({"event": "limit_intent", "code": code, "sec": st.sec, "price": st.bid})
        elif act == 3 and st.pos == 1:
            st.pending = "sell"

    def flush(self) -> None:
        for code, st in self.state.items():
            if st.pos == 1 and st.bid > 0:
                self.sink({"event": "sell", "code": code, "sec": st.sec, "price": st.bid,
                           "ret": st.bid / st.entry - 1 - COST, "hold": st.sec - st.entry_sec,
                           "reason": "마감"})
                st.pos = 0
