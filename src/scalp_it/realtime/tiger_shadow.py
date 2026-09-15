"""86번 호랑이 규칙 — 소켓 실시간 그림자(주문 없음). 규칙 정본 docs/research/manju/86 §2~§4.

리플레이 `scripts/tiger_86/engine.py` 와 같은 판정을 **초 마감 시점**에 한다. 종목마다 새 초의 첫 이벤트가 오면
직전 초를 마감해 (1) 대기 주문 체결·취소, (2) 보유 청산, (3) 신호를 판정한다. 이벤트가 없는 초는 값이 그대로라
리플레이의 전방채움과 같은 결과가 된다.

**주문 경로에는 닿지 않는다.** 결과는 sink 에 dict 한 줄씩(신호·주문·체결·취소·청산).
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

__all__ = ["TigerShadow"]

T0 = 9 * 3600
ENTRY_LO, ENTRY_HI = 300, 22800 - 600        # 09:05 ~ 15:10
WARMUP_SEC = 2100                            # 09:35 전은 전일 분위
Q_SIG = 0.99
SAME_SIZE_MIN = 3
CANCEL_SECS = 10
STOP_TICKS = 2
SDROP = 1.0
COST = 0.0023
MIN_PRICE = 1000


def _tick(price: float) -> float:
    from kiwoom_client.tick_size import tick_size

    return float(tick_size(Decimal(int(max(price, 1)))))


@dataclass
class _Code:
    sec: int = -1
    bid: float = float("nan")
    ask: float = float("nan")
    bidqty1: float = float("nan")
    strength: float = float("nan")
    last: float = float("nan")
    cumval: float = 0.0
    n_tr: int = 0
    buyval: float = 0.0
    sellval: float = 0.0
    sells_at: list[tuple[float, float]] = field(default_factory=list)   # 이번 초 (가격, 수량) 매도 체결
    min_px: float = float("inf")
    buys: deque = field(default_factory=deque)          # (초, 수량) 최근 10초 매수 체결
    flow: deque = field(default_factory=deque)          # (초, 매수대금, 매도대금) 최근 3초
    st_secs: list[int] = field(default_factory=list)
    st_vals: list[float] = field(default_factory=list)
    pending: dict | None = None
    open: dict | None = None
    free: int = -1


class TigerShadow:
    def __init__(self, *, prev_quantile: float | None, top_n: int = 30,
                 sink: Callable[[dict[str, Any]], None], min_quantile_rows: int = 1000) -> None:
        self.prev_q = prev_quantile
        self.top_n = top_n
        self.sink = sink
        self.min_rows = min_quantile_rows
        self.codes: dict[str, _Code] = {}
        self._dstr_rows: list[float] = []
        self._q_minute = -1
        self._q_value = float("inf")

    # ------------------------------------------------------------------ 입력
    def pending_count(self) -> int:
        return sum(1 for c in self.codes.values() if c.pending is not None)

    def open_count(self) -> int:
        return sum(1 for c in self.codes.values() if c.open is not None)

    def on_quote(self, code: str, ts: datetime, *, bid1: float | None, ask1: float | None,
                 bidqty1: float | None) -> None:
        st = self._roll(code, ts)
        if bid1 and ask1:
            st.bid, st.ask = float(bid1), float(ask1)
        if bidqty1 is not None:
            st.bidqty1 = float(bidqty1)

    def on_tick(self, code: str, ts: datetime, price: float, *, volume: float | None, side: int | None,
                strength: float | None, best_bid: float | None, best_ask: float | None) -> None:
        st = self._roll(code, ts)
        if best_bid and best_ask:
            st.bid, st.ask = float(best_bid), float(best_ask)
        vol = float(volume or 0)
        v = float(price) * vol
        st.last = float(price)
        st.cumval += v
        st.n_tr += 1
        st.min_px = min(st.min_px, float(price))
        if side is not None and side > 0:
            st.buyval += v
            st.buys.append((st.sec, vol))
        elif side is not None and side < 0:
            st.sellval += v
            st.sells_at.append((float(price), vol))
        if strength is not None:
            st.strength = float(strength)

    # ------------------------------------------------------------------ 초 마감
    def _roll(self, code: str, ts: datetime) -> _Code:
        sec = ts.hour * 3600 + ts.minute * 60 + ts.second - T0
        st = self.codes.setdefault(code, _Code())
        if st.sec != sec:
            if st.sec >= 0 and sec > st.sec:
                self._finalize(code, st)
            st.sec = sec
            st.n_tr, st.buyval, st.sellval, st.min_px = 0, 0.0, 0.0, float("inf")
            st.sells_at = []
        return st

    def _threshold(self, sec: int) -> float:
        if sec < WARMUP_SEC:
            return self.prev_q if self.prev_q is not None else float("inf")
        minute = sec // 60
        if minute != self._q_minute:
            self._q_minute = minute
            vals = sorted(self._dstr_rows)
            if len(vals) >= self.min_rows:
                k = Q_SIG * (len(vals) - 1)
                lo = int(k)
                hi = min(lo + 1, len(vals) - 1)
                self._q_value = vals[lo] + (vals[hi] - vals[lo]) * (k - lo)
            else:
                self._q_value = float("inf")
        return self._q_value

    def _rank(self, code: str) -> int:
        mine = self.codes[code].cumval
        return 1 + sum(1 for c in self.codes.values() if c.cumval > mine)

    def _emit(self, **kw: Any) -> None:
        self.sink(kw)

    def _finalize(self, code: str, st: _Code) -> None:
        s = st.sec
        if st.strength == st.strength:
            st.st_secs.append(s)
            st.st_vals.append(st.strength)
        st.flow.append((s, st.buyval, st.sellval))
        while st.flow and st.flow[0][0] < s - 2:
            st.flow.popleft()
        while st.buys and st.buys[0][0] < s - 9:
            st.buys.popleft()

        # (1) 대기 주문
        p = st.pending
        if p is not None:
            if p.get("P") is None:
                if s >= p["j"]:
                    if st.bid > 0 and st.bidqty1 == st.bidqty1:
                        p.update(P=st.bid, Qa=st.bidqty1, peak=st.strength, tick=_tick(st.bid))
                        self._emit(event="order", code=code, sec=s, price=st.bid, queue_ahead=st.bidqty1)
                    else:
                        st.pending = None
            else:
                P = p["P"]
                p["cum"] += sum(q for px, q in st.sells_at if px == P)
                if st.min_px < P or p["cum"] >= p["Qa"] + 1:
                    st.open = {"P": P, "e": s, "peak": st.strength, "stop": P - STOP_TICKS * p["tick"]}
                    st.pending = None
                    self._emit(event="fill", code=code, sec=s, price=P)
                else:
                    if st.strength == st.strength:
                        p["peak"] = st.strength if p["peak"] != p["peak"] else max(p["peak"], st.strength)
                    if st.strength <= p["peak"] - SDROP or s >= p["j"] + CANCEL_SECS:
                        st.pending = None
                        st.free = s + 1
                        self._emit(event="cancel", code=code, sec=s, price=P)
        # (2) 보유
        o = st.open
        if o is not None and s > o["e"]:
            reason = None
            if st.bid <= o["stop"]:
                reason = "손절"
            elif st.strength == st.strength:
                o["peak"] = st.strength if o["peak"] != o["peak"] else max(o["peak"], st.strength)
                if st.strength <= o["peak"] - SDROP:
                    reason = "불꺼짐"
            if reason:
                self._emit(event="close", code=code, sec=s, entry=o["P"], exit=st.bid, reason=reason,
                           hold=s - o["e"], ret=st.bid / o["P"] - 1 - COST)
                st.open = None
                st.free = s + 1

        # (3) 신호 — 결정 행 조건
        if not (st.n_tr > 0 and ENTRY_LO <= s <= ENTRY_HI and st.last >= MIN_PRICE and st.ask > st.bid > 0):
            return
        if self._rank(code) > self.top_n:
            return
        i = bisect_right(st.st_secs, s - 30)
        ago = st.st_vals[i - 1] if i else float("nan")
        dstr30 = st.strength - ago
        if dstr30 == dstr30:
            self._dstr_rows.append(dstr30)
        if st.pending is not None or st.open is not None or s < st.free:
            return
        buy3 = sum(b for _, b, _ in st.flow)
        sell3 = sum(x for _, _, x in st.flow)
        same = max(Counter(v for _, v in st.buys).values(), default=0)
        if dstr30 == dstr30 and dstr30 >= self._threshold(s) and same >= SAME_SIZE_MIN and buy3 > sell3:
            st.pending = {"j": s + 1, "cum": 0.0, "P": None}
            self._emit(event="signal", code=code, sec=s, dstr30=dstr30, same_size=same)

    def flush(self) -> None:
        for code, st in self.codes.items():
            if st.open is not None:
                self._emit(event="close", code=code, sec=st.sec, entry=st.open["P"], exit=st.bid, reason="마감",
                           hold=st.sec - st.open["e"], ret=st.bid / st.open["P"] - 1 - COST)
                st.open = None
