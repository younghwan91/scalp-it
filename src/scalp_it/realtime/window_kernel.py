"""창 재계산 커널 — 실시간에서도 **백테스트와 같은 함수**를 호출해 피처를 만든다.

core v2(증분 커널)가 나오기 전까지 쓰는 다리다. 증분 구현을 새로 짜면 값이 어긋날 위험이 있어,
여기서는 초마다 최근 창(기본 700초)의 원시 틱·호가로 **배치 함수 자체**를 다시 돌리고 마지막 행만 쓴다.
정의상 값이 같다(같은 코드 경로) — 대신 CPU 를 쓰므로 종목 수를 제한해서 붙인다.

**부하 실측**: 종목 1개·1초 재계산 **81ms**(trader, 2026-09-16, 평균 종목) / **55~129ms**(simnode, 2026-09-22,
거래대금 상위 3종목 · 첫 호출은 numba 컴파일로 1,481ms). 산술로는 3종목이 초당 300ms 지만, **실제로는 3종목에서도
프레임을 버렸다**(09-22 실측 34,034건) — 소켓 이벤트 루프와 GIL 을 나눠 쓰기 때문이다. 그래서 운용은 **1종목**이다.
그래서 이 커널은 "소수 종목으로 의사결정 패리티를 먼저 모으는" 용도이고, core `SecondFeatureStreamV2`(증분)가
나오면 교체한다(패리티 검사로 동치 확인 후 넓힌다).
"""
from __future__ import annotations

import importlib.util
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["WindowKernelStream", "load_batch_modules"]

WINDOW_SEC = 700          # 가장 긴 창(600초 기준선 + 여유)
T0 = 9 * 3600


def load_batch_modules(root: Path | None = None) -> tuple[Any, Any]:
    """백테스트가 쓰는 바로 그 모듈(flow_83.features, flow_84.library)을 불러온다."""
    base = Path(root or Path(__file__).resolve().parents[3])
    mods = []
    for name, rel in (("f83_features", "scripts/flow_83/features.py"), ("f84_library", "scripts/flow_84/library.py")):
        if name in sys.modules:
            mods.append(sys.modules[name])
            continue
        spec = importlib.util.spec_from_file_location(name, base / rel)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        sys.path.insert(0, str(base / "scripts" / "flow_83"))
        spec.loader.exec_module(mod)
        mods.append(mod)
    return mods[0], mods[1]


class WindowKernelStream:
    """한 종목의 실시간 피처. `update(...)` 한 번이 그 초의 피처 dict 를 돌려준다."""

    def __init__(self, code: str, *, day: str, window_sec: int = WINDOW_SEC, root: Path | None = None) -> None:
        self.code = code
        self.day = day
        self.window = window_sec
        self.f83, self.f84 = load_batch_modules(root)
        self.ticks: deque[tuple[datetime, float, float, int, float, float, float]] = deque()
        self.quotes: deque[tuple[datetime, float, float, float, float, float, float, float, float]] = deque()

    def add_tick(self, ts: datetime, price: float, volume: float, side: int, strength: float,
                 best_bid: float, best_ask: float) -> None:
        self.ticks.append((ts, price, volume, side, strength, best_bid, best_ask))

    def add_quote(self, ts: datetime, bid1: float, ask1: float, bq1: float, bq2: float, bq3: float,
                  aq1: float, aq2: float, aq3: float) -> None:
        self.quotes.append((ts, bid1, ask1, bq1, bq2, bq3, aq1, aq2, aq3))

    @staticmethod
    def _rel(sec: int) -> int:
        """09:00 기준 초로 맞춘다 — 호출자가 자정 기준(32400+) 을 줘도 받는다."""
        return sec - T0 if sec >= T0 else sec

    def _trim(self, sec: int) -> None:
        lo = sec - self.window
        while self.ticks and (self.ticks[0][0].hour * 3600 + self.ticks[0][0].minute * 60
                              + self.ticks[0][0].second - T0) < lo:
            self.ticks.popleft()
        while self.quotes and (self.quotes[0][0].hour * 3600 + self.quotes[0][0].minute * 60
                               + self.quotes[0][0].second - T0) < lo:
            self.quotes.popleft()

    def update(self, sec: int) -> dict[str, float]:
        """초 `sec` 마감 시점의 피처. 배치와 같은 함수를 창 데이터로 돌린다."""
        sec = self._rel(sec)
        self._trim(sec)
        if not self.ticks:
            return {}
        tk = pd.DataFrame(self.ticks, columns=["ts", "price", "volume", "side", "strength", "best_bid", "best_ask"])
        tk.insert(0, "code", self.code)
        tk["seq"] = range(len(tk))
        qt = pd.DataFrame(self.quotes, columns=["ts", "bid1", "ask1", "bidqty1", "bidqty2", "bidqty3",
                                                "askqty1", "askqty2", "askqty3"])
        if len(qt):
            qt.insert(0, "code", self.code)
        f, p = self.f83.build_code(tk, qt)          # ← 백테스트가 쓰는 바로 그 함수
        x = self.f84._extra(tk, qt, f, p)
        out: dict[str, float] = {}
        for k in self.f83.FEATURES:
            if k in f:
                out[k] = float(f[k][sec])
        for k, v in x.items():
            out[k] = float(v[sec])
        out["log_cumval"] = float(__import__("numpy").log1p(f["cumval"][sec]))
        out["sec"] = float(sec)
        return out
