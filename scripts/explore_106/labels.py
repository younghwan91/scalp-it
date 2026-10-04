"""106번 라벨 — 95번 `ceiling.py` 의 oracle / fill_second / maker_pessimistic 을 그대로 쓰고,
**사전확정 청산**(H초 뒤 매수1호가 매도) 두 개만 여기서 더한다.

- 테이커(사전확정): t+1 매도1호가 매수 → t+1+H 매수1호가 매도 − 23bp.
- 메이커(사전확정): t+1 매수1호가 지정가, 10초 안에 **뚫려야** 체결(95 의 fill_second) → 체결 초 e 에서
  e+H 매수1호가 매도 − 23bp. 미체결은 NaN(체결 부분집합은 조건부다 — 99번 규칙 2).
- 천장: 95 의 oracle(테이커·메이커 사후 최적 매도) · maker_pessimistic(뚫림 체결 + 사후 최적 매도).

비용 23bp = 세금 20 + 수수료 3. 스프레드는 매도1호가에 사고 매수1호가에 파는 가격 자체에 들어 있다
(99번의 41.5bp = 23 + 스프레드 14~31 과 같은 모형이다).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "ceiling95", Path(__file__).resolve().parents[1] / "ceiling_95" / "ceiling.py")
C = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("ceiling95", C)
_spec.loader.exec_module(C)

COST = C.COST
FILL_SEC = C.FILL_SEC
oracle = C.oracle
fill_second = C.fill_second
maker_pessimistic = C.maker_pessimistic
window_max = C.window_max


def taker_fixed(bid: np.ndarray, ask: np.ndarray, h: int) -> np.ndarray:
    """r[t] = bid[t+1+h] / ask[t+1] − 1 − COST. 뒤가 모자라거나 호가가 없으면 NaN."""
    n = len(bid)
    out = np.full(n, np.nan)
    k = n - 1 - h
    if k <= 0:
        return out
    entry, exit_ = ask[1: 1 + k], bid[1 + h: 1 + h + k]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = exit_ / entry - 1.0 - COST
    r[~(entry > 0) | ~(exit_ > 0)] = np.nan
    out[:k] = r
    return out


def maker_fixed(bid: np.ndarray, e: np.ndarray, h: int) -> np.ndarray:
    """r[t] = bid[e[t]+h] / bid[t+1] − 1 − COST (e = fill_second 의 뚫림 체결 초). 미체결·범위 밖은 NaN."""
    n = len(bid)
    out = np.full(n, np.nan)
    idx = np.nonzero(e >= 0)[0]
    ex = e[idx] + h
    ok = ex < n
    idx, ex = idx[ok], ex[ok]
    P, X = bid[idx + 1], bid[ex]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = X / P - 1.0 - COST
    r[~(P > 0) | ~(X > 0)] = np.nan
    out[idx] = r
    return out
