"""autoresearch 실험 파일 — 에이전트가 고치는 유일한 파일(program.md). 실행은 평가기: ``uv run python research/autoresearch/prepare.py``.

기준선 = 97번 표의 칸 그대로(이미 본 것): 전일까지 거래대금이 큰 종목을 종가에 사서 다음 날 시가에 판다. 순위만 있고 조건은 없다.
숫자 파라미터는 ``params`` 범위만 정하고 값은 평가기가 폴드마다 학습 구간에서 고른다 — 여기서 숫자를 손으로 맞추지 않는다.

계약(prepare.load_strategy — AST 허용 목록: numpy·pandas·math·functools·itertools·collections 만, 파일·OS·getattr·난수·
더블 언더스코어·연도·종목코드 리터럴 금지, params 최대 6개·범위 폭 가드):
  get_strategy() → {"params": {이름: ("int"|"float", lo, hi) | ("choice", (값, ...))},
                    "signals": f(V, x) → {"cand": bool[N,T], "score": float[N,T]},
                    "cfg": f(x) → {"slots": 5|10|20|40}}
  V(읽기 전용, 날짜·종목코드 없음). t 열 = t 일 종가에 살지 정하는 순간 아는 것:
    O    t 일 시가                      C1·H1·L1·O1·V1·TV1  t−1 일 종가·고가·저가·시가·거래량·거래대금(수정가)
    uni  평가기 유니버스(t−1 정보)       kxo·kqo  t 일 KOSPI·KOSDAQ 시가    kx1·kq1  t−1 일 종가
    lim  그날 가격제한폭(0.15|0.30)      mkt      종목별 0 = 유가증권, 1 = 코스닥 (N,)
  cand 는 평가기가 uni ∧ 체결 가능(상한가 종가·정지일 제외)과 AND 한다. score 가 클수록 먼저 산다(NaN = 안 산다).
  신호는 열 t 까지의 값만 — 평가기가 절단 재계산으로 검사한다(앞쪽 shift·중심 rolling·전 기간 정규화는 void).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rmean(a: np.ndarray, w: int) -> np.ndarray:
    return pd.DataFrame(a.T).rolling(w, min_periods=w).mean().to_numpy().T


def signals(V: dict, x: dict) -> dict:
    liq = _rmean(V["TV1"], x["w"])
    cand = V["uni"] & np.isfinite(liq)
    return {"cand": cand, "score": np.where(cand, np.log(np.maximum(liq, 1.0)), np.nan)}


def cfg(x: dict) -> dict:
    return {"slots": x["slots"]}


def get_strategy() -> dict:
    return {"params": {"w": ("int", 1, 20), "slots": ("choice", (5, 10, 20, 40))},
            "signals": signals, "cfg": cfg}
