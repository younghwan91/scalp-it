"""autoresearch 실험 파일 — 에이전트가 고치는 유일한 파일(program.md). 실행은 평가기: ``uv run python research/autoresearch/prepare.py``.

기준선 = 97번 표의 칸 그대로(이미 본 것): 전일까지 거래대금이 큰 종목을 종가에 사서 다음 날 시가에 판다. 순위만 있고 조건은 없다.
숫자 파라미터는 ``params`` 범위만 정하고 값은 평가기가 폴드마다 학습 구간에서 고른다 — 여기서 숫자를 손으로 맞추지 않는다.

계약(ar_sandbox — train.py 는 평가기 데이터가 없는 별도 프로세스에서만 돈다. AST 허용 목록: numpy·pandas·math·functools·itertools·
collections·dataclasses·typing 의 최상위 import 만, 파일·OS·getattr·난수·print·class·yield·밑줄로 시작하는 속성·속성 대입·
연도·종목코드 리터럴 금지, params 최대 6개·범위 폭 ≥ 1/3):
  get_strategy() → {"params": {이름: ("int"|"float", lo, hi) | ("choice", (값, ...))},
                    "signals": f(V, x) → {"cand": bool[N,T], "score": float[N,T]},
                    "cfg": f(x) → {"slots": 5|10|20|40}}
  V(읽기 전용, 날짜·종목코드·시장 구분·지수 수준 없음). t 열 = t 일 종가에 살지 정하는 순간 아는 것:
    O       t 일 시가                          C1·H1·L1·O1·V1·TV1  t−1 일 종가·고가·저가·시가·거래량·거래대금(수정가)
    uni     평가기 유니버스(t−1 정보)          lim                 그날 가격제한폭(0.15|0.30) (T,)
    kx_gap·kq_gap  t 일 KOSPI·KOSDAQ 시가 갭    kx_r1·kq_r1         t−1 일 KOSPI·KOSDAQ 일간 수익률 (T,)
  cand 는 평가기가 uni ∧ 체결 가능(상한가 종가·정지일 제외)과 AND 한다. score 가 클수록 먼저 산다(NaN = 안 산다).
  평가기 검사 — 어기면 void: ① 미래정보(열 t 까지의 값만: 앞쪽 shift·중심 rolling·전 기간 정규화 금지)
  ② 이동 불변(최근 300열 안의 값만: 누적·expanding·열 번호 금지) ③ 행 치환(종목 순서·번호로 고르지 않는다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rmean(a: np.ndarray, w: int, mp: int) -> np.ndarray:
    return pd.DataFrame(a.T).rolling(w, min_periods=mp).mean().to_numpy().T


def _rstd(a: np.ndarray, w: int, mp: int) -> np.ndarray:
    return pd.DataFrame(a.T).rolling(w, min_periods=mp).std().to_numpy().T


def _xrank(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    """열(날)마다 후보 안 백분위 순위 — 그날 값만 쓴다."""
    return pd.DataFrame(np.where(m, a, np.nan)).rank(axis=0, pct=True).to_numpy()


def _tick_rel(p: np.ndarray) -> np.ndarray:
    """편도 1틱 / 가격 — 옛 호가표(새 표보다 어디서나 크거나 같다 — 시대를 모르므로 보수). 기대 비용을 점수에 넣는 데만 쓴다."""
    with np.errstate(invalid="ignore"):
        old = np.select([p < 1000, p < 5000, p < 10000, p < 50000, p < 100000, p < 500000], [1, 5, 10, 50, 100, 500], 1000)
        return old / p


def _cir(gap: np.ndarray, tk: np.ndarray, lam: float, w: int) -> np.ndarray:
    mp = max(3, w // 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (_rmean(gap, w, mp) - lam * tk) / _rstd(gap, w, mp)


def signals(V: dict, x: dict) -> dict:
    # 비용 조정 야간 IR((평균 갭 − λ·틱/가격)/갭 표준편차)을 창 w/2·w·2w 세 개로 평균 — 창 선택 잡음을 줄인다.
    gap = np.log(V["O"] / V["C1"])
    tk = _tick_rel(V["C1"])
    w = x["w"]
    on = (_cir(gap, tk, x["lam"], max(3, w // 2)) + _cir(gap, tk, x["lam"], w) + _cir(gap, tk, x["lam"], min(2 * w, 250))) / 3
    gp = V["O"] / V["C1"] - 1
    ret1 = V["C1"] / pd.DataFrame(V["C1"].T).shift(1).to_numpy().T - 1
    ev = (gp >= V["lim"][None, :] / 2) | (np.abs(ret1) >= V["lim"][None, :] - 0.02)   # 전일 상·하한가 마감도 사건
    cand = V["uni"] & np.isfinite(on) & ~ev       # 오늘 시가가 제한폭 절반 이상 갭업 = 사건 종목(상한가 근처 종가는 어차피 못 산다)
    return {"cand": cand, "score": np.where(cand, on, np.nan)}


def cfg(x: dict) -> dict:
    return {"slots": x["slots"]}


def get_strategy() -> dict:
    return {"params": {"w": ("int", 5, 120), "lam": ("float", 0.0, 3.0), "slots": ("choice", (5, 10, 20, 40))},
            "signals": signals, "cfg": cfg}
