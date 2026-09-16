"""95번 천장 측정 — 오라클 계산이 **미래를 정확히 그만큼만** 보는지 확인한다.

여기서 한 칸이 밀리면 상한이 부풀거나 줄어 결론이 뒤집힌다. 그래서 값을 손으로 아는 경우로 고정한다.
"""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "ceiling95", Path(__file__).resolve().parents[1] / "scripts" / "ceiling_95" / "ceiling.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def test_oracle_picks_best_exit_in_window():
    #        t:  0    1     2     3     4
    bid = np.array([100, 100, 110, 130, 100], float)
    ask = np.array([101, 101, 111, 131, 101], float)
    tk, mk = C.oracle(bid, ask, h=2)              # t=0 → 진입 ask[1]=101, 매도 max(bid[2],bid[3])=130
    assert abs(tk[0] - (130 / 101 - 1 - C.COST)) < 1e-12
    assert abs(mk[0] - (131 / 100 - 1 - C.COST)) < 1e-12   # 메이커 진입 bid[1]=100, 매도 max(ask)=131


def test_window_does_not_reach_beyond_horizon():
    bid = np.array([100, 100, 100, 100, 999], float)      # 큰 값이 창 밖(t+3)에 있다
    ask = bid + 1
    tk, _ = C.oracle(bid, ask, h=1)                       # t=0 창 = [t+2, t+2]
    assert tk[0] < 0, "창 밖 미래를 보면 안 된다"


def test_missing_prices_become_nan():
    bid = np.array([0, 0, 0, 0], float)
    ask = np.array([0, 0, 0, 0], float)
    tk, mk = C.oracle(bid, ask, h=2)
    assert np.isnan(tk).all() and np.isnan(mk).all()


def test_cost_is_subtracted_once():
    bid = np.full(6, 100.0)
    ask = np.full(6, 100.0)                                # 스프레드 0 · 움직임 없음
    tk, _ = C.oracle(bid, ask, h=3)
    assert abs(tk[0] + C.COST) < 1e-12
