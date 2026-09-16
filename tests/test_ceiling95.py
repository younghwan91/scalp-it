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


def test_pessimistic_fill_requires_price_through():
    bid = np.array([100.0] * 20)
    e = C.fill_second(bid)
    assert (e < 0).all(), "가격이 안 뚫렸는데 체결로 보면 역선택을 과소평가한다"
    bid2 = bid.copy()
    bid2[5] = 99.0                                     # t=3 지정가 bid[4]=100 → t=5 에 뚫린다
    e2 = C.fill_second(bid2)
    assert e2[3] == 5


def test_pessimistic_return_measured_from_fill_second():
    bid = np.array([100.0] * 20)
    bid[5] = 99.0
    ask = np.array([101.0] * 20)
    ask[7] = 120.0
    e = C.fill_second(bid)
    mp = C.maker_pessimistic(bid, e, C.window_max(ask, 3))   # 체결 t=5 → 매도창 [6,8] 최댓값 120
    assert abs(mp[3] - (120 / 100 - 1 - C.COST)) < 1e-12


def test_short_oracle_profits_when_price_falls():
    #          t: 0    1    2    3    4
    bid = np.array([100, 100, 90, 80, 80], float)
    ask = np.array([101, 101, 91, 81, 81], float)
    tk, mk = C.oracle_short(bid, ask, h=2)      # t=0 → 진입 bid[1]=100, 환매수 min(ask[2],ask[3])=81
    assert abs(tk[0] - (100 / 81 - 1 - C.COST)) < 1e-12
    assert abs(mk[0] - (101 / 80 - 1 - C.COST)) < 1e-12


def test_short_loses_when_price_rises():
    bid = np.array([100, 100, 110, 120, 120], float)
    tk, _ = C.oracle_short(bid, bid + 1, h=2)
    assert tk[0] < 0
